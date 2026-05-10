from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import imageio
import numpy as np
import torch
from torchvision.transforms import functional as F


DEFAULT_WIDTH = 280
DEFAULT_HEIGHT = 168
DEFAULT_FPS = 16


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"Failed to read json: {path}\nError: {exc!r}")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _prepare_imports(neoverse_repo: Path) -> None:
    repo_path = str(neoverse_repo.resolve())
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)


def _resolve_repo_relative_path(repo_root: Path, path_value: str) -> Path:
    path = Path(str(path_value))
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def _parse_torch_dtype(name: str) -> torch.dtype:
    value = str(name).strip().lower()
    if value in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if value in {"fp16", "float16"}:
        return torch.float16
    if value in {"fp32", "float32"}:
        return torch.float32
    raise SystemExit(f"Unsupported --torch_dtype: {name}")


def _make_gaussian(gaussian_cls, item: dict[str, Any]):
    kwargs = {
        "means": item["means"],
        "harmonics": item["harmonics"],
        "opacities": item["opacities"],
        "scales": item["scales"],
        "rotations": item["rotations"],
        "confidences": item.get("confidences"),
        "timestamp": int(item.get("timestamp", -1)),
        "life_span": item.get("life_span", 1.0),
        "life_span_gamma": float(item.get("life_span_gamma", 0.0)),
        "forward_timestamp": item.get("forward_timestamp"),
        "forward_vel": item.get("forward_vel"),
        "forward_scales": item.get("forward_scales"),
        "forward_rotations": item.get("forward_rotations"),
        "backward_timestamp": item.get("backward_timestamp"),
        "backward_vel": item.get("backward_vel"),
        "backward_scales": item.get("backward_scales"),
        "backward_rotations": item.get("backward_rotations"),
    }
    return gaussian_cls(**kwargs)


def _bundle_camera_arrays(bundle: dict[str, Any], camera_source: str) -> tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]:
    if camera_source == "rendered":
        pose_keys = ["rendered_cam2world", "rendered_extrinsics"]
        intr_keys = ["rendered_intrinsics"]
        ts_keys = ["rendered_timestamps"]
        source_label = "rendered"
    elif camera_source == "predicted_camera":
        pose_keys = ["predicted_camera_cam2world", "predicted_cam2world", "predicted_extrinsics"]
        intr_keys = ["predicted_camera_intrinsics", "predicted_intrinsics"]
        ts_keys = ["predicted_timestamps", "rendered_timestamps"]
        source_label = "predicted_camera"
    else:
        raise SystemExit(f"Unsupported --camera_source: {camera_source}")

    pose = None
    for key in pose_keys:
        value = bundle.get(key)
        if isinstance(value, torch.Tensor):
            pose = value
            break
    intr = None
    for key in intr_keys:
        value = bundle.get(key)
        if isinstance(value, torch.Tensor):
            intr = value
            break
    ts = None
    for key in ts_keys:
        value = bundle.get(key)
        if isinstance(value, torch.Tensor):
            ts = value
            break

    if pose is None or intr is None or ts is None:
        raise SystemExit(f"Bundle missing camera arrays for camera_source={camera_source}")

    if pose.ndim == 3:
        pose = pose.unsqueeze(0)
    if intr.ndim == 3:
        intr = intr.unsqueeze(0)
    if ts.ndim == 1:
        ts = ts.unsqueeze(0)

    pose = pose.float()
    intr = intr.float()
    ts = ts.long()

    return source_label, pose, intr, ts


def _read_rgb_image(path: Path):
    from PIL import Image

    try:
        return Image.open(path).convert("RGB")
    except Exception as exc:
        raise SystemExit(f"Failed to read image: {path}; error={exc!r}")


def _resize_with_intrinsics(img, K: np.ndarray, target_width: int, target_height: int):
    from PIL import Image

    src_width, src_height = img.size
    scale_x = float(target_width) / float(src_width)
    scale_y = float(target_height) / float(src_height)
    out = img.resize((target_width, target_height), resample=Image.LANCZOS)

    K_out = np.asarray(K, dtype=np.float32).copy()
    K_out[0, 0] *= scale_x
    K_out[1, 1] *= scale_y
    K_out[0, 2] *= scale_x
    K_out[1, 2] *= scale_y
    return out, K_out


def _center_crop_with_intrinsics(img, K: np.ndarray, target_width: int, target_height: int):
    from PIL import Image

    src_width, src_height = img.size
    scale = max(float(target_width) / float(src_width), float(target_height) / float(src_height))
    scaled_width = int(round(src_width * scale))
    scaled_height = int(round(src_height * scale))
    scaled = img.resize((scaled_width, scaled_height), resample=Image.LANCZOS)

    left = (scaled_width - target_width) // 2
    top = (scaled_height - target_height) // 2
    out = scaled.crop((left, top, left + target_width, top + target_height))

    K_out = np.asarray(K, dtype=np.float32).copy()
    K_out[0, 0] *= scale
    K_out[1, 1] *= scale
    K_out[0, 2] = K_out[0, 2] * scale - float(left)
    K_out[1, 2] = K_out[1, 2] * scale - float(top)
    return out, K_out


def _prepare_image_and_intrinsics(img, K: np.ndarray, target_width: int, target_height: int, resize_mode: str):
    if resize_mode == "resize":
        return _resize_with_intrinsics(img, K, target_width, target_height)
    if resize_mode == "center_crop":
        return _center_crop_with_intrinsics(img, K, target_width, target_height)
    raise SystemExit(f"Unsupported --resize_mode: {resize_mode}")


def _build_gaussians(bundle: dict[str, Any], neoverse_repo: Path) -> list[Any]:
    _prepare_imports(neoverse_repo)
    from diffsynth.auxiliary_models.worldmirror.models.models.rasterization import Gaussians

    splats_serialized = list(bundle.get("splats_serialized") or [])
    if not splats_serialized:
        raise SystemExit("Bundle has empty splats_serialized")
    return [_make_gaussian(Gaussians, item) for item in splats_serialized]


def _move_gaussians_to_device(gaussians: list[Any], device: str, torch_dtype: torch.dtype) -> list[Any]:
    moved = []
    for gaussian in gaussians:
        for name, value in list(gaussian.__dict__.items()):
            if isinstance(value, torch.Tensor):
                target_dtype = torch_dtype if value.is_floating_point() else value.dtype
                setattr(gaussian, name, value.to(device=device, dtype=target_dtype))
        moved.append(gaussian)
    return moved


def _load_reconstructor(reconstructor_path: Path, device: str, torch_dtype: torch.dtype):
    from diffsynth.models import ModelManager

    model_manager = ModelManager(torch_dtype=torch_dtype, device=device)
    model_manager.load_model(str(reconstructor_path), device=device, torch_dtype=torch_dtype)
    reconstructor = model_manager.fetch_model("reconstructor")
    if reconstructor is None:
        raise SystemExit(f"Failed to load reconstructor from: {reconstructor_path}")
    if device.startswith("cuda"):
        reconstructor.to(device)
    reconstructor.eval()
    return reconstructor


def _tensor_to_uint8_frame(frame: np.ndarray) -> np.ndarray:
    frame = np.asarray(frame)
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0.0, 1.0)
        frame = (frame * 255.0).round().astype(np.uint8)
    if frame.ndim == 2:
        frame = np.repeat(frame[:, :, None], 3, axis=2)
    if frame.shape[-1] == 1:
        frame = np.repeat(frame, 3, axis=2)
    return frame


def _colorize_depth_sequence(depth_seq: torch.Tensor) -> list[np.ndarray]:
    depth = depth_seq.detach().float().cpu()
    while depth.ndim > 4 and depth.shape[0] == 1:
        depth = depth[0]
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    finite = depth[torch.isfinite(depth)]
    finite = finite[finite > 0]
    if finite.numel() == 0:
        min_depth, max_depth = 0.0, 1.0
    else:
        min_depth = float(torch.quantile(finite, 0.02).item())
        max_depth = float(torch.quantile(finite, 0.98).item())
        if max_depth <= min_depth:
            max_depth = min_depth + 1.0
    depth = depth.clamp(min=min_depth, max=max_depth)
    depth = (depth - min_depth) / (max_depth - min_depth + 1e-6)
    frames = []
    for frame in depth:
        arr = (frame.numpy() * 255.0).round().astype(np.uint8)
        frames.append(np.repeat(arr[:, :, None], 3, axis=2))
    return frames


def _alpha_to_frames(alpha_seq: torch.Tensor) -> list[np.ndarray]:
    alpha = alpha_seq.detach().float().cpu()
    while alpha.ndim > 4 and alpha.shape[0] == 1:
        alpha = alpha[0]
    if alpha.ndim == 4 and alpha.shape[-1] == 1:
        alpha = alpha[..., 0]
    alpha = alpha.clamp(0.0, 1.0)
    frames = []
    for frame in alpha:
        arr = (frame.numpy() * 255.0).round().astype(np.uint8)
        frames.append(np.repeat(arr[:, :, None], 3, axis=2))
    return frames


def _rgb_to_frames(rgb_seq: torch.Tensor) -> list[np.ndarray]:
    rgb = rgb_seq.detach().float().cpu()
    while rgb.ndim > 4 and rgb.shape[0] == 1:
        rgb = rgb[0]
    if rgb.ndim == 4 and rgb.shape[-1] != 3:
        raise SystemExit(f"Unexpected rgb tensor shape: {tuple(rgb.shape)}")
    frames = []
    for frame in rgb:
        frames.append(_tensor_to_uint8_frame(frame.numpy()))
    return frames


def _render_sequence(
    rasterizer,
    gaussians: list[Any],
    render_viewmats: torch.Tensor,
    render_Ks: torch.Tensor,
    render_timestamps: torch.Tensor,
    width: int,
    height: int,
    device: str,
    torch_dtype: torch.dtype,
):
    render_viewmats = render_viewmats.to(device)
    render_Ks = render_Ks.to(device)
    render_timestamps = render_timestamps.to(device)

    use_amp = device.startswith("cuda")
    with torch.no_grad():
        if use_amp:
            with torch.amp.autocast("cuda", dtype=torch_dtype):
                rgb, depth, alpha = rasterizer.forward(
                    [gaussians],
                    render_viewmats=[render_viewmats],
                    render_Ks=[render_Ks],
                    render_timestamps=[render_timestamps],
                    sh_degree=0,
                    width=width,
                    height=height,
                )
        else:
            rgb, depth, alpha = rasterizer.forward(
                [gaussians],
                render_viewmats=[render_viewmats],
                render_Ks=[render_Ks],
                render_timestamps=[render_timestamps],
                sh_degree=0,
                width=width,
                height=height,
            )
    return rgb, depth, alpha


def _compose_compare_frames(input_frames: list[np.ndarray], rendered_frames: list[np.ndarray], alpha_frames: list[np.ndarray]) -> list[np.ndarray]:
    if not (len(input_frames) == len(rendered_frames) == len(alpha_frames)):
        raise SystemExit("Compare video frame count mismatch")
    composed = []
    for left, middle, right in zip(input_frames, rendered_frames, alpha_frames):
        composed.append(np.concatenate([left, middle, right], axis=1))
    return composed


def _save_video_exact_size(frames: list[np.ndarray], save_path: Path, fps: int) -> None:
    writer = imageio.get_writer(
        str(save_path),
        fps=fps,
        quality=9,
        macro_block_size=1,
    )
    for frame in frames:
        writer.append_data(np.asarray(frame))
    writer.close()


def _group_views(manifest: dict[str, Any], camera_source: str, bundle: dict[str, Any]) -> dict[str, dict[str, Any]]:
    source_label, camera_poses, camera_intrs, timestamps = _bundle_camera_arrays(bundle, camera_source)
    views = list(manifest.get("views") or [])
    if not views:
        raise SystemExit("source_manifest.views is empty")

    grouped: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(views):
        cam_id = str(item["cam_id"])
        grouped.setdefault(cam_id, {"items": [], "indices": []})
        grouped[cam_id]["items"].append(item)
        grouped[cam_id]["indices"].append(index)

    for cam_id, payload in grouped.items():
        ordered_pairs = sorted(
            zip(payload["items"], payload["indices"]),
            key=lambda pair: (int(pair[0]["logical_t_idx"]), int(pair[1])),
        )
        payload["items"] = [item for item, _ in ordered_pairs]
        indices = [index for _, index in ordered_pairs]
        payload["indices"] = indices
        payload["camera_poses"] = camera_poses[0, indices]
        payload["camera_intrs"] = camera_intrs[0, indices]
        payload["timestamps"] = timestamps[0, indices]
        payload["source_label"] = source_label
    return grouped


def _resolve_dimensions(bundle_dir: Path, bundle: dict[str, Any], args: argparse.Namespace) -> tuple[int, int, str]:
    probe_meta_path = bundle_dir / "probe_meta.json"
    probe_meta = _load_json(probe_meta_path) if probe_meta_path.exists() else {}
    input_resolution = probe_meta.get("input_resolution") or {}
    width = int(args.width) if args.width is not None else int(input_resolution.get("width", DEFAULT_WIDTH))
    height = int(args.height) if args.height is not None else int(input_resolution.get("height", DEFAULT_HEIGHT))
    resize_mode = str(args.resize_mode) if args.resize_mode is not None else str(probe_meta.get("resize_mode", "center_crop"))
    return width, height, resize_mode


def _load_reference_image_sequence(
    items: list[dict[str, Any]],
    scene_dir: Path,
    target_width: int,
    target_height: int,
    resize_mode: str,
) -> list[np.ndarray]:
    prepared_frames = []
    for item in items:
        frame_abs = scene_dir / str(item["frame_rel"])
        img = _read_rgb_image(frame_abs)
        K = np.asarray(item["camera_K"], dtype=np.float32)
        prepared_img, _ = _prepare_image_and_intrinsics(img, K, target_width, target_height, resize_mode)
        prepared_frames.append(_tensor_to_uint8_frame(np.asarray(prepared_img)))
    return prepared_frames


def main() -> None:
    ap = argparse.ArgumentParser(description="Render lightweight NeoVerse multiview previews from reconstruction_bundle.pt.")
    ap.add_argument("--bundle", required=True, type=str)
    ap.add_argument("--neoverse_repo", default="third_party/NeoVerse", type=str)
    ap.add_argument(
        "--reconstructor_path",
        default="third_party/NeoVerse/models/NeoVerse/reconstructor.ckpt",
        type=str,
    )
    ap.add_argument("--device", default="cuda", type=str)
    ap.add_argument("--torch_dtype", default="float16", type=str)
    ap.add_argument("--out_dir", default="", type=str)
    ap.add_argument("--camera_source", default="rendered", choices=["rendered", "predicted_camera"], type=str)
    ap.add_argument("--preview_mode", default="both", choices=["original", "orbit", "both"], type=str)
    ap.add_argument("--width", default=None, type=int)
    ap.add_argument("--height", default=None, type=int)
    ap.add_argument("--resize_mode", default=None, choices=["center_crop", "resize"], type=str)
    ap.add_argument("--reference_cam", default="cam0", type=str)
    ap.add_argument("--trajectory", default="orbit_left", type=str)
    ap.add_argument("--angle", default=12.0, type=float)
    ap.add_argument("--orbit_radius", default=0.08, type=float)
    ap.add_argument("--fps", default=DEFAULT_FPS, type=int)
    args = ap.parse_args()

    repo_root = _repo_root()
    bundle_path = Path(str(args.bundle))
    if not bundle_path.is_absolute():
        bundle_path = repo_root / bundle_path
    bundle_path = bundle_path.resolve()
    if not bundle_path.exists():
        raise SystemExit(f"Missing --bundle: {bundle_path}")

    bundle_dir = bundle_path.parent
    if str(args.out_dir).strip():
        out_dir = Path(str(args.out_dir))
        if not out_dir.is_absolute():
            out_dir = bundle_dir / out_dir
    else:
        out_dir = bundle_dir / "render_preview"
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle = torch.load(bundle_path, map_location="cpu")
    if not isinstance(bundle, dict):
        raise SystemExit("Invalid bundle format: expected dict")

    manifest = bundle.get("source_manifest")
    if not isinstance(manifest, dict):
        raise SystemExit("Bundle missing source_manifest dict")

    scene_dir = Path(str(manifest.get("scene_dir", "")))
    if not scene_dir.is_absolute():
        scene_dir = repo_root / scene_dir
    scene_dir = scene_dir.resolve()
    if not scene_dir.exists():
        raise SystemExit(f"source_manifest.scene_dir not found: {scene_dir}")

    width, height, resize_mode = _resolve_dimensions(bundle_dir, bundle, args)

    neoverse_repo = _resolve_repo_relative_path(repo_root, str(args.neoverse_repo))
    reconstructor_path = _resolve_repo_relative_path(repo_root, str(args.reconstructor_path))

    _prepare_imports(neoverse_repo)
    from diffsynth.utils.auxiliary import CameraTrajectory, homo_matrix_inverse

    gaussians = _build_gaussians(bundle, neoverse_repo)
    gaussians = _move_gaussians_to_device(gaussians, str(args.device), torch.float32)
    reconstructor = _load_reconstructor(reconstructor_path, str(args.device), _parse_torch_dtype(str(args.torch_dtype)))
    rasterizer = reconstructor.gs_renderer.rasterizer

    grouped_views = _group_views(manifest, str(args.camera_source), bundle)
    if str(args.reference_cam) not in grouped_views:
        raise SystemExit(f"reference_cam not found in source_manifest views: {args.reference_cam}")

    output_paths: dict[str, Any] = {
        "original": {},
        "original_compare": {},
        "orbit": {},
    }

    if args.preview_mode in {"original", "both"}:
        original_dir = out_dir / "original"
        compare_dir = out_dir / "original_compare"
        original_dir.mkdir(parents=True, exist_ok=True)
        compare_dir.mkdir(parents=True, exist_ok=True)

        for cam_id in sorted(grouped_views.keys()):
            payload = grouped_views[cam_id]
            items = list(payload["items"])
            prepared_frames = _load_reference_image_sequence(items, scene_dir, width, height, resize_mode)
            camera_poses = payload["camera_poses"]
            camera_intrs = payload["camera_intrs"]
            timestamps = payload["timestamps"]

            pose_w2c = homo_matrix_inverse(camera_poses)
            rgb, depth, alpha = _render_sequence(
                rasterizer=rasterizer,
                gaussians=gaussians,
                render_viewmats=pose_w2c,
                render_Ks=camera_intrs,
                render_timestamps=timestamps,
                width=width,
                height=height,
                device=str(args.device),
                torch_dtype=_parse_torch_dtype(str(args.torch_dtype)),
            )

            rgb_frames = _rgb_to_frames(rgb)
            depth_frames = _colorize_depth_sequence(depth)
            alpha_frames = _alpha_to_frames(alpha)
            compare_frames = _compose_compare_frames(prepared_frames, rgb_frames, alpha_frames)

            rgb_path = original_dir / f"{cam_id}_rgb.mp4"
            mask_path = original_dir / f"{cam_id}_mask.mp4"
            depth_path = original_dir / f"{cam_id}_depth.mp4"
            compare_path = compare_dir / f"{cam_id}_compare.mp4"

            _save_video_exact_size(rgb_frames, rgb_path, fps=int(args.fps))
            _save_video_exact_size(alpha_frames, mask_path, fps=int(args.fps))
            _save_video_exact_size(depth_frames, depth_path, fps=int(args.fps))
            _save_video_exact_size(compare_frames, compare_path, fps=int(args.fps))

            output_paths["original"][cam_id] = {
                "rgb": rgb_path.as_posix(),
                "mask": mask_path.as_posix(),
                "depth": depth_path.as_posix(),
            }
            output_paths["original_compare"][cam_id] = {
                "compare": compare_path.as_posix(),
            }

    if args.preview_mode in {"orbit", "both"}:
        orbit_dir = out_dir / "orbit"
        orbit_dir.mkdir(parents=True, exist_ok=True)

        ref_payload = grouped_views[str(args.reference_cam)]
        ref_items = list(ref_payload["items"])
        ref_camera_poses = ref_payload["camera_poses"]
        ref_camera_intrs = ref_payload["camera_intrs"]
        ref_timestamps = ref_payload["timestamps"]
        ref_pose = ref_camera_poses[0]
        ref_intr = ref_camera_intrs[0]

        cam_traj = CameraTrajectory.from_predefined(
            str(args.trajectory),
            num_frames=len(ref_items),
            mode="relative",
            angle=float(args.angle),
            orbit_radius=float(args.orbit_radius),
        )

        target_cam2world = cam_traj.c2w.to(ref_pose.device)
        if cam_traj.mode == "relative":
            target_cam2world = ref_pose.unsqueeze(0) @ target_cam2world
        target_world2cam = homo_matrix_inverse(target_cam2world)

        target_intrs = ref_intr.unsqueeze(0).repeat(len(ref_items), 1, 1)
        target_timestamps = ref_timestamps

        rgb, depth, alpha = _render_sequence(
            rasterizer=rasterizer,
            gaussians=gaussians,
            render_viewmats=target_world2cam,
            render_Ks=target_intrs,
            render_timestamps=target_timestamps,
            width=width,
            height=height,
            device=str(args.device),
            torch_dtype=_parse_torch_dtype(str(args.torch_dtype)),
        )

        rgb_frames = _rgb_to_frames(rgb)
        depth_frames = _colorize_depth_sequence(depth)
        alpha_frames = _alpha_to_frames(alpha)

        rgb_path = orbit_dir / f"{args.trajectory}_rgb.mp4"
        mask_path = orbit_dir / f"{args.trajectory}_mask.mp4"
        depth_path = orbit_dir / f"{args.trajectory}_depth.mp4"

        _save_video_exact_size(rgb_frames, rgb_path, fps=int(args.fps))
        _save_video_exact_size(alpha_frames, mask_path, fps=int(args.fps))
        _save_video_exact_size(depth_frames, depth_path, fps=int(args.fps))

        output_paths["orbit"] = {
            "rgb": rgb_path.as_posix(),
            "mask": mask_path.as_posix(),
            "depth": depth_path.as_posix(),
        }

    preview_meta = {
        "bundle": bundle_path.as_posix(),
        "camera_source": str(args.camera_source),
        "preview_mode": str(args.preview_mode),
        "reference_cam": str(args.reference_cam),
        "trajectory": str(args.trajectory),
        "trajectory_args": {
            "angle": float(args.angle),
            "orbit_radius": float(args.orbit_radius),
        },
        "width": int(width),
        "height": int(height),
        "resize_mode": str(resize_mode),
        "fps": int(args.fps),
        "outputs": output_paths,
    }
    preview_meta_path = out_dir / "preview_meta.json"
    _write_json(preview_meta_path, preview_meta)

    print(f"Wrote preview outputs to: {out_dir}")
    print(f"Wrote: {preview_meta_path}")


if __name__ == "__main__":
    main()