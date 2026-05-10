from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from neoverse_fused_utils import (
    alpha_to_frames,
    build_gaussians,
    bundle_camera_arrays,
    colorize_depth_sequence,
    crop_image_and_intrinsics,
    load_bundle,
    load_reconstructor,
    move_gaussians_to_device,
    parse_torch_dtype,
    prepare_image_and_intrinsics,
    read_rgb_image,
    render_sequence,
    repo_root,
    resolve_repo_relative_path,
    rgb_to_frames,
    save_video_exact_size,
    to_rel,
    write_json,
)


DEFAULT_CAMS = ["cam0", "cam1", "cam2"]
SCHEMA_VERSION = "neoverse_per_camera_gaussian_export_v1"
COORDINATE_FRAME = "neoverse_per_camera_local"


def _save_uint8_png(path: Path, frame: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(frame, dtype=np.uint8)).save(path)


def _ensure_scene_output_root(fused_root: Path, scene_id: str) -> Path:
    candidates = [fused_root, fused_root / scene_id]
    valid: list[Path] = []
    for candidate in candidates:
        if (candidate / "per_camera").exists():
            valid.append(candidate)
    if len(valid) == 1:
        return valid[0].resolve()
    if len(valid) > 1:
        for candidate in valid:
            if candidate.name == scene_id:
                return candidate.resolve()
        raise SystemExit(f"Ambiguous --fused_root for scene_id={scene_id}: {[p.as_posix() for p in valid]}")
    raise SystemExit(
        f"Could not resolve scene output root under --fused_root={fused_root}. "
        f"Expected either <fused_root>/per_camera or <fused_root>/{scene_id}/per_camera."
    )


def _cpu_clone(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    return value


def _serialize_raw_splats(splats_serialized: list[dict[str, Any]]) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []
    for item in splats_serialized:
        serialized.append({key: _cpu_clone(value) for key, value in item.items()})
    return serialized


def _raw_gaussian_count(splats_serialized: list[dict[str, Any]]) -> int:
    count = 0
    for item in splats_serialized:
        means = item.get("means")
        if isinstance(means, torch.Tensor):
            count += int(means.shape[0])
    return count


def _active_opacity_count(splats_serialized: list[dict[str, Any]], threshold: float) -> int:
    count = 0
    for item in splats_serialized:
        opacities = item.get("opacities")
        if isinstance(opacities, torch.Tensor):
            count += int((opacities.detach().cpu().float().reshape(-1) > float(threshold)).sum().item())
    return count


def _gaussian_to_cpu_dict(gaussian: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for name, value in gaussian.__dict__.items():
        payload[name] = _cpu_clone(value)
    return payload


def _should_render_gaussian_group(splats: Any, timestamp: int, bidirection: bool) -> bool:
    source_timestamp = int(getattr(splats, "timestamp", -1))
    forward_timestamp = getattr(splats, "forward_timestamp", None)
    backward_timestamp = getattr(splats, "backward_timestamp", None)

    if source_timestamp == -1 or source_timestamp == timestamp:
        return True
    if timestamp > source_timestamp and forward_timestamp is not None and timestamp < int(forward_timestamp):
        if bool(bidirection):
            return True
        return abs(timestamp - source_timestamp) <= abs(timestamp - int(forward_timestamp))
    if timestamp < source_timestamp and backward_timestamp is not None and timestamp > int(backward_timestamp):
        if bool(bidirection):
            return True
        return abs(timestamp - source_timestamp) < abs(timestamp - int(backward_timestamp))
    return False


def _render_prune_mask(
    splats: Any,
    opacity_prune_threshold: float,
    confidence_prune_threshold: float,
) -> torch.Tensor:
    mask = torch.ones_like(splats.opacities, dtype=torch.bool)
    if float(opacity_prune_threshold) >= 0:
        mask = mask & (splats.opacities >= float(opacity_prune_threshold))
    if float(confidence_prune_threshold) >= 0 and splats.confidences is not None:
        mask = mask & (splats.confidences >= float(confidence_prune_threshold))
    return mask


def _materialize_render_snapshot(
    gaussians: list[Any],
    timestamp: int,
    opacity_prune_threshold: float,
    confidence_prune_threshold: float,
    bidirection: bool,
) -> tuple[list[Any], dict[str, int]]:
    materialized: list[Any] = []
    num_source_groups = 0
    num_frame_raw = 0
    num_frame_active = 0
    num_materialized = 0

    with torch.no_grad():
        for splats in gaussians:
            if not _should_render_gaussian_group(splats, timestamp=timestamp, bidirection=bidirection):
                continue
            num_source_groups += 1
            num_frame_raw += int(splats.means.shape[0])
            mask = _render_prune_mask(
                splats=splats,
                opacity_prune_threshold=opacity_prune_threshold,
                confidence_prune_threshold=confidence_prune_threshold,
            )
            num_frame_active += int(mask.sum().item())
            transitioned = splats.transition(timestamp, mask=mask)
            num_materialized += int(transitioned.means.shape[0])
            materialized.append(transitioned)

    return materialized, {
        "num_render_source_groups": int(num_source_groups),
        "num_frame_raw_gaussians": int(num_frame_raw),
        "num_frame_active_gaussians": int(num_frame_active),
        "num_materialized_gaussians": int(num_materialized),
    }


def _prepare_source_crop(view_item: dict[str, Any], scene_dir: Path, render_width: int, render_height: int, resize_mode: str) -> np.ndarray:
    frame_path = scene_dir / str(view_item["frame_rel"])
    img = read_rgb_image(frame_path)
    K = np.asarray(view_item.get("camera_K") or view_item.get("prepared_camera_K"), dtype=np.float32)
    crop_box = view_item.get("crop_box_xyxy")
    if bool(view_item.get("crop_applied")) and crop_box:
        img, K = crop_image_and_intrinsics(img, K, crop_box)
    prepared_img, _ = prepare_image_and_intrinsics(img, K, render_width, render_height, resize_mode)
    return np.asarray(prepared_img, dtype=np.uint8)


def _resize_rgb(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    arr = np.asarray(frame, dtype=np.uint8)
    if arr.shape[1] == width and arr.shape[0] == height:
        return arr
    return np.asarray(Image.fromarray(arr).resize((width, height), resample=Image.BILINEAR), dtype=np.uint8)


def _resize_alpha(alpha: np.ndarray, width: int, height: int) -> np.ndarray:
    arr = np.asarray(alpha, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.shape[1] == width and arr.shape[0] == height:
        return arr
    pil = Image.fromarray(np.clip(arr * 255.0, 0.0, 255.0).astype(np.uint8))
    out = pil.resize((width, height), resample=Image.BILINEAR)
    return np.asarray(out, dtype=np.float32) / 255.0


def _make_full_frame_overlay(
    view_item: dict[str, Any],
    scene_dir: Path,
    render_rgb: np.ndarray,
    render_alpha: np.ndarray,
) -> np.ndarray:
    frame_path = scene_dir / str(view_item["frame_rel"])
    base = np.asarray(read_rgb_image(frame_path), dtype=np.uint8)
    overlay = base.astype(np.float32)

    crop_box = view_item.get("crop_box_xyxy")
    if bool(view_item.get("crop_applied")) and crop_box:
        x0, y0, x1, y1 = [int(v) for v in crop_box]
        crop_w = max(1, x1 - x0)
        crop_h = max(1, y1 - y0)
        render_rgb = _resize_rgb(render_rgb, crop_w, crop_h)
        render_alpha = _resize_alpha(render_alpha, crop_w, crop_h)
        target = overlay[y0:y1, x0:x1, :]
        alpha3 = np.clip(render_alpha[..., None], 0.0, 1.0)
        overlay[y0:y1, x0:x1, :] = target * (1.0 - alpha3) + render_rgb.astype(np.float32) * alpha3
    else:
        full_w, full_h = int(base.shape[1]), int(base.shape[0])
        render_rgb = _resize_rgb(render_rgb, full_w, full_h)
        render_alpha = _resize_alpha(render_alpha, full_w, full_h)
        alpha3 = np.clip(render_alpha[..., None], 0.0, 1.0)
        overlay = overlay * (1.0 - alpha3) + render_rgb.astype(np.float32) * alpha3
    return np.clip(overlay, 0.0, 255.0).astype(np.uint8)


def _make_side_by_side(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_u8 = np.asarray(left, dtype=np.uint8)
    right_u8 = np.asarray(right, dtype=np.uint8)
    if left_u8.shape[0] != right_u8.shape[0]:
        raise SystemExit(f"Height mismatch for side-by-side preview: {left_u8.shape} vs {right_u8.shape}")
    return np.concatenate([left_u8, right_u8], axis=1)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Export per-camera NeoVerse Gaussian parameters and per-timestamp render previews from existing bundles."
    )
    ap.add_argument("--scene_dir", required=True, type=str)
    ap.add_argument("--fused_root", required=True, type=str)
    ap.add_argument("--cams", default="cam0,cam1,cam2", type=str)
    ap.add_argument("--neoverse_repo", default="third_party/NeoVerse", type=str)
    ap.add_argument("--reconstructor_path", default="third_party/NeoVerse/models/NeoVerse/reconstructor.ckpt", type=str)
    ap.add_argument("--camera_source", default="rendered", choices=["rendered"])
    ap.add_argument("--device", default="cuda", type=str)
    ap.add_argument("--torch_dtype", default="float32", type=str)
    ap.add_argument("--fps", default=16, type=int)
    ap.add_argument("--opacity_threshold", default=0.05, type=float)
    ap.add_argument("--export_snapshot_pt", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    repo = repo_root()
    scene_dir = Path(str(args.scene_dir)).resolve()
    if not scene_dir.exists():
        raise SystemExit(f"Missing --scene_dir: {scene_dir}")

    fused_root = resolve_repo_relative_path(str(args.fused_root), repo)
    neoverse_repo = resolve_repo_relative_path(str(args.neoverse_repo), repo)
    reconstructor_path = resolve_repo_relative_path(str(args.reconstructor_path), repo)
    if not neoverse_repo.exists():
        raise SystemExit(f"Missing --neoverse_repo: {neoverse_repo}")
    if not reconstructor_path.exists():
        raise SystemExit(f"Missing --reconstructor_path: {reconstructor_path}")

    cams = [cam.strip() for cam in str(args.cams).split(",") if cam.strip()]
    if not cams:
        raise SystemExit("Empty --cams")

    scene_id = scene_dir.name
    scene_out_root = _ensure_scene_output_root(fused_root, scene_id)
    per_camera_root = scene_out_root / "per_camera"
    out_root = scene_out_root / "per_camera_gaussians"
    out_root.mkdir(parents=True, exist_ok=True)

    torch_dtype = parse_torch_dtype(str(args.torch_dtype))
    reconstructor = load_reconstructor(
        neoverse_repo=neoverse_repo,
        reconstructor_path=reconstructor_path,
        device=str(args.device),
        torch_dtype=torch_dtype,
    )
    rasterizer = reconstructor.gs_renderer.rasterizer
    rasterizer_opacity_prune_threshold = float(getattr(rasterizer, "opacity_prune_threshold", -1))
    rasterizer_confidence_prune_threshold = float(getattr(rasterizer, "confidence_prune_threshold", -1))
    rasterizer_bidirection = bool(getattr(rasterizer, "bidirection", True))

    for cam_id in cams:
        bundle_path = per_camera_root / cam_id / "reconstruction_bundle.pt"
        if not bundle_path.exists():
            raise SystemExit(f"Missing bundle for {cam_id}: {bundle_path}")

        cam_out = out_root / cam_id
        if cam_out.exists():
            if not bool(args.overwrite):
                raise SystemExit(f"Output already exists for {cam_id}: {cam_out}. Pass --overwrite to replace it.")
            shutil.rmtree(cam_out)
        cam_out.mkdir(parents=True, exist_ok=True)

        bundle = load_bundle(bundle_path)
        manifest = bundle.get("source_manifest")
        if not isinstance(manifest, dict):
            raise SystemExit(f"Bundle missing source_manifest dict: {bundle_path}")
        views = list(manifest.get("views") or [])
        if not views:
            raise SystemExit(f"Bundle missing source_manifest.views: {bundle_path}")
        if not isinstance(bundle.get("rendered_cam2world"), torch.Tensor):
            raise SystemExit(f"Bundle missing rendered_cam2world: {bundle_path}")
        if not isinstance(bundle.get("rendered_intrinsics"), torch.Tensor):
            raise SystemExit(f"Bundle missing rendered_intrinsics: {bundle_path}")
        if not isinstance(bundle.get("rendered_timestamps"), torch.Tensor):
            raise SystemExit(f"Bundle missing rendered_timestamps: {bundle_path}")

        source_label, camera_poses, camera_intrs, timestamps = bundle_camera_arrays(bundle, str(args.camera_source))
        if source_label != "rendered":
            raise SystemExit(f"Expected rendered camera arrays only, got: {source_label}")

        render_width = int(manifest.get("input_resolution", {}).get("width", 280))
        render_height = int(manifest.get("input_resolution", {}).get("height", 168))
        resize_mode = str(manifest.get("resize_mode") or "resize")

        raw_splats = list(bundle.get("splats_serialized") or [])
        if not raw_splats:
            raise SystemExit(f"Bundle has empty splats_serialized: {bundle_path}")
        raw_splats_cpu = _serialize_raw_splats(raw_splats)

        gaussians = build_gaussians(bundle, neoverse_repo)
        gaussians = move_gaussians_to_device(gaussians, str(args.device), torch_dtype)

        camera_c2w = camera_poses[0].detach().cpu().float()
        camera_w2c = torch.linalg.inv(camera_c2w)
        rgb, depth, alpha = render_sequence(
            rasterizer=rasterizer,
            gaussians=gaussians,
            render_viewmats=camera_w2c,
            render_Ks=camera_intrs[0],
            render_timestamps=timestamps[0],
            width=render_width,
            height=render_height,
            device=str(args.device),
            torch_dtype=torch_dtype,
        )

        rgb_frames = rgb_to_frames(rgb)
        depth_frames = colorize_depth_sequence(depth)
        alpha_frames = alpha_to_frames(alpha)
        alpha_np = alpha[0].detach().cpu().float().numpy()
        depth_np = depth[0].detach().cpu().float().numpy()
        if alpha_np.ndim == 4 and alpha_np.shape[-1] == 1:
            alpha_np = alpha_np[..., 0]
        if depth_np.ndim == 4 and depth_np.shape[-1] == 1:
            depth_np = depth_np[..., 0]

        gaussians_dir = cam_out / "gaussians"
        renders_rgb_dir = cam_out / "renders" / "rgb"
        renders_depth_dir = cam_out / "renders" / "depth"
        renders_alpha_dir = cam_out / "renders" / "alpha"
        preview_dir = cam_out / "preview"
        snapshots_dir = gaussians_dir / "snapshots"

        gaussians_dir.mkdir(parents=True, exist_ok=True)
        renders_rgb_dir.mkdir(parents=True, exist_ok=True)
        renders_depth_dir.mkdir(parents=True, exist_ok=True)
        renders_alpha_dir.mkdir(parents=True, exist_ok=True)
        preview_dir.mkdir(parents=True, exist_ok=True)
        if bool(args.export_snapshot_pt):
            snapshots_dir.mkdir(parents=True, exist_ok=True)

        raw_splats_path = gaussians_dir / "raw_splats.pt"
        torch.save(
            {
                "schema_version": SCHEMA_VERSION,
                "scene_id": scene_id,
                "cam_id": cam_id,
                "coordinate_frame": COORDINATE_FRAME,
                "source_bundle": bundle_path.as_posix(),
                "splats_serialized": raw_splats_cpu,
            },
            raw_splats_path,
        )

        source_crop_compare_frames: list[np.ndarray] = []
        full_frame_overlay_frames: list[np.ndarray] = []
        index_rows: list[dict[str, object]] = []

        total_raw_gaussians = _raw_gaussian_count(raw_splats_cpu)
        total_active_opacity = _active_opacity_count(raw_splats_cpu, float(args.opacity_threshold))

        for frame_idx, view_item in enumerate(views):
            scene_stem = str(view_item.get("scene_stem") or f"{frame_idx:06d}")
            logical_t_idx = int(view_item.get("logical_t_idx", frame_idx))
            timestamp = int(timestamps[0, frame_idx].item())
            rgb_path = renders_rgb_dir / f"{scene_stem}.png"
            depth_path = renders_depth_dir / f"{scene_stem}.npy"
            alpha_path = renders_alpha_dir / f"{scene_stem}.npy"
            materialized_splats, frame_stats = _materialize_render_snapshot(
                gaussians=gaussians,
                timestamp=timestamp,
                opacity_prune_threshold=rasterizer_opacity_prune_threshold,
                confidence_prune_threshold=rasterizer_confidence_prune_threshold,
                bidirection=rasterizer_bidirection,
            )

            _save_uint8_png(rgb_path, rgb_frames[frame_idx])
            np.save(depth_path, np.asarray(depth_np[frame_idx], dtype=np.float32))
            np.save(alpha_path, np.asarray(alpha_np[frame_idx], dtype=np.float32))

            source_crop = _prepare_source_crop(view_item, scene_dir, render_width, render_height, resize_mode)
            source_crop_compare_frames.append(_make_side_by_side(source_crop, rgb_frames[frame_idx]))
            full_frame_overlay_frames.append(
                _make_full_frame_overlay(
                    view_item=view_item,
                    scene_dir=scene_dir,
                    render_rgb=rgb_frames[frame_idx],
                    render_alpha=np.asarray(alpha_np[frame_idx], dtype=np.float32),
                )
            )

            if bool(args.export_snapshot_pt):
                snapshot_path = snapshots_dir / f"{scene_stem}.pt"
                torch.save(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "snapshot_export_mode": "materialized_render_snapshot",
                        "scene_id": scene_id,
                        "cam_id": cam_id,
                        "logical_t_idx": logical_t_idx,
                        "scene_stem": scene_stem,
                        "timestamp": timestamp,
                        "coordinate_frame": COORDINATE_FRAME,
                        "rasterizer_opacity_prune_threshold": rasterizer_opacity_prune_threshold,
                        "rasterizer_confidence_prune_threshold": rasterizer_confidence_prune_threshold,
                        "rasterizer_bidirection": rasterizer_bidirection,
                        "snapshot_stats": dict(frame_stats),
                        "materialized_render_snapshot": [_gaussian_to_cpu_dict(splat) for splat in materialized_splats],
                    },
                    snapshot_path,
                )

            index_rows.append(
                {
                    "cam_id": cam_id,
                    "logical_t_idx": logical_t_idx,
                    "scene_stem": scene_stem,
                    "timestamp": timestamp,
                    "raw_splats_rel": to_rel(raw_splats_path, cam_out),
                    "rgb_rel": to_rel(rgb_path, cam_out),
                    "depth_rel": to_rel(depth_path, cam_out),
                    "alpha_rel": to_rel(alpha_path, cam_out),
                    "num_render_source_groups": frame_stats["num_render_source_groups"],
                    "num_frame_raw_gaussians": frame_stats["num_frame_raw_gaussians"],
                    "num_frame_active_gaussians": frame_stats["num_frame_active_gaussians"],
                    "num_materialized_gaussians": frame_stats["num_materialized_gaussians"],
                    "camera_source": "rendered",
                    "coordinate_frame": COORDINATE_FRAME,
                }
            )

        index_path = gaussians_dir / "index.csv"
        with index_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "cam_id",
                    "logical_t_idx",
                    "scene_stem",
                    "timestamp",
                    "raw_splats_rel",
                    "rgb_rel",
                    "depth_rel",
                    "alpha_rel",
                    "num_render_source_groups",
                    "num_frame_raw_gaussians",
                    "num_frame_active_gaussians",
                    "num_materialized_gaussians",
                    "camera_source",
                    "coordinate_frame",
                ],
            )
            writer.writeheader()
            writer.writerows(index_rows)

        save_video_exact_size(rgb_frames, preview_dir / "rgb_gaussian.mp4", fps=int(args.fps))
        save_video_exact_size(depth_frames, preview_dir / "depth_gaussian.mp4", fps=int(args.fps))
        save_video_exact_size(alpha_frames, preview_dir / "alpha_gaussian.mp4", fps=int(args.fps))
        save_video_exact_size(source_crop_compare_frames, preview_dir / "source_crop_compare.mp4", fps=int(args.fps))
        save_video_exact_size(full_frame_overlay_frames, preview_dir / "full_frame_overlay.mp4", fps=int(args.fps))

        meta = {
            "schema_version": SCHEMA_VERSION,
            "scene_id": scene_id,
            "cam_id": cam_id,
            "source_bundle": bundle_path.as_posix(),
            "camera_source": "rendered",
            "coordinate_frame": COORDINATE_FRAME,
            "num_frames": int(len(index_rows)),
            "num_raw_splat_groups": int(len(raw_splats_cpu)),
            "snapshot_export_mode": "materialized_render_snapshot" if bool(args.export_snapshot_pt) else "disabled",
            "total_raw_gaussians": int(total_raw_gaussians),
            "total_active_opacity_gt_threshold": int(total_active_opacity),
            "opacity_count_threshold": float(args.opacity_threshold),
            "rasterizer_opacity_prune_threshold": rasterizer_opacity_prune_threshold,
            "rasterizer_confidence_prune_threshold": rasterizer_confidence_prune_threshold,
            "rasterizer_bidirection": rasterizer_bidirection,
            "render_width": render_width,
            "render_height": render_height,
            "uses_reconstructor_only": True,
            "is_multiview_fused": False,
            "is_world_aligned": False,
            "notes": [
                "Rendered from existing per-camera reconstruction_bundle.pt only.",
                "No multiview Gaussian fusion or rig/world alignment is applied in this export.",
                "raw_splats.pt stores the complete per-camera Gaussian bundle for the whole sequence.",
                "index.csv per-frame counts follow the rasterizer timestamp selection and prune mask before rendering.",
                "Snapshot .pt files, when enabled, store materialized Gaussian groups after timestamp transition and rasterizer pruning.",
                f"total_active_opacity_gt_threshold uses raw opacity tensor > {float(args.opacity_threshold):.4f}.",
            ],
            "outputs": {
                "raw_splats_pt": to_rel(raw_splats_path, cam_out),
                "index_csv": to_rel(index_path, cam_out),
                "preview_rgb": to_rel(preview_dir / 'rgb_gaussian.mp4', cam_out),
                "preview_depth": to_rel(preview_dir / 'depth_gaussian.mp4', cam_out),
                "preview_alpha": to_rel(preview_dir / 'alpha_gaussian.mp4', cam_out),
                "preview_source_crop_compare": to_rel(preview_dir / 'source_crop_compare.mp4', cam_out),
                "preview_full_frame_overlay": to_rel(preview_dir / 'full_frame_overlay.mp4', cam_out),
            },
        }
        write_json(cam_out / "gaussian_meta.json", meta)
        print(f"Wrote per-camera Gaussian export for {cam_id}: {cam_out}")


if __name__ == "__main__":
    main()
