from __future__ import annotations

import importlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imageio
import numpy as np
import torch
from torchvision.transforms import functional as F


DEFAULT_WIDTH = 280
DEFAULT_HEIGHT = 168
DEFAULT_FPS = 16


@dataclass
class RuntimeConfig:
    device: str
    torch_dtype: torch.dtype
    enable_vram_management: bool = False


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"Failed to read json: {path}\nError: {exc!r}")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def to_rel(path: Path, base: Path) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except Exception:
        return path.resolve().as_posix()


def resolve_repo_relative_path(path_value: str, base: Path | None = None) -> Path:
    root = repo_root() if base is None else base
    path = Path(str(path_value))
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def prepare_imports(neoverse_repo: Path) -> None:
    repo_path = str(neoverse_repo.resolve())
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)


def parse_torch_dtype(name: str) -> torch.dtype:
    value = str(name).strip().lower()
    if value in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if value in {"fp16", "float16"}:
        return torch.float16
    if value in {"fp32", "float32"}:
        return torch.float32
    raise SystemExit(f"Unsupported --torch_dtype: {name}")


def read_rgb_image(path: Path):
    from PIL import Image

    try:
        return Image.open(path).convert("RGB")
    except Exception as exc:
        raise SystemExit(f"Failed to read image: {path}; error={exc!r}")


def read_gray_image(path: Path) -> np.ndarray:
    from PIL import Image

    try:
        return np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
    except Exception as exc:
        raise SystemExit(f"Failed to read mask image: {path}; error={exc!r}")


def resolve_scene_mask_path(
    scene_dir: Path,
    cam_id: str,
    scene_stem: str,
    mask_source: str = "auto",
) -> tuple[Path | None, str | None]:
    source_text = str(mask_source).strip()
    source = source_text.lower()
    ordered_sources = ["masks_gt", "masks"] if source == "auto" else [source_text]
    for source_name in ordered_sources:
        candidate = scene_dir / "cams" / cam_id / source_name / f"{scene_stem}.png"
        if candidate.exists():
            return candidate, source_name
    return None, None


def compute_mask_bbox_xyxy(mask_u8: np.ndarray) -> list[int] | None:
    mask = np.asarray(mask_u8, dtype=np.uint8)
    if mask.ndim == 3:
        mask = mask[..., 0]
    if mask.ndim != 2:
        raise SystemExit(f"Unsupported mask shape: {mask.shape}")
    ys, xs = np.where(mask > 0)
    if xs.size == 0 or ys.size == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def expand_square_crop_box(
    bbox_xyxy: list[int],
    image_width: int,
    image_height: int,
    crop_padding: float,
) -> list[int]:
    if image_width <= 0 or image_height <= 0:
        raise SystemExit(f"Invalid image size for crop: width={image_width}, height={image_height}")
    x0, y0, x1, y1 = [int(v) for v in bbox_xyxy]
    if x1 <= x0 or y1 <= y0:
        raise SystemExit(f"Invalid bbox_xyxy for crop: {bbox_xyxy}")

    pad = max(float(crop_padding), 0.0)
    width = float(x1 - x0)
    height = float(y1 - y0)
    side = max(width, height)
    side = max(1.0, side * (1.0 + 2.0 * pad))
    side_int = max(1, int(np.ceil(side)))

    center_x = 0.5 * (float(x0) + float(x1))
    center_y = 0.5 * (float(y0) + float(y1))
    left = int(np.floor(center_x - 0.5 * side_int))
    top = int(np.floor(center_y - 0.5 * side_int))
    right = left + side_int
    bottom = top + side_int

    if left < 0:
        right -= left
        left = 0
    if top < 0:
        bottom -= top
        top = 0
    if right > image_width:
        left -= (right - image_width)
        right = image_width
    if bottom > image_height:
        top -= (bottom - image_height)
        bottom = image_height

    left = max(0, left)
    top = max(0, top)
    right = min(image_width, right)
    bottom = min(image_height, bottom)
    if right <= left or bottom <= top:
        return [0, 0, image_width, image_height]
    return [int(left), int(top), int(right), int(bottom)]


def crop_image_and_intrinsics(img, K: np.ndarray, crop_box_xyxy: list[int]):
    left, top, right, bottom = [int(v) for v in crop_box_xyxy]
    cropped = img.crop((left, top, right, bottom))
    K_out = np.asarray(K, dtype=np.float32).copy()
    K_out[0, 2] -= float(left)
    K_out[1, 2] -= float(top)
    return cropped, K_out


def resize_with_intrinsics(img, K: np.ndarray, target_width: int, target_height: int):
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


def center_crop_with_intrinsics(img, K: np.ndarray, target_width: int, target_height: int):
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


def uniform_indices(total: int, count: int) -> list[int]:
    if total <= 0:
        return []
    if count >= total:
        return list(range(total))
    idx = np.linspace(0, total - 1, num=count)
    rounded = np.round(idx).astype(int)
    uniq: list[int] = []
    seen: set[int] = set()
    for i in rounded.tolist():
        if i not in seen:
            seen.add(i)
            uniq.append(i)
    if len(uniq) < count:
        for i in range(total):
            if i not in seen:
                uniq.append(i)
                seen.add(i)
            if len(uniq) >= count:
                break
    return sorted(uniq[:count])


def prepare_image_and_intrinsics(img, K: np.ndarray, target_width: int, target_height: int, resize_mode: str):
    if resize_mode == "resize":
        return resize_with_intrinsics(img, K, target_width, target_height)
    if resize_mode == "center_crop":
        return center_crop_with_intrinsics(img, K, target_width, target_height)
    raise SystemExit(f"Unsupported --resize_mode: {resize_mode}")


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


def bundle_camera_arrays(bundle: dict[str, Any], camera_source: str) -> tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]:
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

    return source_label, pose.float(), intr.float(), ts.long()


def build_gaussians(bundle: dict[str, Any], neoverse_repo: Path) -> list[Any]:
    prepare_imports(neoverse_repo)
    from diffsynth.auxiliary_models.worldmirror.models.models.rasterization import Gaussians

    splats_serialized = list(bundle.get("splats_serialized") or [])
    if not splats_serialized:
        raise SystemExit("Bundle has empty splats_serialized")
    return [_make_gaussian(Gaussians, item) for item in splats_serialized]


def move_gaussians_to_device(gaussians: list[Any], device: str, torch_dtype: torch.dtype) -> list[Any]:
    moved = []
    for gaussian in gaussians:
        for name, value in list(gaussian.__dict__.items()):
            if isinstance(value, torch.Tensor):
                target_dtype = torch_dtype if value.is_floating_point() else value.dtype
                setattr(gaussian, name, value.to(device=device, dtype=target_dtype))
        moved.append(gaussian)
    return moved


def load_reconstructor(neoverse_repo: Path, reconstructor_path: Path, device: str, torch_dtype: torch.dtype):
    prepare_imports(neoverse_repo)
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


def build_views_from_scene(
    scene_dir: Path,
    cam_id: str,
    device: str,
    target_width: int,
    target_height: int,
    resize_mode: str,
    scene_mode: str = "general",
    num_frames: int = 81,
    input_variant: str = "full_frame",
    crop_padding: float = 0.25,
    crop_mask_source: str = "auto",
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]], dict[str, Any]]:
    rig_path = scene_dir / "calib" / "rig.json"
    if not rig_path.exists():
        raise SystemExit(f"Missing rig json: {rig_path}")
    rig = load_json(rig_path)
    cameras = rig.get("cameras", {})
    if cam_id not in cameras:
        raise SystemExit(f"Camera {cam_id} missing in rig.json. Available: {sorted(cameras.keys())}")

    cam_meta = cameras[cam_id]
    camera_K = np.asarray(cam_meta.get("K"), dtype=np.float32)
    camera_pose_c2w = np.asarray(cam_meta.get("T_node_from_cam"), dtype=np.float32)
    if camera_K.shape != (3, 3):
        raise SystemExit(f"Invalid K shape for {cam_id}: {camera_K.shape}")
    if camera_pose_c2w.shape != (4, 4):
        raise SystemExit(f"Invalid T_node_from_cam shape for {cam_id}: {camera_pose_c2w.shape}")

    frame_dir = scene_dir / "cams" / cam_id / "frames"
    if not frame_dir.exists():
        raise SystemExit(f"Missing frames dir: {frame_dir}")

    frame_files = sorted(list(frame_dir.glob("*.jpg")) + list(frame_dir.glob("*.png")))
    if not frame_files:
        raise SystemExit(f"No frames found in: {frame_dir}")

    num_frames_req = int(num_frames)
    if num_frames_req <= 0:
        selected_indices = list(range(len(frame_files)))
        frame_sampling_rule = "all_frames_requested_non_positive"
    else:
        selected_indices = uniform_indices(total=len(frame_files), count=num_frames_req)
        frame_sampling_rule = "uniform"
    frame_files = [frame_files[i] for i in selected_indices]

    scene_mode_norm = str(scene_mode).strip().lower()
    if scene_mode_norm not in {"general", "static"}:
        raise SystemExit(f"Unsupported scene_mode: {scene_mode}; expected one of ['general', 'static']")

    input_variant_norm = str(input_variant).strip().lower()
    if input_variant_norm not in {"full_frame", "object_crop"}:
        raise SystemExit(
            f"Unsupported input_variant: {input_variant}; expected one of ['full_frame', 'object_crop']"
        )

    crop_mask_source_text = str(crop_mask_source).strip()
    if not crop_mask_source_text:
        raise SystemExit("--crop_mask_source is empty")
    crop_mask_source_norm = crop_mask_source_text.lower()

    views_meta: list[dict[str, Any]] = []
    tensors = []
    timestamps = []
    camera_intrs = []
    camera_poses = []

    for view_idx, frame_path in enumerate(frame_files):
        img = read_rgb_image(frame_path)
        scene_stem = frame_path.stem
        frame_rel = frame_path.relative_to(scene_dir).as_posix()
        mask_path, resolved_mask_source = resolve_scene_mask_path(
            scene_dir=scene_dir,
            cam_id=cam_id,
            scene_stem=scene_stem,
            mask_source=crop_mask_source_text,
        )
        mask_rel = None if mask_path is None else mask_path.relative_to(scene_dir).as_posix()
        working_img = img
        working_K = np.asarray(camera_K, dtype=np.float32).copy()
        crop_applied = False
        crop_box_xyxy = None
        input_variant_used = "full_frame"
        if input_variant_norm == "object_crop" and mask_path is not None:
            mask_u8 = read_gray_image(mask_path)
            bbox_xyxy = compute_mask_bbox_xyxy(mask_u8)
            if bbox_xyxy is not None:
                crop_box_xyxy = expand_square_crop_box(
                    bbox_xyxy=bbox_xyxy,
                    image_width=img.size[0],
                    image_height=img.size[1],
                    crop_padding=float(crop_padding),
                )
                working_img, working_K = crop_image_and_intrinsics(
                    img=working_img,
                    K=working_K,
                    crop_box_xyxy=crop_box_xyxy,
                )
                crop_applied = True
                input_variant_used = "object_crop"
        prepared_img, prepared_K = prepare_image_and_intrinsics(
            working_img,
            working_K,
            target_width=target_width,
            target_height=target_height,
            resize_mode=resize_mode,
        )

        views_meta.append(
            {
                "view_idx": int(view_idx),
                "scene_stem": scene_stem,
                "logical_t_idx": int(view_idx),
                "cam_id": cam_id,
                "frame_rel": frame_rel,
                "mask_rel": mask_rel,
                "camera_K": camera_K.tolist(),
                "prepared_camera_K": prepared_K.tolist(),
                "camera_pose_c2w": camera_pose_c2w.tolist(),
                "scene_mode": scene_mode_norm,
                "input_variant": input_variant_used,
                "input_variant_requested": input_variant_norm,
                "crop_applied": bool(crop_applied),
                "crop_box_xyxy": crop_box_xyxy,
                "crop_padding": float(crop_padding),
                "crop_mask_source": resolved_mask_source,
            }
        )
        tensors.append(F.to_tensor(prepared_img)[None])
        timestamps.append(0 if scene_mode_norm == "static" else int(view_idx))
        camera_intrs.append(torch.tensor(prepared_K, dtype=torch.float32))
        camera_poses.append(torch.tensor(camera_pose_c2w, dtype=torch.float32))

    img_tensor = torch.stack(tensors, dim=1).to(device)
    S = img_tensor.shape[1]
    is_static_value = True if scene_mode_norm == "static" else False
    views = {
        "img": img_tensor,
        "is_target": torch.zeros((1, S), dtype=torch.bool, device=device),
        "is_static": torch.full((1, S), is_static_value, dtype=torch.bool, device=device),
        "timestamp": torch.tensor(timestamps, dtype=torch.int64, device=device).unsqueeze(0),
        "camera_intrs": torch.stack(camera_intrs, dim=0).to(device).unsqueeze(0),
        "camera_poses": torch.stack(camera_poses, dim=0).to(device).unsqueeze(0),
    }
    selected_frame_stems = [item["scene_stem"] for item in views_meta]
    sampling_meta = {
        "num_frames_requested": int(num_frames_req),
        "num_frames_selected": int(len(views_meta)),
        "selected_frame_stems": selected_frame_stems,
        "frame_sampling_rule": frame_sampling_rule,
        "input_variant_requested": input_variant_norm,
        "crop_padding": float(crop_padding),
        "crop_mask_source": crop_mask_source_text,
    }
    return views, views_meta, sampling_meta


def apply_rig_anchor_geometry_patch(neoverse_repo: Path) -> None:
    prepare_imports(neoverse_repo)
    raster_mod = importlib.import_module("diffsynth.auxiliary_models.worldmirror.models.models.rasterization")
    GaussianSplatRenderer = getattr(raster_mod, "GaussianSplatRenderer", None)
    if GaussianSplatRenderer is None:
        raise SystemExit("Missing GaussianSplatRenderer in NeoVerse rasterization module")
    if getattr(GaussianSplatRenderer, "_neoverse_rig_anchor_patched", False):
        return

    original_render = getattr(GaussianSplatRenderer, "render", None)
    if original_render is None or not callable(original_render):
        raise SystemExit("GaussianSplatRenderer.render is not patchable")

    def rig_anchor_render(
        self,
        gs_feats: torch.Tensor,
        images: torch.Tensor,
        predictions: dict[str, torch.Tensor],
        views: dict[str, torch.Tensor],
        context_predictions: dict[str, torch.Tensor],
        is_inference: bool = True,
    ) -> dict[str, torch.Tensor]:
        if self.training or not is_inference:
            return original_render(
                self,
                gs_feats,
                images,
                predictions,
                views,
                context_predictions,
                is_inference=is_inference,
            )

        B = images.shape[0]
        S = gs_feats.shape[1]
        H = gs_feats.shape[3]
        W = gs_feats.shape[4]

        gs_feats_reshape = gs_feats.reshape(B * S, gs_feats.shape[2], H, W)
        gs_params_static = self.gs_head(gs_feats_reshape)
        if self.is_4dgs:
            gs_params_dynamic = self.gs_head_dynamic(gs_feats_reshape)
            is_static = views["is_static"][:, :S].reshape(-1)
            gs_params = torch.where(is_static[:, None, None, None], gs_params_static, gs_params_dynamic)
        else:
            gs_params = gs_params_static

        splats = self.prepare_splats(
            views,
            predictions,
            images,
            gs_params,
            S,
            position_from="gsdepth+gtcamera",
        )

        predictions["splats"] = splats
        predictions["rendered_extrinsics"] = views["camera_poses"]
        predictions["rendered_intrinsics"] = views["camera_intrs"]
        predictions["rendered_timestamps"] = views["timestamp"]
        return predictions

    GaussianSplatRenderer.render = rig_anchor_render
    GaussianSplatRenderer._neoverse_rig_anchor_patched = True


def run_reconstruction(
    scene_dir: Path,
    cam_id: str,
    neoverse_repo: Path,
    reconstructor_path: Path,
    cfg: RuntimeConfig,
    target_width: int,
    target_height: int,
    resize_mode: str,
    scene_mode: str = "general",
    num_frames: int = 81,
    input_variant: str = "full_frame",
    crop_padding: float = 0.25,
    crop_mask_source: str = "auto",
) -> tuple[dict[str, Any], dict[str, torch.Tensor], list[dict[str, Any]]]:
    apply_rig_anchor_geometry_patch(neoverse_repo)
    models_mod = importlib.import_module("diffsynth.models")
    ModelManager = getattr(models_mod, "ModelManager")

    views, views_meta, sampling_meta = build_views_from_scene(
        scene_dir=scene_dir,
        cam_id=cam_id,
        device=cfg.device,
        target_width=target_width,
        target_height=target_height,
        resize_mode=resize_mode,
        scene_mode=scene_mode,
        num_frames=num_frames,
        input_variant=input_variant,
        crop_padding=crop_padding,
        crop_mask_source=crop_mask_source,
    )

    reconstructor_device = "cpu" if cfg.enable_vram_management else cfg.device
    model_manager = ModelManager(torch_dtype=cfg.torch_dtype, device=reconstructor_device)
    model_manager.load_model(str(reconstructor_path), device=reconstructor_device, torch_dtype=cfg.torch_dtype)
    reconstructor = model_manager.fetch_model("reconstructor")
    if cfg.enable_vram_management:
        reconstructor.to(cfg.device)

    use_amp = str(cfg.device).startswith("cuda")
    t0 = time.perf_counter()
    with torch.no_grad():
        if use_amp:
            with torch.amp.autocast("cuda", dtype=cfg.torch_dtype):
                predictions = reconstructor(views, cond_flags=[0, 1, 1], is_inference=True, use_motion=False)
        else:
            predictions = reconstructor(views, cond_flags=[0, 1, 1], is_inference=True, use_motion=False)
    dt = time.perf_counter() - t0

    if cfg.enable_vram_management:
        reconstructor.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {"runtime_sec": float(dt), "sampling_meta": dict(sampling_meta)}, predictions, views_meta


def serialize_tensor(x: Any) -> Any:
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.detach().cpu()
    return x


def serialize_splats(predictions: dict[str, Any]) -> list[dict[str, Any]]:
    splats_batch = predictions["splats"][0]
    serialized: list[dict[str, Any]] = []
    for gs in splats_batch:
        serialized.append(
            {
                "means": serialize_tensor(gs.means),
                "harmonics": serialize_tensor(gs.harmonics),
                "opacities": serialize_tensor(gs.opacities),
                "scales": serialize_tensor(gs.scales),
                "rotations": serialize_tensor(gs.rotations),
                "confidences": serialize_tensor(getattr(gs, "confidences", None)),
                "timestamp": int(getattr(gs, "timestamp", -1)),
                "life_span": serialize_tensor(getattr(gs, "life_span", 1.0)),
                "life_span_gamma": float(getattr(gs, "life_span_gamma", 0.0)),
                "forward_timestamp": getattr(gs, "forward_timestamp", None),
                "forward_vel": serialize_tensor(getattr(gs, "forward_vel", None)),
                "forward_scales": serialize_tensor(getattr(gs, "forward_scales", None)),
                "forward_rotations": serialize_tensor(getattr(gs, "forward_rotations", None)),
                "backward_timestamp": getattr(gs, "backward_timestamp", None),
                "backward_vel": serialize_tensor(getattr(gs, "backward_vel", None)),
                "backward_scales": serialize_tensor(getattr(gs, "backward_scales", None)),
                "backward_rotations": serialize_tensor(getattr(gs, "backward_rotations", None)),
            }
        )
    return serialized


def build_scene_glb(predictions: dict[str, Any], out_dir: Path, neoverse_repo: Path) -> Path:
    prepare_imports(neoverse_repo)
    app_mod = importlib.import_module("diffsynth.utils.app")
    build_scene_glb_fn = getattr(app_mod, "build_scene_glb")
    extract_point_cloud = getattr(app_mod, "extract_point_cloud")

    points, colors, frame_indices = extract_point_cloud(predictions)
    cam2world = predictions["rendered_extrinsics"][0].detach().cpu().numpy()
    scene = build_scene_glb_fn(points, colors, frame_indices, cam2world)
    glb_path = out_dir / "scene.glb"
    scene.export(str(glb_path))
    return glb_path


def load_bundle(bundle_path: Path) -> dict[str, Any]:
    bundle = torch.load(bundle_path, map_location="cpu")
    if not isinstance(bundle, dict):
        raise SystemExit(f"Invalid bundle format: {bundle_path}")
    return bundle


def resolve_bundle_dir(bundle_path: Path) -> Path:
    return bundle_path.resolve().parent


def resolve_bundle_manifest(bundle: dict[str, Any]) -> dict[str, Any]:
    manifest = bundle.get("source_manifest")
    if not isinstance(manifest, dict):
        raise SystemExit("Bundle missing source_manifest dict")
    return manifest


def render_sequence(
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


def tensor_to_uint8_frame(frame: np.ndarray) -> np.ndarray:
    frame = np.asarray(frame)
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0.0, 1.0)
        frame = (frame * 255.0).round().astype(np.uint8)
    if frame.ndim == 2:
        frame = np.repeat(frame[:, :, None], 3, axis=2)
    if frame.shape[-1] == 1:
        frame = np.repeat(frame, 3, axis=2)
    return frame


def rgb_to_frames(rgb_seq: torch.Tensor) -> list[np.ndarray]:
    rgb = rgb_seq.detach().float().cpu()
    while rgb.ndim > 4 and rgb.shape[0] == 1:
        rgb = rgb[0]
    if rgb.ndim == 4 and rgb.shape[-1] != 3:
        raise SystemExit(f"Unexpected rgb tensor shape: {tuple(rgb.shape)}")
    return [tensor_to_uint8_frame(frame.numpy()) for frame in rgb]


def colorize_depth_sequence(depth_seq: torch.Tensor) -> list[np.ndarray]:
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


def alpha_to_frames(alpha_seq: torch.Tensor) -> list[np.ndarray]:
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


def save_video_exact_size(frames: list[np.ndarray], save_path: Path, fps: int) -> None:
    writer = imageio.get_writer(str(save_path), fps=fps, quality=9, macro_block_size=1)
    for frame in frames:
        writer.append_data(np.asarray(frame))
    writer.close()


def load_reference_image_sequence(
    items: list[dict[str, Any]],
    scene_dir: Path,
    target_width: int,
    target_height: int,
    resize_mode: str,
) -> list[np.ndarray]:
    prepared_frames = []
    for item in items:
        frame_abs = scene_dir / str(item["frame_rel"])
        img = read_rgb_image(frame_abs)
        K = np.asarray(item.get("camera_K") or item.get("prepared_camera_K"), dtype=np.float32)
        prepared_img, _ = prepare_image_and_intrinsics(img, K, target_width, target_height, resize_mode)
        prepared_frames.append(tensor_to_uint8_frame(np.asarray(prepared_img)))
    return prepared_frames
