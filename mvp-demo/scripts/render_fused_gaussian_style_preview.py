from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import imageio
import numpy as np
from PIL import Image


DEFAULT_CAMS = ["cam0", "cam1", "cam2"]
DEFAULT_PRESENTATION_WIDTH = 640
DEFAULT_PRESENTATION_HEIGHT = 360


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"Failed to read json: {path}\nError: {exc!r}")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _read_rgb(path: Path) -> np.ndarray:
    try:
        return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    except Exception as exc:
        raise SystemExit(f"Failed to read image: {path}; error={exc!r}")


def _write_video(frames: list[np.ndarray], out_path: Path, fps: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        writer = imageio.get_writer(str(out_path), format="FFMPEG", fps=fps, quality=9, macro_block_size=1)
        for frame in frames:
            writer.append_data(np.asarray(frame, dtype=np.uint8))
        writer.close()
        return
    except Exception:
        pass

    try:
        import cv2  # type: ignore
    except Exception as exc:
        raise SystemExit(f"Failed to open MP4 writer for {out_path}; missing fallback dependency cv2. Error: {exc!r}")

    if not frames:
        raise SystemExit(f"Cannot write empty video: {out_path}")
    first = np.asarray(frames[0], dtype=np.uint8)
    height, width = first.shape[:2]
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))
    if not writer.isOpened():
        raise SystemExit(f"Failed to open cv2 VideoWriter for: {out_path}")
    try:
        for frame in frames:
            rgb = np.asarray(frame, dtype=np.uint8)
            if rgb.shape[:2] != (height, width):
                raise SystemExit(f"Inconsistent frame shape for {out_path}: expected {(height, width)}, got {rgb.shape[:2]}")
            writer.write(rgb[:, :, ::-1])
    finally:
        writer.release()


def _write_first_frame_png(frames: list[np.ndarray], out_path: Path) -> None:
    if not frames:
        return
    try:
        Image.fromarray(np.asarray(frames[0], dtype=np.uint8)).save(out_path)
    except Exception as exc:
        raise SystemExit(f"Failed to write first-frame PNG: {out_path}; error={exc!r}")


def _scene_root_has_expected_artifacts(scene_root: Path) -> bool:
    bg_candidates = [scene_root / "fused" / "background_world.npy", scene_root / "background_world.npy"]
    index_candidates = [scene_root / "points_by_timestamp" / "index.csv", scene_root / "fused" / "points_by_timestamp" / "index.csv"]
    return any(path.exists() for path in bg_candidates) and any(path.exists() for path in index_candidates)


def _resolve_run_scene_root(fused_root: Path, scene_id: str) -> Path:
    direct_candidates = [fused_root, fused_root / scene_id]
    for candidate in direct_candidates:
        if candidate.exists() and candidate.is_dir() and _scene_root_has_expected_artifacts(candidate):
            return candidate

    child_candidates = [child for child in fused_root.iterdir() if child.is_dir() and _scene_root_has_expected_artifacts(child)]
    if len(child_candidates) == 1:
        return child_candidates[0]
    if len(child_candidates) > 1:
        formatted = "\n".join(f"- {child.as_posix()}" for child in child_candidates)
        raise SystemExit(
            "Ambiguous --fused_root: multiple scene roots match. Please pass the exact scene root or run_root/scene_id. "
            f"Candidates:\n{formatted}"
        )

    raise SystemExit(
        "Could not locate a unique scene root under --fused_root. "
        f"Expected fused/background_world.npy and points_by_timestamp/index.csv for scene_id={scene_id!r}."
    )


def _resolve_background_path(run_scene_root: Path) -> Path:
    candidates = [
        run_scene_root / "fused" / "background_world.npy",
        run_scene_root / "background_world.npy",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise SystemExit(f"Missing background_world.npy under: {run_scene_root}")


def _resolve_points_dir(run_scene_root: Path) -> Path:
    candidates = [
        run_scene_root / "points_by_timestamp",
        run_scene_root / "fused" / "points_by_timestamp",
        run_scene_root / "points_export" / "points_by_timestamp",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise SystemExit(f"Missing points_by_timestamp directory under: {run_scene_root}")


def _resolve_rig_path(scene_dir: Path) -> Path:
    path = scene_dir / "calib" / "rig.json"
    if not path.exists():
        raise SystemExit(f"Missing rig json: {path}")
    return path


def _resolve_frame_times_path(scene_dir: Path) -> Path:
    path = scene_dir / "frame_times.csv"
    if not path.exists():
        raise SystemExit(f"Missing frame_times.csv: {path}")
    return path


def _row_first(row: dict[str, str], keys: list[str], required: bool = True) -> str | None:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip() != "":
            return str(value).strip()
    if required:
        raise SystemExit(f"Missing expected columns in CSV row. Tried keys={keys}; row={row}")
    return None


def _load_points_index(index_csv: Path) -> list[dict[str, str]]:
    with index_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"No rows found in: {index_csv}")
    rows.sort(key=lambda row: int(float(row.get("logical_t_idx", 0))))
    return rows


def _load_frame_times(frame_times_csv: Path, cams: list[str]) -> dict[str, dict[str, str]]:
    by_stem: dict[str, dict[str, str]] = {}
    with frame_times_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cam_id = str(row.get("cam_id") or "").strip()
            if cam_id not in cams:
                continue
            filename = _row_first(row, ["filename", "frame_rel", "rgb_path"])
            stem = Path(filename).stem
            by_stem.setdefault(stem, {})[cam_id] = filename
    return by_stem


def _load_rig(scene_dir: Path) -> dict[str, Any]:
    return _load_json(_resolve_rig_path(scene_dir))


def _native_camera_size(scene_dir: Path, cam_id: str) -> tuple[int, int]:
    frames_dir = scene_dir / "cams" / cam_id / "frames"
    if not frames_dir.exists():
        return 280, 168
    frame_files = sorted(list(frames_dir.glob("*.png")) + list(frames_dir.glob("*.jpg")) + list(frames_dir.glob("*.jpeg")))
    if not frame_files:
        return 280, 168
    rgb = _read_rgb(frame_files[0])
    return int(rgb.shape[1]), int(rgb.shape[0])


def _scale_intrinsics(K: np.ndarray, src_size: tuple[int, int], dst_size: tuple[int, int]) -> np.ndarray:
    src_w, src_h = src_size
    dst_w, dst_h = dst_size
    if src_w <= 0 or src_h <= 0:
        raise SystemExit(f"Invalid source size: {src_size}")
    if dst_w <= 0 or dst_h <= 0:
        raise SystemExit(f"Invalid target size: {dst_size}")
    scale_x = float(dst_w) / float(src_w)
    scale_y = float(dst_h) / float(src_h)
    out = np.asarray(K, dtype=np.float32).copy()
    out[0, 0] *= scale_x
    out[1, 1] *= scale_y
    out[0, 2] *= scale_x
    out[1, 2] *= scale_y
    return out


def _resize_rgb(rgb: np.ndarray, dst_size: tuple[int, int]) -> np.ndarray:
    dst_w, dst_h = dst_size
    if rgb.shape[1] == dst_w and rgb.shape[0] == dst_h:
        return np.asarray(rgb, dtype=np.uint8)
    return np.asarray(Image.fromarray(np.asarray(rgb, dtype=np.uint8)).resize((dst_w, dst_h), resample=Image.BILINEAR), dtype=np.uint8)


def _parse_rgb_triplet(text: str, flag_name: str) -> np.ndarray:
    parts = [p.strip() for p in str(text).split(",")]
    if len(parts) != 3:
        raise SystemExit(f"{flag_name} must have exactly 3 comma-separated integers in [0,255], got: {text!r}")
    try:
        vals = [int(p) for p in parts]
    except ValueError:
        raise SystemExit(f"{flag_name} must contain integers, got: {text!r}")
    if any(v < 0 or v > 255 for v in vals):
        raise SystemExit(f"{flag_name} values must be in [0,255], got: {text!r}")
    return np.asarray(vals, dtype=np.uint8)


def _read_mask_binary(path: Path, dst_size: tuple[int, int]) -> np.ndarray:
    try:
        mask = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
    except Exception as exc:
        raise SystemExit(f"Failed to read mask image: {path}; error={exc!r}")
    dst_w, dst_h = dst_size
    if mask.shape[1] != dst_w or mask.shape[0] != dst_h:
        mask = np.asarray(Image.fromarray(mask).resize((dst_w, dst_h), resample=Image.NEAREST), dtype=np.uint8)
    return mask > 0


def _resolve_mask_path(scene_dir: Path, cam_id: str, scene_stem: str) -> Path | None:
    candidates = [
        scene_dir / "cams" / cam_id / "masks_gt" / f"{scene_stem}.png",
        scene_dir / "cams" / cam_id / "masks" / f"{scene_stem}.png",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _sample_point_colors_from_cameras(
    points_world: np.ndarray,
    scene_stem: str,
    scene_dir: Path,
    frame_times_by_stem: dict[str, dict[str, str]],
    rig: dict[str, Any],
    cams: list[str],
    width: int,
    height: int,
    require_mask: bool,
    fallback_color: np.ndarray,
) -> dict[str, Any]:
    pts = np.asarray(points_world, dtype=np.float32)
    num_points = int(pts.shape[0])
    if num_points == 0:
        return {
            "median_colors": np.zeros((0, 3), dtype=np.uint8),
            "per_camera_colors": {cam_id: np.zeros((0, 3), dtype=np.uint8) for cam_id in cams},
            "per_camera_valid": {cam_id: np.zeros((0,), dtype=bool) for cam_id in cams},
            "valid_any": np.zeros((0,), dtype=bool),
            "valid_ratio": 0.0,
            "per_camera_valid_ratio": {cam_id: 0.0 for cam_id in cams},
        }

    sample_cube = np.full((num_points, len(cams), 3), np.nan, dtype=np.float32)
    for cam_idx, cam_id in enumerate(cams):
        frame_rel = frame_times_by_stem.get(scene_stem, {}).get(cam_id)
        if frame_rel is None:
            continue

        frame_path = scene_dir / frame_rel
        rgb_native = _read_rgb(frame_path)
        rgb_frame = _resize_rgb(rgb_native, (width, height))

        cam_meta = dict(rig.get("cameras", {}).get(cam_id) or {})
        if not cam_meta:
            continue
        cam_K_native = np.asarray(cam_meta.get("K"), dtype=np.float32)
        cam_c2w = np.asarray(cam_meta.get("T_node_from_cam"), dtype=np.float32)
        if cam_K_native.shape != (3, 3) or cam_c2w.shape != (4, 4):
            continue
        cam_K = _scale_intrinsics(cam_K_native, (int(rgb_native.shape[1]), int(rgb_native.shape[0])), (width, height))

        mask_valid: np.ndarray | None = None
        if require_mask:
            mask_path = _resolve_mask_path(scene_dir, cam_id, scene_stem)
            if mask_path is None:
                continue
            mask_valid = _read_mask_binary(mask_path, (width, height))

        w2c = np.linalg.inv(cam_c2w).astype(np.float32)
        R = w2c[:3, :3]
        t = w2c[:3, 3]
        pts_cam = pts @ R.T + t[None, :]
        z = pts_cam[:, 2]
        valid_depth = np.isfinite(z) & (z > 1e-6)
        if not np.any(valid_depth):
            continue

        idx_depth = np.nonzero(valid_depth)[0]
        pts_cam_d = pts_cam[valid_depth]
        z_d = z[valid_depth]
        u = np.round(cam_K[0, 0] * (pts_cam_d[:, 0] / z_d) + cam_K[0, 2]).astype(np.int32)
        v = np.round(cam_K[1, 1] * (pts_cam_d[:, 1] / z_d) + cam_K[1, 2]).astype(np.int32)
        valid_uv = (u >= 0) & (u < width) & (v >= 0) & (v < height)
        if not np.any(valid_uv):
            continue

        idx = idx_depth[valid_uv]
        uu = u[valid_uv]
        vv = v[valid_uv]

        if mask_valid is not None:
            in_fg = mask_valid[vv, uu]
            if not np.any(in_fg):
                continue
            idx = idx[in_fg]
            uu = uu[in_fg]
            vv = vv[in_fg]

        sampled_rgb = rgb_frame[vv, uu, :].astype(np.float32)
        sample_cube[idx, cam_idx, :] = sampled_rgb

    sampled_counts = np.sum(np.isfinite(sample_cube[..., 0]), axis=1)
    valid_any = sampled_counts > 0
    out = np.tile(np.asarray(fallback_color, dtype=np.uint8)[None, :], (num_points, 1))
    if np.any(valid_any):
        med = np.nanmedian(sample_cube[valid_any], axis=1)
        out[valid_any] = np.clip(np.round(med), 0.0, 255.0).astype(np.uint8)

    per_camera_colors: dict[str, np.ndarray] = {}
    per_camera_valid: dict[str, np.ndarray] = {}
    per_camera_valid_ratio: dict[str, float] = {}
    for cam_idx, cam_id in enumerate(cams):
        cam_valid = np.isfinite(sample_cube[:, cam_idx, 0])
        cam_colors = np.tile(np.asarray(fallback_color, dtype=np.uint8)[None, :], (num_points, 1))
        if np.any(cam_valid):
            cam_rgb = sample_cube[cam_valid, cam_idx, :]
            cam_colors[cam_valid] = np.clip(np.round(cam_rgb), 0.0, 255.0).astype(np.uint8)
        per_camera_colors[cam_id] = cam_colors
        per_camera_valid[cam_id] = cam_valid
        per_camera_valid_ratio[cam_id] = float(cam_valid.mean()) if num_points > 0 else 0.0

    valid_ratio = float(valid_any.mean()) if num_points > 0 else 0.0
    return {
        "median_colors": out,
        "per_camera_colors": per_camera_colors,
        "per_camera_valid": per_camera_valid,
        "valid_any": valid_any,
        "valid_ratio": valid_ratio,
        "per_camera_valid_ratio": per_camera_valid_ratio,
    }


def _look_at_c2w(camera_pos: np.ndarray, target: np.ndarray, up_hint: np.ndarray) -> np.ndarray:
    forward = np.asarray(target - camera_pos, dtype=np.float32)
    forward_norm = float(np.linalg.norm(forward))
    if forward_norm < 1e-6:
        forward = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    else:
        forward = forward / forward_norm
    up = np.asarray(up_hint, dtype=np.float32)
    right = np.cross(forward, up)
    if float(np.linalg.norm(right)) < 1e-6:
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        right = np.cross(forward, up)
    right = right / max(float(np.linalg.norm(right)), 1e-6)
    true_up = np.cross(right, forward)
    true_up = true_up / max(float(np.linalg.norm(true_up)), 1e-6)
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 0] = right
    c2w[:3, 1] = true_up
    c2w[:3, 2] = forward
    c2w[:3, 3] = camera_pos
    return c2w


def _project_points(points_world: np.ndarray, colors: np.ndarray, K: np.ndarray, c2w: np.ndarray, width: int, height: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if points_world.size == 0:
        empty_idx = np.zeros((0,), dtype=np.int64)
        empty_int = np.zeros((0,), dtype=np.int32)
        empty_float = np.zeros((0,), dtype=np.float32)
        empty_cols = np.zeros((0, 3), dtype=np.uint8)
        return empty_idx, empty_int, empty_int, empty_float, empty_cols
    w2c = np.linalg.inv(c2w).astype(np.float32)
    R = w2c[:3, :3]
    t = w2c[:3, 3]
    pts = np.asarray(points_world, dtype=np.float32)
    pts_cam = pts @ R.T + t[None, :]
    z = pts_cam[:, 2]
    valid = np.isfinite(z) & (z > 1e-6)
    if not np.any(valid):
        empty_idx = np.zeros((0,), dtype=np.int64)
        empty_int = np.zeros((0,), dtype=np.int32)
        empty_float = np.zeros((0,), dtype=np.float32)
        empty_cols = np.zeros((0, 3), dtype=np.uint8)
        return empty_idx, empty_int, empty_int, empty_float, empty_cols
    pts_cam = pts_cam[valid]
    z = z[valid]
    cols = np.asarray(colors, dtype=np.uint8)[valid]
    u = (K[0, 0] * (pts_cam[:, 0] / z) + K[0, 2]).round().astype(np.int32)
    v = (K[1, 1] * (pts_cam[:, 1] / z) + K[1, 2]).round().astype(np.int32)
    valid = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    if not np.any(valid):
        empty_idx = np.zeros((0,), dtype=np.int64)
        empty_int = np.zeros((0,), dtype=np.int32)
        empty_float = np.zeros((0,), dtype=np.float32)
        empty_cols = np.zeros((0, 3), dtype=np.uint8)
        return empty_idx, empty_int, empty_int, empty_float, empty_cols
    return (
        (v[valid] * width + u[valid]).astype(np.int64),
        u[valid].astype(np.int32),
        v[valid].astype(np.int32),
        z[valid].astype(np.float32),
        cols[valid],
    )


def _soft_layer_from_points(
    points_world: np.ndarray,
    colors: np.ndarray,
    K: np.ndarray,
    c2w: np.ndarray,
    width: int,
    height: int,
    sigma_px: float,
    point_alpha: float,
    base_rgb: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    flat_idx, u, v, z, cols = _project_points(points_world, colors, K, c2w, width, height)
    if flat_idx.size == 0:
        rgb = np.zeros((height, width, 3), dtype=np.uint8) if base_rgb is None else np.asarray(base_rgb, dtype=np.uint8)
        alpha = np.zeros((height, width), dtype=np.float32)
        return rgb, alpha

    total_px = width * height
    alpha_weights = np.full(flat_idx.shape[0], float(point_alpha), dtype=np.float32)
    alpha_sum = np.bincount(flat_idx, weights=alpha_weights, minlength=total_px).astype(np.float32).reshape(height, width)
    color_sum = np.zeros((height, width, 3), dtype=np.float32)
    for channel in range(3):
        color_sum[..., channel] = np.bincount(
            flat_idx,
            weights=alpha_weights * cols[:, channel].astype(np.float32),
            minlength=total_px,
        ).astype(np.float32).reshape(height, width)

    try:
        import cv2  # type: ignore
    except Exception as exc:
        raise SystemExit(f"cv2 is required for Gaussian-style soft point rendering. Error: {exc!r}")

    if sigma_px > 0:
        color_blur = cv2.GaussianBlur(color_sum, (0, 0), sigmaX=float(sigma_px), sigmaY=float(sigma_px), borderType=cv2.BORDER_REFLECT)
        alpha_blur = cv2.GaussianBlur(alpha_sum, (0, 0), sigmaX=float(sigma_px), sigmaY=float(sigma_px), borderType=cv2.BORDER_REFLECT)
    else:
        color_blur = color_sum
        alpha_blur = alpha_sum

    if base_rgb is not None:
        base_rgb_arr = np.asarray(base_rgb, dtype=np.uint8)
        if base_rgb_arr.shape[:2] != (height, width):
            raise SystemExit(f"base_rgb shape mismatch: {base_rgb_arr.shape} vs {(height, width)}")
    alpha_blur_raw = np.asarray(alpha_blur, dtype=np.float32)
    alpha_for_color = np.maximum(alpha_blur_raw[..., None], 1e-6)
    layer_rgb = np.clip(color_blur / alpha_for_color, 0.0, 255.0).astype(np.uint8)
    layer_alpha = np.clip(alpha_blur_raw, 0.0, 1.0)
    return layer_rgb, layer_alpha


def _compose_layer(base_rgb: np.ndarray, layer_rgb: np.ndarray, layer_alpha: np.ndarray, global_alpha: float) -> np.ndarray:
    base = np.asarray(base_rgb, dtype=np.uint8)
    rgb = np.asarray(layer_rgb, dtype=np.uint8)
    alpha = np.clip(np.asarray(layer_alpha, dtype=np.float32) * float(global_alpha), 0.0, 1.0)[..., None]
    out = base.astype(np.float32) * (1.0 - alpha) + rgb.astype(np.float32) * alpha
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def _orbit_camera(center: np.ndarray, radius: float, angle: float) -> np.ndarray:
    pos = np.asarray(center, dtype=np.float32) + np.array(
        [np.cos(angle) * radius, np.sin(angle) * radius, radius * 0.35],
        dtype=np.float32,
    )
    return _look_at_c2w(pos, np.asarray(center, dtype=np.float32), np.array([0.0, 0.0, 1.0], dtype=np.float32))


def _select_orbit_indices(total_frames: int, orbit_frames: int) -> list[int]:
    if total_frames <= 0:
        return []
    if orbit_frames <= 0:
        raise SystemExit(f"--orbit_frames must be positive, got: {orbit_frames}")
    if orbit_frames == total_frames:
        return list(range(total_frames))
    samples = np.linspace(0, total_frames - 1, num=orbit_frames)
    indices = np.clip(np.round(samples).astype(np.int32), 0, total_frames - 1)
    return indices.tolist()


def _scene_center_and_radius(*point_sets: np.ndarray) -> tuple[np.ndarray, float]:
    valid_sets = [np.asarray(points, dtype=np.float32) for points in point_sets if points is not None and np.asarray(points).size]
    if not valid_sets:
        return np.zeros(3, dtype=np.float32), 1.0
    combined = np.concatenate(valid_sets, axis=0)
    min_corner = combined.min(axis=0)
    max_corner = combined.max(axis=0)
    center = 0.5 * (min_corner + max_corner)
    extent = float(np.max(max_corner - min_corner))
    return center.astype(np.float32), max(extent, 1.0)


def _load_points_series(points_dir: Path) -> list[dict[str, Any]]:
    index_csv = points_dir / "index.csv"
    rows = _load_points_index(index_csv)
    for row in rows:
        rel = _row_first(row, ["points_rel", "points_path", "npy_rel", "rel_path", "points"])
        row["_points_rel"] = rel
    return rows


def _find_point_file(points_dir: Path, rel_path: str) -> Path:
    candidates = [
        points_dir / rel_path,
        points_dir.parent / rel_path,
        points_dir.parent.parent / rel_path,
        points_dir / Path(rel_path).name,
    ]
    for path in candidates:
        if path.exists():
            return path
    raise SystemExit(f"Could not resolve point file {rel_path!r} under {points_dir}")


def _load_point_cloud(path: Path) -> np.ndarray:
    pts = np.asarray(np.load(str(path)), dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise SystemExit(f"Expected Nx3 point cloud in {path}, got {pts.shape}")
    return pts


def _select_reference_camera(scene_dir: Path, rig: dict[str, Any], cams: list[str]) -> tuple[str, int, int, np.ndarray]:
    cam_id = cams[0]
    native_w, native_h = _native_camera_size(scene_dir, cam_id)
    cam_meta = dict(rig.get("cameras", {}).get(cam_id) or {})
    if not cam_meta:
        raise SystemExit(f"Camera {cam_id} missing in rig.json")
    K = np.asarray(cam_meta.get("K"), dtype=np.float32)
    if K.shape != (3, 3):
        raise SystemExit(f"Invalid K for {cam_id}: {K.shape}")
    return cam_id, native_w, native_h, K


def _render_camera_overlay_frame(
    rgb_frame: np.ndarray,
    background_points: np.ndarray,
    dynamic_points: np.ndarray,
    dynamic_colors: np.ndarray,
    cam_K: np.ndarray,
    cam_c2w: np.ndarray,
    width: int,
    height: int,
    sigma_bg: float,
    sigma_dyn: float,
    alpha_bg: float,
    alpha_dyn: float,
) -> np.ndarray:
    base = _resize_rgb(rgb_frame, (width, height))
    bg_colors = np.full((background_points.shape[0], 3), np.asarray([170, 170, 170], dtype=np.uint8), dtype=np.uint8)
    dyn_colors = np.asarray(dynamic_colors, dtype=np.uint8)

    bg_rgb, bg_alpha_map = _soft_layer_from_points(
        points_world=background_points,
        colors=bg_colors,
        K=cam_K,
        c2w=cam_c2w,
        width=width,
        height=height,
        sigma_px=sigma_bg,
        point_alpha=alpha_bg,
        base_rgb=base,
    )
    dyn_rgb, dyn_alpha_map = _soft_layer_from_points(
        points_world=dynamic_points,
        colors=dyn_colors,
        K=cam_K,
        c2w=cam_c2w,
        width=width,
        height=height,
        sigma_px=sigma_dyn,
        point_alpha=alpha_dyn,
        base_rgb=base,
    )
    composed = _compose_layer(base, bg_rgb, bg_alpha_map, global_alpha=1.0)
    composed = _compose_layer(composed, dyn_rgb, dyn_alpha_map, global_alpha=1.0)
    return composed


def _orbit_background_points(background_points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    bg_colors = np.full((background_points.shape[0], 3), np.asarray([170, 170, 170], dtype=np.uint8), dtype=np.uint8)
    return background_points, bg_colors


def _project_scene_points_for_orbit(
    background_points: np.ndarray,
    dynamic_points: np.ndarray,
    dynamic_colors: np.ndarray,
    cam_K: np.ndarray,
    cam_c2w: np.ndarray,
    width: int,
    height: int,
    sigma_bg: float,
    sigma_dyn: float,
    alpha_bg: float,
    alpha_dyn: float,
) -> np.ndarray:
    bg_points, bg_colors = _orbit_background_points(background_points)
    base = np.zeros((height, width, 3), dtype=np.uint8)
    bg_rgb, bg_alpha = _soft_layer_from_points(
        points_world=bg_points,
        colors=bg_colors,
        K=cam_K,
        c2w=cam_c2w,
        width=width,
        height=height,
        sigma_px=sigma_bg,
        point_alpha=alpha_bg,
        base_rgb=base,
    )
    dyn_colors = np.asarray(dynamic_colors, dtype=np.uint8)
    dyn_rgb, dyn_alpha = _soft_layer_from_points(
        points_world=dynamic_points,
        colors=dyn_colors,
        K=cam_K,
        c2w=cam_c2w,
        width=width,
        height=height,
        sigma_px=sigma_dyn,
        point_alpha=alpha_dyn,
        base_rgb=base,
    )
    composed = _compose_layer(base, bg_rgb, bg_alpha, global_alpha=1.0)
    composed = _compose_layer(composed, dyn_rgb, dyn_alpha, global_alpha=1.0)
    return composed


def main() -> None:
    ap = argparse.ArgumentParser(description="Render fused points_by_timestamp with Gaussian-style soft point videos.")
    ap.add_argument("--scene_dir", required=True, type=str)
    ap.add_argument("--fused_root", "--fused_dir", required=True, type=str)
    ap.add_argument("--cams", default="cam0,cam1,cam2", type=str)
    ap.add_argument("--fps", default=16, type=int)
    ap.add_argument("--orbit_frames", default=81, type=int)
    ap.add_argument("--orbit_radius_scale", default=1.8, type=float)
    ap.add_argument("--max_dynamic_points", default=120000, type=int)
    ap.add_argument("--max_background_points", default=80000, type=int)
    ap.add_argument("--dynamic_sigma_px", default=3.5, type=float)
    ap.add_argument("--background_sigma_px", default=1.8, type=float)
    ap.add_argument("--dynamic_alpha", default=0.95, type=float)
    ap.add_argument("--background_alpha", default=0.28, type=float)
    ap.add_argument("--dynamic_color_mode", choices=["fixed", "sample_rgb"], default="sample_rgb")
    ap.add_argument("--sample_rgb_fusion", choices=["median", "view"], default="median")
    ap.add_argument("--sample_rgb_require_mask", action="store_true")
    ap.add_argument("--sample_rgb_fallback_color", default="255,120,60", type=str)
    ap.add_argument("--output_subdir", default="", type=str)
    ap.add_argument("--width", default=None, type=int)
    ap.add_argument("--height", default=None, type=int)
    args = ap.parse_args()

    width_provided = args.width is not None
    height_provided = args.height is not None
    if width_provided != height_provided:
        raise SystemExit("--width and --height must be provided together")

    target_width = int(args.width) if args.width is not None else DEFAULT_PRESENTATION_WIDTH
    target_height = int(args.height) if args.height is not None else DEFAULT_PRESENTATION_HEIGHT
    sample_rgb_fallback_color = _parse_rgb_triplet(args.sample_rgb_fallback_color, "--sample_rgb_fallback_color")

    repo = _repo_root()
    scene_dir = Path(str(args.scene_dir)).resolve()
    if not scene_dir.exists():
        raise SystemExit(f"Missing --scene_dir: {scene_dir}")

    fused_root = Path(str(args.fused_root))
    if not fused_root.is_absolute():
        fused_root = repo / fused_root
    fused_root = fused_root.resolve()

    scene_id = scene_dir.name
    run_scene_root = _resolve_run_scene_root(fused_root, scene_id=scene_id)
    background_path = _resolve_background_path(run_scene_root)
    points_dir = _resolve_points_dir(run_scene_root)
    frame_times_csv = _resolve_frame_times_path(scene_dir)
    rig = _load_rig(scene_dir)

    cams = [c.strip() for c in str(args.cams).split(",") if c.strip()]
    if cams != DEFAULT_CAMS:
        raise SystemExit(f"This script currently expects cams={DEFAULT_CAMS}. Got: {cams}")

    points_rows = _load_points_series(points_dir)
    frame_times_by_stem = _load_frame_times(frame_times_csv, cams)

    background_points = np.asarray(np.load(str(background_path)), dtype=np.float32)
    background_points = background_points.reshape(-1, 3) if background_points.size else np.zeros((0, 3), dtype=np.float32)
    background_sample_seed = 7
    if background_points.shape[0] > int(args.max_background_points):
        rng = np.random.default_rng(background_sample_seed)
        idx = rng.choice(background_points.shape[0], size=int(args.max_background_points), replace=False)
        idx.sort()
        background_points = background_points[idx]

    output_subdir = str(args.output_subdir).strip().replace("\\", "/")
    if output_subdir:
        output_subdir_rel = output_subdir.lstrip("/")
        if output_subdir_rel.startswith("preview/"):
            output_root = run_scene_root / Path(output_subdir_rel)
        else:
            output_root = run_scene_root / "preview" / "gaussian_style_compare" / Path(output_subdir_rel)
    else:
        output_root = run_scene_root / "preview" / "gaussian_style"
    output_root.mkdir(parents=True, exist_ok=True)

    cam0_id, native_w, native_h, cam0_K = _select_reference_camera(scene_dir, rig, cams)
    orbit_width = target_width
    orbit_height = target_height
    overlay_width = target_width
    overlay_height = target_height
    cam0_K_orbit = _scale_intrinsics(cam0_K, (native_w, native_h), (orbit_width, orbit_height))
    overlay_shape_note = f"{overlay_width}x{overlay_height}"

    scene_stems = [str(row["scene_stem"]) for row in points_rows]
    orbit_source_indices = _select_orbit_indices(len(points_rows), int(args.orbit_frames))
    orbit_frames: list[np.ndarray] = []
    overlay_frames: dict[str, list[np.ndarray]] = {cam: [] for cam in cams}
    sampled_dynamic_counts: list[int] = []
    sampled_background_counts: list[int] = []
    sample_rgb_valid_ratios: list[float] = []
    sample_rgb_per_camera_valid_ratios: dict[str, list[float]] = {cam: [] for cam in cams}
    per_frame_dynamic_points: list[np.ndarray] = []
    per_frame_dynamic_colors_median: list[np.ndarray] = []
    per_frame_dynamic_colors_per_cam: list[dict[str, np.ndarray]] = []

    for idx, row in enumerate(points_rows):
        scene_stem = str(row["scene_stem"])
        logical_t_idx = int(float(row.get("logical_t_idx", idx)))
        points_rel = row["_points_rel"]
        point_path = _find_point_file(points_dir, points_rel)
        dynamic_points = _load_point_cloud(point_path)
        if dynamic_points.shape[0] > int(args.max_dynamic_points):
            rng = np.random.default_rng(logical_t_idx * 7919 + 17)
            keep = rng.choice(dynamic_points.shape[0], size=int(args.max_dynamic_points), replace=False)
            keep.sort()
            dynamic_points = dynamic_points[keep]

        if args.dynamic_color_mode == "fixed":
            dynamic_colors_median = np.full((dynamic_points.shape[0], 3), sample_rgb_fallback_color, dtype=np.uint8)
            dynamic_colors_per_cam = {cam_id: dynamic_colors_median for cam_id in cams}
            sample_rgb_valid_ratio = 1.0 if dynamic_points.shape[0] > 0 else 0.0
            per_cam_valid_ratio = {cam_id: sample_rgb_valid_ratio for cam_id in cams}
        else:
            sample_info = _sample_point_colors_from_cameras(
                points_world=dynamic_points,
                scene_stem=scene_stem,
                scene_dir=scene_dir,
                frame_times_by_stem=frame_times_by_stem,
                rig=rig,
                cams=cams,
                width=int(overlay_width),
                height=int(overlay_height),
                require_mask=bool(args.sample_rgb_require_mask),
                fallback_color=sample_rgb_fallback_color,
            )
            dynamic_colors_median = np.asarray(sample_info["median_colors"], dtype=np.uint8)
            sample_rgb_valid_ratio = float(sample_info["valid_ratio"])
            per_cam_valid_ratio = dict(sample_info["per_camera_valid_ratio"])
            dynamic_colors_per_cam = {}
            for cam_id in cams:
                cam_colors = np.asarray(sample_info["per_camera_colors"][cam_id], dtype=np.uint8)
                cam_valid = np.asarray(sample_info["per_camera_valid"][cam_id], dtype=bool)
                dynamic_colors_per_cam[cam_id] = np.where(cam_valid[:, None], cam_colors, dynamic_colors_median)

        per_frame_dynamic_points.append(dynamic_points)
        per_frame_dynamic_colors_median.append(dynamic_colors_median)
        per_frame_dynamic_colors_per_cam.append(dynamic_colors_per_cam)
        sample_rgb_valid_ratios.append(float(sample_rgb_valid_ratio))
        for cam_id in cams:
            sample_rgb_per_camera_valid_ratios[cam_id].append(float(per_cam_valid_ratio[cam_id]))

        sampled_dynamic_counts.append(int(dynamic_points.shape[0]))
        sampled_background_counts.append(int(background_points.shape[0]))

        for cam_id in cams:
            frame_rel = frame_times_by_stem.get(scene_stem, {}).get(cam_id)
            if frame_rel is None:
                raise SystemExit(f"Missing synchronized frame for cam={cam_id}, scene_stem={scene_stem}")
            frame_path = scene_dir / frame_rel
            rgb_frame = _read_rgb(frame_path)

            cam_meta = dict(rig.get("cameras", {}).get(cam_id) or {})
            if not cam_meta:
                raise SystemExit(f"Camera {cam_id} missing in rig.json")
            cam_K = np.asarray(cam_meta.get("K"), dtype=np.float32)
            cam_c2w = np.asarray(cam_meta.get("T_node_from_cam"), dtype=np.float32)
            if cam_K.shape != (3, 3):
                raise SystemExit(f"Invalid K for {cam_id}: {cam_K.shape}")
            if cam_c2w.shape != (4, 4):
                raise SystemExit(f"Invalid T_node_from_cam for {cam_id}: {cam_c2w.shape}")

            target_w, target_h = int(overlay_width), int(overlay_height)
            overlay_rgb = _resize_rgb(rgb_frame, (target_w, target_h))
            cam_K_overlay = _scale_intrinsics(cam_K, (int(rgb_frame.shape[1]), int(rgb_frame.shape[0])), (target_w, target_h))

            if args.dynamic_color_mode == "fixed":
                dynamic_colors_overlay = dynamic_colors_median
            elif args.sample_rgb_fusion == "median":
                dynamic_colors_overlay = dynamic_colors_median
            else:
                dynamic_colors_overlay = dynamic_colors_per_cam[cam_id]

            overlay_frame = _render_camera_overlay_frame(
                rgb_frame=overlay_rgb,
                background_points=background_points,
                dynamic_points=dynamic_points,
                dynamic_colors=dynamic_colors_overlay,
                cam_K=cam_K_overlay,
                cam_c2w=cam_c2w,
                width=target_w,
                height=target_h,
                sigma_bg=float(args.background_sigma_px),
                sigma_dyn=float(args.dynamic_sigma_px),
                alpha_bg=float(args.background_alpha),
                alpha_dyn=float(args.dynamic_alpha),
            )
            overlay_frames[cam_id].append(overlay_frame)

    orbit_map_logical_t_idx: list[int] = []
    orbit_map_scene_stem: list[str] = []
    for orbit_idx, source_idx in enumerate(orbit_source_indices):
        row = points_rows[source_idx]
        scene_stem = str(row["scene_stem"])
        logical_t_idx = int(float(row.get("logical_t_idx", source_idx)))
        dynamic_points = per_frame_dynamic_points[source_idx]
        dynamic_colors = per_frame_dynamic_colors_median[source_idx]

        orbit_map_logical_t_idx.append(logical_t_idx)
        orbit_map_scene_stem.append(scene_stem)
        orbit_angle = 2.0 * np.pi * float(orbit_idx) / max(int(args.orbit_frames), 1)
        orbit_center, orbit_extent = _scene_center_and_radius(background_points, dynamic_points)
        orbit_radius = max(float(orbit_extent) * float(args.orbit_radius_scale), 1.0)
        orbit_c2w = _orbit_camera(center=orbit_center, radius=orbit_radius, angle=orbit_angle)
        orbit_frame = _project_scene_points_for_orbit(
            background_points=background_points,
            dynamic_points=dynamic_points,
            dynamic_colors=dynamic_colors,
            cam_K=cam0_K_orbit,
            cam_c2w=orbit_c2w,
            width=orbit_width,
            height=orbit_height,
            sigma_bg=float(args.background_sigma_px),
            sigma_dyn=float(args.dynamic_sigma_px),
            alpha_bg=float(args.background_alpha),
            alpha_dyn=float(args.dynamic_alpha),
        )
        orbit_frames.append(orbit_frame)

    orbit_video_path = output_root / "orbit_gaussian.mp4"
    cam_video_paths: dict[str, str] = {}
    _write_video(orbit_frames, orbit_video_path, fps=int(args.fps))
    _write_first_frame_png(orbit_frames, output_root / "orbit_gaussian_first.png")
    for cam_id, frames in overlay_frames.items():
        video_path = output_root / f"{cam_id}_overlay_gaussian.mp4"
        _write_video(frames, video_path, fps=int(args.fps))
        _write_first_frame_png(frames, output_root / f"{cam_id}_overlay_gaussian_first.png")
        cam_video_paths[cam_id] = video_path.as_posix()

    meta = {
        "schema_version": "gaussian_style_fused_preview_v1",
        "scene_id": scene_dir.name,
        "scene_dir": scene_dir.as_posix(),
        "fused_root": fused_root.as_posix(),
        "run_scene_root": run_scene_root.as_posix(),
        "points_by_timestamp_root": points_dir.as_posix(),
        "num_frames": int(len(points_rows)),
        "num_rendered_orbit_frames": int(len(orbit_frames)),
        "num_rendered_overlay_frames": int(len(overlay_frames[cams[0]])),
        "max_dynamic_points": int(args.max_dynamic_points),
        "max_background_points": int(args.max_background_points),
        "dynamic_sigma_px": float(args.dynamic_sigma_px),
        "background_sigma_px": float(args.background_sigma_px),
        "dynamic_alpha": float(args.dynamic_alpha),
        "background_alpha": float(args.background_alpha),
        "dynamic_color_mode": str(args.dynamic_color_mode),
        "sample_rgb_fusion": str(args.sample_rgb_fusion),
        "sample_rgb_require_mask": bool(args.sample_rgb_require_mask),
        "sample_rgb_fallback_color": [int(x) for x in sample_rgb_fallback_color.tolist()],
        "sample_rgb_valid_ratio_mean": float(np.mean(sample_rgb_valid_ratios)) if sample_rgb_valid_ratios else 0.0,
        "sample_rgb_valid_ratio_min": float(np.min(sample_rgb_valid_ratios)) if sample_rgb_valid_ratios else 0.0,
        "sample_rgb_valid_ratio_max": float(np.max(sample_rgb_valid_ratios)) if sample_rgb_valid_ratios else 0.0,
        "sample_rgb_per_camera_valid_ratio_mean": {
            cam_id: float(np.mean(sample_rgb_per_camera_valid_ratios[cam_id])) if sample_rgb_per_camera_valid_ratios[cam_id] else 0.0
            for cam_id in cams
        },
        "alpha_color_normalization": "raw_alpha_for_color_clipped_alpha_for_compositing",
        "output_subdir": output_subdir,
        "orbit_frames": int(args.orbit_frames),
        "orbit_radius_scale": float(args.orbit_radius_scale),
        "orbit_frame_to_logical_t_idx": orbit_map_logical_t_idx,
        "orbit_frame_to_scene_stem": orbit_map_scene_stem,
        "overlay_size": None if overlay_width is None or overlay_height is None else {"width": int(overlay_width), "height": int(overlay_height)},
        "orbit_size": {"width": int(orbit_width), "height": int(orbit_height)},
        "reference_camera": cam0_id,
        "reference_camera_native_size": {"width": int(native_w), "height": int(native_h)},
        "frame_times_csv": frame_times_csv.as_posix(),
        "video_paths": {
            "orbit_gaussian": orbit_video_path.as_posix(),
            "cam0_overlay_gaussian": cam_video_paths["cam0"],
            "cam1_overlay_gaussian": cam_video_paths["cam1"],
            "cam2_overlay_gaussian": cam_video_paths["cam2"],
        },
        "notes": [
            "This is a Gaussian-style soft-point visualization of fused point clouds.",
            "It is not a true fused Gaussian reconstruction or a new geometry model.",
            "Each frame renders only the current timestamp's dynamic points plus static background points.",
            "Default parameters prioritize presentation readability over metric-faithful point size.",
        ],
        "debug": {
            "scene_stems": scene_stems,
            "sampled_dynamic_counts": sampled_dynamic_counts,
            "sampled_background_counts": sampled_background_counts,
            "output_dir": output_root.as_posix(),
            "overlay_mode": overlay_shape_note,
        },
    }
    meta_path = output_root / "gaussian_style_meta.json"
    _write_json(meta_path, meta)

    print(f"Wrote: {orbit_video_path}")
    print(f"Wrote: {cam_video_paths['cam0']}")
    print(f"Wrote: {cam_video_paths['cam1']}")
    print(f"Wrote: {cam_video_paths['cam2']}")
    print(f"Wrote: {meta_path}")


if __name__ == "__main__":
    main()