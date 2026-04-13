from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:  # pragma: no cover
        raise SystemExit(f"Failed to read json: {path}\nError: {e!r}")


def _to_2d_depth(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        if arr.shape[-1] == 1:
            return arr[..., 0]
        if arr.shape[0] == 1:
            return arr[0]
    raise ValueError(f"Unsupported depth shape: {arr.shape}")


def _read_depth_npy(path: Path) -> np.ndarray:
    try:
        depth = np.load(str(path))
    except Exception as e:  # pragma: no cover
        raise SystemExit(f"Failed to load depth npy: {path}\nError: {e!r}")
    return np.asarray(_to_2d_depth(np.asarray(depth)), dtype=np.float32)


def _read_mask_u8(path: Path) -> np.ndarray:
    try:
        import cv2  # type: ignore
    except Exception as e:  # pragma: no cover
        raise SystemExit(f"Missing dependency: cv2 (opencv-python). Error: {e!r}")

    path_str = str(path)
    if not path_str.isascii():
        try:
            data = np.fromfile(path_str, dtype=np.uint8)
            mask = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
        except Exception:
            mask = None
        if mask is not None:
            return np.asarray(mask, dtype=np.uint8)

    mask = cv2.imread(path_str, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        try:
            data = np.fromfile(path_str, dtype=np.uint8)
            mask = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
        except Exception:
            mask = None
    if mask is None:
        raise SystemExit(f"Failed to read mask image: {path}")
    return np.asarray(mask, dtype=np.uint8)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _maybe_erode_mask(mask_u8: np.ndarray, erode_px: int) -> np.ndarray:
    erode_px = int(erode_px)
    if erode_px <= 0:
        return np.asarray(mask_u8, dtype=np.uint8)

    try:
        import cv2  # type: ignore
    except Exception as e:  # pragma: no cover
        raise SystemExit(f"Missing dependency: cv2 (opencv-python). Error: {e!r}")

    kernel_size = (2 * erode_px) + 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    src = np.asarray(mask_u8, dtype=np.uint8)
    eroded = cv2.erode(src, kernel, iterations=1)
    if int(np.count_nonzero(eroded)) <= 0:
        return src
    return np.asarray(eroded, dtype=np.uint8)


def _resolve_mask_path(scene_dir: Path, cam_id: str, mask_subdir: str, stem: str) -> Path | None:
    mask_root = scene_dir / "cams" / cam_id / str(mask_subdir)
    direct = mask_root / f"{stem}.png"
    if direct.exists():
        return direct
    nested = sorted(mask_root.glob(f"obj_*/{stem}.png"))
    if nested:
        return nested[0]
    return None


def _read_frame_index(frame_times_csv: Path, cams: list[str]) -> list[tuple[int, dict[str, str]]]:
    rows: dict[int, dict[str, str]] = {}
    with frame_times_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ts_us = int(row["ts_us"])
            except Exception as e:
                raise SystemExit(f"Invalid ts_us row in {frame_times_csv}: {row!r}\nError: {e!r}")
            cam_id = str(row["cam_id"]).strip()
            filename = str(row["filename"]).strip()
            if cam_id not in cams:
                continue
            rows.setdefault(ts_us, {})[cam_id] = filename
    return sorted(rows.items(), key=lambda item: item[0])


def _normalize_xyz(value: Any, *, field_name: str) -> np.ndarray:
    if isinstance(value, str):
        tokens = [tok for tok in str(value).replace(",", " ").split() if tok]
        seq = tokens
    else:
        seq = value
    try:
        arr = np.asarray(seq, dtype=np.float32).reshape(-1)
    except Exception as e:
        raise SystemExit(f"Invalid {field_name}: {value!r}\nError: {e!r}")
    if arr.size != 3:
        raise SystemExit(f"Invalid {field_name}: expected 3 values, got {arr.size} from {value!r}")
    return arr.astype(np.float32)


def _float_or(value: Any, default: float) -> float:
    try:
        if value is None or value == "":
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _canonical_support_candidates(
    capture_meta: dict[str, Any],
    *,
    mask_subdir: str,
    depth_subdir: str,
    out_subdir: str,
    requested_support: int,
) -> tuple[list[int], str]:
    requested_support = int(requested_support)
    if requested_support > 0:
        return [requested_support], "explicit"

    target_meta = dict(capture_meta.get("target") or {})
    traj = str(target_meta.get("traj") or "")
    is_gt = any("_gt" in str(value) for value in (mask_subdir, depth_subdir, out_subdir))
    if is_gt:
        return [1], "auto_gt"
    if "circle" in traj:
        return [4, 3, 2], "auto_pred_circle_descend"
    return [3, 2], "auto_pred_static_descend"


def _is_gt_branch(*, mask_subdir: str, depth_subdir: str, out_subdir: str) -> bool:
    return any("_gt" in str(value) for value in (mask_subdir, depth_subdir, out_subdir))


def _is_predicted_static_branch(
    capture_meta: dict[str, Any],
    *,
    mask_subdir: str,
    depth_subdir: str,
    out_subdir: str,
) -> bool:
    if _is_gt_branch(mask_subdir=mask_subdir, depth_subdir=depth_subdir, out_subdir=out_subdir):
        return False
    target_meta = dict(capture_meta.get("target") or {})
    traj = str(target_meta.get("traj") or "")
    return "static" in traj and "circle" not in traj


def _rot_x_deg(angle_deg: float) -> np.ndarray:
    a = math.radians(float(angle_deg))
    c = math.cos(a)
    s = math.sin(a)
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, c, -s],
            [0.0, s, c],
        ],
        dtype=np.float32,
    )


def _rot_z_deg(angle_deg: float) -> np.ndarray:
    a = math.radians(float(angle_deg))
    c = math.cos(a)
    s = math.sin(a)
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def _make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = np.asarray(R, dtype=np.float32)
    T[:3, 3] = np.asarray(t, dtype=np.float32)
    return T


def _invert_T(T: np.ndarray) -> np.ndarray:
    R = np.asarray(T[:3, :3], dtype=np.float32)
    t = np.asarray(T[:3, 3], dtype=np.float32)
    Ti = np.eye(4, dtype=np.float32)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def _maybe_subsample(points_xyz: np.ndarray, max_points: int, rng: np.random.Generator) -> np.ndarray:
    if max_points <= 0 or points_xyz.shape[0] <= max_points:
        return points_xyz
    idx = rng.choice(points_xyz.shape[0], size=int(max_points), replace=False)
    return points_xyz[idx]


def _voxel_downsample_first(points_xyz: np.ndarray, voxel_size: float) -> np.ndarray:
    if points_xyz.size == 0:
        return points_xyz
    if voxel_size <= 0:
        return points_xyz

    pts = np.asarray(points_xyz, dtype=np.float32)
    vox = np.floor(pts / float(voxel_size)).astype(np.int32)
    vox = np.ascontiguousarray(vox)
    vox_view = vox.view([("x", np.int32), ("y", np.int32), ("z", np.int32)])
    _, uniq_idx = np.unique(vox_view, return_index=True)
    uniq_idx = np.sort(uniq_idx)
    return pts[uniq_idx]


def _voxelize_points(points_xyz: np.ndarray, voxel_size: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pts = np.asarray(points_xyz, dtype=np.float32)
    if pts.size == 0:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.int32),
            np.zeros((0,), dtype=np.int32),
            np.zeros((0,), dtype=np.int32),
        )
    if float(voxel_size) <= 0:
        raise ValueError(f"voxel_size must be > 0, got {voxel_size}")

    vox = np.floor(pts / float(voxel_size)).astype(np.int32)
    vox = np.ascontiguousarray(vox)
    vox_view = vox.view([("x", np.int32), ("y", np.int32), ("z", np.int32)]).reshape(-1)
    uniq_view, inverse, counts = np.unique(vox_view, return_inverse=True, return_counts=True)
    uniq_coords = np.stack([uniq_view["x"], uniq_view["y"], uniq_view["z"]], axis=1).astype(np.int32, copy=False)

    sums = np.zeros((uniq_coords.shape[0], 3), dtype=np.float64)
    np.add.at(sums, inverse, pts.astype(np.float64, copy=False))
    centroids = (sums / counts[:, None]).astype(np.float32, copy=False)
    return centroids, uniq_coords, counts.astype(np.int32, copy=False), inverse.astype(np.int32, copy=False)


def _support_histogram_dict(support_counts: np.ndarray) -> dict[str, int]:
    support_counts = np.asarray(support_counts, dtype=np.int32).reshape(-1)
    if support_counts.size == 0:
        return {}
    counts = np.bincount(support_counts)
    return {str(i): int(v) for i, v in enumerate(counts) if v > 0}


def _largest_component_mask(voxel_coords: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    coords = np.asarray(voxel_coords, dtype=np.int32)
    if coords.size == 0:
        return np.zeros((0,), dtype=bool), {"connected_components": 0, "largest_component_voxels": 0}

    keys = [tuple(int(v) for v in row) for row in coords]
    index_by_key = {key: idx for idx, key in enumerate(keys)}
    visited = np.zeros((coords.shape[0],), dtype=bool)
    best_component: list[int] = []
    component_count = 0
    neighbor_offsets = [
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
        if not (dx == 0 and dy == 0 and dz == 0)
    ]

    for start_idx, key in enumerate(keys):
        if visited[start_idx]:
            continue
        component_count += 1
        visited[start_idx] = True
        stack = [start_idx]
        component: list[int] = []
        while stack:
            idx = stack.pop()
            component.append(idx)
            x, y, z = keys[idx]
            for dx, dy, dz in neighbor_offsets:
                next_idx = index_by_key.get((x + dx, y + dy, z + dz))
                if next_idx is None or visited[next_idx]:
                    continue
                visited[next_idx] = True
                stack.append(next_idx)
        if len(component) > len(best_component):
            best_component = component

    keep_mask = np.zeros((coords.shape[0],), dtype=bool)
    if best_component:
        keep_mask[np.asarray(best_component, dtype=np.int32)] = True
    return keep_mask, {
        "connected_components": int(component_count),
        "largest_component_voxels": int(len(best_component)),
    }


def _component_indices_list(voxel_coords: np.ndarray) -> list[np.ndarray]:
    coords = np.asarray(voxel_coords, dtype=np.int32)
    if coords.size == 0:
        return []

    keys = [tuple(int(v) for v in row) for row in coords]
    index_by_key = {key: idx for idx, key in enumerate(keys)}
    visited = np.zeros((coords.shape[0],), dtype=bool)
    components: list[np.ndarray] = []
    neighbor_offsets = [
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
        if not (dx == 0 and dy == 0 and dz == 0)
    ]

    for start_idx in range(coords.shape[0]):
        if visited[start_idx]:
            continue
        visited[start_idx] = True
        stack = [start_idx]
        component: list[int] = []
        while stack:
            idx = stack.pop()
            component.append(idx)
            x, y, z = keys[idx]
            for dx, dy, dz in neighbor_offsets:
                next_idx = index_by_key.get((x + dx, y + dy, z + dz))
                if next_idx is None or visited[next_idx]:
                    continue
                visited[next_idx] = True
                stack.append(next_idx)
        components.append(np.asarray(component, dtype=np.int32))
    return components


def _positive_tail_limit(values: np.ndarray, percentile: float, scale: float, floor_value: float) -> float:
    vals = np.asarray(values, dtype=np.float32).reshape(-1)
    if vals.size == 0:
        return float(floor_value)
    tail = vals[vals > 0]
    if tail.size == 0:
        return float(floor_value)
    limit = float(np.percentile(tail, percentile)) * float(scale)
    return max(float(floor_value), float(limit))


def _median_abs_deviation(values: np.ndarray) -> float:
    vals = np.asarray(values, dtype=np.float32).reshape(-1)
    if vals.size == 0:
        return 0.0
    med = float(np.median(vals))
    return float(np.median(np.abs(vals - med)))


def _cleanup_predicted_static_frame_points(
    points_xyz: np.ndarray,
    *,
    voxel_size_m: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    pts = np.asarray(points_xyz, dtype=np.float32)
    meta: dict[str, Any] = {
        "enabled": True,
        "input_points": int(pts.shape[0]),
    }
    if pts.shape[0] < 128:
        meta["reason"] = "too_few_points"
        meta["output_points"] = int(pts.shape[0])
        return pts, meta

    component_voxel_size = max(float(voxel_size_m) * 1.5, float(voxel_size_m) + 1e-6)
    centroids, uniq_coords, _, inverse = _voxelize_points(pts, component_voxel_size)
    components = _component_indices_list(uniq_coords)
    if not components:
        meta["reason"] = "no_component"
        meta["output_points"] = int(pts.shape[0])
        return pts, meta

    median_center = np.median(centroids, axis=0).astype(np.float32, copy=False)
    best_idx: np.ndarray | None = None
    best_score = -1.0
    best_center = median_center
    best_extent = np.zeros((3,), dtype=np.float32)
    for comp_idx in components:
        comp_pts = centroids[comp_idx]
        comp_center = np.median(comp_pts, axis=0).astype(np.float32, copy=False)
        comp_extent = (comp_pts.max(axis=0) - comp_pts.min(axis=0)).astype(np.float32, copy=False)
        center_dist = float(np.linalg.norm(comp_center - median_center))
        extent_max = float(comp_extent.max()) if comp_extent.size else 0.0
        score = float(comp_pts.shape[0]) / (1.0 + (0.6 * center_dist * center_dist) + (0.15 * extent_max))
        if score > best_score:
            best_score = float(score)
            best_idx = comp_idx
            best_center = comp_center
            best_extent = comp_extent

    component_mask = np.ones((pts.shape[0],), dtype=bool)
    if best_idx is not None:
        keep_vox = np.zeros((uniq_coords.shape[0],), dtype=bool)
        keep_vox[best_idx] = True
        component_mask = keep_vox[inverse]
    component_points = pts[component_mask]
    if component_points.shape[0] < 96:
        meta["reason"] = "component_too_small"
        meta["output_points"] = int(pts.shape[0])
        return pts, meta

    center = np.median(component_points, axis=0).astype(np.float32, copy=False)
    dx = component_points[:, 0] - center[0]
    dy = component_points[:, 1] - center[1]
    dz = component_points[:, 2] - center[2]
    min_tail = max(float(voxel_size_m) * 6.0, 0.08)
    x_limit = max(float(np.percentile(np.abs(dx), 96.0)) * 1.35, min_tail)
    z_limit = max(float(np.percentile(np.abs(dz), 96.0)) * 1.35, min_tail)
    y_low_limit = _positive_tail_limit(-dy, 97.0, 1.12, min_tail)
    y_high_limit = _positive_tail_limit(dy, 97.0, 1.18, min_tail)

    crop_mask = (
        (np.abs(dx) <= x_limit)
        & (np.abs(dz) <= z_limit)
        & (dy >= (-1.0 * y_low_limit))
        & (dy <= y_high_limit)
    )
    cropped = component_points[crop_mask]
    min_kept_points = max(96, int(round(component_points.shape[0] * 0.45)))
    if cropped.shape[0] >= min_kept_points:
        output_points = cropped.astype(np.float32, copy=False)
        crop_applied = True
    else:
        output_points = component_points.astype(np.float32, copy=False)
        crop_applied = False

    meta.update(
        {
            "component_voxel_size_m": float(component_voxel_size),
            "component_count": int(len(components)),
            "selected_component_points": int(component_points.shape[0]),
            "selected_component_center": best_center.astype(float).tolist(),
            "selected_component_extent": best_extent.astype(float).tolist(),
            "crop_center": center.astype(float).tolist(),
            "crop_limits": {
                "x_abs": float(x_limit),
                "y_low": float(y_low_limit),
                "y_high": float(y_high_limit),
                "z_abs": float(z_limit),
            },
            "crop_applied": bool(crop_applied),
            "output_points": int(output_points.shape[0]),
        }
    )
    return output_points, meta


def _filter_predicted_static_canonical_frames(
    points_xyz: np.ndarray,
    frame_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    pts = np.asarray(points_xyz, dtype=np.float32)
    ids = np.asarray(frame_ids, dtype=np.int32).reshape(-1)
    meta: dict[str, Any] = {
        "enabled": True,
        "frame_count": int(np.unique(ids).shape[0]) if ids.size else 0,
        "input_points": int(pts.shape[0]),
    }
    if pts.shape[0] == 0 or ids.size == 0:
        meta["reason"] = "empty_input"
        meta["output_points"] = int(pts.shape[0])
        return pts, ids, meta

    unique_ids = np.unique(ids)
    if unique_ids.size < 24:
        meta["reason"] = "too_few_frames"
        meta["output_points"] = int(pts.shape[0])
        return pts, ids, meta

    centers: list[np.ndarray] = []
    frame_counts: list[int] = []
    for frame_id in unique_ids.tolist():
        frame_mask = ids == int(frame_id)
        frame_pts = pts[frame_mask]
        frame_counts.append(int(frame_pts.shape[0]))
        center = np.median(frame_pts, axis=0).astype(np.float32, copy=False)
        centers.append(center)

    centers_arr = np.asarray(centers, dtype=np.float32)
    counts_arr = np.asarray(frame_counts, dtype=np.int32)
    consensus_center = np.median(centers_arr, axis=0).astype(np.float32, copy=False)
    center_dist = np.linalg.norm(centers_arr - consensus_center[None, :], axis=1).astype(np.float32, copy=False)
    y_dev = np.abs(centers_arr[:, 1] - consensus_center[1]).astype(np.float32, copy=False)
    strict_l2_thresh = 0.6
    strict_y_thresh = 0.4
    relaxed_l2_thresh = 0.9
    relaxed_y_thresh = 0.6
    min_keep_ratio = 0.70

    keep_frames = (center_dist <= strict_l2_thresh) & (y_dev <= strict_y_thresh)
    relaxed = False
    if float(np.count_nonzero(keep_frames)) / float(max(unique_ids.size, 1)) < float(min_keep_ratio):
        keep_frames = (center_dist <= relaxed_l2_thresh) & (y_dev <= relaxed_y_thresh)
        relaxed = True

    kept_frames = unique_ids[keep_frames]
    removed_frames = unique_ids[~keep_frames]
    if kept_frames.size <= 0:
        meta.update(
            {
                "reason": "fallback_keep_original_no_frames",
                "strict_l2_thresh": float(strict_l2_thresh),
                "strict_y_thresh": float(strict_y_thresh),
                "relaxed_l2_thresh": float(relaxed_l2_thresh),
                "relaxed_y_thresh": float(relaxed_y_thresh),
                "relaxed": bool(relaxed),
                "frames_kept": 0,
                "frames_removed": int(unique_ids.size),
                "frame_centroid_keep_ratio": 1.0,
                "frame_centroid_spread_l2": float(np.percentile(center_dist, 90.0)) if center_dist.size else 0.0,
                "output_points": int(pts.shape[0]),
            }
        )
        return pts, ids, meta

    keep_mask = np.isin(ids, kept_frames)
    kept_points = pts[keep_mask]
    kept_ids = ids[keep_mask]

    meta.update(
        {
            "consensus_center": consensus_center.astype(float).tolist(),
            "strict_l2_thresh": float(strict_l2_thresh),
            "strict_y_thresh": float(strict_y_thresh),
            "relaxed_l2_thresh": float(relaxed_l2_thresh),
            "relaxed_y_thresh": float(relaxed_y_thresh),
            "applied_l2_thresh": float(relaxed_l2_thresh if relaxed else strict_l2_thresh),
            "applied_y_thresh": float(relaxed_y_thresh if relaxed else strict_y_thresh),
            "relaxed": bool(relaxed),
            "frames_kept": int(kept_frames.size),
            "frames_removed": int(removed_frames.size),
            "frame_centroid_keep_ratio": float(kept_frames.size) / float(max(unique_ids.size, 1)),
            "frame_centroid_spread_l2": float(np.percentile(center_dist, 90.0)) if center_dist.size else 0.0,
            "frame_centroid_dist_median": float(np.median(center_dist)) if center_dist.size else 0.0,
            "frame_centroid_dist_max": float(center_dist.max()) if center_dist.size else 0.0,
            "frame_centroid_y_dev_median": float(np.median(y_dev)) if y_dev.size else 0.0,
            "frame_centroid_y_dev_max": float(y_dev.max()) if y_dev.size else 0.0,
            "frame_count_median": float(np.median(counts_arr)) if counts_arr.size else 0.0,
            "removed_frame_ids": [int(v) for v in removed_frames.tolist()[:32]],
            "output_points": int(kept_points.shape[0]),
        }
    )
    return kept_points.astype(np.float32, copy=False), kept_ids.astype(np.int32, copy=False), meta


def _trim_predicted_static_canonical_points(
    points_xyz: np.ndarray,
    *,
    voxel_size_m: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    pts = np.asarray(points_xyz, dtype=np.float32)
    meta: dict[str, Any] = {
        "enabled": True,
        "input_points": int(pts.shape[0]),
    }
    if pts.shape[0] < 256:
        meta["reason"] = "too_few_points"
        meta["output_points"] = int(pts.shape[0])
        return pts, meta

    center = np.median(pts, axis=0).astype(np.float32, copy=False)
    dx = pts[:, 0] - center[0]
    dy = pts[:, 1] - center[1]
    dz = pts[:, 2] - center[2]
    min_tail = max(float(voxel_size_m) * 8.0, 0.10)
    x_limit = max(float(np.percentile(np.abs(dx), 98.0)) * 1.08, min_tail)
    z_limit = max(float(np.percentile(np.abs(dz), 98.0)) * 1.08, min_tail)
    y_low_limit = _positive_tail_limit(-dy, 96.0, 1.04, min_tail)
    y_high_limit = _positive_tail_limit(dy, 99.0, 1.08, min_tail)

    keep_mask = (
        (np.abs(dx) <= x_limit)
        & (np.abs(dz) <= z_limit)
        & (dy >= (-1.0 * y_low_limit))
        & (dy <= y_high_limit)
    )
    kept = pts[keep_mask]
    min_keep = max(192, int(round(pts.shape[0] * 0.8)))
    if kept.shape[0] < min_keep:
        meta["reason"] = "fallback_keep_original"
        meta["output_points"] = int(pts.shape[0])
        return pts, meta

    meta.update(
        {
            "crop_center": center.astype(float).tolist(),
            "crop_limits": {
                "x_abs": float(x_limit),
                "y_low": float(y_low_limit),
                "y_high": float(y_high_limit),
                "z_abs": float(z_limit),
            },
            "output_points": int(kept.shape[0]),
        }
    )
    return kept.astype(np.float32, copy=False), meta


def _coarse_crop_predicted_static_canonical_points(
    points_xyz: np.ndarray,
    *,
    gate_meta: dict[str, Any] | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    pts = np.asarray(points_xyz, dtype=np.float32)
    meta: dict[str, Any] = {
        "enabled": True,
        "input_points": int(pts.shape[0]),
    }
    if pts.shape[0] < 512:
        meta["reason"] = "too_few_points"
        meta["output_points"] = int(pts.shape[0])
        return pts, meta

    payload = dict(gate_meta or {})
    center_value = payload.get("consensus_center")
    if not isinstance(center_value, (list, tuple)) or len(center_value) < 3:
        meta["reason"] = "missing_consensus_center"
        meta["output_points"] = int(pts.shape[0])
        return pts, meta

    center = np.asarray([float(center_value[0]), float(center_value[1]), float(center_value[2])], dtype=np.float32)
    dx = pts[:, 0] - float(center[0])
    dy = pts[:, 1] - float(center[1])
    dz = pts[:, 2] - float(center[2])
    below_center = (-dy[dy < 0.0]).astype(np.float32, copy=False)
    above_center = dy[dy > 0.0].astype(np.float32, copy=False)

    x_limit = float(np.clip(max(float(np.percentile(np.abs(dx), 90.0)) * 1.20, 2.60), 2.60, 3.60))
    z_limit = float(np.clip(max(float(np.percentile(np.abs(dz), 92.0)) * 1.20, 0.80), 0.80, 1.40))
    y_low_limit = float(
        np.clip(
            max(float(np.percentile(below_center, 72.5)) * 1.10 if below_center.size else 0.55, 0.55),
            0.55,
            1.10,
        )
    )
    y_high_limit = float(
        np.clip(
            max(float(np.percentile(above_center, 96.0)) * 1.10 if above_center.size else 1.20, 1.20),
            1.20,
            2.40,
        )
    )
    keep_mask = (
        (np.abs(dx) <= x_limit)
        & (np.abs(dz) <= z_limit)
        & (dy >= (-1.0 * y_low_limit))
        & (dy <= y_high_limit)
    )
    kept = pts[keep_mask]
    min_keep_points = max(4096, int(round(float(pts.shape[0]) * 0.18)))
    if kept.shape[0] < int(min_keep_points):
        meta.update(
            {
                "reason": "fallback_keep_original",
                "crop_center": center.astype(float).tolist(),
                "bounds": {
                    "x_abs": float(x_limit),
                    "y_low": float(y_low_limit),
                    "y_high": float(y_high_limit),
                    "z_abs": float(z_limit),
                },
                "min_keep_points": int(min_keep_points),
                "output_points": int(pts.shape[0]),
            }
        )
        return pts, meta

    meta.update(
        {
            "reason": "applied",
            "crop_center": center.astype(float).tolist(),
            "bounds": {
                "x_abs": float(x_limit),
                "y_low": float(y_low_limit),
                "y_high": float(y_high_limit),
                "z_abs": float(z_limit),
            },
            "keep_ratio": float(kept.shape[0]) / float(max(pts.shape[0], 1)),
            "output_points": int(kept.shape[0]),
        }
    )
    return kept.astype(np.float32, copy=False), meta


def _augment_predicted_static_canonical_points(
    core_points_xyz: np.ndarray,
    aux_points_xyz: np.ndarray,
    *,
    voxel_size_m: float,
    max_points: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    core = np.asarray(core_points_xyz, dtype=np.float32)
    aux = np.asarray(aux_points_xyz, dtype=np.float32)
    meta: dict[str, Any] = {
        "enabled": True,
        "core_points": int(core.shape[0]),
        "aux_points": int(aux.shape[0]),
    }
    if core.shape[0] < 256:
        meta["reason"] = "core_too_small"
        meta["output_points"] = int(core.shape[0])
        return core, meta
    if aux.shape[0] <= core.shape[0]:
        meta["reason"] = "no_larger_aux"
        meta["output_points"] = int(core.shape[0])
        return core, meta

    center = np.median(core, axis=0).astype(np.float32, copy=False)
    dx = core[:, 0] - center[0]
    dy = core[:, 1] - center[1]
    dz = core[:, 2] - center[2]
    min_tail = max(float(voxel_size_m) * 10.0, 0.12)
    x_limit = max(float(np.percentile(np.abs(dx), 98.0)) * 2.45, min_tail)
    z_limit = max(float(np.percentile(np.abs(dz), 98.0)) * 2.45, min_tail)
    y_low_limit = _positive_tail_limit(-dy, 97.0, 1.55, min_tail)
    y_high_limit = _positive_tail_limit(dy, 99.0, 1.75, min_tail)

    aux_dx = aux[:, 0] - center[0]
    aux_dy = aux[:, 1] - center[1]
    aux_dz = aux[:, 2] - center[2]
    aux_mask = (
        (np.abs(aux_dx) <= x_limit)
        & (np.abs(aux_dz) <= z_limit)
        & (aux_dy >= (-1.0 * y_low_limit))
        & (aux_dy <= y_high_limit)
    )
    aux_selected = aux[aux_mask]
    min_added = max(128, int(round(core.shape[0] * 0.08)))
    if aux_selected.shape[0] < min_added:
        meta["reason"] = "too_few_aux_points_in_core_box"
        meta["output_points"] = int(core.shape[0])
        return core, meta

    merged = np.concatenate([core, aux_selected], axis=0).astype(np.float32, copy=False)
    merged = _voxel_downsample_first(merged, float(voxel_size_m))
    if int(max_points) > 0 and merged.shape[0] > int(max_points):
        merged_center = np.median(merged, axis=0).astype(np.float32, copy=False)
        dist = np.linalg.norm(merged - merged_center[None, :], axis=1)
        order = np.argsort(dist, kind="stable")
        merged = merged[order[: int(max_points)]]
    if merged.shape[0] <= int(round(core.shape[0] * 1.05)):
        meta["reason"] = "insufficient_growth"
        meta["output_points"] = int(core.shape[0])
        return core, meta

    meta.update(
        {
            "crop_center": center.astype(float).tolist(),
            "crop_limits": {
                "x_abs": float(x_limit),
                "y_low": float(y_low_limit),
                "y_high": float(y_high_limit),
                "z_abs": float(z_limit),
            },
            "aux_points_selected": int(aux_selected.shape[0]),
            "output_points": int(merged.shape[0]),
        }
    )
    return merged.astype(np.float32, copy=False), meta


def _core_clip_canonical_voxels(
    kept_points: np.ndarray,
    kept_vox: np.ndarray,
    kept_support: np.ndarray,
    kept_total_counts: np.ndarray,
    *,
    min_support: int,
    keep_largest_component: bool,
    min_canonical_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    points = np.asarray(kept_points, dtype=np.float32)
    vox = np.asarray(kept_vox, dtype=np.int32)
    support = np.asarray(kept_support, dtype=np.int32).reshape(-1)
    total_counts = np.asarray(kept_total_counts, dtype=np.int32).reshape(-1)
    input_count = int(points.shape[0])
    meta: dict[str, Any] = {
        "enabled": True,
        "input_points": int(input_count),
        "min_canonical_points": int(min_canonical_points),
        "min_support": int(min_support),
    }
    if input_count <= 0:
        meta["reason"] = "empty_input"
        meta["output_points"] = 0
        meta["clip_keep_ratio"] = 0.0
        return points, vox, support, total_counts, meta

    core_support_thresh = max(int(min_support) + 1, 4)
    core_mask = support >= int(core_support_thresh)
    core_source = "support_threshold"
    min_core_points = max(1, int(math.ceil(max(int(min_canonical_points), 1) / 4.0)))
    if int(np.count_nonzero(core_mask)) < int(min_core_points):
        top_count = max(1, int(math.ceil(float(input_count) * 0.20)))
        order = np.lexsort((total_counts, support))[::-1]
        core_mask = np.zeros((input_count,), dtype=bool)
        core_mask[order[:top_count]] = True
        core_source = "top20_support_fallback"

    core_points = points[core_mask]
    if core_points.shape[0] <= 0:
        meta["reason"] = "fallback_no_core"
        meta["core_support_thresh"] = int(core_support_thresh)
        meta["core_source"] = str(core_source)
        meta["core_point_count"] = 0
        meta["output_points"] = int(input_count)
        meta["clip_keep_ratio"] = 1.0
        return points, vox, support, total_counts, meta

    def _bounds(values: np.ndarray, lo_pct: float, hi_pct: float, pad_lo: float, pad_hi: float) -> tuple[float, float]:
        vals = np.asarray(values, dtype=np.float32).reshape(-1)
        lo = float(np.percentile(vals, float(lo_pct)))
        hi = float(np.percentile(vals, float(hi_pct)))
        return float(lo - pad_lo), float(hi + pad_hi)

    x_lo_raw = float(np.percentile(core_points[:, 0], 2.5))
    x_hi_raw = float(np.percentile(core_points[:, 0], 97.5))
    z_lo_raw = float(np.percentile(core_points[:, 2], 2.5))
    z_hi_raw = float(np.percentile(core_points[:, 2], 97.5))
    x_pad = min(max((x_hi_raw - x_lo_raw) * 0.15, 0.08), 0.40)
    z_pad = min(max((z_hi_raw - z_lo_raw) * 0.15, 0.08), 0.40)
    x_lo, x_hi = float(x_lo_raw - x_pad), float(x_hi_raw + x_pad)
    y_lo, y_hi = _bounds(core_points[:, 1], 5.0, 97.5, 0.05, 0.20)
    z_lo, z_hi = float(z_lo_raw - z_pad), float(z_hi_raw + z_pad)

    clip_mask = (
        (points[:, 0] >= x_lo)
        & (points[:, 0] <= x_hi)
        & (points[:, 1] >= y_lo)
        & (points[:, 1] <= y_hi)
        & (points[:, 2] >= z_lo)
        & (points[:, 2] <= z_hi)
    )
    clipped_points = points[clip_mask]
    clipped_vox = vox[clip_mask]
    clipped_support = support[clip_mask]
    clipped_total_counts = total_counts[clip_mask]
    clip_count = int(clipped_points.shape[0])
    clip_keep_ratio = float(clip_count) / float(max(input_count, 1))

    if clip_count < int(min_canonical_points):
        meta.update(
            {
                "reason": "fallback_clip_too_small",
                "core_support_thresh": int(core_support_thresh),
                "core_source": str(core_source),
                "core_point_count": int(core_points.shape[0]),
                "clip_points": int(clip_count),
                "clip_keep_ratio": float(clip_keep_ratio),
                "output_points": int(input_count),
            }
        )
        return points, vox, support, total_counts, meta

    component_meta = {"connected_components": 1, "largest_component_voxels": int(clip_count)}
    if bool(keep_largest_component) and clipped_points.shape[0] > 0:
        component_mask, component_meta = _largest_component_mask(clipped_vox)
        component_points = clipped_points[component_mask]
        component_vox = clipped_vox[component_mask]
        component_support = clipped_support[component_mask]
        component_total_counts = clipped_total_counts[component_mask]
        if component_points.shape[0] < int(min_canonical_points):
            meta.update(
                {
                    "reason": "fallback_component_too_small",
                    "core_support_thresh": int(core_support_thresh),
                    "core_source": str(core_source),
                    "core_point_count": int(core_points.shape[0]),
                    "clip_points": int(clip_count),
                    "clip_keep_ratio": float(clip_keep_ratio),
                    "component_points": int(component_points.shape[0]),
                    "component_connected_components": int(component_meta["connected_components"]),
                    "component_largest_voxels": int(component_meta["largest_component_voxels"]),
                    "output_points": int(input_count),
                }
            )
            return points, vox, support, total_counts, meta
        clipped_points = component_points
        clipped_vox = component_vox
        clipped_support = component_support
        clipped_total_counts = component_total_counts

    meta.update(
        {
            "reason": "applied",
            "core_support_thresh": int(core_support_thresh),
            "core_source": str(core_source),
            "core_point_count": int(core_points.shape[0]),
            "core_center": np.median(core_points, axis=0).astype(float).tolist(),
            "clip_bounds": {
                "x": [float(x_lo), float(x_hi)],
                "y": [float(y_lo), float(y_hi)],
                "z": [float(z_lo), float(z_hi)],
            },
            "clip_points": int(clip_count),
            "clip_keep_ratio": float(clip_keep_ratio),
            "component_connected_components": int(component_meta["connected_components"]),
            "component_largest_voxels": int(component_meta["largest_component_voxels"]),
            "output_points": int(clipped_points.shape[0]),
        }
    )
    return (
        clipped_points.astype(np.float32, copy=False),
        clipped_vox.astype(np.int32, copy=False),
        clipped_support.astype(np.int32, copy=False),
        clipped_total_counts.astype(np.int32, copy=False),
        meta,
    )


def _extent_xyz(points_xyz: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_xyz, dtype=np.float32)
    if pts.size == 0:
        return np.zeros((3,), dtype=np.float32)
    return (pts.max(axis=0) - pts.min(axis=0)).astype(np.float32, copy=False)


def _top_fraction_mask(
    support_counts: np.ndarray,
    total_counts: np.ndarray,
    *,
    fraction: float,
) -> np.ndarray:
    support = np.asarray(support_counts, dtype=np.int32).reshape(-1)
    total = np.asarray(total_counts, dtype=np.int32).reshape(-1)
    mask = np.zeros((support.shape[0],), dtype=bool)
    if support.size == 0:
        return mask
    top_k = max(1, int(math.ceil(float(support.size) * float(fraction))))
    order = np.lexsort((total, support))[::-1]
    mask[order[:top_k]] = True
    return mask


def _within_voxel_radius_mask(
    candidate_vox: np.ndarray,
    anchor_vox: np.ndarray,
    *,
    radius: int,
) -> np.ndarray:
    cand = np.asarray(candidate_vox, dtype=np.int32)
    anchor = np.asarray(anchor_vox, dtype=np.int32)
    mask = np.zeros((cand.shape[0],), dtype=bool)
    if cand.size == 0 or anchor.size == 0:
        return mask
    anchor_set = {tuple(int(v) for v in row) for row in anchor}
    offsets = [
        (dx, dy, dz)
        for dx in range(-int(radius), int(radius) + 1)
        for dy in range(-int(radius), int(radius) + 1)
        for dz in range(-int(radius), int(radius) + 1)
    ]
    for idx, row in enumerate(cand.tolist()):
        x, y, z = (int(row[0]), int(row[1]), int(row[2]))
        keep = False
        for dx, dy, dz in offsets:
            if (x + dx, y + dy, z + dz) in anchor_set:
                keep = True
                break
        mask[idx] = keep
    return mask


def _shell_priority_order(
    points_xyz: np.ndarray,
    support_counts: np.ndarray,
    total_counts: np.ndarray,
    *,
    core_center: np.ndarray,
) -> np.ndarray:
    pts = np.asarray(points_xyz, dtype=np.float32)
    support = np.asarray(support_counts, dtype=np.int32).reshape(-1)
    total = np.asarray(total_counts, dtype=np.int32).reshape(-1)
    if pts.shape[0] <= 0:
        return np.zeros((0,), dtype=np.int32)
    center = np.asarray(core_center, dtype=np.float32).reshape(3)
    edge_radius = np.linalg.norm(pts[:, [0, 2]] - center[None, [0, 2]], axis=1).astype(np.float32, copy=False)
    upper_bonus = np.clip(pts[:, 1] - float(center[1]), 0.0, None).astype(np.float32, copy=False)
    lower_penalty = np.clip(float(center[1]) - pts[:, 1], 0.0, None).astype(np.float32, copy=False)
    edge_score = (edge_radius + (0.25 * upper_bonus) - (0.20 * lower_penalty)).astype(np.float32, copy=False)
    return np.lexsort((total, support, edge_score))[::-1].astype(np.int32, copy=False)


def _canonicalize_predicted_static_core_shell(
    points_xyz: np.ndarray,
    frame_ids: np.ndarray,
    *,
    voxel_size_m: float,
    min_support: int,
    keep_largest_component: bool,
    max_points: int,
    min_canonical_points: int,
) -> tuple[np.ndarray, dict[str, Any], dict[str, np.ndarray]]:
    pts = np.asarray(points_xyz, dtype=np.float32)
    ids = np.asarray(frame_ids, dtype=np.int32).reshape(-1)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise SystemExit(f"Invalid canonical points shape: {pts.shape}")
    if pts.shape[0] != ids.shape[0]:
        raise SystemExit(f"points/frame_ids length mismatch: points={pts.shape[0]} frame_ids={ids.shape[0]}")
    if pts.shape[0] == 0:
        empty = np.zeros((0, 3), dtype=np.float32)
        return empty, {
            "canonical_raw_point_count": 0,
            "canonical_voxel_count_before_support": 0,
            "canonical_voxel_count_after_support": 0,
            "canonical_voxel_count_after_component": 0,
            "canonical_voxel_count_after_core_clip": 0,
            "canonical_point_count_after_cap": 0,
            "canonical_connected_components": 0,
            "canonical_largest_component_voxels": 0,
            "canonical_support_histogram": {},
            "canonical_support_min": 0,
            "canonical_support_median": 0.0,
            "canonical_support_p90": 0.0,
            "canonical_support_max": 0,
            "canonical_cap_strategy": "support_then_density",
            "static_core_clip_meta": {},
            "component_fallback_used": False,
        }, {
            "raw": empty,
            "core": empty,
            "shell": empty,
        }
    if float(voxel_size_m) <= 0:
        raise SystemExit("--canonical_voxel_size_m must be > 0 when canonical support filtering is enabled.")

    centroids, uniq_coords, total_counts, inverse = _voxelize_points(pts, float(voxel_size_m))
    pair_view = np.rec.fromarrays([inverse.astype(np.int32, copy=False), ids], names="vox,frame")
    uniq_pairs = np.unique(pair_view)
    support_counts = np.bincount(uniq_pairs["vox"], minlength=uniq_coords.shape[0]).astype(np.int32, copy=False)

    support_mask = support_counts >= max(1, int(min_support))
    support_points = centroids[support_mask]
    support_vox = uniq_coords[support_mask]
    support_support = support_counts[support_mask]
    support_total = total_counts[support_mask]

    support_component_points = support_points
    support_component_vox = support_vox
    support_component_support = support_support
    support_component_total = support_total
    support_component_meta = {
        "connected_components": int(1 if support_component_points.shape[0] > 0 else 0),
        "largest_component_voxels": int(support_component_points.shape[0]),
    }
    if bool(keep_largest_component) and support_points.shape[0] > 0:
        support_component_mask, support_component_meta = _largest_component_mask(support_vox)
        support_component_points = support_points[support_component_mask]
        support_component_vox = support_vox[support_component_mask]
        support_component_support = support_support[support_component_mask]
        support_component_total = support_total[support_component_mask]

    core_support_thresh = max(int(min_support) + 1, 4)
    core_mask = support_component_support >= int(core_support_thresh)
    core_source = "support_threshold"
    min_core_points = max(1, int(math.ceil(float(max(int(min_canonical_points), 1)) / 4.0)))
    if int(np.count_nonzero(core_mask)) < int(min_core_points):
        core_mask = _top_fraction_mask(
            support_component_support,
            support_component_total,
            fraction=0.25,
        )
        core_source = "top25_support_fallback"
    if not np.any(core_mask) and support_component_points.shape[0] > 0:
        core_mask = np.ones((support_component_points.shape[0],), dtype=bool)
        core_source = "full_support_component_fallback"

    core_points = support_component_points[core_mask]
    core_vox = support_component_vox[core_mask]
    core_support = support_component_support[core_mask]
    core_total = support_component_total[core_mask]
    core_component_meta = {
        "connected_components": int(1 if core_points.shape[0] > 0 else 0),
        "largest_component_voxels": int(core_points.shape[0]),
    }
    if bool(keep_largest_component) and core_points.shape[0] > 0:
        core_component_mask, core_component_meta = _largest_component_mask(core_vox)
        core_points = core_points[core_component_mask]
        core_vox = core_vox[core_component_mask]
        core_support = core_support[core_component_mask]
        core_total = core_total[core_component_mask]

    shell_points = np.zeros((0, 3), dtype=np.float32)
    shell_vox = np.zeros((0, 3), dtype=np.int32)
    shell_support = np.zeros((0,), dtype=np.int32)
    shell_total = np.zeros((0,), dtype=np.int32)
    final_points = support_component_points
    final_vox = support_component_vox
    final_support = support_component_support
    final_total = support_component_total
    final_component_meta = dict(support_component_meta)
    component_fallback_used = False

    if core_points.shape[0] > 0:
        raw_extent = _extent_xyz(pts).astype(np.float32, copy=False)
        moderate_raw_extent = bool(float(raw_extent.max()) <= 8.0 and float(np.mean(raw_extent)) <= 3.5)
        core_bbox_min = core_points.min(axis=0).astype(np.float32, copy=False)
        core_bbox_max = core_points.max(axis=0).astype(np.float32, copy=False)
        core_span = np.maximum(core_bbox_max - core_bbox_min, 1e-6).astype(np.float32, copy=False)
        x_pad = float(np.clip(core_span[0] * 0.35, 0.12, 0.60))
        z_pad = float(np.clip(core_span[2] * 0.35, 0.12, 0.60))
        support_q_x = np.percentile(support_component_points[:, 0], [2.5, 97.5]).astype(np.float32, copy=False)
        support_q_y = np.percentile(support_component_points[:, 1], [12.5, 97.5]).astype(np.float32, copy=False)
        support_q_z = np.percentile(support_component_points[:, 2], [2.5, 97.5]).astype(np.float32, copy=False)
        support_x_pad = float(np.clip((float(support_q_x[1] - support_q_x[0])) * 0.18, 0.10, 0.60))
        support_z_pad = float(np.clip((float(support_q_z[1] - support_q_z[0])) * 0.18, 0.10, 0.60))
        expanded_bbox_min = np.array(
            [
                float(min(float(core_bbox_min[0] - x_pad), float(support_q_x[0] - support_x_pad))),
                float(max(float(np.percentile(core_points[:, 1], 5.0) - 0.12), float(support_q_y[0] - 0.08))),
                float(min(float(core_bbox_min[2] - z_pad), float(support_q_z[0] - support_z_pad))),
            ],
            dtype=np.float32,
        )
        expanded_bbox_max = np.array(
            [
                float(max(float(core_bbox_max[0] + x_pad), float(support_q_x[1] + support_x_pad))),
                float(max(float(core_bbox_max[1] + 0.45), float(support_q_y[1] + 0.10))),
                float(max(float(core_bbox_max[2] + z_pad), float(support_q_z[1] + support_z_pad))),
            ],
            dtype=np.float32,
        )
        if moderate_raw_extent:
            raw_q_x = np.percentile(pts[:, 0], [2.5, 97.5]).astype(np.float32, copy=False)
            raw_q_y = np.percentile(pts[:, 1], [10.0, 97.5]).astype(np.float32, copy=False)
            raw_q_z = np.percentile(pts[:, 2], [2.5, 97.5]).astype(np.float32, copy=False)
            raw_x_pad = float(np.clip((float(raw_q_x[1] - raw_q_x[0])) * 0.08, 0.08, 0.35))
            raw_z_pad = float(np.clip((float(raw_q_z[1] - raw_q_z[0])) * 0.08, 0.06, 0.20))
            expanded_bbox_min[0] = float(min(float(expanded_bbox_min[0]), float(raw_q_x[0] - raw_x_pad)))
            expanded_bbox_max[0] = float(max(float(expanded_bbox_max[0]), float(raw_q_x[1] + raw_x_pad)))
            expanded_bbox_max[1] = float(max(float(expanded_bbox_max[1]), float(raw_q_y[1] + 0.10)))
            expanded_bbox_min[2] = float(min(float(expanded_bbox_min[2]), float(raw_q_z[0] - raw_z_pad)))
            expanded_bbox_max[2] = float(max(float(expanded_bbox_max[2]), float(raw_q_z[1] + raw_z_pad)))
        core_y_p05 = float(np.percentile(core_points[:, 1], 5.0))
        bottom_tail_y_thresh = float(core_y_p05 - 0.12)
        support_component_vox_set = {tuple(int(v) for v in row) for row in support_component_vox}
        core_center = np.median(core_points, axis=0).astype(np.float32, copy=False)

        shell_support_thresh = 1 if moderate_raw_extent else max(1, int(min_support) - 2)
        component_candidate_mask = support_component_support >= int(shell_support_thresh)
        if core_vox.shape[0] > 0:
            core_vox_set = {tuple(int(v) for v in row) for row in core_vox}
            component_candidate_mask &= np.asarray(
                [tuple(int(v) for v in row) not in core_vox_set for row in support_component_vox.tolist()],
                dtype=bool,
            )
        else:
            core_vox_set = set()

        component_candidate_points = support_component_points[component_candidate_mask]
        component_candidate_vox = support_component_vox[component_candidate_mask]
        component_candidate_support = support_component_support[component_candidate_mask]
        component_candidate_total = support_component_total[component_candidate_mask]

        halo_candidate_mask = support_counts >= int(shell_support_thresh)
        if core_vox.shape[0] > 0:
            halo_candidate_mask &= np.asarray(
                [tuple(int(v) for v in row) not in core_vox_set for row in uniq_coords.tolist()],
                dtype=bool,
            )
        halo_candidate_mask &= np.asarray(
            [tuple(int(v) for v in row) not in support_component_vox_set for row in uniq_coords.tolist()],
            dtype=bool,
        )
        halo_candidate_points = centroids[halo_candidate_mask]
        halo_candidate_vox = uniq_coords[halo_candidate_mask]
        halo_candidate_support = support_counts[halo_candidate_mask]
        halo_candidate_total = total_counts[halo_candidate_mask]
        halo_adjacent_mask = np.zeros((halo_candidate_points.shape[0],), dtype=bool)
        if halo_candidate_points.shape[0] > 0 and support_component_vox.shape[0] > 0:
            halo_adjacent_mask = _within_voxel_radius_mask(
                halo_candidate_vox,
                support_component_vox,
                radius=2,
            )
            if core_vox.shape[0] > 0:
                halo_adjacent_mask &= _within_voxel_radius_mask(
                    halo_candidate_vox,
                    core_vox,
                    radius=3,
                )

        shell_candidate_points = np.concatenate([component_candidate_points, halo_candidate_points[halo_adjacent_mask]], axis=0).astype(
            np.float32,
            copy=False,
        )
        shell_candidate_vox = np.concatenate([component_candidate_vox, halo_candidate_vox[halo_adjacent_mask]], axis=0).astype(
            np.int32,
            copy=False,
        )
        shell_candidate_support = np.concatenate(
            [component_candidate_support, halo_candidate_support[halo_adjacent_mask]],
            axis=0,
        ).astype(np.int32, copy=False)
        shell_candidate_total = np.concatenate(
            [component_candidate_total, halo_candidate_total[halo_adjacent_mask]],
            axis=0,
        ).astype(np.int32, copy=False)

        if shell_candidate_points.shape[0] > 0:
            bbox_mask = (
                np.all(shell_candidate_points >= expanded_bbox_min[None, :], axis=1)
                & np.all(shell_candidate_points <= expanded_bbox_max[None, :], axis=1)
            )
            near_mask = np.ones((shell_candidate_points.shape[0],), dtype=bool)
            if core_vox.shape[0] > 0:
                near_mask &= _within_voxel_radius_mask(
                    shell_candidate_vox,
                    core_vox,
                    radius=3,
                )
            if support_component_vox.shape[0] > 0:
                near_mask &= _within_voxel_radius_mask(
                    shell_candidate_vox,
                    support_component_vox,
                    radius=2,
                )
            inside_core_xz = (
                (shell_candidate_points[:, 0] >= float(core_bbox_min[0]))
                & (shell_candidate_points[:, 0] <= float(core_bbox_max[0]))
                & (shell_candidate_points[:, 2] >= float(core_bbox_min[2]))
                & (shell_candidate_points[:, 2] <= float(core_bbox_max[2]))
            )
            bottom_mask = (shell_candidate_points[:, 1] >= float(bottom_tail_y_thresh)) | inside_core_xz
            shell_keep_mask = bbox_mask & near_mask & bottom_mask
            shell_points = shell_candidate_points[shell_keep_mask].astype(np.float32, copy=False)
            shell_vox = shell_candidate_vox[shell_keep_mask].astype(np.int32, copy=False)
            shell_support = shell_candidate_support[shell_keep_mask].astype(np.int32, copy=False)
            shell_total = shell_candidate_total[shell_keep_mask].astype(np.int32, copy=False)

        shell_soft_cap_points = -1
        shell_trimmed_points = 0
        if int(max_points) > 0 and shell_points.shape[0] > 0:
            soft_target_total = int(max(core_points.shape[0], min(int(max_points), int(round(float(max_points) * 0.90)))))
            shell_soft_cap_points = int(max(soft_target_total - int(core_points.shape[0]), 0))
            if shell_points.shape[0] > int(shell_soft_cap_points) >= 0:
                order = _shell_priority_order(
                    shell_points,
                    shell_support,
                    shell_total,
                    core_center=core_center,
                )
                keep_idx = order[: int(shell_soft_cap_points)]
                shell_trimmed_points = int(shell_points.shape[0] - keep_idx.shape[0])
                shell_points = shell_points[keep_idx].astype(np.float32, copy=False)
                shell_vox = shell_vox[keep_idx].astype(np.int32, copy=False)
                shell_support = shell_support[keep_idx].astype(np.int32, copy=False)
                shell_total = shell_total[keep_idx].astype(np.int32, copy=False)

        union_points = np.concatenate([core_points, shell_points], axis=0).astype(np.float32, copy=False)
        union_vox = np.concatenate([core_vox, shell_vox], axis=0).astype(np.int32, copy=False)
        union_support = np.concatenate([core_support, shell_support], axis=0).astype(np.int32, copy=False)
        union_total = np.concatenate([core_total, shell_total], axis=0).astype(np.int32, copy=False)
        if union_points.shape[0] > 0:
            final_points = union_points
            final_vox = union_vox
            final_support = union_support
            final_total = union_total
            final_component_meta = {
                "connected_components": int(1 if final_points.shape[0] > 0 else 0),
                "largest_component_voxels": int(final_points.shape[0]),
            }
            if bool(keep_largest_component):
                final_component_mask, final_component_meta = _largest_component_mask(final_vox)
                final_points = final_points[final_component_mask]
                final_vox = final_vox[final_component_mask]
                final_support = final_support[final_component_mask]
                final_total = final_total[final_component_mask]

            if final_points.shape[0] < int(min_canonical_points):
                final_points = support_component_points
                final_vox = support_component_vox
                final_support = support_component_support
                final_total = support_component_total
                final_component_meta = dict(support_component_meta)
                component_fallback_used = True

        static_core_clip_meta: dict[str, Any] = {
            "enabled": True,
            "reason": "applied" if not component_fallback_used else "fallback_support_component",
            "core_support_thresh": int(core_support_thresh),
            "shell_support_thresh": int(shell_support_thresh),
            "core_source": str(core_source),
            "core_point_count": int(core_points.shape[0]),
            "shell_added_points": int(shell_points.shape[0]),
            "core_shell_ratio": float(core_points.shape[0]) / float(max(shell_points.shape[0], 1)),
            "raw_extent_xyz": _extent_xyz(pts).astype(float).tolist(),
            "core_extent_xyz": _extent_xyz(core_points).astype(float).tolist(),
            "shell_extent_xyz": _extent_xyz(shell_points).astype(float).tolist(),
            "shell_added_points_count": int(shell_points.shape[0]),
            "y_downshift_vs_raw": float(np.median(pts[:, 1]) - np.median(final_points[:, 1])) if final_points.shape[0] > 0 else 0.0,
            "support_component_point_count": int(support_component_points.shape[0]),
            "support_component_extent_xyz": _extent_xyz(support_component_points).astype(float).tolist(),
            "moderate_raw_extent": bool(moderate_raw_extent),
            "core_bbox_min": core_bbox_min.astype(float).tolist(),
            "core_bbox_max": core_bbox_max.astype(float).tolist(),
            "expanded_bbox_min": expanded_bbox_min.astype(float).tolist(),
            "expanded_bbox_max": expanded_bbox_max.astype(float).tolist(),
            "bottom_tail_y_thresh": float(bottom_tail_y_thresh),
            "shell_component_candidate_points": int(component_candidate_points.shape[0]),
            "shell_halo_candidate_points": int(np.count_nonzero(halo_adjacent_mask)),
            "shell_soft_cap_points": int(shell_soft_cap_points),
            "shell_trimmed_points": int(shell_trimmed_points),
            "output_points": int(final_points.shape[0]),
            "final_component_connected_components": int(final_component_meta["connected_components"]),
            "final_component_largest_voxels": int(final_component_meta["largest_component_voxels"]),
            "component_fallback_used": bool(component_fallback_used),
        }
    else:
        static_core_clip_meta = {
            "enabled": True,
            "reason": "fallback_no_core",
            "core_support_thresh": int(core_support_thresh),
            "shell_support_thresh": int(max(1, int(min_support) - 2)),
            "core_source": str(core_source),
            "core_point_count": 0,
            "shell_added_points": 0,
            "core_shell_ratio": float("inf"),
            "raw_extent_xyz": _extent_xyz(pts).astype(float).tolist(),
            "core_extent_xyz": [0.0, 0.0, 0.0],
            "shell_extent_xyz": [0.0, 0.0, 0.0],
            "y_downshift_vs_raw": float(np.median(pts[:, 1]) - np.median(final_points[:, 1])) if final_points.shape[0] > 0 else 0.0,
            "output_points": int(final_points.shape[0]),
            "component_fallback_used": True,
        }
        component_fallback_used = True

    if int(max_points) > 0 and final_points.shape[0] > int(max_points):
        order = np.lexsort((final_total, final_support))[::-1]
        keep_idx = order[: int(max_points)]
        final_points = final_points[keep_idx]
        final_support = final_support[keep_idx]
        final_total = final_total[keep_idx]

    support_hist = _support_histogram_dict(final_support)
    meta = {
        "canonical_raw_point_count": int(pts.shape[0]),
        "canonical_voxel_count_before_support": int(centroids.shape[0]),
        "canonical_voxel_count_after_support": int(support_points.shape[0]),
        "canonical_voxel_count_after_component": int(support_component_points.shape[0]),
        "canonical_voxel_count_after_core_clip": int(final_support.shape[0]),
        "canonical_point_count_after_cap": int(final_points.shape[0]),
        "canonical_connected_components": int(final_component_meta["connected_components"]),
        "canonical_largest_component_voxels": int(final_component_meta["largest_component_voxels"]),
        "canonical_support_histogram": support_hist,
        "canonical_support_min": int(final_support.min()) if final_support.size else 0,
        "canonical_support_median": float(np.median(final_support)) if final_support.size else 0.0,
        "canonical_support_p90": float(np.percentile(final_support, 90.0)) if final_support.size else 0.0,
        "canonical_support_max": int(final_support.max()) if final_support.size else 0,
        "canonical_cap_strategy": "support_then_density",
        "static_core_clip_meta": static_core_clip_meta,
        "component_fallback_used": bool(component_fallback_used),
        "raw_extent_xyz": _extent_xyz(pts).astype(float).tolist(),
        "core_extent_xyz": list(static_core_clip_meta.get("core_extent_xyz") or [0.0, 0.0, 0.0]),
        "shell_extent_xyz": list(static_core_clip_meta.get("shell_extent_xyz") or [0.0, 0.0, 0.0]),
        "core_shell_ratio": static_core_clip_meta.get("core_shell_ratio"),
        "shell_added_points": int(static_core_clip_meta.get("shell_added_points", 0)),
        "y_downshift_vs_raw": float(static_core_clip_meta.get("y_downshift_vs_raw", 0.0)),
    }
    debug = {
        "raw": pts.astype(np.float32, copy=False),
        "core": core_points.astype(np.float32, copy=False),
        "shell": shell_points.astype(np.float32, copy=False),
    }
    return final_points.astype(np.float32, copy=False), meta, debug


def _canonicalize_points(
    points_xyz: np.ndarray,
    frame_ids: np.ndarray,
    *,
    voxel_size_m: float,
    min_support: int,
    keep_largest_component: bool,
    max_points: int,
    core_clip: bool = False,
    min_canonical_points: int = 128,
) -> tuple[np.ndarray, dict[str, Any]]:
    pts = np.asarray(points_xyz, dtype=np.float32)
    ids = np.asarray(frame_ids, dtype=np.int32).reshape(-1)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise SystemExit(f"Invalid canonical points shape: {pts.shape}")
    if pts.shape[0] != ids.shape[0]:
        raise SystemExit(f"points/frame_ids length mismatch: points={pts.shape[0]} frame_ids={ids.shape[0]}")
    if pts.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32), {
            "canonical_raw_point_count": 0,
            "canonical_voxel_count_before_support": 0,
            "canonical_voxel_count_after_support": 0,
            "canonical_voxel_count_after_component": 0,
            "canonical_point_count_after_cap": 0,
            "canonical_connected_components": 0,
            "canonical_largest_component_voxels": 0,
            "canonical_support_histogram": {},
            "canonical_support_min": 0,
            "canonical_support_median": 0.0,
            "canonical_support_p90": 0.0,
            "canonical_support_max": 0,
            "canonical_cap_strategy": "support_then_density",
            "canonical_voxel_count_after_core_clip": 0,
            "static_core_clip_meta": {},
        }
    if float(voxel_size_m) <= 0:
        raise SystemExit("--canonical_voxel_size_m must be > 0 when canonical support filtering is enabled.")

    centroids, uniq_coords, total_counts, inverse = _voxelize_points(pts, float(voxel_size_m))

    pair_view = np.rec.fromarrays([inverse.astype(np.int32, copy=False), ids], names="vox,frame")
    uniq_pairs = np.unique(pair_view)
    support_counts = np.bincount(uniq_pairs["vox"], minlength=uniq_coords.shape[0]).astype(np.int32, copy=False)

    keep_mask = support_counts >= max(1, int(min_support))
    kept_points = centroids[keep_mask]
    kept_vox = uniq_coords[keep_mask]
    kept_support = support_counts[keep_mask]
    kept_total_counts = total_counts[keep_mask]

    component_meta = {
        "connected_components": int(1 if kept_points.shape[0] > 0 else 0),
        "largest_component_voxels": int(kept_points.shape[0]),
    }
    if bool(keep_largest_component) and kept_points.shape[0] > 0:
        component_mask, component_meta = _largest_component_mask(kept_vox)
        kept_points = kept_points[component_mask]
        kept_vox = kept_vox[component_mask]
        kept_support = kept_support[component_mask]
        kept_total_counts = kept_total_counts[component_mask]

    component_count_before_core_clip = int(kept_support.shape[0])
    static_core_clip_meta: dict[str, Any] = {"enabled": False}
    if bool(core_clip) and kept_points.shape[0] > 0:
        kept_points, kept_vox, kept_support, kept_total_counts, static_core_clip_meta = _core_clip_canonical_voxels(
            kept_points,
            kept_vox,
            kept_support,
            kept_total_counts,
            min_support=int(min_support),
            keep_largest_component=bool(keep_largest_component),
            min_canonical_points=int(min_canonical_points),
        )

    if int(max_points) > 0 and kept_points.shape[0] > int(max_points):
        order = np.lexsort((kept_total_counts, kept_support))[::-1]
        keep_idx = order[: int(max_points)]
        kept_points = kept_points[keep_idx]
        kept_support = kept_support[keep_idx]
        kept_total_counts = kept_total_counts[keep_idx]

    support_hist = _support_histogram_dict(kept_support)
    return kept_points.astype(np.float32, copy=False), {
        "canonical_raw_point_count": int(pts.shape[0]),
        "canonical_voxel_count_before_support": int(centroids.shape[0]),
        "canonical_voxel_count_after_support": int(np.count_nonzero(keep_mask)),
        "canonical_voxel_count_after_component": int(component_count_before_core_clip),
        "canonical_voxel_count_after_core_clip": int(kept_support.shape[0]),
        "canonical_point_count_after_cap": int(kept_points.shape[0]),
        "canonical_connected_components": int(component_meta["connected_components"]),
        "canonical_largest_component_voxels": int(component_meta["largest_component_voxels"]),
        "canonical_support_histogram": support_hist,
        "canonical_support_min": int(kept_support.min()) if kept_support.size else 0,
        "canonical_support_median": float(np.median(kept_support)) if kept_support.size else 0.0,
        "canonical_support_p90": float(np.percentile(kept_support, 90.0)) if kept_support.size else 0.0,
        "canonical_support_max": int(kept_support.max()) if kept_support.size else 0,
        "canonical_cap_strategy": "support_then_density",
        "static_core_clip_meta": static_core_clip_meta,
    }


def _write_ply_xyz(path: Path, points_xyz: np.ndarray) -> None:
    pts = np.asarray(points_xyz, dtype=np.float32)
    _ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {pts.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for x, y, z in pts:
            f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")


def _backproject(
    depth: np.ndarray,
    mask_u8: np.ndarray,
    K: np.ndarray,
    *,
    depth_mode: str,
    depth_min_m: float,
    depth_max_m: float,
    max_points: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if depth.shape != mask_u8.shape:
        raise ValueError(f"depth/mask shape mismatch: depth={depth.shape} mask={mask_u8.shape}")

    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])
    if fx <= 0 or fy <= 0:
        raise ValueError(f"Invalid intrinsics: fx={fx} fy={fy}")

    m = mask_u8 > 0
    d = np.asarray(depth, dtype=np.float32)
    valid = m & np.isfinite(d) & (d > float(depth_min_m)) & (d < float(depth_max_m))
    if not np.any(valid):
        return np.zeros((0, 3), dtype=np.float32)

    v, u = np.where(valid)
    u_f = u.astype(np.float32)
    v_f = v.astype(np.float32)
    d_f = d[v, u].astype(np.float32)

    x_n = (u_f - cx) / fx
    y_n = (v_f - cy) / fy

    if depth_mode == "z":
        z = d_f
        x = x_n * z
        y = y_n * z
    elif depth_mode == "range":
        ray_norm = np.sqrt(x_n * x_n + y_n * y_n + 1.0)
        z = d_f / ray_norm
        x = x_n * z
        y = y_n * z
    else:
        raise ValueError(f"Unknown depth_mode: {depth_mode}")

    pts = np.stack([x, y, z], axis=1).astype(np.float32)
    return _maybe_subsample(pts, int(max_points), rng)


def _transform_points(T_dst_from_src: np.ndarray, points_src: np.ndarray) -> np.ndarray:
    R = np.asarray(T_dst_from_src[:3, :3], dtype=np.float32)
    t = np.asarray(T_dst_from_src[:3, 3], dtype=np.float32)
    pts = np.asarray(points_src, dtype=np.float32)
    return (pts @ R.T) + t[None, :]


def _target_position(target_meta: dict[str, Any], t_sec: float) -> np.ndarray:
    traj = str(target_meta.get("traj") or "")
    center = _normalize_xyz(target_meta.get("traj_center") or [0.0, 0.0, 0.0], field_name="target.traj_center")
    radius = _float_or(target_meta.get("traj_radius"), 0.0)
    period = max(1e-6, _float_or(target_meta.get("traj_period"), 1.0))

    if traj in {"static", "static_spin_yaw_pitch"}:
        return center
    if traj in {"circle_xz", "circle_xz_spin_yaw_pitch"}:
        theta = 2.0 * math.pi * (float(t_sec) / period)
        delta = np.array([radius * math.cos(theta), 0.0, radius * math.sin(theta)], dtype=np.float32)
        return center + delta
    if traj == "line_x":
        s = math.sin(2.0 * math.pi * (float(t_sec) / period))
        return center + np.array([radius * s, 0.0, 0.0], dtype=np.float32)
    if traj == "line_y":
        s = math.sin(2.0 * math.pi * (float(t_sec) / period))
        return center + np.array([0.0, radius * s, 0.0], dtype=np.float32)

    raise SystemExit(f'Unsupported target trajectory for recon_spin_points.py: "{traj}"')


def _target_rotation(target_meta: dict[str, Any], t_sec: float, capture_seconds: float) -> np.ndarray:
    traj = str(target_meta.get("traj") or "")
    if "spin" not in traj:
        return np.eye(3, dtype=np.float32)

    yaw_start_deg = _float_or(target_meta.get("yaw_start_deg"), 0.0)
    yaw_end_deg = _float_or(target_meta.get("yaw_end_deg"), 0.0)
    pitch_amp_deg = _float_or(target_meta.get("pitch_amp_deg"), 0.0)
    pitch_period = max(1e-6, _float_or(target_meta.get("pitch_period"), capture_seconds if capture_seconds > 0 else 1.0))

    alpha = 0.0 if capture_seconds <= 1e-6 else min(max(float(t_sec) / float(capture_seconds), 0.0), 1.0)
    yaw_deg = yaw_start_deg + alpha * (yaw_end_deg - yaw_start_deg)
    pitch_deg = pitch_amp_deg * math.sin(2.0 * math.pi * (float(t_sec) / pitch_period))
    return _rot_z_deg(yaw_deg) @ _rot_x_deg(pitch_deg)


def _pose_from_capture_meta(capture_meta: dict[str, Any], t_sec: float) -> tuple[np.ndarray, np.ndarray]:
    render_meta = dict(capture_meta.get("render") or {})
    target_meta = dict(capture_meta.get("target") or {})
    capture_seconds = _float_or(render_meta.get("seconds"), 0.0)
    pos = _target_position(target_meta, t_sec)
    R = _target_rotation(target_meta, t_sec, capture_seconds)
    T_node_from_target = _make_T(R, pos)
    return T_node_from_target, _invert_T(T_node_from_target)


def _build_points_node_for_stem(
    scene_dir: Path,
    *,
    rig_cams: dict[str, Any],
    cams: list[str],
    stem: str,
    depth_subdir: str,
    mask_subdir: str,
    depth_mode: str,
    depth_min_m: float,
    depth_max_m: float,
    max_points_per_cam: int,
    voxel_size_m: float,
    mask_erode_px: int,
    depth_scale_by_cam: dict[str, float] | None,
    depth_scale_by_stem_cam: dict[str, dict[str, float]] | None,
    allowed_cams: set[str] | None,
    apply_predicted_static_cleanup: bool,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, int]]:
    pts_all: list[np.ndarray] = []
    per_cam_counts: dict[str, int] = {}

    for cam_id in cams:
        if allowed_cams is not None and cam_id not in allowed_cams:
            per_cam_counts[cam_id] = 0
            continue
        cam_entry = dict(rig_cams.get(cam_id) or {})
        K = np.asarray(cam_entry.get("K"), dtype=np.float32)
        if K.shape != (3, 3):
            raise SystemExit(f'Invalid K shape for "{cam_id}": {K.shape}')
        T_node_from_cam = np.asarray(cam_entry.get("T_node_from_cam"), dtype=np.float32)
        if T_node_from_cam.shape != (4, 4):
            raise SystemExit(f'Invalid T_node_from_cam shape for "{cam_id}": {T_node_from_cam.shape}')

        depth_path = scene_dir / "cams" / cam_id / str(depth_subdir) / f"{stem}.npy"
        mask_path = _resolve_mask_path(scene_dir, cam_id, str(mask_subdir), stem)
        if not depth_path.exists():
            raise SystemExit(f"Missing depth: {depth_path}")
        if mask_path is None:
            raise SystemExit(f'Missing mask for "{cam_id}" stem={stem} under {scene_dir / "cams" / cam_id / str(mask_subdir)}')

        depth = _read_depth_npy(depth_path)
        frame_scale = float(((depth_scale_by_stem_cam or {}).get(stem) or {}).get(cam_id, 1.0))
        if not math.isfinite(frame_scale) or frame_scale <= 0:
            frame_scale = 1.0
        depth_scale = frame_scale * float((depth_scale_by_cam or {}).get(cam_id, 1.0))
        if not math.isfinite(depth_scale) or depth_scale <= 0:
            depth_scale = 1.0
        if abs(depth_scale - 1.0) > 1e-6:
            depth = np.asarray(depth, dtype=np.float32) * float(depth_scale)
        mask_u8 = _maybe_erode_mask(_read_mask_u8(mask_path), int(mask_erode_px))

        try:
            w_exp, h_exp = int(cam_entry["image_size"][0]), int(cam_entry["image_size"][1])
            if (depth.shape[1], depth.shape[0]) != (w_exp, h_exp):
                raise SystemExit(f'Shape mismatch for "{cam_id}": depth={depth.shape} rig.image_size={[w_exp, h_exp]}')
        except KeyError:
            pass

        pts_cam = _backproject(
            depth,
            mask_u8,
            K,
            depth_mode=str(depth_mode),
            depth_min_m=float(depth_min_m),
            depth_max_m=float(depth_max_m),
            max_points=int(max_points_per_cam),
            rng=rng,
        )
        per_cam_counts[cam_id] = int(pts_cam.shape[0])
        if pts_cam.shape[0] == 0:
            continue
        pts_all.append(_transform_points(T_node_from_cam, pts_cam))

    if not pts_all:
        return np.zeros((0, 3), dtype=np.float32), per_cam_counts

    fused = np.concatenate(pts_all, axis=0).astype(np.float32, copy=False)
    fused = _voxel_downsample_first(fused, float(voxel_size_m))
    if bool(apply_predicted_static_cleanup) and fused.shape[0] > 0:
        fused, _ = _cleanup_predicted_static_frame_points(
            fused,
            voxel_size_m=float(voxel_size_m),
        )
    return fused, per_cam_counts


def _parse_scale_candidates(raw_value: str) -> list[float]:
    values: list[float] = []
    for token in str(raw_value).split(","):
        token = str(token).strip()
        if not token:
            continue
        try:
            scale = float(token)
        except ValueError:
            continue
        if math.isfinite(scale) and scale > 0:
            values.append(float(scale))
    return values


def _select_probe_rows(
    rows: list[tuple[int, dict[str, str]]],
    *,
    max_frames: int,
) -> list[tuple[int, dict[str, str]]]:
    if len(rows) <= int(max_frames):
        return list(rows)
    idx = np.linspace(0, len(rows) - 1, num=max(1, int(max_frames)), dtype=np.int32)
    unique_idx = sorted(set(int(i) for i in idx.tolist()))
    return [rows[i] for i in unique_idx]


def _estimate_static_frame_depth_norms(
    scene_dir: Path,
    *,
    cams: list[str],
    selected_rows: list[tuple[int, dict[str, str]]],
    depth_subdir: str,
    mask_subdir: str,
    scale_clip_min: float,
    scale_clip_max: float,
) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    min_valid_pixels = 32
    clip_min = float(scale_clip_min)
    clip_max = float(scale_clip_max)
    if clip_min <= 0:
        clip_min = 0.5
    if clip_max <= 0 or clip_max < clip_min:
        clip_max = max(clip_min, 2.0)

    scale_by_stem_cam: dict[str, dict[str, float]] = {}
    meta: dict[str, Any] = {
        "enabled": True,
        "min_valid_pixels": int(min_valid_pixels),
        "scale_clip_min": float(clip_min),
        "scale_clip_max": float(clip_max),
        "per_cam": {},
    }

    for cam_id in cams:
        frame_stats: list[tuple[str, float]] = []
        for ts_us, _ in selected_rows:
            stem = f"{ts_us:012d}"
            depth_path = scene_dir / "cams" / cam_id / str(depth_subdir) / f"{stem}.npy"
            mask_path = _resolve_mask_path(scene_dir, cam_id, str(mask_subdir), stem)
            if not depth_path.exists() or mask_path is None:
                continue

            depth = _read_depth_npy(depth_path)
            mask_u8 = _read_mask_u8(mask_path)
            valid = (mask_u8 > 0) & np.isfinite(depth) & (depth > 1e-6)
            if int(np.count_nonzero(valid)) < int(min_valid_pixels):
                continue

            depth_stat = float(np.median(depth[valid].astype(np.float32, copy=False)))
            if not math.isfinite(depth_stat) or depth_stat <= 0:
                continue
            frame_stats.append((stem, depth_stat))

        if not frame_stats:
            continue

        per_frame_values = np.asarray([value for _, value in frame_stats], dtype=np.float32)
        reference_value = float(np.median(per_frame_values))
        if not math.isfinite(reference_value) or reference_value <= 0:
            continue

        resolved_scales: list[float] = []
        for stem, frame_value in frame_stats:
            scale = float(reference_value / max(frame_value, 1e-6))
            scale = min(max(scale, clip_min), clip_max)
            scale_by_stem_cam.setdefault(stem, {})[cam_id] = float(scale)
            resolved_scales.append(float(scale))

        scales_arr = np.asarray(resolved_scales, dtype=np.float32)
        meta["per_cam"][cam_id] = {
            "reference_depth_stat": float(reference_value),
            "valid_frames": int(len(frame_stats)),
            "frame_depth_stat_p10": float(np.percentile(per_frame_values, 10.0)),
            "frame_depth_stat_p90": float(np.percentile(per_frame_values, 90.0)),
            "frame_scale_min": float(scales_arr.min()) if scales_arr.size else 1.0,
            "frame_scale_median": float(np.median(scales_arr)) if scales_arr.size else 1.0,
            "frame_scale_max": float(scales_arr.max()) if scales_arr.size else 1.0,
        }

    if not meta["per_cam"]:
        meta["enabled"] = False
        meta["reason"] = "no_valid_frame_depth_stats"
    return scale_by_stem_cam, meta


def _estimate_static_depth_scales(
    scene_dir: Path,
    *,
    rig_cams: dict[str, Any],
    cams: list[str],
    selected_rows: list[tuple[int, dict[str, str]]],
    capture_meta: dict[str, Any],
    depth_subdir: str,
    mask_subdir: str,
    depth_mode: str,
    depth_min_m: float,
    depth_max_m: float,
    voxel_size_m: float,
    canonical_voxel_size_m: float,
    depth_scale_by_stem_cam: dict[str, dict[str, float]] | None,
    scale_candidates: list[float],
    probe_frames: int,
) -> tuple[dict[str, float], set[str], dict[str, Any]]:
    probe_rows = _select_probe_rows(selected_rows, max_frames=int(probe_frames))
    min_support = 2
    min_component_points = 16
    max_component_volume = 8.0
    max_component_extent = 4.5
    max_center_norm = 3.0
    consensus_radius = 1.75

    meta: dict[str, Any] = {
        "enabled": True,
        "probe_frames_requested": int(probe_frames),
        "probe_frames_used": int(len(probe_rows)),
        "scale_candidates": [float(v) for v in scale_candidates],
        "min_support": int(min_support),
        "candidate_filters": {
            "min_component_points": int(min_component_points),
            "max_component_volume": float(max_component_volume),
            "max_component_extent": float(max_component_extent),
            "max_center_norm": float(max_center_norm),
            "consensus_radius": float(consensus_radius),
        },
        "per_cam_candidates": {},
        "selected_scales": {},
        "selected_cams": [],
        "consensus_center": None,
        "consensus_rejected_cams": [],
    }

    best_by_cam: dict[str, dict[str, Any]] = {}
    for cam_id in cams:
        accepted_candidates: list[dict[str, Any]] = []
        for scale in scale_candidates:
            probe_rng = np.random.default_rng(0)
            canonical_chunks: list[np.ndarray] = []
            canonical_frame_ids: list[np.ndarray] = []
            for frame_idx, (ts_us, _) in enumerate(probe_rows):
                stem = f"{ts_us:012d}"
                t_sec = float(ts_us) / 1e6
                _, T_target_from_node = _pose_from_capture_meta(capture_meta, t_sec)
                points_node, _ = _build_points_node_for_stem(
                    scene_dir,
                    rig_cams=rig_cams,
                    cams=cams,
                    stem=stem,
                    depth_subdir=str(depth_subdir),
                    mask_subdir=str(mask_subdir),
                    depth_mode=str(depth_mode),
                    depth_min_m=float(depth_min_m),
                    depth_max_m=float(depth_max_m),
                    max_points_per_cam=0,
                    voxel_size_m=float(voxel_size_m),
                    mask_erode_px=0,
                    depth_scale_by_cam={cam_id: float(scale)},
                    depth_scale_by_stem_cam=depth_scale_by_stem_cam,
                    allowed_cams={cam_id},
                    apply_predicted_static_cleanup=False,
                    rng=probe_rng,
                )
                if points_node.shape[0] == 0:
                    continue
                canonical_chunks.append(_transform_points(T_target_from_node, points_node))
                canonical_frame_ids.append(np.full((points_node.shape[0],), frame_idx, dtype=np.int32))

            if not canonical_chunks:
                continue

            probe_points = np.concatenate(canonical_chunks, axis=0).astype(np.float32, copy=False)
            probe_frame_ids = np.concatenate(canonical_frame_ids, axis=0).astype(np.int32, copy=False)
            canonical_points, canonical_meta = _canonicalize_points(
                probe_points,
                probe_frame_ids,
                voxel_size_m=float(canonical_voxel_size_m),
                min_support=int(min_support),
                keep_largest_component=True,
                max_points=0,
            )
            if canonical_points.shape[0] < int(min_component_points):
                continue

            bbox_min = canonical_points.min(axis=0)
            bbox_max = canonical_points.max(axis=0)
            extent = bbox_max - bbox_min
            volume = float(np.prod(np.maximum(extent, 1e-6)))
            center = np.median(canonical_points, axis=0).astype(np.float32, copy=False)
            center_norm = float(np.linalg.norm(center))
            if (
                float(volume) > float(max_component_volume)
                or float(extent.max()) > float(max_component_extent)
                or float(center_norm) > float(max_center_norm)
            ):
                continue

            score = float(canonical_points.shape[0]) / ((1.0 + volume) * (1.0 + (0.15 * center_norm * center_norm)))
            accepted_candidates.append(
                {
                    "scale": float(scale),
                    "score": float(score),
                    "component_points": int(canonical_points.shape[0]),
                    "component_volume": float(volume),
                    "component_center": center.astype(float).tolist(),
                    "component_extent": extent.astype(float).tolist(),
                    "support_kept_voxels": int(canonical_meta["canonical_voxel_count_after_support"]),
                }
            )

        accepted_candidates.sort(key=lambda item: (float(item["score"]), int(item["component_points"])), reverse=True)
        meta["per_cam_candidates"][cam_id] = accepted_candidates[:5]
        if accepted_candidates:
            best_by_cam[cam_id] = accepted_candidates[0]

    if not best_by_cam:
        meta["reason"] = "no_valid_scale_candidate"
        return {}, set(), meta

    selected_by_cam = dict(best_by_cam)
    if len(selected_by_cam) >= 2:
        centers = np.asarray([selected_by_cam[cam_id]["component_center"] for cam_id in sorted(selected_by_cam)], dtype=np.float32)
        consensus_center = np.median(centers, axis=0).astype(np.float32, copy=False)
        rejected_cams: list[str] = []
        for cam_id, payload in list(selected_by_cam.items()):
            center = np.asarray(payload["component_center"], dtype=np.float32)
            dist = float(np.linalg.norm(center - consensus_center))
            payload["consensus_distance"] = float(dist)
            if dist > float(consensus_radius):
                rejected_cams.append(str(cam_id))
                selected_by_cam.pop(cam_id, None)
        if selected_by_cam:
            meta["consensus_center"] = consensus_center.astype(float).tolist()
            meta["consensus_rejected_cams"] = rejected_cams

    if not selected_by_cam:
        selected_by_cam = dict(best_by_cam)
        meta["consensus_center"] = None
        meta["consensus_rejected_cams"] = []

    selected_scales = {cam_id: float(payload["scale"]) for cam_id, payload in selected_by_cam.items()}
    enabled_cams = set(selected_scales.keys())
    meta["selected_scales"] = {cam_id: float(scale) for cam_id, scale in selected_scales.items()}
    meta["selected_cams"] = sorted(enabled_cams)
    return selected_scales, enabled_cams, meta


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Aggregate spin-scene depth(+mask) across timestamps into a canonical target cloud, then reproject per timestamp."
    )
    ap.add_argument(
        "--scene_dir",
        required=True,
        type=str,
        help="Scene folder like: mvp-demo/data/nodes/<node_id>/scenes/<scene_id>",
    )
    ap.add_argument(
        "--rig_json",
        default="",
        type=str,
        help="Optional override path to rig.json. Default: <scene_dir>/calib/rig.json",
    )
    ap.add_argument(
        "--capture_meta_json",
        default="",
        type=str,
        help="Optional override path to capture_meta.json. Default: <scene_dir>/capture_meta.json",
    )
    ap.add_argument(
        "--cams",
        default="cam0,cam1,cam2",
        type=str,
        help='Comma-separated cam ids. Default: "cam0,cam1,cam2".',
    )
    ap.add_argument(
        "--depth_subdir",
        default="depth_gt",
        type=str,
        help='Depth folder name under each cam. Default: "depth_gt".',
    )
    ap.add_argument(
        "--mask_subdir",
        default="masks_gt",
        type=str,
        help='Mask folder name under each cam. Default: "masks_gt".',
    )
    ap.add_argument(
        "--mask_erode_px",
        default=0,
        type=int,
        help="Optional binary-mask erosion radius in pixels before backprojection. 0=disable.",
    )
    ap.add_argument(
        "--depth_mode",
        default="z",
        choices=["z", "range"],
        help='Interpretation of depth values. "z" or "range".',
    )
    ap.add_argument("--depth_min_m", default=1e-3, type=float, help="Min valid depth (meters).")
    ap.add_argument(
        "--depth_max_m",
        default=0.0,
        type=float,
        help="Optional max depth (meters). 0=use zfar_m from capture_meta.json if available, else +inf.",
    )
    ap.add_argument(
        "--max_points_per_cam",
        default=0,
        type=int,
        help="Optional per-camera random subsample cap. 0=disable.",
    )
    ap.add_argument(
        "--voxel_size_m",
        default=0.02,
        type=float,
        help="Voxel size for per-timestamp fused points before canonical aggregation. 0=disable.",
    )
    ap.add_argument(
        "--canonical_voxel_size_m",
        default=0.015,
        type=float,
        help="Voxel size for the aggregated canonical target cloud. 0=disable.",
    )
    ap.add_argument(
        "--canonical_min_support",
        default=0,
        type=int,
        help="Min number of unique timestamps that must hit a canonical voxel to keep it. 0=auto policy.",
    )
    ap.add_argument(
        "--canonical_keep_largest_component",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Keep only the largest connected canonical voxel component after support filtering.",
    )
    ap.add_argument(
        "--min_canonical_points",
        default=128,
        type=int,
        help="Fail if the canonical cloud has fewer than this many points after downsampling.",
    )
    ap.add_argument(
        "--max_canonical_points",
        default=50000,
        type=int,
        help="Optional cap applied after canonical downsampling. 0=disable.",
    )
    ap.add_argument(
        "--static_auto_depth_scale",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="For predicted static scenes, probe per-camera depth scales that maximize canonical consistency.",
    )
    ap.add_argument(
        "--static_depth_scale_candidates",
        default="0.5,1,2,4,6,8,10,12,16,24,32,48,64,96",
        type=str,
        help='Comma-separated scale candidates used by --static_auto_depth_scale. Default: "0.5,1,2,4,6,8,10,12,16,24,32,48,64,96".',
    )
    ap.add_argument(
        "--static_scale_probe_frames",
        default=96,
        type=int,
        help="Max number of timestamps used to probe static depth scales. 0=all selected timestamps.",
    )
    ap.add_argument(
        "--static_frame_depth_norm",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="For predicted static scenes, normalize each camera stream by the masked-depth median of each frame before global scale probing.",
    )
    ap.add_argument(
        "--static_frame_scale_clip_min",
        default=0.5,
        type=float,
        help="Lower clamp for predicted-static per-frame depth normalization scales.",
    )
    ap.add_argument(
        "--static_frame_scale_clip_max",
        default=2.0,
        type=float,
        help="Upper clamp for predicted-static per-frame depth normalization scales.",
    )
    ap.add_argument("--limit", default=0, type=int, help="Process only first N timestamps. 0=all.")
    ap.add_argument(
        "--ts",
        default="",
        type=str,
        help='Optional single timestamp stem to process (e.g. "000000133333").',
    )
    ap.add_argument("--seed", default=0, type=int, help="RNG seed for subsampling.")
    ap.add_argument(
        "--out_subdir",
        default="recon/points_recon_spin_gt",
        type=str,
        help='Relative output subdir under scene_dir. Default: "recon/points_recon_spin_gt".',
    )
    ap.add_argument("--write_ply", action="store_true", help="Also write per-timestamp and canonical PLYs for debug.")
    args = ap.parse_args()

    scene_dir = Path(args.scene_dir).resolve()
    if not scene_dir.exists():
        raise SystemExit(f"--scene_dir does not exist: {scene_dir}")

    rig_path = Path(args.rig_json).resolve() if str(args.rig_json).strip() else (scene_dir / "calib" / "rig.json")
    if not rig_path.exists():
        raise SystemExit(f"rig.json not found: {rig_path}")
    capture_meta_path = (
        Path(args.capture_meta_json).resolve()
        if str(args.capture_meta_json).strip()
        else (scene_dir / "capture_meta.json")
    )
    if not capture_meta_path.exists():
        raise SystemExit(f"capture_meta.json not found: {capture_meta_path}")

    frame_times_csv = scene_dir / "frame_times.csv"
    if not frame_times_csv.exists():
        raise SystemExit(f"frame_times.csv not found: {frame_times_csv}")

    rig = _load_json(rig_path)
    capture_meta = _load_json(capture_meta_path)

    cams = [c.strip() for c in str(args.cams).split(",") if c.strip()]
    if not cams:
        raise SystemExit("--cams is empty")

    rig_cams = dict(rig.get("cameras") or {})
    for cam_id in cams:
        if cam_id not in rig_cams:
            raise SystemExit(f'Camera "{cam_id}" not found in rig.json. Available: {sorted(rig_cams.keys())}')

    depth_max_m = float(args.depth_max_m)
    if depth_max_m <= 0:
        try:
            depth_max_m = float(capture_meta["render"]["zfar_m"])
        except Exception:
            depth_max_m = float("inf")

    rows = _read_frame_index(frame_times_csv, cams)
    if not rows:
        raise SystemExit(f"No synchronized timestamps found in: {frame_times_csv}")

    selected_rows: list[tuple[int, dict[str, str]]] = []
    ts_filter = str(args.ts).strip()
    for ts_us, by_cam in rows:
        if any(cam_id not in by_cam for cam_id in cams):
            raise SystemExit(f"Incomplete synchronized timestamp in {frame_times_csv}: ts_us={ts_us} cams={sorted(by_cam.keys())}")
        stem = f"{ts_us:012d}"
        if ts_filter and stem != ts_filter:
            continue
        selected_rows.append((ts_us, by_cam))

    if int(args.limit) > 0:
        selected_rows = selected_rows[: int(args.limit)]

    if not selected_rows:
        raise SystemExit("No timestamps selected to process.")

    rng = np.random.default_rng(int(args.seed))
    active_cams: set[str] | None = None
    depth_scale_by_cam: dict[str, float] | None = None
    depth_scale_by_stem_cam: dict[str, dict[str, float]] | None = None
    static_scale_meta: dict[str, Any] | None = None
    static_frame_depth_norm_meta: dict[str, Any] | None = None
    if bool(args.static_auto_depth_scale) and _is_predicted_static_branch(
        capture_meta,
        mask_subdir=str(args.mask_subdir),
        depth_subdir=str(args.depth_subdir),
        out_subdir=str(args.out_subdir),
    ):
        if bool(args.static_frame_depth_norm):
            depth_scale_by_stem_cam, static_frame_depth_norm_meta = _estimate_static_frame_depth_norms(
                scene_dir,
                cams=cams,
                selected_rows=selected_rows,
                depth_subdir=str(args.depth_subdir),
                mask_subdir=str(args.mask_subdir),
                scale_clip_min=float(args.static_frame_scale_clip_min),
                scale_clip_max=float(args.static_frame_scale_clip_max),
            )
            if not depth_scale_by_stem_cam:
                depth_scale_by_stem_cam = None
        scale_candidates = _parse_scale_candidates(str(args.static_depth_scale_candidates))
        if scale_candidates:
            probe_frames = int(args.static_scale_probe_frames) if int(args.static_scale_probe_frames) > 0 else len(selected_rows)
            depth_scale_by_cam, active_cams, static_scale_meta = _estimate_static_depth_scales(
                scene_dir,
                rig_cams=rig_cams,
                cams=cams,
                selected_rows=selected_rows,
                capture_meta=capture_meta,
                depth_subdir=str(args.depth_subdir),
                mask_subdir=str(args.mask_subdir),
                depth_mode=str(args.depth_mode),
                depth_min_m=float(args.depth_min_m),
                depth_max_m=float(depth_max_m),
                voxel_size_m=float(args.voxel_size_m),
                canonical_voxel_size_m=float(args.canonical_voxel_size_m),
                depth_scale_by_stem_cam=depth_scale_by_stem_cam,
                scale_candidates=scale_candidates,
                probe_frames=int(probe_frames),
            )
            if not active_cams:
                active_cams = None
                depth_scale_by_cam = None

    out_rel = Path(str(args.out_subdir))
    out_dir = out_rel if out_rel.is_absolute() else (scene_dir / out_rel)
    ply_rel = out_rel.parent / f"{out_rel.name}_ply"
    ply_dir = ply_rel if ply_rel.is_absolute() else (scene_dir / ply_rel)
    _ensure_dir(out_dir)
    if args.write_ply:
        _ensure_dir(ply_dir)

    canonical_chunks: list[np.ndarray] = []
    canonical_frame_ids: list[np.ndarray] = []
    input_point_rows: list[tuple[str, int, dict[str, int]]] = []
    input_frames_with_points = 0
    predicted_static_cleanup_enabled = bool(
        _is_predicted_static_branch(
            capture_meta,
            mask_subdir=str(args.mask_subdir),
            depth_subdir=str(args.depth_subdir),
            out_subdir=str(args.out_subdir),
        )
    )

    for frame_idx, (ts_us, _) in enumerate(selected_rows):
        stem = f"{ts_us:012d}"
        t_sec = float(ts_us) / 1e6
        _, T_target_from_node = _pose_from_capture_meta(capture_meta, t_sec)
        points_node, per_cam_counts = _build_points_node_for_stem(
            scene_dir,
            rig_cams=rig_cams,
            cams=cams,
            stem=stem,
            depth_subdir=str(args.depth_subdir),
            mask_subdir=str(args.mask_subdir),
            depth_mode=str(args.depth_mode),
            depth_min_m=float(args.depth_min_m),
            depth_max_m=float(depth_max_m),
            max_points_per_cam=int(args.max_points_per_cam),
            voxel_size_m=float(args.voxel_size_m),
            mask_erode_px=int(args.mask_erode_px),
            depth_scale_by_cam=depth_scale_by_cam,
            depth_scale_by_stem_cam=depth_scale_by_stem_cam,
            allowed_cams=active_cams,
            apply_predicted_static_cleanup=False,
            rng=rng,
        )
        input_point_rows.append((stem, int(points_node.shape[0]), per_cam_counts))
        if points_node.shape[0] == 0:
            continue
        input_frames_with_points += 1
        canonical_chunks.append(_transform_points(T_target_from_node, points_node))
        canonical_frame_ids.append(np.full((points_node.shape[0],), frame_idx, dtype=np.int32))

    if not canonical_chunks:
        raise SystemExit("Canonical aggregation failed: all timestamps produced empty point clouds.")

    canonical_points_raw = np.concatenate(canonical_chunks, axis=0).astype(np.float32, copy=False)
    canonical_frame_ids_raw = np.concatenate(canonical_frame_ids, axis=0).astype(np.int32, copy=False)
    static_frame_centroid_gate_meta: dict[str, Any] = {}
    static_canonical_crop_meta: dict[str, Any] = {}
    if bool(predicted_static_cleanup_enabled):
        canonical_points_raw, canonical_frame_ids_raw, static_frame_centroid_gate_meta = _filter_predicted_static_canonical_frames(
            canonical_points_raw,
            canonical_frame_ids_raw,
        )
        crop_keep_mask = np.ones((canonical_points_raw.shape[0],), dtype=bool)
        canonical_points_cropped, static_canonical_crop_meta = _coarse_crop_predicted_static_canonical_points(
            canonical_points_raw,
            gate_meta=static_frame_centroid_gate_meta,
        )
        if canonical_points_cropped.shape[0] != canonical_points_raw.shape[0]:
            center_value = np.asarray(static_canonical_crop_meta.get("crop_center") or [0.0, 0.0, 0.0], dtype=np.float32).reshape(3)
            bounds = dict(static_canonical_crop_meta.get("bounds") or {})
            dx = canonical_points_raw[:, 0] - float(center_value[0])
            dy = canonical_points_raw[:, 1] - float(center_value[1])
            dz = canonical_points_raw[:, 2] - float(center_value[2])
            crop_keep_mask = (
                (np.abs(dx) <= float(bounds.get("x_abs", float("inf"))))
                & (np.abs(dz) <= float(bounds.get("z_abs", float("inf"))))
                & (dy >= (-1.0 * float(bounds.get("y_low", float("inf")))))
                & (dy <= float(bounds.get("y_high", float("inf"))))
            )
        canonical_points_raw = canonical_points_raw[crop_keep_mask].astype(np.float32, copy=False)
        canonical_frame_ids_raw = canonical_frame_ids_raw[crop_keep_mask].astype(np.int32, copy=False)
    support_candidates, support_policy = _canonical_support_candidates(
        capture_meta,
        mask_subdir=str(args.mask_subdir),
        depth_subdir=str(args.depth_subdir),
        out_subdir=str(args.out_subdir),
        requested_support=int(args.canonical_min_support),
    )
    canonical_attempts: list[dict[str, Any]] = []
    canonical_points = np.zeros((0, 3), dtype=np.float32)
    canonical_meta: dict[str, Any] = {}
    canonical_component_fallback_used = False
    resolved_canonical_support = int(support_candidates[-1])
    static_structure_debug: dict[str, np.ndarray] = {
        "raw": np.zeros((0, 3), dtype=np.float32),
        "core": np.zeros((0, 3), dtype=np.float32),
        "shell": np.zeros((0, 3), dtype=np.float32),
    }

    for support_candidate in support_candidates:
        if bool(predicted_static_cleanup_enabled):
            candidate_points, candidate_meta, candidate_debug = _canonicalize_predicted_static_core_shell(
                canonical_points_raw,
                canonical_frame_ids_raw,
                voxel_size_m=float(args.canonical_voxel_size_m),
                min_support=int(support_candidate),
                keep_largest_component=bool(args.canonical_keep_largest_component),
                max_points=int(args.max_canonical_points),
                min_canonical_points=int(args.min_canonical_points),
            )
            fallback_used = bool(candidate_meta.get("component_fallback_used", False))
        else:
            candidate_points, candidate_meta = _canonicalize_points(
                canonical_points_raw,
                canonical_frame_ids_raw,
                voxel_size_m=float(args.canonical_voxel_size_m),
                min_support=int(support_candidate),
                keep_largest_component=bool(args.canonical_keep_largest_component),
                max_points=int(args.max_canonical_points),
            )
            candidate_debug = {
                "raw": np.zeros((0, 3), dtype=np.float32),
                "core": np.zeros((0, 3), dtype=np.float32),
                "shell": np.zeros((0, 3), dtype=np.float32),
            }
            fallback_used = False
            if candidate_points.shape[0] < int(args.min_canonical_points) and bool(args.canonical_keep_largest_component):
                fallback_points, fallback_meta = _canonicalize_points(
                    canonical_points_raw,
                    canonical_frame_ids_raw,
                    voxel_size_m=float(args.canonical_voxel_size_m),
                    min_support=int(support_candidate),
                    keep_largest_component=False,
                    max_points=int(args.max_canonical_points),
                )
                if fallback_points.shape[0] >= int(args.min_canonical_points):
                    candidate_points = fallback_points
                    candidate_meta = fallback_meta
                    fallback_used = True

        canonical_attempts.append(
            {
                "min_support": int(support_candidate),
                "point_count": int(candidate_points.shape[0]),
                "component_fallback_used": bool(fallback_used),
            }
        )
        canonical_points = candidate_points
        canonical_meta = candidate_meta
        canonical_component_fallback_used = bool(fallback_used)
        resolved_canonical_support = int(support_candidate)
        static_structure_debug = candidate_debug
        if canonical_points.shape[0] >= int(args.min_canonical_points):
            break
    if canonical_points.shape[0] < int(args.min_canonical_points):
        raise SystemExit(
            f"Canonical cloud too small after downsampling: {canonical_points.shape[0]} < --min_canonical_points={args.min_canonical_points}"
        )

    index_rows: list[tuple[str, int]] = []
    for ts_us, _ in selected_rows:
        stem = f"{ts_us:012d}"
        t_sec = float(ts_us) / 1e6
        T_node_from_target, _ = _pose_from_capture_meta(capture_meta, t_sec)
        recon_points = _transform_points(T_node_from_target, canonical_points)
        out_path = out_dir / f"{stem}.npy"
        np.save(str(out_path), recon_points.astype(np.float32))
        if args.write_ply:
            _write_ply_xyz(ply_dir / f"{stem}.ply", recon_points)
        index_rows.append((stem, int(recon_points.shape[0])))

    np.save(str(out_dir / "canonical_target.npy"), canonical_points.astype(np.float32, copy=False))
    if bool(predicted_static_cleanup_enabled):
        np.save(str(out_dir / "canonical_raw.npy"), np.asarray(static_structure_debug.get("raw"), dtype=np.float32))
        np.save(str(out_dir / "canonical_core.npy"), np.asarray(static_structure_debug.get("core"), dtype=np.float32))
        np.save(str(out_dir / "canonical_shell.npy"), np.asarray(static_structure_debug.get("shell"), dtype=np.float32))
    if args.write_ply:
        _write_ply_xyz(ply_dir / "canonical_target.ply", canonical_points)

    target_meta = dict(capture_meta.get("target") or {})
    meta = {
        "scene_dir": str(scene_dir),
        "rig_json": str(rig_path),
        "capture_meta_json": str(capture_meta_path),
        "cams": cams,
        "out_subdir": str(args.out_subdir),
        "depth_subdir": str(args.depth_subdir),
        "mask_subdir": str(args.mask_subdir),
        "mask_erode_px": int(args.mask_erode_px),
        "static_auto_depth_scale": bool(args.static_auto_depth_scale),
        "static_scale_probe_frames": int(args.static_scale_probe_frames),
        "static_depth_scale_candidates": _parse_scale_candidates(str(args.static_depth_scale_candidates)),
        "static_frame_depth_norm": bool(args.static_frame_depth_norm),
        "static_frame_scale_clip_min": float(args.static_frame_scale_clip_min),
        "static_frame_scale_clip_max": float(args.static_frame_scale_clip_max),
        "predicted_static_cleanup_enabled": bool(predicted_static_cleanup_enabled),
        "depth_mode": str(args.depth_mode),
        "depth_min_m": float(args.depth_min_m),
        "depth_max_m": float(depth_max_m),
        "max_points_per_cam": int(args.max_points_per_cam),
        "voxel_size_m": float(args.voxel_size_m),
        "canonical_voxel_size_m": float(args.canonical_voxel_size_m),
        "canonical_min_support_requested": int(args.canonical_min_support),
        "canonical_min_support": int(resolved_canonical_support),
        "canonical_min_support_policy": str(support_policy),
        "canonical_min_support_attempts": canonical_attempts,
        "canonical_keep_largest_component": bool(args.canonical_keep_largest_component),
        "canonical_component_fallback_used": bool(canonical_component_fallback_used),
        "min_canonical_points": int(args.min_canonical_points),
        "max_canonical_points": int(args.max_canonical_points),
        "count": int(len(index_rows)),
        "frames_with_input_points": int(input_frames_with_points),
        "frames_without_input_points": int(len(index_rows) - input_frames_with_points),
        "canonical_point_count": int(canonical_points.shape[0]),
        "canonical_raw_point_count": int(canonical_meta["canonical_raw_point_count"]),
        "canonical_voxel_count_before_support": int(canonical_meta["canonical_voxel_count_before_support"]),
        "canonical_voxel_count_after_support": int(canonical_meta["canonical_voxel_count_after_support"]),
        "canonical_voxel_count_after_component": int(canonical_meta["canonical_voxel_count_after_component"]),
        "canonical_voxel_count_after_core_clip": int(canonical_meta.get("canonical_voxel_count_after_core_clip", canonical_meta["canonical_voxel_count_after_component"])),
        "canonical_point_count_after_cap": int(canonical_meta["canonical_point_count_after_cap"]),
        "canonical_connected_components": int(canonical_meta["canonical_connected_components"]),
        "canonical_largest_component_voxels": int(canonical_meta["canonical_largest_component_voxels"]),
        "canonical_support_histogram": dict(canonical_meta["canonical_support_histogram"]),
        "canonical_support_min": int(canonical_meta["canonical_support_min"]),
        "canonical_support_median": float(canonical_meta["canonical_support_median"]),
        "canonical_support_p90": float(canonical_meta["canonical_support_p90"]),
        "canonical_support_max": int(canonical_meta["canonical_support_max"]),
        "canonical_cap_strategy": str(canonical_meta["canonical_cap_strategy"]),
        "raw_extent_xyz": list(canonical_meta.get("raw_extent_xyz") or [0.0, 0.0, 0.0]),
        "core_extent_xyz": list(canonical_meta.get("core_extent_xyz") or [0.0, 0.0, 0.0]),
        "shell_extent_xyz": list(canonical_meta.get("shell_extent_xyz") or [0.0, 0.0, 0.0]),
        "core_shell_ratio": canonical_meta.get("core_shell_ratio"),
        "shell_added_points": int(canonical_meta.get("shell_added_points", 0)),
        "y_downshift_vs_raw": float(canonical_meta.get("y_downshift_vs_raw", 0.0)),
        "traj": str(target_meta.get("traj") or ""),
        "yaw_start_deg": _float_or(target_meta.get("yaw_start_deg"), 0.0),
        "yaw_end_deg": _float_or(target_meta.get("yaw_end_deg"), 0.0),
        "pitch_amp_deg": _float_or(target_meta.get("pitch_amp_deg"), 0.0),
        "pitch_period": _float_or(target_meta.get("pitch_period"), 0.0),
        "active_cams": sorted(active_cams) if active_cams is not None else list(cams),
        "depth_scale_by_cam": {cam_id: float(scale) for cam_id, scale in (depth_scale_by_cam or {}).items()},
        "static_frame_depth_norm_meta": static_frame_depth_norm_meta or {},
        "static_depth_scale_meta": static_scale_meta or {},
        "static_frame_centroid_gate_meta": static_frame_centroid_gate_meta or {},
        "static_temporal_cleanup_meta": static_frame_centroid_gate_meta or {},
        "static_core_clip_meta": dict(canonical_meta.get("static_core_clip_meta") or {}),
        "static_canonical_crop_meta": static_canonical_crop_meta or {},
        "static_canonical_augment_meta": {},
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    with (out_dir / "points_index.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ts_stem", "n_points", "filename"])
        for stem, n_points in index_rows:
            w.writerow([stem, n_points, f"{stem}.npy"])

    with (out_dir / "input_points_index.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ts_stem", "n_input_points", *[f"{cam_id}_points" for cam_id in cams]])
        for stem, n_points, per_cam_counts in input_point_rows:
            w.writerow([stem, n_points, *[int(per_cam_counts.get(cam_id, 0)) for cam_id in cams]])

    print(f"[ok] scene={scene_dir.name}")
    print(f"[ok] out_dir={out_dir}")
    print(f"[ok] timestamps={len(index_rows)} canonical_points={canonical_points.shape[0]}")


if __name__ == "__main__":
    main()
