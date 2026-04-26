from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from backproject_neoverse_observations import (
    _dilate_mask as _obs_dilate_mask,
    _parse_crop_box_json as _obs_parse_crop_box_json,
    _prepare_mask as _obs_prepare_mask,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_scene_points_root(root_value: str, scene_id: str, repo: Path) -> Path:
    root = Path(str(root_value))
    if not root.is_absolute():
        root = repo / root
    return root / scene_id / "points_per_view"


def _resolve_scene_output_root(root_value: str, scene_id: str, repo: Path, leaf: str) -> Path:
    root = Path(str(root_value))
    if not root.is_absolute():
        root = repo / root
    return root / scene_id / leaf


def _resolve_optional_path(path_value: str | Path | None, repo: Path, fallback: Path) -> Path:
    if path_value is None:
        return fallback
    path = Path(str(path_value))
    if not path.is_absolute():
        path = repo / path
    return path


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"Failed to read json: {path}\nError: {exc!r}")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _read_gray(path: Path) -> np.ndarray:
    try:
        return np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
    except Exception as exc:
        raise SystemExit(f"Failed to read image: {path}\nError: {exc!r}")


def _read_npy(path: Path) -> np.ndarray:
    try:
        return np.asarray(np.load(str(path)), dtype=np.float32)
    except Exception as exc:
        raise SystemExit(f"Failed to load npy: {path}\nError: {exc!r}")


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _save_npy(path: Path, points_xyz: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(points_xyz, dtype=np.float32))


def _mask_path(scene_dir: Path, cam_id: str, scene_stem: str) -> Path:
    candidates = [
        scene_dir / "cams" / cam_id / "masks_gt" / f"{scene_stem}.png",
        scene_dir / "cams" / cam_id / "masks" / f"{scene_stem}.png",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise SystemExit(f"Missing mask for {cam_id}/{scene_stem}; checked: {[p.as_posix() for p in candidates]}")


def _load_fg_points_by_key(points_root: Path) -> dict[tuple[str, int, str], Path]:
    index_path = points_root / "points_index.csv"
    if not index_path.exists():
        raise SystemExit(f"Missing fg points index: {index_path}")
    rows = _read_csv_rows(index_path)
    mapping: dict[tuple[str, int, str], Path] = {}
    for row in rows:
        key = (str(row["cam_id"]), int(float(row["logical_t_idx"])), str(row["scene_stem"]))
        mapping[key] = points_root / str(row["fg_path"])
    return mapping


def _load_observation_rows(observations_root: Path, cams: list[str]) -> dict[tuple[str, int, str], dict[str, str]]:
    index_path = observations_root / "index.csv"
    if not index_path.exists():
        raise SystemExit(f"Missing observations index: {index_path}")
    mapping: dict[tuple[str, int, str], dict[str, str]] = {}
    for row in _read_csv_rows(index_path):
        cam_id = str(row.get("cam_id", "")).strip()
        if cam_id not in cams:
            continue
        key = (cam_id, int(float(row["logical_t_idx"])), str(row["scene_stem"]))
        if key in mapping:
            raise SystemExit(f"Duplicate observation row for key={key} in {index_path}")
        mapping[key] = row
    return mapping


def _load_dynamic_rows(dynamic_index_path: Path) -> list[dict[str, str]]:
    rows = _read_csv_rows(dynamic_index_path)
    if not rows:
        raise SystemExit(f"No rows found in: {dynamic_index_path}")
    rows.sort(key=lambda row: (int(float(row.get("logical_t_idx", 0))), str(row.get("scene_stem", ""))))
    return rows


def _load_rig(scene_dir: Path, cams: list[str]) -> dict[str, dict[str, np.ndarray]]:
    rig_path = scene_dir / "calib" / "rig.json"
    if not rig_path.exists():
        raise SystemExit(f"Missing rig json: {rig_path}")
    rig = _load_json(rig_path)
    cameras = rig.get("cameras", {})
    out: dict[str, dict[str, np.ndarray]] = {}
    for cam_id in cams:
        cam_meta = cameras.get(cam_id)
        if cam_meta is None:
            raise SystemExit(f"Camera {cam_id} missing in rig.json")
        K = np.asarray(cam_meta.get("K"), dtype=np.float32)
        c2w = np.asarray(cam_meta.get("T_node_from_cam"), dtype=np.float32)
        if K.shape != (3, 3):
            raise SystemExit(f"Invalid K shape for {cam_id}: {K.shape}")
        if c2w.shape != (4, 4):
            raise SystemExit(f"Invalid T_node_from_cam shape for {cam_id}: {c2w.shape}")
        out[cam_id] = {"K": K, "c2w": c2w}
    return out


def _load_points_contract(points_root: Path, repo: Path) -> tuple[dict[str, Any], Path, bool]:
    meta_path = points_root / "meta.json"
    if not meta_path.exists():
        raise SystemExit(f"Missing points_per_view meta: {meta_path}")
    meta = _load_json(meta_path)
    point_coordinate_frame = str(meta.get("point_coordinate_frame") or "").strip()
    legacy_frame_contract_assumed = False
    if not point_coordinate_frame:
        point_coordinate_frame = "neoverse_render_world"
        legacy_frame_contract_assumed = True
    if point_coordinate_frame != "neoverse_render_world":
        raise SystemExit(
            "Unsupported points_per_view coordinate contract: "
            f"{point_coordinate_frame!r}. Expected 'neoverse_render_world'."
        )
    observations_root = _resolve_optional_path(
        meta.get("observations_root"),
        repo=repo,
        fallback=points_root.parent / "observations",
    )
    if not observations_root.exists():
        raise SystemExit(f"Missing observations root inferred from points_per_view meta: {observations_root}")
    return meta, observations_root, legacy_frame_contract_assumed


def _compute_roi_bounds(points_world: np.ndarray, padding_m: float) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points_world, dtype=np.float32)
    if pts.size == 0:
        raise SystemExit("Cannot compute ROI bounds on empty points")
    pad = float(max(padding_m, 0.0))
    roi_min = pts.min(axis=0) - pad
    roi_max = pts.max(axis=0) + pad
    roi_max = np.maximum(roi_max, roi_min + 1e-6)
    return roi_min.astype(np.float32), roi_max.astype(np.float32)


def _grid_dims_for_bounds(roi_min: np.ndarray, roi_max: np.ndarray, voxel_size_m: float) -> np.ndarray:
    size = float(voxel_size_m)
    if size <= 0:
        raise SystemExit(f"Invalid voxel_size_m: {voxel_size_m}")
    extents = np.maximum(np.asarray(roi_max, dtype=np.float32) - np.asarray(roi_min, dtype=np.float32), 1e-6)
    dims = np.ceil(extents / size).astype(np.int32)
    dims = np.maximum(dims, 1)
    return dims


def _choose_roi_voxel_size(
    roi_min: np.ndarray,
    roi_max: np.ndarray,
    base_voxel_size_m: float,
    max_roi_voxels: int,
) -> tuple[float, np.ndarray]:
    voxel_size = float(base_voxel_size_m)
    limit = max(int(max_roi_voxels), 1)
    for _ in range(64):
        dims = _grid_dims_for_bounds(roi_min, roi_max, voxel_size)
        if int(np.prod(dims, dtype=np.int64)) <= limit:
            return float(voxel_size), dims
        voxel_size *= 1.25
    raise SystemExit(
        "Failed to adapt ROI voxel size within limit; "
        f"roi_min={roi_min.tolist()} roi_max={roi_max.tolist()} max_roi_voxels={limit}"
    )


def _flatten_grid(roi_min: np.ndarray, dims: np.ndarray, voxel_size_m: float) -> np.ndarray:
    grid_idx = np.indices(tuple(int(v) for v in dims.tolist()), dtype=np.int32)
    ijk = np.stack(grid_idx, axis=-1).reshape(-1, 3)
    centers = np.asarray(roi_min, dtype=np.float32)[None, :] + (ijk.astype(np.float32) + 0.5) * float(voxel_size_m)
    return centers.astype(np.float32)


def _mask_bbox(mask_u8: np.ndarray) -> list[int] | None:
    ys, xs = np.nonzero(np.asarray(mask_u8, dtype=np.uint8) > 0)
    if xs.size == 0 or ys.size == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def _bbox_center_xy(bbox_xyxy: list[int]) -> tuple[float, float]:
    x0, y0, x1, y1 = [int(v) for v in bbox_xyxy]
    return 0.5 * float(x0 + x1), 0.5 * float(y0 + y1)


def _camera_ray_world(K: np.ndarray, c2w: np.ndarray, u: float, v: float) -> tuple[np.ndarray, np.ndarray]:
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])
    if fx <= 0 or fy <= 0:
        raise SystemExit(f"Invalid camera intrinsics for ray build: fx={fx}, fy={fy}")
    direction_cam = np.asarray([(float(u) - cx) / fx, (float(v) - cy) / fy, 1.0], dtype=np.float32)
    direction_cam /= max(float(np.linalg.norm(direction_cam)), 1e-8)
    origin_world = np.asarray(c2w[:3, 3], dtype=np.float32)
    direction_world = direction_cam @ np.asarray(c2w[:3, :3], dtype=np.float32).T
    direction_world /= max(float(np.linalg.norm(direction_world)), 1e-8)
    return origin_world.astype(np.float32), direction_world.astype(np.float32)


def _triangulate_rays(origins_world: list[np.ndarray], directions_world: list[np.ndarray]) -> tuple[np.ndarray | None, list[float] | None]:
    if len(origins_world) < 2 or len(directions_world) < 2:
        return None, None
    A = np.zeros((3, 3), dtype=np.float64)
    b = np.zeros((3,), dtype=np.float64)
    for origin, direction in zip(origins_world, directions_world):
        d = np.asarray(direction, dtype=np.float64)
        d /= max(float(np.linalg.norm(d)), 1e-12)
        c = np.asarray(origin, dtype=np.float64)
        projector = np.eye(3, dtype=np.float64) - np.outer(d, d)
        A += projector
        b += projector @ c
    try:
        point = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None, None
    errors: list[float] = []
    for origin, direction in zip(origins_world, directions_world):
        d = np.asarray(direction, dtype=np.float64)
        d /= max(float(np.linalg.norm(d)), 1e-12)
        c = np.asarray(origin, dtype=np.float64)
        t = float((point - c) @ d)
        closest = c + t * d
        errors.append(float(np.linalg.norm(point - closest)))
    return point.astype(np.float32), errors


def _world_to_camera(point_world: np.ndarray, c2w: np.ndarray) -> np.ndarray:
    w2c = np.linalg.inv(np.asarray(c2w, dtype=np.float32)).astype(np.float32)
    point = np.asarray(point_world, dtype=np.float32).reshape(1, 3)
    return (point @ w2c[:3, :3].T + w2c[:3, 3][None, :]).reshape(3).astype(np.float32)


def _backproject_pixels_at_depth(
    K: np.ndarray,
    c2w: np.ndarray,
    pixels_uv: list[tuple[float, float]],
    depth_z: float,
) -> np.ndarray:
    depth = float(depth_z)
    if depth <= 0:
        raise SystemExit(f"Invalid positive depth required for backprojection, got: {depth}")
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])
    pts_cam = []
    for u, v in pixels_uv:
        pts_cam.append([(float(u) - cx) * depth / fx, (float(v) - cy) * depth / fy, depth])
    pts_cam_arr = np.asarray(pts_cam, dtype=np.float32)
    return (
        pts_cam_arr @ np.asarray(c2w[:3, :3], dtype=np.float32).T
        + np.asarray(c2w[:3, 3], dtype=np.float32)[None, :]
    ).astype(np.float32)


def _project_mask_support(
    centers_world: np.ndarray,
    masks_by_cam: dict[str, np.ndarray],
    rig_by_cam: dict[str, dict[str, np.ndarray]],
) -> np.ndarray:
    points = np.asarray(centers_world, dtype=np.float32)
    support = np.zeros((points.shape[0],), dtype=np.uint8)
    for cam_id, mask_u8 in masks_by_cam.items():
        mask = np.asarray(mask_u8 > 0, dtype=bool)
        K = np.asarray(rig_by_cam[cam_id]["K"], dtype=np.float32)
        c2w = np.asarray(rig_by_cam[cam_id]["c2w"], dtype=np.float32)
        w2c = np.linalg.inv(c2w).astype(np.float32)
        pts_cam = points @ w2c[:3, :3].T + w2c[:3, 3][None, :]
        z = pts_cam[:, 2]
        valid = np.isfinite(z) & (z > 1e-6)
        if not np.any(valid):
            continue
        u = np.round(K[0, 0] * (pts_cam[valid, 0] / z[valid]) + K[0, 2]).astype(np.int32)
        v = np.round(K[1, 1] * (pts_cam[valid, 1] / z[valid]) + K[1, 2]).astype(np.int32)
        in_bounds = (u >= 0) & (u < mask.shape[1]) & (v >= 0) & (v < mask.shape[0])
        if not np.any(in_bounds):
            continue
        valid_indices = np.flatnonzero(valid)
        hit_indices = valid_indices[in_bounds]
        hits = mask[v[in_bounds], u[in_bounds]]
        if np.any(hits):
            support[hit_indices[hits]] += 1
    return support


def _surface_mask(occupancy: np.ndarray) -> np.ndarray:
    occ = np.asarray(occupancy, dtype=bool)
    if occ.size == 0 or not np.any(occ):
        return np.zeros_like(occ, dtype=bool)
    padded = np.pad(occ, 1, mode="constant", constant_values=False)
    all_neighbors = (
        padded[2:, 1:-1, 1:-1]
        & padded[:-2, 1:-1, 1:-1]
        & padded[1:-1, 2:, 1:-1]
        & padded[1:-1, :-2, 1:-1]
        & padded[1:-1, 1:-1, 2:]
        & padded[1:-1, 1:-1, :-2]
    )
    return occ & (~all_neighbors)


def _point_support_grid(
    candidate_points: np.ndarray,
    roi_min: np.ndarray,
    dims: np.ndarray,
    voxel_size_m: float,
    point_support_radius_m: float,
) -> np.ndarray:
    points = np.asarray(candidate_points, dtype=np.float32)
    support = np.zeros(tuple(int(v) for v in dims.tolist()), dtype=bool)
    if points.size == 0:
        return support
    roi_min_f = np.asarray(roi_min, dtype=np.float32)
    dims_i = dims.astype(np.int32)
    voxel = float(voxel_size_m)
    radius = float(point_support_radius_m)
    radius_sq = radius * radius
    for point in points:
        min_idx = np.ceil(((point - radius) - roi_min_f) / voxel - 0.5).astype(np.int32)
        max_idx = np.floor(((point + radius) - roi_min_f) / voxel - 0.5).astype(np.int32)
        min_idx = np.maximum(min_idx, 0)
        max_idx = np.minimum(max_idx, dims_i - 1)
        if np.any(max_idx < min_idx):
            continue
        xs = np.arange(int(min_idx[0]), int(max_idx[0]) + 1, dtype=np.int32)
        ys = np.arange(int(min_idx[1]), int(max_idx[1]) + 1, dtype=np.int32)
        zs = np.arange(int(min_idx[2]), int(max_idx[2]) + 1, dtype=np.int32)
        grid = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1).reshape(-1, 3)
        centers = roi_min_f[None, :] + (grid.astype(np.float32) + 0.5) * voxel
        dist_sq = np.sum((centers - point[None, :]) ** 2, axis=1)
        near = dist_sq <= radius_sq
        if np.any(near):
            support[tuple(grid[near].T)] = True
    return support


def _connected_components_from_indices(voxel_indices: np.ndarray) -> list[np.ndarray]:
    indices = np.asarray(voxel_indices, dtype=np.int32)
    if indices.size == 0:
        return []
    key_to_idx = {tuple(row.tolist()): idx for idx, row in enumerate(indices)}
    parent = list(range(indices.shape[0]))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for idx, (x, y, z) in enumerate(indices.tolist()):
        for dx, dy, dz in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
            other = key_to_idx.get((x + dx, y + dy, z + dz))
            if other is not None:
                union(idx, other)

    groups: dict[int, list[int]] = defaultdict(list)
    for idx in range(indices.shape[0]):
        groups[find(idx)].append(idx)
    components = [indices[np.asarray(member_idx, dtype=np.int32)] for member_idx in groups.values()]
    components.sort(key=lambda arr: int(arr.shape[0]), reverse=True)
    return components


def _select_component_mask(
    occupancy: np.ndarray,
    anchor_world: np.ndarray | None,
    roi_min: np.ndarray,
    voxel_size_m: float,
) -> tuple[np.ndarray, str]:
    occ_indices = np.argwhere(occupancy)
    if occ_indices.size == 0:
        return np.zeros_like(occupancy, dtype=bool), "empty"
    components = _connected_components_from_indices(occ_indices)
    if not components:
        return np.zeros_like(occupancy, dtype=bool), "empty"
    selected = components[0]
    selection_mode = "largest_component_fallback"
    if anchor_world is not None:
        anchor_grid = (
            (np.asarray(anchor_world, dtype=np.float32) - np.asarray(roi_min, dtype=np.float32)) / float(voxel_size_m)
        ) - 0.5
        anchor_idx = np.floor(anchor_grid).astype(np.int32)
        if np.all(anchor_idx >= 0) and np.all(anchor_idx < np.asarray(occupancy.shape, dtype=np.int32)) and bool(
            occupancy[tuple(anchor_idx.tolist())]
        ):
            for component in components:
                if np.any(np.all(component == anchor_idx[None, :], axis=1)):
                    selected = component
                    selection_mode = "anchor_contains_component"
                    break
        else:
            threshold_cells = max(float(0.12) / max(float(voxel_size_m), 1e-6), 2.0)
            best_dist = float("inf")
            best_component = None
            anchor_grid_f = anchor_grid.astype(np.float32)
            for component in components:
                dist = float(np.min(np.linalg.norm(component.astype(np.float32) - anchor_grid_f[None, :], axis=1)))
                if dist < best_dist:
                    best_dist = dist
                    best_component = component
            if best_component is not None and best_dist <= threshold_cells:
                selected = best_component
                selection_mode = "anchor_nearest_component"
    selected_mask = np.zeros_like(occupancy, dtype=bool)
    selected_mask[tuple(selected.T)] = True
    return selected_mask, selection_mode


def _dilate_bool_grid(mask_bool: np.ndarray, radius_m: float, voxel_size_m: float) -> np.ndarray:
    src = np.asarray(mask_bool, dtype=bool)
    if not np.any(src):
        return np.zeros_like(src, dtype=bool)
    radius = float(radius_m)
    voxel = max(float(voxel_size_m), 1e-6)
    radius_cells = int(np.ceil(radius / voxel))
    offsets: list[tuple[int, int, int]] = []
    for dx in range(-radius_cells, radius_cells + 1):
        for dy in range(-radius_cells, radius_cells + 1):
            for dz in range(-radius_cells, radius_cells + 1):
                dist = np.linalg.norm(np.asarray([dx, dy, dz], dtype=np.float32) * voxel)
                if dist <= radius + 1e-6:
                    offsets.append((dx, dy, dz))
    out = np.zeros_like(src, dtype=bool)
    src_idx = np.argwhere(src)
    shape = np.asarray(src.shape, dtype=np.int32)
    for dx, dy, dz in offsets:
        shifted = src_idx + np.asarray([dx, dy, dz], dtype=np.int32)[None, :]
        valid = np.all((shifted >= 0) & (shifted < shape[None, :]), axis=1)
        if np.any(valid):
            out[tuple(shifted[valid].T)] = True
    return out


def _voxel_downsample_centroids(points_xyz: np.ndarray, voxel_size_m: float) -> np.ndarray:
    pts = np.asarray(points_xyz, dtype=np.float32)
    if pts.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    size = float(voxel_size_m)
    if size <= 0:
        return pts.astype(np.float32)
    vox = np.floor(pts / size).astype(np.int64)
    groups: dict[tuple[int, int, int], list[int]] = {}
    for idx, key in enumerate(map(tuple, vox.tolist())):
        groups.setdefault(key, []).append(idx)
    out = [pts[indices].mean(axis=0).astype(np.float32) for indices in groups.values()]
    if not out:
        return np.zeros((0, 3), dtype=np.float32)
    return np.stack(out, axis=0).astype(np.float32)


def _serialize_csv_value(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return value


def _ordered_unique_floats(values: list[float]) -> list[float]:
    out: list[float] = []
    for value in values:
        value_f = float(value)
        if not any(abs(value_f - existing) <= 1e-9 for existing in out):
            out.append(value_f)
    return out


def _npz_matrix_or_none(npz_file: Any, key: str, expected_shape: tuple[int, int]) -> np.ndarray | None:
    if key not in npz_file.files:
        return None
    value = np.asarray(npz_file[key], dtype=np.float32)
    if value.shape != expected_shape:
        raise SystemExit(f"Invalid {key} shape in npz: expected {expected_shape}, got {value.shape}")
    return value


def _npz_str_or_none(npz_file: Any, key: str) -> str | None:
    if key not in npz_file.files:
        return None
    value = np.asarray(npz_file[key])
    try:
        item = value.item()
    except Exception:
        return None
    return str(item)


def _parse_matrix_json(raw_value: str, expected_shape: tuple[int, int], field_name: str) -> np.ndarray:
    try:
        mat = np.asarray(json.loads(str(raw_value)), dtype=np.float32)
    except Exception as exc:
        raise SystemExit(f"Failed to parse {field_name}: {raw_value!r}; error={exc!r}")
    if mat.shape != expected_shape:
        raise SystemExit(f"Invalid {field_name} shape: expected {expected_shape}, got {mat.shape}")
    return mat


def _select_depth_scale(raw_scale: float | None, camera_median_scale: float, scale_guard_ratio: float) -> tuple[float, str]:
    median_scale = float(camera_median_scale)
    if not np.isfinite(median_scale) or median_scale <= 0:
        raise SystemExit(f"Invalid camera median scale: {camera_median_scale}")
    if raw_scale is None or (not np.isfinite(float(raw_scale))) or float(raw_scale) <= 0:
        return median_scale, "camera_median"
    raw = float(raw_scale)
    rel = abs(raw / median_scale - 1.0)
    if rel <= float(scale_guard_ratio):
        return raw, "frame"
    return median_scale, "camera_median"


def _align_local_points_to_rig(
    points_local: np.ndarray,
    render_c2w_local: np.ndarray,
    rig_c2w: np.ndarray,
    scale: float,
) -> np.ndarray:
    pts = np.asarray(points_local, dtype=np.float32)
    if pts.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    local_c2w = np.asarray(render_c2w_local, dtype=np.float32)
    rig_c2w = np.asarray(rig_c2w, dtype=np.float32)
    R_local = local_c2w[:3, :3]
    t_local = local_c2w[:3, 3]
    R_rig = rig_c2w[:3, :3]
    t_rig = rig_c2w[:3, 3]
    aligned = float(scale) * ((pts - t_local[None, :]) @ R_local @ R_rig.T) + t_rig[None, :]
    return np.asarray(aligned, dtype=np.float32)


def _points_in_roi_stats(points_xyz: np.ndarray, roi_min: np.ndarray | None, roi_max: np.ndarray | None) -> tuple[int, float]:
    pts = np.asarray(points_xyz, dtype=np.float32)
    if pts.size == 0 or roi_min is None or roi_max is None:
        return 0, 0.0
    lower = np.asarray(roi_min, dtype=np.float32)[None, :]
    upper = np.asarray(roi_max, dtype=np.float32)[None, :]
    inside = np.all((pts >= lower) & (pts <= upper), axis=1)
    count = int(np.count_nonzero(inside))
    ratio = float(count / max(int(pts.shape[0]), 1))
    return count, ratio


def _depth_stats_for_observation(
    scene_dir: Path,
    observations_root: Path,
    cam_id: str,
    scene_stem: str,
    observation_row: dict[str, str],
    mask_dilate_px: int,
) -> tuple[float | None, int]:
    depth_rel = str(observation_row.get("depth_path") or "").strip()
    if not depth_rel:
        raise SystemExit(f"Observation row missing depth_path for {cam_id}/{scene_stem}")
    depth_path = Path(depth_rel)
    if not depth_path.is_absolute():
        depth_path = observations_root / depth_path
    depth = _read_npy(depth_path)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 2:
        raise SystemExit(f"Unsupported depth shape for {cam_id}/{scene_stem}: {depth.shape}")

    mask_u8 = _read_gray(_mask_path(scene_dir, cam_id, scene_stem))
    crop_applied = bool(int(float(str(observation_row.get("crop_applied") or "0"))))
    crop_box_xyxy = _obs_parse_crop_box_json(str(observation_row.get("crop_box_xyxy") or "")) if crop_applied else None
    row_width_raw = str(observation_row.get("width") or "").strip()
    row_height_raw = str(observation_row.get("height") or "").strip()
    row_resize_mode = str(observation_row.get("resize_mode") or "").strip().lower()
    if not row_width_raw or not row_height_raw:
        raise SystemExit(f"Observation row missing width/height for {cam_id}/{scene_stem}")
    if row_resize_mode not in {"resize", "center_crop"}:
        raise SystemExit(f"Observation row has invalid resize_mode={row_resize_mode!r} for {cam_id}/{scene_stem}")

    prepared_mask = _obs_prepare_mask(
        mask_u8,
        target_width=int(float(row_width_raw)),
        target_height=int(float(row_height_raw)),
        resize_mode=row_resize_mode,
        crop_box_xyxy=crop_box_xyxy,
    )
    if prepared_mask.shape != depth.shape:
        raise SystemExit(
            f"Prepared mask/depth mismatch for {cam_id}/{scene_stem}: mask={prepared_mask.shape}, depth={depth.shape}"
        )
    mask_bool = _obs_dilate_mask(prepared_mask > 0, radius_px=int(mask_dilate_px))
    valid = np.asarray(mask_bool, dtype=bool) & np.isfinite(depth) & (depth > 0)
    depth_mask_pixels = int(np.count_nonzero(valid))
    if depth_mask_pixels <= 0:
        return None, 0
    return float(np.median(depth[valid])), depth_mask_pixels


def _attempt_constraint(
    base_roi_min: np.ndarray,
    base_roi_max: np.ndarray,
    roi_padding_m: float,
    hull_voxel_size_m: float,
    required_support: int,
    anchor_world: np.ndarray | None,
    masks_by_cam: dict[str, np.ndarray],
    rig_by_cam: dict[str, dict[str, np.ndarray]],
    fg_points_per_cam: dict[str, np.ndarray],
    point_support_radius_m: float,
    depth_trim_radius_m: float,
    min_trimmed_points: int,
    output_voxel_size_m: float,
    max_roi_voxels: int,
) -> dict[str, Any]:
    roi_min = np.asarray(base_roi_min, dtype=np.float32) - float(roi_padding_m)
    roi_max = np.asarray(base_roi_max, dtype=np.float32) + float(roi_padding_m)
    roi_voxel_size_m, dims = _choose_roi_voxel_size(
        roi_min=roi_min,
        roi_max=roi_max,
        base_voxel_size_m=float(hull_voxel_size_m),
        max_roi_voxels=int(max_roi_voxels),
    )
    centers_flat = _flatten_grid(roi_min=roi_min, dims=dims, voxel_size_m=roi_voxel_size_m)
    support_counts_flat = _project_mask_support(
        centers_world=centers_flat,
        masks_by_cam=masks_by_cam,
        rig_by_cam=rig_by_cam,
    )
    support_counts = support_counts_flat.reshape(tuple(int(v) for v in dims.tolist()))
    support_ge_2_voxels = int(np.count_nonzero(support_counts >= 2))
    occupancy = support_counts >= int(required_support)
    mask_supported_voxels = int(np.count_nonzero(occupancy))
    if mask_supported_voxels == 0:
        return {
            "success": False,
            "roi_min": roi_min,
            "roi_max": roi_max,
            "roi_voxel_size_m": float(roi_voxel_size_m),
            "support_ge_2_voxels": int(support_ge_2_voxels),
            "mask_supported_voxels": 0,
            "surface_voxels": 0,
            "selected_component_voxels": 0,
            "depth_supported_voxels": 0,
            "depth_support_ratio": 0.0,
            "trim_applied": False,
            "trim_rejected": False,
            "trim_removed_voxels": 0,
            "selection_mode": "empty",
            "output_points_xyz": np.zeros((0, 3), dtype=np.float32),
        }

    surface = _surface_mask(occupancy)
    surface_voxels = int(np.count_nonzero(surface))
    if surface_voxels == 0:
        return {
            "success": False,
            "roi_min": roi_min,
            "roi_max": roi_max,
            "roi_voxel_size_m": float(roi_voxel_size_m),
            "support_ge_2_voxels": int(support_ge_2_voxels),
            "mask_supported_voxels": int(mask_supported_voxels),
            "surface_voxels": 0,
            "selected_component_voxels": 0,
            "depth_supported_voxels": 0,
            "depth_support_ratio": 0.0,
            "trim_applied": False,
            "trim_rejected": False,
            "trim_removed_voxels": 0,
            "selection_mode": "empty_surface",
            "output_points_xyz": np.zeros((0, 3), dtype=np.float32),
        }

    selected_component_mask, selection_mode = _select_component_mask(
        occupancy=occupancy,
        anchor_world=anchor_world,
        roi_min=roi_min,
        voxel_size_m=float(roi_voxel_size_m),
    )
    surface_keep_base = surface & selected_component_mask
    selected_component_voxels = int(np.count_nonzero(surface_keep_base))
    if selected_component_voxels == 0:
        return {
            "success": False,
            "roi_min": roi_min,
            "roi_max": roi_max,
            "roi_voxel_size_m": float(roi_voxel_size_m),
            "support_ge_2_voxels": int(support_ge_2_voxels),
            "mask_supported_voxels": int(mask_supported_voxels),
            "surface_voxels": int(surface_voxels),
            "selected_component_voxels": 0,
            "depth_supported_voxels": 0,
            "depth_support_ratio": 0.0,
            "trim_applied": False,
            "trim_rejected": False,
            "trim_removed_voxels": 0,
            "selection_mode": str(selection_mode),
            "output_points_xyz": np.zeros((0, 3), dtype=np.float32),
        }

    depth_support_any = np.zeros_like(surface_keep_base, dtype=bool)
    for points_xyz in fg_points_per_cam.values():
        depth_support_any |= _point_support_grid(
            candidate_points=points_xyz,
            roi_min=roi_min,
            dims=dims,
            voxel_size_m=float(roi_voxel_size_m),
            point_support_radius_m=float(point_support_radius_m),
        )
    depth_supported_surface = surface_keep_base & depth_support_any
    depth_supported_voxels = int(np.count_nonzero(depth_supported_surface))
    depth_support_ratio = float(depth_supported_voxels / max(selected_component_voxels, 1))

    final_keep = surface_keep_base
    trim_applied = False
    trim_rejected = False
    trim_removed_voxels = 0
    if depth_supported_voxels > 0:
        depth_trim_region = _dilate_bool_grid(
            mask_bool=depth_supported_surface,
            radius_m=float(depth_trim_radius_m),
            voxel_size_m=float(roi_voxel_size_m),
        )
        trimmed_keep = surface_keep_base & depth_trim_region
        trimmed_count = int(np.count_nonzero(trimmed_keep))
        removed_count = int(selected_component_voxels - trimmed_count)
        if trimmed_count >= int(min_trimmed_points):
            if removed_count > 0:
                final_keep = trimmed_keep
                trim_applied = True
                trim_removed_voxels = removed_count
        elif removed_count > 0:
            trim_rejected = True
            trim_removed_voxels = removed_count

    if not np.any(final_keep):
        return {
            "success": False,
            "roi_min": roi_min,
            "roi_max": roi_max,
            "roi_voxel_size_m": float(roi_voxel_size_m),
            "support_ge_2_voxels": int(support_ge_2_voxels),
            "mask_supported_voxels": int(mask_supported_voxels),
            "surface_voxels": int(surface_voxels),
            "selected_component_voxels": int(selected_component_voxels),
            "depth_supported_voxels": int(depth_supported_voxels),
            "depth_support_ratio": float(depth_support_ratio),
            "trim_applied": False,
            "trim_rejected": bool(trim_rejected),
            "trim_removed_voxels": int(trim_removed_voxels),
            "selection_mode": str(selection_mode),
            "output_points_xyz": np.zeros((0, 3), dtype=np.float32),
        }

    keep_idx = np.argwhere(final_keep)
    output_points_xyz = roi_min[None, :] + (keep_idx.astype(np.float32) + 0.5) * float(roi_voxel_size_m)
    output_points_xyz = _voxel_downsample_centroids(output_points_xyz, voxel_size_m=float(output_voxel_size_m))
    return {
        "success": bool(output_points_xyz.size > 0),
        "roi_min": roi_min,
        "roi_max": roi_max,
        "roi_voxel_size_m": float(roi_voxel_size_m),
        "support_ge_2_voxels": int(support_ge_2_voxels),
        "mask_supported_voxels": int(mask_supported_voxels),
        "surface_voxels": int(surface_voxels),
        "selected_component_voxels": int(selected_component_voxels),
        "depth_supported_voxels": int(depth_supported_voxels),
        "depth_support_ratio": float(depth_support_ratio),
        "trim_applied": bool(trim_applied),
        "trim_rejected": bool(trim_rejected),
        "trim_removed_voxels": int(trim_removed_voxels),
        "selection_mode": str(selection_mode),
        "output_points_xyz": np.asarray(output_points_xyz, dtype=np.float32),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Apply calibrated multiview dynamic constraints on NeoVerse fg point candidates.")
    ap.add_argument("--scene_dir", required=True, type=str)
    ap.add_argument("--fg_points_root", required=True, type=str)
    ap.add_argument("--fused_root", default="mvp-demo/output/neoverse_fused", type=str)
    ap.add_argument("--points_by_timestamp_root", default="mvp-demo/output/neoverse_fused", type=str)
    ap.add_argument("--cams", default="cam0,cam1,cam2", type=str)
    ap.add_argument("--hull_voxel_size_m", default=0.02, type=float)
    ap.add_argument("--output_voxel_size_m", default=0.01, type=float)
    ap.add_argument("--roi_padding_m", default=0.12, type=float)
    ap.add_argument("--min_mask_cam_support", default=2, type=int)
    ap.add_argument("--point_support_radius_m", default=0.03, type=float)
    ap.add_argument("--depth_trim_radius_m", default=0.06, type=float)
    ap.add_argument("--min_trimmed_points", default=40, type=int)
    ap.add_argument("--scale_guard_ratio", default=0.25, type=float)
    ap.add_argument("--min_depth_mask_pixels", default=24, type=int)
    ap.add_argument("--depth_support_source", default="aligned_fg_points", choices=["aligned_fg_points"], type=str)
    ap.add_argument(
        "--depth_support_mode",
        default="aligned_fg_points",
        type=str,
        help="Compatibility alias retained for old callers; trim support now always uses aligned_fg_points.",
    )
    ap.add_argument("--max_roi_voxels", default=400000, type=int)
    args = ap.parse_args()

    repo = _repo_root()
    scene_dir = Path(str(args.scene_dir)).resolve()
    if not scene_dir.exists():
        raise SystemExit(f"Missing --scene_dir: {scene_dir}")

    scene_id = scene_dir.name
    cams = [c.strip() for c in str(args.cams).split(",") if c.strip()]
    if cams != ["cam0", "cam1", "cam2"]:
        raise SystemExit(f"This first version only supports cams=['cam0','cam1','cam2']. Got: {cams}")

    fg_points_root = _resolve_scene_points_root(str(args.fg_points_root), scene_id, repo)
    if not fg_points_root.exists():
        raise SystemExit(f"Missing fg_points_root: {fg_points_root}")

    scene_fused_root = _resolve_scene_output_root(str(args.fused_root), scene_id, repo, "fused")
    if not scene_fused_root.exists():
        raise SystemExit(f"Missing fused root: {scene_fused_root}")

    points_by_timestamp_dir = _resolve_scene_output_root(str(args.points_by_timestamp_root), scene_id, repo, "points_by_timestamp")
    points_by_timestamp_dir.mkdir(parents=True, exist_ok=True)

    dynamic_dir = scene_fused_root / "dynamic"
    dynamic_dir.mkdir(parents=True, exist_ok=True)

    dynamic_index_path = scene_fused_root / "dynamic_index.csv"
    fusion_meta_path = scene_fused_root / "fusion_meta.json"
    if not dynamic_index_path.exists():
        raise SystemExit(f"Missing dynamic_index.csv: {dynamic_index_path}")
    if not fusion_meta_path.exists():
        raise SystemExit(f"Missing fusion_meta.json: {fusion_meta_path}")

    rig_by_cam = _load_rig(scene_dir, cams)
    fg_points_by_key = _load_fg_points_by_key(fg_points_root)
    dynamic_rows = _load_dynamic_rows(dynamic_index_path)
    fusion_meta = _load_json(fusion_meta_path)
    points_meta, observations_root, legacy_frame_contract_assumed = _load_points_contract(fg_points_root, repo)
    observation_rows = _load_observation_rows(observations_root, cams)
    mask_dilate_px = int(points_meta.get("mask_dilate_px", 0))

    padding_schedule = _ordered_unique_floats([float(args.roi_padding_m), 0.20, 0.32])
    voxel_schedule = _ordered_unique_floats([float(args.hull_voxel_size_m), 0.03, 0.04])

    frame_infos: list[dict[str, Any]] = []
    raw_scale_values_by_cam: dict[str, list[float]] = {cam_id: [] for cam_id in cams}
    anchor_ray_error_means: list[float] = []

    for row in dynamic_rows:
        scene_stem = str(row["scene_stem"])
        logical_t_idx = int(float(row["logical_t_idx"]))
        bboxes_by_cam: dict[str, list[int]] = {}
        ray_origins: list[np.ndarray] = []
        ray_dirs: list[np.ndarray] = []
        obs_rows_by_cam: dict[str, dict[str, str]] = {}
        depth_local_median_by_cam: dict[str, float | None] = {}
        depth_mask_pixels_by_cam: dict[str, int] = {}
        depth_scale_raw_by_cam: dict[str, float | None] = {}
        anchor_depths: dict[str, float | None] = {cam_id: None for cam_id in cams}
        anchor_ray_error_per_cam: dict[str, float | None] = {cam_id: None for cam_id in cams}
        roi_source = "mask_anchor_world_slice"

        for cam_id in cams:
            mask_u8 = _read_gray(_mask_path(scene_dir, cam_id, scene_stem))
            bbox = _mask_bbox(mask_u8)
            if bbox is None:
                raise SystemExit(f"Empty mask for {cam_id}/{scene_stem}")
            bboxes_by_cam[cam_id] = bbox

            key = (cam_id, logical_t_idx, scene_stem)
            observation_row = observation_rows.get(key)
            if observation_row is None:
                raise SystemExit(f"Missing observation row for key={key}")
            obs_rows_by_cam[cam_id] = observation_row

            local_depth_median, depth_mask_pixels = _depth_stats_for_observation(
                scene_dir=scene_dir,
                observations_root=observations_root,
                cam_id=cam_id,
                scene_stem=scene_stem,
                observation_row=observation_row,
                mask_dilate_px=mask_dilate_px,
            )
            depth_local_median_by_cam[cam_id] = local_depth_median
            depth_mask_pixels_by_cam[cam_id] = int(depth_mask_pixels)

            u, v = _bbox_center_xy(bbox)
            origin, direction = _camera_ray_world(
                K=rig_by_cam[cam_id]["K"],
                c2w=rig_by_cam[cam_id]["c2w"],
                u=u,
                v=v,
            )
            ray_origins.append(origin)
            ray_dirs.append(direction)

        mask_anchor_world, anchor_errors = _triangulate_rays(ray_origins, ray_dirs)
        if mask_anchor_world is not None and anchor_errors is not None:
            for cam_id, error in zip(cams, anchor_errors):
                anchor_ray_error_per_cam[cam_id] = float(error)
            anchor_ray_error_means.append(float(np.mean(anchor_errors)))
            for cam_id in cams:
                point_cam = _world_to_camera(mask_anchor_world, rig_by_cam[cam_id]["c2w"])
                anchor_depth = float(point_cam[2])
                anchor_depths[cam_id] = anchor_depth if np.isfinite(anchor_depth) else None
        else:
            roi_source = "fg_union_fallback"

        anchor_depth_valid = all(
            (anchor_depths[cam_id] is not None) and (float(anchor_depths[cam_id]) > 0.0) for cam_id in cams
        )
        if not anchor_depth_valid:
            roi_source = "fg_union_fallback"

        for cam_id in cams:
            local_depth_median = depth_local_median_by_cam[cam_id]
            anchor_depth = anchor_depths[cam_id]
            depth_mask_pixels = depth_mask_pixels_by_cam[cam_id]
            if (
                local_depth_median is not None
                and anchor_depth is not None
                and float(anchor_depth) > 0.0
                and int(depth_mask_pixels) >= int(args.min_depth_mask_pixels)
            ):
                raw_scale = float(anchor_depth) / float(local_depth_median)
                if np.isfinite(raw_scale) and raw_scale > 0:
                    depth_scale_raw_by_cam[cam_id] = raw_scale
                    raw_scale_values_by_cam[cam_id].append(raw_scale)
                    continue
            depth_scale_raw_by_cam[cam_id] = None

        anchor_ray_error_mean = None
        valid_anchor_errors = [float(v) for v in anchor_ray_error_per_cam.values() if v is not None]
        if valid_anchor_errors:
            anchor_ray_error_mean = float(np.mean(valid_anchor_errors))

        frame_infos.append(
            {
                "scene_stem": scene_stem,
                "logical_t_idx": logical_t_idx,
                "bboxes_by_cam": bboxes_by_cam,
                "obs_rows_by_cam": obs_rows_by_cam,
                "depth_local_median_by_cam": depth_local_median_by_cam,
                "depth_mask_pixels_by_cam": depth_mask_pixels_by_cam,
                "depth_scale_raw_by_cam": depth_scale_raw_by_cam,
                "mask_anchor_world": None if mask_anchor_world is None else np.asarray(mask_anchor_world, dtype=np.float32),
                "anchor_depths": anchor_depths,
                "anchor_depth_valid": bool(anchor_depth_valid),
                "anchor_ray_error_per_cam": anchor_ray_error_per_cam,
                "anchor_ray_error_mean": anchor_ray_error_mean,
                "roi_source": roi_source,
            }
        )

    camera_median_scale: dict[str, float] = {}
    per_camera_scale_stats: dict[str, dict[str, float | int]] = {}
    for cam_id in cams:
        values = np.asarray(raw_scale_values_by_cam[cam_id], dtype=np.float32)
        if values.size == 0:
            raise SystemExit(f"No valid raw scale samples for {cam_id}; cannot align NeoVerse local points into rig world.")
        median_scale = float(np.median(values))
        camera_median_scale[cam_id] = median_scale
        per_camera_scale_stats[cam_id] = {
            "num_valid_raw_scales": int(values.size),
            "camera_median_scale": median_scale,
            "raw_scale_mean": float(np.mean(values)),
            "raw_scale_median": median_scale,
            "raw_scale_min": float(np.min(values)),
            "raw_scale_max": float(np.max(values)),
        }

    constraint_rows: list[dict[str, Any]] = []
    retrieval_index_rows: list[dict[str, Any]] = []
    updated_dynamic_rows: list[dict[str, Any]] = []
    mode_counter: Counter[str] = Counter()
    roi_source_counter: Counter[str] = Counter()
    output_counts: list[int] = []
    trim_applied_frames = 0
    trim_rejected_frames = 0
    trim_zero_support_frames = 0
    before_align_ratio_by_cam: dict[str, list[float]] = {cam_id: [] for cam_id in cams}
    after_align_ratio_by_cam: dict[str, list[float]] = {cam_id: [] for cam_id in cams}

    for row, frame_info in zip(dynamic_rows, frame_infos):
        scene_stem = str(frame_info["scene_stem"])
        logical_t_idx = int(frame_info["logical_t_idx"])
        masks_by_cam = {cam_id: _read_gray(_mask_path(scene_dir, cam_id, scene_stem)) for cam_id in cams}
        mask_anchor_world = frame_info["mask_anchor_world"]
        anchor_depths = dict(frame_info["anchor_depths"])
        roi_source = str(frame_info["roi_source"])
        roi_source_counter[roi_source] += 1

        depth_local_median_by_cam = dict(frame_info["depth_local_median_by_cam"])
        depth_mask_pixels_by_cam = dict(frame_info["depth_mask_pixels_by_cam"])
        depth_scale_raw_by_cam = dict(frame_info["depth_scale_raw_by_cam"])
        depth_scale_used_by_cam: dict[str, float] = {}
        depth_scale_source_by_cam: dict[str, str] = {}

        raw_fg_points_per_cam: dict[str, np.ndarray] = {}
        aligned_fg_points_per_cam: dict[str, np.ndarray] = {}
        aligned_union_parts: list[np.ndarray] = []
        total_candidate_points = 0

        for cam_id in cams:
            key = (cam_id, logical_t_idx, scene_stem)
            points_path = fg_points_by_key.get(key)
            if points_path is None:
                raise SystemExit(f"Missing fg points index row for key={key}")
            if not points_path.exists():
                raise SystemExit(f"Missing fg points file for key={key}: {points_path}")
            observation_row = frame_info["obs_rows_by_cam"][cam_id]
            with np.load(points_path) as fg_npz:
                fg_xyz_local = np.asarray(fg_npz["xyz"], dtype=np.float32)
                point_coordinate_frame = _npz_str_or_none(fg_npz, "coordinate_frame") or str(
                    points_meta.get("point_coordinate_frame") or "neoverse_render_world"
                )
                if point_coordinate_frame != "neoverse_render_world":
                    raise SystemExit(
                        f"Unsupported point coordinate frame for {cam_id}/{scene_stem}: {point_coordinate_frame!r}"
                    )
                render_c2w = _npz_matrix_or_none(fg_npz, "render_c2w", (4, 4))
                if render_c2w is None:
                    render_c2w = _parse_matrix_json(str(observation_row.get("render_c2w")), (4, 4), "render_c2w")

            scale_used, scale_source = _select_depth_scale(
                raw_scale=depth_scale_raw_by_cam.get(cam_id),
                camera_median_scale=camera_median_scale[cam_id],
                scale_guard_ratio=float(args.scale_guard_ratio),
            )
            depth_scale_used_by_cam[cam_id] = float(scale_used)
            depth_scale_source_by_cam[cam_id] = str(scale_source)

            aligned_fg = _align_local_points_to_rig(
                points_local=fg_xyz_local,
                render_c2w_local=render_c2w,
                rig_c2w=rig_by_cam[cam_id]["c2w"],
                scale=float(scale_used),
            )
            raw_fg_points_per_cam[cam_id] = fg_xyz_local
            aligned_fg_points_per_cam[cam_id] = aligned_fg
            total_candidate_points += int(fg_xyz_local.shape[0])
            if aligned_fg.size:
                aligned_union_parts.append(aligned_fg)

        mask_slice_world_points: list[np.ndarray] = []
        base_points = np.zeros((0, 3), dtype=np.float32)
        if mask_anchor_world is not None and bool(frame_info["anchor_depth_valid"]):
            for cam_id in cams:
                x0, y0, x1, y1 = frame_info["bboxes_by_cam"][cam_id]
                corners = [
                    (float(x0), float(y0)),
                    (float(x1), float(y0)),
                    (float(x0), float(y1)),
                    (float(x1), float(y1)),
                ]
                mask_slice_world_points.append(
                    _backproject_pixels_at_depth(
                        K=rig_by_cam[cam_id]["K"],
                        c2w=rig_by_cam[cam_id]["c2w"],
                        pixels_uv=corners,
                        depth_z=float(anchor_depths[cam_id]),
                    )
                )
            base_points = np.concatenate(mask_slice_world_points, axis=0).astype(np.float32)
        elif aligned_union_parts:
            base_points = np.concatenate(aligned_union_parts, axis=0).astype(np.float32)

        constraint_mode = "empty_fg_candidates"
        best_support_ge_2_voxels = 0
        selected_result: dict[str, Any] | None = None

        if base_points.size:
            base_roi_min, base_roi_max = _compute_roi_bounds(base_points, padding_m=0.0)
            for requested_voxel_size in voxel_schedule:
                for padding_m in padding_schedule:
                    attempt = _attempt_constraint(
                        base_roi_min=base_roi_min,
                        base_roi_max=base_roi_max,
                        roi_padding_m=float(padding_m),
                        hull_voxel_size_m=float(requested_voxel_size),
                        required_support=int(args.min_mask_cam_support),
                        anchor_world=mask_anchor_world,
                        masks_by_cam=masks_by_cam,
                        rig_by_cam=rig_by_cam,
                        fg_points_per_cam=aligned_fg_points_per_cam,
                        point_support_radius_m=float(args.point_support_radius_m),
                        depth_trim_radius_m=float(args.depth_trim_radius_m),
                        min_trimmed_points=int(args.min_trimmed_points),
                        output_voxel_size_m=float(args.output_voxel_size_m),
                        max_roi_voxels=int(args.max_roi_voxels),
                    )
                    best_support_ge_2_voxels = max(best_support_ge_2_voxels, int(attempt["support_ge_2_voxels"]))
                    if not bool(attempt["success"]):
                        continue
                    selected_result = attempt
                    if roi_source == "fg_union_fallback":
                        constraint_mode = "fg_union_fallback"
                    elif float(requested_voxel_size) > float(args.hull_voxel_size_m) + 1e-9:
                        constraint_mode = "coarse_voxel_multiview"
                    elif float(padding_m) > float(args.roi_padding_m) + 1e-9:
                        constraint_mode = "expanded_roi_multiview"
                    elif bool(attempt["trim_applied"]):
                        constraint_mode = "multiview_mask_hull_trimmed"
                    elif bool(attempt["trim_rejected"]):
                        constraint_mode = "multiview_mask_hull_trim_rejected"
                    else:
                        constraint_mode = "multiview_mask_hull_no_trim"
                    break
                if selected_result is not None:
                    break

            if selected_result is None:
                selected_result = _attempt_constraint(
                    base_roi_min=base_roi_min,
                    base_roi_max=base_roi_max,
                    roi_padding_m=float(padding_schedule[-1]),
                    hull_voxel_size_m=float(voxel_schedule[-1]),
                    required_support=1,
                    anchor_world=mask_anchor_world,
                    masks_by_cam=masks_by_cam,
                    rig_by_cam=rig_by_cam,
                    fg_points_per_cam=aligned_fg_points_per_cam,
                    point_support_radius_m=float(args.point_support_radius_m),
                    depth_trim_radius_m=float(args.depth_trim_radius_m),
                    min_trimmed_points=int(args.min_trimmed_points),
                    output_voxel_size_m=float(args.output_voxel_size_m),
                    max_roi_voxels=int(args.max_roi_voxels),
                )
                constraint_mode = "degraded_single_view_fallback"
        else:
            selected_result = {
                "success": False,
                "roi_min": None,
                "roi_max": None,
                "roi_voxel_size_m": float(args.hull_voxel_size_m),
                "support_ge_2_voxels": 0,
                "mask_supported_voxels": 0,
                "surface_voxels": 0,
                "selected_component_voxels": 0,
                "depth_supported_voxels": 0,
                "depth_support_ratio": 0.0,
                "trim_applied": False,
                "trim_rejected": False,
                "trim_removed_voxels": 0,
                "selection_mode": "empty",
                "output_points_xyz": np.zeros((0, 3), dtype=np.float32),
            }

        output_points_xyz = np.asarray(selected_result["output_points_xyz"], dtype=np.float32)
        dynamic_out_path = dynamic_dir / f"{scene_stem}.npy"
        retrieval_out_path = points_by_timestamp_dir / f"{scene_stem}.npy"
        _save_npy(dynamic_out_path, output_points_xyz)
        _save_npy(retrieval_out_path, output_points_xyz)

        output_count = int(output_points_xyz.shape[0])
        output_counts.append(output_count)
        mode_counter[str(constraint_mode)] += 1
        if bool(selected_result["trim_applied"]):
            trim_applied_frames += 1
        if bool(selected_result["trim_rejected"]):
            trim_rejected_frames += 1
        if int(selected_result["depth_supported_voxels"]) == 0:
            trim_zero_support_frames += 1

        roi_bounds_xyz = None
        roi_min = selected_result["roi_min"]
        roi_max = selected_result["roi_max"]
        if roi_min is not None and roi_max is not None:
            roi_bounds_xyz = [
                np.asarray(roi_min, dtype=np.float32).astype(float).tolist(),
                np.asarray(roi_max, dtype=np.float32).astype(float).tolist(),
            ]

        fg_points_in_roi_before_align: dict[str, float] = {}
        fg_points_in_roi_after_align: dict[str, float] = {}
        fg_points_in_roi_before_align_count: dict[str, int] = {}
        fg_points_in_roi_after_align_count: dict[str, int] = {}
        fg_points_total: dict[str, int] = {}
        for cam_id in cams:
            before_count, before_ratio = _points_in_roi_stats(raw_fg_points_per_cam[cam_id], roi_min, roi_max)
            after_count, after_ratio = _points_in_roi_stats(aligned_fg_points_per_cam[cam_id], roi_min, roi_max)
            fg_points_in_roi_before_align[cam_id] = float(before_ratio)
            fg_points_in_roi_after_align[cam_id] = float(after_ratio)
            fg_points_in_roi_before_align_count[cam_id] = int(before_count)
            fg_points_in_roi_after_align_count[cam_id] = int(after_count)
            fg_points_total[cam_id] = int(raw_fg_points_per_cam[cam_id].shape[0])
            if fg_points_total[cam_id] > 0:
                before_align_ratio_by_cam[cam_id].append(float(before_ratio))
                after_align_ratio_by_cam[cam_id].append(float(after_ratio))

        anchor_world_list = None
        if mask_anchor_world is not None:
            anchor_world_list = np.asarray(mask_anchor_world, dtype=np.float32).astype(float).tolist()

        constraint_row = {
            "scene_stem": scene_stem,
            "logical_t_idx": logical_t_idx,
            "roi_source": str(roi_source),
            "roi_bounds_xyz": roi_bounds_xyz,
            "roi_voxel_size_m": float(selected_result["roi_voxel_size_m"]),
            "mask_anchor_world": anchor_world_list,
            "anchor_ray_error_per_cam": frame_info["anchor_ray_error_per_cam"],
            "anchor_ray_error_mean": frame_info["anchor_ray_error_mean"],
            "anchor_depth_cam0": anchor_depths["cam0"],
            "anchor_depth_cam1": anchor_depths["cam1"],
            "anchor_depth_cam2": anchor_depths["cam2"],
            "depth_local_median": depth_local_median_by_cam,
            "depth_mask_pixels": depth_mask_pixels_by_cam,
            "depth_scale_raw": depth_scale_raw_by_cam,
            "depth_scale_used": depth_scale_used_by_cam,
            "depth_scale_source": depth_scale_source_by_cam,
            "fg_points_in_roi_before_align": fg_points_in_roi_before_align,
            "fg_points_in_roi_after_align": fg_points_in_roi_after_align,
            "fg_points_in_roi_before_align_count": fg_points_in_roi_before_align_count,
            "fg_points_in_roi_after_align_count": fg_points_in_roi_after_align_count,
            "fg_points_total": fg_points_total,
            "support_ge_2_voxels": int(
                best_support_ge_2_voxels if best_support_ge_2_voxels > 0 else selected_result["support_ge_2_voxels"]
            ),
            "mask_supported_voxels": int(selected_result["mask_supported_voxels"]),
            "surface_voxels": int(selected_result["surface_voxels"]),
            "selected_component_voxels": int(selected_result["selected_component_voxels"]),
            "depth_supported_voxels": int(selected_result["depth_supported_voxels"]),
            "depth_support_ratio": float(selected_result["depth_support_ratio"]),
            "trim_applied": int(bool(selected_result["trim_applied"])),
            "trim_rejected": int(bool(selected_result["trim_rejected"])),
            "trim_removed_voxels": int(selected_result["trim_removed_voxels"]),
            "depth_support_source": str(args.depth_support_source),
            "output_points": int(output_count),
            "constraint_mode": str(constraint_mode),
            "effective_min_mask_cam_support": int(1 if constraint_mode == "degraded_single_view_fallback" else args.min_mask_cam_support),
            "raw_candidate_points": int(total_candidate_points),
            "selection_mode": str(selected_result["selection_mode"]),
            "points_path": dynamic_out_path.relative_to(scene_fused_root).as_posix(),
            "points_by_timestamp_path": retrieval_out_path.relative_to(points_by_timestamp_dir.parent).as_posix(),
        }
        constraint_rows.append(constraint_row)
        retrieval_index_rows.append(
            {
                "logical_t_idx": int(logical_t_idx),
                "scene_stem": scene_stem,
                "points_rel": retrieval_out_path.relative_to(points_by_timestamp_dir).as_posix(),
                "num_points": int(output_count),
                "source_points_path": dynamic_out_path.relative_to(scene_fused_root).as_posix(),
                "constraint_mode": str(constraint_mode),
            }
        )

        updated_row = dict(row)
        updated_row["points_path"] = dynamic_out_path.relative_to(scene_fused_root).as_posix()
        updated_row["raw_points"] = int(total_candidate_points)
        updated_row["fused_points"] = int(output_count)
        updated_row["constraint_mode"] = str(constraint_mode)
        updated_row["constraint_output_points"] = int(output_count)
        updated_row["constraint_roi_source"] = str(roi_source)
        updated_row["constraint_roi_bounds_xyz"] = roi_bounds_xyz
        updated_row["constraint_roi_voxel_size_m"] = float(selected_result["roi_voxel_size_m"])
        updated_row["constraint_mask_anchor_world"] = anchor_world_list
        updated_row["constraint_anchor_ray_error_mean"] = frame_info["anchor_ray_error_mean"]
        updated_row["constraint_anchor_ray_error_per_cam"] = frame_info["anchor_ray_error_per_cam"]
        updated_row["constraint_anchor_depth_cam0"] = anchor_depths["cam0"]
        updated_row["constraint_anchor_depth_cam1"] = anchor_depths["cam1"]
        updated_row["constraint_anchor_depth_cam2"] = anchor_depths["cam2"]
        updated_row["constraint_depth_local_median"] = depth_local_median_by_cam
        updated_row["constraint_depth_mask_pixels"] = depth_mask_pixels_by_cam
        updated_row["constraint_depth_scale_raw"] = depth_scale_raw_by_cam
        updated_row["constraint_depth_scale_used"] = depth_scale_used_by_cam
        updated_row["constraint_depth_scale_source"] = depth_scale_source_by_cam
        updated_row["constraint_fg_points_in_roi_before_align"] = fg_points_in_roi_before_align
        updated_row["constraint_fg_points_in_roi_after_align"] = fg_points_in_roi_after_align
        updated_row["constraint_support_ge_2_voxels"] = int(
            best_support_ge_2_voxels if best_support_ge_2_voxels > 0 else selected_result["support_ge_2_voxels"]
        )
        updated_row["constraint_mask_supported_voxels"] = int(selected_result["mask_supported_voxels"])
        updated_row["constraint_surface_voxels"] = int(selected_result["surface_voxels"])
        updated_row["constraint_selected_component_voxels"] = int(selected_result["selected_component_voxels"])
        updated_row["constraint_depth_supported_voxels"] = int(selected_result["depth_supported_voxels"])
        updated_row["constraint_depth_support_ratio"] = float(selected_result["depth_support_ratio"])
        updated_row["constraint_effective_min_mask_cam_support"] = int(
            1 if constraint_mode == "degraded_single_view_fallback" else args.min_mask_cam_support
        )
        updated_row["constraint_trim_applied"] = int(bool(selected_result["trim_applied"]))
        updated_row["constraint_trim_rejected"] = int(bool(selected_result["trim_rejected"]))
        updated_row["constraint_trim_removed_voxels"] = int(selected_result["trim_removed_voxels"])
        updated_row["constraint_depth_support_source"] = str(args.depth_support_source)
        updated_row["constraint_selection_mode"] = str(selected_result["selection_mode"])
        updated_dynamic_rows.append(updated_row)

    per_camera_mean_after_align = {
        cam_id: float(np.mean(values)) if values else 0.0 for cam_id, values in after_align_ratio_by_cam.items()
    }
    per_camera_mean_before_align = {
        cam_id: float(np.mean(values)) if values else 0.0 for cam_id, values in before_align_ratio_by_cam.items()
    }
    all_after_align = [value for values in after_align_ratio_by_cam.values() for value in values]
    mean_fg_in_roi_after_align: dict[str, Any] = {
        "overall": float(np.mean(all_after_align)) if all_after_align else 0.0,
        "per_camera": per_camera_mean_after_align,
    }
    mean_fg_in_roi_before_align: dict[str, Any] = {
        "overall": float(np.mean([value for values in before_align_ratio_by_cam.values() for value in values]))
        if any(before_align_ratio_by_cam.values())
        else 0.0,
        "per_camera": per_camera_mean_before_align,
    }

    bad_cams = [cam_id for cam_id, mean_value in per_camera_mean_after_align.items() if float(mean_value) < 0.01]
    if bad_cams:
        raise SystemExit(
            "Aligned fg points still fail ROI guard after local→rig similarity alignment; "
            f"bad_cams={bad_cams}, mean_fg_in_roi_after_align={mean_fg_in_roi_after_align}"
        )

    updated_fieldnames = list(dynamic_rows[0].keys())
    extra_dynamic_fields = [
        "constraint_mode",
        "constraint_output_points",
        "constraint_roi_source",
        "constraint_roi_bounds_xyz",
        "constraint_roi_voxel_size_m",
        "constraint_mask_anchor_world",
        "constraint_anchor_ray_error_mean",
        "constraint_anchor_ray_error_per_cam",
        "constraint_anchor_depth_cam0",
        "constraint_anchor_depth_cam1",
        "constraint_anchor_depth_cam2",
        "constraint_depth_local_median",
        "constraint_depth_mask_pixels",
        "constraint_depth_scale_raw",
        "constraint_depth_scale_used",
        "constraint_depth_scale_source",
        "constraint_fg_points_in_roi_before_align",
        "constraint_fg_points_in_roi_after_align",
        "constraint_support_ge_2_voxels",
        "constraint_mask_supported_voxels",
        "constraint_surface_voxels",
        "constraint_selected_component_voxels",
        "constraint_depth_supported_voxels",
        "constraint_depth_support_ratio",
        "constraint_effective_min_mask_cam_support",
        "constraint_trim_applied",
        "constraint_trim_rejected",
        "constraint_trim_removed_voxels",
        "constraint_depth_support_source",
        "constraint_selection_mode",
    ]
    for field in extra_dynamic_fields:
        if field not in updated_fieldnames:
            updated_fieldnames.append(field)

    with dynamic_index_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=updated_fieldnames)
        writer.writeheader()
        for row in updated_dynamic_rows:
            writer.writerow({key: _serialize_csv_value(row.get(key, "")) for key in updated_fieldnames})

    constraint_index_path = scene_fused_root / "dynamic_constraint_index.csv"
    constraint_fieldnames = [
        "scene_stem",
        "logical_t_idx",
        "roi_source",
        "roi_bounds_xyz",
        "roi_voxel_size_m",
        "mask_anchor_world",
        "anchor_ray_error_per_cam",
        "anchor_ray_error_mean",
        "anchor_depth_cam0",
        "anchor_depth_cam1",
        "anchor_depth_cam2",
        "depth_local_median",
        "depth_mask_pixels",
        "depth_scale_raw",
        "depth_scale_used",
        "depth_scale_source",
        "fg_points_in_roi_before_align",
        "fg_points_in_roi_after_align",
        "fg_points_in_roi_before_align_count",
        "fg_points_in_roi_after_align_count",
        "fg_points_total",
        "support_ge_2_voxels",
        "mask_supported_voxels",
        "surface_voxels",
        "selected_component_voxels",
        "depth_supported_voxels",
        "depth_support_ratio",
        "trim_applied",
        "trim_rejected",
        "trim_removed_voxels",
        "depth_support_source",
        "output_points",
        "constraint_mode",
        "effective_min_mask_cam_support",
        "raw_candidate_points",
        "selection_mode",
        "points_path",
        "points_by_timestamp_path",
    ]
    with constraint_index_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=constraint_fieldnames)
        writer.writeheader()
        for row in constraint_rows:
            writer.writerow({key: _serialize_csv_value(row.get(key, "")) for key in constraint_fieldnames})

    retrieval_index_path = points_by_timestamp_dir / "index.csv"
    retrieval_fieldnames = [
        "logical_t_idx",
        "scene_stem",
        "points_rel",
        "num_points",
        "source_points_path",
        "constraint_mode",
    ]
    with retrieval_index_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=retrieval_fieldnames)
        writer.writeheader()
        for row in retrieval_index_rows:
            writer.writerow({key: _serialize_csv_value(row.get(key, "")) for key in retrieval_fieldnames})

    nonempty_dynamic_frames = int(sum(1 for count in output_counts if count > 0))
    degraded_frames = int(mode_counter.get("degraded_single_view_fallback", 0))
    fg_union_fallback_frames = int(roi_source_counter.get("fg_union_fallback", 0))
    multiview_supported_frames = int(len(constraint_rows) - degraded_frames)
    mean_anchor_ray_error = float(np.mean(anchor_ray_error_means)) if anchor_ray_error_means else None

    meta_payload = {
        "schema_version": "neoverse_multiview_dynamic_constraint_v4",
        "scene_id": scene_id,
        "scene_dir": scene_dir.as_posix(),
        "fg_points_root": fg_points_root.as_posix(),
        "fused_root": scene_fused_root.as_posix(),
        "points_by_timestamp_root": points_by_timestamp_dir.as_posix(),
        "dynamic_index_csv": dynamic_index_path.as_posix(),
        "dynamic_constraint_index_csv": constraint_index_path.as_posix(),
        "cams": cams,
        "legacy_frame_contract_assumed": bool(legacy_frame_contract_assumed),
        "point_coordinate_frame": str(points_meta.get("point_coordinate_frame") or "neoverse_render_world"),
        "point_frame_source": str(points_meta.get("point_frame_source") or "observation_render_c2w"),
        "render_depth_unit": str(points_meta.get("render_depth_unit") or "neoverse_local_metric_like"),
        "depth_support_source": str(args.depth_support_source),
        "params": {
            "hull_voxel_size_m": float(args.hull_voxel_size_m),
            "output_voxel_size_m": float(args.output_voxel_size_m),
            "roi_padding_m": float(args.roi_padding_m),
            "roi_padding_schedule_m": padding_schedule,
            "min_mask_cam_support": int(args.min_mask_cam_support),
            "point_support_radius_m": float(args.point_support_radius_m),
            "depth_trim_radius_m": float(args.depth_trim_radius_m),
            "min_trimmed_points": int(args.min_trimmed_points),
            "scale_guard_ratio": float(args.scale_guard_ratio),
            "min_depth_mask_pixels": int(args.min_depth_mask_pixels),
            "depth_support_source": str(args.depth_support_source),
            "legacy_depth_support_mode_arg": str(args.depth_support_mode),
            "hull_voxel_size_schedule_m": voxel_schedule,
            "max_roi_voxels": int(args.max_roi_voxels),
        },
        "num_timestamps": int(len(constraint_rows)),
        "nonempty_dynamic_frames": nonempty_dynamic_frames,
        "empty_dynamic_frames": int(len(constraint_rows) - nonempty_dynamic_frames),
        "multiview_supported_frames": int(multiview_supported_frames),
        "degraded_single_view_fallback_frames": int(degraded_frames),
        "fg_union_fallback_frames": int(fg_union_fallback_frames),
        "constraint_mode_counts": dict(sorted(mode_counter.items())),
        "roi_source_counts": dict(sorted(roi_source_counter.items())),
        "average_output_points": float(np.mean(output_counts)) if output_counts else 0.0,
        "total_output_points": int(sum(output_counts)),
        "mean_anchor_ray_error": mean_anchor_ray_error,
        "per_camera_scale_stats": per_camera_scale_stats,
        "trim_applied_frames": int(trim_applied_frames),
        "trim_rejected_frames": int(trim_rejected_frames),
        "trim_zero_support_frames": int(trim_zero_support_frames),
        "mean_fg_in_roi_before_align": mean_fg_in_roi_before_align,
        "mean_fg_in_roi_after_align": mean_fg_in_roi_after_align,
    }
    constraint_meta_path = scene_fused_root / "dynamic_constraint_meta.json"
    _write_json(constraint_meta_path, meta_payload)

    retrieval_meta_path = points_by_timestamp_dir / "meta.json"
    retrieval_meta = {
        "schema_version": "neoverse_points_by_timestamp_v1",
        "scene_id": scene_id,
        "scene_dir": scene_dir.as_posix(),
        "points_root": points_by_timestamp_dir.as_posix(),
        "num_timestamps": int(len(retrieval_index_rows)),
        "source_fused_root": scene_fused_root.as_posix(),
        "index_csv": retrieval_index_path.as_posix(),
    }
    _write_json(retrieval_meta_path, retrieval_meta)

    fusion_meta["dynamic_constraint"] = {
        "constraint_applied": True,
        "dynamic_constraint_index_csv": constraint_index_path.as_posix(),
        "dynamic_constraint_meta_json": constraint_meta_path.as_posix(),
        "points_by_timestamp_root": points_by_timestamp_dir.as_posix(),
        "points_by_timestamp_index_csv": retrieval_index_path.as_posix(),
        "points_by_timestamp_meta_json": retrieval_meta_path.as_posix(),
        "multiview_supported_frames": int(multiview_supported_frames),
        "degraded_single_view_fallback_frames": int(degraded_frames),
        "fg_union_fallback_frames": int(fg_union_fallback_frames),
        "constraint_mode_counts": dict(sorted(mode_counter.items())),
        "roi_source_counts": dict(sorted(roi_source_counter.items())),
        "mean_anchor_ray_error": mean_anchor_ray_error,
        "trim_applied_frames": int(trim_applied_frames),
        "trim_rejected_frames": int(trim_rejected_frames),
        "trim_zero_support_frames": int(trim_zero_support_frames),
        "per_camera_scale_stats": per_camera_scale_stats,
        "mean_fg_in_roi_after_align": mean_fg_in_roi_after_align,
        "params": dict(meta_payload["params"]),
    }
    dynamic_meta = dict(fusion_meta.get("dynamic", {}))
    dynamic_meta["constraint_applied"] = True
    dynamic_meta["constraint_stage"] = "mask_anchor_multiview_visual_hull_aligned_fg_depth_trim"
    dynamic_meta["points_by_timestamp_root"] = points_by_timestamp_dir.as_posix()
    dynamic_meta["points_by_timestamp_index_csv"] = retrieval_index_path.as_posix()
    dynamic_meta["points_by_timestamp_meta_json"] = retrieval_meta_path.as_posix()
    dynamic_meta["num_timestamps"] = int(len(updated_dynamic_rows))
    dynamic_meta["timestamps"] = updated_dynamic_rows
    fusion_meta["dynamic"] = dynamic_meta
    _write_json(fusion_meta_path, fusion_meta)

    print(f"Wrote constrained dynamic outputs to: {dynamic_dir}")
    print(f"Wrote retrieval points to: {points_by_timestamp_dir}")
    print(f"Wrote: {retrieval_index_path}")
    print(f"Wrote: {retrieval_meta_path}")
    print(f"Wrote: {dynamic_index_path}")
    print(f"Wrote: {constraint_index_path}")
    print(f"Wrote: {constraint_meta_path}")


if __name__ == "__main__":
    main()
