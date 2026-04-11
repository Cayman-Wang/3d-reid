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
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, int]]:
    pts_all: list[np.ndarray] = []
    per_cam_counts: dict[str, int] = {}

    for cam_id in cams:
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
        mask_u8 = _read_mask_u8(mask_path)

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
    return fused, per_cam_counts


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

    out_rel = Path(str(args.out_subdir))
    out_dir = out_rel if out_rel.is_absolute() else (scene_dir / out_rel)
    ply_rel = out_rel.parent / f"{out_rel.name}_ply"
    ply_dir = ply_rel if ply_rel.is_absolute() else (scene_dir / ply_rel)
    _ensure_dir(out_dir)
    if args.write_ply:
        _ensure_dir(ply_dir)

    canonical_chunks: list[np.ndarray] = []
    input_point_rows: list[tuple[str, int]] = []
    input_frames_with_points = 0

    for ts_us, _ in selected_rows:
        stem = f"{ts_us:012d}"
        t_sec = float(ts_us) / 1e6
        _, T_target_from_node = _pose_from_capture_meta(capture_meta, t_sec)
        points_node, _ = _build_points_node_for_stem(
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
            rng=rng,
        )
        input_point_rows.append((stem, int(points_node.shape[0])))
        if points_node.shape[0] == 0:
            continue
        input_frames_with_points += 1
        canonical_chunks.append(_transform_points(T_target_from_node, points_node))

    if not canonical_chunks:
        raise SystemExit("Canonical aggregation failed: all timestamps produced empty point clouds.")

    canonical_points = np.concatenate(canonical_chunks, axis=0).astype(np.float32, copy=False)
    canonical_points = _voxel_downsample_first(canonical_points, float(args.canonical_voxel_size_m))
    canonical_points = _maybe_subsample(canonical_points, int(args.max_canonical_points), rng)
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
        "depth_mode": str(args.depth_mode),
        "depth_min_m": float(args.depth_min_m),
        "depth_max_m": float(depth_max_m),
        "max_points_per_cam": int(args.max_points_per_cam),
        "voxel_size_m": float(args.voxel_size_m),
        "canonical_voxel_size_m": float(args.canonical_voxel_size_m),
        "min_canonical_points": int(args.min_canonical_points),
        "max_canonical_points": int(args.max_canonical_points),
        "count": int(len(index_rows)),
        "frames_with_input_points": int(input_frames_with_points),
        "frames_without_input_points": int(len(index_rows) - input_frames_with_points),
        "canonical_point_count": int(canonical_points.shape[0]),
        "traj": str(target_meta.get("traj") or ""),
        "yaw_start_deg": _float_or(target_meta.get("yaw_start_deg"), 0.0),
        "yaw_end_deg": _float_or(target_meta.get("yaw_end_deg"), 0.0),
        "pitch_amp_deg": _float_or(target_meta.get("pitch_amp_deg"), 0.0),
        "pitch_period": _float_or(target_meta.get("pitch_period"), 0.0),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    with (out_dir / "points_index.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ts_stem", "n_points", "filename"])
        for stem, n_points in index_rows:
            w.writerow([stem, n_points, f"{stem}.npy"])

    with (out_dir / "input_points_index.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ts_stem", "n_input_points"])
        for stem, n_points in input_point_rows:
            w.writerow([stem, n_points])

    print(f"[ok] scene={scene_dir.name}")
    print(f"[ok] out_dir={out_dir}")
    print(f"[ok] timestamps={len(index_rows)} canonical_points={canonical_points.shape[0]}")


if __name__ == "__main__":
    main()
