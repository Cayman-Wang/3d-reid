from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:  # pragma: no cover
        raise SystemExit(f"Failed to read json: {path}\nError: {e!r}")


def _to_2d_depth(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        # Common variants: (H,W,1) or (1,H,W)
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
    depth2d = _to_2d_depth(np.asarray(depth))
    return np.asarray(depth2d, dtype=np.float32)


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


def _sorted_common_stems(per_cam_stems: List[set[str]]) -> List[str]:
    if not per_cam_stems:
        return []
    common = set.intersection(*per_cam_stems)
    # Prefer numeric sort if stems are timestamps.
    try:
        return [s for s in sorted(common, key=lambda x: int(x))]
    except Exception:
        return sorted(common)


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


def _maybe_subsample(points_xyz: np.ndarray, max_points: int, rng: np.random.Generator) -> np.ndarray:
    if max_points <= 0 or points_xyz.shape[0] <= max_points:
        return points_xyz
    idx = rng.choice(points_xyz.shape[0], size=int(max_points), replace=False)
    return points_xyz[idx]


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
        # depth is Euclidean range along the pixel ray; convert to z-depth via ray normalization.
        ray_norm = np.sqrt(x_n * x_n + y_n * y_n + 1.0)
        z = d_f / ray_norm
        x = x_n * z
        y = y_n * z
    else:
        raise ValueError(f"Unknown depth_mode: {depth_mode}")

    pts = np.stack([x, y, z], axis=1).astype(np.float32)
    pts = _maybe_subsample(pts, int(max_points), rng)
    return pts


def _transform_points(T_dst_from_src: np.ndarray, points_src: np.ndarray) -> np.ndarray:
    R = T_dst_from_src[:3, :3].astype(np.float32)
    t = T_dst_from_src[:3, 3].astype(np.float32)
    pts = np.asarray(points_src, dtype=np.float32)
    return (pts @ R.T) + t[None, :]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Fuse multi-camera depth(+mask) into per-timestamp point clouds in the node frame."
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
        "--cams",
        default="cam0,cam1,cam2",
        type=str,
        help='Comma-separated cam ids (must match rig.json keys). Default: "cam0,cam1,cam2".',
    )
    ap.add_argument(
        "--depth_subdir",
        default="depth",
        type=str,
        help='Depth folder name under each cam. Default: "depth" (expects cams/<cam>/depth/*.npy).',
    )
    ap.add_argument(
        "--mask_subdir",
        default="masks",
        type=str,
        help='Mask folder name under each cam. Default: "masks" (expects cams/<cam>/masks/*.png).',
    )
    ap.add_argument(
        "--depth_mode",
        default="z",
        choices=["z", "range"],
        help='Interpretation of depth values. "z": z-depth in camera coords (default). "range": Euclidean range.',
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
        help="Voxel size for fused downsampling (meters). 0=disable.",
    )
    ap.add_argument("--limit", default=0, type=int, help="Process only first N timestamps. 0=all.")
    ap.add_argument(
        "--ts",
        default="",
        type=str,
        help='Optional single timestamp stem to process (e.g. "000000123456").',
    )
    ap.add_argument("--seed", default=0, type=int, help="RNG seed for subsampling.")
    ap.add_argument(
        "--out_subdir",
        default="recon/points_fused",
        type=str,
        help='Relative output subdir under scene_dir. Default: "recon/points_fused".',
    )
    ap.add_argument("--write_ply", action="store_true", help="Also write recon/points_fused_ply/<ts>.ply (debug).")
    args = ap.parse_args()

    scene_dir = Path(args.scene_dir).resolve()
    if not scene_dir.exists():
        raise SystemExit(f"--scene_dir does not exist: {scene_dir}")

    rig_path = Path(args.rig_json).resolve() if str(args.rig_json).strip() else (scene_dir / "calib" / "rig.json")
    if not rig_path.exists():
        raise SystemExit(f"rig.json not found: {rig_path}")
    rig = _load_json(rig_path)

    cams = [c.strip() for c in str(args.cams).split(",") if c.strip()]
    if not cams:
        raise SystemExit("--cams is empty")

    rig_cams = rig.get("cameras", {})
    for c in cams:
        if c not in rig_cams:
            raise SystemExit(f'Camera "{c}" not found in rig.json. Available: {sorted(rig_cams.keys())}')

    # Depth max: prefer scene/capture_meta.json zfar if present.
    depth_max_m = float(args.depth_max_m)
    if depth_max_m <= 0:
        cap_meta_path = scene_dir / "capture_meta.json"
        if cap_meta_path.exists():
            cap_meta = _load_json(cap_meta_path)
            try:
                depth_max_m = float(cap_meta["render"]["zfar_m"])
            except Exception:
                depth_max_m = float("inf")
        else:
            depth_max_m = float("inf")

    rng = np.random.default_rng(int(args.seed))

    # Collect common stems across selected cameras.
    stems_per_cam: List[set[str]] = []
    for c in cams:
        depth_dir = scene_dir / "cams" / c / str(args.depth_subdir)
        if not depth_dir.exists():
            raise SystemExit(f"Depth dir not found for {c}: {depth_dir} (did you capture with --save_depth?)")
        stems_per_cam.append({p.stem for p in depth_dir.glob("*.npy")})

    stems = _sorted_common_stems(stems_per_cam)
    if str(args.ts).strip():
        stems = [str(args.ts).strip()]
    if int(args.limit) > 0:
        stems = stems[: int(args.limit)]
    if not stems:
        raise SystemExit("No timestamps found to process (empty depth folders?)")

    out_rel = Path(str(args.out_subdir))
    out_dir = out_rel if out_rel.is_absolute() else (scene_dir / out_rel)
    ply_rel = out_rel.parent / f"{out_rel.name}_ply"
    ply_dir = ply_rel if ply_rel.is_absolute() else (scene_dir / ply_rel)
    _ensure_dir(out_dir)
    if args.write_ply:
        _ensure_dir(ply_dir)

    index_csv_path = out_dir / "points_index.csv"
    index_rows: List[Tuple[str, int]] = []

    for stem in stems:
        pts_all: List[np.ndarray] = []
        per_cam_counts: Dict[str, int] = {}
        for c in cams:
            cam_entry = rig_cams[c]
            K = np.asarray(cam_entry["K"], dtype=np.float32)
            if K.shape != (3, 3):
                raise SystemExit(f'Invalid K shape for "{c}": {K.shape}')
            T_n_c = np.asarray(cam_entry["T_node_from_cam"], dtype=np.float32)
            if T_n_c.shape != (4, 4):
                raise SystemExit(f'Invalid T_node_from_cam shape for "{c}": {T_n_c.shape}')

            depth_path = scene_dir / "cams" / c / str(args.depth_subdir) / f"{stem}.npy"
            mask_path = scene_dir / "cams" / c / str(args.mask_subdir) / f"{stem}.png"
            if not depth_path.exists():
                raise SystemExit(f"Missing depth: {depth_path}")
            if not mask_path.exists():
                raise SystemExit(f"Missing mask: {mask_path}")

            depth = _read_depth_npy(depth_path)
            mask_u8 = _read_mask_u8(mask_path)

            # Optional sanity check vs rig.json image_size.
            try:
                w_exp, h_exp = int(cam_entry["image_size"][0]), int(cam_entry["image_size"][1])
                if (depth.shape[1], depth.shape[0]) != (w_exp, h_exp):
                    raise SystemExit(
                        f'Shape mismatch for "{c}": depth={depth.shape} rig.image_size={[w_exp, h_exp]}'
                    )
            except KeyError:
                pass

            pts_cam = _backproject(
                depth,
                mask_u8,
                K,
                depth_mode=str(args.depth_mode),
                depth_min_m=float(args.depth_min_m),
                depth_max_m=float(depth_max_m),
                max_points=int(args.max_points_per_cam),
                rng=rng,
            )
            per_cam_counts[c] = int(pts_cam.shape[0])
            if pts_cam.shape[0] == 0:
                continue
            pts_node = _transform_points(T_n_c, pts_cam)
            pts_all.append(pts_node)

        if not pts_all:
            # Still write an empty file so downstream can keep a consistent timeline.
            fused = np.zeros((0, 3), dtype=np.float32)
        else:
            fused = np.concatenate(pts_all, axis=0).astype(np.float32, copy=False)
            fused = _voxel_downsample_first(fused, float(args.voxel_size_m))

        out_path = out_dir / f"{stem}.npy"
        np.save(str(out_path), fused.astype(np.float32))
        if args.write_ply:
            _write_ply_xyz(ply_dir / f"{stem}.ply", fused)

        index_rows.append((stem, int(fused.shape[0])))

    # Write small metadata files for reproducibility.
    meta = {
        "scene_dir": str(scene_dir),
        "rig_json": str(rig_path),
        "cams": cams,
        "out_subdir": str(args.out_subdir),
        "depth_subdir": str(args.depth_subdir),
        "mask_subdir": str(args.mask_subdir),
        "depth_mode": str(args.depth_mode),
        "depth_min_m": float(args.depth_min_m),
        "depth_max_m": float(depth_max_m),
        "max_points_per_cam": int(args.max_points_per_cam),
        "voxel_size_m": float(args.voxel_size_m),
        "count": int(len(index_rows)),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    with index_csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ts_stem", "n_points", "filename"])
        for stem, n in index_rows:
            w.writerow([stem, n, f"{stem}.npy"])


if __name__ == "__main__":
    main()
