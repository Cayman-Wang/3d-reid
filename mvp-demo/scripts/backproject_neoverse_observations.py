from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"Failed to read json: {path}\nError: {exc!r}")


def _read_npy(path: Path) -> np.ndarray:
    try:
        return np.asarray(np.load(str(path)), dtype=np.float32)
    except Exception as exc:
        raise SystemExit(f"Failed to load npy: {path}\nError: {exc!r}")


def _read_image(path: Path, flags: int) -> np.ndarray:
    try:
        import cv2  # type: ignore
    except Exception as exc:
        raise SystemExit(f"Missing dependency: cv2 (opencv-python). Error: {exc!r}")

    path_str = str(path)
    if not path_str.isascii():
        try:
            data = np.fromfile(path_str, dtype=np.uint8)
            img = cv2.imdecode(data, flags)
        except Exception:
            img = None
        if img is not None:
            return np.asarray(img)

    img = cv2.imread(path_str, flags)
    if img is not None:
        return np.asarray(img)
    try:
        data = np.fromfile(path_str, dtype=np.uint8)
        img = cv2.imdecode(data, flags)
    except Exception:
        img = None
    if img is None:
        raise SystemExit(f"Failed to read image: {path}")
    return np.asarray(img)


def _read_gray(path: Path) -> np.ndarray:
    return _read_image(path, flags=0)


def _parse_crop_box_json(raw_value: str) -> list[int] | None:
    text = str(raw_value or "").strip()
    if not text or text.lower() == "null":
        return None
    try:
        payload = json.loads(text)
    except Exception as exc:
        raise SystemExit(f"Failed to parse crop_box_xyxy from observation index: {raw_value!r}; error={exc!r}")
    if payload is None:
        return None
    if not isinstance(payload, list) or len(payload) != 4:
        raise SystemExit(f"Invalid crop_box_xyxy payload: {payload!r}")
    box = [int(v) for v in payload]
    if box[2] <= box[0] or box[3] <= box[1]:
        raise SystemExit(f"Invalid crop_box_xyxy extents: {box}")
    return box


def _prepare_mask(
    mask_u8: np.ndarray,
    target_width: int,
    target_height: int,
    resize_mode: str,
    crop_box_xyxy: list[int] | None = None,
) -> np.ndarray:
    try:
        from PIL import Image
    except Exception as exc:
        raise SystemExit(f"Missing PIL dependency: {exc!r}")

    src = np.asarray(mask_u8, dtype=np.uint8)
    if src.ndim == 3:
        src = src[..., 0]
    if src.ndim != 2:
        raise SystemExit(f"Unsupported mask shape: {src.shape}")

    if crop_box_xyxy is not None:
        left, top, right, bottom = [int(v) for v in crop_box_xyxy]
        left = max(0, left)
        top = max(0, top)
        right = min(src.shape[1], right)
        bottom = min(src.shape[0], bottom)
        if right <= left or bottom <= top:
            raise SystemExit(f"Invalid crop_box_xyxy after clipping: {crop_box_xyxy}")
        src = src[top:bottom, left:right]

    tw = int(target_width)
    th = int(target_height)
    if tw <= 0 or th <= 0:
        raise SystemExit(f"Invalid mask target size: width={tw}, height={th}")

    mode = str(resize_mode or "").strip().lower()
    img = Image.fromarray(src)
    src_h, src_w = src.shape[:2]

    if mode == "resize":
        out = img.resize((tw, th), resample=Image.NEAREST)
        return np.asarray(out, dtype=np.uint8)

    if mode == "center_crop":
        scale = max(float(tw) / float(src_w), float(th) / float(src_h))
        resized_w = max(1, int(round(src_w * scale)))
        resized_h = max(1, int(round(src_h * scale)))
        resized = img.resize((resized_w, resized_h), resample=Image.NEAREST)
        left = (resized_w - tw) // 2
        top = (resized_h - th) // 2
        right = left + tw
        bottom = top + th
        cropped = resized.crop((left, top, right, bottom))
        return np.asarray(cropped, dtype=np.uint8)

    raise SystemExit(f"Unsupported resize_mode in observation index: {resize_mode!r}")


def _dilate_mask(mask_bool: np.ndarray, radius_px: int) -> np.ndarray:
    mask = np.asarray(mask_bool, dtype=bool)
    radius = int(radius_px)
    if radius <= 0 or mask.size == 0:
        return mask
    try:
        from PIL import Image, ImageFilter
    except Exception as exc:
        raise SystemExit(f"Missing PIL dependency for mask dilation: {exc!r}")
    mask_u8 = (mask.astype(np.uint8) * 255)
    filter_size = int(radius * 2 + 1)
    dilated = Image.fromarray(mask_u8).filter(ImageFilter.MaxFilter(size=filter_size))
    return np.asarray(dilated, dtype=np.uint8) > 0


def _depth_to_camera_points(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 2:
        raise SystemExit(f"Unsupported depth shape: {depth.shape}")

    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])
    if fx <= 0 or fy <= 0:
        raise SystemExit(f"Invalid camera intrinsics: fx={fx} fy={fy}")

    h, w = depth.shape
    v, u = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")
    z = depth
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    pts = np.stack([x, y, z], axis=-1)
    return pts.reshape(-1, 3)


def _transform_points(T_node_from_cam: np.ndarray, points_cam: np.ndarray) -> np.ndarray:
    R = np.asarray(T_node_from_cam, dtype=np.float32)[:3, :3]
    t = np.asarray(T_node_from_cam, dtype=np.float32)[:3, 3]
    return (points_cam @ R.T) + t[None, :]


def _voxel_downsample(points_xyz: np.ndarray, voxel_size_m: float) -> np.ndarray:
    pts = np.asarray(points_xyz, dtype=np.float32)
    if pts.size == 0 or voxel_size_m <= 0:
        return pts
    vox = np.floor(pts / float(voxel_size_m)).astype(np.int64)
    _, uniq_idx = np.unique(vox, axis=0, return_index=True)
    uniq_idx = np.sort(uniq_idx)
    return pts[uniq_idx]


def _cap_points(points_xyz: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    pts = np.asarray(points_xyz, dtype=np.float32)
    if max_points <= 0 or pts.shape[0] <= max_points:
        return pts
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(pts.shape[0], size=int(max_points), replace=False)
    idx.sort()
    return pts[idx]


def _voxel_downsample_indices(points_xyz: np.ndarray, voxel_size_m: float) -> np.ndarray:
    pts = np.asarray(points_xyz, dtype=np.float32)
    if pts.size == 0:
        return np.zeros((0,), dtype=np.int64)
    if voxel_size_m <= 0:
        return np.arange(pts.shape[0], dtype=np.int64)
    vox = np.floor(pts / float(voxel_size_m)).astype(np.int64)
    _, uniq_idx = np.unique(vox, axis=0, return_index=True)
    return np.sort(uniq_idx).astype(np.int64)


def _cap_indices(num_points: int, max_points: int, seed: int) -> np.ndarray:
    if num_points <= 0:
        return np.zeros((0,), dtype=np.int64)
    if max_points <= 0 or num_points <= max_points:
        return np.arange(num_points, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(num_points, size=int(max_points), replace=False)
    return np.sort(idx).astype(np.int64)


def _parse_matrix_json(raw_value: str, expected_shape: tuple[int, int], field_name: str) -> np.ndarray:
    try:
        mat = np.asarray(json.loads(str(raw_value)), dtype=np.float32)
    except Exception as exc:
        raise SystemExit(f"Failed to parse {field_name} from observation index: {raw_value!r}; error={exc!r}")
    if mat.shape != expected_shape:
        raise SystemExit(f"Invalid {field_name} shape: expected {expected_shape}, got {mat.shape}")
    return mat


def _reduce_points_with_attrs(
    points_world: np.ndarray,
    alpha_values: np.ndarray,
    rgb_values: np.ndarray | None,
    voxel_size_m: float,
    max_points: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    idx = _voxel_downsample_indices(points_world, voxel_size_m)
    pts = np.asarray(points_world[idx], dtype=np.float32)
    alp = np.asarray(alpha_values[idx], dtype=np.float32)
    rgb = None if rgb_values is None else np.asarray(rgb_values[idx], dtype=np.uint8)

    idx_cap = _cap_indices(pts.shape[0], max_points=max_points, seed=seed)
    pts = np.asarray(pts[idx_cap], dtype=np.float32)
    alp = np.asarray(alp[idx_cap], dtype=np.float32)
    rgb = None if rgb is None else np.asarray(rgb[idx_cap], dtype=np.uint8)
    return pts, alp, rgb


def _resolve_mask_path(scene_dir: Path, cam_id: str, scene_stem: str, mask_subdir: str) -> tuple[Path | None, str | None]:
    requested = str(mask_subdir).strip()
    if not requested:
        raise SystemExit("--mask_subdir is empty")
    requested_norm = requested.lower()
    candidates = ["masks_gt", "masks"] if requested_norm == "auto" else [requested]
    for subdir in candidates:
        candidate = scene_dir / "cams" / cam_id / subdir / f"{scene_stem}.png"
        if candidate.exists():
            return candidate, subdir
    return None, None


def main() -> None:
    ap = argparse.ArgumentParser(description="Backproject NeoVerse observations into rig-world bg/fg point clouds.")
    ap.add_argument("--scene_dir", required=True, type=str)
    ap.add_argument("--observations_root", default="mvp-demo/output/neoverse_fused", type=str)
    ap.add_argument("--cams", default="cam0,cam1,cam2", type=str)
    ap.add_argument("--alpha_thresh", default=None, type=float, help="Compatibility alias used when fg/bg thresholds are not provided")
    ap.add_argument("--voxel_size_m", default=None, type=float, help="Compatibility alias used when fg/bg voxel sizes are not provided")
    ap.add_argument("--fg_alpha_thresh", default=None, type=float)
    ap.add_argument("--bg_alpha_thresh", default=None, type=float)
    ap.add_argument("--fg_voxel_size_m", default=None, type=float)
    ap.add_argument("--bg_voxel_size_m", default=None, type=float)
    ap.add_argument("--mask_dilate_px", default=3, type=int)
    ap.add_argument("--mask_subdir", default="auto", type=str)
    ap.add_argument("--max_points_per_view", default=50000, type=int)
    ap.add_argument("--camera_source", default="rendered", choices=["rendered"], type=str)
    ap.add_argument("--out_root", default="mvp-demo/output/neoverse_fused", type=str)
    args = ap.parse_args()

    repo = _repo_root()
    scene_dir = Path(str(args.scene_dir)).resolve()
    if not scene_dir.exists():
        raise SystemExit(f"Missing --scene_dir: {scene_dir}")

    observations_root = Path(str(args.observations_root))
    if not observations_root.is_absolute():
        observations_root = repo / observations_root
    out_root = Path(str(args.out_root))
    if not out_root.is_absolute():
        out_root = repo / out_root

    cams = [c.strip() for c in str(args.cams).split(",") if c.strip()]
    if cams != ["cam0", "cam1", "cam2"]:
        raise SystemExit(f"This first version only supports cams=['cam0','cam1','cam2']. Got: {cams}")

    fg_alpha_thresh = (
        float(args.fg_alpha_thresh)
        if args.fg_alpha_thresh is not None
        else float(args.alpha_thresh)
        if args.alpha_thresh is not None
        else 0.01
    )
    bg_alpha_thresh = (
        float(args.bg_alpha_thresh)
        if args.bg_alpha_thresh is not None
        else float(args.alpha_thresh)
        if args.alpha_thresh is not None
        else 0.02
    )
    fg_voxel_size_m = (
        float(args.fg_voxel_size_m)
        if args.fg_voxel_size_m is not None
        else float(args.voxel_size_m)
        if args.voxel_size_m is not None
        else 0.005
    )
    bg_voxel_size_m = (
        float(args.bg_voxel_size_m)
        if args.bg_voxel_size_m is not None
        else float(args.voxel_size_m)
        if args.voxel_size_m is not None
        else 0.02
    )
    mask_dilate_px = max(int(args.mask_dilate_px), 0)

    scene_id = scene_dir.name
    obs_root = observations_root / scene_id / "observations"
    points_root = out_root / scene_id / "points_per_view"
    points_root.mkdir(parents=True, exist_ok=True)
    (points_root / "bg").mkdir(parents=True, exist_ok=True)
    (points_root / "fg").mkdir(parents=True, exist_ok=True)

    index_path = obs_root / "index.csv"
    if not index_path.exists():
        raise SystemExit(f"Missing observations index: {index_path}")

    rows = list(csv.DictReader(index_path.open("r", encoding="utf-8", newline="")))
    if not rows:
        raise SystemExit(f"No observations found in: {index_path}")

    index_rows: list[dict[str, Any]] = []
    totals = {"bg": 0, "fg": 0}
    skipped = 0

    for row in rows:
        cam_id = str(row["cam_id"])
        scene_stem = str(row["scene_stem"])
        logical_t_idx = int(float(row["logical_t_idx"]))
        row_camera_source = str(row.get("camera_source") or "")
        if row_camera_source != str(args.camera_source):
            raise SystemExit(
                f"Unsupported camera_source row={row_camera_source!r} for {cam_id}/{scene_stem}; "
                f"this script currently requires camera_source={args.camera_source!r}."
            )

        depth_path = obs_root / str(row["depth_path"])
        alpha_path = obs_root / str(row["alpha_path"])
        rgb_path = obs_root / str(row["rgb_path"])
        if not depth_path.exists() or not alpha_path.exists():
            raise SystemExit(f"Missing depth/alpha for row: {row}")

        depth = _read_npy(depth_path)
        alpha = _read_npy(alpha_path)
        if alpha.ndim == 3 and alpha.shape[-1] == 1:
            alpha = alpha[..., 0]
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        if depth.shape != alpha.shape:
            raise SystemExit(f"Depth/alpha shape mismatch: depth={depth.shape} alpha={alpha.shape} path={depth_path}")

        mask_path, resolved_mask_subdir = _resolve_mask_path(scene_dir, cam_id, scene_stem, str(args.mask_subdir))
        if mask_path is None:
            raise SystemExit(
                f"Missing mask for {cam_id} {scene_stem} under requested mask_subdir={args.mask_subdir!r}"
            )

        row_width_raw = str(row.get("width") or "").strip()
        row_height_raw = str(row.get("height") or "").strip()
        row_resize_mode = str(row.get("resize_mode") or "").strip().lower()
        crop_applied = bool(int(float(str(row.get("crop_applied") or "0"))))
        crop_box_xyxy = _parse_crop_box_json(str(row.get("crop_box_xyxy") or "")) if crop_applied else None
        if not row_width_raw or not row_height_raw:
            raise SystemExit(f"Observation index missing width/height for {cam_id}/{scene_stem}")
        if row_resize_mode not in {"resize", "center_crop"}:
            raise SystemExit(f"Observation index has invalid resize_mode={row_resize_mode!r} for {cam_id}/{scene_stem}")

        target_w = int(float(row_width_raw))
        target_h = int(float(row_height_raw))
        mask_u8 = _prepare_mask(
            _read_gray(mask_path),
            target_width=target_w,
            target_height=target_h,
            resize_mode=row_resize_mode,
            crop_box_xyxy=crop_box_xyxy,
        )
        if mask_u8.shape != depth.shape:
            raise SystemExit(
                f"Mask/depth shape mismatch after preprocess for {cam_id}/{scene_stem}: "
                f"mask={mask_u8.shape}, depth={depth.shape}, resize_mode={row_resize_mode}"
            )

        if not str(row.get("render_K") or "").strip():
            raise SystemExit(f"Observation index missing render_K for {cam_id}/{scene_stem}")
        if not str(row.get("render_c2w") or "").strip():
            raise SystemExit(f"Observation index missing render_c2w for {cam_id}/{scene_stem}")
        K = _parse_matrix_json(str(row.get("render_K")), (3, 3), "render_K")
        render_c2w = _parse_matrix_json(str(row.get("render_c2w")), (4, 4), "render_c2w")

        pts_cam = _depth_to_camera_points(depth, K)
        alpha_flat = alpha.reshape(-1)
        mask_bool = _dilate_mask(mask_u8 > 0, radius_px=mask_dilate_px)
        mask_flat = mask_bool.reshape(-1)
        valid_geom = np.isfinite(pts_cam[:, 2]) & (pts_cam[:, 2] > 0)

        fg_valid = valid_geom & mask_flat & (alpha_flat > float(fg_alpha_thresh))
        bg_valid = valid_geom & (~mask_flat) & (alpha_flat > float(bg_alpha_thresh))

        pts_fg = pts_cam[fg_valid]
        pts_bg = pts_cam[bg_valid]
        alpha_fg = np.asarray(alpha_flat[fg_valid], dtype=np.float32)
        alpha_bg = np.asarray(alpha_flat[bg_valid], dtype=np.float32)
        pts_fg_world = _transform_points(render_c2w, pts_fg) if pts_fg.size else np.zeros((0, 3), dtype=np.float32)
        pts_bg_world = _transform_points(render_c2w, pts_bg) if pts_bg.size else np.zeros((0, 3), dtype=np.float32)

        rgb = None
        if rgb_path.exists():
            rgb_bgr = _read_image(rgb_path, flags=1)
            rgb = np.asarray(rgb_bgr, dtype=np.uint8).reshape(-1, rgb_bgr.shape[-1])
            rgb = rgb[:, ::-1]

        rgb_fg = None if rgb is None else np.asarray(rgb[fg_valid], dtype=np.uint8)
        rgb_bg = None if rgb is None else np.asarray(rgb[bg_valid], dtype=np.uint8)

        pts_fg_world, alpha_fg, rgb_fg = _reduce_points_with_attrs(
            points_world=pts_fg_world,
            alpha_values=alpha_fg,
            rgb_values=rgb_fg,
            voxel_size_m=float(fg_voxel_size_m),
            max_points=int(args.max_points_per_view),
            seed=logical_t_idx * 31 + 7,
        )
        pts_bg_world, alpha_bg, rgb_bg = _reduce_points_with_attrs(
            points_world=pts_bg_world,
            alpha_values=alpha_bg,
            rgb_values=rgb_bg,
            voxel_size_m=float(bg_voxel_size_m),
            max_points=int(args.max_points_per_view),
            seed=logical_t_idx * 31 + 13,
        )

        if pts_bg_world.shape[0] != alpha_bg.shape[0]:
            raise SystemExit(f"Contract violation bg xyz/alpha length mismatch for {cam_id}/{scene_stem}")
        if pts_fg_world.shape[0] != alpha_fg.shape[0]:
            raise SystemExit(f"Contract violation fg xyz/alpha length mismatch for {cam_id}/{scene_stem}")
        if rgb_bg is not None and pts_bg_world.shape[0] != rgb_bg.shape[0]:
            raise SystemExit(f"Contract violation bg xyz/rgb length mismatch for {cam_id}/{scene_stem}")
        if rgb_fg is not None and pts_fg_world.shape[0] != rgb_fg.shape[0]:
            raise SystemExit(f"Contract violation fg xyz/rgb length mismatch for {cam_id}/{scene_stem}")

        bg_out = points_root / "bg" / cam_id
        fg_out = points_root / "fg" / cam_id
        bg_out.mkdir(parents=True, exist_ok=True)
        fg_out.mkdir(parents=True, exist_ok=True)

        bg_path = bg_out / f"{scene_stem}.npz"
        fg_path = fg_out / f"{scene_stem}.npz"
        bg_payload = {
            "xyz": pts_bg_world.astype(np.float32),
            "alpha": np.asarray(alpha_bg, dtype=np.float32),
            "cam_id": np.array(cam_id),
            "logical_t_idx": np.array(logical_t_idx, dtype=np.int32),
            "scene_stem": np.array(scene_stem),
            "coordinate_frame": np.array("neoverse_render_world"),
            "render_c2w": np.asarray(render_c2w, dtype=np.float32),
            "render_K": np.asarray(K, dtype=np.float32),
            "camera_center_world": np.asarray(render_c2w[:3, 3], dtype=np.float32),
        }
        fg_payload = {
            "xyz": pts_fg_world.astype(np.float32),
            "alpha": np.asarray(alpha_fg, dtype=np.float32),
            "cam_id": np.array(cam_id),
            "logical_t_idx": np.array(logical_t_idx, dtype=np.int32),
            "scene_stem": np.array(scene_stem),
            "coordinate_frame": np.array("neoverse_render_world"),
            "render_c2w": np.asarray(render_c2w, dtype=np.float32),
            "render_K": np.asarray(K, dtype=np.float32),
            "camera_center_world": np.asarray(render_c2w[:3, 3], dtype=np.float32),
        }
        if rgb_bg is not None:
            bg_payload["rgb"] = np.asarray(rgb_bg, dtype=np.uint8)
            fg_payload["rgb"] = np.asarray(rgb_fg, dtype=np.uint8)

        np.savez_compressed(bg_path, **bg_payload)
        np.savez_compressed(fg_path, **fg_payload)

        totals["bg"] += int(pts_bg_world.shape[0])
        totals["fg"] += int(pts_fg_world.shape[0])
        index_rows.append(
            {
                "cam_id": cam_id,
                "scene_stem": scene_stem,
                "logical_t_idx": logical_t_idx,
                "bg_points": int(pts_bg_world.shape[0]),
                "fg_points": int(pts_fg_world.shape[0]),
                "bg_path": bg_path.relative_to(points_root).as_posix(),
                "fg_path": fg_path.relative_to(points_root).as_posix(),
                "mask_subdir": str(resolved_mask_subdir or args.mask_subdir),
            }
        )

    with (points_root / "points_index.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["cam_id", "scene_stem", "logical_t_idx", "bg_points", "fg_points", "bg_path", "fg_path", "mask_subdir"],
        )
        writer.writeheader()
        for row in index_rows:
            writer.writerow(row)

    meta = {
        "schema_version": "neoverse_fused_backproject_v2",
        "scene_id": scene_id,
        "scene_dir": scene_dir.as_posix(),
        "observations_root": obs_root.as_posix(),
        "points_root": points_root.as_posix(),
        "cams": cams,
        "point_coordinate_frame": "neoverse_render_world",
        "point_frame_source": "observation_render_c2w",
        "render_depth_unit": "neoverse_local_metric_like",
        "alpha_thresh": None if args.alpha_thresh is None else float(args.alpha_thresh),
        "fg_alpha_thresh": float(fg_alpha_thresh),
        "bg_alpha_thresh": float(bg_alpha_thresh),
        "camera_source": str(args.camera_source),
        "voxel_size_m": None if args.voxel_size_m is None else float(args.voxel_size_m),
        "fg_voxel_size_m": float(fg_voxel_size_m),
        "bg_voxel_size_m": float(bg_voxel_size_m),
        "mask_dilate_px": int(mask_dilate_px),
        "mask_subdir": str(args.mask_subdir),
        "max_points_per_view": int(args.max_points_per_view),
        "num_rows": len(index_rows),
        "num_skipped": skipped,
        "total_bg_points": int(totals["bg"]),
        "total_fg_points": int(totals["fg"]),
    }
    (points_root / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote backprojected points to: {points_root}")
    print(f"Wrote: {points_root / 'points_index.csv'}")
    print(f"Wrote: {points_root / 'meta.json'}")


if __name__ == "__main__":
    main()
