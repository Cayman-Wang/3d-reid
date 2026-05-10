from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import imageio
import numpy as np
from PIL import Image, ImageDraw


DEFAULT_CAMS = ["cam0", "cam1", "cam2"]
FALLBACK_POINT_COLOR = np.asarray([255, 120, 60], dtype=np.uint8)


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


def _read_mask_binary(path: Path) -> np.ndarray:
    try:
        return np.asarray(Image.open(path).convert("L"), dtype=np.uint8) > 0
    except Exception as exc:
        raise SystemExit(f"Failed to read mask image: {path}; error={exc!r}")


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
        raise SystemExit(f"Cannot write first-frame PNG for empty video: {out_path}")
    try:
        Image.fromarray(np.asarray(frames[0], dtype=np.uint8)).save(out_path)
    except Exception as exc:
        raise SystemExit(f"Failed to write first-frame PNG: {out_path}; error={exc!r}")


def _scene_root_has_expected_artifacts(scene_root: Path) -> bool:
    index_candidates = [
        scene_root / "points_by_timestamp" / "index.csv",
        scene_root / "fused" / "points_by_timestamp" / "index.csv",
    ]
    return any(path.exists() for path in index_candidates)


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
        f"Expected points_by_timestamp/index.csv for scene_id={scene_id!r}."
    )


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


def _resolve_frame_times_path(scene_dir: Path) -> Path:
    path = scene_dir / "frame_times.csv"
    if not path.exists():
        raise SystemExit(f"Missing frame_times.csv: {path}")
    return path


def _resolve_rig_path(scene_dir: Path) -> Path:
    path = scene_dir / "calib" / "rig.json"
    if not path.exists():
        raise SystemExit(f"Missing rig json: {path}")
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


def _load_points_series(points_dir: Path) -> list[dict[str, Any]]:
    rows = _load_points_index(points_dir / "index.csv")
    for row in rows:
        row["_points_rel"] = _row_first(row, ["points_rel", "points_path", "npy_rel", "rel_path", "points"])
    return rows


def _find_point_file(points_dir: Path, rel_path: str) -> Path:
    rel = Path(str(rel_path))
    candidates = [
        rel if rel.is_absolute() else points_dir / rel,
        points_dir.parent / rel,
        points_dir.parent.parent / rel,
        points_dir / rel.name,
    ]
    for path in candidates:
        if path.exists():
            return path
    raise SystemExit(f"Could not resolve point file {rel_path!r} under {points_dir}")


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


def _resolve_mask_path(scene_dir: Path, cam_id: str, mask_subdir: str, stem: str) -> Path | None:
    mask_root = scene_dir / "cams" / cam_id / str(mask_subdir)
    direct = mask_root / f"{stem}.png"
    if direct.exists():
        return direct
    nested = sorted(mask_root.glob(f"obj_*/{stem}.png"))
    if nested:
        return nested[0]
    return None


def _mask_bbox_xyxy(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def _expand_bbox_xyxy(bbox: list[int], width: int, height: int, padding_ratio: float) -> list[int] | None:
    x1, y1, x2, y2 = bbox
    box_w = max(1.0, float(x2 - x1))
    box_h = max(1.0, float(y2 - y1))
    pad_w = box_w * max(float(padding_ratio), 0.0) * 0.5
    pad_h = box_h * max(float(padding_ratio), 0.0) * 0.5
    out = [
        int(np.floor(float(x1) - pad_w)),
        int(np.floor(float(y1) - pad_h)),
        int(np.ceil(float(x2) + pad_w)),
        int(np.ceil(float(y2) + pad_h)),
    ]
    out[0] = max(0, min(out[0], width - 1))
    out[1] = max(0, min(out[1], height - 1))
    out[2] = max(0, min(out[2], width))
    out[3] = max(0, min(out[3], height))
    if out[2] <= out[0] or out[3] <= out[1]:
        return None
    return out


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


def _fit_panel(rgb: np.ndarray, panel_width: int, panel_height: int) -> np.ndarray:
    src = np.asarray(rgb, dtype=np.uint8)
    src_h, src_w = src.shape[:2]
    scale = min(float(panel_width) / max(src_w, 1), float(panel_height) / max(src_h, 1))
    dst_w = max(1, int(round(float(src_w) * scale)))
    dst_h = max(1, int(round(float(src_h) * scale)))
    resized = np.asarray(Image.fromarray(src).resize((dst_w, dst_h), resample=Image.BILINEAR), dtype=np.uint8)
    canvas = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
    x0 = (panel_width - dst_w) // 2
    y0 = (panel_height - dst_h) // 2
    canvas[y0 : y0 + dst_h, x0 : x0 + dst_w] = resized
    return canvas


def _label_panel(rgb: np.ndarray, label: str) -> np.ndarray:
    img = Image.fromarray(np.asarray(rgb, dtype=np.uint8))
    draw = ImageDraw.Draw(img)
    x0 = 8
    y0 = 8
    try:
        text_box = draw.textbbox((x0, y0), label)
        draw.rectangle(
            [text_box[0] - 4, text_box[1] - 2, text_box[2] + 4, text_box[3] + 2],
            fill=(0, 0, 0),
        )
    except Exception:
        pass
    draw.text((x0, y0), label, fill=(255, 255, 255))
    return np.asarray(img, dtype=np.uint8)


def _layout_notes(layout: str, point_color_mode: str) -> list[str]:
    notes = [
        "RGB crop keeps original image texture from the source frame.",
        "The points-overlay panel shows fused 4D point projections over the same RGB crop.",
        "The overlay panel is not a re-rendered reconstruction result.",
        "Projected fused points explain why Gaussian-style soft previews can look mosaic or blurry locally.",
    ]
    if layout == "rgb_mask_overlay":
        notes.insert(1, "Mask crop removes background but does not invent new texture.")
    if point_color_mode == "fixed_orange":
        notes.append("Projected points use a fixed orange display color for presentation clarity.")
    else:
        notes.append("Projected points sample color from the source RGB frame when available.")
    return notes


def _load_point_cloud(path: Path) -> np.ndarray:
    pts = np.asarray(np.load(str(path)), dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise SystemExit(f"Expected Nx3 point cloud in {path}, got {pts.shape}")
    return pts


def _project_points(points_world: np.ndarray, K: np.ndarray, c2w: np.ndarray, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    if points_world.size == 0:
        return np.zeros((0, 2), dtype=np.int32), np.zeros((0,), dtype=bool)
    w2c = np.linalg.inv(c2w).astype(np.float32)
    R = w2c[:3, :3]
    t = w2c[:3, 3]
    pts = np.asarray(points_world, dtype=np.float32)
    pts_cam = pts @ R.T + t[None, :]
    z = pts_cam[:, 2]
    valid = np.isfinite(z) & (z > 1e-6)
    uv = np.full((pts.shape[0], 2), -1, dtype=np.int32)
    if not np.any(valid):
        return uv, valid
    idx = np.nonzero(valid)[0]
    pts_valid = pts_cam[valid]
    z_valid = z[valid]
    u = np.round(K[0, 0] * (pts_valid[:, 0] / z_valid) + K[0, 2]).astype(np.int32)
    v = np.round(K[1, 1] * (pts_valid[:, 1] / z_valid) + K[1, 2]).astype(np.int32)
    valid_uv = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    valid[idx] &= valid_uv
    uv[idx[valid_uv], 0] = u[valid_uv]
    uv[idx[valid_uv], 1] = v[valid_uv]
    return uv, valid


def _disk_offsets(radius_px: int) -> list[tuple[int, int]]:
    radius = max(int(radius_px), 0)
    offsets: list[tuple[int, int]] = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy <= radius * radius:
                offsets.append((dx, dy))
    return offsets or [(0, 0)]


def _render_overlay_crop(
    crop_rgb: np.ndarray,
    local_uv: np.ndarray,
    point_colors: np.ndarray,
    point_radius_px: int,
    point_alpha: float,
) -> np.ndarray:
    base = np.asarray(crop_rgb, dtype=np.uint8)
    h, w = base.shape[:2]
    if local_uv.size == 0:
        return base.copy()
    offsets = _disk_offsets(point_radius_px)
    color_sum = np.zeros((h, w, 3), dtype=np.float32)
    alpha_sum = np.zeros((h, w), dtype=np.float32)
    colors = np.asarray(point_colors, dtype=np.float32)
    alpha = float(point_alpha)
    xs0 = np.asarray(local_uv[:, 0], dtype=np.int32)
    ys0 = np.asarray(local_uv[:, 1], dtype=np.int32)
    for dx, dy in offsets:
        xs = xs0 + int(dx)
        ys = ys0 + int(dy)
        valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        if not np.any(valid):
            continue
        xv = xs[valid]
        yv = ys[valid]
        cv = colors[valid]
        np.add.at(alpha_sum, (yv, xv), alpha)
        for channel in range(3):
            np.add.at(color_sum[..., channel], (yv, xv), alpha * cv[:, channel])

    out = base.astype(np.float32)
    alpha_clip = np.clip(alpha_sum, 0.0, 1.0)
    point_pixels = alpha_sum > 0
    if np.any(point_pixels):
        blended = color_sum[point_pixels] / np.maximum(alpha_sum[point_pixels, None], 1e-6)
        alpha_local = alpha_clip[point_pixels, None]
        out[point_pixels] = out[point_pixels] * (1.0 - alpha_local) + blended * alpha_local
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser(description="Render local RGB/mask/projected-point comparison previews for fused points_by_timestamp.")
    ap.add_argument("--scene_dir", required=True, type=str)
    ap.add_argument("--fused_root", required=True, type=str)
    ap.add_argument("--cams", default="cam0,cam1,cam2", type=str)
    ap.add_argument("--output_subdir", default="preview/local_target_compare_v1", type=str)
    ap.add_argument("--layout", default="rgb_mask_overlay", choices=["rgb_mask_overlay", "rgb_overlay"], type=str)
    ap.add_argument("--fps", default=16, type=int)
    ap.add_argument("--width", default=512, type=int, help="Per-panel output width.")
    ap.add_argument("--bbox_padding_ratio", default=1.0, type=float)
    ap.add_argument("--point_radius_px", default=1, type=int)
    ap.add_argument("--point_alpha", default=0.35, type=float)
    ap.add_argument("--point_color_mode", default="sampled_rgb", choices=["sampled_rgb", "fixed_orange"], type=str)
    ap.add_argument("--max_frames", default=0, type=int)
    ap.add_argument("--sample_rgb_require_mask", action="store_true")
    ap.add_argument("--mask_subdir", default="masks_gt", type=str)
    args = ap.parse_args()

    repo_root = _repo_root()
    scene_dir = Path(str(args.scene_dir)).resolve()
    if not scene_dir.exists():
        raise SystemExit(f"Missing --scene_dir: {scene_dir}")

    fused_root = Path(str(args.fused_root))
    if not fused_root.is_absolute():
        fused_root = repo_root / fused_root
    fused_root = fused_root.resolve()
    if not fused_root.exists():
        raise SystemExit(f"Missing --fused_root: {fused_root}")

    output_subdir = Path(str(args.output_subdir).replace("\\", "/"))
    if output_subdir.is_absolute():
        raise SystemExit(f"--output_subdir must be relative to --fused_root, got: {output_subdir}")

    scene_id = scene_dir.name
    run_scene_root = _resolve_run_scene_root(fused_root, scene_id=scene_id)
    points_dir = _resolve_points_dir(run_scene_root)
    frame_times_csv = _resolve_frame_times_path(scene_dir)
    rig = _load_json(_resolve_rig_path(scene_dir))
    rig_cams = dict(rig.get("cameras") or {})

    cams = [c.strip() for c in str(args.cams).split(",") if c.strip()]
    if not cams:
        raise SystemExit("--cams is empty")
    for cam_id in cams:
        if cam_id not in rig_cams:
            raise SystemExit(f'Camera "{cam_id}" missing from rig.json. Available: {sorted(rig_cams.keys())}')

    points_rows = _load_points_series(points_dir)
    if int(args.max_frames) > 0:
        points_rows = points_rows[: int(args.max_frames)]
    if not points_rows:
        raise SystemExit(f"No points_by_timestamp frames selected under: {points_dir}")

    frame_times_by_stem = _load_frame_times(frame_times_csv, cams)
    panel_heights: dict[str, int] = {cam_id: 1 for cam_id in cams}
    cam_image_sizes: dict[str, tuple[int, int]] = {}
    records: list[dict[str, Any]] = []

    for row in points_rows:
        stem = str(row["scene_stem"])
        frame_record: dict[str, Any] = {
            "stem": stem,
            "points_path": _find_point_file(points_dir, str(row["_points_rel"])),
            "per_camera": {},
        }
        for cam_id in cams:
            frame_rel = frame_times_by_stem.get(stem, {}).get(cam_id)
            if frame_rel is None:
                raise SystemExit(f"Missing synchronized frame for cam={cam_id}, scene_stem={stem}")
            frame_path = scene_dir / frame_rel
            if not frame_path.exists():
                raise SystemExit(f"Missing RGB frame for cam={cam_id}, scene_stem={stem}: {frame_path}")
            mask_path = _resolve_mask_path(scene_dir, cam_id, str(args.mask_subdir), stem)
            if mask_path is None:
                raise SystemExit(f"Missing mask for cam={cam_id}, scene_stem={stem} under {args.mask_subdir}")
            mask = _read_mask_binary(mask_path)
            bbox = _mask_bbox_xyxy(mask)
            if bbox is None:
                raise SystemExit(f"Foreground bbox is empty for cam={cam_id}, scene_stem={stem}: {mask_path}")
            bbox = _expand_bbox_xyxy(bbox, width=int(mask.shape[1]), height=int(mask.shape[0]), padding_ratio=float(args.bbox_padding_ratio))
            if bbox is None:
                raise SystemExit(f"Expanded bbox is invalid for cam={cam_id}, scene_stem={stem}: {mask_path}")
            cam_size = (int(mask.shape[1]), int(mask.shape[0]))
            if cam_id in cam_image_sizes and cam_image_sizes[cam_id] != cam_size:
                raise SystemExit(
                    f"Inconsistent image size for {cam_id}: expected {cam_image_sizes[cam_id]}, got {cam_size} from {mask_path}"
                )
            cam_image_sizes[cam_id] = cam_size
            crop_w = max(1, int(bbox[2] - bbox[0]))
            crop_h = max(1, int(bbox[3] - bbox[1]))
            panel_heights[cam_id] = max(panel_heights[cam_id], int(round(float(args.width) * float(crop_h) / float(crop_w))))
            frame_record["per_camera"][cam_id] = {
                "frame_path": frame_path,
                "mask_path": mask_path,
                "bbox_xyxy": bbox,
            }
        records.append(frame_record)

    cam_intrinsics: dict[str, np.ndarray] = {}
    cam_transforms: dict[str, np.ndarray] = {}
    for cam_id in cams:
        cam_meta = dict(rig_cams.get(cam_id) or {})
        K_native = np.asarray(cam_meta.get("K"), dtype=np.float32)
        c2w = np.asarray(cam_meta.get("T_node_from_cam"), dtype=np.float32)
        if K_native.shape != (3, 3):
            raise SystemExit(f"Invalid K for {cam_id}: {K_native.shape}")
        if c2w.shape != (4, 4):
            raise SystemExit(f"Invalid T_node_from_cam for {cam_id}: {c2w.shape}")
        src_size_raw = cam_meta.get("image_size") or []
        if isinstance(src_size_raw, list) and len(src_size_raw) == 2:
            src_size = (int(src_size_raw[0]), int(src_size_raw[1]))
        else:
            src_size = cam_image_sizes[cam_id]
        cam_intrinsics[cam_id] = _scale_intrinsics(K_native, src_size=src_size, dst_size=cam_image_sizes[cam_id])
        cam_transforms[cam_id] = c2w.astype(np.float32)

    frames_by_cam: dict[str, list[np.ndarray]] = {cam_id: [] for cam_id in cams}
    per_cam_overlay_points: dict[str, list[int]] = {cam_id: [] for cam_id in cams}
    selected_stems: list[str] = []

    for record in records:
        stem = str(record["stem"])
        selected_stems.append(stem)
        points = _load_point_cloud(Path(record["points_path"]))
        for cam_id in cams:
            frame_path = Path(record["per_camera"][cam_id]["frame_path"])
            mask_path = Path(record["per_camera"][cam_id]["mask_path"])
            x1, y1, x2, y2 = [int(v) for v in record["per_camera"][cam_id]["bbox_xyxy"]]

            rgb = _read_rgb(frame_path)
            mask = _read_mask_binary(mask_path)
            if mask.shape != rgb.shape[:2]:
                mask_img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
                mask = np.asarray(mask_img.resize((int(rgb.shape[1]), int(rgb.shape[0])), resample=Image.NEAREST), dtype=np.uint8) > 0

            crop_rgb = np.asarray(rgb[y1:y2, x1:x2], dtype=np.uint8)
            crop_mask = np.asarray(mask[y1:y2, x1:x2], dtype=bool)
            crop_masked = crop_rgb.copy()
            crop_masked[~crop_mask] = 0

            uv, valid_proj = _project_points(
                points_world=points,
                K=cam_intrinsics[cam_id],
                c2w=cam_transforms[cam_id],
                width=int(rgb.shape[1]),
                height=int(rgb.shape[0]),
            )
            proj_uv = uv[valid_proj]
            proj_colors = np.tile(FALLBACK_POINT_COLOR[None, :], (proj_uv.shape[0], 1))
            if proj_uv.size > 0:
                sample_ok = np.ones((proj_uv.shape[0],), dtype=bool)
                if str(args.point_color_mode) == "sampled_rgb" and bool(args.sample_rgb_require_mask):
                    sample_ok = mask[proj_uv[:, 1], proj_uv[:, 0]]
                if str(args.point_color_mode) == "sampled_rgb" and np.any(sample_ok):
                    proj_colors[sample_ok] = rgb[proj_uv[sample_ok, 1], proj_uv[sample_ok, 0], :]
                in_crop = (
                    (proj_uv[:, 0] >= x1)
                    & (proj_uv[:, 0] < x2)
                    & (proj_uv[:, 1] >= y1)
                    & (proj_uv[:, 1] < y2)
                )
                local_uv = np.stack([proj_uv[in_crop, 0] - x1, proj_uv[in_crop, 1] - y1], axis=1) if np.any(in_crop) else np.zeros((0, 2), dtype=np.int32)
                overlay_points = int(np.count_nonzero(in_crop))
                overlay_rgb = _render_overlay_crop(
                    crop_rgb=crop_rgb,
                    local_uv=local_uv,
                    point_colors=proj_colors[in_crop],
                    point_radius_px=int(args.point_radius_px),
                    point_alpha=float(args.point_alpha),
                )
            else:
                overlay_points = 0
                overlay_rgb = crop_rgb.copy()

            per_cam_overlay_points[cam_id].append(overlay_points)
            panel_height = max(1, int(panel_heights[cam_id]))
            panel_rgb = _label_panel(
                _fit_panel(crop_rgb, panel_width=int(args.width), panel_height=panel_height),
                "Original RGB crop",
            )
            panel_overlay = _label_panel(
                _fit_panel(overlay_rgb, panel_width=int(args.width), panel_height=panel_height),
                "RGB + fused 4D points",
            )
            if str(args.layout) == "rgb_overlay":
                frame_panels = [panel_rgb, panel_overlay]
            else:
                panel_mask = _label_panel(
                    _fit_panel(crop_masked, panel_width=int(args.width), panel_height=panel_height),
                    "Mask crop",
                )
                frame_panels = [panel_rgb, panel_mask, panel_overlay]
            frames_by_cam[cam_id].append(np.concatenate(frame_panels, axis=1))

    output_root = run_scene_root / output_subdir
    output_root.mkdir(parents=True, exist_ok=True)
    video_paths: dict[str, str] = {}
    first_frame_paths: dict[str, str] = {}
    for cam_id in cams:
        video_path = output_root / f"{cam_id}_local_compare.mp4"
        first_path = output_root / f"{cam_id}_local_compare_first.png"
        _write_video(frames_by_cam[cam_id], video_path, fps=int(args.fps))
        _write_first_frame_png(frames_by_cam[cam_id], first_path)
        video_paths[cam_id] = video_path.as_posix()
        first_frame_paths[cam_id] = first_path.as_posix()

    meta = {
        "schema_version": "local_target_preview_v1",
        "scene_id": scene_id,
        "scene_dir": scene_dir.as_posix(),
        "fused_root": fused_root.as_posix(),
        "run_scene_root": run_scene_root.as_posix(),
        "points_by_timestamp_root": points_dir.as_posix(),
        "num_frames": int(len(records)),
        "cams": cams,
        "layout": str(args.layout),
        "mask_subdir": str(args.mask_subdir),
        "bbox_padding_ratio": float(args.bbox_padding_ratio),
        "point_radius_px": int(args.point_radius_px),
        "point_alpha": float(args.point_alpha),
        "point_color_mode": str(args.point_color_mode),
        "sample_rgb_require_mask": bool(args.sample_rgb_require_mask),
        "panel_width": int(args.width),
        "panel_heights": {cam_id: int(panel_heights[cam_id]) for cam_id in cams},
        "frame_times_csv": frame_times_csv.as_posix(),
        "output_subdir": output_subdir.as_posix(),
        "video_paths": video_paths,
        "first_frame_paths": first_frame_paths,
        "debug": {
            "selected_stems": selected_stems,
            "overlay_points_per_frame": per_cam_overlay_points,
        },
        "notes": _layout_notes(str(args.layout), str(args.point_color_mode)),
    }
    meta_path = output_root / "local_target_preview_meta.json"
    _write_json(meta_path, meta)

    for cam_id in cams:
        print(f"Wrote: {video_paths[cam_id]}")
        print(f"Wrote: {first_frame_paths[cam_id]}")
    print(f"Wrote: {meta_path}")


if __name__ == "__main__":
    main()
