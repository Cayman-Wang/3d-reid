from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import imageio
import numpy as np
from PIL import Image, ImageFilter


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"Failed to read json: {path}\nError: {exc!r}")


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


def _numeric_sort(items: list[str]) -> list[str]:
    try:
        return sorted(items, key=lambda x: int(x))
    except Exception:
        return sorted(items)


def _read_sync_rows(frame_times_csv: Path, cams: list[str]) -> list[tuple[int, dict[str, str]]]:
    rows_by_ts: dict[int, dict[str, str]] = {}
    with frame_times_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts_us = int(row["ts_us"])
            cam_id = str(row["cam_id"]).strip()
            if cam_id not in cams:
                continue
            rows_by_ts.setdefault(ts_us, {})[cam_id] = str(row["filename"]).strip()
    ordered = sorted(rows_by_ts.items(), key=lambda kv: kv[0])
    return [(ts_us, by_cam) for ts_us, by_cam in ordered if all(cam in by_cam for cam in cams)]


def _load_sampled_scene_stems(dynamic_index_path: Path) -> list[str]:
    rows = list(csv.DictReader(dynamic_index_path.open("r", encoding="utf-8", newline="")))
    if not rows:
        raise SystemExit(f"No dynamic rows found in: {dynamic_index_path}")
    rows.sort(key=lambda row: int(float(row.get("logical_t_idx", 0))))
    stems: list[str] = []
    seen: set[str] = set()
    for row in rows:
        scene_stem = str(row["scene_stem"])
        if scene_stem in seen:
            continue
        seen.add(scene_stem)
        stems.append(scene_stem)
    return stems


def _frame_times_by_stem(frame_times_csv: Path, cams: list[str]) -> dict[str, dict[str, str]]:
    by_stem: dict[str, dict[str, str]] = {}
    with frame_times_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cam_id = str(row["cam_id"]).strip()
            if cam_id not in cams:
                continue
            stem = Path(str(row["filename"]).strip()).stem
            by_stem.setdefault(stem, {})[cam_id] = str(row["filename"]).strip()
    return by_stem


def _get_rig(scene_dir: Path) -> dict[str, Any]:
    rig_path = scene_dir / "calib" / "rig.json"
    if not rig_path.exists():
        raise SystemExit(f"Missing rig json: {rig_path}")
    return _load_json(rig_path)


def _resize_image(rgb: np.ndarray, size_hw: tuple[int, int]) -> np.ndarray:
    target_h, target_w = size_hw
    img = Image.fromarray(np.asarray(rgb, dtype=np.uint8))
    if img.size == (target_w, target_h):
        return rgb
    return np.asarray(img.resize((target_w, target_h), resample=Image.BILINEAR), dtype=np.uint8)


def _look_at_c2w(camera_pos: np.ndarray, target: np.ndarray, up_hint: np.ndarray) -> np.ndarray:
    forward = np.asarray(target - camera_pos, dtype=np.float32)
    forward_norm = np.linalg.norm(forward)
    if forward_norm < 1e-6:
        forward = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    else:
        forward = forward / forward_norm
    up = np.asarray(up_hint, dtype=np.float32)
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-6:
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        right = np.cross(forward, up)
    right = right / max(np.linalg.norm(right), 1e-6)
    true_up = np.cross(right, forward)
    true_up = true_up / max(np.linalg.norm(true_up), 1e-6)
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 0] = right
    c2w[:3, 1] = true_up
    c2w[:3, 2] = forward
    c2w[:3, 3] = camera_pos
    return c2w


def _project_points(points_world: np.ndarray, colors: np.ndarray, K: np.ndarray, c2w: np.ndarray, width: int, height: int) -> np.ndarray:
    if points_world.size == 0:
        return np.zeros((height, width, 3), dtype=np.uint8)
    w2c = np.linalg.inv(c2w).astype(np.float32)
    R = w2c[:3, :3]
    t = w2c[:3, 3]
    pts = np.asarray(points_world, dtype=np.float32)
    pts_cam = pts @ R.T + t[None, :]
    z = pts_cam[:, 2]
    valid = np.isfinite(z) & (z > 1e-6)
    if not np.any(valid):
        return np.zeros((height, width, 3), dtype=np.uint8)
    pts_cam = pts_cam[valid]
    z = z[valid]
    cols = np.asarray(colors, dtype=np.uint8)[valid]

    u = (K[0, 0] * (pts_cam[:, 0] / z) + K[0, 2]).round().astype(np.int32)
    v = (K[1, 1] * (pts_cam[:, 1] / z) + K[1, 2]).round().astype(np.int32)
    valid = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    if not np.any(valid):
        return np.zeros((height, width, 3), dtype=np.uint8)
    u = u[valid]
    v = v[valid]
    z = z[valid]
    cols = cols[valid]

    order = np.argsort(z)
    u = u[order]
    v = v[order]
    cols = cols[order]
    pix = v * width + u
    _, first = np.unique(pix, return_index=True)
    sel = first
    image = np.zeros((height * width, 3), dtype=np.uint8)
    image[pix[sel]] = cols[sel]
    return image.reshape(height, width, 3)


def _render_overlay_frame(base_rgb: np.ndarray, sparse_points_rgb: np.ndarray, point_radius: int = 2, alpha: float = 0.7) -> np.ndarray:
    base = np.asarray(base_rgb, dtype=np.uint8)
    sparse = np.asarray(sparse_points_rgb, dtype=np.uint8)
    radius = max(int(point_radius), 0)
    if sparse.shape != base.shape:
        raise SystemExit(f"Overlay sparse/base shape mismatch: sparse={sparse.shape}, base={base.shape}")
    if radius > 0:
        filter_size = int(radius * 2 + 1)
        sparse = np.asarray(Image.fromarray(sparse).filter(ImageFilter.MaxFilter(size=filter_size)), dtype=np.uint8)
        mask_u8 = ((sparse_points_rgb.sum(axis=-1) > 0).astype(np.uint8) * 255)
        mask_u8 = np.asarray(Image.fromarray(mask_u8).filter(ImageFilter.MaxFilter(size=filter_size)), dtype=np.uint8)
    else:
        mask_u8 = (sparse.sum(axis=-1) > 0).astype(np.uint8) * 255
    mask = (mask_u8 > 0).astype(np.float32)[..., None] * float(alpha)
    blended = base.astype(np.float32) * (1.0 - mask) + sparse.astype(np.float32) * mask
    return np.clip(np.round(blended), 0, 255).astype(np.uint8)


def _combine_points(background: np.ndarray, dynamic: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    bg = np.asarray(background, dtype=np.float32)
    if bg.size:
        bg_col = np.full((bg.shape[0], 3), 175, dtype=np.uint8)
    else:
        bg_col = np.zeros((0, 3), dtype=np.uint8)
    if dynamic is None or dynamic.size == 0:
        return bg, bg_col
    dy = np.asarray(dynamic, dtype=np.float32)
    dy_col = np.tile(np.asarray([255, 120, 60], dtype=np.uint8), (dy.shape[0], 1))
    points = np.concatenate([bg, dy], axis=0) if bg.size else dy
    colors = np.concatenate([bg_col, dy_col], axis=0) if bg.size else dy_col
    return points, colors


def _sample_points(points: np.ndarray, colors: np.ndarray, max_points: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    if points.shape[0] <= max_points:
        return points, colors
    rng = np.random.default_rng(seed)
    idx = rng.choice(points.shape[0], size=max_points, replace=False)
    idx.sort()
    return points[idx], colors[idx]


def _build_glb(scene_root: Path, background: np.ndarray, dynamics: dict[str, np.ndarray]) -> Path:
    try:
        import trimesh  # type: ignore
    except Exception as exc:
        glb_path = scene_root / "fused_scene.glb"
        if glb_path.exists():
            return glb_path
        raise SystemExit(f"Missing trimesh dependency: {exc!r}")

    all_points = [np.asarray(background, dtype=np.float32)]
    all_colors = [np.full((background.shape[0], 3), 175, dtype=np.uint8)] if background.size else []
    for dyn in dynamics.values():
        dyn = np.asarray(dyn, dtype=np.float32)
        if dyn.size:
            all_points.append(dyn)
            all_colors.append(np.tile(np.asarray([255, 120, 60], dtype=np.uint8), (dyn.shape[0], 1)))
    if all_points:
        points = np.concatenate(all_points, axis=0)
        colors = np.concatenate(all_colors, axis=0) if all_colors else np.zeros((0, 3), dtype=np.uint8)
    else:
        points = np.zeros((0, 3), dtype=np.float32)
        colors = np.zeros((0, 3), dtype=np.uint8)
    if points.shape[0] > 500000:
        points, colors = _sample_points(points, colors, 500000, seed=7)
    rgba = np.concatenate([colors, np.full((colors.shape[0], 1), 255, dtype=np.uint8)], axis=1) if colors.size else np.zeros((0, 4), dtype=np.uint8)
    scene = trimesh.Scene()
    if points.size:
        scene.add_geometry(trimesh.PointCloud(vertices=points, colors=rgba))
    glb_path = scene_root / "fused_scene.glb"
    scene.export(glb_path)
    return glb_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Render fused NeoVerse world points into original-compare and orbit previews.")
    ap.add_argument("--scene_dir", required=True, type=str)
    ap.add_argument("--fused_root", default="mvp-demo/output/neoverse_fused", type=str)
    ap.add_argument("--cams", default="cam0,cam1,cam2", type=str)
    ap.add_argument("--fps", default=16, type=int)
    ap.add_argument("--orbit_frames", default=81, type=int)
    ap.add_argument("--orbit_radius_scale", default=1.8, type=float)
    ap.add_argument("--max_render_points", default=300000, type=int)
    args = ap.parse_args()

    repo = _repo_root()
    scene_dir = Path(str(args.scene_dir)).resolve()
    if not scene_dir.exists():
        raise SystemExit(f"Missing --scene_dir: {scene_dir}")

    fused_root = Path(str(args.fused_root))
    if not fused_root.is_absolute():
        fused_root = repo / fused_root
    scene_id = scene_dir.name
    scene_fused_root = fused_root / scene_id / "fused"
    if not scene_fused_root.exists():
        raise SystemExit(f"Missing fused root: {scene_fused_root}")

    background_path = scene_fused_root / "background_world.npy"
    meta_path = scene_fused_root / "fusion_meta.json"
    dynamic_index_path = scene_fused_root / "dynamic_index.csv"
    if not background_path.exists() or not meta_path.exists() or not dynamic_index_path.exists():
        raise SystemExit(f"Missing fused artifacts under: {scene_fused_root}")

    background = np.asarray(np.load(str(background_path)), dtype=np.float32)
    meta = _load_json(meta_path)
    dynamic_rows = list(csv.DictReader(dynamic_index_path.open("r", encoding="utf-8", newline="")))
    dynamic_by_stem: dict[str, np.ndarray] = {}
    for row in dynamic_rows:
        scene_stem = str(row["scene_stem"])
        points_path = scene_fused_root / str(row["points_path"])
        if points_path.exists():
            dynamic_by_stem[scene_stem] = np.asarray(np.load(str(points_path)), dtype=np.float32)

    rig = _get_rig(scene_dir)
    rig_cameras = rig.get("cameras", {})
    cams = [c.strip() for c in str(args.cams).split(",") if c.strip()]
    if cams != ["cam0", "cam1", "cam2"]:
        raise SystemExit(f"This first version only supports cams=['cam0','cam1','cam2']. Got: {cams}")

    frame_times_csv = scene_dir / "frame_times.csv"
    if not frame_times_csv.exists():
        raise SystemExit(f"Missing frame_times.csv: {frame_times_csv}")
    sampled_scene_stems = _load_sampled_scene_stems(dynamic_index_path)
    frame_times_by_stem = _frame_times_by_stem(frame_times_csv, cams)
    sampled_sync_rows: list[tuple[str, dict[str, str]]] = []
    for scene_stem in sampled_scene_stems:
        by_cam = frame_times_by_stem.get(scene_stem)
        if by_cam is None or not all(cam in by_cam for cam in cams):
            raise SystemExit(f"Missing complete three-camera sync row for sampled scene_stem={scene_stem!r}")
        sampled_sync_rows.append((scene_stem, by_cam))

    preview_root = scene_fused_root.parent / "preview"
    compare_root = preview_root / "original_compare"
    overlay_root = preview_root / "original_overlay"
    orbit_root = preview_root / "orbit"
    compare_root.mkdir(parents=True, exist_ok=True)
    overlay_root.mkdir(parents=True, exist_ok=True)
    orbit_root.mkdir(parents=True, exist_ok=True)

    # Original-view compare videos.
    for cam_id in cams:
        cam_frames: list[np.ndarray] = []
        overlay_frames: list[np.ndarray] = []
        for scene_stem, by_cam in sampled_sync_rows:
            frame_rel = by_cam[cam_id]
            frame_path = scene_dir / frame_rel
            orig = _read_rgb(frame_path)
            cam_meta = rig_cameras[cam_id]
            K = np.asarray(cam_meta["K"], dtype=np.float32)
            c2w = np.asarray(cam_meta["T_node_from_cam"], dtype=np.float32)

            dyn = dynamic_by_stem.get(scene_stem)
            pts, cols = _combine_points(background, dyn)
            pts, cols = _sample_points(pts, cols, int(args.max_render_points), seed=3)
            fused_rgb = _project_points(pts, cols, K, c2w, orig.shape[1], orig.shape[0])
            orig = _resize_image(orig, (orig.shape[0], orig.shape[1]))
            compare = np.concatenate([orig, fused_rgb], axis=1)
            cam_frames.append(compare)
            overlay_frames.append(_render_overlay_frame(orig, fused_rgb, point_radius=2, alpha=0.7))

        _write_video(cam_frames, compare_root / f"{cam_id}_compare.mp4", fps=int(args.fps))
        _write_video(overlay_frames, overlay_root / f"{cam_id}_overlay.mp4", fps=int(args.fps))

    # Orbit preview around the fused world point cloud.
    # Keep the same sampled temporal order as compare/overlay outputs.
    if sampled_scene_stems:
        active_key_list = list(sampled_scene_stems)
    else:
        active_key_list = [""]

    if background.size:
        center = background.mean(axis=0)
        extent = np.max(background.max(axis=0) - background.min(axis=0))
    else:
        center = np.zeros(3, dtype=np.float32)
        extent = 1.0
    radius = max(float(extent) * float(args.orbit_radius_scale), 1.0)
    orbit_frames: list[np.ndarray] = []
    # Reuse the first camera intrinsics as a stable framing reference.
    ref_cam = rig_cameras[cams[0]]
    ref_K = np.asarray(ref_cam["K"], dtype=np.float32)
    ref_w = int(ref_cam.get("image_size", [280, 168])[0])
    ref_h = int(ref_cam.get("image_size", [280, 168])[1])
    if ref_w <= 0 or ref_h <= 0:
        ref_w, ref_h = 280, 168

    for frame_idx in range(int(args.orbit_frames)):
        angle = 2.0 * np.pi * float(frame_idx) / max(int(args.orbit_frames), 1)
        cam_pos = center + np.array([np.cos(angle) * radius, np.sin(angle) * radius, radius * 0.35], dtype=np.float32)
        c2w = _look_at_c2w(cam_pos, center, np.array([0.0, 0.0, 1.0], dtype=np.float32))
        dyn = dynamic_by_stem.get(active_key_list[frame_idx % len(active_key_list)]) if active_key_list else None
        pts, cols = _combine_points(background, dyn)
        pts, cols = _sample_points(pts, cols, int(args.max_render_points), seed=11 + frame_idx)
        frame = _project_points(pts, cols, ref_K, c2w, ref_w, ref_h)
        orbit_frames.append(frame)

    _write_video(orbit_frames, orbit_root / "orbit_left_rgb.mp4", fps=int(args.fps))

    glb_path = _build_glb(scene_fused_root, background, dynamic_by_stem)

    preview_meta = {
        "scene_id": scene_id,
        "scene_dir": scene_dir.as_posix(),
        "fused_root": scene_fused_root.as_posix(),
        "background_source_branch": str(meta.get("background_source_branch", "unknown")),
        "dynamic_source_branch": str(meta.get("dynamic_source_branch", "unknown")),
        "background_points": int(background.shape[0]),
        "num_dynamic_frames": int(len(dynamic_by_stem)),
        "num_preview_frames": int(len(sampled_sync_rows)),
        "sampled_scene_stems": sampled_scene_stems,
        "outputs": {
            "original_compare": compare_root.as_posix(),
            "original_overlay": overlay_root.as_posix(),
            "orbit_rgb": (orbit_root / "orbit_left_rgb.mp4").as_posix(),
            "fused_scene_glb": glb_path.as_posix(),
        },
        "fusion_meta": meta,
    }
    (preview_root / "preview_meta.json").write_text(json.dumps(preview_meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote preview outputs to: {preview_root}")
    print(f"Wrote: {glb_path}")
    print(f"Wrote: {orbit_root / 'orbit_left_rgb.mp4'}")


if __name__ == "__main__":
    main()
