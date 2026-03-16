from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:  # pragma: no cover
        raise SystemExit(f"Failed to read json: {path}\nError: {e!r}")


def _read_image(path: Path, flags: int) -> np.ndarray:
    try:
        import cv2  # type: ignore
    except Exception as e:  # pragma: no cover
        raise SystemExit(f"Missing dependency: cv2 (opencv-python). Error: {e!r}")

    img = cv2.imread(str(path), flags)
    if img is not None:
        return img

    # cv2.imread can fail on Windows unicode paths; fall back to imdecode.
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(data, flags)
    except Exception:
        img = None
    if img is None:
        raise SystemExit(f"Failed to read image: {path}")
    return img


def _read_color(path: Path) -> np.ndarray:
    return _read_image(path, flags=1)


def _read_gray(path: Path) -> np.ndarray:
    return _read_image(path, flags=0)


def _to_2d_depth(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        if arr.shape[-1] == 1:
            return arr[..., 0]
        if arr.shape[0] == 1:
            return arr[0]
    raise SystemExit(f"Unsupported depth shape: {arr.shape}")


def _read_depth(path: Path) -> np.ndarray:
    try:
        depth = np.load(str(path))
    except Exception as e:  # pragma: no cover
        raise SystemExit(f"Failed to load depth npy: {path}\nError: {e!r}")
    return np.asarray(_to_2d_depth(depth), dtype=np.float32)


def _numeric_sort(stems: set[str]) -> list[str]:
    try:
        return sorted(stems, key=lambda x: int(x))
    except Exception:
        return sorted(stems)


def _pick_existing_subdir(scene_dir: Path, cam_id: str, candidates: list[str], suffix: str) -> str:
    for name in candidates:
        candidate_dir = scene_dir / "cams" / cam_id / name
        if not candidate_dir.exists():
            continue
        if any(candidate_dir.glob(f"*{suffix}")):
            return name
    raise SystemExit(
        f"Could not find a valid subdir under {scene_dir / 'cams' / cam_id} "
        f"for candidates={candidates!r} suffix={suffix!r}"
    )


def _find_common_frame_stems(scene_dir: Path, cams: list[str]) -> list[str]:
    per_cam: list[set[str]] = []
    for cam_id in cams:
        frame_dir = scene_dir / "cams" / cam_id / "frames"
        if not frame_dir.exists():
            raise SystemExit(f"Missing frames dir: {frame_dir}")
        stems = {p.stem for p in frame_dir.glob("*.jpg")}
        if not stems:
            stems = {p.stem for p in frame_dir.glob("*.png")}
        if not stems:
            raise SystemExit(f"No frames found in: {frame_dir}")
        per_cam.append(stems)
    return _numeric_sort(set.intersection(*per_cam))


def _pick_key_stems(stems: list[str], explicit: str | None) -> list[str]:
    if explicit:
        requested = [s.strip() for s in explicit.split(",") if s.strip()]
        missing = [s for s in requested if s not in stems]
        if missing:
            raise SystemExit(f"Requested key stems not found: {missing!r}")
        return requested
    if len(stems) == 1:
        return [stems[0]]
    if len(stems) == 2:
        return [stems[0], stems[1]]
    mid = stems[len(stems) // 2]
    return [stems[0], mid, stems[-1]]


def _label_tile(img: np.ndarray, text: str) -> np.ndarray:
    import cv2  # type: ignore

    out = img.copy()
    org = (8, 24)
    cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _make_mask_panel(rgb_bgr: np.ndarray, mask_u8: np.ndarray) -> np.ndarray:
    mask = mask_u8 > 0
    rgb = np.asarray(rgb_bgr, dtype=np.float32)
    red_tint = np.zeros_like(rgb)
    red_tint[..., 2] = 180.0
    out = rgb * 0.40 + red_tint * 0.60
    if np.any(mask):
        warm = np.zeros_like(rgb)
        warm[..., 1] = 90.0
        warm[..., 2] = 255.0
        out[mask] = rgb[mask] * 0.55 + warm[mask] * 0.45
    return np.clip(out, 0, 255).astype(np.uint8)


def _make_depth_panel(depth_m: np.ndarray) -> np.ndarray:
    import cv2  # type: ignore

    depth = np.asarray(depth_m, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0)
    if not np.any(valid):
        return np.zeros((*depth.shape, 3), dtype=np.uint8)

    vals = depth[valid]
    lo = float(np.percentile(vals, 2))
    hi = float(np.percentile(vals, 98))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(vals.min())
        hi = float(vals.max())
    denom = max(hi - lo, 1e-6)
    scaled = np.clip((depth - lo) / denom, 0.0, 1.0)
    scaled_u8 = np.round(scaled * 255.0).astype(np.uint8)
    colored = cv2.applyColorMap(scaled_u8, cv2.COLORMAP_INFERNO)
    colored[~valid] = 0
    return colored


def _resize_tile(img: np.ndarray, width: int, height: int) -> np.ndarray:
    import cv2  # type: ignore

    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)


def _build_overview(
    scene_dir: Path,
    stem: str,
    cams: list[str],
    mask_subdir: str,
    depth_subdir: str,
    tile_width: int,
    tile_height: int,
) -> np.ndarray:
    rows: list[np.ndarray] = []
    row_specs = [
        ("RGB", "frames", ".jpg"),
        ("Mask", mask_subdir, ".png"),
        ("Depth", depth_subdir, ".npy"),
    ]

    for label_suffix, subdir, suffix in row_specs:
        tiles: list[np.ndarray] = []
        for cam_id in cams:
            path = scene_dir / "cams" / cam_id / subdir / f"{stem}{suffix}"
            if suffix == ".jpg":
                if not path.exists():
                    alt = scene_dir / "cams" / cam_id / subdir / f"{stem}.png"
                    path = alt if alt.exists() else path
                tile = _read_color(path)
            elif suffix == ".png":
                rgb = _read_color(scene_dir / "cams" / cam_id / "frames" / f"{stem}.jpg")
                if rgb is None:
                    rgb = _read_color(scene_dir / "cams" / cam_id / "frames" / f"{stem}.png")
                mask = _read_gray(path)
                tile = _make_mask_panel(rgb, mask)
            else:
                depth = _read_depth(path)
                tile = _make_depth_panel(depth)

            tile = _resize_tile(tile, tile_width, tile_height)
            tile = _label_tile(tile, f"{cam_id} {label_suffix}")
            tiles.append(tile)
        rows.append(np.hstack(tiles))

    return np.vstack(rows)


def _write_triview_video(scene_dir: Path, stems: list[str], cams: list[str], out_path: Path, fps: float) -> int:
    import cv2  # type: ignore

    first_frames = [_read_color(scene_dir / "cams" / cam_id / "frames" / f"{stems[0]}.jpg") for cam_id in cams]
    height, width = first_frames[0].shape[:2]
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width * len(cams), height))
    if not writer.isOpened():
        raise SystemExit(f"Failed to open video writer: {out_path}")

    try:
        for stem in stems:
            tiles: list[np.ndarray] = []
            for cam_id in cams:
                frame_path = scene_dir / "cams" / cam_id / "frames" / f"{stem}.jpg"
                if not frame_path.exists():
                    frame_path = scene_dir / "cams" / cam_id / "frames" / f"{stem}.png"
                tile = _read_color(frame_path)
                tile = _label_tile(tile, f"{cam_id} RGB")
                tiles.append(tile)
            frame = np.hstack(tiles)
            frame = _label_tile(frame, f"{scene_dir.name}  ts={stem}")
            writer.write(frame)
    finally:
        writer.release()
    return len(stems)


def main() -> None:
    ap = argparse.ArgumentParser(description="Export presentation-ready triview video and keyframe overviews.")
    ap.add_argument("--scene_dir", required=True, type=str)
    ap.add_argument("--out_dir", default="presentation_assets", type=str)
    ap.add_argument("--cams", default="cam0,cam1,cam2", type=str)
    ap.add_argument("--mask_subdir", default="", type=str)
    ap.add_argument("--depth_subdir", default="", type=str)
    ap.add_argument("--key_stems", default="", type=str)
    ap.add_argument("--canvas_width", default=1024, type=int)
    ap.add_argument("--canvas_height", default=768, type=int)
    args = ap.parse_args()

    scene_dir = Path(args.scene_dir).resolve()
    if not scene_dir.exists():
        raise SystemExit(f"Missing scene dir: {scene_dir}")

    cams = [s.strip() for s in str(args.cams).split(",") if s.strip()]
    if not cams:
        raise SystemExit("No cameras specified.")

    capture_meta_path = scene_dir / "capture_meta.json"
    capture_meta = _load_json(capture_meta_path) if capture_meta_path.exists() else {}
    fps = float(capture_meta.get("render", {}).get("fps", 10.0))

    mask_subdir = str(args.mask_subdir).strip() or _pick_existing_subdir(scene_dir, cams[0], ["masks", "masks_gt"], ".png")
    depth_subdir = str(args.depth_subdir).strip() or _pick_existing_subdir(scene_dir, cams[0], ["depth", "depth_gt"], ".npy")

    stems = _find_common_frame_stems(scene_dir, cams)
    key_stems = _pick_key_stems(stems, str(args.key_stems).strip() or None)

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = scene_dir / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    tile_width = int(args.canvas_width) // len(cams)
    tile_height = int(args.canvas_height) // 3

    for stem in key_stems:
        overview = _build_overview(
            scene_dir=scene_dir,
            stem=stem,
            cams=cams,
            mask_subdir=mask_subdir,
            depth_subdir=depth_subdir,
            tile_width=tile_width,
            tile_height=tile_height,
        )
        overview_path = out_dir / f"overview_{stem}.png"
        try:
            import cv2  # type: ignore
        except Exception as e:  # pragma: no cover
            raise SystemExit(f"Missing dependency: cv2 (opencv-python). Error: {e!r}")
        ok = cv2.imwrite(str(overview_path), overview)
        if not ok:
            encoded = cv2.imencode(".png", overview)[1]
            encoded.tofile(str(overview_path))

    video_path = out_dir / "triview_video.mp4"
    frame_count = _write_triview_video(scene_dir=scene_dir, stems=stems, cams=cams, out_path=video_path, fps=fps)

    manifest = {
        "scene_dir": str(scene_dir),
        "key_stems": key_stems,
        "mask_subdir": mask_subdir,
        "depth_subdir": depth_subdir,
        "generated_files": [f"overview_{stem}.png" for stem in key_stems] + ["triview_video.mp4"],
        "num_video_frames": frame_count,
        "fps": fps,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote presentation assets to: {out_dir}")
    print(f"[summary] keyframes={len(key_stems)} video_frames={frame_count} fps={fps:g}")


if __name__ == "__main__":
    main()
