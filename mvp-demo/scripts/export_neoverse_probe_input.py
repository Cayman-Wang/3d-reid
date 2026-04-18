from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


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

    path_str = str(path)
    if not path_str.isascii():
        try:
            data = np.fromfile(path_str, dtype=np.uint8)
            img = cv2.imdecode(data, flags)
        except Exception:
            img = None
        if img is not None:
            return img

    img = cv2.imread(path_str, flags)
    if img is not None:
        return img
    try:
        data = np.fromfile(path_str, dtype=np.uint8)
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


def _numeric_sort(stems: set[str]) -> list[str]:
    try:
        return sorted(stems, key=lambda x: int(x))
    except Exception:
        return sorted(stems)


def _find_frame_stems(frame_dir: Path) -> tuple[list[str], str]:
    jpg_stems = {p.stem for p in frame_dir.glob("*.jpg")}
    png_stems = {p.stem for p in frame_dir.glob("*.png")}
    if jpg_stems and png_stems:
        stems = jpg_stems | png_stems
        return _numeric_sort(stems), "mixed"
    if jpg_stems:
        return _numeric_sort(jpg_stems), "jpg"
    if png_stems:
        return _numeric_sort(png_stems), "png"
    raise SystemExit(f"No frames found in: {frame_dir}")


def _pick_mask_subdir(scene_dir: Path, cam_id: str, explicit: str) -> str:
    if explicit.strip():
        return explicit.strip()
    cam_dir = scene_dir / "cams" / cam_id
    for name in ["masks", "masks_gt"]:
        candidate = cam_dir / name
        if candidate.exists() and any(candidate.glob("*.png")):
            return name
    raise SystemExit(f"Could not find masks or masks_gt under: {cam_dir}")


def _frame_path(frame_dir: Path, stem: str, ext_mode: str) -> Path:
    if ext_mode == "jpg":
        return frame_dir / f"{stem}.jpg"
    if ext_mode == "png":
        return frame_dir / f"{stem}.png"
    jpg = frame_dir / f"{stem}.jpg"
    return jpg if jpg.exists() else (frame_dir / f"{stem}.png")


def _bbox_from_mask(mask_u8: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask_u8 > 0)
    if ys.size == 0 or xs.size == 0:
        return None
    x0 = int(xs.min())
    x1 = int(xs.max())
    y0 = int(ys.min())
    y1 = int(ys.max())
    return x0, y0, x1, y1


def _compute_global_crop(
    frame_width: int,
    frame_height: int,
    boxes: list[tuple[int, int, int, int]],
    pad_ratio: float,
) -> tuple[int, int, int, int]:
    if not boxes:
        return 0, 0, frame_width, frame_height

    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[2] for b in boxes)
    y1 = max(b[3] for b in boxes)

    bw = max(1, x1 - x0 + 1)
    bh = max(1, y1 - y0 + 1)
    pad_x = int(round(bw * pad_ratio))
    pad_y = int(round(bh * pad_ratio))

    cx = (x0 + x1) // 2
    cy = (y0 + y1) // 2
    crop_w = min(frame_width, bw + 2 * pad_x)
    crop_h = min(frame_height, bh + 2 * pad_y)

    left = cx - crop_w // 2
    top = cy - crop_h // 2
    left = max(0, min(left, frame_width - crop_w))
    top = max(0, min(top, frame_height - crop_h))
    return left, top, left + crop_w, top + crop_h


def _write_video(frames: list[np.ndarray], out_path: Path, fps: float) -> None:
    try:
        import cv2  # type: ignore
    except Exception as e:  # pragma: no cover
        raise SystemExit(f"Missing dependency: cv2 (opencv-python). Error: {e!r}")

    if not frames:
        raise SystemExit("No frames to write.")
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise SystemExit(f"Failed to open video writer: {out_path}")
    try:
        for frame in frames:
            if frame.shape[0] != h or frame.shape[1] != w:
                frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
            writer.write(frame)
    finally:
        writer.release()


def main() -> None:
    ap = argparse.ArgumentParser(description="Export NeoVerse probe input videos from a scene/camera.")
    ap.add_argument("--scene_dir", required=True, type=str)
    ap.add_argument("--cam_id", default="cam0", type=str)
    ap.add_argument("--mask_subdir", default="", type=str, help='Default: auto pick "masks" then "masks_gt".')
    ap.add_argument("--pad_ratio", default=0.20, type=float, help="Padding ratio around object bbox. Default: 0.20")
    ap.add_argument(
        "--out_root",
        default="mvp-demo/output/neoverse_probe",
        type=str,
        help="Output root. Scene/cam/input will be appended automatically.",
    )
    args = ap.parse_args()

    scene_dir = Path(args.scene_dir).resolve()
    if not scene_dir.exists():
        raise SystemExit(f"Missing scene dir: {scene_dir}")

    cam_id = str(args.cam_id).strip() or "cam0"
    cam_dir = scene_dir / "cams" / cam_id
    frame_dir = cam_dir / "frames"
    if not frame_dir.exists():
        raise SystemExit(f"Missing frames dir: {frame_dir}")

    stems, ext_mode = _find_frame_stems(frame_dir)
    if not stems:
        raise SystemExit(f"No frame stems found in: {frame_dir}")

    mask_subdir = _pick_mask_subdir(scene_dir, cam_id, str(args.mask_subdir))
    mask_dir = cam_dir / mask_subdir
    if not mask_dir.exists():
        raise SystemExit(f"Missing mask dir: {mask_dir}")

    capture_meta = _load_json(scene_dir / "capture_meta.json") if (scene_dir / "capture_meta.json").exists() else {}
    fps = float(capture_meta.get("render", {}).get("fps", 10.0))

    full_frames: list[np.ndarray] = []
    boxes: list[tuple[int, int, int, int]] = []
    missing_masks = 0
    empty_masks = 0

    for stem in stems:
        frame = _read_color(_frame_path(frame_dir, stem, ext_mode))
        full_frames.append(frame)

        mask_path = mask_dir / f"{stem}.png"
        if not mask_path.exists():
            missing_masks += 1
            continue
        mask = _read_gray(mask_path)
        box = _bbox_from_mask(mask)
        if box is None:
            empty_masks += 1
            continue
        boxes.append(box)

    if not full_frames:
        raise SystemExit("No readable RGB frames.")

    h, w = full_frames[0].shape[:2]
    crop_l, crop_t, crop_r, crop_b = _compute_global_crop(
        frame_width=w,
        frame_height=h,
        boxes=boxes,
        pad_ratio=max(0.0, float(args.pad_ratio)),
    )
    crop_frames = [f[crop_t:crop_b, crop_l:crop_r].copy() for f in full_frames]

    repo_root = _repo_root()
    scene_id = scene_dir.name
    out_root = Path(str(args.out_root))
    if not out_root.is_absolute():
        out_root = repo_root / out_root
    input_dir = out_root / scene_id / cam_id / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    full_path = input_dir / "full_frame.mp4"
    crop_path = input_dir / "object_crop.mp4"
    _write_video(full_frames, full_path, fps=fps)
    _write_video(crop_frames, crop_path, fps=fps)

    prep_meta = {
        "scene_id": scene_id,
        "scene_dir": str(scene_dir),
        "cam_id": cam_id,
        "mask_subdir": mask_subdir,
        "fps": fps,
        "num_frames": len(stems),
        "frame_extension_mode": ext_mode,
        "crop": {
            "left": crop_l,
            "top": crop_t,
            "right": crop_r,
            "bottom": crop_b,
            "width": crop_r - crop_l,
            "height": crop_b - crop_t,
            "pad_ratio": float(args.pad_ratio),
        },
        "mask_stats": {
            "boxes_used": len(boxes),
            "missing_masks": missing_masks,
            "empty_masks": empty_masks,
        },
        "outputs": {
            "full_frame": str(full_path),
            "object_crop": str(crop_path),
        },
    }
    (input_dir / "input_prep_meta.json").write_text(
        json.dumps(prep_meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote NeoVerse input videos to: {input_dir}")
    print(
        "[summary] "
        f"scene={scene_id} cam={cam_id} frames={len(stems)} "
        f"mask_subdir={mask_subdir} crop=({crop_l},{crop_t})-({crop_r},{crop_b})"
    )


if __name__ == "__main__":
    main()
