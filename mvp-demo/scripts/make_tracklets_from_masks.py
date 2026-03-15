from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _load_mask(path: Path) -> np.ndarray:
    """
    Load a binary mask as bool(H,W). Supports typical 0/255 PNG outputs.
    """
    try:
        import cv2  # type: ignore

        m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise FileNotFoundError(path)
        return (m > 0)
    except Exception:
        # Fallback for environments without opencv.
        from PIL import Image  # type: ignore

        m = Image.open(path).convert("L")
        return (np.array(m) > 0)


def _mask_bbox_xyxy(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max()) + 1
    y2 = int(ys.max()) + 1
    return [x1, y1, x2, y2]


def _try_int_stem(p: Path) -> tuple[int, str]:
    try:
        return (int(p.stem), p.stem)
    except Exception:
        return (10**18, p.stem)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build tracklets.json from SAM2-style masks/obj_*/<stem>.png")
    ap.add_argument("--scene_dir", required=True, type=str)
    ap.add_argument("--images_dir", default="images", type=str)
    ap.add_argument("--masks_dir", default="masks", type=str)
    ap.add_argument("--out", default="tracklets.json", type=str)
    ap.add_argument("--min_frames", default=2, type=int, help="Drop tracks shorter than this")
    args = ap.parse_args()

    scene_dir = Path(args.scene_dir).resolve()
    images_dir = scene_dir / args.images_dir
    masks_dir = scene_dir / args.masks_dir
    out_path = scene_dir / args.out

    if not images_dir.exists():
        raise SystemExit(f"images dir not found: {images_dir}")
    if not masks_dir.exists():
        raise SystemExit(f"masks dir not found: {masks_dir}")

    obj_dirs = sorted([p for p in masks_dir.glob("obj_*") if p.is_dir()])
    if not obj_dirs:
        raise SystemExit(f"No obj_* dirs found under: {masks_dir}")

    tracklets: list[dict] = []
    for obj_dir in obj_dirs:
        mask_paths = sorted(obj_dir.glob("*.png"), key=_try_int_stem)
        frame_names: list[str] = []
        rel_masks: list[str] = []
        bboxes_xyxy: list[list[int]] = []

        for mp in mask_paths:
            stem = mp.stem
            img = images_dir / f"{stem}.jpg"
            if not img.exists():
                img = images_dir / f"{stem}.png"
            if not img.exists():
                continue

            mask = _load_mask(mp)
            bbox = _mask_bbox_xyxy(mask)
            if bbox is None:
                continue

            frame_names.append(img.name)
            rel_masks.append(str(mp.relative_to(scene_dir)))
            bboxes_xyxy.append(bbox)

        if len(frame_names) < int(args.min_frames):
            continue

        tracklets.append(
            {
                "track_id": f"{scene_dir.name}_{obj_dir.name}",
                "scene_dir": str(scene_dir),
                "object_id": obj_dir.name,
                "frame_names": frame_names,
                "mask_paths": rel_masks,
                "bboxes_xyxy": bboxes_xyxy,
            }
        )

    out_path.write_text(json.dumps(tracklets, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(tracklets)} tracklets to: {out_path}")


if __name__ == "__main__":
    main()

