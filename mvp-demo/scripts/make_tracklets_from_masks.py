from __future__ import annotations

import argparse
import hashlib
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


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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
    ap.add_argument("--source", default="mask_directory_tracklets", type=str)
    ap.add_argument("--identity_id", default="", type=str)
    ap.add_argument("--diagnostic_only", action="store_true")
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
        input_lineage: list[dict] = []

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
            frame_rel = str(img.relative_to(scene_dir))
            mask_rel = str(mp.relative_to(scene_dir))
            rel_masks.append(mask_rel)
            bboxes_xyxy.append(bbox)
            input_lineage.append(
                {
                    "schema_version": "mask_tracklet_input_lineage_v1",
                    "frame_name": img.name,
                    "frame_path": frame_rel,
                    "frame_sha256": _sha256_file(img),
                    "mask_path": mask_rel,
                    "mask_sha256": _sha256_file(mp),
                    "bbox_xyxy": bbox,
                }
            )

        if len(frame_names) < int(args.min_frames):
            continue

        source = str(args.source)
        diagnostic_only = bool(args.diagnostic_only)
        identity_id = str(args.identity_id).strip() or obj_dir.name
        tracklets.append(
            {
                "schema_version": "mask_tracklet_v2",
                "track_id": f"{scene_dir.name}_{obj_dir.name}",
                "scene_dir": str(scene_dir),
                "object_id": obj_dir.name,
                "identity_id": identity_id,
                "frame_names": frame_names,
                "mask_paths": rel_masks,
                "bboxes_xyxy": bboxes_xyxy,
                "source": source,
                "diagnostic_only": diagnostic_only,
                "formal_scene_outputs_modified": False,
                "updates_pipeline_contract": False,
                "identity_proof": False,
                "pixel_accurate": False,
                "formal_synthetic_annotation_ready": False,
                "input_lineage": input_lineage,
                "non_promotion_policy": {
                    "mask_directory_tracklet_is_not_identity_proof": True,
                    "requires_external_target_selection_before_formal_use": True,
                    "does_not_update_pipeline_contract": True,
                },
            }
        )

    out_path.write_text(json.dumps(tracklets, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(tracklets)} tracklets to: {out_path}")


if __name__ == "__main__":
    main()
