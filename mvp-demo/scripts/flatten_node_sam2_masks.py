from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def _pick_obj_dir(mask_root: Path, obj_dir_name: str) -> Path:
    if obj_dir_name:
        obj_dir = mask_root / obj_dir_name
        if not obj_dir.is_dir():
            raise SystemExit(f"Requested obj dir not found: {obj_dir}")
        return obj_dir

    obj_dirs = sorted(p for p in mask_root.glob("obj_*") if p.is_dir())
    if not obj_dirs:
        raise SystemExit(f"No obj_* dirs found under: {mask_root}")
    return obj_dirs[0]


def main() -> None:
    ap = argparse.ArgumentParser(description="Flatten SAM2 obj_XXX masks into cams/<cam>/masks/<stem>.png.")
    ap.add_argument("--scene_dir", required=True, type=str)
    ap.add_argument("--cams", default="cam0,cam1,cam2", type=str)
    ap.add_argument("--masks_subdir", default="masks", type=str)
    ap.add_argument("--obj_dir_name", default="", type=str)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    scene_dir = Path(args.scene_dir).resolve()
    if not scene_dir.exists():
        raise SystemExit(f"--scene_dir not found: {scene_dir}")

    cams = [c.strip() for c in str(args.cams).split(",") if c.strip()]
    if not cams:
        raise SystemExit("--cams is empty")

    for cam_id in cams:
        mask_root = scene_dir / "cams" / cam_id / str(args.masks_subdir)
        if not mask_root.is_dir():
            raise SystemExit(f"Mask root not found: {mask_root}")

        obj_dir = _pick_obj_dir(mask_root, str(args.obj_dir_name).strip())
        wrote = 0
        skipped = 0
        for src_path in sorted(obj_dir.glob("*.png")):
            dst_path = mask_root / src_path.name
            if dst_path.exists() and not args.overwrite:
                skipped += 1
                continue
            shutil.copyfile(src_path, dst_path)
            wrote += 1

        print(f"[flatten-node-sam2] cam={cam_id} obj_dir={obj_dir.name} wrote={wrote} skipped={skipped}")


if __name__ == "__main__":
    main()
