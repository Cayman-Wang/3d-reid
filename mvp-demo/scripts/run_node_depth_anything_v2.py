from __future__ import annotations

import argparse
from pathlib import Path

from run_depth_anything_v2 import build_depth_anything_v2, infer_images_to_dir


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run Depth Anything V2 over each camera stream in a MuJoCo node scene."
    )
    ap.add_argument("--scene_dir", required=True, type=str)
    ap.add_argument("--cams", default="cam0,cam1,cam2", type=str)
    ap.add_argument("--frames_subdir", default="frames", type=str)
    ap.add_argument("--depth_subdir", default="depth", type=str)
    ap.add_argument(
        "--model_id",
        default="depth-anything/Depth-Anything-V2-Small-hf",
        type=str,
        help="Hugging Face model id for Depth Anything V2.",
    )
    ap.add_argument("--device", default="auto", type=str, help="auto|cpu|cuda")
    ap.add_argument("--max_images", default=0, type=int)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    scene_dir = Path(args.scene_dir).resolve()
    if not scene_dir.exists():
        raise SystemExit(f"--scene_dir not found: {scene_dir}")

    cams = [c.strip() for c in str(args.cams).split(",") if c.strip()]
    if not cams:
        raise SystemExit("--cams is empty")

    processor, model, torch, device = build_depth_anything_v2(str(args.model_id), str(args.device))

    total_wrote = 0
    total_skipped = 0
    for cam_id in cams:
        images_dir = scene_dir / "cams" / cam_id / str(args.frames_subdir)
        out_dir = scene_dir / "cams" / cam_id / str(args.depth_subdir)
        meta = infer_images_to_dir(
            images_dir=images_dir,
            out_dir=out_dir,
            model_id=str(args.model_id),
            processor=processor,
            model=model,
            torch=torch,
            device=device,
            overwrite=bool(args.overwrite),
            max_images=int(args.max_images),
        )
        total_wrote += int(meta["wrote"])
        total_skipped += int(meta["skipped"])
        print(
            f"[node-depth] cam={cam_id} wrote={meta['wrote']} skipped={meta['skipped']} "
            f"count={meta['count']} out_dir={out_dir}"
        )

    print(f"[node-depth] total_wrote={total_wrote} total_skipped={total_skipped}")


if __name__ == "__main__":
    main()
