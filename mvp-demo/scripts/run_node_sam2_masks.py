from __future__ import annotations

import argparse
import contextlib
from pathlib import Path

import numpy as np


def _parse_box(text: str) -> np.ndarray:
    parts = [p for p in str(text).replace(",", " ").split() if p.strip()]
    if len(parts) != 4:
        raise ValueError(f"expected 4 numbers, got {len(parts)}: {text!r}")
    return np.array([float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float32)


def _parse_camera_boxes(values: list[str]) -> dict[str, np.ndarray]:
    boxes: dict[str, np.ndarray] = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f'Invalid --camera_box value: {value!r}. Expected "cam0=x1,y1,x2,y2".')
        cam_id, box_text = value.split("=", 1)
        cam_id = cam_id.strip()
        if not cam_id:
            raise SystemExit(f'Invalid --camera_box value: {value!r}. Camera id is empty.')
        boxes[cam_id] = _parse_box(box_text)
    return boxes


def _sorted_frames(images_dir: Path) -> list[Path]:
    exts = [".jpg", ".jpeg", ".png"]
    frames: list[Path] = []
    for ext in exts:
        frames.extend(sorted(images_dir.glob(f"*{ext}")))

    def key(p: Path):
        try:
            return (0, int(p.stem))
        except Exception:
            return (1, p.stem)

    return sorted(frames, key=key)


def _resolve_cfg_name(cfg_arg: str) -> str:
    cfg_name = cfg_arg
    cfg_path = Path(cfg_arg).expanduser()
    if not cfg_path.exists():
        return cfg_name

    cfg_path = cfg_path.resolve()
    cfg_posix = cfg_path.as_posix()
    for marker in ("/sam2/configs/", "/configs/"):
        if marker in cfg_posix:
            return "configs/" + cfg_posix.split(marker, 1)[1]
    raise SystemExit(
        f"--model_cfg looks like a file path but cannot be mapped to a SAM2 config name: {cfg_path}\n"
        "Pass the config name instead, e.g. 'configs/sam2.1/sam2.1_hiera_l.yaml'."
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run SAM2 video masks over each camera stream in a MuJoCo node scene."
    )
    ap.add_argument("--scene_dir", required=True, type=str)
    ap.add_argument("--cams", default="cam0,cam1,cam2", type=str)
    ap.add_argument("--frames_subdir", default="frames", type=str)
    ap.add_argument("--masks_subdir", default="masks", type=str)
    ap.add_argument("--camera_box", action="append", default=[], type=str, help='Repeat: "cam0=x1,y1,x2,y2"')
    ap.add_argument("--obj_id", default=0, type=int)
    ap.add_argument("--init_frame", default=0, type=int)
    ap.add_argument("--checkpoint", required=True, type=str)
    ap.add_argument(
        "--model_cfg",
        required=True,
        type=str,
        help="SAM2 config name or a local yaml path.",
    )
    ap.add_argument("--device", default="auto", type=str, help="auto|cpu|cuda")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    scene_dir = Path(args.scene_dir).resolve()
    if not scene_dir.exists():
        raise SystemExit(f"--scene_dir not found: {scene_dir}")

    cams = [c.strip() for c in str(args.cams).split(",") if c.strip()]
    if not cams:
        raise SystemExit("--cams is empty")

    boxes = _parse_camera_boxes(list(args.camera_box))
    missing = [c for c in cams if c not in boxes]
    if missing:
        raise SystemExit(f"Missing --camera_box for cameras: {missing}")

    ckpt = Path(args.checkpoint).expanduser().resolve()
    if not ckpt.exists():
        raise SystemExit(f"--checkpoint not found: {ckpt}")
    cfg_name = _resolve_cfg_name(str(args.model_cfg).strip())

    try:
        import os
        import sys

        if sys.platform.startswith("linux") and hasattr(sys, "setdlopenflags") and hasattr(os, "RTLD_LAZY"):
            try:
                sys.setdlopenflags(os.RTLD_GLOBAL | os.RTLD_LAZY)
            except Exception:
                pass

        try:
            import ctypes

            mode = getattr(ctypes, "RTLD_GLOBAL", None) or getattr(os, "RTLD_GLOBAL", 0)
            prefix = Path(sys.prefix)
            for name in ("libittnotify.so", "libittnotify.so.1"):
                for cand in (prefix / "lib" / name, Path(name)):
                    try:
                        ctypes.CDLL(str(cand), mode=mode)
                        break
                    except Exception:
                        continue
        except Exception:
            pass

        import cv2  # type: ignore
        import torch  # type: ignore
        from sam2.build_sam import build_sam2_video_predictor  # type: ignore
    except Exception as e:
        raise SystemExit(
            "SAM2 dependencies are not available in this environment. "
            "Install torch and the local SAM2 package before running this script. "
            f"Original import error: {e!r}"
        )

    device = str(args.device)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    predictor = build_sam2_video_predictor(cfg_name, str(ckpt), device=device)
    autocast_ctx = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if device.startswith("cuda") and torch.cuda.is_available()
        else contextlib.nullcontext()
    )

    with torch.inference_mode(), autocast_ctx:
        for cam_id in cams:
            images_dir = scene_dir / "cams" / cam_id / str(args.frames_subdir)
            out_root = scene_dir / "cams" / cam_id / str(args.masks_subdir)
            if not images_dir.exists():
                raise SystemExit(f"frames dir not found for {cam_id}: {images_dir}")

            frames = _sorted_frames(images_dir)
            if not frames:
                raise SystemExit(f"no frames found for {cam_id}: {images_dir}")
            if not (0 <= int(args.init_frame) < len(frames)):
                raise SystemExit(f"--init_frame out of range for {cam_id}: {args.init_frame} (frames={len(frames)})")

            out_root.mkdir(parents=True, exist_ok=True)
            state = predictor.init_state(str(images_dir))
            predictor.add_new_points_or_box(
                state,
                box=boxes[cam_id],
                frame_idx=int(args.init_frame),
                obj_id=int(args.obj_id),
            )

            for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
                frame_idx = int(frame_idx)
                if not (0 <= frame_idx < len(frames)):
                    continue
                stem = frames[frame_idx].stem
                for oi, oid in enumerate(object_ids):
                    obj_dir = out_root / f"obj_{int(oid):03d}"
                    obj_dir.mkdir(parents=True, exist_ok=True)
                    out_path = obj_dir / f"{stem}.png"
                    if out_path.exists() and not args.overwrite:
                        continue
                    mask_i = masks[oi]
                    if getattr(mask_i, "ndim", None) == 3 and mask_i.shape[0] == 1:
                        mask_i = mask_i[0]
                    mask_u8 = (mask_i > 0).to(torch.uint8).cpu().numpy() * 255
                    ok = cv2.imwrite(str(out_path), mask_u8)
                    if not ok:
                        raise RuntimeError(f"Failed to write mask png: {out_path}")

            print(f"[node-sam2] cam={cam_id} frames={len(frames)} out_dir={out_root}")


if __name__ == "__main__":
    main()
