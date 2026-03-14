from __future__ import annotations

import argparse
import contextlib
from pathlib import Path

import numpy as np


def _parse_box(s: str) -> np.ndarray:
    s = (s or "").strip()
    if not s:
        raise ValueError("empty")
    parts = [p for p in s.replace(",", " ").split() if p.strip()]
    if len(parts) != 4:
        raise ValueError(f"expected 4 numbers (x1 y1 x2 y2), got {len(parts)}")
    x1, y1, x2, y2 = [float(p) for p in parts]
    return np.array([x1, y1, x2, y2], dtype=np.float32)


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


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run SAM2 VideoPredictor on SCENE_DIR/images and export masks/obj_XXX/<stem>.png"
    )
    ap.add_argument("--scene_dir", required=True, type=str)
    ap.add_argument("--images_dir", default="images", type=str)
    ap.add_argument("--out_dir", default="masks", type=str, help="Relative to scene_dir")
    ap.add_argument("--obj_id", default=0, type=int)
    ap.add_argument("--init_frame", default=0, type=int, help="0-based index into the sorted images list")
    ap.add_argument("--init_box", required=True, type=str, help='x1,y1,x2,y2 in pixel coords of SCENE_DIR/images/*')
    ap.add_argument("--checkpoint", required=True, type=str, help="SAM2 checkpoint path (e.g. third_party/sam2/checkpoints/*.pt)")
    ap.add_argument(
        "--model_cfg",
        required=True,
        type=str,
        help="SAM2 config name (recommended, e.g. 'configs/sam2.1/sam2.1_hiera_l.yaml') or a local yaml path",
    )
    ap.add_argument("--device", default="auto", type=str, help="auto|cpu|cuda")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    scene_dir = Path(args.scene_dir).resolve()
    images_dir = scene_dir / args.images_dir
    out_root = scene_dir / args.out_dir

    if not images_dir.exists():
        raise SystemExit(f"images dir not found: {images_dir}")

    frames = _sorted_frames(images_dir)
    if not frames:
        raise SystemExit(f"no images found under: {images_dir}")
    if not (0 <= int(args.init_frame) < len(frames)):
        raise SystemExit(f"--init_frame out of range: {args.init_frame} (frames={len(frames)})")

    init_box = _parse_box(str(args.init_box))
    ckpt = Path(args.checkpoint).expanduser().resolve()
    cfg_arg = str(args.model_cfg).strip()
    if not ckpt.exists():
        raise SystemExit(f"--checkpoint not found: {ckpt}")

    # `build_sam2_video_predictor()` expects a Hydra config name (e.g. "configs/sam2.1/...yaml").
    # For convenience we also accept a local file path and convert it to the config name when possible.
    cfg_name = cfg_arg
    cfg_path = Path(cfg_arg).expanduser()
    if cfg_path.exists():
        cfg_path = cfg_path.resolve()
        cfg_posix = cfg_path.as_posix()
        marker = "/sam2/configs/"
        if marker in cfg_posix:
            cfg_name = "configs/" + cfg_posix.split(marker, 1)[1]
        else:
            marker2 = "/configs/"
            if marker2 in cfg_posix:
                cfg_name = "configs/" + cfg_posix.split(marker2, 1)[1]
            else:
                raise SystemExit(
                    f"--model_cfg looks like a file path but cannot be mapped to a SAM2 config name: {cfg_path}\n"
                    f"Pass the config name instead, e.g. 'configs/sam2.1/sam2.1_hiera_l.yaml'."
                )

    try:
        import os
        import sys

        # Workaround: some PyTorch builds reference optional iJIT symbols (Intel JIT profiling).
        # Using RTLD_LAZY defers symbol resolution so `import torch` can succeed.
        if sys.platform.startswith("linux") and hasattr(sys, "setdlopenflags") and hasattr(os, "RTLD_LAZY"):
            try:
                sys.setdlopenflags(os.RTLD_GLOBAL | os.RTLD_LAZY)
            except Exception:
                pass

        # If available (e.g. conda-forge `ittapi`), preload ITT notify so iJIT_* symbols resolve.
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
            "SAM2 dependencies not available in this environment. "
            "Run this script inside your sam2 env (torch + sam2 installed). "
            f"Original import error: {e}"
        )

    device = str(args.device)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    predictor = build_sam2_video_predictor(cfg_name, str(ckpt), device=device)

    # SAM2 prefers inference_mode + autocast(bfloat16) on CUDA. CPU is supported but slow.
    autocast_ctx = (
        torch.autocast("cuda", dtype=torch.bfloat16) if device.startswith("cuda") and torch.cuda.is_available() else contextlib.nullcontext()
    )

    out_root.mkdir(parents=True, exist_ok=True)
    print(f"[sam2] frames={len(frames)} images_dir={images_dir}")
    print(f"[sam2] init_frame={args.init_frame} init_box={init_box.tolist()} obj_id={args.obj_id}")
    print(f"[sam2] out_dir={out_root} device={device}")

    with torch.inference_mode(), autocast_ctx:
        state = predictor.init_state(str(images_dir))
        predictor.add_new_points_or_box(state, box=init_box, frame_idx=int(args.init_frame), obj_id=int(args.obj_id))

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
                m = (masks[oi] > 0).to(torch.uint8).cpu().numpy() * 255
                cv2.imwrite(str(out_path), m)

    print(f"[sam2] done: {out_root}")


if __name__ == "__main__":
    main()
