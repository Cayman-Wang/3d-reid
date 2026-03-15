from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np


def _sorted_images(images_dir: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png"}
    paths = [p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]

    def _key(p: Path):
        try:
            return (0, int(p.stem))
        except Exception:
            return (1, p.stem)

    return sorted(paths, key=_key)


def _resolve_device(device: str, torch) -> str:
    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_depth_anything_v2(model_id: str, device: str):
    try:
        import torch  # type: ignore
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation  # type: ignore
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            "Depth Anything V2 dependencies are not available. Install torch, transformers, pillow and numpy. "
            f"Original import error: {e!r}"
        )

    device_resolved = _resolve_device(device, torch)
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForDepthEstimation.from_pretrained(model_id).to(device_resolved).eval()
    return processor, model, torch, device_resolved


def infer_images_to_dir(
    *,
    images_dir: Path,
    out_dir: Path,
    model_id: str,
    processor,
    model,
    torch,
    device: str,
    overwrite: bool,
    max_images: int,
) -> dict:
    try:
        from PIL import Image  # type: ignore
    except Exception as e:  # pragma: no cover
        raise SystemExit(f"Missing Pillow dependency. Original import error: {e!r}")

    if not images_dir.exists():
        raise SystemExit(f"images dir not found: {images_dir}")

    image_paths = _sorted_images(images_dir)
    if max_images > 0:
        image_paths = image_paths[:max_images]
    if not image_paths:
        raise SystemExit(f"no images found under: {images_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    wrote = 0
    skipped = 0
    stems: list[str] = []
    for image_path in image_paths:
        stem = image_path.stem
        stems.append(stem)
        out_path = out_dir / f"{stem}.npy"
        if out_path.exists() and not overwrite:
            skipped += 1
            continue

        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
        post_processed = processor.post_process_depth_estimation(
            outputs,
            target_sizes=[(image.height, image.width)],
        )
        predicted_depth = post_processed[0]["predicted_depth"].detach().float().cpu().numpy().astype(np.float32)
        np.save(str(out_path), predicted_depth)
        wrote += 1

    meta = {
        "model_id": model_id,
        "device": device,
        "images_dir": str(images_dir),
        "count": len(image_paths),
        "wrote": wrote,
        "skipped": skipped,
        "depth_type": "relative",
        "image_stems": stems,
    }
    (out_dir / "depth_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return meta


def main(argv: Iterable[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Run Depth Anything V2 on an image directory and export per-frame .npy depth maps."
    )
    ap.add_argument("--scene_dir", required=True, type=str)
    ap.add_argument("--images_dir", default="images", type=str, help="Relative to scene_dir")
    ap.add_argument("--out_dir", default="depth", type=str, help="Relative to scene_dir")
    ap.add_argument(
        "--model_id",
        default="depth-anything/Depth-Anything-V2-Small-hf",
        type=str,
        help="Hugging Face model id for Depth Anything V2.",
    )
    ap.add_argument("--device", default="auto", type=str, help="auto|cpu|cuda")
    ap.add_argument("--max_images", default=0, type=int, help="Optional cap on the number of frames to process.")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(list(argv) if argv is not None else None)

    scene_dir = Path(args.scene_dir).resolve()
    images_dir = scene_dir / args.images_dir
    out_dir = scene_dir / args.out_dir

    processor, model, torch, device = build_depth_anything_v2(str(args.model_id), str(args.device))
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
    print(
        f"[depth-anything-v2] wrote={meta['wrote']} skipped={meta['skipped']} "
        f"count={meta['count']} out_dir={out_dir}"
    )


if __name__ == "__main__":
    main()
