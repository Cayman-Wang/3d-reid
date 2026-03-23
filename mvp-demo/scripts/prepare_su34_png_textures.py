from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert su34 JPG diffuse textures to PNG for MuJoCo.")
    ap.add_argument(
        "--map_dir",
        default=str(Path(__file__).resolve().parents[1] / "assets" / "models" / "su34" / "Map"),
        type=str,
    )
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    try:
        from PIL import Image
    except Exception as e:  # pragma: no cover
        raise SystemExit(f"Pillow is required. Install it with `python -m pip install pillow`. Import error: {e!r}")

    map_dir = Path(str(args.map_dir))
    if not map_dir.is_absolute():
        map_dir = Path.cwd() / map_dir
    map_dir = map_dir.resolve()
    if not map_dir.is_dir():
        raise SystemExit(f"--map_dir not found: {map_dir}")

    wrote = 0
    skipped = 0
    for jpg_path in sorted(map_dir.glob("*.jpg")):
        png_path = jpg_path.with_suffix(".png")
        if png_path.exists() and not bool(args.overwrite):
            skipped += 1
            continue
        with Image.open(jpg_path) as img:
            img.save(png_path)
        wrote += 1
        print(f"[su34-texture] wrote {png_path.name}")

    print(f"[su34-texture] wrote={wrote} skipped={skipped} map_dir={map_dir}")


if __name__ == "__main__":
    main()
