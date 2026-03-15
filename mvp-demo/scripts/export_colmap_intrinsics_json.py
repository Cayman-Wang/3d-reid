from __future__ import annotations

import argparse
import json
import struct
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class _Camera:
    camera_id: int
    model: str
    width: int
    height: int
    params: tuple[float, ...]


# Mirrors COLMAP's camera model table (see colmap/scripts/python/read_write_model.py).
_CAMERA_MODEL_ID_TO_NAME = {
    0: "SIMPLE_PINHOLE",
    1: "PINHOLE",
    2: "SIMPLE_RADIAL",
    3: "RADIAL",
    4: "OPENCV",
    5: "OPENCV_FISHEYE",
    6: "FULL_OPENCV",
    7: "FOV",
    8: "SIMPLE_RADIAL_FISHEYE",
    9: "RADIAL_FISHEYE",
    10: "THIN_PRISM_FISHEYE",
}

_CAMERA_MODEL_NUM_PARAMS = {
    "SIMPLE_PINHOLE": 3,
    "PINHOLE": 4,
    "SIMPLE_RADIAL": 4,
    "RADIAL": 5,
    "OPENCV": 8,
    "OPENCV_FISHEYE": 8,
    "FULL_OPENCV": 12,
    "FOV": 5,
    "SIMPLE_RADIAL_FISHEYE": 4,
    "RADIAL_FISHEYE": 5,
    "THIN_PRISM_FISHEYE": 12,
}


def _read_next_bytes(fid, num_bytes: int, fmt: str, endian: str = "<"):
    data = fid.read(num_bytes)
    if len(data) != num_bytes:
        raise EOFError("Unexpected EOF while reading COLMAP binary model.")
    return struct.unpack(endian + fmt, data)


def _read_cameras_bin(path: Path) -> dict[int, _Camera]:
    cams: dict[int, _Camera] = {}
    with path.open("rb") as f:
        (num_cams,) = _read_next_bytes(f, 8, "Q")
        for _ in range(int(num_cams)):
            camera_id, model_id, width, height = _read_next_bytes(f, 24, "iiQQ")
            model = _CAMERA_MODEL_ID_TO_NAME.get(int(model_id))
            if model is None:
                raise ValueError(f"Unsupported/unknown COLMAP camera model id: {model_id}")
            n = _CAMERA_MODEL_NUM_PARAMS[model]
            params = _read_next_bytes(f, 8 * n, "d" * n)
            cams[int(camera_id)] = _Camera(
                camera_id=int(camera_id),
                model=str(model),
                width=int(width),
                height=int(height),
                params=tuple(float(p) for p in params),
            )
    return cams


def _camera_to_fx_fy_cx_cy(cam: _Camera) -> dict[str, float]:
    p = cam.params
    if cam.model == "PINHOLE":
        fx, fy, cx, cy = p[0], p[1], p[2], p[3]
    elif cam.model in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL", "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE"}:
        f, cx, cy = p[0], p[1], p[2]
        fx, fy = f, f
    elif cam.model in {"OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV", "FOV", "THIN_PRISM_FISHEYE"}:
        fx, fy, cx, cy = p[0], p[1], p[2], p[3]
    else:
        raise ValueError(f"Unsupported camera model for MVP: {cam.model}")

    return {"fx": float(fx), "fy": float(fy), "cx": float(cx), "cy": float(cy)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Export per-frame intrinsics.json from COLMAP sparse model (cameras.bin).")
    ap.add_argument("--scene_dir", required=True, type=str, help="Scene dir (contains sparse/0/cameras.bin + images/)")
    ap.add_argument("--sparse_dir", default="sparse/0", type=str, help="Relative sparse dir under scene_dir")
    ap.add_argument("--images_dir", default="images", type=str, help="Relative images dir under scene_dir")
    ap.add_argument("--camera_id", default="", type=str, help="Optional: pick a specific camera id (when multiple)")
    ap.add_argument("--out", default="intrinsics.json", type=str, help="Output file name under scene_dir")
    args = ap.parse_args()

    scene_dir = Path(args.scene_dir).resolve()
    cameras_bin = scene_dir / args.sparse_dir / "cameras.bin"
    images_dir = scene_dir / args.images_dir
    out_path = scene_dir / args.out

    if not cameras_bin.exists():
        raise SystemExit(f"COLMAP cameras.bin not found: {cameras_bin}")
    if not images_dir.exists():
        raise SystemExit(f"images dir not found: {images_dir}")

    cams = _read_cameras_bin(cameras_bin)
    if not cams:
        raise SystemExit(f"No cameras found in: {cameras_bin}")

    if args.camera_id:
        cid = int(args.camera_id)
        if cid not in cams:
            raise SystemExit(f"--camera_id {cid} not found in cameras.bin; available: {sorted(cams.keys())}")
        cam = cams[cid]
    else:
        if len(cams) != 1:
            raise SystemExit(f"Multiple cameras in model ({sorted(cams.keys())}); pass --camera_id to choose.")
        cam = next(iter(cams.values()))

    k = _camera_to_fx_fy_cx_cy(cam)
    k_meta = {"model": cam.model, "width": cam.width, "height": cam.height, "camera_id": cam.camera_id}

    imgs = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))
    if not imgs:
        raise SystemExit(f"No images found under: {images_dir}")

    out: dict[str, dict[str, object]] = {}
    for p in imgs:
        out[p.stem] = {
            "fx": k["fx"],
            "fy": k["fy"],
            "cx": k["cx"],
            "cy": k["cy"],
            # Keep width/height/model per entry so downstream can be stateless.
            "width": int(k_meta["width"]),
            "height": int(k_meta["height"]),
            "model": str(k_meta["model"]),
        }

    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(out)} intrinsics to: {out_path}")
    print(f"Camera: id={cam.camera_id} model={cam.model} size={cam.width}x{cam.height}")


if __name__ == "__main__":
    main()
