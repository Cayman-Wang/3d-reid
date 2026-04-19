from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:  # pragma: no cover
        raise SystemExit(f"Failed to read json: {path}\nError: {e!r}")


def _read_image(path: Path, flags: int):
    import cv2  # type: ignore

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
    return img


def _load_mask(path: Path) -> np.ndarray:
    try:
        import cv2  # type: ignore

        mask = _read_image(path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(path)
        return np.asarray(mask) > 0
    except Exception:
        from PIL import Image  # type: ignore

        return np.asarray(Image.open(path).convert("L")) > 0


def _resolve_mask_path(scene_dir: Path, cam_id: str, mask_subdir: str, stem: str) -> Path | None:
    mask_root = scene_dir / "cams" / cam_id / str(mask_subdir)
    direct = mask_root / f"{stem}.png"
    if direct.exists():
        return direct

    # SAM2 node outputs are nested under obj_XXX/. Under the current single-target
    # assumption, use the first matching object folder if present.
    nested = sorted(mask_root.glob(f"obj_*/{stem}.png"))
    if nested:
        return nested[0]
    return None


def _mask_bbox_xyxy(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def _points_meet_threshold(points_path: Path, min_points: int) -> bool:
    try:
        points = np.load(points_path, mmap_mode="r")
    except Exception as exc:
        raise SystemExit(f"Failed to load points file: {points_path}\nError: {exc!r}")
    if points.ndim != 2:
        return False
    if points.shape[1] != 3:
        return False
    return int(points.shape[0]) >= int(min_points)


def _read_frame_index(frame_times_csv: Path, cams: list[str]) -> list[tuple[int, dict[str, str]]]:
    rows: dict[int, dict[str, str]] = {}
    with frame_times_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ts_us = int(row["ts_us"])
            except Exception as e:
                raise SystemExit(f"Invalid ts_us row in {frame_times_csv}: {row!r}\nError: {e!r}")
            cam_id = str(row["cam_id"]).strip()
            filename = str(row["filename"]).strip()
            if cam_id not in cams:
                continue
            rows.setdefault(ts_us, {})[cam_id] = filename

    ordered = sorted(rows.items(), key=lambda item: item[0])
    return ordered


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a single node-level tracklet from a MuJoCo node scene.")
    ap.add_argument("--scene_dir", required=True, type=str)
    ap.add_argument("--cams", default="cam0,cam1,cam2", type=str)
    ap.add_argument("--mask_subdir", default="masks", type=str)
    ap.add_argument("--depth_subdir", default="depth", type=str)
    ap.add_argument("--points_subdir", default="recon/points_fused", type=str)
    ap.add_argument("--out", default="tracks/tracklets.json", type=str)
    ap.add_argument("--identity_id", default="", type=str)
    ap.add_argument("--min_timestamps", default=1, type=int)
    ap.add_argument("--require_points", action="store_true")
    ap.add_argument("--min_points", default=1, type=int)
    args = ap.parse_args()

    scene_dir = Path(args.scene_dir).resolve()
    if not scene_dir.exists():
        raise SystemExit(f"--scene_dir not found: {scene_dir}")

    cams = [c.strip() for c in str(args.cams).split(",") if c.strip()]
    if not cams:
        raise SystemExit("--cams is empty")

    frame_times_csv = scene_dir / "frame_times.csv"
    rig_json = scene_dir / "calib" / "rig.json"
    capture_meta_json = scene_dir / "capture_meta.json"
    if not frame_times_csv.exists():
        raise SystemExit(f"frame_times.csv not found: {frame_times_csv}")
    if not rig_json.exists():
        raise SystemExit(f"rig.json not found: {rig_json}")
    if not capture_meta_json.exists():
        raise SystemExit(f"capture_meta.json not found: {capture_meta_json}")

    rig = _load_json(rig_json)
    rig_cams = rig.get("cameras", {})
    for cam_id in cams:
        if cam_id not in rig_cams:
            raise SystemExit(f'Camera "{cam_id}" missing from rig.json. Available: {sorted(rig_cams.keys())}')

    capture_meta = _load_json(capture_meta_json)
    node_id = str(capture_meta.get("node_id") or rig.get("node_id") or scene_dir.parent.parent.name)
    identity_id = str(args.identity_id).strip()
    if not identity_id:
        identity_id = str(capture_meta.get("target", {}).get("identity_id") or scene_dir.name)

    timestamps = _read_frame_index(frame_times_csv, cams)
    if not timestamps:
        raise SystemExit(f"No synchronized timestamps found in: {frame_times_csv}")

    out_per_cam: dict[str, dict[str, list[Any]]] = {
        cam_id: {"frame_paths": [], "mask_paths": [], "depth_paths": [], "bboxes_xyxy": []} for cam_id in cams
    }
    timestamps_us: list[int] = []
    timestamp_stems: list[str] = []
    fused_points_paths: list[str | None] = []

    for ts_us, frames_by_cam in timestamps:
        if any(cam_id not in frames_by_cam for cam_id in cams):
            continue

        stem = f"{ts_us:012d}"
        per_cam_bboxes: dict[str, list[int]] = {}
        per_cam_paths: dict[str, tuple[str, str, str]] = {}

        valid = True
        for cam_id in cams:
            frame_rel = frames_by_cam[cam_id]
            frame_path = scene_dir / frame_rel
            mask_path = _resolve_mask_path(scene_dir, cam_id, str(args.mask_subdir), stem)
            depth_path = scene_dir / "cams" / cam_id / str(args.depth_subdir) / f"{stem}.npy"
            if not frame_path.exists() or mask_path is None or not depth_path.exists():
                valid = False
                break

            bbox = _mask_bbox_xyxy(_load_mask(mask_path))
            if bbox is None:
                valid = False
                break

            per_cam_bboxes[cam_id] = bbox
            per_cam_paths[cam_id] = (
                frame_path.relative_to(scene_dir).as_posix(),
                mask_path.relative_to(scene_dir).as_posix(),
                depth_path.relative_to(scene_dir).as_posix(),
            )

        if not valid:
            continue

        points_rel: str | None = None
        points_path = scene_dir / str(args.points_subdir) / f"{stem}.npy"
        if points_path.exists():
            if bool(args.require_points):
                if not _points_meet_threshold(points_path, int(args.min_points)):
                    continue
            points_rel = points_path.relative_to(scene_dir).as_posix()
        if bool(args.require_points) and points_rel is None:
            continue

        timestamps_us.append(ts_us)
        timestamp_stems.append(stem)
        fused_points_paths.append(points_rel)
        for cam_id in cams:
            frame_rel, mask_rel, depth_rel = per_cam_paths[cam_id]
            out_per_cam[cam_id]["frame_paths"].append(frame_rel)
            out_per_cam[cam_id]["mask_paths"].append(mask_rel)
            out_per_cam[cam_id]["depth_paths"].append(depth_rel)
            out_per_cam[cam_id]["bboxes_xyxy"].append(per_cam_bboxes[cam_id])

    if len(timestamps_us) < int(args.min_timestamps):
        raise SystemExit(
            f"Only found {len(timestamps_us)} valid synchronized timestamps, less than --min_timestamps={args.min_timestamps}"
        )

    track = {
        "track_id": f"{node_id}_{scene_dir.name}",
        "scene_dir": str(scene_dir),
        "node_id": node_id,
        "identity_id": identity_id,
        "timestamp_stems": timestamp_stems,
        "timestamps_us": timestamps_us,
        "fused_points_paths": fused_points_paths,
        "per_camera": out_per_cam,
    }

    out_path = scene_dir / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps([track], ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote 1 track to: {out_path}")


if __name__ == "__main__":
    main()
