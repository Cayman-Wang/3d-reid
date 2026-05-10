from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_CAMS = ["cam0", "cam1", "cam2"]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"Failed to read json: {path}\nError: {exc!r}")


def _uniform_indices(total: int, count: int) -> list[int]:
    if total <= 0:
        return []
    if count >= total:
        return list(range(total))
    idx = np.linspace(0, total - 1, num=count)
    rounded = np.round(idx).astype(int)
    uniq: list[int] = []
    seen: set[int] = set()
    for i in rounded.tolist():
        if i not in seen:
            seen.add(i)
            uniq.append(i)
    if len(uniq) < count:
        for i in range(total):
            if i not in seen:
                uniq.append(i)
                seen.add(i)
            if len(uniq) >= count:
                break
    return sorted(uniq[:count])


def _read_sync_rows(frame_times_csv: Path, cams: list[str]) -> list[tuple[int, dict[str, str]]]:
    rows_by_ts: dict[int, dict[str, str]] = {}
    with frame_times_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ts_us = int(row["ts_us"])
            except Exception as exc:
                raise SystemExit(f"Invalid ts_us in {frame_times_csv}: {row!r}; error={exc!r}")
            cam_id = str(row["cam_id"]).strip()
            if cam_id not in cams:
                continue
            filename = str(row["filename"]).strip()
            rows_by_ts.setdefault(ts_us, {})[cam_id] = filename

    ordered = sorted(rows_by_ts.items(), key=lambda kv: kv[0])
    out: list[tuple[int, dict[str, str]]] = []
    for ts_us, by_cam in ordered:
        if all(cam in by_cam for cam in cams):
            out.append((ts_us, by_cam))
    return out


def _frame_stem_from_relpath(path_str: str) -> str:
    return Path(path_str).stem


def _extract_camera_priors(rig_cam: dict[str, Any], cam_id: str) -> tuple[list[list[float]], list[list[float]]]:
    K = rig_cam.get("K")
    pose = rig_cam.get("T_node_from_cam")
    if not isinstance(K, list) or len(K) != 3:
        raise SystemExit(f"Invalid rig.K for {cam_id}; expected 3x3 list")
    if not isinstance(pose, list) or len(pose) != 4:
        raise SystemExit(f"Invalid rig.T_node_from_cam for {cam_id}; expected 4x4 list")
    return K, pose


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare tri-camera NeoVerse multiview manifest from one scene.")
    ap.add_argument("--scene_dir", required=True, type=str)
    ap.add_argument("--cams", default="cam0,cam1,cam2", type=str)
    ap.add_argument("--num_sync_steps", default=27, type=int)
    ap.add_argument("--mask_subdir", default="masks_gt", type=str)
    ap.add_argument("--frame_times_csv", default="frame_times.csv", type=str)
    ap.add_argument("--rig_json", default="calib/rig.json", type=str)
    ap.add_argument("--out_root", default="mvp-demo/output/neoverse_multiview", type=str)
    args = ap.parse_args()

    scene_dir = Path(args.scene_dir).resolve()
    if not scene_dir.exists():
        raise SystemExit(f"Missing scene dir: {scene_dir}")

    cams = [c.strip() for c in str(args.cams).split(",") if c.strip()]
    if cams != DEFAULT_CAMS:
        raise SystemExit(
            f"This first version only supports cams={DEFAULT_CAMS}. Got: {cams}."
        )

    frame_times_csv = scene_dir / str(args.frame_times_csv)
    if not frame_times_csv.exists():
        raise SystemExit(f"Missing frame_times.csv: {frame_times_csv}")

    rig_json_path = scene_dir / str(args.rig_json)
    if not rig_json_path.exists():
        raise SystemExit(f"Missing rig json: {rig_json_path}")

    rig = _load_json(rig_json_path)
    rig_cameras = rig.get("cameras", {})
    for cam in cams:
        if cam not in rig_cameras:
            raise SystemExit(f"Camera {cam} missing in rig.json. Available: {sorted(rig_cameras.keys())}")

    sync_rows = _read_sync_rows(frame_times_csv, cams)
    if not sync_rows:
        raise SystemExit(f"No complete synchronized rows found in: {frame_times_csv}")

    chosen_idx = _uniform_indices(total=len(sync_rows), count=max(1, int(args.num_sync_steps)))
    selected_rows = [sync_rows[i] for i in chosen_idx]

    repo_root = _repo_root()
    out_root = Path(str(args.out_root))
    if not out_root.is_absolute():
        out_root = repo_root / out_root

    scene_id = scene_dir.name
    manifest_path = out_root / scene_id / "input" / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    views: list[dict[str, Any]] = []
    sync_steps: list[dict[str, Any]] = []
    repeated_timestamps: list[int] = []

    view_idx = 0
    for logical_t_idx, (ts_us, by_cam) in enumerate(selected_rows):
        stems = [_frame_stem_from_relpath(by_cam[cam]) for cam in cams]
        scene_stem = stems[0]
        if any(stem != scene_stem for stem in stems):
            raise SystemExit(
                f"Inconsistent frame stem at ts_us={ts_us}: {dict(zip(cams, stems))}"
            )

        sync_steps.append(
            {
                "logical_t_idx": int(logical_t_idx),
                "ts_us": int(ts_us),
                "scene_stem": scene_stem,
            }
        )

        for cam in cams:
            frame_rel = by_cam[cam]
            frame_abs = scene_dir / frame_rel
            if not frame_abs.exists():
                raise SystemExit(f"Frame missing: {frame_abs}")

            mask_rel = f"cams/{cam}/{args.mask_subdir}/{scene_stem}.png"
            mask_abs = scene_dir / mask_rel
            mask_rel_or_null: str | None = Path(mask_rel).as_posix() if mask_abs.exists() else None
            camera_K, camera_pose_c2w = _extract_camera_priors(rig_cameras[cam], cam)

            views.append(
                {
                    "view_idx": int(view_idx),
                    "scene_stem": scene_stem,
                    "logical_t_idx": int(logical_t_idx),
                    "cam_id": cam,
                    "frame_rel": Path(frame_rel).as_posix(),
                    "mask_rel": mask_rel_or_null,
                    "camera_K": camera_K,
                    "camera_pose_c2w": camera_pose_c2w,
                    "rig_cam": rig_cameras[cam],
                }
            )
            repeated_timestamps.append(int(logical_t_idx))
            view_idx += 1

    manifest = {
        "schema_version": "neoverse_multiview_manifest_v1",
        "scene_id": scene_id,
        "scene_dir": scene_dir.as_posix(),
        "frame_times_csv": frame_times_csv.as_posix(),
        "rig_json": rig_json_path.as_posix(),
        "cams": cams,
        "mask_subdir": str(args.mask_subdir),
        "timestamp_rule": "shared_logical_t_idx_per_sync_step",
        "num_sync_steps": len(sync_steps),
        "num_views": len(views),
        "sync_steps": sync_steps,
        "timestamps": repeated_timestamps,
        "views": views,
    }

    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote manifest: {manifest_path}")
    print(f"[summary] scene={scene_id} num_sync_steps={len(sync_steps)} num_views={len(views)} cams={cams}")


if __name__ == "__main__":
    main()
