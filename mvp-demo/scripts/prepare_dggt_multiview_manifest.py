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
            ts_us = int(row["ts_us"])
            cam_id = str(row["cam_id"]).strip()
            if cam_id not in cams:
                continue
            filename = str(row["filename"]).strip()
            rows_by_ts.setdefault(ts_us, {})[cam_id] = filename

    ordered = sorted(rows_by_ts.items(), key=lambda kv: kv[0])
    out: list[tuple[int, dict[str, str]]] = []
    for ts_us, by_cam in ordered:
        missing = [cam for cam in cams if cam not in by_cam]
        if missing:
            raise SystemExit(
                f"Synchronized timestep missing cameras at ts_us={ts_us}: missing={missing}, present={sorted(by_cam.keys())}"
            )
        out.append((ts_us, by_cam))
    return out


def _frame_stem_from_relpath(path_str: str) -> str:
    return Path(path_str).stem


def _inverse_4x4(matrix_4x4: list[list[float]]) -> np.ndarray:
    m = np.asarray(matrix_4x4, dtype=np.float64)
    if m.shape != (4, 4):
        raise SystemExit(f"Expected 4x4 matrix, got shape {m.shape}")
    return np.linalg.inv(m)


def _resolve_mask_rel(scene_dir: Path, cam_id: str, scene_stem: str, mask_source: str) -> tuple[str | None, str]:
    candidates: list[tuple[str, str]]
    if mask_source == "masks_gt":
        candidates = [("masks_gt", f"cams/{cam_id}/masks_gt/{scene_stem}.png")]
    elif mask_source == "masks":
        candidates = [("masks", f"cams/{cam_id}/masks/{scene_stem}.png")]
    else:
        candidates = [
            ("masks_gt", f"cams/{cam_id}/masks_gt/{scene_stem}.png"),
            ("masks", f"cams/{cam_id}/masks/{scene_stem}.png"),
        ]

    for source_name, rel in candidates:
        if (scene_dir / rel).exists():
            return Path(rel).as_posix(), source_name
    return None, "dynamic_conf"


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare tri-camera DGGT multiview manifest from one scene.")
    ap.add_argument("--scene_dir", required=True, type=str)
    ap.add_argument("--cam_ids", default="cam0,cam1,cam2", type=str)
    ap.add_argument("--num_sync_steps", default=27, type=int)
    ap.add_argument("--mask_source", default="auto", choices=["auto", "masks_gt", "masks"], type=str)
    ap.add_argument("--frame_times_csv", default="frame_times.csv", type=str)
    ap.add_argument("--rig_json", default="calib/rig.json", type=str)
    ap.add_argument("--out_root", default="mvp-demo/output/dggt_multiview", type=str)
    args = ap.parse_args()

    scene_dir = Path(args.scene_dir).resolve()
    if not scene_dir.exists():
        raise SystemExit(f"Missing scene dir: {scene_dir}")

    cams = [c.strip() for c in str(args.cam_ids).split(",") if c.strip()]
    if cams != DEFAULT_CAMS:
        raise SystemExit(f"This first version only supports cam_ids={DEFAULT_CAMS}. Got: {cams}")

    frame_times_csv = scene_dir / str(args.frame_times_csv)
    if not frame_times_csv.exists():
        raise SystemExit(f"Missing frame_times.csv: {frame_times_csv}")

    rig_json_path = scene_dir / str(args.rig_json)
    if not rig_json_path.exists():
        raise SystemExit(f"Missing rig json: {rig_json_path}")

    rig = _load_json(rig_json_path)
    rig_cameras = dict(rig.get("cameras") or {})
    for cam in cams:
        if cam not in rig_cameras:
            raise SystemExit(f"Camera {cam} missing in rig.json. Available: {sorted(rig_cameras.keys())}")

    sync_rows = _read_sync_rows(frame_times_csv, cams)
    if not sync_rows:
        raise SystemExit(f"No synchronized rows found in: {frame_times_csv}")

    chosen_idx = _uniform_indices(total=len(sync_rows), count=max(1, int(args.num_sync_steps)))
    selected_rows = [sync_rows[i] for i in chosen_idx]

    repo_root = _repo_root()
    out_root = Path(str(args.out_root))
    if not out_root.is_absolute():
        out_root = repo_root / out_root

    scene_id = scene_dir.name
    manifest_path = out_root / scene_id / "input" / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    sync_steps: list[dict[str, Any]] = []
    views: list[dict[str, Any]] = []
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
            frame_rel = Path(by_cam[cam]).as_posix()
            frame_abs = scene_dir / frame_rel
            if not frame_abs.exists():
                raise SystemExit(f"Frame missing for synchronized step: {frame_abs}")

            rig_cam = dict(rig_cameras[cam])
            K = np.asarray(rig_cam.get("K"), dtype=np.float64)
            if K.shape != (3, 3):
                raise SystemExit(f"Invalid rig K for {cam}; expected 3x3, got {K.shape}")

            T_node_from_cam = rig_cam.get("T_node_from_cam")
            T_w_from_c = np.asarray(T_node_from_cam, dtype=np.float64)
            if T_w_from_c.shape != (4, 4):
                raise SystemExit(f"Invalid rig T_node_from_cam for {cam}; expected 4x4, got {T_w_from_c.shape}")
            T_c_from_w = _inverse_4x4(T_w_from_c)
            mask_rel, mask_source_used = _resolve_mask_rel(
                scene_dir=scene_dir,
                cam_id=cam,
                scene_stem=scene_stem,
                mask_source=str(args.mask_source),
            )

            views.append(
                {
                    "view_idx": int(view_idx),
                    "logical_t_idx": int(logical_t_idx),
                    "scene_stem": scene_stem,
                    "cam_id": cam,
                    "frame_rel": frame_rel,
                    "mask_rel": mask_rel,
                    "mask_source_used": mask_source_used,
                    "camera_intrinsic_3x3": K.tolist(),
                    "camera_extrinsic_w2c_3x4": T_c_from_w[:3, :4].tolist(),
                }
            )
            repeated_timestamps.append(int(logical_t_idx))
            view_idx += 1

    manifest = {
        "schema_version": "dggt_multiview_manifest_v1",
        "scene_id": scene_id,
        "scene_dir": scene_dir.as_posix(),
        "cams": cams,
        "frame_times_csv": frame_times_csv.as_posix(),
        "rig_json": rig_json_path.as_posix(),
        "mask_source": str(args.mask_source),
        "timestamp_rule": "shared_logical_t_idx_per_sync_step",
        "num_sync_steps": len(sync_steps),
        "num_views": len(views),
        "sync_steps": sync_steps,
        "timestamps": repeated_timestamps,
        "views": views,
    }

    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote manifest: {manifest_path}")
    print(
        f"[summary] scene={scene_id} num_sync_steps={len(sync_steps)} num_views={len(views)} cams={cams} mask_source={args.mask_source}"
    )


if __name__ == "__main__":
    main()
