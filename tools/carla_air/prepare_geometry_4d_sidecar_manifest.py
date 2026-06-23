#!/usr/bin/env python3
"""Prepare a dry-run input manifest for CARLA-Air 4D geometry sidecars.

This tool only reads existing capture artifacts and writes a lightweight
input_manifest.json. It does not run any geometry model and does not copy
images or large assets.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "carla_air_geometry_4d_sidecar_input_manifest_v1"
ALLOWED_METHODS = ("dggt", "mapanything", "dggt_mapanything_aligned")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"Failed to read JSON: {path}\nError: {exc!r}")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    except Exception as exc:
        raise SystemExit(f"Failed to read CSV: {path}\nError: {exc!r}")


def _repo_rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _resolve_image_path(capture_root: Path, row: dict[str, str], node_id: str, camera_id: str, ts_us: int) -> Path:
    raw_map = row.get("image_files_json") or ""
    try:
        image_files = json.loads(raw_map) if raw_map else {}
    except Exception as exc:
        raise SystemExit(
            f"Invalid image_files_json for node={node_id} ts_us={ts_us} under {capture_root}\nError: {exc!r}"
        )
    if not isinstance(image_files, dict):
        raise SystemExit(f"image_files_json is not an object for node={node_id} ts_us={ts_us}")

    raw_rel = str(image_files.get(camera_id) or "").strip()
    candidates: list[Path] = []
    if raw_rel:
        raw_path = Path(raw_rel)
        if raw_path.is_absolute():
            candidates.append(raw_path)
        node_dir_raw = str(row.get("node_dir") or "").strip()
        if node_dir_raw:
            candidates.append((Path(node_dir_raw) / raw_path).resolve())
        candidates.append((capture_root / "nodes" / node_id / raw_path).resolve())

    stem = f"{ts_us:016d}"
    for suffix in (".png", ".jpg", ".jpeg"):
        candidates.append((capture_root / "nodes" / node_id / "cams" / camera_id / "frames" / f"{stem}{suffix}").resolve())

    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise SystemExit(
        f"Could not resolve frame for node={node_id} camera={camera_id} ts_us={ts_us} under {capture_root}"
    )


def _rotation_matrix_from_carla(pitch: float, yaw: float, roll: float) -> np.ndarray:
    p = math.radians(float(pitch))
    y = math.radians(float(yaw))
    r = math.radians(float(roll))
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    cr, sr = math.cos(r), math.sin(r)

    forward = [cp * cy, cp * sy, sp]
    right = [cy * sp * sr - sy * cr, sy * sp * sr + cy * cr, -cp * sr]
    up = [-cy * sp * cr - sy * sr, -sy * sp * cr + cy * sr, cp * cr]
    return np.asarray(
        [
            [forward[0], right[0], up[0]],
            [forward[1], right[1], up[1]],
            [forward[2], right[2], up[2]],
        ],
        dtype=np.float64,
    )


def _camera_pose_c2w_and_w2c_from_carla(pose: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    r_world_from_carla_cam = _rotation_matrix_from_carla(
        pitch=float(pose.get("pitch", 0.0)),
        yaw=float(pose.get("yaw", 0.0)),
        roll=float(pose.get("roll", 0.0)),
    )
    t_world = np.asarray(
        [float(pose["x"]), float(pose["y"]), float(pose["z"])],
        dtype=np.float64,
    )

    # OpenCV camera basis (+X right, +Y down, +Z forward) expressed in CARLA camera basis
    r_carla_cam_from_cv_cam = np.asarray(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ],
        dtype=np.float64,
    )
    r_world_from_cv_cam = r_world_from_carla_cam @ r_carla_cam_from_cv_cam

    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = r_world_from_cv_cam
    c2w[:3, 3] = t_world

    w2c = np.eye(4, dtype=np.float64)
    w2c[:3, :3] = r_world_from_cv_cam.T
    w2c[:3, 3] = -(r_world_from_cv_cam.T @ t_world)
    return c2w, w2c


def _infer_capture_id(capture_root: Path, capture_meta: dict[str, Any]) -> str:
    for key in ("run_id", "capture_id"):
        value = capture_meta.get(key)
        if value:
            return str(value)
    return capture_root.name


def _infer_identity_id(capture_meta: dict[str, Any], rows: list[dict[str, str]]) -> str:
    for key in ("identity_id", "object_identity_id"):
        value = capture_meta.get(key)
        if value:
            return str(value)
    target = capture_meta.get("target")
    if isinstance(target, dict) and target.get("identity_id"):
        return str(target["identity_id"])
    for row in rows:
        value = row.get("identity_id")
        if value:
            return str(value)
    raise SystemExit("Could not infer identity_id from capture_meta.json or trajectory_frame_groups.csv")


def _infer_trajectory_id(capture_meta: dict[str, Any], rows: list[dict[str, str]]) -> str:
    for key in ("trajectory_id", "trajectory_name", "route_id"):
        value = capture_meta.get(key)
        if value:
            return str(value)
    nested = capture_meta.get("trajectory")
    if isinstance(nested, dict):
        nested_traj = nested.get("trajectory")
        if isinstance(nested_traj, dict) and nested_traj.get("trajectory_id"):
            return str(nested_traj["trajectory_id"])
    for row in rows:
        value = row.get("trajectory_id")
        if value:
            return str(value)
    raise SystemExit("Could not infer trajectory_id from capture_meta.json or trajectory_frame_groups.csv")


def _parse_valid_timestamp_group(
    capture_root: Path,
    ts_us: int,
    rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: (str(item.get("node_id") or ""), str(item.get("planned_frame_index") or ""))):
        node_id = str(row.get("node_id") or "").strip()
        if not node_id:
            raise SystemExit(f"Missing node_id for ts_us={ts_us} under {capture_root}")
        raw_camera_meta = row.get("camera_meta_json") or ""
        raw_pose = row.get("drone_carla_pose_json") or ""
        try:
            camera_meta = json.loads(raw_camera_meta)
        except Exception as exc:
            raise SystemExit(f"Invalid camera_meta_json for node={node_id} ts_us={ts_us}\nError: {exc!r}")
        try:
            drone_pose = json.loads(raw_pose)
        except Exception as exc:
            raise SystemExit(f"Invalid drone_carla_pose_json for node={node_id} ts_us={ts_us}\nError: {exc!r}")
        if not isinstance(camera_meta, dict) or not camera_meta:
            raise SystemExit(f"camera_meta_json missing usable camera entries for node={node_id} ts_us={ts_us}")
        if not isinstance(drone_pose, dict):
            raise SystemExit(f"drone_carla_pose_json is not an object for node={node_id} ts_us={ts_us}")

        for camera_id in sorted(camera_meta.keys()):
            raw_entry = camera_meta[camera_id]
            if not isinstance(raw_entry, dict):
                raise SystemExit(f"camera_meta_json[{camera_id}] is not an object for node={node_id} ts_us={ts_us}")
            if "K" not in raw_entry or "carla_world_transform" not in raw_entry:
                raise SystemExit(
                    f"camera_meta_json[{camera_id}] missing K/carla_world_transform for node={node_id} ts_us={ts_us}"
                )
            frame_path = _resolve_image_path(capture_root, row, node_id=node_id, camera_id=str(camera_id), ts_us=ts_us)
            camera_pose = raw_entry["carla_world_transform"]
            if not isinstance(camera_pose, dict):
                raise SystemExit(
                    f"camera_meta_json[{camera_id}].carla_world_transform is not an object for node={node_id} ts_us={ts_us}"
                )
            c2w, w2c = _camera_pose_c2w_and_w2c_from_carla(camera_pose)
            out.append(
                {
                    "ts_us": int(ts_us),
                    "planned_frame_index": str(row.get("planned_frame_index") or ""),
                    "planned_t_sec": float(row.get("planned_t_sec") or 0.0),
                    "node_id": node_id,
                    "camera_id": str(camera_id),
                    "frame_path": _repo_rel(frame_path),
                    "K": raw_entry["K"],
                    "image_size": raw_entry.get("image_size"),
                    "camera_pose_c2w": c2w.tolist(),
                    "camera_extrinsic_w2c": w2c.tolist(),
                    "camera_pose_convention": "opencv_rdf_cam2world_in_carla_world_frame",
                    "camera_extrinsic_convention": "opencv_rdf_world2cam_in_carla_world_frame",
                    "camera_carla_world_transform": camera_pose,
                    "drone_gt_pose": drone_pose,
                }
            )
    if not out:
        raise SystemExit(f"No usable views parsed for ts_us={ts_us} under {capture_root}")
    return out


def build_manifest(capture_root: Path, method: str, max_timestamps: int, output_root: Path) -> tuple[Path, dict[str, Any]]:
    groups_path = capture_root / "trajectory_frame_groups.csv"
    capture_meta_path = capture_root / "capture_meta.json"
    for path in (groups_path, capture_meta_path):
        if not path.exists():
            raise SystemExit(f"Missing required file: {path}")

    rows = _read_csv_rows(groups_path)
    if not rows:
        raise SystemExit(f"No rows in trajectory_frame_groups.csv: {groups_path}")
    capture_meta = _load_json(capture_meta_path)

    capture_id = _infer_capture_id(capture_root, capture_meta)
    identity_id = _infer_identity_id(capture_meta, rows)
    trajectory_id = _infer_trajectory_id(capture_meta, rows)

    grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        raw_ts = str(row.get("ts_us") or "").strip()
        if not raw_ts:
            continue
        try:
            grouped[int(raw_ts)].append(row)
        except Exception as exc:
            raise SystemExit(f"Invalid ts_us value in {groups_path}: {raw_ts!r}\nError: {exc!r}")

    if not grouped:
        raise SystemExit(f"No usable ts_us groups in trajectory_frame_groups.csv: {groups_path}")

    selected_ts = sorted(grouped.keys())[: max(1, int(max_timestamps))]
    if len(selected_ts) < 1:
        raise SystemExit(f"Need at least one valid timestamp, got zero from {groups_path}")

    views_by_ts: list[dict[str, Any]] = []
    flat_views: list[dict[str, Any]] = []
    for ts_us in selected_ts:
        views = _parse_valid_timestamp_group(capture_root, ts_us=ts_us, rows=grouped[ts_us])
        flat_views.extend(views)
        views_by_ts.append(
            {
                "ts_us": int(ts_us),
                "view_count": int(len(views)),
                "views": views,
            }
        )

    method_root = output_root / capture_id / method
    output_path = method_root / "input_manifest.json"
    payload = {
        "schema_version": SCHEMA_VERSION,
        "capture_id": capture_id,
        "method": method,
        "identity_id": identity_id,
        "trajectory_id": trajectory_id,
        "selected_ts_us": [int(v) for v in selected_ts],
        "views": flat_views,
        "views_by_timestamp": views_by_ts,
        "diagnostic_only": True,
        "non_promotion": True,
        "not_formal_geometry": True,
        "source_capture_root": _repo_rel(capture_root),
        "source_capture_meta": _repo_rel(capture_meta_path),
        "source_trajectory_frame_groups": _repo_rel(groups_path),
        "created_at": _utc_now_iso(),
    }
    return output_path, payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare dry-run input manifests for CARLA-Air 4D geometry sidecars.")
    parser.add_argument("--capture-root", required=True, type=str, help="Capture root containing trajectory_frame_groups.csv")
    parser.add_argument("--method", required=True, choices=ALLOWED_METHODS, type=str)
    parser.add_argument("--max-timestamps", default=10, type=int)
    parser.add_argument("--output-root", default="local/carla_air/geometry_4d", type=str)
    args = parser.parse_args()

    capture_root = Path(str(args.capture_root))
    if not capture_root.is_absolute():
        capture_root = (REPO_ROOT / capture_root).resolve()
    output_root = Path(str(args.output_root))
    if not output_root.is_absolute():
        output_root = (REPO_ROOT / output_root).resolve()

    output_path, payload = build_manifest(
        capture_root=capture_root,
        method=str(args.method),
        max_timestamps=max(1, int(args.max_timestamps)),
        output_root=output_root,
    )
    _write_json(output_path, payload)
    print(f"Wrote: {output_path}")
    print(
        "[summary] "
        f"capture_id={payload['capture_id']} method={payload['method']} "
        f"selected_ts={len(payload['selected_ts_us'])} "
        f"total_views={sum(int(item['view_count']) for item in payload['views_by_timestamp'])}"
    )


if __name__ == "__main__":
    main()
