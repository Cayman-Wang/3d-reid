#!/usr/bin/env python3
"""Export placeholder CARLA-Air 4D geometry sidecars.

This script only reads an existing input_manifest.json and writes lightweight
diagnostic metadata. It does not run any model, does not copy images, and does
not generate fake depth or point outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_SCHEMA_VERSION = "carla_air_geometry_4d_sidecar_input_manifest_v1"
MANIFEST_SCHEMA_VERSION = "carla_air_geometry_4d_sidecar_placeholder_manifest_v1"
ALIGNMENT_SCHEMA_VERSION = "carla_air_geometry_4d_sidecar_placeholder_camera_alignment_v1"
QUALITY_SCHEMA_VERSION = "carla_air_geometry_4d_sidecar_placeholder_quality_summary_v1"
ALLOWED_METHODS = ("dggt", "mapanything", "dggt_mapanything_aligned")
OUTPUT_FILENAMES = ("manifest.json", "camera_alignment.json", "quality_summary.json")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _repo_rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"Failed to read JSON: {path}\nError: {exc!r}")
    if not isinstance(payload, dict):
        raise SystemExit(f"JSON root is not an object: {path}")
    return payload


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_keys(obj: dict[str, Any], path_label: str, keys: tuple[str, ...]) -> None:
    missing = [key for key in keys if key not in obj]
    if missing:
        raise SystemExit(f"Missing required field(s) in {path_label}: {', '.join(missing)}")


def _coerce_int(value: Any, label: str) -> int:
    try:
        return int(value)
    except Exception as exc:
        raise SystemExit(f"Invalid integer for {label}: {value!r}\nError: {exc!r}")


def _normalize_manifest(input_manifest_path: Path) -> dict[str, Any]:
    manifest = _load_json(input_manifest_path)
    _require_keys(
        manifest,
        str(input_manifest_path),
        ("schema_version", "capture_id", "method", "identity_id", "trajectory_id", "selected_ts_us", "views"),
    )
    if manifest["schema_version"] != INPUT_SCHEMA_VERSION:
        raise SystemExit(
            f"Unexpected input manifest schema_version in {input_manifest_path}: "
            f"{manifest['schema_version']!r} != {INPUT_SCHEMA_VERSION!r}"
        )
    method = str(manifest["method"])
    if method not in ALLOWED_METHODS:
        raise SystemExit(f"Unsupported method in {input_manifest_path}: {method!r}")

    selected_ts_raw = manifest["selected_ts_us"]
    if not isinstance(selected_ts_raw, list) or not selected_ts_raw:
        raise SystemExit(f"selected_ts_us must be a non-empty list in {input_manifest_path}")
    selected_ts_us = [_coerce_int(value, "selected_ts_us") for value in selected_ts_raw]

    views_raw = manifest["views"]
    if not isinstance(views_raw, list) or not views_raw:
        raise SystemExit(f"views must be a non-empty list in {input_manifest_path}")

    normalized_views: list[dict[str, Any]] = []
    for idx, view in enumerate(views_raw):
        if not isinstance(view, dict):
            raise SystemExit(f"View #{idx} is not an object in {input_manifest_path}")
        _require_keys(
            view,
            f"{input_manifest_path} view #{idx}",
            (
                "ts_us",
                "node_id",
                "camera_id",
                "frame_path",
                "K",
                "camera_pose_c2w",
                "camera_extrinsic_w2c",
                "drone_gt_pose",
            ),
        )
        normalized_view = dict(view)
        normalized_view["ts_us"] = _coerce_int(view["ts_us"], f"views[{idx}].ts_us")
        normalized_view["node_id"] = str(view["node_id"])
        normalized_view["camera_id"] = str(view["camera_id"])
        normalized_view["frame_path"] = str(view["frame_path"])
        if not isinstance(view["K"], list):
            raise SystemExit(f"views[{idx}].K must be a list in {input_manifest_path}")
        if not isinstance(view["camera_pose_c2w"], list):
            raise SystemExit(f"views[{idx}].camera_pose_c2w must be a list in {input_manifest_path}")
        if not isinstance(view["camera_extrinsic_w2c"], list):
            raise SystemExit(f"views[{idx}].camera_extrinsic_w2c must be a list in {input_manifest_path}")
        if not isinstance(view["drone_gt_pose"], dict):
            raise SystemExit(f"views[{idx}].drone_gt_pose must be an object in {input_manifest_path}")
        normalized_views.append(normalized_view)

    view_ts_set = {int(view["ts_us"]) for view in normalized_views}
    selected_ts_set = set(selected_ts_us)
    if view_ts_set != selected_ts_set:
        raise SystemExit(
            "selected_ts_us and views ts_us do not match in "
            f"{input_manifest_path}\n"
            f"selected_ts_us_only={sorted(selected_ts_set - view_ts_set)}\n"
            f"views_ts_only={sorted(view_ts_set - selected_ts_set)}"
        )

    manifest["capture_id"] = str(manifest["capture_id"])
    manifest["method"] = method
    manifest["identity_id"] = str(manifest["identity_id"])
    manifest["trajectory_id"] = str(manifest["trajectory_id"])
    manifest["selected_ts_us"] = selected_ts_us
    manifest["views"] = normalized_views
    return manifest


def _shared_flags() -> dict[str, bool]:
    return {
        "diagnostic_only": True,
        "non_promotion": True,
        "not_formal_geometry": True,
        "formal_annotation_ready": False,
        "final_4d_geometry_ready": False,
        "benchmark_ready": False,
        "inference_executed": False,
    }


def _build_outputs(input_manifest_path: Path, manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    input_manifest_sha256 = _sha256(input_manifest_path)
    input_manifest_ref = _repo_rel(input_manifest_path)
    capture_id = str(manifest["capture_id"])
    method = str(manifest["method"])
    identity_id = str(manifest["identity_id"])
    trajectory_id = str(manifest["trajectory_id"])
    views = list(manifest["views"])
    selected_ts_us = list(manifest["selected_ts_us"])
    carla_pose_available = all(isinstance(view.get("camera_carla_world_transform"), dict) for view in views)

    base_fields = {
        "capture_id": capture_id,
        "method": method,
        "identity_id": identity_id,
        "trajectory_id": trajectory_id,
        "input_manifest_path": input_manifest_ref,
        "input_manifest_sha256": input_manifest_sha256,
        "created_at": _utc_now_iso(),
        **_shared_flags(),
    }

    manifest_payload = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        **base_fields,
        "geometry_ready": False,
    }

    alignment_payload = {
        "schema_version": ALIGNMENT_SCHEMA_VERSION,
        **base_fields,
        "alignment_ready": False,
        "reason": "inference_not_executed",
        "carla_pose_available": bool(carla_pose_available),
    }

    quality_payload = {
        "schema_version": QUALITY_SCHEMA_VERSION,
        **base_fields,
        "geometry_ready": False,
        "depth_ready": False,
        "points_ready": False,
        "input_timestamp_count": len(selected_ts_us),
        "input_view_count": len(views),
        "blockers": ["inference_not_executed"],
    }

    return {
        "manifest.json": manifest_payload,
        "camera_alignment.json": alignment_payload,
        "quality_summary.json": quality_payload,
    }


def _ensure_placeholder_dirs(output_root: Path) -> None:
    for rel in ("depth_by_frame", "points_by_timestamp"):
        path = output_root / rel
        if path.exists() and not path.is_dir():
            raise SystemExit(f"Refusing to use non-directory placeholder path: {path}")
        path.mkdir(parents=True, exist_ok=True)


def _check_overwrite(output_root: Path, force: bool) -> None:
    existing = [output_root / name for name in OUTPUT_FILENAMES if (output_root / name).exists()]
    if existing and not force:
        rels = ", ".join(_repo_rel(path) for path in existing)
        raise SystemExit(f"Refusing to overwrite existing file(s) without --force: {rels}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Export placeholder CARLA-Air 4D geometry sidecars.")
    parser.add_argument("--input-manifest", required=True, type=str, help="Path to an existing input_manifest.json")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output sidecar files")
    args = parser.parse_args()

    input_manifest_path = Path(str(args.input_manifest))
    if not input_manifest_path.is_absolute():
        input_manifest_path = (REPO_ROOT / input_manifest_path).resolve()
    if not input_manifest_path.exists():
        raise SystemExit(f"Missing input manifest: {input_manifest_path}")
    if not input_manifest_path.is_file():
        raise SystemExit(f"Input manifest is not a file: {input_manifest_path}")

    output_root = input_manifest_path.parent
    manifest = _normalize_manifest(input_manifest_path)
    _check_overwrite(output_root, bool(args.force))
    _ensure_placeholder_dirs(output_root)

    outputs = _build_outputs(input_manifest_path, manifest)
    for filename, payload in outputs.items():
        _write_json(output_root / filename, payload)

    print(f"Wrote placeholder sidecars under: {_repo_rel(output_root)}")
    print(
        "[summary] "
        f"capture_id={manifest['capture_id']} method={manifest['method']} "
        f"input_view_count={len(manifest['views'])} "
        f"input_timestamp_count={len(manifest['selected_ts_us'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
