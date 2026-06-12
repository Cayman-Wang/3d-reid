#!/usr/bin/env python3
"""Offline verifier for CARLA-Air Dataset Generation Pipeline V1 run outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import re
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
LOCAL_ROOT = REPO_ROOT / "local"
FORBIDDEN_ROOT_TOKENS = {"weak", "proxy"}

REQUIRED_ARTIFACTS = {
    "dataset_plan.json": "dataset_plan",
    "dataset_manifest.json": "dataset_manifest",
    "dataset_samples.jsonl": "dataset_samples_jsonl",
    "dataset_splits.json": "dataset_splits",
    "deployment_episodes.json": "deployment_episodes",
}

SCHEMA_INDEX_MANIFEST = "carla_air_dataset_index_manifest_v1"
SCHEMA_SCENE_SAMPLE_INDEX_MANIFEST = "carla_air_scene_sample_index_manifest_v1"
SCHEMA_INDEX_STRICT_CONTRACT = "carla_air_dataset_index_strict_contract_v1"
SUMMARY_SAMPLE_ID_PREVIEW_LIMIT = 12

MIN_REQUIRED_SAMPLE_KEYS = [
    "schema_version",
    "sample_id",
    "identity_id",
    "trajectory_id",
    "node_id",
    "camera_id",
    "split",
    "rgb",
    "pose",
    "calib",
    "mask_gt",
    "plan_alignment",
]


def _repo_or_abs(raw: str | Path) -> Path:
    path = Path(str(raw))
    return path if path.is_absolute() else REPO_ROOT / path


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _collect_id_set(items: Any, key: str) -> set[str]:
    ids: set[str] = set()
    for item in _as_list(items):
        if isinstance(item, dict):
            token = str(item.get(key) or "").strip()
        else:
            token = str(item or "").strip()
        if token:
            ids.add(token)
    return ids


def _preview_items(values: list[str], limit: int = SUMMARY_SAMPLE_ID_PREVIEW_LIMIT) -> dict[str, Any]:
    return {
        "count": len(values),
        "preview": values[:limit],
        "preview_limit": limit,
        "truncated": len(values) > limit,
    }


def _load_json(path: Path) -> tuple[dict[str, Any], str | None]:
    if not path.is_file():
        return {}, "missing_file"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {}, f"invalid_json:{type(exc).__name__}"
    if not isinstance(payload, dict):
        return {}, "json_root_not_object"
    return payload, None


def _load_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    if not path.is_file():
        failures.append({"code": "samples_jsonl_missing", "detail": "dataset_samples.jsonl is missing."})
        return rows, failures
    for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        text = line.strip()
        if not text:
            continue
        try:
            item = json.loads(text)
        except Exception as exc:
            failures.append(
                {
                    "code": "samples_jsonl_line_invalid_json",
                    "detail": "dataset_samples.jsonl contains invalid JSON line.",
                    "line": idx,
                    "error": f"{type(exc).__name__}",
                }
            )
            continue
        if not isinstance(item, dict):
            failures.append(
                {
                    "code": "samples_jsonl_line_not_object",
                    "detail": "dataset_samples.jsonl line is not a JSON object.",
                    "line": idx,
                }
            )
            continue
        rows.append(item)
    return rows, failures


def _load_object_jsonl(path: Path, missing_code: str, invalid_prefix: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    if not path.is_file():
        failures.append({"code": missing_code, "detail": f"{path.name} is missing."})
        return rows, failures
    for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        text = line.strip()
        if not text:
            continue
        try:
            item = json.loads(text)
        except Exception as exc:
            failures.append(
                {
                    "code": f"{invalid_prefix}_line_invalid_json",
                    "detail": f"{path.name} contains invalid JSON line.",
                    "line": idx,
                    "error": f"{type(exc).__name__}",
                }
            )
            continue
        if not isinstance(item, dict):
            failures.append(
                {
                    "code": f"{invalid_prefix}_line_not_object",
                    "detail": f"{path.name} line is not a JSON object.",
                    "line": idx,
                }
            )
            continue
        rows.append(item)
    return rows, failures


def _add_issue(items: list[dict[str, Any]], code: str, detail: str, **extra: Any) -> None:
    issue: dict[str, Any] = {"code": code, "detail": detail}
    issue.update(extra)
    items.append(issue)


def _to_int(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def _hash_text_parts(parts: list[str]) -> str:
    hasher = hashlib.sha256()
    for part in parts:
        hasher.update(part.encode("utf-8"))
        hasher.update(b"\n")
    return hasher.hexdigest()


def _canonical_json_sha256(payload: Any) -> str:
    canonical = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _normalize_split_policy_summary(value: Any) -> dict[str, Any]:
    obj = _as_dict(value)
    split_names = sorted(str(name).strip() for name in _as_list(obj.get("split_names")) if str(name).strip())
    return {
        "split_strategy": str(obj.get("split_strategy") or "").strip() or "deployment_oriented_node_layout_v1",
        "not_random_frame_split": bool(obj.get("not_random_frame_split", True)),
        "split_names": split_names,
        "split_count": len(split_names),
    }


def _compute_split_policy_summary_from_splits_payload(splits_payload: dict[str, Any]) -> dict[str, Any]:
    split_map = _as_dict(splits_payload.get("splits"))
    split_names = sorted(str(name).strip() for name in split_map.keys() if str(name).strip())
    return {
        "split_strategy": str(splits_payload.get("split_strategy") or "").strip() or "deployment_oriented_node_layout_v1",
        "not_random_frame_split": bool(splits_payload.get("not_random_frame_split", True)),
        "split_names": split_names,
        "split_count": len(split_names),
    }


def _normalize_dataset_run_contract_summary(value: Any) -> dict[str, Any]:
    obj = _as_dict(value)
    split_distribution = {
        str(k).strip(): _to_int(v)
        for k, v in _as_dict(obj.get("split_distribution")).items()
        if str(k).strip()
    }
    sidecar_missing = {
        key: _to_int(_as_dict(obj.get("sidecar_missing_count_by_modality")).get(key))
        for key in ("rgb", "depth", "semantic", "instance", "pose", "calib")
    }
    planned_ids = sorted({str(x).strip() for x in _as_list(obj.get("planned_identity_ids")) if str(x).strip()})
    observed_ids = sorted({str(x).strip() for x in _as_list(obj.get("observed_identity_ids")) if str(x).strip()})
    normalized = {
        "schema_version": str(obj.get("schema_version") or "").strip(),
        "sample_count": _to_int(obj.get("sample_count")),
        "scene_count": _to_int(obj.get("scene_count")),
        "split_distribution": split_distribution,
        "capture_task_count": _to_int(obj.get("capture_task_count")),
        "sample_with_capture_task_count": _to_int(obj.get("sample_with_capture_task_count")),
        "sample_without_capture_task_count": _to_int(obj.get("sample_without_capture_task_count")),
        "strict_matrix_entry_sample_count": _to_int(obj.get("strict_matrix_entry_sample_count")),
        "legacy_or_observed_scene_passthrough_count": _to_int(obj.get("legacy_or_observed_scene_passthrough_count")),
        "mask_gt_available_count": _to_int(obj.get("mask_gt_available_count")),
        "no_mask_sample_count": _to_int(obj.get("no_mask_sample_count")),
        "sidecar_complete_count": _to_int(obj.get("sidecar_complete_count")),
        "sidecar_complete_fraction": float(obj.get("sidecar_complete_fraction") or 0.0),
        "sidecar_missing_count_by_modality": sidecar_missing,
        "planned_identity_ids": planned_ids,
        "observed_identity_ids": observed_ids,
        "identity_mismatch_count": _to_int(obj.get("identity_mismatch_count")),
        "strict_planned_identity_sample_count": _to_int(obj.get("strict_planned_identity_sample_count")),
        "observed_passthrough_identity_sample_count": _to_int(obj.get("observed_passthrough_identity_sample_count")),
        "starts_runtime": obj.get("starts_runtime"),
        "writes_scene_outputs": obj.get("writes_scene_outputs"),
        "non_promotion": obj.get("non_promotion"),
        "full_v1_live_dataset_ready": obj.get("full_v1_live_dataset_ready"),
    }
    for bridge_key in (
        "sample_with_trajectory_node_camera_bridge_count",
        "sample_without_trajectory_node_camera_bridge_count",
        "planned_capture_task_candidate_reference_count",
    ):
        if bridge_key in obj:
            normalized[bridge_key] = _to_int(obj.get(bridge_key))
    return normalized


def _normalize_dataset_run_closure_manifest(value: Any) -> dict[str, Any]:
    obj = _as_dict(value)
    split_distribution = {
        str(k).strip(): _to_int(v)
        for k, v in _as_dict(obj.get("split_distribution")).items()
        if str(k).strip()
    }
    return {
        "schema_version": str(obj.get("schema_version") or "").strip(),
        "run_id": str(obj.get("run_id") or "").strip(),
        "run_dir": str(obj.get("run_dir") or "").strip(),
        "sample_count": _to_int(obj.get("sample_count")),
        "scene_count": _to_int(obj.get("scene_count")),
        "split_distribution": split_distribution,
        "capture_task_count": _to_int(obj.get("capture_task_count")),
        "capture_queue_item_count": _to_int(obj.get("capture_queue_item_count")),
        "blocked_capture_queue_item_count": _to_int(obj.get("blocked_capture_queue_item_count")),
        "scene_discovery_accepted_scene_root_count": _to_int(obj.get("scene_discovery_accepted_scene_root_count")),
        "scene_sample_index_hash": str(obj.get("scene_sample_index_hash") or "").strip(),
        "scene_split_membership_hash": str(obj.get("scene_split_membership_hash") or "").strip(),
        "scene_keys_sorted_hash": str(obj.get("scene_keys_sorted_hash") or "").strip(),
        "artifact_manifest_entry_count_excluding_self": _to_int(obj.get("artifact_manifest_entry_count_excluding_self")),
        "contract_artifact_count_including_self": _to_int(obj.get("contract_artifact_count_including_self")),
        "count_gap_explained_as_self_reference": obj.get("count_gap_explained_as_self_reference") is True,
        "mask_gt_available_count": _to_int(obj.get("mask_gt_available_count")),
        "no_mask_sample_count": _to_int(obj.get("no_mask_sample_count")),
        "identity_mismatch_count": _to_int(obj.get("identity_mismatch_count")),
        "observed_identity_ids": sorted(
            {str(x).strip() for x in _as_list(obj.get("observed_identity_ids")) if str(x).strip()}
        ),
        "planned_identity_ids": sorted(
            {str(x).strip() for x in _as_list(obj.get("planned_identity_ids")) if str(x).strip()}
        ),
        "starts_runtime": obj.get("starts_runtime"),
        "writes_scene_outputs": obj.get("writes_scene_outputs"),
        "non_promotion": obj.get("non_promotion"),
        "full_v1_live_dataset_ready": obj.get("full_v1_live_dataset_ready"),
    }


def _normalize_capture_queue_manifest(value: Any) -> dict[str, Any]:
    obj = _as_dict(value)
    state_counts = {
        str(k).strip(): _to_int(v)
        for k, v in _as_dict(obj.get("state_counts")).items()
        if str(k).strip()
    }
    block_reason_counts = {
        str(k).strip(): _to_int(v)
        for k, v in _as_dict(obj.get("block_reason_counts")).items()
        if str(k).strip()
    }
    return {
        "schema_version": str(obj.get("schema_version") or "").strip(),
        "run_id": str(obj.get("run_id") or "").strip(),
        "capture_queue_item_count": _to_int(obj.get("capture_queue_item_count")),
        "blocked_capture_queue_item_count": _to_int(obj.get("blocked_capture_queue_item_count")),
        "queued_capture_queue_item_count": _to_int(obj.get("queued_capture_queue_item_count")),
        "state_counts": state_counts,
        "block_reason_counts": block_reason_counts,
        "capture_task_id_order_sha256": str(obj.get("capture_task_id_order_sha256") or "").strip(),
        "expected_scene_root_order_sha256": str(obj.get("expected_scene_root_order_sha256") or "").strip(),
        "identity_ids": sorted({str(x).strip() for x in _as_list(obj.get("identity_ids")) if str(x).strip()}),
        "trajectory_ids": sorted({str(x).strip() for x in _as_list(obj.get("trajectory_ids")) if str(x).strip()}),
        "node_ids": sorted({str(x).strip() for x in _as_list(obj.get("node_ids")) if str(x).strip()}),
        "camera_ids": sorted({str(x).strip() for x in _as_list(obj.get("camera_ids")) if str(x).strip()}),
        "scene_group_count": _to_int(obj.get("scene_group_count")),
        "starts_runtime": obj.get("starts_runtime"),
        "writes_scene_outputs": obj.get("writes_scene_outputs"),
        "non_promotion": obj.get("non_promotion"),
        "full_v1_live_dataset_ready": obj.get("full_v1_live_dataset_ready"),
        "source_capture_queue_path": str(obj.get("source_capture_queue_path") or "").strip(),
    }


def _compute_capture_queue_manifest_from_rows(
    *, run_id: str, capture_queue_rows: list[dict[str, Any]], capture_queue_path: Path
) -> dict[str, Any]:
    state_counts: dict[str, int] = {}
    block_reason_counts: dict[str, int] = {}
    capture_task_id_rows: list[str] = []
    expected_scene_root_rows: list[str] = []
    identity_ids: set[str] = set()
    trajectory_ids: set[str] = set()
    node_ids: set[str] = set()
    camera_ids: set[str] = set()
    scene_groups: set[tuple[str, str, str]] = set()
    blocked_count = 0
    queued_count = 0

    for row in [_as_dict(item) for item in capture_queue_rows if isinstance(item, dict)]:
        state = str(row.get("state") or "").strip().lower()
        if state:
            state_counts[state] = _to_int(state_counts.get(state)) + 1
        if state == "blocked":
            blocked_count += 1
        if state == "queued":
            queued_count += 1
        block_reason = str(row.get("block_reason") or "").strip()
        if block_reason:
            block_reason_counts[block_reason] = _to_int(block_reason_counts.get(block_reason)) + 1

        capture_task_id_rows.append(str(row.get("capture_task_id") or "").strip())
        expected_scene_root_rows.append(str(row.get("expected_scene_root") or "").strip())

        identity_id = str(row.get("identity_id") or "").strip()
        trajectory_id = str(row.get("trajectory_id") or "").strip()
        node_id = str(row.get("node_id") or "").strip()
        camera_id = str(row.get("camera_id") or "").strip()
        if identity_id:
            identity_ids.add(identity_id)
        if trajectory_id:
            trajectory_ids.add(trajectory_id)
        if node_id:
            node_ids.add(node_id)
        if camera_id:
            camera_ids.add(camera_id)
        scene_groups.add((identity_id, trajectory_id, node_id))

    return {
        "schema_version": "carla_air_capture_queue_manifest_v1",
        "run_id": str(run_id or "").strip(),
        "capture_queue_item_count": len(capture_queue_rows),
        "blocked_capture_queue_item_count": blocked_count,
        "queued_capture_queue_item_count": queued_count,
        "state_counts": {k: _to_int(v) for k, v in sorted(state_counts.items())},
        "block_reason_counts": {k: _to_int(v) for k, v in sorted(block_reason_counts.items())},
        "capture_task_id_order_sha256": _canonical_json_sha256(capture_task_id_rows),
        "expected_scene_root_order_sha256": _canonical_json_sha256(expected_scene_root_rows),
        "identity_ids": sorted(identity_ids),
        "trajectory_ids": sorted(trajectory_ids),
        "node_ids": sorted(node_ids),
        "camera_ids": sorted(camera_ids),
        "scene_group_count": len(scene_groups),
        "starts_runtime": False,
        "writes_scene_outputs": False,
        "non_promotion": True,
        "full_v1_live_dataset_ready": False,
        "source_capture_queue_path": str(capture_queue_path.resolve()),
    }


def _count_samples_by_scene(samples: list[dict[str, Any]]) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    by_scene: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for sample in samples:
        source = _as_dict(sample.get("source"))
        scene_id = str(sample.get("scene_id") or source.get("scene_id") or "").strip()
        key = (
            str(sample.get("identity_id") or "unknown_identity").strip() or "unknown_identity",
            str(sample.get("trajectory_id") or "unknown_trajectory").strip() or "unknown_trajectory",
            str(sample.get("node_id") or "unknown_node").strip() or "unknown_node",
            scene_id,
        )
        rec = by_scene.setdefault(
            key,
            {
                "sample_count": 0,
                "camera_rows": {},
                "depth_count": 0,
                "semantic_count": 0,
                "instance_count": 0,
                "calib_count": 0,
            },
        )
        rec["sample_count"] += 1
        cam_id = str(sample.get("camera_id") or "").strip()
        if cam_id:
            camera_rows = _as_dict(rec.get("camera_rows"))
            camera_rows[cam_id] = _to_int(camera_rows.get(cam_id)) + 1
            rec["camera_rows"] = camera_rows
        if sample.get("depth") not in (None, ""):
            rec["depth_count"] += 1
        if sample.get("semantic") not in (None, ""):
            rec["semantic_count"] += 1
        if sample.get("instance") not in (None, ""):
            rec["instance_count"] += 1
        if sample.get("calib") not in (None, ""):
            rec["calib_count"] += 1
    return by_scene


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _jsonl_row_count(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def _schema_version_if_json(path: Path) -> str | None:
    if path.suffix.lower() != ".json":
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if isinstance(payload, dict):
        schema = payload.get("schema_version")
        if schema is not None:
            return str(schema)
    return None


def _safe_token(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def _expected_scene_id(identity_id: str, trajectory_id: str, node_id: str) -> str:
    return ".".join([str(identity_id).strip(), str(trajectory_id).strip(), str(node_id).strip()])


def _expected_scene_root(identity_id: str, trajectory_id: str, node_id: str) -> str:
    return f"data/carla_air/nodes/{node_id}/scenes/{_expected_scene_id(identity_id, trajectory_id, node_id)}"


def _scene_key_for_fields(identity_id: str, trajectory_id: str, node_id: str, scene_id: str, scene_dir: str) -> str:
    stable_scene = str(scene_id or "").strip() or str(scene_dir or "").strip()
    return "|".join(
        [
            str(identity_id or "").strip() or "unknown_identity",
            str(trajectory_id or "").strip() or "unknown_trajectory",
            str(node_id or "").strip() or "unknown_node",
            stable_scene,
        ]
    )


def _compute_sample_schema_coverage_summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    required_fields = [
        "sample_id",
        "scene_id",
        "scene_key",
        "identity_id",
        "trajectory_id",
        "node_id",
        "camera_id",
        "timestamp",
        "split",
        "rgb",
        "pose",
        "calib",
        "depth",
        "semantic",
        "instance",
        "mask_gt",
    ]
    field_present_count = {key: 0 for key in required_fields}
    field_missing_count = {key: 0 for key in required_fields}
    for sample in samples:
        source = _as_dict(sample.get("source"))
        derived_scene_key = _scene_key_for_fields(
            str(sample.get("identity_id") or "").strip(),
            str(sample.get("trajectory_id") or "").strip(),
            str(sample.get("node_id") or "").strip(),
            str(sample.get("scene_id") or source.get("scene_id") or "").strip(),
            str(source.get("scene_dir") or "").strip(),
        )
        for key in required_fields:
            present = sample.get(key) not in (None, "")
            if key in ("depth", "semantic", "instance"):
                present = key in sample
            if key == "scene_key":
                present = key in sample or bool(derived_scene_key)
            if present:
                field_present_count[key] += 1
            else:
                field_missing_count[key] += 1
    return {
        "schema_version": "carla_air_dataset_sample_schema_coverage_summary_v1",
        "sample_count": len(samples),
        "required_fields": required_fields,
        "field_present_count": field_present_count,
        "field_missing_count": field_missing_count,
        "field_presence_required_even_when_sidecar_unavailable": {
            "depth": True,
            "semantic": True,
            "instance": True,
        },
        "sidecar_unavailable_is_reference_or_availability_not_mask_gt": True,
        "candidate_proxy_pseudo_legacy_not_promoted_to_mask_gt": True,
    }


def _build_sample_schema_coverage_manifest_from_summary(
    *, run_id: str, sample_schema_coverage_summary: dict[str, Any]
) -> dict[str, Any]:
    core_payload = {
        "schema_version": "carla_air_sample_schema_coverage_manifest_v1",
        "run_id": str(run_id or "").strip(),
        "sample_count": _to_int(sample_schema_coverage_summary.get("sample_count")),
        "required_fields": list(_as_list(sample_schema_coverage_summary.get("required_fields"))),
        "field_present_count": {
            str(k): _to_int(v)
            for k, v in _as_dict(sample_schema_coverage_summary.get("field_present_count")).items()
            if str(k).strip()
        },
        "field_missing_count": {
            str(k): _to_int(v)
            for k, v in _as_dict(sample_schema_coverage_summary.get("field_missing_count")).items()
            if str(k).strip()
        },
        "field_presence_required_even_when_sidecar_unavailable": _as_dict(
            sample_schema_coverage_summary.get("field_presence_required_even_when_sidecar_unavailable")
        ),
        "sidecar_unavailable_is_reference_or_availability_not_mask_gt": True,
        "candidate_proxy_pseudo_legacy_not_promoted_to_mask_gt": True,
        "starts_runtime": False,
        "writes_scene_outputs": False,
        "non_promotion": True,
        "full_v1_live_dataset_ready": False,
    }
    return {
        **core_payload,
        "stable_hashes": {
            "canonical_payload_sha256": _canonical_json_sha256(core_payload),
        },
    }


def _sidecar_quality_record(samples: list[dict[str, Any]]) -> dict[str, Any]:
    modalities = ["rgb", "depth", "semantic", "instance", "pose", "calib"]
    sample_count = len(samples)
    present = {key: 0 for key in modalities}
    missing = {key: 0 for key in modalities}
    missing_sample_ids = {key: [] for key in modalities}
    complete_count = 0
    for sample in samples:
        refs = _as_dict(sample.get("refs"))
        sample_complete = True
        sample_id = str(sample.get("sample_id") or "").strip()
        for key in modalities:
            if refs.get(key):
                present[key] += 1
            else:
                missing[key] += 1
                if sample_id:
                    missing_sample_ids[key].append(sample_id)
                sample_complete = False
        if sample_complete:
            complete_count += 1
    return {
        "sample_count": sample_count,
        "complete_rgb_depth_semantic_instance_pose_calib_count": complete_count,
        "complete_fraction": (complete_count / sample_count) if sample_count else 0.0,
        "present_count_by_modality": present,
        "missing_count_by_modality": missing,
        "missing_sample_id_preview_by_modality": {
            key: _preview_items(values)
            for key, values in missing_sample_ids.items()
            if values
        },
    }


def _compute_sidecar_quality_matrix(samples: list[dict[str, Any]]) -> dict[str, Any]:
    by_scene: dict[str, list[dict[str, Any]]] = {}
    by_split: dict[str, list[dict[str, Any]]] = {}
    by_scene_split: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for sample in samples:
        source = _as_dict(sample.get("source"))
        scene_key = str(sample.get("scene_key") or "").strip()
        if not scene_key:
            scene_key = _scene_key_for_fields(
                str(sample.get("identity_id") or "").strip(),
                str(sample.get("trajectory_id") or "").strip(),
                str(sample.get("node_id") or "").strip(),
                str(sample.get("scene_id") or source.get("scene_id") or "").strip(),
                str(source.get("scene_dir") or "").strip(),
            )
        split_name = str(sample.get("split") or "unknown").strip() or "unknown"
        by_scene.setdefault(scene_key or "unknown_scene", []).append(sample)
        by_split.setdefault(split_name, []).append(sample)
        by_scene_split.setdefault((scene_key or "unknown_scene", split_name), []).append(sample)

    scene_entries: list[dict[str, Any]] = []
    for scene_key in sorted(by_scene):
        rows = by_scene[scene_key]
        first = _as_dict(rows[0]) if rows else {}
        source = _as_dict(first.get("source"))
        scene_entries.append(
            {
                "scene_key": scene_key,
                "scene_id": str(first.get("scene_id") or source.get("scene_id") or "").strip() or None,
                "identity_id": str(first.get("identity_id") or "").strip() or None,
                "trajectory_id": str(first.get("trajectory_id") or "").strip() or None,
                "node_id": str(first.get("node_id") or "").strip() or None,
                "split_names": sorted({str(sample.get("split") or "unknown").strip() or "unknown" for sample in rows}),
                "quality": _sidecar_quality_record(rows),
            }
        )

    split_entries = [
        {
            "split": split_name,
            "quality": _sidecar_quality_record(by_split[split_name]),
        }
        for split_name in sorted(by_split)
    ]

    scene_split_entries: list[dict[str, Any]] = []
    for scene_key, split_name in sorted(by_scene_split):
        rows = by_scene_split[(scene_key, split_name)]
        first = _as_dict(rows[0]) if rows else {}
        source = _as_dict(first.get("source"))
        scene_split_entries.append(
            {
                "scene_key": scene_key,
                "scene_id": str(first.get("scene_id") or source.get("scene_id") or "").strip() or None,
                "split": split_name,
                "quality": _sidecar_quality_record(rows),
            }
        )

    return {
        "schema_version": "carla_air_sidecar_quality_matrix_v1",
        "sample_count": len(samples),
        "modalities": ["rgb", "depth", "semantic", "instance", "pose", "calib"],
        "scene_count": len(scene_entries),
        "split_count": len(split_entries),
        "scene_split_count": len(scene_split_entries),
        "overall": _sidecar_quality_record(samples),
        "by_scene": scene_entries,
        "by_split": split_entries,
        "by_scene_split": scene_split_entries,
        "sidecar_unavailable_is_reference_or_availability_not_mask_gt": True,
        "non_promotion": True,
        "full_v1_live_dataset_ready": False,
    }


def _compute_scene_membership_alignment(
    run_id: str,
    scene_output_manifest_payload: dict[str, Any],
    scene_membership_manifest_payload: dict[str, Any],
) -> dict[str, Any]:
    scene_outputs = [item for item in _as_list(scene_output_manifest_payload.get("scene_outputs")) if isinstance(item, dict)]
    scene_entries = [item for item in _as_list(scene_membership_manifest_payload.get("scene_entries")) if isinstance(item, dict)]

    observed_by_tn: dict[tuple[str, str], list[dict[str, Any]]] = {}
    observed_by_scene: dict[str, list[dict[str, Any]]] = {}
    observed_passthrough_count = 0
    for item in scene_entries:
        trajectory_id = str(item.get("trajectory_id") or "").strip()
        node_id = str(item.get("node_id") or "").strip()
        scene_id = str(item.get("scene_id") or "").strip()
        if trajectory_id and node_id:
            observed_by_tn.setdefault((trajectory_id, node_id), []).append(item)
        if scene_id:
            observed_by_scene.setdefault(scene_id, []).append(item)
        alignment = _as_dict(item.get("capture_task_alignment"))
        if _to_int(alignment.get("identity_passthrough_mismatch_count")) > 0:
            observed_passthrough_count += 1

    def _pick_observation(planned: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
        planned_trajectory_id = str(planned.get("trajectory_id") or "").strip()
        planned_node_id = str(planned.get("node_id") or "").strip()
        planned_scene_id = str(planned.get("scene_id") or "").strip()
        planned_identity_id = str(planned.get("identity_id") or "").strip()
        candidates: list[dict[str, Any]] = []
        join_mode = "unmatched"
        if planned_trajectory_id and planned_node_id:
            candidates = list(observed_by_tn.get((planned_trajectory_id, planned_node_id), []))
            if candidates:
                join_mode = "trajectory_node"
        if not candidates and planned_scene_id:
            candidates = list(observed_by_scene.get(planned_scene_id, []))
            if candidates:
                join_mode = "scene_id"
        if not candidates:
            return None, join_mode

        def _sort_key(obs: dict[str, Any]) -> tuple[int, int, str]:
            observed_identity_id = str(obs.get("identity_id") or "").strip()
            exact_identity = 1 if (planned_identity_id and observed_identity_id == planned_identity_id) else 0
            sample_count = _to_int(obs.get("sample_count"))
            obs_scene_id = str(obs.get("scene_id") or "").strip()
            return (exact_identity, sample_count, obs_scene_id)

        return sorted(candidates, key=_sort_key, reverse=True)[0], join_mode

    rows: list[dict[str, Any]] = []
    planned_blocked_count = 0
    trajectory_node_match_count = 0
    exact_identity_match_count = 0
    identity_mismatch_count = 0
    missing_observation_count = 0

    for item in scene_outputs:
        planned_identity_id = str(item.get("identity_id") or "").strip()
        planned_scene_id = str(item.get("scene_id") or "").strip()
        trajectory_id = str(item.get("trajectory_id") or "").strip()
        node_id = str(item.get("node_id") or "").strip()
        scene_output_state = str(item.get("state") or "").strip() or "unknown"
        if scene_output_state == "blocked":
            planned_blocked_count += 1

        observed, join_mode = _pick_observation(item)
        observed_scene_id = None
        observed_scene_root = None
        observed_identity_id = None
        membership_sample_count = 0
        membership_split = None
        membership_match_status = "missing_observation"
        if join_mode == "trajectory_node" and observed is not None:
            trajectory_node_match_count += 1

        if observed is None:
            missing_observation_count += 1
        else:
            observed_scene_id = str(observed.get("scene_id") or "").strip() or None
            observed_scene_root = str(observed.get("scene_root") or observed.get("scene_dir") or "").strip() or None
            observed_identity_id = str(observed.get("identity_id") or "").strip() or None
            membership_sample_count = _to_int(observed.get("sample_count"))
            split_value = str(observed.get("split") or "").strip()
            if split_value:
                membership_split = split_value
            else:
                split_names = [str(x).strip() for x in _as_list(observed.get("split_names")) if str(x).strip()]
                membership_split = ",".join(sorted(set(split_names))) if split_names else None
            if observed_identity_id and planned_identity_id and observed_identity_id == planned_identity_id:
                membership_match_status = "exact_identity_match"
                exact_identity_match_count += 1
            else:
                membership_match_status = (
                    "observed_default_airsim_drone_non_promotion_passthrough"
                    if observed_identity_id == "default_airsim_drone"
                    else "identity_mismatch_non_promotion_passthrough"
                )
                identity_mismatch_count += 1

        rows.append(
            {
                "join_key": f"{join_mode}:{trajectory_id}:{node_id}:{planned_scene_id}",
                "trajectory_id": trajectory_id or None,
                "node_id": node_id or None,
                "planned_identity_id": planned_identity_id or None,
                "planned_scene_id": planned_scene_id or None,
                "expected_scene_root": str(item.get("expected_scene_root") or "").strip() or None,
                "scene_output_state": scene_output_state,
                "requires_ue_carla_import_readback": bool(item.get("requires_ue_carla_import_readback")),
                "observed_scene_id": observed_scene_id,
                "observed_scene_root": observed_scene_root,
                "observed_identity_id": observed_identity_id,
                "membership_sample_count": membership_sample_count,
                "membership_split": membership_split,
                "membership_match_status": membership_match_status,
                "non_promotion": True,
                "full_v1_live_dataset_ready": False,
            }
        )

    summary = {
        "planned_scene_output_count": len(scene_outputs),
        "observed_scene_membership_count": len(scene_entries),
        "planned_blocked_count": planned_blocked_count,
        "observed_passthrough_count": observed_passthrough_count,
        "trajectory_node_match_count": trajectory_node_match_count,
        "exact_identity_match_count": exact_identity_match_count,
        "identity_mismatch_count": identity_mismatch_count,
        "missing_observation_count": missing_observation_count,
    }
    return {
        "schema_version": "carla_air_scene_membership_alignment_manifest_v1",
        "run_id": str(run_id or "").strip(),
        "starts_runtime": False,
        "writes_scene_outputs": False,
        "non_promotion": True,
        "full_v1_live_dataset_ready": False,
        "summary": summary,
        "rows": rows,
    }


def _compute_capture_matrix_alignment(
    run_id: str,
    capture_queue_rows: list[dict[str, Any]],
    samples: list[dict[str, Any]],
) -> dict[str, Any]:
    observed_by_tnc: dict[tuple[str, str, str], dict[str, Any]] = {}
    observed_sample_count = 0
    mask_gt_available_count = 0
    no_mask_sample_count = 0
    for sample in samples:
        sample_obj = _as_dict(sample)
        observed_sample_count += 1
        trajectory_id = str(sample_obj.get("trajectory_id") or "").strip()
        node_id = str(sample_obj.get("node_id") or "").strip()
        camera_id = str(sample_obj.get("camera_id") or "").strip()
        key = (trajectory_id, node_id, camera_id)
        rec = observed_by_tnc.setdefault(
            key,
            {
                "sample_count": 0,
                "identity_ids": set(),
                "scene_ids": set(),
                "split_names": set(),
                "mask_gt_available_count": 0,
                "no_mask_sample_count": 0,
            },
        )
        rec["sample_count"] += 1
        identity_id = str(sample_obj.get("identity_id") or "").strip()
        if identity_id:
            rec["identity_ids"].add(identity_id)
        source = _as_dict(sample_obj.get("source"))
        scene_id = str(sample_obj.get("scene_id") or source.get("scene_id") or "").strip()
        if scene_id:
            rec["scene_ids"].add(scene_id)
        split_name = str(sample_obj.get("split") or "").strip()
        if split_name:
            rec["split_names"].add(split_name)
        mask_gt = _as_dict(sample_obj.get("mask_gt"))
        if str(mask_gt.get("availability") or "").strip() == "available":
            rec["mask_gt_available_count"] += 1
            mask_gt_available_count += 1
        else:
            rec["no_mask_sample_count"] += 1
            no_mask_sample_count += 1

    rows: list[dict[str, Any]] = []
    exact_capture_task_sample_count = 0
    trajectory_node_camera_match_task_count = 0
    identity_mismatch_task_count = 0
    missing_observation_task_count = 0
    blocked_capture_task_count = 0
    planned_rows = [_as_dict(row) for row in capture_queue_rows if isinstance(row, dict)]
    for item in planned_rows:
        capture_task_id = str(item.get("capture_task_id") or "").strip()
        planned_identity_id = str(item.get("identity_id") or "").strip()
        trajectory_id = str(item.get("trajectory_id") or "").strip()
        node_id = str(item.get("node_id") or "").strip()
        camera_id = str(item.get("camera_id") or "").strip()
        state = str(item.get("state") or "").strip() or "unknown"
        if state == "blocked":
            blocked_capture_task_count += 1
        rec = observed_by_tnc.get((trajectory_id, node_id, camera_id), {})
        observed_sample_count_for_task = _to_int(rec.get("sample_count"))
        observed_identity_ids = sorted(str(x) for x in rec.get("identity_ids", set()) if str(x))
        observed_scene_ids = sorted(str(x) for x in rec.get("scene_ids", set()) if str(x))
        observed_split_names = sorted(str(x) for x in rec.get("split_names", set()) if str(x))
        observed_mask_gt_available_count = _to_int(rec.get("mask_gt_available_count"))
        observed_no_mask_sample_count = _to_int(rec.get("no_mask_sample_count"))

        if observed_sample_count_for_task <= 0:
            match_status = "missing_observation"
            missing_observation_task_count += 1
        elif planned_identity_id in observed_identity_ids:
            match_status = "exact_match"
            trajectory_node_camera_match_task_count += 1
            exact_capture_task_sample_count += observed_sample_count_for_task
        else:
            match_status = "observed_scene_passthrough_identity_mismatch"
            trajectory_node_camera_match_task_count += 1
            identity_mismatch_task_count += 1

        rows.append(
            {
                "capture_task_id": capture_task_id,
                "planned_identity_id": planned_identity_id or None,
                "identity_id": planned_identity_id or None,
                "trajectory_id": trajectory_id or None,
                "node_id": node_id or None,
                "camera_id": camera_id or None,
                "planned_state": state,
                "state": state,
                "capture_allowed_now": item.get("capture_allowed_now") is True,
                "requires_ue_carla_import_readback": bool(item.get("requires_ue_carla_import_readback")),
                "observed_sample_count": observed_sample_count_for_task,
                "observed_identity_ids": observed_identity_ids,
                "observed_scene_ids": observed_scene_ids,
                "observed_split_names": observed_split_names,
                "observed_mask_gt_available_count": observed_mask_gt_available_count,
                "observed_no_mask_sample_count": observed_no_mask_sample_count,
                "match_status": match_status,
                "non_promotion": True,
                "full_v1_live_dataset_ready": False,
            }
        )

    return {
        "schema_version": "carla_air_capture_matrix_alignment_manifest_v1",
        "run_id": str(run_id or "").strip(),
        "starts_runtime": False,
        "writes_scene_outputs": False,
        "non_promotion": True,
        "full_v1_live_dataset_ready": False,
        "summary": {
            "planned_capture_task_count": len(planned_rows),
            "observed_camera_group_count": len(observed_by_tnc),
            "observed_sample_count": observed_sample_count,
            "exact_capture_task_sample_count": exact_capture_task_sample_count,
            "trajectory_node_camera_match_task_count": trajectory_node_camera_match_task_count,
            "identity_mismatch_task_count": identity_mismatch_task_count,
            "missing_observation_task_count": missing_observation_task_count,
            "blocked_capture_task_count": blocked_capture_task_count,
            "mask_gt_available_count": mask_gt_available_count,
            "no_mask_sample_count": no_mask_sample_count,
        },
        "rows": rows,
    }


def _episode_filter_sets(episode: dict[str, Any]) -> dict[str, set[str]]:
    split_value = str(episode.get("split") or "").strip()
    node_ids = {str(v).strip() for v in _as_list(episode.get("node_ids")) if str(v).strip()}
    node_id = str(episode.get("node_id") or episode.get("node") or "").strip()
    if node_id:
        node_ids.add(node_id)
    camera_ids = {str(v).strip() for v in _as_list(episode.get("camera_ids")) if str(v).strip()}
    trajectory_ids = {str(v).strip() for v in _as_list(episode.get("trajectory_ids")) if str(v).strip()}
    return {
        "splits": {split_value} if split_value else set(),
        "node_ids": node_ids,
        "camera_ids": camera_ids,
        "trajectory_ids": trajectory_ids,
    }


def _sample_matches_episode(sample: dict[str, Any], filters: dict[str, set[str]]) -> bool:
    split_token = str(sample.get("split") or "").strip()
    node_token = str(sample.get("node_id") or "").strip()
    camera_token = str(sample.get("camera_id") or "").strip()
    trajectory_token = str(sample.get("trajectory_id") or "").strip()
    if filters["splits"] and split_token not in filters["splits"]:
        return False
    if filters["node_ids"] and node_token not in filters["node_ids"]:
        return False
    if filters["camera_ids"] and camera_token not in filters["camera_ids"]:
        return False
    if filters["trajectory_ids"] and trajectory_token not in filters["trajectory_ids"]:
        return False
    return True


def _readiness_from_state(state: str, block_reasons: list[str]) -> dict[str, Any]:
    blocked = str(state).strip().lower() == "blocked"
    return {
        "status": "blocked" if blocked else "ready_for_execution",
        "blocked": blocked,
        "blocked_reasons": list(block_reasons) if blocked else [],
        "evidence_ready": not blocked,
    }


def _normalize_profile_ref(value: Any) -> dict[str, Any]:
    obj = _as_dict(value)
    return {
        "identity_model_profile_id": str(obj.get("identity_model_profile_id") or "").strip(),
        "identity_id": str(obj.get("identity_id") or "").strip(),
        "model_label": str(obj.get("model_label") or "").strip(),
        "capture_profile": str(obj.get("capture_profile") or "").strip(),
        "switch_method": str(obj.get("switch_method") or "").strip(),
        "requires_ue_carla_import_readback": obj.get("requires_ue_carla_import_readback") is True,
        "ready_for_capture": obj.get("ready_for_capture") is True,
        "non_promotion": obj.get("non_promotion") is True,
    }


def _validate_out_path(raw_out: str, allow_nonlocal_out: bool) -> Path:
    out_path = _repo_or_abs(raw_out)
    if out_path.suffix.lower() != ".json":
        raise SystemExit("--out must point to a .json file")
    if (not allow_nonlocal_out) and (not _is_under(out_path, LOCAL_ROOT)):
        raise SystemExit("--out must stay under repository local/ unless --allow-nonlocal-out is set")
    return out_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify offline dataset generation run artifacts for V1 contract consistency.")
    parser.add_argument("--run-dir", required=True, help="Run directory containing dataset artifacts (relative path allowed).")
    parser.add_argument("--out", default="", help="Optional JSON output path.")
    parser.add_argument("--allow-nonlocal-out", action="store_true")
    parser.add_argument("--require-samples", action="store_true")
    parser.add_argument("--require-run-contract", action="store_true")
    parser.add_argument("--allow-fail", action="store_true", help="Exit 0 even if failures exist.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_dir = _repo_or_abs(args.run_dir)

    failures: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    artifact_presence: dict[str, bool] = {}
    for filename, key in REQUIRED_ARTIFACTS.items():
        artifact_presence[key] = (run_dir / filename).is_file()
    run_contract_path = run_dir / "run_contract.json"
    run_summary_path = run_dir / "run_summary.json"
    capture_queue_path = run_dir / "capture_queue.jsonl"
    scene_discovery_manifest_path = run_dir / "scene_discovery_manifest.json"
    scene_output_manifest_path = run_dir / "scene_output_manifest.json"
    batch_run_manifest_path = run_dir / "batch_run_manifest.json"
    capture_queue_manifest_path = run_dir / "capture_queue_manifest.json"
    identity_model_switch_manifest_path = run_dir / "identity_model_switch_manifest.json"
    existing_scene_index_bridge_manifest_path = run_dir / "existing_scene_index_bridge_manifest.json"
    sidecar_quality_manifest_path = run_dir / "sidecar_quality_manifest.json"
    dataset_gap_manifest_path = run_dir / "dataset_gap_manifest.json"
    no_mask_non_promotion_manifest_path = run_dir / "no_mask_non_promotion_manifest.json"
    deployment_episode_visibility_manifest_path = run_dir / "deployment_episode_visibility_manifest.json"
    scene_membership_manifest_path = run_dir / "scene_membership_manifest.json"
    scene_membership_alignment_manifest_path = run_dir / "scene_membership_alignment_manifest.json"
    capture_matrix_alignment_manifest_path = run_dir / "capture_matrix_alignment_manifest.json"
    dataset_run_closure_manifest_path = run_dir / "dataset_run_closure_manifest.json"
    scene_sample_index_manifest_path = run_dir / "scene_sample_index_manifest.json"
    sample_schema_coverage_manifest_path = run_dir / "sample_schema_coverage_manifest.json"
    artifact_presence["run_contract"] = run_contract_path.is_file()
    artifact_presence["run_summary"] = run_summary_path.is_file()
    artifact_presence["capture_queue"] = capture_queue_path.is_file()
    artifact_presence["scene_discovery_manifest"] = scene_discovery_manifest_path.is_file()
    artifact_presence["scene_output_manifest"] = scene_output_manifest_path.is_file()
    artifact_presence["batch_run_manifest"] = batch_run_manifest_path.is_file()
    artifact_presence["capture_queue_manifest"] = capture_queue_manifest_path.is_file()
    artifact_presence["identity_model_switch_manifest"] = identity_model_switch_manifest_path.is_file()
    artifact_presence["existing_scene_index_bridge_manifest"] = existing_scene_index_bridge_manifest_path.is_file()
    artifact_presence["sidecar_quality_manifest"] = sidecar_quality_manifest_path.is_file()
    artifact_presence["dataset_gap_manifest"] = dataset_gap_manifest_path.is_file()
    artifact_presence["no_mask_non_promotion_manifest"] = no_mask_non_promotion_manifest_path.is_file()
    artifact_presence["deployment_episode_visibility_manifest"] = deployment_episode_visibility_manifest_path.is_file()
    artifact_presence["scene_membership_manifest"] = scene_membership_manifest_path.is_file()
    artifact_presence["scene_membership_alignment_manifest"] = scene_membership_alignment_manifest_path.is_file()
    artifact_presence["capture_matrix_alignment_manifest"] = capture_matrix_alignment_manifest_path.is_file()
    artifact_presence["dataset_run_closure_manifest"] = dataset_run_closure_manifest_path.is_file()
    artifact_presence["scene_sample_index_manifest"] = scene_sample_index_manifest_path.is_file()
    artifact_presence["sample_schema_coverage_manifest"] = sample_schema_coverage_manifest_path.is_file()
    artifact_manifest_path = run_dir / "artifact_manifest.json"
    artifact_presence["artifact_manifest"] = artifact_manifest_path.is_file()
    dataset_index_manifest_path = run_dir / "dataset_index_manifest.json"
    artifact_presence["dataset_index_manifest"] = dataset_index_manifest_path.is_file()

    missing_required = [name for name, key in REQUIRED_ARTIFACTS.items() if not artifact_presence[key]]
    if missing_required:
        _add_issue(
            failures,
            "required_artifacts_missing",
            "Required dataset artifacts are missing.",
            missing_artifacts=missing_required,
        )

    if args.require_run_contract and (not artifact_presence["run_contract"]):
        _add_issue(
            failures,
            "run_contract_required_missing",
            "--require-run-contract set but run_contract.json is missing.",
        )

    plan_payload, plan_err = _load_json(run_dir / "dataset_plan.json")
    if plan_err:
        _add_issue(failures, "dataset_plan_invalid", "dataset_plan.json is missing or invalid.", error=plan_err)

    manifest_payload, manifest_err = _load_json(run_dir / "dataset_manifest.json")
    if manifest_err:
        _add_issue(failures, "dataset_manifest_invalid", "dataset_manifest.json is missing or invalid.", error=manifest_err)

    splits_payload, splits_err = _load_json(run_dir / "dataset_splits.json")
    if splits_err:
        _add_issue(failures, "dataset_splits_invalid", "dataset_splits.json is missing or invalid.", error=splits_err)
    split_policy_summary_computed = (
        _compute_split_policy_summary_from_splits_payload(splits_payload) if splits_payload else {}
    )
    split_policy_digest_computed = (
        _canonical_json_sha256(split_policy_summary_computed) if split_policy_summary_computed else None
    )

    episodes_payload, episodes_err = _load_json(run_dir / "deployment_episodes.json")
    if episodes_err:
        _add_issue(
            failures,
            "deployment_episodes_invalid",
            "deployment_episodes.json is missing or invalid.",
            error=episodes_err,
        )

    if episodes_payload:
        if episodes_payload.get("schema_version") != "carla_air_deployment_episodes_v1":
            _add_issue(
                failures,
                "deployment_episodes_schema_mismatch",
                "deployment_episodes schema_version mismatch.",
                got=episodes_payload.get("schema_version"),
                expected="carla_air_deployment_episodes_v1",
            )
        for ep_idx, ep in enumerate(_as_list(episodes_payload.get("episodes")), start=1):
            ep_obj = _as_dict(ep)
            camera_ids = [str(x).strip() for x in _as_list(ep_obj.get("camera_ids")) if str(x).strip()]
            camera_ids_by_node = _as_dict(ep_obj.get("camera_ids_by_node"))
            if camera_ids and len(camera_ids) != len(set(camera_ids)):
                _add_issue(
                    failures,
                    "deployment_episode_camera_ids_not_unique",
                    "deployment_episodes[].camera_ids must be de-duplicated.",
                    episode_index=ep_idx,
                    episode_id=ep_obj.get("episode_id"),
                )
            union_camera_ids: list[str] = []
            seen_union: set[str] = set()
            for node_cameras in camera_ids_by_node.values():
                for raw_cam in _as_list(node_cameras):
                    cam = str(raw_cam).strip()
                    if cam and cam not in seen_union:
                        union_camera_ids.append(cam)
                        seen_union.add(cam)
            if union_camera_ids and camera_ids and (camera_ids != union_camera_ids):
                _add_issue(
                    failures,
                    "deployment_episode_camera_ids_union_mismatch",
                    "deployment_episodes[].camera_ids must equal flattened union of camera_ids_by_node when both are present.",
                    episode_index=ep_idx,
                    episode_id=ep_obj.get("episode_id"),
                    manifest_value=camera_ids,
                    computed_value=union_camera_ids,
                )

    samples, jsonl_failures = _load_jsonl(run_dir / "dataset_samples.jsonl")
    failures.extend(jsonl_failures)

    deployment_visibility_present_count = 0
    deployment_sample_count_total_from_episodes = 0
    deployment_scene_ids_union: set[str] = set()
    deployment_episode_with_visibility_gap_count = 0
    deployment_episode_without_samples_count = 0
    deployment_episode_scene_rows: list[str] = []
    deployment_episode_sample_order_rows: list[str] = []
    deployment_episode_sample_sorted_rows: list[str] = []
    deployment_episode_hash_strict_required = False
    if episodes_payload:
        for ep_idx, ep in enumerate(_as_list(episodes_payload.get("episodes")), start=1):
            ep_obj = _as_dict(ep)
            visibility = _as_dict(ep_obj.get("sample_scene_visibility"))
            if not visibility:
                continue
            deployment_visibility_present_count += 1
            filters = _episode_filter_sets(ep_obj)
            matched = [sample for sample in samples if _sample_matches_episode(sample, filters)]
            scene_ids_set = {
                str(_as_dict(sample.get("source")).get("scene_id") or sample.get("scene_id") or "").strip()
                for sample in matched
                if str(_as_dict(sample.get("source")).get("scene_id") or sample.get("scene_id") or "").strip()
            }
            scene_ids = sorted(scene_ids_set)
            sample_ids_ordered = [str(_as_dict(sample).get("sample_id") or "").strip() for sample in matched]
            sample_ids_ordered = [sid for sid in sample_ids_ordered if sid]
            sample_ids_sorted = sorted(set(sample_ids_ordered))
            if any(
                key in visibility
                for key in ("scene_ids_sorted_hash", "sample_id_order_hash", "sample_ids_sorted_hash")
            ):
                deployment_episode_hash_strict_required = True
            recomputed_sample_count = len(matched)
            recomputed_scene_count = len(scene_ids_set)
            deployment_sample_count_total_from_episodes += recomputed_sample_count
            deployment_scene_ids_union.update(scene_ids_set)
            if recomputed_sample_count == 0:
                deployment_episode_without_samples_count += 1
            coverage_gaps = _as_dict(visibility.get("coverage_gaps"))
            if coverage_gaps.get("has_missing_configured_filter_values") is True:
                deployment_episode_with_visibility_gap_count += 1
            if _to_int(visibility.get("sample_count")) != recomputed_sample_count:
                _add_issue(
                    failures,
                    "deployment_episode_visibility_sample_count_mismatch",
                    "deployment_episodes[].sample_scene_visibility.sample_count must match recomputed sample count.",
                    episode_index=ep_idx,
                    episode_id=ep_obj.get("episode_id"),
                    manifest_value=visibility.get("sample_count"),
                    computed_value=recomputed_sample_count,
                )
            if _to_int(visibility.get("scene_count")) != recomputed_scene_count:
                _add_issue(
                    failures,
                    "deployment_episode_visibility_scene_count_mismatch",
                    "deployment_episodes[].sample_scene_visibility.scene_count must match recomputed scene count.",
                    episode_index=ep_idx,
                    episode_id=ep_obj.get("episode_id"),
                    manifest_value=visibility.get("scene_count"),
                    computed_value=recomputed_scene_count,
                )
            if visibility.get("non_promotion") is not True:
                _add_issue(
                    failures,
                    "deployment_episode_visibility_non_promotion_mismatch",
                    "deployment_episodes[].sample_scene_visibility.non_promotion must be true.",
                    episode_index=ep_idx,
                    episode_id=ep_obj.get("episode_id"),
                    got=visibility.get("non_promotion"),
                )
            if visibility.get("full_v1_live_dataset_ready") is not False:
                _add_issue(
                    failures,
                    "deployment_episode_visibility_full_v1_live_dataset_ready_mismatch",
                    "deployment_episodes[].sample_scene_visibility.full_v1_live_dataset_ready must be false.",
                    episode_index=ep_idx,
                    episode_id=ep_obj.get("episode_id"),
                    got=visibility.get("full_v1_live_dataset_ready"),
                )
            mask_count = _to_int(visibility.get("mask_gt_available_count"))
            if mask_count != 0:
                _add_issue(
                    failures,
                    "deployment_episode_visibility_mask_gt_available_count_mismatch",
                    "deployment_episodes[].sample_scene_visibility.mask_gt_available_count must be 0 in non-promotion mode.",
                    episode_index=ep_idx,
                    episode_id=ep_obj.get("episode_id"),
                    got=visibility.get("mask_gt_available_count"),
                )
            required_hash_fields = [
                "scene_ids",
                "scene_ids_sorted_hash",
                "sample_id_order_hash",
                "sample_ids_sorted_hash",
                "first_sample_id",
                "last_sample_id",
            ]
            missing_hash_fields = [key for key in required_hash_fields if key not in visibility]
            if missing_hash_fields:
                issue_list = failures if deployment_episode_hash_strict_required else warnings
                issue_code = (
                    "deployment_episode_visibility_hash_fields_missing"
                    if deployment_episode_hash_strict_required
                    else "deployment_episode_visibility_hash_fields_missing_legacy_compatible"
                )
                _add_issue(
                    issue_list,
                    issue_code,
                    "deployment_episodes[].sample_scene_visibility hash/audit fields missing.",
                    episode_index=ep_idx,
                    episode_id=ep_obj.get("episode_id"),
                    missing_fields=missing_hash_fields,
                )
            else:
                if [str(v).strip() for v in _as_list(visibility.get("scene_ids")) if str(v).strip()] != scene_ids:
                    _add_issue(
                        failures,
                        "deployment_episode_visibility_scene_ids_mismatch",
                        "deployment_episodes[].sample_scene_visibility.scene_ids must match recomputed sorted scene IDs.",
                        episode_index=ep_idx,
                        episode_id=ep_obj.get("episode_id"),
                        manifest_value=visibility.get("scene_ids"),
                        computed_value=scene_ids,
                    )
                if str(visibility.get("scene_ids_sorted_hash") or "").strip() != _hash_text_parts(scene_ids):
                    _add_issue(
                        failures,
                        "deployment_episode_visibility_scene_ids_sorted_hash_mismatch",
                        "deployment_episodes[].sample_scene_visibility.scene_ids_sorted_hash mismatch.",
                        episode_index=ep_idx,
                        episode_id=ep_obj.get("episode_id"),
                        manifest_value=visibility.get("scene_ids_sorted_hash"),
                        computed_value=_hash_text_parts(scene_ids),
                    )
                if str(visibility.get("sample_id_order_hash") or "").strip() != _hash_text_parts(sample_ids_ordered):
                    _add_issue(
                        failures,
                        "deployment_episode_visibility_sample_id_order_hash_mismatch",
                        "deployment_episodes[].sample_scene_visibility.sample_id_order_hash mismatch.",
                        episode_index=ep_idx,
                        episode_id=ep_obj.get("episode_id"),
                        manifest_value=visibility.get("sample_id_order_hash"),
                        computed_value=_hash_text_parts(sample_ids_ordered),
                    )
                if str(visibility.get("sample_ids_sorted_hash") or "").strip() != _hash_text_parts(sample_ids_sorted):
                    _add_issue(
                        failures,
                        "deployment_episode_visibility_sample_ids_sorted_hash_mismatch",
                        "deployment_episodes[].sample_scene_visibility.sample_ids_sorted_hash mismatch.",
                        episode_index=ep_idx,
                        episode_id=ep_obj.get("episode_id"),
                        manifest_value=visibility.get("sample_ids_sorted_hash"),
                        computed_value=_hash_text_parts(sample_ids_sorted),
                    )
                if visibility.get("first_sample_id") != (sample_ids_ordered[0] if sample_ids_ordered else None):
                    _add_issue(
                        failures,
                        "deployment_episode_visibility_first_sample_id_mismatch",
                        "deployment_episodes[].sample_scene_visibility.first_sample_id mismatch.",
                        episode_index=ep_idx,
                        episode_id=ep_obj.get("episode_id"),
                        manifest_value=visibility.get("first_sample_id"),
                        computed_value=sample_ids_ordered[0] if sample_ids_ordered else None,
                    )
                if visibility.get("last_sample_id") != (sample_ids_ordered[-1] if sample_ids_ordered else None):
                    _add_issue(
                        failures,
                        "deployment_episode_visibility_last_sample_id_mismatch",
                        "deployment_episodes[].sample_scene_visibility.last_sample_id mismatch.",
                        episode_index=ep_idx,
                        episode_id=ep_obj.get("episode_id"),
                        manifest_value=visibility.get("last_sample_id"),
                        computed_value=sample_ids_ordered[-1] if sample_ids_ordered else None,
                    )
            episode_id = str(ep_obj.get("episode_id") or "").strip()
            deployment_episode_scene_rows.append(f"{episode_id}|{_hash_text_parts(scene_ids)}")
            deployment_episode_sample_order_rows.append(f"{episode_id}|{_hash_text_parts(sample_ids_ordered)}")
            deployment_episode_sample_sorted_rows.append(f"{episode_id}|{_hash_text_parts(sample_ids_sorted)}")
        if _as_list(episodes_payload.get("episodes")) and deployment_visibility_present_count == 0:
            _add_issue(
                warnings,
                "deployment_episode_visibility_missing_legacy_compatible",
                "deployment_episodes[].sample_scene_visibility missing; treated as legacy-compatible run.",
            )

    run_contract_payload: dict[str, Any] = {}
    if artifact_presence["run_contract"]:
        run_contract_payload, run_contract_err = _load_json(run_contract_path)
        if run_contract_err:
            _add_issue(
                failures,
                "run_contract_invalid",
                "run_contract.json exists but is invalid.",
                error=run_contract_err,
            )
    else:
        _add_issue(
            warnings,
            "run_contract_missing",
            "run_contract.json is absent; contract cross-checks were skipped.",
        )

    run_summary_payload: dict[str, Any] = {}
    if artifact_presence["run_summary"]:
        run_summary_payload, run_summary_err = _load_json(run_summary_path)
        if run_summary_err:
            _add_issue(
                failures,
                "run_summary_invalid",
                "run_summary.json exists but is invalid.",
                error=run_summary_err,
            )
    else:
        _add_issue(
            warnings,
            "run_summary_missing_legacy_compatible",
            "run_summary.json is absent; treated as legacy run for compatibility.",
        )

    if plan_payload:
        if plan_payload.get("schema_version") != "carla_air_dataset_generation_plan_v1":
            _add_issue(
                failures,
                "plan_schema_mismatch",
                "dataset_plan.json schema_version mismatch.",
                got=plan_payload.get("schema_version"),
                expected="carla_air_dataset_generation_plan_v1",
            )
        if plan_payload.get("read_only") is not True:
            _add_issue(
                failures,
                "plan_read_only_mismatch",
                "dataset_plan.read_only must be true.",
                got=plan_payload.get("read_only"),
            )
        if plan_payload.get("starts_runtime") is not False:
            _add_issue(
                failures,
                "plan_starts_runtime_mismatch",
                "dataset_plan.starts_runtime must be false.",
                got=plan_payload.get("starts_runtime"),
            )
        if plan_payload.get("writes_scene_outputs") is not False:
            _add_issue(
                failures,
                "plan_writes_scene_outputs_mismatch",
                "dataset_plan.writes_scene_outputs must be false.",
                got=plan_payload.get("writes_scene_outputs"),
            )

    if manifest_payload:
        if manifest_payload.get("schema_version") != "carla_air_dataset_training_index_v1":
            _add_issue(
                failures,
                "manifest_schema_mismatch",
                "dataset_manifest.json schema_version mismatch.",
                got=manifest_payload.get("schema_version"),
                expected="carla_air_dataset_training_index_v1",
            )
        for guard_key, expected in (
            ("starts_runtime", False),
            ("writes_scene_outputs", False),
            ("non_promotion", True),
            ("full_v1_live_dataset_ready", False),
            ("ue_carla_import_externalized", True),
        ):
            if guard_key not in manifest_payload:
                _add_issue(
                    warnings,
                    "dataset_manifest_guard_field_missing_legacy_compatible",
                    "dataset_manifest top-level guard field missing; treated as legacy-compatible.",
                    field=guard_key,
                    expected=expected,
                )
            elif manifest_payload.get(guard_key) is not expected:
                _add_issue(
                    failures,
                    "dataset_manifest_guard_field_mismatch",
                    "dataset_manifest top-level guard field must match contract.",
                    field=guard_key,
                    got=manifest_payload.get(guard_key),
                    expected=expected,
                )

    if run_contract_payload:
        if run_contract_payload.get("schema_version") != "carla_air_dataset_generation_run_contract_v1":
            _add_issue(
                failures,
                "run_contract_schema_mismatch",
                "run_contract schema_version mismatch.",
                got=run_contract_payload.get("schema_version"),
                expected="carla_air_dataset_generation_run_contract_v1",
            )
        if run_contract_payload.get("starts_runtime") is not False:
            _add_issue(
                failures,
                "run_contract_starts_runtime_mismatch",
                "run_contract.starts_runtime must be false.",
                got=run_contract_payload.get("starts_runtime"),
            )
        if run_contract_payload.get("writes_scene_outputs") is not False:
            _add_issue(
                failures,
                "run_contract_writes_scene_outputs_mismatch",
                "run_contract.writes_scene_outputs must be false.",
                got=run_contract_payload.get("writes_scene_outputs"),
            )

    if run_summary_payload:
        if run_summary_payload.get("schema_version") != "carla_air_dataset_generation_orchestrator_run_summary_v1":
            _add_issue(
                failures,
                "run_summary_schema_mismatch",
                "run_summary schema_version mismatch.",
                got=run_summary_payload.get("schema_version"),
                expected="carla_air_dataset_generation_orchestrator_run_summary_v1",
            )
        for flag_key in ("starts_runtime", "writes_scene_outputs"):
            if flag_key in run_summary_payload and run_summary_payload.get(flag_key) is not False:
                _add_issue(
                    failures,
                    f"run_summary_{flag_key}_mismatch",
                    f"run_summary.{flag_key} must be false.",
                    got=run_summary_payload.get(flag_key),
                )

    manifest_sample_count = int(_as_dict(manifest_payload).get("sample_count") or 0)
    sample_count = len(samples)
    if manifest_payload and (manifest_sample_count != sample_count):
        _add_issue(
            failures,
            "sample_count_mismatch_manifest_vs_jsonl",
            "manifest.sample_count does not match JSONL line count.",
            manifest_sample_count=manifest_sample_count,
            jsonl_sample_count=sample_count,
        )

    if args.require_samples and sample_count == 0:
        _add_issue(
            failures,
            "samples_required_but_zero",
            "--require-samples set but sample count is 0.",
        )

    split_distribution: dict[str, int] = {}
    split_ids_total = 0
    split_seen_ids: set[str] = set()
    if splits_payload:
        if splits_payload.get("schema_version") != "carla_air_dataset_splits_v1":
            _add_issue(
                failures,
                "dataset_splits_schema_mismatch",
                "dataset_splits schema_version mismatch.",
                got=splits_payload.get("schema_version"),
                expected="carla_air_dataset_splits_v1",
            )
        if ("split_strategy" in splits_payload) and (splits_payload.get("split_strategy") != "deployment_oriented_node_layout_v1"):
            _add_issue(
                failures,
                "dataset_splits_split_strategy_mismatch",
                "dataset_splits.split_strategy must be deployment_oriented_node_layout_v1 when present.",
                got=splits_payload.get("split_strategy"),
            )
        if ("not_random_frame_split" in splits_payload) and (splits_payload.get("not_random_frame_split") is not True):
            _add_issue(
                failures,
                "dataset_splits_not_random_frame_split_mismatch",
                "dataset_splits.not_random_frame_split must be true when present.",
                got=splits_payload.get("not_random_frame_split"),
            )
        split_map = _as_dict(splits_payload.get("splits"))
        split_names_from_map = sorted(str(name).strip() for name in split_map.keys() if str(name).strip())
        if "split_names" in splits_payload:
            split_names = sorted(str(x).strip() for x in _as_list(splits_payload.get("split_names")) if str(x).strip())
            if split_names != split_names_from_map:
                _add_issue(
                    failures,
                    "dataset_splits_split_names_mismatch",
                    "dataset_splits.split_names must align with splits map keys.",
                    manifest_value=split_names,
                    computed_value=split_names_from_map,
                )
        for split_name, ids in split_map.items():
            if not isinstance(ids, list):
                _add_issue(
                    failures,
                    "split_ids_not_list",
                    "dataset_splits.splits entry is not a list.",
                    split=split_name,
                )
                continue
            split_distribution[str(split_name)] = len(ids)
            split_ids_total += len(ids)
            for sample_id in ids:
                sid = str(sample_id or "").strip()
                if sid:
                    split_seen_ids.add(sid)
        splits_manifest = _as_dict(splits_payload.get("manifest"))
        if "split_count" in splits_payload and _to_int(splits_payload.get("split_count")) != len(split_map):
            _add_issue(
                failures,
                "dataset_splits_split_count_mismatch",
                "dataset_splits.split_count must equal number of splits map entries.",
                manifest_value=splits_payload.get("split_count"),
                computed_value=len(split_map),
            )
        if splits_manifest and ("split_count" in splits_manifest) and (_to_int(splits_manifest.get("split_count")) != len(split_map)):
            _add_issue(
                failures,
                "dataset_splits_manifest_split_count_mismatch",
                "dataset_splits.manifest.split_count must equal number of splits map entries.",
                manifest_value=splits_manifest.get("split_count"),
                computed_value=len(split_map),
            )
        if "sample_count" in splits_payload and _to_int(splits_payload.get("sample_count")) != split_ids_total:
            _add_issue(
                failures,
                "dataset_splits_sample_count_mismatch",
                "dataset_splits.sample_count must equal total sample IDs in splits map.",
                manifest_value=splits_payload.get("sample_count"),
                computed_value=split_ids_total,
            )
        if splits_manifest and ("sample_count" in splits_manifest) and (_to_int(splits_manifest.get("sample_count")) != split_ids_total):
            _add_issue(
                failures,
                "dataset_splits_manifest_sample_count_mismatch",
                "dataset_splits.manifest.sample_count must equal total sample IDs in splits map.",
                manifest_value=splits_manifest.get("sample_count"),
                computed_value=split_ids_total,
            )
        if splits_manifest and ("split_strategy" in splits_manifest) and (
            str(splits_manifest.get("split_strategy") or "").strip() != str(splits_payload.get("split_strategy") or "").strip()
        ):
            _add_issue(
                failures,
                "dataset_splits_manifest_split_strategy_mismatch",
                "dataset_splits.manifest.split_strategy must match dataset_splits.split_strategy.",
                manifest_value=splits_manifest.get("split_strategy"),
                computed_value=splits_payload.get("split_strategy"),
            )
        if splits_manifest and ("not_random_frame_split" in splits_manifest) and (
            bool(splits_manifest.get("not_random_frame_split")) != bool(splits_payload.get("not_random_frame_split"))
        ):
            _add_issue(
                failures,
                "dataset_splits_manifest_not_random_frame_split_mismatch",
                "dataset_splits.manifest.not_random_frame_split must match dataset_splits.not_random_frame_split.",
                manifest_value=splits_manifest.get("not_random_frame_split"),
                computed_value=splits_payload.get("not_random_frame_split"),
            )
        for dist_source, dist_obj in (
            ("dataset_splits.split_distribution", _as_dict(splits_payload.get("split_distribution"))),
            ("dataset_splits.manifest.split_distribution", _as_dict(splits_manifest.get("split_distribution"))),
        ):
            if not dist_obj:
                continue
            computed_dist = {k: split_distribution.get(k, 0) for k in split_names_from_map}
            observed_dist = {str(k).strip(): _to_int(v) for k, v in dist_obj.items() if str(k).strip()}
            if observed_dist != computed_dist:
                _add_issue(
                    failures,
                    "dataset_splits_split_distribution_mismatch",
                    f"{dist_source} must align with splits map lengths.",
                    manifest_value=observed_dist,
                    computed_value=computed_dist,
                )
        if split_ids_total != manifest_sample_count:
            _add_issue(
                failures,
                "split_total_mismatch_sample_count",
                "Total sample IDs in splits does not equal sample_count.",
                split_sample_id_total=split_ids_total,
                sample_count=manifest_sample_count,
            )
        if split_ids_total != sample_count:
            _add_issue(
                failures,
                "split_total_mismatch_jsonl_sample_count",
                "Total sample IDs in splits does not equal dataset_samples.jsonl sample count.",
                split_sample_id_total=split_ids_total,
                jsonl_sample_count=sample_count,
            )
    else:
        for sample in samples:
            split_name = str(sample.get("split") or "unknown")
            split_distribution[split_name] = split_distribution.get(split_name, 0) + 1

    if samples and split_seen_ids:
        sample_ids_in_jsonl = {str(item.get("sample_id") or "").strip() for item in samples}
        sample_ids_in_jsonl.discard("")
        if sample_ids_in_jsonl != split_seen_ids:
            _add_issue(
                failures,
                "split_sample_id_set_mismatch",
                "Sample IDs between dataset_samples.jsonl and dataset_splits.json differ.",
                sample_ids_jsonl_count=len(sample_ids_in_jsonl),
                sample_ids_splits_count=len(split_seen_ids),
            )

    dataset_index_manifest_payload: dict[str, Any] = {}
    dataset_index_manifest_contract_required = False
    if run_contract_payload:
        contract_artifacts_early = _as_dict(run_contract_payload.get("artifacts"))
        dataset_index_manifest_raw = contract_artifacts_early.get("dataset_index_manifest_json")
        if dataset_index_manifest_raw:
            dataset_index_manifest_contract_required = True
    strict_index_contract_presence = {
        "present": False,
        "required_by_run_contract_artifacts": bool(dataset_index_manifest_contract_required),
        "legacy_compatible_warning_emitted": False,
    }
    strict_index_contract_report: dict[str, Any] = {}
    should_validate_dataset_index_manifest = artifact_presence["dataset_index_manifest"] or dataset_index_manifest_contract_required
    if should_validate_dataset_index_manifest:
        dataset_index_manifest_payload, dataset_index_manifest_err = _load_json(dataset_index_manifest_path)
        if dataset_index_manifest_err:
            _add_issue(
                failures,
                "dataset_index_manifest_invalid",
                "dataset_index_manifest.json is missing or invalid while present/contract-required.",
                error=dataset_index_manifest_err,
            )
        else:
            if dataset_index_manifest_payload.get("schema_version") != SCHEMA_INDEX_MANIFEST:
                _add_issue(
                    failures,
                    "dataset_index_manifest_schema_mismatch",
                    "dataset_index_manifest schema_version mismatch.",
                    got=dataset_index_manifest_payload.get("schema_version"),
                    expected=SCHEMA_INDEX_MANIFEST,
                )
            if dataset_index_manifest_payload.get("non_promotion") is not True:
                _add_issue(
                    failures,
                    "dataset_index_manifest_non_promotion_mismatch",
                    "dataset_index_manifest.non_promotion must be true.",
                    got=dataset_index_manifest_payload.get("non_promotion"),
                )
            if dataset_index_manifest_payload.get("full_v1_live_dataset_ready") is not False:
                _add_issue(
                    failures,
                    "dataset_index_manifest_full_v1_live_dataset_ready_mismatch",
                    "dataset_index_manifest.full_v1_live_dataset_ready must be false.",
                    got=dataset_index_manifest_payload.get("full_v1_live_dataset_ready"),
                )
            if _to_int(dataset_index_manifest_payload.get("sample_count")) != sample_count:
                _add_issue(
                    failures,
                    "dataset_index_manifest_sample_count_mismatch",
                    "dataset_index_manifest.sample_count must match dataset_samples.jsonl row count.",
                    manifest_value=dataset_index_manifest_payload.get("sample_count"),
                    computed_value=sample_count,
                )
            sample_ids_in_order = [str(_as_dict(s).get("sample_id") or "").strip() for s in samples]
            sample_ids_non_empty = [sid for sid in sample_ids_in_order if sid]
            sample_ids_set = set(sample_ids_non_empty)
            split_map_local = _as_dict(splits_payload.get("splits"))
            split_distribution_local = {
                str(split_name): len(_as_list(split_ids))
                for split_name, split_ids in split_map_local.items()
                if str(split_name).strip()
            }
            split_ids_set_local: set[str] = set()
            for split_ids in split_map_local.values():
                for sample_id in _as_list(split_ids):
                    sid = str(sample_id or "").strip()
                    if sid:
                        split_ids_set_local.add(sid)
            canonical_rows = [
                json.dumps(_as_dict(sample), ensure_ascii=True, sort_keys=True, separators=(",", ":"))
                for sample in samples
            ]
            scene_membership: dict[str, dict[str, Any]] = {}
            for sample in samples:
                sample_obj = _as_dict(sample)
                source = _as_dict(sample_obj.get("source"))
                identity_id = str(sample_obj.get("identity_id") or "unknown_identity").strip() or "unknown_identity"
                trajectory_id = str(sample_obj.get("trajectory_id") or "unknown_trajectory").strip() or "unknown_trajectory"
                node_id = str(sample_obj.get("node_id") or "unknown_node").strip() or "unknown_node"
                scene_id = str(sample_obj.get("scene_id") or source.get("scene_id") or "").strip()
                scene_dir = str(source.get("scene_dir") or source.get("scene_root") or "").strip()
                if not scene_id:
                    scene_id = "unknown_scene"
                scene_key = _scene_key_for_fields(identity_id, trajectory_id, node_id, scene_id, scene_dir)
                split_name = str(sample_obj.get("split") or "unknown").strip() or "unknown"
                sample_id = str(sample_obj.get("sample_id") or "").strip()
                rec = scene_membership.setdefault(
                    scene_key,
                    {
                        "scene_key": scene_key,
                        "identity_id": identity_id,
                        "trajectory_id": trajectory_id,
                        "node_id": node_id,
                        "scene_id": scene_id,
                        "scene_dir": scene_dir,
                        "source_scene_root": scene_dir,
                        "sample_ids": [],
                        "split_names": set(),
                    },
                )
                if sample_id:
                    rec["sample_ids"].append(sample_id)
                rec["split_names"].add(split_name)
            scene_split_membership_index: list[dict[str, Any]] = []
            for scene_key in sorted(scene_membership):
                rec = scene_membership[scene_key]
                scene_sample_ids = [str(v).strip() for v in _as_list(rec.get("sample_ids")) if str(v).strip()]
                scene_split_names = sorted({str(v).strip() for v in rec.get("split_names", set()) if str(v).strip()})
                scene_split_membership_index.append(
                    {
                        "scene_key": str(rec.get("scene_key") or ""),
                        "identity_id": str(rec.get("identity_id") or ""),
                        "trajectory_id": str(rec.get("trajectory_id") or ""),
                        "node_id": str(rec.get("node_id") or ""),
                        "scene_id": str(rec.get("scene_id") or ""),
                        "scene_dir": str(rec.get("scene_dir") or ""),
                        "source_scene_root": str(rec.get("source_scene_root") or ""),
                        "split_names": scene_split_names,
                        "sample_count": len(scene_sample_ids),
                        "sample_id_order_hash": _hash_text_parts(scene_sample_ids),
                        "sample_ids_sorted_hash": _hash_text_parts(sorted(scene_sample_ids)),
                        "first_sample_id": scene_sample_ids[0] if scene_sample_ids else None,
                        "last_sample_id": scene_sample_ids[-1] if scene_sample_ids else None,
                    }
                )
            scene_split_membership_hash = _hash_text_parts(
                [
                    json.dumps(entry, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
                    for entry in scene_split_membership_index
                ]
            )
            computed = {
                "sample_id_count": len(sample_ids_non_empty),
                "sample_id_unique_count": len(sample_ids_set),
                "duplicate_sample_id_count": len(sample_ids_non_empty) - len(sample_ids_set),
                "missing_from_splits_count": len(sample_ids_set - split_ids_set_local),
                "extra_in_splits_count": len(split_ids_set_local - sample_ids_set),
                "sample_id_order_hash": _hash_text_parts(sample_ids_non_empty),
                "sample_content_hash": _hash_text_parts(canonical_rows),
                "split_sample_id_set_hash": _hash_text_parts(sorted(split_ids_set_local)),
                "sample_ids_sorted_hash": _hash_text_parts(sorted(sample_ids_set)),
            }
            scene_split_membership_computed = {
                "scene_count": len(scene_split_membership_index),
                "scene_ids_sorted_hash": _hash_text_parts([entry["scene_id"] for entry in scene_split_membership_index]),
                "scene_keys_sorted_hash": _hash_text_parts([entry["scene_key"] for entry in scene_split_membership_index]),
                "scene_split_membership_hash": scene_split_membership_hash,
            }
            scene_membership_legacy: dict[str, dict[str, Any]] = {}
            for sample in samples:
                sample_obj = _as_dict(sample)
                source = _as_dict(sample_obj.get("source"))
                scene_id = str(sample_obj.get("scene_id") or source.get("scene_id") or source.get("scene_dir") or "").strip()
                if not scene_id:
                    scene_id = "unknown_scene"
                split_name = str(sample_obj.get("split") or "unknown").strip() or "unknown"
                sample_id = str(sample_obj.get("sample_id") or "").strip()
                rec = scene_membership_legacy.setdefault(scene_id, {"sample_ids": [], "split_names": set()})
                if sample_id:
                    rec["sample_ids"].append(sample_id)
                rec["split_names"].add(split_name)
            scene_split_membership_index_legacy: list[dict[str, Any]] = []
            for scene_id in sorted(scene_membership_legacy):
                rec = scene_membership_legacy[scene_id]
                scene_sample_ids = [str(v).strip() for v in _as_list(rec.get("sample_ids")) if str(v).strip()]
                scene_split_names = sorted({str(v).strip() for v in rec.get("split_names", set()) if str(v).strip()})
                scene_split_membership_index_legacy.append(
                    {
                        "scene_id": scene_id,
                        "split_names": scene_split_names,
                        "sample_count": len(scene_sample_ids),
                        "sample_id_order_hash": _hash_text_parts(scene_sample_ids),
                        "sample_ids_sorted_hash": _hash_text_parts(sorted(scene_sample_ids)),
                        "first_sample_id": scene_sample_ids[0] if scene_sample_ids else None,
                        "last_sample_id": scene_sample_ids[-1] if scene_sample_ids else None,
                    }
                )
            scene_split_membership_hash_legacy = _hash_text_parts(
                [
                    json.dumps(entry, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
                    for entry in scene_split_membership_index_legacy
                ]
            )
            scene_split_membership_computed_legacy = {
                "scene_count": len(scene_split_membership_index_legacy),
                "scene_ids_sorted_hash": _hash_text_parts([entry["scene_id"] for entry in scene_split_membership_index_legacy]),
                "scene_split_membership_hash": scene_split_membership_hash_legacy,
            }
            computed_int_fields = {
                "sample_id_count",
                "sample_id_unique_count",
                "duplicate_sample_id_count",
                "missing_from_splits_count",
                "extra_in_splits_count",
            }
            computed_hash_fields = {
                "sample_id_order_hash",
                "sample_content_hash",
                "split_sample_id_set_hash",
                "sample_ids_sorted_hash",
            }
            for key, expected in computed.items():
                observed = dataset_index_manifest_payload.get(key)
                mismatch = False
                if key in computed_int_fields:
                    mismatch = _to_int(observed) != int(expected)
                elif key in computed_hash_fields:
                    mismatch = str(observed).strip() != str(expected)
                else:
                    mismatch = observed != expected
                if mismatch:
                    _add_issue(
                        failures,
                        "dataset_index_manifest_field_mismatch",
                        "dataset_index_manifest field mismatch against recomputed value.",
                        field=key,
                        manifest_value=dataset_index_manifest_payload.get(key),
                        computed_value=expected,
                    )
            manifest_scene_split_membership_index = _as_list(dataset_index_manifest_payload.get("scene_split_membership_index"))
            normalized_manifest_scene_index = [_as_dict(item) for item in manifest_scene_split_membership_index]
            manifest_has_scene_key = bool(normalized_manifest_scene_index) and all(
                bool(str(item.get("scene_key") or "").strip()) for item in normalized_manifest_scene_index
            )
            scene_split_membership_computed_selected = (
                scene_split_membership_computed if manifest_has_scene_key else scene_split_membership_computed_legacy
            )
            scene_split_membership_keys_present = any(
                key in dataset_index_manifest_payload for key in scene_split_membership_computed_selected
            )
            if scene_split_membership_keys_present:
                for key, expected in scene_split_membership_computed_selected.items():
                    observed = dataset_index_manifest_payload.get(key)
                    if manifest_has_scene_key and key == "scene_keys_sorted_hash" and key not in dataset_index_manifest_payload:
                        _add_issue(
                            warnings,
                            "dataset_index_manifest_scene_keys_sorted_hash_missing_legacy_compatible",
                            "dataset_index_manifest.scene_keys_sorted_hash is missing; treated as transition-compatible.",
                            computed_value=expected,
                        )
                        continue
                    if key == "scene_count":
                        mismatch = _to_int(observed) != int(expected)
                    else:
                        mismatch = str(observed or "").strip() != str(expected)
                    if mismatch:
                        if (
                            manifest_has_scene_key
                            and key == "scene_ids_sorted_hash"
                            and "scene_keys_sorted_hash" not in dataset_index_manifest_payload
                            and str(observed or "").strip()
                            == str(scene_split_membership_computed.get("scene_keys_sorted_hash") or "").strip()
                        ):
                            _add_issue(
                                warnings,
                                "dataset_index_manifest_scene_ids_sorted_hash_scene_key_semantics_legacy_compatible",
                                "dataset_index_manifest.scene_ids_sorted_hash uses transition scene_key semantics; "
                                "treated as legacy-compatible when scene_keys_sorted_hash is absent.",
                                manifest_value=observed,
                                computed_scene_id_hash=expected,
                                computed_scene_key_hash=scene_split_membership_computed.get("scene_keys_sorted_hash"),
                            )
                            continue
                        _add_issue(
                            failures,
                            "dataset_index_manifest_scene_split_membership_field_mismatch",
                            "dataset_index_manifest scene/split membership field mismatch against recomputed value.",
                            field=key,
                            manifest_value=observed,
                            computed_value=expected,
                        )
            else:
                _add_issue(
                    warnings,
                    "dataset_index_manifest_scene_split_membership_fields_missing_legacy_compatible",
                    "dataset_index_manifest scene/split membership hash fields are missing; treated as legacy-compatible.",
                )
            if _as_dict(dataset_index_manifest_payload.get("split_distribution")) != split_distribution_local:
                _add_issue(
                    failures,
                    "dataset_index_manifest_split_distribution_mismatch",
                    "dataset_index_manifest.split_distribution must match dataset_splits.json.",
                    manifest_value=_as_dict(dataset_index_manifest_payload.get("split_distribution")),
                    computed_value=split_distribution_local,
                )
            if manifest_scene_split_membership_index:
                if manifest_has_scene_key:
                    if normalized_manifest_scene_index != scene_split_membership_index:
                        _add_issue(
                            failures,
                            "dataset_index_manifest_scene_split_membership_index_mismatch",
                            "dataset_index_manifest.scene_split_membership_index must match recomputed scene/split membership.",
                            manifest_count=len(normalized_manifest_scene_index),
                            computed_count=len(scene_split_membership_index),
                        )
                else:
                    _add_issue(
                        warnings,
                        "dataset_index_manifest_scene_split_membership_index_missing_scene_key_legacy_compatible",
                        "dataset_index_manifest.scene_split_membership_index is missing scene_key; treated as legacy-compatible.",
                    )
                    if normalized_manifest_scene_index != scene_split_membership_index_legacy:
                        _add_issue(
                            failures,
                            "dataset_index_manifest_scene_split_membership_index_legacy_mismatch",
                            "legacy dataset_index_manifest.scene_split_membership_index must match legacy recomputed scene/split membership.",
                            manifest_count=len(normalized_manifest_scene_index),
                            computed_count=len(scene_split_membership_index_legacy),
                        )
            elif dataset_index_manifest_payload:
                _add_issue(
                    warnings,
                    "dataset_index_manifest_scene_split_membership_index_missing_legacy_compatible",
                    "dataset_index_manifest.scene_split_membership_index is missing; treated as legacy-compatible.",
                )
            if manifest_payload:
                manifest_mask_summary = _as_dict(manifest_payload.get("mask_gt_availability_summary"))
                manifest_mask_count = _to_int(manifest_mask_summary.get("available_count"))
                if _to_int(dataset_index_manifest_payload.get("mask_gt_available_count")) != manifest_mask_count:
                    _add_issue(
                        failures,
                        "dataset_index_manifest_mask_gt_available_count_mismatch",
                        "dataset_index_manifest.mask_gt_available_count must match dataset_manifest mask_gt summary.",
                        manifest_value=dataset_index_manifest_payload.get("mask_gt_available_count"),
                        computed_value=manifest_mask_count,
                    )
            strict_index_contract_obj = _as_dict(dataset_index_manifest_payload.get("strict_index_contract"))
            strict_index_contract_present = bool(strict_index_contract_obj)
            strict_index_contract_presence["present"] = strict_index_contract_present
            strict_required = strict_index_contract_present or bool(dataset_index_manifest_contract_required)
            strict_expected = {
                "schema_version": SCHEMA_INDEX_STRICT_CONTRACT,
                "sample_id_integrity_enforced": True,
                "sample_id_complete": len(sample_ids_non_empty) == sample_count,
                "duplicate_sample_id_count": _to_int(computed.get("duplicate_sample_id_count")),
                "missing_from_splits_count": _to_int(computed.get("missing_from_splits_count")),
                "extra_in_splits_count": _to_int(computed.get("extra_in_splits_count")),
                "scene_split_membership_index_required": True,
                "scene_key_required": True,
                "scene_keys_sorted_hash_required": True,
                "scene_split_membership_hash_required": True,
                "all_required_scene_membership_fields_present": True,
                "non_promotion": True,
                "full_v1_live_dataset_ready": False,
                "mask_gt_available_count": _to_int(dataset_index_manifest_payload.get("mask_gt_available_count")),
            }
            required_scene_membership_fields = {
                "scene_key",
                "identity_id",
                "trajectory_id",
                "node_id",
                "scene_id",
                "scene_dir",
                "source_scene_root",
                "split_names",
                "sample_count",
                "sample_id_order_hash",
                "sample_ids_sorted_hash",
                "first_sample_id",
                "last_sample_id",
            }
            all_required_scene_membership_fields_present = bool(manifest_scene_split_membership_index) and all(
                required_scene_membership_fields.issubset(set(_as_dict(item).keys()))
                for item in manifest_scene_split_membership_index
            )
            strict_expected["all_required_scene_membership_fields_present"] = all_required_scene_membership_fields_present
            strict_index_contract_report = {
                "present": strict_index_contract_present,
                "required": strict_required,
                "computed": strict_expected,
                "manifest": strict_index_contract_obj if strict_index_contract_present else None,
            }
            if strict_required:
                required_top_level_fields = (
                    "scene_keys_sorted_hash",
                    "scene_split_membership_index",
                    "scene_split_membership_hash",
                )
                for top_field in required_top_level_fields:
                    if top_field not in dataset_index_manifest_payload:
                        _add_issue(
                            failures,
                            "dataset_index_manifest_strict_required_field_missing",
                            "strict dataset_index_contract requires scene membership fields at top-level.",
                            field=top_field,
                        )
                if "scene_split_membership_index" in dataset_index_manifest_payload:
                    if (not manifest_scene_split_membership_index) or (
                        not all(bool(str(_as_dict(item).get("scene_key") or "").strip()) for item in manifest_scene_split_membership_index)
                    ):
                        _add_issue(
                            failures,
                            "dataset_index_manifest_strict_required_scene_key_missing",
                            "strict dataset_index_contract requires scene_key in every scene_split_membership_index entry.",
                        )
                if not strict_index_contract_present:
                    _add_issue(
                        failures,
                        "dataset_index_manifest_strict_index_contract_missing",
                        "strict dataset index contract is required for this run but strict_index_contract is missing.",
                    )
                else:
                    for key, expected_value in strict_expected.items():
                        observed_value = strict_index_contract_obj.get(key)
                        if isinstance(expected_value, bool):
                            mismatch = observed_value is not expected_value
                        elif isinstance(expected_value, int):
                            mismatch = _to_int(observed_value) != expected_value
                        else:
                            mismatch = str(observed_value or "").strip() != str(expected_value)
                        if mismatch:
                            _add_issue(
                                failures,
                                "dataset_index_manifest_strict_index_contract_mismatch",
                                "strict_index_contract field mismatch against recomputed verifier value.",
                                field=key,
                                manifest_value=observed_value,
                                computed_value=expected_value,
                            )
            elif not strict_index_contract_present:
                strict_index_contract_presence["legacy_compatible_warning_emitted"] = True
                _add_issue(
                    warnings,
                    "dataset_index_manifest_strict_index_contract_missing_legacy_compatible",
                    "strict_index_contract is absent; treated as legacy-compatible run.",
                )
    else:
        _add_issue(
            warnings,
            "dataset_index_manifest_missing_legacy_compatible",
            "dataset_index_manifest.json is absent; treated as legacy-compatible run.",
        )

    plan_counts = _as_dict(plan_payload.get("counts"))
    plan_capture_task_count = _to_int(plan_counts.get("capture_task_count"))
    plan_capture_tasks = _as_list(plan_payload.get("capture_tasks"))
    plan_capture_task_id_set: set[str] = set()
    for task in plan_capture_tasks:
        task_obj = _as_dict(task)
        task_id = str(task_obj.get("capture_task_id") or task_obj.get("task_id") or task_obj.get("id") or "").strip()
        if task_id:
            plan_capture_task_id_set.add(task_id)
    if plan_payload and (plan_capture_task_count != len(plan_capture_tasks)):
        _add_issue(
            failures,
            "plan_capture_task_count_mismatch",
            "plan.counts.capture_task_count must equal len(plan.capture_tasks).",
            counts_capture_task_count=plan_capture_task_count,
            capture_tasks_len=len(plan_capture_tasks),
        )
    if plan_payload:
        count_fields = {
            "identity_count": "identities",
            "trajectory_count": "trajectories",
            "camera_layout_count": "camera_layouts",
            "matrix_count": "matrix",
            "capture_task_count": "capture_tasks",
        }
        for count_key, list_key in count_fields.items():
            list_len = len(_as_list(plan_payload.get(list_key)))
            count_val = _to_int(plan_counts.get(count_key))
            if count_val != list_len:
                _add_issue(
                    failures,
                    "plan_count_list_mismatch",
                    "plan.counts field must equal corresponding list length.",
                    count_key=count_key,
                    count_value=count_val,
                    list_key=list_key,
                    list_len=list_len,
                )

    allowed_nodes = {f"node0{i}" for i in range(1, 6)}
    allowed_cameras = {f"cam{i}" for i in range(0, 3)}
    plan_identities = _collect_id_set(plan_payload.get("identities"), "identity_id")
    plan_trajectories = _collect_id_set(plan_payload.get("trajectories"), "trajectory_id")
    plan_filters = _as_dict(plan_payload.get("selected_filters"))
    filter_nodes = [str(x).strip() for x in _as_list(plan_filters.get("node_ids")) if str(x).strip()]
    filter_cameras = [str(x).strip() for x in _as_list(plan_filters.get("camera_ids")) if str(x).strip()]
    filter_identities = [str(x).strip() for x in _as_list(plan_filters.get("identity_ids")) if str(x).strip()]
    filter_trajectories = [str(x).strip() for x in _as_list(plan_filters.get("trajectory_ids")) if str(x).strip()]
    if plan_payload:
        for node in filter_nodes:
            if node not in allowed_nodes:
                _add_issue(
                    failures,
                    "plan_selected_filters_node_invalid",
                    "selected_filters.node_ids contains invalid node id.",
                    node_id=node,
                )
        for camera in filter_cameras:
            if camera not in allowed_cameras:
                _add_issue(
                    failures,
                    "plan_selected_filters_camera_invalid",
                    "selected_filters.camera_ids contains invalid camera id.",
                    camera_id=camera,
                )
        if not filter_identities:
            _add_issue(
                failures,
                "plan_selected_filters_identity_empty",
                "selected_filters.identity_ids must be non-empty.",
            )
        if not filter_trajectories:
            _add_issue(
                failures,
                "plan_selected_filters_trajectory_empty",
                "selected_filters.trajectory_ids must be non-empty.",
            )
        if set(filter_identities) != plan_identities:
            _add_issue(
                failures,
                "plan_selected_filters_identity_mismatch",
                "selected_filters.identity_ids must align with plan.identities.",
                selected_filters_identity_ids=sorted(set(filter_identities)),
                plan_identities=sorted(plan_identities),
            )
        if set(filter_trajectories) != plan_trajectories:
            _add_issue(
                failures,
                "plan_selected_filters_trajectory_mismatch",
                "selected_filters.trajectory_ids must align with plan.trajectories.",
                selected_filters_trajectory_ids=sorted(set(filter_trajectories)),
                plan_trajectories=sorted(plan_trajectories),
            )

    plan_profiles = _as_list(plan_payload.get("identity_model_profiles"))
    plan_profile_map: dict[str, dict[str, Any]] = {}
    for idx, profile in enumerate(plan_profiles, start=1):
        pobj = _as_dict(profile)
        profile_id = str(pobj.get("identity_model_profile_id") or "").strip()
        identity_id = str(pobj.get("identity_id") or "").strip()
        if not profile_id or not identity_id:
            _add_issue(
                warnings,
                "plan_identity_model_profile_incomplete_legacy_compatible",
                "identity_model_profiles entry missing profile_id/identity_id; treated as legacy-compatible.",
                index=idx,
            )
            continue
        plan_profile_map[profile_id] = _normalize_profile_ref(pobj)
    plan_profile_contract_present = bool(plan_profile_map)
    if plan_payload and not plan_profile_contract_present:
        _add_issue(
            warnings,
            "plan_identity_model_profiles_missing_legacy_compatible",
            "plan.identity_model_profiles missing/empty; treated as legacy-compatible run.",
        )

    contract_capture_task_count = None
    contract_artifact_count_including_self_reported: int | None = None
    contract_self_artifact_key_reported = ""
    contract_excluded_self_reference_from_hashed_entries: bool | None = None
    capture_queue_rows: list[dict[str, Any]] = []
    capture_queue_blocked_count = 0
    capture_queue_contract_required = False
    capture_queue_manifest_payload: dict[str, Any] = {}
    capture_queue_manifest_contract_declared = False
    capture_queue_manifest_contract_required = False
    scene_output_manifest_payload: dict[str, Any] = {}
    scene_output_manifest_contract_required = False
    scene_discovery_manifest_payload: dict[str, Any] = {}
    scene_discovery_manifest_contract_required = False
    batch_run_manifest_payload: dict[str, Any] = {}
    batch_run_manifest_contract_required = False
    scene_output_count = 0
    blocked_scene_output_count = 0
    identity_model_switch_manifest_payload: dict[str, Any] = {}
    identity_model_switch_manifest_contract_declared = False
    identity_model_switch_manifest_contract_required = False
    scene_membership_manifest_payload: dict[str, Any] = {}
    scene_membership_manifest_contract_declared = False
    scene_membership_manifest_contract_required = False
    scene_membership_alignment_manifest_payload: dict[str, Any] = {}
    scene_membership_alignment_manifest_contract_declared = False
    scene_membership_alignment_manifest_contract_required = False
    capture_matrix_alignment_manifest_payload: dict[str, Any] = {}
    capture_matrix_alignment_manifest_contract_declared = False
    capture_matrix_alignment_manifest_contract_required = False
    dataset_run_closure_manifest_payload: dict[str, Any] = {}
    dataset_run_closure_manifest_contract_declared = False
    dataset_run_closure_manifest_contract_required = False
    existing_scene_index_bridge_manifest_payload: dict[str, Any] = {}
    existing_scene_index_bridge_manifest_contract_declared = False
    existing_scene_index_bridge_manifest_contract_required = False
    scene_sample_index_manifest_payload: dict[str, Any] = {}
    scene_sample_index_manifest_contract_declared = False
    scene_sample_index_manifest_contract_required = False
    sidecar_quality_manifest_payload: dict[str, Any] = {}
    sidecar_quality_manifest_contract_declared = False
    sidecar_quality_manifest_contract_required = False
    dataset_gap_manifest_payload: dict[str, Any] = {}
    dataset_gap_manifest_contract_declared = False
    dataset_gap_manifest_contract_required = False
    sample_schema_coverage_manifest_payload: dict[str, Any] = {}
    sample_schema_coverage_manifest_contract_declared = False
    sample_schema_coverage_manifest_contract_required = False
    no_mask_non_promotion_manifest_payload: dict[str, Any] = {}
    no_mask_non_promotion_manifest_contract_declared = False
    no_mask_non_promotion_manifest_contract_required = False
    deployment_episode_visibility_manifest_payload: dict[str, Any] = {}
    deployment_episode_visibility_manifest_contract_declared = False
    deployment_episode_visibility_manifest_contract_required = False
    if run_contract_payload:
        contract_counts = _as_dict(run_contract_payload.get("counts"))
        contract_capture_task_count = _to_int(contract_counts.get("capture_task_count"))
        if contract_capture_task_count != plan_capture_task_count:
            _add_issue(
                failures,
                "run_contract_capture_task_count_mismatch",
                "run_contract.counts.capture_task_count must equal plan count.",
                run_contract_capture_task_count=contract_capture_task_count,
                plan_capture_task_count=plan_capture_task_count,
            )
        contract_artifacts = _as_dict(run_contract_payload.get("artifacts"))
        contract_accounting = _as_dict(run_contract_payload.get("artifact_accounting"))
        if contract_accounting:
            contract_artifact_count_including_self_reported = _to_int(
                contract_accounting.get("contract_artifact_count_including_self")
            )
            contract_self_artifact_key_reported = str(contract_accounting.get("self_artifact_key") or "").strip()
            excluded_self_raw = contract_accounting.get("excluded_self_reference_from_hashed_entries")
            if isinstance(excluded_self_raw, bool):
                contract_excluded_self_reference_from_hashed_entries = excluded_self_raw
            observed_contract_artifact_count = len(contract_artifacts)
            if contract_artifact_count_including_self_reported != observed_contract_artifact_count:
                _add_issue(
                    failures,
                    "run_contract_artifact_accounting_count_mismatch",
                    "run_contract.artifact_accounting.contract_artifact_count_including_self must equal len(run_contract.artifacts).",
                    reported_count=contract_artifact_count_including_self_reported,
                    observed_count=observed_contract_artifact_count,
                )
            if not contract_self_artifact_key_reported:
                _add_issue(
                    failures,
                    "run_contract_artifact_accounting_self_key_missing",
                    "run_contract.artifact_accounting.self_artifact_key must be non-empty when artifact_accounting is present.",
                )
            elif contract_self_artifact_key_reported not in contract_artifacts:
                _add_issue(
                    failures,
                    "run_contract_artifact_accounting_self_key_unknown",
                    "run_contract.artifact_accounting.self_artifact_key must exist in run_contract.artifacts.",
                    self_artifact_key=contract_self_artifact_key_reported,
                )
            if contract_excluded_self_reference_from_hashed_entries is not True:
                _add_issue(
                    failures,
                    "run_contract_artifact_accounting_exclusion_flag_mismatch",
                    "run_contract.artifact_accounting.excluded_self_reference_from_hashed_entries must be true when present.",
                    got=excluded_self_raw,
                )
        else:
            _add_issue(
                warnings,
                "run_contract_artifact_accounting_missing_legacy_compatible",
                "run_contract.artifact_accounting missing; treated as legacy-compatible run.",
            )
        required_contract_artifact_keys = [
            "dataset_plan_json",
            "dataset_manifest_json",
            "dataset_samples_jsonl",
            "dataset_splits_json",
            "deployment_episodes_json",
            "run_summary_json",
            "run_contract_json",
        ]
        for key in required_contract_artifact_keys:
            artifact_path_raw = contract_artifacts.get(key)
            artifact_path = _repo_or_abs(str(artifact_path_raw or "")) if artifact_path_raw else Path("")
            if not artifact_path_raw or (not artifact_path.is_file()):
                _add_issue(
                    failures,
                    "run_contract_artifact_missing",
                    "run_contract.artifacts entry is missing or points to a non-file path.",
                    artifact_key=key,
                    artifact_path=str(artifact_path_raw or ""),
                )
        dataset_index_manifest_artifact_raw = contract_artifacts.get("dataset_index_manifest_json")
        if dataset_index_manifest_artifact_raw:
            dataset_index_manifest_artifact_path = _repo_or_abs(str(dataset_index_manifest_artifact_raw))
            if not dataset_index_manifest_artifact_path.is_file():
                _add_issue(
                    failures,
                    "run_contract_artifact_dataset_index_manifest_missing",
                    "run_contract.artifacts.dataset_index_manifest_json points to a non-file path.",
                    artifact_key="dataset_index_manifest_json",
                    artifact_path=str(dataset_index_manifest_artifact_raw),
                )
        else:
            _add_issue(
                warnings,
                "run_contract_artifact_dataset_index_manifest_missing_legacy_compatible",
                "run_contract.artifacts.dataset_index_manifest_json is absent; treated as legacy-compatible run.",
            )
        contract_artifacts = _as_dict(run_contract_payload.get("artifacts"))
        capture_queue_artifact_raw = contract_artifacts.get("capture_queue_jsonl")
        if capture_queue_artifact_raw:
            capture_queue_contract_required = True
            capture_queue_artifact_path = _repo_or_abs(str(capture_queue_artifact_raw))
            if not capture_queue_artifact_path.is_file():
                _add_issue(
                    failures,
                    "run_contract_artifact_capture_queue_missing",
                    "run_contract.artifacts.capture_queue_jsonl points to a non-file path.",
                    artifact_key="capture_queue_jsonl",
                    artifact_path=str(capture_queue_artifact_raw),
                )
        scene_output_manifest_artifact_raw = contract_artifacts.get("scene_output_manifest_json")
        if scene_output_manifest_artifact_raw:
            scene_output_manifest_contract_required = True
            scene_output_manifest_artifact_path = _repo_or_abs(str(scene_output_manifest_artifact_raw))
            if not scene_output_manifest_artifact_path.is_file():
                _add_issue(
                    failures,
                    "run_contract_artifact_scene_output_manifest_missing",
                    "run_contract.artifacts.scene_output_manifest_json points to a non-file path.",
                    artifact_key="scene_output_manifest_json",
                    artifact_path=str(scene_output_manifest_artifact_raw),
                )
        scene_discovery_manifest_artifact_raw = contract_artifacts.get("scene_discovery_manifest_json")
        if scene_discovery_manifest_artifact_raw:
            scene_discovery_manifest_contract_required = True
            scene_discovery_manifest_artifact_path = _repo_or_abs(str(scene_discovery_manifest_artifact_raw))
            if not scene_discovery_manifest_artifact_path.is_file():
                _add_issue(
                    failures,
                    "run_contract_artifact_scene_discovery_manifest_missing",
                    "run_contract.artifacts.scene_discovery_manifest_json points to a non-file path.",
                    artifact_key="scene_discovery_manifest_json",
                    artifact_path=str(scene_discovery_manifest_artifact_raw),
                )
        batch_run_manifest_artifact_raw = contract_artifacts.get("batch_run_manifest_json")
        if batch_run_manifest_artifact_raw:
            batch_run_manifest_contract_required = True
            batch_run_manifest_artifact_path = _repo_or_abs(str(batch_run_manifest_artifact_raw))
            if not batch_run_manifest_artifact_path.is_file():
                _add_issue(
                    failures,
                    "run_contract_artifact_batch_run_manifest_missing",
                    "run_contract.artifacts.batch_run_manifest_json points to a non-file path.",
                    artifact_key="batch_run_manifest_json",
                    artifact_path=str(batch_run_manifest_artifact_raw),
                )
        capture_queue_manifest_artifact_raw = contract_artifacts.get("capture_queue_manifest_json")
        if capture_queue_manifest_artifact_raw:
            capture_queue_manifest_contract_declared = True
            capture_queue_manifest_contract_required = True
            capture_queue_manifest_artifact_path = _repo_or_abs(str(capture_queue_manifest_artifact_raw))
            if not capture_queue_manifest_artifact_path.is_file():
                _add_issue(
                    failures,
                    "run_contract_artifact_capture_queue_manifest_missing",
                    "run_contract.artifacts.capture_queue_manifest_json points to a non-file path.",
                    artifact_key="capture_queue_manifest_json",
                    artifact_path=str(capture_queue_manifest_artifact_raw),
                )
        identity_model_switch_manifest_artifact_raw = contract_artifacts.get("identity_model_switch_manifest_json")
        if identity_model_switch_manifest_artifact_raw:
            identity_model_switch_manifest_contract_declared = True
            identity_model_switch_manifest_contract_required = True
            identity_model_switch_manifest_artifact_path = _repo_or_abs(str(identity_model_switch_manifest_artifact_raw))
            if not identity_model_switch_manifest_artifact_path.is_file():
                _add_issue(
                    failures,
                    "run_contract_artifact_identity_model_switch_manifest_missing",
                    "run_contract.artifacts.identity_model_switch_manifest_json points to a non-file path.",
                    artifact_key="identity_model_switch_manifest_json",
                    artifact_path=str(identity_model_switch_manifest_artifact_raw),
                )
        scene_membership_manifest_artifact_raw = contract_artifacts.get("scene_membership_manifest_json")
        if scene_membership_manifest_artifact_raw:
            scene_membership_manifest_contract_declared = True
            scene_membership_manifest_contract_required = True
            scene_membership_manifest_artifact_path = _repo_or_abs(str(scene_membership_manifest_artifact_raw))
            if not scene_membership_manifest_artifact_path.is_file():
                _add_issue(
                    failures,
                    "run_contract_artifact_scene_membership_manifest_missing",
                    "run_contract.artifacts.scene_membership_manifest_json points to a non-file path.",
                    artifact_key="scene_membership_manifest_json",
                    artifact_path=str(scene_membership_manifest_artifact_raw),
                )
        scene_membership_alignment_manifest_artifact_raw = contract_artifacts.get("scene_membership_alignment_manifest_json")
        if scene_membership_alignment_manifest_artifact_raw:
            scene_membership_alignment_manifest_contract_declared = True
            scene_membership_alignment_manifest_contract_required = True
            scene_membership_alignment_manifest_artifact_path = _repo_or_abs(str(scene_membership_alignment_manifest_artifact_raw))
            if not scene_membership_alignment_manifest_artifact_path.is_file():
                _add_issue(
                    failures,
                    "run_contract_artifact_scene_membership_alignment_manifest_missing",
                    "run_contract.artifacts.scene_membership_alignment_manifest_json points to a non-file path.",
                    artifact_key="scene_membership_alignment_manifest_json",
                    artifact_path=str(scene_membership_alignment_manifest_artifact_raw),
                )
        capture_matrix_alignment_manifest_artifact_raw = contract_artifacts.get("capture_matrix_alignment_manifest_json")
        if capture_matrix_alignment_manifest_artifact_raw:
            capture_matrix_alignment_manifest_contract_declared = True
            capture_matrix_alignment_manifest_contract_required = True
            capture_matrix_alignment_manifest_artifact_path = _repo_or_abs(str(capture_matrix_alignment_manifest_artifact_raw))
            if not capture_matrix_alignment_manifest_artifact_path.is_file():
                _add_issue(
                    failures,
                    "run_contract_artifact_capture_matrix_alignment_manifest_missing",
                    "run_contract.artifacts.capture_matrix_alignment_manifest_json points to a non-file path.",
                    artifact_key="capture_matrix_alignment_manifest_json",
                    artifact_path=str(capture_matrix_alignment_manifest_artifact_raw),
                )
        dataset_run_closure_manifest_artifact_raw = contract_artifacts.get("dataset_run_closure_manifest_json")
        if dataset_run_closure_manifest_artifact_raw:
            dataset_run_closure_manifest_contract_declared = True
            dataset_run_closure_manifest_contract_required = True
            dataset_run_closure_manifest_artifact_path = _repo_or_abs(str(dataset_run_closure_manifest_artifact_raw))
            if not dataset_run_closure_manifest_artifact_path.is_file():
                _add_issue(
                    failures,
                    "run_contract_artifact_dataset_run_closure_manifest_missing",
                    "run_contract.artifacts.dataset_run_closure_manifest_json points to a non-file path.",
                    artifact_key="dataset_run_closure_manifest_json",
                    artifact_path=str(dataset_run_closure_manifest_artifact_raw),
                )
        existing_scene_index_bridge_manifest_artifact_raw = contract_artifacts.get(
            "existing_scene_index_bridge_manifest_json"
        )
        if existing_scene_index_bridge_manifest_artifact_raw:
            existing_scene_index_bridge_manifest_contract_declared = True
            existing_scene_index_bridge_manifest_contract_required = True
            existing_scene_index_bridge_manifest_artifact_path = _repo_or_abs(
                str(existing_scene_index_bridge_manifest_artifact_raw)
            )
            if not existing_scene_index_bridge_manifest_artifact_path.is_file():
                _add_issue(
                    failures,
                    "run_contract_artifact_existing_scene_index_bridge_manifest_missing",
                    "run_contract.artifacts.existing_scene_index_bridge_manifest_json points to a non-file path.",
                    artifact_key="existing_scene_index_bridge_manifest_json",
                    artifact_path=str(existing_scene_index_bridge_manifest_artifact_raw),
                )
        sidecar_quality_manifest_artifact_raw = contract_artifacts.get("sidecar_quality_manifest_json")
        if sidecar_quality_manifest_artifact_raw:
            sidecar_quality_manifest_contract_declared = True
            sidecar_quality_manifest_contract_required = True
            sidecar_quality_manifest_artifact_path = _repo_or_abs(str(sidecar_quality_manifest_artifact_raw))
            if not sidecar_quality_manifest_artifact_path.is_file():
                _add_issue(
                    failures,
                    "run_contract_artifact_sidecar_quality_manifest_missing",
                    "run_contract.artifacts.sidecar_quality_manifest_json points to a non-file path.",
                    artifact_key="sidecar_quality_manifest_json",
                    artifact_path=str(sidecar_quality_manifest_artifact_raw),
                )
        dataset_gap_manifest_artifact_raw = contract_artifacts.get("dataset_gap_manifest_json")
        if dataset_gap_manifest_artifact_raw:
            dataset_gap_manifest_contract_declared = True
            dataset_gap_manifest_contract_required = True
            dataset_gap_manifest_artifact_path = _repo_or_abs(str(dataset_gap_manifest_artifact_raw))
            if not dataset_gap_manifest_artifact_path.is_file():
                _add_issue(
                    failures,
                    "run_contract_artifact_dataset_gap_manifest_missing",
                    "run_contract.artifacts.dataset_gap_manifest_json points to a non-file path.",
                    artifact_key="dataset_gap_manifest_json",
                    artifact_path=str(dataset_gap_manifest_artifact_raw),
                )
        sample_schema_coverage_manifest_artifact_raw = contract_artifacts.get("sample_schema_coverage_manifest_json")
        if sample_schema_coverage_manifest_artifact_raw:
            sample_schema_coverage_manifest_contract_declared = True
            sample_schema_coverage_manifest_contract_required = True
            sample_schema_coverage_manifest_artifact_path = _repo_or_abs(
                str(sample_schema_coverage_manifest_artifact_raw)
            )
            if not sample_schema_coverage_manifest_artifact_path.is_file():
                _add_issue(
                    failures,
                    "run_contract_artifact_sample_schema_coverage_manifest_missing",
                    "run_contract.artifacts.sample_schema_coverage_manifest_json points to a non-file path.",
                    artifact_key="sample_schema_coverage_manifest_json",
                    artifact_path=str(sample_schema_coverage_manifest_artifact_raw),
                )
        no_mask_non_promotion_manifest_artifact_raw = contract_artifacts.get("no_mask_non_promotion_manifest_json")
        if no_mask_non_promotion_manifest_artifact_raw:
            no_mask_non_promotion_manifest_contract_declared = True
            no_mask_non_promotion_manifest_contract_required = True
            no_mask_non_promotion_manifest_artifact_path = _repo_or_abs(
                str(no_mask_non_promotion_manifest_artifact_raw)
            )
            if not no_mask_non_promotion_manifest_artifact_path.is_file():
                _add_issue(
                    failures,
                    "run_contract_artifact_no_mask_non_promotion_manifest_missing",
                    "run_contract.artifacts.no_mask_non_promotion_manifest_json points to a non-file path.",
                    artifact_key="no_mask_non_promotion_manifest_json",
                    artifact_path=str(no_mask_non_promotion_manifest_artifact_raw),
                )
        deployment_episode_visibility_manifest_artifact_raw = contract_artifacts.get(
            "deployment_episode_visibility_manifest_json"
        )
        if deployment_episode_visibility_manifest_artifact_raw:
            deployment_episode_visibility_manifest_contract_declared = True
            deployment_episode_visibility_manifest_contract_required = True
            deployment_episode_visibility_manifest_artifact_path = _repo_or_abs(
                str(deployment_episode_visibility_manifest_artifact_raw)
            )
            if not deployment_episode_visibility_manifest_artifact_path.is_file():
                _add_issue(
                    failures,
                    "run_contract_artifact_deployment_episode_visibility_manifest_missing",
                    "run_contract.artifacts.deployment_episode_visibility_manifest_json points to a non-file path.",
                    artifact_key="deployment_episode_visibility_manifest_json",
                    artifact_path=str(deployment_episode_visibility_manifest_artifact_raw),
                )
        scene_sample_index_manifest_artifact_raw = contract_artifacts.get("scene_sample_index_manifest_json")
        if scene_sample_index_manifest_artifact_raw:
            scene_sample_index_manifest_contract_declared = True
            scene_sample_index_manifest_contract_required = True
            scene_sample_index_manifest_artifact_path = _repo_or_abs(str(scene_sample_index_manifest_artifact_raw))
            if not scene_sample_index_manifest_artifact_path.is_file():
                _add_issue(
                    failures,
                    "run_contract_artifact_scene_sample_index_manifest_missing",
                    "run_contract.artifacts.scene_sample_index_manifest_json points to a non-file path.",
                    artifact_key="scene_sample_index_manifest_json",
                    artifact_path=str(scene_sample_index_manifest_artifact_raw),
                )
        artifact_manifest_artifact_raw = contract_artifacts.get("artifact_manifest_json")
        artifact_manifest_contract_required = bool(artifact_manifest_artifact_raw)
        if artifact_manifest_contract_required:
            artifact_manifest_artifact_path = _repo_or_abs(str(artifact_manifest_artifact_raw))
            if not artifact_manifest_artifact_path.is_file():
                _add_issue(
                    failures,
                    "run_contract_artifact_artifact_manifest_missing",
                    "run_contract.artifacts.artifact_manifest_json points to a non-file path.",
                    artifact_key="artifact_manifest_json",
                    artifact_path=str(artifact_manifest_artifact_raw),
                )
        scene_roots = _as_list(run_contract_payload.get("scene_roots"))
        scene_root_count = _to_int(contract_counts.get("scene_root_count"))
        if scene_root_count != len(scene_roots):
            _add_issue(
                failures,
                "run_contract_scene_root_count_mismatch",
                "run_contract.counts.scene_root_count must equal len(scene_roots).",
                counts_scene_root_count=scene_root_count,
                scene_roots_len=len(scene_roots),
            )
    else:
        artifact_manifest_contract_required = False
        batch_run_manifest_contract_required = False
        capture_queue_manifest_contract_declared = False
        capture_queue_manifest_contract_required = False
        identity_model_switch_manifest_contract_declared = False
        identity_model_switch_manifest_contract_required = False
        scene_membership_manifest_contract_declared = False
        scene_membership_manifest_contract_required = False
        scene_membership_alignment_manifest_contract_declared = False
        scene_membership_alignment_manifest_contract_required = False
        capture_matrix_alignment_manifest_contract_declared = False
        capture_matrix_alignment_manifest_contract_required = False
        dataset_run_closure_manifest_contract_declared = False
        dataset_run_closure_manifest_contract_required = False
        existing_scene_index_bridge_manifest_contract_declared = False
        existing_scene_index_bridge_manifest_contract_required = False
        sidecar_quality_manifest_contract_declared = False
        sidecar_quality_manifest_contract_required = False
        dataset_gap_manifest_contract_declared = False
        dataset_gap_manifest_contract_required = False
        sample_schema_coverage_manifest_contract_declared = False
        sample_schema_coverage_manifest_contract_required = False
        no_mask_non_promotion_manifest_contract_declared = False
        no_mask_non_promotion_manifest_contract_required = False
        deployment_episode_visibility_manifest_contract_declared = False
        deployment_episode_visibility_manifest_contract_required = False
        scene_sample_index_manifest_contract_declared = False
        scene_sample_index_manifest_contract_required = False

    should_validate_batch_run_manifest = artifact_presence["batch_run_manifest"] or batch_run_manifest_contract_required
    if should_validate_batch_run_manifest:
        batch_run_manifest_payload, batch_run_manifest_err = _load_json(batch_run_manifest_path)
        if batch_run_manifest_err:
            _add_issue(
                failures,
                "batch_run_manifest_invalid",
                "batch_run_manifest.json is missing or invalid while present/contract-required.",
                error=batch_run_manifest_err,
            )
        else:
            if batch_run_manifest_payload.get("schema_version") != "carla_air_batch_run_manifest_v1":
                _add_issue(
                    failures,
                    "batch_run_manifest_schema_mismatch",
                    "batch_run_manifest schema_version mismatch.",
                    got=batch_run_manifest_payload.get("schema_version"),
                    expected="carla_air_batch_run_manifest_v1",
                )
            if batch_run_manifest_payload.get("starts_runtime") is not False:
                _add_issue(
                    failures,
                    "batch_run_manifest_starts_runtime_mismatch",
                    "batch_run_manifest.starts_runtime must be false.",
                    got=batch_run_manifest_payload.get("starts_runtime"),
                )
            if batch_run_manifest_payload.get("writes_scene_outputs") is not False:
                _add_issue(
                    failures,
                    "batch_run_manifest_writes_scene_outputs_mismatch",
                    "batch_run_manifest.writes_scene_outputs must be false.",
                    got=batch_run_manifest_payload.get("writes_scene_outputs"),
                )
            if batch_run_manifest_payload.get("non_promotion") is not True:
                _add_issue(
                    failures,
                    "batch_run_manifest_non_promotion_mismatch",
                    "batch_run_manifest.non_promotion must be true.",
                    got=batch_run_manifest_payload.get("non_promotion"),
                )
            if batch_run_manifest_payload.get("full_v1_live_dataset_ready") is not False:
                _add_issue(
                    failures,
                    "batch_run_manifest_full_v1_live_dataset_ready_mismatch",
                    "batch_run_manifest.full_v1_live_dataset_ready must be false.",
                    got=batch_run_manifest_payload.get("full_v1_live_dataset_ready"),
                )
            aligned_run_id = str(run_contract_payload.get("run_id") or "").strip() if run_contract_payload else ""
            observed_batch_run_id = str(batch_run_manifest_payload.get("run_id") or "").strip()
            if aligned_run_id and observed_batch_run_id and observed_batch_run_id != aligned_run_id:
                _add_issue(
                    failures,
                    "batch_run_manifest_run_id_mismatch",
                    "batch_run_manifest.run_id must align with run_contract.run_id.",
                    expected=aligned_run_id,
                    got=observed_batch_run_id,
                )
            if run_contract_payload:
                contract_counts = _as_dict(run_contract_payload.get("counts"))
                batch_counts = _as_dict(batch_run_manifest_payload.get("counts"))
                for count_key, expected_raw in contract_counts.items():
                    if count_key not in batch_counts:
                        continue
                    if _to_int(batch_counts.get(count_key)) != _to_int(expected_raw):
                        _add_issue(
                            failures,
                            "batch_run_manifest_count_mismatch",
                            "batch_run_manifest.counts field must align with run_contract.counts.",
                            count_key=count_key,
                            expected=_to_int(expected_raw),
                            got=_to_int(batch_counts.get(count_key)),
                        )
                contract_artifacts = _as_dict(run_contract_payload.get("artifacts"))
                batch_artifacts = _as_dict(batch_run_manifest_payload.get("artifact_paths"))
                for key in (
                    "run_contract_json",
                    "artifact_manifest_json",
                    "scene_discovery_manifest_json",
                    "dataset_plan_json",
                    "dataset_manifest_json",
                    "dataset_index_manifest_json",
                    "scene_sample_index_manifest_json",
                    "dataset_samples_jsonl",
                    "dataset_gap_manifest_json",
                    "dataset_splits_json",
                    "deployment_episodes_json",
                    "capture_queue_jsonl",
                    "capture_queue_manifest_json",
                    "scene_output_manifest_json",
                    "scene_membership_manifest_json",
                    "scene_membership_alignment_manifest_json",
                    "capture_matrix_alignment_manifest_json",
                    "dataset_run_closure_manifest_json",
                    "batch_run_manifest_json",
                ):
                    contract_path = str(contract_artifacts.get(key) or "").strip()
                    if not contract_path:
                        continue
                    batch_path = str(batch_artifacts.get(key) or "").strip()
                    if not batch_path:
                        _add_issue(
                            failures,
                            "batch_run_manifest_artifact_path_missing",
                            "batch_run_manifest.artifact_paths must cover contract-declared artifact keys.",
                            artifact_key=key,
                        )
                    elif batch_path != contract_path:
                        _add_issue(
                            failures,
                            "batch_run_manifest_artifact_path_mismatch",
                            "batch_run_manifest.artifact_paths entry must align with run_contract.artifacts.",
                            artifact_key=key,
                            expected=contract_path,
                            got=batch_path,
                        )
            child_runs = _as_list(batch_run_manifest_payload.get("child_runs"))
            if child_runs:
                child = _as_dict(child_runs[0])
                child_run_id = str(child.get("run_id") or "").strip()
                if observed_batch_run_id and child_run_id and child_run_id != observed_batch_run_id:
                    _add_issue(
                        failures,
                        "batch_run_manifest_child_run_id_mismatch",
                        "batch_run_manifest.child_runs[0].run_id must align with batch run_id for single-run batch.",
                        expected=observed_batch_run_id,
                        got=child_run_id,
                    )
                child_run_dir = str(child.get("run_dir") or "").strip()
                manifest_run_dir = str(batch_run_manifest_payload.get("run_dir") or "").strip()
                if manifest_run_dir and child_run_dir and child_run_dir != manifest_run_dir:
                    _add_issue(
                        failures,
                        "batch_run_manifest_child_run_dir_mismatch",
                        "batch_run_manifest.child_runs[0].run_dir must align with batch run_dir for single-run batch.",
                        expected=manifest_run_dir,
                        got=child_run_dir,
                    )
                child_scene_root_count = child.get("scene_root_count")
                if child_scene_root_count is not None and run_contract_payload:
                    contract_scene_root_count = _to_int(_as_dict(run_contract_payload.get("counts")).get("scene_root_count"))
                    if _to_int(child_scene_root_count) != contract_scene_root_count:
                        _add_issue(
                            failures,
                            "batch_run_manifest_child_scene_root_count_mismatch",
                            "batch_run_manifest child run scene_root_count must align with run_contract counts.",
                            expected=contract_scene_root_count,
                            got=_to_int(child_scene_root_count),
                        )
                child_sample_count = child.get("sample_count")
                if child_sample_count is not None and _to_int(child_sample_count) != sample_count:
                    _add_issue(
                        failures,
                        "batch_run_manifest_child_sample_count_mismatch",
                        "batch_run_manifest child run sample_count must align with observed sample count.",
                        expected=sample_count,
                        got=_to_int(child_sample_count),
                    )

    should_validate_scene_sample_index_manifest = (
        artifact_presence["scene_sample_index_manifest"] or scene_sample_index_manifest_contract_required
    )
    if should_validate_scene_sample_index_manifest:
        scene_sample_index_manifest_payload, scene_sample_index_manifest_err = _load_json(scene_sample_index_manifest_path)
        if scene_sample_index_manifest_err:
            _add_issue(
                failures,
                "scene_sample_index_manifest_invalid",
                "scene_sample_index_manifest.json is missing or invalid while present/contract-declared.",
                error=scene_sample_index_manifest_err,
            )
        else:
            if scene_sample_index_manifest_payload.get("schema_version") != SCHEMA_SCENE_SAMPLE_INDEX_MANIFEST:
                _add_issue(
                    failures,
                    "scene_sample_index_manifest_schema_mismatch",
                    "scene_sample_index_manifest schema_version mismatch.",
                    got=scene_sample_index_manifest_payload.get("schema_version"),
                    expected=SCHEMA_SCENE_SAMPLE_INDEX_MANIFEST,
                )
            for flag_name, expected_value in (
                ("starts_runtime", False),
                ("writes_scene_outputs", False),
                ("non_promotion", True),
                ("full_v1_live_dataset_ready", False),
            ):
                if scene_sample_index_manifest_payload.get(flag_name) is not expected_value:
                    _add_issue(
                        failures,
                        "scene_sample_index_manifest_guard_flag_mismatch",
                        "scene_sample_index_manifest guard flag mismatch.",
                        field=flag_name,
                        got=scene_sample_index_manifest_payload.get(flag_name),
                        expected=expected_value,
                    )
            aligned_run_id = str(run_contract_payload.get("run_id") or "").strip() if run_contract_payload else ""
            if not aligned_run_id:
                aligned_run_id = str(manifest_payload.get("run_id") or "").strip()
            observed_run_id = str(scene_sample_index_manifest_payload.get("run_id") or "").strip()
            if aligned_run_id and observed_run_id and observed_run_id != aligned_run_id:
                _add_issue(
                    failures,
                    "scene_sample_index_manifest_run_id_mismatch",
                    "scene_sample_index_manifest.run_id must align with run_contract/manifest run_id.",
                    expected=aligned_run_id,
                    got=observed_run_id,
                )
            if _to_int(scene_sample_index_manifest_payload.get("sample_count")) != sample_count:
                _add_issue(
                    failures,
                    "scene_sample_index_manifest_sample_count_mismatch",
                    "scene_sample_index_manifest.sample_count must equal dataset_samples.jsonl sample count.",
                    manifest_value=scene_sample_index_manifest_payload.get("sample_count"),
                    computed_value=sample_count,
                )
            split_distribution_from_manifest = {
                str(k).strip(): _to_int(v)
                for k, v in _as_dict(scene_sample_index_manifest_payload.get("split_distribution")).items()
                if str(k).strip()
            }
            if split_distribution_from_manifest != split_distribution:
                _add_issue(
                    failures,
                    "scene_sample_index_manifest_split_distribution_mismatch",
                    "scene_sample_index_manifest.split_distribution must match recomputed split distribution.",
                    manifest_value=split_distribution_from_manifest,
                    computed_value=split_distribution,
                )
            scene_entries = [_as_dict(item) for item in _as_list(scene_sample_index_manifest_payload.get("scene_entries"))]
            if _to_int(scene_sample_index_manifest_payload.get("scene_count")) != len(scene_entries):
                _add_issue(
                    failures,
                    "scene_sample_index_manifest_scene_count_entry_mismatch",
                    "scene_sample_index_manifest.scene_count must equal len(scene_entries).",
                    manifest_value=scene_sample_index_manifest_payload.get("scene_count"),
                    computed_value=len(scene_entries),
                )
            scene_membership: dict[str, dict[str, Any]] = {}
            for sample in samples:
                sample_obj = _as_dict(sample)
                source = _as_dict(sample_obj.get("source"))
                identity_id = str(sample_obj.get("identity_id") or "unknown_identity").strip() or "unknown_identity"
                trajectory_id = str(sample_obj.get("trajectory_id") or "unknown_trajectory").strip() or "unknown_trajectory"
                node_id = str(sample_obj.get("node_id") or "unknown_node").strip() or "unknown_node"
                scene_id = str(sample_obj.get("scene_id") or source.get("scene_id") or "").strip() or "unknown_scene"
                scene_dir = str(source.get("scene_dir") or source.get("scene_root") or "").strip()
                scene_key = _scene_key_for_fields(identity_id, trajectory_id, node_id, scene_id, scene_dir)
                split_name = str(sample_obj.get("split") or "unknown").strip() or "unknown"
                sample_id = str(sample_obj.get("sample_id") or "").strip()
                camera_id = str(sample_obj.get("camera_id") or "").strip()
                timestamp_obj = _as_dict(sample_obj.get("timestamp"))
                timestamp_us = str(sample_obj.get("timestamp_us") or timestamp_obj.get("timestamp_us") or "").strip()
                mask_gt_obj = _as_dict(sample_obj.get("mask_gt"))
                availability = str(mask_gt_obj.get("availability") or "").strip().lower()
                has_mask_gt = availability == "available"
                sidecar_complete = all(sample_obj.get(k) not in (None, "") for k in ("rgb", "depth", "semantic", "instance", "pose", "calib"))
                rec = scene_membership.setdefault(
                    scene_key,
                    {
                        "scene_key": scene_key,
                        "scene_id": scene_id,
                        "scene_dir": scene_dir,
                        "source_scene_root": scene_dir,
                        "identity_id": identity_id,
                        "trajectory_id": trajectory_id,
                        "node_id": node_id,
                        "sample_ids": [],
                        "split_names": set(),
                        "camera_ids": set(),
                        "timestamps_us": set(),
                        "first_timestamp_us": None,
                        "last_timestamp_us": None,
                        "mask_gt_available_count": 0,
                        "no_mask_sample_count": 0,
                        "sidecar_complete_count": 0,
                    },
                )
                if sample_id:
                    rec["sample_ids"].append(sample_id)
                rec["split_names"].add(split_name)
                if camera_id:
                    rec["camera_ids"].add(camera_id)
                if timestamp_us:
                    rec["timestamps_us"].add(timestamp_us)
                    rec["first_timestamp_us"] = timestamp_us if rec["first_timestamp_us"] is None else min(rec["first_timestamp_us"], timestamp_us)
                    rec["last_timestamp_us"] = timestamp_us if rec["last_timestamp_us"] is None else max(rec["last_timestamp_us"], timestamp_us)
                if has_mask_gt:
                    rec["mask_gt_available_count"] += 1
                else:
                    rec["no_mask_sample_count"] += 1
                if sidecar_complete:
                    rec["sidecar_complete_count"] += 1
            recomputed_scene_entries: list[dict[str, Any]] = []
            for scene_key in sorted(scene_membership):
                rec = scene_membership[scene_key]
                scene_sample_ids = [str(v).strip() for v in _as_list(rec.get("sample_ids")) if str(v).strip()]
                split_names = sorted({str(v).strip() for v in rec.get("split_names", set()) if str(v).strip()})
                camera_ids = sorted({str(v).strip() for v in rec.get("camera_ids", set()) if str(v).strip()})
                timestamp_us_sorted = sorted({str(v).strip() for v in rec.get("timestamps_us", set()) if str(v).strip()})
                recomputed_scene_entries.append(
                    {
                        "scene_key": str(rec.get("scene_key") or ""),
                        "scene_id": str(rec.get("scene_id") or ""),
                        "scene_dir": str(rec.get("scene_dir") or ""),
                        "source_scene_root": str(rec.get("source_scene_root") or ""),
                        "identity_id": str(rec.get("identity_id") or ""),
                        "trajectory_id": str(rec.get("trajectory_id") or ""),
                        "node_id": str(rec.get("node_id") or ""),
                        "split_names": split_names,
                        "camera_ids": camera_ids,
                        "sample_count": len(scene_sample_ids),
                        "timestamp_count": len(timestamp_us_sorted),
                        "first_sample_id": scene_sample_ids[0] if scene_sample_ids else None,
                        "last_sample_id": scene_sample_ids[-1] if scene_sample_ids else None,
                        "first_timestamp_us": rec.get("first_timestamp_us"),
                        "last_timestamp_us": rec.get("last_timestamp_us"),
                        "sample_ids_sorted_hash": _hash_text_parts(sorted(scene_sample_ids)),
                        "sample_id_order_hash": _hash_text_parts(scene_sample_ids),
                        "timestamp_us_sorted_hash": _hash_text_parts(timestamp_us_sorted),
                        "mask_gt_available_count": int(rec.get("mask_gt_available_count") or 0),
                        "no_mask_sample_count": int(rec.get("no_mask_sample_count") or 0),
                        "sidecar_complete_count": int(rec.get("sidecar_complete_count") or 0),
                    }
                )
            recomputed_scene_entries_by_key = {
                str(item.get("scene_key") or ""): item for item in recomputed_scene_entries if str(item.get("scene_key") or "").strip()
            }
            manifest_scene_entries_by_key = {
                str(item.get("scene_key") or ""): item for item in scene_entries if str(item.get("scene_key") or "").strip()
            }
            if sorted(manifest_scene_entries_by_key.keys()) != sorted(recomputed_scene_entries_by_key.keys()):
                _add_issue(
                    failures,
                    "scene_sample_index_manifest_scene_key_set_mismatch",
                    "scene_sample_index_manifest scene_key set must match recomputed scene keys.",
                    manifest_scene_count=len(manifest_scene_entries_by_key),
                    computed_scene_count=len(recomputed_scene_entries_by_key),
                )
            for scene_key, expected_entry in recomputed_scene_entries_by_key.items():
                observed_entry = _as_dict(manifest_scene_entries_by_key.get(scene_key))
                if not observed_entry:
                    continue
                if observed_entry != expected_entry:
                    _add_issue(
                        failures,
                        "scene_sample_index_manifest_scene_entry_mismatch",
                        "scene_sample_index_manifest scene entry mismatch against recomputed sample/index facts.",
                        scene_key=scene_key,
                    )
            recomputed_scene_keys_sorted_hash = _hash_text_parts([str(item.get("scene_key") or "") for item in recomputed_scene_entries])
            if str(scene_sample_index_manifest_payload.get("scene_keys_sorted_hash") or "").strip() != recomputed_scene_keys_sorted_hash:
                _add_issue(
                    failures,
                    "scene_sample_index_manifest_scene_keys_sorted_hash_mismatch",
                    "scene_sample_index_manifest.scene_keys_sorted_hash mismatch.",
                    manifest_value=scene_sample_index_manifest_payload.get("scene_keys_sorted_hash"),
                    computed_value=recomputed_scene_keys_sorted_hash,
                )
            recomputed_scene_split_membership_hash = _hash_text_parts(
                [
                    json.dumps(entry, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
                    for entry in recomputed_scene_entries
                ]
            )
            observed_scene_split_membership_hash = str(scene_sample_index_manifest_payload.get("scene_split_membership_hash") or "").strip()
            observed_index_scene_split_membership_hash = str(
                scene_sample_index_manifest_payload.get("dataset_index_scene_split_membership_hash") or ""
            ).strip()
            observed_scene_sample_index_hash = str(
                scene_sample_index_manifest_payload.get("scene_sample_index_hash") or ""
            ).strip()
            index_scene_split_membership_hash = str(dataset_index_manifest_payload.get("scene_split_membership_hash") or "").strip()
            index_scene_sample_index_hash = str(dataset_index_manifest_payload.get("scene_sample_index_hash") or "").strip()
            index_scene_keys_sorted_hash = str(dataset_index_manifest_payload.get("scene_keys_sorted_hash") or "").strip()
            index_scene_count = _to_int(dataset_index_manifest_payload.get("scene_count"))
            if observed_scene_split_membership_hash and index_scene_split_membership_hash and observed_scene_split_membership_hash != index_scene_split_membership_hash:
                _add_issue(
                    failures,
                    "scene_sample_index_manifest_scene_split_membership_hash_mismatch_index",
                    "scene_sample_index_manifest.scene_split_membership_hash must match dataset_index_manifest.scene_split_membership_hash.",
                    manifest_value=observed_scene_split_membership_hash,
                    index_manifest_value=index_scene_split_membership_hash,
                )
            if (
                observed_index_scene_split_membership_hash
                and index_scene_split_membership_hash
                and observed_index_scene_split_membership_hash != index_scene_split_membership_hash
            ):
                _add_issue(
                    failures,
                    "scene_sample_index_manifest_dataset_index_scene_split_membership_hash_mismatch",
                    "scene_sample_index_manifest.dataset_index_scene_split_membership_hash must match dataset_index_manifest.scene_split_membership_hash.",
                    manifest_value=observed_index_scene_split_membership_hash,
                    index_manifest_value=index_scene_split_membership_hash,
                )
            if observed_scene_sample_index_hash != recomputed_scene_split_membership_hash:
                _add_issue(
                    failures,
                    "scene_sample_index_manifest_scene_sample_index_hash_mismatch",
                    "scene_sample_index_manifest.scene_sample_index_hash mismatch against recomputed scene entries hash.",
                    manifest_value=observed_scene_sample_index_hash,
                    computed_value=recomputed_scene_split_membership_hash,
                )
            if dataset_index_manifest_payload and scene_sample_index_manifest_payload:
                if index_scene_sample_index_hash:
                    if index_scene_sample_index_hash != observed_scene_sample_index_hash:
                        _add_issue(
                            failures,
                            "dataset_index_manifest_scene_sample_index_hash_mismatch_scene_sample_index_manifest",
                            "dataset_index_manifest.scene_sample_index_hash must match scene_sample_index_manifest.scene_sample_index_hash.",
                            dataset_index_manifest_value=index_scene_sample_index_hash,
                            scene_sample_index_manifest_value=observed_scene_sample_index_hash,
                        )
                elif scene_sample_index_manifest_contract_required:
                    _add_issue(
                        failures,
                        "dataset_index_manifest_scene_sample_index_hash_missing_strict",
                        "dataset_index_manifest.scene_sample_index_hash is required for contract-declared scene_sample_index_manifest runs.",
                    )
                else:
                    _add_issue(
                        warnings,
                        "dataset_index_manifest_scene_sample_index_hash_missing_legacy_compatible",
                        "dataset_index_manifest.scene_sample_index_hash is missing; treated as legacy-compatible when scene_sample_index_manifest is not contract-required.",
                    )
                observed_scene_count = _to_int(scene_sample_index_manifest_payload.get("scene_count"))
                if index_scene_count and observed_scene_count and index_scene_count != observed_scene_count:
                    _add_issue(
                        failures,
                        "dataset_index_manifest_scene_count_mismatch_scene_sample_index_manifest",
                        "dataset_index_manifest.scene_count must match scene_sample_index_manifest.scene_count.",
                        dataset_index_manifest_value=index_scene_count,
                        scene_sample_index_manifest_value=observed_scene_count,
                    )
                observed_scene_keys_sorted_hash = str(scene_sample_index_manifest_payload.get("scene_keys_sorted_hash") or "").strip()
                if index_scene_keys_sorted_hash and observed_scene_keys_sorted_hash and (
                    index_scene_keys_sorted_hash != observed_scene_keys_sorted_hash
                ):
                    _add_issue(
                        failures,
                        "dataset_index_manifest_scene_keys_sorted_hash_mismatch_scene_sample_index_manifest",
                        "dataset_index_manifest.scene_keys_sorted_hash must match scene_sample_index_manifest.scene_keys_sorted_hash.",
                        dataset_index_manifest_value=index_scene_keys_sorted_hash,
                        scene_sample_index_manifest_value=observed_scene_keys_sorted_hash,
                    )
    else:
        _add_issue(
            warnings,
            "scene_sample_index_manifest_missing_legacy_compatible",
            "scene_sample_index_manifest.json is absent and not contract-declared; treated as legacy-compatible run.",
        )

    if not should_validate_batch_run_manifest:
        _add_issue(
            warnings,
            "batch_run_manifest_missing_legacy_compatible",
            "batch_run_manifest.json is absent; treated as legacy-compatible run.",
        )

    should_validate_scene_membership_manifest = artifact_presence["scene_membership_manifest"] or scene_membership_manifest_contract_required
    if should_validate_scene_membership_manifest:
        scene_membership_manifest_payload, scene_membership_manifest_err = _load_json(scene_membership_manifest_path)
        if scene_membership_manifest_err:
            _add_issue(
                failures,
                "scene_membership_manifest_invalid",
                "scene_membership_manifest.json is missing or invalid while present/contract-required.",
                error=scene_membership_manifest_err,
            )
    elif run_contract_payload:
        _add_issue(
            warnings,
            "scene_membership_manifest_missing_legacy_compatible",
            "scene_membership_manifest.json is absent and not contract-declared; treated as legacy-compatible run.",
        )

    should_validate_scene_membership_alignment_manifest = (
        artifact_presence["scene_membership_alignment_manifest"] or scene_membership_alignment_manifest_contract_required
    )
    if should_validate_scene_membership_alignment_manifest:
        scene_membership_alignment_manifest_payload, scene_membership_alignment_manifest_err = _load_json(
            scene_membership_alignment_manifest_path
        )
        if scene_membership_alignment_manifest_err:
            _add_issue(
                failures,
                "scene_membership_alignment_manifest_invalid",
                "scene_membership_alignment_manifest.json is missing or invalid while present/contract-required.",
                error=scene_membership_alignment_manifest_err,
            )
    elif run_contract_payload:
        _add_issue(
            warnings,
            "scene_membership_alignment_manifest_missing_legacy_compatible",
            "scene_membership_alignment_manifest.json is absent and not contract-declared; treated as legacy-compatible run.",
        )

    should_validate_capture_matrix_alignment_manifest = (
        artifact_presence["capture_matrix_alignment_manifest"] or capture_matrix_alignment_manifest_contract_required
    )
    if should_validate_capture_matrix_alignment_manifest:
        capture_matrix_alignment_manifest_payload, capture_matrix_alignment_manifest_err = _load_json(
            capture_matrix_alignment_manifest_path
        )
        if capture_matrix_alignment_manifest_err:
            _add_issue(
                failures,
                "capture_matrix_alignment_manifest_invalid",
                "capture_matrix_alignment_manifest.json is missing or invalid while present/contract-required.",
                error=capture_matrix_alignment_manifest_err,
            )
    elif run_contract_payload:
        _add_issue(
            warnings,
            "capture_matrix_alignment_manifest_missing_legacy_compatible",
            "capture_matrix_alignment_manifest.json is absent and not contract-declared; treated as legacy-compatible run.",
        )
    should_validate_dataset_run_closure_manifest = (
        artifact_presence["dataset_run_closure_manifest"] or dataset_run_closure_manifest_contract_required
    )
    if should_validate_dataset_run_closure_manifest:
        dataset_run_closure_manifest_payload, dataset_run_closure_manifest_err = _load_json(
            dataset_run_closure_manifest_path
        )
        if dataset_run_closure_manifest_err:
            _add_issue(
                failures,
                "dataset_run_closure_manifest_invalid",
                "dataset_run_closure_manifest.json is missing or invalid while present/contract-declared.",
                error=dataset_run_closure_manifest_err,
            )
    elif run_contract_payload:
        _add_issue(
            warnings,
            "dataset_run_closure_manifest_missing_legacy_compatible",
            "dataset_run_closure_manifest.json is absent and not contract-declared; treated as legacy-compatible run.",
        )

    accepted_scene_root_count = 0
    accepted_scene_root_keys: set[tuple[str, str, str]] = set()
    accepted_scene_root_paths: set[str] = set()
    should_validate_scene_discovery_manifest = artifact_presence["scene_discovery_manifest"] or scene_discovery_manifest_contract_required
    if should_validate_scene_discovery_manifest:
        scene_discovery_manifest_payload, scene_discovery_manifest_err = _load_json(scene_discovery_manifest_path)
        if scene_discovery_manifest_err:
            _add_issue(
                failures,
                "scene_discovery_manifest_invalid",
                "scene_discovery_manifest.json is missing or invalid.",
                error=scene_discovery_manifest_err,
            )
        else:
            if scene_discovery_manifest_payload.get("schema_version") != "carla_air_scene_discovery_manifest_v1":
                _add_issue(
                    failures,
                    "scene_discovery_manifest_schema_mismatch",
                    "scene_discovery_manifest schema_version mismatch.",
                    got=scene_discovery_manifest_payload.get("schema_version"),
                    expected="carla_air_scene_discovery_manifest_v1",
                )
            if run_contract_payload:
                expected_run_id = str(run_contract_payload.get("run_id") or "").strip()
                observed_run_id = str(scene_discovery_manifest_payload.get("run_id") or "").strip()
                if expected_run_id and observed_run_id and observed_run_id != expected_run_id:
                    _add_issue(
                        failures,
                        "scene_discovery_manifest_run_id_mismatch",
                        "scene_discovery_manifest.run_id must align with run_contract.run_id.",
                        expected=expected_run_id,
                        got=observed_run_id,
                    )
            scene_roots_value = scene_discovery_manifest_payload.get("scene_roots")
            scene_roots = _as_list(scene_roots_value)
            if not isinstance(scene_roots_value, list):
                _add_issue(
                    failures,
                    "scene_discovery_manifest_scene_roots_not_list",
                    "scene_discovery_manifest.scene_roots must be a list.",
                    got_type=type(scene_roots_value).__name__,
                )
            else:
                for idx, entry in enumerate(scene_roots, start=1):
                    obj = _as_dict(entry)
                    path_text = str(obj.get("path") or "").strip()
                    accepted = obj.get("accepted") is True
                    reject_reason = str(obj.get("reject_reason") or "").strip()
                    if accepted and reject_reason:
                        _add_issue(
                            failures,
                            "scene_discovery_manifest_accept_reject_conflict",
                            "accepted scene root must not carry reject_reason.",
                            index=idx,
                        )
                    if (not accepted) and ("reject_reason" in obj) and (not reject_reason):
                        _add_issue(
                            failures,
                            "scene_discovery_manifest_reject_reason_missing",
                            "rejected scene root should carry reject_reason when field is present.",
                            index=idx,
                        )
                    if accepted:
                        lowered_parts = {part.lower() for part in Path(path_text).parts}
                        if any(token in lowered_parts for token in FORBIDDEN_ROOT_TOKENS):
                            _add_issue(
                                failures,
                                "scene_discovery_manifest_accepted_forbidden_root",
                                "accepted scene root must not include weak/proxy token.",
                                index=idx,
                                path=path_text,
                            )
                        if path_text:
                            accepted_scene_root_paths.add(str(_repo_or_abs(path_text).resolve()))
                        accepted_scene_root_count += 1
                        accepted_scene_root_keys.add(
                            (
                                str(obj.get("node_id") or "").strip(),
                                str(obj.get("trajectory_id") or "").strip(),
                                str(obj.get("scene_id") or "").strip(),
                            )
                        )
            declared_count = _to_int(scene_discovery_manifest_payload.get("scene_root_count"))
            if declared_count != accepted_scene_root_count:
                _add_issue(
                    failures,
                    "scene_discovery_manifest_scene_root_count_mismatch",
                    "scene_discovery_manifest.scene_root_count must equal accepted scene roots count.",
                    declared_count=declared_count,
                    accepted_count=accepted_scene_root_count,
                )
            if run_contract_payload:
                contract_counts = _as_dict(run_contract_payload.get("counts"))
                contract_scene_root_count = _to_int(contract_counts.get("scene_root_count"))
                if contract_scene_root_count != accepted_scene_root_count:
                    _add_issue(
                        failures,
                        "scene_discovery_manifest_run_contract_scene_root_count_mismatch",
                        "scene_discovery_manifest accepted count must match run_contract.counts.scene_root_count.",
                        run_contract_scene_root_count=contract_scene_root_count,
                        accepted_count=accepted_scene_root_count,
                    )
                contract_scene_root_paths = {
                    str(_repo_or_abs(str(raw)).resolve())
                    for raw in _as_list(run_contract_payload.get("scene_roots"))
                    if str(raw or "").strip()
                }
                if contract_scene_root_paths and accepted_scene_root_paths != contract_scene_root_paths:
                    _add_issue(
                        failures,
                        "scene_discovery_manifest_run_contract_scene_roots_mismatch",
                        "scene_discovery_manifest accepted roots must match run_contract.scene_roots.",
                        run_contract_scene_root_count=len(contract_scene_root_paths),
                        accepted_scene_root_count=len(accepted_scene_root_paths),
                    )
            scene_observations = _as_list(manifest_payload.get("scene_observations")) if manifest_payload else []
            if scene_observations and len(scene_observations) != accepted_scene_root_count:
                _add_issue(
                    failures,
                    "scene_discovery_manifest_scene_observation_count_mismatch",
                    "scene_discovery_manifest accepted roots must match dataset_manifest.scene_observations count.",
                    accepted_scene_root_count=accepted_scene_root_count,
                    scene_observation_count=len(scene_observations),
                )
    else:
        _add_issue(
            warnings,
            "scene_discovery_manifest_missing_legacy_compatible",
            "scene_discovery_manifest.json is absent; treated as legacy run for compatibility.",
        )

    artifact_manifest_payload: dict[str, Any] = {}
    artifact_manifest_entry_count_excluding_self_reported: int | None = None
    artifact_manifest_self_artifact_key_reported = ""
    artifact_manifest_contract_artifact_count_including_self_reported: int | None = None
    artifact_manifest_excluded_self_reference_from_hashed_entries: bool | None = None
    should_validate_artifact_manifest = artifact_presence["artifact_manifest"] or artifact_manifest_contract_required
    if should_validate_artifact_manifest:
        artifact_manifest_payload, artifact_manifest_err = _load_json(artifact_manifest_path)
        if artifact_manifest_err:
            _add_issue(
                failures,
                "artifact_manifest_invalid",
                "artifact_manifest.json is missing or invalid.",
                error=artifact_manifest_err,
            )
        else:
            if artifact_manifest_payload.get("schema_version") != "carla_air_dataset_run_artifact_manifest_v1":
                _add_issue(
                    failures,
                    "artifact_manifest_schema_mismatch",
                    "artifact_manifest schema_version mismatch.",
                    got=artifact_manifest_payload.get("schema_version"),
                    expected="carla_air_dataset_run_artifact_manifest_v1",
                )
            if artifact_manifest_payload.get("starts_runtime") is not False:
                _add_issue(
                    failures,
                    "artifact_manifest_starts_runtime_mismatch",
                    "artifact_manifest.starts_runtime must be false.",
                    got=artifact_manifest_payload.get("starts_runtime"),
                )
            if artifact_manifest_payload.get("writes_scene_outputs") is not False:
                _add_issue(
                    failures,
                    "artifact_manifest_writes_scene_outputs_mismatch",
                    "artifact_manifest.writes_scene_outputs must be false.",
                    got=artifact_manifest_payload.get("writes_scene_outputs"),
                )
            expected_run_id = str(run_contract_payload.get("run_id") or "").strip() or str(plan_payload.get("run_id") or "").strip() or str(manifest_payload.get("run_id") or "").strip()
            if expected_run_id and str(artifact_manifest_payload.get("run_id") or "").strip() != expected_run_id:
                _add_issue(
                    failures,
                    "artifact_manifest_run_id_mismatch",
                    "artifact_manifest.run_id must align with run_contract/plan/manifest run_id.",
                    manifest_value=artifact_manifest_payload.get("run_id"),
                    expected=expected_run_id,
                )
            artifact_map = _as_dict(artifact_manifest_payload.get("artifacts"))
            declared_count = _to_int(artifact_manifest_payload.get("artifact_count"))
            if declared_count != len(artifact_map):
                _add_issue(
                    failures,
                    "artifact_manifest_count_mismatch",
                    "artifact_manifest.artifact_count must equal number of artifacts entries.",
                    declared_count=declared_count,
                    observed_count=len(artifact_map),
                )
            manifest_accounting = _as_dict(artifact_manifest_payload.get("artifact_accounting"))
            if manifest_accounting:
                artifact_manifest_entry_count_excluding_self_reported = _to_int(
                    manifest_accounting.get("artifact_manifest_entry_count_excluding_self")
                )
                artifact_manifest_self_artifact_key_reported = str(manifest_accounting.get("self_artifact_key") or "").strip()
                artifact_manifest_contract_artifact_count_including_self_reported = _to_int(
                    manifest_accounting.get("contract_artifact_count_including_self")
                )
                excluded_self_raw = manifest_accounting.get("excluded_self_reference_from_hashed_entries")
                if isinstance(excluded_self_raw, bool):
                    artifact_manifest_excluded_self_reference_from_hashed_entries = excluded_self_raw
                if artifact_manifest_entry_count_excluding_self_reported != len(artifact_map):
                    _add_issue(
                        failures,
                        "artifact_manifest_artifact_accounting_entry_count_mismatch",
                        "artifact_manifest.artifact_accounting.artifact_manifest_entry_count_excluding_self must equal len(artifact_manifest.artifacts).",
                        reported_count=artifact_manifest_entry_count_excluding_self_reported,
                        observed_count=len(artifact_map),
                    )
                if artifact_manifest_entry_count_excluding_self_reported != declared_count:
                    _add_issue(
                        failures,
                        "artifact_manifest_artifact_accounting_declared_count_mismatch",
                        "artifact_manifest.artifact_accounting.artifact_manifest_entry_count_excluding_self must equal artifact_manifest.artifact_count.",
                        reported_count=artifact_manifest_entry_count_excluding_self_reported,
                        declared_count=declared_count,
                    )
                if not artifact_manifest_self_artifact_key_reported:
                    _add_issue(
                        failures,
                        "artifact_manifest_artifact_accounting_self_key_missing",
                        "artifact_manifest.artifact_accounting.self_artifact_key must be non-empty when artifact_accounting is present.",
                    )
                elif artifact_manifest_self_artifact_key_reported in artifact_map:
                    _add_issue(
                        failures,
                        "artifact_manifest_artifact_accounting_self_key_in_hashed_entries",
                        "artifact_manifest.artifact_accounting.self_artifact_key must be excluded from artifact_manifest.artifacts hashed entries.",
                        self_artifact_key=artifact_manifest_self_artifact_key_reported,
                    )
                if artifact_manifest_excluded_self_reference_from_hashed_entries is not True:
                    _add_issue(
                        failures,
                        "artifact_manifest_artifact_accounting_exclusion_flag_mismatch",
                        "artifact_manifest.artifact_accounting.excluded_self_reference_from_hashed_entries must be true when present.",
                        got=excluded_self_raw,
                    )
            else:
                _add_issue(
                    warnings,
                    "artifact_manifest_artifact_accounting_missing_legacy_compatible",
                    "artifact_manifest.artifact_accounting missing; treated as legacy-compatible run.",
                )
            for key, raw_entry in artifact_map.items():
                entry = _as_dict(raw_entry)
                path_raw = entry.get("path")
                path = _repo_or_abs(str(path_raw or "")) if path_raw else Path("")
                exists = bool(entry.get("exists"))
                if (not path_raw) or (not path.is_file()):
                    _add_issue(
                        failures,
                        "artifact_manifest_entry_file_missing",
                        "artifact_manifest artifact entry path is missing or not a file.",
                        artifact_key=key,
                        artifact_path=str(path_raw or ""),
                    )
                    continue
                if exists is not True:
                    _add_issue(
                        failures,
                        "artifact_manifest_entry_exists_mismatch",
                        "artifact_manifest artifact entry exists must be true when file exists.",
                        artifact_key=key,
                        manifest_exists=entry.get("exists"),
                    )
                size_bytes = _to_int(entry.get("size_bytes"))
                actual_size_bytes = path.stat().st_size
                if size_bytes != actual_size_bytes:
                    _add_issue(
                        failures,
                        "artifact_manifest_entry_size_mismatch",
                        "artifact_manifest artifact size_bytes mismatch.",
                        artifact_key=key,
                        manifest_value=size_bytes,
                        actual_value=actual_size_bytes,
                    )
                expected_sha = str(entry.get("sha256") or "").strip()
                if not expected_sha:
                    _add_issue(
                        failures,
                        "artifact_manifest_entry_sha256_missing",
                        "artifact_manifest artifact sha256 must be non-empty.",
                        artifact_key=key,
                    )
                else:
                    actual_sha = _sha256_file(path)
                    if expected_sha != actual_sha:
                        _add_issue(
                            failures,
                            "artifact_manifest_entry_sha256_mismatch",
                            "artifact_manifest artifact sha256 mismatch.",
                            artifact_key=key,
                            manifest_value=expected_sha,
                            actual_value=actual_sha,
                        )
                expected_schema = entry.get("schema_version")
                actual_schema = _schema_version_if_json(path)
                if expected_schema is not None and str(expected_schema) != str(actual_schema):
                    _add_issue(
                        failures,
                        "artifact_manifest_entry_schema_version_mismatch",
                        "artifact_manifest artifact schema_version mismatch for JSON artifact.",
                        artifact_key=key,
                        manifest_value=expected_schema,
                        actual_value=actual_schema,
                    )
                expected_rows = entry.get("row_count")
                if path.suffix.lower() == ".jsonl":
                    actual_rows = _jsonl_row_count(path)
                    if _to_int(expected_rows) != actual_rows:
                        _add_issue(
                            failures,
                            "artifact_manifest_entry_row_count_mismatch",
                            "artifact_manifest artifact row_count mismatch for JSONL artifact.",
                            artifact_key=key,
                            manifest_value=expected_rows,
                            actual_value=actual_rows,
                        )
            if run_contract_payload:
                contract_artifacts = _as_dict(run_contract_payload.get("artifacts"))
                inferred_self_key = artifact_manifest_self_artifact_key_reported or contract_self_artifact_key_reported or "artifact_manifest_json"
                if inferred_self_key in artifact_map:
                    _add_issue(
                        failures,
                        "artifact_manifest_self_reference_not_excluded",
                        "artifact_manifest self artifact key must be excluded from artifact_manifest.artifacts hashed entries.",
                        self_artifact_key=inferred_self_key,
                    )
                if (
                    artifact_manifest_contract_artifact_count_including_self_reported is not None
                    and artifact_manifest_contract_artifact_count_including_self_reported != len(contract_artifacts)
                ):
                    _add_issue(
                        failures,
                        "artifact_manifest_contract_artifact_count_including_self_mismatch",
                        "artifact_manifest.artifact_accounting.contract_artifact_count_including_self must align with len(run_contract.artifacts).",
                        reported_count=artifact_manifest_contract_artifact_count_including_self_reported,
                        observed_count=len(contract_artifacts),
                    )
                if (
                    contract_artifact_count_including_self_reported is not None
                    and artifact_manifest_entry_count_excluding_self_reported is not None
                    and contract_artifact_count_including_self_reported
                    != artifact_manifest_entry_count_excluding_self_reported + 1
                ):
                    _add_issue(
                        failures,
                        "artifact_accounting_cross_file_count_mismatch",
                        "artifact accounting must satisfy contract_including_self = manifest_excluding_self + 1.",
                        contract_artifact_count_including_self=contract_artifact_count_including_self_reported,
                        artifact_manifest_entry_count_excluding_self=artifact_manifest_entry_count_excluding_self_reported,
                    )
                for key in contract_artifacts.keys():
                    if key == "artifact_manifest_json":
                        continue
                    if key not in artifact_map:
                        _add_issue(
                            failures,
                            "artifact_manifest_missing_contract_artifact_key",
                            "artifact_manifest.artifacts missing key declared by run_contract.artifacts.",
                            artifact_key=key,
                        )
    else:
        _add_issue(
            warnings,
            "artifact_manifest_missing_legacy_compatible",
            "artifact_manifest.json is absent; treated as legacy run for compatibility.",
        )

    should_validate_identity_model_switch_manifest = (
        artifact_presence["identity_model_switch_manifest"] or identity_model_switch_manifest_contract_declared
    )
    if should_validate_identity_model_switch_manifest:
        identity_model_switch_manifest_payload, identity_model_switch_manifest_err = _load_json(identity_model_switch_manifest_path)
        if identity_model_switch_manifest_err:
            _add_issue(
                failures,
                "identity_model_switch_manifest_invalid",
                "identity_model_switch_manifest.json is missing or invalid while present/contract-declared.",
                error=identity_model_switch_manifest_err,
            )
        else:
            if identity_model_switch_manifest_payload.get("schema_version") != "carla_air_identity_model_switch_manifest_v1":
                _add_issue(
                    failures,
                    "identity_model_switch_manifest_schema_mismatch",
                    "identity_model_switch_manifest schema_version mismatch.",
                    got=identity_model_switch_manifest_payload.get("schema_version"),
                    expected="carla_air_identity_model_switch_manifest_v1",
                )
            for flag_name, expected_value in (
                ("starts_runtime", False),
                ("writes_scene_outputs", False),
                ("non_promotion", True),
                ("full_v1_live_dataset_ready", False),
                ("no_silent_identity_rewrite", True),
                ("legacy_or_observed_scene_passthrough_allowed_for_no_mask_index", True),
            ):
                if identity_model_switch_manifest_payload.get(flag_name) is not expected_value:
                    _add_issue(
                        failures,
                        "identity_model_switch_manifest_guard_flag_mismatch",
                        "identity_model_switch_manifest guard flag mismatch.",
                        field=flag_name,
                        got=identity_model_switch_manifest_payload.get(flag_name),
                        expected=expected_value,
                    )
            manifest_run_id = str(identity_model_switch_manifest_payload.get("run_id") or "").strip()
            expected_run_id = (
                str(run_contract_payload.get("run_id") or "").strip()
                or str(plan_payload.get("run_id") or "").strip()
                or str(manifest_payload.get("run_id") or "").strip()
            )
            if expected_run_id and manifest_run_id != expected_run_id:
                _add_issue(
                    failures,
                    "identity_model_switch_manifest_run_id_mismatch",
                    "identity_model_switch_manifest.run_id must align with run_contract/plan/manifest run_id.",
                    manifest_value=manifest_run_id,
                    expected=expected_run_id,
                )
            if identity_model_switch_manifest_payload.get("capture_profile") != plan_payload.get("capture_profile"):
                _add_issue(
                    failures,
                    "identity_model_switch_manifest_capture_profile_mismatch",
                    "identity_model_switch_manifest.capture_profile must match plan.capture_profile.",
                    manifest_value=identity_model_switch_manifest_payload.get("capture_profile"),
                    plan_value=plan_payload.get("capture_profile"),
                )
            manifest_planned_identity_ids = sorted(
                {str(x).strip() for x in _as_list(identity_model_switch_manifest_payload.get("planned_identity_ids")) if str(x).strip()}
            )
            expected_planned_identity_ids = sorted(plan_identities)
            if manifest_planned_identity_ids != expected_planned_identity_ids:
                _add_issue(
                    failures,
                    "identity_model_switch_manifest_planned_identity_ids_mismatch",
                    "identity_model_switch_manifest.planned_identity_ids must match plan identities.",
                    manifest_value=manifest_planned_identity_ids,
                    expected=expected_planned_identity_ids,
                )
            manifest_profiles = [p for p in _as_list(identity_model_switch_manifest_payload.get("identity_model_profiles")) if isinstance(p, dict)]
            if _to_int(identity_model_switch_manifest_payload.get("profile_count")) != len(plan_profiles):
                _add_issue(
                    failures,
                    "identity_model_switch_manifest_profile_count_mismatch",
                    "identity_model_switch_manifest.profile_count must match len(plan.identity_model_profiles).",
                    manifest_value=identity_model_switch_manifest_payload.get("profile_count"),
                    expected=len(plan_profiles),
                )
            if _to_int(identity_model_switch_manifest_payload.get("identity_count")) != len(expected_planned_identity_ids):
                _add_issue(
                    failures,
                    "identity_model_switch_manifest_identity_count_mismatch",
                    "identity_model_switch_manifest.identity_count must match planned identity count.",
                    manifest_value=identity_model_switch_manifest_payload.get("identity_count"),
                    expected=len(expected_planned_identity_ids),
                )
            plan_switch_methods = sorted(
                {str(_as_dict(p).get("switch_method") or "").strip() for p in plan_profiles if str(_as_dict(p).get("switch_method") or "").strip()}
            )
            plan_model_labels = sorted(
                {str(_as_dict(p).get("model_label") or "").strip() for p in plan_profiles if str(_as_dict(p).get("model_label") or "").strip()}
            )
            manifest_switch_methods = sorted(
                {str(x).strip() for x in _as_list(identity_model_switch_manifest_payload.get("switch_methods")) if str(x).strip()}
            )
            manifest_model_labels = sorted(
                {str(x).strip() for x in _as_list(identity_model_switch_manifest_payload.get("model_labels")) if str(x).strip()}
            )
            if manifest_switch_methods != plan_switch_methods:
                _add_issue(
                    failures,
                    "identity_model_switch_manifest_switch_methods_mismatch",
                    "identity_model_switch_manifest.switch_methods must match plan identity_model_profiles switch methods.",
                    manifest_value=manifest_switch_methods,
                    expected=plan_switch_methods,
                )
            if manifest_model_labels != plan_model_labels:
                _add_issue(
                    failures,
                    "identity_model_switch_manifest_model_labels_mismatch",
                    "identity_model_switch_manifest.model_labels must match plan identity_model_profiles model labels.",
                    manifest_value=manifest_model_labels,
                    expected=plan_model_labels,
                )
            plan_requires_readback_flags = sorted(
                {(_as_dict(p).get("requires_ue_carla_import_readback") is True) for p in plan_profiles}
            )
            manifest_requires_readback_flags = sorted(
                {bool(x) for x in _as_list(identity_model_switch_manifest_payload.get("requires_ue_carla_import_readback_flags"))}
            )
            if manifest_requires_readback_flags != plan_requires_readback_flags:
                _add_issue(
                    failures,
                    "identity_model_switch_manifest_requires_import_readback_flags_mismatch",
                    "identity_model_switch_manifest.requires_ue_carla_import_readback_flags must match plan profiles.",
                    manifest_value=manifest_requires_readback_flags,
                    expected=plan_requires_readback_flags,
                )
            if _to_int(identity_model_switch_manifest_payload.get("capture_task_count")) != plan_capture_task_count:
                _add_issue(
                    failures,
                    "identity_model_switch_manifest_capture_task_count_mismatch",
                    "identity_model_switch_manifest.capture_task_count must match plan capture task count.",
                    manifest_value=identity_model_switch_manifest_payload.get("capture_task_count"),
                    expected=plan_capture_task_count,
                )
            blocked_capture_task_count_expected = _to_int(plan_counts.get("blocked_capture_task_count"))
            if blocked_capture_task_count_expected == 0:
                blocked_capture_task_count_expected = sum(1 for task in plan_capture_tasks if _as_dict(task).get("capture_allowed_now") is not True)
            if _to_int(identity_model_switch_manifest_payload.get("blocked_capture_task_count")) != blocked_capture_task_count_expected:
                _add_issue(
                    failures,
                    "identity_model_switch_manifest_blocked_capture_task_count_mismatch",
                    "identity_model_switch_manifest.blocked_capture_task_count must match plan blocked capture task count.",
                    manifest_value=identity_model_switch_manifest_payload.get("blocked_capture_task_count"),
                    expected=blocked_capture_task_count_expected,
                )
            manifest_profile_ids = sorted(
                {str(_as_dict(p).get("identity_model_profile_id") or "").strip() for p in manifest_profiles if str(_as_dict(p).get("identity_model_profile_id") or "").strip()}
            )
            plan_profile_ids = sorted(
                {str(_as_dict(p).get("identity_model_profile_id") or "").strip() for p in plan_profiles if str(_as_dict(p).get("identity_model_profile_id") or "").strip()}
            )
            if manifest_profile_ids != plan_profile_ids:
                _add_issue(
                    failures,
                    "identity_model_switch_manifest_profile_ids_mismatch",
                    "identity_model_switch_manifest.identity_model_profiles profile ids must match plan identity_model_profiles.",
                    manifest_value=manifest_profile_ids,
                    expected=plan_profile_ids,
                )
            manifest_identity_switch_contract = _as_dict(manifest_payload.get("identity_model_switch_contract"))
            if manifest_identity_switch_contract:
                expected_observed_ids = sorted(
                    {str(x).strip() for x in _as_list(manifest_identity_switch_contract.get("observed_sample_identity_ids")) if str(x).strip()}
                )
                observed_ids = sorted(
                    {str(x).strip() for x in _as_list(identity_model_switch_manifest_payload.get("observed_sample_identity_ids")) if str(x).strip()}
                )
                if observed_ids != expected_observed_ids:
                    _add_issue(
                        failures,
                        "identity_model_switch_manifest_observed_ids_mismatch",
                        "identity_model_switch_manifest.observed_sample_identity_ids must match dataset_manifest.identity_model_switch_contract.",
                        manifest_value=observed_ids,
                        expected=expected_observed_ids,
                    )
                if _to_int(identity_model_switch_manifest_payload.get("identity_mismatch_count")) != _to_int(
                    manifest_identity_switch_contract.get("identity_mismatch_count")
                ):
                    _add_issue(
                        failures,
                        "identity_model_switch_manifest_identity_mismatch_count_mismatch",
                        "identity_model_switch_manifest.identity_mismatch_count must match dataset_manifest.identity_model_switch_contract.",
                        manifest_value=identity_model_switch_manifest_payload.get("identity_mismatch_count"),
                        expected=manifest_identity_switch_contract.get("identity_mismatch_count"),
                    )
    else:
        _add_issue(
            warnings,
            "identity_model_switch_manifest_missing_legacy_compatible",
            "identity_model_switch_manifest.json is absent; treated as legacy run for compatibility.",
        )

    should_validate_existing_scene_index_bridge_manifest = (
        artifact_presence["existing_scene_index_bridge_manifest"] or existing_scene_index_bridge_manifest_contract_declared
    )
    if should_validate_existing_scene_index_bridge_manifest:
        existing_scene_index_bridge_manifest_payload, existing_scene_index_bridge_manifest_err = _load_json(
            existing_scene_index_bridge_manifest_path
        )
        if existing_scene_index_bridge_manifest_err:
            _add_issue(
                failures,
                "existing_scene_index_bridge_manifest_invalid",
                "existing_scene_index_bridge_manifest.json is missing or invalid while present/contract-declared.",
                error=existing_scene_index_bridge_manifest_err,
            )
        else:
            if (
                existing_scene_index_bridge_manifest_payload.get("schema_version")
                != "carla_air_existing_scene_index_bridge_manifest_v1"
            ):
                _add_issue(
                    failures,
                    "existing_scene_index_bridge_manifest_schema_mismatch",
                    "existing_scene_index_bridge_manifest schema_version mismatch.",
                    got=existing_scene_index_bridge_manifest_payload.get("schema_version"),
                    expected="carla_air_existing_scene_index_bridge_manifest_v1",
                )
            for flag_name, expected_value in (
                ("starts_runtime", False),
                ("writes_scene_outputs", False),
                ("non_promotion", True),
                ("full_v1_live_dataset_ready", False),
            ):
                if existing_scene_index_bridge_manifest_payload.get(flag_name) is not expected_value:
                    _add_issue(
                        failures,
                        "existing_scene_index_bridge_manifest_guard_flag_mismatch",
                        "existing_scene_index_bridge_manifest guard flag mismatch.",
                        field=flag_name,
                        got=existing_scene_index_bridge_manifest_payload.get(flag_name),
                        expected=expected_value,
                    )
            expected_run_id = (
                str(run_contract_payload.get("run_id") or "").strip()
                or str(plan_payload.get("run_id") or "").strip()
                or str(manifest_payload.get("run_id") or "").strip()
            )
            observed_run_id = str(existing_scene_index_bridge_manifest_payload.get("run_id") or "").strip()
            if expected_run_id and observed_run_id != expected_run_id:
                _add_issue(
                    failures,
                    "existing_scene_index_bridge_manifest_run_id_mismatch",
                    "existing_scene_index_bridge_manifest.run_id must align with run_contract/plan/manifest run_id.",
                    manifest_value=observed_run_id,
                    expected=expected_run_id,
                )

            computed_scene_entries: dict[str, dict[str, Any]] = {}
            computed_mask_gt_available_count = 0
            computed_no_mask_sample_count = 0
            for sample in samples:
                sample_obj = _as_dict(sample)
                source = _as_dict(sample_obj.get("source"))
                identity_id = str(sample_obj.get("identity_id") or "").strip()
                trajectory_id = str(sample_obj.get("trajectory_id") or "").strip()
                node_id = str(sample_obj.get("node_id") or "").strip()
                scene_id = str(sample_obj.get("scene_id") or source.get("scene_id") or "").strip()
                scene_dir = str(source.get("scene_dir") or source.get("scene_root") or "").strip()
                scene_key = _scene_key_for_fields(identity_id, trajectory_id, node_id, scene_id, scene_dir)
                rec = computed_scene_entries.setdefault(
                    scene_key,
                    {
                        "scene_id": scene_id,
                        "scene_key": scene_key,
                        "scene_dir": scene_dir,
                        "scene_root": scene_dir,
                        "identity_id": identity_id or "unknown_identity",
                        "trajectory_id": trajectory_id or "unknown_trajectory",
                        "node_id": node_id or "unknown_node",
                        "split_names": set(),
                        "camera_ids": set(),
                        "sample_count": 0,
                        "timestamp_us_values": set(),
                        "mask_gt_available_count": 0,
                        "no_mask_sample_count": 0,
                        "sidecar_complete_count": 0,
                    },
                )
                split_name = str(sample_obj.get("split") or "").strip()
                if split_name:
                    rec["split_names"].add(split_name)
                camera_id = str(sample_obj.get("camera_id") or "").strip()
                if camera_id:
                    rec["camera_ids"].add(camera_id)
                rec["sample_count"] += 1
                ts = _as_dict(sample_obj.get("timestamp"))
                ts_us_raw = ts.get("timestamp_us")
                ts_us = None
                try:
                    if ts_us_raw is not None and str(ts_us_raw).strip():
                        ts_us = int(ts_us_raw)
                except Exception:
                    ts_us = None
                if ts_us is not None:
                    rec["timestamp_us_values"].add(ts_us)
                mask_gt = _as_dict(sample_obj.get("mask_gt"))
                mask_gt_available = mask_gt.get("present") is True or str(mask_gt.get("availability") or "").strip() == "available"
                if mask_gt_available:
                    rec["mask_gt_available_count"] += 1
                    computed_mask_gt_available_count += 1
                else:
                    rec["no_mask_sample_count"] += 1
                    computed_no_mask_sample_count += 1
                refs = _as_dict(sample_obj.get("refs"))
                if all(bool(refs.get(k)) for k in ("rgb", "depth", "semantic", "instance", "pose", "calib")):
                    rec["sidecar_complete_count"] += 1

            computed_scene_keys_sorted = sorted(computed_scene_entries.keys())
            computed_scene_keys_sorted_hash = _hash_text_parts(computed_scene_keys_sorted)
            computed_scene_split_membership_hash = ""
            if dataset_index_manifest_payload:
                computed_scene_split_membership_hash = str(
                    dataset_index_manifest_payload.get("scene_split_membership_hash") or ""
                ).strip()
            if not computed_scene_split_membership_hash and scene_membership_manifest_payload:
                scene_entries_for_hash = [
                    _as_dict(x) for x in _as_list(scene_membership_manifest_payload.get("scene_entries")) if isinstance(x, dict)
                ]
                hash_rows = []
                for entry in scene_entries_for_hash:
                    row_scene_key = _scene_key_for_fields(
                        str(entry.get("identity_id") or "").strip(),
                        str(entry.get("trajectory_id") or "").strip(),
                        str(entry.get("node_id") or "").strip(),
                        str(entry.get("scene_id") or "").strip(),
                        str(entry.get("scene_dir") or entry.get("scene_root") or "").strip(),
                    )
                    hash_rows.append(
                        json.dumps(
                            {
                                "scene_key": row_scene_key,
                                "split_names": sorted(
                                    {str(v).strip() for v in _as_list(entry.get("split_names")) if str(v).strip()}
                                ),
                                "sample_count": _to_int(entry.get("sample_count")),
                            },
                            ensure_ascii=True,
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                    )
                computed_scene_split_membership_hash = _hash_text_parts(sorted(hash_rows))

            if _to_int(existing_scene_index_bridge_manifest_payload.get("scene_root_count")) != len(
                _as_list(manifest_payload.get("scene_observations"))
            ):
                _add_issue(
                    failures,
                    "existing_scene_index_bridge_manifest_scene_root_count_mismatch",
                    "existing_scene_index_bridge_manifest.scene_root_count must match dataset_manifest.scene_observations count.",
                    manifest_value=existing_scene_index_bridge_manifest_payload.get("scene_root_count"),
                    computed_value=len(_as_list(manifest_payload.get("scene_observations"))),
                )
            if _to_int(existing_scene_index_bridge_manifest_payload.get("indexed_scene_count")) != len(computed_scene_entries):
                _add_issue(
                    failures,
                    "existing_scene_index_bridge_manifest_indexed_scene_count_mismatch",
                    "existing_scene_index_bridge_manifest.indexed_scene_count mismatch.",
                    manifest_value=existing_scene_index_bridge_manifest_payload.get("indexed_scene_count"),
                    computed_value=len(computed_scene_entries),
                )
            if _to_int(existing_scene_index_bridge_manifest_payload.get("sample_count")) != sample_count:
                _add_issue(
                    failures,
                    "existing_scene_index_bridge_manifest_sample_count_mismatch",
                    "existing_scene_index_bridge_manifest.sample_count mismatch.",
                    manifest_value=existing_scene_index_bridge_manifest_payload.get("sample_count"),
                    computed_value=sample_count,
                )
            if _to_int(existing_scene_index_bridge_manifest_payload.get("mask_gt_available_count")) != computed_mask_gt_available_count:
                _add_issue(
                    failures,
                    "existing_scene_index_bridge_manifest_mask_gt_available_count_mismatch",
                    "existing_scene_index_bridge_manifest.mask_gt_available_count mismatch.",
                    manifest_value=existing_scene_index_bridge_manifest_payload.get("mask_gt_available_count"),
                    computed_value=computed_mask_gt_available_count,
                )
            if _to_int(existing_scene_index_bridge_manifest_payload.get("no_mask_sample_count")) != computed_no_mask_sample_count:
                _add_issue(
                    failures,
                    "existing_scene_index_bridge_manifest_no_mask_sample_count_mismatch",
                    "existing_scene_index_bridge_manifest.no_mask_sample_count mismatch.",
                    manifest_value=existing_scene_index_bridge_manifest_payload.get("no_mask_sample_count"),
                    computed_value=computed_no_mask_sample_count,
                )
            observed_scene_keys_sorted_hash = str(
                existing_scene_index_bridge_manifest_payload.get("scene_keys_sorted_hash") or ""
            ).strip()
            if observed_scene_keys_sorted_hash != computed_scene_keys_sorted_hash:
                _add_issue(
                    failures,
                    "existing_scene_index_bridge_manifest_scene_keys_sorted_hash_mismatch",
                    "existing_scene_index_bridge_manifest.scene_keys_sorted_hash mismatch.",
                    manifest_value=observed_scene_keys_sorted_hash,
                    computed_value=computed_scene_keys_sorted_hash,
                )
            observed_scene_split_membership_hash = str(
                existing_scene_index_bridge_manifest_payload.get("scene_split_membership_hash") or ""
            ).strip()
            if computed_scene_split_membership_hash and observed_scene_split_membership_hash != computed_scene_split_membership_hash:
                _add_issue(
                    failures,
                    "existing_scene_index_bridge_manifest_scene_split_membership_hash_mismatch",
                    "existing_scene_index_bridge_manifest.scene_split_membership_hash mismatch.",
                    manifest_value=observed_scene_split_membership_hash,
                    computed_value=computed_scene_split_membership_hash,
                )
            observed_scene_entries = _as_list(existing_scene_index_bridge_manifest_payload.get("scene_entries"))
            if len(observed_scene_entries) != len(computed_scene_entries):
                _add_issue(
                    failures,
                    "existing_scene_index_bridge_manifest_scene_entries_count_mismatch",
                    "existing_scene_index_bridge_manifest.scene_entries count mismatch.",
                    manifest_value=len(observed_scene_entries),
                    computed_value=len(computed_scene_entries),
                )
            for idx, entry in enumerate(observed_scene_entries, start=1):
                obj = _as_dict(entry)
                row_scene_key = str(obj.get("scene_key") or "").strip()
                if not row_scene_key:
                    row_scene_key = _scene_key_for_fields(
                        str(obj.get("identity_id") or "").strip(),
                        str(obj.get("trajectory_id") or "").strip(),
                        str(obj.get("node_id") or "").strip(),
                        str(obj.get("scene_id") or "").strip(),
                        str(obj.get("scene_dir") or obj.get("scene_root") or "").strip(),
                    )
                expected_entry = _as_dict(computed_scene_entries.get(row_scene_key))
                if not expected_entry:
                    _add_issue(
                        failures,
                        "existing_scene_index_bridge_manifest_scene_entry_unknown_scene_key",
                        "existing_scene_index_bridge_manifest.scene_entries contains unknown scene key.",
                        scene_entry_index=idx,
                        scene_key=row_scene_key,
                    )
                    continue
                if _to_int(obj.get("sample_count")) != _to_int(expected_entry.get("sample_count")):
                    _add_issue(
                        failures,
                        "existing_scene_index_bridge_manifest_scene_entry_sample_count_mismatch",
                        "existing_scene_index_bridge_manifest.scene_entries[].sample_count mismatch.",
                        scene_entry_index=idx,
                        scene_key=row_scene_key,
                        manifest_value=obj.get("sample_count"),
                        computed_value=expected_entry.get("sample_count"),
                    )
                observed_splits = sorted({str(v).strip() for v in _as_list(obj.get("split_names")) if str(v).strip()})
                expected_splits = sorted({str(v).strip() for v in expected_entry.get("split_names", set()) if str(v).strip()})
                if observed_splits != expected_splits:
                    _add_issue(
                        failures,
                        "existing_scene_index_bridge_manifest_scene_entry_split_names_mismatch",
                        "existing_scene_index_bridge_manifest.scene_entries[].split_names mismatch.",
                        scene_entry_index=idx,
                        scene_key=row_scene_key,
                        manifest_value=observed_splits,
                        computed_value=expected_splits,
                    )
                observed_cameras = sorted({str(v).strip() for v in _as_list(obj.get("camera_ids")) if str(v).strip()})
                expected_cameras = sorted({str(v).strip() for v in expected_entry.get("camera_ids", set()) if str(v).strip()})
                if observed_cameras != expected_cameras:
                    _add_issue(
                        failures,
                        "existing_scene_index_bridge_manifest_scene_entry_camera_ids_mismatch",
                        "existing_scene_index_bridge_manifest.scene_entries[].camera_ids mismatch.",
                        scene_entry_index=idx,
                        scene_key=row_scene_key,
                        manifest_value=observed_cameras,
                        computed_value=expected_cameras,
                    )
                for field in ("mask_gt_available_count", "no_mask_sample_count", "sidecar_complete_count"):
                    if _to_int(obj.get(field)) != _to_int(expected_entry.get(field)):
                        _add_issue(
                            failures,
                            "existing_scene_index_bridge_manifest_scene_entry_count_field_mismatch",
                            "existing_scene_index_bridge_manifest.scene_entries[] count field mismatch.",
                            scene_entry_index=idx,
                            scene_key=row_scene_key,
                            field=field,
                            manifest_value=obj.get(field),
                            computed_value=expected_entry.get(field),
                        )
                expected_ts_count = len(expected_entry.get("timestamp_us_values", set()))
                if _to_int(obj.get("timestamp_count")) != expected_ts_count:
                    _add_issue(
                        failures,
                        "existing_scene_index_bridge_manifest_scene_entry_timestamp_count_mismatch",
                        "existing_scene_index_bridge_manifest.scene_entries[].timestamp_count mismatch.",
                        scene_entry_index=idx,
                        scene_key=row_scene_key,
                        manifest_value=obj.get("timestamp_count"),
                        computed_value=expected_ts_count,
                    )
    else:
        _add_issue(
            warnings,
            "existing_scene_index_bridge_manifest_missing_legacy_compatible",
            "existing_scene_index_bridge_manifest.json is absent; treated as legacy run for compatibility.",
        )
    should_validate_sidecar_quality_manifest = (
        artifact_presence["sidecar_quality_manifest"] or sidecar_quality_manifest_contract_declared
    )
    if should_validate_sidecar_quality_manifest:
        sidecar_quality_manifest_payload, sidecar_quality_manifest_err = _load_json(sidecar_quality_manifest_path)
        if sidecar_quality_manifest_err:
            _add_issue(
                failures,
                "sidecar_quality_manifest_invalid",
                "sidecar_quality_manifest.json is missing or invalid while present/contract-declared.",
                error=sidecar_quality_manifest_err,
            )
        elif sidecar_quality_manifest_payload.get("schema_version") != "carla_air_sidecar_quality_manifest_v1":
            _add_issue(
                failures,
                "sidecar_quality_manifest_schema_mismatch",
                "sidecar_quality_manifest schema_version mismatch.",
                got=sidecar_quality_manifest_payload.get("schema_version"),
                expected="carla_air_sidecar_quality_manifest_v1",
            )
    elif sidecar_quality_manifest_contract_required:
        _add_issue(
            failures,
            "sidecar_quality_manifest_missing_contract_required",
            "sidecar_quality_manifest.json is required by run_contract.artifacts but missing.",
        )
    else:
        _add_issue(
            warnings,
            "sidecar_quality_manifest_missing_legacy_compatible",
            "sidecar_quality_manifest.json is absent and not contract-declared; treated as legacy-compatible.",
        )
    should_validate_sample_schema_coverage_manifest = (
        artifact_presence["sample_schema_coverage_manifest"] or sample_schema_coverage_manifest_contract_declared
    )
    if should_validate_sample_schema_coverage_manifest:
        sample_schema_coverage_manifest_payload, sample_schema_coverage_manifest_err = _load_json(
            sample_schema_coverage_manifest_path
        )
        if sample_schema_coverage_manifest_err:
            _add_issue(
                failures,
                "sample_schema_coverage_manifest_invalid",
                "sample_schema_coverage_manifest.json is missing or invalid while present/contract-declared.",
                error=sample_schema_coverage_manifest_err,
            )
        elif sample_schema_coverage_manifest_payload.get("schema_version") != "carla_air_sample_schema_coverage_manifest_v1":
            _add_issue(
                failures,
                "sample_schema_coverage_manifest_schema_mismatch",
                "sample_schema_coverage_manifest schema_version mismatch.",
                got=sample_schema_coverage_manifest_payload.get("schema_version"),
                expected="carla_air_sample_schema_coverage_manifest_v1",
            )
    elif sample_schema_coverage_manifest_contract_required:
        _add_issue(
            failures,
            "sample_schema_coverage_manifest_missing_contract_required",
            "sample_schema_coverage_manifest.json is required by run_contract.artifacts but missing.",
        )
    else:
        _add_issue(
            warnings,
            "sample_schema_coverage_manifest_missing_legacy_compatible",
            "sample_schema_coverage_manifest.json is absent and not contract-declared; treated as legacy-compatible.",
        )
    should_validate_no_mask_non_promotion_manifest = (
        artifact_presence["no_mask_non_promotion_manifest"] or no_mask_non_promotion_manifest_contract_declared
    )
    if should_validate_no_mask_non_promotion_manifest:
        no_mask_non_promotion_manifest_payload, no_mask_non_promotion_manifest_err = _load_json(
            no_mask_non_promotion_manifest_path
        )
        if no_mask_non_promotion_manifest_err:
            _add_issue(
                failures,
                "no_mask_non_promotion_manifest_invalid",
                "no_mask_non_promotion_manifest.json is missing or invalid while present/contract-declared.",
                error=no_mask_non_promotion_manifest_err,
            )
        elif no_mask_non_promotion_manifest_payload.get("schema_version") != "carla_air_no_mask_non_promotion_manifest_v1":
            _add_issue(
                failures,
                "no_mask_non_promotion_manifest_schema_mismatch",
                "no_mask_non_promotion_manifest schema_version mismatch.",
                got=no_mask_non_promotion_manifest_payload.get("schema_version"),
                expected="carla_air_no_mask_non_promotion_manifest_v1",
            )
    elif no_mask_non_promotion_manifest_contract_required:
        _add_issue(
            failures,
            "no_mask_non_promotion_manifest_missing_contract_required",
            "no_mask_non_promotion_manifest.json is required by run_contract.artifacts but missing.",
        )
    else:
        _add_issue(
            warnings,
            "no_mask_non_promotion_manifest_missing_legacy_compatible",
            "no_mask_non_promotion_manifest.json is absent and not contract-declared; treated as legacy-compatible.",
        )
    should_validate_deployment_episode_visibility_manifest = (
        artifact_presence["deployment_episode_visibility_manifest"]
        or deployment_episode_visibility_manifest_contract_declared
    )
    if should_validate_deployment_episode_visibility_manifest:
        (
            deployment_episode_visibility_manifest_payload,
            deployment_episode_visibility_manifest_err,
        ) = _load_json(deployment_episode_visibility_manifest_path)
        if deployment_episode_visibility_manifest_err:
            _add_issue(
                failures,
                "deployment_episode_visibility_manifest_invalid",
                "deployment_episode_visibility_manifest.json is missing or invalid while present/contract-declared.",
                error=deployment_episode_visibility_manifest_err,
            )
        elif (
            deployment_episode_visibility_manifest_payload.get("schema_version")
            != "carla_air_deployment_episode_visibility_manifest_v1"
        ):
            _add_issue(
                failures,
                "deployment_episode_visibility_manifest_schema_mismatch",
                "deployment_episode_visibility_manifest schema_version mismatch.",
                got=deployment_episode_visibility_manifest_payload.get("schema_version"),
                expected="carla_air_deployment_episode_visibility_manifest_v1",
            )
    elif deployment_episode_visibility_manifest_contract_required:
        _add_issue(
            failures,
            "deployment_episode_visibility_manifest_missing_contract_required",
            "deployment_episode_visibility_manifest.json is required by run_contract.artifacts but missing.",
        )
    else:
        _add_issue(
            warnings,
            "deployment_episode_visibility_manifest_missing_legacy_compatible",
            "deployment_episode_visibility_manifest.json is absent and not contract-declared; treated as legacy-compatible.",
        )

    should_validate_capture_queue = artifact_presence["capture_queue"] or capture_queue_contract_required
    if should_validate_capture_queue:
        capture_queue_rows, capture_queue_failures = _load_object_jsonl(
            capture_queue_path,
            "capture_queue_missing",
            "capture_queue",
        )
        failures.extend(capture_queue_failures)
        queue_capture_task_id_set: set[str] = set()
        queue_scene_by_itn: dict[tuple[str, str, str], dict[str, Any]] = {}
        queue_has_expected_scene_root_field = False
        queue_has_readiness_field = False
        for idx, item in enumerate(capture_queue_rows, start=1):
            if item.get("schema_version") != "carla_air_capture_queue_item_v1":
                _add_issue(
                    failures,
                    "capture_queue_schema_mismatch",
                    "capture_queue.jsonl item schema_version mismatch.",
                    line=idx,
                    got=item.get("schema_version"),
                    expected="carla_air_capture_queue_item_v1",
                )
            required_queue_fields = [
                "capture_task_id",
                "identity_id",
                "trajectory_id",
                "node_id",
                "camera_id",
                "state",
                "capture_allowed_now",
                "starts_runtime",
                "writes_scene_outputs",
            ]
            missing_fields = [k for k in required_queue_fields if k not in item]
            if missing_fields:
                _add_issue(
                    failures,
                    "capture_queue_required_field_missing",
                    "capture_queue.jsonl item misses required fields.",
                    line=idx,
                    missing_fields=missing_fields,
                )
                continue
            capture_task_id = str(item.get("capture_task_id") or "").strip()
            identity_id = str(item.get("identity_id") or "").strip()
            trajectory_id = str(item.get("trajectory_id") or "").strip()
            node_id = str(item.get("node_id") or "").strip()
            block_reason = str(item.get("block_reason") or "").strip()
            state_value = str(item.get("state") or "").strip().lower()
            if state_value not in {"queued", "blocked"}:
                _add_issue(
                    failures,
                    "capture_queue_state_invalid",
                    "capture_queue state must be queued or blocked.",
                    line=idx,
                    state=item.get("state"),
                )
            if capture_task_id:
                queue_capture_task_id_set.add(capture_task_id)
            else:
                _add_issue(
                    failures,
                    "capture_queue_capture_task_id_empty",
                    "capture_queue.jsonl item capture_task_id must be non-empty.",
                    line=idx,
                )
            if item.get("starts_runtime") is not False:
                _add_issue(
                    failures,
                    "capture_queue_starts_runtime_mismatch",
                    "capture_queue item starts_runtime must be false.",
                    line=idx,
                    got=item.get("starts_runtime"),
                )
            if item.get("writes_scene_outputs") is not False:
                _add_issue(
                    failures,
                    "capture_queue_writes_scene_outputs_mismatch",
                    "capture_queue item writes_scene_outputs must be false.",
                    line=idx,
                    got=item.get("writes_scene_outputs"),
                )
            if state_value == "blocked":
                capture_queue_blocked_count += 1
                if not block_reason:
                    _add_issue(
                        failures,
                        "capture_queue_blocked_missing_block_reason",
                        "blocked capture_queue item must carry non-empty block_reason.",
                        line=idx,
                    )
            if state_value == "queued" and block_reason:
                _add_issue(
                    failures,
                    "capture_queue_queued_has_block_reason",
                    "queued capture_queue item should not carry block_reason.",
                    line=idx,
                    block_reason=block_reason,
                )

            expected_scene_root_raw = item.get("expected_scene_root")
            expected_scene_root = str(expected_scene_root_raw or "").strip()
            if expected_scene_root_raw is not None:
                queue_has_expected_scene_root_field = True
                expected_scene_root_ref = _expected_scene_root(identity_id, trajectory_id, node_id)
                if not expected_scene_root:
                    _add_issue(
                        failures,
                        "capture_queue_expected_scene_root_empty",
                        "capture_queue expected_scene_root must be non-empty when present.",
                        line=idx,
                    )
                elif expected_scene_root != expected_scene_root_ref:
                    _add_issue(
                        failures,
                        "capture_queue_expected_scene_root_mismatch",
                        "capture_queue expected_scene_root does not match deterministic convention.",
                        line=idx,
                        expected=expected_scene_root_ref,
                        got=expected_scene_root,
                    )

            readiness_obj = _as_dict(item.get("scene_output_readiness"))
            if "scene_output_readiness" in item:
                queue_has_readiness_field = True
                expected_readiness = _readiness_from_state(state_value, [block_reason] if block_reason else [])
                for key in ("status", "blocked", "evidence_ready"):
                    if readiness_obj.get(key) != expected_readiness.get(key):
                        _add_issue(
                            failures,
                            "capture_queue_readiness_mismatch",
                            "capture_queue scene_output_readiness is inconsistent with state/block_reason.",
                            line=idx,
                            field=key,
                            expected=expected_readiness.get(key),
                            got=readiness_obj.get(key),
                        )
                blocked_reasons = [str(x).strip() for x in _as_list(readiness_obj.get("blocked_reasons")) if str(x).strip()]
                expected_blocked_reasons = expected_readiness["blocked_reasons"]
                if blocked_reasons != expected_blocked_reasons:
                    _add_issue(
                        failures,
                        "capture_queue_readiness_block_reasons_mismatch",
                        "capture_queue readiness blocked_reasons must align with state/block_reason.",
                        line=idx,
                        expected=expected_blocked_reasons,
                        got=blocked_reasons,
                    )

            scene_key = (identity_id, trajectory_id, node_id)
            scene_rec = queue_scene_by_itn.get(scene_key)
            if scene_rec is None:
                scene_rec = {
                    "expected_scene_root": expected_scene_root,
                    "states": [],
                    "block_reasons": [],
                }
                queue_scene_by_itn[scene_key] = scene_rec
            if expected_scene_root:
                scene_rec["expected_scene_root"] = expected_scene_root
            scene_rec["states"].append(state_value)
            if block_reason:
                scene_rec["block_reasons"].append(block_reason)
            if plan_profile_contract_present:
                queue_profile_id = str(item.get("identity_model_profile_id") or "").strip()
                if not queue_profile_id:
                    _add_issue(
                        failures,
                        "capture_queue_profile_id_missing",
                        "capture_queue item must include identity_model_profile_id when plan.identity_model_profiles exists.",
                        line=idx,
                        capture_task_id=capture_task_id,
                    )
                else:
                    planned_profile = plan_profile_map.get(queue_profile_id)
                    if planned_profile is None:
                        _add_issue(
                            failures,
                            "capture_queue_profile_id_unknown",
                            "capture_queue item identity_model_profile_id not found in plan.identity_model_profiles.",
                            line=idx,
                            identity_model_profile_id=queue_profile_id,
                        )
                    else:
                        queue_model_label = str(item.get("model_label") or "").strip()
                        if planned_profile["model_label"] and queue_model_label != planned_profile["model_label"]:
                            _add_issue(
                                failures,
                                "capture_queue_profile_model_label_mismatch",
                                "capture_queue item model_label must align with referenced identity_model_profile.",
                                line=idx,
                                queue_model_label=queue_model_label,
                                profile_model_label=planned_profile["model_label"],
                            )
                        queue_switch_method = str(item.get("switch_method") or "").strip()
                        if planned_profile["switch_method"] and queue_switch_method != planned_profile["switch_method"]:
                            _add_issue(
                                failures,
                                "capture_queue_profile_switch_method_mismatch",
                                "capture_queue item switch_method must align with referenced identity_model_profile.",
                                line=idx,
                                queue_switch_method=queue_switch_method,
                                profile_switch_method=planned_profile["switch_method"],
                            )
                        if bool(item.get("requires_ue_carla_import_readback")) != bool(planned_profile["requires_ue_carla_import_readback"]):
                            _add_issue(
                                failures,
                                "capture_queue_profile_requires_readback_mismatch",
                                "capture_queue item requires_ue_carla_import_readback must align with profile.",
                                line=idx,
                            )
                queue_profile_obj = _as_dict(item.get("identity_model_profile"))
                if queue_profile_obj:
                    queue_profile_obj_id = str(queue_profile_obj.get("identity_model_profile_id") or "").strip()
                    if queue_profile_obj_id and queue_profile_id and queue_profile_obj_id != queue_profile_id:
                        _add_issue(
                            failures,
                            "capture_queue_profile_object_id_mismatch",
                            "capture_queue identity_model_profile.identity_model_profile_id must match identity_model_profile_id.",
                            line=idx,
                            expected=queue_profile_id,
                            got=queue_profile_obj_id,
                        )
        if plan_payload:
            if len(capture_queue_rows) != len(plan_capture_tasks):
                _add_issue(
                    failures,
                    "capture_queue_count_mismatch_plan",
                    "capture_queue item count must equal len(plan.capture_tasks).",
                    capture_queue_count=len(capture_queue_rows),
                    plan_capture_tasks_len=len(plan_capture_tasks),
                )
            if queue_capture_task_id_set != plan_capture_task_id_set:
                _add_issue(
                    failures,
                    "capture_queue_capture_task_id_set_mismatch_plan",
                    "capture_queue capture_task_id set must equal plan capture task id set.",
                    capture_queue_capture_task_id_count=len(queue_capture_task_id_set),
                    plan_capture_task_id_count=len(plan_capture_task_id_set),
                )

        if run_contract_payload:
            contract_counts = _as_dict(run_contract_payload.get("counts"))
            if "capture_queue_item_count" in contract_counts:
                expected = _to_int(contract_counts.get("capture_queue_item_count"))
                if expected != len(capture_queue_rows):
                    _add_issue(
                        failures,
                        "run_contract_capture_queue_item_count_mismatch",
                        "run_contract.counts.capture_queue_item_count must match capture_queue.jsonl rows.",
                        expected=expected,
                        actual=len(capture_queue_rows),
                    )
            for blocked_count_key in ("blocked_capture_queue_item_count", "capture_queue_blocked_item_count"):
                if blocked_count_key not in contract_counts:
                    continue
                expected_blocked = _to_int(contract_counts.get(blocked_count_key))
                if expected_blocked != capture_queue_blocked_count:
                    _add_issue(
                        failures,
                        "run_contract_capture_queue_blocked_item_count_mismatch",
                        "run_contract.counts blocked capture queue count must match blocked rows in capture_queue.jsonl.",
                        count_key=blocked_count_key,
                        expected=expected_blocked,
                        actual=capture_queue_blocked_count,
                    )

    should_validate_capture_queue_manifest = (
        artifact_presence["capture_queue_manifest"] or capture_queue_manifest_contract_declared
    )
    if should_validate_capture_queue_manifest:
        capture_queue_manifest_payload, capture_queue_manifest_err = _load_json(capture_queue_manifest_path)
        if capture_queue_manifest_err:
            _add_issue(
                failures,
                "capture_queue_manifest_invalid",
                "capture_queue_manifest.json is missing or invalid while present/contract-declared.",
                error=capture_queue_manifest_err,
            )
        elif capture_queue_manifest_payload.get("schema_version") != "carla_air_capture_queue_manifest_v1":
            _add_issue(
                failures,
                "capture_queue_manifest_schema_mismatch",
                "capture_queue_manifest schema_version mismatch.",
                got=capture_queue_manifest_payload.get("schema_version"),
                expected="carla_air_capture_queue_manifest_v1",
            )
    elif capture_queue_manifest_contract_required:
        _add_issue(
            failures,
            "capture_queue_manifest_missing_contract_required",
            "capture_queue_manifest.json is required by run_contract.artifacts but missing.",
        )
    elif run_contract_payload:
        _add_issue(
            warnings,
            "capture_queue_manifest_missing_legacy_compatible",
            "capture_queue_manifest.json is absent and not contract-declared; treated as legacy-compatible run.",
        )

    queue_expected_scene_root_required = False
    queue_readiness_required = False
    if should_validate_capture_queue and capture_queue_rows:
        queue_expected_scene_root_required = any("expected_scene_root" in row for row in capture_queue_rows)
        queue_readiness_required = any("scene_output_readiness" in row for row in capture_queue_rows)

    should_validate_scene_output_manifest = artifact_presence["scene_output_manifest"] or scene_output_manifest_contract_required
    if should_validate_scene_output_manifest:
        scene_output_manifest_payload, scene_output_manifest_err = _load_json(scene_output_manifest_path)
        if scene_output_manifest_err:
            _add_issue(
                failures,
                "scene_output_manifest_invalid",
                "scene_output_manifest.json is missing or invalid.",
                error=scene_output_manifest_err,
            )
        else:
            if scene_output_manifest_payload.get("schema_version") != "carla_air_scene_output_manifest_v1":
                _add_issue(
                    failures,
                    "scene_output_manifest_schema_mismatch",
                    "scene_output_manifest.json schema_version mismatch.",
                    got=scene_output_manifest_payload.get("schema_version"),
                    expected="carla_air_scene_output_manifest_v1",
                )
            if scene_output_manifest_payload.get("starts_runtime") is not False:
                _add_issue(
                    failures,
                    "scene_output_manifest_starts_runtime_mismatch",
                    "scene_output_manifest.starts_runtime must be false.",
                    got=scene_output_manifest_payload.get("starts_runtime"),
                )
            if scene_output_manifest_payload.get("writes_scene_outputs") is not False:
                _add_issue(
                    failures,
                    "scene_output_manifest_writes_scene_outputs_mismatch",
                    "scene_output_manifest.writes_scene_outputs must be false.",
                    got=scene_output_manifest_payload.get("writes_scene_outputs"),
                )
            scene_outputs = scene_output_manifest_payload.get("scene_outputs")
            if not isinstance(scene_outputs, list):
                _add_issue(
                    failures,
                    "scene_output_manifest_scene_outputs_not_list",
                    "scene_output_manifest.scene_outputs must be a list.",
                    got_type=type(scene_outputs).__name__,
                )
            else:
                scene_output_count = len(scene_outputs)
                manifest_has_expected_scene_root_field = any(isinstance(obj, dict) and ("expected_scene_root" in obj) for obj in scene_outputs)
                manifest_has_readiness_field = any(isinstance(obj, dict) and ("scene_output_readiness" in obj) for obj in scene_outputs)
                expected_scene_root_required = queue_expected_scene_root_required or manifest_has_expected_scene_root_field
                readiness_required = queue_readiness_required or manifest_has_readiness_field
                scene_manifest_by_itn: dict[tuple[str, str, str], dict[str, Any]] = {}
                for idx, obj in enumerate(scene_outputs, start=1):
                    item = _as_dict(obj)
                    identity_id = str(item.get("identity_id") or "").strip()
                    trajectory_id = str(item.get("trajectory_id") or "").strip()
                    node_id = str(item.get("node_id") or "").strip()
                    state_value = str(item.get("state") or "").strip().lower()
                    block_reasons = sorted({str(x).strip() for x in _as_list(item.get("block_reasons")) if str(x).strip()})
                    if state_value not in {"queued", "blocked"}:
                        _add_issue(
                            failures,
                            "scene_output_manifest_state_invalid",
                            "scene_output state must be queued or blocked.",
                            index=idx,
                            state=item.get("state"),
                        )
                    if state_value == "blocked" and not block_reasons:
                        _add_issue(
                            failures,
                            "scene_output_manifest_blocked_missing_block_reasons",
                            "blocked scene_output must carry non-empty block_reasons.",
                            index=idx,
                        )
                    if state_value == "queued" and block_reasons:
                        _add_issue(
                            failures,
                            "scene_output_manifest_queued_has_block_reasons",
                            "queued scene_output should not carry block_reasons.",
                            index=idx,
                            block_reasons=block_reasons,
                        )
                    if state_value == "blocked":
                        blocked_scene_output_count += 1
                    expected_root = str(item.get("expected_scene_root") or "").strip()
                    expected_root_ref = _expected_scene_root(identity_id, trajectory_id, node_id)
                    if expected_scene_root_required:
                        if not expected_root:
                            _add_issue(
                                failures,
                                "scene_output_manifest_expected_scene_root_empty",
                                "scene_output expected_scene_root must be non-empty when queue/manifest declares it.",
                                index=idx,
                            )
                        elif expected_root != expected_root_ref:
                            _add_issue(
                                failures,
                                "scene_output_manifest_expected_scene_root_mismatch",
                                "scene_output expected_scene_root does not match deterministic convention.",
                                index=idx,
                                expected=expected_root_ref,
                                got=expected_root,
                            )

                    if readiness_required:
                        readiness_obj = _as_dict(item.get("scene_output_readiness"))
                        expected_readiness = _readiness_from_state(state_value, block_reasons)
                        for key in ("status", "blocked", "evidence_ready"):
                            if readiness_obj.get(key) != expected_readiness.get(key):
                                _add_issue(
                                    failures,
                                    "scene_output_manifest_readiness_mismatch",
                                    "scene_output_readiness is inconsistent with state/block_reasons.",
                                    index=idx,
                                    field=key,
                                    expected=expected_readiness.get(key),
                                    got=readiness_obj.get(key),
                                )
                        readiness_blocked_reasons = [str(x).strip() for x in _as_list(readiness_obj.get("blocked_reasons")) if str(x).strip()]
                        if readiness_blocked_reasons != expected_readiness["blocked_reasons"]:
                            _add_issue(
                                failures,
                                "scene_output_manifest_readiness_block_reasons_mismatch",
                                "scene_output readiness blocked_reasons must align with state/block_reasons.",
                                index=idx,
                                expected=expected_readiness["blocked_reasons"],
                                got=readiness_blocked_reasons,
                            )

                    scene_manifest_by_itn[(identity_id, trajectory_id, node_id)] = {
                        "expected_scene_root": expected_root,
                        "state": state_value,
                        "block_reasons": block_reasons,
                    }
                    if plan_profile_contract_present:
                        manifest_profile_id = str(item.get("identity_model_profile_id") or "").strip()
                        if not manifest_profile_id:
                            _add_issue(
                                failures,
                                "scene_output_manifest_profile_id_missing",
                                "scene_output must include identity_model_profile_id when plan.identity_model_profiles exists.",
                                index=idx,
                            )
                        else:
                            planned_profile = plan_profile_map.get(manifest_profile_id)
                            if planned_profile is None:
                                _add_issue(
                                    failures,
                                    "scene_output_manifest_profile_id_unknown",
                                    "scene_output identity_model_profile_id not found in plan.identity_model_profiles.",
                                    index=idx,
                                    identity_model_profile_id=manifest_profile_id,
                                )
                            else:
                                manifest_model_label = str(item.get("model_label") or "").strip()
                                if planned_profile["model_label"] and manifest_model_label != planned_profile["model_label"]:
                                    _add_issue(
                                        failures,
                                        "scene_output_manifest_profile_model_label_mismatch",
                                        "scene_output model_label must align with referenced identity_model_profile.",
                                        index=idx,
                                        scene_model_label=manifest_model_label,
                                        profile_model_label=planned_profile["model_label"],
                                    )
                                manifest_switch_method = str(item.get("switch_method") or "").strip()
                                if planned_profile["switch_method"] and manifest_switch_method != planned_profile["switch_method"]:
                                    _add_issue(
                                        failures,
                                        "scene_output_manifest_profile_switch_method_mismatch",
                                        "scene_output switch_method must align with referenced identity_model_profile.",
                                        index=idx,
                                        scene_switch_method=manifest_switch_method,
                                        profile_switch_method=planned_profile["switch_method"],
                                    )
                                if bool(item.get("requires_ue_carla_import_readback")) != bool(planned_profile["requires_ue_carla_import_readback"]):
                                    _add_issue(
                                        failures,
                                        "scene_output_manifest_profile_requires_readback_mismatch",
                                        "scene_output requires_ue_carla_import_readback must align with profile.",
                                        index=idx,
                                    )
                        manifest_profile_ids = [str(x).strip() for x in _as_list(item.get("identity_model_profile_ids")) if str(x).strip()]
                        if manifest_profile_id and manifest_profile_ids and manifest_profile_id not in manifest_profile_ids:
                            _add_issue(
                                failures,
                                "scene_output_manifest_profile_ids_mismatch",
                                "scene_output identity_model_profile_id must be included in identity_model_profile_ids.",
                                index=idx,
                                identity_model_profile_id=manifest_profile_id,
                                identity_model_profile_ids=manifest_profile_ids,
                            )
                        manifest_profiles = _as_list(item.get("identity_model_profiles"))
                        if manifest_profiles:
                            manifest_profile_obj_ids = sorted(
                                {
                                    str(_as_dict(profile).get("identity_model_profile_id") or "").strip()
                                    for profile in manifest_profiles
                                    if str(_as_dict(profile).get("identity_model_profile_id") or "").strip()
                                }
                            )
                            if manifest_profile_ids and manifest_profile_obj_ids and manifest_profile_obj_ids != sorted(set(manifest_profile_ids)):
                                _add_issue(
                                    failures,
                                    "scene_output_manifest_profile_objects_ids_mismatch",
                                    "scene_output identity_model_profiles ids must align with identity_model_profile_ids.",
                                    index=idx,
                                    identity_model_profile_ids=sorted(set(manifest_profile_ids)),
                                    identity_model_profiles_ids=manifest_profile_obj_ids,
                                )

                if should_validate_capture_queue and capture_queue_rows:
                    queue_scene_by_itn_local: dict[tuple[str, str, str], dict[str, Any]] = {}
                    for row in capture_queue_rows:
                        identity_id = str(row.get("identity_id") or "").strip()
                        trajectory_id = str(row.get("trajectory_id") or "").strip()
                        node_id = str(row.get("node_id") or "").strip()
                        state_value = str(row.get("state") or "").strip().lower()
                        block_reason = str(row.get("block_reason") or "").strip()
                        key = (identity_id, trajectory_id, node_id)
                        rec = queue_scene_by_itn_local.get(key)
                        if rec is None:
                            rec = {"states": [], "block_reasons": [], "expected_scene_root": ""}
                            queue_scene_by_itn_local[key] = rec
                        rec["states"].append(state_value)
                        if block_reason:
                            rec["block_reasons"].append(block_reason)
                        root = str(row.get("expected_scene_root") or "").strip()
                        if root:
                            rec["expected_scene_root"] = root

                    queue_scene_key_set = set(queue_scene_by_itn_local.keys())
                    manifest_scene_key_set = set(scene_manifest_by_itn.keys())
                    if queue_scene_key_set != manifest_scene_key_set:
                        _add_issue(
                            failures,
                            "scene_output_manifest_scene_group_mismatch_queue",
                            "scene_output groups must align with queue identity/trajectory/node groups.",
                            queue_group_count=len(queue_scene_key_set),
                            manifest_group_count=len(manifest_scene_key_set),
                        )
                    for key in sorted(queue_scene_key_set & manifest_scene_key_set):
                        queue_rec = queue_scene_by_itn_local[key]
                        manifest_rec = scene_manifest_by_itn[key]
                        queue_scene_state = "blocked" if queue_rec["states"] and all(s == "blocked" for s in queue_rec["states"]) else "queued"
                        queue_block_reasons = sorted({str(x) for x in queue_rec["block_reasons"] if str(x)})
                        if manifest_rec["state"] != queue_scene_state:
                            _add_issue(
                                failures,
                                "scene_output_manifest_state_mismatch_queue",
                                "scene_output state must align with aggregated queue states.",
                                identity_id=key[0],
                                trajectory_id=key[1],
                                node_id=key[2],
                                expected=queue_scene_state,
                                got=manifest_rec["state"],
                            )
                        if queue_scene_state == "blocked" and manifest_rec["block_reasons"] != queue_block_reasons:
                            _add_issue(
                                failures,
                                "scene_output_manifest_block_reasons_mismatch_queue",
                                "blocked scene_output block_reasons must align with queue blocked reasons.",
                                identity_id=key[0],
                                trajectory_id=key[1],
                                node_id=key[2],
                                expected=queue_block_reasons,
                                got=manifest_rec["block_reasons"],
                            )
                        if expected_scene_root_required:
                            queue_root = str(queue_rec["expected_scene_root"] or "").strip()
                            manifest_root = str(manifest_rec["expected_scene_root"] or "").strip()
                            if queue_root and manifest_root and queue_root != manifest_root:
                                _add_issue(
                                    failures,
                                    "scene_output_manifest_expected_scene_root_mismatch_queue",
                                    "scene_output expected_scene_root must align with queue expected_scene_root.",
                                    identity_id=key[0],
                                    trajectory_id=key[1],
                                    node_id=key[2],
                                    queue_expected_scene_root=queue_root,
                                    manifest_expected_scene_root=manifest_root,
                                )
        if run_contract_payload:
            contract_counts = _as_dict(run_contract_payload.get("counts"))
            if "scene_output_count" in contract_counts:
                expected_scene_output_count = _to_int(contract_counts.get("scene_output_count"))
                if expected_scene_output_count != scene_output_count:
                    _add_issue(
                        failures,
                        "run_contract_scene_output_count_mismatch",
                        "run_contract.counts.scene_output_count must match scene_output_manifest scene_outputs count.",
                        expected=expected_scene_output_count,
                        actual=scene_output_count,
                    )
    capture_queue_manifest_computed: dict[str, Any] = {}
    if should_validate_capture_queue and should_validate_capture_queue_manifest and capture_queue_manifest_payload:
        aligned_run_id = (
            str(run_contract_payload.get("run_id") or "").strip()
            or str(plan_payload.get("run_id") or "").strip()
            or str(manifest_payload.get("run_id") or "").strip()
        )
        capture_queue_manifest_computed = _compute_capture_queue_manifest_from_rows(
            run_id=aligned_run_id,
            capture_queue_rows=capture_queue_rows,
            capture_queue_path=capture_queue_path,
        )
        normalized_capture_queue_manifest = _normalize_capture_queue_manifest(capture_queue_manifest_payload)
        for field_name in (
            "run_id",
            "capture_queue_item_count",
            "blocked_capture_queue_item_count",
            "queued_capture_queue_item_count",
            "state_counts",
            "block_reason_counts",
            "capture_task_id_order_sha256",
            "expected_scene_root_order_sha256",
            "identity_ids",
            "trajectory_ids",
            "node_ids",
            "camera_ids",
            "scene_group_count",
            "source_capture_queue_path",
        ):
            if normalized_capture_queue_manifest.get(field_name) != capture_queue_manifest_computed.get(field_name):
                _add_issue(
                    failures,
                    "capture_queue_manifest_field_mismatch",
                    "capture_queue_manifest field mismatch against recomputed capture_queue facts.",
                    field=field_name,
                    manifest_value=normalized_capture_queue_manifest.get(field_name),
                    computed_value=capture_queue_manifest_computed.get(field_name),
                )
        for bool_key, expected in (
            ("starts_runtime", False),
            ("writes_scene_outputs", False),
            ("non_promotion", True),
            ("full_v1_live_dataset_ready", False),
        ):
            if normalized_capture_queue_manifest.get(bool_key) is not expected:
                _add_issue(
                    failures,
                    "capture_queue_manifest_offline_boundary_mismatch",
                    "capture_queue_manifest violates offline non-promotion boundary.",
                    field=bool_key,
                    observed_value=normalized_capture_queue_manifest.get(bool_key),
                    expected_value=expected,
                )
        if run_contract_payload:
            contract_artifacts = _as_dict(run_contract_payload.get("artifacts"))
            contract_manifest_path = str(contract_artifacts.get("capture_queue_manifest_json") or "").strip()
            if contract_manifest_path and str(capture_queue_manifest_path.resolve()) != str(_repo_or_abs(contract_manifest_path).resolve()):
                _add_issue(
                    failures,
                    "capture_queue_manifest_run_contract_artifact_path_mismatch",
                    "run_contract.artifacts.capture_queue_manifest_json must align with verifier run_dir manifest path.",
                    manifest_path=str(capture_queue_manifest_path.resolve()),
                    run_contract_path=contract_manifest_path,
                )
        if batch_run_manifest_payload:
            batch_paths = _as_dict(batch_run_manifest_payload.get("artifact_paths"))
            batch_manifest_path = str(batch_paths.get("capture_queue_manifest_json") or "").strip()
            contract_manifest_path = str(_as_dict(run_contract_payload.get("artifacts")).get("capture_queue_manifest_json") or "").strip()
            if contract_manifest_path:
                if not batch_manifest_path:
                    _add_issue(
                        failures,
                        "batch_run_manifest_capture_queue_manifest_path_missing",
                        "batch_run_manifest.artifact_paths.capture_queue_manifest_json must exist when contract declares it.",
                    )
                elif batch_manifest_path != contract_manifest_path:
                    _add_issue(
                        failures,
                        "batch_run_manifest_capture_queue_manifest_path_mismatch",
                        "batch_run_manifest.artifact_paths.capture_queue_manifest_json must align with run_contract.artifacts.",
                        expected=contract_manifest_path,
                        got=batch_manifest_path,
                    )
        if artifact_manifest_payload:
            artifact_map = _as_dict(artifact_manifest_payload.get("artifacts"))
            if capture_queue_manifest_contract_declared and ("capture_queue_manifest_json" not in artifact_map):
                _add_issue(
                    failures,
                    "artifact_manifest_missing_capture_queue_manifest_key",
                    "artifact_manifest.artifacts missing capture_queue_manifest_json declared by run_contract.",
                    artifact_key="capture_queue_manifest_json",
                )

    if manifest_payload:
        outputs = _as_dict(manifest_payload.get("outputs"))
        required_manifest_output_keys = [
            "dataset_plan_json",
            "dataset_samples_jsonl",
            "dataset_splits_json",
            "deployment_episodes_json",
            "dataset_manifest_json",
        ]
        for key in required_manifest_output_keys:
            raw_path = outputs.get(key)
            path = _repo_or_abs(str(raw_path or "")) if raw_path else Path("")
            if not raw_path or (not path.is_file()):
                _add_issue(
                    failures,
                    "manifest_output_missing",
                    "manifest.outputs entry is missing or points to a non-file path.",
                    output_key=key,
                    output_path=str(raw_path or ""),
                )
        dataset_index_manifest_output_raw = outputs.get("dataset_index_manifest_json")
        if dataset_index_manifest_output_raw:
            dataset_index_manifest_output_path = _repo_or_abs(str(dataset_index_manifest_output_raw))
            if not dataset_index_manifest_output_path.is_file():
                _add_issue(
                    failures,
                    "manifest_output_dataset_index_manifest_missing",
                    "manifest.outputs.dataset_index_manifest_json points to a non-file path.",
                    output_key="dataset_index_manifest_json",
                    output_path=str(dataset_index_manifest_output_raw),
                )
        else:
            _add_issue(
                warnings,
                "manifest_output_dataset_index_manifest_missing_legacy_compatible",
                "manifest.outputs.dataset_index_manifest_json is absent; treated as legacy-compatible run.",
            )
        scene_sample_index_manifest_output_raw = outputs.get("scene_sample_index_manifest_json")
        if scene_sample_index_manifest_output_raw:
            scene_sample_index_manifest_output_path = _repo_or_abs(str(scene_sample_index_manifest_output_raw))
            if not scene_sample_index_manifest_output_path.is_file():
                _add_issue(
                    failures,
                    "manifest_output_scene_sample_index_manifest_missing",
                    "manifest.outputs.scene_sample_index_manifest_json points to a non-file path.",
                    output_key="scene_sample_index_manifest_json",
                    output_path=str(scene_sample_index_manifest_output_raw),
                )
        else:
            _add_issue(
                warnings,
                "manifest_output_scene_sample_index_manifest_missing_legacy_compatible",
                "manifest.outputs.scene_sample_index_manifest_json is absent; treated as legacy-compatible run.",
            )
        sample_schema_coverage_manifest_output_raw = outputs.get("sample_schema_coverage_manifest_json")
        if sample_schema_coverage_manifest_output_raw:
            sample_schema_coverage_manifest_output_path = _repo_or_abs(str(sample_schema_coverage_manifest_output_raw))
            if not sample_schema_coverage_manifest_output_path.is_file():
                _add_issue(
                    failures,
                    "manifest_output_sample_schema_coverage_manifest_missing",
                    "manifest.outputs.sample_schema_coverage_manifest_json points to a non-file path.",
                    output_key="sample_schema_coverage_manifest_json",
                    output_path=str(sample_schema_coverage_manifest_output_raw),
                )
        else:
            _add_issue(
                warnings,
                "manifest_output_sample_schema_coverage_manifest_missing_legacy_compatible",
                "manifest.outputs.sample_schema_coverage_manifest_json is absent; treated as legacy-compatible run.",
            )
        if run_contract_payload:
            contract_artifacts = _as_dict(run_contract_payload.get("artifacts"))
            for key in ("run_contract_json", "run_summary_json"):
                if key in contract_artifacts:
                    raw_path = contract_artifacts.get(key)
                    path = _repo_or_abs(str(raw_path or "")) if raw_path else Path("")
                    if not raw_path or (not path.is_file()):
                        _add_issue(
                            failures,
                            "manifest_related_contract_output_missing",
                            "run_contract.artifacts output path is missing or invalid.",
                            output_key=key,
                            output_path=str(raw_path or ""),
                        )

    capture_tasks_by_itn: dict[tuple[str, str, str], set[str]] = {}
    capture_task_id_set: set[str] = set()
    for idx, task in enumerate(plan_capture_tasks, start=1):
        task_obj = _as_dict(task)
        if task_obj.get("schema_version") != "carla_air_capture_task_v1":
            _add_issue(
                failures,
                "plan_capture_task_schema_mismatch",
                "capture_tasks[].schema_version mismatch.",
                index=idx,
                got=task_obj.get("schema_version"),
            )
        required_task_fields = ["capture_task_id", "identity_id", "trajectory_id", "node_id", "camera_id"]
        missing_fields = [k for k in required_task_fields if not str(task_obj.get(k) or "").strip()]
        if missing_fields:
            _add_issue(
                failures,
                "plan_capture_task_required_field_missing",
                "capture_tasks[] entry misses required non-empty fields.",
                index=idx,
                missing_fields=missing_fields,
            )
            continue
        capture_task_id = str(task_obj.get("capture_task_id")).strip()
        identity_id = str(task_obj.get("identity_id")).strip()
        trajectory_id = str(task_obj.get("trajectory_id")).strip()
        node_id = str(task_obj.get("node_id")).strip()
        camera_id = str(task_obj.get("camera_id")).strip()
        capture_task_id_set.add(capture_task_id)
        if node_id not in allowed_nodes:
            _add_issue(
                failures,
                "plan_capture_task_node_invalid",
                "capture_tasks[].node_id is out of allowed domain.",
                index=idx,
                node_id=node_id,
            )
        if camera_id not in allowed_cameras:
            _add_issue(
                failures,
                "plan_capture_task_camera_invalid",
                "capture_tasks[].camera_id is out of allowed domain.",
                index=idx,
                camera_id=camera_id,
            )
        capture_profile = task_obj.get("capture_profile")
        safe_tokens = [
            _safe_token(capture_profile),
            _safe_token(identity_id),
            _safe_token(trajectory_id),
            _safe_token(node_id),
            _safe_token(camera_id),
        ]
        safe_capture_id = _safe_token(capture_task_id)
        missing_tokens = [tok for tok in safe_tokens if tok and tok not in safe_capture_id]
        if missing_tokens:
            _add_issue(
                failures,
                "plan_capture_task_id_token_mismatch",
                "capture_task_id should include safe tokens of capture_profile/identity/trajectory/node/camera.",
                index=idx,
                capture_task_id=capture_task_id,
                missing_tokens=missing_tokens,
            )
        if plan_profile_contract_present:
            profile_id = str(task_obj.get("identity_model_profile_id") or "").strip()
            profile_obj = _normalize_profile_ref(task_obj.get("identity_model_profile"))
            if not profile_id:
                _add_issue(
                    failures,
                    "plan_capture_task_profile_id_missing",
                    "capture_tasks[] must include identity_model_profile_id when plan.identity_model_profiles exists.",
                    index=idx,
                    capture_task_id=capture_task_id,
                )
            else:
                planned_profile = plan_profile_map.get(profile_id)
                if planned_profile is None:
                    _add_issue(
                        failures,
                        "plan_capture_task_profile_id_unknown",
                        "capture_tasks[] identity_model_profile_id not found in plan.identity_model_profiles.",
                        index=idx,
                        capture_task_id=capture_task_id,
                        identity_model_profile_id=profile_id,
                    )
                else:
                    if planned_profile["identity_id"] and planned_profile["identity_id"] != identity_id:
                        _add_issue(
                            failures,
                            "plan_capture_task_profile_identity_mismatch",
                            "capture_tasks[] profile identity_id must match task identity_id.",
                            index=idx,
                            capture_task_id=capture_task_id,
                            profile_identity_id=planned_profile["identity_id"],
                            task_identity_id=identity_id,
                        )
                    task_model_label = str(task_obj.get("model_label") or "").strip()
                    if planned_profile["model_label"] and task_model_label != planned_profile["model_label"]:
                        _add_issue(
                            failures,
                            "plan_capture_task_profile_model_label_mismatch",
                            "capture_tasks[] model_label must align with referenced identity_model_profile.",
                            index=idx,
                            capture_task_id=capture_task_id,
                            task_model_label=task_model_label,
                            profile_model_label=planned_profile["model_label"],
                        )
                    task_switch_method = str(task_obj.get("switch_method") or "").strip()
                    if planned_profile["switch_method"] and task_switch_method != planned_profile["switch_method"]:
                        _add_issue(
                            failures,
                            "plan_capture_task_profile_switch_method_mismatch",
                            "capture_tasks[] switch_method must align with referenced identity_model_profile.",
                            index=idx,
                            capture_task_id=capture_task_id,
                            task_switch_method=task_switch_method,
                            profile_switch_method=planned_profile["switch_method"],
                        )
                    if bool(task_obj.get("requires_ue_carla_import_readback")) != bool(planned_profile["requires_ue_carla_import_readback"]):
                        _add_issue(
                            failures,
                            "plan_capture_task_profile_requires_readback_mismatch",
                            "capture_tasks[] requires_ue_carla_import_readback must align with referenced identity_model_profile.",
                            index=idx,
                            capture_task_id=capture_task_id,
                        )
            if profile_obj and profile_obj["identity_model_profile_id"] and profile_id and profile_obj["identity_model_profile_id"] != profile_id:
                _add_issue(
                    failures,
                    "plan_capture_task_profile_object_id_mismatch",
                    "capture_tasks[] identity_model_profile.identity_model_profile_id must match identity_model_profile_id.",
                    index=idx,
                    capture_task_id=capture_task_id,
                    field_value=profile_obj["identity_model_profile_id"],
                    expected=profile_id,
                )
        key = (identity_id, trajectory_id, node_id)
        cameras = capture_tasks_by_itn.setdefault(key, set())
        cameras.add(camera_id)

    matrix_entries = _as_list(plan_payload.get("matrix"))
    for idx, entry in enumerate(matrix_entries, start=1):
        cell = _as_dict(entry)
        required_matrix_fields = ["matrix_cell_id", "identity_id", "trajectory_id", "camera_layout_id", "node_id"]
        missing_fields = [k for k in required_matrix_fields if not str(cell.get(k) or "").strip()]
        if missing_fields:
            _add_issue(
                failures,
                "plan_matrix_required_field_missing",
                "matrix[] entry misses required non-empty fields.",
                index=idx,
                missing_fields=missing_fields,
            )
            continue
        identity_id = str(cell.get("identity_id")).strip()
        trajectory_id = str(cell.get("trajectory_id")).strip()
        node_id = str(cell.get("node_id")).strip()
        camera_layout = _as_dict(cell.get("camera_layout"))
        layout_camera_ids = [str(x).strip() for x in _as_list(camera_layout.get("camera_ids")) if str(x).strip()]
        if not layout_camera_ids:
            # fallback to plan selected filters camera ids when matrix camera layout object is absent.
            layout_camera_ids = filter_cameras[:]
        observed_cameras = capture_tasks_by_itn.get((identity_id, trajectory_id, node_id), set())
        missing_cameras = [cam for cam in layout_camera_ids if cam not in observed_cameras]
        if missing_cameras:
            _add_issue(
                failures,
                "plan_matrix_capture_task_missing_for_camera_layout",
                "matrix entry cannot find matching capture tasks for all cameras in camera layout.",
                index=idx,
                identity_id=identity_id,
                trajectory_id=trajectory_id,
                node_id=node_id,
                missing_camera_ids=missing_cameras,
            )

    modality_present = {"rgb": 0, "depth": 0, "semantic": 0, "instance": 0, "pose": 0, "calib": 0}
    modality_missing = {"rgb": 0, "depth": 0, "semantic": 0, "instance": 0, "pose": 0, "calib": 0}
    mask_gt_true_count = 0
    sample_with_capture_task_in_plan_count = 0
    sample_with_capture_task_id_count = 0
    sample_capture_matrix_bridge_present_count = 0
    sample_with_trajectory_node_camera_bridge_count = 0
    planned_capture_task_candidate_reference_count = 0
    capture_matrix_bridge_status_counts: dict[str, int] = {}
    sample_identity_ids: set[str] = set()
    identity_mismatch_count = 0
    strict_planned_identity_sample_count = 0
    observed_passthrough_identity_sample_count = 0
    strict_matrix_entry_sample_count = 0
    legacy_or_observed_scene_passthrough_count = 0
    scene_qualification_summary = _as_dict(manifest_payload.get("scene_qualification_summary"))
    has_scene_qualification_summary = bool(scene_qualification_summary)
    sample_scene_key_map: dict[str, dict[str, Any]] = {}
    samples_with_top_level_scene_id = 0

    scene_qualification_field_seen = any(("scene_qualification" in _as_dict(sample)) for sample in samples)
    scene_qualification_required_keys = [
        "minimum_index_artifacts_ready",
        "formal_mask_gt_available",
        "legacy_proxy_candidate_not_promoted",
        "readiness_status",
        "readiness_blocked",
        "readiness_blocked_reasons",
    ]
    for idx, sample in enumerate(samples, start=1):
        missing_keys = [key for key in MIN_REQUIRED_SAMPLE_KEYS if key not in sample]
        if scene_qualification_field_seen and ("scene_qualification" not in sample):
            missing_keys.append("scene_qualification")
        if missing_keys:
            _add_issue(
                failures,
                "sample_min_required_keys_missing",
                "Sample is missing minimum required contract keys.",
                line=idx,
                sample_id=sample.get("sample_id"),
                missing_keys=missing_keys,
            )

        mask_gt = _as_dict(sample.get("mask_gt"))
        if mask_gt.get("is_mask_gt") is True:
            mask_gt_true_count += 1
            _add_issue(
                failures,
                "mask_gt_promotion_forbidden",
                "mask_gt.is_mask_gt=true is currently forbidden without formal evidence integration.",
                line=idx,
                sample_id=sample.get("sample_id"),
            )
        if scene_qualification_field_seen:
            scene_qualification = _as_dict(sample.get("scene_qualification"))
            missing_scene_qualification_keys = [
                key for key in scene_qualification_required_keys if key not in scene_qualification
            ]
            if missing_scene_qualification_keys:
                _add_issue(
                    failures,
                    "sample_scene_qualification_keys_missing",
                    "sample.scene_qualification is missing required keys.",
                    line=idx,
                    sample_id=sample.get("sample_id"),
                    missing_keys=missing_scene_qualification_keys,
                )
            readiness_status = str(scene_qualification.get("readiness_status") or "").strip()
            readiness_blocked = scene_qualification.get("readiness_blocked") is True
            readiness_blocked_reasons = [
                str(x).strip()
                for x in _as_list(scene_qualification.get("readiness_blocked_reasons"))
                if str(x).strip()
            ]
            if (readiness_status == "blocked") != readiness_blocked:
                _add_issue(
                    failures,
                    "sample_scene_qualification_readiness_status_blocked_inconsistent",
                    "scene_qualification.readiness_status and readiness_blocked are inconsistent.",
                    line=idx,
                    sample_id=sample.get("sample_id"),
                    readiness_status=readiness_status,
                    readiness_blocked=readiness_blocked,
                )
            if readiness_blocked and not readiness_blocked_reasons:
                _add_issue(
                    failures,
                    "sample_scene_qualification_blocked_reasons_missing",
                    "scene_qualification.readiness_blocked_reasons must be non-empty when blocked.",
                    line=idx,
                    sample_id=sample.get("sample_id"),
                )
            if (not readiness_blocked) and readiness_blocked_reasons:
                _add_issue(
                    failures,
                    "sample_scene_qualification_blocked_reasons_unexpected",
                    "scene_qualification.readiness_blocked_reasons must be empty when not blocked.",
                    line=idx,
                    sample_id=sample.get("sample_id"),
                )
            if scene_qualification.get("legacy_proxy_candidate_not_promoted") is not True:
                _add_issue(
                    failures,
                    "sample_scene_qualification_non_promotion_guard_mismatch",
                    "scene_qualification.legacy_proxy_candidate_not_promoted must be true.",
                    line=idx,
                    sample_id=sample.get("sample_id"),
                )
            if (scene_qualification.get("formal_mask_gt_available") is True) != (mask_gt.get("availability") == "available"):
                _add_issue(
                    failures,
                    "sample_scene_qualification_formal_mask_gt_available_mismatch",
                    "scene_qualification.formal_mask_gt_available must align with mask_gt.availability.",
                    line=idx,
                    sample_id=sample.get("sample_id"),
                    scene_qualification_formal_mask_gt_available=scene_qualification.get("formal_mask_gt_available"),
                    mask_gt_availability=mask_gt.get("availability"),
                )
            if (scene_qualification.get("formal_mask_gt_available") is True) and (mask_gt.get("is_mask_gt") is True):
                _add_issue(
                    failures,
                    "sample_scene_qualification_forbidden_mask_gt_promotion",
                    "formal mask evidence must not auto-promote sample.mask_gt.is_mask_gt.",
                    line=idx,
                    sample_id=sample.get("sample_id"),
                )

        alignment = _as_dict(sample.get("plan_alignment"))
        sample_identity_id = str(sample.get("identity_id") or "").strip()
        if sample_identity_id:
            sample_identity_ids.add(sample_identity_id)
        identity_matches = alignment.get("observed_identity_matches_planned")
        if identity_matches is None:
            identity_matches = alignment.get("planned_identity_match")
        if identity_matches is None:
            identity_matches = alignment.get("identity_in_plan")
        identity_matches = identity_matches is True
        if identity_matches:
            strict_planned_identity_sample_count += 1
        else:
            identity_mismatch_count += 1
        sample_passthrough = alignment.get("legacy_or_observed_scene_passthrough") is True
        if alignment.get("matrix_entry_in_plan") is True:
            strict_matrix_entry_sample_count += 1
        if sample_passthrough:
            legacy_or_observed_scene_passthrough_count += 1
        if (not identity_matches) or sample_passthrough:
            observed_passthrough_identity_sample_count += 1
        if alignment.get("capture_task_in_plan") is True:
            sample_with_capture_task_in_plan_count += 1
        if str(alignment.get("capture_task_id") or "").strip():
            sample_with_capture_task_id_count += 1
        bridge = _as_dict(sample.get("capture_matrix_bridge"))
        if bridge:
            sample_capture_matrix_bridge_present_count += 1
            if bridge.get("schema_version") != "carla_air_sample_capture_matrix_bridge_v1":
                _add_issue(
                    failures,
                    "sample_capture_matrix_bridge_schema_mismatch",
                    "sample.capture_matrix_bridge schema_version mismatch.",
                    line=idx,
                    sample_id=sample.get("sample_id"),
                    got=bridge.get("schema_version"),
                )
            for bool_key, expected in (
                ("no_silent_identity_rewrite", True),
                ("legacy_or_observed_scene_passthrough_allowed_for_no_mask_index", True),
                ("non_promotion", True),
                ("full_v1_live_dataset_ready", False),
            ):
                if bridge.get(bool_key) is not expected:
                    _add_issue(
                        failures,
                        "sample_capture_matrix_bridge_guard_mismatch",
                        "sample.capture_matrix_bridge guard flag mismatch.",
                        line=idx,
                        sample_id=sample.get("sample_id"),
                        field=bool_key,
                        got=bridge.get(bool_key),
                        expected=expected,
                    )
            bridge_in_plan = bridge.get("trajectory_node_camera_in_plan") is True
            alignment_in_plan = alignment.get("trajectory_node_camera_in_capture_matrix") is True
            if bridge_in_plan != alignment_in_plan:
                _add_issue(
                    failures,
                    "sample_capture_matrix_bridge_plan_alignment_mismatch",
                    "capture_matrix_bridge.trajectory_node_camera_in_plan must align with plan_alignment.",
                    line=idx,
                    sample_id=sample.get("sample_id"),
                    bridge_value=bridge.get("trajectory_node_camera_in_plan"),
                    alignment_value=alignment.get("trajectory_node_camera_in_capture_matrix"),
                )
            bridge_candidate_count = _to_int(bridge.get("planned_capture_task_candidate_count"))
            alignment_candidate_count = _to_int(alignment.get("planned_capture_task_candidate_count"))
            if bridge_candidate_count != alignment_candidate_count:
                _add_issue(
                    failures,
                    "sample_capture_matrix_bridge_candidate_count_mismatch",
                    "capture_matrix_bridge candidate count must align with plan_alignment.",
                    line=idx,
                    sample_id=sample.get("sample_id"),
                    bridge_value=bridge_candidate_count,
                    alignment_value=alignment_candidate_count,
                )
            bridge_candidate_ids = sorted(
                {
                    str(x).strip()
                    for x in _as_list(bridge.get("planned_capture_task_candidate_ids"))
                    if str(x).strip()
                }
            )
            alignment_candidate_ids = sorted(
                {
                    str(x).strip()
                    for x in _as_list(alignment.get("planned_capture_task_candidate_ids"))
                    if str(x).strip()
                }
            )
            if bridge_candidate_ids != alignment_candidate_ids:
                _add_issue(
                    failures,
                    "sample_capture_matrix_bridge_candidate_ids_mismatch",
                    "capture_matrix_bridge candidate ids must align with plan_alignment.",
                    line=idx,
                    sample_id=sample.get("sample_id"),
                    bridge_value=bridge_candidate_ids,
                    alignment_value=alignment_candidate_ids,
                )
            bridge_status = str(bridge.get("bridge_status") or "").strip()
            alignment_status = str(alignment.get("capture_matrix_bridge_status") or "").strip()
            if bridge_status != alignment_status:
                _add_issue(
                    failures,
                    "sample_capture_matrix_bridge_status_mismatch",
                    "capture_matrix_bridge.bridge_status must align with plan_alignment.capture_matrix_bridge_status.",
                    line=idx,
                    sample_id=sample.get("sample_id"),
                    bridge_value=bridge_status,
                    alignment_value=alignment_status,
                )
            allowed_bridge_statuses = {
                "exact_capture_task_match",
                "trajectory_node_camera_passthrough_identity_mismatch",
                "missing_capture_matrix_entry",
            }
            if bridge_status and bridge_status not in allowed_bridge_statuses:
                _add_issue(
                    failures,
                    "sample_capture_matrix_bridge_status_invalid",
                    "capture_matrix_bridge.bridge_status is not recognized.",
                    line=idx,
                    sample_id=sample.get("sample_id"),
                    bridge_status=bridge_status,
                )
            if bridge_in_plan:
                sample_with_trajectory_node_camera_bridge_count += 1
            planned_capture_task_candidate_reference_count += bridge_candidate_count
            if bridge_status:
                capture_matrix_bridge_status_counts[bridge_status] = (
                    capture_matrix_bridge_status_counts.get(bridge_status, 0) + 1
                )
        elif "capture_matrix_bridge" in sample:
            _add_issue(
                failures,
                "sample_capture_matrix_bridge_invalid",
                "sample.capture_matrix_bridge must be an object when present.",
                line=idx,
                sample_id=sample.get("sample_id"),
            )
        identity_switch_contract_present = bool(_as_dict(manifest_payload.get("identity_model_switch_contract")))
        if identity_switch_contract_present and alignment.get("no_silent_identity_rewrite") is not True:
            _add_issue(
                failures,
                "sample_plan_alignment_no_silent_identity_rewrite_flag_missing",
                "sample.plan_alignment.no_silent_identity_rewrite must be true.",
                line=idx,
                sample_id=sample.get("sample_id"),
            )
        if identity_switch_contract_present and alignment.get("planned_identity_rewrite_applied") is not False:
            _add_issue(
                failures,
                "sample_plan_alignment_identity_rewrite_forbidden",
                "sample.plan_alignment.planned_identity_rewrite_applied must be false.",
                line=idx,
                sample_id=sample.get("sample_id"),
                got=alignment.get("planned_identity_rewrite_applied"),
            )

        for key in ("rgb", "depth", "semantic", "instance", "pose", "calib"):
            value = sample.get(key)
            if value in (None, ""):
                modality_missing[key] += 1
            else:
                modality_present[key] += 1

        if sample.get("schema_version") != "carla_air_dataset_sample_v1":
            _add_issue(
                failures,
                "sample_schema_mismatch",
                "sample.schema_version mismatch.",
                line=idx,
                sample_id=sample.get("sample_id"),
                got=sample.get("schema_version"),
            )
        if not str(sample.get("sample_id") or "").strip():
            _add_issue(
                failures,
                "sample_id_missing",
                "sample_id must be non-empty.",
                line=idx,
            )
        source = _as_dict(sample.get("source"))
        if not str(source.get("scene_id") or "").strip():
            _add_issue(
                failures,
                "sample_source_scene_id_missing",
                "sample.source.scene_id must be non-empty.",
                line=idx,
                sample_id=sample.get("sample_id"),
            )
        top_scene_id = str(sample.get("scene_id") or "").strip()
        if top_scene_id:
            samples_with_top_level_scene_id += 1
        if has_scene_qualification_summary and (not top_scene_id):
            _add_issue(
                failures,
                "sample_scene_id_missing_when_scene_qualification_summary_present",
                "sample.scene_id must be non-empty when scene_qualification_summary exists.",
                line=idx,
                sample_id=sample.get("sample_id"),
            )
        if has_scene_qualification_summary:
            source_scene_id = str(source.get("scene_id") or "").strip()
            scene_dir = str(source.get("scene_dir") or "").strip()
            scene_key = _scene_key_for_fields(
                str(sample.get("identity_id") or "").strip(),
                str(sample.get("trajectory_id") or "").strip(),
                str(sample.get("node_id") or "").strip(),
                top_scene_id or source_scene_id,
                scene_dir,
            )
            rec = sample_scene_key_map.setdefault(scene_key, {"splits": set(), "sample_count": 0})
            rec["splits"].add(str(sample.get("split") or "").strip() or "unknown")
            rec["sample_count"] += 1
        if not str(source.get("scene_dir") or "").strip():
            _add_issue(
                failures,
                "sample_source_scene_dir_missing",
                "sample.source.scene_dir must be non-empty.",
                line=idx,
                sample_id=sample.get("sample_id"),
            )
        refs = _as_dict(sample.get("refs"))
        if not str(refs.get("rgb") or "").strip():
            _add_issue(
                failures,
                "sample_refs_rgb_missing",
                "sample.refs.rgb must be non-empty.",
                line=idx,
                sample_id=sample.get("sample_id"),
            )
        timestamp = _as_dict(sample.get("timestamp"))
        if (
            (timestamp.get("timestamp_us") in (None, ""))
            and (timestamp.get("frame_index") in (None, ""))
            and (timestamp.get("unix_ns") in (None, ""))
        ):
            _add_issue(
                failures,
                "sample_timestamp_contract_missing",
                "sample.timestamp must include at least one of timestamp_us/frame_index/unix_ns.",
                line=idx,
                sample_id=sample.get("sample_id"),
            )
        pose_ref = _as_dict(sample.get("pose_ref"))
        if not str(pose_ref.get("row_key") or "").strip():
            _add_issue(
                failures,
                "sample_pose_ref_row_key_missing",
                "sample.pose_ref.row_key must be present.",
                line=idx,
                sample_id=sample.get("sample_id"),
            )

        mask_gt = _as_dict(sample.get("mask_gt"))
        mask_gt_audit = _as_dict(sample.get("mask_gt_audit"))
        if mask_gt.get("pseudo_or_candidate_never_mask_gt") is not True:
            _add_issue(
                failures,
                "sample_mask_gt_pseudo_candidate_contract_mismatch",
                "mask_gt.pseudo_or_candidate_never_mask_gt must be true.",
                line=idx,
                sample_id=sample.get("sample_id"),
            )
        mask_gt_availability = str(mask_gt.get("availability") or "").strip()
        formal_mask_gt_found = mask_gt_audit.get("formal_mask_gt_found") is True
        if (mask_gt_availability == "available") and (not formal_mask_gt_found):
            _add_issue(
                failures,
                "sample_mask_gt_availability_without_formal_evidence",
                "mask_gt.availability=available requires mask_gt_audit.formal_mask_gt_found=true.",
                line=idx,
                sample_id=sample.get("sample_id"),
            )
        if mask_gt_audit.get("legacy_masks_gt_directory_is_not_formal_mask_gt_without_evidence") is not True:
            _add_issue(
                failures,
                "sample_mask_gt_legacy_directory_contract_mismatch",
                "mask_gt_audit.legacy_masks_gt_directory_is_not_formal_mask_gt_without_evidence must be true.",
                line=idx,
                sample_id=sample.get("sample_id"),
            )
    if sample_count > 0 and sample_capture_matrix_bridge_present_count == 0:
        _add_issue(
            warnings,
            "sample_capture_matrix_bridge_missing_legacy_compatible",
            "sample.capture_matrix_bridge is absent from all samples; treated as legacy-compatible run.",
        )
    if (not scene_qualification_field_seen) and samples:
        _add_issue(
            warnings,
            "sample_scene_qualification_missing_legacy_compatible",
            "scene_qualification not found in samples; treated as legacy-compatible run.",
        )

    manifest_modality_summary = _as_dict(manifest_payload.get("modality_summary"))
    manifest_present_by_modality = _as_dict(manifest_modality_summary.get("present_count_by_modality"))
    manifest_missing_by_modality = _as_dict(manifest_modality_summary.get("missing_count_by_modality"))

    resolved_present_by_modality: dict[str, int] = {}
    resolved_missing_by_modality: dict[str, int] = {}
    for key in ("rgb", "depth", "semantic", "instance", "pose", "calib"):
        manifest_present_count = _to_int(manifest_present_by_modality.get(key))
        manifest_missing_count = _to_int(manifest_missing_by_modality.get(key))
        resolved_present_by_modality[key] = manifest_present_count if manifest_present_count >= 0 else modality_present.get(key, 0)
        resolved_missing_by_modality[key] = manifest_missing_count if manifest_missing_count >= 0 else modality_missing.get(key, 0)

    complete_required_modalities = ("rgb", "depth", "semantic", "instance", "pose", "calib")
    complete_sample_count = 0
    sidecar_missing_modality_ids: dict[str, list[str]] = {"depth": [], "semantic": [], "instance": []}
    sidecar_missing_modality_id_summary: dict[str, dict[str, Any]] = {}
    mask_gt_available_count_from_samples = 0
    for sample in samples:
        sample_complete = True
        for key in complete_required_modalities:
            if sample.get(key) in (None, ""):
                sample_complete = False
                if key in sidecar_missing_modality_ids:
                    sample_id = str(sample.get("sample_id") or "").strip()
                    sidecar_missing_modality_ids[key].append(sample_id or "<missing_sample_id>")
        if sample_complete:
            complete_sample_count += 1

        sample_mask_gt = _as_dict(sample.get("mask_gt"))
        if str(sample_mask_gt.get("availability") or "").strip() == "available":
            mask_gt_available_count_from_samples += 1

    manifest_mask_gt_summary = _as_dict(manifest_payload.get("mask_gt_availability_summary"))
    manifest_mask_gt_available_count = _to_int(manifest_mask_gt_summary.get("available_count"))
    mask_gt_available_count = (
        manifest_mask_gt_available_count if manifest_mask_gt_available_count >= 0 else mask_gt_available_count_from_samples
    )
    mask_gt_unavailable_sample_count = max(sample_count - mask_gt_available_count, 0)
    complete_fraction = (float(complete_sample_count) / float(sample_count)) if sample_count > 0 else 0.0
    for key, missing_ids in sidecar_missing_modality_ids.items():
        sidecar_missing_modality_id_summary[key] = _preview_items(missing_ids)

    sidecar_quality_summary = {
        "sample_count": sample_count,
        "complete_rgb_depth_semantic_instance_pose_calib_count": complete_sample_count,
        "complete_fraction": complete_fraction,
        "present_count_by_modality": resolved_present_by_modality,
        "missing_count_by_modality": resolved_missing_by_modality,
        "sidecar_missing_modality_id_summary": sidecar_missing_modality_id_summary,
        "mask_gt_available_count": mask_gt_available_count,
        "mask_gt_unavailable_sample_count": mask_gt_unavailable_sample_count,
        "no_mask_sample_count": mask_gt_unavailable_sample_count,
    }

    manifest_schema_coverage_summary = _as_dict(manifest_payload.get("sample_schema_coverage_summary"))
    computed_schema_coverage_summary = _compute_sample_schema_coverage_summary(samples)
    expected_schema_coverage_manifest_run_id = (
        str(run_contract_payload.get("run_id") or "").strip()
        or str(plan_payload.get("run_id") or "").strip()
        or str(manifest_payload.get("run_id") or "").strip()
    )
    recomputed_schema_coverage_manifest = _build_sample_schema_coverage_manifest_from_summary(
        run_id=expected_schema_coverage_manifest_run_id,
        sample_schema_coverage_summary=computed_schema_coverage_summary,
    )
    if manifest_payload:
        if not manifest_schema_coverage_summary:
            _add_issue(
                warnings,
                "sample_schema_coverage_summary_missing_legacy_compatible",
                "dataset_manifest.sample_schema_coverage_summary is missing; treated as legacy/transition-compatible run.",
            )
        else:
            for key in (
                "schema_version",
                "sample_count",
                "required_fields",
                "field_present_count",
                "field_missing_count",
                "field_presence_required_even_when_sidecar_unavailable",
                "sidecar_unavailable_is_reference_or_availability_not_mask_gt",
                "candidate_proxy_pseudo_legacy_not_promoted_to_mask_gt",
            ):
                observed = manifest_schema_coverage_summary.get(key)
                expected = computed_schema_coverage_summary.get(key)
                if observed != expected:
                    _add_issue(
                        failures,
                        "sample_schema_coverage_summary_mismatch",
                        "dataset_manifest.sample_schema_coverage_summary mismatch against recomputed coverage.",
                        field=key,
                        manifest_value=observed,
                        computed_value=expected,
                    )
    if sample_schema_coverage_manifest_payload:
        observed_run_id = str(sample_schema_coverage_manifest_payload.get("run_id") or "").strip()
        if expected_schema_coverage_manifest_run_id and observed_run_id != expected_schema_coverage_manifest_run_id:
            _add_issue(
                failures,
                "sample_schema_coverage_manifest_run_id_mismatch",
                "sample_schema_coverage_manifest.run_id must align with run_contract/plan/manifest run_id.",
                expected=expected_schema_coverage_manifest_run_id,
                got=observed_run_id,
            )
        for flag_name, expected_value in (
            ("starts_runtime", False),
            ("writes_scene_outputs", False),
            ("non_promotion", True),
            ("full_v1_live_dataset_ready", False),
            ("sidecar_unavailable_is_reference_or_availability_not_mask_gt", True),
            ("candidate_proxy_pseudo_legacy_not_promoted_to_mask_gt", True),
        ):
            if sample_schema_coverage_manifest_payload.get(flag_name) is not expected_value:
                _add_issue(
                    failures,
                    "sample_schema_coverage_manifest_guard_flag_mismatch",
                    "sample_schema_coverage_manifest guard flag mismatch.",
                    field=flag_name,
                    got=sample_schema_coverage_manifest_payload.get(flag_name),
                    expected=expected_value,
                )
        for key in (
            "sample_count",
            "required_fields",
            "field_present_count",
            "field_missing_count",
            "field_presence_required_even_when_sidecar_unavailable",
        ):
            manifest_value = sample_schema_coverage_manifest_payload.get(key)
            computed_value = recomputed_schema_coverage_manifest.get(key)
            if manifest_value != computed_value:
                _add_issue(
                    failures,
                    "sample_schema_coverage_manifest_mismatch",
                    "sample_schema_coverage_manifest mismatch against recomputed schema coverage facts.",
                    field=key,
                    manifest_value=manifest_value,
                    computed_value=computed_value,
                )
        stable_hashes = _as_dict(sample_schema_coverage_manifest_payload.get("stable_hashes"))
        observed_sha = str(stable_hashes.get("canonical_payload_sha256") or "").strip()
        expected_sha = str(_as_dict(recomputed_schema_coverage_manifest.get("stable_hashes")).get("canonical_payload_sha256") or "").strip()
        if not observed_sha:
            _add_issue(
                failures,
                "sample_schema_coverage_manifest_stable_hash_missing",
                "sample_schema_coverage_manifest.stable_hashes.canonical_payload_sha256 must be non-empty.",
            )
        elif observed_sha != expected_sha:
            _add_issue(
                failures,
                "sample_schema_coverage_manifest_stable_hash_mismatch",
                "sample_schema_coverage_manifest stable hash mismatch against canonical payload hash.",
                manifest_value=observed_sha,
                computed_value=expected_sha,
            )
        if manifest_schema_coverage_summary:
            for key in (
                "sample_count",
                "required_fields",
                "field_present_count",
                "field_missing_count",
                "field_presence_required_even_when_sidecar_unavailable",
            ):
                if sample_schema_coverage_manifest_payload.get(key) != manifest_schema_coverage_summary.get(key):
                    _add_issue(
                        failures,
                        "sample_schema_coverage_manifest_vs_dataset_manifest_summary_mismatch",
                        "standalone sample_schema_coverage_manifest must align with dataset_manifest.sample_schema_coverage_summary.",
                        field=key,
                    )

    manifest_sidecar_quality_matrix = _as_dict(manifest_payload.get("sidecar_quality_matrix"))
    computed_sidecar_quality_matrix = _compute_sidecar_quality_matrix(samples)
    if manifest_payload:
        if not manifest_sidecar_quality_matrix:
            _add_issue(
                warnings,
                "sidecar_quality_matrix_missing_legacy_compatible",
                "dataset_manifest.sidecar_quality_matrix is missing; treated as legacy/transition-compatible run.",
            )
        else:
            if manifest_sidecar_quality_matrix != computed_sidecar_quality_matrix:
                _add_issue(
                    failures,
                    "sidecar_quality_matrix_mismatch",
                    "dataset_manifest.sidecar_quality_matrix mismatch against recomputed sidecar quality matrix.",
                    manifest_digest=_canonical_json_sha256(manifest_sidecar_quality_matrix),
                    computed_digest=_canonical_json_sha256(computed_sidecar_quality_matrix),
                )
            for flag_name, expected_value in (
                ("sidecar_unavailable_is_reference_or_availability_not_mask_gt", True),
                ("non_promotion", True),
                ("full_v1_live_dataset_ready", False),
            ):
                if manifest_sidecar_quality_matrix.get(flag_name) is not expected_value:
                    _add_issue(
                        failures,
                        "sidecar_quality_matrix_guard_flag_mismatch",
                        "dataset_manifest.sidecar_quality_matrix guard flag mismatch.",
                        field=flag_name,
                        got=manifest_sidecar_quality_matrix.get(flag_name),
                        expected=expected_value,
                    )
    if sidecar_quality_manifest_payload:
        expected_run_id = (
            str(run_contract_payload.get("run_id") or "").strip()
            or str(plan_payload.get("run_id") or "").strip()
            or str(manifest_payload.get("run_id") or "").strip()
        )
        observed_run_id = str(sidecar_quality_manifest_payload.get("run_id") or "").strip()
        if expected_run_id and observed_run_id != expected_run_id:
            _add_issue(
                failures,
                "sidecar_quality_manifest_run_id_mismatch",
                "sidecar_quality_manifest.run_id must align with run_contract/plan/manifest run_id.",
                manifest_value=observed_run_id,
                expected=expected_run_id,
            )
        for flag_name, expected_value in (
            ("starts_runtime", False),
            ("writes_scene_outputs", False),
            ("non_promotion", True),
            ("full_v1_live_dataset_ready", False),
        ):
            if sidecar_quality_manifest_payload.get(flag_name) is not expected_value:
                _add_issue(
                    failures,
                    "sidecar_quality_manifest_guard_flag_mismatch",
                    "sidecar_quality_manifest guard flag mismatch.",
                    field=flag_name,
                    got=sidecar_quality_manifest_payload.get(flag_name),
                    expected=expected_value,
                )
        expected_sidecar_quality_manifest = {
            "sample_count": sample_count,
            "complete_rgb_depth_semantic_instance_pose_calib_count": _to_int(
                sidecar_quality_summary.get("complete_rgb_depth_semantic_instance_pose_calib_count")
            ),
            "complete_fraction": float(sidecar_quality_summary.get("complete_fraction") or 0.0),
            "present_count_by_modality": {
                key: _to_int(_as_dict(sidecar_quality_summary.get("present_count_by_modality")).get(key))
                for key in ("rgb", "depth", "semantic", "instance", "pose", "calib")
            },
            "missing_count_by_modality": {
                key: _to_int(_as_dict(sidecar_quality_summary.get("missing_count_by_modality")).get(key))
                for key in ("rgb", "depth", "semantic", "instance", "pose", "calib")
            },
            "mask_gt_available_count": _to_int(sidecar_quality_summary.get("mask_gt_available_count")),
            "no_mask_sample_count": _to_int(sidecar_quality_summary.get("no_mask_sample_count")),
            "by_split": _as_list(computed_sidecar_quality_matrix.get("by_split")),
            "by_scene": _as_list(computed_sidecar_quality_matrix.get("by_scene")),
            "by_scene_split": _as_list(computed_sidecar_quality_matrix.get("by_scene_split")),
            "split_count": _to_int(computed_sidecar_quality_matrix.get("split_count")),
            "scene_count": _to_int(computed_sidecar_quality_matrix.get("scene_count")),
            "scene_split_count": _to_int(computed_sidecar_quality_matrix.get("scene_split_count")),
        }
        for key, expected_value in expected_sidecar_quality_manifest.items():
            observed_value = sidecar_quality_manifest_payload.get(key)
            if observed_value != expected_value:
                _add_issue(
                    failures,
                    "sidecar_quality_manifest_computed_mismatch",
                    "sidecar_quality_manifest field mismatch against verifier recomputation.",
                    field=key,
                    manifest_value=observed_value,
                    computed_value=expected_value,
                )
        observed_hashes = _as_dict(sidecar_quality_manifest_payload.get("stable_hashes"))
        expected_hashes = {
            "overall_digest": _canonical_json_sha256(_as_dict(computed_sidecar_quality_matrix.get("overall"))),
            "by_split_digest": _canonical_json_sha256(_as_list(computed_sidecar_quality_matrix.get("by_split"))),
            "by_scene_digest": _canonical_json_sha256(_as_list(computed_sidecar_quality_matrix.get("by_scene"))),
            "by_scene_split_digest": _canonical_json_sha256(
                _as_list(computed_sidecar_quality_matrix.get("by_scene_split"))
            ),
        }
        manifest_without_payload_digest = dict(sidecar_quality_manifest_payload)
        stable_without_payload_digest = dict(observed_hashes)
        stable_without_payload_digest.pop("manifest_payload_digest_without_manifest_digest", None)
        manifest_without_payload_digest["stable_hashes"] = stable_without_payload_digest
        expected_hashes["manifest_payload_digest_without_manifest_digest"] = _canonical_json_sha256(
            manifest_without_payload_digest
        )
        for key, expected_value in expected_hashes.items():
            observed_value = observed_hashes.get(key)
            if observed_value != expected_value:
                _add_issue(
                    failures,
                    "sidecar_quality_manifest_stable_hash_mismatch",
                    "sidecar_quality_manifest.stable_hashes entry mismatch against verifier recomputation.",
                    field=key,
                    manifest_value=observed_value,
                    computed_value=expected_value,
                )
    if no_mask_non_promotion_manifest_payload:
        expected_run_id = (
            str(run_contract_payload.get("run_id") or "").strip()
            or str(plan_payload.get("run_id") or "").strip()
            or str(manifest_payload.get("run_id") or "").strip()
        )
        observed_run_id = str(no_mask_non_promotion_manifest_payload.get("run_id") or "").strip()
        if expected_run_id and observed_run_id != expected_run_id:
            _add_issue(
                failures,
                "no_mask_non_promotion_manifest_run_id_mismatch",
                "no_mask_non_promotion_manifest.run_id must align with run_contract/plan/manifest run_id.",
                manifest_value=observed_run_id,
                expected=expected_run_id,
            )
        for flag_name, expected_value in (
            ("starts_runtime", False),
            ("writes_scene_outputs", False),
            ("non_promotion", True),
            ("full_v1_live_dataset_ready", False),
        ):
            if no_mask_non_promotion_manifest_payload.get(flag_name) is not expected_value:
                _add_issue(
                    failures,
                    "no_mask_non_promotion_manifest_guard_flag_mismatch",
                    "no_mask_non_promotion_manifest guard flag mismatch.",
                    field=flag_name,
                    got=no_mask_non_promotion_manifest_payload.get(flag_name),
                    expected=expected_value,
                )
        expected_mask_gt_available_count = mask_gt_available_count_from_samples
        expected_no_mask_sample_count = sample_count - mask_gt_available_count_from_samples
        for key, expected_value in (
            ("sample_count", sample_count),
            ("mask_gt_available_count", expected_mask_gt_available_count),
            ("no_mask_sample_count", expected_no_mask_sample_count),
        ):
            if _to_int(no_mask_non_promotion_manifest_payload.get(key)) != int(expected_value):
                _add_issue(
                    failures,
                    "no_mask_non_promotion_manifest_count_mismatch",
                    "no_mask_non_promotion_manifest count field mismatch against verifier recomputation.",
                    field=key,
                    manifest_value=no_mask_non_promotion_manifest_payload.get(key),
                    computed_value=expected_value,
                )
        policy = _as_dict(no_mask_non_promotion_manifest_payload.get("policy"))
        for flag_name in (
            "proxy_candidate_pseudo_legacy_not_promoted_to_mask_gt",
            "legacy_masks_gt_directory_is_not_formal_mask_gt_without_evidence",
            "trusted_mask_gt_requires_explicit_formal_evidence",
            "mask_gt_availability_unavailable_is_not_candidate_proxy_pseudo_promotion",
        ):
            if policy.get(flag_name) is not True:
                _add_issue(
                    failures,
                    "no_mask_non_promotion_manifest_policy_flag_mismatch",
                    "no_mask_non_promotion_manifest.policy flag mismatch.",
                    field=flag_name,
                    got=policy.get(flag_name),
                    expected=True,
                )
        cross_checks = _as_dict(no_mask_non_promotion_manifest_payload.get("cross_checks"))
        manifest_contract_summary_for_cross_check = _as_dict(manifest_payload.get("dataset_run_contract_summary"))
        expected_cross_checks = {
            "dataset_run_contract_summary_sample_count": _to_int(
                manifest_contract_summary_for_cross_check.get("sample_count")
            ),
            "dataset_run_contract_summary_mask_gt_available_count": _to_int(
                manifest_contract_summary_for_cross_check.get("mask_gt_available_count")
            ),
            "dataset_run_contract_summary_no_mask_sample_count": _to_int(
                manifest_contract_summary_for_cross_check.get("no_mask_sample_count")
            ),
        }
        for key, expected_value in expected_cross_checks.items():
            if _to_int(cross_checks.get(key)) != expected_value:
                _add_issue(
                    failures,
                    "no_mask_non_promotion_manifest_cross_check_mismatch",
                    "no_mask_non_promotion_manifest.cross_checks mismatch.",
                    field=key,
                    manifest_value=cross_checks.get(key),
                    computed_value=expected_value,
                )
        observed_hashes = _as_dict(no_mask_non_promotion_manifest_payload.get("stable_hashes"))
        digest_payload = {
            "run_id": observed_run_id,
            "sample_count": sample_count,
            "mask_gt_available_count": expected_mask_gt_available_count,
            "no_mask_sample_count": expected_no_mask_sample_count,
            "starts_runtime": False,
            "writes_scene_outputs": False,
            "non_promotion": True,
            "full_v1_live_dataset_ready": False,
            "proxy_candidate_pseudo_legacy_not_promoted_to_mask_gt": True,
            "legacy_masks_gt_directory_is_not_formal_mask_gt_without_evidence": True,
            "trusted_mask_gt_requires_explicit_formal_evidence": True,
        }
        expected_core_digest = _canonical_json_sha256(digest_payload)
        if str(observed_hashes.get("core_digest_sha256") or "") != expected_core_digest:
            _add_issue(
                failures,
                "no_mask_non_promotion_manifest_core_digest_mismatch",
                "no_mask_non_promotion_manifest.stable_hashes.core_digest_sha256 mismatch.",
                manifest_value=observed_hashes.get("core_digest_sha256"),
                computed_value=expected_core_digest,
            )
        manifest_without_stable_hashes = dict(no_mask_non_promotion_manifest_payload)
        manifest_without_stable_hashes.pop("stable_hashes", None)
        expected_manifest_digest = _canonical_json_sha256(manifest_without_stable_hashes)
        if str(observed_hashes.get("manifest_digest_sha256") or "") != expected_manifest_digest:
            _add_issue(
                failures,
                "no_mask_non_promotion_manifest_digest_mismatch",
                "no_mask_non_promotion_manifest.stable_hashes.manifest_digest_sha256 mismatch.",
                manifest_value=observed_hashes.get("manifest_digest_sha256"),
                computed_value=expected_manifest_digest,
            )
    if deployment_episode_visibility_manifest_payload:
        expected_run_id = (
            str(run_contract_payload.get("run_id") or "").strip()
            or str(plan_payload.get("run_id") or "").strip()
            or str(manifest_payload.get("run_id") or "").strip()
        )
        observed_run_id = str(deployment_episode_visibility_manifest_payload.get("run_id") or "").strip()
        if expected_run_id and observed_run_id != expected_run_id:
            _add_issue(
                failures,
                "deployment_episode_visibility_manifest_run_id_mismatch",
                "deployment_episode_visibility_manifest.run_id must align with run_contract/plan/manifest run_id.",
                manifest_value=observed_run_id,
                expected=expected_run_id,
            )
        for flag_name, expected_value in (
            ("starts_runtime", False),
            ("writes_scene_outputs", False),
            ("non_promotion", True),
            ("full_v1_live_dataset_ready", False),
        ):
            if deployment_episode_visibility_manifest_payload.get(flag_name) is not expected_value:
                _add_issue(
                    failures,
                    "deployment_episode_visibility_manifest_guard_flag_mismatch",
                    "deployment_episode_visibility_manifest guard flag mismatch.",
                    field=flag_name,
                    got=deployment_episode_visibility_manifest_payload.get(flag_name),
                    expected=expected_value,
                )
        expected_top_fields = {
            "episode_count": len(_as_list(episodes_payload.get("episodes"))) if episodes_payload else 0,
            "sample_count_total_from_episodes": deployment_sample_count_total_from_episodes,
            "scene_count_total_from_episodes": len(deployment_scene_ids_union),
            "episode_with_visibility_gap_count": deployment_episode_with_visibility_gap_count,
            "episode_without_samples_count": deployment_episode_without_samples_count,
            "episode_scene_visibility_hash": _hash_text_parts(deployment_episode_scene_rows),
            "episode_sample_order_hash": _hash_text_parts(deployment_episode_sample_order_rows),
            "episode_sample_sorted_hash": _hash_text_parts(deployment_episode_sample_sorted_rows),
        }
        for key, expected_value in expected_top_fields.items():
            observed_value = deployment_episode_visibility_manifest_payload.get(key)
            if key.endswith("_hash"):
                mismatch = str(observed_value or "").strip() != str(expected_value)
            else:
                mismatch = _to_int(observed_value) != int(expected_value)
            if mismatch:
                _add_issue(
                    failures,
                    "deployment_episode_visibility_manifest_top_level_mismatch",
                    "deployment_episode_visibility_manifest top-level field mismatch against verifier recomputation.",
                    field=key,
                    manifest_value=observed_value,
                    computed_value=expected_value,
                )
        observed_split_policy_summary = _normalize_split_policy_summary(
            deployment_episode_visibility_manifest_payload.get("split_policy_summary")
        )
        if split_policy_summary_computed and observed_split_policy_summary != split_policy_summary_computed:
            _add_issue(
                failures,
                "deployment_episode_visibility_manifest_split_policy_summary_mismatch",
                "deployment_episode_visibility_manifest.split_policy_summary mismatch.",
                manifest_value=observed_split_policy_summary,
                computed_value=split_policy_summary_computed,
            )
        observed_split_policy_digest = str(
            deployment_episode_visibility_manifest_payload.get("split_policy_digest") or ""
        ).strip()
        if split_policy_digest_computed and observed_split_policy_digest != split_policy_digest_computed:
            _add_issue(
                failures,
                "deployment_episode_visibility_manifest_split_policy_digest_mismatch",
                "deployment_episode_visibility_manifest.split_policy_digest mismatch.",
                manifest_value=observed_split_policy_digest,
                computed_value=split_policy_digest_computed,
            )
        episode_entries = _as_list(deployment_episode_visibility_manifest_payload.get("episode_entries"))
        episodes_list = _as_list(episodes_payload.get("episodes")) if episodes_payload else []
        if len(episode_entries) != len(episodes_list):
            _add_issue(
                failures,
                "deployment_episode_visibility_manifest_episode_entry_count_mismatch",
                "deployment_episode_visibility_manifest.episode_entries count must match deployment_episodes count.",
                manifest_value=len(episode_entries),
                computed_value=len(episodes_list),
            )
        episodes_by_id: dict[str, dict[str, Any]] = {}
        for ep in episodes_list:
            ep_obj = _as_dict(ep)
            ep_id = str(ep_obj.get("episode_id") or "").strip()
            if ep_id:
                episodes_by_id[ep_id] = ep_obj
        for idx, entry in enumerate(episode_entries, start=1):
            entry_obj = _as_dict(entry)
            ep_id = str(entry_obj.get("episode_id") or "").strip()
            ep_obj = episodes_by_id.get(ep_id)
            if not ep_obj:
                _add_issue(
                    failures,
                    "deployment_episode_visibility_manifest_episode_id_unknown",
                    "deployment_episode_visibility_manifest.episode_entries contains unknown episode_id.",
                    episode_index=idx,
                    episode_id=ep_id,
                )
                continue
            visibility = _as_dict(ep_obj.get("sample_scene_visibility"))
            expected_entry = {
                "split": ep_obj.get("split"),
                "node_ids": _as_list(ep_obj.get("node_ids")),
                "node_pair_id": ep_obj.get("node_pair_id"),
                "primary_node_id": ep_obj.get("primary_node_id"),
                "camera_ids": _as_list(ep_obj.get("camera_ids")),
                "camera_ids_by_node": _as_dict(ep_obj.get("camera_ids_by_node")),
                "configured_filters": _as_dict(visibility.get("configured_filters")),
                "sample_count": _to_int(visibility.get("sample_count")),
                "scene_count": _to_int(visibility.get("scene_count")),
                "scene_ids_sorted_hash": str(visibility.get("scene_ids_sorted_hash") or ""),
                "sample_id_order_hash": str(visibility.get("sample_id_order_hash") or ""),
                "sample_ids_sorted_hash": str(visibility.get("sample_ids_sorted_hash") or ""),
                "first_sample_id": visibility.get("first_sample_id"),
                "last_sample_id": visibility.get("last_sample_id"),
                "visibility_gap": _as_dict(visibility.get("coverage_gaps")),
                "non_promotion": True,
                "full_v1_live_dataset_ready": False,
            }
            for field, expected_value in expected_entry.items():
                observed_value = entry_obj.get(field)
                if isinstance(expected_value, dict):
                    mismatch = _as_dict(observed_value) != expected_value
                elif isinstance(expected_value, list):
                    mismatch = _as_list(observed_value) != expected_value
                elif isinstance(expected_value, int):
                    mismatch = _to_int(observed_value) != expected_value
                else:
                    mismatch = observed_value != expected_value
                if mismatch:
                    _add_issue(
                        failures,
                        "deployment_episode_visibility_manifest_entry_field_mismatch",
                        "deployment_episode_visibility_manifest.episode_entries field mismatch against deployment_episodes visibility.",
                        episode_id=ep_id,
                        field=field,
                        manifest_value=observed_value,
                        computed_value=expected_value,
                    )
        observed_hashes = _as_dict(deployment_episode_visibility_manifest_payload.get("stable_hashes"))
        manifest_without_stable_hashes = dict(deployment_episode_visibility_manifest_payload)
        manifest_without_stable_hashes.pop("stable_hashes", None)
        expected_manifest_digest = _canonical_json_sha256(manifest_without_stable_hashes)
        if str(observed_hashes.get("core_digest_sha256") or "") != expected_manifest_digest:
            _add_issue(
                failures,
                "deployment_episode_visibility_manifest_core_digest_mismatch",
                "deployment_episode_visibility_manifest.stable_hashes.core_digest_sha256 mismatch.",
                manifest_value=observed_hashes.get("core_digest_sha256"),
                computed_value=expected_manifest_digest,
            )
        if str(observed_hashes.get("manifest_digest_sha256") or "") != expected_manifest_digest:
            _add_issue(
                failures,
                "deployment_episode_visibility_manifest_digest_mismatch",
                "deployment_episode_visibility_manifest.stable_hashes.manifest_digest_sha256 mismatch.",
                manifest_value=observed_hashes.get("manifest_digest_sha256"),
                computed_value=expected_manifest_digest,
            )

    mask_gt_non_promotion_verified = mask_gt_true_count == 0

    identity_switch_contract = _as_dict(manifest_payload.get("identity_model_switch_contract"))
    if identity_switch_contract:
        plan_capture_profile = plan_payload.get("capture_profile")
        plan_model_label = None
        if isinstance(plan_capture_profile, dict):
            plan_model_label = plan_capture_profile.get("model_label") or plan_capture_profile.get("label") or plan_capture_profile.get("name")
        elif plan_capture_profile is not None:
            plan_model_label = plan_capture_profile

        expected_planned_identity_ids = sorted(plan_identities)
        expected_observed_identity_ids = sorted(sample_identity_ids)
        expected_all_samples_match = identity_mismatch_count == 0

        if sorted(str(x).strip() for x in _as_list(identity_switch_contract.get("planned_identity_ids")) if str(x).strip()) != expected_planned_identity_ids:
            _add_issue(
                failures,
                "identity_model_switch_contract_planned_identity_ids_mismatch",
                "identity_model_switch_contract.planned_identity_ids mismatch.",
                manifest_value=identity_switch_contract.get("planned_identity_ids"),
                computed_value=expected_planned_identity_ids,
            )
        if sorted(str(x).strip() for x in _as_list(identity_switch_contract.get("observed_sample_identity_ids")) if str(x).strip()) != expected_observed_identity_ids:
            _add_issue(
                failures,
                "identity_model_switch_contract_observed_identity_ids_mismatch",
                "identity_model_switch_contract.observed_sample_identity_ids mismatch.",
                manifest_value=identity_switch_contract.get("observed_sample_identity_ids"),
                computed_value=expected_observed_identity_ids,
            )
        if _to_int(identity_switch_contract.get("strict_planned_identity_sample_count")) != strict_planned_identity_sample_count:
            _add_issue(
                failures,
                "identity_model_switch_contract_strict_count_mismatch",
                "identity_model_switch_contract.strict_planned_identity_sample_count mismatch.",
                manifest_value=identity_switch_contract.get("strict_planned_identity_sample_count"),
                computed_value=strict_planned_identity_sample_count,
            )
        if _to_int(identity_switch_contract.get("observed_passthrough_identity_sample_count")) != observed_passthrough_identity_sample_count:
            _add_issue(
                failures,
                "identity_model_switch_contract_passthrough_count_mismatch",
                "identity_model_switch_contract.observed_passthrough_identity_sample_count mismatch.",
                manifest_value=identity_switch_contract.get("observed_passthrough_identity_sample_count"),
                computed_value=observed_passthrough_identity_sample_count,
            )
        if _to_int(identity_switch_contract.get("identity_mismatch_count")) != identity_mismatch_count:
            _add_issue(
                failures,
                "identity_model_switch_contract_mismatch_count_mismatch",
                "identity_model_switch_contract.identity_mismatch_count mismatch.",
                manifest_value=identity_switch_contract.get("identity_mismatch_count"),
                computed_value=identity_mismatch_count,
            )
        if (identity_switch_contract.get("all_samples_match_planned_identities") is True) != expected_all_samples_match:
            _add_issue(
                failures,
                "identity_model_switch_contract_match_flag_mismatch",
                "identity_model_switch_contract.all_samples_match_planned_identities mismatch.",
                manifest_value=identity_switch_contract.get("all_samples_match_planned_identities"),
                computed_value=expected_all_samples_match,
            )
        if identity_switch_contract.get("no_silent_identity_rewrite") is not True:
            _add_issue(
                failures,
                "identity_model_switch_contract_no_silent_rewrite_flag_missing",
                "identity_model_switch_contract.no_silent_identity_rewrite must be true.",
            )
        if identity_switch_contract.get("non_promotion") is not True:
            _add_issue(
                failures,
                "identity_model_switch_contract_non_promotion_flag_missing",
                "identity_model_switch_contract.non_promotion must be true.",
            )
        if identity_switch_contract.get("requires_ue_carla_import_readback") is not True:
            _add_issue(
                failures,
                "identity_model_switch_contract_requires_import_readback_flag_mismatch",
                "identity_model_switch_contract.requires_ue_carla_import_readback must be true.",
            )
        if identity_switch_contract.get("capture_profile") != plan_capture_profile:
            _add_issue(
                failures,
                "identity_model_switch_contract_capture_profile_mismatch",
                "identity_model_switch_contract.capture_profile must match plan.capture_profile.",
                manifest_value=identity_switch_contract.get("capture_profile"),
                plan_value=plan_capture_profile,
            )
        if identity_switch_contract.get("model_label") != plan_model_label:
            _add_issue(
                failures,
                "identity_model_switch_contract_model_label_mismatch",
                "identity_model_switch_contract.model_label must match derived plan capture_profile model label.",
                manifest_value=identity_switch_contract.get("model_label"),
                plan_value=plan_model_label,
            )
        if identity_model_switch_manifest_payload:
            contract_planned_ids = sorted(
                {str(x).strip() for x in _as_list(identity_switch_contract.get("planned_identity_ids")) if str(x).strip()}
            )
            manifest_planned_ids = sorted(
                {str(x).strip() for x in _as_list(identity_model_switch_manifest_payload.get("planned_identity_ids")) if str(x).strip()}
            )
            if manifest_planned_ids != contract_planned_ids:
                _add_issue(
                    failures,
                    "identity_model_switch_standalone_vs_embedded_planned_ids_mismatch",
                    "identity_model_switch_manifest.planned_identity_ids must match dataset_manifest.identity_model_switch_contract.",
                    standalone_value=manifest_planned_ids,
                    embedded_value=contract_planned_ids,
                )
            contract_observed_ids = sorted(
                {str(x).strip() for x in _as_list(identity_switch_contract.get("observed_sample_identity_ids")) if str(x).strip()}
            )
            manifest_observed_ids = sorted(
                {
                    str(x).strip()
                    for x in _as_list(identity_model_switch_manifest_payload.get("observed_sample_identity_ids"))
                    if str(x).strip()
                }
            )
            if manifest_observed_ids != contract_observed_ids:
                _add_issue(
                    failures,
                    "identity_model_switch_standalone_vs_embedded_observed_ids_mismatch",
                    "identity_model_switch_manifest.observed_sample_identity_ids must match dataset_manifest.identity_model_switch_contract.",
                    standalone_value=manifest_observed_ids,
                    embedded_value=contract_observed_ids,
                )
            if _to_int(identity_model_switch_manifest_payload.get("identity_mismatch_count")) != _to_int(
                identity_switch_contract.get("identity_mismatch_count")
            ):
                _add_issue(
                    failures,
                    "identity_model_switch_standalone_vs_embedded_mismatch_count_mismatch",
                    "identity_model_switch_manifest.identity_mismatch_count must match dataset_manifest.identity_model_switch_contract.",
                    standalone_value=identity_model_switch_manifest_payload.get("identity_mismatch_count"),
                    embedded_value=identity_switch_contract.get("identity_mismatch_count"),
                )
            for flag_key, expected_value in (
                ("non_promotion", True),
                ("no_silent_identity_rewrite", True),
                ("legacy_or_observed_scene_passthrough_allowed_for_no_mask_index", True),
                ("full_v1_live_dataset_ready", False),
                ("starts_runtime", False),
                ("writes_scene_outputs", False),
            ):
                if identity_model_switch_manifest_payload.get(flag_key) is not expected_value:
                    _add_issue(
                        failures,
                        "identity_model_switch_standalone_guard_flag_mismatch",
                        "identity_model_switch_manifest guard flag mismatch in standalone vs embedded cross-check.",
                        field=flag_key,
                        got=identity_model_switch_manifest_payload.get(flag_key),
                        expected=expected_value,
                    )
    elif manifest_payload:
        _add_issue(
            warnings,
            "identity_model_switch_contract_missing_legacy_compatible",
            "identity_model_switch_contract missing; treated as legacy run for compatibility.",
        )

    sample_without_capture_task_in_plan_count = max(sample_count - sample_with_capture_task_in_plan_count, 0)
    identity_alignment_status = (
        "strict_planned_identity_match"
        if sample_count > 0 and identity_mismatch_count == 0
        else "observed_scene_passthrough"
        if sample_count > 0 and identity_mismatch_count > 0
        else "plan_only_or_no_samples"
    )
    identity_model_switch_alignment_summary = {
        "sample_count": sample_count,
        "planned_identity_ids": sorted(plan_identities),
        "observed_sample_identity_ids": sorted(sample_identity_ids),
        "strict_planned_identity_sample_count": strict_planned_identity_sample_count,
        "observed_passthrough_identity_sample_count": observed_passthrough_identity_sample_count,
        "identity_mismatch_count": identity_mismatch_count,
        "all_samples_match_planned_identities": identity_mismatch_count == 0,
        "status": identity_alignment_status,
        "requires_ue_carla_import_readback": True,
        "legacy_or_observed_scene_passthrough_allowed_for_no_mask_index": True,
        "full_v1_live_dataset_ready": False,
        "non_promotion": True,
    }
    manifest_identity_alignment_summary = _as_dict(identity_model_switch_manifest_payload.get("identity_alignment_summary"))
    if manifest_identity_alignment_summary:
        manifest_planned_ids = sorted(
            {str(x).strip() for x in _as_list(manifest_identity_alignment_summary.get("planned_identity_ids")) if str(x).strip()}
        )
        manifest_observed_ids = sorted(
            {str(x).strip() for x in _as_list(manifest_identity_alignment_summary.get("observed_sample_identity_ids")) if str(x).strip()}
        )
        expected_fields = {
            "sample_count": sample_count,
            "strict_planned_identity_sample_count": strict_planned_identity_sample_count,
            "observed_passthrough_identity_sample_count": observed_passthrough_identity_sample_count,
            "identity_mismatch_count": identity_mismatch_count,
        }
        for field, expected in expected_fields.items():
            if _to_int(manifest_identity_alignment_summary.get(field)) != expected:
                _add_issue(
                    failures,
                    "identity_model_switch_manifest_alignment_summary_count_mismatch",
                    "identity_model_switch_manifest.identity_alignment_summary count field mismatch.",
                    field=field,
                    manifest_value=manifest_identity_alignment_summary.get(field),
                    computed_value=expected,
                )
        if manifest_planned_ids != sorted(plan_identities):
            _add_issue(
                failures,
                "identity_model_switch_manifest_alignment_summary_planned_ids_mismatch",
                "identity_alignment_summary.planned_identity_ids must match plan identities.",
                manifest_value=manifest_planned_ids,
                computed_value=sorted(plan_identities),
            )
        if manifest_observed_ids != sorted(sample_identity_ids):
            _add_issue(
                failures,
                "identity_model_switch_manifest_alignment_summary_observed_ids_mismatch",
                "identity_alignment_summary.observed_sample_identity_ids must match sample identities.",
                manifest_value=manifest_observed_ids,
                computed_value=sorted(sample_identity_ids),
            )
        for bool_key, expected in (
            ("all_samples_match_planned_identities", identity_mismatch_count == 0),
            ("requires_ue_carla_import_readback", True),
            ("legacy_or_observed_scene_passthrough_allowed_for_no_mask_index", True),
            ("full_v1_live_dataset_ready", False),
            ("non_promotion", True),
        ):
            if manifest_identity_alignment_summary.get(bool_key) is not expected:
                _add_issue(
                    failures,
                    "identity_model_switch_manifest_alignment_summary_guard_mismatch",
                    "identity_alignment_summary boolean guard mismatch.",
                    field=bool_key,
                    manifest_value=manifest_identity_alignment_summary.get(bool_key),
                    computed_value=expected,
                )
        if str(manifest_identity_alignment_summary.get("status") or "").strip() != identity_alignment_status:
            _add_issue(
                failures,
                "identity_model_switch_manifest_alignment_summary_status_mismatch",
                "identity_alignment_summary.status mismatch.",
                manifest_value=manifest_identity_alignment_summary.get("status"),
                computed_value=identity_alignment_status,
            )
    elif identity_model_switch_manifest_payload:
        _add_issue(
            warnings,
            "identity_model_switch_manifest_alignment_summary_missing_legacy_compatible",
            "identity_model_switch_manifest.identity_alignment_summary is missing; treated as legacy-compatible.",
        )

    passthrough_or_mismatch_observed = (identity_mismatch_count > 0) or (observed_passthrough_identity_sample_count > 0)
    readiness_obj = _as_dict(manifest_payload.get("readiness"))
    if passthrough_or_mismatch_observed:
        for readiness_key in ("multi_identity_live_capture_ready", "full_v1_live_dataset_ready"):
            if readiness_key in readiness_obj and readiness_obj.get(readiness_key) is not False:
                _add_issue(
                    failures,
                    "manifest_readiness_non_promotion_guard_mismatch",
                    "readiness must stay false when observed passthrough/mismatch exists.",
                    readiness_key=readiness_key,
                    got=readiness_obj.get(readiness_key),
                    expected=False,
                )
        if identity_switch_contract:
            for required_true_key in ("non_promotion", "no_silent_identity_rewrite", "requires_ue_carla_import_readback"):
                if identity_switch_contract.get(required_true_key) is not True:
                    _add_issue(
                        failures,
                        "identity_model_switch_contract_non_promotion_guard_mismatch",
                        "identity_model_switch_contract non-promotion guard flags must remain true.",
                        field=required_true_key,
                        got=identity_switch_contract.get(required_true_key),
                        expected=True,
                    )

    if run_summary_payload:
        aligned_run_id = (
            str(run_contract_payload.get("run_id") or "").strip()
            or str(plan_payload.get("run_id") or "").strip()
            or str(manifest_payload.get("run_id") or "").strip()
        )
        field_expectations = {
            "sample_count": sample_count,
            "scene_root_count": len(_as_list(run_contract_payload.get("scene_roots"))) if run_contract_payload else None,
            "scene_output_count": scene_output_count if should_validate_scene_output_manifest else None,
            "blocked_scene_output_count": blocked_scene_output_count if should_validate_scene_output_manifest else None,
            "capture_queue_item_count": len(capture_queue_rows) if should_validate_capture_queue else None,
            "blocked_capture_queue_item_count": capture_queue_blocked_count if should_validate_capture_queue else None,
        }
        if aligned_run_id and ("run_id" in run_summary_payload) and (str(run_summary_payload.get("run_id") or "").strip() != aligned_run_id):
            _add_issue(
                failures,
                "run_summary_run_id_mismatch",
                "run_summary.run_id must align with run_contract/plan/manifest run_id.",
                manifest_value=run_summary_payload.get("run_id"),
                computed_value=aligned_run_id,
            )
        for field_name, expected_value in field_expectations.items():
            if expected_value is None or (field_name not in run_summary_payload):
                continue
            if _to_int(run_summary_payload.get(field_name)) != int(expected_value):
                _add_issue(
                    failures,
                    "run_summary_count_mismatch",
                    "run_summary count field must align with observed artifacts.",
                    field=field_name,
                    manifest_value=run_summary_payload.get(field_name),
                    computed_value=expected_value,
                )
    if identity_mismatch_count > 0:
        _add_issue(
            warnings,
            "identity_model_switch_mismatch_observed_scene_passthrough",
            "Observed sample identity differs from planned identities; passthrough is kept for no-mask index but full live readiness remains blocked.",
            identity_mismatch_count=identity_mismatch_count,
            observed_sample_identity_ids=sorted(sample_identity_ids),
            planned_identity_ids=sorted(plan_identities),
        )

    manifest_capture_alignment = _as_dict(manifest_payload.get("capture_task_alignment_summary"))
    if manifest_payload and manifest_capture_alignment:
        if int(manifest_capture_alignment.get("capture_task_count") or 0) != plan_capture_task_count:
            _add_issue(
                failures,
                "manifest_capture_task_count_mismatch",
                "manifest.capture_task_alignment_summary.capture_task_count mismatch.",
                manifest_capture_task_count=manifest_capture_alignment.get("capture_task_count"),
                plan_capture_task_count=plan_capture_task_count,
            )
        if int(manifest_capture_alignment.get("sample_count") or 0) != sample_count:
            _add_issue(
                failures,
                "manifest_capture_alignment_sample_count_mismatch",
                "manifest.capture_task_alignment_summary.sample_count mismatch.",
                manifest_sample_count=manifest_capture_alignment.get("sample_count"),
                sample_count=sample_count,
            )
        if int(manifest_capture_alignment.get("sample_with_capture_task_count") or 0) != sample_with_capture_task_in_plan_count:
            _add_issue(
                failures,
                "manifest_capture_alignment_with_count_mismatch",
                "manifest sample_with_capture_task_count mismatch.",
                manifest_value=manifest_capture_alignment.get("sample_with_capture_task_count"),
                computed_value=sample_with_capture_task_in_plan_count,
            )
        if int(manifest_capture_alignment.get("sample_without_capture_task_count") or 0) != (
            sample_count - sample_with_capture_task_in_plan_count
        ):
            _add_issue(
                failures,
                "manifest_capture_alignment_without_count_mismatch",
                "manifest sample_without_capture_task_count mismatch.",
                manifest_value=manifest_capture_alignment.get("sample_without_capture_task_count"),
                computed_value=(sample_count - sample_with_capture_task_in_plan_count),
            )
        if "sample_with_trajectory_node_camera_bridge_count" in manifest_capture_alignment:
            if _to_int(manifest_capture_alignment.get("sample_with_trajectory_node_camera_bridge_count")) != (
                sample_with_trajectory_node_camera_bridge_count
            ):
                _add_issue(
                    failures,
                    "manifest_capture_alignment_bridge_count_mismatch",
                    "manifest sample_with_trajectory_node_camera_bridge_count mismatch.",
                    manifest_value=manifest_capture_alignment.get("sample_with_trajectory_node_camera_bridge_count"),
                    computed_value=sample_with_trajectory_node_camera_bridge_count,
                )
        if "sample_without_trajectory_node_camera_bridge_count" in manifest_capture_alignment:
            computed_without_bridge = sample_count - sample_with_trajectory_node_camera_bridge_count
            if _to_int(manifest_capture_alignment.get("sample_without_trajectory_node_camera_bridge_count")) != (
                computed_without_bridge
            ):
                _add_issue(
                    failures,
                    "manifest_capture_alignment_without_bridge_count_mismatch",
                    "manifest sample_without_trajectory_node_camera_bridge_count mismatch.",
                    manifest_value=manifest_capture_alignment.get("sample_without_trajectory_node_camera_bridge_count"),
                    computed_value=computed_without_bridge,
                )
        if "planned_capture_task_candidate_reference_count" in manifest_capture_alignment:
            if _to_int(manifest_capture_alignment.get("planned_capture_task_candidate_reference_count")) != (
                planned_capture_task_candidate_reference_count
            ):
                _add_issue(
                    failures,
                    "manifest_capture_alignment_candidate_reference_count_mismatch",
                    "manifest planned_capture_task_candidate_reference_count mismatch.",
                    manifest_value=manifest_capture_alignment.get("planned_capture_task_candidate_reference_count"),
                    computed_value=planned_capture_task_candidate_reference_count,
                )
        status_counts = _as_dict(manifest_capture_alignment.get("capture_matrix_bridge_status_counts"))
        if status_counts:
            normalized_status_counts = {str(k): _to_int(v) for k, v in status_counts.items() if str(k).strip()}
            if normalized_status_counts != capture_matrix_bridge_status_counts:
                _add_issue(
                    failures,
                    "manifest_capture_alignment_bridge_status_counts_mismatch",
                    "manifest capture_matrix_bridge_status_counts mismatch.",
                    manifest_value=normalized_status_counts,
                    computed_value=capture_matrix_bridge_status_counts,
                )
    elif manifest_payload:
        _add_issue(
            warnings,
            "manifest_capture_alignment_missing",
            "capture_task_alignment_summary missing from dataset_manifest.json.",
        )

    if manifest_payload:
        manifest_modality_summary = _as_dict(manifest_payload.get("modality_summary"))
        present_by_modality = _as_dict(manifest_modality_summary.get("present_count_by_modality"))
        missing_by_modality = _as_dict(manifest_modality_summary.get("missing_count_by_modality"))
        for key in ("rgb", "depth", "semantic", "instance", "pose", "calib"):
            if present_by_modality and (_to_int(present_by_modality.get(key)) != modality_present.get(key, 0)):
                _add_issue(
                    failures,
                    "manifest_modality_present_count_mismatch",
                    "manifest.modality_summary present count mismatch.",
                    modality=key,
                    manifest_value=present_by_modality.get(key),
                    computed_value=modality_present.get(key, 0),
                )
            if missing_by_modality and (_to_int(missing_by_modality.get(key)) != modality_missing.get(key, 0)):
                _add_issue(
                    failures,
                    "manifest_modality_missing_count_mismatch",
                    "manifest.modality_summary missing count mismatch.",
                    modality=key,
                    manifest_value=missing_by_modality.get(key),
                    computed_value=modality_missing.get(key, 0),
                )

        mask_gt_summary = _as_dict(manifest_payload.get("mask_gt_availability_summary"))
        scene_observations = _as_list(manifest_payload.get("scene_observations"))
        if mask_gt_summary and scene_observations:
            avail_stats: dict[str, int] = {}
            for obs in scene_observations:
                probe = _as_dict(_as_dict(obs).get("mask_gt_probe"))
                availability = str(probe.get("availability") or "").strip()
                if availability:
                    avail_stats[availability] = avail_stats.get(availability, 0) + 1
            available_count_from_probe = avail_stats.get("available", 0)
            manifest_available_count = _to_int(mask_gt_summary.get("available_count"))
            if manifest_available_count != available_count_from_probe:
                _add_issue(
                    failures,
                    "manifest_mask_gt_available_count_mismatch_scene_observations",
                    "mask_gt_availability_summary.available_count must match scene_observations mask_gt_probe availability stats.",
                    manifest_available_count=manifest_available_count,
                    scene_observations_available_count=available_count_from_probe,
                )
        if has_scene_qualification_summary:
            summary_scene_count = _to_int(scene_qualification_summary.get("scene_count"))
            if summary_scene_count != len(scene_observations):
                _add_issue(
                    failures,
                    "scene_qualification_summary_scene_count_mismatch",
                    "scene_qualification_summary.scene_count must equal len(scene_observations).",
                    summary_scene_count=summary_scene_count,
                    scene_observations_count=len(scene_observations),
                )
            summary_sample_scene_key_count = _to_int(scene_qualification_summary.get("sample_scene_key_count"))
            if summary_sample_scene_key_count != len(sample_scene_key_map):
                _add_issue(
                    failures,
                    "scene_qualification_summary_sample_scene_key_count_mismatch",
                    "scene_qualification_summary.sample_scene_key_count mismatch.",
                    summary_value=summary_sample_scene_key_count,
                    computed_value=len(sample_scene_key_map),
                )
            computed_duplicate_scene_key_count = sum(1 for rec in sample_scene_key_map.values() if int(rec["sample_count"]) > 1)
            if "multi_sample_scene_key_count" in scene_qualification_summary:
                if _to_int(scene_qualification_summary.get("multi_sample_scene_key_count")) != computed_duplicate_scene_key_count:
                    _add_issue(
                        failures,
                        "scene_qualification_summary_multi_sample_scene_key_count_mismatch",
                        "scene_qualification_summary.multi_sample_scene_key_count mismatch.",
                        summary_value=scene_qualification_summary.get("multi_sample_scene_key_count"),
                        computed_value=computed_duplicate_scene_key_count,
                    )
            if _to_int(scene_qualification_summary.get("duplicate_scene_key_count")) != computed_duplicate_scene_key_count:
                _add_issue(
                    failures,
                    "scene_qualification_summary_duplicate_scene_key_count_mismatch",
                    "scene_qualification_summary.duplicate_scene_key_count mismatch.",
                    summary_value=scene_qualification_summary.get("duplicate_scene_key_count"),
                    computed_value=computed_duplicate_scene_key_count,
                )
            computed_cross_split_scene_key_count = sum(1 for rec in sample_scene_key_map.values() if len(rec["splits"]) > 1)
            if _to_int(scene_qualification_summary.get("cross_split_scene_key_count")) != computed_cross_split_scene_key_count:
                _add_issue(
                    failures,
                    "scene_qualification_summary_cross_split_scene_key_count_mismatch",
                    "scene_qualification_summary.cross_split_scene_key_count mismatch.",
                    summary_value=scene_qualification_summary.get("cross_split_scene_key_count"),
                    computed_value=computed_cross_split_scene_key_count,
                )
            summary_has_cross_split_conflict = scene_qualification_summary.get("has_cross_split_scene_key_conflict") is True
            if summary_has_cross_split_conflict != (computed_cross_split_scene_key_count > 0):
                _add_issue(
                    failures,
                    "scene_qualification_summary_cross_split_conflict_flag_mismatch",
                    "scene_qualification_summary.has_cross_split_scene_key_conflict mismatch.",
                    summary_value=summary_has_cross_split_conflict,
                    computed_value=(computed_cross_split_scene_key_count > 0),
                )
            summary_collisions = _as_list(scene_qualification_summary.get("scene_key_split_collisions"))
            summary_collision_map: dict[str, set[str]] = {}
            for item in summary_collisions:
                obj = _as_dict(item)
                scene_key = str(obj.get("scene_key") or "").strip()
                if not scene_key:
                    continue
                splits = {
                    str(v).strip()
                    for v in _as_list(obj.get("splits"))
                    if str(v).strip()
                }
                if splits:
                    summary_collision_map[scene_key] = splits
            computed_collision_map = {
                scene_key: set(rec["splits"])
                for scene_key, rec in sample_scene_key_map.items()
                if len(rec["splits"]) > 1
            }
            if summary_collision_map != computed_collision_map:
                _add_issue(
                    failures,
                    "scene_qualification_summary_scene_key_split_collisions_mismatch",
                    "scene_qualification_summary.scene_key_split_collisions mismatch.",
                    summary_collision_count=len(summary_collision_map),
                    computed_collision_count=len(computed_collision_map),
                )
            split_total_from_samples = len(samples)
            split_total_from_distribution = sum(split_distribution.values())
            if split_total_from_distribution != split_total_from_samples:
                _add_issue(
                    failures,
                    "split_distribution_total_mismatch_sample_count",
                    "split distribution total must equal sample count.",
                    split_distribution_total=split_total_from_distribution,
                    sample_count=split_total_from_samples,
                )
            split_by_scene_summary = _as_dict(scene_qualification_summary.get("readiness_status_counts"))
            if split_by_scene_summary and (sum(_to_int(v) for v in split_by_scene_summary.values()) != len(scene_observations)):
                _add_issue(
                    failures,
                    "scene_qualification_summary_readiness_status_total_mismatch",
                    "scene_qualification_summary.readiness_status_counts must sum to scene_count.",
                    readiness_status_total=sum(_to_int(v) for v in split_by_scene_summary.values()),
                    scene_count=len(scene_observations),
                )
        if scene_observations:
            samples_by_scene = _count_samples_by_scene(samples)
            for obs in scene_observations:
                obs_obj = _as_dict(obs)
                scene_id = str(obs_obj.get("scene_id") or "").strip()
                scene_key = (
                    str(obs_obj.get("identity_id") or "unknown_identity").strip() or "unknown_identity",
                    str(obs_obj.get("trajectory_id") or "unknown_trajectory").strip() or "unknown_trajectory",
                    str(obs_obj.get("node_id") or "unknown_node").strip() or "unknown_node",
                    scene_id,
                )
                scene_sample_rec = _as_dict(samples_by_scene.get(scene_key))
                if not scene_sample_rec:
                    continue

                camera_coverage = _as_dict(obs_obj.get("camera_coverage"))
                if camera_coverage:
                    camera_ids = sorted(
                        {
                            str(x).strip()
                            for x in _as_list(camera_coverage.get("camera_ids"))
                            if str(x).strip()
                        }
                    )
                    expected_rows_by_camera = {
                        str(k).strip(): _to_int(v)
                        for k, v in _as_dict(camera_coverage.get("rows_by_camera")).items()
                        if str(k).strip()
                    }
                    expected_valid_rows_by_camera = {
                        str(k).strip(): _to_int(v)
                        for k, v in _as_dict(camera_coverage.get("valid_rows_by_camera")).items()
                        if str(k).strip()
                    }
                    sample_camera_rows = {
                        str(k).strip(): _to_int(v)
                        for k, v in _as_dict(scene_sample_rec.get("camera_rows")).items()
                        if str(k).strip()
                    }
                    if camera_ids and camera_ids != sorted(sample_camera_rows.keys()):
                        _add_issue(
                            failures,
                            "scene_observation_camera_coverage_camera_ids_mismatch",
                            "scene_observations[].camera_coverage.camera_ids must match sample camera_id set for scene.",
                            scene_id=scene_id,
                            manifest_value=camera_ids,
                            computed_value=sorted(sample_camera_rows.keys()),
                        )
                    if expected_rows_by_camera and expected_rows_by_camera != sample_camera_rows:
                        _add_issue(
                            failures,
                            "scene_observation_camera_coverage_rows_by_camera_mismatch",
                            "scene_observations[].camera_coverage.rows_by_camera must match per-scene sample camera counts.",
                            scene_id=scene_id,
                            manifest_value=expected_rows_by_camera,
                            computed_value=sample_camera_rows,
                        )
                    if expected_valid_rows_by_camera and expected_valid_rows_by_camera != sample_camera_rows:
                        _add_issue(
                            failures,
                            "scene_observation_camera_coverage_valid_rows_by_camera_mismatch",
                            "scene_observations[].camera_coverage.valid_rows_by_camera must align with indexed sample camera counts.",
                            scene_id=scene_id,
                            manifest_value=expected_valid_rows_by_camera,
                            computed_value=sample_camera_rows,
                        )
                else:
                    _add_issue(
                        warnings,
                        "scene_observation_camera_coverage_missing_legacy_compatible",
                        "scene_observations[].camera_coverage is missing; treated as legacy-compatible.",
                        scene_id=scene_id,
                    )

                timestamp_coverage = _as_dict(obs_obj.get("timestamp_coverage"))
                if timestamp_coverage:
                    valid_row_count = _to_int(timestamp_coverage.get("valid_row_count"))
                    scene_qualification = _as_dict(obs_obj.get("scene_qualification"))
                    scene_valid_rows = _to_int(scene_qualification.get("frame_rows_valid_for_index"))
                    scene_sample_count = _to_int(scene_sample_rec.get("sample_count"))
                    if valid_row_count not in {scene_sample_count, scene_valid_rows}:
                        _add_issue(
                            failures,
                            "scene_observation_timestamp_coverage_valid_row_count_mismatch",
                            "scene_observations[].timestamp_coverage.valid_row_count must align with scene sample_count or frame_rows_valid_for_index.",
                            scene_id=scene_id,
                            manifest_value=valid_row_count,
                            sample_count=scene_sample_count,
                            frame_rows_valid_for_index=scene_valid_rows,
                        )
                else:
                    _add_issue(
                        warnings,
                        "scene_observation_timestamp_coverage_missing_legacy_compatible",
                        "scene_observations[].timestamp_coverage is missing; treated as legacy-compatible.",
                        scene_id=scene_id,
                    )

                calib_rig_summary = _as_dict(obs_obj.get("calib_rig_summary"))
                if calib_rig_summary:
                    rig_exists = calib_rig_summary.get("exists") is True
                    calib_present = _as_dict(obs_obj.get("modality_availability")).get("calib") is True
                    if rig_exists and not calib_present:
                        _add_issue(
                            failures,
                            "scene_observation_calib_rig_summary_exists_mismatch",
                            "scene_observations[].calib_rig_summary.exists=true requires modality_availability.calib=true.",
                            scene_id=scene_id,
                            rig_exists=rig_exists,
                            modality_calib=calib_present,
                        )
                else:
                    _add_issue(
                        warnings,
                        "scene_observation_calib_rig_summary_missing_legacy_compatible",
                        "scene_observations[].calib_rig_summary is missing; treated as legacy-compatible.",
                        scene_id=scene_id,
                    )

                sidecar_missing_summary = _as_dict(obs_obj.get("sidecar_missing_summary"))
                if sidecar_missing_summary:
                    present_set = {
                        str(x).strip()
                        for x in _as_list(sidecar_missing_summary.get("present"))
                        if str(x).strip()
                    }
                    artifact_counts = _as_dict(obs_obj.get("artifact_counts"))
                    if ("depth" in present_set) != (_to_int(artifact_counts.get("depth")) > 0):
                        _add_issue(
                            failures,
                            "scene_observation_sidecar_depth_mismatch",
                            "scene_observations[].sidecar_missing_summary depth presence must align with artifact_counts.depth.",
                            scene_id=scene_id,
                        )
                    if ("semantic" in present_set) != (_to_int(artifact_counts.get("semantic")) > 0):
                        _add_issue(
                            failures,
                            "scene_observation_sidecar_semantic_mismatch",
                            "scene_observations[].sidecar_missing_summary semantic presence must align with artifact_counts.semantic.",
                            scene_id=scene_id,
                        )
                    if ("instance" in present_set) != (_to_int(artifact_counts.get("instance")) > 0):
                        _add_issue(
                            failures,
                            "scene_observation_sidecar_instance_mismatch",
                            "scene_observations[].sidecar_missing_summary instance presence must align with artifact_counts.instance.",
                            scene_id=scene_id,
                        )
                else:
                    _add_issue(
                        warnings,
                        "scene_observation_sidecar_missing_summary_missing_legacy_compatible",
                        "scene_observations[].sidecar_missing_summary is missing; treated as legacy-compatible.",
                        scene_id=scene_id,
                    )

        capture_task_alignment_present_count = 0
        capture_task_exact_match_scene_count = 0
        trajectory_node_camera_plan_match_scene_count = 0
        identity_passthrough_mismatch_scene_count = 0
        scene_capture_alignment_observed_identity_ids: set[str] = set()
        scene_capture_alignment_planned_identity_ids: set[str] = set()
        if should_validate_scene_membership_manifest and scene_membership_manifest_payload:
            recomputed_cross_split_scene_key_count = sum(1 for rec in sample_scene_key_map.values() if len(rec["splits"]) > 1)
            if scene_membership_manifest_payload.get("schema_version") != "carla_air_scene_membership_manifest_v1":
                _add_issue(
                    failures,
                    "scene_membership_manifest_schema_mismatch",
                    "scene_membership_manifest schema_version mismatch.",
                    got=scene_membership_manifest_payload.get("schema_version"),
                    expected="carla_air_scene_membership_manifest_v1",
                )
            for bool_key, expected in (
                ("starts_runtime", False),
                ("writes_scene_outputs", False),
                ("non_promotion", True),
                ("full_v1_live_dataset_ready", False),
            ):
                if bool_key in scene_membership_manifest_payload and scene_membership_manifest_payload.get(bool_key) is not expected:
                    _add_issue(
                        failures,
                        "scene_membership_manifest_guard_flag_mismatch",
                        "scene_membership_manifest guard boolean mismatch.",
                        field=bool_key,
                        got=scene_membership_manifest_payload.get(bool_key),
                        expected=expected,
                    )

            scene_entries = _as_list(scene_membership_manifest_payload.get("scene_entries"))
            if _to_int(scene_membership_manifest_payload.get("sample_count")) != sample_count:
                _add_issue(
                    failures,
                    "scene_membership_manifest_sample_count_mismatch",
                    "scene_membership_manifest.sample_count must equal dataset sample count.",
                    manifest_value=scene_membership_manifest_payload.get("sample_count"),
                    computed_value=sample_count,
                )
            if _to_int(scene_membership_manifest_payload.get("scene_count")) != len(scene_entries):
                _add_issue(
                    failures,
                    "scene_membership_manifest_scene_count_entry_mismatch",
                    "scene_membership_manifest.scene_count must equal len(scene_entries).",
                    manifest_value=scene_membership_manifest_payload.get("scene_count"),
                    computed_value=len(scene_entries),
                )

            if _to_int(scene_membership_manifest_payload.get("cross_split_scene_key_count")) != recomputed_cross_split_scene_key_count:
                _add_issue(
                    failures,
                    "scene_membership_manifest_cross_split_scene_key_count_mismatch",
                    "scene_membership_manifest.cross_split_scene_key_count mismatch.",
                    manifest_value=scene_membership_manifest_payload.get("cross_split_scene_key_count"),
                    computed_value=recomputed_cross_split_scene_key_count,
                )

            summary_collision_map: dict[str, set[str]] = {}
            for item in _as_list(scene_membership_manifest_payload.get("scene_key_split_collisions")):
                obj = _as_dict(item)
                scene_key = str(obj.get("scene_key") or "").strip()
                if not scene_key:
                    continue
                splits_set = {str(v).strip() for v in _as_list(obj.get("splits")) if str(v).strip()}
                if splits_set:
                    summary_collision_map[scene_key] = splits_set
            computed_collision_map = {
                scene_key: set(rec["splits"])
                for scene_key, rec in sample_scene_key_map.items()
                if len(rec["splits"]) > 1
            }
            if summary_collision_map != computed_collision_map:
                _add_issue(
                    failures,
                    "scene_membership_manifest_scene_key_split_collisions_mismatch",
                    "scene_membership_manifest.scene_key_split_collisions mismatch.",
                    summary_collision_count=len(summary_collision_map),
                    computed_collision_count=len(computed_collision_map),
                )

            manifest_split_distribution = {
                str(k).strip(): _to_int(v)
                for k, v in _as_dict(scene_membership_manifest_payload.get("split_distribution")).items()
                if str(k).strip()
            }
            if manifest_split_distribution and manifest_split_distribution != split_distribution:
                _add_issue(
                    failures,
                    "scene_membership_manifest_split_distribution_mismatch",
                    "scene_membership_manifest.split_distribution must match dataset_splits/sample distribution.",
                    manifest_value=manifest_split_distribution,
                    computed_value=split_distribution,
                )
            split_strategy = str(splits_payload.get("split_strategy") or "").strip()
            if split_strategy:
                for strategy_key in ("split_strategy", "selected_split_strategy"):
                    observed = str(scene_membership_manifest_payload.get(strategy_key) or "").strip()
                    if observed and observed != split_strategy:
                        _add_issue(
                            failures,
                            "scene_membership_manifest_split_strategy_mismatch",
                            "scene_membership_manifest split strategy must match dataset_splits.",
                            field=strategy_key,
                            manifest_value=observed,
                            expected=split_strategy,
                        )
            if "not_random_frame_split" in scene_membership_manifest_payload and (
                scene_membership_manifest_payload.get("not_random_frame_split") != splits_payload.get("not_random_frame_split")
            ):
                _add_issue(
                    failures,
                    "scene_membership_manifest_not_random_frame_split_mismatch",
                    "scene_membership_manifest.not_random_frame_split must match dataset_splits.not_random_frame_split.",
                    manifest_value=scene_membership_manifest_payload.get("not_random_frame_split"),
                    expected=splits_payload.get("not_random_frame_split"),
                )

            planned_identity_ids_by_tnc: dict[tuple[str, str, str], set[str]] = {}
            planned_tnc_keys: set[tuple[str, str, str]] = set()
            for task in plan_capture_tasks:
                task_obj = _as_dict(task)
                trajectory_id = str(task_obj.get("trajectory_id") or "").strip()
                node_id = str(task_obj.get("node_id") or "").strip()
                camera_id = str(task_obj.get("camera_id") or "").strip()
                identity_id = str(task_obj.get("identity_id") or "").strip()
                if trajectory_id and node_id and camera_id:
                    planned_tnc_keys.add((trajectory_id, node_id, camera_id))
                    if identity_id:
                        planned_identity_ids_by_tnc.setdefault((trajectory_id, node_id, camera_id), set()).add(identity_id)

            sample_scene_aggregate: dict[tuple[str, str, str, str], dict[str, Any]] = {}
            for sample in samples:
                source = _as_dict(sample.get("source"))
                scene_id = str(sample.get("scene_id") or source.get("scene_id") or "").strip()
                scene_dir = str(source.get("scene_dir") or "").strip()
                identity_id = str(sample.get("identity_id") or "unknown_identity").strip() or "unknown_identity"
                trajectory_id = str(sample.get("trajectory_id") or "unknown_trajectory").strip() or "unknown_trajectory"
                node_id = str(sample.get("node_id") or "unknown_node").strip() or "unknown_node"
                key = (identity_id, trajectory_id, node_id, scene_id or scene_dir)
                rec = sample_scene_aggregate.setdefault(
                    key,
                    {
                        "sample_count": 0,
                        "split_names": set(),
                        "sample_with_capture_task_count": 0,
                        "sample_without_capture_task_count": 0,
                        "capture_task_ids": set(),
                        "trajectory_node_camera_plan_match_count": 0,
                        "trajectory_node_camera_plan_missing_count": 0,
                        "identity_exact_match_count": 0,
                        "identity_passthrough_mismatch_count": 0,
                        "observed_identity_ids": set(),
                        "planned_identity_ids_for_trajectory_node_camera": set(),
                    },
                )
                rec["sample_count"] += 1
                rec["split_names"].add(str(sample.get("split") or "").strip() or "unknown")
                alignment = _as_dict(sample.get("plan_alignment"))
                capture_task_in_plan = alignment.get("capture_task_in_plan") is True
                if capture_task_in_plan:
                    rec["sample_with_capture_task_count"] += 1
                else:
                    rec["sample_without_capture_task_count"] += 1
                capture_task_id = str(alignment.get("capture_task_id") or "").strip()
                if capture_task_id:
                    rec["capture_task_ids"].add(capture_task_id)
                cam_id = str(sample.get("camera_id") or "").strip()
                tnc_key = (trajectory_id, node_id, cam_id)
                tnc_in_plan = tnc_key in planned_tnc_keys
                if tnc_in_plan:
                    rec["trajectory_node_camera_plan_match_count"] += 1
                else:
                    rec["trajectory_node_camera_plan_missing_count"] += 1
                if identity_id:
                    rec["observed_identity_ids"].add(identity_id)
                planned_ids = planned_identity_ids_by_tnc.get(tnc_key, set())
                rec["planned_identity_ids_for_trajectory_node_camera"].update(planned_ids)
                if tnc_in_plan and identity_id and identity_id in planned_ids:
                    rec["identity_exact_match_count"] += 1
                elif tnc_in_plan and identity_id and planned_ids and identity_id not in planned_ids:
                    rec["identity_passthrough_mismatch_count"] += 1

            per_scene_sum = 0
            capture_task_alignment_present_count = 0
            capture_task_exact_match_scene_count = 0
            trajectory_node_camera_plan_match_scene_count = 0
            identity_passthrough_mismatch_scene_count = 0
            capture_task_alignment_missing_scene_count = 0
            for idx, entry in enumerate(scene_entries, start=1):
                obj = _as_dict(entry)
                identity_id = str(obj.get("identity_id") or "unknown_identity").strip() or "unknown_identity"
                trajectory_id = str(obj.get("trajectory_id") or "unknown_trajectory").strip() or "unknown_trajectory"
                node_id = str(obj.get("node_id") or "unknown_node").strip() or "unknown_node"
                scene_id = str(obj.get("scene_id") or "").strip()
                scene_dir = str(obj.get("scene_dir") or obj.get("scene_root") or "").strip()
                key = (identity_id, trajectory_id, node_id, scene_id or scene_dir)
                sample_rec = _as_dict(sample_scene_aggregate.get(key))
                entry_sample_count = _to_int(obj.get("sample_count"))
                per_scene_sum += entry_sample_count
                if _to_int(obj.get("sample_count")) != _to_int(sample_rec.get("sample_count")):
                    _add_issue(
                        failures,
                        "scene_membership_manifest_scene_sample_count_mismatch",
                        "scene_membership_manifest scene sample_count must match sample rows.",
                        scene_entry_index=idx,
                        scene_id=scene_id,
                        manifest_value=obj.get("sample_count"),
                        computed_value=sample_rec.get("sample_count"),
                    )
                split_names = sorted({str(v).strip() for v in _as_list(obj.get("split_names")) if str(v).strip()})
                if not split_names:
                    split_one = str(obj.get("split") or "").strip()
                    split_names = [split_one] if split_one else []
                computed_split_names = sorted({str(v).strip() for v in sample_rec.get("split_names", set()) if str(v).strip()})
                if split_names != computed_split_names:
                    _add_issue(
                        failures,
                        "scene_membership_manifest_scene_split_names_mismatch",
                        "scene_membership_manifest per-scene split names must align with sample rows.",
                        scene_entry_index=idx,
                        scene_id=scene_id,
                        manifest_value=split_names,
                        computed_value=computed_split_names,
                    )
                capture_alignment = _as_dict(obj.get("capture_task_alignment"))
                if capture_alignment:
                    capture_task_alignment_present_count += 1
                    if _to_int(capture_alignment.get("sample_count")) != _to_int(obj.get("sample_count")):
                        _add_issue(
                            failures,
                            "scene_membership_manifest_capture_alignment_sample_count_mismatch",
                            "capture_task_alignment.sample_count must equal scene entry sample_count.",
                            scene_entry_index=idx,
                            scene_id=scene_id,
                            manifest_value=capture_alignment.get("sample_count"),
                            scene_sample_count=obj.get("sample_count"),
                        )
                    if _to_int(capture_alignment.get("sample_with_capture_task_count")) != _to_int(
                        sample_rec.get("sample_with_capture_task_count")
                    ):
                        _add_issue(
                            failures,
                            "scene_membership_manifest_capture_alignment_with_capture_task_count_mismatch",
                            "capture_task_alignment.sample_with_capture_task_count mismatch.",
                            scene_entry_index=idx,
                            scene_id=scene_id,
                            manifest_value=capture_alignment.get("sample_with_capture_task_count"),
                            computed_value=sample_rec.get("sample_with_capture_task_count"),
                        )
                    if _to_int(capture_alignment.get("sample_without_capture_task_count")) != _to_int(
                        sample_rec.get("sample_without_capture_task_count")
                    ):
                        _add_issue(
                            failures,
                            "scene_membership_manifest_capture_alignment_without_capture_task_count_mismatch",
                            "capture_task_alignment.sample_without_capture_task_count mismatch.",
                            scene_entry_index=idx,
                            scene_id=scene_id,
                            manifest_value=capture_alignment.get("sample_without_capture_task_count"),
                            computed_value=sample_rec.get("sample_without_capture_task_count"),
                        )
                    for bool_key, expected in (
                        ("non_promotion", True),
                        ("full_v1_live_dataset_ready", False),
                    ):
                        if bool_key in capture_alignment and capture_alignment.get(bool_key) is not expected:
                            _add_issue(
                                failures,
                                "scene_membership_manifest_capture_alignment_guard_flag_mismatch",
                                "capture_task_alignment guard boolean mismatch.",
                                scene_entry_index=idx,
                                scene_id=scene_id,
                                field=bool_key,
                                got=capture_alignment.get(bool_key),
                                expected=expected,
                            )
                    if _to_int(capture_alignment.get("identity_exact_match_count")) > 0:
                        capture_task_exact_match_scene_count += 1
                    if _to_int(capture_alignment.get("trajectory_node_camera_plan_match_count")) > 0:
                        trajectory_node_camera_plan_match_scene_count += 1
                    if _to_int(capture_alignment.get("identity_passthrough_mismatch_count")) > 0:
                        identity_passthrough_mismatch_scene_count += 1
                    scene_capture_alignment_observed_identity_ids.update(
                        {
                            str(v).strip()
                            for v in _as_list(capture_alignment.get("observed_identity_ids"))
                            if str(v).strip()
                        }
                    )
                    scene_capture_alignment_planned_identity_ids.update(
                        {
                            str(v).strip()
                            for v in _as_list(capture_alignment.get("planned_identity_ids_for_trajectory_node_camera"))
                            if str(v).strip()
                        }
                    )
                elif scene_membership_manifest_payload:
                    capture_task_alignment_missing_scene_count += 1
            if per_scene_sum != sample_count:
                _add_issue(
                    failures,
                    "scene_membership_manifest_scene_sample_sum_mismatch",
                    "Sum of scene_membership_manifest scene sample_count must equal sample_count.",
                    per_scene_sum=per_scene_sum,
                    sample_count=sample_count,
                )

            scene_obs_count = len(_as_list(manifest_payload.get("scene_observations")))
            if scene_obs_count > 0 and _to_int(scene_membership_manifest_payload.get("scene_count")) != scene_obs_count:
                _add_issue(
                    failures,
                    "scene_membership_manifest_scene_count_scene_observations_mismatch",
                    "scene_membership_manifest.scene_count must match dataset_manifest.scene_observations count when present.",
                    manifest_value=scene_membership_manifest_payload.get("scene_count"),
                    scene_observations_count=scene_obs_count,
                )
            if run_contract_payload:
                scene_root_count = len(_as_list(run_contract_payload.get("scene_roots")))
                if scene_root_count > 0 and _to_int(scene_membership_manifest_payload.get("scene_count")) != scene_root_count:
                    _add_issue(
                        failures,
                        "scene_membership_manifest_scene_count_run_contract_scene_roots_mismatch",
                        "scene_membership_manifest.scene_count must match run_contract.scene_roots count when present.",
                        manifest_value=scene_membership_manifest_payload.get("scene_count"),
                        run_contract_scene_root_count=scene_root_count,
                    )
            if capture_task_alignment_missing_scene_count > 0:
                _add_issue(
                    warnings,
                    "scene_membership_manifest_capture_alignment_missing_legacy_compatible",
                    "scene entry capture_task_alignment missing; treated as legacy-compatible.",
                    missing_scene_count=capture_task_alignment_missing_scene_count,
                    scene_entry_count=len(scene_entries),
                )
        else:
            pass

    capture_matrix_alignment_computed: dict[str, Any] = {}
    if should_validate_capture_matrix_alignment_manifest and capture_matrix_alignment_manifest_payload:
        if capture_matrix_alignment_manifest_payload.get("schema_version") != "carla_air_capture_matrix_alignment_manifest_v1":
            _add_issue(
                failures,
                "capture_matrix_alignment_manifest_schema_mismatch",
                "capture_matrix_alignment_manifest schema_version mismatch.",
                got=capture_matrix_alignment_manifest_payload.get("schema_version"),
                expected="carla_air_capture_matrix_alignment_manifest_v1",
            )
        expected_run_id = (
            str(run_contract_payload.get("run_id") or "").strip()
            or str(plan_payload.get("run_id") or "").strip()
            or str(manifest_payload.get("run_id") or "").strip()
        )
        if expected_run_id and str(capture_matrix_alignment_manifest_payload.get("run_id") or "").strip() != expected_run_id:
            _add_issue(
                failures,
                "capture_matrix_alignment_manifest_run_id_mismatch",
                "capture_matrix_alignment_manifest.run_id must align with run_contract/plan/manifest run_id.",
                manifest_value=capture_matrix_alignment_manifest_payload.get("run_id"),
                expected=expected_run_id,
            )
        if not str(capture_matrix_alignment_manifest_payload.get("generated_at_utc") or "").strip():
            _add_issue(
                failures,
                "capture_matrix_alignment_manifest_generated_at_utc_missing",
                "capture_matrix_alignment_manifest.generated_at_utc must be non-empty.",
            )
        for bool_key, expected in (
            ("starts_runtime", False),
            ("writes_scene_outputs", False),
            ("non_promotion", True),
            ("full_v1_live_dataset_ready", False),
        ):
            if capture_matrix_alignment_manifest_payload.get(bool_key) is not expected:
                _add_issue(
                    failures,
                    "capture_matrix_alignment_manifest_guard_flag_mismatch",
                    "capture_matrix_alignment_manifest guard boolean mismatch.",
                    field=bool_key,
                    got=capture_matrix_alignment_manifest_payload.get(bool_key),
                    expected=expected,
                )
        dependency_payloads_available = bool(capture_queue_rows) or sample_count == 0
        if dependency_payloads_available:
            capture_matrix_alignment_computed = _compute_capture_matrix_alignment(
                expected_run_id or str(capture_matrix_alignment_manifest_payload.get("run_id") or "").strip(),
                capture_queue_rows,
                samples,
            )
            manifest_summary = _as_dict(capture_matrix_alignment_manifest_payload.get("summary"))
            computed_summary = _as_dict(capture_matrix_alignment_computed.get("summary"))
            if manifest_summary != computed_summary:
                _add_issue(
                    failures,
                    "capture_matrix_alignment_manifest_summary_mismatch",
                    "capture_matrix_alignment_manifest.summary mismatch against recomputed capture_queue/dataset_samples join.",
                    manifest_value=manifest_summary,
                    computed_value=computed_summary,
                )
            manifest_rows = _as_list(capture_matrix_alignment_manifest_payload.get("rows"))
            computed_rows = _as_list(capture_matrix_alignment_computed.get("rows"))
            if len(manifest_rows) != len(computed_rows):
                _add_issue(
                    failures,
                    "capture_matrix_alignment_manifest_row_count_mismatch",
                    "capture_matrix_alignment_manifest rows count mismatch.",
                    manifest_row_count=len(manifest_rows),
                    computed_row_count=len(computed_rows),
                )
            if manifest_rows != computed_rows:
                _add_issue(
                    failures,
                    "capture_matrix_alignment_manifest_rows_mismatch",
                    "capture_matrix_alignment_manifest.rows mismatch against recomputed capture_queue/dataset_samples join.",
                    manifest_row_count=len(manifest_rows),
                    computed_row_count=len(computed_rows),
                    manifest_preview=manifest_rows[:3],
                    computed_preview=computed_rows[:3],
                )
            allowed_statuses = {
                "exact_match",
                "observed_scene_passthrough_identity_mismatch",
                "missing_observation",
            }
            for idx, row in enumerate(manifest_rows, start=1):
                row_obj = _as_dict(row)
                if row_obj.get("non_promotion") is not True:
                    _add_issue(
                        failures,
                        "capture_matrix_alignment_manifest_row_non_promotion_mismatch",
                        "capture_matrix_alignment_manifest.rows[].non_promotion must be true.",
                        row_index=idx,
                        got=row_obj.get("non_promotion"),
                    )
                if row_obj.get("full_v1_live_dataset_ready") is not False:
                    _add_issue(
                        failures,
                        "capture_matrix_alignment_manifest_row_live_ready_mismatch",
                        "capture_matrix_alignment_manifest.rows[].full_v1_live_dataset_ready must be false.",
                        row_index=idx,
                        got=row_obj.get("full_v1_live_dataset_ready"),
                    )
                match_status = str(row_obj.get("match_status") or "").strip()
                if match_status not in allowed_statuses:
                    _add_issue(
                        failures,
                        "capture_matrix_alignment_manifest_row_status_invalid",
                        "capture_matrix_alignment_manifest.rows[].match_status is not recognized.",
                        row_index=idx,
                        match_status=match_status,
                    )
        else:
            _add_issue(
                warnings,
                "capture_matrix_alignment_manifest_dependencies_unavailable",
                "capture_matrix_alignment_manifest present but capture_queue rows are unavailable; row recomputation skipped.",
                capture_queue_available=bool(capture_queue_rows),
            )

    scene_membership_alignment_computed: dict[str, Any] = {}
    if should_validate_scene_membership_alignment_manifest and scene_membership_alignment_manifest_payload:
        if scene_membership_alignment_manifest_payload.get("schema_version") != "carla_air_scene_membership_alignment_manifest_v1":
            _add_issue(
                failures,
                "scene_membership_alignment_manifest_schema_mismatch",
                "scene_membership_alignment_manifest schema_version mismatch.",
                got=scene_membership_alignment_manifest_payload.get("schema_version"),
                expected="carla_air_scene_membership_alignment_manifest_v1",
            )
        expected_run_id = (
            str(run_contract_payload.get("run_id") or "").strip()
            or str(plan_payload.get("run_id") or "").strip()
            or str(manifest_payload.get("run_id") or "").strip()
        )
        if expected_run_id and str(scene_membership_alignment_manifest_payload.get("run_id") or "").strip() != expected_run_id:
            _add_issue(
                failures,
                "scene_membership_alignment_manifest_run_id_mismatch",
                "scene_membership_alignment_manifest.run_id must align with run_contract/plan/manifest run_id.",
                manifest_value=scene_membership_alignment_manifest_payload.get("run_id"),
                expected=expected_run_id,
            )
        if not str(scene_membership_alignment_manifest_payload.get("generated_at_utc") or "").strip():
            _add_issue(
                failures,
                "scene_membership_alignment_manifest_generated_at_utc_missing",
                "scene_membership_alignment_manifest.generated_at_utc must be non-empty.",
            )
        for bool_key, expected in (
            ("starts_runtime", False),
            ("writes_scene_outputs", False),
            ("non_promotion", True),
            ("full_v1_live_dataset_ready", False),
        ):
            if scene_membership_alignment_manifest_payload.get(bool_key) is not expected:
                _add_issue(
                    failures,
                    "scene_membership_alignment_manifest_guard_flag_mismatch",
                    "scene_membership_alignment_manifest guard boolean mismatch.",
                    field=bool_key,
                    got=scene_membership_alignment_manifest_payload.get(bool_key),
                    expected=expected,
                )
        dependency_payloads_available = bool(scene_output_manifest_payload) and bool(scene_membership_manifest_payload)
        if dependency_payloads_available:
            scene_membership_alignment_computed = _compute_scene_membership_alignment(
                expected_run_id or str(scene_membership_alignment_manifest_payload.get("run_id") or "").strip(),
                scene_output_manifest_payload,
                scene_membership_manifest_payload,
            )
            manifest_summary = _as_dict(scene_membership_alignment_manifest_payload.get("summary"))
            computed_summary = _as_dict(scene_membership_alignment_computed.get("summary"))
            if manifest_summary != computed_summary:
                _add_issue(
                    failures,
                    "scene_membership_alignment_manifest_summary_mismatch",
                    "scene_membership_alignment_manifest.summary mismatch against recomputed scene_output/scene_membership join.",
                    manifest_value=manifest_summary,
                    computed_value=computed_summary,
                )
            manifest_rows = _as_list(scene_membership_alignment_manifest_payload.get("rows"))
            computed_rows = _as_list(scene_membership_alignment_computed.get("rows"))
            if len(manifest_rows) != len(computed_rows):
                _add_issue(
                    failures,
                    "scene_membership_alignment_manifest_row_count_mismatch",
                    "scene_membership_alignment_manifest rows count mismatch.",
                    manifest_row_count=len(manifest_rows),
                    computed_row_count=len(computed_rows),
                )
            if manifest_rows != computed_rows:
                _add_issue(
                    failures,
                    "scene_membership_alignment_manifest_rows_mismatch",
                    "scene_membership_alignment_manifest.rows mismatch against recomputed scene_output/scene_membership join.",
                    manifest_row_count=len(manifest_rows),
                    computed_row_count=len(computed_rows),
                    manifest_preview=manifest_rows[:3],
                    computed_preview=computed_rows[:3],
                )
            for idx, row in enumerate(manifest_rows, start=1):
                row_obj = _as_dict(row)
                if row_obj.get("non_promotion") is not True:
                    _add_issue(
                        failures,
                        "scene_membership_alignment_manifest_row_non_promotion_mismatch",
                        "scene_membership_alignment_manifest.rows[].non_promotion must be true.",
                        row_index=idx,
                        got=row_obj.get("non_promotion"),
                    )
                if row_obj.get("full_v1_live_dataset_ready") is not False:
                    _add_issue(
                        failures,
                        "scene_membership_alignment_manifest_row_live_ready_mismatch",
                        "scene_membership_alignment_manifest.rows[].full_v1_live_dataset_ready must be false.",
                        row_index=idx,
                        got=row_obj.get("full_v1_live_dataset_ready"),
                    )
                match_status = str(row_obj.get("membership_match_status") or "").strip()
                allowed_statuses = {
                    "exact_identity_match",
                    "observed_default_airsim_drone_non_promotion_passthrough",
                    "identity_mismatch_non_promotion_passthrough",
                    "missing_observation",
                }
                if match_status not in allowed_statuses:
                    _add_issue(
                        failures,
                        "scene_membership_alignment_manifest_row_status_invalid",
                        "scene_membership_alignment_manifest.rows[].membership_match_status is not recognized.",
                        row_index=idx,
                        membership_match_status=match_status,
                    )
        else:
            _add_issue(
                warnings,
                "scene_membership_alignment_manifest_dependencies_unavailable",
                "scene_membership_alignment_manifest present but scene_output_manifest or scene_membership_manifest is unavailable; row recomputation skipped.",
                scene_output_manifest_available=bool(scene_output_manifest_payload),
                scene_membership_manifest_available=bool(scene_membership_manifest_payload),
            )

    scene_membership_manifest_summary = {
        "present": artifact_presence["scene_membership_manifest"],
        "contract_declared": scene_membership_manifest_contract_declared,
        "contract_required": scene_membership_manifest_contract_required,
        "schema_version": scene_membership_manifest_payload.get("schema_version") if scene_membership_manifest_payload else None,
        "scene_count": _to_int(scene_membership_manifest_payload.get("scene_count")) if scene_membership_manifest_payload else None,
        "sample_count": _to_int(scene_membership_manifest_payload.get("sample_count")) if scene_membership_manifest_payload else None,
        "cross_split_scene_key_count": _to_int(scene_membership_manifest_payload.get("cross_split_scene_key_count"))
        if scene_membership_manifest_payload
        else None,
        "scene_key_split_collision_count": len(_as_list(scene_membership_manifest_payload.get("scene_key_split_collisions")))
        if scene_membership_manifest_payload
        else None,
        "split_distribution": _as_dict(scene_membership_manifest_payload.get("split_distribution"))
        if scene_membership_manifest_payload
        else {},
        "not_random_frame_split": scene_membership_manifest_payload.get("not_random_frame_split")
        if scene_membership_manifest_payload
        else None,
        "non_promotion": scene_membership_manifest_payload.get("non_promotion") if scene_membership_manifest_payload else None,
        "mask_gt_available": scene_membership_manifest_payload.get("mask_gt_available")
        if scene_membership_manifest_payload
        else None,
        "full_v1_live_dataset_ready": scene_membership_manifest_payload.get("full_v1_live_dataset_ready")
        if scene_membership_manifest_payload
        else None,
        "capture_task_alignment_present_count": capture_task_alignment_present_count
        if scene_membership_manifest_payload
        else 0,
        "capture_task_exact_match_scene_count": capture_task_exact_match_scene_count
        if scene_membership_manifest_payload
        else 0,
        "trajectory_node_camera_plan_match_scene_count": trajectory_node_camera_plan_match_scene_count
        if scene_membership_manifest_payload
        else 0,
        "identity_passthrough_mismatch_scene_count": identity_passthrough_mismatch_scene_count
        if scene_membership_manifest_payload
        else 0,
    }
    scene_membership_alignment_manifest_summary = {
        "present": artifact_presence["scene_membership_alignment_manifest"],
        "contract_declared": scene_membership_alignment_manifest_contract_declared,
        "contract_required": scene_membership_alignment_manifest_contract_required,
        "schema_version": scene_membership_alignment_manifest_payload.get("schema_version")
        if scene_membership_alignment_manifest_payload
        else None,
        "row_count": len(_as_list(scene_membership_alignment_manifest_payload.get("rows")))
        if scene_membership_alignment_manifest_payload
        else 0,
        "summary": _as_dict(scene_membership_alignment_manifest_payload.get("summary"))
        if scene_membership_alignment_manifest_payload
        else {},
        "computed_summary": _as_dict(scene_membership_alignment_computed.get("summary"))
        if scene_membership_alignment_computed
        else {},
        "non_promotion": scene_membership_alignment_manifest_payload.get("non_promotion")
        if scene_membership_alignment_manifest_payload
        else None,
        "full_v1_live_dataset_ready": scene_membership_alignment_manifest_payload.get("full_v1_live_dataset_ready")
        if scene_membership_alignment_manifest_payload
        else None,
    }
    capture_matrix_alignment_manifest_summary = {
        "present": artifact_presence["capture_matrix_alignment_manifest"],
        "contract_declared": capture_matrix_alignment_manifest_contract_declared,
        "contract_required": capture_matrix_alignment_manifest_contract_required,
        "schema_version": capture_matrix_alignment_manifest_payload.get("schema_version")
        if capture_matrix_alignment_manifest_payload
        else None,
        "row_count": len(_as_list(capture_matrix_alignment_manifest_payload.get("rows")))
        if capture_matrix_alignment_manifest_payload
        else 0,
        "summary": _as_dict(capture_matrix_alignment_manifest_payload.get("summary"))
        if capture_matrix_alignment_manifest_payload
        else {},
        "computed_summary": _as_dict(capture_matrix_alignment_computed.get("summary"))
        if capture_matrix_alignment_computed
        else {},
        "non_promotion": capture_matrix_alignment_manifest_payload.get("non_promotion")
        if capture_matrix_alignment_manifest_payload
        else None,
        "full_v1_live_dataset_ready": capture_matrix_alignment_manifest_payload.get("full_v1_live_dataset_ready")
        if capture_matrix_alignment_manifest_payload
        else None,
    }
    manifest_capture_alignment_summary = _as_dict(manifest_payload.get("capture_task_alignment_summary")) if manifest_payload else {}
    manifest_observed_identity_ids = sorted(
        {
            str(x).strip()
            for x in _as_list(manifest_capture_alignment_summary.get("observed_identity_ids"))
            if str(x).strip()
        }
    )
    manifest_planned_identity_ids = sorted(
        {
            str(x).strip()
            for x in _as_list(manifest_capture_alignment_summary.get("planned_identity_ids"))
            if str(x).strip()
        }
    )
    if "scene_capture_alignment_observed_identity_ids" not in locals():
        scene_capture_alignment_observed_identity_ids = set()
    if "scene_capture_alignment_planned_identity_ids" not in locals():
        scene_capture_alignment_planned_identity_ids = set()
    observed_identity_ids = manifest_observed_identity_ids or sorted(scene_capture_alignment_observed_identity_ids) or sorted(sample_identity_ids)
    planned_identity_ids = manifest_planned_identity_ids or sorted(scene_capture_alignment_planned_identity_ids) or sorted(plan_identities)
    scene_capture_alignment_summary = {
        "total_scene_count": _to_int(scene_membership_manifest_summary.get("scene_count"))
        if scene_membership_manifest_payload
        else 0,
        "alignment_present_scene_count": capture_task_alignment_present_count
        if scene_membership_manifest_payload
        else 0,
        "exact_match_scene_count": capture_task_exact_match_scene_count
        if scene_membership_manifest_payload
        else 0,
        "trajectory_node_camera_plan_match_scene_count": trajectory_node_camera_plan_match_scene_count
        if scene_membership_manifest_payload
        else 0,
        "identity_passthrough_mismatch_scene_count": identity_passthrough_mismatch_scene_count
        if scene_membership_manifest_payload
        else 0,
        "observed_identity_ids": observed_identity_ids,
        "planned_identity_ids": planned_identity_ids,
        "sample_with_capture_task_count": _to_int(manifest_capture_alignment_summary.get("sample_with_capture_task_count"))
        if manifest_capture_alignment_summary
        else sample_with_capture_task_in_plan_count,
        "sample_without_capture_task_count": _to_int(manifest_capture_alignment_summary.get("sample_without_capture_task_count"))
        if manifest_capture_alignment_summary
        else sample_without_capture_task_in_plan_count,
        "capture_task_count": _to_int(manifest_capture_alignment_summary.get("capture_task_count"))
        if manifest_capture_alignment_summary
        else plan_capture_task_count,
    }
    split_policy_summary_report = split_policy_summary_computed if split_policy_summary_computed else None
    split_policy_digest_report = split_policy_digest_computed
    split_policy_digest_sources: dict[str, str] = {}
    split_policy_artifact_count_with_digest = 0
    split_policy_artifact_count_with_any = 0
    if splits_payload:
        splits_manifest = _as_dict(splits_payload.get("manifest"))
        split_policy_checks = [
            ("dataset_splits", splits_payload),
            ("dataset_splits.manifest", splits_manifest),
            ("scene_membership_manifest", scene_membership_manifest_payload),
            ("deployment_episodes.summary", _as_dict(episodes_payload.get("summary")) if episodes_payload else {}),
            ("deployment_episode_visibility_manifest", deployment_episode_visibility_manifest_payload),
        ]
        for source_name, source_obj in split_policy_checks:
            has_summary = "split_policy_summary" in source_obj
            has_digest = "split_policy_digest" in source_obj
            if has_summary or has_digest:
                split_policy_artifact_count_with_any += 1
            if not has_summary and not has_digest:
                continue
            if has_summary:
                observed_summary = _normalize_split_policy_summary(source_obj.get("split_policy_summary"))
                if observed_summary != split_policy_summary_computed:
                    _add_issue(
                        failures,
                        "split_policy_summary_mismatch",
                        "split policy summary mismatch against recomputed dataset_splits policy.",
                        source=source_name,
                        observed_summary=observed_summary,
                        computed_summary=split_policy_summary_computed,
                    )
                if not split_policy_summary_report:
                    split_policy_summary_report = observed_summary
            else:
                _add_issue(
                    warnings,
                    "split_policy_summary_missing_legacy_compatible",
                    "split policy summary field missing; treated as legacy-compatible.",
                    source=source_name,
                )
            if has_digest:
                observed_digest = str(source_obj.get("split_policy_digest") or "").strip()
                split_policy_artifact_count_with_digest += 1
                if observed_digest:
                    split_policy_digest_sources[source_name] = observed_digest
                    if split_policy_digest_computed and observed_digest != split_policy_digest_computed:
                        _add_issue(
                            failures,
                            "split_policy_digest_mismatch",
                            "split policy digest mismatch against recomputed canonical digest.",
                            source=source_name,
                            observed_digest=observed_digest,
                            computed_digest=split_policy_digest_computed,
                        )
                else:
                    _add_issue(
                        warnings,
                        "split_policy_digest_empty_legacy_compatible",
                        "split policy digest is empty; treated as legacy-compatible.",
                        source=source_name,
                    )
            else:
                _add_issue(
                    warnings,
                    "split_policy_digest_missing_legacy_compatible",
                    "split policy digest field missing; treated as legacy-compatible.",
                    source=source_name,
                )
        if split_policy_artifact_count_with_digest > 0 and split_policy_artifact_count_with_digest < len(split_policy_checks):
            _add_issue(
                failures,
                "split_policy_digest_partial_update_detected",
                "split policy digest exists in only a subset of related artifacts; require all-or-none to avoid partial updates.",
                digest_source_count=split_policy_artifact_count_with_digest,
                expected_source_count=len(split_policy_checks),
            )
        if split_policy_artifact_count_with_any == 0:
            _add_issue(
                warnings,
                "split_policy_fields_missing_legacy_compatible",
                "split policy summary/digest fields are missing across related artifacts; treated as legacy-compatible run.",
            )
        if len(set(split_policy_digest_sources.values())) > 1:
            _add_issue(
                failures,
                "split_policy_digest_cross_artifact_mismatch",
                "split policy digest must be identical across related artifacts.",
                digests_by_source=split_policy_digest_sources,
            )

    dataset_run_contract_sources = [
        ("dataset_manifest", _as_dict(manifest_payload.get("dataset_run_contract_summary"))),
        ("dataset_index_manifest", _as_dict(dataset_index_manifest_payload.get("dataset_run_contract_summary"))),
        ("run_contract", _as_dict(run_contract_payload.get("dataset_run_contract_summary"))),
        ("batch_run_manifest", _as_dict(batch_run_manifest_payload.get("dataset_run_contract_summary"))),
    ]
    dataset_run_contract_present_sources = [(name, payload) for name, payload in dataset_run_contract_sources if payload]
    dataset_run_contract_source_digests: dict[str, str] = {}
    dataset_run_contract_summary_computed = {
        "schema_version": "carla_air_dataset_run_contract_summary_v1",
        "sample_count": sample_count,
        "scene_count": _to_int(scene_membership_manifest_summary.get("scene_count"))
        if scene_membership_manifest_payload
        else len(_as_list(manifest_payload.get("scene_observations"))),
        "split_distribution": {str(k): _to_int(v) for k, v in split_distribution.items()},
        "capture_task_count": plan_capture_task_count,
        "sample_with_capture_task_count": sample_with_capture_task_in_plan_count,
        "sample_without_capture_task_count": sample_without_capture_task_in_plan_count,
        "strict_matrix_entry_sample_count": strict_matrix_entry_sample_count,
        "legacy_or_observed_scene_passthrough_count": legacy_or_observed_scene_passthrough_count,
        "mask_gt_available_count": _to_int(sidecar_quality_summary.get("mask_gt_available_count")),
        "no_mask_sample_count": _to_int(sidecar_quality_summary.get("no_mask_sample_count")),
        "sidecar_complete_count": _to_int(sidecar_quality_summary.get("complete_rgb_depth_semantic_instance_pose_calib_count")),
        "sidecar_complete_fraction": float(sidecar_quality_summary.get("complete_fraction") or 0.0),
        "sidecar_missing_count_by_modality": {
            key: _to_int(_as_dict(sidecar_quality_summary.get("missing_count_by_modality")).get(key))
            for key in ("rgb", "depth", "semantic", "instance", "pose", "calib")
        },
        "planned_identity_ids": sorted(plan_identities),
        "observed_identity_ids": sorted(sample_identity_ids),
        "identity_mismatch_count": identity_mismatch_count,
        "strict_planned_identity_sample_count": strict_planned_identity_sample_count,
        "observed_passthrough_identity_sample_count": observed_passthrough_identity_sample_count,
        "starts_runtime": False,
        "writes_scene_outputs": False,
        "non_promotion": True,
        "full_v1_live_dataset_ready": False,
    }
    dataset_run_contract_source_payloads = [
        payload for _, payload in dataset_run_contract_sources if isinstance(payload, dict) and payload
    ]
    dataset_run_contract_bridge_fields_present = any(
        any(
            bridge_key in payload
            for bridge_key in (
                "sample_with_trajectory_node_camera_bridge_count",
                "sample_without_trajectory_node_camera_bridge_count",
                "planned_capture_task_candidate_reference_count",
            )
        )
        for payload in dataset_run_contract_source_payloads
    )
    if sample_capture_matrix_bridge_present_count > 0 or dataset_run_contract_bridge_fields_present:
        dataset_run_contract_summary_computed.update(
            {
                "sample_with_trajectory_node_camera_bridge_count": sample_with_trajectory_node_camera_bridge_count,
                "sample_without_trajectory_node_camera_bridge_count": (
                    sample_count - sample_with_trajectory_node_camera_bridge_count
                ),
                "planned_capture_task_candidate_reference_count": planned_capture_task_candidate_reference_count,
            }
        )
    dataset_run_contract_summary_report = dataset_run_contract_summary_computed
    dataset_run_contract_summary_digest = _canonical_json_sha256(dataset_run_contract_summary_computed)
    if not dataset_run_contract_present_sources:
        _add_issue(
            warnings,
            "dataset_run_contract_summary_missing_legacy_compatible",
            "dataset_run_contract_summary is absent across dataset_manifest/dataset_index_manifest/run_contract/batch_run_manifest; treated as legacy-compatible run.",
        )
    else:
        if len(dataset_run_contract_present_sources) != len(dataset_run_contract_sources):
            _add_issue(
                failures,
                "dataset_run_contract_summary_partial_update_detected",
                "dataset_run_contract_summary exists in only a subset of required artifacts; require all-or-none for non-legacy runs.",
                observed_source_count=len(dataset_run_contract_present_sources),
                expected_source_count=len(dataset_run_contract_sources),
                present_sources=[name for name, _ in dataset_run_contract_present_sources],
            )
        normalized_by_source: dict[str, dict[str, Any]] = {}
        for source_name, source_payload in dataset_run_contract_present_sources:
            normalized = _normalize_dataset_run_contract_summary(source_payload)
            normalized_by_source[source_name] = normalized
            dataset_run_contract_source_digests[source_name] = _canonical_json_sha256(normalized)
        if len(set(dataset_run_contract_source_digests.values())) > 1:
            _add_issue(
                failures,
                "dataset_run_contract_summary_cross_artifact_mismatch",
                "dataset_run_contract_summary must be identical across all present artifacts.",
                digests_by_source=dataset_run_contract_source_digests,
            )
        for source_name, observed_summary in normalized_by_source.items():
            for bool_key, expected in (
                ("starts_runtime", False),
                ("writes_scene_outputs", False),
                ("non_promotion", True),
                ("full_v1_live_dataset_ready", False),
            ):
                if observed_summary.get(bool_key) is not expected:
                    _add_issue(
                        failures,
                        "dataset_run_contract_summary_offline_boundary_mismatch",
                        "dataset_run_contract_summary violates offline non-promotion boundary.",
                        source=source_name,
                        field=bool_key,
                        observed_value=observed_summary.get(bool_key),
                        expected_value=expected,
                    )
            if observed_summary != dataset_run_contract_summary_computed:
                _add_issue(
                    failures,
                    "dataset_run_contract_summary_computed_mismatch",
                    "dataset_run_contract_summary mismatch against verifier recomputed facts.",
                    source=source_name,
                    observed_summary=observed_summary,
                    computed_summary=dataset_run_contract_summary_computed,
                )
        if normalized_by_source:
            dataset_run_contract_summary_report = next(iter(normalized_by_source.values()))

    should_validate_dataset_gap_manifest = (
        artifact_presence["dataset_gap_manifest"] or dataset_gap_manifest_contract_required
    )
    if should_validate_dataset_gap_manifest:
        dataset_gap_manifest_payload, dataset_gap_manifest_err = _load_json(dataset_gap_manifest_path)
        if dataset_gap_manifest_err:
            _add_issue(
                failures,
                "dataset_gap_manifest_invalid",
                "dataset_gap_manifest.json is missing or invalid while present/contract-required.",
                error=dataset_gap_manifest_err,
            )
        else:
            if dataset_gap_manifest_payload.get("schema_version") != "carla_air_dataset_gap_manifest_v1":
                _add_issue(
                    failures,
                    "dataset_gap_manifest_schema_mismatch",
                    "dataset_gap_manifest schema_version mismatch.",
                    got=dataset_gap_manifest_payload.get("schema_version"),
                    expected="carla_air_dataset_gap_manifest_v1",
                )
            expected_run_id = (
                str(run_contract_payload.get("run_id") or "").strip()
                or str(plan_payload.get("run_id") or "").strip()
                or str(manifest_payload.get("run_id") or "").strip()
            )
            observed_run_id = str(dataset_gap_manifest_payload.get("run_id") or "").strip()
            if expected_run_id and observed_run_id != expected_run_id:
                _add_issue(
                    failures,
                    "dataset_gap_manifest_run_id_mismatch",
                    "dataset_gap_manifest.run_id must align with run_contract/plan/manifest run_id.",
                    manifest_value=observed_run_id,
                    expected=expected_run_id,
                )
            for flag_name, expected_value in (
                ("starts_runtime", False),
                ("writes_scene_outputs", False),
                ("non_promotion", True),
                ("full_v1_live_dataset_ready", False),
            ):
                if dataset_gap_manifest_payload.get(flag_name) is not expected_value:
                    _add_issue(
                        failures,
                        "dataset_gap_manifest_guard_flag_mismatch",
                        "dataset_gap_manifest guard flag mismatch.",
                        field=flag_name,
                        got=dataset_gap_manifest_payload.get(flag_name),
                        expected=expected_value,
                    )
            gap_formal_ready_sample_count = 0
            for sample in samples:
                sample_obj = _as_dict(sample)
                mask_gt_obj = _as_dict(sample_obj.get("mask_gt"))
                alignment = _as_dict(sample_obj.get("plan_alignment"))
                refs = _as_dict(sample_obj.get("refs"))
                has_all_sidecars = all(bool(refs.get(key)) for key in ("rgb", "depth", "semantic", "instance", "pose", "calib"))
                identity_matches = alignment.get("observed_identity_matches_planned")
                if identity_matches is None:
                    identity_matches = alignment.get("planned_identity_match")
                if identity_matches is None:
                    identity_matches = alignment.get("identity_in_plan")
                if (
                    str(mask_gt_obj.get("availability") or "").strip().lower() == "available"
                    and has_all_sidecars
                    and identity_matches is True
                ):
                    gap_formal_ready_sample_count += 1
            expected_gap_manifest_core = {
                "run_id": expected_run_id or observed_run_id,
                "starts_runtime": False,
                "writes_scene_outputs": False,
                "non_promotion": True,
                "full_v1_live_dataset_ready": False,
                "sample_count": _to_int(dataset_run_contract_summary_computed.get("sample_count")),
                "scene_count": _to_int(dataset_run_contract_summary_computed.get("scene_count")),
                "split_distribution": _as_dict(dataset_run_contract_summary_computed.get("split_distribution")),
                "planned_identity_ids": sorted(
                    {str(v).strip() for v in _as_list(dataset_run_contract_summary_computed.get("planned_identity_ids")) if str(v).strip()}
                ),
                "observed_identity_ids": sorted(
                    {str(v).strip() for v in _as_list(dataset_run_contract_summary_computed.get("observed_identity_ids")) if str(v).strip()}
                ),
                "identity_mismatch_count": _to_int(dataset_run_contract_summary_computed.get("identity_mismatch_count")),
                "strict_planned_identity_sample_count": _to_int(
                    dataset_run_contract_summary_computed.get("strict_planned_identity_sample_count")
                ),
                "observed_passthrough_identity_sample_count": _to_int(
                    dataset_run_contract_summary_computed.get("observed_passthrough_identity_sample_count")
                ),
                "mask_gt_available_count": _to_int(dataset_run_contract_summary_computed.get("mask_gt_available_count")),
                "no_mask_sample_count": _to_int(dataset_run_contract_summary_computed.get("no_mask_sample_count")),
                "sidecar_complete_count": _to_int(dataset_run_contract_summary_computed.get("sidecar_complete_count")),
                "sidecar_incomplete_sample_count": (
                    _to_int(dataset_run_contract_summary_computed.get("sample_count"))
                    - _to_int(dataset_run_contract_summary_computed.get("sidecar_complete_count"))
                ),
                "sidecar_missing_count_by_modality": {
                    key: _to_int(_as_dict(dataset_run_contract_summary_computed.get("sidecar_missing_count_by_modality")).get(key))
                    for key in ("rgb", "depth", "semantic", "instance", "pose", "calib")
                },
                "gap_counts": {
                    "no_mask_non_promotion_sample_count": _to_int(
                        dataset_run_contract_summary_computed.get("no_mask_sample_count")
                    ),
                    "sidecar_incomplete_sample_count": (
                        _to_int(dataset_run_contract_summary_computed.get("sample_count"))
                        - _to_int(dataset_run_contract_summary_computed.get("sidecar_complete_count"))
                    ),
                    "identity_passthrough_mismatch_sample_count": _to_int(
                        dataset_run_contract_summary_computed.get("observed_passthrough_identity_sample_count")
                    ),
                    "formal_ready_sample_count": gap_formal_ready_sample_count,
                },
                "gap_policy": {
                    "sidecar_missing_is_not_mask_gt_failure": True,
                    "no_mask_allowed_in_index": True,
                    "identity_passthrough_not_live_identity_evidence": True,
                    "proxy_candidate_pseudo_legacy_not_promoted_to_mask_gt": True,
                },
            }
            expected_gap_manifest = {
                "schema_version": "carla_air_dataset_gap_manifest_v1",
                **expected_gap_manifest_core,
                "stable_hashes": {
                    "core_payload_sha256": _canonical_json_sha256(expected_gap_manifest_core),
                },
            }
            for key, expected_value in expected_gap_manifest.items():
                observed_value = dataset_gap_manifest_payload.get(key)
                if observed_value != expected_value:
                    _add_issue(
                        failures,
                        "dataset_gap_manifest_computed_mismatch",
                        "dataset_gap_manifest field mismatch against verifier recomputation.",
                        field=key,
                        manifest_value=observed_value,
                        computed_value=expected_value,
                    )
            if run_contract_payload:
                contract_artifacts = _as_dict(run_contract_payload.get("artifacts"))
                contract_gap_path = str(contract_artifacts.get("dataset_gap_manifest_json") or "").strip()
                if contract_gap_path and str(dataset_gap_manifest_path.resolve()) != str(_repo_or_abs(contract_gap_path).resolve()):
                    _add_issue(
                        failures,
                        "dataset_gap_manifest_run_contract_artifact_path_mismatch",
                        "run_contract.artifacts.dataset_gap_manifest_json must align with verifier run_dir manifest path.",
                        manifest_path=str(dataset_gap_manifest_path.resolve()),
                        run_contract_path=contract_gap_path,
                    )
                if batch_run_manifest_payload:
                    batch_paths = _as_dict(batch_run_manifest_payload.get("artifact_paths"))
                    batch_gap_path = str(batch_paths.get("dataset_gap_manifest_json") or "").strip()
                    if contract_gap_path:
                        if not batch_gap_path:
                            _add_issue(
                                failures,
                                "batch_run_manifest_dataset_gap_manifest_path_missing",
                                "batch_run_manifest.artifact_paths.dataset_gap_manifest_json must exist when contract declares it.",
                            )
                        elif batch_gap_path != contract_gap_path:
                            _add_issue(
                                failures,
                                "batch_run_manifest_dataset_gap_manifest_path_mismatch",
                                "batch_run_manifest.artifact_paths.dataset_gap_manifest_json must align with run_contract.artifacts.",
                                expected=contract_gap_path,
                                got=batch_gap_path,
                            )
                if artifact_manifest_payload:
                    artifact_map = _as_dict(artifact_manifest_payload.get("artifacts"))
                    if contract_gap_path and ("dataset_gap_manifest_json" not in artifact_map):
                        _add_issue(
                            failures,
                            "artifact_manifest_missing_dataset_gap_manifest_key",
                            "artifact_manifest.artifacts missing dataset_gap_manifest_json declared by run_contract.",
                            artifact_key="dataset_gap_manifest_json",
                        )
    elif not dataset_gap_manifest_contract_declared:
        _add_issue(
            warnings,
            "dataset_gap_manifest_missing_legacy_compatible",
            "dataset_gap_manifest.json is absent and run_contract does not declare dataset_gap_manifest_json; treated as legacy-compatible run.",
        )

    episodes_top_summary = _as_dict(episodes_payload.get("summary")) if episodes_payload else {}
    deployment_episodes_summary = {
        "present": artifact_presence["deployment_episodes"],
        "schema_version": episodes_payload.get("schema_version") if episodes_payload else None,
        "episode_count": len(_as_list(episodes_payload.get("episodes"))) if episodes_payload else 0,
        "sample_count_total_from_episodes": _to_int(episodes_top_summary.get("sample_count_total_from_episodes"))
        if episodes_top_summary
        else deployment_sample_count_total_from_episodes,
        "scene_count_total_from_episodes": _to_int(episodes_top_summary.get("scene_count_total_from_episodes"))
        if episodes_top_summary
        else len(deployment_scene_ids_union),
        "visibility_present_count": deployment_visibility_present_count,
        "episode_with_visibility_gap_count": _to_int(episodes_top_summary.get("episode_with_visibility_gap_count"))
        if episodes_top_summary
        else deployment_episode_with_visibility_gap_count,
        "episode_without_samples_count": _to_int(episodes_top_summary.get("episode_without_samples_count"))
        if episodes_top_summary
        else deployment_episode_without_samples_count,
        "episode_scene_visibility_hash": episodes_top_summary.get("episode_scene_visibility_hash")
        if episodes_top_summary
        else _hash_text_parts(deployment_episode_scene_rows),
        "episode_sample_order_hash": episodes_top_summary.get("episode_sample_order_hash")
        if episodes_top_summary
        else _hash_text_parts(deployment_episode_sample_order_rows),
        "episode_sample_sorted_hash": episodes_top_summary.get("episode_sample_sorted_hash")
        if episodes_top_summary
        else _hash_text_parts(deployment_episode_sample_sorted_rows),
    }
    if episodes_top_summary:
        if any(
            key in episodes_top_summary
            for key in (
                "episode_scene_visibility_hash",
                "episode_sample_order_hash",
                "episode_sample_sorted_hash",
            )
        ):
            deployment_episode_hash_strict_required = True
        if episodes_top_summary.get("non_promotion") is not True:
            _add_issue(
                failures,
                "deployment_episodes_summary_non_promotion_mismatch",
                "deployment_episodes.summary.non_promotion must be true.",
                got=episodes_top_summary.get("non_promotion"),
            )
        if episodes_top_summary.get("full_v1_live_dataset_ready") is not False:
            _add_issue(
                failures,
                "deployment_episodes_summary_full_v1_live_dataset_ready_mismatch",
                "deployment_episodes.summary.full_v1_live_dataset_ready must be false.",
                got=episodes_top_summary.get("full_v1_live_dataset_ready"),
            )
        if (
            deployment_visibility_present_count > 0
            and _to_int(episodes_top_summary.get("sample_count_total_from_episodes"))
            != deployment_sample_count_total_from_episodes
        ):
            _add_issue(
                failures,
                "deployment_episodes_summary_sample_count_total_mismatch",
                "deployment_episodes.summary.sample_count_total_from_episodes must match recomputed episode-summed sample count.",
                manifest_value=episodes_top_summary.get("sample_count_total_from_episodes"),
                computed_value=deployment_sample_count_total_from_episodes,
            )
        if (
            deployment_visibility_present_count > 0
            and _to_int(episodes_top_summary.get("scene_count_total_from_episodes")) != len(deployment_scene_ids_union)
        ):
            _add_issue(
                failures,
                "deployment_episodes_summary_scene_count_total_mismatch",
                "deployment_episodes.summary.scene_count_total_from_episodes must match recomputed unique scene count across episodes.",
                manifest_value=episodes_top_summary.get("scene_count_total_from_episodes"),
                computed_value=len(deployment_scene_ids_union),
            )
        if (
            deployment_visibility_present_count > 0
            and _to_int(episodes_top_summary.get("episode_with_visibility_gap_count"))
            != deployment_episode_with_visibility_gap_count
        ):
            _add_issue(
                failures,
                "deployment_episodes_summary_visibility_gap_count_mismatch",
                "deployment_episodes.summary.episode_with_visibility_gap_count must match recomputed visibility gap count.",
                manifest_value=episodes_top_summary.get("episode_with_visibility_gap_count"),
                computed_value=deployment_episode_with_visibility_gap_count,
            )
        if (
            deployment_visibility_present_count > 0
            and _to_int(episodes_top_summary.get("episode_without_samples_count")) != deployment_episode_without_samples_count
        ):
            _add_issue(
                failures,
                "deployment_episodes_summary_episode_without_samples_count_mismatch",
                "deployment_episodes.summary.episode_without_samples_count must match recomputed zero-sample episode count.",
                manifest_value=episodes_top_summary.get("episode_without_samples_count"),
                computed_value=deployment_episode_without_samples_count,
            )
        episode_hash_summary_fields = {
            "episode_scene_visibility_hash": _hash_text_parts(deployment_episode_scene_rows),
            "episode_sample_order_hash": _hash_text_parts(deployment_episode_sample_order_rows),
            "episode_sample_sorted_hash": _hash_text_parts(deployment_episode_sample_sorted_rows),
        }
        missing_episode_hash_fields = [key for key in episode_hash_summary_fields if key not in episodes_top_summary]
        if missing_episode_hash_fields:
            issue_list = failures if deployment_episode_hash_strict_required else warnings
            issue_code = (
                "deployment_episodes_summary_hash_fields_missing"
                if deployment_episode_hash_strict_required
                else "deployment_episodes_summary_hash_fields_missing_legacy_compatible"
            )
            _add_issue(
                issue_list,
                issue_code,
                "deployment_episodes.summary hash bridge fields missing.",
                missing_fields=missing_episode_hash_fields,
            )
        for key, computed_value in episode_hash_summary_fields.items():
            if key in episodes_top_summary and str(episodes_top_summary.get(key) or "").strip() != computed_value:
                _add_issue(
                    failures,
                    f"deployment_episodes_summary_{key}_mismatch",
                    f"deployment_episodes.summary.{key} mismatch.",
                    manifest_value=episodes_top_summary.get(key),
                    computed_value=computed_value,
                )

    artifact_manifest_entries = _as_dict(artifact_manifest_payload.get("artifacts"))
    artifact_count = _to_int(artifact_manifest_payload.get("artifact_count"))
    if artifact_count <= 0 and artifact_manifest_entries:
        artifact_count = len(artifact_manifest_entries)
    contract_artifacts_summary = _as_dict(run_contract_payload.get("artifacts")) if run_contract_payload else {}
    closure_manifest_computed: dict[str, Any] = {}
    if should_validate_dataset_run_closure_manifest and dataset_run_closure_manifest_payload:
        normalized_closure = _normalize_dataset_run_closure_manifest(dataset_run_closure_manifest_payload)
        if normalized_closure.get("schema_version") != "carla_air_dataset_run_closure_manifest_v1":
            _add_issue(
                failures,
                "dataset_run_closure_manifest_schema_mismatch",
                "dataset_run_closure_manifest schema_version mismatch.",
                got=normalized_closure.get("schema_version"),
                expected="carla_air_dataset_run_closure_manifest_v1",
            )
        aligned_run_id = str(run_contract_payload.get("run_id") or "").strip() or str(plan_payload.get("run_id") or "").strip() or str(manifest_payload.get("run_id") or "").strip()
        if aligned_run_id and normalized_closure.get("run_id") != aligned_run_id:
            _add_issue(
                failures,
                "dataset_run_closure_manifest_run_id_mismatch",
                "dataset_run_closure_manifest.run_id must align with run_contract/plan/manifest run_id.",
                manifest_value=normalized_closure.get("run_id"),
                expected=aligned_run_id,
            )
        if str(normalized_closure.get("run_dir") or "").strip() != str(run_dir.resolve()):
            _add_issue(
                failures,
                "dataset_run_closure_manifest_run_dir_mismatch",
                "dataset_run_closure_manifest.run_dir must match verifier run_dir.",
                manifest_value=normalized_closure.get("run_dir"),
                expected=str(run_dir.resolve()),
            )
        closure_manifest_computed = {
            "schema_version": "carla_air_dataset_run_closure_manifest_v1",
            "run_id": aligned_run_id or normalized_closure.get("run_id"),
            "run_dir": str(run_dir.resolve()),
            "sample_count": sample_count,
            "scene_count": _to_int(scene_sample_index_manifest_payload.get("scene_count")) or _to_int(manifest_payload.get("scene_count")),
            "split_distribution": split_distribution,
            "capture_task_count": len(capture_queue_rows),
            "capture_queue_item_count": len(capture_queue_rows),
            "blocked_capture_queue_item_count": capture_queue_blocked_count,
            "scene_discovery_accepted_scene_root_count": accepted_scene_root_count if should_validate_scene_discovery_manifest else 0,
            "scene_sample_index_hash": str(
                scene_sample_index_manifest_payload.get("scene_sample_index_hash")
                or dataset_index_manifest_payload.get("scene_sample_index_hash")
                or ""
            ).strip(),
            "scene_split_membership_hash": str(
                scene_sample_index_manifest_payload.get("scene_split_membership_hash")
                or dataset_index_manifest_payload.get("scene_split_membership_hash")
                or ""
            ).strip(),
            "scene_keys_sorted_hash": str(
                scene_sample_index_manifest_payload.get("scene_keys_sorted_hash")
                or dataset_index_manifest_payload.get("scene_keys_sorted_hash")
                or ""
            ).strip(),
            "artifact_manifest_entry_count_excluding_self": (
                artifact_manifest_entry_count_excluding_self_reported
                if artifact_manifest_entry_count_excluding_self_reported is not None
                else artifact_count
            ),
            "contract_artifact_count_including_self": (
                contract_artifact_count_including_self_reported
                if contract_artifact_count_including_self_reported is not None
                else len(contract_artifacts_summary)
            ),
            "count_gap_explained_as_self_reference": (
                (
                    (contract_artifact_count_including_self_reported if contract_artifact_count_including_self_reported is not None else len(contract_artifacts_summary))
                    - (artifact_manifest_entry_count_excluding_self_reported if artifact_manifest_entry_count_excluding_self_reported is not None else artifact_count)
                )
                == 1
            ),
            "mask_gt_available_count": mask_gt_available_count_from_samples,
            "no_mask_sample_count": sample_count - mask_gt_available_count_from_samples,
            "identity_mismatch_count": _to_int(dataset_run_contract_summary_report.get("identity_mismatch_count")),
            "observed_identity_ids": observed_identity_ids,
            "planned_identity_ids": planned_identity_ids,
            "starts_runtime": False,
            "writes_scene_outputs": False,
            "non_promotion": True,
            "full_v1_live_dataset_ready": False,
        }
        closure_manifest_computed_stable_hashes = {
            "core_payload_sha256": _canonical_json_sha256(closure_manifest_computed)
        }
        observed_stable_hashes = _as_dict(dataset_run_closure_manifest_payload.get("stable_hashes"))
        if observed_stable_hashes != closure_manifest_computed_stable_hashes:
            _add_issue(
                failures,
                "dataset_run_closure_manifest_stable_hash_mismatch",
                "dataset_run_closure_manifest.stable_hashes mismatch against recomputed core payload digest.",
                manifest_value=observed_stable_hashes,
                computed_value=closure_manifest_computed_stable_hashes,
            )
        for bool_key, expected in (
            ("starts_runtime", False),
            ("writes_scene_outputs", False),
            ("non_promotion", True),
            ("full_v1_live_dataset_ready", False),
        ):
            if normalized_closure.get(bool_key) is not expected:
                _add_issue(
                    failures,
                    "dataset_run_closure_manifest_guard_flag_mismatch",
                    "dataset_run_closure_manifest offline/non-promotion guard mismatch.",
                    field=bool_key,
                    got=normalized_closure.get(bool_key),
                    expected=expected,
                )
        if normalized_closure != closure_manifest_computed:
            _add_issue(
                failures,
                "dataset_run_closure_manifest_computed_mismatch",
                "dataset_run_closure_manifest mismatch against verifier recomputed closure facts.",
                manifest_value=normalized_closure,
                computed_value=closure_manifest_computed,
            )
    required_artifact_present_count = sum(1 for key in REQUIRED_ARTIFACTS.values() if artifact_presence.get(key))
    missing_required_artifact_count = len(REQUIRED_ARTIFACTS) - required_artifact_present_count
    artifact_integrity_summary = {
        "artifact_manifest_present": artifact_presence.get("artifact_manifest", False),
        "artifact_count": artifact_count,
        "contract_artifact_count": len(contract_artifacts_summary),
        "artifact_accounting": {
            "self_artifact_key": (
                artifact_manifest_self_artifact_key_reported
                or contract_self_artifact_key_reported
                or "artifact_manifest_json"
            ),
            "contract_artifact_count_including_self": (
                contract_artifact_count_including_self_reported
                if contract_artifact_count_including_self_reported is not None
                else len(contract_artifacts_summary)
            ),
            "artifact_manifest_entry_count_excluding_self": (
                artifact_manifest_entry_count_excluding_self_reported
                if artifact_manifest_entry_count_excluding_self_reported is not None
                else artifact_count
            ),
            "excluded_self_reference_from_hashed_entries": (
                artifact_manifest_excluded_self_reference_from_hashed_entries
                if artifact_manifest_excluded_self_reference_from_hashed_entries is not None
                else contract_excluded_self_reference_from_hashed_entries
            ),
            "count_gap_explained_as_self_reference": (
                (
                    (contract_artifact_count_including_self_reported if contract_artifact_count_including_self_reported is not None else len(contract_artifacts_summary))
                    - (artifact_manifest_entry_count_excluding_self_reported if artifact_manifest_entry_count_excluding_self_reported is not None else artifact_count)
                )
                == 1
            ),
        },
        "required_artifact_present_count": required_artifact_present_count,
        "missing_required_artifact_count": missing_required_artifact_count,
        "dataset_index_manifest_present": artifact_presence.get("dataset_index_manifest", False),
        "dataset_index_manifest_schema_version": dataset_index_manifest_payload.get("schema_version") if dataset_index_manifest_payload else None,
        "dataset_index_manifest_scene_count": _to_int(dataset_index_manifest_payload.get("scene_count"))
        if dataset_index_manifest_payload
        else None,
        "dataset_index_manifest_scene_split_membership_hash": dataset_index_manifest_payload.get("scene_split_membership_hash")
        if dataset_index_manifest_payload
        else None,
        "dataset_index_manifest_scene_sample_index_hash": dataset_index_manifest_payload.get("scene_sample_index_hash")
        if dataset_index_manifest_payload
        else None,
        "sha256_available_count": sum(
            1 for entry in artifact_manifest_entries.values() if str(_as_dict(entry).get("sha256") or "").strip()
        ),
        "jsonl_row_count_available_count": sum(
            1
            for entry in artifact_manifest_entries.values()
            if _as_dict(entry).get("row_count") is not None
        ),
        "run_dir": str(run_dir.resolve()),
    }

    report = {
        "ok": len(failures) == 0,
        "failure_count": len(failures),
        "failures": failures,
        "warning_count": len(warnings),
        "warnings": warnings,
        "run_dir": str(run_dir.resolve()),
        "artifact_integrity_summary": artifact_integrity_summary,
        "strict_index_contract_summary": {
            "presence": strict_index_contract_presence,
            "report": strict_index_contract_report,
        },
        "sample_count": sample_count,
        "split_distribution": split_distribution,
        "artifact_presence": artifact_presence,
        "capture_task_count": {
            "plan_counts_capture_task_count": plan_capture_task_count,
            "plan_capture_tasks_len": len(plan_capture_tasks),
            "run_contract_counts_capture_task_count": contract_capture_task_count,
            "sample_with_capture_task_in_plan_count": sample_with_capture_task_in_plan_count,
            "sample_with_capture_task_id_count": sample_with_capture_task_id_count,
            "sample_with_trajectory_node_camera_bridge_count": sample_with_trajectory_node_camera_bridge_count,
            "sample_without_trajectory_node_camera_bridge_count": (
                sample_count - sample_with_trajectory_node_camera_bridge_count
            ),
            "planned_capture_task_candidate_reference_count": planned_capture_task_candidate_reference_count,
            "capture_matrix_bridge_status_counts": capture_matrix_bridge_status_counts,
            "samples_with_top_level_scene_id_count": samples_with_top_level_scene_id,
        },
        "modality_distribution": {
            "present_count_by_modality": modality_present,
            "missing_count_by_modality": modality_missing,
        },
        "sidecar_quality_summary": sidecar_quality_summary,
        "sidecar_quality_matrix": {
            "present": bool(manifest_sidecar_quality_matrix),
            "schema_version": manifest_sidecar_quality_matrix.get("schema_version")
            if manifest_sidecar_quality_matrix
            else None,
            "scene_count": _to_int(manifest_sidecar_quality_matrix.get("scene_count"))
            if manifest_sidecar_quality_matrix
            else None,
            "split_count": _to_int(manifest_sidecar_quality_matrix.get("split_count"))
            if manifest_sidecar_quality_matrix
            else None,
            "scene_split_count": _to_int(manifest_sidecar_quality_matrix.get("scene_split_count"))
            if manifest_sidecar_quality_matrix
            else None,
            "computed_scene_count": _to_int(computed_sidecar_quality_matrix.get("scene_count")),
            "computed_split_count": _to_int(computed_sidecar_quality_matrix.get("split_count")),
            "computed_scene_split_count": _to_int(computed_sidecar_quality_matrix.get("scene_split_count")),
            "overall": _as_dict(computed_sidecar_quality_matrix.get("overall")),
        },
        "sidecar_quality_manifest": {
            "present": artifact_presence["sidecar_quality_manifest"],
            "contract_declared": sidecar_quality_manifest_contract_declared,
            "contract_required": sidecar_quality_manifest_contract_required,
            "schema_version": sidecar_quality_manifest_payload.get("schema_version") if sidecar_quality_manifest_payload else None,
            "sample_count": _to_int(sidecar_quality_manifest_payload.get("sample_count")) if sidecar_quality_manifest_payload else None,
            "complete_count": _to_int(sidecar_quality_manifest_payload.get("complete_rgb_depth_semantic_instance_pose_calib_count"))
            if sidecar_quality_manifest_payload
            else None,
            "complete_fraction": float(sidecar_quality_manifest_payload.get("complete_fraction") or 0.0)
            if sidecar_quality_manifest_payload
            else None,
            "mask_gt_available_count": _to_int(sidecar_quality_manifest_payload.get("mask_gt_available_count"))
            if sidecar_quality_manifest_payload
            else None,
            "no_mask_sample_count": _to_int(sidecar_quality_manifest_payload.get("no_mask_sample_count"))
            if sidecar_quality_manifest_payload
            else None,
            "computed_sample_count": sample_count,
            "computed_complete_count": _to_int(sidecar_quality_summary.get("complete_rgb_depth_semantic_instance_pose_calib_count")),
            "computed_complete_fraction": float(sidecar_quality_summary.get("complete_fraction") or 0.0),
            "computed_mask_gt_available_count": _to_int(sidecar_quality_summary.get("mask_gt_available_count")),
            "computed_no_mask_sample_count": _to_int(sidecar_quality_summary.get("no_mask_sample_count")),
        },
        "dataset_gap_manifest": {
            "present": artifact_presence["dataset_gap_manifest"],
            "contract_declared": dataset_gap_manifest_contract_declared,
            "contract_required": dataset_gap_manifest_contract_required,
            "schema_version": dataset_gap_manifest_payload.get("schema_version") if dataset_gap_manifest_payload else None,
            "run_id": dataset_gap_manifest_payload.get("run_id") if dataset_gap_manifest_payload else None,
            "sample_count": _to_int(dataset_gap_manifest_payload.get("sample_count")) if dataset_gap_manifest_payload else None,
            "scene_count": _to_int(dataset_gap_manifest_payload.get("scene_count")) if dataset_gap_manifest_payload else None,
            "mask_gt_available_count": _to_int(dataset_gap_manifest_payload.get("mask_gt_available_count"))
            if dataset_gap_manifest_payload
            else None,
            "no_mask_sample_count": _to_int(dataset_gap_manifest_payload.get("no_mask_sample_count"))
            if dataset_gap_manifest_payload
            else None,
            "sidecar_complete_count": _to_int(dataset_gap_manifest_payload.get("sidecar_complete_count"))
            if dataset_gap_manifest_payload
            else None,
            "sidecar_incomplete_sample_count": _to_int(dataset_gap_manifest_payload.get("sidecar_incomplete_sample_count"))
            if dataset_gap_manifest_payload
            else None,
            "stable_hashes": _as_dict(dataset_gap_manifest_payload.get("stable_hashes")) if dataset_gap_manifest_payload else {},
        },
        "no_mask_non_promotion_manifest": {
            "present": artifact_presence["no_mask_non_promotion_manifest"],
            "contract_declared": no_mask_non_promotion_manifest_contract_declared,
            "contract_required": no_mask_non_promotion_manifest_contract_required,
            "schema_version": no_mask_non_promotion_manifest_payload.get("schema_version")
            if no_mask_non_promotion_manifest_payload
            else None,
            "run_id": no_mask_non_promotion_manifest_payload.get("run_id")
            if no_mask_non_promotion_manifest_payload
            else None,
            "sample_count": _to_int(no_mask_non_promotion_manifest_payload.get("sample_count"))
            if no_mask_non_promotion_manifest_payload
            else None,
            "mask_gt_available_count": _to_int(no_mask_non_promotion_manifest_payload.get("mask_gt_available_count"))
            if no_mask_non_promotion_manifest_payload
            else None,
            "no_mask_sample_count": _to_int(no_mask_non_promotion_manifest_payload.get("no_mask_sample_count"))
            if no_mask_non_promotion_manifest_payload
            else None,
            "computed_sample_count": sample_count,
            "computed_mask_gt_available_count": mask_gt_available_count_from_samples,
            "computed_no_mask_sample_count": sample_count - mask_gt_available_count_from_samples,
            "policy": _as_dict(no_mask_non_promotion_manifest_payload.get("policy"))
            if no_mask_non_promotion_manifest_payload
            else {},
        },
        "deployment_episode_visibility_manifest": {
            "present": artifact_presence["deployment_episode_visibility_manifest"],
            "contract_declared": deployment_episode_visibility_manifest_contract_declared,
            "contract_required": deployment_episode_visibility_manifest_contract_required,
            "schema_version": deployment_episode_visibility_manifest_payload.get("schema_version")
            if deployment_episode_visibility_manifest_payload
            else None,
            "run_id": deployment_episode_visibility_manifest_payload.get("run_id")
            if deployment_episode_visibility_manifest_payload
            else None,
            "episode_count": _to_int(deployment_episode_visibility_manifest_payload.get("episode_count"))
            if deployment_episode_visibility_manifest_payload
            else None,
            "sample_count_total_from_episodes": _to_int(
                deployment_episode_visibility_manifest_payload.get("sample_count_total_from_episodes")
            )
            if deployment_episode_visibility_manifest_payload
            else None,
            "scene_count_total_from_episodes": _to_int(
                deployment_episode_visibility_manifest_payload.get("scene_count_total_from_episodes")
            )
            if deployment_episode_visibility_manifest_payload
            else None,
            "episode_with_visibility_gap_count": _to_int(
                deployment_episode_visibility_manifest_payload.get("episode_with_visibility_gap_count")
            )
            if deployment_episode_visibility_manifest_payload
            else None,
            "episode_without_samples_count": _to_int(
                deployment_episode_visibility_manifest_payload.get("episode_without_samples_count")
            )
            if deployment_episode_visibility_manifest_payload
            else None,
            "computed_episode_count": len(_as_list(episodes_payload.get("episodes"))) if episodes_payload else 0,
            "computed_sample_count_total_from_episodes": deployment_sample_count_total_from_episodes,
            "computed_scene_count_total_from_episodes": len(deployment_scene_ids_union),
            "computed_episode_with_visibility_gap_count": deployment_episode_with_visibility_gap_count,
            "computed_episode_without_samples_count": deployment_episode_without_samples_count,
            "episode_scene_visibility_hash": deployment_episode_visibility_manifest_payload.get(
                "episode_scene_visibility_hash"
            )
            if deployment_episode_visibility_manifest_payload
            else None,
            "episode_sample_order_hash": deployment_episode_visibility_manifest_payload.get(
                "episode_sample_order_hash"
            )
            if deployment_episode_visibility_manifest_payload
            else None,
            "episode_sample_sorted_hash": deployment_episode_visibility_manifest_payload.get(
                "episode_sample_sorted_hash"
            )
            if deployment_episode_visibility_manifest_payload
            else None,
            "computed_episode_scene_visibility_hash": _hash_text_parts(deployment_episode_scene_rows),
            "computed_episode_sample_order_hash": _hash_text_parts(deployment_episode_sample_order_rows),
            "computed_episode_sample_sorted_hash": _hash_text_parts(deployment_episode_sample_sorted_rows),
        },
        "sample_schema_coverage_summary": computed_schema_coverage_summary,
        "sample_schema_coverage_manifest": {
            "present": artifact_presence["sample_schema_coverage_manifest"],
            "contract_declared": sample_schema_coverage_manifest_contract_declared,
            "contract_required": sample_schema_coverage_manifest_contract_required,
            "schema_version": sample_schema_coverage_manifest_payload.get("schema_version")
            if sample_schema_coverage_manifest_payload
            else None,
            "run_id": sample_schema_coverage_manifest_payload.get("run_id")
            if sample_schema_coverage_manifest_payload
            else None,
            "sample_count": _to_int(sample_schema_coverage_manifest_payload.get("sample_count"))
            if sample_schema_coverage_manifest_payload
            else None,
            "computed_sample_count": _to_int(recomputed_schema_coverage_manifest.get("sample_count")),
            "field_missing_count": _as_dict(sample_schema_coverage_manifest_payload.get("field_missing_count"))
            if sample_schema_coverage_manifest_payload
            else {},
            "computed_field_missing_count": _as_dict(recomputed_schema_coverage_manifest.get("field_missing_count")),
            "stable_hashes": _as_dict(sample_schema_coverage_manifest_payload.get("stable_hashes"))
            if sample_schema_coverage_manifest_payload
            else {},
            "computed_stable_hashes": _as_dict(recomputed_schema_coverage_manifest.get("stable_hashes")),
        },
        "identity_model_switch_alignment_summary": identity_model_switch_alignment_summary,
        "mask_gt_non_promotion_verified": mask_gt_non_promotion_verified,
        "starts_runtime": False,
        "writes_scene_outputs": False,
        "capture_queue": {
            "present": artifact_presence["capture_queue"],
            "contract_required": capture_queue_contract_required,
            "item_count": len(capture_queue_rows),
            "blocked_item_count": capture_queue_blocked_count,
        },
        "capture_queue_manifest": {
            "present": artifact_presence["capture_queue_manifest"],
            "contract_declared": capture_queue_manifest_contract_declared,
            "contract_required": capture_queue_manifest_contract_required,
            "schema_version": capture_queue_manifest_payload.get("schema_version")
            if capture_queue_manifest_payload
            else None,
            "run_id": capture_queue_manifest_payload.get("run_id")
            if capture_queue_manifest_payload
            else None,
            "capture_queue_item_count": _to_int(capture_queue_manifest_payload.get("capture_queue_item_count"))
            if capture_queue_manifest_payload
            else None,
            "blocked_capture_queue_item_count": _to_int(
                capture_queue_manifest_payload.get("blocked_capture_queue_item_count")
            )
            if capture_queue_manifest_payload
            else None,
            "queued_capture_queue_item_count": _to_int(
                capture_queue_manifest_payload.get("queued_capture_queue_item_count")
            )
            if capture_queue_manifest_payload
            else None,
            "state_counts": _as_dict(capture_queue_manifest_payload.get("state_counts"))
            if capture_queue_manifest_payload
            else {},
            "block_reason_counts": _as_dict(capture_queue_manifest_payload.get("block_reason_counts"))
            if capture_queue_manifest_payload
            else {},
            "capture_task_id_order_sha256": capture_queue_manifest_payload.get("capture_task_id_order_sha256")
            if capture_queue_manifest_payload
            else None,
            "expected_scene_root_order_sha256": capture_queue_manifest_payload.get("expected_scene_root_order_sha256")
            if capture_queue_manifest_payload
            else None,
            "source_capture_queue_path": capture_queue_manifest_payload.get("source_capture_queue_path")
            if capture_queue_manifest_payload
            else None,
            "computed": capture_queue_manifest_computed if capture_queue_manifest_computed else {},
        },
        "scene_discovery_manifest": {
            "present": artifact_presence["scene_discovery_manifest"],
            "contract_required": scene_discovery_manifest_contract_required,
            "accepted_scene_root_count": accepted_scene_root_count if should_validate_scene_discovery_manifest else None,
        },
        "scene_output_manifest": {
            "present": artifact_presence["scene_output_manifest"],
            "contract_required": scene_output_manifest_contract_required,
            "scene_output_count": scene_output_count,
            "blocked_scene_output_count": blocked_scene_output_count,
        },
        "batch_run_manifest": {
            "present": artifact_presence["batch_run_manifest"],
            "contract_required": batch_run_manifest_contract_required,
            "schema_version": batch_run_manifest_payload.get("schema_version") if batch_run_manifest_payload else None,
        },
        "deployment_episodes": deployment_episodes_summary,
        "split_policy": {
            "summary": split_policy_summary_report,
            "digest": split_policy_digest_report,
            "digest_sources": split_policy_digest_sources,
            "artifact_count_with_any_field": split_policy_artifact_count_with_any,
            "artifact_count_with_digest": split_policy_artifact_count_with_digest,
        },
        "dataset_run_contract_summary": {
            "observed_source_count": len(dataset_run_contract_present_sources),
            "expected_source_count": len(dataset_run_contract_sources),
            "sources": [name for name, _ in dataset_run_contract_present_sources],
            "source_digests": dataset_run_contract_source_digests,
            "summary": dataset_run_contract_summary_report,
            "digest": dataset_run_contract_summary_digest,
        },
        "scene_capture_alignment_summary": scene_capture_alignment_summary,
        "scene_membership_manifest": scene_membership_manifest_summary,
        "scene_sample_index_manifest": {
            "present": artifact_presence["scene_sample_index_manifest"],
            "contract_declared": scene_sample_index_manifest_contract_declared,
            "contract_required": scene_sample_index_manifest_contract_required,
            "schema_version": scene_sample_index_manifest_payload.get("schema_version") if scene_sample_index_manifest_payload else None,
            "scene_count": _to_int(scene_sample_index_manifest_payload.get("scene_count")) if scene_sample_index_manifest_payload else None,
            "sample_count": _to_int(scene_sample_index_manifest_payload.get("sample_count")) if scene_sample_index_manifest_payload else None,
            "scene_split_membership_hash": scene_sample_index_manifest_payload.get("scene_split_membership_hash")
            if scene_sample_index_manifest_payload
            else None,
            "dataset_index_scene_split_membership_hash": scene_sample_index_manifest_payload.get(
                "dataset_index_scene_split_membership_hash"
            )
            if scene_sample_index_manifest_payload
            else None,
            "scene_sample_index_hash": scene_sample_index_manifest_payload.get("scene_sample_index_hash")
            if scene_sample_index_manifest_payload
            else None,
            "scene_keys_sorted_hash": scene_sample_index_manifest_payload.get("scene_keys_sorted_hash")
            if scene_sample_index_manifest_payload
            else None,
        },
        "scene_membership_alignment_manifest": scene_membership_alignment_manifest_summary,
        "capture_matrix_alignment_manifest": capture_matrix_alignment_manifest_summary,
        "dataset_run_closure_manifest": {
            "present": artifact_presence["dataset_run_closure_manifest"],
            "contract_declared": dataset_run_closure_manifest_contract_declared,
            "contract_required": dataset_run_closure_manifest_contract_required,
            "schema_version": dataset_run_closure_manifest_payload.get("schema_version")
            if dataset_run_closure_manifest_payload
            else None,
            "stable_hashes": _as_dict(dataset_run_closure_manifest_payload.get("stable_hashes"))
            if dataset_run_closure_manifest_payload
            else {},
            "computed_stable_hashes": (
                {"core_payload_sha256": _canonical_json_sha256(closure_manifest_computed)}
                if closure_manifest_computed
                else {}
            ),
        },
        "identity_model_switch_manifest": {
            "present": artifact_presence["identity_model_switch_manifest"],
            "contract_declared": identity_model_switch_manifest_contract_declared,
            "contract_required": identity_model_switch_manifest_contract_required,
            "schema_version": identity_model_switch_manifest_payload.get("schema_version") if identity_model_switch_manifest_payload else None,
        },
        "existing_scene_index_bridge_manifest": {
            "present": artifact_presence["existing_scene_index_bridge_manifest"],
            "contract_declared": existing_scene_index_bridge_manifest_contract_declared,
            "contract_required": existing_scene_index_bridge_manifest_contract_required,
            "schema_version": (
                existing_scene_index_bridge_manifest_payload.get("schema_version")
                if existing_scene_index_bridge_manifest_payload
                else None
            ),
        },
    }

    if args.out:
        out_path = _validate_out_path(args.out, bool(args.allow_nonlocal_out))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(report, ensure_ascii=True, indent=2))
    if args.allow_fail:
        return 0
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
