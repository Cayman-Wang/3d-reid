#!/usr/bin/env python3
"""Offline CARLA-Air Dataset Generation Pipeline V1 orchestrator.

This script is read-only with respect to runtime/assets/scene outputs:
- It does not start CARLA/AirSim.
- It does not modify scene contents.
- It only plans and indexes existing scene artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
LOCAL_ROOT = REPO_ROOT / "local"
DEFAULT_RUN_ROOT = REPO_ROOT / "local/carla_air/dataset_runs"
ALLOWED_NODES = {"node01", "node02", "node03", "node04", "node05"}
FORBIDDEN_ROOT_TOKENS = {"weak", "proxy"}
SCENE_ROOT_RE = re.compile(r"^carla_air_\d{8}_\d{6}_traj_[A-Za-z0-9_]+_node\d{2}$")


def _load_planner_module() -> Any:
    planner_path = REPO_ROOT / "tools/carla_air/plan_dataset_generation_v1.py"
    spec = importlib.util.spec_from_file_location("plan_dataset_generation_v1", planner_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load planner module from {planner_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


planner = _load_planner_module()


def _parse_id_filters(raw_values: list[str] | None) -> list[str]:
    if not raw_values:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for raw in raw_values:
        for part in str(raw).split(","):
            item = part.strip()
            if not item or item in seen:
                continue
            seen.add(item)
            out.append(item)
    return out


def _repo_or_abs(raw: str | Path) -> Path:
    path = Path(str(raw))
    return path if path.is_absolute() else REPO_ROOT / path


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _validate_run_root(raw_run_root: str, allow_nonlocal_out: bool) -> Path:
    run_root = _repo_or_abs(raw_run_root)
    if not allow_nonlocal_out and not _is_under(run_root, LOCAL_ROOT):
        raise SystemExit("--run-root must stay under repository local/ unless --allow-nonlocal-out is set")
    return run_root


def _forbidden_scene_root(path: Path) -> bool:
    parts = {part.lower() for part in path.parts}
    return any(token in parts for token in FORBIDDEN_ROOT_TOKENS)


def _scene_meta(scene_dir: Path) -> dict[str, Any]:
    for name in ("capture_meta.json", "pipeline_contract.json"):
        path = scene_dir / name
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(payload, dict):
            return payload
    return {}


def _path_for_manifest(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except Exception:
        return str(path.resolve())


def _scene_fields_from_root(path: Path) -> dict[str, str]:
    meta = _scene_meta(path)
    node_id = str(meta.get("node_id") or "").strip()
    trajectory_id = str(meta.get("trajectory_id") or "").strip()
    scene_id = str(meta.get("scene_id") or "").strip()
    if not scene_id:
        scene_id = path.name
    if not node_id and ".node" in scene_id:
        maybe_node = scene_id.split(".")[-1].strip()
        if maybe_node:
            node_id = maybe_node
    if not trajectory_id and "." in scene_id:
        parts = [p.strip() for p in scene_id.split(".") if p.strip()]
        if len(parts) >= 2:
            trajectory_id = parts[-2]
    return {
        "node_id": node_id,
        "trajectory_id": trajectory_id,
        "scene_id": scene_id,
        "has_capture_meta": (path / "capture_meta.json").is_file(),
        "has_pipeline_contract": (path / "pipeline_contract.json").is_file(),
    }


def _discover_scene_roots(selected_nodes: list[str], selected_trajectories: list[str]) -> list[Path]:
    roots: list[Path] = []
    seen: set[Path] = set()
    trajectory_filter = {str(item) for item in selected_trajectories if str(item)}
    for node_id in selected_nodes:
        if node_id not in ALLOWED_NODES:
            continue
        scenes_root = REPO_ROOT / "data/carla_air/nodes" / node_id / "scenes"
        if not scenes_root.is_dir():
            continue
        for candidate in sorted(scenes_root.iterdir()):
            if not candidate.is_dir():
                continue
            if not SCENE_ROOT_RE.match(candidate.name):
                continue
            has_contract = (candidate / "capture_meta.json").is_file() or (candidate / "pipeline_contract.json").is_file()
            if not has_contract:
                continue
            meta = _scene_meta(candidate)
            scene_node = str(meta.get("node_id") or "").strip()
            scene_trajectory = str(meta.get("trajectory_id") or "").strip()
            if not scene_node or scene_node != node_id:
                continue
            if trajectory_filter and (not scene_trajectory or scene_trajectory not in trajectory_filter):
                continue
            resolved = candidate.resolve()
            if _forbidden_scene_root(resolved):
                continue
            if resolved in seen:
                continue
            seen.add(resolved)
            roots.append(resolved)
    return roots


def _extract_last_json_object(raw: str) -> dict[str, Any]:
    text = str(raw or "").strip()
    if not text:
        return {}
    try:
        payload = json.loads(text)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        pass
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in reversed(lines):
        try:
            payload = json.loads(line)
            if isinstance(payload, dict):
                return payload
        except Exception:
            continue
    return {}


def _normalize_run_id(raw: str | None) -> str:
    if raw:
        return str(raw).strip()
    return f"dataset_v1_orchestrated_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True))
            f.write("\n")


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


def _artifact_metadata_entry(key: str, path: Path, *, include_hash: bool = True) -> dict[str, Any]:
    exists = path.is_file()
    entry: dict[str, Any] = {
        "key": key,
        "path": str(path.resolve()),
        "exists": exists,
        "size_bytes": path.stat().st_size if exists else 0,
        "sha256": _sha256_file(path) if (exists and include_hash) else None,
        "schema_version": None,
        "row_count": None,
    }
    if not exists:
        return entry
    if path.suffix.lower() == ".jsonl":
        entry["row_count"] = _jsonl_row_count(path)
        return entry
    if path.suffix.lower() == ".json":
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return entry
        if isinstance(payload, dict):
            schema_version = payload.get("schema_version")
            if schema_version is not None:
                entry["schema_version"] = str(schema_version)
    return entry


def _scene_output_readiness(capture_allowed_now: bool, block_reason: str) -> dict[str, Any]:
    blocked_reasons = [block_reason] if block_reason else []
    return {
        "status": "ready_for_execution" if capture_allowed_now else "blocked",
        "blocked": not capture_allowed_now,
        "blocked_reasons": blocked_reasons,
        "evidence_ready": bool(capture_allowed_now),
    }


def _safe_json_dict(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _canonical_json_sha256(payload: Any) -> str:
    canonical = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _build_dataset_run_closure_manifest(
    *,
    run_id: str,
    run_dir: Path,
    sample_count: int,
    capture_queue_items: list[dict[str, Any]],
    scene_discovery_manifest_payload: dict[str, Any],
    dataset_manifest_payload: dict[str, Any],
    dataset_index_manifest_payload: dict[str, Any],
    scene_sample_index_manifest_payload: dict[str, Any],
    scene_membership_manifest_payload: dict[str, Any],
    capture_matrix_alignment_manifest_payload: dict[str, Any],
    dataset_run_contract_summary: dict[str, Any],
    contract_artifact_count_including_self: int,
    artifact_manifest_entry_count_excluding_self: int,
) -> dict[str, Any]:
    split_distribution_source = (
        dataset_manifest_payload.get("split_distribution")
        if isinstance(dataset_manifest_payload.get("split_distribution"), dict)
        else scene_sample_index_manifest_payload.get("split_distribution")
        if isinstance(scene_sample_index_manifest_payload.get("split_distribution"), dict)
        else {}
    )
    split_distribution = {
        str(k).strip(): _safe_int(v)
        for k, v in split_distribution_source.items()
        if str(k).strip()
    }
    capture_queue_item_count = len(capture_queue_items)
    blocked_capture_queue_item_count = sum(1 for item in capture_queue_items if _safe_int(item.get("state") == "blocked"))
    accepted_scene_root_count = _safe_int(scene_discovery_manifest_payload.get("scene_root_count"))
    scene_count = _safe_int(scene_sample_index_manifest_payload.get("scene_count")) or _safe_int(dataset_manifest_payload.get("scene_count"))
    scene_split_membership_hash = str(
        scene_sample_index_manifest_payload.get("scene_split_membership_hash")
        or dataset_index_manifest_payload.get("scene_split_membership_hash")
        or ""
    ).strip()
    scene_sample_index_hash = str(
        scene_sample_index_manifest_payload.get("scene_sample_index_hash")
        or dataset_index_manifest_payload.get("scene_sample_index_hash")
        or ""
    ).strip()
    scene_keys_sorted_hash = str(
        scene_sample_index_manifest_payload.get("scene_keys_sorted_hash")
        or dataset_index_manifest_payload.get("scene_keys_sorted_hash")
        or ""
    ).strip()
    no_mask_manifest = dataset_manifest_payload.get("no_mask_non_promotion_summary")
    no_mask_manifest_obj = no_mask_manifest if isinstance(no_mask_manifest, dict) else {}
    mask_gt_available_count = _safe_int(no_mask_manifest_obj.get("mask_gt_available_count"))
    no_mask_sample_count = _safe_int(no_mask_manifest_obj.get("no_mask_sample_count"))
    if not no_mask_manifest_obj:
        capture_summary = capture_matrix_alignment_manifest_payload.get("summary")
        capture_summary_obj = capture_summary if isinstance(capture_summary, dict) else {}
        mask_gt_available_count = _safe_int(capture_summary_obj.get("mask_gt_available_count"))
        no_mask_sample_count = _safe_int(capture_summary_obj.get("no_mask_sample_count"))
    identity_alignment = dataset_manifest_payload.get("capture_task_alignment_summary")
    identity_alignment_obj = identity_alignment if isinstance(identity_alignment, dict) else {}
    scene_membership_summary = scene_membership_manifest_payload.get("capture_task_alignment_summary")
    scene_membership_summary_obj = scene_membership_summary if isinstance(scene_membership_summary, dict) else {}
    contract_observed_identity_ids = sorted(
        {
            str(x).strip()
            for x in (
                dataset_run_contract_summary.get("observed_identity_ids")
                if isinstance(dataset_run_contract_summary.get("observed_identity_ids"), list)
                else []
            )
            if str(x).strip()
        }
    )
    contract_planned_identity_ids = sorted(
        {
            str(x).strip()
            for x in (
                dataset_run_contract_summary.get("planned_identity_ids")
                if isinstance(dataset_run_contract_summary.get("planned_identity_ids"), list)
                else []
            )
            if str(x).strip()
        }
    )
    observed_identity_ids = sorted(
        {
            str(x).strip()
            for x in (
                identity_alignment_obj.get("observed_identity_ids")
                if isinstance(identity_alignment_obj.get("observed_identity_ids"), list)
                else scene_membership_summary_obj.get("observed_identity_ids")
                if isinstance(scene_membership_summary_obj.get("observed_identity_ids"), list)
                else []
            )
            if str(x).strip()
        }
    )
    if contract_observed_identity_ids:
        observed_identity_ids = contract_observed_identity_ids
    planned_identity_ids = sorted(
        {
            str(x).strip()
            for x in (
                identity_alignment_obj.get("planned_identity_ids")
                if isinstance(identity_alignment_obj.get("planned_identity_ids"), list)
                else scene_membership_summary_obj.get("planned_identity_ids")
                if isinstance(scene_membership_summary_obj.get("planned_identity_ids"), list)
                else []
            )
            if str(x).strip()
        }
    )
    if contract_planned_identity_ids:
        planned_identity_ids = contract_planned_identity_ids
    if "identity_mismatch_count" in dataset_run_contract_summary:
        identity_mismatch_count = _safe_int(dataset_run_contract_summary.get("identity_mismatch_count"))
    else:
        identity_mismatch_count = _safe_int(
            scene_membership_summary_obj.get("identity_passthrough_mismatch_scene_count")
        ) or _safe_int(identity_alignment_obj.get("identity_passthrough_mismatch_scene_count"))

    core_payload = {
        "schema_version": "carla_air_dataset_run_closure_manifest_v1",
        "run_id": run_id,
        "run_dir": str(run_dir.resolve()),
        "sample_count": sample_count,
        "scene_count": scene_count,
        "split_distribution": split_distribution,
        "capture_task_count": capture_queue_item_count,
        "capture_queue_item_count": capture_queue_item_count,
        "blocked_capture_queue_item_count": blocked_capture_queue_item_count,
        "scene_discovery_accepted_scene_root_count": accepted_scene_root_count,
        "scene_sample_index_hash": scene_sample_index_hash,
        "scene_split_membership_hash": scene_split_membership_hash,
        "scene_keys_sorted_hash": scene_keys_sorted_hash,
        "artifact_manifest_entry_count_excluding_self": artifact_manifest_entry_count_excluding_self,
        "contract_artifact_count_including_self": contract_artifact_count_including_self,
        "count_gap_explained_as_self_reference": (
            contract_artifact_count_including_self - artifact_manifest_entry_count_excluding_self
        )
        == 1,
        "mask_gt_available_count": mask_gt_available_count,
        "no_mask_sample_count": no_mask_sample_count,
        "identity_mismatch_count": identity_mismatch_count,
        "observed_identity_ids": observed_identity_ids,
        "planned_identity_ids": planned_identity_ids,
        "starts_runtime": False,
        "writes_scene_outputs": False,
        "non_promotion": True,
        "full_v1_live_dataset_ready": False,
    }
    return {
        **core_payload,
        "stable_hashes": {
            "core_payload_sha256": _canonical_json_sha256(core_payload),
        },
    }


def _build_identity_model_switch_manifest_fallback(
    *,
    run_id: str,
    plan_payload: dict[str, Any],
    capture_queue_item_count: int,
    blocked_capture_queue_item_count: int,
    identity_switch_contract: dict[str, Any],
) -> dict[str, Any]:
    plan_identity_model_profiles = [
        item
        for item in (plan_payload.get("identity_model_profiles") if isinstance(plan_payload.get("identity_model_profiles"), list) else [])
        if isinstance(item, dict)
    ]
    planned_identity_ids = sorted(
        {str(item.get("identity_id") or "").strip() for item in plan_identity_model_profiles if str(item.get("identity_id") or "").strip()}
    )
    profile_switch_methods = sorted(
        {str(item.get("switch_method") or "").strip() for item in plan_identity_model_profiles if str(item.get("switch_method") or "").strip()}
    )
    profile_model_labels = sorted(
        {str(item.get("model_label") or "").strip() for item in plan_identity_model_profiles if str(item.get("model_label") or "").strip()}
    )
    profile_requires_import_readback_flags = sorted(
        {bool(item.get("requires_ue_carla_import_readback")) for item in plan_identity_model_profiles}
    )
    observed_sample_identity_ids = sorted(
        {
            str(x).strip()
            for x in (identity_switch_contract.get("observed_sample_identity_ids") if isinstance(identity_switch_contract.get("observed_sample_identity_ids"), list) else [])
            if str(x).strip()
        }
    )
    sample_count = _safe_int(identity_switch_contract.get("strict_planned_identity_sample_count")) + _safe_int(
        identity_switch_contract.get("observed_passthrough_identity_sample_count")
    )
    identity_mismatch_count = _safe_int(identity_switch_contract.get("identity_mismatch_count"))
    strict_planned_identity_sample_count = _safe_int(identity_switch_contract.get("strict_planned_identity_sample_count"))
    observed_passthrough_identity_sample_count = _safe_int(
        identity_switch_contract.get("observed_passthrough_identity_sample_count")
    )
    all_samples_match_planned_identities = identity_switch_contract.get("all_samples_match_planned_identities") is True
    identity_alignment_status = (
        "strict_planned_identity_match"
        if sample_count > 0 and all_samples_match_planned_identities
        else "observed_scene_passthrough"
        if sample_count > 0 and identity_mismatch_count > 0
        else "plan_only_or_no_samples"
    )
    identity_alignment_summary = {
        "sample_count": sample_count,
        "planned_identity_ids": planned_identity_ids,
        "observed_sample_identity_ids": observed_sample_identity_ids,
        "strict_planned_identity_sample_count": strict_planned_identity_sample_count,
        "observed_passthrough_identity_sample_count": observed_passthrough_identity_sample_count,
        "identity_mismatch_count": identity_mismatch_count,
        "all_samples_match_planned_identities": all_samples_match_planned_identities,
        "status": identity_alignment_status,
        "requires_ue_carla_import_readback": True,
        "legacy_or_observed_scene_passthrough_allowed_for_no_mask_index": True,
        "full_v1_live_dataset_ready": False,
        "non_promotion": True,
    }
    return {
        "schema_version": "carla_air_identity_model_switch_manifest_v1",
        "run_id": run_id,
        "starts_runtime": False,
        "writes_scene_outputs": False,
        "non_promotion": True,
        "full_v1_live_dataset_ready": False,
        "capture_profile": plan_payload.get("capture_profile"),
        "planned_identity_ids": planned_identity_ids,
        "identity_model_profiles": plan_identity_model_profiles,
        "profile_count": len(plan_identity_model_profiles),
        "identity_count": len(planned_identity_ids),
        "switch_methods": profile_switch_methods,
        "model_labels": profile_model_labels,
        "requires_ue_carla_import_readback_flags": profile_requires_import_readback_flags,
        "capture_task_count": capture_queue_item_count,
        "blocked_capture_task_count": blocked_capture_queue_item_count,
        "observed_sample_identity_ids": observed_sample_identity_ids,
        "identity_mismatch_count": identity_mismatch_count,
        "identity_alignment_summary": identity_alignment_summary,
        "no_silent_identity_rewrite": True,
        "legacy_or_observed_scene_passthrough_allowed_for_no_mask_index": True,
    }


def _extract_dataset_run_contract_summary(
    dataset_manifest_payload: dict[str, Any],
    dataset_index_manifest_payload: dict[str, Any],
) -> dict[str, Any]:
    index_summary = (
        dataset_index_manifest_payload.get("dataset_run_contract_summary")
        if isinstance(dataset_index_manifest_payload.get("dataset_run_contract_summary"), dict)
        else {}
    )
    manifest_summary = (
        dataset_manifest_payload.get("dataset_run_contract_summary")
        if isinstance(dataset_manifest_payload.get("dataset_run_contract_summary"), dict)
        else {}
    )
    if index_summary:
        return dict(index_summary)
    if manifest_summary:
        return dict(manifest_summary)
    return {
        "schema_version": "carla_air_dataset_run_contract_summary_v1",
        "sample_count": 0,
        "scene_count": 0,
        "split_distribution": {},
        "capture_task_count": 0,
        "sample_with_capture_task_count": 0,
        "sample_without_capture_task_count": 0,
        "strict_matrix_entry_sample_count": 0,
        "legacy_or_observed_scene_passthrough_count": 0,
        "mask_gt_available_count": 0,
        "no_mask_sample_count": 0,
        "sidecar_complete_count": 0,
        "sidecar_complete_fraction": 0.0,
        "sidecar_missing_count_by_modality": {},
        "planned_identity_ids": [],
        "observed_identity_ids": [],
        "identity_mismatch_count": 0,
        "strict_planned_identity_sample_count": 0,
        "observed_passthrough_identity_sample_count": 0,
        "starts_runtime": False,
        "writes_scene_outputs": False,
        "non_promotion": True,
        "full_v1_live_dataset_ready": False,
    }


def _build_scene_membership_alignment_manifest(
    *,
    run_id: str,
    scene_output_manifest: dict[str, Any],
    scene_membership_manifest: dict[str, Any],
) -> dict[str, Any]:
    planned_rows = scene_output_manifest.get("scene_outputs")
    if not isinstance(planned_rows, list):
        planned_rows = []
    observed_rows = scene_membership_manifest.get("scene_entries")
    if not isinstance(observed_rows, list):
        observed_rows = []

    observed_by_tn: dict[tuple[str, str], list[dict[str, Any]]] = {}
    observed_by_scene: dict[str, list[dict[str, Any]]] = {}
    observed_passthrough_count = 0
    for item in observed_rows:
        if not isinstance(item, dict):
            continue
        trajectory_id = str(item.get("trajectory_id") or "").strip()
        node_id = str(item.get("node_id") or "").strip()
        scene_id = str(item.get("scene_id") or "").strip()
        if trajectory_id and node_id:
            observed_by_tn.setdefault((trajectory_id, node_id), []).append(item)
        if scene_id:
            observed_by_scene.setdefault(scene_id, []).append(item)
        alignment = item.get("capture_task_alignment")
        if isinstance(alignment, dict):
            if _safe_int(alignment.get("identity_passthrough_mismatch_count")) > 0:
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
            sample_count = _safe_int(obs.get("sample_count"))
            obs_scene_id = str(obs.get("scene_id") or "").strip()
            return (exact_identity, sample_count, obs_scene_id)

        chosen = sorted(candidates, key=_sort_key, reverse=True)[0]
        return chosen, join_mode

    rows: list[dict[str, Any]] = []
    planned_blocked_count = 0
    trajectory_node_match_count = 0
    exact_identity_match_count = 0
    identity_mismatch_count = 0
    missing_observation_count = 0

    for planned in planned_rows:
        if not isinstance(planned, dict):
            continue
        planned_identity_id = str(planned.get("identity_id") or "").strip()
        planned_scene_id = str(planned.get("scene_id") or "").strip()
        planned_trajectory_id = str(planned.get("trajectory_id") or "").strip()
        planned_node_id = str(planned.get("node_id") or "").strip()
        expected_scene_root = str(planned.get("expected_scene_root") or "").strip()
        scene_output_state = str(planned.get("state") or "").strip() or "unknown"
        requires_import_readback = bool(planned.get("requires_ue_carla_import_readback"))
        if scene_output_state == "blocked":
            planned_blocked_count += 1

        observed, join_mode = _pick_observation(planned)
        observed_scene_id = None
        observed_scene_root = None
        observed_identity_id = None
        membership_sample_count = 0
        membership_split = None
        membership_match_status = "missing_observation"
        if join_mode == "trajectory_node":
            trajectory_node_match_count += 1 if observed is not None else 0

        if observed is None:
            missing_observation_count += 1
        else:
            observed_scene_id = str(observed.get("scene_id") or "").strip() or None
            observed_scene_root = str(observed.get("scene_root") or observed.get("scene_dir") or "").strip() or None
            observed_identity_id = str(observed.get("identity_id") or "").strip() or None
            membership_sample_count = _safe_int(observed.get("sample_count"))
            split_value = str(observed.get("split") or "").strip()
            if split_value:
                membership_split = split_value
            else:
                split_names = observed.get("split_names")
                if isinstance(split_names, list):
                    names = [str(x).strip() for x in split_names if str(x).strip()]
                    membership_split = ",".join(sorted(set(names))) if names else None
            if observed_identity_id and planned_identity_id and observed_identity_id == planned_identity_id:
                membership_match_status = "exact_identity_match"
                exact_identity_match_count += 1
            else:
                if observed_identity_id == "default_airsim_drone":
                    membership_match_status = "observed_default_airsim_drone_non_promotion_passthrough"
                else:
                    membership_match_status = "identity_mismatch_non_promotion_passthrough"
                identity_mismatch_count += 1

        join_key = f"{join_mode}:{planned_trajectory_id}:{planned_node_id}:{planned_scene_id}"
        rows.append(
            {
                "join_key": join_key,
                "trajectory_id": planned_trajectory_id or None,
                "node_id": planned_node_id or None,
                "planned_identity_id": planned_identity_id or None,
                "planned_scene_id": planned_scene_id or None,
                "expected_scene_root": expected_scene_root or None,
                "scene_output_state": scene_output_state,
                "requires_ue_carla_import_readback": requires_import_readback,
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

    return {
        "schema_version": "carla_air_scene_membership_alignment_manifest_v1",
        "run_id": run_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "starts_runtime": False,
        "writes_scene_outputs": False,
        "non_promotion": True,
        "full_v1_live_dataset_ready": False,
        "summary": {
            "planned_scene_output_count": len([x for x in planned_rows if isinstance(x, dict)]),
            "observed_scene_membership_count": len([x for x in observed_rows if isinstance(x, dict)]),
            "planned_blocked_count": planned_blocked_count,
            "observed_passthrough_count": observed_passthrough_count,
            "trajectory_node_match_count": trajectory_node_match_count,
            "exact_identity_match_count": exact_identity_match_count,
            "identity_mismatch_count": identity_mismatch_count,
            "missing_observation_count": missing_observation_count,
        },
        "rows": rows,
    }


def _load_jsonl_objects(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        try:
            payload = json.loads(text)
        except Exception:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _build_capture_matrix_alignment_manifest(
    *,
    run_id: str,
    capture_queue_items: list[dict[str, Any]],
    samples: list[dict[str, Any]],
) -> dict[str, Any]:
    observed_by_tnc: dict[tuple[str, str, str], dict[str, Any]] = {}
    observed_sample_count = 0
    mask_gt_available_count = 0
    no_mask_sample_count = 0
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        observed_sample_count += 1
        trajectory_id = str(sample.get("trajectory_id") or "").strip()
        node_id = str(sample.get("node_id") or "").strip()
        camera_id = str(sample.get("camera_id") or "").strip()
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
        identity_id = str(sample.get("identity_id") or "").strip()
        if identity_id:
            rec["identity_ids"].add(identity_id)
        scene_id = str(sample.get("scene_id") or "").strip()
        source = sample.get("source")
        if not scene_id and isinstance(source, dict):
            scene_id = str(source.get("scene_id") or "").strip()
        if scene_id:
            rec["scene_ids"].add(scene_id)
        split_name = str(sample.get("split") or "").strip()
        if split_name:
            rec["split_names"].add(split_name)
        mask_gt = sample.get("mask_gt")
        mask_available = isinstance(mask_gt, dict) and str(mask_gt.get("availability") or "").strip() == "available"
        if mask_available:
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

    planned_rows = [item for item in capture_queue_items if isinstance(item, dict)]
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
        observed_sample_count_for_task = _safe_int(rec.get("sample_count"))
        observed_identity_ids = sorted(str(x) for x in rec.get("identity_ids", set()) if str(x))
        observed_scene_ids = sorted(str(x) for x in rec.get("scene_ids", set()) if str(x))
        observed_split_names = sorted(str(x) for x in rec.get("split_names", set()) if str(x))
        observed_mask_gt_available_count = _safe_int(rec.get("mask_gt_available_count"))
        observed_no_mask_sample_count = _safe_int(rec.get("no_mask_sample_count"))

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
        "run_id": run_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
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


def _build_capture_queue_manifest(
    *,
    run_id: str,
    capture_queue_path: Path,
    capture_queue_items: list[dict[str, Any]],
) -> dict[str, Any]:
    planned_rows = [item for item in capture_queue_items if isinstance(item, dict)]
    blocked_count = 0
    queued_count = 0
    state_counts: dict[str, int] = {}
    block_reason_counts: dict[str, int] = {}
    capture_task_id_order: list[str] = []
    expected_scene_root_order: list[str] = []
    identity_ids: set[str] = set()
    trajectory_ids: set[str] = set()
    node_ids: set[str] = set()
    camera_ids: set[str] = set()
    scene_groups: set[tuple[str, str, str]] = set()

    for item in planned_rows:
        state = str(item.get("state") or "").strip() or "unknown"
        state_counts[state] = state_counts.get(state, 0) + 1
        if state == "blocked":
            blocked_count += 1
        if state == "queued":
            queued_count += 1

        block_reason = str(item.get("block_reason") or "").strip()
        if block_reason:
            block_reason_counts[block_reason] = block_reason_counts.get(block_reason, 0) + 1

        capture_task_id = str(item.get("capture_task_id") or "").strip()
        if capture_task_id:
            capture_task_id_order.append(capture_task_id)

        expected_scene_root = str(item.get("expected_scene_root") or "").strip()
        if expected_scene_root:
            expected_scene_root_order.append(expected_scene_root)

        identity_id = str(item.get("identity_id") or "").strip()
        trajectory_id = str(item.get("trajectory_id") or "").strip()
        node_id = str(item.get("node_id") or "").strip()
        camera_id = str(item.get("camera_id") or "").strip()
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
        "run_id": run_id,
        "capture_queue_item_count": len(planned_rows),
        "blocked_capture_queue_item_count": blocked_count,
        "queued_capture_queue_item_count": queued_count,
        "state_counts": {k: state_counts[k] for k in sorted(state_counts.keys())},
        "block_reason_counts": {k: block_reason_counts[k] for k in sorted(block_reason_counts.keys())},
        "capture_task_id_order_sha256": _canonical_json_sha256(capture_task_id_order),
        "expected_scene_root_order_sha256": _canonical_json_sha256(expected_scene_root_order),
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline orchestrator for CARLA-Air Dataset Generation Pipeline V1.")
    parser.add_argument("--readiness-json", default=str(planner.DEFAULT_READINESS_JSON))
    parser.add_argument("--roster-plan", default=str(planner.DEFAULT_ROSTER_PLAN))
    parser.add_argument("--ue-carla-import-readiness", default=str(planner.DEFAULT_UE_IMPORT_READINESS))
    parser.add_argument("--trajectory-config", default=str(planner.DEFAULT_TRAJECTORY_CONFIG))
    parser.add_argument("--run-id", default="")
    parser.add_argument("--out", default="")
    parser.add_argument("--run-root", default=str(DEFAULT_RUN_ROOT))
    parser.add_argument("--allow-nonlocal-out", action="store_true")
    parser.add_argument("--allow-fail", action="store_true")
    parser.add_argument("--node-id", dest="node_ids", action="append", default=[], help="Repeat or comma-separate.")
    parser.add_argument("--camera-id", dest="camera_ids", action="append", default=[], help="Repeat or comma-separate.")
    parser.add_argument("--identity-id", dest="identity_ids", action="append", default=[], help="Repeat or comma-separate.")
    parser.add_argument("--trajectory-id", dest="trajectory_ids", action="append", default=[], help="Repeat or comma-separate.")
    parser.add_argument("--capture-profile", "--model-label", dest="capture_profile", default="v1_default")
    parser.add_argument("--scene-root", action="append", default=[], help="Explicit scene root (repeatable).")
    parser.add_argument("--no-auto-scene-roots", action="store_true", help="Disable auto-discovery from node scenes.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = _normalize_run_id(args.run_id)
    run_root = _validate_run_root(args.run_root, bool(args.allow_nonlocal_out))
    run_dir = run_root / run_id
    dataset_plan_path = planner._validate_out_path(
        args.out if args.out else str(run_dir / "dataset_plan.json"),
        run_id,
        bool(args.allow_nonlocal_out),
    )

    planner_argv = [
        "--readiness-json",
        str(args.readiness_json),
        "--roster-plan",
        str(args.roster_plan),
        "--ue-carla-import-readiness",
        str(args.ue_carla_import_readiness),
        "--trajectory-config",
        str(args.trajectory_config),
        "--run-id",
        run_id,
        "--out",
        str(dataset_plan_path),
        "--capture-profile",
        str(args.capture_profile),
    ]
    for raw in args.node_ids:
        planner_argv.extend(["--node-id", str(raw)])
    for raw in args.camera_ids:
        planner_argv.extend(["--camera-id", str(raw)])
    for raw in args.identity_ids:
        planner_argv.extend(["--identity-id", str(raw)])
    for raw in args.trajectory_ids:
        planner_argv.extend(["--trajectory-id", str(raw)])
    if args.allow_nonlocal_out:
        planner_argv.append("--allow-nonlocal-out")
    if args.allow_fail:
        planner_argv.append("--allow-fail")

    planner_args = planner.build_parser().parse_args(planner_argv)
    planner_args.run_id = run_id
    plan_payload = planner.build_plan(planner_args)
    planner._write_json(dataset_plan_path, plan_payload)

    selected_filters = plan_payload.get("selected_filters") if isinstance(plan_payload.get("selected_filters"), dict) else {}
    selected_nodes = [str(node) for node in selected_filters.get("node_ids", []) if str(node) in ALLOWED_NODES]
    if not selected_nodes:
        selected_nodes = list(planner.DEFAULT_NODE_IDS)
    selected_trajectories = [str(item) for item in selected_filters.get("trajectory_ids", []) if str(item)]

    scene_roots: list[Path] = []
    seen_roots: set[Path] = set()
    auto_discovery_enabled = not args.no_auto_scene_roots
    auto_discovery_records: list[dict[str, Any]] = []
    explicit_discovery_records: list[dict[str, Any]] = []

    if auto_discovery_enabled:
        for scene_root in _discover_scene_roots(selected_nodes, selected_trajectories):
            if scene_root in seen_roots:
                continue
            seen_roots.add(scene_root)
            scene_roots.append(scene_root)
            fields = _scene_fields_from_root(scene_root)
            auto_discovery_records.append(
                {
                    "path": _path_for_manifest(scene_root),
                    "exists": scene_root.exists(),
                    "node_id": fields["node_id"],
                    "trajectory_id": fields["trajectory_id"],
                    "scene_id": fields["scene_id"],
                    "has_capture_meta": fields["has_capture_meta"],
                    "has_pipeline_contract": fields["has_pipeline_contract"],
                    "source": "auto_discovered",
                    "accepted": True,
                    "reject_reason": None,
                }
            )

    for raw in args.scene_root:
        path = _repo_or_abs(raw).resolve()
        fields = _scene_fields_from_root(path)
        exists = path.exists()
        reject_reason = None
        accepted = True
        if _forbidden_scene_root(path):
            reject_reason = "forbidden_root_token_weak_or_proxy"
            accepted = False
        elif path in seen_roots:
            reject_reason = "duplicate_scene_root"
            accepted = False
        elif not exists:
            reject_reason = "scene_root_missing"
            accepted = False
        explicit_discovery_records.append(
            {
                "path": _path_for_manifest(path),
                "exists": exists,
                "node_id": fields["node_id"],
                "trajectory_id": fields["trajectory_id"],
                "scene_id": fields["scene_id"],
                "has_capture_meta": fields["has_capture_meta"],
                "has_pipeline_contract": fields["has_pipeline_contract"],
                "source": "explicit",
                "accepted": accepted,
                "reject_reason": reject_reason,
            }
        )
        if not accepted:
            continue
        seen_roots.add(path)
        scene_roots.append(path)

    scene_discovery_manifest = {
        "schema_version": "carla_air_scene_discovery_manifest_v1",
        "run_id": run_id,
        "auto_discovery_enabled": auto_discovery_enabled,
        "selected_nodes": selected_nodes,
        "selected_trajectories": selected_trajectories,
        "explicit_scene_root_count": len(explicit_discovery_records),
        "auto_scene_root_count": len(auto_discovery_records),
        "scene_root_count": len(scene_roots),
        "scene_roots": auto_discovery_records + explicit_discovery_records,
    }
    planner._write_json(run_dir / "scene_discovery_manifest.json", scene_discovery_manifest)

    index_script = REPO_ROOT / "tools/carla_air/build_dataset_training_index_v1.py"
    cmd = [sys.executable, str(index_script), "--dataset-plan", str(dataset_plan_path), "--run-root", str(run_root)]
    for scene_root in scene_roots:
        cmd.extend(["--scene-root", str(scene_root)])
    if args.allow_nonlocal_out:
        cmd.append("--allow-nonlocal-out")
    if args.allow_fail:
        cmd.append("--allow-fail")

    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
    index_result = _extract_last_json_object(proc.stdout)
    sample_count = int(index_result.get("sample_count") or 0)

    plan_counts = plan_payload.get("counts") if isinstance(plan_payload.get("counts"), dict) else {}
    capture_tasks = plan_payload.get("capture_tasks") if isinstance(plan_payload.get("capture_tasks"), list) else []
    capture_task_ids_sample: list[str] = []
    capture_queue_items: list[dict[str, Any]] = []
    scene_groups: dict[tuple[str, str, str], dict[str, Any]] = {}
    for task in capture_tasks:
        if not isinstance(task, dict):
            continue
        task_id = str(task.get("capture_task_id") or task.get("task_id") or task.get("id") or "").strip()
        if not task_id:
            continue
        capture_task_ids_sample.append(task_id)
        if len(capture_task_ids_sample) >= 20:
            break
    for task in capture_tasks:
        if not isinstance(task, dict):
            continue
        capture_task_id = str(task.get("capture_task_id") or task.get("task_id") or task.get("id") or "").strip()
        if not capture_task_id:
            continue
        identity_id = str(task.get("identity_id") or "").strip()
        trajectory_id = str(task.get("trajectory_id") or "").strip()
        node_id = str(task.get("node_id") or "").strip()
        camera_id = str(task.get("camera_id") or "").strip()
        capture_profile = str(task.get("capture_profile") or args.capture_profile).strip()
        capture_allowed_now = bool(task.get("capture_allowed_now"))
        model_label = str(task.get("model_label") or identity_id).strip() or identity_id
        switch_method = str(task.get("switch_method") or "ue_carla_import_readback").strip() or "ue_carla_import_readback"
        identity_model_profile_id = str(task.get("identity_model_profile_id") or identity_id).strip() or identity_id
        requires_ue_carla_import_readback = bool(task.get("requires_ue_carla_import_readback"))
        identity_model_profile = task.get("identity_model_profile")
        if not isinstance(identity_model_profile, dict):
            identity_model_profile = {
                "identity_model_profile_id": identity_model_profile_id,
                "identity_id": identity_id,
                "model_label": model_label,
                "capture_profile": capture_profile,
                "switch_method": switch_method,
                "requires_ue_carla_import_readback": requires_ue_carla_import_readback,
                "ready_for_capture": capture_allowed_now,
                "non_promotion": True,
            }
        block_reason = str(task.get("block_reason") or "").strip()
        expected_scene_id = str(task.get("expected_scene_id") or f"{identity_id}.{trajectory_id}.{node_id}").strip()
        expected_scene_root = str(task.get("expected_scene_root") or "").strip()
        state = "queued" if capture_allowed_now else "blocked"
        readiness = _scene_output_readiness(capture_allowed_now, block_reason)
        queue_item = {
            "schema_version": "carla_air_capture_queue_item_v1",
            "run_id": run_id,
            "capture_task_id": capture_task_id,
            "identity_id": identity_id,
            "trajectory_id": trajectory_id,
            "node_id": node_id,
            "camera_id": camera_id,
            "capture_profile": capture_profile,
            "model_label": model_label,
            "switch_method": switch_method,
            "identity_model_profile_id": identity_model_profile_id,
            "identity_model_profile": identity_model_profile,
            "requires_ue_carla_import_readback": requires_ue_carla_import_readback,
            "expected_scene_id": expected_scene_id,
            "expected_scene_root": expected_scene_root,
            "state": state,
            "block_reason": block_reason,
            "capture_allowed_now": capture_allowed_now,
            "scene_output_readiness": readiness,
            "starts_runtime": False,
            "writes_scene_outputs": False,
            "evidence_paths": [],
        }
        capture_queue_items.append(queue_item)

        group_key = (identity_id, trajectory_id, node_id)
        group = scene_groups.get(group_key)
        if group is None:
            group = {
                "scene_id": expected_scene_id,
                "expected_scene_root": expected_scene_root,
                "identity_id": identity_id,
                "trajectory_id": trajectory_id,
                "node_id": node_id,
                "camera_ids": [],
                "capture_task_ids": [],
                "states": [],
                "block_reasons": [],
                "readiness_items": [],
                "model_labels": [],
                "switch_methods": [],
                "identity_model_profile_ids": [],
                "identity_model_profiles": [],
                "requires_ue_carla_import_readback_flags": [],
            }
            scene_groups[group_key] = group
        group["camera_ids"].append(camera_id)
        group["capture_task_ids"].append(capture_task_id)
        group["states"].append(state)
        if block_reason:
            group["block_reasons"].append(block_reason)
        group["readiness_items"].append(readiness)
        group["model_labels"].append(model_label)
        group["switch_methods"].append(switch_method)
        group["identity_model_profile_ids"].append(identity_model_profile_id)
        group["identity_model_profiles"].append(identity_model_profile)
        group["requires_ue_carla_import_readback_flags"].append(requires_ue_carla_import_readback)

    scene_outputs: list[dict[str, Any]] = []
    blocked_scene_output_count = 0
    for group_key in sorted(scene_groups.keys()):
        group = scene_groups[group_key]
        unique_cameras = sorted({str(x) for x in group["camera_ids"] if str(x)})
        unique_capture_task_ids = sorted({str(x) for x in group["capture_task_ids"] if str(x)})
        unique_block_reasons = sorted({str(x) for x in group["block_reasons"] if str(x)})
        unique_model_labels = sorted({str(x) for x in group["model_labels"] if str(x)})
        unique_switch_methods = sorted({str(x) for x in group["switch_methods"] if str(x)})
        unique_identity_model_profile_ids = sorted({str(x) for x in group["identity_model_profile_ids"] if str(x)})
        unique_requires_import_readback = sorted({bool(x) for x in group["requires_ue_carla_import_readback_flags"]})
        profile_map: dict[str, dict[str, Any]] = {}
        for profile in group["identity_model_profiles"]:
            if not isinstance(profile, dict):
                continue
            profile_id = str(profile.get("identity_model_profile_id") or "").strip()
            if profile_id and profile_id not in profile_map:
                profile_map[profile_id] = profile
        group_states = [str(x) for x in group["states"]]
        scene_state = "blocked" if group_states and all(s == "blocked" for s in group_states) else "queued"
        scene_readiness_status = "blocked" if scene_state == "blocked" else "ready_for_execution"
        scene_readiness = {
            "status": scene_readiness_status,
            "blocked": scene_state == "blocked",
            "blocked_reasons": unique_block_reasons if scene_state == "blocked" else [],
            "evidence_ready": scene_state != "blocked",
        }
        if scene_state == "blocked":
            blocked_scene_output_count += 1
        scene_outputs.append(
            {
                "scene_id": str(group["scene_id"]),
                "expected_scene_root": str(group["expected_scene_root"]),
                "identity_id": str(group["identity_id"]),
                "trajectory_id": str(group["trajectory_id"]),
                "node_id": str(group["node_id"]),
                "camera_ids": unique_cameras,
                "capture_task_ids": unique_capture_task_ids,
                "state": scene_state,
                "block_reasons": unique_block_reasons,
                "model_label": unique_model_labels[0] if unique_model_labels else "",
                "model_labels": unique_model_labels,
                "switch_method": unique_switch_methods[0] if unique_switch_methods else "",
                "switch_methods": unique_switch_methods,
                "identity_model_profile_id": unique_identity_model_profile_ids[0] if unique_identity_model_profile_ids else "",
                "identity_model_profile_ids": unique_identity_model_profile_ids,
                "identity_model_profiles": [profile_map[k] for k in sorted(profile_map.keys())],
                "requires_ue_carla_import_readback": (True in unique_requires_import_readback),
                "scene_output_readiness": scene_readiness,
                "expected_modalities": ["rgb", "depth", "semantic", "instance", "pose", "calib"],
                "mask_gt_expected": False,
            }
        )

    capture_queue_path = run_dir / "capture_queue.jsonl"
    _write_jsonl(capture_queue_path, capture_queue_items)
    capture_queue_manifest_path = run_dir / "capture_queue_manifest.json"
    capture_queue_manifest = _build_capture_queue_manifest(
        run_id=run_id,
        capture_queue_path=capture_queue_path,
        capture_queue_items=capture_queue_items,
    )
    planner._write_json(capture_queue_manifest_path, capture_queue_manifest)
    scene_output_manifest = {
        "schema_version": "carla_air_scene_output_manifest_v1",
        "run_id": run_id,
        "starts_runtime": False,
        "writes_scene_outputs": False,
        "scene_output_readiness_schema_version": "carla_air_scene_output_readiness_v1",
        "scene_outputs": scene_outputs,
    }
    planner._write_json(run_dir / "scene_output_manifest.json", scene_output_manifest)
    scene_membership_manifest_payload = _safe_json_dict(run_dir / "scene_membership_manifest.json")
    scene_membership_alignment_manifest = _build_scene_membership_alignment_manifest(
        run_id=run_id,
        scene_output_manifest=scene_output_manifest,
        scene_membership_manifest=scene_membership_manifest_payload,
    )
    planner._write_json(run_dir / "scene_membership_alignment_manifest.json", scene_membership_alignment_manifest)
    dataset_samples_rows = _load_jsonl_objects(run_dir / "dataset_samples.jsonl")
    capture_matrix_alignment_manifest = _build_capture_matrix_alignment_manifest(
        run_id=run_id,
        capture_queue_items=capture_queue_items,
        samples=dataset_samples_rows,
    )
    planner._write_json(run_dir / "capture_matrix_alignment_manifest.json", capture_matrix_alignment_manifest)

    blocked_capture_queue_item_count = sum(1 for item in capture_queue_items if item.get("state") == "blocked")
    capture_queue_item_count = len(capture_queue_items)
    scene_output_count = len(scene_outputs)

    plan_identity_model_profiles = [
        item
        for item in (plan_payload.get("identity_model_profiles") if isinstance(plan_payload.get("identity_model_profiles"), list) else [])
        if isinstance(item, dict)
    ]
    dataset_manifest_payload = _safe_json_dict(run_dir / "dataset_manifest.json")
    dataset_index_manifest_payload = _safe_json_dict(run_dir / "dataset_index_manifest.json")
    dataset_run_contract_summary = _extract_dataset_run_contract_summary(
        dataset_manifest_payload=dataset_manifest_payload,
        dataset_index_manifest_payload=dataset_index_manifest_payload,
    )
    identity_switch_contract = (
        dataset_manifest_payload.get("identity_model_switch_contract")
        if isinstance(dataset_manifest_payload.get("identity_model_switch_contract"), dict)
        else {}
    )
    identity_model_switch_manifest_path = run_dir / "identity_model_switch_manifest.json"
    identity_model_switch_manifest = _safe_json_dict(identity_model_switch_manifest_path)
    if not identity_model_switch_manifest:
        identity_model_switch_manifest = _build_identity_model_switch_manifest_fallback(
            run_id=run_id,
            plan_payload=plan_payload,
            capture_queue_item_count=capture_queue_item_count,
            blocked_capture_queue_item_count=blocked_capture_queue_item_count,
            identity_switch_contract=identity_switch_contract,
        )
        planner._write_json(identity_model_switch_manifest_path, identity_model_switch_manifest)

    contract_counts = {
        "identity_count": _safe_int(plan_counts.get("identity_count")),
        "trajectory_count": _safe_int(plan_counts.get("trajectory_count")),
        "camera_layout_count": _safe_int(plan_counts.get("camera_layout_count")),
        "matrix_count": _safe_int(plan_counts.get("matrix_count")),
        "capture_task_count": _safe_int(plan_counts.get("capture_task_count")) or len(capture_tasks),
        "capture_queue_item_count": capture_queue_item_count,
        "scene_output_count": scene_output_count,
        "blocked_capture_queue_item_count": blocked_capture_queue_item_count,
        "blocked_scene_output_count": blocked_scene_output_count,
        "scene_root_count": len(scene_roots),
    }

    artifact_paths = {
        "dataset_plan_json": str((run_dir / "dataset_plan.json").resolve()),
        "dataset_manifest_json": str((run_dir / "dataset_manifest.json").resolve()),
        "dataset_index_manifest_json": str((run_dir / "dataset_index_manifest.json").resolve()),
        "scene_sample_index_manifest_json": str((run_dir / "scene_sample_index_manifest.json").resolve()),
        "scene_membership_manifest_json": str((run_dir / "scene_membership_manifest.json").resolve()),
        "scene_membership_alignment_manifest_json": str(
            (run_dir / "scene_membership_alignment_manifest.json").resolve()
        ),
        "capture_matrix_alignment_manifest_json": str(
            (run_dir / "capture_matrix_alignment_manifest.json").resolve()
        ),
        "dataset_samples_jsonl": str((run_dir / "dataset_samples.jsonl").resolve()),
        "dataset_splits_json": str((run_dir / "dataset_splits.json").resolve()),
        "deployment_episodes_json": str((run_dir / "deployment_episodes.json").resolve()),
        "deployment_episode_visibility_manifest_json": str(
            (run_dir / "deployment_episode_visibility_manifest.json").resolve()
        ),
        "capture_queue_jsonl": str(capture_queue_path.resolve()),
        "capture_queue_manifest_json": str(capture_queue_manifest_path.resolve()),
        "identity_model_switch_manifest_json": str((run_dir / "identity_model_switch_manifest.json").resolve()),
        "existing_scene_index_bridge_manifest_json": str(
            (run_dir / "existing_scene_index_bridge_manifest.json").resolve()
        ),
        "sidecar_quality_manifest_json": str((run_dir / "sidecar_quality_manifest.json").resolve()),
        "no_mask_non_promotion_manifest_json": str((run_dir / "no_mask_non_promotion_manifest.json").resolve()),
        "sample_schema_coverage_manifest_json": str((run_dir / "sample_schema_coverage_manifest.json").resolve()),
        "scene_discovery_manifest_json": str((run_dir / "scene_discovery_manifest.json").resolve()),
        "scene_output_manifest_json": str((run_dir / "scene_output_manifest.json").resolve()),
        "dataset_gap_manifest_json": str((run_dir / "dataset_gap_manifest.json").resolve()),
        "dataset_run_closure_manifest_json": str((run_dir / "dataset_run_closure_manifest.json").resolve()),
        "batch_run_manifest_json": str((run_dir / "batch_run_manifest.json").resolve()),
        "run_summary_json": str((run_dir / "run_summary.json").resolve()),
        "run_contract_json": str((run_dir / "run_contract.json").resolve()),
        "artifact_manifest_json": str((run_dir / "artifact_manifest.json").resolve()),
    }
    self_artifact_key = "artifact_manifest_json"
    contract_artifact_count_including_self = len(artifact_paths)

    batch_run_manifest = {
        "schema_version": "carla_air_batch_run_manifest_v1",
        "run_id": run_id,
        "run_dir": str(run_dir.resolve()),
        "batch_id": run_id,
        "batch_root": str(run_root.resolve()),
        "dataset_run_role": "offline_orchestrated_index_run",
        "starts_runtime": False,
        "writes_scene_outputs": False,
        "ue_carla_import_externalized": True,
        "full_v1_live_dataset_ready": False,
        "non_promotion": True,
        "selected_filters": selected_filters,
        "capture_profile": str(args.capture_profile),
        "artifact_paths": artifact_paths,
        "counts": contract_counts,
        "dataset_run_contract_summary": dataset_run_contract_summary,
        "child_runs": [
            {
                "run_id": run_id,
                "run_dir": str(run_dir.resolve()),
                "status": "index_completed" if proc.returncode == 0 else "index_failed",
                "sample_count": sample_count,
                "scene_root_count": len(scene_roots),
            }
        ],
        "run_order": [run_id],
    }
    planner._write_json(run_dir / "batch_run_manifest.json", batch_run_manifest)

    run_contract = {
        "schema_version": "carla_air_dataset_generation_run_contract_v1",
        "run_id": run_id,
        "run_dir": str(run_dir.resolve()),
        "starts_runtime": False,
        "writes_scene_outputs": False,
        "capture_profile": str(args.capture_profile),
        "selected_filters": selected_filters,
        "counts": contract_counts,
        "dataset_run_contract_summary": dataset_run_contract_summary,
        "capture_task_ids_sample": capture_task_ids_sample,
        "scene_roots": [str(path) for path in scene_roots],
        "artifacts": artifact_paths,
        "artifact_accounting": {
            "self_artifact_key": self_artifact_key,
            "contract_artifact_count_including_self": contract_artifact_count_including_self,
            "excluded_self_reference_from_hashed_entries": True,
        },
    }
    planner._write_json(run_dir / "run_contract.json", run_contract)

    summary = {
        "schema_version": "carla_air_dataset_generation_orchestrator_run_summary_v1",
        "run_id": run_id,
        "run_dir": str(run_dir.resolve()),
        "planner_ok": bool(plan_payload.get("ok")),
        "planner_allow_fail_used": bool(args.allow_fail),
        "scene_root_count": len(scene_roots),
        "capture_queue_item_count": capture_queue_item_count,
        "scene_output_count": scene_output_count,
        "blocked_capture_queue_item_count": blocked_capture_queue_item_count,
        "blocked_scene_output_count": blocked_scene_output_count,
        "scene_roots": [str(path) for path in scene_roots],
        "index_returncode": int(proc.returncode),
        "sample_count": sample_count,
        "artifacts": artifact_paths,
    }
    planner._write_json(run_dir / "run_summary.json", summary)

    closure_manifest_path = run_dir / "dataset_run_closure_manifest.json"
    artifact_manifest_entry_count_excluding_self = len(
        [key for key in artifact_paths.keys() if key != self_artifact_key]
    )
    scene_sample_index_manifest_payload = _safe_json_dict(run_dir / "scene_sample_index_manifest.json")
    closure_manifest_payload = _build_dataset_run_closure_manifest(
        run_id=run_id,
        run_dir=run_dir,
        sample_count=sample_count,
        capture_queue_items=capture_queue_items,
        scene_discovery_manifest_payload=_safe_json_dict(run_dir / "scene_discovery_manifest.json"),
        dataset_manifest_payload=dataset_manifest_payload,
        dataset_index_manifest_payload=dataset_index_manifest_payload,
        scene_sample_index_manifest_payload=scene_sample_index_manifest_payload,
        scene_membership_manifest_payload=scene_membership_manifest_payload,
        capture_matrix_alignment_manifest_payload=capture_matrix_alignment_manifest,
        dataset_run_contract_summary=dataset_run_contract_summary,
        contract_artifact_count_including_self=contract_artifact_count_including_self,
        artifact_manifest_entry_count_excluding_self=artifact_manifest_entry_count_excluding_self,
    )
    planner._write_json(closure_manifest_path, closure_manifest_payload)

    artifact_manifest_path = run_dir / "artifact_manifest.json"
    artifact_entries: list[dict[str, Any]] = []
    for key in sorted(artifact_paths.keys()):
        if key == self_artifact_key:
            continue
        artifact_entries.append(_artifact_metadata_entry(key, Path(artifact_paths[key]), include_hash=True))
    artifact_manifest_entry_count_excluding_self = len(artifact_entries)
    artifact_manifest = {
        "schema_version": "carla_air_dataset_run_artifact_manifest_v1",
        "run_id": run_id,
        "run_dir": str(run_dir.resolve()),
        "starts_runtime": False,
        "writes_scene_outputs": False,
        "artifact_count": artifact_manifest_entry_count_excluding_self,
        "self_path": str(artifact_manifest_path.resolve()),
        "artifact_accounting": {
            "self_artifact_key": self_artifact_key,
            "artifact_manifest_entry_count_excluding_self": artifact_manifest_entry_count_excluding_self,
            "contract_artifact_count_including_self": contract_artifact_count_including_self,
            "excluded_self_reference_from_hashed_entries": True,
        },
        "artifacts": {entry["key"]: entry for entry in artifact_entries},
    }
    planner._write_json(artifact_manifest_path, artifact_manifest)

    required_present = all(Path(p).is_file() for p in artifact_paths.values())
    ok = bool(required_present and (proc.returncode == 0 or args.allow_fail))
    result = {
        "ok": ok,
        "run_id": run_id,
        "run_dir": str(run_dir.resolve()),
        "scene_root_count": len(scene_roots),
        "sample_count": sample_count,
        "artifacts": artifact_paths,
    }
    if not required_present:
        result["error"] = "required_artifacts_missing"
    if proc.returncode != 0 and not args.allow_fail:
        result["index_stderr"] = proc.stderr.strip()

    print(json.dumps(result, ensure_ascii=True, indent=2))
    if ok:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
