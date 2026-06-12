#!/usr/bin/env python3
"""Plan-only CARLA-Air dataset generation pipeline V1 preflight.

This script is read-only with respect to runtime, assets, and scene outputs.
It builds a dataset planning matrix (identity x trajectory x camera_layout)
from existing readiness/roster/import-readiness evidence when available.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
LOCAL_ROOT = REPO_ROOT / "local"
SCHEMA_VERSION = "carla_air_dataset_generation_plan_v1"

DEFAULT_READINESS_JSON = REPO_ROOT / "local/carla_air/tmp/aircraft_identity_readiness_private_policy_final_20260530.json"
DEFAULT_ROSTER_PLAN = REPO_ROOT / "local/carla_air/tmp/aircraft_identity_roster_plan_private_policy_final_20260530.json"
DEFAULT_UE_IMPORT_READINESS = (
    REPO_ROOT / "local/carla_air/tmp/aircraft_identity_ue_carla_import_readiness_procedural_canard_poc_20260529.json"
)
DEFAULT_TRAJECTORY_CONFIG = REPO_ROOT / "configs/carla_air/trajectories/town10hd_coverage_first_v1.json"

DEFAULT_TOWN = "Town10HD"
DEFAULT_NODE_IDS = ["node01", "node02", "node03", "node04", "node05"]
DEFAULT_CAMERA_IDS = ["cam0", "cam1", "cam2"]
EXPECTED_PRIVATE_LOCAL_IDENTITY_COUNT = 6
DEFAULT_SPLIT_NAMES = ["train", "val_in_domain", "test_cross_layout"]


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _repo_or_abs(raw: str | Path) -> Path:
    path = Path(str(raw))
    return path if path.is_absolute() else REPO_ROOT / path


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_token(raw: Any) -> str:
    text = str(raw or "")
    return re.sub(r"[^A-Za-z0-9_.-]", "_", text)


def _expected_scene_id(identity_id: str, trajectory_id: str, node_id: str) -> str:
    return ".".join([_safe_token(identity_id), _safe_token(trajectory_id), _safe_token(node_id)])


def _expected_scene_root(identity_id: str, trajectory_id: str, node_id: str) -> str:
    scene_id = _expected_scene_id(identity_id, trajectory_id, node_id)
    return f"data/carla_air/nodes/{node_id}/scenes/{scene_id}"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


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


def _load_json(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    evidence: dict[str, Any] = {"path": str(path), "exists": path.is_file(), "readable": False}
    if not path.is_file():
        evidence["error"] = "missing input JSON"
        return {}, evidence
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        evidence["error"] = repr(exc)
        return {}, evidence
    if not isinstance(payload, dict):
        evidence["error"] = "JSON root is not an object"
        return {}, evidence
    evidence["readable"] = True
    evidence["sha256"] = _sha256_file(path)
    evidence["schema_version"] = payload.get("schema_version")
    evidence["ok"] = payload.get("ok")
    return payload, evidence


def _validate_out_path(raw_out: str | None, run_id: str, allow_nonlocal_out: bool) -> Path:
    if raw_out:
        out = _repo_or_abs(raw_out)
    else:
        out = REPO_ROOT / f"local/carla_air/dataset_runs/{run_id}/dataset_plan.json"
    if out.suffix.lower() != ".json":
        raise SystemExit("--out must point to a .json file")
    if not allow_nonlocal_out and not _is_under(out, LOCAL_ROOT):
        raise SystemExit("--out must stay under repository local/ unless --allow-nonlocal-out is set")
    return out


def _normalize_run_id(raw: str | None) -> str:
    if raw:
        return str(raw).strip()
    return f"v1_plan_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"


def _identity_from_readiness(readiness: dict[str, Any]) -> dict[str, dict[str, Any]]:
    identities: dict[str, dict[str, Any]] = {}
    normalized_inventory = _as_dict(readiness.get("normalized_inventory"))
    for item in _as_list(normalized_inventory.get("identities")):
        rec = _as_dict(item)
        identity_id = str(rec.get("identity_id") or "").strip()
        if not identity_id:
            continue
        identities[identity_id] = {
            "identity_id": identity_id,
            "technical_ready": rec.get("technical_ready") is True,
            "local_poc_eligible": rec.get("local_poc_eligible") is True,
            "private_local_benchmark_eligible": rec.get("private_local_benchmark_eligible") is True,
            "private_benchmark_eligible": rec.get("private_benchmark_eligible") is True,
            "benchmark_eligible": rec.get("benchmark_eligible") is True,
            "blockers": list(rec.get("blockers") or []),
            "normalized_dir": rec.get("normalized_dir"),
        }
    return identities


def _merge_roster_identities(identities: dict[str, dict[str, Any]], roster: dict[str, Any]) -> None:
    roster_groups = (
        ("local_poc_roster", "local_technical_poc"),
        ("private_benchmark_roster", "private_local_benchmark"),
        ("benchmark_roster", "formal_benchmark"),
    )
    for key, roster_kind in roster_groups:
        for item in _as_list(roster.get(key)):
            rec = _as_dict(item)
            identity_id = str(rec.get("identity_id") or "").strip()
            if not identity_id:
                continue
            base = identities.get(identity_id, {"identity_id": identity_id, "blockers": []})
            base["roster_kind"] = roster_kind
            for flag_key in (
                "technical_ready",
                "local_poc_eligible",
                "private_local_benchmark_eligible",
                "private_benchmark_eligible",
                "benchmark_eligible",
            ):
                if flag_key in rec:
                    base[flag_key] = rec.get(flag_key) is True
            if "blockers" in rec and not base.get("blockers"):
                base["blockers"] = list(rec.get("blockers") or [])
            if "normalized_dir" in rec and not base.get("normalized_dir"):
                base["normalized_dir"] = rec.get("normalized_dir")
            identities[identity_id] = base


def _import_readiness_map(import_readiness: dict[str, Any]) -> dict[str, dict[str, Any]]:
    by_identity: dict[str, dict[str, Any]] = {}
    for item in _as_list(import_readiness.get("identity_imports")):
        rec = _as_dict(item)
        identity_id = str(rec.get("identity_id") or "").strip()
        if not identity_id:
            continue
        by_identity[identity_id] = rec
    return by_identity


def _load_trajectory_items(trajectory_cfg: dict[str, Any], fallback_town: str) -> tuple[str, list[dict[str, Any]]]:
    town = str(trajectory_cfg.get("town") or trajectory_cfg.get("map") or fallback_town or DEFAULT_TOWN)
    trajectories_raw = _as_list(trajectory_cfg.get("trajectories"))
    trajectories: list[dict[str, Any]] = []
    for idx, item in enumerate(trajectories_raw):
        rec = _as_dict(item)
        tid = str(rec.get("trajectory_id") or rec.get("id") or f"traj_{idx + 1:02d}")
        trajectories.append(
            {
                "trajectory_id": tid,
                "town": str(rec.get("town") or town),
                "source": str(rec.get("source") or trajectory_cfg.get("schema_version") or "trajectory_config"),
                "record": rec,
            }
        )
    if not trajectories:
        trajectories.append(
            {
                "trajectory_id": "town10hd_default_v1",
                "town": town,
                "source": "default_fallback",
                "record": {},
            }
        )
    return town, trajectories


def _camera_layout_entries(node_ids: list[str], camera_ids: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for node_id in node_ids:
        rows.append(
            {
                "camera_layout_id": f"{node_id}_tri_cam_parallel_v1",
                "node_id": node_id,
                "camera_ids": list(camera_ids),
                "cameras": [{"camera_id": camera_id, "role": "fixed_ground_to_air"} for camera_id in camera_ids],
                "camera_count": len(camera_ids),
                "modality": "rgb_instance_semantic_depth",
                "status": "planned",
            }
        )
    return rows


def _build_dataset_splits(node_ids: list[str]) -> dict[str, Any]:
    return {
        "schema_version": "carla_air_dataset_splits_plan_v1",
        "strategy": "deployment_oriented_node_layout_v1",
        "names": DEFAULT_SPLIT_NAMES,
        "plan_only": True,
        "node_groups": {
            "train": node_ids[:3],
            "val_in_domain": node_ids[3:4] or node_ids[:1],
            "test_cross_layout": node_ids[4:5] or node_ids[-1:],
        },
        "policy": {
            "not_random_frame_split": True,
            "hold_out_camera_layouts_for_cross_layout_test": True,
            "future_two_camera_node_deployment_target": True,
        },
    }


def _build_deployment_episodes(node_ids: list[str], camera_ids: list[str], trajectories: list[dict[str, Any]]) -> list[dict[str, Any]]:
    episodes: list[dict[str, Any]] = []
    trajectory_ids = [str(item.get("trajectory_id") or "") for item in trajectories if str(item.get("trajectory_id") or "")]
    for index in range(max(0, len(node_ids) - 1)):
        pair = node_ids[index : index + 2]
        episodes.append(
            {
                "episode_id": f"two_node_pair_{pair[0]}_{pair[1]}",
                "mode": "two_camera_node_deployment_plan",
                "node_ids": pair,
                "camera_ids_by_node": {node_id: list(camera_ids) for node_id in pair},
                "trajectory_ids": trajectory_ids,
                "split_hint": "test_cross_layout" if index >= max(0, len(node_ids) - 2) else "train",
                "plan_only": True,
            }
        )
    return episodes


def _identity_enabled(identity: dict[str, Any]) -> bool:
    return bool(
        identity.get("technical_ready") is True
        and (
            identity.get("local_poc_eligible") is True
            or identity.get("private_local_benchmark_eligible") is True
            or identity.get("private_benchmark_eligible") is True
        )
    )


def _must_exist_or_die(kind: str, requested: list[str], available: list[str]) -> None:
    missing = [x for x in requested if x not in available]
    if not missing:
        return
    available_hint = ", ".join(available) if available else "(none)"
    missing_hint = ", ".join(missing)
    raise SystemExit(f"Requested {kind} not found: {missing_hint}. Available {kind}: {available_hint}")


def _nullable_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text if text else None


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    readiness_json = _repo_or_abs(args.readiness_json)
    roster_json = _repo_or_abs(args.roster_plan)
    import_json = _repo_or_abs(args.ue_carla_import_readiness)
    trajectory_json = _repo_or_abs(args.trajectory_config)

    readiness, readiness_ev = _load_json(readiness_json)
    roster, roster_ev = _load_json(roster_json)
    import_ready, import_ev = _load_json(import_json)
    trajectory_cfg, trajectory_ev = _load_json(trajectory_json)

    identities = _identity_from_readiness(readiness)
    _merge_roster_identities(identities, roster)
    import_map = _import_readiness_map(import_ready)

    town, trajectories_all = _load_trajectory_items(trajectory_cfg, DEFAULT_TOWN)

    selected_node_ids = _parse_id_filters(getattr(args, "node_ids", [])) or list(DEFAULT_NODE_IDS)
    selected_camera_ids = _parse_id_filters(getattr(args, "camera_ids", [])) or list(DEFAULT_CAMERA_IDS)
    selected_identity_ids = _parse_id_filters(getattr(args, "identity_ids", []))
    selected_trajectory_ids = _parse_id_filters(getattr(args, "trajectory_ids", []))
    capture_profile = str(getattr(args, "capture_profile", "") or "v1_default")

    _must_exist_or_die("node_ids", selected_node_ids, list(DEFAULT_NODE_IDS))
    _must_exist_or_die("camera_ids", selected_camera_ids, list(DEFAULT_CAMERA_IDS))

    all_identity_ids = sorted(identities.keys())
    if selected_identity_ids:
        _must_exist_or_die("identity_ids", selected_identity_ids, all_identity_ids)
        identity_ids = list(selected_identity_ids)
    else:
        identity_ids = all_identity_ids

    trajectory_ids_all = [str(item.get("trajectory_id") or "") for item in trajectories_all if str(item.get("trajectory_id") or "")]
    if selected_trajectory_ids:
        _must_exist_or_die("trajectory_ids", selected_trajectory_ids, trajectory_ids_all)
        requested = set(selected_trajectory_ids)
        trajectories = [item for item in trajectories_all if str(item.get("trajectory_id") or "") in requested]
    else:
        trajectories = trajectories_all

    camera_layouts = _camera_layout_entries(selected_node_ids, selected_camera_ids)
    dataset_splits = _build_dataset_splits(selected_node_ids)
    deployment_episodes = _build_deployment_episodes(selected_node_ids, selected_camera_ids, trajectories)

    identity_rows: list[dict[str, Any]] = []
    identity_model_profiles: list[dict[str, Any]] = []
    matrix: list[dict[str, Any]] = []
    capture_tasks: list[dict[str, Any]] = []
    blockers: list[dict[str, Any]] = []
    ue_import_missing = 0

    for identity_id in identity_ids:
        info = identities[identity_id]
        import_rec = _as_dict(import_map.get(identity_id))
        ue_verified = import_rec.get("ue_carla_import_verified") is True
        import_blockers = [str(x) for x in _as_list(import_rec.get("blockers"))]
        ready_for_capture = _identity_enabled(info) and ue_verified
        model_label = str(import_rec.get("model_label") or identity_id).strip() or identity_id
        profile_id = identity_id
        switch_method = "ue_carla_import_readback"
        block_reason = "" if ready_for_capture else "await_ue_carla_import_readback_evidence"
        requires_ue_carla_import_readback = not ready_for_capture
        if not ue_verified:
            ue_import_missing += 1
            blockers.append(
                {
                    "code": "missing_ue_carla_import_readback_evidence",
                    "identity_id": identity_id,
                    "detail": "Identity lacks ue_carla_import_verified=true evidence; keep in plan-only blocked state.",
                    "current_import_blockers": import_blockers,
                }
            )
        identity_rows.append(
            {
                "identity_id": identity_id,
                "roster_kind": info.get("roster_kind", "unclassified"),
                "technical_ready": info.get("technical_ready") is True,
                "local_poc_eligible": info.get("local_poc_eligible") is True,
                "private_local_benchmark_eligible": info.get("private_local_benchmark_eligible") is True,
                "private_benchmark_eligible": info.get("private_benchmark_eligible") is True,
                "benchmark_eligible": info.get("benchmark_eligible") is True,
                "ue_carla_import_verified": ue_verified,
                "identity_blockers": list(info.get("blockers") or []),
                "import_blockers": import_blockers,
                "ready_for_capture": ready_for_capture,
            }
        )
        normalized_dir = info.get("normalized_dir")
        identity_model_profiles.append(
            {
                "identity_model_profile_id": profile_id,
                "identity_id": identity_id,
                "model_label": model_label,
                "capture_profile": capture_profile,
                "normalized_dir": normalized_dir,
                "switch_method": switch_method,
                "requires_ue_carla_import_readback": requires_ue_carla_import_readback,
                "ue_carla_import_verified": ue_verified,
                "ue_import_readback_status": "verified" if ue_verified else "missing_or_unverified",
                "ue_import_readback_evidence": _nullable_text(import_rec.get("ue_import_readback_evidence")),
                "carla_import_status": _nullable_text(import_rec.get("carla_import_status")),
                "carla_readback_status": _nullable_text(import_rec.get("carla_readback_status")),
                "blueprint_evidence": _nullable_text(import_rec.get("blueprint_evidence")),
                "asset_evidence": _nullable_text(import_rec.get("asset_evidence")),
                "spawn_evidence": _nullable_text(import_rec.get("spawn_evidence")),
                "switch_evidence": _nullable_text(import_rec.get("switch_evidence")),
                "ready_for_capture": ready_for_capture,
                "blockers": list(import_blockers),
                "non_promotion": True,
            }
        )
        for traj in trajectories:
            for cam in camera_layouts:
                matrix_cell_id = ".".join(
                    [
                        "matrix",
                        _safe_token(capture_profile),
                        _safe_token(identity_id),
                        _safe_token(traj["trajectory_id"]),
                        _safe_token(cam["node_id"]),
                    ]
                )
                matrix.append(
                    {
                        "identity_id": identity_id,
                        "trajectory_id": traj["trajectory_id"],
                        "camera_layout_id": cam["camera_layout_id"],
                        "node_id": cam["node_id"],
                        "camera_ids": list(cam["camera_ids"]),
                        "capture_profile": capture_profile,
                        "model_label": model_label,
                        "identity_model_profile_id": profile_id,
                        "identity_model_profile": {
                            "identity_model_profile_id": profile_id,
                            "identity_id": identity_id,
                            "model_label": model_label,
                            "capture_profile": capture_profile,
                            "switch_method": switch_method,
                            "requires_ue_carla_import_readback": requires_ue_carla_import_readback,
                            "ready_for_capture": ready_for_capture,
                            "non_promotion": True,
                        },
                        "matrix_cell_id": matrix_cell_id,
                        "town": traj["town"],
                        "capture_allowed_now": ready_for_capture,
                        "block_reason": block_reason,
                    }
                )
                for camera in _as_list(cam.get("cameras")):
                    cam_rec = _as_dict(camera)
                    camera_id = str(cam_rec.get("camera_id") or "")
                    camera_role = str(cam_rec.get("role") or "unknown")
                    capture_task_id = ".".join(
                        [
                            "capture_task",
                            _safe_token(capture_profile),
                            _safe_token(identity_id),
                            _safe_token(traj["trajectory_id"]),
                            _safe_token(cam["node_id"]),
                            _safe_token(camera_id),
                        ]
                    )
                    capture_tasks.append(
                        {
                            "schema_version": "carla_air_capture_task_v1",
                            "capture_task_id": capture_task_id,
                            "capture_profile": capture_profile,
                            "model_label": model_label,
                            "identity_model_profile_id": profile_id,
                            "identity_model_profile": {
                                "identity_model_profile_id": profile_id,
                                "identity_id": identity_id,
                                "model_label": model_label,
                                "capture_profile": capture_profile,
                                "switch_method": switch_method,
                                "requires_ue_carla_import_readback": requires_ue_carla_import_readback,
                                "ready_for_capture": ready_for_capture,
                                "non_promotion": True,
                            },
                            "switch_method": switch_method,
                            "identity_id": identity_id,
                            "normalized_dir": normalized_dir,
                            "trajectory_id": traj["trajectory_id"],
                            "town": traj["town"],
                            "camera_layout_id": cam["camera_layout_id"],
                            "node_id": cam["node_id"],
                            "camera_id": camera_id,
                            "camera_role": camera_role,
                            "expected_scene_id": _expected_scene_id(identity_id, traj["trajectory_id"], cam["node_id"]),
                            "expected_scene_root": _expected_scene_root(identity_id, traj["trajectory_id"], cam["node_id"]),
                            "capture_allowed_now": ready_for_capture,
                            "block_reason": block_reason,
                            "requires_ue_carla_import_readback": requires_ue_carla_import_readback,
                            "writes_scene_outputs": False,
                            "starts_runtime": False,
                        }
                    )

    if len(identity_rows) < EXPECTED_PRIVATE_LOCAL_IDENTITY_COUNT:
        blockers.append(
            {
                "code": "insufficient_private_local_identities_for_v1",
                "detail": "V1 expects 6 normalized private/local identities from readiness/roster.",
                "actual": len(identity_rows),
                "required": EXPECTED_PRIVATE_LOCAL_IDENTITY_COUNT,
            }
        )

    if not trajectory_ev.get("readable"):
        blockers.append(
            {
                "code": "trajectory_config_missing_or_invalid",
                "detail": "Trajectory config JSON is missing/invalid; default fallback trajectory used.",
                "path": str(trajectory_json),
            }
        )

    ok = len(blockers) == 0
    run_id = _normalize_run_id(args.run_id)

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _now_iso(),
        "run_id": run_id,
        "repo_root": str(REPO_ROOT),
        "mode": "plan_only_preflight",
        "read_only": True,
        "starts_runtime": False,
        "writes_assets": False,
        "writes_scene_outputs": False,
        "updates_pipeline_contract": False,
        "capture_profile": capture_profile,
        "town": town or DEFAULT_TOWN,
        "camera_layout_defaults": DEFAULT_NODE_IDS,
        "v1_defaults": {"town": DEFAULT_TOWN, "camera_nodes": DEFAULT_NODE_IDS, "camera_ids": DEFAULT_CAMERA_IDS},
        "selected_filters": {
            "node_ids": selected_node_ids,
            "camera_ids": selected_camera_ids,
            "identity_ids": identity_ids,
            "trajectory_ids": [str(item.get("trajectory_id") or "") for item in trajectories],
            "requested_identity_ids": selected_identity_ids,
            "requested_trajectory_ids": selected_trajectory_ids,
        },
        "input_evidence": {
            "readiness_json": readiness_ev,
            "roster_plan": roster_ev,
            "ue_carla_import_readiness": import_ev,
            "trajectory_config": trajectory_ev,
        },
        "counts": {
            "identity_count": len(identity_rows),
            "expected_private_local_identity_count": EXPECTED_PRIVATE_LOCAL_IDENTITY_COUNT,
            "trajectory_count": len(trajectories),
            "camera_layout_count": len(camera_layouts),
            "identity_model_profile_count": len(identity_model_profiles),
            "matrix_count": len(matrix),
            "capture_task_count": len(capture_tasks),
            "blocked_capture_task_count": sum(1 for task in capture_tasks if not task["capture_allowed_now"]),
            "ue_carla_import_missing_count": ue_import_missing,
        },
        "identities": identity_rows,
        "identity_model_profiles": identity_model_profiles,
        "trajectories": trajectories,
        "camera_layouts": camera_layouts,
        "dataset_splits": dataset_splits,
        "deployment_episodes": deployment_episodes,
        "matrix": matrix,
        "capture_tasks": capture_tasks,
        "mask_gt_audit_policy": {
            "mode": "availability_audit_first",
            "availability": "unknown_until_live_or_export_audit",
            "mask_gt_source_required": "formal annotation export with auditable actor-to-pixel evidence",
            "disallow_pseudo_or_candidate_as_mask_gt": True,
            "disallow_proxy_as_mask_gt": True,
            "pseudo_candidate_only_diagnostic": True,
            "must_not_promote_candidate_without_verifier": True,
            "claim_restriction": "Do not label pseudo/candidate/proxy outputs as mask_gt.",
        },
        "blockers": blockers,
        "ok": ok,
        "allow_fail_used": bool(args.allow_fail),
        "next_actions": [
            "Run UE/CARLA import smoke/readback evidence for identities with ue_carla_import_verified=false.",
            "Re-run this planner after readiness/roster/import-readiness updates.",
            "Only run capture/export scripts after blockers are cleared.",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plan-only CARLA-Air Dataset Generation Pipeline V1 preflight.")
    parser.add_argument("--readiness-json", default=str(DEFAULT_READINESS_JSON))
    parser.add_argument("--roster-plan", default=str(DEFAULT_ROSTER_PLAN))
    parser.add_argument("--ue-carla-import-readiness", default=str(DEFAULT_UE_IMPORT_READINESS))
    parser.add_argument("--trajectory-config", default=str(DEFAULT_TRAJECTORY_CONFIG))
    parser.add_argument("--run-id", default="")
    parser.add_argument("--out", default="")
    parser.add_argument("--node-id", dest="node_ids", action="append", default=[], help="Repeat or comma-separate.")
    parser.add_argument("--camera-id", dest="camera_ids", action="append", default=[], help="Repeat or comma-separate.")
    parser.add_argument("--identity-id", dest="identity_ids", action="append", default=[], help="Repeat or comma-separate.")
    parser.add_argument("--trajectory-id", dest="trajectory_ids", action="append", default=[], help="Repeat or comma-separate.")
    parser.add_argument("--capture-profile", "--model-label", dest="capture_profile", default="v1_default")
    parser.add_argument("--allow-nonlocal-out", action="store_true")
    parser.add_argument("--allow-fail", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_id = _normalize_run_id(args.run_id)
    out = _validate_out_path(args.out, run_id, bool(args.allow_nonlocal_out))
    args.run_id = run_id
    payload = build_plan(args)
    _write_json(out, payload)
    print(
        json.dumps(
            {
                "ok": payload["ok"],
                "run_id": run_id,
                "identity_count": payload["counts"]["identity_count"],
                "trajectory_count": payload["counts"]["trajectory_count"],
                "camera_layout_count": payload["counts"]["camera_layout_count"],
                "matrix_count": payload["counts"]["matrix_count"],
                "blocker_count": len(payload["blockers"]),
                "out": str(out),
            },
            ensure_ascii=True,
            indent=2,
        )
    )
    if not payload["ok"] and not args.allow_fail:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
