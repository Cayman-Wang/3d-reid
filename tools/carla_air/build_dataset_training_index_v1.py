#!/usr/bin/env python3
"""Build CARLA-Air Dataset Generation Pipeline V1 training index artifacts.

This tool consumes a dataset_plan.json and optional scene roots, then writes
training-index artifacts under local/carla_air/dataset_runs/<run_id>/ by
default. If scene artifacts are absent, it emits a plan-only manifest with
empty samples and explicit blockers instead of failing.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
LOCAL_ROOT = REPO_ROOT / "local"
DEFAULT_RUN_ROOT = REPO_ROOT / "local/carla_air/dataset_runs"
SCHEMA_MANIFEST = "carla_air_dataset_training_index_v1"
SCHEMA_INDEX_MANIFEST = "carla_air_dataset_index_manifest_v1"
SCHEMA_SCENE_SAMPLE_INDEX_MANIFEST = "carla_air_scene_sample_index_manifest_v1"
SCHEMA_INDEX_STRICT_CONTRACT = "carla_air_dataset_index_strict_contract_v1"
SCHEMA_SAMPLE = "carla_air_dataset_sample_v1"
DEFAULT_CAMERA_IDS = ["cam0", "cam1", "cam2"]
DEFAULT_SPLIT_NAMES = ["train", "val_in_domain", "test_cross_layout"]
FORBIDDEN_MASK_GT_SOURCE_TOKENS = (
    "proxy",
    "candidate",
    "pseudo",
    "sam",
    "detector",
    "bbox",
    "projected",
    "semantic_lidar",
    "airsim_onboard",
    "onboard_camera",
)
FORBIDDEN_PIXEL_ACCURACY_TOKENS = (
    "quality_audit",
    "mask_quality",
    "candidate_stat",
    "candidate_statistics",
)
LEGACY_MASK_GT_SUBDIRS = {"masks_gt"}


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


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("json root is not an object")
    return payload


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _scene_rel(scene_dir: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(scene_dir.resolve()))
    except ValueError:
        return _repo_rel(path)


def _load_optional_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _path_exists_from_scene(scene_dir: Path, raw: Any) -> tuple[bool, str | None]:
    text = str(raw or "").strip()
    if not text:
        return False, None
    path = Path(text)
    candidates = [path] if path.is_absolute() else [scene_dir / path, REPO_ROOT / path]
    for candidate in candidates:
        if candidate.is_file():
            return True, _repo_rel(candidate)
    return False, _repo_rel(candidates[0])


def _contains_forbidden_token(*values: Any, tokens: tuple[str, ...] = FORBIDDEN_MASK_GT_SOURCE_TOKENS) -> bool:
    text = " ".join(str(value or "") for value in values).lower()
    return any(token in text for token in tokens)


def _positive_int(value: Any) -> int | None:
    if isinstance(value, int) and value > 0:
        return value
    if isinstance(value, str) and value.strip().isdigit() and int(value.strip()) > 0:
        return int(value.strip())
    return None


def _lineage_records(payload: dict[str, Any]) -> list[dict[str, Any]]:
    raw = payload.get("camera_lineage")
    if raw is None:
        raw = payload.get("lineage")
    if isinstance(raw, dict):
        return [raw]
    return [item for item in _as_list(raw) if isinstance(item, dict)]


def _lineage_satisfies_node_camera(payload: dict[str, Any]) -> bool:
    if payload.get("uses_airsim_onboard_camera") is True:
        return False
    for rec in _lineage_records(payload):
        if rec.get("uses_airsim_onboard_camera") is True:
            return False
        node_id = str(rec.get("node_id") or rec.get("node") or "").strip()
        camera_id = str(rec.get("camera_id") or rec.get("cam_id") or "").strip()
        has_time = any(
            rec.get(key) not in (None, "")
            for key in ("timestamp", "timestamp_us", "unix_ns", "frame_index", "carla_frame")
        )
        if node_id and camera_id and has_time:
            return True
    return False


def _actor_binding_satisfies_contract(payload: dict[str, Any]) -> bool:
    binding = _as_dict(payload.get("actor_binding"))
    actor_id = _positive_int(binding.get("actor_id"))
    if actor_id is None:
        actor_id = _positive_int(binding.get("target_actor_id"))
    has_binding_evidence = any(
        str(binding.get(key) or "").strip()
        for key in ("binding_source", "binding_evidence", "evidence_path", "evidence_file")
    )
    return actor_id is not None and has_binding_evidence


def _pixel_accuracy_satisfies_contract(scene_dir: Path, payload: dict[str, Any]) -> tuple[bool, list[str]]:
    raw = payload.get("pixel_accuracy_evidence")
    if raw is None:
        raw = payload.get("pixel_accuracy")
    records = raw if isinstance(raw, list) else [raw]
    existing_paths: list[str] = []
    for item in records:
        rec = item if isinstance(item, dict) else {"evidence_path": item}
        method = str(rec.get("method") or "").strip()
        source_type = str(rec.get("source_type") or rec.get("source") or "").strip()
        if not (method or source_type):
            continue
        if _contains_forbidden_token(
            method,
            source_type,
            rec.get("summary"),
            tokens=FORBIDDEN_PIXEL_ACCURACY_TOKENS + FORBIDDEN_MASK_GT_SOURCE_TOKENS,
        ):
            continue
        candidate_paths: list[Any] = []
        for key in ("evidence_path", "path", "file"):
            if rec.get(key):
                candidate_paths.append(rec.get(key))
        for key in ("evidence_paths", "paths", "files"):
            for value in _as_list(rec.get(key)):
                candidate_paths.append(value)
        for candidate in candidate_paths:
            exists, resolved = _path_exists_from_scene(scene_dir, candidate)
            if exists and resolved:
                existing_paths.append(resolved)
    return bool(existing_paths), sorted(set(existing_paths))


def _mask_subdir_satisfies_contract(scene_dir: Path, raw_subdir: Any) -> tuple[bool, str, int]:
    subdir = str(raw_subdir or "").strip().strip("/")
    if not subdir:
        return False, subdir, 0
    if subdir in LEGACY_MASK_GT_SUBDIRS:
        return False, subdir, 0
    if _contains_forbidden_token(subdir):
        return False, subdir, 0
    files = sorted(scene_dir.glob(f"cams/*/{subdir}/*.png"))
    return bool(files), subdir, len(files)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _parse_int_or_none(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    text = str(value or "").strip()
    if not text:
        return None
    if text.isdigit() or (text.startswith("-") and text[1:].isdigit()):
        try:
            return int(text)
        except Exception:
            return None
    return None


def _camera_ids_from_rig_payload(payload: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    for key in ("cameras", "camera_items"):
        for item in _as_list(payload.get(key)):
            obj = _as_dict(item)
            cam_id = str(obj.get("camera_id") or obj.get("id") or obj.get("name") or "").strip()
            if cam_id:
                ids.append(cam_id)
    for key in ("camera_ids",):
        for item in _as_list(payload.get(key)):
            cam_id = str(item or "").strip()
            if cam_id:
                ids.append(cam_id)
    # stable dedup while preserving order
    seen: set[str] = set()
    out: list[str] = []
    for cam_id in ids:
        if cam_id not in seen:
            seen.add(cam_id)
            out.append(cam_id)
    return out


def _json_loads_or_empty(raw: Any) -> Any:
    if raw in (None, ""):
        return None
    try:
        return json.loads(str(raw))
    except Exception:
        return None


def _split_for_node(node_id: str, plan: dict[str, Any]) -> str:
    configured = _as_dict(plan.get("dataset_splits")) or _as_dict(_as_dict(plan.get("dataset")).get("splits"))
    node_groups = _as_dict(configured.get("node_groups"))
    for split_name in DEFAULT_SPLIT_NAMES:
        if node_id and node_id in [str(item) for item in _as_list(node_groups.get(split_name))]:
            return split_name
    return "train"


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


def _validate_run_root(raw: str, allow_nonlocal_out: bool) -> Path:
    run_root = _repo_or_abs(raw)
    if (not allow_nonlocal_out) and (not _is_under(run_root, LOCAL_ROOT)):
        raise SystemExit("--run-root must stay under repository local/ unless --allow-nonlocal-out is set")
    return run_root


def _infer_run_id(plan: dict[str, Any]) -> str:
    for key in ("run_id", "dataset_run_id", "plan_id"):
        value = str(plan.get(key) or "").strip()
        if value:
            return value
    return datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S")


def _extract_identities(plan: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = (
        _as_list(plan.get("identities"))
        or _as_list(_as_dict(plan.get("identity")).get("identities"))
        or _as_list(_as_dict(plan.get("dataset")).get("identities"))
    )
    out: list[dict[str, Any]] = []
    for item in candidates:
        if not isinstance(item, dict):
            continue
        identity_id = str(item.get("identity_id") or item.get("id") or "").strip()
        if not identity_id:
            continue
        out.append({"identity_id": identity_id, "meta": item})
    return out


def _extract_trajectories(plan: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = (
        _as_list(plan.get("trajectories"))
        or _as_list(_as_dict(plan.get("trajectory")).get("trajectories"))
        or _as_list(_as_dict(plan.get("dataset")).get("trajectories"))
    )
    out: list[dict[str, Any]] = []
    for item in candidates:
        if not isinstance(item, dict):
            continue
        trajectory_id = str(item.get("trajectory_id") or item.get("id") or "").strip()
        if not trajectory_id:
            continue
        out.append({"trajectory_id": trajectory_id, "meta": item})
    return out


def _camera_items_from_layout(layout: dict[str, Any]) -> list[dict[str, Any]]:
    cameras = _as_list(layout.get("cameras"))
    if not cameras:
        cameras = [{"camera_id": camera_id} for camera_id in _as_list(layout.get("camera_ids"))]
    camera_items: list[dict[str, Any]] = []
    for entry in cameras:
        if not isinstance(entry, dict):
            continue
        camera_id = str(entry.get("camera_id") or entry.get("id") or "").strip()
        if not camera_id:
            continue
        camera_items.append({"camera_id": camera_id, "meta": entry})
    return camera_items


def _extract_camera_layouts(plan: dict[str, Any]) -> list[dict[str, Any]]:
    raw_layouts = _as_list(plan.get("camera_layouts")) or _as_list(_as_dict(plan.get("dataset")).get("camera_layouts"))
    if not raw_layouts:
        single = (
            _as_dict(plan.get("camera_layout"))
            or _as_dict(_as_dict(plan.get("dataset")).get("camera_layout"))
            or _as_dict(_as_dict(plan.get("camera")).get("layout"))
        )
        raw_layouts = [single] if single else []

    layouts: list[dict[str, Any]] = []
    for item in raw_layouts:
        layout = _as_dict(item)
        node_id = str(layout.get("node_id") or layout.get("node") or "").strip()
        layout_id = str(layout.get("camera_layout_id") or layout.get("layout_id") or node_id or "").strip()
        camera_items = _camera_items_from_layout(layout)
        if not camera_items and node_id:
            camera_items = [{"camera_id": camera_id, "meta": {"camera_id": camera_id}} for camera_id in DEFAULT_CAMERA_IDS]
        layouts.append({"layout_id": layout_id, "node_id": node_id, "cameras": camera_items, "raw": layout})
    return layouts


def _extract_capture_tasks(plan: dict[str, Any]) -> list[dict[str, Any]]:
    """Read capture_tasks from plan top-level; return [] when absent/invalid."""
    out: list[dict[str, Any]] = []
    for item in _as_list(plan.get("capture_tasks")):
        if not isinstance(item, dict):
            continue
        identity_id = str(item.get("identity_id") or "").strip()
        trajectory_id = str(item.get("trajectory_id") or "").strip()
        node_id = str(item.get("node_id") or "").strip()
        camera_id = str(item.get("camera_id") or "").strip()
        task_id = str(item.get("id") or item.get("capture_task_id") or item.get("task_id") or "").strip()
        out.append(
            {
                "id": task_id if task_id else None,
                "identity_id": identity_id,
                "trajectory_id": trajectory_id,
                "node_id": node_id,
                "camera_id": camera_id,
                "capture_profile": str(item.get("capture_profile") or "").strip(),
                "model_label": str(item.get("model_label") or "").strip(),
                "identity_model_profile_id": str(item.get("identity_model_profile_id") or "").strip(),
                "switch_method": str(item.get("switch_method") or "").strip(),
                "requires_ue_carla_import_readback": item.get("requires_ue_carla_import_readback") is True,
                "capture_allowed_now": item.get("capture_allowed_now") is True,
            }
        )
    return out


def _mask_gt_contract(plan: dict[str, Any]) -> dict[str, Any]:
    policy = (
        _as_dict(plan.get("mask_gt_policy"))
        or _as_dict(plan.get("mask_gt_audit_policy"))
        or _as_dict(_as_dict(plan.get("mask")).get("mask_gt_policy"))
        or _as_dict(_as_dict(plan.get("dataset")).get("mask_gt_policy"))
    )
    mode = str(policy.get("mode") or "availability_audit_first")
    availability = str(policy.get("availability") or "unknown")
    return {
        "mode": mode,
        "availability": availability,
        "mask_gt_required_for_formal_training": bool(policy.get("mask_gt_required_for_formal_training", False)),
        "pseudo_candidate_proxy_never_mask_gt": True,
        "forbidden_mask_gt_sources": [
            "proxy",
            "candidate",
            "pseudo",
            "sam",
            "detector",
            "projected_bbox",
            "semantic_lidar_actor_idx",
        ],
        "raw": policy,
    }


def _scene_contract_meta(scene_dir: Path) -> dict[str, Any]:
    contract = _load_optional_json(scene_dir / "pipeline_contract.json")
    capture_meta = _load_optional_json(scene_dir / "capture_meta.json")
    annotation_meta = _load_optional_json(scene_dir / "annotations" / "annotation_meta.json")
    synthetic_meta = _load_optional_json(scene_dir / "annotations" / "annotation_meta_synth.json")
    return {
        "contract": contract,
        "capture_meta": capture_meta,
        "annotation_meta": annotation_meta,
        "synthetic_meta": synthetic_meta,
        "scene_id": str(
            contract.get("scene_id")
            or capture_meta.get("scene_id")
            or annotation_meta.get("scene_id")
            or scene_dir.name
        ),
        "node_id": str(
            contract.get("node_id")
            or capture_meta.get("node_id")
            or annotation_meta.get("node_id")
            or ""
        ),
        "identity_id": str(
            contract.get("identity_id")
            or capture_meta.get("identity_id")
            or annotation_meta.get("identity_id")
            or "unknown_identity"
        ),
        "trajectory_id": str(
            contract.get("trajectory_id")
            or capture_meta.get("trajectory_id")
            or ""
        ),
    }


def _formal_mask_gt_evidence(scene_dir: Path) -> dict[str, Any]:
    """Return explicit mask_gt evidence only.

    Legacy directories named masks_gt are not enough here: existing CARLA-Air
    scenes used that name for proxy annotations. Future true mask_gt must be
    accompanied by an explicit evidence JSON before it is treated as ground
    truth by the dataset index.
    """

    evidence_paths = [
        scene_dir / "annotations" / "mask_gt_evidence.json",
        scene_dir / "annotations" / "formal_mask_gt_evidence.json",
        scene_dir / "annotations" / "mask_gt_source.json",
    ]
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for path in evidence_paths:
        payload = _load_optional_json(path)
        if not payload:
            continue
        source = str(payload.get("source") or payload.get("mask_source") or "")
        source_kind = str(payload.get("source_kind") or payload.get("type") or "")
        mask_subdir_ok, mask_subdir, mask_file_count = _mask_subdir_satisfies_contract(scene_dir, payload.get("mask_subdir"))
        pixel_accuracy_ok, pixel_accuracy_evidence_paths = _pixel_accuracy_satisfies_contract(scene_dir, payload)
        failure_reasons: list[str] = []
        if payload.get("is_mask_gt") is not True:
            failure_reasons.append("is_mask_gt_not_true")
        if payload.get("formal_mask_gt_ready") is not True:
            failure_reasons.append("formal_mask_gt_ready_not_true")
        if payload.get("pixel_accurate") is not True:
            failure_reasons.append("pixel_accurate_not_true")
        if _contains_forbidden_token(source, source_kind):
            failure_reasons.append("forbidden_mask_gt_source")
        if not _actor_binding_satisfies_contract(payload):
            failure_reasons.append("actor_binding_missing_or_incomplete")
        if not pixel_accuracy_ok:
            failure_reasons.append("pixel_accuracy_evidence_missing_or_unusable")
        if not _lineage_satisfies_node_camera(payload):
            failure_reasons.append("node_camera_lineage_missing_or_uses_airsim_onboard")
        if not mask_subdir_ok:
            failure_reasons.append("formal_mask_subdir_missing_forbidden_or_empty")
        ok = not failure_reasons
        rec = {
            "path": _repo_rel(path),
            "sha256": _sha256_file(path),
            "source": source,
            "source_kind": source_kind,
            "is_mask_gt": payload.get("is_mask_gt"),
            "formal_mask_gt_ready": payload.get("formal_mask_gt_ready"),
            "pixel_accurate": payload.get("pixel_accurate"),
            "mask_subdir": mask_subdir,
            "mask_file_count": mask_file_count,
            "pixel_accuracy_evidence_paths": pixel_accuracy_evidence_paths,
            "actor_binding_present": bool(_as_dict(payload.get("actor_binding"))),
            "node_camera_lineage_present": bool(_lineage_records(payload)),
            "failure_reasons": failure_reasons,
        }
        if ok:
            accepted.append(rec)
        else:
            rejected.append(rec)
    return {
        "available": bool(accepted),
        "accepted": accepted,
        "rejected": rejected,
        "required": (
            "explicit annotations/*mask_gt*_evidence.json with is_mask_gt=true, "
            "formal_mask_gt_ready=true, pixel_accurate=true, actor_binding, "
            "node-camera lineage, pixel-accuracy evidence file, and non-legacy mask_subdir"
        ),
    }


def _first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.is_file():
            return path
    return None


def _candidate_mask_paths(scene_dir: Path, cam_id: str, stem: str) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    checks = [
        ("legacy_misnamed_mask_gt_proxy_or_bootstrap", scene_dir / "cams" / cam_id / "masks_gt" / f"{stem}.png"),
        ("synthetic_candidate_not_identity_proof", scene_dir / "cams" / cam_id / "masks_synth" / f"{stem}.png"),
    ]
    for source, path in checks:
        if path.is_file():
            candidates.append({"source": source, "path": _scene_rel(scene_dir, path), "is_mask_gt": False})
    for path in sorted((scene_dir / "cams" / cam_id).glob(f"masks_actor_bbox*/*{stem}.png")):
        if path.is_file():
            candidates.append({"source": "projected_bbox_candidate", "path": _scene_rel(scene_dir, path), "is_mask_gt": False})
    return candidates


def _formal_mask_subdir(mask_gt_probe: dict[str, Any]) -> str | None:
    evidence = _as_dict(mask_gt_probe.get("explicit_evidence"))
    for rec in _as_list(evidence.get("accepted")):
        subdir = str(_as_dict(rec).get("mask_subdir") or "").strip()
        if subdir:
            return subdir
    return None


def _formal_mask_path(scene_dir: Path, cam_id: str, stem: str, mask_gt_probe: dict[str, Any]) -> Path | None:
    if mask_gt_probe.get("availability") != "available":
        return None
    subdir = _formal_mask_subdir(mask_gt_probe)
    if not subdir:
        return None
    return _first_existing([scene_dir / "cams" / cam_id / subdir / f"{stem}.png"])


def _depth_path(scene_dir: Path, cam_id: str, stem: str) -> Path | None:
    return _first_existing(
        [
            scene_dir / "cams" / cam_id / "depth" / f"{stem}.npy",
            scene_dir / "cams" / cam_id / "depth_synth" / f"{stem}.npy",
            scene_dir / "cams" / cam_id / "depth_gt" / f"{stem}.npy",
        ]
    )


def _sensor_png_path(scene_dir: Path, cam_id: str, subdir: str, stem: str) -> Path | None:
    return _first_existing([scene_dir / "cams" / cam_id / subdir / f"{stem}.png"])


def _plan_id_set(rows: list[dict[str, Any]], key: str) -> set[str]:
    values: set[str] = set()
    for row in rows:
        value = str(row.get(key) or "").strip()
        if value:
            values.add(value)
    return values


def _plan_layout_set(camera_layouts: list[dict[str, Any]]) -> set[tuple[str, str]]:
    values: set[tuple[str, str]] = set()
    for layout in camera_layouts:
        node_id = str(layout.get("node_id") or "").strip()
        for camera in _as_list(layout.get("cameras")):
            camera_id = str(_as_dict(camera).get("camera_id") or "").strip()
            if node_id and camera_id:
                values.add((node_id, camera_id))
    return values


def _collect_scene_observations(scene_roots: list[Path]) -> tuple[list[dict[str, Any]], list[str]]:
    observations: list[dict[str, Any]] = []
    blockers: list[str] = []
    if not scene_roots:
        blockers.append("no_scene_root_provided")
        return observations, blockers

    found_any_scene = False
    found_artifacts = False
    for raw_root in scene_roots:
        root = raw_root.resolve()
        exists = root.is_dir()
        if not exists:
            blockers.append(f"scene_root_missing:{root}")
            continue
        scene_dirs = sorted(path.parent for path in root.rglob("capture_meta.json"))
        if not scene_dirs:
            scene_dirs = [p for p in sorted(root.iterdir()) if p.is_dir()]
        if not scene_dirs:
            blockers.append(f"scene_root_has_no_scene_dirs:{root}")
            continue
        found_any_scene = True
        for scene_dir in scene_dirs:
            meta = _scene_contract_meta(scene_dir)
            rgb_files = (
                list(scene_dir.rglob("rgb*.png"))
                + list(scene_dir.rglob("*_rgb.png"))
                + list(scene_dir.glob("cams/*/frames/*.png"))
            )
            depth_files = (
                list(scene_dir.rglob("depth*.npy"))
                + list(scene_dir.rglob("*_depth.npy"))
                + list(scene_dir.glob("cams/*/depth/*.npy"))
                + list(scene_dir.glob("cams/*/depth_synth/*.npy"))
                + list(scene_dir.glob("cams/*/depth_gt/*.npy"))
            )
            semantic_files = list(scene_dir.glob("cams/*/semantic_synth/*.png"))
            instance_files = list(scene_dir.glob("cams/*/instance_synth/*.png"))
            pose_files = list(scene_dir.rglob("*pose*.json")) + list(scene_dir.rglob("object_pose_by_timestamp*.csv"))
            calib_files = list(scene_dir.rglob("*calib*.json"))
            if (scene_dir / "calib" / "rig.json").is_file():
                calib_files.append(scene_dir / "calib" / "rig.json")
            explicit_mask_gt = _formal_mask_gt_evidence(scene_dir)
            formal_mask_gt_files: list[Path] = []
            for rec in _as_list(explicit_mask_gt.get("accepted")):
                subdir = str(_as_dict(rec).get("mask_subdir") or "").strip()
                if subdir:
                    formal_mask_gt_files.extend(scene_dir.glob(f"cams/*/{subdir}/*.png"))
            non_gt_mask_candidate_files = (
                list(scene_dir.rglob("masks_gt/*.png"))
                + list(scene_dir.rglob("*mask*proxy*.png"))
                + list(scene_dir.rglob("*mask*candidate*.png"))
                + list(scene_dir.rglob("*mask*pseudo*.png"))
                + list(scene_dir.rglob("*projected_bbox*.png"))
                + list(scene_dir.rglob("masks_synth/*.png"))
                + list(scene_dir.rglob("masks_actor_bbox*/*.png"))
            )
            has_minimum = bool(rgb_files and (pose_files or calib_files))
            found_artifacts = found_artifacts or has_minimum
            if has_minimum:
                if explicit_mask_gt["available"] and formal_mask_gt_files:
                    mask_gt_availability = "available"
                    mask_gt_unavailable_reason = None
                elif explicit_mask_gt["available"]:
                    mask_gt_availability = "unavailable"
                    mask_gt_unavailable_reason = "explicit_formal_mask_gt_evidence_present_but_mask_files_not_found"
                else:
                    mask_gt_availability = "unavailable"
                    mask_gt_unavailable_reason = "explicit_formal_mask_gt_evidence_not_found"
            else:
                mask_gt_availability = "unknown"
                mask_gt_unavailable_reason = "minimum_scene_artifacts_missing_for_mask_gt_audit"
            frame_times_csv = scene_dir / "frame_times.csv"
            trajectory_groups_csv = scene_dir / "trajectory_frame_groups.csv"
            rig_json = scene_dir / "calib" / "rig.json"
            frame_rows = _read_csv_rows(frame_times_csv)
            valid_frame_rows = [
                row
                for row in frame_rows
                if str(row.get("cam_id") or "").strip() and str(row.get("filename") or "").strip()
            ]
            rows_by_camera: dict[str, int] = {}
            valid_rows_by_camera: dict[str, int] = {}
            timestamp_values: set[int] = set()
            carla_frame_values: set[int] = set()
            for row in frame_rows:
                cam_id = str(row.get("cam_id") or "").strip()
                has_valid = bool(cam_id and str(row.get("filename") or "").strip())
                if cam_id:
                    rows_by_camera[cam_id] = rows_by_camera.get(cam_id, 0) + 1
                    if has_valid:
                        valid_rows_by_camera[cam_id] = valid_rows_by_camera.get(cam_id, 0) + 1
                ts_us = _parse_int_or_none(row.get("ts_us"))
                if ts_us is not None:
                    timestamp_values.add(ts_us)
                carla_frame = _parse_int_or_none(row.get("carla_frame"))
                if carla_frame is not None:
                    carla_frame_values.add(carla_frame)
            camera_ids_observed = sorted(rows_by_camera.keys())
            rig_payload = _load_optional_json(rig_json)
            rig_camera_ids = _camera_ids_from_rig_payload(rig_payload)
            expected_camera_ids = rig_camera_ids if rig_camera_ids else (camera_ids_observed if camera_ids_observed else None)
            sidecar_presence = {
                "annotations": (scene_dir / "annotations").is_dir(),
                "tracks": trajectory_groups_csv.is_file(),
                "embeddings": bool(list(scene_dir.rglob("*embedding*.npy")) or list(scene_dir.rglob("*embeddings*.json*"))),
                "reconstruction_or_points": bool(list(scene_dir.rglob("*reconstruction*")) or list(scene_dir.rglob("*point*cloud*"))),
                "depth": bool(depth_files),
                "semantic": bool(semantic_files),
                "instance": bool(instance_files),
            }
            present_sidecar_keys = sorted([key for key, present in sidecar_presence.items() if present])
            missing_sidecar_keys = sorted([key for key, present in sidecar_presence.items() if not present])
            minimum_index_artifacts_ready = bool(has_minimum and valid_frame_rows)
            observations.append(
                {
                    "schema_version": "carla_air_dataset_scene_observation_v1",
                    "scene_id": meta["scene_id"],
                    "scene_dir": str(scene_dir.resolve()),
                    "node_id": meta["node_id"],
                    "identity_id": meta["identity_id"],
                    "trajectory_id": meta["trajectory_id"],
                    "artifact_counts": {
                        "rgb": len(rgb_files),
                        "depth": len(depth_files),
                        "semantic": len(semantic_files),
                        "instance": len(instance_files),
                        "pose": len(pose_files),
                        "calib": len(calib_files),
                        "formal_mask_gt": len(formal_mask_gt_files),
                        "non_gt_mask_candidate": len(non_gt_mask_candidate_files),
                    },
                    "modality_availability": {
                        "rgb": bool(rgb_files),
                        "depth": bool(depth_files),
                        "semantic": bool(semantic_files),
                        "instance": bool(instance_files),
                        "pose": bool(pose_files or trajectory_groups_csv.is_file()),
                        "calib": bool(calib_files or rig_json.is_file()),
                    },
                    "sample_source_files": {
                        "frame_times_csv": _repo_rel(frame_times_csv) if frame_times_csv.is_file() else None,
                        "trajectory_frame_groups_csv": _repo_rel(trajectory_groups_csv) if trajectory_groups_csv.is_file() else None,
                        "rig_json": _repo_rel(rig_json) if rig_json.is_file() else None,
                    },
                    "camera_coverage": {
                        "camera_ids": camera_ids_observed,
                        "camera_count": len(camera_ids_observed),
                        "rows_by_camera": rows_by_camera,
                        "valid_rows_by_camera": valid_rows_by_camera,
                        "expected_camera_ids": expected_camera_ids,
                    },
                    "timestamp_coverage": {
                        "row_count": len(frame_rows),
                        "valid_row_count": len(valid_frame_rows),
                        "unique_timestamp_count": len(timestamp_values),
                        "min_timestamp_us": min(timestamp_values) if timestamp_values else None,
                        "max_timestamp_us": max(timestamp_values) if timestamp_values else None,
                        "min_carla_frame": min(carla_frame_values) if carla_frame_values else None,
                        "max_carla_frame": max(carla_frame_values) if carla_frame_values else None,
                    },
                    "calib_rig_summary": {
                        "exists": rig_json.is_file(),
                        "path": _repo_rel(rig_json) if rig_json.is_file() else None,
                        "camera_count": len(rig_camera_ids) if rig_camera_ids else (len(camera_ids_observed) if camera_ids_observed else 0),
                        "camera_ids": rig_camera_ids or camera_ids_observed,
                    },
                    "sidecar_missing_summary": {
                        "present": present_sidecar_keys,
                        "present_count": len(present_sidecar_keys),
                        "missing": missing_sidecar_keys,
                        "missing_count": len(missing_sidecar_keys),
                    },
                    "has_minimum_artifacts": has_minimum,
                    "scene_qualification": {
                        "minimum_index_artifacts_ready": minimum_index_artifacts_ready,
                        "frame_rows_total": len(frame_rows),
                        "frame_rows_valid_for_index": len(valid_frame_rows),
                        "formal_mask_gt_available": bool(explicit_mask_gt["available"] and formal_mask_gt_files),
                        "legacy_proxy_candidate_not_promoted": True,
                    },
                    "readiness": {
                        "status": "ready_for_index_no_mask" if minimum_index_artifacts_ready else "blocked",
                        "blocked": not minimum_index_artifacts_ready,
                        "blocked_reasons": (
                            []
                            if minimum_index_artifacts_ready
                            else ["minimum_index_artifacts_or_frame_rows_missing"]
                        ),
                    },
                    "mask_gt_probe": {
                        "availability": mask_gt_availability,
                        "formal_mask_gt_present": bool(explicit_mask_gt["available"]),
                        "explicit_evidence": explicit_mask_gt,
                        "formal_mask_gt_unavailable_reason": mask_gt_unavailable_reason,
                        "non_gt_mask_candidates_present": bool(non_gt_mask_candidate_files),
                        "non_gt_mask_candidate_count": len(non_gt_mask_candidate_files),
                        "candidate_pseudo_proxy_never_promoted_to_mask_gt": True,
                        "legacy_masks_gt_directory_is_not_formal_mask_gt_without_evidence": True,
                        "audit_only": True,
                    },
                }
            )
    if not found_any_scene:
        blockers.append("no_scene_dir_discovered")
    if not found_artifacts:
        blockers.append("no_minimum_scene_artifacts_for_samples")
    return observations, blockers


def _build_samples(
    *,
    identities: list[dict[str, Any]],
    trajectories: list[dict[str, Any]],
    camera_layouts: list[dict[str, Any]],
    scene_observations: list[dict[str, Any]],
    capture_tasks: list[dict[str, Any]],
    plan: dict[str, Any],
    mask_gt: dict[str, Any],
) -> list[dict[str, Any]]:
    if not any(layout.get("cameras") for layout in camera_layouts):
        return []

    layout_by_node = {str(layout.get("node_id") or ""): layout for layout in camera_layouts if layout.get("node_id")}
    plan_identity_ids = _plan_id_set(identities, "identity_id")
    plan_trajectory_ids = _plan_id_set(trajectories, "trajectory_id")
    plan_node_ids = {str(layout.get("node_id") or "").strip() for layout in camera_layouts if layout.get("node_id")}
    plan_node_camera_pairs = _plan_layout_set(camera_layouts)
    capture_task_lookup: dict[tuple[str, str, str, str], str | None] = {}
    planned_capture_tasks_by_tnc: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for task in capture_tasks:
        key = (
            str(task.get("identity_id") or "").strip(),
            str(task.get("trajectory_id") or "").strip(),
            str(task.get("node_id") or "").strip(),
            str(task.get("camera_id") or "").strip(),
        )
        if all(key) and key not in capture_task_lookup:
            capture_task_lookup[key] = task.get("capture_task_id") or task.get("id")
        tnc_key = key[1:]
        if all(tnc_key):
            planned_capture_tasks_by_tnc.setdefault(tnc_key, []).append(task)
    samples: list[dict[str, Any]] = []
    for observation in scene_observations:
        if not observation.get("has_minimum_artifacts"):
            continue
        scene_dir = Path(str(observation["scene_dir"]))
        frame_times_csv = scene_dir / "frame_times.csv"
        frame_rows = _read_csv_rows(frame_times_csv)
        if not frame_rows:
            continue
        trajectory_groups_csv = scene_dir / "trajectory_frame_groups.csv"
        rig_json = scene_dir / "calib" / "rig.json"
        node_id = str(observation.get("node_id") or "")
        identity_id = str(observation.get("identity_id") or "unknown_identity")
        trajectory_id = str(observation.get("trajectory_id") or "unknown_trajectory")
        layout = layout_by_node.get(node_id, {"layout_id": f"{node_id}_observed_layout", "node_id": node_id, "cameras": []})
        camera_ids = [str(item.get("camera_id")) for item in _as_list(layout.get("cameras")) if _as_dict(item).get("camera_id")]
        split = _split_for_node(node_id, plan)
        for row in frame_rows:
            cam_id = str(row.get("cam_id") or "").strip()
            filename = str(row.get("filename") or "").strip()
            if not cam_id or not filename:
                continue
            if camera_ids and cam_id not in camera_ids:
                continue
            rgb_path = scene_dir / filename
            if not rgb_path.is_file():
                continue
            stem = rgb_path.stem
            depth_path = _depth_path(scene_dir, cam_id, stem)
            semantic_path = _sensor_png_path(scene_dir, cam_id, "semantic_synth", stem)
            instance_path = _sensor_png_path(scene_dir, cam_id, "instance_synth", stem)
            candidate_paths = _candidate_mask_paths(scene_dir, cam_id, stem)
            mask_gt_probe = _as_dict(observation.get("mask_gt_probe"))
            formal_mask_path = _formal_mask_path(scene_dir, cam_id, stem, mask_gt_probe)
            mask_gt_available = formal_mask_path is not None
            timestamp_us = int(row["ts_us"]) if str(row.get("ts_us") or "").isdigit() else None
            frame_index = int(row["carla_frame"]) if str(row.get("carla_frame") or "").isdigit() else None
            planned_index = None
            if str(row.get("planned_frame_index") or "").isdigit():
                planned_index = int(row["planned_frame_index"])
            refs = {
                "rgb": _repo_rel(rgb_path),
                "depth": _repo_rel(depth_path) if depth_path is not None else None,
                "semantic": _repo_rel(semantic_path) if semantic_path is not None else None,
                "instance": _repo_rel(instance_path) if instance_path is not None else None,
                "pose": _repo_rel(trajectory_groups_csv) if trajectory_groups_csv.is_file() else None,
                "calib": _repo_rel(rig_json) if rig_json.is_file() else None,
            }
            modality_availability = {key: value is not None for key, value in refs.items()}
            matrix_entry_in_plan = (
                identity_id in plan_identity_ids
                and trajectory_id in plan_trajectory_ids
                and (node_id, cam_id) in plan_node_camera_pairs
            )
            capture_task_key = (identity_id, trajectory_id, node_id, cam_id)
            capture_task_id = capture_task_lookup.get(capture_task_key)
            planned_task_candidates = sorted(
                planned_capture_tasks_by_tnc.get((trajectory_id, node_id, cam_id), []),
                key=lambda item: str(item.get("capture_task_id") or item.get("id") or ""),
            )
            planned_capture_task_candidates = [
                {
                    "capture_task_id": str(task.get("capture_task_id") or task.get("id") or "").strip() or None,
                    "planned_identity_id": str(task.get("identity_id") or "").strip() or None,
                    "identity_model_profile_id": str(task.get("identity_model_profile_id") or "").strip() or None,
                    "model_label": str(task.get("model_label") or "").strip() or None,
                    "capture_profile": str(task.get("capture_profile") or "").strip() or None,
                    "switch_method": str(task.get("switch_method") or "").strip() or None,
                    "requires_ue_carla_import_readback": task.get("requires_ue_carla_import_readback") is True,
                    "capture_allowed_now": task.get("capture_allowed_now") is True,
                }
                for task in planned_task_candidates
            ]
            planned_candidate_identity_ids = sorted(
                {
                    str(item.get("planned_identity_id") or "").strip()
                    for item in planned_capture_task_candidates
                    if str(item.get("planned_identity_id") or "").strip()
                }
            )
            planned_candidate_capture_task_ids = sorted(
                {
                    str(item.get("capture_task_id") or "").strip()
                    for item in planned_capture_task_candidates
                    if str(item.get("capture_task_id") or "").strip()
                }
            )
            identity_in_plan = identity_id in plan_identity_ids
            trajectory_node_camera_in_plan = bool(planned_capture_task_candidates)
            observed_identity_in_candidate_set = identity_id in planned_candidate_identity_ids
            plan_alignment = {
                "identity_in_plan": identity_in_plan,
                "trajectory_in_plan": trajectory_id in plan_trajectory_ids,
                "node_in_plan": node_id in plan_node_ids,
                "node_camera_in_plan": (node_id, cam_id) in plan_node_camera_pairs,
                "matrix_entry_in_plan": matrix_entry_in_plan,
                "capture_task_in_plan": capture_task_key in capture_task_lookup,
                "capture_task_id": capture_task_id,
                "trajectory_node_camera_in_capture_matrix": trajectory_node_camera_in_plan,
                "planned_capture_task_candidate_count": len(planned_capture_task_candidates),
                "planned_capture_task_candidate_ids": planned_candidate_capture_task_ids,
                "planned_identity_ids_for_trajectory_node_camera": planned_candidate_identity_ids,
                "observed_identity_in_planned_capture_candidates": observed_identity_in_candidate_set,
                "capture_matrix_bridge_status": (
                    "exact_capture_task_match"
                    if capture_task_id
                    else "trajectory_node_camera_passthrough_identity_mismatch"
                    if trajectory_node_camera_in_plan
                    else "missing_capture_matrix_entry"
                ),
                "observed_identity_id": identity_id,
                "planned_identity_match": identity_in_plan,
                "observed_identity_matches_planned": identity_in_plan,
                "planned_identity_rewrite_applied": False,
                "no_silent_identity_rewrite": True,
                "legacy_or_observed_scene_passthrough": (not matrix_entry_in_plan) or (not identity_in_plan),
            }
            scene_qualification_obj = _as_dict(observation.get("scene_qualification"))
            readiness_obj = _as_dict(observation.get("readiness"))
            sample_id = "__".join(
                part
                for part in [
                    str(observation.get("scene_id") or scene_dir.name),
                    identity_id,
                    trajectory_id,
                    node_id or "unknown_node",
                    cam_id,
                    stem,
                ]
                if part
            )
            scene_key = _scene_key_for_fields(
                identity_id,
                trajectory_id,
                node_id,
                str(observation.get("scene_id") or scene_dir.name),
                _repo_rel(scene_dir),
            )
            samples.append(
                {
                    "schema_version": SCHEMA_SAMPLE,
                    "sample_id": sample_id,
                    "scene_key": scene_key,
                    "scene_id": str(observation.get("scene_id") or scene_dir.name),
                    "split": split,
                    "identity_id": identity_id,
                    "trajectory_id": trajectory_id,
                    "node_id": node_id,
                    "camera_id": cam_id,
                    "rgb": refs["rgb"],
                    "depth": refs["depth"],
                    "semantic": refs["semantic"],
                    "instance": refs["instance"],
                    "pose": refs["pose"],
                    "calib": refs["calib"],
                    "source": {
                        "mode": "existing_scene_probe_materialized",
                        "scene_id": observation.get("scene_id"),
                        "scene_dir": _repo_rel(scene_dir),
                    },
                    "identity": {"identity_id": identity_id},
                    "trajectory": {"trajectory_id": trajectory_id},
                    "camera_layout": {
                        "layout_id": layout.get("layout_id") or f"{node_id}_observed_layout",
                        "node_id": node_id,
                        "camera_ids": camera_ids or [cam_id],
                    },
                    "view": {"node_id": node_id, "camera_id": cam_id},
                    "timestamp": {
                        "frame_index": frame_index,
                        "planned_frame_index": planned_index,
                        "unix_ns": timestamp_us * 1000 if timestamp_us is not None else None,
                        "timestamp_us": timestamp_us,
                        "carla_timestamp": float(row["carla_timestamp"]) if str(row.get("carla_timestamp") or "") else None,
                        "iso8601": None,
                    },
                    "refs": refs,
                    "modality_availability": modality_availability,
                    "plan_alignment": plan_alignment,
                    "capture_matrix_bridge": {
                        "schema_version": "carla_air_sample_capture_matrix_bridge_v1",
                        "trajectory_node_camera_in_plan": trajectory_node_camera_in_plan,
                        "observed_identity_id": identity_id,
                        "observed_identity_in_planned_capture_candidates": observed_identity_in_candidate_set,
                        "planned_capture_task_candidate_count": len(planned_capture_task_candidates),
                        "planned_capture_task_candidate_ids": planned_candidate_capture_task_ids,
                        "planned_identity_ids_for_trajectory_node_camera": planned_candidate_identity_ids,
                        "planned_capture_task_candidates": planned_capture_task_candidates,
                        "exact_capture_task_id": capture_task_id,
                        "bridge_status": plan_alignment["capture_matrix_bridge_status"],
                        "no_silent_identity_rewrite": True,
                        "legacy_or_observed_scene_passthrough_allowed_for_no_mask_index": True,
                        "non_promotion": True,
                        "full_v1_live_dataset_ready": False,
                    },
                    "scene_qualification": {
                        "minimum_index_artifacts_ready": scene_qualification_obj.get("minimum_index_artifacts_ready") is True,
                        "formal_mask_gt_available": bool(mask_gt_available),
                        "legacy_proxy_candidate_not_promoted": scene_qualification_obj.get("legacy_proxy_candidate_not_promoted")
                        is True,
                        "readiness_status": str(readiness_obj.get("status") or "unknown").strip() or "unknown",
                        "readiness_blocked": readiness_obj.get("blocked") is True,
                        "readiness_blocked_reasons": [
                            str(reason).strip()
                            for reason in _as_list(readiness_obj.get("blocked_reasons"))
                            if str(reason).strip()
                        ],
                    },
                    "pose_ref": {
                        "trajectory_frame_groups_csv": _repo_rel(trajectory_groups_csv) if trajectory_groups_csv.is_file() else None,
                        "row_key": {"ts_us": timestamp_us, "cam_id": cam_id, "planned_frame_index": planned_index},
                    },
                    "mask_gt": {
                        "policy_mode": mask_gt["mode"],
                        "availability": "available" if mask_gt_available else "unavailable",
                        "present": bool(mask_gt_available),
                        "source": "ground_truth" if mask_gt_available else "none",
                        "path": _repo_rel(formal_mask_path) if formal_mask_path is not None else None,
                        "is_mask_gt": bool(mask_gt_available),
                        "unavailable_reason": None
                        if mask_gt_available
                        else (
                            mask_gt_probe.get("formal_mask_gt_unavailable_reason")
                            or "explicit_formal_mask_gt_evidence_not_found"
                        ),
                        "audit_state": "verified" if mask_gt_available else "missing_formal_gt",
                        "non_gt_candidates_seen": bool(candidate_paths),
                        "pseudo_or_candidate_never_mask_gt": True,
                    },
                    "mask_gt_audit": {
                        "scene_probe_has_rgb_pose_calib": bool(observation.get("has_minimum_artifacts")),
                        "formal_mask_gt_found": bool(mask_gt_available),
                        "formal_mask_gt_path": _repo_rel(formal_mask_path) if formal_mask_path is not None else None,
                        "candidate_or_pseudo_or_proxy_paths": candidate_paths,
                        "legacy_masks_gt_directory_is_not_formal_mask_gt_without_evidence": True,
                        "contract_enforcement": "candidate/pseudo/proxy must not be promoted to mask_gt",
                    },
                    "raw_row": {
                        "carla_frame": row.get("carla_frame"),
                        "planned_frame_index": row.get("planned_frame_index"),
                        "camera_meta": _json_loads_or_empty(row.get("camera_meta_json")),
                    },
                }
            )
    return samples


def _build_modality_summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    keys = ["rgb", "depth", "semantic", "instance", "pose", "calib"]
    present = {key: 0 for key in keys}
    missing = {key: 0 for key in keys}
    for sample in samples:
        refs = _as_dict(sample.get("refs"))
        for key in keys:
            if refs.get(key):
                present[key] += 1
            else:
                missing[key] += 1
    complete_count = sum(1 for sample in samples if all(_as_dict(sample.get("refs")).get(key) for key in keys))
    return {
        "sample_count": len(samples),
        "present_count_by_modality": present,
        "missing_count_by_modality": missing,
        "rgb_depth_semantic_instance_pose_calib_complete_count": complete_count,
        "rgb_depth_semantic_instance_pose_calib_complete_fraction": (complete_count / len(samples)) if samples else 0.0,
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
            key: {
                "count": len(values),
                "preview": values[:12],
                "preview_limit": 12,
                "truncated": len(values) > 12,
            }
            for key, values in missing_sample_ids.items()
            if values
        },
    }


def _build_sidecar_quality_matrix(samples: list[dict[str, Any]]) -> dict[str, Any]:
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

    scene_entries = []
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

    scene_split_entries = []
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


def _build_sidecar_quality_manifest(run_id: str, samples: list[dict[str, Any]]) -> dict[str, Any]:
    matrix = _build_sidecar_quality_matrix(samples)
    overall = _as_dict(matrix.get("overall"))
    by_split = _as_list(matrix.get("by_split"))
    by_scene = _as_list(matrix.get("by_scene"))
    by_scene_split = _as_list(matrix.get("by_scene_split"))
    stable_hashes = {
        "overall_digest": _canonical_json_sha256(overall),
        "by_split_digest": _canonical_json_sha256(by_split),
        "by_scene_digest": _canonical_json_sha256(by_scene),
        "by_scene_split_digest": _canonical_json_sha256(by_scene_split),
    }
    manifest = {
        "schema_version": "carla_air_sidecar_quality_manifest_v1",
        "run_id": str(run_id or "").strip(),
        "starts_runtime": False,
        "writes_scene_outputs": False,
        "non_promotion": True,
        "full_v1_live_dataset_ready": False,
        "sample_count": int(overall.get("sample_count") or 0),
        "complete_rgb_depth_semantic_instance_pose_calib_count": int(
            overall.get("complete_rgb_depth_semantic_instance_pose_calib_count") or 0
        ),
        "complete_fraction": float(overall.get("complete_fraction") or 0.0),
        "present_count_by_modality": {
            key: int(_as_dict(overall.get("present_count_by_modality")).get(key) or 0)
            for key in ("rgb", "depth", "semantic", "instance", "pose", "calib")
        },
        "missing_count_by_modality": {
            key: int(_as_dict(overall.get("missing_count_by_modality")).get(key) or 0)
            for key in ("rgb", "depth", "semantic", "instance", "pose", "calib")
        },
        "mask_gt_available_count": sum(
            1 for sample in samples if str(_as_dict(sample.get("mask_gt")).get("availability") or "").strip() == "available"
        ),
        "no_mask_sample_count": sum(
            1 for sample in samples if str(_as_dict(sample.get("mask_gt")).get("availability") or "").strip() != "available"
        ),
        "by_split": by_split,
        "by_scene": by_scene,
        "by_scene_split": by_scene_split,
        "split_count": len(by_split),
        "scene_count": len(by_scene),
        "scene_split_count": len(by_scene_split),
        "stable_hashes": stable_hashes,
        "notes": {
            "sidecar_absence_is_not_mask_gt_promotion": True,
            "proxy_candidate_pseudo_legacy_masks_not_trusted_mask_gt": True,
            "default_airsim_drone_or_passthrough_identity_not_formal_annotation": True,
            "semantic_lidar_or_actor_relative_points_not_real_geometry": True,
            "trusted_mask_gt_requires_explicit_formal_evidence": True,
        },
    }
    stable_hashes["manifest_payload_digest_without_manifest_digest"] = _canonical_json_sha256(manifest)
    return manifest


def _build_sample_schema_coverage_summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
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
        for key in required_fields:
            present = sample.get(key) not in (None, "")
            if key in ("depth", "semantic", "instance"):
                # Sidecars can be unavailable, but the field must still be present.
                present = key in sample
            if key == "scene_key":
                source = _as_dict(sample.get("source"))
                derived_scene_key = _scene_key_for_fields(
                    str(sample.get("identity_id") or "").strip(),
                    str(sample.get("trajectory_id") or "").strip(),
                    str(sample.get("node_id") or "").strip(),
                    str(sample.get("scene_id") or source.get("scene_id") or "").strip(),
                    str(source.get("scene_dir") or "").strip(),
                )
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


def _build_sample_schema_coverage_manifest(
    *, run_id: str, sample_schema_coverage_summary: dict[str, Any]
) -> dict[str, Any]:
    required_fields = list(_as_list(sample_schema_coverage_summary.get("required_fields")))
    field_present_count = {
        str(k): int(v)
        for k, v in _as_dict(sample_schema_coverage_summary.get("field_present_count")).items()
        if str(k).strip()
    }
    field_missing_count = {
        str(k): int(v)
        for k, v in _as_dict(sample_schema_coverage_summary.get("field_missing_count")).items()
        if str(k).strip()
    }
    core_payload = {
        "schema_version": "carla_air_sample_schema_coverage_manifest_v1",
        "run_id": str(run_id or "").strip(),
        "sample_count": int(sample_schema_coverage_summary.get("sample_count") or 0),
        "required_fields": required_fields,
        "field_present_count": field_present_count,
        "field_missing_count": field_missing_count,
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


def _build_plan_alignment_summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    keys = ["identity_in_plan", "trajectory_in_plan", "node_in_plan", "node_camera_in_plan", "matrix_entry_in_plan"]
    true_counts = {key: 0 for key in keys}
    passthrough_count = 0
    for sample in samples:
        alignment = _as_dict(sample.get("plan_alignment"))
        for key in keys:
            if alignment.get(key) is True:
                true_counts[key] += 1
        if alignment.get("legacy_or_observed_scene_passthrough") is True:
            passthrough_count += 1
    return {
        "sample_count": len(samples),
        "true_count_by_check": true_counts,
        "strict_matrix_entry_sample_count": true_counts["matrix_entry_in_plan"],
        "legacy_or_observed_scene_passthrough_count": passthrough_count,
    }


def _build_identity_model_switch_contract(samples: list[dict[str, Any]], identities: list[dict[str, Any]], plan: dict[str, Any]) -> dict[str, Any]:
    planned_identity_ids = sorted(_plan_id_set(identities, "identity_id"))
    observed_sample_identity_ids = sorted(
        {
            str(sample.get("identity_id") or "").strip()
            for sample in samples
            if str(sample.get("identity_id") or "").strip()
        }
    )
    strict_planned_identity_sample_count = 0
    observed_passthrough_identity_sample_count = 0
    identity_mismatch_count = 0
    for sample in samples:
        alignment = _as_dict(sample.get("plan_alignment"))
        identity_matches = alignment.get("observed_identity_matches_planned") is True
        passthrough = alignment.get("legacy_or_observed_scene_passthrough") is True
        if identity_matches:
            strict_planned_identity_sample_count += 1
        else:
            identity_mismatch_count += 1
            passthrough = True
        if passthrough:
            observed_passthrough_identity_sample_count += 1

    capture_profile = plan.get("capture_profile")
    model_label = None
    if isinstance(capture_profile, dict):
        model_label = capture_profile.get("model_label") or capture_profile.get("label") or capture_profile.get("name")
    elif capture_profile is not None:
        model_label = capture_profile

    return {
        "planned_identity_ids": planned_identity_ids,
        "observed_sample_identity_ids": observed_sample_identity_ids,
        "strict_planned_identity_sample_count": strict_planned_identity_sample_count,
        "observed_passthrough_identity_sample_count": observed_passthrough_identity_sample_count,
        "identity_mismatch_count": identity_mismatch_count,
        "all_samples_match_planned_identities": identity_mismatch_count == 0,
        "capture_profile": capture_profile,
        "model_label": model_label,
        "requires_ue_carla_import_readback": True,
        "non_promotion": True,
        "no_silent_identity_rewrite": True,
        "legacy_or_observed_scene_passthrough_allowed_for_no_mask_index": True,
        "full_live_readiness_blocked_on_identity_mismatch": identity_mismatch_count > 0,
    }


def _build_identity_model_switch_manifest(
    *,
    run_id: str,
    samples: list[dict[str, Any]],
    identities: list[dict[str, Any]],
    plan: dict[str, Any],
    capture_tasks: list[dict[str, Any]],
) -> dict[str, Any]:
    identity_switch_contract = _build_identity_model_switch_contract(samples, identities, plan)
    plan_profiles = [
        item
        for item in _as_list(plan.get("identity_model_profiles"))
        if isinstance(item, dict)
    ]
    planned_identity_ids = sorted(_plan_id_set(identities, "identity_id"))
    profile_switch_methods = sorted(
        {str(item.get("switch_method") or "").strip() for item in plan_profiles if str(item.get("switch_method") or "").strip()}
    )
    profile_model_labels = sorted(
        {str(item.get("model_label") or "").strip() for item in plan_profiles if str(item.get("model_label") or "").strip()}
    )
    profile_requires_import_readback_flags = sorted(
        {bool(item.get("requires_ue_carla_import_readback")) for item in plan_profiles}
    )
    blocked_capture_task_count = sum(
        1 for task in capture_tasks if isinstance(task, dict) and task.get("capture_allowed_now") is not True
    )
    sample_count = len(samples)
    identity_mismatch_count = int(identity_switch_contract.get("identity_mismatch_count") or 0)
    all_samples_match = identity_switch_contract.get("all_samples_match_planned_identities") is True
    identity_alignment_status = (
        "strict_planned_identity_match"
        if sample_count > 0 and all_samples_match
        else "observed_scene_passthrough"
        if sample_count > 0 and identity_mismatch_count > 0
        else "plan_only_or_no_samples"
    )
    identity_alignment_summary = {
        "sample_count": sample_count,
        "planned_identity_ids": planned_identity_ids,
        "observed_sample_identity_ids": list(identity_switch_contract.get("observed_sample_identity_ids") or []),
        "strict_planned_identity_sample_count": int(identity_switch_contract.get("strict_planned_identity_sample_count") or 0),
        "observed_passthrough_identity_sample_count": int(
            identity_switch_contract.get("observed_passthrough_identity_sample_count") or 0
        ),
        "identity_mismatch_count": identity_mismatch_count,
        "all_samples_match_planned_identities": all_samples_match,
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
        "capture_profile": plan.get("capture_profile"),
        "planned_identity_ids": planned_identity_ids,
        "identity_model_profiles": plan_profiles,
        "profile_count": len(plan_profiles),
        "identity_count": len(planned_identity_ids),
        "switch_methods": profile_switch_methods,
        "model_labels": profile_model_labels,
        "requires_ue_carla_import_readback_flags": profile_requires_import_readback_flags,
        "capture_task_count": len(capture_tasks),
        "blocked_capture_task_count": blocked_capture_task_count,
        "observed_sample_identity_ids": list(identity_switch_contract.get("observed_sample_identity_ids") or []),
        "identity_mismatch_count": identity_mismatch_count,
        "identity_alignment_summary": identity_alignment_summary,
        "no_silent_identity_rewrite": True,
        "legacy_or_observed_scene_passthrough_allowed_for_no_mask_index": True,
    }


def _build_existing_scene_index_bridge_manifest(
    *,
    run_id: str,
    samples: list[dict[str, Any]],
    scene_observations: list[dict[str, Any]],
    scene_membership_manifest: dict[str, Any],
    dataset_index_manifest: dict[str, Any],
) -> dict[str, Any]:
    scene_membership_entries = _as_list(scene_membership_manifest.get("scene_entries"))
    membership_index: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for entry in scene_membership_entries:
        obj = _as_dict(entry)
        key = (
            str(obj.get("identity_id") or "").strip(),
            str(obj.get("trajectory_id") or "").strip(),
            str(obj.get("node_id") or "").strip(),
            str(obj.get("scene_id") or "").strip(),
        )
        if any(key):
            membership_index[key] = obj

    obs_index: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for obs in scene_observations:
        obj = _as_dict(obs)
        key = (
            str(obj.get("identity_id") or "").strip(),
            str(obj.get("trajectory_id") or "").strip(),
            str(obj.get("node_id") or "").strip(),
            str(obj.get("scene_id") or "").strip(),
        )
        if any(key):
            obs_index[key] = obj

    per_scene: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    total_mask_gt_available_count = 0
    total_no_mask_sample_count = 0
    for sample in samples:
        sample_obj = _as_dict(sample)
        source = _as_dict(sample_obj.get("source"))
        identity_id = str(sample_obj.get("identity_id") or "").strip()
        trajectory_id = str(sample_obj.get("trajectory_id") or "").strip()
        node_id = str(sample_obj.get("node_id") or "").strip()
        scene_id = str(sample_obj.get("scene_id") or source.get("scene_id") or "").strip()
        scene_dir = str(source.get("scene_dir") or source.get("scene_root") or "").strip()
        split_name = str(sample_obj.get("split") or "").strip()
        camera_id = str(sample_obj.get("camera_id") or "").strip()
        sample_id = str(sample_obj.get("sample_id") or "").strip()
        ts = _as_dict(sample_obj.get("timestamp"))
        timestamp_us = _parse_int_or_none(ts.get("timestamp_us"))
        mask_gt = _as_dict(sample_obj.get("mask_gt"))
        mask_gt_available = mask_gt.get("present") is True or str(mask_gt.get("availability") or "").strip() == "available"
        refs = _as_dict(sample_obj.get("refs"))
        sidecar_complete = all(bool(refs.get(key)) for key in ("rgb", "depth", "semantic", "instance", "pose", "calib"))
        key = (identity_id, trajectory_id, node_id, scene_id)
        rec = per_scene.setdefault(
            key,
            {
                "scene_id": scene_id,
                "scene_key": _scene_key_for_fields(identity_id, trajectory_id, node_id, scene_id, scene_dir),
                "scene_dir": scene_dir or None,
                "scene_root": scene_dir or None,
                "identity_id": identity_id or "unknown_identity",
                "trajectory_id": trajectory_id or "unknown_trajectory",
                "node_id": node_id or "unknown_node",
                "split_names": set(),
                "camera_ids": set(),
                "sample_ids": [],
                "sample_count": 0,
                "timestamp_us_values": set(),
                "mask_gt_available_count": 0,
                "no_mask_sample_count": 0,
                "sidecar_complete_count": 0,
            },
        )
        rec["sample_count"] += 1
        if split_name:
            rec["split_names"].add(split_name)
        if camera_id:
            rec["camera_ids"].add(camera_id)
        if sample_id:
            rec["sample_ids"].append(sample_id)
        if timestamp_us is not None:
            rec["timestamp_us_values"].add(timestamp_us)
        if mask_gt_available:
            rec["mask_gt_available_count"] += 1
            total_mask_gt_available_count += 1
        else:
            rec["no_mask_sample_count"] += 1
            total_no_mask_sample_count += 1
        if sidecar_complete:
            rec["sidecar_complete_count"] += 1

    scene_entries: list[dict[str, Any]] = []
    for key in sorted(per_scene.keys()):
        rec = per_scene[key]
        membership_entry = _as_dict(membership_index.get(key))
        obs_entry = _as_dict(obs_index.get(key))
        split_names = sorted({str(v).strip() for v in rec.get("split_names", set()) if str(v).strip()})
        camera_ids = sorted({str(v).strip() for v in rec.get("camera_ids", set()) if str(v).strip()})
        sample_ids = [str(v).strip() for v in _as_list(rec.get("sample_ids")) if str(v).strip()]
        scene_entries.append(
            {
                "scene_id": rec.get("scene_id"),
                "scene_key": rec.get("scene_key"),
                "scene_dir": rec.get("scene_dir"),
                "scene_root": rec.get("scene_root"),
                "identity_id": rec.get("identity_id"),
                "trajectory_id": rec.get("trajectory_id"),
                "node_id": rec.get("node_id"),
                "split_names": split_names,
                "sample_count": int(rec.get("sample_count") or 0),
                "camera_ids": camera_ids,
                "timestamp_count": len(_as_list(list(rec.get("timestamp_us_values", set())))),
                "mask_gt_available_count": int(rec.get("mask_gt_available_count") or 0),
                "no_mask_sample_count": int(rec.get("no_mask_sample_count") or 0),
                "sidecar_complete_count": int(rec.get("sidecar_complete_count") or 0),
                "sample_id_order_hash": _hash_text_parts(sample_ids),
                "sample_ids_sorted_hash": _hash_text_parts(sorted(sample_ids)),
                "source_scene_membership_present": bool(membership_entry),
                "source_scene_observation_present": bool(obs_entry),
            }
        )

    scene_keys_sorted_hash = _hash_text_parts([str(_as_dict(x).get("scene_key") or "") for x in scene_entries])
    scene_split_membership_hash = str(dataset_index_manifest.get("scene_split_membership_hash") or "").strip() or _hash_text_parts(
        [
            json.dumps(
                {
                    "scene_key": str(_as_dict(entry).get("scene_key") or ""),
                    "split_names": _as_list(_as_dict(entry).get("split_names")),
                    "sample_count": _parse_int_or_none(_as_dict(entry).get("sample_count")) or 0,
                },
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
            for entry in scene_entries
        ]
    )
    indexed_scene_count = _parse_int_or_none(dataset_index_manifest.get("scene_count"))
    if indexed_scene_count is None:
        indexed_scene_count = len(scene_entries)
    scene_root_count = len(scene_observations)
    return {
        "schema_version": "carla_air_existing_scene_index_bridge_manifest_v1",
        "run_id": run_id,
        "starts_runtime": False,
        "writes_scene_outputs": False,
        "non_promotion": True,
        "full_v1_live_dataset_ready": False,
        "scene_root_count": scene_root_count,
        "indexed_scene_count": indexed_scene_count,
        "sample_count": len(samples),
        "scene_entries": scene_entries,
        "scene_keys_sorted_hash": scene_keys_sorted_hash,
        "scene_split_membership_hash": scene_split_membership_hash,
        "mask_gt_available_count": total_mask_gt_available_count,
        "no_mask_sample_count": total_no_mask_sample_count,
    }


def _build_capture_task_alignment_summary(samples: list[dict[str, Any]], capture_tasks: list[dict[str, Any]]) -> dict[str, Any]:
    sample_with_capture_task_count = 0
    sample_with_trajectory_node_camera_bridge_count = 0
    bridge_status_counts: dict[str, int] = {}
    planned_candidate_count_total = 0
    planned_candidate_identity_ids: set[str] = set()
    for sample in samples:
        alignment = _as_dict(sample.get("plan_alignment"))
        if alignment.get("capture_task_in_plan") is True:
            sample_with_capture_task_count += 1
        bridge = _as_dict(sample.get("capture_matrix_bridge"))
        if alignment.get("trajectory_node_camera_in_capture_matrix") is True or bridge.get("trajectory_node_camera_in_plan") is True:
            sample_with_trajectory_node_camera_bridge_count += 1
        status = str(alignment.get("capture_matrix_bridge_status") or bridge.get("bridge_status") or "").strip()
        if status:
            bridge_status_counts[status] = bridge_status_counts.get(status, 0) + 1
        candidate_count = _positive_int(alignment.get("planned_capture_task_candidate_count"))
        if candidate_count is None:
            candidate_count = _positive_int(bridge.get("planned_capture_task_candidate_count")) or 0
        planned_candidate_count_total += int(candidate_count)
        for identity_id in _as_list(
            alignment.get("planned_identity_ids_for_trajectory_node_camera")
            or bridge.get("planned_identity_ids_for_trajectory_node_camera")
        ):
            text = str(identity_id or "").strip()
            if text:
                planned_candidate_identity_ids.add(text)
    sample_count = len(samples)
    return {
        "capture_task_count": len(capture_tasks),
        "sample_count": sample_count,
        "sample_with_capture_task_count": sample_with_capture_task_count,
        "sample_without_capture_task_count": sample_count - sample_with_capture_task_count,
        "sample_with_trajectory_node_camera_bridge_count": sample_with_trajectory_node_camera_bridge_count,
        "sample_without_trajectory_node_camera_bridge_count": sample_count - sample_with_trajectory_node_camera_bridge_count,
        "planned_capture_task_candidate_reference_count": planned_candidate_count_total,
        "planned_identity_ids_seen_in_capture_matrix_bridge": sorted(planned_candidate_identity_ids),
        "capture_matrix_bridge_status_counts": dict(sorted(bridge_status_counts.items())),
        "bridge_semantics": "trajectory/node/camera bridge is not an exact capture_task_id match and does not rewrite observed identity",
    }


def _build_dataset_run_contract_summary(
    *,
    samples: list[dict[str, Any]],
    scene_observations: list[dict[str, Any]],
    capture_tasks: list[dict[str, Any]],
    identities: list[dict[str, Any]],
    splits: dict[str, Any],
) -> dict[str, Any]:
    sample_count = len(samples)
    scene_count = len(scene_observations)
    split_distribution = {
        str(name): len(_as_list(sample_ids))
        for name, sample_ids in _as_dict(splits.get("splits")).items()
        if str(name).strip()
    }

    sample_with_capture_task_count = 0
    sample_with_trajectory_node_camera_bridge_count = 0
    planned_capture_task_candidate_reference_count = 0
    strict_matrix_entry_sample_count = 0
    legacy_or_observed_scene_passthrough_count = 0
    mask_gt_available_count = 0
    sidecar_complete_count = 0
    sidecar_modalities = ["rgb", "depth", "semantic", "instance", "pose", "calib"]
    sidecar_missing_count_by_modality = {key: 0 for key in sidecar_modalities}
    strict_planned_identity_sample_count = 0
    observed_passthrough_identity_sample_count = 0

    for sample in samples:
        alignment = _as_dict(sample.get("plan_alignment"))
        refs = _as_dict(sample.get("refs"))
        mask_gt_obj = _as_dict(sample.get("mask_gt"))
        if alignment.get("capture_task_in_plan") is True:
            sample_with_capture_task_count += 1
        bridge = _as_dict(sample.get("capture_matrix_bridge"))
        if alignment.get("trajectory_node_camera_in_capture_matrix") is True or bridge.get("trajectory_node_camera_in_plan") is True:
            sample_with_trajectory_node_camera_bridge_count += 1
        candidate_count = _positive_int(alignment.get("planned_capture_task_candidate_count"))
        if candidate_count is None:
            candidate_count = _positive_int(bridge.get("planned_capture_task_candidate_count")) or 0
        planned_capture_task_candidate_reference_count += int(candidate_count)
        if alignment.get("matrix_entry_in_plan") is True:
            strict_matrix_entry_sample_count += 1
        identity_matches = alignment.get("observed_identity_matches_planned")
        if identity_matches is None:
            identity_matches = alignment.get("planned_identity_match")
        if identity_matches is None:
            identity_matches = alignment.get("identity_in_plan")
        identity_matches = identity_matches is True
        if alignment.get("legacy_or_observed_scene_passthrough") is True:
            legacy_or_observed_scene_passthrough_count += 1
        if identity_matches:
            strict_planned_identity_sample_count += 1
        else:
            observed_passthrough_identity_sample_count += 1
        if mask_gt_obj.get("present") is True:
            mask_gt_available_count += 1

        has_all_sidecars = True
        for key in sidecar_modalities:
            if refs.get(key):
                continue
            sidecar_missing_count_by_modality[key] += 1
            has_all_sidecars = False
        if has_all_sidecars:
            sidecar_complete_count += 1

    planned_identity_ids = sorted(_plan_id_set(identities, "identity_id"))
    observed_identity_ids = sorted(
        {
            str(sample.get("identity_id") or "").strip()
            for sample in samples
            if str(sample.get("identity_id") or "").strip()
        }
    )
    identity_mismatch_count = sample_count - strict_planned_identity_sample_count

    return {
        "schema_version": "carla_air_dataset_run_contract_summary_v1",
        "sample_count": sample_count,
        "scene_count": scene_count,
        "split_distribution": split_distribution,
        "capture_task_count": len(capture_tasks),
        "sample_with_capture_task_count": sample_with_capture_task_count,
        "sample_without_capture_task_count": sample_count - sample_with_capture_task_count,
        "sample_with_trajectory_node_camera_bridge_count": sample_with_trajectory_node_camera_bridge_count,
        "sample_without_trajectory_node_camera_bridge_count": sample_count - sample_with_trajectory_node_camera_bridge_count,
        "planned_capture_task_candidate_reference_count": planned_capture_task_candidate_reference_count,
        "strict_matrix_entry_sample_count": strict_matrix_entry_sample_count,
        "legacy_or_observed_scene_passthrough_count": legacy_or_observed_scene_passthrough_count,
        "mask_gt_available_count": mask_gt_available_count,
        "no_mask_sample_count": sample_count - mask_gt_available_count,
        "sidecar_complete_count": sidecar_complete_count,
        "sidecar_complete_fraction": (sidecar_complete_count / sample_count) if sample_count else 0.0,
        "sidecar_missing_count_by_modality": sidecar_missing_count_by_modality,
        "planned_identity_ids": planned_identity_ids,
        "observed_identity_ids": observed_identity_ids,
        "identity_mismatch_count": identity_mismatch_count,
        "strict_planned_identity_sample_count": strict_planned_identity_sample_count,
        "observed_passthrough_identity_sample_count": observed_passthrough_identity_sample_count,
        "starts_runtime": False,
        "writes_scene_outputs": False,
        "non_promotion": True,
        "full_v1_live_dataset_ready": False,
    }


def _build_no_mask_non_promotion_manifest(
    *,
    run_id: str,
    samples: list[dict[str, Any]],
    dataset_run_contract_summary: dict[str, Any],
    dataset_manifest_outputs: dict[str, Any],
) -> dict[str, Any]:
    sample_count = len(samples)
    mask_gt_available_count = 0
    for sample in samples:
        mask_gt = _as_dict(_as_dict(sample).get("mask_gt"))
        if str(mask_gt.get("availability") or "").strip() == "available":
            mask_gt_available_count += 1
    no_mask_sample_count = sample_count - mask_gt_available_count
    digest_payload = {
        "run_id": run_id,
        "sample_count": sample_count,
        "mask_gt_available_count": mask_gt_available_count,
        "no_mask_sample_count": no_mask_sample_count,
        "starts_runtime": False,
        "writes_scene_outputs": False,
        "non_promotion": True,
        "full_v1_live_dataset_ready": False,
        "proxy_candidate_pseudo_legacy_not_promoted_to_mask_gt": True,
        "legacy_masks_gt_directory_is_not_formal_mask_gt_without_evidence": True,
        "trusted_mask_gt_requires_explicit_formal_evidence": True,
    }
    manifest = {
        "schema_version": "carla_air_no_mask_non_promotion_manifest_v1",
        "run_id": run_id,
        "starts_runtime": False,
        "writes_scene_outputs": False,
        "non_promotion": True,
        "full_v1_live_dataset_ready": False,
        "sample_count": sample_count,
        "mask_gt_available_count": mask_gt_available_count,
        "no_mask_sample_count": no_mask_sample_count,
        "policy": {
            "no_mask_samples_allowed_in_index": True,
            "mask_gt_availability_unavailable_is_not_candidate_proxy_pseudo_promotion": True,
            "proxy_candidate_pseudo_legacy_not_promoted_to_mask_gt": True,
            "legacy_masks_gt_directory_is_not_formal_mask_gt_without_evidence": True,
            "trusted_mask_gt_requires_explicit_formal_evidence": True,
        },
        "cross_checks": {
            "dataset_manifest_outputs_dataset_samples_jsonl": str(dataset_manifest_outputs.get("dataset_samples_jsonl") or ""),
            "dataset_manifest_outputs_dataset_manifest_json": str(dataset_manifest_outputs.get("dataset_manifest_json") or ""),
            "dataset_run_contract_summary_sample_count": int(dataset_run_contract_summary.get("sample_count") or 0),
            "dataset_run_contract_summary_mask_gt_available_count": int(
                dataset_run_contract_summary.get("mask_gt_available_count") or 0
            ),
            "dataset_run_contract_summary_no_mask_sample_count": int(
                dataset_run_contract_summary.get("no_mask_sample_count") or 0
            ),
        },
    }
    manifest["stable_hashes"] = {
        "core_digest_sha256": _canonical_json_sha256(digest_payload),
        "manifest_digest_sha256": _canonical_json_sha256(manifest),
    }
    return manifest


def _build_dataset_gap_manifest(
    *,
    run_id: str,
    samples: list[dict[str, Any]],
    dataset_run_contract_summary: dict[str, Any],
) -> dict[str, Any]:
    summary = _as_dict(dataset_run_contract_summary)
    sample_count = int(summary.get("sample_count") or len(samples))
    scene_count = int(summary.get("scene_count") or 0)
    split_distribution = _as_dict(summary.get("split_distribution"))
    planned_identity_ids = sorted({str(v).strip() for v in _as_list(summary.get("planned_identity_ids")) if str(v).strip()})
    observed_identity_ids = sorted({str(v).strip() for v in _as_list(summary.get("observed_identity_ids")) if str(v).strip()})
    identity_mismatch_count = int(summary.get("identity_mismatch_count") or 0)
    strict_planned_identity_sample_count = int(summary.get("strict_planned_identity_sample_count") or 0)
    observed_passthrough_identity_sample_count = int(summary.get("observed_passthrough_identity_sample_count") or 0)
    mask_gt_available_count = int(summary.get("mask_gt_available_count") or 0)
    no_mask_sample_count = int(summary.get("no_mask_sample_count") or (sample_count - mask_gt_available_count))
    sidecar_complete_count = int(summary.get("sidecar_complete_count") or 0)
    sidecar_incomplete_sample_count = sample_count - sidecar_complete_count
    sidecar_missing_count_by_modality = _as_dict(summary.get("sidecar_missing_count_by_modality"))
    formal_ready_sample_count = 0
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
            str(mask_gt_obj.get("availability") or "").strip() == "available"
            and has_all_sidecars
            and identity_matches is True
        ):
            formal_ready_sample_count += 1

    core_payload = {
        "run_id": run_id,
        "starts_runtime": False,
        "writes_scene_outputs": False,
        "non_promotion": True,
        "full_v1_live_dataset_ready": False,
        "sample_count": sample_count,
        "scene_count": scene_count,
        "split_distribution": split_distribution,
        "planned_identity_ids": planned_identity_ids,
        "observed_identity_ids": observed_identity_ids,
        "identity_mismatch_count": identity_mismatch_count,
        "strict_planned_identity_sample_count": strict_planned_identity_sample_count,
        "observed_passthrough_identity_sample_count": observed_passthrough_identity_sample_count,
        "mask_gt_available_count": mask_gt_available_count,
        "no_mask_sample_count": no_mask_sample_count,
        "sidecar_complete_count": sidecar_complete_count,
        "sidecar_incomplete_sample_count": sidecar_incomplete_sample_count,
        "sidecar_missing_count_by_modality": sidecar_missing_count_by_modality,
        "gap_counts": {
            "no_mask_non_promotion_sample_count": no_mask_sample_count,
            "sidecar_incomplete_sample_count": sidecar_incomplete_sample_count,
            "identity_passthrough_mismatch_sample_count": observed_passthrough_identity_sample_count,
            "formal_ready_sample_count": formal_ready_sample_count,
        },
        "gap_policy": {
            "sidecar_missing_is_not_mask_gt_failure": True,
            "no_mask_allowed_in_index": True,
            "identity_passthrough_not_live_identity_evidence": True,
            "proxy_candidate_pseudo_legacy_not_promoted_to_mask_gt": True,
        },
    }
    manifest = {
        "schema_version": "carla_air_dataset_gap_manifest_v1",
        **core_payload,
        "stable_hashes": {
            "core_payload_sha256": _canonical_json_sha256(core_payload),
        },
    }
    return manifest


def _build_scene_qualification_summary(scene_observations: list[dict[str, Any]], samples: list[dict[str, Any]]) -> dict[str, Any]:
    sample_scene_keys: dict[str, dict[str, Any]] = {}
    for sample in samples:
        scene_id = str(sample.get("scene_id") or "").strip()
        source = _as_dict(sample.get("source"))
        scene_dir = str(source.get("scene_dir") or "").strip()
        key = _scene_key_for_fields(
            str(sample.get("identity_id") or "").strip(),
            str(sample.get("trajectory_id") or "").strip(),
            str(sample.get("node_id") or "").strip(),
            scene_id,
            scene_dir,
        )
        rec = sample_scene_keys.setdefault(key, {"splits": set(), "sample_count": 0})
        split_name = str(sample.get("split") or "").strip() or "unknown"
        rec["splits"].add(split_name)
        rec["sample_count"] += 1

    multi_sample_scene_key_count = sum(1 for rec in sample_scene_keys.values() if int(rec["sample_count"]) > 1)
    cross_split_scene_key_count = sum(1 for rec in sample_scene_keys.values() if len(rec["splits"]) > 1)
    scene_key_split_collisions = [
        {"scene_key": scene_key, "splits": sorted(rec["splits"]), "sample_count": int(rec["sample_count"])}
        for scene_key, rec in sorted(sample_scene_keys.items())
        if len(rec["splits"]) > 1
    ]
    readiness_counts: dict[str, int] = {}
    qualified_count = 0
    for obs in scene_observations:
        readiness = _as_dict(_as_dict(obs).get("readiness"))
        status = str(readiness.get("status") or "unknown").strip() or "unknown"
        readiness_counts[status] = readiness_counts.get(status, 0) + 1
        if _as_dict(_as_dict(obs).get("scene_qualification")).get("minimum_index_artifacts_ready") is True:
            qualified_count += 1

    return {
        "scene_count": len(scene_observations),
        "qualified_no_mask_index_scene_count": qualified_count,
        "readiness_status_counts": readiness_counts,
        "sample_scene_key_count": len(sample_scene_keys),
        "multi_sample_scene_key_count": multi_sample_scene_key_count,
        "duplicate_scene_key_count": multi_sample_scene_key_count,
        "duplicate_scene_key_count_semantics": "compatibility alias for multi_sample_scene_key_count; not a duplicate-scene error by itself",
        "cross_split_scene_key_count": cross_split_scene_key_count,
        "has_cross_split_scene_key_conflict": cross_split_scene_key_count > 0,
        "scene_key_split_collisions": scene_key_split_collisions,
        "minimum_index_artifacts_contract": "rgb+pose_or_calib+frame_rows_valid_for_index",
        "legacy_proxy_candidate_never_promoted_to_mask_gt": True,
        "formal_mask_gt_is_evidence_driven_only": True,
    }


def _build_scene_membership_manifest(
    *,
    run_id: str,
    run_dir: Path,
    scene_observations: list[dict[str, Any]],
    samples: list[dict[str, Any]],
    capture_tasks: list[dict[str, Any]],
    splits: dict[str, Any],
    scene_qualification_summary: dict[str, Any],
) -> dict[str, Any]:
    split_distribution = {
        str(name): len(_as_list(sample_ids))
        for name, sample_ids in _as_dict(splits.get("splits")).items()
        if str(name).strip()
    }
    split_strategy = str(splits.get("split_strategy") or "").strip() or None
    not_random_frame_split = bool(splits.get("not_random_frame_split")) if "not_random_frame_split" in splits else None

    planned_identity_ids_by_tnc: dict[tuple[str, str, str], set[str]] = {}
    planned_tnc_keys: set[tuple[str, str, str]] = set()
    for task in capture_tasks:
        task_obj = _as_dict(task)
        trajectory_id = str(task_obj.get("trajectory_id") or "").strip()
        node_id = str(task_obj.get("node_id") or "").strip()
        camera_id = str(task_obj.get("camera_id") or "").strip()
        identity_id = str(task_obj.get("identity_id") or "").strip()
        if trajectory_id and node_id and camera_id:
            planned_tnc_keys.add((trajectory_id, node_id, camera_id))
            if identity_id:
                planned_identity_ids_by_tnc.setdefault((trajectory_id, node_id, camera_id), set()).add(identity_id)

    per_scene: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for sample in samples:
        source = _as_dict(sample.get("source"))
        scene_id = str(sample.get("scene_id") or source.get("scene_id") or "").strip()
        scene_dir = str(source.get("scene_dir") or "").strip()
        identity_id = str(sample.get("identity_id") or "unknown_identity").strip() or "unknown_identity"
        trajectory_id = str(sample.get("trajectory_id") or "unknown_trajectory").strip() or "unknown_trajectory"
        node_id = str(sample.get("node_id") or "unknown_node").strip() or "unknown_node"
        key = (identity_id, trajectory_id, node_id, scene_id or scene_dir)
        rec = per_scene.setdefault(
            key,
            {
                "scene_id": scene_id or (scene_dir.split("/")[-1] if scene_dir else ""),
                "scene_dir": scene_dir or None,
                "scene_root": scene_dir or None,
                "identity_id": identity_id,
                "trajectory_id": trajectory_id,
                "node_id": node_id,
                "sample_count": 0,
                "camera_ids": set(),
                "split_names": set(),
                "timestamp_summary": {
                    "count_with_timestamp_us": 0,
                    "count_with_frame_index": 0,
                    "min_timestamp_us": None,
                    "max_timestamp_us": None,
                    "min_frame_index": None,
                    "max_frame_index": None,
                },
                "capture_task_alignment": {
                    "sample_with_capture_task_count": 0,
                    "sample_without_capture_task_count": 0,
                    "sample_with_trajectory_node_camera_bridge_count": 0,
                    "sample_without_trajectory_node_camera_bridge_count": 0,
                    "planned_capture_task_candidate_reference_count": 0,
                    "capture_task_ids": set(),
                    "planned_capture_task_candidate_ids": set(),
                    "trajectory_node_camera_plan_match_count": 0,
                    "trajectory_node_camera_plan_missing_count": 0,
                    "identity_exact_match_count": 0,
                    "identity_passthrough_mismatch_count": 0,
                    "observed_identity_ids": set(),
                    "planned_identity_ids_for_trajectory_node_camera": set(),
                },
            },
        )
        rec["sample_count"] += 1
        cam_id = str(sample.get("camera_id") or "").strip()
        if cam_id:
            rec["camera_ids"].add(cam_id)
        split_name = str(sample.get("split") or "").strip() or "unknown"
        rec["split_names"].add(split_name)
        ts_obj = _as_dict(sample.get("timestamp"))
        ts_us = _parse_int_or_none(ts_obj.get("timestamp_us"))
        frame_idx = _parse_int_or_none(ts_obj.get("frame_index"))
        ts_summary = _as_dict(rec.get("timestamp_summary"))
        if ts_us is not None:
            ts_summary["count_with_timestamp_us"] = int(ts_summary.get("count_with_timestamp_us") or 0) + 1
            min_ts = _parse_int_or_none(ts_summary.get("min_timestamp_us"))
            max_ts = _parse_int_or_none(ts_summary.get("max_timestamp_us"))
            ts_summary["min_timestamp_us"] = ts_us if min_ts is None else min(min_ts, ts_us)
            ts_summary["max_timestamp_us"] = ts_us if max_ts is None else max(max_ts, ts_us)
        if frame_idx is not None:
            ts_summary["count_with_frame_index"] = int(ts_summary.get("count_with_frame_index") or 0) + 1
            min_frame = _parse_int_or_none(ts_summary.get("min_frame_index"))
            max_frame = _parse_int_or_none(ts_summary.get("max_frame_index"))
            ts_summary["min_frame_index"] = frame_idx if min_frame is None else min(min_frame, frame_idx)
            ts_summary["max_frame_index"] = frame_idx if max_frame is None else max(max_frame, frame_idx)
        rec["timestamp_summary"] = ts_summary
        alignment = _as_dict(sample.get("plan_alignment"))
        capture_task_alignment = _as_dict(rec.get("capture_task_alignment"))
        capture_task_in_plan = alignment.get("capture_task_in_plan") is True
        if capture_task_in_plan:
            capture_task_alignment["sample_with_capture_task_count"] = (
                int(capture_task_alignment.get("sample_with_capture_task_count") or 0) + 1
            )
        else:
            capture_task_alignment["sample_without_capture_task_count"] = (
                int(capture_task_alignment.get("sample_without_capture_task_count") or 0) + 1
            )
        capture_task_id = str(alignment.get("capture_task_id") or "").strip()
        if capture_task_id:
            capture_task_alignment.setdefault("capture_task_ids", set()).add(capture_task_id)
        bridge = _as_dict(sample.get("capture_matrix_bridge"))
        if bridge.get("trajectory_node_camera_in_plan") is True or alignment.get("trajectory_node_camera_in_capture_matrix") is True:
            capture_task_alignment["sample_with_trajectory_node_camera_bridge_count"] = (
                int(capture_task_alignment.get("sample_with_trajectory_node_camera_bridge_count") or 0) + 1
            )
        else:
            capture_task_alignment["sample_without_trajectory_node_camera_bridge_count"] = (
                int(capture_task_alignment.get("sample_without_trajectory_node_camera_bridge_count") or 0) + 1
            )
        candidate_count = _positive_int(alignment.get("planned_capture_task_candidate_count"))
        if candidate_count is None:
            candidate_count = _positive_int(bridge.get("planned_capture_task_candidate_count")) or 0
        capture_task_alignment["planned_capture_task_candidate_reference_count"] = (
            int(capture_task_alignment.get("planned_capture_task_candidate_reference_count") or 0) + int(candidate_count)
        )
        for candidate_id in _as_list(
            alignment.get("planned_capture_task_candidate_ids") or bridge.get("planned_capture_task_candidate_ids")
        ):
            candidate_text = str(candidate_id or "").strip()
            if candidate_text:
                capture_task_alignment.setdefault("planned_capture_task_candidate_ids", set()).add(candidate_text)
        tnc_key = (trajectory_id, node_id, cam_id)
        tnc_in_plan = tnc_key in planned_tnc_keys
        if tnc_in_plan:
            capture_task_alignment["trajectory_node_camera_plan_match_count"] = (
                int(capture_task_alignment.get("trajectory_node_camera_plan_match_count") or 0) + 1
            )
        else:
            capture_task_alignment["trajectory_node_camera_plan_missing_count"] = (
                int(capture_task_alignment.get("trajectory_node_camera_plan_missing_count") or 0) + 1
            )
        observed_identity_id = str(sample.get("identity_id") or "").strip()
        if observed_identity_id:
            capture_task_alignment.setdefault("observed_identity_ids", set()).add(observed_identity_id)
        planned_ids = planned_identity_ids_by_tnc.get((trajectory_id, node_id, cam_id), set())
        for planned_id in planned_ids:
            capture_task_alignment.setdefault("planned_identity_ids_for_trajectory_node_camera", set()).add(planned_id)
        if tnc_in_plan and observed_identity_id and observed_identity_id in planned_ids:
            capture_task_alignment["identity_exact_match_count"] = (
                int(capture_task_alignment.get("identity_exact_match_count") or 0) + 1
            )
        elif tnc_in_plan and observed_identity_id and planned_ids and observed_identity_id not in planned_ids:
            capture_task_alignment["identity_passthrough_mismatch_count"] = (
                int(capture_task_alignment.get("identity_passthrough_mismatch_count") or 0) + 1
            )
        rec["capture_task_alignment"] = capture_task_alignment

    observation_index: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for obs in scene_observations:
        obj = _as_dict(obs)
        obs_identity_id = str(obj.get("identity_id") or "unknown_identity").strip() or "unknown_identity"
        obs_trajectory_id = str(obj.get("trajectory_id") or "unknown_trajectory").strip() or "unknown_trajectory"
        obs_node_id = str(obj.get("node_id") or "unknown_node").strip() or "unknown_node"
        obs_scene_id = str(obj.get("scene_id") or "").strip()
        obs_scene_dir = str(obj.get("scene_dir") or "").strip()
        observation_index[(obs_identity_id, obs_trajectory_id, obs_node_id, obs_scene_id or obs_scene_dir)] = obj

    scene_entries: list[dict[str, Any]] = []
    for key in sorted(per_scene.keys()):
        rec = per_scene[key]
        obs_obj = _as_dict(observation_index.get(key))
        split_names = sorted({str(v).strip() for v in rec.get("split_names", set()) if str(v).strip()})
        entry = {
            "scene_id": rec.get("scene_id"),
            "scene_dir": rec.get("scene_dir"),
            "scene_root": rec.get("scene_root"),
            "identity_id": rec.get("identity_id"),
            "trajectory_id": rec.get("trajectory_id"),
            "node_id": rec.get("node_id"),
            "split_names": split_names,
            "split": split_names[0] if len(split_names) == 1 else None,
            "sample_count": int(rec.get("sample_count") or 0),
            "camera_ids": sorted({str(v).strip() for v in rec.get("camera_ids", set()) if str(v).strip()}),
            "timestamp_summary": _as_dict(rec.get("timestamp_summary")),
            "mask_gt_available": False,
            "mask_gt_available_derived_from_formal_gt_evidence": False,
            "non_promotion": True,
            "scene_qualification_minimum_index_artifacts_ready": _as_dict(obs_obj.get("scene_qualification")).get(
                "minimum_index_artifacts_ready"
            )
            is True,
        }
        capture_task_alignment = _as_dict(rec.get("capture_task_alignment"))
        sample_count = int(rec.get("sample_count") or 0)
        trajectory_node_camera_plan_match_count = int(capture_task_alignment.get("trajectory_node_camera_plan_match_count") or 0)
        entry["capture_task_alignment"] = {
            "sample_count": sample_count,
            "sample_with_capture_task_count": int(capture_task_alignment.get("sample_with_capture_task_count") or 0),
            "sample_without_capture_task_count": int(capture_task_alignment.get("sample_without_capture_task_count") or 0),
            "sample_with_trajectory_node_camera_bridge_count": int(
                capture_task_alignment.get("sample_with_trajectory_node_camera_bridge_count") or 0
            ),
            "sample_without_trajectory_node_camera_bridge_count": int(
                capture_task_alignment.get("sample_without_trajectory_node_camera_bridge_count") or 0
            ),
            "planned_capture_task_candidate_reference_count": int(
                capture_task_alignment.get("planned_capture_task_candidate_reference_count") or 0
            ),
            "capture_task_ids": sorted(
                {str(v).strip() for v in capture_task_alignment.get("capture_task_ids", set()) if str(v).strip()}
            ),
            "planned_capture_task_candidate_ids": sorted(
                {
                    str(v).strip()
                    for v in capture_task_alignment.get("planned_capture_task_candidate_ids", set())
                    if str(v).strip()
                }
            ),
            "trajectory_node_camera_in_plan": trajectory_node_camera_plan_match_count > 0,
            "trajectory_node_camera_plan_match_count": trajectory_node_camera_plan_match_count,
            "trajectory_node_camera_plan_missing_count": int(
                capture_task_alignment.get("trajectory_node_camera_plan_missing_count") or 0
            ),
            "identity_exact_match_count": int(capture_task_alignment.get("identity_exact_match_count") or 0),
            "identity_passthrough_mismatch_count": int(
                capture_task_alignment.get("identity_passthrough_mismatch_count") or 0
            ),
            "observed_identity_ids": sorted(
                {str(v).strip() for v in capture_task_alignment.get("observed_identity_ids", set()) if str(v).strip()}
            ),
            "planned_identity_ids_for_trajectory_node_camera": sorted(
                {
                    str(v).strip()
                    for v in capture_task_alignment.get("planned_identity_ids_for_trajectory_node_camera", set())
                    if str(v).strip()
                }
            ),
            "legacy_or_observed_scene_passthrough_allowed_for_no_mask_index": True,
            "non_promotion": True,
            "full_v1_live_dataset_ready": False,
        }
        scene_entries.append(entry)

    return {
        "schema_version": "carla_air_scene_membership_manifest_v1",
        "generated_at": _now_iso(),
        "run_id": run_id,
        "run_dir": str(run_dir.resolve()),
        "starts_runtime": False,
        "writes_scene_outputs": False,
        "non_promotion": True,
        "full_v1_live_dataset_ready": False,
        "scene_count": len(scene_entries),
        "sample_count": len(samples),
        "split_distribution": split_distribution,
        "split_strategy": split_strategy,
        "not_random_frame_split": not_random_frame_split,
        "selected_split_strategy": split_strategy,
        "scene_entries": scene_entries,
        "cross_split_scene_key_count": int(scene_qualification_summary.get("cross_split_scene_key_count") or 0),
        "scene_key_split_collisions": _as_list(scene_qualification_summary.get("scene_key_split_collisions")),
    }


def _build_splits(samples: list[dict[str, Any]], plan: dict[str, Any]) -> dict[str, Any]:
    configured = _as_dict(plan.get("dataset_splits")) or _as_dict(_as_dict(plan.get("dataset")).get("splits"))
    split_names = _as_list(configured.get("names")) or DEFAULT_SPLIT_NAMES
    split_map: dict[str, list[str]] = {}
    for name in split_names:
        split_map[str(name)] = []
    for sample in samples:
        split_name = str(sample.get("split") or "train")
        split_map.setdefault(split_name, [])
        split_map[split_name].append(str(sample.get("sample_id") or ""))
    split_distribution = {str(name): len(_as_list(ids)) for name, ids in split_map.items()}
    split_strategy = str(configured.get("strategy") or "deployment_oriented_node_layout_v1")
    not_random_frame_split = bool(_as_dict(configured.get("policy")).get("not_random_frame_split", True))
    split_count = len(split_map)
    sample_count = len(samples)
    return {
        "schema_version": "carla_air_dataset_splits_v1",
        "split_strategy": split_strategy,
        "not_random_frame_split": not_random_frame_split,
        "split_count": split_count,
        "sample_count": sample_count,
        "split_distribution": split_distribution,
        "splits": split_map,
        "manifest": {
            "split_count": split_count,
            "sample_count": sample_count,
            "split_distribution": split_distribution,
            "not_random_frame_split": not_random_frame_split,
            "split_strategy": split_strategy,
        },
    }


def _build_deployment_episodes(plan: dict[str, Any]) -> list[dict[str, Any]]:
    episodes = _as_list(plan.get("deployment_episodes")) or _as_list(_as_dict(plan.get("deployment")).get("episodes"))
    out: list[dict[str, Any]] = []
    for item in episodes:
        if not isinstance(item, dict):
            continue
        split_value = item.get("split") or item.get("split_hint")
        raw_node_ids = _as_list(item.get("node_ids"))
        node_ids: list[str] = []
        for v in raw_node_ids:
            node = str(v or "").strip()
            if node:
                node_ids.append(node)
        primary_node_id = str(item.get("primary_node_id") or "").strip()
        if not primary_node_id and node_ids:
            primary_node_id = node_ids[0]
        node_id_value = str(item.get("node_id") or item.get("node") or "").strip()
        if not primary_node_id and node_id_value:
            primary_node_id = node_id_value
        if not node_id_value and primary_node_id:
            node_id_value = primary_node_id
        if not node_ids and node_id_value:
            node_ids = [node_id_value]
        node_pair_id = str(item.get("node_pair_id") or "").strip()
        if not node_pair_id and len(node_ids) >= 2:
            node_pair_id = f"{node_ids[0]}__{node_ids[1]}"
        camera_ids = [str(v).strip() for v in _as_list(item.get("camera_ids")) if str(v).strip()]
        camera_ids_by_node = _as_dict(item.get("camera_ids_by_node"))
        if (not camera_ids) and camera_ids_by_node:
            flattened: list[str] = []
            seen: set[str] = set()
            for node_cameras in camera_ids_by_node.values():
                for raw_cam in _as_list(node_cameras):
                    cam = str(raw_cam).strip()
                    if cam and cam not in seen:
                        flattened.append(cam)
                        seen.add(cam)
            camera_ids = flattened
        out.append(
            {
                "episode_id": item.get("episode_id") or item.get("id"),
                "split": split_value,
                "node_id": node_id_value or None,
                "node_ids": node_ids,
                "primary_node_id": primary_node_id or None,
                "node_pair_id": node_pair_id or None,
                "camera_ids": camera_ids,
                "camera_ids_by_node": camera_ids_by_node,
                "trajectory_ids": _as_list(item.get("trajectory_ids")),
                "timestamps": _as_dict(item.get("timestamps")),
            }
        )
    return out


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


def _timestamp_summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    frame_values: list[int] = []
    ts_us_values: list[int] = []
    unix_ns_values: list[int] = []
    iso_values: list[str] = []
    for sample in samples:
        ts = _as_dict(sample.get("timestamp"))
        frame_idx = _parse_int_or_none(ts.get("frame_index"))
        ts_us = _parse_int_or_none(ts.get("timestamp_us"))
        unix_ns = _parse_int_or_none(ts.get("unix_ns"))
        iso8601 = str(ts.get("iso8601") or "").strip()
        if frame_idx is not None:
            frame_values.append(frame_idx)
        if ts_us is not None:
            ts_us_values.append(ts_us)
        if unix_ns is not None:
            unix_ns_values.append(unix_ns)
        if iso8601:
            iso_values.append(iso8601)
    out: dict[str, Any] = {"count": len(samples)}
    if frame_values:
        out["frame_index_min"] = min(frame_values)
        out["frame_index_max"] = max(frame_values)
    if ts_us_values:
        out["timestamp_us_min"] = min(ts_us_values)
        out["timestamp_us_max"] = max(ts_us_values)
    if unix_ns_values:
        out["unix_ns_min"] = min(unix_ns_values)
        out["unix_ns_max"] = max(unix_ns_values)
    if iso_values:
        out["iso8601_min"] = min(iso_values)
        out["iso8601_max"] = max(iso_values)
    return out


def _with_episode_visibility(episodes: list[dict[str, Any]], samples: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    episodes_out: list[dict[str, Any]] = []
    sample_total = 0
    scene_union: set[str] = set()
    episode_scene_rows: list[str] = []
    episode_sample_order_rows: list[str] = []
    episode_sample_sorted_rows: list[str] = []
    for episode in episodes:
        filters = _episode_filter_sets(episode)
        matched = [sample for sample in samples if _sample_matches_episode(sample, filters)]
        scene_ids = sorted({str(_as_dict(sample.get("source")).get("scene_id") or sample.get("scene_id") or "").strip() for sample in matched if str(_as_dict(sample.get("source")).get("scene_id") or sample.get("scene_id") or "").strip()})
        sample_ids_ordered = [str(_as_dict(sample).get("sample_id") or "").strip() for sample in matched]
        sample_ids_ordered = [sid for sid in sample_ids_ordered if sid]
        sample_ids_sorted = sorted(set(sample_ids_ordered))
        node_ids_observed = sorted({str(sample.get("node_id") or "").strip() for sample in matched if str(sample.get("node_id") or "").strip()})
        trajectory_ids_observed = sorted({str(sample.get("trajectory_id") or "").strip() for sample in matched if str(sample.get("trajectory_id") or "").strip()})
        camera_ids_observed = sorted({str(sample.get("camera_id") or "").strip() for sample in matched if str(sample.get("camera_id") or "").strip()})
        split_distribution: dict[str, int] = {}
        for sample in matched:
            split_name = str(sample.get("split") or "unknown").strip() or "unknown"
            split_distribution[split_name] = split_distribution.get(split_name, 0) + 1
        sample_scene_visibility: dict[str, Any] = {
            "sample_count": len(matched),
            "scene_count": len(scene_ids),
            "scene_id_head": scene_ids[:20],
            "scene_id_sample_count": len(scene_ids),
            "configured_filters": {
                "split_names": sorted(filters["splits"]),
                "node_ids": sorted(filters["node_ids"]),
                "trajectory_ids": sorted(filters["trajectory_ids"]),
                "camera_ids": sorted(filters["camera_ids"]),
            },
            "node_ids_observed": node_ids_observed,
            "trajectory_ids_observed": trajectory_ids_observed,
            "camera_ids_observed": camera_ids_observed,
            "split_distribution": split_distribution,
            "coverage_gaps": {
                "missing_split_names": sorted(filters["splits"] - set(split_distribution.keys())),
                "missing_node_ids": sorted(filters["node_ids"] - set(node_ids_observed)),
                "missing_trajectory_ids": sorted(filters["trajectory_ids"] - set(trajectory_ids_observed)),
                "missing_camera_ids": sorted(filters["camera_ids"] - set(camera_ids_observed)),
                "has_missing_configured_filter_values": bool(
                    (filters["splits"] - set(split_distribution.keys()))
                    or (filters["node_ids"] - set(node_ids_observed))
                    or (filters["trajectory_ids"] - set(trajectory_ids_observed))
                    or (filters["camera_ids"] - set(camera_ids_observed))
                ),
            },
            "timestamp_summary": _timestamp_summary(matched),
            "scene_ids_sorted_hash": _hash_text_parts(scene_ids),
            "sample_id_order_hash": _hash_text_parts(sample_ids_ordered),
            "sample_ids_sorted_hash": _hash_text_parts(sample_ids_sorted),
            "first_sample_id": sample_ids_ordered[0] if sample_ids_ordered else None,
            "last_sample_id": sample_ids_ordered[-1] if sample_ids_ordered else None,
            "non_promotion": True,
            "mask_gt_available_count": 0,
            "full_v1_live_dataset_ready": False,
        }
        sample_scene_visibility["scene_ids"] = scene_ids
        ep_out = dict(episode)
        ep_out["sample_scene_visibility"] = sample_scene_visibility
        episodes_out.append(ep_out)
        sample_total += len(matched)
        scene_union.update(scene_ids)
        episode_id = str(episode.get("episode_id") or "").strip()
        episode_scene_rows.append(f"{episode_id}|{sample_scene_visibility['scene_ids_sorted_hash']}")
        episode_sample_order_rows.append(f"{episode_id}|{sample_scene_visibility['sample_id_order_hash']}")
        episode_sample_sorted_rows.append(f"{episode_id}|{sample_scene_visibility['sample_ids_sorted_hash']}")
    episode_with_visibility_gap_count = sum(
        1
        for episode in episodes_out
        if _as_dict(_as_dict(episode.get("sample_scene_visibility")).get("coverage_gaps")).get(
            "has_missing_configured_filter_values"
        )
        is True
    )
    episode_without_samples_count = sum(
        1 for episode in episodes_out if int(_as_dict(episode.get("sample_scene_visibility")).get("sample_count") or 0) == 0
    )
    top_summary = {
        "episode_count": len(episodes_out),
        "sample_count_total_from_episodes_policy": "sum_over_episodes_samples_may_repeat",
        "sample_count_total_from_episodes": sample_total,
        "scene_count_total_from_episodes_policy": "unique_scene_ids_across_episodes",
        "scene_count_total_from_episodes": len(scene_union),
        "episode_scene_visibility_hash": _hash_text_parts(episode_scene_rows),
        "episode_sample_order_hash": _hash_text_parts(episode_sample_order_rows),
        "episode_sample_sorted_hash": _hash_text_parts(episode_sample_sorted_rows),
        "episode_with_visibility_gap_count": episode_with_visibility_gap_count,
        "episode_without_samples_count": episode_without_samples_count,
        "non_promotion": True,
        "full_v1_live_dataset_ready": False,
    }
    return episodes_out, top_summary


def _build_deployment_episode_visibility_manifest(
    *,
    run_id: str,
    deployment_episodes: list[dict[str, Any]],
    deployment_episodes_summary: dict[str, Any],
    split_policy_summary: dict[str, Any],
    split_policy_digest: str,
) -> dict[str, Any]:
    episode_entries: list[dict[str, Any]] = []
    for episode in deployment_episodes:
        ep_obj = _as_dict(episode)
        episode_id = str(ep_obj.get("episode_id") or "").strip()
        visibility = _as_dict(ep_obj.get("sample_scene_visibility"))
        configured_filters = _as_dict(visibility.get("configured_filters"))
        episode_entries.append(
            {
                "episode_id": episode_id,
                "split": ep_obj.get("split"),
                "node_ids": _as_list(ep_obj.get("node_ids")),
                "node_pair_id": ep_obj.get("node_pair_id"),
                "primary_node_id": ep_obj.get("primary_node_id"),
                "camera_ids": _as_list(ep_obj.get("camera_ids")),
                "camera_ids_by_node": _as_dict(ep_obj.get("camera_ids_by_node")),
                "configured_filters": {
                    "split_names": _as_list(configured_filters.get("split_names")),
                    "node_ids": _as_list(configured_filters.get("node_ids")),
                    "trajectory_ids": _as_list(configured_filters.get("trajectory_ids")),
                    "camera_ids": _as_list(configured_filters.get("camera_ids")),
                },
                "sample_count": int(visibility.get("sample_count") or 0),
                "scene_count": int(visibility.get("scene_count") or 0),
                "scene_ids_sorted_hash": str(visibility.get("scene_ids_sorted_hash") or ""),
                "sample_id_order_hash": str(visibility.get("sample_id_order_hash") or ""),
                "sample_ids_sorted_hash": str(visibility.get("sample_ids_sorted_hash") or ""),
                "first_sample_id": visibility.get("first_sample_id"),
                "last_sample_id": visibility.get("last_sample_id"),
                "visibility_gap": _as_dict(visibility.get("coverage_gaps")),
                "non_promotion": True,
                "full_v1_live_dataset_ready": False,
            }
        )
    core_payload = {
        "schema_version": "carla_air_deployment_episode_visibility_manifest_v1",
        "run_id": run_id,
        "starts_runtime": False,
        "writes_scene_outputs": False,
        "non_promotion": True,
        "full_v1_live_dataset_ready": False,
        "episode_count": int(deployment_episodes_summary.get("episode_count") or len(deployment_episodes)),
        "sample_count_total_from_episodes": int(deployment_episodes_summary.get("sample_count_total_from_episodes") or 0),
        "scene_count_total_from_episodes": int(deployment_episodes_summary.get("scene_count_total_from_episodes") or 0),
        "episode_with_visibility_gap_count": int(
            deployment_episodes_summary.get("episode_with_visibility_gap_count") or 0
        ),
        "episode_without_samples_count": int(deployment_episodes_summary.get("episode_without_samples_count") or 0),
        "episode_scene_visibility_hash": str(deployment_episodes_summary.get("episode_scene_visibility_hash") or ""),
        "episode_sample_order_hash": str(deployment_episodes_summary.get("episode_sample_order_hash") or ""),
        "episode_sample_sorted_hash": str(deployment_episodes_summary.get("episode_sample_sorted_hash") or ""),
        "split_policy_summary": split_policy_summary,
        "split_policy_digest": str(split_policy_digest or ""),
        "episode_entries": episode_entries,
    }
    core_digest = _canonical_json_sha256(core_payload)
    manifest_payload = dict(core_payload)
    manifest_without_stable_hashes = dict(manifest_payload)
    manifest_without_stable_hashes.pop("stable_hashes", None)
    manifest_payload["stable_hashes"] = {
        "core_digest_sha256": core_digest,
        "manifest_digest_sha256": _canonical_json_sha256(manifest_without_stable_hashes),
    }
    return manifest_payload


def _sample_contract_template(mask_gt: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_SAMPLE,
        "sample_id": "<string>",
        "identity_id": "<string>",
        "trajectory_id": "<string>",
        "node_id": "<string>",
        "camera_id": "<string>",
        "rgb": "<path_or_uri_or_null>",
        "depth": "<path_or_uri_or_null>",
        "semantic": "<path_or_uri_or_null>",
        "instance": "<path_or_uri_or_null>",
        "pose": "<path_or_uri_or_null>",
        "calib": "<path_or_uri_or_null>",
        "identity": {"identity_id": "<string>"},
        "trajectory": {"trajectory_id": "<string>"},
        "camera_layout": {"layout_id": "<string_optional>", "node_id": "<string>", "camera_ids": ["<string>"]},
        "view": {"node_id": "<string>", "camera_id": "<string>"},
        "timestamp": {"frame_index": "<int_or_null>", "unix_ns": "<int_or_null>", "iso8601": "<string_or_null>"},
        "refs": {
            "rgb": "<path_or_uri_or_null>",
            "depth": "<path_or_uri_or_null>",
            "semantic": "<path_or_uri_or_null>",
            "instance": "<path_or_uri_or_null>",
            "pose": "<path_or_uri_or_null>",
            "calib": "<path_or_uri_or_null>",
        },
        "modality_availability": {
            "rgb": "<bool>",
            "depth": "<bool>",
            "semantic": "<bool>",
            "instance": "<bool>",
            "pose": "<bool>",
            "calib": "<bool>",
        },
        "plan_alignment": {
            "identity_in_plan": "<bool>",
            "trajectory_in_plan": "<bool>",
            "node_in_plan": "<bool>",
            "node_camera_in_plan": "<bool>",
            "matrix_entry_in_plan": "<bool>",
            "capture_task_in_plan": "<bool>",
            "capture_task_id": "<string_or_null>",
            "legacy_or_observed_scene_passthrough": "<bool>",
        },
        "mask_gt": {
            "policy_mode": mask_gt["mode"],
            "availability": "<available|unavailable|unknown>",
            "present": "<bool>",
            "source": "<ground_truth|none>",
            "is_mask_gt": "<bool>",
            "unavailable_reason": "<string_or_null>",
            "audit_state": "<verified|missing_formal_gt|unknown>",
            "non_gt_candidates_seen": "<bool>",
            "pseudo_or_candidate_never_mask_gt": True,
        },
        "mask_gt_audit": {
            "scene_probe_has_rgb_pose_calib": "<bool>",
            "formal_mask_gt_found": "<bool>",
            "formal_mask_gt_path": "<path_or_null>",
            "candidate_or_pseudo_or_proxy_paths": ["<path_optional>"],
            "contract_enforcement": "candidate/pseudo/proxy must not be promoted to mask_gt",
        },
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def _hash_text_parts(parts: list[str]) -> str:
    hasher = hashlib.sha256()
    for part in parts:
        hasher.update(part.encode("utf-8"))
        hasher.update(b"\n")
    return hasher.hexdigest()


def _canonical_json_sha256(payload: Any) -> str:
    canonical = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _build_scene_sample_entries_and_hashes(samples: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str, str]:
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
    scene_entries: list[dict[str, Any]] = []
    for scene_key in sorted(scene_membership):
        rec = scene_membership[scene_key]
        scene_sample_ids = [str(v).strip() for v in _as_list(rec.get("sample_ids")) if str(v).strip()]
        split_names = sorted({str(v).strip() for v in rec.get("split_names", set()) if str(v).strip()})
        camera_ids = sorted({str(v).strip() for v in rec.get("camera_ids", set()) if str(v).strip()})
        timestamp_us_sorted = sorted({str(v).strip() for v in rec.get("timestamps_us", set()) if str(v).strip()})
        scene_entries.append(
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
    scene_sample_index_hash = _hash_text_parts(
        [
            json.dumps(entry, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
            for entry in scene_entries
        ]
    )
    scene_keys_sorted_hash = _hash_text_parts([str(entry.get("scene_key") or "") for entry in scene_entries])
    return scene_entries, scene_sample_index_hash, scene_keys_sorted_hash


def _build_split_policy_summary(splits: dict[str, Any]) -> dict[str, Any]:
    split_map = _as_dict(splits.get("splits"))
    split_names = sorted(str(name).strip() for name in split_map.keys() if str(name).strip())
    return {
        "split_strategy": str(splits.get("split_strategy") or "").strip() or "deployment_oriented_node_layout_v1",
        "not_random_frame_split": bool(splits.get("not_random_frame_split", True)),
        "split_names": split_names,
        "split_count": len(split_names),
    }


def _build_dataset_index_manifest(
    run_id: str,
    run_dir: Path,
    dataset_samples_path: Path,
    dataset_splits_path: Path,
    dataset_manifest_path: Path,
    samples: list[dict[str, Any]],
    splits: dict[str, Any],
    mask_gt_available_count: int,
    dataset_run_contract_summary: dict[str, Any],
) -> dict[str, Any]:
    _scene_entries, scene_sample_index_hash, _scene_keys_sorted_hash = _build_scene_sample_entries_and_hashes(samples)
    sample_ids_ordered = [str(_as_dict(sample).get("sample_id") or "").strip() for sample in samples]
    non_empty_sample_ids_ordered = [sid for sid in sample_ids_ordered if sid]
    unique_sample_ids = set(non_empty_sample_ids_ordered)

    canonical_rows = [
        json.dumps(_as_dict(sample), ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        for sample in samples
    ]
    sample_content_hash = _hash_text_parts(canonical_rows)
    sample_id_order_hash = _hash_text_parts(non_empty_sample_ids_ordered)

    split_map = _as_dict(splits.get("splits"))
    split_distribution = {
        str(split_name): len(_as_list(split_ids))
        for split_name, split_ids in split_map.items()
        if str(split_name).strip()
    }
    split_id_set: set[str] = set()
    for split_ids in split_map.values():
        for sample_id in _as_list(split_ids):
            sid = str(sample_id or "").strip()
            if sid:
                split_id_set.add(sid)
    split_sample_id_set_hash = _hash_text_parts(sorted(split_id_set))

    missing_from_splits = sorted(unique_sample_ids - split_id_set)
    extra_in_splits = sorted(split_id_set - unique_sample_ids)
    sample_ids_sorted = sorted(unique_sample_ids)
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
    duplicate_sample_id_count = len(non_empty_sample_ids_ordered) - len(unique_sample_ids)
    missing_from_splits_count = len(missing_from_splits)
    extra_in_splits_count = len(extra_in_splits)
    sample_id_complete = len(non_empty_sample_ids_ordered) == len(samples)
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
    all_required_scene_membership_fields_present = bool(scene_split_membership_index) and all(
        required_scene_membership_fields.issubset(set(entry.keys()))
        for entry in scene_split_membership_index
    )
    strict_index_contract = {
        "schema_version": SCHEMA_INDEX_STRICT_CONTRACT,
        "sample_id_integrity_enforced": True,
        "sample_id_complete": bool(sample_id_complete),
        "duplicate_sample_id_count": int(duplicate_sample_id_count),
        "missing_from_splits_count": int(missing_from_splits_count),
        "extra_in_splits_count": int(extra_in_splits_count),
        "scene_split_membership_index_required": True,
        "scene_key_required": True,
        "scene_keys_sorted_hash_required": True,
        "scene_split_membership_hash_required": True,
        "all_required_scene_membership_fields_present": bool(all_required_scene_membership_fields_present),
        "non_promotion": True,
        "full_v1_live_dataset_ready": False,
        "mask_gt_available_count": int(mask_gt_available_count),
    }

    return {
        "schema_version": SCHEMA_INDEX_MANIFEST,
        "generated_at": _now_iso(),
        "run_id": run_id,
        "run_dir": str(run_dir.resolve()),
        "dataset_samples_jsonl": str(dataset_samples_path.resolve()),
        "dataset_splits_json": str(dataset_splits_path.resolve()),
        "dataset_manifest_json": str(dataset_manifest_path.resolve()),
        "sample_count": len(samples),
        "sample_id_count": len(non_empty_sample_ids_ordered),
        "sample_id_unique_count": len(unique_sample_ids),
        "sample_ordering_policy": "existing scene/root/frame order from indexed sample materialization",
        "sample_id_order_hash": sample_id_order_hash,
        "sample_content_hash": sample_content_hash,
        "split_sample_id_set_hash": split_sample_id_set_hash,
        "split_distribution": split_distribution,
        "sample_ids_sorted_hash": _hash_text_parts(sample_ids_sorted),
        "scene_count": len(scene_split_membership_index),
        "scene_ids_sorted_hash": _hash_text_parts([entry["scene_id"] for entry in scene_split_membership_index]),
        "scene_keys_sorted_hash": _hash_text_parts([entry["scene_key"] for entry in scene_split_membership_index]),
        "scene_sample_index_hash": scene_sample_index_hash,
        "scene_split_membership_hash": scene_split_membership_hash,
        "scene_split_membership_index": scene_split_membership_index,
        "duplicate_sample_id_count": int(duplicate_sample_id_count),
        "missing_from_splits_count": int(missing_from_splits_count),
        "extra_in_splits_count": int(extra_in_splits_count),
        "strict_index_contract": strict_index_contract,
        "non_promotion": True,
        "mask_gt_available_count": int(mask_gt_available_count),
        "full_v1_live_dataset_ready": False,
        "dataset_run_contract_summary": dataset_run_contract_summary,
        "first_sample_id": non_empty_sample_ids_ordered[0] if non_empty_sample_ids_ordered else None,
        "last_sample_id": non_empty_sample_ids_ordered[-1] if non_empty_sample_ids_ordered else None,
        "sample_id_head": non_empty_sample_ids_ordered[:5],
        "sample_id_tail": non_empty_sample_ids_ordered[-5:] if non_empty_sample_ids_ordered else [],
    }


def _build_scene_sample_index_manifest(
    *,
    run_id: str,
    run_dir: Path,
    samples: list[dict[str, Any]],
    splits: dict[str, Any],
    dataset_index_manifest: dict[str, Any],
) -> dict[str, Any]:
    split_distribution = {
        str(split_name): len(_as_list(split_ids))
        for split_name, split_ids in _as_dict(splits.get("splits")).items()
        if str(split_name).strip()
    }
    scene_entries, scene_sample_index_hash, scene_keys_sorted_hash = _build_scene_sample_entries_and_hashes(samples)
    dataset_index_scene_split_membership_hash = str(dataset_index_manifest.get("scene_split_membership_hash") or "")
    return {
        "schema_version": SCHEMA_SCENE_SAMPLE_INDEX_MANIFEST,
        "generated_at": _now_iso(),
        "run_id": run_id,
        "run_dir": str(run_dir.resolve()),
        "starts_runtime": False,
        "writes_scene_outputs": False,
        "non_promotion": True,
        "full_v1_live_dataset_ready": False,
        "sample_count": len(samples),
        "scene_count": len(scene_entries),
        "split_distribution": split_distribution,
        "scene_split_membership_hash": dataset_index_scene_split_membership_hash,
        "dataset_index_scene_split_membership_hash": dataset_index_scene_split_membership_hash,
        "scene_sample_index_hash": scene_sample_index_hash,
        "scene_keys_sorted_hash": scene_keys_sorted_hash,
        "notes": {
            "mask_policy": "no_mask samples are allowed and auditable in offline V1; no candidate/proxy/pseudo promotion.",
            "non_promotion_boundary": "offline index-only artifacts; no trusted mask_gt promotion, no runtime writes.",
        },
        "scene_entries": scene_entries,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build CARLA-Air Dataset Generation Pipeline V1 training index artifacts.")
    parser.add_argument("--dataset-plan", required=True, help="Path to dataset_plan.json")
    parser.add_argument("--scene-root", action="append", default=[], help="Optional scene root (repeatable).")
    parser.add_argument("--run-root", default=str(DEFAULT_RUN_ROOT), help="Output root; run data is written under <run-root>/<run_id>/.")
    parser.add_argument("--allow-nonlocal-out", action="store_true", help="Allow --run-root outside repository local/.")
    parser.add_argument("--allow-fail", action="store_true", help="Exit 0 even when blockers exist.")
    args = parser.parse_args()

    try:
        dataset_plan_path = _repo_or_abs(args.dataset_plan)
        run_root = _validate_run_root(args.run_root, bool(args.allow_nonlocal_out))
        plan = _load_json(dataset_plan_path)
    except Exception as exc:
        if args.allow_fail:
            print(json.dumps({"ok": False, "error": repr(exc), "allow_fail": True}, ensure_ascii=True, indent=2))
            return 0
        print(json.dumps({"ok": False, "error": repr(exc)}, ensure_ascii=True, indent=2))
        return 1

    run_id = _infer_run_id(plan)
    run_dir = run_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    scene_roots = [_repo_or_abs(raw) for raw in args.scene_root]
    identities = _extract_identities(plan)
    trajectories = _extract_trajectories(plan)
    camera_layouts = _extract_camera_layouts(plan)
    capture_tasks = _extract_capture_tasks(plan)
    mask_gt = _mask_gt_contract(plan)
    scene_observations, scene_blockers = _collect_scene_observations(scene_roots)
    samples = _build_samples(
        identities=identities,
        trajectories=trajectories,
        camera_layouts=camera_layouts,
        scene_observations=scene_observations,
        capture_tasks=capture_tasks,
        plan=plan,
        mask_gt=mask_gt,
    )
    splits = _build_splits(samples, plan)
    split_policy_summary = _build_split_policy_summary(splits)
    split_policy_digest = _canonical_json_sha256(split_policy_summary)
    splits["split_policy_summary"] = split_policy_summary
    splits["split_policy_digest"] = split_policy_digest
    splits_manifest = _as_dict(splits.get("manifest"))
    splits_manifest["split_policy_summary"] = split_policy_summary
    splits_manifest["split_policy_digest"] = split_policy_digest
    splits["manifest"] = splits_manifest

    deployment_episodes = _build_deployment_episodes(plan)
    deployment_episodes, deployment_episodes_summary = _with_episode_visibility(deployment_episodes, samples)
    deployment_episodes_summary["split_policy_summary"] = split_policy_summary
    deployment_episodes_summary["split_policy_digest"] = split_policy_digest
    scene_qualification_summary = _build_scene_qualification_summary(scene_observations, samples)
    dataset_run_contract_summary = _build_dataset_run_contract_summary(
        samples=samples,
        scene_observations=scene_observations,
        capture_tasks=capture_tasks,
        identities=identities,
        splits=splits,
    )

    plan_counts = _as_dict(plan.get("counts"))
    ue_import_missing_count = int(plan_counts.get("ue_carla_import_missing_count") or 0)
    full_v1_capture_ready = ue_import_missing_count == 0 and bool(samples)

    blockers: list[dict[str, Any]] = []
    if not identities:
        blockers.append({"code": "identities_missing", "detail": "dataset_plan has no resolved identities"})
    if not trajectories:
        blockers.append({"code": "trajectories_missing", "detail": "dataset_plan has no resolved trajectories"})
    if not camera_layouts:
        blockers.append({"code": "camera_layouts_missing", "detail": "dataset_plan has no camera_layouts"})
    if camera_layouts and not any(layout.get("node_id") for layout in camera_layouts):
        blockers.append({"code": "camera_layout_node_missing", "detail": "dataset_plan camera_layouts missing node_id"})
    if camera_layouts and not any(layout.get("cameras") for layout in camera_layouts):
        blockers.append({"code": "camera_layout_cameras_missing", "detail": "dataset_plan camera_layouts have no cameras"})
    for code in scene_blockers:
        blockers.append({"code": code, "detail": "scene artifacts unavailable for sample materialization"})
    if not samples:
        blockers.append({"code": "samples_empty_plan_only", "detail": "no materialized captured samples; generated plan-only training index"})
    if scene_qualification_summary.get("has_cross_split_scene_key_conflict") is True:
        blockers.append(
            {
                "code": "scene_key_cross_split_conflict",
                "detail": "at least one scene_key maps to multiple splits in indexed samples",
                "cross_split_scene_key_count": scene_qualification_summary.get("cross_split_scene_key_count"),
            }
        )

    dataset_plan_copy_path = run_dir / "dataset_plan.json"
    dataset_samples_path = run_dir / "dataset_samples.jsonl"
    dataset_splits_path = run_dir / "dataset_splits.json"
    deployment_episodes_path = run_dir / "deployment_episodes.json"
    manifest_path = run_dir / "dataset_manifest.json"
    dataset_index_manifest_path = run_dir / "dataset_index_manifest.json"
    scene_sample_index_manifest_path = run_dir / "scene_sample_index_manifest.json"
    scene_membership_manifest_path = run_dir / "scene_membership_manifest.json"
    identity_model_switch_manifest_path = run_dir / "identity_model_switch_manifest.json"
    existing_scene_index_bridge_manifest_path = run_dir / "existing_scene_index_bridge_manifest.json"
    sidecar_quality_manifest_path = run_dir / "sidecar_quality_manifest.json"
    no_mask_non_promotion_manifest_path = run_dir / "no_mask_non_promotion_manifest.json"
    dataset_gap_manifest_path = run_dir / "dataset_gap_manifest.json"
    deployment_episode_visibility_manifest_path = run_dir / "deployment_episode_visibility_manifest.json"
    sample_schema_coverage_manifest_path = run_dir / "sample_schema_coverage_manifest.json"

    if dataset_plan_path.resolve() != dataset_plan_copy_path.resolve():
        shutil.copyfile(dataset_plan_path, dataset_plan_copy_path)
    _write_jsonl(dataset_samples_path, samples)
    _write_json(dataset_splits_path, splits)
    _write_json(
        deployment_episodes_path,
        {
            "schema_version": "carla_air_deployment_episodes_v1",
            "summary": deployment_episodes_summary,
            "episodes": deployment_episodes,
        },
    )
    deployment_episode_visibility_manifest = _build_deployment_episode_visibility_manifest(
        run_id=run_id,
        deployment_episodes=deployment_episodes,
        deployment_episodes_summary=deployment_episodes_summary,
        split_policy_summary=split_policy_summary,
        split_policy_digest=split_policy_digest,
    )
    _write_json(deployment_episode_visibility_manifest_path, deployment_episode_visibility_manifest)
    mask_gt_available_count = sum(
        1 for obs in scene_observations if _as_dict(obs.get("mask_gt_probe")).get("availability") == "available"
    )
    index_manifest = _build_dataset_index_manifest(
        run_id=run_id,
        run_dir=run_dir,
        dataset_samples_path=dataset_samples_path,
        dataset_splits_path=dataset_splits_path,
        dataset_manifest_path=manifest_path,
        samples=samples,
        splits=splits,
        mask_gt_available_count=mask_gt_available_count,
        dataset_run_contract_summary=dataset_run_contract_summary,
    )
    _write_json(dataset_index_manifest_path, index_manifest)
    scene_sample_index_manifest = _build_scene_sample_index_manifest(
        run_id=run_id,
        run_dir=run_dir,
        samples=samples,
        splits=splits,
        dataset_index_manifest=index_manifest,
    )
    _write_json(scene_sample_index_manifest_path, scene_sample_index_manifest)
    duplicate_sample_id_count = int(index_manifest.get("duplicate_sample_id_count") or 0)
    missing_from_splits_count = int(index_manifest.get("missing_from_splits_count") or 0)
    extra_in_splits_count = int(index_manifest.get("extra_in_splits_count") or 0)
    if duplicate_sample_id_count != 0:
        blockers.append(
            {
                "code": "dataset_index_duplicate_sample_ids_detected",
                "detail": "dataset_index_manifest duplicate_sample_id_count must be 0 for strict index contract.",
                "duplicate_sample_id_count": duplicate_sample_id_count,
            }
        )
    if missing_from_splits_count != 0:
        blockers.append(
            {
                "code": "dataset_index_missing_from_splits_detected",
                "detail": "dataset_index_manifest missing_from_splits_count must be 0 for strict index contract.",
                "missing_from_splits_count": missing_from_splits_count,
            }
        )
    if extra_in_splits_count != 0:
        blockers.append(
            {
                "code": "dataset_index_extra_in_splits_detected",
                "detail": "dataset_index_manifest extra_in_splits_count must be 0 for strict index contract.",
                "extra_in_splits_count": extra_in_splits_count,
            }
        )
    strict_index_contract = _as_dict(index_manifest.get("strict_index_contract"))
    if strict_index_contract.get("sample_id_complete") is not True:
        blockers.append(
            {
                "code": "dataset_index_sample_id_incomplete",
                "detail": "strict_index_contract.sample_id_complete must be true.",
                "sample_id_count": index_manifest.get("sample_id_count"),
                "sample_count": index_manifest.get("sample_count"),
            }
        )
    if strict_index_contract.get("all_required_scene_membership_fields_present") is not True:
        blockers.append(
            {
                "code": "dataset_index_scene_membership_fields_incomplete",
                "detail": "strict_index_contract requires complete scene_split_membership_index fields.",
            }
        )
    scene_membership_manifest = _build_scene_membership_manifest(
        run_id=run_id,
        run_dir=run_dir,
        scene_observations=scene_observations,
        samples=samples,
        capture_tasks=capture_tasks,
        splits=splits,
        scene_qualification_summary=scene_qualification_summary,
    )
    scene_membership_manifest["split_policy_summary"] = split_policy_summary
    scene_membership_manifest["split_policy_digest"] = split_policy_digest
    _write_json(scene_membership_manifest_path, scene_membership_manifest)
    identity_model_switch_manifest = _build_identity_model_switch_manifest(
        run_id=run_id,
        samples=samples,
        identities=identities,
        plan=plan,
        capture_tasks=capture_tasks,
    )
    _write_json(identity_model_switch_manifest_path, identity_model_switch_manifest)
    existing_scene_index_bridge_manifest = _build_existing_scene_index_bridge_manifest(
        run_id=run_id,
        samples=samples,
        scene_observations=scene_observations,
        scene_membership_manifest=scene_membership_manifest,
        dataset_index_manifest=index_manifest,
    )
    _write_json(existing_scene_index_bridge_manifest_path, existing_scene_index_bridge_manifest)
    sidecar_quality_manifest = _build_sidecar_quality_manifest(run_id, samples)
    _write_json(sidecar_quality_manifest_path, sidecar_quality_manifest)
    sample_schema_coverage_summary = _build_sample_schema_coverage_summary(samples)
    sample_schema_coverage_manifest = _build_sample_schema_coverage_manifest(
        run_id=run_id,
        sample_schema_coverage_summary=sample_schema_coverage_summary,
    )
    _write_json(sample_schema_coverage_manifest_path, sample_schema_coverage_manifest)
    dataset_gap_manifest = _build_dataset_gap_manifest(
        run_id=run_id,
        samples=samples,
        dataset_run_contract_summary=dataset_run_contract_summary,
    )
    _write_json(dataset_gap_manifest_path, dataset_gap_manifest)

    outputs = {
        "dataset_plan_json": str(dataset_plan_copy_path.resolve()),
        "dataset_samples_jsonl": str(dataset_samples_path.resolve()),
        "dataset_splits_json": str(dataset_splits_path.resolve()),
        "deployment_episodes_json": str(deployment_episodes_path.resolve()),
        "deployment_episode_visibility_manifest_json": str(deployment_episode_visibility_manifest_path.resolve()),
        "dataset_manifest_json": str(manifest_path.resolve()),
        "dataset_index_manifest_json": str(dataset_index_manifest_path.resolve()),
        "scene_sample_index_manifest_json": str(scene_sample_index_manifest_path.resolve()),
        "scene_membership_manifest_json": str(scene_membership_manifest_path.resolve()),
        "identity_model_switch_manifest_json": str(identity_model_switch_manifest_path.resolve()),
        "existing_scene_index_bridge_manifest_json": str(existing_scene_index_bridge_manifest_path.resolve()),
        "sidecar_quality_manifest_json": str(sidecar_quality_manifest_path.resolve()),
        "no_mask_non_promotion_manifest_json": str(no_mask_non_promotion_manifest_path.resolve()),
        "dataset_gap_manifest_json": str(dataset_gap_manifest_path.resolve()),
        "sample_schema_coverage_manifest_json": str(sample_schema_coverage_manifest_path.resolve()),
    }

    no_mask_non_promotion_manifest = _build_no_mask_non_promotion_manifest(
        run_id=run_id,
        samples=samples,
        dataset_run_contract_summary=dataset_run_contract_summary,
        dataset_manifest_outputs=outputs,
    )
    _write_json(no_mask_non_promotion_manifest_path, no_mask_non_promotion_manifest)

    manifest = {
        "schema_version": SCHEMA_MANIFEST,
        "generated_at": _now_iso(),
        "ok": len(blockers) == 0,
        "plan_only": len(samples) == 0,
        "run_id": run_id,
        "run_dir": str(run_dir.resolve()),
        "starts_runtime": False,
        "writes_scene_outputs": False,
        "non_promotion": True,
        "full_v1_live_dataset_ready": False,
        "ue_carla_import_externalized": True,
        "dataset_plan_path": str(dataset_plan_path.resolve()),
        "scene_roots": [str(path.resolve()) for path in scene_roots],
        "plan_selection": {
            "capture_profile": plan.get("capture_profile"),
            "selected_filters": _as_dict(plan.get("selected_filters")),
            "matrix_count": _as_dict(plan.get("counts")).get("matrix_count"),
            "plan_ok": plan.get("ok"),
        },
        "sample_count": len(samples),
        "identity_count": len(identities),
        "trajectory_count": len(trajectories),
        "camera_layout_count": len(camera_layouts),
        "capture_task_count": (
            int(plan_counts.get("capture_task_count"))
            if str(plan_counts.get("capture_task_count") or "").isdigit()
            else len(capture_tasks)
        ),
        "split_count": len(_as_dict(splits.get("splits"))),
        "deployment_episode_count": len(deployment_episodes),
        "blocker_count": len(blockers),
        "blockers": blockers,
        "mask_gt_policy": mask_gt,
        "sample_contract_template": _sample_contract_template(mask_gt),
        "readiness": {
            "index_materialized": bool(samples),
            "existing_scene_index_ready": bool(samples),
            "full_v1_live_dataset_ready": full_v1_capture_ready,
            "ue_carla_import_missing_count": ue_import_missing_count,
            "multi_identity_live_capture_ready": full_v1_capture_ready,
            "mask_gt_required_for_index_materialization": False,
            "mask_gt_required_for_full_v1_capture": bool(mask_gt.get("mask_gt_required_for_formal_training")),
            "note": (
                "Index materialization success only means existing scene RGB/depth/semantic/instance/pose/calib "
                "references were indexed. It does not prove full multi-identity live capture, formal mask_gt, "
                "UE/CARLA import/readback, or final 4D geometry readiness."
            ),
        },
        "outputs": outputs,
        "scene_observations": scene_observations,
        "camera_layouts": camera_layouts,
        "modality_summary": _build_modality_summary(samples),
        "sidecar_quality_matrix": _build_sidecar_quality_matrix(samples),
        "sample_schema_coverage_summary": sample_schema_coverage_summary,
        "plan_alignment_summary": _build_plan_alignment_summary(samples),
        "identity_model_switch_contract": _build_identity_model_switch_contract(samples, identities, plan),
        "capture_task_alignment_summary": _build_capture_task_alignment_summary(samples, capture_tasks),
        "scene_qualification_summary": scene_qualification_summary,
        "dataset_run_contract_summary": dataset_run_contract_summary,
        "mask_gt_availability_summary": {
            "scene_probe_count": len(scene_observations),
            "available_count": mask_gt_available_count,
            "unavailable_count": sum(1 for obs in scene_observations if _as_dict(obs.get("mask_gt_probe")).get("availability") == "unavailable"),
            "unknown_count": sum(1 for obs in scene_observations if _as_dict(obs.get("mask_gt_probe")).get("availability") == "unknown"),
            "scene_probe_rgb_pose_calib_without_formal_mask_gt_count": sum(
                1
                for obs in scene_observations
                if bool(obs.get("has_minimum_artifacts"))
                and _as_dict(obs.get("mask_gt_probe")).get("availability") == "unavailable"
            ),
            "candidate_pseudo_proxy_never_promoted_to_mask_gt": True,
            "mask_gt_unavailable_is_auditable_not_auto_promotion": True,
            "mask_gt_unavailable_is_not_hard_failure_when_scene_probe_has_minimum_artifacts": True,
        },
        "contract_notes": {
            "pseudo_candidate_proxy_not_mask_gt": True,
            "treat_mask_gt_only_as_true_ground_truth": True,
            "mask_gt_unavailable_allowed_in_v1_contract": True,
            "mask_gt_unavailable_requires_audit_fields": True,
            "mask_gt_unavailable_does_not_imply_candidate_promotion": True,
            "deployment_splits": DEFAULT_SPLIT_NAMES,
            "required_sample_fields": [
                "identity_id",
                "trajectory_id",
                "node_id",
                "camera_id",
                "rgb",
                "depth",
                "semantic",
                "instance",
                "pose",
                "calib",
                "split",
                "identity",
                "trajectory",
                "camera_layout",
                "view.node_id",
                "view.camera_id",
                "timestamp",
                "refs.rgb",
                "refs.depth",
                "refs.semantic",
                "refs.instance",
                "refs.pose",
                "refs.calib",
                "mask_gt.policy_mode",
                "mask_gt.availability",
                "mask_gt.unavailable_reason",
                "mask_gt.audit_state",
            ],
        },
    }
    _write_json(manifest_path, manifest)

    result = {
        "ok": manifest["ok"],
        "plan_only": manifest["plan_only"],
        "sample_count": manifest["sample_count"],
        "blocker_count": manifest["blocker_count"],
        "run_dir": manifest["run_dir"],
        "manifest": str(manifest_path.resolve()),
    }
    print(json.dumps(result, ensure_ascii=True, indent=2))
    if manifest["ok"] or args.allow_fail:
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
