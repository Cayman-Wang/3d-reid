from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
LOCAL_ROOT = REPO_ROOT / "local"
WEAK_VARIANT_NAME = "actor_bbox_semantic_lidar_diag"
WEAK_VARIANT_MARKERS = (
    "actor_bbox_semantic_lidar_diag",
    "tracklets_actor_bbox_semantic_lidar_diag",
    "embeddings_actor_bbox_semantic_lidar_diag",
    "masks_actor_bbox",
    "semantic_lidar_actor_points",
    "carla_semantic_lidar_actor_idx_v1",
)
WEAK_READINESS_SCHEMA = "carla_air_weak_variant_official_readiness_verification_v1"
WEAK_CONTRACT_VERIFICATION_SCHEMA = "carla_air_weak_variant_contract_update_verification_v1"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def repo_or_abs(raw: str | Path) -> Path:
    path = Path(str(raw))
    return path if path.is_absolute() else REPO_ROOT / path


def is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def path_uses_weak_variant(value: str | Path) -> bool:
    text = str(value).replace("\\", "/")
    return any(marker in text for marker in WEAK_VARIANT_MARKERS)


def json_uses_weak_variant(value: Any) -> bool:
    if isinstance(value, str):
        return path_uses_weak_variant(value)
    if isinstance(value, list):
        return any(json_uses_weak_variant(item) for item in value)
    if isinstance(value, dict):
        return any(json_uses_weak_variant(item) for item in value.values())
    return False


def validate_weak_variant_readiness(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"--weak-variant-readiness not found: {path}")
    if not is_under(path, LOCAL_ROOT):
        raise SystemExit(f"--weak-variant-readiness must stay under repo local/: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"--weak-variant-readiness is not readable JSON: {path} ({exc!r})")
    if not isinstance(payload, dict):
        raise SystemExit(f"--weak-variant-readiness JSON root must be an object: {path}")

    required = {
        "schema_version": WEAK_READINESS_SCHEMA,
        "ok": True,
        "failure_count": 0,
        "verifier_scope": "pre_writer_readiness_only",
        "writes_scene_outputs": False,
        "modifies_pipeline_contract": False,
        "formalization_ready": False,
        "goal_complete": False,
        "real_writer_implemented": False,
        "promotion_evidence_satisfied": False,
        "not_identity_proof": True,
        "not_pixel_accurate_mask_evidence": True,
        "not_real_or_final_geometry": True,
    }
    mismatches = {
        key: {"expected": expected, "actual": payload.get(key)}
        for key, expected in required.items()
        if payload.get(key) != expected
    }
    if mismatches:
        raise SystemExit(
            "--weak-variant-readiness is not an accepted non-promotion guard: "
            f"{path} mismatches={json.dumps(mismatches, ensure_ascii=True, sort_keys=True)}"
        )

    diag = payload.get("diagnostic_reid_smoke")
    if not isinstance(diag, dict) or diag.get("ok") is not True or int(diag.get("failure_count") or 0) != 0:
        raise SystemExit("--weak-variant-readiness must include a passing diagnostic_reid_smoke summary")

    return {
        "weak_variant": WEAK_VARIANT_NAME,
        "diagnostic_only": True,
        "not_formal_benchmark": True,
        "formalization_ready": False,
        "goal_complete": False,
        "readiness_report_path": str(path),
        "readiness_report_sha256": sha256_file(path),
        "readiness_verifier_scope": payload.get("verifier_scope"),
        "real_writer_implemented": payload.get("real_writer_implemented"),
        "promotion_evidence_satisfied": payload.get("promotion_evidence_satisfied"),
    }


def validate_weak_contract_verification(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"--weak-contract-verification not found: {path}")
    if not is_under(path, LOCAL_ROOT):
        raise SystemExit(f"--weak-contract-verification must stay under repo local/: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"--weak-contract-verification is not readable JSON: {path} ({exc!r})")
    if not isinstance(payload, dict):
        raise SystemExit(f"--weak-contract-verification JSON root must be an object: {path}")

    required = {
        "schema_version": WEAK_CONTRACT_VERIFICATION_SCHEMA,
        "ok": True,
        "failure_count": 0,
        "writes_scene_outputs": False,
        "modifies_pipeline_contract": False,
        "formalization_ready": False,
        "goal_complete": False,
        "not_identity_proof": True,
        "not_pixel_accurate_mask_evidence": True,
        "not_real_or_final_geometry": True,
        "promotion_evidence_satisfied": False,
    }
    mismatches = {
        key: {"expected": expected, "actual": payload.get(key)}
        for key, expected in required.items()
        if payload.get(key) != expected
    }
    if mismatches:
        raise SystemExit(
            "--weak-contract-verification is not an accepted non-promotion guard: "
            f"{path} mismatches={json.dumps(mismatches, ensure_ascii=True, sort_keys=True)}"
        )

    registered_weak_variant_count = int(payload.get("registered_weak_variant_count") or 0)
    weak_marker_scene_count = int(payload.get("weak_marker_scene_count") or 0)
    if registered_weak_variant_count < 4:
        raise SystemExit(
            "--weak-contract-verification requires registered_weak_variant_count >= 4; "
            f"got {registered_weak_variant_count} ({path})"
        )
    if weak_marker_scene_count < 2:
        raise SystemExit(
            "--weak-contract-verification requires weak_marker_scene_count >= 2; "
            f"got {weak_marker_scene_count} ({path})"
        )

    return {
        "contract_verification_path": str(path),
        "contract_verification_sha256": sha256_file(path),
        "registered_weak_variant_count": registered_weak_variant_count,
        "weak_marker_scene_count": weak_marker_scene_count,
    }


def resolve_embedding_weak_variant_guard(
    args: argparse.Namespace,
    scene_dir: Path,
    tracklets: list[Any],
) -> dict[str, Any] | None:
    uses_weak_paths = (
        path_uses_weak_variant(args.tracklets)
        or path_uses_weak_variant(args.out_dir)
        or json_uses_weak_variant(tracklets)
    )
    weak_variant = str(args.weak_variant or "")
    readiness_arg = str(args.weak_variant_readiness or "")
    contract_verification_arg = str(args.weak_contract_verification or "")

    if not uses_weak_paths:
        if weak_variant or readiness_arg or contract_verification_arg:
            raise SystemExit(
                "--weak-variant, --weak-variant-readiness and --weak-contract-verification "
                "are only valid for weak diagnostic paths"
            )
        return None

    if weak_variant != WEAK_VARIANT_NAME:
        raise SystemExit(
            f"weak diagnostic paths require --weak-variant {WEAK_VARIANT_NAME}; got {weak_variant or '<missing>'}"
        )
    if not readiness_arg:
        raise SystemExit("weak diagnostic paths require --weak-variant-readiness from verify_weak_variant_official_readiness.py")
    if not contract_verification_arg:
        raise SystemExit(
            "weak diagnostic paths require --weak-contract-verification "
            "from weak_variant_contract_update_verification_post_execute"
        )
    if not is_under(scene_dir, LOCAL_ROOT):
        raise SystemExit(f"weak diagnostic embedding extraction is local-only; scene_dir must be under repo local/: {scene_dir}")
    tracklets_path = (scene_dir / str(args.tracklets)).resolve()
    out_dir_path = (scene_dir / str(args.out_dir)).resolve()
    if not is_under(tracklets_path, LOCAL_ROOT):
        raise SystemExit(f"weak diagnostic tracklets must stay under repo local/: {tracklets_path}")
    if not is_under(out_dir_path, LOCAL_ROOT):
        raise SystemExit(f"weak diagnostic embeddings output must stay under repo local/: {out_dir_path}")
    weak_variant_guard = validate_weak_variant_readiness(repo_or_abs(readiness_arg))
    weak_variant_guard["contract_verification"] = validate_weak_contract_verification(repo_or_abs(contract_verification_arg))
    return weak_variant_guard


def resolve_eval_weak_variant_guard(args: argparse.Namespace, scene_dirs: list[Path]) -> dict[str, Any] | None:
    weak_variant = str(args.weak_variant or "")
    readiness_arg = str(args.weak_variant_readiness or "")
    contract_verification_arg = str(args.weak_contract_verification or "")
    uses_weak_paths = path_uses_weak_variant(args.embeddings_subdir) or bool(
        weak_variant or readiness_arg or contract_verification_arg
    )

    if not uses_weak_paths:
        return None

    if weak_variant != WEAK_VARIANT_NAME:
        raise SystemExit(
            f"weak diagnostic paths require --weak-variant {WEAK_VARIANT_NAME}; got {weak_variant or '<missing>'}"
        )
    if not readiness_arg:
        raise SystemExit("weak diagnostic paths require --weak-variant-readiness from verify_weak_variant_official_readiness.py")
    if not contract_verification_arg:
        raise SystemExit(
            "weak diagnostic paths require --weak-contract-verification "
            "from weak_variant_contract_update_verification_post_execute"
        )
    for scene_dir in scene_dirs:
        if not is_under(scene_dir, LOCAL_ROOT):
            raise SystemExit(f"weak diagnostic retrieval eval is local-only; scene_dir must be under repo local/: {scene_dir}")
        embeddings_dir = (scene_dir / str(args.embeddings_subdir)).resolve()
        if not is_under(embeddings_dir, LOCAL_ROOT):
            raise SystemExit(f"weak diagnostic embeddings input must stay under repo local/: {embeddings_dir}")
    if args.out:
        out_path = repo_or_abs(str(args.out)).resolve()
        if not is_under(out_path, LOCAL_ROOT):
            raise SystemExit(f"weak diagnostic eval output must stay under repo local/: {out_path}")
    weak_variant_guard = validate_weak_variant_readiness(repo_or_abs(readiness_arg))
    weak_variant_guard["contract_verification"] = validate_weak_contract_verification(repo_or_abs(contract_verification_arg))
    return weak_variant_guard


def validate_weak_embedding_meta(
    meta: list[dict],
    meta_path: Path,
    weak_variant_guard: dict[str, Any] | None,
) -> None:
    if weak_variant_guard is None:
        return
    for idx, item in enumerate(meta):
        if not isinstance(item, dict):
            raise SystemExit(f"weak diagnostic tracks_meta item must be an object: {meta_path} index={idx}")
        weak_meta = item.get("weak_variant")
        if not isinstance(weak_meta, dict):
            raise SystemExit(
                f"weak diagnostic embeddings must be regenerated with --weak-variant metadata: {meta_path} index={idx}"
            )
        required = {
            "weak_variant": WEAK_VARIANT_NAME,
            "diagnostic_only": True,
            "not_formal_benchmark": True,
            "formalization_ready": False,
            "goal_complete": False,
            "readiness_report_sha256": weak_variant_guard["readiness_report_sha256"],
        }
        mismatches = {
            key: {"expected": expected, "actual": weak_meta.get(key)}
            for key, expected in required.items()
            if weak_meta.get(key) != expected
        }
        contract_meta = weak_meta.get("contract_verification")
        contract_guard = weak_variant_guard.get("contract_verification")
        if not isinstance(contract_meta, dict) or not isinstance(contract_guard, dict):
            raise SystemExit(
                f"weak diagnostic embeddings metadata missing contract verification evidence: {meta_path} index={idx}"
            )
        contract_required = {
            "contract_verification_path": contract_guard["contract_verification_path"],
            "contract_verification_sha256": contract_guard["contract_verification_sha256"],
            "registered_weak_variant_count": contract_guard["registered_weak_variant_count"],
            "weak_marker_scene_count": contract_guard["weak_marker_scene_count"],
        }
        contract_mismatches = {
            key: {"expected": expected, "actual": contract_meta.get(key)}
            for key, expected in contract_required.items()
            if contract_meta.get(key) != expected
        }
        if contract_mismatches:
            raise SystemExit(
                "weak diagnostic embeddings contract evidence does not match --weak-contract-verification: "
                f"{meta_path} index={idx} mismatches={json.dumps(contract_mismatches, ensure_ascii=True, sort_keys=True)}"
            )
        if mismatches:
            raise SystemExit(
                "weak diagnostic embeddings metadata does not match --weak-variant-readiness: "
                f"{meta_path} index={idx} mismatches={json.dumps(mismatches, ensure_ascii=True, sort_keys=True)}"
            )
