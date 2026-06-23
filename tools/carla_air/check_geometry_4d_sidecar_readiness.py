#!/usr/bin/env python3
"""Check local readiness for CARLA-Air geometry 4D sidecar inputs.

This script performs only local filesystem and environment checks. It does not
run inference, does not start any runtime, and does not promote any geometry
output beyond diagnostic readiness.
"""

from __future__ import annotations

import argparse
import importlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "carla_air_geometry_4d_sidecar_readiness_v1"
DEFAULT_OUT = REPO_ROOT / "local/carla_air/geometry_4d/readiness/readiness_summary.json"

REQUIRED_REPOS = {
    "dggt_repo_present": REPO_ROOT / "third_party/dggt",
    "mapanything_repo_present": REPO_ROOT / "third_party/map-anything",
}

REQUIRED_DGGT_WEIGHTS = [
    "model_latest_waymo.pt",
    "model_difix.pkl",
    "tapip3d_final.pth",
    "model.pt",
]

REQUIRED_MAPANYTHING_WEIGHTS = [
    "model.safetensors",
    "config.json",
]

SYMLINK_PATHS = {
    "dggt": REPO_ROOT / "local/carla_air/models/dggt",
    "mapanything-apache": REPO_ROOT / "local/carla_air/models/mapanything-apache",
}

OPTIONAL_PYTHON_IMPORTS = ("torch", "numpy", "cv2", "huggingface_hub")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _repo_rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _load_optional_import(module_name: str) -> dict[str, Any]:
    record: dict[str, Any] = {
        "module": module_name,
        "present": False,
        "error": None,
    }
    try:
        importlib.import_module(module_name)
    except Exception as exc:  # noqa: BLE001 - readiness check must not crash
        record["error"] = f"{type(exc).__name__}: {exc}"
        return record
    record["present"] = True
    return record


def _check_required_file(path: Path) -> dict[str, Any]:
    exists = path.is_file()
    size = path.stat().st_size if exists else None
    return {
        "path": _repo_rel(path),
        "exists": exists,
        "non_empty": bool(exists and size and size > 0),
        "size_bytes": size,
    }


def _check_repo_dir(path: Path) -> dict[str, Any]:
    return {
        "path": _repo_rel(path),
        "exists": path.is_dir(),
    }


def _check_symlink(path: Path) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": _repo_rel(path),
        "exists": path.exists(),
        "is_symlink": path.is_symlink(),
        "target": None,
        "resolved_target": None,
        "target_exists": None,
        "target_is_dir": None,
        "broken": None,
    }
    if path.is_symlink():
        try:
            target = Path(path.readlink())
            record["target"] = str(target)
            resolved = path.resolve(strict=False)
            record["resolved_target"] = str(resolved)
            record["target_exists"] = resolved.exists()
            record["target_is_dir"] = resolved.is_dir()
            record["broken"] = not resolved.exists()
        except Exception as exc:  # noqa: BLE001 - readiness check must not crash
            record["broken"] = True
            record["error"] = f"{type(exc).__name__}: {exc}"
    elif path.exists():
        record["broken"] = False
    else:
        record["broken"] = True
    return record


def build_readiness_summary(out_path: Path) -> dict[str, Any]:
    blockers: list[str] = []

    repo_checks: dict[str, Any] = {}
    for key, repo_path in REQUIRED_REPOS.items():
        record = _check_repo_dir(repo_path)
        repo_checks[key] = record
        if not record["exists"]:
            blockers.append(f"missing_repo_dir:{record['path']}")

    dggt_root = REPO_ROOT / "local/models/dggt"
    mapanything_root = REPO_ROOT / "local/models/mapanything-apache"

    dggt_weights: dict[str, Any] = {}
    for name in REQUIRED_DGGT_WEIGHTS:
        file_record = _check_required_file(dggt_root / name)
        dggt_weights[name] = file_record
        if not file_record["exists"]:
            blockers.append(f"missing_weight:{file_record['path']}")
        elif not file_record["non_empty"]:
            blockers.append(f"empty_weight:{file_record['path']}")

    mapanything_weights: dict[str, Any] = {}
    for name in REQUIRED_MAPANYTHING_WEIGHTS:
        file_record = _check_required_file(mapanything_root / name)
        mapanything_weights[name] = file_record
        if not file_record["exists"]:
            blockers.append(f"missing_weight:{file_record['path']}")
        elif not file_record["non_empty"]:
            blockers.append(f"empty_weight:{file_record['path']}")

    symlinks = {name: _check_symlink(path) for name, path in SYMLINK_PATHS.items()}
    for name, record in symlinks.items():
        if record["broken"]:
            blockers.append(f"broken_symlink:{record['path']}")
        elif not record["is_symlink"]:
            blockers.append(f"not_a_symlink:{record['path']}")

    python_imports = {
        name: _load_optional_import(name) for name in OPTIONAL_PYTHON_IMPORTS
    }
    for record in python_imports.values():
        if not record["present"]:
            blockers.append(f"missing_python_import:{record['module']}")

    payload = {
        "schema_version": SCHEMA_VERSION,
        "checked_at": _utc_now_iso(),
        "ok": not blockers,
        "dggt_repo_present": repo_checks["dggt_repo_present"]["exists"],
        "mapanything_repo_present": repo_checks["mapanything_repo_present"]["exists"],
        "dggt_weights": dggt_weights,
        "mapanything_weights": mapanything_weights,
        "symlinks": symlinks,
        "python_imports": python_imports,
        "blockers": blockers,
        "diagnostic_only": True,
        "non_promotion": True,
        "not_formal_geometry": True,
        "inference_executed": False,
        "runtime_started": False,
        "output_path": _repo_rel(out_path),
        "check_scope": "local_readiness_only",
        "source_roots": {
            "third_party_dggt": _repo_rel(REPO_ROOT / "third_party/dggt"),
            "third_party_map_anything": _repo_rel(REPO_ROOT / "third_party/map-anything"),
            "local_models_dggt": _repo_rel(dggt_root),
            "local_models_mapanything_apache": _repo_rel(mapanything_root),
        },
        "required_repos": repo_checks,
    }
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Output JSON path.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    out_path = Path(str(args.out))
    if not out_path.is_absolute():
        out_path = (REPO_ROOT / out_path).resolve()
    payload = build_readiness_summary(out_path)
    _write_json(out_path, payload)
    print(f"Wrote: {out_path}")
    print(
        "[summary] "
        f"ok={payload['ok']} "
        f"blockers={len(payload['blockers'])} "
        f"python_imports_ok={sum(1 for item in payload['python_imports'].values() if item['present'])}"
    )


if __name__ == "__main__":
    main()
