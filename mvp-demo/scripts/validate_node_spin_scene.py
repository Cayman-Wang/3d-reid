from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from spin_scene_checks import lexical_abspath, preflight_mjcf_assets


def _repo_root() -> Path:
    return Path(__file__).absolute().parents[2]


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:  # pragma: no cover
        raise SystemExit(f"Failed to read json: {path}\nError: {e!r}")


def _normalize_cams(value: str) -> list[str]:
    cams = [cam.strip() for cam in str(value).split(",") if cam.strip()]
    if not cams:
        raise SystemExit("--cams is empty")
    return cams


def _resolve_manifest_entry(manifest_path: Path, scene_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = _load_json(manifest_path)
    entries = list(manifest.get("entries") or [])
    for entry in entries:
        if str(entry.get("scene_id")) == scene_id:
            return manifest, dict(entry)
    available = sorted(str(entry.get("scene_id")) for entry in entries)
    raise SystemExit(f"Could not find scene_id={scene_id!r} in manifest {manifest_path}\nAvailable: {available}")


def _float_close(actual: Any, expected: Any, tol: float = 1e-6) -> bool:
    try:
        return abs(float(actual) - float(expected)) <= tol
    except Exception:
        return False


def _normalize_float_seq(value: Any) -> list[float] | None:
    if value is None:
        return None
    if isinstance(value, str):
        tokens = [tok for tok in str(value).replace(",", " ").split() if tok]
        if not tokens:
            return []
        try:
            return [float(tok) for tok in tokens]
        except Exception:
            return None
    if isinstance(value, (list, tuple)):
        try:
            return [float(item) for item in value]
        except Exception:
            return None
    return None


def _normalize_str_seq(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        tokens = [tok.strip() for tok in str(value).replace(",", " ").split() if tok.strip()]
        return tokens
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value]
    return None


def _float_seq_close(actual: Any, expected: Any, tol: float = 1e-6) -> bool:
    actual_seq = _normalize_float_seq(actual)
    expected_seq = _normalize_float_seq(expected)
    if actual_seq is None or expected_seq is None or len(actual_seq) != len(expected_seq):
        return False
    return all(_float_close(a, e, tol=tol) for a, e in zip(actual_seq, expected_seq))


def _capture_param_matches(field: str, actual: Any, expected: Any) -> bool:
    if expected is None:
        return True
    if field in {"traj", "target_body", "yaw_profile", "pitch_profile"}:
        return str(actual) == str(expected)
    if field in {
        "traj_period",
        "traj_radius",
        "fps",
        "seconds",
        "yaw_start_deg",
        "yaw_end_deg",
        "pitch_amp_deg",
        "pitch_period",
        "roll_amp_deg",
    }:
        return _float_close(actual, expected)
    if field == "traj_center":
        return _float_seq_close(actual, expected)
    if field == "spin_axes":
        return _normalize_str_seq(actual) == _normalize_str_seq(expected)
    return actual == expected


def _count_files(path: Path, suffixes: tuple[str, ...]) -> int:
    total = 0
    for suffix in suffixes:
        total += len(list(path.glob(f"*{suffix}")))
    return total


def _frame_map(frame_times_csv: Path, cams: list[str]) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    with frame_times_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ts_us = int(row["ts_us"])
            except Exception as e:
                raise SystemExit(f"Invalid ts_us row in {frame_times_csv}: {row!r}\nError: {e!r}")
            cam_id = str(row["cam_id"]).strip()
            if cam_id not in cams:
                continue
            rows.setdefault(f"{ts_us:012d}", {})[cam_id] = str(row["filename"]).strip()
    return rows


def _check_stem_exists(base_dir: Path, stem: str, suffixes: tuple[str, ...]) -> bool:
    for suffix in suffixes:
        if (base_dir / f"{stem}{suffix}").exists():
            return True
    return False


def _pick_gt_branch_config(manifest: dict[str, Any]) -> dict[str, Any]:
    branch_configs = manifest.get("branch_configs") or {}
    return dict(branch_configs.get("gt_upper_bound") or {})


def _scene_validation_report(
    scene_dir: Path,
    *,
    cams: list[str],
    mask_subdir: str,
    depth_subdir: str,
    points_subdir: str,
    require_presentation_assets: bool,
    require_points: bool,
    expected_node_id: str | None,
    expected_scene_id: str | None,
    expected_identity_id: str | None,
    expected_mjcf_name: str | None,
    expected_capture_plan: dict[str, Any] | None,
    expected_unique_timestamps: int | None,
    max_empty_pointclouds: int,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "scene_dir": str(scene_dir),
        "checks": {},
        "counts": {},
        "capture_meta": {},
    }

    frame_times_csv = scene_dir / "frame_times.csv"
    rig_json = scene_dir / "calib" / "rig.json"
    capture_meta_json = scene_dir / "capture_meta.json"
    report["checks"]["scene_dir_exists"] = scene_dir.exists()
    report["checks"]["frame_times_exists"] = frame_times_csv.exists()
    report["checks"]["rig_json_exists"] = rig_json.exists()
    report["checks"]["capture_meta_exists"] = capture_meta_json.exists()
    if not all(bool(report["checks"][key]) for key in ("scene_dir_exists", "frame_times_exists", "rig_json_exists", "capture_meta_exists")):
        report["ok"] = False
        return report

    frame_rows = _frame_map(frame_times_csv, cams)
    stems = sorted(frame_rows.keys())
    unique_ts = len(stems)
    report["counts"]["unique_timestamps"] = unique_ts
    report["counts"]["frame_index_rows"] = sum(len(v) for v in frame_rows.values())
    report["checks"]["unique_timestamps_ok"] = (
        unique_ts == int(expected_unique_timestamps) if expected_unique_timestamps is not None else unique_ts > 0
    )
    report["checks"]["frame_index_complete"] = all(len(frame_rows[stem]) == len(cams) for stem in stems)

    rig = _load_json(rig_json)
    rig_cams = set((rig.get("cameras") or {}).keys())
    report["checks"]["rig_contains_cams"] = all(cam in rig_cams for cam in cams)

    capture_meta = _load_json(capture_meta_json)
    target_meta = dict(capture_meta.get("target") or {})
    render_meta = dict(capture_meta.get("render") or {})
    report["capture_meta"]["node_id"] = capture_meta.get("node_id")
    report["capture_meta"]["scene_id"] = capture_meta.get("scene_id")
    report["capture_meta"]["identity_id"] = target_meta.get("identity_id")
    report["capture_meta"]["target_body"] = target_meta.get("body")
    report["capture_meta"]["mjcf"] = capture_meta.get("mjcf")

    if expected_node_id is not None:
        report["checks"]["node_id_matches"] = str(capture_meta.get("node_id")) == expected_node_id

    if expected_scene_id is not None:
        report["checks"]["scene_id_matches"] = str(capture_meta.get("scene_id")) == expected_scene_id == scene_dir.name
    else:
        report["checks"]["scene_id_matches"] = str(capture_meta.get("scene_id")) == scene_dir.name

    if expected_identity_id is not None:
        report["checks"]["identity_id_matches"] = str(target_meta.get("identity_id")) == expected_identity_id

    if expected_mjcf_name is not None:
        actual_mjcf_name = Path(str(capture_meta.get("mjcf") or "")).name
        report["checks"]["mjcf_name_matches"] = actual_mjcf_name == expected_mjcf_name
        report["capture_meta"]["mjcf_name"] = actual_mjcf_name

    if expected_capture_plan is not None:
        expected_params = {
            "target_body": expected_capture_plan.get("target_body"),
            "traj": expected_capture_plan.get("traj"),
            "traj_center": expected_capture_plan.get("traj_center"),
            "traj_radius": expected_capture_plan.get("traj_radius"),
            "traj_period": expected_capture_plan.get("traj_period"),
            "fps": expected_capture_plan.get("fps"),
            "seconds": expected_capture_plan.get("seconds"),
            "spin_axes": expected_capture_plan.get("spin_axes"),
            "yaw_start_deg": expected_capture_plan.get("yaw_start_deg"),
            "yaw_end_deg": expected_capture_plan.get("yaw_end_deg"),
            "yaw_profile": expected_capture_plan.get("yaw_profile") or "linear_across_capture_duration",
            "pitch_amp_deg": expected_capture_plan.get("pitch_amp_deg"),
            "pitch_period": expected_capture_plan.get("pitch_period"),
            "pitch_profile": expected_capture_plan.get("pitch_profile") or "sine",
            "roll_amp_deg": expected_capture_plan.get("roll_amp_deg"),
        }
        actual_params = {
            "target_body": target_meta.get("body"),
            "traj": target_meta.get("traj"),
            "traj_center": target_meta.get("traj_center"),
            "traj_radius": target_meta.get("traj_radius"),
            "traj_period": target_meta.get("traj_period"),
            "fps": render_meta.get("fps"),
            "seconds": render_meta.get("seconds"),
            "spin_axes": target_meta.get("spin_axes"),
            "yaw_start_deg": target_meta.get("yaw_start_deg"),
            "yaw_end_deg": target_meta.get("yaw_end_deg"),
            "yaw_profile": target_meta.get("yaw_profile"),
            "pitch_amp_deg": target_meta.get("pitch_amp_deg"),
            "pitch_period": target_meta.get("pitch_period"),
            "pitch_profile": target_meta.get("pitch_profile"),
            "roll_amp_deg": target_meta.get("roll_amp_deg"),
        }
        field_matches = {
            key: _capture_param_matches(key, actual_params.get(key), expected_params.get(key)) for key in expected_params
        }
        report["capture_meta"]["expected"] = expected_params
        report["capture_meta"]["actual"] = actual_params
        report["capture_meta"]["field_matches"] = field_matches
        report["checks"]["capture_plan_matches"] = all(field_matches.values())

    per_cam: dict[str, Any] = {}
    for cam_id in cams:
        frame_dir = scene_dir / "cams" / cam_id / "frames"
        mask_dir = scene_dir / "cams" / cam_id / mask_subdir
        depth_dir = scene_dir / "cams" / cam_id / depth_subdir
        missing_frames = [stem for stem in stems if not _check_stem_exists(frame_dir, stem, (".jpg", ".png"))]
        missing_masks = [stem for stem in stems if not (mask_dir / f"{stem}.png").exists()]
        missing_depth = [stem for stem in stems if not (depth_dir / f"{stem}.npy").exists()]
        per_cam[cam_id] = {
            "frames": _count_files(frame_dir, (".jpg", ".png")),
            "masks": _count_files(mask_dir, (".png",)),
            "depth": _count_files(depth_dir, (".npy",)),
            "missing_frames": len(missing_frames),
            "missing_masks": len(missing_masks),
            "missing_depth": len(missing_depth),
        }
    report["counts"]["per_cam"] = per_cam
    report["checks"]["per_cam_complete"] = all(
        cam_report["missing_frames"] == 0 and cam_report["missing_masks"] == 0 and cam_report["missing_depth"] == 0
        for cam_report in per_cam.values()
    )

    presentation_dir = scene_dir / "presentation_assets"
    video_path = presentation_dir / "triview_video.mp4"
    overview_count = len(list(presentation_dir.glob("overview_*.png")))
    report["counts"]["presentation_overviews"] = overview_count
    report["checks"]["presentation_assets_ok"] = (
        True
        if not require_presentation_assets
        else video_path.exists() and overview_count >= 3
    )

    points_dir = scene_dir / points_subdir
    point_files = sorted(points_dir.glob("*.npy"))
    empty_points = 0
    for path in point_files:
        arr = np.load(str(path))
        if arr.size == 0 or (arr.ndim >= 2 and arr.shape[0] == 0):
            empty_points += 1
    report["counts"]["points_files"] = len(point_files)
    report["counts"]["empty_pointclouds"] = empty_points
    report["checks"]["points_ok"] = (
        True
        if not require_points
        else len(point_files) == unique_ts and empty_points <= int(max_empty_pointclouds)
    )

    required_keys = [
        "unique_timestamps_ok",
        "frame_index_complete",
        "rig_contains_cams",
        "scene_id_matches",
        "per_cam_complete",
        "presentation_assets_ok",
        "points_ok",
    ]
    if expected_node_id is not None:
        required_keys.append("node_id_matches")
    if expected_identity_id is not None:
        required_keys.append("identity_id_matches")
    if expected_mjcf_name is not None:
        required_keys.append("mjcf_name_matches")
    if expected_capture_plan is not None:
        required_keys.append("capture_plan_matches")

    report["ok"] = all(bool(report["checks"].get(key)) for key in required_keys)
    return report


def _print_human(report: dict[str, Any]) -> None:
    scene_dir = report.get("scene_dir")
    if scene_dir:
        print(f"scene_dir: {scene_dir}")
    if "mjcf_preflight" in report:
        preflight = report["mjcf_preflight"]
        print(f"mjcf_preflight_ok: {bool(preflight.get('ok'))}")
        if not bool(preflight.get("ok")):
            print(f"  mjcf: {preflight.get('mjcf')}")
            print(f"  missing_refs: {len(preflight.get('missing_refs') or [])}")
            print(f"  ascii_path: {bool(preflight.get('ascii_path'))}")
    if scene_dir:
        print(f"unique_timestamps: {report.get('counts', {}).get('unique_timestamps')}")
        print(f"frame_index_rows: {report.get('counts', {}).get('frame_index_rows')}")
        for cam_id, cam_report in sorted((report.get("counts", {}).get("per_cam") or {}).items()):
            print(
                f"{cam_id}: frames={cam_report['frames']} masks={cam_report['masks']} "
                f"depth={cam_report['depth']} missing=({cam_report['missing_frames']},"
                f"{cam_report['missing_masks']},{cam_report['missing_depth']})"
            )
        print(
            f"presentation_assets: overviews={report.get('counts', {}).get('presentation_overviews')} "
            f"ok={bool(report.get('checks', {}).get('presentation_assets_ok'))}"
        )
        print(
            f"points: files={report.get('counts', {}).get('points_files')} "
            f"empty={report.get('counts', {}).get('empty_pointclouds')} "
            f"ok={bool(report.get('checks', {}).get('points_ok'))}"
        )
        capture_meta = report.get("capture_meta") or {}
        if capture_meta:
            if capture_meta.get("node_id") is not None:
                print(f"capture_meta.node_id: {capture_meta.get('node_id')}")
            print(f"capture_meta.scene_id: {capture_meta.get('scene_id')}")
            print(f"capture_meta.identity_id: {capture_meta.get('identity_id')}")
            print(f"capture_meta.target_body: {capture_meta.get('target_body')}")
            print(f"capture_meta.mjcf: {capture_meta.get('mjcf')}")
            if "expected" in capture_meta and "actual" in capture_meta:
                print(f"capture_plan_matches: {bool(report.get('checks', {}).get('capture_plan_matches'))}")
                print(f"  expected: {capture_meta['expected']}")
                print(f"  actual:   {capture_meta['actual']}")
                if "field_matches" in capture_meta:
                    print(f"  field_matches: {capture_meta['field_matches']}")
    print(f"result: {bool(report.get('overall_ok'))}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Preflight MJCF assets and validate a frozen node spin scene.")
    ap.add_argument("--scene_dir", default="", type=str, help="Scene dir like mvp-demo/data/nodes/node01/scenes/<scene_id>.")
    ap.add_argument(
        "--manifest",
        default="research/plans/tri_camera_node_3d_aware_reid/benchmarks/node01_recon_spin_v1.json",
        type=str,
        help="Frozen benchmark manifest used to derive expected capture parameters.",
    )
    ap.add_argument("--scene_id", default="", type=str, help="Scene id to resolve from --manifest.")
    ap.add_argument("--mjcf", default="", type=str, help="Optional MJCF to preflight before running MuJoCo.")
    ap.add_argument("--cams", default="", type=str, help='Comma-separated cams. Default: manifest gt_upper_bound or "cam0,cam1,cam2".')
    ap.add_argument("--mask_subdir", default="", type=str, help='Default: manifest gt_upper_bound or "masks_gt".')
    ap.add_argument("--depth_subdir", default="", type=str, help='Default: manifest gt_upper_bound or "depth_gt".')
    ap.add_argument("--points_subdir", default="", type=str, help='Default: manifest gt_upper_bound or "recon/points_fused_gt".')
    ap.add_argument("--expected_unique_timestamps", default=0, type=int)
    ap.add_argument("--max_empty_pointclouds", default=0, type=int)
    ap.add_argument("--skip_presentation_assets", action="store_true")
    ap.add_argument("--skip_points", action="store_true")
    ap.add_argument("--json", action="store_true", help="Also print a machine-readable JSON report.")
    args = ap.parse_args()

    if not str(args.scene_dir).strip() and not str(args.mjcf).strip() and not str(args.scene_id).strip():
        raise SystemExit("Provide at least one of --scene_dir, --scene_id, or --mjcf.")

    repo_root = _repo_root()
    manifest_path = Path(str(args.manifest))
    if not manifest_path.is_absolute():
        manifest_path = repo_root / manifest_path

    manifest: dict[str, Any] | None = None
    entry: dict[str, Any] | None = None
    if str(args.scene_id).strip():
        manifest, entry = _resolve_manifest_entry(manifest_path, str(args.scene_id).strip())
    elif manifest_path.exists():
        manifest = _load_json(manifest_path)

    scene_dir = Path(str(args.scene_dir)) if str(args.scene_dir).strip() else None
    if scene_dir is None and entry is not None:
        scene_dir = Path(str(entry["scene_dir"]))
        if not scene_dir.is_absolute():
            scene_dir = repo_root / scene_dir
    if scene_dir is not None:
        scene_dir = lexical_abspath(scene_dir)

    gt_cfg = _pick_gt_branch_config(manifest or {})
    cams = _normalize_cams(str(args.cams).strip() or ",".join(gt_cfg.get("cams") or ["cam0", "cam1", "cam2"]))
    mask_subdir = str(args.mask_subdir).strip() or str(gt_cfg.get("mask_subdir") or "masks_gt")
    depth_subdir = str(args.depth_subdir).strip() or str(gt_cfg.get("depth_subdir") or "depth_gt")
    points_subdir = str(args.points_subdir).strip() or str(gt_cfg.get("points_subdir") or "recon/points_fused_gt")

    full_report: dict[str, Any] = {}
    if str(args.mjcf).strip():
        mjcf_path = Path(str(args.mjcf))
        if not mjcf_path.is_absolute():
            mjcf_path = repo_root / mjcf_path
        full_report["mjcf_preflight"] = preflight_mjcf_assets(mjcf_path)

    if scene_dir is not None:
        expected_capture_plan = dict((entry or {}).get("capture_plan") or {})
        expected_unique_timestamps = (
            int(args.expected_unique_timestamps)
            if int(args.expected_unique_timestamps) > 0
            else (
                int(round(float(expected_capture_plan["fps"]) * float(expected_capture_plan["seconds"])))
                if expected_capture_plan.get("fps") is not None and expected_capture_plan.get("seconds") is not None
                else None
            )
        )
        expected_mjcf_name = (
            Path(str(expected_capture_plan.get("mjcf"))).name if expected_capture_plan.get("mjcf") else None
        )
        full_report.update(
            _scene_validation_report(
                scene_dir,
                cams=cams,
                mask_subdir=mask_subdir,
                depth_subdir=depth_subdir,
                points_subdir=points_subdir,
                require_presentation_assets=not bool(args.skip_presentation_assets),
                require_points=not bool(args.skip_points),
                expected_node_id=str((entry or {}).get("node_id")) if entry is not None else None,
                expected_scene_id=str((entry or {}).get("scene_id")) if entry is not None else None,
                expected_identity_id=str((entry or {}).get("identity_id")) if entry is not None else None,
                expected_mjcf_name=expected_mjcf_name,
                expected_capture_plan=expected_capture_plan or None,
                expected_unique_timestamps=expected_unique_timestamps,
                max_empty_pointclouds=int(args.max_empty_pointclouds),
            )
        )

    checks_ok = True
    if "mjcf_preflight" in full_report:
        checks_ok = checks_ok and bool(full_report["mjcf_preflight"].get("ok"))
    if "ok" in full_report:
        checks_ok = checks_ok and bool(full_report["ok"])
    full_report["overall_ok"] = bool(checks_ok)

    _print_human(full_report)
    if args.json:
        print("")
        print(json.dumps(full_report, indent=2, ensure_ascii=True))
    if not checks_ok:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
