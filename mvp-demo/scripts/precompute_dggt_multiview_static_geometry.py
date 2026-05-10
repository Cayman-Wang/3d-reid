from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _run(cmd: list[str], cwd: Path) -> None:
    printable = " ".join(f'"{x}"' if " " in x else x for x in cmd)
    print(f"[run] {printable}")
    env = dict(os.environ)
    env.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    subprocess.run(cmd, cwd=str(cwd), check=True, env=env)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _resolve_path(repo_root: Path, value: str) -> Path:
    p = Path(str(value))
    if not p.is_absolute():
        p = repo_root / p
    return p.resolve()


def _check_outputs(stage_name: str, expected_paths: list[Path]) -> list[str]:
    missing: list[str] = []
    for p in expected_paths:
        if not p.exists():
            missing.append(p.as_posix())
    if missing:
        raise RuntimeError(f"Stage {stage_name} missing outputs: {missing}")
    return missing


def main() -> None:
    ap = argparse.ArgumentParser(description="Precompute DGGT multiview geometry for one static scene.")
    ap.add_argument("--scene_dir", default="mvp-demo/data/nodes/node01/scenes/mj_node01_j10_spin_static_yp_a", type=str)
    ap.add_argument("--dggt_python", required=True, type=str)
    ap.add_argument("--dggt_repo", default="third_party/dggt", type=str)
    ap.add_argument("--ckpt_path", required=True, type=str)
    ap.add_argument("--num_sync_steps", default=27, type=int)
    ap.add_argument("--mask_source", default="auto", choices=["auto", "masks_gt", "masks", "dynamic_conf"], type=str)
    ap.add_argument("--device", default="cuda", type=str)
    ap.add_argument("--torch_dtype", default="float32", type=str)
    ap.add_argument("--use_input_calib", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--voxel_size_m", default=0.02, type=float)
    ap.add_argument("--max_points", default=50000, type=int)
    ap.add_argument("--min_points", default=32, type=int)
    args = ap.parse_args()

    repo_root = _repo_root()
    scripts_dir = repo_root / "mvp-demo" / "scripts"

    dggt_python = _resolve_path(repo_root, str(args.dggt_python))
    scene_id = Path(str(args.scene_dir)).name if str(args.scene_dir).strip() else "unknown_scene"
    out_root = repo_root / "mvp-demo" / "output" / "dggt_multiview"
    failure_path = out_root / "precompute_failures.json"

    precheck_failures: list[dict[str, Any]] = []
    if not dggt_python.exists():
        precheck_failures.append(
            {
                "scene_id": scene_id,
                "status": "error",
                "failed_stage": "precheck",
                "error_type": "MissingPath",
                "error": f"Missing dggt_python: {dggt_python}",
            }
        )

    scene_dir = _resolve_path(repo_root, str(args.scene_dir))
    if not scene_dir.exists():
        precheck_failures.append(
            {
                "scene_id": scene_id,
                "status": "error",
                "failed_stage": "precheck",
                "error_type": "MissingPath",
                "error": f"Missing scene_dir: {scene_dir}",
            }
        )

    dggt_repo = _resolve_path(repo_root, str(args.dggt_repo))
    if not dggt_repo.exists():
        precheck_failures.append(
            {
                "scene_id": scene_id,
                "status": "error",
                "failed_stage": "precheck",
                "error_type": "MissingPath",
                "error": f"Missing dggt_repo: {dggt_repo}",
            }
        )

    ckpt_path = _resolve_path(repo_root, str(args.ckpt_path))
    if not ckpt_path.exists():
        precheck_failures.append(
            {
                "scene_id": scene_id,
                "status": "error",
                "failed_stage": "precheck",
                "error_type": "MissingPath",
                "error": f"Missing ckpt_path: {ckpt_path}",
            }
        )

    if precheck_failures:
        _write_json(
            failure_path,
            {
                "scene_id": scene_id,
                "num_failures": len(precheck_failures),
                "failures": precheck_failures,
                "stages": [],
            },
        )
        print(f"Wrote: {failure_path}")
        return

    scene_id = scene_dir.name
    manifest_path = out_root / scene_id / "input" / "manifest.json"
    bundle_path = out_root / scene_id / "run_full_frame_joint" / "reconstruction_bundle.npz"

    failures: list[dict[str, Any]] = []
    stage_commands: list[dict[str, Any]] = []

    pose_report_path = out_root / scene_id / "run_full_frame_joint" / "pose_alignment_report.json"
    probe_meta_path = out_root / scene_id / "run_full_frame_joint" / "probe_meta.json"
    points_meta_path = out_root / scene_id / "run_full_frame_joint" / "points_export" / "meta.json"
    points_index_path = out_root / scene_id / "run_full_frame_joint" / "points_export" / "points_index.csv"

    try:
        cmd_prepare = [
                str(dggt_python),
                str(scripts_dir / "prepare_dggt_multiview_manifest.py"),
                "--scene_dir",
                str(scene_dir),
                "--cam_ids",
                "cam0,cam1,cam2",
                "--num_sync_steps",
                str(args.num_sync_steps),
                "--mask_source",
                str(args.mask_source if args.mask_source in {"auto", "masks_gt", "masks"} else "auto"),
            ]
        stage_commands.append({"stage": "prepare", "cmd": cmd_prepare})
        _run(cmd_prepare, cwd=repo_root)
        _check_outputs("prepare", [manifest_path])

        cmd_joint = [
                str(dggt_python),
                str(scripts_dir / "run_dggt_multiview_joint.py"),
                "--manifest",
                str(manifest_path),
                "--dggt_repo",
                str(dggt_repo),
                "--ckpt_path",
                str(ckpt_path),
                "--device",
                str(args.device),
                "--torch_dtype",
                str(args.torch_dtype),
                "--use_input_calib" if bool(args.use_input_calib) else "--no-use_input_calib",
            ]
        stage_commands.append({"stage": "joint_run", "cmd": cmd_joint})
        _run(cmd_joint, cwd=repo_root)
        _check_outputs("joint_run", [bundle_path, pose_report_path, probe_meta_path])

        cmd_export = [
                str(dggt_python),
                str(scripts_dir / "export_dggt_multiview_points.py"),
                "--bundle",
                str(bundle_path),
                "--scene_dir",
                str(scene_dir),
                "--dggt_repo",
                str(dggt_repo),
                "--mask_source",
                str(args.mask_source),
                "--voxel_size_m",
                str(args.voxel_size_m),
                "--max_points",
                str(args.max_points),
                "--min_points",
                str(args.min_points),
            ]
        stage_commands.append({"stage": "export", "cmd": cmd_export})
        _run(cmd_export, cwd=repo_root)
        _check_outputs("export", [points_meta_path, points_index_path])
    except Exception as exc:
        failed_stage = "unknown"
        if stage_commands:
            failed_stage = str(stage_commands[-1].get("stage") or "unknown")
        failures.append(
            {
                "scene_id": scene_id,
                "status": "error",
                "failed_stage": failed_stage,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "last_cmd": stage_commands[-1]["cmd"] if stage_commands else None,
                "expected_outputs": {
                    "prepare": [manifest_path.as_posix()],
                    "joint_run": [bundle_path.as_posix(), pose_report_path.as_posix(), probe_meta_path.as_posix()],
                    "export": [points_meta_path.as_posix(), points_index_path.as_posix()],
                },
            }
        )

    _write_json(
        failure_path,
        {
            "scene_id": scene_id,
            "num_failures": len(failures),
            "failures": failures,
            "stages": stage_commands,
        },
    )
    print(f"Wrote: {failure_path}")


if __name__ == "__main__":
    main()
