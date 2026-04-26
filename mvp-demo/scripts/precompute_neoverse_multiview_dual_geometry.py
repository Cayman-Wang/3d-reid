from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_path(path_value: str, repo: Path) -> Path:
    path = Path(str(path_value))
    if not path.is_absolute():
        path = repo / path
    return path.resolve()


def _run(cmd: list[str], cwd: Path) -> None:
    printable = " ".join(f'"{part}"' if " " in part else part for part in cmd)
    print(f"[run] {printable}")
    env = dict(os.environ)
    env.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    subprocess.run(cmd, cwd=str(cwd), check=True, env=env)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run NeoVerse dual-branch multiview world fusion: full_frame background + object_crop dynamic."
    )
    ap.add_argument("--scene_dir", required=True, type=str)
    ap.add_argument("--neoverse_python", required=True, type=str)
    ap.add_argument("--neoverse_repo", default="third_party/NeoVerse", type=str)
    ap.add_argument("--reconstructor_path", default="third_party/NeoVerse/models/NeoVerse/reconstructor.ckpt", type=str)
    ap.add_argument("--export_torch_dtype", default="float32", type=str)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    repo = _repo_root()
    scripts = repo / "mvp-demo" / "scripts"
    scene_dir = _resolve_path(str(args.scene_dir), repo)
    neoverse_python = _resolve_path(str(args.neoverse_python), repo)
    neoverse_repo = _resolve_path(str(args.neoverse_repo), repo)
    reconstructor_path = _resolve_path(str(args.reconstructor_path), repo)

    if not scene_dir.exists():
        raise SystemExit(f"Missing --scene_dir: {scene_dir}")
    if not neoverse_python.exists():
        raise SystemExit(f"Missing --neoverse_python: {neoverse_python}")
    if not neoverse_repo.exists():
        raise SystemExit(f"Missing --neoverse_repo: {neoverse_repo}")
    if not reconstructor_path.exists():
        raise SystemExit(f"Missing --reconstructor_path: {reconstructor_path}")

    scene_id = scene_dir.name
    dual_root = repo / "mvp-demo" / "output" / "neoverse_dual"
    full_root = dual_root / "branches" / "full_frame"
    crop_root = dual_root / "branches" / "object_crop"
    merged_root = dual_root / "merged"

    # NeoVerse input size constraint: width/height must be divisible by 14.
    full_w, full_h = 280, 168
    crop_w, crop_h = 224, 224
    if (full_w % 14) != 0 or (full_h % 14) != 0 or (crop_w % 14) != 0 or (crop_h % 14) != 0:
        raise SystemExit("Configured branch resolutions must be multiples of 14.")

    started_at = datetime.now(timezone.utc).isoformat()

    common_prefix = [
        str(neoverse_python),
    ]

    run_flag = ["--overwrite"] if bool(args.overwrite) else []

    # Branch A: full_frame for background fidelity.
    _run(
        common_prefix
        + [
            str(scripts / "run_neoverse_per_camera_bundle.py"),
            "--scene_dir",
            str(scene_dir),
            "--neoverse_repo",
            str(neoverse_repo),
            "--reconstructor_path",
            str(reconstructor_path),
            "--scene_mode",
            "general",
            "--width",
            str(full_w),
            "--height",
            str(full_h),
            "--resize_mode",
            "center_crop",
            "--num_frames",
            "81",
            "--input_variant",
            "full_frame",
            "--out_root",
            str(full_root),
        ]
        + run_flag,
        cwd=repo,
    )
    _run(
        common_prefix
        + [
            str(scripts / "export_neoverse_view_observations.py"),
            "--scene_dir",
            str(scene_dir),
            "--bundle_root",
            str(full_root),
            "--neoverse_repo",
            str(neoverse_repo),
            "--reconstructor_path",
            str(reconstructor_path),
            "--torch_dtype",
            str(args.export_torch_dtype),
            "--camera_source",
            "rendered",
            "--out_root",
            str(full_root),
        ]
        + run_flag,
        cwd=repo,
    )
    _run(
        common_prefix
        + [
            str(scripts / "backproject_neoverse_observations.py"),
            "--scene_dir",
            str(scene_dir),
            "--observations_root",
            str(full_root),
            "--camera_source",
            "rendered",
            "--out_root",
            str(full_root),
        ],
        cwd=repo,
    )

    # Branch B: object_crop for dynamic enhancement on 8GB GPU-safe resolution.
    _run(
        common_prefix
        + [
            str(scripts / "run_neoverse_per_camera_bundle.py"),
            "--scene_dir",
            str(scene_dir),
            "--neoverse_repo",
            str(neoverse_repo),
            "--reconstructor_path",
            str(reconstructor_path),
            "--scene_mode",
            "general",
            "--width",
            str(crop_w),
            "--height",
            str(crop_h),
            "--resize_mode",
            "resize",
            "--num_frames",
            "81",
            "--input_variant",
            "object_crop",
            "--crop_padding",
            "0.25",
            "--crop_mask_source",
            "auto",
            "--out_root",
            str(crop_root),
        ]
        + run_flag,
        cwd=repo,
    )
    _run(
        common_prefix
        + [
            str(scripts / "export_neoverse_view_observations.py"),
            "--scene_dir",
            str(scene_dir),
            "--bundle_root",
            str(crop_root),
            "--neoverse_repo",
            str(neoverse_repo),
            "--reconstructor_path",
            str(reconstructor_path),
            "--torch_dtype",
            str(args.export_torch_dtype),
            "--camera_source",
            "rendered",
            "--out_root",
            str(crop_root),
        ]
        + run_flag,
        cwd=repo,
    )
    _run(
        common_prefix
        + [
            str(scripts / "backproject_neoverse_observations.py"),
            "--scene_dir",
            str(scene_dir),
            "--observations_root",
            str(crop_root),
            "--camera_source",
            "rendered",
            "--fg_alpha_thresh",
            "0.01",
            "--bg_alpha_thresh",
            "0.02",
            "--fg_voxel_size_m",
            "0.005",
            "--bg_voxel_size_m",
            "0.02",
            "--mask_dilate_px",
            "3",
            "--out_root",
            str(crop_root),
        ],
        cwd=repo,
    )

    # Merge: background from full_frame, dynamic from object_crop.
    _run(
        common_prefix
        + [
            str(scripts / "fuse_neoverse_multiview_world_points.py"),
            "--scene_dir",
            str(scene_dir),
            "--bg_points_root",
            str(full_root),
            "--fg_points_root",
            str(crop_root),
            "--out_root",
            str(merged_root),
            "--bg_voxel_size_m",
            "0.02",
            "--min_bg_cam_support",
            "2",
            "--dynamic_voxel_size_m",
            "0.01",
            "--dynamic_track_radius_m",
            "0.40",
            "--dynamic_merge_radius_m",
            "0.08",
            "--dynamic_min_component_points",
            "12",
            "--background_source_branch",
            "full_frame",
            "--dynamic_source_branch",
            "object_crop",
        ],
        cwd=repo,
    )
    _run(
        common_prefix
        + [
            str(scripts / "constrain_neoverse_multiview_dynamic.py"),
            "--scene_dir",
            str(scene_dir),
            "--fg_points_root",
            str(crop_root),
            "--fused_root",
            str(merged_root),
            "--points_by_timestamp_root",
            str(merged_root),
            "--hull_voxel_size_m",
            "0.02",
            "--output_voxel_size_m",
            "0.01",
            "--roi_padding_m",
            "0.12",
            "--min_mask_cam_support",
            "2",
            "--point_support_radius_m",
            "0.03",
            "--depth_trim_radius_m",
            "0.06",
            "--min_trimmed_points",
            "40",
            "--depth_support_mode",
            "anchor_depth_reproject",
            "--scale_guard_ratio",
            "0.25",
            "--min_depth_mask_pixels",
            "24",
            "--depth_support_source",
            "aligned_fg_points",
            "--max_roi_voxels",
            "400000",
        ],
        cwd=repo,
    )
    _run(
        common_prefix
        + [
            str(scripts / "render_fused_world_preview.py"),
            "--scene_dir",
            str(scene_dir),
            "--fused_root",
            str(merged_root),
        ],
        cwd=repo,
    )
    _run(
        common_prefix
        + [
            str(scripts / "analyze_fused_multiview_quality.py"),
            "--scene_dir",
            str(scene_dir),
            "--fused_root",
            str(merged_root),
            "--out_root",
            str(merged_root),
        ],
        cwd=repo,
    )

    finished_at = datetime.now(timezone.utc).isoformat()
    run_report = {
        "schema_version": "neoverse_dual_orchestrator_v1",
        "scene_id": scene_id,
        "scene_dir": scene_dir.as_posix(),
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
        "export_torch_dtype": str(args.export_torch_dtype),
        "branches": {
            "full_frame": {
                "out_root": full_root.as_posix(),
                "input_variant": "full_frame",
                "scene_mode": "general",
                "width": full_w,
                "height": full_h,
                "resize_mode": "center_crop",
                "num_frames": 81,
                "bg_used_for_merge": True,
                "fg_used_for_merge": False,
            },
            "object_crop": {
                "out_root": crop_root.as_posix(),
                "input_variant": "object_crop",
                "crop_padding": 0.25,
                "crop_mask_source": "auto",
                "scene_mode": "general",
                "width": crop_w,
                "height": crop_h,
                "resize_mode": "resize",
                "num_frames": 81,
                "fg_alpha_thresh": 0.01,
                "bg_alpha_thresh": 0.02,
                "fg_voxel_size_m": 0.005,
                "bg_voxel_size_m": 0.02,
                "mask_dilate_px": 3,
                "bg_used_for_merge": False,
                "fg_used_for_merge": True,
            },
        },
        "merged": {
            "out_root": merged_root.as_posix(),
            "background_source_branch": "full_frame",
            "dynamic_source_branch": "object_crop",
            "dynamic_constraint": {
                "script": (scripts / "constrain_neoverse_multiview_dynamic.py").as_posix(),
                "fg_points_root": crop_root.as_posix(),
                "fused_root": merged_root.as_posix(),
                "points_by_timestamp_root": merged_root.as_posix(),
                "hull_voxel_size_m": 0.02,
                "output_voxel_size_m": 0.01,
                "roi_padding_m": 0.12,
                "min_mask_cam_support": 2,
                "point_support_radius_m": 0.03,
                "depth_trim_radius_m": 0.06,
                "min_trimmed_points": 40,
                "depth_support_mode": "anchor_depth_reproject",
                "scale_guard_ratio": 0.25,
                "min_depth_mask_pixels": 24,
                "depth_support_source": "aligned_fg_points",
                "max_roi_voxels": 400000,
            },
        },
    }

    report_path = merged_root / scene_id / "run_dual_branch_report.json"
    _write_json(report_path, run_report)
    print(f"Wrote dual-branch run report to: {report_path}")


if __name__ == "__main__":
    main()
