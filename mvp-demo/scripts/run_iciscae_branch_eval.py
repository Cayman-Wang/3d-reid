from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


BRANCH_CONFIGS = {
    "rgb_predicted_depth_geometry": {
        "cams": "cam0",
        "depth_subdir": "depth",
        "mask_subdir": "masks",
        "points_subdir": "recon/points_depth_cam0",
        "tracklets_rel": "tracks_rgb_predicted_depth_geometry/tracklets.json",
        "embeddings_subdir": "embeddings_rgb_predicted_depth_geometry",
        "eval_subdir": "rgb_predicted_depth_geometry",
        "geo_backend": "open3d_fpfh",
    },
    "rgb_fused_geometry": {
        "cams": "cam0,cam1,cam2",
        "depth_subdir": "depth",
        "mask_subdir": "masks",
        "points_subdir": "recon/points_fused",
        "tracklets_rel": "tracks_rgb_fused_geometry/tracklets.json",
        "embeddings_subdir": "embeddings_rgb_fused_geometry",
        "eval_subdir": "rgb_fused_geometry",
        "geo_backend": "open3d_fpfh",
    },
    "gt_upper_bound": {
        "cams": "cam0,cam1,cam2",
        "depth_subdir": "depth_gt",
        "mask_subdir": "masks_gt",
        "points_subdir": "recon/points_fused_gt",
        "tracklets_rel": "tracks_gt_upper_bound/tracklets.json",
        "embeddings_subdir": "embeddings_gt_upper_bound",
        "eval_subdir": "gt_upper_bound",
        "geo_backend": "open3d_fpfh",
    },
}


def _run(cmd: list[str], cwd: Path) -> None:
    printable = " ".join(f'"{p}"' if " " in p else p for p in cmd)
    print(f"[run] {printable}")
    env = dict(os.environ)
    env.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    subprocess.run(cmd, cwd=str(cwd), check=True, env=env)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser(description="Run one ICISCAE branch end-to-end over the frozen 6-scene manifest.")
    ap.add_argument(
        "--manifest",
        default="research/plans/tri_camera_node_3d_aware_reid/benchmarks/iciscae_node01_uav_v1.json",
        type=str,
    )
    ap.add_argument("--branch", required=True, choices=sorted(BRANCH_CONFIGS.keys()), type=str)
    ap.add_argument("--topk", default=5, type=int)
    args = ap.parse_args()

    repo_root = _repo_root()
    manifest_path = Path(str(args.manifest))
    if not manifest_path.is_absolute():
        manifest_path = repo_root / manifest_path
    manifest = _load_manifest(manifest_path)

    benchmark_id = str(manifest["benchmark_id"])
    entries = list(manifest.get("entries") or [])
    if not entries:
        raise SystemExit(f"No entries found in manifest: {manifest_path}")

    cfg = dict(BRANCH_CONFIGS[str(args.branch)])
    scene_items: list[tuple[str, Path]] = []
    for entry in entries:
        scene_id = str(entry["scene_id"])
        scene_dir = repo_root / str(entry["scene_dir"])
        if not scene_dir.exists():
            raise SystemExit(f"Scene dir missing: {scene_dir}")
        scene_items.append((scene_id, scene_dir))

    scripts_dir = repo_root / "mvp-demo" / "scripts"
    eval_root = repo_root / "mvp-demo" / "output" / "evals" / benchmark_id / str(cfg["eval_subdir"])
    eval_root.mkdir(parents=True, exist_ok=True)

    run_meta = {
        "benchmark_id": benchmark_id,
        "branch": str(args.branch),
        "branch_config": cfg,
        "scene_ids": [scene_id for scene_id, _ in scene_items],
        "manifest": str(manifest_path),
    }
    (eval_root / "run_meta.json").write_text(json.dumps(run_meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    for scene_id, scene_dir in scene_items:
        _run(
            [
                sys.executable,
                str(scripts_dir / "recon_fuse_depth_points.py"),
                "--scene_dir",
                str(scene_dir),
                "--cams",
                str(cfg["cams"]),
                "--depth_subdir",
                str(cfg["depth_subdir"]),
                "--mask_subdir",
                str(cfg["mask_subdir"]),
                "--out_subdir",
                str(cfg["points_subdir"]),
            ],
            cwd=repo_root,
        )

        _run(
            [
                sys.executable,
                str(scripts_dir / "build_node_tracklets.py"),
                "--scene_dir",
                str(scene_dir),
                "--mask_subdir",
                str(cfg["mask_subdir"]),
                "--depth_subdir",
                str(cfg["depth_subdir"]),
                "--points_subdir",
                str(cfg["points_subdir"]),
                "--out",
                str(cfg["tracklets_rel"]),
                "--min_timestamps",
                str(manifest.get("defaults", {}).get("min_valid_timestamps", 5)),
            ],
            cwd=repo_root,
        )

        _run(
            [
                sys.executable,
                str(scripts_dir / "extract_node_track_embeddings.py"),
                "--scene_dir",
                str(scene_dir),
                "--tracklets",
                str(cfg["tracklets_rel"]),
                "--out_dir",
                str(cfg["embeddings_subdir"]),
                "--rgb_backend",
                "clip",
                "--geo_backend",
                str(cfg["geo_backend"]),
            ],
            cwd=repo_root,
        )

    for scene_id, scene_dir in scene_items:
        per_query_out = eval_root / f"{scene_id}.json"
        cmd = [
            sys.executable,
            str(scripts_dir / "eval_node_track_retrieval.py"),
            "--query_scene_dir",
            str(scene_dir),
        ]
        for _, gallery_scene_dir in scene_items:
            cmd.extend(["--gallery_scene_dir", str(gallery_scene_dir)])
        cmd.extend(
            [
                "--topk",
                str(args.topk),
                "--exclude_same_track_id",
                "--exclude_same_scene",
                "--embeddings_subdir",
                str(cfg["embeddings_subdir"]),
                "--out",
                str(per_query_out),
            ]
        )
        _run(cmd, cwd=repo_root)

    all_out = eval_root / "all_queries_vs_all_scenes.json"
    all_cmd = [
        sys.executable,
        str(scripts_dir / "eval_node_track_retrieval.py"),
    ]
    for _, scene_dir in scene_items:
        all_cmd.extend(["--query_scene_dir", str(scene_dir)])
    for _, scene_dir in scene_items:
        all_cmd.extend(["--gallery_scene_dir", str(scene_dir)])
    all_cmd.extend(
        [
            "--topk",
            str(args.topk),
            "--exclude_same_track_id",
            "--exclude_same_scene",
            "--embeddings_subdir",
            str(cfg["embeddings_subdir"]),
            "--out",
            str(all_out),
        ]
    )
    _run(all_cmd, cwd=repo_root)


if __name__ == "__main__":
    main()
