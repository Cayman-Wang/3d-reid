from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


DEFAULT_BRANCH_CONFIGS = {
    "rgb_only": {
        "cams": "cam0,cam1,cam2",
        "depth_subdir": "depth",
        "mask_subdir": "masks",
        "points_subdir": "recon/points_rgb_only_unused",
        "tracklets_rel": "tracks/tracklets.json",
        "embeddings_subdir": "embeddings",
        "eval_subdir": "rgb_only",
        "rgb_backend": "clip",
        "geo_backend": "none",
        "run_recon": False,
    },
    "rgb_predicted_depth_geometry": {
        "cams": "cam0",
        "depth_subdir": "depth",
        "mask_subdir": "masks",
        "points_subdir": "recon/points_depth_cam0",
        "tracklets_rel": "tracks_rgb_predicted_depth_geometry/tracklets.json",
        "embeddings_subdir": "embeddings_rgb_predicted_depth_geometry",
        "eval_subdir": "rgb_predicted_depth_geometry",
        "rgb_backend": "clip",
        "geo_backend": "open3d_fpfh",
        "run_recon": True,
    },
    "rgb_fused_geometry": {
        "cams": "cam0,cam1,cam2",
        "depth_subdir": "depth",
        "mask_subdir": "masks",
        "points_subdir": "recon/points_fused",
        "tracklets_rel": "tracks_rgb_fused_geometry/tracklets.json",
        "embeddings_subdir": "embeddings_rgb_fused_geometry",
        "eval_subdir": "rgb_fused_geometry",
        "rgb_backend": "clip",
        "geo_backend": "open3d_fpfh",
        "run_recon": True,
    },
    "gt_upper_bound": {
        "cams": "cam0,cam1,cam2",
        "depth_subdir": "depth_gt",
        "mask_subdir": "masks_gt",
        "points_subdir": "recon/points_fused_gt",
        "tracklets_rel": "tracks_gt_upper_bound/tracklets.json",
        "embeddings_subdir": "embeddings_gt_upper_bound",
        "eval_subdir": "gt_upper_bound",
        "rgb_backend": "clip",
        "geo_backend": "open3d_fpfh",
        "run_recon": True,
    },
}


def _run(cmd: list[str], cwd: Path) -> None:
    printable = " ".join(f'"{p}"' if " " in p else p for p in cmd)
    print(f"[run] {printable}")
    env = dict(os.environ)
    env.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    subprocess.run(cmd, cwd=str(cwd), check=True, env=env)


def _repo_root() -> Path:
    return Path(__file__).absolute().parents[2]


def _load_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_manifest_path(repo_root: Path, path_text: str) -> Path:
    path = Path(str(path_text))
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def _resolve_entry_scene_dir(repo_root: Path, scene_dir_text: str) -> Path:
    path = Path(str(scene_dir_text))
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def _resolve_scene_subdir(scene_dir: Path, subdir_text: str) -> Path:
    path = Path(str(subdir_text))
    if path.is_absolute():
        return path
    return scene_dir / path


def _format_cfg_value(value, context: dict[str, str], key: str):
    if not isinstance(value, str):
        return value
    if "{" not in value:
        return value
    try:
        return value.format_map(context)
    except KeyError as exc:
        raise SystemExit(f"Branch config field {key!r} references missing template key: {exc.args[0]!r}") from exc


def _resolve_entry_cfg(cfg: dict, entry: dict, scene_id: str, scene_dir: Path, repo_root: Path) -> dict:
    context = {str(k): str(v) for k, v in entry.items() if not isinstance(v, (dict, list))}
    context.setdefault("scene_id", str(scene_id))
    context.setdefault("scene_name", scene_dir.name)
    context.setdefault("scene_dir", str(scene_dir))
    context.setdefault("repo_root", str(repo_root))
    return {key: _format_cfg_value(value, context, key) for key, value in cfg.items()}


def _ensure_points_ready(scene_dir: Path, points_subdir: str, branch: str) -> None:
    points_dir = _resolve_scene_subdir(scene_dir, str(points_subdir))
    if not points_dir.exists():
        raise SystemExit(
            f'Branch "{branch}" requires precomputed geometry, but points dir is missing: {points_dir}'
        )
    if not any(points_dir.glob("*.npy")):
        raise SystemExit(
            f'Branch "{branch}" requires precomputed geometry, but no *.npy files were found in: {points_dir}'
        )


def _resolve_branch_config(manifest: dict, branch: str) -> tuple[dict, list[str]]:
    manifest_cfgs = manifest.get("branch_configs") or {}
    available = sorted(set(DEFAULT_BRANCH_CONFIGS.keys()) | set(manifest_cfgs.keys()))

    if branch in manifest_cfgs:
        cfg = dict(manifest_cfgs[branch])
    elif branch in DEFAULT_BRANCH_CONFIGS:
        cfg = dict(DEFAULT_BRANCH_CONFIGS[branch])
    else:
        raise SystemExit(f"Unknown --branch={branch!r}. Available: {available}")

    cfg.setdefault("eval_subdir", branch)
    cfg.setdefault("mask_subdir", "masks")
    cfg.setdefault("depth_subdir", "depth")
    cfg.setdefault("tracklets_rel", "tracks/tracklets.json")
    cfg.setdefault("embeddings_subdir", "embeddings")
    cfg.setdefault("rgb_backend", "clip")
    cfg.setdefault("geo_backend", "none")
    cfg.setdefault("require_points", False)
    cfg.setdefault("require_points_per_timestamp", False)
    cfg.setdefault("allow_missing_depth", False)
    cfg.setdefault("min_points_per_timestamp", 1)
    cfg.setdefault(
        "points_subdir",
        "recon/points_rgb_only_unused" if str(cfg["geo_backend"]) == "none" else "recon/points_fused",
    )
    cfg.setdefault("run_recon", str(cfg["geo_backend"]) != "none")

    cams = cfg.get("cams", "cam0,cam1,cam2")
    if isinstance(cams, (list, tuple)):
        cfg["cams"] = ",".join(str(cam).strip() for cam in cams if str(cam).strip())
    else:
        cfg["cams"] = str(cams)

    return cfg, available


def main() -> None:
    ap = argparse.ArgumentParser(description="Run one ICISCAE branch end-to-end over the frozen 6-scene manifest.")
    ap.add_argument(
        "--manifest",
        default="research/plans/tri_camera_node_3d_aware_reid/benchmarks/iciscae_node01_uav_v3_clean.json",
        type=str,
    )
    ap.add_argument("--branch", required=True, type=str)
    ap.add_argument("--topk", default=5, type=int)
    args = ap.parse_args()

    repo_root = _repo_root()
    manifest_path = _resolve_manifest_path(repo_root, str(args.manifest))
    manifest = _load_manifest(manifest_path)

    benchmark_id = str(manifest["benchmark_id"])
    entries = list(manifest.get("entries") or [])
    if not entries:
        raise SystemExit(f"No entries found in manifest: {manifest_path}")

    cfg, available_branches = _resolve_branch_config(manifest, str(args.branch))
    print(f"[cfg] branch={args.branch} available={available_branches}")
    scene_items: list[tuple[str, Path, dict]] = []
    for entry in entries:
        scene_id = str(entry["scene_id"])
        scene_dir = _resolve_entry_scene_dir(repo_root, str(entry["scene_dir"]))
        if not scene_dir.exists():
            raise SystemExit(f"Scene dir missing: {scene_dir}")
        scene_items.append((scene_id, scene_dir, dict(entry)))

    scripts_dir = repo_root / "mvp-demo" / "scripts"
    eval_root = repo_root / "mvp-demo" / "output" / "evals" / benchmark_id / str(cfg["eval_subdir"])
    eval_root.mkdir(parents=True, exist_ok=True)

    run_meta = {
        "benchmark_id": benchmark_id,
        "branch": str(args.branch),
        "branch_config": cfg,
        "resolved_branch_configs": {
            scene_id: _resolve_entry_cfg(cfg, entry, scene_id, scene_dir, repo_root)
            for scene_id, scene_dir, entry in scene_items
        },
        "scene_ids": [scene_id for scene_id, _, _ in scene_items],
        "manifest": str(manifest_path),
    }
    (eval_root / "run_meta.json").write_text(json.dumps(run_meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    for scene_id, scene_dir, entry in scene_items:
        scene_cfg = _resolve_entry_cfg(cfg, entry, scene_id, scene_dir, repo_root)
        if bool(scene_cfg.get("run_recon")):
            _run(
                [
                    sys.executable,
                    str(scripts_dir / "recon_fuse_depth_points.py"),
                    "--scene_dir",
                    str(scene_dir),
                    "--cams",
                    str(scene_cfg["cams"]),
                    "--depth_subdir",
                    str(scene_cfg["depth_subdir"]),
                    "--mask_subdir",
                    str(scene_cfg["mask_subdir"]),
                    "--out_subdir",
                    str(scene_cfg["points_subdir"]),
                ],
                cwd=repo_root,
            )

        if bool(scene_cfg.get("require_points")):
            _ensure_points_ready(scene_dir, str(scene_cfg["points_subdir"]), str(args.branch))

        _run(
            [
                sys.executable,
                str(scripts_dir / "build_node_tracklets.py"),
                "--scene_dir",
                str(scene_dir),
                "--cams",
                str(scene_cfg["cams"]),
                "--mask_subdir",
                str(scene_cfg["mask_subdir"]),
                "--depth_subdir",
                str(scene_cfg["depth_subdir"]),
                "--points_subdir",
                str(scene_cfg["points_subdir"]),
                "--out",
                str(scene_cfg["tracklets_rel"]),
                "--min_timestamps",
                str(manifest.get("defaults", {}).get("min_valid_timestamps", 5)),
            ]
            + (["--points_contract", str(scene_cfg["points_contract"])] if str(scene_cfg.get("points_contract", "")).strip() else [])
            + (["--allow_missing_depth"] if bool(scene_cfg.get("allow_missing_depth")) else [])
            + (
                ["--require_points", "--min_points", str(scene_cfg.get("min_points_per_timestamp", 1))]
                if bool(scene_cfg.get("require_points_per_timestamp"))
                else []
            ),
            cwd=repo_root,
        )

        embed_cmd = [
            sys.executable,
            str(scripts_dir / "extract_node_track_embeddings.py"),
            "--scene_dir",
            str(scene_dir),
            "--tracklets",
            str(scene_cfg["tracklets_rel"]),
            "--out_dir",
            str(scene_cfg["embeddings_subdir"]),
            "--rgb_backend",
            str(scene_cfg["rgb_backend"]),
            "--geo_backend",
            str(scene_cfg["geo_backend"]),
        ]
        passthrough_keys = [
            "max_timestamps_per_track",
            "max_points_per_timestamp",
            "seed",
            "geo_bins",
            "device",
            "clip_model",
            "clip_pretrained",
            "rgb_weight",
            "geo_weight",
        ]
        for key in passthrough_keys:
            value = scene_cfg.get(key)
            if value is None:
                continue
            embed_cmd.extend([f"--{key}", str(value)])
        if bool(scene_cfg.get("apply_mask_to_rgb")):
            embed_cmd.append("--apply_mask_to_rgb")
        _run(embed_cmd, cwd=repo_root)

    for scene_id, scene_dir, _ in scene_items:
        per_query_out = eval_root / f"{scene_id}.json"
        cmd = [
            sys.executable,
            str(scripts_dir / "eval_node_track_retrieval.py"),
            "--query_scene_dir",
            str(scene_dir),
        ]
        for _, gallery_scene_dir, _ in scene_items:
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
    for _, scene_dir, _ in scene_items:
        all_cmd.extend(["--query_scene_dir", str(scene_dir)])
    for _, scene_dir, _ in scene_items:
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
