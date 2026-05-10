from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_manifest_from_bundle(bundle: dict[str, Any]) -> dict[str, Any]:
    manifest = bundle.get("source_manifest")
    if not isinstance(manifest, dict):
        raise SystemExit("Bundle missing source_manifest dict")
    return manifest


def _prepare_imports(neoverse_repo: Path) -> None:
    p = str(neoverse_repo.resolve())
    if p not in sys.path:
        sys.path.insert(0, p)


def _to_rel(path: Path, scene_dir: Path) -> str:
    try:
        return path.resolve().relative_to(scene_dir.resolve()).as_posix()
    except Exception:
        return path.resolve().as_posix()


def _make_gaussian(gaussian_cls, item: dict[str, Any]):
    kwargs = {
        "means": item["means"],
        "harmonics": item["harmonics"],
        "opacities": item["opacities"],
        "scales": item["scales"],
        "rotations": item["rotations"],
        "confidences": item.get("confidences"),
        "timestamp": int(item.get("timestamp", -1)),
        "life_span": item.get("life_span", 1.0),
        "life_span_gamma": float(item.get("life_span_gamma", 0.0)),
        "forward_timestamp": item.get("forward_timestamp"),
        "forward_vel": item.get("forward_vel"),
        "forward_scales": item.get("forward_scales"),
        "forward_rotations": item.get("forward_rotations"),
        "backward_timestamp": item.get("backward_timestamp"),
        "backward_vel": item.get("backward_vel"),
        "backward_scales": item.get("backward_scales"),
        "backward_rotations": item.get("backward_rotations"),
    }
    return gaussian_cls(**kwargs)


def _voxel_downsample(points_xyz: np.ndarray, voxel_size_m: float) -> np.ndarray:
    if points_xyz.size == 0:
        return points_xyz
    if voxel_size_m <= 0:
        return points_xyz
    vox = np.floor(points_xyz / float(voxel_size_m)).astype(np.int64)
    _, unique_idx = np.unique(vox, axis=0, return_index=True)
    unique_idx = np.sort(unique_idx)
    return points_xyz[unique_idx]


def _cap_points(points_xyz: np.ndarray, max_points: int, seed: int = 42) -> np.ndarray:
    if points_xyz.shape[0] <= max_points:
        return points_xyz
    rng = np.random.default_rng(seed)
    idx = rng.choice(points_xyz.shape[0], size=max_points, replace=False)
    idx.sort()
    return points_xyz[idx]


def _count_points(points_xyz: np.ndarray) -> int:
    return int(points_xyz.shape[0])


def main() -> None:
    ap = argparse.ArgumentParser(description="Export NeoVerse multiview 4D bundle to timestamped point clouds for retrieval.")
    ap.add_argument("--bundle", required=True, type=str)
    ap.add_argument("--scene_dir", required=True, type=str)
    ap.add_argument("--neoverse_repo", default="third_party/NeoVerse", type=str)
    ap.add_argument("--out_subdir", default="recon/points_neoverse_multiview", type=str)
    ap.add_argument("--opacity_thresh", default=0.05, type=float)
    ap.add_argument("--confidence_thresh", default=0.0, type=float)
    ap.add_argument("--voxel_size_m", default=0.02, type=float)
    ap.add_argument("--max_points", default=50000, type=int)
    ap.add_argument("--min_points", default=32, type=int)
    args = ap.parse_args()

    repo_root = _repo_root()
    bundle_path = Path(str(args.bundle))
    if not bundle_path.is_absolute():
        bundle_path = repo_root / bundle_path
    bundle_path = bundle_path.resolve()
    if not bundle_path.exists():
        raise SystemExit(f"Missing bundle: {bundle_path}")

    scene_dir = Path(str(args.scene_dir)).resolve()
    if not scene_dir.exists():
        raise SystemExit(f"Missing scene dir: {scene_dir}")

    neoverse_repo = Path(str(args.neoverse_repo))
    if not neoverse_repo.is_absolute():
        neoverse_repo = repo_root / neoverse_repo
    neoverse_repo = neoverse_repo.resolve()
    if not neoverse_repo.exists():
        raise SystemExit(f"Missing neoverse repo: {neoverse_repo}")

    _prepare_imports(neoverse_repo)
    from diffsynth.auxiliary_models.worldmirror.models.models.rasterization import Gaussians

    bundle = torch.load(bundle_path, map_location="cpu")
    if not isinstance(bundle, dict):
        raise SystemExit("Invalid bundle format: expected dict")

    manifest = _load_manifest_from_bundle(bundle)
    sync_steps = list(manifest.get("sync_steps") or [])
    if not sync_steps:
        raise SystemExit("source_manifest.sync_steps is empty")

    splats_serialized = list(bundle.get("splats_serialized") or [])
    if not splats_serialized:
        raise SystemExit("Bundle has empty splats_serialized")

    out_dir = scene_dir / str(args.out_subdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale_path in out_dir.glob("*.npy"):
        stale_path.unlink()
    for stale_name in ["meta.json", "points_index.csv"]:
        stale_path = out_dir / stale_name
        if stale_path.exists():
            stale_path.unlink()

    gaussians = [_make_gaussian(Gaussians, item) for item in splats_serialized]

    index_rows: list[dict[str, Any]] = []
    total_points = 0
    skipped_small = 0
    for step in sync_steps:
        logical_t_idx = int(step["logical_t_idx"])
        scene_stem = str(step["scene_stem"])

        transitioned_chunks: list[np.ndarray] = []
        for gs in gaussians:
            base_mask = gs.opacities >= float(args.opacity_thresh)
            if gs.confidences is not None:
                base_mask = base_mask & (gs.confidences > float(args.confidence_thresh))

            transitioned = gs.transition(target_timestamp=logical_t_idx, mask=base_mask)
            if transitioned.means.numel() == 0:
                continue
            transitioned_chunks.append(transitioned.means.detach().cpu().numpy().astype(np.float32))

        if transitioned_chunks:
            points_xyz = np.concatenate(transitioned_chunks, axis=0)
        else:
            points_xyz = np.zeros((0, 3), dtype=np.float32)

        points_xyz = _voxel_downsample(points_xyz, voxel_size_m=float(args.voxel_size_m))
        points_xyz = _cap_points(points_xyz, max_points=int(args.max_points), seed=42 + logical_t_idx)
        num_points = _count_points(points_xyz)
        if num_points < int(args.min_points):
            skipped_small += 1
            continue

        out_npy = out_dir / f"{scene_stem}.npy"
        np.save(out_npy, points_xyz)

        total_points += num_points
        index_rows.append(
            {
                "logical_t_idx": logical_t_idx,
                "scene_stem": scene_stem,
                "points_rel": _to_rel(out_npy, scene_dir),
                "num_points": num_points,
            }
        )

    meta = {
        "schema_version": "points_neoverse_multiview_v2",
        "scene_id": str(manifest.get("scene_id") or scene_dir.name),
        "source_bundle": bundle_path.as_posix(),
        "out_subdir": str(args.out_subdir),
        "num_sync_steps": len(sync_steps),
        "num_files": len(index_rows),
        "total_points": total_points,
        "num_skipped_empty_or_too_small": skipped_small,
        "cleared_stale_outputs": True,
        "filters": {
            "opacity_thresh": float(args.opacity_thresh),
            "confidence_thresh": float(args.confidence_thresh),
            "voxel_size_m": float(args.voxel_size_m),
            "max_points": int(args.max_points),
            "min_points": int(args.min_points),
        },
    }

    meta_path = out_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    index_path = out_dir / "points_index.csv"
    with index_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["logical_t_idx", "scene_stem", "points_rel", "num_points"])
        writer.writeheader()
        for row in index_rows:
            writer.writerow(row)

    print(f"Wrote points dir: {out_dir}")
    print(f"Wrote: {meta_path}")
    print(f"Wrote: {index_path}")


if __name__ == "__main__":
    main()
