from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_path(repo_root: Path, value: str) -> Path:
    p = Path(str(value))
    if not p.is_absolute():
        p = repo_root / p
    return p.resolve()


def _prepare_imports(dggt_repo: Path) -> None:
    p = str(dggt_repo.resolve())
    if p not in sys.path:
        sys.path.insert(0, p)


def _to_rel(path: Path, base: Path) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except Exception:
        return path.resolve().as_posix()


def _voxel_downsample(points_xyz: np.ndarray, voxel_size_m: float) -> np.ndarray:
    if points_xyz.size == 0 or voxel_size_m <= 0:
        return points_xyz
    vox = np.floor(points_xyz / float(voxel_size_m)).astype(np.int64)
    _, unique_idx = np.unique(vox, axis=0, return_index=True)
    unique_idx = np.sort(unique_idx)
    return points_xyz[unique_idx]


def _cap_points(points_xyz: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    if points_xyz.shape[0] <= max_points:
        return points_xyz
    rng = np.random.default_rng(seed)
    idx = rng.choice(points_xyz.shape[0], size=max_points, replace=False)
    idx.sort()
    return points_xyz[idx]


def _write_ascii_ply_xyz(points_xyz: np.ndarray, out_path: Path) -> None:
    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {points_xyz.shape[0]}",
        "property float x",
        "property float y",
        "property float z",
        "end_header",
    ]
    lines = ["{:.6f} {:.6f} {:.6f}".format(float(p[0]), float(p[1]), float(p[2])) for p in points_xyz]
    out_path.write_text("\n".join(header + lines) + "\n", encoding="utf-8")


def _load_bundle(bundle_path: Path) -> dict[str, Any]:
    bundle_npz = np.load(bundle_path, allow_pickle=True)
    keys = set(bundle_npz.files)
    required = {
        "scene_id",
        "logical_t_indices",
        "scene_stem_per_view",
        "view_cam_ids",
        "frame_paths",
        "prepared_intrinsics",
        "input_extrinsics_w2c",
        "pred_depth",
        "pred_dynamic_conf",
    }
    missing = sorted(required - keys)
    if missing:
        raise SystemExit(f"Bundle missing keys: {missing}")
    return {k: bundle_npz[k] for k in bundle_npz.files}


def _resolve_bundle_scalar(bundle: dict[str, Any], key: str, default_value: Any) -> Any:
    if key not in bundle:
        return default_value
    value = np.asarray(bundle[key])
    if value.shape == ():
        return value.item()
    return value


def _build_world_points_from_bundle(bundle: dict[str, Any], dggt_repo: Path) -> tuple[np.ndarray, str]:
    if "world_points_geometry" in bundle:
        return np.asarray(bundle["world_points_geometry"], dtype=np.float32), "world_points_geometry"

    _prepare_imports(dggt_repo)
    from dggt.utils.geometry import unproject_depth_map_to_point_map

    pred_depth = np.asarray(bundle["pred_depth"], dtype=np.float32)
    if "geometry_extrinsics_w2c" in bundle and "geometry_intrinsics" in bundle:
        extr = np.asarray(bundle["geometry_extrinsics_w2c"], dtype=np.float32)
        intr = np.asarray(bundle["geometry_intrinsics"], dtype=np.float32)
        return (
            unproject_depth_map_to_point_map(depth_map=pred_depth, extrinsics_cam=extr, intrinsics_cam=intr).astype(np.float32),
            "reconstructed_from_geometry_extrinsics_w2c_and_geometry_intrinsics",
        )

    extr = np.asarray(bundle["input_extrinsics_w2c"], dtype=np.float32)
    intr = np.asarray(bundle["prepared_intrinsics"], dtype=np.float32)
    return (
        unproject_depth_map_to_point_map(depth_map=pred_depth, extrinsics_cam=extr, intrinsics_cam=intr).astype(np.float32),
        "reconstructed_from_legacy_input_extrinsics_w2c_and_prepared_intrinsics",
    )


def _quantile_filter_mask(
    conf_map: np.ndarray,
    candidate_mask: np.ndarray,
    drop_quantile: float,
) -> tuple[np.ndarray, float | None, str | None]:
    if conf_map is None:
        return np.ones(candidate_mask.shape, dtype=bool), None, "missing_conf_map"

    q = float(drop_quantile)
    if q <= 0:
        return np.ones(candidate_mask.shape, dtype=bool), None, "quantile_disabled"

    values = conf_map[candidate_mask]
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.ones(candidate_mask.shape, dtype=bool), None, "no_candidate_values"

    threshold = float(np.quantile(values, q))
    return conf_map >= threshold, threshold, None


def _resolve_mask_candidate(scene_dir: Path, frame_path: str, mask_source: str, scene_stem: str, cam_id: str) -> tuple[Path | None, str]:
    _ = frame_path

    if mask_source == "masks_gt":
        candidates = [("masks_gt", scene_dir / f"cams/{cam_id}/masks_gt/{scene_stem}.png")]
    elif mask_source == "masks":
        candidates = [("masks", scene_dir / f"cams/{cam_id}/masks/{scene_stem}.png")]
    else:
        candidates = [
            ("masks_gt", scene_dir / f"cams/{cam_id}/masks_gt/{scene_stem}.png"),
            ("masks", scene_dir / f"cams/{cam_id}/masks/{scene_stem}.png"),
        ]

    for source_name, path in candidates:
        if path.exists():
            return path.resolve(), source_name
    return None, "dynamic_conf"


def main() -> None:
    ap = argparse.ArgumentParser(description="Export DGGT multiview bundle to per-timestamp fused world-coordinate point clouds.")
    ap.add_argument("--bundle", required=True, type=str)
    ap.add_argument("--scene_dir", required=True, type=str)
    ap.add_argument("--dggt_repo", default="third_party/dggt", type=str)
    ap.add_argument("--mask_source", default="auto", choices=["auto", "masks_gt", "masks", "dynamic_conf"], type=str)
    ap.add_argument("--dynamic_thresh", default=0.5, type=float)
    ap.add_argument("--depth_conf_drop_quantile", default=0.2, type=float)
    ap.add_argument("--gs_conf_drop_quantile", default=0.2, type=float)
    ap.add_argument("--voxel_size_m", default=0.02, type=float)
    ap.add_argument("--max_points", default=50000, type=int)
    ap.add_argument("--min_points", default=32, type=int)
    args = ap.parse_args()

    repo_root = _repo_root()
    bundle_path = _resolve_path(repo_root, str(args.bundle))
    if not bundle_path.exists():
        raise SystemExit(f"Missing bundle: {bundle_path}")

    scene_dir = _resolve_path(repo_root, str(args.scene_dir))
    if not scene_dir.exists():
        raise SystemExit(f"Missing scene_dir: {scene_dir}")

    dggt_repo = _resolve_path(repo_root, str(args.dggt_repo))
    if not dggt_repo.exists():
        raise SystemExit(f"Missing dggt_repo: {dggt_repo}")

    _prepare_imports(dggt_repo)
    from dggt.utils.inference_adapter import apply_preprocess_to_mask_with_meta, PreparedImageMeta

    bundle = _load_bundle(bundle_path)

    logical_t_indices = np.asarray(bundle["logical_t_indices"], dtype=np.int32)
    scene_stem_per_view = np.asarray(bundle["scene_stem_per_view"], dtype=object)
    view_cam_ids = np.asarray(bundle["view_cam_ids"], dtype=object)
    frame_paths = np.asarray(bundle["frame_paths"], dtype=object)
    pred_depth = np.asarray(bundle["pred_depth"], dtype=np.float32)
    pred_depth_conf = np.asarray(bundle["pred_depth_conf"], dtype=np.float32) if "pred_depth_conf" in bundle else None
    pred_gs_conf = np.asarray(bundle["pred_gs_conf"], dtype=np.float32) if "pred_gs_conf" in bundle else None
    pred_dynamic_conf = np.asarray(bundle["pred_dynamic_conf"], dtype=np.float32)

    # Reconstruct preprocess metadata from probe_meta.json if available.
    probe_meta_path = bundle_path.parent / "probe_meta.json"
    preprocess_meta_by_view: dict[int, PreparedImageMeta] = {}
    if probe_meta_path.exists():
        probe_meta = json.loads(probe_meta_path.read_text(encoding="utf-8"))
        for view_idx, meta_dict in enumerate(list(probe_meta.get("preprocess_meta") or [])):
            md = dict(meta_dict)
            md.setdefault("extra_pad_top", 0)
            md.setdefault("extra_pad_bottom", 0)
            md.setdefault("extra_pad_left", 0)
            md.setdefault("extra_pad_right", 0)
            md.setdefault("final_height", int(md.get("crop_height", md.get("resized_height", 0)) + md.get("pad_top", 0) + md.get("pad_bottom", 0)))
            md.setdefault("final_width", int(md.get("crop_width", md.get("resized_width", 0)) + md.get("pad_left", 0) + md.get("pad_right", 0)))
            preprocess_meta_by_view[view_idx] = PreparedImageMeta(**md)

    world_points, geometry_source_used = _build_world_points_from_bundle(bundle, dggt_repo)

    out_root = bundle_path.parent / "points_export"
    points_dir = out_root / "points_by_timestamp"
    raw_by_view_dir = out_root / "debug" / "raw_by_view"
    fused_ply_dir = out_root / "debug" / "fused_preview_ply"
    points_dir.mkdir(parents=True, exist_ok=True)
    raw_by_view_dir.mkdir(parents=True, exist_ok=True)
    fused_ply_dir.mkdir(parents=True, exist_ok=True)

    # Clear stale outputs.
    for p in points_dir.glob("*.npy"):
        p.unlink()
    for p in raw_by_view_dir.glob("*.npy"):
        p.unlink()
    for p in fused_ply_dir.glob("*.ply"):
        p.unlink()

    unique_t = sorted(set(int(x) for x in logical_t_indices.tolist()))
    first_mid_last = set()
    if unique_t:
        first_mid_last.add(unique_t[0])
        first_mid_last.add(unique_t[len(unique_t) // 2])
        first_mid_last.add(unique_t[-1])

    index_rows: list[dict[str, Any]] = []
    total_points = 0
    skipped_count = 0
    per_view_thresholds: dict[str, dict[str, Any]] = {}
    per_step_filter_counts: list[dict[str, Any]] = []

    depth_filter_enabled = pred_depth_conf is not None
    gs_filter_enabled = pred_gs_conf is not None
    depth_filter_disabled_reason = None if depth_filter_enabled else "bundle_missing_pred_depth_conf"
    gs_filter_disabled_reason = None if gs_filter_enabled else "bundle_missing_pred_gs_conf"

    for logical_t_idx in unique_t:
        idxs = [i for i, t in enumerate(logical_t_indices.tolist()) if int(t) == logical_t_idx]
        if not idxs:
            continue

        scene_stem = str(scene_stem_per_view[idxs[0]])
        per_view_points: list[np.ndarray] = []
        valid_views = 0
        chosen_mask_sources: list[str] = []

        for i in idxs:
            depth_i = pred_depth[i]
            if depth_i.ndim == 3 and depth_i.shape[-1] == 1:
                depth_i = depth_i[..., 0]
            depth_i = np.asarray(depth_i, dtype=np.float32)

            xyz_i = world_points[i]
            cam_id = str(view_cam_ids[i])

            finite_mask = np.isfinite(xyz_i).all(axis=-1)
            positive_depth_mask = depth_i > 0

            if str(args.mask_source) == "dynamic_conf":
                dynamic_i = np.asarray(pred_dynamic_conf[i], dtype=np.float32)
                if dynamic_i.ndim == 3 and dynamic_i.shape[-1] == 1:
                    dynamic_i = dynamic_i[..., 0]
                fg_mask = 1.0 / (1.0 + np.exp(-dynamic_i)) >= float(args.dynamic_thresh)
                mask_source_used = "dynamic_conf"
            else:
                mask_path, mask_source_used = _resolve_mask_candidate(
                    scene_dir=scene_dir,
                    frame_path=str(frame_paths[i]),
                    mask_source=str(args.mask_source),
                    scene_stem=scene_stem,
                    cam_id=cam_id,
                )
                if mask_path is None:
                    dynamic_i = np.asarray(pred_dynamic_conf[i], dtype=np.float32)
                    if dynamic_i.ndim == 3 and dynamic_i.shape[-1] == 1:
                        dynamic_i = dynamic_i[..., 0]
                    fg_mask = 1.0 / (1.0 + np.exp(-dynamic_i)) >= float(args.dynamic_thresh)
                    mask_source_used = "dynamic_conf"
                else:
                    meta = preprocess_meta_by_view.get(i)
                    if meta is None:
                        raise SystemExit(
                            f"Missing preprocess metadata for view_idx={i}; cannot apply shared DGGT preprocess to mask."
                        )
                    prepared_mask = apply_preprocess_to_mask_with_meta(str(mask_path), meta)
                    fg_mask = prepared_mask > 0

            candidate_mask = finite_mask & positive_depth_mask & fg_mask

            depth_conf_i = None
            if pred_depth_conf is not None:
                depth_conf_i = np.asarray(pred_depth_conf[i], dtype=np.float32)
                if depth_conf_i.ndim == 3 and depth_conf_i.shape[-1] == 1:
                    depth_conf_i = depth_conf_i[..., 0]

            gs_conf_i = None
            if pred_gs_conf is not None:
                gs_conf_i = np.asarray(pred_gs_conf[i], dtype=np.float32)
                if gs_conf_i.ndim == 3 and gs_conf_i.shape[-1] == 1:
                    gs_conf_i = gs_conf_i[..., 0]

            depth_conf_mask, depth_thr, depth_reason = _quantile_filter_mask(
                conf_map=depth_conf_i,
                candidate_mask=candidate_mask,
                drop_quantile=float(args.depth_conf_drop_quantile),
            )
            gs_conf_mask, gs_thr, gs_reason = _quantile_filter_mask(
                conf_map=gs_conf_i,
                candidate_mask=candidate_mask,
                drop_quantile=float(args.gs_conf_drop_quantile),
            )

            chosen_mask_sources.append(mask_source_used)
            valid_mask = finite_mask & positive_depth_mask & fg_mask & depth_conf_mask & gs_conf_mask
            xyz_valid = xyz_i[valid_mask]

            per_view_key = f"{scene_stem}:{cam_id}"
            per_view_thresholds[per_view_key] = {
                "logical_t_idx": int(logical_t_idx),
                "cam_id": cam_id,
                "depth_conf_threshold": (None if depth_thr is None else float(depth_thr)),
                "gs_conf_threshold": (None if gs_thr is None else float(gs_thr)),
                "depth_conf_disabled_reason": depth_reason,
                "gs_conf_disabled_reason": gs_reason,
            }

            per_step_filter_counts.append(
                {
                    "logical_t_idx": int(logical_t_idx),
                    "cam_id": cam_id,
                    "scene_stem": scene_stem,
                    "candidate_before_conf": int(candidate_mask.sum()),
                    "after_depth_conf": int((candidate_mask & depth_conf_mask).sum()),
                    "after_gs_conf": int((candidate_mask & gs_conf_mask).sum()),
                    "final_valid": int(valid_mask.sum()),
                }
            )

            raw_path = raw_by_view_dir / f"{scene_stem}_{cam_id}.npy"
            np.save(raw_path, xyz_valid.astype(np.float32))

            if xyz_valid.shape[0] > 0:
                valid_views += 1
                per_view_points.append(xyz_valid.astype(np.float32))

        if per_view_points:
            fused_xyz = np.concatenate(per_view_points, axis=0)
        else:
            fused_xyz = np.zeros((0, 3), dtype=np.float32)

        fused_xyz = _voxel_downsample(fused_xyz, voxel_size_m=float(args.voxel_size_m))
        fused_xyz = _cap_points(fused_xyz, max_points=int(args.max_points), seed=42 + logical_t_idx)

        num_points = int(fused_xyz.shape[0])
        status = "kept"
        points_rel = ""
        if num_points < int(args.min_points):
            status = "skipped_small_or_empty"
            skipped_count += 1
        else:
            out_npy = points_dir / f"{scene_stem}.npy"
            np.save(out_npy, fused_xyz.astype(np.float32))
            points_rel = _to_rel(out_npy, out_root)
            total_points += num_points

            if logical_t_idx in first_mid_last:
                ply_path = fused_ply_dir / f"{scene_stem}.ply"
                _write_ascii_ply_xyz(fused_xyz, ply_path)

        index_rows.append(
            {
                "logical_t_idx": int(logical_t_idx),
                "scene_stem": scene_stem,
                "status": status,
                "num_valid_views": int(valid_views),
                "num_points": int(num_points),
                "points_rel": points_rel,
                "mask_source_used": "|".join(sorted(set(chosen_mask_sources))),
            }
        )

    index_path = out_root / "points_index.csv"
    with index_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "logical_t_idx",
                "scene_stem",
                "status",
                "num_valid_views",
                "num_points",
                "points_rel",
                "mask_source_used",
            ],
        )
        writer.writeheader()
        for row in index_rows:
            writer.writerow(row)

    meta = {
        "schema_version": "dggt_multiview_points_v1",
        "scene_id": str(bundle["scene_id"].item() if np.asarray(bundle["scene_id"]).shape == () else bundle["scene_id"]),
        "source_bundle": bundle_path.as_posix(),
        "scene_dir": scene_dir.as_posix(),
        "output_root": out_root.as_posix(),
        "num_timestamps": len(unique_t),
        "num_kept": int(sum(1 for r in index_rows if r["status"] == "kept")),
        "num_skipped": int(skipped_count),
        "total_points": int(total_points),
        "geometry_source_used": str(geometry_source_used),
        "filters": {
            "mask_source": str(args.mask_source),
            "dynamic_thresh": float(args.dynamic_thresh),
            "depth_conf_drop_quantile": float(args.depth_conf_drop_quantile),
            "gs_conf_drop_quantile": float(args.gs_conf_drop_quantile),
            "voxel_size_m": float(args.voxel_size_m),
            "max_points": int(args.max_points),
            "min_points": int(args.min_points),
        },
        "confidence_filtering": {
            "depth_conf_enabled": bool(depth_filter_enabled),
            "gs_conf_enabled": bool(gs_filter_enabled),
            "depth_conf_disabled_reason": depth_filter_disabled_reason,
            "gs_conf_disabled_reason": gs_filter_disabled_reason,
            "per_view_thresholds": per_view_thresholds,
            "per_step_filter_counts": per_step_filter_counts,
        },
        "outputs": {
            "points_by_timestamp": _to_rel(points_dir, out_root),
            "points_index_csv": _to_rel(index_path, out_root),
            "raw_by_view": _to_rel(raw_by_view_dir, out_root),
            "fused_preview_ply": _to_rel(fused_ply_dir, out_root),
        },
    }

    meta_path = out_root / "meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote: {index_path}")
    print(f"Wrote: {meta_path}")
    print(f"Wrote points dir: {points_dir}")


if __name__ == "__main__":
    main()
