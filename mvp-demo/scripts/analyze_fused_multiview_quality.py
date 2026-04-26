from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"Failed to read json: {path}\nError: {exc!r}")


def _read_sync_rows(frame_times_csv: Path, cams: list[str]) -> list[tuple[int, dict[str, str]]]:
    rows_by_ts: dict[int, dict[str, str]] = {}
    with frame_times_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts_us = int(row["ts_us"])
            cam_id = str(row["cam_id"]).strip()
            if cam_id not in cams:
                continue
            rows_by_ts.setdefault(ts_us, {})[cam_id] = str(row["filename"]).strip()
    ordered = sorted(rows_by_ts.items(), key=lambda kv: kv[0])
    return [(ts_us, by_cam) for ts_us, by_cam in ordered if all(cam in by_cam for cam in cams)]


def _load_sampled_scene_stems(dynamic_index_path: Path) -> list[str]:
    rows = list(csv.DictReader(dynamic_index_path.open("r", encoding="utf-8", newline="")))
    if not rows:
        raise SystemExit(f"No dynamic rows found in: {dynamic_index_path}")
    rows.sort(key=lambda row: int(float(row.get("logical_t_idx", 0))))
    stems: list[str] = []
    seen: set[str] = set()
    for row in rows:
        scene_stem = str(row["scene_stem"])
        if scene_stem in seen:
            continue
        seen.add(scene_stem)
        stems.append(scene_stem)
    return stems


def _frame_times_by_stem(frame_times_csv: Path, cams: list[str]) -> dict[str, dict[str, str]]:
    by_stem: dict[str, dict[str, str]] = {}
    with frame_times_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cam_id = str(row["cam_id"]).strip()
            if cam_id not in cams:
                continue
            stem = Path(str(row["filename"]).strip()).stem
            by_stem.setdefault(stem, {})[cam_id] = str(row["filename"]).strip()
    return by_stem


def _read_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def _read_gray(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8)


def _project_hit_mask(points_world: np.ndarray, K: np.ndarray, c2w: np.ndarray, width: int, height: int) -> np.ndarray:
    if points_world.size == 0:
        return np.zeros((height, width), dtype=bool)
    w2c = np.linalg.inv(c2w).astype(np.float32)
    R = w2c[:3, :3]
    t = w2c[:3, 3]
    pts = np.asarray(points_world, dtype=np.float32)
    pts_cam = pts @ R.T + t[None, :]
    z = pts_cam[:, 2]
    valid = np.isfinite(z) & (z > 1e-6)
    if not np.any(valid):
        return np.zeros((height, width), dtype=bool)
    pts_cam = pts_cam[valid]
    z = z[valid]
    u = (K[0, 0] * (pts_cam[:, 0] / z) + K[0, 2]).round().astype(np.int32)
    v = (K[1, 1] * (pts_cam[:, 1] / z) + K[1, 2]).round().astype(np.int32)
    valid = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    hit = np.zeros((height, width), dtype=bool)
    hit[v[valid], u[valid]] = True
    return hit


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze quality of fused NeoVerse multiview world points.")
    ap.add_argument("--scene_dir", required=True, type=str)
    ap.add_argument("--fused_root", default="mvp-demo/output/neoverse_fused", type=str)
    ap.add_argument("--cams", default="cam0,cam1,cam2", type=str)
    ap.add_argument("--out_root", default="mvp-demo/output/neoverse_fused", type=str)
    args = ap.parse_args()

    repo = _repo_root()
    scene_dir = Path(str(args.scene_dir)).resolve()
    if not scene_dir.exists():
        raise SystemExit(f"Missing --scene_dir: {scene_dir}")

    fused_root = Path(str(args.fused_root))
    if not fused_root.is_absolute():
        fused_root = repo / fused_root
    out_root = Path(str(args.out_root))
    if not out_root.is_absolute():
        out_root = repo / out_root

    scene_id = scene_dir.name
    scene_fused_root = fused_root / scene_id / "fused"
    if not scene_fused_root.exists():
        raise SystemExit(f"Missing fused root: {scene_fused_root}")

    meta_path = scene_fused_root / "fusion_meta.json"
    dynamic_index_path = scene_fused_root / "dynamic_index.csv"
    background_path = scene_fused_root / "background_world.npy"
    if not meta_path.exists() or not dynamic_index_path.exists() or not background_path.exists():
        raise SystemExit(f"Missing fused outputs under: {scene_fused_root}")

    meta = _load_json(meta_path)
    background = np.asarray(np.load(str(background_path)), dtype=np.float32)
    dynamic_rows = list(csv.DictReader(dynamic_index_path.open("r", encoding="utf-8", newline="")))
    dynamic_points = {row["scene_stem"]: np.asarray(np.load(str(scene_fused_root / str(row["points_path"]))), dtype=np.float32) for row in dynamic_rows}
    dynamic_constraint_summary: dict[str, Any] | None = None
    depth_trim_status = "not_engaged"
    constraint_meta_ref = meta.get("dynamic_constraint", {}).get("dynamic_constraint_meta_json")
    constraint_meta_path = Path(str(constraint_meta_ref)) if constraint_meta_ref else (scene_fused_root / "dynamic_constraint_meta.json")
    if not constraint_meta_path.is_absolute():
        constraint_meta_path = scene_fused_root / constraint_meta_path
    if constraint_meta_path.exists():
        constraint_meta = _load_json(constraint_meta_path)
        trim_applied_frames = int(constraint_meta.get("trim_applied_frames", 0))
        trim_rejected_frames = int(constraint_meta.get("trim_rejected_frames", 0))
        trim_zero_support_frames = int(constraint_meta.get("trim_zero_support_frames", 0))
        per_camera_scale_stats = dict(constraint_meta.get("per_camera_scale_stats", {}))
        mean_fg_in_roi_after_align = constraint_meta.get("mean_fg_in_roi_after_align")
        dynamic_constraint_summary = {
            "schema_version": constraint_meta.get("schema_version"),
            "dynamic_constraint_index_csv": constraint_meta.get("dynamic_constraint_index_csv"),
            "num_timestamps": int(constraint_meta.get("num_timestamps", 0)),
            "multiview_supported_frames": int(constraint_meta.get("multiview_supported_frames", 0)),
            "degraded_single_view_fallback_frames": int(constraint_meta.get("degraded_single_view_fallback_frames", 0)),
            "fg_union_fallback_frames": int(constraint_meta.get("fg_union_fallback_frames", 0)),
            "nonempty_dynamic_frames": int(constraint_meta.get("nonempty_dynamic_frames", 0)),
            "empty_dynamic_frames": int(constraint_meta.get("empty_dynamic_frames", 0)),
            "constraint_mode_counts": dict(constraint_meta.get("constraint_mode_counts", {})),
            "roi_source_counts": dict(constraint_meta.get("roi_source_counts", {})),
            "mean_anchor_ray_error": constraint_meta.get("mean_anchor_ray_error"),
            "average_output_points": float(constraint_meta.get("average_output_points", 0.0)),
            "total_output_points": int(constraint_meta.get("total_output_points", 0)),
            "params": dict(constraint_meta.get("params", {})),
            "cams": list(constraint_meta.get("cams", [])),
            "trim_applied_frames": trim_applied_frames,
            "trim_rejected_frames": trim_rejected_frames,
            "trim_zero_support_frames": trim_zero_support_frames,
            "per_camera_scale_stats": per_camera_scale_stats,
            "mean_fg_in_roi_after_align": mean_fg_in_roi_after_align,
        }
        depth_trim_status = "engaged" if trim_applied_frames > 0 else "not_engaged"

    rig_path = scene_dir / "calib" / "rig.json"
    if not rig_path.exists():
        raise SystemExit(f"Missing rig json: {rig_path}")
    rig = _load_json(rig_path)
    rig_cameras = rig.get("cameras", {})
    cams = [c.strip() for c in str(args.cams).split(",") if c.strip()]
    if cams != ["cam0", "cam1", "cam2"]:
        raise SystemExit(f"This first version only supports cams=['cam0','cam1','cam2']. Got: {cams}")

    frame_times_csv = scene_dir / "frame_times.csv"
    if not frame_times_csv.exists():
        raise SystemExit(f"Missing frame_times.csv: {frame_times_csv}")
    sampled_scene_stems = _load_sampled_scene_stems(dynamic_index_path)
    frame_times_by_stem = _frame_times_by_stem(frame_times_csv, cams)
    sync_rows: list[tuple[str, dict[str, str]]] = []
    for scene_stem in sampled_scene_stems:
        by_cam = frame_times_by_stem.get(scene_stem)
        if by_cam is None or not all(cam in by_cam for cam in cams):
            raise SystemExit(f"Missing complete three-camera sync row for sampled scene_stem={scene_stem!r}")
        sync_rows.append((scene_stem, by_cam))

    mask_support = meta.get("background", {})
    bg_support_rate = float(mask_support.get("support_rate_ge_2", 0.0))
    bg_support_mean = float(mask_support.get("mean_cam_support", 0.0))
    bg_support_histogram = dict(mask_support.get("support_histogram", {}))
    selected_bg_support_rate = float(mask_support.get("selected_support_rate_ge_2", bg_support_rate))
    selected_bg_support_mean = float(mask_support.get("selected_mean_cam_support", bg_support_mean))
    selected_bg_support_histogram = dict(mask_support.get("selected_support_histogram", bg_support_histogram))
    raw_bg = int(meta.get("raw_counts", {}).get("bg", 0))
    raw_fg = int(meta.get("raw_counts", {}).get("fg", 0))
    fused_bg = int(meta.get("background", {}).get("fused_points", int(background.shape[0])))

    dynamic_ratios = [float(row.get("main_component_ratio", 0.0)) for row in dynamic_rows]
    dynamic_main_component_ratio_mean = float(np.mean(dynamic_ratios)) if dynamic_ratios else 0.0
    dynamic_main_component_ratio_min = float(np.min(dynamic_ratios)) if dynamic_ratios else 0.0

    centroid_series = []
    for row in dynamic_rows:
        points = np.asarray(np.load(str(scene_fused_root / str(row["points_path"]))), dtype=np.float32)
        if points.size:
            centroid_series.append(points.mean(axis=0))
    centroid_metrics = {"mean_step_l2": 0.0, "median_step_l2": 0.0, "max_step_l2": 0.0}
    if len(centroid_series) >= 2:
        centroids = np.stack(centroid_series, axis=0)
        steps = np.linalg.norm(np.diff(centroids, axis=0), axis=1)
        centroid_metrics = {
            "mean_step_l2": float(np.mean(steps)),
            "median_step_l2": float(np.median(steps)),
            "max_step_l2": float(np.max(steps)),
        }

    coverage_stats: dict[str, dict[str, float]] = {cam: {"mean_coverage": 0.0, "mean_hit_rate": 0.0} for cam in cams}
    scene_hit_rates: dict[str, dict[str, float]] = {cam: {"scene_mean_hit_rate": 0.0, "scene_reprojection_hit_rate": 0.0} for cam in cams}
    for cam_id in cams:
        cam_meta = rig_cameras[cam_id]
        K = np.asarray(cam_meta.get("K"), dtype=np.float32)
        c2w = np.asarray(cam_meta.get("T_node_from_cam"), dtype=np.float32)
        coverages: list[float] = []
        hit_rates: list[float] = []
        scene_coverages: list[float] = []
        scene_hit_rate_samples: list[float] = []
        for scene_stem, by_cam in sync_rows:
            frame_rel = by_cam[cam_id]
            scene_stem = Path(frame_rel).stem
            rgb_path = scene_dir / frame_rel
            mask_candidates = [scene_dir / "cams" / cam_id / "masks_gt" / f"{scene_stem}.png", scene_dir / "cams" / cam_id / "masks" / f"{scene_stem}.png"]
            mask_path = next((p for p in mask_candidates if p.exists()), None)
            if mask_path is None:
                continue
            mask = _read_gray(mask_path)
            dyn = dynamic_points.get(scene_stem)
            if dyn is None:
                continue
            rgb = _read_rgb(rgb_path)
            dyn_hit = _project_hit_mask(dyn, K, c2w, rgb.shape[1], rgb.shape[0])
            scene_hit = _project_hit_mask(np.concatenate([background, dyn], axis=0), K, c2w, rgb.shape[1], rgb.shape[0]) if background.size else dyn_hit
            mask_bool = mask > 0
            if mask_bool.any():
                coverages.append(float(np.logical_and(dyn_hit, mask_bool).sum()) / float(mask_bool.sum()))
                scene_coverages.append(float(np.logical_and(scene_hit, mask_bool).sum()) / float(mask_bool.sum()))
            hit_rates.append(float(dyn_hit.sum()) / float(dyn_hit.size))
            scene_hit_rate_samples.append(float(scene_hit.sum()) / float(scene_hit.size))
        if coverages:
            coverage_stats[cam_id]["mean_coverage"] = float(np.mean(coverages))
        if hit_rates:
            coverage_stats[cam_id]["mean_hit_rate"] = float(np.mean(hit_rates))
        if scene_coverages:
            scene_hit_rates[cam_id]["scene_mean_hit_rate"] = float(np.mean(scene_coverages))
        if scene_hit_rate_samples:
            scene_hit_rates[cam_id]["scene_reprojection_hit_rate"] = float(np.mean(scene_hit_rate_samples))

    fused_dynamic_total = int(sum(int(row.get("fused_points", 0)) for row in dynamic_rows))
    analysis = {
        "schema_version": "neoverse_fused_quality_v1",
        "scene_id": scene_id,
        "scene_dir": scene_dir.as_posix(),
        "fused_root": scene_fused_root.as_posix(),
        "background_support": {
            "mean_cam_support": bg_support_mean,
            "support_rate_ge_2": bg_support_rate,
            "raw_mean_cam_support": bg_support_mean,
            "raw_support_rate_ge_2": bg_support_rate,
            "raw_support_histogram": bg_support_histogram,
            "selected_mean_cam_support": selected_bg_support_mean,
            "selected_support_rate_ge_2": selected_bg_support_rate,
            "selected_support_histogram": selected_bg_support_histogram,
        },
        "dynamic_main_component": {
            "mean_ratio": dynamic_main_component_ratio_mean,
            "min_ratio": dynamic_main_component_ratio_min,
        },
        "reprojection_coverage": coverage_stats,
        "scene_reprojection": scene_hit_rates,
        "dynamic_centroid_continuity": centroid_metrics,
        "num_eval_frames": int(len(sampled_scene_stems)),
        "sampled_scene_stems": sampled_scene_stems,
        "point_count_comparison": {
            "raw_bg_points": raw_bg,
            "raw_fg_points": raw_fg,
            "fused_bg_points": fused_bg,
            "fused_dynamic_points": fused_dynamic_total,
            "bg_reduction_ratio": float(1.0 - (fused_bg / max(raw_bg, 1))),
            "fg_reduction_ratio": float(1.0 - (fused_dynamic_total / max(raw_fg, 1))),
        },
        "depth_trim_status": depth_trim_status,
        **({"dynamic_constraint_summary": dynamic_constraint_summary} if dynamic_constraint_summary is not None else {}),
        "dynamic_timestamps": dynamic_rows,
    }

    analysis_root = out_root / scene_id / "analysis"
    analysis_root.mkdir(parents=True, exist_ok=True)
    report_path = analysis_root / "quality_report.json"
    report_path.write_text(json.dumps(analysis, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote quality report to: {report_path}")


if __name__ == "__main__":
    main()
