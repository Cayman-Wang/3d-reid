from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover
    plt = None

from recon_spin_points import _pose_from_capture_meta, _transform_points


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:  # pragma: no cover
        raise SystemExit(f"Failed to read json: {path}\nError: {e!r}")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _load_recon_branch(scene_dir: Path, subdir: str) -> dict[str, Any]:
    branch_dir = scene_dir / str(subdir)
    meta_path = branch_dir / "meta.json"
    input_index_path = branch_dir / "input_points_index.csv"
    points_index_path = branch_dir / "points_index.csv"
    if not meta_path.exists():
        raise SystemExit(f"Missing meta.json: {meta_path}")
    if not input_index_path.exists():
        raise SystemExit(f"Missing input_points_index.csv: {input_index_path}")
    if not points_index_path.exists():
        raise SystemExit(f"Missing points_index.csv: {points_index_path}")
    meta = _load_json(meta_path)
    input_rows = _read_csv_rows(input_index_path)
    output_rows = _read_csv_rows(points_index_path)
    if not output_rows:
        raise SystemExit(f"No output rows found in: {points_index_path}")

    ts_stems = [str(row["ts_stem"]).strip() for row in output_rows]
    sample_ts = ts_stems[len(ts_stems) // 2]
    points_path = branch_dir / f"{sample_ts}.npy"
    if not points_path.exists():
        raise SystemExit(f"Missing sampled points file: {points_path}")

    return {
        "subdir": str(subdir),
        "dir": branch_dir,
        "meta": meta,
        "input_rows": input_rows,
        "output_rows": output_rows,
        "sample_ts": sample_ts,
        "sample_points": np.asarray(np.load(str(points_path)), dtype=np.float32),
    }


def _recover_canonical(points_node: np.ndarray, capture_meta: dict[str, Any], ts_stem: str) -> np.ndarray:
    ts_us = int(str(ts_stem))
    t_sec = float(ts_us) / 1e6
    _, T_target_from_node = _pose_from_capture_meta(capture_meta, t_sec)
    return _transform_points(T_target_from_node, np.asarray(points_node, dtype=np.float32))


def _points_stats(points_xyz: np.ndarray) -> dict[str, Any]:
    pts = np.asarray(points_xyz, dtype=np.float32)
    if pts.size == 0:
        return {
            "count": 0,
            "bbox_min": [0.0, 0.0, 0.0],
            "bbox_max": [0.0, 0.0, 0.0],
            "extent": [0.0, 0.0, 0.0],
            "center": [0.0, 0.0, 0.0],
            "volume": 0.0,
        }
    bbox_min = pts.min(axis=0)
    bbox_max = pts.max(axis=0)
    extent = bbox_max - bbox_min
    center = pts.mean(axis=0)
    return {
        "count": int(pts.shape[0]),
        "bbox_min": bbox_min.astype(float).tolist(),
        "bbox_max": bbox_max.astype(float).tolist(),
        "extent": extent.astype(float).tolist(),
        "center": center.astype(float).tolist(),
        "volume": float(np.prod(np.maximum(extent, 1e-6))),
    }


def _sample_points(points_xyz: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    pts = np.asarray(points_xyz, dtype=np.float32)
    if pts.shape[0] <= int(max_points):
        return pts
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(pts.shape[0], size=int(max_points), replace=False)
    return pts[idx]


def _draw_scatter_panel(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    *,
    gt_points: np.ndarray,
    pred_points: np.ndarray,
    title: str,
) -> None:
    x0, y0, x1, y1 = rect
    draw.rectangle(rect, outline=(160, 160, 160), width=1)
    draw.text((x0 + 6, y0 + 4), title, fill=(0, 0, 0))
    plot_x0, plot_y0 = x0 + 20, y0 + 26
    plot_x1, plot_y1 = x1 - 10, y1 - 16

    def _norm(points: np.ndarray) -> list[tuple[int, int]]:
        pts = np.asarray(points, dtype=np.float32)
        if pts.size == 0:
            return []
        mins = pts.min(axis=0)
        maxs = pts.max(axis=0)
        span = np.maximum(maxs - mins, 1e-6)
        out: list[tuple[int, int]] = []
        for x, y in pts:
            px = plot_x0 + int(round(((x - mins[0]) / span[0]) * max(plot_x1 - plot_x0, 1)))
            py = plot_y1 - int(round(((y - mins[1]) / span[1]) * max(plot_y1 - plot_y0, 1)))
            out.append((px, py))
        return out

    all_points = []
    if gt_points.size:
        all_points.append(np.asarray(gt_points, dtype=np.float32))
    if pred_points.size:
        all_points.append(np.asarray(pred_points, dtype=np.float32))
    if not all_points:
        return

    pts = np.concatenate(all_points, axis=0)
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    span = np.maximum(maxs - mins, 1e-6)

    def _norm_shared(points: np.ndarray) -> list[tuple[int, int]]:
        out: list[tuple[int, int]] = []
        for x, y in np.asarray(points, dtype=np.float32):
            px = plot_x0 + int(round(((x - mins[0]) / span[0]) * max(plot_x1 - plot_x0, 1)))
            py = plot_y1 - int(round(((y - mins[1]) / span[1]) * max(plot_y1 - plot_y0, 1)))
            out.append((px, py))
        return out

    for px, py in _norm_shared(gt_points):
        draw.point((px, py), fill=(31, 119, 180))
    for px, py in _norm_shared(pred_points):
        draw.point((px, py), fill=(214, 39, 40))


def _draw_line_panel(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    *,
    gt_values: np.ndarray,
    pred_values: np.ndarray,
    title: str,
) -> None:
    x0, y0, x1, y1 = rect
    draw.rectangle(rect, outline=(160, 160, 160), width=1)
    draw.text((x0 + 6, y0 + 4), title, fill=(0, 0, 0))
    plot_x0, plot_y0 = x0 + 20, y0 + 26
    plot_x1, plot_y1 = x1 - 10, y1 - 16

    max_len = max(int(gt_values.size), int(pred_values.size), 1)
    max_val = max(float(gt_values.max()) if gt_values.size else 0.0, float(pred_values.max()) if pred_values.size else 0.0, 1.0)

    def _poly(values: np.ndarray) -> list[tuple[int, int]]:
        vals = np.asarray(values, dtype=np.float32)
        if vals.size == 0:
            return []
        out: list[tuple[int, int]] = []
        denom_x = max(max_len - 1, 1)
        for idx, value in enumerate(vals):
            px = plot_x0 + int(round((idx / denom_x) * max(plot_x1 - plot_x0, 1)))
            py = plot_y1 - int(round((float(value) / max_val) * max(plot_y1 - plot_y0, 1)))
            out.append((px, py))
        return out

    gt_poly = _poly(gt_values)
    pred_poly = _poly(pred_values)
    if len(gt_poly) >= 2:
        draw.line(gt_poly, fill=(31, 119, 180), width=2)
    if len(pred_poly) >= 2:
        draw.line(pred_poly, fill=(214, 39, 40), width=2)


def _count_series(rows: list[dict[str, str]], field: str) -> np.ndarray:
    return np.asarray([int(float(row.get(field, "0") or 0)) for row in rows], dtype=np.int32)


def _summarize_branch(branch: dict[str, Any], canonical_points: np.ndarray) -> dict[str, Any]:
    input_counts = _count_series(branch["input_rows"], "n_input_points")
    output_counts = _count_series(branch["output_rows"], "n_points")
    stats = {
        "input_mean": float(input_counts.mean()) if input_counts.size else 0.0,
        "input_min": int(input_counts.min()) if input_counts.size else 0,
        "input_max": int(input_counts.max()) if input_counts.size else 0,
        "output_mean": float(output_counts.mean()) if output_counts.size else 0.0,
        "output_min": int(output_counts.min()) if output_counts.size else 0,
        "output_max": int(output_counts.max()) if output_counts.size else 0,
    }
    stats.update(_points_stats(canonical_points))
    return stats


def _diagnose_scene(
    gt_branch: dict[str, Any], pred_branch: dict[str, Any], gt_stats: dict[str, Any], pred_stats: dict[str, Any]
) -> tuple[list[str], dict[str, Any], dict[str, bool]]:
    notes: list[str] = []

    gt_input_mean = max(float(gt_stats["input_mean"]), 1e-6)
    input_ratio = float(pred_stats["input_mean"]) / gt_input_mean

    gt_extent = np.asarray(gt_stats["extent"], dtype=np.float32)
    pred_extent = np.asarray(pred_stats["extent"], dtype=np.float32)
    gt_center = np.asarray(gt_stats["center"], dtype=np.float32)
    pred_center = np.asarray(pred_stats["center"], dtype=np.float32)
    gt_bbox_min = np.asarray(gt_stats["bbox_min"], dtype=np.float32)
    pred_bbox_min = np.asarray(pred_stats["bbox_min"], dtype=np.float32)

    gt_volume = max(float(gt_stats["volume"]), 1e-6)
    volume_ratio = float(pred_stats["volume"]) / gt_volume
    extent_ratio = pred_extent / np.maximum(gt_extent, 1e-6)
    mean_extent_ratio = float(np.mean(extent_ratio))

    center_delta = pred_center - gt_center
    center_shift_l2 = float(np.linalg.norm(center_delta))
    gt_diag = float(np.linalg.norm(gt_extent))
    center_shift_thresh = max(0.35, 0.2 * gt_diag)
    center_y_shift_thresh = max(0.25, 0.2 * max(float(gt_extent[1]), 1e-6))

    bottom_tail_delta = float(gt_bbox_min[1] - pred_bbox_min[1])  # >0 means predicted bottom is lower
    bottom_tail_thresh = max(0.30, 0.18 * max(float(gt_extent[1]), 1e-6))

    pred_meta = dict(pred_branch["meta"])
    max_cap = int(pred_meta.get("max_canonical_points", 0))
    pred_canonical_points = int(pred_meta.get("canonical_point_count", pred_stats["count"]))
    support_after = int(pred_meta.get("canonical_voxel_count_after_support", pred_canonical_points))
    support_before = int(pred_meta.get("canonical_voxel_count_before_support", support_after))
    support_keep_ratio = float(support_after) / float(max(support_before, 1))
    centroid_gate_meta = dict(pred_meta.get("static_frame_centroid_gate_meta") or {})
    core_clip_meta = dict(pred_meta.get("static_core_clip_meta") or {})

    def _as_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except Exception:
            return float(default)

    def _as_int(value: Any, default: int = 0) -> int:
        try:
            return int(value)
        except Exception:
            return int(default)

    def _pick_float(payload: dict[str, Any], keys: list[str], default: float = 0.0) -> float:
        for key in keys:
            if key in payload and payload.get(key) is not None:
                return _as_float(payload.get(key), default)
        return float(default)

    def _ratio_from_counts(payload: dict[str, Any], num_keys: list[str], den_keys: list[str], default: float = 0.0) -> float:
        numerator = 0
        denominator = 0
        for key in num_keys:
            if key in payload and payload.get(key) is not None:
                numerator = _as_int(payload.get(key), 0)
                break
        for key in den_keys:
            if key in payload and payload.get(key) is not None:
                denominator = _as_int(payload.get(key), 0)
                break
        if denominator > 0:
            return float(numerator) / float(denominator)
        return float(default)

    def _pick_array3(payloads: list[dict[str, Any]], keys: list[str], fallback: list[float] | None = None) -> np.ndarray:
        for payload in payloads:
            for key in keys:
                value = payload.get(key)
                if isinstance(value, (list, tuple)) and len(value) >= 3:
                    try:
                        return np.asarray([float(value[0]), float(value[1]), float(value[2])], dtype=np.float32)
                    except Exception:
                        continue
        if fallback is not None and len(fallback) >= 3:
            return np.asarray([float(fallback[0]), float(fallback[1]), float(fallback[2])], dtype=np.float32)
        return np.zeros(3, dtype=np.float32)

    frame_centroid_keep_ratio = _pick_float(
        centroid_gate_meta,
        ["frame_centroid_keep_ratio", "keep_ratio", "kept_ratio", "nonempty_keep_ratio"],
        default=-1.0,
    )
    if frame_centroid_keep_ratio < 0.0:
        frame_centroid_keep_ratio = _ratio_from_counts(
            centroid_gate_meta,
            ["kept_nonempty_frames", "kept_frames", "num_kept_frames"],
            ["total_nonempty_frames", "nonempty_frames", "total_frames", "num_frames"],
            default=0.0,
        )
    frame_centroid_spread_l2 = _pick_float(
        centroid_gate_meta,
        ["frame_centroid_spread_l2", "centroid_spread_l2", "spread_l2", "median_centroid_l2"],
        default=0.0,
    )

    core_point_count = _as_int(
        core_clip_meta.get("core_point_count", core_clip_meta.get("core_points", core_clip_meta.get("core_count", 0))),
        0,
    )

    raw_extent_xyz = _pick_array3(
        [pred_meta, core_clip_meta],
        ["raw_extent_xyz", "canonical_raw_extent_xyz", "extent_raw_xyz"],
        fallback=list(pred_stats.get("extent") or [0.0, 0.0, 0.0]),
    )
    core_extent_xyz = _pick_array3(
        [pred_meta, core_clip_meta],
        ["core_extent_xyz", "canonical_core_extent_xyz", "extent_core_xyz"],
        fallback=[0.0, 0.0, 0.0],
    )
    shell_extent_xyz = _pick_array3(
        [pred_meta, core_clip_meta],
        ["shell_extent_xyz", "canonical_shell_extent_xyz", "extent_shell_xyz"],
        fallback=[0.0, 0.0, 0.0],
    )
    shell_added_points = _as_int(
        pred_meta.get(
            "shell_added_points",
            core_clip_meta.get("shell_added_points", core_clip_meta.get("shell_points", core_clip_meta.get("added_shell_points", 0))),
        ),
        0,
    )
    core_shell_ratio = _pick_float(pred_meta, ["core_shell_ratio"], default=-1.0)
    if core_shell_ratio < 0.0:
        core_shell_ratio = _pick_float(core_clip_meta, ["core_shell_ratio", "core_to_shell_ratio"], default=-1.0)
    if core_shell_ratio < 0.0:
        if shell_added_points > 0:
            core_shell_ratio = float(core_point_count) / float(max(shell_added_points, 1))
        elif core_point_count > 0:
            core_shell_ratio = float(core_point_count)
        else:
            core_shell_ratio = 0.0
    y_downshift_vs_raw = _pick_float(
        pred_meta,
        ["y_downshift_vs_raw", "canonical_y_downshift_vs_raw", "downshift_vs_raw_y"],
        default=-999.0,
    )
    if y_downshift_vs_raw <= -998.0:
        y_downshift_vs_raw = _pick_float(core_clip_meta, ["y_downshift_vs_raw", "downshift_vs_raw_y"], default=-999.0)
    if y_downshift_vs_raw <= -998.0:
        raw_center_y = _pick_float(pred_meta, ["raw_center_y", "canonical_raw_center_y"], default=0.0)
        if abs(raw_center_y) > 1e-6:
            y_downshift_vs_raw = float(raw_center_y - pred_center[1])
        else:
            y_downshift_vs_raw = 0.0
    clip_keep_ratio = _pick_float(core_clip_meta, ["clip_keep_ratio", "keep_ratio", "clip_ratio"], default=-1.0)
    if clip_keep_ratio < 0.0:
        clip_keep_ratio = _ratio_from_counts(
            core_clip_meta,
            ["clipped_point_count", "kept_point_count", "post_clip_count", "point_count_after_clip"],
            ["pre_clip_count", "point_count_before_clip", "input_point_count"],
            default=-1.0,
        )
    if clip_keep_ratio < 0.0 and core_point_count > 0:
        clipped_points = _as_int(
            core_clip_meta.get("clipped_point_count", core_clip_meta.get("kept_point_count", core_clip_meta.get("post_clip_count", 0))),
            0,
        )
        if clipped_points > 0:
            clip_keep_ratio = float(clipped_points) / float(max(core_point_count, 1))
    if clip_keep_ratio < 0.0:
        clip_keep_ratio = 0.0

    flags = {
        "input_leakage": bool(input_ratio >= 1.8),
        "input_insufficient": bool(input_ratio <= 0.7),
        "center_shift": bool(center_shift_l2 >= center_shift_thresh),
        "bottom_tail": bool(bottom_tail_delta >= bottom_tail_thresh),
        "volume_expansion": bool(volume_ratio >= 1.35 and mean_extent_ratio >= 1.1),
        "main_structure_shrink": bool(volume_ratio <= 0.72 or int(np.sum(extent_ratio <= 0.85)) >= 2),
        "low_support_keep_ratio": bool(support_before > 0 and support_keep_ratio <= 0.55),
        "cap_saturation": bool(max_cap > 0 and pred_canonical_points >= int(round(0.98 * max_cap))),
        "low_frame_centroid_keep_ratio": bool(frame_centroid_keep_ratio > 0.0 and frame_centroid_keep_ratio <= 0.70),
        "high_frame_centroid_spread": bool(frame_centroid_spread_l2 >= 0.60),
        "low_clip_keep_ratio": bool(clip_keep_ratio > 0.0 and clip_keep_ratio <= 0.60),
        "shell_sparse": bool(shell_added_points <= 0),
    }

    raw_shell_margin = np.maximum(raw_extent_xyz - np.maximum(core_extent_xyz, 1e-6), 0.0)
    raw_shell_margin_max = float(np.max(raw_shell_margin))
    raw_shell_margin_ratio = float(np.mean(raw_extent_xyz) / max(float(np.mean(np.maximum(core_extent_xyz, 1e-6))), 1e-6))
    shell_extent_effective = float(np.mean(shell_extent_xyz)) / max(float(np.mean(np.maximum(core_extent_xyz, 1e-6))), 1e-6)
    raw_has_outer_shell = bool(raw_shell_margin_max >= 0.20 or raw_shell_margin_ratio >= 1.20)
    shell_recovered = bool(shell_added_points > 0 and (shell_extent_effective >= 1.08 or core_shell_ratio <= 6.0))
    raw_stage_polluted = bool(
        centroid_gate_meta.get("reason") == "fallback_keep_original_no_frames"
        or raw_shell_margin_ratio >= 8.0
        or raw_shell_margin_max >= 4.0
    )
    if raw_stage_polluted:
        shell_loss_stage = "raw canonical polluted before support/core-shell"
    elif raw_has_outer_shell and not shell_recovered:
        shell_loss_stage = "support/core-shell filtering removed edge structure"
    else:
        shell_loss_stage = "input stage lacks edge structure"
    flags["edge_missing_in_input"] = bool(shell_loss_stage == "input stage lacks edge structure")
    flags["edge_removed_by_filter"] = bool(shell_loss_stage == "support/core-shell filtering removed edge structure")
    flags["raw_stage_polluted"] = bool(shell_loss_stage == "raw canonical polluted before support/core-shell")

    metrics = {
        "input_ratio": float(input_ratio),
        "volume_ratio": float(volume_ratio),
        "mean_extent_ratio": float(mean_extent_ratio),
        "center_shift_l2": float(center_shift_l2),
        "center_shift_y": float(center_delta[1]),
        "bottom_tail_delta_ymin": float(bottom_tail_delta),
        "support_keep_ratio": float(support_keep_ratio),
        "cap_fill_ratio": float(pred_canonical_points / max(max_cap, 1)) if max_cap > 0 else 0.0,
        "frame_centroid_keep_ratio": float(frame_centroid_keep_ratio),
        "frame_centroid_spread_l2": float(frame_centroid_spread_l2),
        "core_point_count": float(core_point_count),
        "clip_keep_ratio": float(clip_keep_ratio),
        "raw_extent_xyz": raw_extent_xyz.astype(float).tolist(),
        "core_extent_xyz": core_extent_xyz.astype(float).tolist(),
        "shell_extent_xyz": shell_extent_xyz.astype(float).tolist(),
        "core_shell_ratio": float(core_shell_ratio),
        "shell_added_points": float(shell_added_points),
        "y_downshift_vs_raw": float(y_downshift_vs_raw),
        "shell_loss_stage": str(shell_loss_stage),
    }

    if flags["input_leakage"]:
        notes.append(f"[输入泄露] 预测输入点数约为 GT 的 {input_ratio:.2f} 倍，优先怀疑 mask/depth 带入背景区域。")
    if flags["input_insufficient"]:
        notes.append(f"[输入不足] 预测输入点数仅为 GT 的 {input_ratio:.2f} 倍，目标覆盖可能不足。")
    if flags["center_shift"]:
        notes.append(
            f"[中心偏移] canonical 中心偏移 L2={center_shift_l2:.3f}m (dy={center_delta[1]:+.3f}m)，跨时刻聚合存在位置漂移。"
        )
    if flags["bottom_tail"]:
        notes.append(
            f"[底部拖尾] 预测 y_min 比 GT 更低 {bottom_tail_delta:.3f}m，优先判定为底部拖尾/背景下沉，而非体积膨胀。"
        )
    if flags["volume_expansion"]:
        notes.append(
            f"[体积膨胀] canonical 体积比 GT 放大到 {volume_ratio:.2f} 倍（平均尺度比 {mean_extent_ratio:.2f}），存在结构外扩。"
        )
    if flags["main_structure_shrink"]:
        notes.append(
            f"[主体收缩] canonical 体积仅为 GT 的 {volume_ratio:.2f} 倍（平均尺度比 {mean_extent_ratio:.2f}），主体结构有塌缩趋势。"
        )
    if flags["low_support_keep_ratio"]:
        notes.append(f"[support 保留率低] 支撑过滤后仅保留 {support_keep_ratio:.2%} 体素，低支撑噪声占比偏高。")
    if flags["cap_saturation"]:
        notes.append(
            f"[cap 饱和] canonical 点数达到上限 {pred_canonical_points}/{max_cap}，输出更可能由噪声上限驱动。"
        )
    if flags["low_frame_centroid_keep_ratio"]:
        notes.append(f"[centroid gate 保留率低] frame_centroid_keep_ratio={frame_centroid_keep_ratio:.2%}，跨帧漂移帧占比偏高。")
    if flags["high_frame_centroid_spread"]:
        notes.append(f"[centroid 离散偏大] frame_centroid_spread_l2={frame_centroid_spread_l2:.3f}m，时序中心稳定性偏弱。")
    if flags["low_clip_keep_ratio"]:
        notes.append(f"[clip 保留率低] clip_keep_ratio={clip_keep_ratio:.2%}，主结构裁剪可能偏激进。")
    if flags["raw_stage_polluted"]:
        notes.append(
            f"[raw canonical 污染] raw_extent/core_extent 均值比约 {raw_shell_margin_ratio:.2f}，frame gate 后仍残留大范围漂移/背景点。"
        )
    notes.append(
        "[外沿归因] "
        + (
            "frame gate 之后的 raw canonical 已被大范围漂移/背景点污染，当前 core/shell 只能在脏 raw 上保结构。"
            if flags["raw_stage_polluted"]
            else (
                "输入阶段就没有外沿，当前 raw->core 结构边缘增量不足。"
                if flags["edge_missing_in_input"]
                else "support/core-shell 过滤删掉了外沿，raw 中可见外沿但最终未保住。"
            )
        )
    )

    active_cams = list(pred_meta.get("active_cams") or [])
    scale_by_cam = dict(pred_meta.get("depth_scale_by_cam") or {})
    frame_norm_meta = dict(pred_meta.get("static_frame_depth_norm_meta") or {})
    frame_norm_by_cam = dict(frame_norm_meta.get("per_cam") or {})
    if frame_norm_by_cam:
        norm_tokens = []
        for cam_id in active_cams or sorted(frame_norm_by_cam):
            payload = dict(frame_norm_by_cam.get(cam_id) or {})
            if not payload:
                continue
            norm_tokens.append(
                f"{cam_id}: ref={float(payload.get('reference_depth_stat', 0.0)):.3f}, "
                f"frame_scale={float(payload.get('frame_scale_min', 1.0)):.2f}-{float(payload.get('frame_scale_max', 1.0)):.2f}"
            )
        if norm_tokens:
            notes.append("predicted static 输入已启用每帧 relative depth 归一化：" + "; ".join(norm_tokens) + "。")
    if active_cams or scale_by_cam:
        scale_tokens = [f"{cam}={float(scale_by_cam[cam]):.2f}" for cam in active_cams if cam in scale_by_cam]
        if scale_tokens:
            notes.append("predicted static 输入已启用每相机 depth scale probe：" + ", ".join(scale_tokens) + "。")
        all_cams = list(pred_meta.get("cams") or [])
        if active_cams and all_cams and len(active_cams) < len(all_cams):
            notes.append("部分相机在 static probe 中被判定为不稳定并被跳过，以减少 canonical 聚合漂移。")

    if not notes:
        notes.append("当前 scene 没有出现单一显著失真，优先继续对比 GT / predicted 的局部结构与检索结果。")
    return notes, metrics, flags


def _write_summary_md(
    out_path: Path,
    *,
    scene_name: str,
    gt_branch: dict[str, Any],
    pred_branch: dict[str, Any],
    gt_stats: dict[str, Any],
    pred_stats: dict[str, Any],
    notes: list[str],
    diag_metrics: dict[str, Any],
    diag_flags: dict[str, bool],
) -> None:
    pred_meta = dict(pred_branch.get("meta") or {})
    active_cams = list(pred_meta.get("active_cams") or [])
    scale_by_cam = dict(pred_meta.get("depth_scale_by_cam") or {})
    frame_norm_meta = dict(pred_meta.get("static_frame_depth_norm_meta") or {})
    frame_norm_by_cam = dict(frame_norm_meta.get("per_cam") or {})

    def _fmt_vec3(value: Any) -> str:
        if isinstance(value, (list, tuple)) and len(value) >= 3:
            try:
                arr = np.asarray([float(value[0]), float(value[1]), float(value[2])], dtype=np.float32)
                return str(np.round(arr, 3).tolist())
            except Exception:
                return str(value)
        return "-"
    lines = [
        f"# {scene_name} spin recon quality summary",
        "",
        "## Branches",
        "",
        f"- GT: `{gt_branch['subdir']}`",
        f"- Predicted: `{pred_branch['subdir']}`",
        f"- Sample timestamp: `{pred_branch['sample_ts']}`",
        "",
    ]
    if active_cams or scale_by_cam:
        lines.extend(
            [
                "## Predicted Static Cleanup",
                "",
                f"- active_cams: `{','.join(active_cams) if active_cams else '-'}`",
                f"- depth_scale_by_cam: `{', '.join(f'{cam}={float(scale_by_cam[cam]):.2f}' for cam in active_cams if cam in scale_by_cam) if scale_by_cam else '-'}`",
                "",
            ]
        )
    if frame_norm_by_cam:
        lines.extend(
            [
                "## Frame Depth Normalization",
                "",
                *[
                    (
                        f"- {cam_id}: "
                        f"`ref={float(payload.get('reference_depth_stat', 0.0)):.3f}, "
                        f"frame_scale={float(payload.get('frame_scale_min', 1.0)):.2f}-"
                        f"{float(payload.get('frame_scale_max', 1.0)):.2f}`"
                    )
                    for cam_id, payload in frame_norm_by_cam.items()
                ],
                "",
            ]
        )
    lines.extend(
        [
            "## Count Stats",
            "",
            "| metric | gt | predicted |",
            "| --- | ---: | ---: |",
            f"| input_mean | {gt_stats['input_mean']:.1f} | {pred_stats['input_mean']:.1f} |",
            f"| input_min | {gt_stats['input_min']} | {pred_stats['input_min']} |",
            f"| input_max | {gt_stats['input_max']} | {pred_stats['input_max']} |",
            f"| output_mean | {gt_stats['output_mean']:.1f} | {pred_stats['output_mean']:.1f} |",
            f"| output_min | {gt_stats['output_min']} | {pred_stats['output_min']} |",
            f"| output_max | {gt_stats['output_max']} | {pred_stats['output_max']} |",
            f"| canonical_count | {gt_stats['count']} | {pred_stats['count']} |",
            "",
            "## Canonical BBox",
            "",
            "| metric | gt | predicted |",
            "| --- | --- | --- |",
            f"| bbox_min | {np.round(np.asarray(gt_stats['bbox_min']), 3).tolist()} | {np.round(np.asarray(pred_stats['bbox_min']), 3).tolist()} |",
            f"| bbox_max | {np.round(np.asarray(gt_stats['bbox_max']), 3).tolist()} | {np.round(np.asarray(pred_stats['bbox_max']), 3).tolist()} |",
            f"| extent | {np.round(np.asarray(gt_stats['extent']), 3).tolist()} | {np.round(np.asarray(pred_stats['extent']), 3).tolist()} |",
            f"| center | {np.round(np.asarray(gt_stats['center']), 3).tolist()} | {np.round(np.asarray(pred_stats['center']), 3).tolist()} |",
            f"| volume | {gt_stats['volume']:.3f} | {pred_stats['volume']:.3f} |",
            "",
            "## Diagnosis Indicators",
            "",
            "| indicator | value | flag |",
            "| --- | ---: | :---: |",
            f"| input_ratio(pred/gt) | {diag_metrics['input_ratio']:.3f} | {'Y' if diag_flags['input_leakage'] or diag_flags['input_insufficient'] else '-'} |",
            f"| volume_ratio(pred/gt) | {diag_metrics['volume_ratio']:.3f} | {'Y' if diag_flags['volume_expansion'] or diag_flags['main_structure_shrink'] else '-'} |",
            f"| mean_extent_ratio(pred/gt) | {diag_metrics['mean_extent_ratio']:.3f} | {'Y' if diag_flags['volume_expansion'] or diag_flags['main_structure_shrink'] else '-'} |",
            f"| center_shift_l2(m) | {diag_metrics['center_shift_l2']:.3f} | {'Y' if diag_flags['center_shift'] else '-'} |",
            f"| center_shift_y(m) | {diag_metrics['center_shift_y']:+.3f} | {'Y' if diag_flags['center_shift'] else '-'} |",
            f"| bottom_tail_delta_ymin(m) | {diag_metrics['bottom_tail_delta_ymin']:.3f} | {'Y' if diag_flags['bottom_tail'] else '-'} |",
            f"| support_keep_ratio | {diag_metrics['support_keep_ratio']:.3%} | {'Y' if diag_flags['low_support_keep_ratio'] else '-'} |",
            f"| cap_fill_ratio | {diag_metrics['cap_fill_ratio']:.3%} | {'Y' if diag_flags['cap_saturation'] else '-'} |",
            f"| frame_centroid_keep_ratio | {diag_metrics['frame_centroid_keep_ratio']:.3%} | {'Y' if diag_flags['low_frame_centroid_keep_ratio'] else '-'} |",
            f"| frame_centroid_spread_l2(m) | {diag_metrics['frame_centroid_spread_l2']:.3f} | {'Y' if diag_flags['high_frame_centroid_spread'] else '-'} |",
            f"| core_point_count | {diag_metrics['core_point_count']:.0f} | - |",
            f"| clip_keep_ratio | {diag_metrics['clip_keep_ratio']:.3%} | {'Y' if diag_flags['low_clip_keep_ratio'] else '-'} |",
            f"| raw_extent_xyz | {_fmt_vec3(diag_metrics.get('raw_extent_xyz'))} | - |",
            f"| core_extent_xyz | {_fmt_vec3(diag_metrics.get('core_extent_xyz'))} | - |",
            f"| shell_extent_xyz | {_fmt_vec3(diag_metrics.get('shell_extent_xyz'))} | {'Y' if diag_flags.get('shell_sparse') else '-'} |",
            f"| core_shell_ratio | {float(diag_metrics.get('core_shell_ratio', 0.0)):.3f} | {'Y' if float(diag_metrics.get('core_shell_ratio', 0.0)) >= 4.0 else '-'} |",
            f"| shell_added_points | {float(diag_metrics.get('shell_added_points', 0.0)):.0f} | {'Y' if diag_flags.get('shell_sparse') else '-'} |",
            f"| y_downshift_vs_raw(m) | {float(diag_metrics.get('y_downshift_vs_raw', 0.0)):+.3f} | {'Y' if float(diag_metrics.get('y_downshift_vs_raw', 0.0)) >= 0.12 else '-'} |",
            f"| shell_loss_stage | {str(diag_metrics.get('shell_loss_stage', '-'))} | {'Y' if diag_flags.get('edge_missing_in_input') or diag_flags.get('edge_removed_by_filter') else '-'} |",
            "",
            "## Failure Attribution",
            "",
        ]
    )
    lines.extend([f"- {note}" for note in notes])
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_canonical_compare(
    out_path: Path,
    *,
    scene_name: str,
    gt_points: np.ndarray,
    pred_points: np.ndarray,
    seed: int,
    max_plot_points: int,
) -> None:
    gt_plot = _sample_points(gt_points, int(max_plot_points), int(seed))
    pred_plot = _sample_points(pred_points, int(max_plot_points), int(seed) + 1)

    if plt is not None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
        axes[0].scatter(gt_plot[:, 0], gt_plot[:, 1], s=1.5, alpha=0.35, label="gt", c="#1f77b4")
        axes[0].scatter(pred_plot[:, 0], pred_plot[:, 1], s=1.5, alpha=0.25, label="pred", c="#d62728")
        axes[0].set_title("Canonical XY")
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("y")
        axes[0].legend(loc="best")
        axes[0].grid(True, alpha=0.2)

        axes[1].scatter(gt_plot[:, 0], gt_plot[:, 2], s=1.5, alpha=0.35, label="gt", c="#1f77b4")
        axes[1].scatter(pred_plot[:, 0], pred_plot[:, 2], s=1.5, alpha=0.25, label="pred", c="#d62728")
        axes[1].set_title("Canonical XZ")
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("z")
        axes[1].legend(loc="best")
        axes[1].grid(True, alpha=0.2)

        fig.suptitle(f"{scene_name} canonical GT vs predicted")
        fig.savefig(str(out_path), dpi=160)
        plt.close(fig)
        return

    img = Image.new("RGB", (1200, 520), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((24, 12), f"{scene_name} canonical GT vs predicted", fill=(0, 0, 0))
    _draw_scatter_panel(draw, (24, 40, 588, 500), gt_points=gt_plot[:, [0, 1]], pred_points=pred_plot[:, [0, 1]], title="Canonical XY")
    _draw_scatter_panel(draw, (612, 40, 1176, 500), gt_points=gt_plot[:, [0, 2]], pred_points=pred_plot[:, [0, 2]], title="Canonical XZ")
    img.save(str(out_path))


def _plot_count_compare(
    out_path: Path,
    *,
    scene_name: str,
    gt_branch: dict[str, Any],
    pred_branch: dict[str, Any],
) -> None:
    gt_input = _count_series(gt_branch["input_rows"], "n_input_points")
    pred_input = _count_series(pred_branch["input_rows"], "n_input_points")
    gt_output = _count_series(gt_branch["output_rows"], "n_points")
    pred_output = _count_series(pred_branch["output_rows"], "n_points")
    x = np.arange(max(gt_input.size, pred_input.size))

    if plt is not None:
        fig, axes = plt.subplots(2, 1, figsize=(11, 7), constrained_layout=True)
        axes[0].plot(x[: gt_input.size], gt_input, label="gt input", color="#1f77b4")
        axes[0].plot(x[: pred_input.size], pred_input, label="pred input", color="#d62728")
        axes[0].set_title("Input points per timestamp")
        axes[0].set_xlabel("timestamp index")
        axes[0].set_ylabel("n_input_points")
        axes[0].legend(loc="best")
        axes[0].grid(True, alpha=0.2)

        axes[1].plot(np.arange(gt_output.size), gt_output, label="gt output", color="#1f77b4")
        axes[1].plot(np.arange(pred_output.size), pred_output, label="pred output", color="#d62728")
        axes[1].set_title("Output points per timestamp")
        axes[1].set_xlabel("timestamp index")
        axes[1].set_ylabel("n_points")
        axes[1].legend(loc="best")
        axes[1].grid(True, alpha=0.2)

        fig.suptitle(f"{scene_name} recon count comparison")
        fig.savefig(str(out_path), dpi=160)
        plt.close(fig)
        return

    img = Image.new("RGB", (1160, 760), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((24, 12), f"{scene_name} recon count comparison", fill=(0, 0, 0))
    _draw_line_panel(draw, (24, 40, 1136, 370), gt_values=gt_input, pred_values=pred_input, title="Input points per timestamp")
    _draw_line_panel(draw, (24, 400, 1136, 730), gt_values=gt_output, pred_values=pred_output, title="Output points per timestamp")
    img.save(str(out_path))


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze one spin-recon scene by comparing GT and predicted reconstructed geometry.")
    ap.add_argument("--scene_dir", required=True, type=str)
    ap.add_argument("--gt_subdir", default="recon/points_recon_spin_gt", type=str)
    ap.add_argument("--pred_subdir", default="recon/points_recon_spin", type=str)
    ap.add_argument("--out_dir", default="", type=str)
    ap.add_argument("--seed", default=0, type=int)
    ap.add_argument("--max_plot_points", default=12000, type=int)
    args = ap.parse_args()

    scene_dir = Path(args.scene_dir).resolve()
    if not scene_dir.exists():
        raise SystemExit(f"--scene_dir does not exist: {scene_dir}")

    out_dir = Path(args.out_dir).resolve() if str(args.out_dir).strip() else (scene_dir / "analysis" / "spin_recon_quality")
    _ensure_dir(out_dir)

    capture_meta_path = scene_dir / "capture_meta.json"
    if not capture_meta_path.exists():
        raise SystemExit(f"Missing capture_meta.json: {capture_meta_path}")
    capture_meta = _load_json(capture_meta_path)

    gt_branch = _load_recon_branch(scene_dir, str(args.gt_subdir))
    pred_branch = _load_recon_branch(scene_dir, str(args.pred_subdir))

    gt_canonical = _recover_canonical(gt_branch["sample_points"], capture_meta, gt_branch["sample_ts"])
    pred_canonical = _recover_canonical(pred_branch["sample_points"], capture_meta, pred_branch["sample_ts"])

    gt_stats = _summarize_branch(gt_branch, gt_canonical)
    pred_stats = _summarize_branch(pred_branch, pred_canonical)
    notes, diag_metrics, diag_flags = _diagnose_scene(gt_branch, pred_branch, gt_stats, pred_stats)

    _plot_canonical_compare(
        out_dir / "canonical_compare.png",
        scene_name=scene_dir.name,
        gt_points=gt_canonical,
        pred_points=pred_canonical,
        seed=int(args.seed),
        max_plot_points=int(args.max_plot_points),
    )
    _plot_count_compare(
        out_dir / "counts_compare.png",
        scene_name=scene_dir.name,
        gt_branch=gt_branch,
        pred_branch=pred_branch,
    )
    _write_summary_md(
        out_dir / "summary.md",
        scene_name=scene_dir.name,
        gt_branch=gt_branch,
        pred_branch=pred_branch,
        gt_stats=gt_stats,
        pred_stats=pred_stats,
        notes=notes,
        diag_metrics=diag_metrics,
        diag_flags=diag_flags,
    )

    report = {
        "scene_dir": str(scene_dir),
        "gt_subdir": str(args.gt_subdir),
        "pred_subdir": str(args.pred_subdir),
        "gt_stats": gt_stats,
        "pred_stats": pred_stats,
        "notes": notes,
        "diagnosis_metrics": diag_metrics,
        "diagnosis_flags": diag_flags,
        "artifacts": {
            "canonical_compare": str(out_dir / "canonical_compare.png"),
            "counts_compare": str(out_dir / "counts_compare.png"),
            "summary": str(out_dir / "summary.md"),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[ok] scene={scene_dir.name}")
    print(f"[ok] out_dir={out_dir}")


if __name__ == "__main__":
    main()
