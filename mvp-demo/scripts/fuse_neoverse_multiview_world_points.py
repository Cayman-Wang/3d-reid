from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"Failed to read json: {path}\nError: {exc!r}")


def _read_index(index_path: Path) -> list[dict[str, str]]:
    with index_path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _resolve_scene_points_root(root_value: str, scene_id: str, repo: Path) -> Path:
    root = Path(str(root_value))
    if not root.is_absolute():
        root = repo / root
    return root / scene_id / "points_per_view"


def _voxel_keys(points_xyz: np.ndarray, voxel_size_m: float) -> np.ndarray:
    if points_xyz.size == 0:
        return np.zeros((0, 3), dtype=np.int64)
    return np.floor(points_xyz / float(voxel_size_m)).astype(np.int64)


def _aggregate_voxels(points_xyz: np.ndarray, voxel_size_m: float, cam_ids: list[str] | None = None) -> list[dict[str, Any]]:
    pts = np.asarray(points_xyz, dtype=np.float32)
    if pts.size == 0:
        return []
    vox = _voxel_keys(pts, voxel_size_m)
    groups: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for idx, key in enumerate(map(tuple, vox.tolist())):
        groups[key].append(idx)

    items: list[dict[str, Any]] = []
    for key, indices in groups.items():
        cluster = pts[indices]
        centroid = cluster.mean(axis=0)
        item = {
            "voxel": key,
            "xyz": centroid.astype(np.float32),
            "count": int(cluster.shape[0]),
        }
        if cam_ids is not None and len(cam_ids) == len(pts):
            item["cam_support"] = int(len(set(cam_ids[i] for i in indices)))
        else:
            item["cam_support"] = 1
        items.append(item)
    items.sort(key=lambda row: (-int(row["cam_support"]), -int(row["count"])))
    return items


def _connected_components_from_voxels(voxels: list[dict[str, Any]]) -> list[list[int]]:
    if not voxels:
        return []
    key_to_idx = {tuple(item["voxel"]): idx for idx, item in enumerate(voxels)}
    parent = list(range(len(voxels)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    neighbor_offsets = [
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
        if not (dx == dy == dz == 0)
    ]
    for idx, item in enumerate(voxels):
        x, y, z = item["voxel"]
        for dx, dy, dz in neighbor_offsets:
            neighbor = (x + dx, y + dy, z + dz)
            other = key_to_idx.get(neighbor)
            if other is not None:
                union(idx, other)

    groups: dict[int, list[int]] = defaultdict(list)
    for idx in range(len(voxels)):
        groups[find(idx)].append(idx)
    return sorted(groups.values(), key=len, reverse=True)


def _select_background_voxels(voxels: list[dict[str, Any]], min_cam_support: int) -> list[dict[str, Any]]:
    if not voxels:
        return []
    supported = [item for item in voxels if int(item.get("cam_support", 1)) >= int(min_cam_support)]
    return supported if supported else voxels


def _summarize_components(voxels: list[dict[str, Any]], components: list[list[int]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for indices in components:
        if not indices:
            continue
        xyz = np.stack([np.asarray(voxels[idx]["xyz"], dtype=np.float32) for idx in indices], axis=0)
        weights = np.asarray([int(voxels[idx]["count"]) for idx in indices], dtype=np.float32)
        if float(weights.sum()) > 0:
            centroid = np.average(xyz, axis=0, weights=weights).astype(np.float32)
        else:
            centroid = xyz.mean(axis=0).astype(np.float32)
        summaries.append(
            {
                "indices": list(indices),
                "centroid": centroid,
                "raw_count": int(weights.sum()),
                "voxel_count": int(len(indices)),
            }
        )
    summaries.sort(key=lambda row: (-int(row["raw_count"]), -int(row["voxel_count"])))
    return summaries


def _choose_dynamic_component(
    component_summaries: list[dict[str, Any]],
    prev_centroid: np.ndarray | None,
    dynamic_track_radius_m: float,
) -> tuple[int | None, str]:
    if not component_summaries:
        return None, "empty"
    if prev_centroid is not None:
        candidates: list[tuple[float, int, int, int]] = []
        for idx, component in enumerate(component_summaries):
            distance = float(np.linalg.norm(np.asarray(component["centroid"], dtype=np.float32) - prev_centroid))
            if distance <= float(dynamic_track_radius_m):
                candidates.append((distance, -int(component["raw_count"]), -int(component["voxel_count"]), idx))
        if candidates:
            candidates.sort()
            return int(candidates[0][3]), "track_nearest"
    return 0, "fallback_largest"


def _merge_component_ids(
    component_summaries: list[dict[str, Any]],
    selected_idx: int | None,
    dynamic_merge_radius_m: float,
) -> list[int]:
    if selected_idx is None or selected_idx < 0 or selected_idx >= len(component_summaries):
        return []
    selected_centroid = np.asarray(component_summaries[selected_idx]["centroid"], dtype=np.float32)
    merged = [int(selected_idx)]
    for idx, component in enumerate(component_summaries):
        if idx == selected_idx:
            continue
        distance = float(np.linalg.norm(np.asarray(component["centroid"], dtype=np.float32) - selected_centroid))
        if distance <= float(dynamic_merge_radius_m):
            merged.append(int(idx))
    return merged


def _save_npy(path: Path, points_xyz: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(points_xyz, dtype=np.float32))


def main() -> None:
    ap = argparse.ArgumentParser(description="Fuse NeoVerse world-coordinate points from per-camera backprojections.")
    ap.add_argument("--scene_dir", required=True, type=str)
    ap.add_argument("--points_root", default="mvp-demo/output/neoverse_fused", type=str)
    ap.add_argument("--bg_points_root", default=None, type=str)
    ap.add_argument("--fg_points_root", default=None, type=str)
    ap.add_argument("--out_root", default="mvp-demo/output/neoverse_fused", type=str)
    ap.add_argument("--cams", default="cam0,cam1,cam2", type=str)
    ap.add_argument("--voxel_size_m", default=None, type=float, help="Compatibility alias used when bg/dynamic voxel sizes are not provided")
    ap.add_argument("--bg_voxel_size_m", default=None, type=float)
    ap.add_argument("--dynamic_voxel_size_m", default=None, type=float)
    ap.add_argument("--min_bg_cam_support", default=2, type=int)
    ap.add_argument("--dynamic_min_component_points", default=12, type=int)
    ap.add_argument("--dynamic_track_radius_m", default=0.40, type=float)
    ap.add_argument("--dynamic_merge_radius_m", default=0.08, type=float)
    ap.add_argument("--max_background_points", default=500000, type=int)
    ap.add_argument("--max_dynamic_points", default=50000, type=int)
    ap.add_argument("--background_source_branch", default="legacy_single_branch", type=str)
    ap.add_argument("--dynamic_source_branch", default="legacy_single_branch", type=str)
    args = ap.parse_args()

    repo = _repo_root()
    scene_dir = Path(str(args.scene_dir)).resolve()
    if not scene_dir.exists():
        raise SystemExit(f"Missing --scene_dir: {scene_dir}")

    scene_id = scene_dir.name
    cams = [str(c.strip()) for c in str(args.cams).split(",") if c.strip()]
    if not cams:
        raise SystemExit("--cams is empty")
    allowed_cams = set(cams)
    bg_voxel_size_m = (
        float(args.bg_voxel_size_m)
        if args.bg_voxel_size_m is not None
        else float(args.voxel_size_m)
        if args.voxel_size_m is not None
        else 0.02
    )
    dynamic_voxel_size_m = (
        float(args.dynamic_voxel_size_m)
        if args.dynamic_voxel_size_m is not None
        else float(args.voxel_size_m)
        if args.voxel_size_m is not None
        else 0.01
    )
    points_root = _resolve_scene_points_root(str(args.points_root), scene_id, repo)
    bg_points_root = _resolve_scene_points_root(str(args.bg_points_root or args.points_root), scene_id, repo)
    fg_points_root = _resolve_scene_points_root(str(args.fg_points_root or args.points_root), scene_id, repo)
    if not bg_points_root.exists():
        raise SystemExit(f"Missing bg_points_root: {bg_points_root}")
    if not fg_points_root.exists():
        raise SystemExit(f"Missing fg_points_root: {fg_points_root}")

    out_root = Path(str(args.out_root))
    if not out_root.is_absolute():
        out_root = repo / out_root
    fused_root = out_root / scene_id / "fused"
    fused_root.mkdir(parents=True, exist_ok=True)

    bg_index_path = bg_points_root / "points_index.csv"
    fg_index_path = fg_points_root / "points_index.csv"
    if not bg_index_path.exists():
        raise SystemExit(f"Missing bg points index: {bg_index_path}")
    if not fg_index_path.exists():
        raise SystemExit(f"Missing fg points index: {fg_index_path}")
    bg_rows = [row for row in _read_index(bg_index_path) if str(row.get("cam_id", "")).strip() in allowed_cams]
    fg_rows = [row for row in _read_index(fg_index_path) if str(row.get("cam_id", "")).strip() in allowed_cams]
    if not bg_rows:
        raise SystemExit(f"No rows in bg points index for cams={cams}: {bg_index_path}")
    if not fg_rows:
        raise SystemExit(f"No rows in fg points index for cams={cams}: {fg_index_path}")

    def _row_key(row: dict[str, str]) -> tuple[str, int, str]:
        return (str(row["cam_id"]), int(float(row["logical_t_idx"])), str(row["scene_stem"]))

    bg_by_key = {_row_key(row): row for row in bg_rows}
    fg_by_key = {_row_key(row): row for row in fg_rows}
    bg_keys = set(bg_by_key.keys())
    fg_keys = set(fg_by_key.keys())
    if bg_keys != fg_keys:
        missing_in_fg = sorted(bg_keys - fg_keys)[:5]
        missing_in_bg = sorted(fg_keys - bg_keys)[:5]
        raise SystemExit(
            "BG/FG branch points index keys mismatch; logical_t_idx/scene_stem/cam_id must align exactly. "
            f"missing_in_fg_sample={missing_in_fg}, missing_in_bg_sample={missing_in_bg}"
        )
    rows = [
        {
            "cam_id": key[0],
            "logical_t_idx": key[1],
            "scene_stem": key[2],
            "bg_row": bg_by_key[key],
            "fg_row": fg_by_key[key],
        }
        for key in sorted(bg_keys, key=lambda item: (item[1], item[2], item[0]))
    ]

    bg_points_all: list[np.ndarray] = []
    bg_cam_ids: list[str] = []
    fg_by_time: dict[str, list[np.ndarray]] = defaultdict(list)
    fg_cam_by_time: dict[str, list[str]] = defaultdict(list)
    raw_counts = {"bg": 0, "fg": 0}

    for row in rows:
        cam_id = str(row["cam_id"])
        scene_stem = str(row["scene_stem"])
        logical_t_idx = int(row["logical_t_idx"])
        bg_row = row["bg_row"]
        fg_row = row["fg_row"]
        bg_path = bg_points_root / str(bg_row["bg_path"])
        fg_path = fg_points_root / str(fg_row["fg_path"])
        if not bg_path.exists() or not fg_path.exists():
            raise SystemExit(f"Missing per-view npz for aligned key {cam_id}/{logical_t_idx}/{scene_stem}")

        bg_npz = np.load(bg_path)
        fg_npz = np.load(fg_path)
        bg_xyz = np.asarray(bg_npz["xyz"], dtype=np.float32)
        fg_xyz = np.asarray(fg_npz["xyz"], dtype=np.float32)
        raw_counts["bg"] += int(bg_xyz.shape[0])
        raw_counts["fg"] += int(fg_xyz.shape[0])

        if bg_xyz.size:
            bg_points_all.append(bg_xyz)
            bg_cam_ids.extend([cam_id] * int(bg_xyz.shape[0]))
        if fg_xyz.size:
            key = f"{logical_t_idx:06d}_{scene_stem}"
            fg_by_time[key].append(fg_xyz)
            fg_cam_by_time[key].extend([cam_id] * int(fg_xyz.shape[0]))

    bg_points = np.concatenate(bg_points_all, axis=0) if bg_points_all else np.zeros((0, 3), dtype=np.float32)
    raw_bg_voxels = _aggregate_voxels(bg_points, float(bg_voxel_size_m), cam_ids=bg_cam_ids)
    selected_bg_voxels = _select_background_voxels(raw_bg_voxels, int(args.min_bg_cam_support))
    bg_points_fused = np.stack([item["xyz"] for item in selected_bg_voxels], axis=0).astype(np.float32) if selected_bg_voxels else np.zeros((0, 3), dtype=np.float32)
    if bg_points_fused.shape[0] > int(args.max_background_points):
        bg_points_fused = bg_points_fused[: int(args.max_background_points)]

    dynamic_dir = fused_root / "dynamic"
    dynamic_dir.mkdir(parents=True, exist_ok=True)
    _save_npy(fused_root / "background_world.npy", bg_points_fused)

    dynamic_rows: list[dict[str, Any]] = []
    dynamic_centroids: list[np.ndarray] = []
    dynamic_keys = sorted(fg_by_time.keys())
    prev_accepted_centroid: np.ndarray | None = None
    for key in dynamic_keys:
        pieces = fg_by_time[key]
        points = np.concatenate(pieces, axis=0) if pieces else np.zeros((0, 3), dtype=np.float32)
        selected_component_size = 0
        selected_centroid = None
        selection_mode = "empty"
        merged_component_count = 0
        if points.size == 0:
            fused = np.zeros((0, 3), dtype=np.float32)
            main_component_ratio = 0.0
            component_count = 0
        else:
            voxels = _aggregate_voxels(points, float(dynamic_voxel_size_m), cam_ids=fg_cam_by_time[key])
            components = _connected_components_from_voxels(voxels)
            if not components:
                fused = np.zeros((0, 3), dtype=np.float32)
                main_component_ratio = 0.0
                component_count = 0
                prev_accepted_centroid = None
            else:
                component_summaries = _summarize_components(voxels, components)
                selected_idx, selection_mode = _choose_dynamic_component(
                    component_summaries=component_summaries,
                    prev_centroid=prev_accepted_centroid,
                    dynamic_track_radius_m=float(args.dynamic_track_radius_m),
                )
                if selected_idx is None:
                    fused = np.zeros((0, 3), dtype=np.float32)
                    main_component_ratio = 0.0
                    component_count = 0
                else:
                    component_count = len(component_summaries)
                    merged_ids = _merge_component_ids(
                        component_summaries=component_summaries,
                        selected_idx=selected_idx,
                        dynamic_merge_radius_m=float(args.dynamic_merge_radius_m),
                    )
                    merged_component_count = int(len(merged_ids))
                    selected_component = component_summaries[selected_idx]
                    selected_component_size = int(selected_component["raw_count"])
                    selected_centroid = np.asarray(selected_component["centroid"], dtype=np.float32)
                    fused_indices: list[int] = []
                    for component_id in merged_ids:
                        fused_indices.extend(int(voxel_idx) for voxel_idx in component_summaries[component_id]["indices"])
                    fused_voxels = [voxels[idx] for idx in fused_indices]
                    fused = np.stack([item["xyz"] for item in fused_voxels], axis=0).astype(np.float32)
                    if fused.shape[0] > int(args.max_dynamic_points):
                        fused = fused[: int(args.max_dynamic_points)]
                    main_component_ratio = float(selected_component_size) / float(max(points.shape[0], 1))
                    if fused.shape[0] < int(args.dynamic_min_component_points):
                        fused = np.zeros((0, 3), dtype=np.float32)
                        main_component_ratio = 0.0
                        prev_accepted_centroid = None
                    else:
                        prev_accepted_centroid = fused.mean(axis=0).astype(np.float32)
                if selected_idx is None:
                    prev_accepted_centroid = None
        if points.size == 0:
            prev_accepted_centroid = None

        scene_stem = key.split("_", 1)[1]
        logical_t_idx = int(key.split("_", 1)[0])
        out_path = dynamic_dir / f"{scene_stem}.npy"
        _save_npy(out_path, fused)
        dynamic_rows.append(
            {
                "scene_stem": scene_stem,
                "logical_t_idx": logical_t_idx,
                "points_path": out_path.relative_to(fused_root).as_posix(),
                "raw_points": int(points.shape[0]),
                "fused_points": int(fused.shape[0]),
                "main_component_ratio": float(main_component_ratio),
                "component_count": int(component_count),
                "selected_component_size": int(selected_component_size),
                "selected_centroid": None if selected_centroid is None else np.asarray(selected_centroid, dtype=np.float32).tolist(),
                "selection_mode": str(selection_mode),
                "merged_component_count": int(merged_component_count),
            }
        )
        if fused.size:
            dynamic_centroids.append(fused.mean(axis=0))

    with (fused_root / "dynamic_index.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "scene_stem",
                "logical_t_idx",
                "points_path",
                "raw_points",
                "fused_points",
                "main_component_ratio",
                "component_count",
                "selected_component_size",
                "selected_centroid",
                "selection_mode",
                "merged_component_count",
            ],
        )
        writer.writeheader()
        for row in dynamic_rows:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False) if isinstance(value, list) else value
                    for key, value in row.items()
                }
            )

    raw_support_counts = [int(item["cam_support"]) for item in raw_bg_voxels] if raw_bg_voxels else []
    raw_support_rate = float(np.mean(np.asarray(raw_support_counts) >= 2)) if raw_support_counts else 0.0
    raw_support_mean = float(np.mean(raw_support_counts)) if raw_support_counts else 0.0
    raw_support_histogram = {str(k): int(sum(1 for item in raw_bg_voxels if int(item["cam_support"]) == k)) for k in range(1, 5)}

    selected_support_counts = [int(item["cam_support"]) for item in selected_bg_voxels] if selected_bg_voxels else []
    selected_support_rate = float(np.mean(np.asarray(selected_support_counts) >= 2)) if selected_support_counts else 0.0
    selected_support_mean = float(np.mean(selected_support_counts)) if selected_support_counts else 0.0
    selected_support_histogram = {str(k): int(sum(1 for item in selected_bg_voxels if int(item["cam_support"]) == k)) for k in range(1, 5)}

    centroid_continuity = {
        "mean_step_l2": 0.0,
        "median_step_l2": 0.0,
        "max_step_l2": 0.0,
    }
    if len(dynamic_centroids) >= 2:
        centroid_arr = np.stack(dynamic_centroids, axis=0).astype(np.float32)
        steps = np.linalg.norm(np.diff(centroid_arr, axis=0), axis=1)
        centroid_continuity = {
            "mean_step_l2": float(np.mean(steps)),
            "median_step_l2": float(np.median(steps)),
            "max_step_l2": float(np.max(steps)),
        }

    meta = {
        "schema_version": "neoverse_fused_world_points_v1",
        "scene_id": scene_id,
        "scene_dir": scene_dir.as_posix(),
        "points_root": points_root.as_posix(),
        "bg_points_root": bg_points_root.as_posix(),
        "fg_points_root": fg_points_root.as_posix(),
        "fused_root": fused_root.as_posix(),
        "points_root_mode": "dual_source" if str(bg_points_root) != str(fg_points_root) else "single_source",
        "background_source_branch": str(args.background_source_branch),
        "dynamic_source_branch": str(args.dynamic_source_branch),
        "cams": cams,
        "voxel_size_m": None if args.voxel_size_m is None else float(args.voxel_size_m),
        "bg_voxel_size_m": float(bg_voxel_size_m),
        "dynamic_voxel_size_m": float(dynamic_voxel_size_m),
        "min_bg_cam_support": int(args.min_bg_cam_support),
        "dynamic_min_component_points": int(args.dynamic_min_component_points),
        "dynamic_track_radius_m": float(args.dynamic_track_radius_m),
        "dynamic_merge_radius_m": float(args.dynamic_merge_radius_m),
        "max_background_points": int(args.max_background_points),
        "max_dynamic_points": int(args.max_dynamic_points),
        "raw_counts": raw_counts,
        "background": {
            "raw_points": int(bg_points.shape[0]),
            "voxels": int(len(selected_bg_voxels)),
            "raw_voxels": int(len(raw_bg_voxels)),
            "selected_voxels": int(len(selected_bg_voxels)),
            "fused_points": int(bg_points_fused.shape[0]),
            "mean_cam_support": raw_support_mean,
            "support_rate_ge_2": raw_support_rate,
            "support_histogram": raw_support_histogram,
            "raw_mean_cam_support": raw_support_mean,
            "raw_support_rate_ge_2": raw_support_rate,
            "raw_support_histogram": raw_support_histogram,
            "selected_mean_cam_support": selected_support_mean,
            "selected_support_rate_ge_2": selected_support_rate,
            "selected_support_histogram": selected_support_histogram,
        },
        "dynamic": {
            "num_timestamps": int(len(dynamic_rows)),
            "centroid_continuity": centroid_continuity,
            "timestamps": dynamic_rows,
        },
    }
    (fused_root / "fusion_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote fused world points to: {fused_root}")
    print(f"Wrote: {fused_root / 'background_world.npy'}")
    print(f"Wrote: {fused_root / 'dynamic_index.csv'}")
    print(f"Wrote: {fused_root / 'fusion_meta.json'}")


if __name__ == "__main__":
    main()
