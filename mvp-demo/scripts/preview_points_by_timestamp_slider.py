from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_index_rows(index_csv: Path) -> list[dict[str, str]]:
    if not index_csv.exists():
        raise SystemExit(f"Missing points_by_timestamp index: {index_csv}")
    with index_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"No rows found in: {index_csv}")
    rows.sort(key=lambda row: int(float(row.get("logical_t_idx", 0))))
    return rows


def _load_frame_times(frame_times_csv: Path) -> dict[str, int]:
    if not frame_times_csv.exists():
        raise SystemExit(f"Missing frame_times.csv: {frame_times_csv}")
    out: dict[str, int] = {}
    with frame_times_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            stem = Path(str(row.get("filename") or "")).stem
            if not stem:
                continue
            try:
                ts_us = int(float(row["ts_us"]))
            except Exception:
                continue
            out.setdefault(stem, ts_us)
    return out


def _sample_points(points_xyz: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    pts = np.asarray(points_xyz, dtype=np.float32)
    if pts.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise SystemExit(f"Expected Nx3 point cloud, got shape={pts.shape}")
    if max_points <= 0 or pts.shape[0] <= int(max_points):
        return pts.astype(np.float32, copy=False)
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(pts.shape[0], size=int(max_points), replace=False)
    idx.sort()
    return np.asarray(pts[idx], dtype=np.float32)


def _update_bounds(bounds_min: np.ndarray | None, bounds_max: np.ndarray | None, points_xyz: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
    pts = np.asarray(points_xyz, dtype=np.float32)
    if pts.size == 0:
        return bounds_min, bounds_max
    cur_min = pts.min(axis=0)
    cur_max = pts.max(axis=0)
    if bounds_min is None or bounds_max is None:
        return cur_min.astype(np.float32), cur_max.astype(np.float32)
    return np.minimum(bounds_min, cur_min), np.maximum(bounds_max, cur_max)


def _set_equal_limits(ax, bounds_min: np.ndarray | None, bounds_max: np.ndarray | None) -> None:
    if bounds_min is None or bounds_max is None:
        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(-1.0, 1.0)
        ax.set_zlim(-1.0, 1.0)
        return
    center = 0.5 * (bounds_min + bounds_max)
    span = np.maximum(bounds_max - bounds_min, 1e-3)
    radius = 0.55 * float(np.max(span))
    ax.set_xlim(float(center[0] - radius), float(center[0] + radius))
    ax.set_ylim(float(center[1] - radius), float(center[1] + radius))
    ax.set_zlim(float(center[2] - radius), float(center[2] + radius))
    try:
        ax.set_box_aspect((1.0, 1.0, 1.0))
    except Exception:
        pass


def main() -> None:
    ap = argparse.ArgumentParser(description="Preview points_by_timestamp/*.npy with a local time slider.")
    ap.add_argument("--scene_dir", required=True, type=str)
    ap.add_argument("--fused_root", default="mvp-demo/output/neoverse_fused", type=str)
    ap.add_argument("--max_dynamic_points", default=30000, type=int)
    ap.add_argument("--max_background_points", default=20000, type=int)
    ap.add_argument("--hide_background", action="store_true")
    ap.add_argument("--seed", default=0, type=int)
    args = ap.parse_args()

    repo = _repo_root()
    scene_dir = Path(str(args.scene_dir)).resolve()
    if not scene_dir.exists():
        raise SystemExit(f"Missing --scene_dir: {scene_dir}")

    fused_root = Path(str(args.fused_root))
    if not fused_root.is_absolute():
        fused_root = repo / fused_root

    scene_id = scene_dir.name
    points_dir = fused_root / scene_id / "points_by_timestamp"
    fused_dir = fused_root / scene_id / "fused"
    index_csv = points_dir / "index.csv"
    background_path = fused_dir / "background_world.npy"
    frame_times_csv = scene_dir / "frame_times.csv"

    index_rows = _load_index_rows(index_csv)
    frame_times_by_stem = _load_frame_times(frame_times_csv)

    background_sample = np.zeros((0, 3), dtype=np.float32)
    if not bool(args.hide_background) and background_path.exists():
        background = np.asarray(np.load(str(background_path), mmap_mode="r"), dtype=np.float32)
        background_sample = _sample_points(background, max_points=int(args.max_background_points), seed=int(args.seed) + 17)

    dynamic_samples: list[np.ndarray] = []
    bounds_min: np.ndarray | None = None
    bounds_max: np.ndarray | None = None
    bounds_min, bounds_max = _update_bounds(bounds_min, bounds_max, background_sample)

    for row in index_rows:
        points_rel = str(row.get("points_rel") or "").strip()
        if not points_rel:
            raise SystemExit(f"Invalid points_rel in {index_csv}: {row}")
        points_path = points_dir / points_rel
        if not points_path.exists():
            raise SystemExit(f"points_rel does not exist: {points_path}")
        points_xyz = np.asarray(np.load(str(points_path), mmap_mode="r"), dtype=np.float32)
        logical_t_idx = int(float(row.get("logical_t_idx", 0)))
        sampled = _sample_points(
            points_xyz,
            max_points=int(args.max_dynamic_points),
            seed=int(args.seed) + logical_t_idx * 9973 + 31,
        )
        dynamic_samples.append(sampled)
        bounds_min, bounds_max = _update_bounds(bounds_min, bounds_max, sampled)

    try:
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Slider
    except Exception as exc:
        raise SystemExit(f"Missing matplotlib dependency for slider preview: {exc!r}")

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    plt.subplots_adjust(bottom=0.18)

    background_artist = None
    if background_sample.size:
        background_artist = ax.scatter(
            background_sample[:, 0],
            background_sample[:, 1],
            background_sample[:, 2],
            s=1.0,
            c="#808080",
            alpha=0.18,
            depthshade=False,
        )

    dynamic_artist = None
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    _set_equal_limits(ax, bounds_min, bounds_max)

    slider_ax = fig.add_axes([0.14, 0.08, 0.72, 0.035])
    slider = Slider(
        ax=slider_ax,
        label="logical_t_idx",
        valmin=0,
        valmax=max(len(index_rows) - 1, 0),
        valinit=0,
        valstep=1,
    )

    help_text = "←/→: prev/next frame"
    fig.text(0.14, 0.03, help_text)

    def _draw_frame(frame_pos: int) -> None:
        nonlocal dynamic_artist
        frame_idx = int(frame_pos)
        frame_idx = max(0, min(frame_idx, len(index_rows) - 1))
        row = index_rows[frame_idx]
        points_xyz = dynamic_samples[frame_idx]
        if dynamic_artist is not None:
            dynamic_artist.remove()
            dynamic_artist = None
        if points_xyz.size:
            dynamic_artist = ax.scatter(
                points_xyz[:, 0],
                points_xyz[:, 1],
                points_xyz[:, 2],
                s=3.0,
                c="#ff8c00",
                alpha=0.85,
                depthshade=False,
            )
        logical_t_idx = int(float(row.get("logical_t_idx", frame_idx)))
        scene_stem = str(row.get("scene_stem") or "")
        ts_us = frame_times_by_stem.get(scene_stem)
        num_points = int(float(row.get("num_points", 0)))
        title = f"{scene_id} | logical_t_idx={logical_t_idx} | scene_stem={scene_stem} | dynamic_points={num_points}"
        if ts_us is not None:
            title += f" | ts_us={ts_us}"
        ax.set_title(title)
        fig.canvas.draw_idle()

    def _on_slider_change(val) -> None:
        _draw_frame(int(val))

    def _on_key(event) -> None:
        if event.key not in {"left", "right"}:
            return
        current = int(slider.val)
        if event.key == "left":
            slider.set_val(max(0, current - 1))
        elif event.key == "right":
            slider.set_val(min(len(index_rows) - 1, current + 1))

    slider.on_changed(_on_slider_change)
    fig.canvas.mpl_connect("key_press_event", _on_key)
    _draw_frame(0)
    plt.show()


if __name__ == "__main__":
    main()
