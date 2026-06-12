#!/usr/bin/env python3
"""Export offline CARLA-Air trajectory/node spatial preview assets."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import re
import shutil
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAJECTORY_CONFIG = REPO_ROOT / "configs" / "carla_air" / "trajectories" / "town10hd_coverage_first_v1.json"
DEFAULT_NODE_CONFIG = REPO_ROOT / "local" / "carla_air" / "camera_nodes" / "Town10HD_ground_to_air_nodes_v1.json"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "\u7ec4\u4f1a\u601d\u8def"
    / f"carla_air_trajectory_spatial_preview_{datetime.now().strftime('%Y_%m_%d')}"
)
DEFAULT_TRAJECTORY_IDS = [
    "traj_cov_01_all_nodes_sweep_return",
    "traj_cov_02_all_nodes_reverse_s_curve",
]
DEFAULT_THREE_JS_SOURCE = REPO_ROOT / "tools" / "carla_air" / "vendor" / "three.min.js"
THREE_JS_VERSION = "0.148.0"
COLORS = [
    "#1f77b4",
    "#d62728",
    "#2ca02c",
    "#9467bd",
    "#ff7f0e",
    "#17becf",
]
CREATED_BY = "tools/carla_air/export_trajectory_spatial_preview.py"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_class(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_-]+", "-", value.strip())
    return value or "item"


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.strip().lstrip("#")
    if len(value) != 6:
        return (0, 0, 0)
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


def _point(raw: dict[str, Any]) -> dict[str, float]:
    return {"x": float(raw["x"]), "y": float(raw["y"]), "z": float(raw["z"])}


def _distance(a: dict[str, float], b: dict[str, float]) -> float:
    return math.sqrt(
        (float(b["x"]) - float(a["x"])) ** 2
        + (float(b["y"]) - float(a["y"])) ** 2
        + (float(b["z"]) - float(a["z"])) ** 2
    )


def _path_length(points: list[dict[str, float]]) -> float:
    return sum(_distance(a, b) for a, b in zip(points[:-1], points[1:]))


def _cumulative_distances(points: list[dict[str, float]]) -> list[float]:
    distances = [0.0]
    for a, b in zip(points[:-1], points[1:]):
        distances.append(distances[-1] + _distance(a, b))
    return distances


def _bounds(points: list[dict[str, float]]) -> dict[str, float]:
    if not points:
        raise SystemExit("Cannot compute bounds for an empty point set")
    xs = [float(point["x"]) for point in points]
    ys = [float(point["y"]) for point in points]
    zs = [float(point["z"]) for point in points]
    return {
        "min_x": min(xs),
        "max_x": max(xs),
        "min_y": min(ys),
        "max_y": max(ys),
        "min_z": min(zs),
        "max_z": max(zs),
    }


def _pad_bounds(bounds: dict[str, float]) -> dict[str, float]:
    width = max(float(bounds["max_x"]) - float(bounds["min_x"]), 1e-6)
    depth = max(float(bounds["max_y"]) - float(bounds["min_y"]), 1e-6)
    height = max(float(bounds["max_z"]) - float(bounds["min_z"]), 1e-6)
    pad_xy = max(8.0, max(width, depth) * 0.12)
    pad_z = max(2.0, height * 0.18)
    return {
        "min_x": float(bounds["min_x"]) - pad_xy,
        "max_x": float(bounds["max_x"]) + pad_xy,
        "min_y": float(bounds["min_y"]) - pad_xy,
        "max_y": float(bounds["max_y"]) + pad_xy,
        "min_z": float(bounds["min_z"]) - pad_z,
        "max_z": float(bounds["max_z"]) + pad_z,
    }


def _bounds_summary(bounds: dict[str, float]) -> dict[str, Any]:
    return {
        "min_x": round(float(bounds["min_x"]), 6),
        "max_x": round(float(bounds["max_x"]), 6),
        "min_y": round(float(bounds["min_y"]), 6),
        "max_y": round(float(bounds["max_y"]), 6),
        "min_z": round(float(bounds["min_z"]), 6),
        "max_z": round(float(bounds["max_z"]), 6),
        "width_m": round(float(bounds["max_x"]) - float(bounds["min_x"]), 6),
        "depth_m": round(float(bounds["max_y"]) - float(bounds["min_y"]), 6),
        "height_m": round(float(bounds["max_z"]) - float(bounds["min_z"]), 6),
    }


def _nice_length(raw: float) -> float:
    if raw <= 0:
        return 1.0
    exponent = math.floor(math.log10(raw))
    base = 10**exponent
    fraction = raw / base
    if fraction <= 1.5:
        nice = 1.0
    elif fraction <= 3.0:
        nice = 2.0
    elif fraction <= 7.0:
        nice = 5.0
    else:
        nice = 10.0
    return nice * base


def _load_trajectories(path: Path, selected_ids: list[str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    config = _load_json(path)
    by_id = {str(item.get("trajectory_id", "")): item for item in config.get("trajectories", []) if isinstance(item, dict)}
    missing = [trajectory_id for trajectory_id in selected_ids if trajectory_id not in by_id]
    if missing:
        available = ", ".join(sorted(by_id))
        raise SystemExit(f"Unknown trajectory id(s): {', '.join(missing)}. Available: {available}")

    trajectories: list[dict[str, Any]] = []
    for index, trajectory_id in enumerate(selected_ids):
        raw = by_id[trajectory_id]
        raw_points = raw.get("waypoints_carla")
        waypoint_source = "waypoints_carla"
        if not isinstance(raw_points, list) or len(raw_points) < 2:
            raw_points = raw.get("waypoints")
            waypoint_source = "waypoints"
        if not isinstance(raw_points, list) or len(raw_points) < 2:
            raise SystemExit(f"Trajectory requires at least two waypoints: {trajectory_id}")
        points = [_point(point) for point in raw_points]
        distances = _cumulative_distances(points)
        z_values = [point["z"] for point in points]
        trajectories.append(
            {
                "trajectory_id": trajectory_id,
                "description": str(raw.get("description", "")),
                "recommended_nodes": [str(node) for node in raw.get("recommended_nodes", [])],
                "target_nodes": [str(node) for node in raw.get("target_nodes", [])],
                "duration_seconds": float(raw.get("duration_seconds", 0.0) or 0.0),
                "waypoint_source": waypoint_source,
                "waypoints": points,
                "distances_m": distances,
                "path_length_m": distances[-1],
                "z_range": [min(z_values), max(z_values)],
                "color": COLORS[index % len(COLORS)],
                "class_name": "traj-" + _safe_class(trajectory_id),
            }
        )
    return config, trajectories


def _load_nodes(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    config = _load_json(path)
    nodes: list[dict[str, Any]] = []
    for raw in config.get("nodes", []):
        if not isinstance(raw, dict):
            continue
        node_id = str(raw.get("node_id", "")).strip()
        anchor = raw.get("anchor_transform")
        if not node_id or not isinstance(anchor, dict):
            continue
        nodes.append(
            {
                "node_id": node_id,
                "x": float(anchor.get("x", 0.0)),
                "y": float(anchor.get("y", 0.0)),
                "z": float(anchor.get("z", 0.0)),
                "pitch": float(anchor.get("pitch", 0.0)),
                "yaw": float(anchor.get("yaw", 0.0)),
                "roll": float(anchor.get("roll", 0.0)),
                "camera_order": [str(cam) for cam in raw.get("camera_order", [])],
            }
        )
    if not nodes:
        raise SystemExit(f"No camera nodes found in: {path}")
    return config, nodes


def _layout(width: int, height: int) -> dict[str, tuple[float, float, float, float]]:
    if width < 900 or height < 650:
        raise SystemExit("--width must be >= 900 and --height must be >= 650")
    legend_w = 310.0
    left = 70.0
    top = 86.0
    right_gap = 35.0
    profile_h = 135.0
    gap = 28.0
    bottom = 56.0
    xy_w = float(width) - left - legend_w - right_gap
    xy_h = float(height) - top - profile_h - gap - bottom
    if xy_w < 430 or xy_h < 300:
        raise SystemExit("Canvas is too small for the requested layout")
    profile_y = top + xy_h + gap
    return {
        "xy": (left, top, xy_w, xy_h),
        "profile": (left, profile_y, xy_w, profile_h),
        "legend": (left + xy_w + 28.0, top, legend_w - 40.0, float(height) - top - bottom),
    }


def _view(bounds: dict[str, float], rect: tuple[float, float, float, float]) -> dict[str, float]:
    x, y, width, height = rect
    data_w = max(float(bounds["max_x"]) - float(bounds["min_x"]), 1e-6)
    data_h = max(float(bounds["max_y"]) - float(bounds["min_y"]), 1e-6)
    scale = min(width / data_w, height / data_h)
    content_w = data_w * scale
    content_h = data_h * scale
    return {
        "min_x": float(bounds["min_x"]),
        "max_x": float(bounds["max_x"]),
        "min_y": float(bounds["min_y"]),
        "max_y": float(bounds["max_y"]),
        "scale": scale,
        "x0": x + (width - content_w) / 2.0,
        "y0": y + (height - content_h) / 2.0,
        "content_w": content_w,
        "content_h": content_h,
    }


def _xy(view: dict[str, float], x: float, y: float) -> tuple[float, float]:
    sx = float(view["x0"]) + (float(x) - float(view["min_x"])) * float(view["scale"])
    sy = float(view["y0"]) + (float(view["max_y"]) - float(y)) * float(view["scale"])
    return sx, sy


def _center_from_bounds(bounds: dict[str, float]) -> dict[str, float]:
    return {
        "x": (float(bounds["min_x"]) + float(bounds["max_x"])) / 2.0,
        "y": (float(bounds["min_y"]) + float(bounds["max_y"])) / 2.0,
        "z": float(bounds["min_z"]),
    }


def _relative_3d(point: dict[str, float], center: dict[str, float], z_scale: float) -> dict[str, float]:
    return {
        "x": float(point["x"]) - float(center["x"]),
        "y": float(point["y"]) - float(center["y"]),
        "z": (float(point["z"]) - float(center["z"])) * float(z_scale),
    }


def _iso_project(point: dict[str, float]) -> tuple[float, float]:
    x = float(point["x"])
    y = float(point["y"])
    z = float(point["z"])
    return ((x - y) * 0.8660254038, (x + y) * 0.5 - z)


def _bbox_corners(bounds: dict[str, float]) -> list[dict[str, float]]:
    return [
        {"x": x, "y": y, "z": z}
        for x in [float(bounds["min_x"]), float(bounds["max_x"])]
        for y in [float(bounds["min_y"]), float(bounds["max_y"])]
        for z in [float(bounds["min_z"]), float(bounds["max_z"])]
    ]


def _bbox_edges() -> list[tuple[int, int]]:
    return [
        (0, 1),
        (2, 3),
        (4, 5),
        (6, 7),
        (0, 2),
        (1, 3),
        (4, 6),
        (5, 7),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]


def _static_3d_view(
    raw_bounds: dict[str, float],
    padded_bounds: dict[str, float],
    trajectories: list[dict[str, Any]],
    nodes: list[dict[str, Any]],
    rect: tuple[float, float, float, float],
    z_scale: float,
) -> dict[str, Any]:
    center = _center_from_bounds(raw_bounds)
    rel_points: list[dict[str, float]] = []
    for point in _all_points(trajectories, nodes):
        rel_points.append(_relative_3d(point, center, z_scale))
    for corner in _bbox_corners(raw_bounds):
        rel_points.append(_relative_3d(corner, center, z_scale))
    for corner in _bbox_corners(padded_bounds):
        rel_points.append(_relative_3d(corner, center, z_scale))
    projected = [_iso_project(point) for point in rel_points]
    min_px = min(point[0] for point in projected)
    max_px = max(point[0] for point in projected)
    min_py = min(point[1] for point in projected)
    max_py = max(point[1] for point in projected)
    x, y, width, height = rect
    scale = min(width / max(max_px - min_px, 1e-6), height / max(max_py - min_py, 1e-6))
    content_w = (max_px - min_px) * scale
    content_h = (max_py - min_py) * scale
    return {
        "center": center,
        "z_scale": float(z_scale),
        "min_px": min_px,
        "max_px": max_px,
        "min_py": min_py,
        "max_py": max_py,
        "scale": scale,
        "x0": x + (width - content_w) / 2.0,
        "y0": y + (height - content_h) / 2.0,
        "content_w": content_w,
        "content_h": content_h,
    }


def _project_3d(view: dict[str, Any], point: dict[str, float]) -> tuple[float, float]:
    rel = _relative_3d(point, view["center"], float(view["z_scale"]))
    px, py = _iso_project(rel)
    sx = float(view["x0"]) + (px - float(view["min_px"])) * float(view["scale"])
    sy = float(view["y0"]) + (py - float(view["min_py"])) * float(view["scale"])
    return sx, sy


def _profile_point(
    rect: tuple[float, float, float, float],
    max_distance: float,
    z_bounds: dict[str, float],
    distance: float,
    z: float,
) -> tuple[float, float]:
    x, y, width, height = rect
    max_distance = max(max_distance, 1e-6)
    z_min = float(z_bounds["min_z"])
    z_max = float(z_bounds["max_z"])
    z_span = max(z_max - z_min, 1e-6)
    sx = x + (float(distance) / max_distance) * width
    sy = y + (z_max - float(z)) / z_span * height
    return sx, sy


def _points_attr(points: list[tuple[float, float]]) -> str:
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)


def _svg_text(x: float, y: float, text: str, *, size: int = 12, anchor: str = "start", klass: str = "") -> str:
    cls = f' class="{klass}"' if klass else ""
    return (
        f'<text{cls} x="{x:.2f}" y="{y:.2f}" font-family="Arial, sans-serif" '
        f'font-size="{size}" text-anchor="{anchor}">{html.escape(text)}</text>'
    )


def _trajectory_summary(trajectory: dict[str, Any]) -> dict[str, Any]:
    bounds = _bounds(trajectory["waypoints"])
    return {
        "trajectory_id": trajectory["trajectory_id"],
        "waypoint_source": trajectory["waypoint_source"],
        "waypoint_count": len(trajectory["waypoints"]),
        "path_length_m": round(float(trajectory["path_length_m"]), 6),
        "duration_seconds": round(float(trajectory["duration_seconds"]), 6),
        "recommended_nodes": trajectory["recommended_nodes"],
        "target_nodes": trajectory["target_nodes"],
        "bounds": _bounds_summary(bounds),
        "z_range_m": [round(float(trajectory["z_range"][0]), 6), round(float(trajectory["z_range"][1]), 6)],
    }


def _node_summary(node: dict[str, Any]) -> dict[str, Any]:
    return {
        "node_id": node["node_id"],
        "anchor_transform": {
            "x": round(float(node["x"]), 6),
            "y": round(float(node["y"]), 6),
            "z": round(float(node["z"]), 6),
            "pitch": round(float(node["pitch"]), 6),
            "yaw": round(float(node["yaw"]), 6),
            "roll": round(float(node["roll"]), 6),
        },
        "camera_count": len(node["camera_order"]),
        "camera_order": node["camera_order"],
    }


def _all_points(trajectories: list[dict[str, Any]], nodes: list[dict[str, Any]]) -> list[dict[str, float]]:
    points: list[dict[str, float]] = []
    for trajectory in trajectories:
        points.extend(trajectory["waypoints"])
    points.extend({"x": node["x"], "y": node["y"], "z": node["z"]} for node in nodes)
    return points


def _covers(bounds: dict[str, float], points: list[dict[str, float]]) -> bool:
    for point in points:
        if not (bounds["min_x"] <= point["x"] <= bounds["max_x"]):
            return False
        if not (bounds["min_y"] <= point["y"] <= bounds["max_y"]):
            return False
        if not (bounds["min_z"] <= point["z"] <= bounds["max_z"]):
            return False
    return True


def _runtime_map_extent(host: str, port: int, timeout: float) -> dict[str, Any]:
    def port_open() -> bool:
        try:
            with socket.create_connection((host, port), timeout=timeout):
                return True
        except OSError:
            return False

    if not port_open():
        return {"available": False, "attempted": True, "reason": f"carla_port_closed:{host}:{port}"}
    try:
        import carla  # type: ignore
    except Exception as exc:
        return {"available": False, "attempted": True, "reason": f"carla_import_failed:{type(exc).__name__}:{exc}"}
    try:
        client = carla.Client(host, int(port))
        client.set_timeout(max(float(timeout), 1.0))
        world = client.get_world()
        carla_map = world.get_map()
        waypoints = carla_map.generate_waypoints(50.0)
        locations = [wp.transform.location for wp in waypoints]
        if not locations:
            return {"available": False, "attempted": True, "reason": "carla_map_waypoints_empty"}
        xs = [float(loc.x) for loc in locations]
        ys = [float(loc.y) for loc in locations]
        zs = [float(loc.z) for loc in locations]
        return {
            "available": True,
            "attempted": True,
            "source": "carla_map_generate_waypoints_50m",
            "map_name": str(carla_map.name),
            "sample_count": len(locations),
            "bounds": _bounds_summary(
                {
                    "min_x": min(xs),
                    "max_x": max(xs),
                    "min_y": min(ys),
                    "max_y": max(ys),
                    "min_z": min(zs),
                    "max_z": max(zs),
                }
            ),
        }
    except Exception as exc:
        return {"available": False, "attempted": True, "reason": f"carla_map_extent_failed:{type(exc).__name__}:{exc}"}


def _draw_grid_svg(parts: list[str], view: dict[str, float]) -> None:
    min_x, max_x = float(view["min_x"]), float(view["max_x"])
    min_y, max_y = float(view["min_y"]), float(view["max_y"])
    spacing = _nice_length(max(max_x - min_x, max_y - min_y) / 6.0)
    start_x = math.floor(min_x / spacing) * spacing
    start_y = math.floor(min_y / spacing) * spacing
    x = start_x
    while x <= max_x + 1e-9:
        sx1, sy1 = _xy(view, x, min_y)
        sx2, sy2 = _xy(view, x, max_y)
        parts.append(f'<line class="grid" x1="{sx1:.2f}" y1="{sy1:.2f}" x2="{sx2:.2f}" y2="{sy2:.2f}"/>')
        if min_x <= x <= max_x:
            parts.append(_svg_text(sx1, sy1 + 16, f"{x:.0f}", size=10, anchor="middle", klass="axis-label"))
        x += spacing
    y = start_y
    while y <= max_y + 1e-9:
        sx1, sy1 = _xy(view, min_x, y)
        sx2, sy2 = _xy(view, max_x, y)
        parts.append(f'<line class="grid" x1="{sx1:.2f}" y1="{sy1:.2f}" x2="{sx2:.2f}" y2="{sy2:.2f}"/>')
        if min_y <= y <= max_y:
            parts.append(_svg_text(sx1 - 8, sy1 + 4, f"{y:.0f}", size=10, anchor="end", klass="axis-label"))
        y += spacing


def _draw_scale_bar_svg(parts: list[str], view: dict[str, float]) -> float:
    width_m = float(view["max_x"]) - float(view["min_x"])
    depth_m = float(view["max_y"]) - float(view["min_y"])
    length_m = _nice_length(min(width_m, depth_m) * 0.18)
    length_px = length_m * float(view["scale"])
    x0 = float(view["x0"]) + 18.0
    y0 = float(view["y0"]) + float(view["content_h"]) - 24.0
    parts.append(
        f'<line class="scale-bar" x1="{x0:.2f}" y1="{y0:.2f}" x2="{x0 + length_px:.2f}" y2="{y0:.2f}"/>'
    )
    parts.append(
        f'<line class="scale-bar" x1="{x0:.2f}" y1="{y0 - 5:.2f}" x2="{x0:.2f}" y2="{y0 + 5:.2f}"/>'
    )
    parts.append(
        f'<line class="scale-bar" x1="{x0 + length_px:.2f}" y1="{y0 - 5:.2f}" x2="{x0 + length_px:.2f}" y2="{y0 + 5:.2f}"/>'
    )
    parts.append(_svg_text(x0 + length_px / 2.0, y0 - 8.0, f"{length_m:g} m", size=11, anchor="middle"))
    return length_m


def _build_svg(
    *,
    width: int,
    height: int,
    trajectory_config: dict[str, Any],
    node_config: dict[str, Any],
    trajectories: list[dict[str, Any]],
    nodes: list[dict[str, Any]],
    raw_bounds: dict[str, float],
    padded_bounds: dict[str, float],
    runtime_extent: dict[str, Any],
) -> tuple[str, float]:
    layout = _layout(width, height)
    xy_rect = layout["xy"]
    profile_rect = layout["profile"]
    legend_rect = layout["legend"]
    view = _view(padded_bounds, xy_rect)
    max_distance = max(float(trajectory["path_length_m"]) for trajectory in trajectories)
    z_bounds = _pad_bounds({"min_x": 0.0, "max_x": 1.0, "min_y": 0.0, "max_y": 1.0, "min_z": raw_bounds["min_z"], "max_z": raw_bounds["max_z"]})
    map_name = str(trajectory_config.get("map") or node_config.get("map") or "unknown")
    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<defs>",
        "<style>",
        "text{fill:#20252b}.title{font-weight:700}.subtitle{fill:#56616f}.grid{stroke:#d8dde3;stroke-width:1}.axis-label{fill:#687382}.axis{stroke:#6f7b88;stroke-width:1.2}.roi{fill:none;stroke:#8c98a8;stroke-width:1.6;stroke-dasharray:6 5}.scale-bar{stroke:#20252b;stroke-width:2.5}.node{fill:#f7f9fb;stroke:#222;stroke-width:2}.node-arrow{stroke:#222;stroke-width:2.2}.waypoint{fill:#fff;stroke-width:1.6}.legend-panel{fill:#ffffff;stroke:#d4dae2;stroke-width:1}.profile-axis{stroke:#6f7b88;stroke-width:1.2}.profile-grid{stroke:#e1e5eb;stroke-width:1}.map-extent-note{fill:#687382}.toggle-hidden{display:none}",
        "</style>",
        "</defs>",
        '<rect x="0" y="0" width="100%" height="100%" fill="#f7f6f1"/>',
        _svg_text(28, 34, "CARLA-Air trajectory spatial preview", size=22, klass="title"),
        _svg_text(28, 58, f"Map={map_name} | local CARLA-world meters | presentation/QC only", size=13, klass="subtitle"),
    ]
    x0, y0, cw, ch = float(view["x0"]), float(view["y0"]), float(view["content_w"]), float(view["content_h"])
    parts.append(f'<rect x="{x0:.2f}" y="{y0:.2f}" width="{cw:.2f}" height="{ch:.2f}" fill="#ffffff" stroke="#c6ccd4"/>')
    parts.append('<g class="layer layer-grid">')
    _draw_grid_svg(parts, view)
    parts.append("</g>")
    parts.append(_svg_text(x0 + cw / 2.0, y0 + ch + 36.0, "CARLA X (m)", size=12, anchor="middle"))
    parts.append(_svg_text(x0 - 46.0, y0 + ch / 2.0, "CARLA Y (m)", size=12, anchor="middle"))

    roi_x1, roi_y1 = _xy(view, raw_bounds["min_x"], raw_bounds["max_y"])
    roi_x2, roi_y2 = _xy(view, raw_bounds["max_x"], raw_bounds["min_y"])
    parts.append('<g class="layer layer-roi">')
    parts.append(
        f'<rect class="roi" x="{roi_x1:.2f}" y="{roi_y1:.2f}" width="{roi_x2 - roi_x1:.2f}" height="{roi_y2 - roi_y1:.2f}"/>'
    )
    parts.append(_svg_text(roi_x1 + 6, roi_y1 - 8, "local ROI", size=11, klass="subtitle"))
    scale_bar_m = _draw_scale_bar_svg(parts, view)
    parts.append("</g>")

    for trajectory in trajectories:
        class_name = trajectory["class_name"]
        color = trajectory["color"]
        points = [_xy(view, point["x"], point["y"]) for point in trajectory["waypoints"]]
        parts.append(f'<g class="layer layer-trajectory {class_name}" data-trajectory-id="{html.escape(trajectory["trajectory_id"])}">')
        parts.append(
            f'<polyline points="{_points_attr(points)}" fill="none" stroke="{color}" stroke-width="4" stroke-linejoin="round" stroke-linecap="round"/>'
        )
        parts.append(f'<g class="layer-waypoints {class_name}-waypoints">')
        for idx, (sx, sy) in enumerate(points):
            parts.append(f'<circle class="waypoint" cx="{sx:.2f}" cy="{sy:.2f}" r="4.4" stroke="{color}"/>')
            label = str(idx + 1)
            if idx == 0:
                label = "S"
            elif idx == len(points) - 1:
                label = "E"
            parts.append(_svg_text(sx + 6, sy - 6, label, size=9, klass="waypoint-label"))
        parts.append("</g>")
        parts.append("</g>")

    parts.append('<g class="layer layer-nodes">')
    for node in nodes:
        sx, sy = _xy(view, node["x"], node["y"])
        yaw = math.radians(float(node["yaw"]))
        arrow_len = 8.0
        tx, ty = _xy(view, float(node["x"]) + math.cos(yaw) * arrow_len, float(node["y"]) + math.sin(yaw) * arrow_len)
        parts.append(f'<line class="node-arrow" x1="{sx:.2f}" y1="{sy:.2f}" x2="{tx:.2f}" y2="{ty:.2f}"/>')
        parts.append(f'<circle class="node" cx="{sx:.2f}" cy="{sy:.2f}" r="7.0"/>')
        parts.append(_svg_text(sx + 9.0, sy - 9.0, str(node["node_id"]), size=11, klass="node-label"))
    parts.append("</g>")

    px, py, pw, ph = profile_rect
    parts.append('<g class="layer layer-profile">')
    parts.append(f'<rect x="{px:.2f}" y="{py:.2f}" width="{pw:.2f}" height="{ph:.2f}" fill="#ffffff" stroke="#c6ccd4"/>')
    for frac in [0.25, 0.5, 0.75]:
        gy = py + ph * frac
        parts.append(f'<line class="profile-grid" x1="{px:.2f}" y1="{gy:.2f}" x2="{px + pw:.2f}" y2="{gy:.2f}"/>')
    parts.append(f'<line class="profile-axis" x1="{px:.2f}" y1="{py + ph:.2f}" x2="{px + pw:.2f}" y2="{py + ph:.2f}"/>')
    parts.append(f'<line class="profile-axis" x1="{px:.2f}" y1="{py:.2f}" x2="{px:.2f}" y2="{py + ph:.2f}"/>')
    parts.append(_svg_text(px, py - 10, "Height profile: z vs path distance", size=12, klass="subtitle"))
    parts.append(_svg_text(px + pw / 2.0, py + ph + 34, "Path distance (m)", size=11, anchor="middle"))
    parts.append(_svg_text(px - 8, py + 4, f"{z_bounds['max_z']:.1f}m", size=10, anchor="end", klass="axis-label"))
    parts.append(_svg_text(px - 8, py + ph, f"{z_bounds['min_z']:.1f}m", size=10, anchor="end", klass="axis-label"))
    parts.append(_svg_text(px + pw, py + ph + 16, f"{max_distance:.1f}m", size=10, anchor="end", klass="axis-label"))
    for trajectory in trajectories:
        profile_points = [
            _profile_point(profile_rect, max_distance, z_bounds, distance, point["z"])
            for distance, point in zip(trajectory["distances_m"], trajectory["waypoints"])
        ]
        parts.append(
            f'<polyline class="{trajectory["class_name"]}" points="{_points_attr(profile_points)}" fill="none" '
            f'stroke="{trajectory["color"]}" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"/>'
        )
    parts.append("</g>")

    lx, ly, lw, lh = legend_rect
    parts.append(f'<rect class="legend-panel" x="{lx:.2f}" y="{ly:.2f}" width="{lw:.2f}" height="{lh:.2f}" rx="4"/>')
    y_cursor = ly + 28.0
    parts.append(_svg_text(lx + 16, y_cursor, "Legend", size=14, klass="title"))
    y_cursor += 26.0
    for trajectory in trajectories:
        parts.append(
            f'<line x1="{lx + 16:.2f}" y1="{y_cursor - 4:.2f}" x2="{lx + 44:.2f}" y2="{y_cursor - 4:.2f}" '
            f'stroke="{trajectory["color"]}" stroke-width="4" stroke-linecap="round"/>'
        )
        parts.append(_svg_text(lx + 54, y_cursor, str(trajectory["trajectory_id"]), size=10))
        y_cursor += 16.0
        parts.append(
            _svg_text(
                lx + 54,
                y_cursor,
                f"wps={len(trajectory['waypoints'])} len={trajectory['path_length_m']:.1f}m",
                size=10,
                klass="subtitle",
            )
        )
        y_cursor += 18.0
    parts.append(f'<circle class="node" cx="{lx + 28:.2f}" cy="{y_cursor - 5:.2f}" r="6"/>')
    parts.append(_svg_text(lx + 54, y_cursor, "camera node anchor + yaw", size=10))
    y_cursor += 24.0
    parts.append(f'<rect class="roi" x="{lx + 18:.2f}" y="{y_cursor - 15:.2f}" width="24" height="14"/>')
    parts.append(_svg_text(lx + 54, y_cursor, "local ROI bounds", size=10))
    y_cursor += 30.0
    parts.append(_svg_text(lx + 16, y_cursor, "Scale", size=12, klass="title"))
    y_cursor += 18.0
    roi_summary = _bounds_summary(raw_bounds)
    parts.append(_svg_text(lx + 16, y_cursor, f"ROI X: {roi_summary['width_m']:.1f}m", size=10, klass="subtitle"))
    y_cursor += 16.0
    parts.append(_svg_text(lx + 16, y_cursor, f"ROI Y: {roi_summary['depth_m']:.1f}m", size=10, klass="subtitle"))
    y_cursor += 16.0
    parts.append(_svg_text(lx + 16, y_cursor, f"Z range: {roi_summary['height_m']:.1f}m", size=10, klass="subtitle"))
    y_cursor += 16.0
    parts.append(_svg_text(lx + 16, y_cursor, f"Scale bar: {scale_bar_m:g}m", size=10, klass="subtitle"))
    y_cursor += 28.0
    runtime_text = "runtime map extent: not requested"
    if runtime_extent.get("attempted"):
        runtime_text = "runtime map extent: available" if runtime_extent.get("available") else "runtime map extent: unavailable"
    parts.append(_svg_text(lx + 16, y_cursor, runtime_text, size=10, klass="map-extent-note"))
    y_cursor += 16.0
    parts.append(_svg_text(lx + 16, y_cursor, "not dataset/annotation evidence", size=10, klass="map-extent-note"))

    parts.append("</svg>")
    return "\n".join(parts) + "\n", scale_bar_m


def _draw_png(
    *,
    path: Path,
    width: int,
    height: int,
    trajectory_config: dict[str, Any],
    node_config: dict[str, Any],
    trajectories: list[dict[str, Any]],
    nodes: list[dict[str, Any]],
    raw_bounds: dict[str, float],
    padded_bounds: dict[str, float],
    runtime_extent: dict[str, Any],
    scale_bar_m: float,
) -> None:
    from PIL import Image, ImageDraw, ImageFont

    image = Image.new("RGB", (width, height), (247, 246, 241))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    layout = _layout(width, height)
    xy_rect = layout["xy"]
    profile_rect = layout["profile"]
    legend_rect = layout["legend"]
    view = _view(padded_bounds, xy_rect)
    z_bounds = _pad_bounds({"min_x": 0.0, "max_x": 1.0, "min_y": 0.0, "max_y": 1.0, "min_z": raw_bounds["min_z"], "max_z": raw_bounds["max_z"]})
    max_distance = max(float(trajectory["path_length_m"]) for trajectory in trajectories)
    map_name = str(trajectory_config.get("map") or node_config.get("map") or "unknown")

    def text(x: float, y: float, value: str, fill: tuple[int, int, int] = (32, 37, 43)) -> None:
        draw.text((float(x), float(y)), value, fill=fill, font=font)

    text(28, 22, "CARLA-Air trajectory spatial preview")
    text(28, 46, f"Map={map_name} | local CARLA-world meters | presentation/QC only", (86, 97, 111))
    x0, y0, cw, ch = float(view["x0"]), float(view["y0"]), float(view["content_w"]), float(view["content_h"])
    draw.rectangle((x0, y0, x0 + cw, y0 + ch), fill=(255, 255, 255), outline=(198, 204, 212), width=1)

    spacing = _nice_length(max(view["max_x"] - view["min_x"], view["max_y"] - view["min_y"]) / 6.0)
    x = math.floor(view["min_x"] / spacing) * spacing
    while x <= view["max_x"] + 1e-9:
        sx1, sy1 = _xy(view, x, view["min_y"])
        sx2, sy2 = _xy(view, x, view["max_y"])
        draw.line((sx1, sy1, sx2, sy2), fill=(216, 221, 227), width=1)
        text(sx1 - 10, sy1 + 4, f"{x:.0f}", (104, 115, 130))
        x += spacing
    y = math.floor(view["min_y"] / spacing) * spacing
    while y <= view["max_y"] + 1e-9:
        sx1, sy1 = _xy(view, view["min_x"], y)
        sx2, sy2 = _xy(view, view["max_x"], y)
        draw.line((sx1, sy1, sx2, sy2), fill=(216, 221, 227), width=1)
        text(sx1 - 38, sy1 - 6, f"{y:.0f}", (104, 115, 130))
        y += spacing

    roi_x1, roi_y1 = _xy(view, raw_bounds["min_x"], raw_bounds["max_y"])
    roi_x2, roi_y2 = _xy(view, raw_bounds["max_x"], raw_bounds["min_y"])
    draw.rectangle((roi_x1, roi_y1, roi_x2, roi_y2), outline=(140, 152, 168), width=2)
    text(roi_x1 + 6, roi_y1 - 16, "local ROI", (86, 97, 111))
    scale_px = scale_bar_m * float(view["scale"])
    sx = x0 + 18.0
    sy = y0 + ch - 24.0
    draw.line((sx, sy, sx + scale_px, sy), fill=(32, 37, 43), width=3)
    draw.line((sx, sy - 5, sx, sy + 5), fill=(32, 37, 43), width=2)
    draw.line((sx + scale_px, sy - 5, sx + scale_px, sy + 5), fill=(32, 37, 43), width=2)
    text(sx + scale_px / 2.0 - 14, sy - 22, f"{scale_bar_m:g} m")

    for trajectory in trajectories:
        color = _hex_to_rgb(trajectory["color"])
        points = [_xy(view, point["x"], point["y"]) for point in trajectory["waypoints"]]
        draw.line(points, fill=color, width=4, joint="curve")
        for idx, (px, py) in enumerate(points):
            draw.ellipse((px - 4, py - 4, px + 4, py + 4), fill=(255, 255, 255), outline=color, width=2)
            label = "S" if idx == 0 else "E" if idx == len(points) - 1 else str(idx + 1)
            text(px + 6, py - 10, label, color)

    for node in nodes:
        nx, ny = _xy(view, node["x"], node["y"])
        yaw = math.radians(float(node["yaw"]))
        arrow_len = 8.0
        tx, ty = _xy(view, float(node["x"]) + math.cos(yaw) * arrow_len, float(node["y"]) + math.sin(yaw) * arrow_len)
        draw.line((nx, ny, tx, ty), fill=(32, 37, 43), width=2)
        draw.ellipse((nx - 7, ny - 7, nx + 7, ny + 7), fill=(247, 249, 251), outline=(32, 37, 43), width=2)
        text(nx + 9, ny - 14, str(node["node_id"]))

    px, py, pw, ph = profile_rect
    draw.rectangle((px, py, px + pw, py + ph), fill=(255, 255, 255), outline=(198, 204, 212), width=1)
    text(px, py - 18, "Height profile: z vs path distance", (86, 97, 111))
    for trajectory in trajectories:
        color = _hex_to_rgb(trajectory["color"])
        points = [
            _profile_point(profile_rect, max_distance, z_bounds, distance, point["z"])
            for distance, point in zip(trajectory["distances_m"], trajectory["waypoints"])
        ]
        draw.line(points, fill=color, width=3)
    text(px - 42, py - 2, f"{z_bounds['max_z']:.1f}m", (104, 115, 130))
    text(px - 42, py + ph - 8, f"{z_bounds['min_z']:.1f}m", (104, 115, 130))
    text(px + pw - 48, py + ph + 8, f"{max_distance:.1f}m", (104, 115, 130))

    lx, ly, lw, lh = legend_rect
    draw.rectangle((lx, ly, lx + lw, ly + lh), fill=(255, 255, 255), outline=(212, 218, 226), width=1)
    y_cursor = ly + 18
    text(lx + 14, y_cursor, "Legend")
    y_cursor += 24
    for trajectory in trajectories:
        color = _hex_to_rgb(trajectory["color"])
        draw.line((lx + 14, y_cursor, lx + 42, y_cursor), fill=color, width=4)
        text(lx + 52, y_cursor - 6, str(trajectory["trajectory_id"]))
        y_cursor += 16
        text(lx + 52, y_cursor - 6, f"wps={len(trajectory['waypoints'])} len={trajectory['path_length_m']:.1f}m", (86, 97, 111))
        y_cursor += 20
    draw.ellipse((lx + 20, y_cursor - 8, lx + 32, y_cursor + 4), fill=(247, 249, 251), outline=(32, 37, 43), width=2)
    text(lx + 52, y_cursor - 7, "camera node anchor + yaw")
    y_cursor += 24
    roi_summary = _bounds_summary(raw_bounds)
    text(lx + 14, y_cursor, f"ROI X: {roi_summary['width_m']:.1f}m", (86, 97, 111))
    y_cursor += 16
    text(lx + 14, y_cursor, f"ROI Y: {roi_summary['depth_m']:.1f}m", (86, 97, 111))
    y_cursor += 16
    text(lx + 14, y_cursor, f"Z range: {roi_summary['height_m']:.1f}m", (86, 97, 111))
    y_cursor += 24
    runtime_text = "runtime map extent: not requested"
    if runtime_extent.get("attempted"):
        runtime_text = "runtime map extent: available" if runtime_extent.get("available") else "runtime map extent: unavailable"
    text(lx + 14, y_cursor, runtime_text, (104, 115, 130))
    y_cursor += 16
    text(lx + 14, y_cursor, "not dataset/annotation evidence", (104, 115, 130))

    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def _build_static_3d_svg(
    *,
    width: int,
    height: int,
    trajectory_config: dict[str, Any],
    node_config: dict[str, Any],
    trajectories: list[dict[str, Any]],
    nodes: list[dict[str, Any]],
    raw_bounds: dict[str, float],
    padded_bounds: dict[str, float],
    z_scale: float,
) -> str:
    map_name = str(trajectory_config.get("map") or node_config.get("map") or "unknown")
    plot_rect = (72.0, 88.0, float(width) - 420.0, float(height) - 170.0)
    legend_rect = (float(width) - 320.0, 88.0, 278.0, float(height) - 170.0)
    view = _static_3d_view(raw_bounds, padded_bounds, trajectories, nodes, plot_rect, z_scale)
    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<defs>",
        "<style>",
        "text{fill:#20252b}.title{font-weight:700}.subtitle{fill:#56616f}.grid3d{stroke:#d8dde3;stroke-width:1}.axis3d{stroke:#20252b;stroke-width:2.2}.bbox3d{fill:none;stroke:#8c98a8;stroke-width:1.8;stroke-dasharray:6 5}.node3d{fill:#f7f9fb;stroke:#222;stroke-width:2}.node-arrow3d{stroke:#222;stroke-width:2.2}.waypoint3d{fill:#fff;stroke-width:1.6}.legend-panel{fill:#ffffff;stroke:#d4dae2;stroke-width:1}.ground{fill:#fff;stroke:#c6ccd4}",
        "</style>",
        "</defs>",
        '<rect x="0" y="0" width="100%" height="100%" fill="#f7f6f1"/>',
        _svg_text(28, 34, "CARLA-Air trajectory spatial preview | static 3D", size=22, klass="title"),
        _svg_text(28, 58, f"Map={map_name} | isometric projection | z-scale={z_scale:g} | presentation/QC only", size=13, klass="subtitle"),
    ]

    ground = [
        {"x": raw_bounds["min_x"], "y": raw_bounds["min_y"], "z": raw_bounds["min_z"]},
        {"x": raw_bounds["max_x"], "y": raw_bounds["min_y"], "z": raw_bounds["min_z"]},
        {"x": raw_bounds["max_x"], "y": raw_bounds["max_y"], "z": raw_bounds["min_z"]},
        {"x": raw_bounds["min_x"], "y": raw_bounds["max_y"], "z": raw_bounds["min_z"]},
    ]
    ground_projected = [_project_3d(view, point) for point in ground]
    parts.append(f'<polygon class="ground" points="{_points_attr(ground_projected)}"/>')

    spacing = _nice_length(max(raw_bounds["max_x"] - raw_bounds["min_x"], raw_bounds["max_y"] - raw_bounds["min_y"]) / 5.0)
    gx = math.floor(raw_bounds["min_x"] / spacing) * spacing
    while gx <= raw_bounds["max_x"] + 1e-9:
        a = _project_3d(view, {"x": gx, "y": raw_bounds["min_y"], "z": raw_bounds["min_z"]})
        b = _project_3d(view, {"x": gx, "y": raw_bounds["max_y"], "z": raw_bounds["min_z"]})
        parts.append(f'<line class="grid3d" x1="{a[0]:.2f}" y1="{a[1]:.2f}" x2="{b[0]:.2f}" y2="{b[1]:.2f}"/>')
        gx += spacing
    gy = math.floor(raw_bounds["min_y"] / spacing) * spacing
    while gy <= raw_bounds["max_y"] + 1e-9:
        a = _project_3d(view, {"x": raw_bounds["min_x"], "y": gy, "z": raw_bounds["min_z"]})
        b = _project_3d(view, {"x": raw_bounds["max_x"], "y": gy, "z": raw_bounds["min_z"]})
        parts.append(f'<line class="grid3d" x1="{a[0]:.2f}" y1="{a[1]:.2f}" x2="{b[0]:.2f}" y2="{b[1]:.2f}"/>')
        gy += spacing

    corners = _bbox_corners(raw_bounds)
    projected_corners = [_project_3d(view, corner) for corner in corners]
    for a_idx, b_idx in _bbox_edges():
        a = projected_corners[a_idx]
        b = projected_corners[b_idx]
        parts.append(f'<line class="bbox3d" x1="{a[0]:.2f}" y1="{a[1]:.2f}" x2="{b[0]:.2f}" y2="{b[1]:.2f}"/>')

    origin = {"x": raw_bounds["min_x"], "y": raw_bounds["min_y"], "z": raw_bounds["min_z"]}
    axis_len = _nice_length(min(raw_bounds["max_x"] - raw_bounds["min_x"], raw_bounds["max_y"] - raw_bounds["min_y"]) * 0.25)
    axes = [
        ("X", "#b23a48", {"x": origin["x"] + axis_len, "y": origin["y"], "z": origin["z"]}),
        ("Y", "#2f7d4f", {"x": origin["x"], "y": origin["y"] + axis_len, "z": origin["z"]}),
        ("Z", "#2f5fb3", {"x": origin["x"], "y": origin["y"], "z": origin["z"] + axis_len}),
    ]
    o = _project_3d(view, origin)
    for label, color, endpoint in axes:
        e = _project_3d(view, endpoint)
        parts.append(f'<line class="axis3d" x1="{o[0]:.2f}" y1="{o[1]:.2f}" x2="{e[0]:.2f}" y2="{e[1]:.2f}" stroke="{color}"/>')
        parts.append(_svg_text(e[0] + 5, e[1] - 5, label, size=12, klass="title"))

    for trajectory in trajectories:
        color = str(trajectory["color"])
        class_name = str(trajectory["class_name"])
        points = [_project_3d(view, point) for point in trajectory["waypoints"]]
        parts.append(f'<g class="layer trajectory3d {class_name}">')
        parts.append(
            f'<polyline points="{_points_attr(points)}" fill="none" stroke="{color}" stroke-width="4" stroke-linejoin="round" stroke-linecap="round"/>'
        )
        for idx, (sx, sy) in enumerate(points):
            parts.append(f'<circle class="waypoint3d" cx="{sx:.2f}" cy="{sy:.2f}" r="4.4" stroke="{color}"/>')
            label = "S" if idx == 0 else "E" if idx == len(points) - 1 else str(idx + 1)
            parts.append(_svg_text(sx + 6, sy - 6, label, size=9))
        parts.append("</g>")

    for node in nodes:
        sx, sy = _project_3d(view, {"x": node["x"], "y": node["y"], "z": node["z"]})
        yaw = math.radians(float(node["yaw"]))
        tx, ty = _project_3d(
            view,
            {
                "x": float(node["x"]) + math.cos(yaw) * 8.0,
                "y": float(node["y"]) + math.sin(yaw) * 8.0,
                "z": float(node["z"]),
            },
        )
        parts.append(f'<line class="node-arrow3d" x1="{sx:.2f}" y1="{sy:.2f}" x2="{tx:.2f}" y2="{ty:.2f}"/>')
        parts.append(f'<circle class="node3d" cx="{sx:.2f}" cy="{sy:.2f}" r="7.0"/>')
        parts.append(_svg_text(sx + 9, sy - 9, str(node["node_id"]), size=11))

    lx, ly, lw, lh = legend_rect
    parts.append(f'<rect class="legend-panel" x="{lx:.2f}" y="{ly:.2f}" width="{lw:.2f}" height="{lh:.2f}" rx="4"/>')
    y_cursor = ly + 28.0
    parts.append(_svg_text(lx + 16, y_cursor, "3D Legend", size=14, klass="title"))
    y_cursor += 26.0
    for trajectory in trajectories:
        parts.append(
            f'<line x1="{lx + 16:.2f}" y1="{y_cursor - 4:.2f}" x2="{lx + 44:.2f}" y2="{y_cursor - 4:.2f}" '
            f'stroke="{trajectory["color"]}" stroke-width="4" stroke-linecap="round"/>'
        )
        parts.append(_svg_text(lx + 54, y_cursor, str(trajectory["trajectory_id"]), size=10))
        y_cursor += 16.0
        parts.append(
            _svg_text(lx + 54, y_cursor, f"wps={len(trajectory['waypoints'])} len={trajectory['path_length_m']:.1f}m", size=10, klass="subtitle")
        )
        y_cursor += 18.0
    roi_summary = _bounds_summary(raw_bounds)
    parts.append(_svg_text(lx + 16, y_cursor + 10, f"ROI: {roi_summary['width_m']:.1f} x {roi_summary['depth_m']:.1f} x {roi_summary['height_m']:.1f}m", size=10, klass="subtitle"))
    parts.append(_svg_text(lx + 16, y_cursor + 30, f"z-scale={z_scale:g}; {'true metric scale' if abs(z_scale - 1.0) < 1e-9 else 'vertical exaggeration'}", size=10, klass="subtitle"))
    parts.append(_svg_text(lx + 16, y_cursor + 50, "not dataset/annotation evidence", size=10, klass="subtitle"))
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def _draw_static_3d_png(
    *,
    path: Path,
    width: int,
    height: int,
    trajectory_config: dict[str, Any],
    node_config: dict[str, Any],
    trajectories: list[dict[str, Any]],
    nodes: list[dict[str, Any]],
    raw_bounds: dict[str, float],
    padded_bounds: dict[str, float],
    z_scale: float,
) -> None:
    from PIL import Image, ImageDraw, ImageFont

    image = Image.new("RGB", (width, height), (247, 246, 241))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    plot_rect = (72.0, 88.0, float(width) - 420.0, float(height) - 170.0)
    view = _static_3d_view(raw_bounds, padded_bounds, trajectories, nodes, plot_rect, z_scale)
    map_name = str(trajectory_config.get("map") or node_config.get("map") or "unknown")

    def text(x: float, y: float, value: str, fill: tuple[int, int, int] = (32, 37, 43)) -> None:
        draw.text((float(x), float(y)), value, fill=fill, font=font)

    text(28, 22, "CARLA-Air trajectory spatial preview | static 3D")
    text(28, 46, f"Map={map_name} | isometric projection | z-scale={z_scale:g} | presentation/QC only", (86, 97, 111))

    ground = [
        {"x": raw_bounds["min_x"], "y": raw_bounds["min_y"], "z": raw_bounds["min_z"]},
        {"x": raw_bounds["max_x"], "y": raw_bounds["min_y"], "z": raw_bounds["min_z"]},
        {"x": raw_bounds["max_x"], "y": raw_bounds["max_y"], "z": raw_bounds["min_z"]},
        {"x": raw_bounds["min_x"], "y": raw_bounds["max_y"], "z": raw_bounds["min_z"]},
    ]
    draw.polygon([_project_3d(view, point) for point in ground], fill=(255, 255, 255), outline=(198, 204, 212))

    spacing = _nice_length(max(raw_bounds["max_x"] - raw_bounds["min_x"], raw_bounds["max_y"] - raw_bounds["min_y"]) / 5.0)
    gx = math.floor(raw_bounds["min_x"] / spacing) * spacing
    while gx <= raw_bounds["max_x"] + 1e-9:
        a = _project_3d(view, {"x": gx, "y": raw_bounds["min_y"], "z": raw_bounds["min_z"]})
        b = _project_3d(view, {"x": gx, "y": raw_bounds["max_y"], "z": raw_bounds["min_z"]})
        draw.line((a, b), fill=(216, 221, 227), width=1)
        gx += spacing
    gy = math.floor(raw_bounds["min_y"] / spacing) * spacing
    while gy <= raw_bounds["max_y"] + 1e-9:
        a = _project_3d(view, {"x": raw_bounds["min_x"], "y": gy, "z": raw_bounds["min_z"]})
        b = _project_3d(view, {"x": raw_bounds["max_x"], "y": gy, "z": raw_bounds["min_z"]})
        draw.line((a, b), fill=(216, 221, 227), width=1)
        gy += spacing

    corners = _bbox_corners(raw_bounds)
    projected_corners = [_project_3d(view, corner) for corner in corners]
    for a_idx, b_idx in _bbox_edges():
        draw.line((projected_corners[a_idx], projected_corners[b_idx]), fill=(140, 152, 168), width=2)

    origin = {"x": raw_bounds["min_x"], "y": raw_bounds["min_y"], "z": raw_bounds["min_z"]}
    axis_len = _nice_length(min(raw_bounds["max_x"] - raw_bounds["min_x"], raw_bounds["max_y"] - raw_bounds["min_y"]) * 0.25)
    o = _project_3d(view, origin)
    for label, color, endpoint in [
        ("X", (178, 58, 72), {"x": origin["x"] + axis_len, "y": origin["y"], "z": origin["z"]}),
        ("Y", (47, 125, 79), {"x": origin["x"], "y": origin["y"] + axis_len, "z": origin["z"]}),
        ("Z", (47, 95, 179), {"x": origin["x"], "y": origin["y"], "z": origin["z"] + axis_len}),
    ]:
        e = _project_3d(view, endpoint)
        draw.line((o, e), fill=color, width=3)
        text(e[0] + 5, e[1] - 5, label, color)

    for trajectory in trajectories:
        color = _hex_to_rgb(str(trajectory["color"]))
        points = [_project_3d(view, point) for point in trajectory["waypoints"]]
        draw.line(points, fill=color, width=4)
        for idx, (px, py) in enumerate(points):
            draw.ellipse((px - 4, py - 4, px + 4, py + 4), fill=(255, 255, 255), outline=color, width=2)
            label = "S" if idx == 0 else "E" if idx == len(points) - 1 else str(idx + 1)
            text(px + 6, py - 10, label, color)

    for node in nodes:
        nx, ny = _project_3d(view, {"x": node["x"], "y": node["y"], "z": node["z"]})
        yaw = math.radians(float(node["yaw"]))
        tx, ty = _project_3d(view, {"x": float(node["x"]) + math.cos(yaw) * 8.0, "y": float(node["y"]) + math.sin(yaw) * 8.0, "z": float(node["z"])})
        draw.line((nx, ny, tx, ty), fill=(32, 37, 43), width=2)
        draw.ellipse((nx - 7, ny - 7, nx + 7, ny + 7), fill=(247, 249, 251), outline=(32, 37, 43), width=2)
        text(nx + 9, ny - 14, str(node["node_id"]))

    lx = float(width) - 320.0
    ly = 88.0
    draw.rectangle((lx, ly, lx + 278.0, float(height) - 82.0), fill=(255, 255, 255), outline=(212, 218, 226), width=1)
    y_cursor = ly + 18
    text(lx + 14, y_cursor, "3D Legend")
    y_cursor += 24
    for trajectory in trajectories:
        color = _hex_to_rgb(str(trajectory["color"]))
        draw.line((lx + 14, y_cursor, lx + 42, y_cursor), fill=color, width=4)
        text(lx + 52, y_cursor - 6, str(trajectory["trajectory_id"]))
        y_cursor += 16
        text(lx + 52, y_cursor - 6, f"wps={len(trajectory['waypoints'])} len={trajectory['path_length_m']:.1f}m", (86, 97, 111))
        y_cursor += 20
    roi_summary = _bounds_summary(raw_bounds)
    text(lx + 14, y_cursor, f"ROI: {roi_summary['width_m']:.1f} x {roi_summary['depth_m']:.1f} x {roi_summary['height_m']:.1f}m", (86, 97, 111))
    y_cursor += 18
    text(lx + 14, y_cursor, f"z-scale={z_scale:g}", (86, 97, 111))
    y_cursor += 18
    text(lx + 14, y_cursor, "not dataset/annotation evidence", (86, 97, 111))

    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def _interactive_3d_payload(
    *,
    trajectory_config: dict[str, Any],
    node_config: dict[str, Any],
    trajectories: list[dict[str, Any]],
    nodes: list[dict[str, Any]],
    raw_bounds: dict[str, float],
    z_scale: float,
) -> dict[str, Any]:
    map_name = str(trajectory_config.get("map") or node_config.get("map") or "unknown")
    return {
        "map_name": map_name,
        "z_scale_default": float(z_scale),
        "center": _center_from_bounds(raw_bounds),
        "bounds": {key: float(value) for key, value in raw_bounds.items()},
        "bounds_summary": _bounds_summary(raw_bounds),
        "trajectories": [
            {
                "trajectory_id": str(trajectory["trajectory_id"]),
                "color": str(trajectory["color"]),
                "path_length_m": float(trajectory["path_length_m"]),
                "waypoints": [
                    {"x": float(point["x"]), "y": float(point["y"]), "z": float(point["z"])}
                    for point in trajectory["waypoints"]
                ],
            }
            for trajectory in trajectories
        ],
        "nodes": [
            {
                "node_id": str(node["node_id"]),
                "x": float(node["x"]),
                "y": float(node["y"]),
                "z": float(node["z"]),
                "yaw": float(node["yaw"]),
            }
            for node in nodes
        ],
    }


def _build_interactive_3d_html(
    *,
    trajectory_config: dict[str, Any],
    node_config: dict[str, Any],
    trajectories: list[dict[str, Any]],
    nodes: list[dict[str, Any]],
    raw_bounds: dict[str, float],
    z_scale: float,
    three_js_relative_path: str,
) -> str:
    payload = _interactive_3d_payload(
        trajectory_config=trajectory_config,
        node_config=node_config,
        trajectories=trajectories,
        nodes=nodes,
        raw_bounds=raw_bounds,
        z_scale=z_scale,
    )
    data_json = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
    three_src = "./" + html.escape(three_js_relative_path.replace("\\", "/"), quote=True)
    template = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CARLA-Air trajectory spatial preview 3D</title>
<style>
body{margin:0;font-family:Arial,sans-serif;background:#f7f6f1;color:#20252b}
.toolbar{position:sticky;top:0;z-index:2;display:flex;gap:14px;align-items:center;flex-wrap:wrap;padding:10px 14px;background:#20252b;color:#fff}
.toolbar label{font-size:13px;white-space:nowrap}.toolbar input[type=range]{width:128px;vertical-align:middle}.toolbar .metric{color:#d8dde3}
.viewer{height:calc(100vh - 56px);min-height:560px;position:relative}.viewer canvas{display:block;width:100%;height:100%;touch-action:none}
.caption{position:absolute;left:14px;bottom:12px;padding:8px 10px;background:rgba(255,255,255,.88);border:1px solid #d4dae2;font-size:12px;line-height:1.45;max-width:min(560px,calc(100% - 28px))}
.caption strong{font-weight:700}.error{padding:18px;color:#9b1c31}
@media (max-width:720px){.viewer{height:calc(100vh - 112px);min-height:420px}.toolbar{gap:10px}.caption{font-size:11px}}
</style>
</head>
<body>
<div class="toolbar">
  <label><input type="checkbox" data-layer="trajectories" checked> trajectories</label>
  <label><input type="checkbox" data-layer="waypoints" checked> waypoints</label>
  <label><input type="checkbox" data-layer="nodes" checked> camera nodes</label>
  <label><input type="checkbox" data-layer="roi" checked> ROI</label>
  <label><input type="checkbox" data-layer="grid" checked> meter grid</label>
  <label>Z scale <input id="zScale" type="range" min="0.25" max="5" step="0.05"><span id="zScaleValue" class="metric"></span></label>
  <span class="metric">drag rotate | wheel zoom</span>
</div>
<div class="viewer">
  <canvas id="scene"></canvas>
  <div class="caption" id="caption"></div>
</div>
<script src="__THREE_SRC__"></script>
<script>
const DATA = __DATA_JSON__;
if (!window.THREE) {
  throw new Error("Three.js vendor did not load from ./vendor/three.min.js");
}

const canvas = document.getElementById("scene");
const renderer = new THREE.WebGLRenderer({canvas, antialias: true, preserveDrawingBuffer: true});
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
renderer.setClearColor(0xf7f6f1, 1);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0xf7f6f1);
const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 10000);

const center = DATA.center;
const bounds = DATA.bounds;
const summary = DATA.bounds_summary;
function toScene(point) {
  return new THREE.Vector3(point.x - center.x, point.z - center.z, point.y - center.y);
}
function niceLength(raw) {
  if (raw <= 0) return 1;
  const exponent = Math.floor(Math.log10(raw));
  const base = Math.pow(10, exponent);
  const fraction = raw / base;
  const nice = fraction <= 1.5 ? 1 : fraction <= 3 ? 2 : fraction <= 7 ? 5 : 10;
  return nice * base;
}
function addLine(group, a, b, material) {
  const geometry = new THREE.BufferGeometry().setFromPoints([a, b]);
  const line = new THREE.Line(geometry, material);
  group.add(line);
  return line;
}

const root = new THREE.Group();
const zRoot = new THREE.Group();
const gridGroup = new THREE.Group();
const trajectoryGroup = new THREE.Group();
const waypointGroup = new THREE.Group();
const nodeGroup = new THREE.Group();
const roiGroup = new THREE.Group();
const axisGroup = new THREE.Group();
root.add(gridGroup);
zRoot.add(trajectoryGroup, waypointGroup, nodeGroup, roiGroup, axisGroup);
root.add(zRoot);
scene.add(root);

const widthM = Math.max(summary.width_m, 1);
const depthM = Math.max(summary.depth_m, 1);
const heightM = Math.max(summary.height_m, 1);
const extent = Math.max(widthM, depthM, heightM * 3, 10);
const gridMaterial = new THREE.LineBasicMaterial({color: 0xd8dde3});
const spacing = niceLength(Math.max(widthM, depthM) / 5);
for (let x = Math.floor(bounds.min_x / spacing) * spacing; x <= bounds.max_x + 1e-9; x += spacing) {
  addLine(gridGroup, toScene({x, y: bounds.min_y, z: bounds.min_z}), toScene({x, y: bounds.max_y, z: bounds.min_z}), gridMaterial);
}
for (let y = Math.floor(bounds.min_y / spacing) * spacing; y <= bounds.max_y + 1e-9; y += spacing) {
  addLine(gridGroup, toScene({x: bounds.min_x, y, z: bounds.min_z}), toScene({x: bounds.max_x, y, z: bounds.min_z}), gridMaterial);
}

const axisLen = niceLength(Math.min(widthM, depthM) * 0.25);
const origin = toScene({x: bounds.min_x, y: bounds.min_y, z: bounds.min_z});
addLine(axisGroup, origin, toScene({x: bounds.min_x + axisLen, y: bounds.min_y, z: bounds.min_z}), new THREE.LineBasicMaterial({color: 0xb23a48}));
addLine(axisGroup, origin, toScene({x: bounds.min_x, y: bounds.min_y + axisLen, z: bounds.min_z}), new THREE.LineBasicMaterial({color: 0x2f7d4f}));
addLine(axisGroup, origin, toScene({x: bounds.min_x, y: bounds.min_y, z: bounds.min_z + axisLen}), new THREE.LineBasicMaterial({color: 0x2f5fb3}));

const waypointRadius = Math.max(extent * 0.008, 0.35);
const nodeRadius = Math.max(extent * 0.012, 0.55);
for (const trajectory of DATA.trajectories) {
  const color = new THREE.Color(trajectory.color);
  const points = trajectory.waypoints.map(toScene);
  const geometry = new THREE.BufferGeometry().setFromPoints(points);
  trajectoryGroup.add(new THREE.Line(geometry, new THREE.LineBasicMaterial({color, linewidth: 4})));
  const waypointMaterial = new THREE.MeshBasicMaterial({color: 0xffffff});
  const waypointEdge = new THREE.MeshBasicMaterial({color});
  for (const point of points) {
    const sphere = new THREE.Mesh(new THREE.SphereGeometry(waypointRadius, 16, 12), waypointMaterial);
    sphere.position.copy(point);
    waypointGroup.add(sphere);
    const ring = new THREE.Mesh(new THREE.SphereGeometry(waypointRadius * 1.08, 16, 12), waypointEdge);
    ring.position.copy(point);
    ring.scale.set(1, 1, 1);
    waypointGroup.add(ring);
  }
}

for (const node of DATA.nodes) {
  const position = toScene(node);
  const marker = new THREE.Mesh(
    new THREE.SphereGeometry(nodeRadius, 18, 14),
    new THREE.MeshBasicMaterial({color: 0xf7f9fb})
  );
  marker.position.copy(position);
  nodeGroup.add(marker);
  const yaw = node.yaw * Math.PI / 180;
  const direction = new THREE.Vector3(Math.cos(yaw), 0, Math.sin(yaw)).normalize();
  const arrow = new THREE.ArrowHelper(direction, position, Math.max(extent * 0.075, 6), 0x20252b, Math.max(extent * 0.018, 1.4), Math.max(extent * 0.012, 0.9));
  nodeGroup.add(arrow);
}

const boxWidth = bounds.max_x - bounds.min_x;
const boxHeight = bounds.max_z - bounds.min_z;
const boxDepth = bounds.max_y - bounds.min_y;
const boxGeometry = new THREE.BoxGeometry(boxWidth, Math.max(boxHeight, 0.01), boxDepth);
const roiMesh = new THREE.Mesh(
  boxGeometry,
  new THREE.MeshBasicMaterial({color: 0x8c98a8, transparent: true, opacity: 0.055, depthWrite: false})
);
roiMesh.position.set((bounds.min_x + bounds.max_x) / 2 - center.x, boxHeight / 2, (bounds.min_y + bounds.max_y) / 2 - center.y);
roiGroup.add(roiMesh);
const roiEdges = new THREE.LineSegments(new THREE.EdgesGeometry(boxGeometry), new THREE.LineBasicMaterial({color: 0x8c98a8}));
roiEdges.position.copy(roiMesh.position);
roiGroup.add(roiEdges);

const hemi = new THREE.HemisphereLight(0xffffff, 0xd8dde3, 0.8);
scene.add(hemi);
root.traverse((object) => {
  if (object.isLine || object.isLineSegments || object.isMesh) {
    object.frustumCulled = false;
  }
});

const layerGroups = {
  trajectories: trajectoryGroup,
  waypoints: waypointGroup,
  nodes: nodeGroup,
  roi: roiGroup,
  grid: gridGroup
};
document.querySelectorAll("[data-layer]").forEach((input) => {
  input.addEventListener("change", () => {
    layerGroups[input.dataset.layer].visible = input.checked;
    render();
  });
});

let theta = Math.PI * 0.78;
let phi = Math.PI * 0.34;
let radius = extent * 1.85;
let zScale = DATA.z_scale_default || 1;
const zInput = document.getElementById("zScale");
const zLabel = document.getElementById("zScaleValue");
zInput.value = String(zScale);
function setZScale(value) {
  zScale = Math.max(0.25, Math.min(5, Number(value) || 1));
  zRoot.scale.y = zScale;
  zLabel.textContent = " " + zScale.toFixed(2) + "x";
  updateCamera();
}
zInput.addEventListener("input", () => setZScale(zInput.value));

function updateCamera() {
  const target = new THREE.Vector3(0, heightM * zScale * 0.32, 0);
  const sinPhi = Math.sin(phi);
  camera.position.set(
    target.x + radius * sinPhi * Math.cos(theta),
    target.y + radius * Math.cos(phi),
    target.z + radius * sinPhi * Math.sin(theta)
  );
  camera.lookAt(target);
  render();
}

let dragging = false;
let lastX = 0;
let lastY = 0;
canvas.addEventListener("pointerdown", (event) => {
  dragging = true;
  lastX = event.clientX;
  lastY = event.clientY;
  canvas.setPointerCapture(event.pointerId);
});
canvas.addEventListener("pointermove", (event) => {
  if (!dragging) return;
  const dx = event.clientX - lastX;
  const dy = event.clientY - lastY;
  lastX = event.clientX;
  lastY = event.clientY;
  theta -= dx * 0.008;
  phi = Math.max(0.12, Math.min(Math.PI - 0.12, phi + dy * 0.008));
  updateCamera();
});
canvas.addEventListener("pointerup", () => { dragging = false; });
canvas.addEventListener("pointercancel", () => { dragging = false; });
canvas.addEventListener("wheel", (event) => {
  event.preventDefault();
  radius *= Math.exp(event.deltaY * 0.001);
  radius = Math.max(extent * 0.45, Math.min(extent * 8, radius));
  updateCamera();
}, {passive: false});

function resize() {
  const parent = canvas.parentElement;
  const width = Math.max(320, parent.clientWidth);
  const height = Math.max(320, parent.clientHeight);
  renderer.setSize(width, height, false);
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
  render();
}
let renderCount = 0;
function render() {
  renderCount += 1;
  window.__trajectoryPreviewRenderCount = renderCount;
  camera.updateMatrixWorld();
  scene.updateMatrixWorld(true);
  renderer.render(scene, camera);
}
window.addEventListener("resize", resize);
window.addEventListener("load", () => {
  resize();
  window.__trajectoryPreviewRender();
});

document.getElementById("caption").innerHTML =
  "<strong>" + DATA.map_name + "</strong> | ROI " +
  summary.width_m.toFixed(1) + " x " + summary.depth_m.toFixed(1) + " x " + summary.height_m.toFixed(1) +
  " m | default z-scale=" + DATA.z_scale_default.toFixed(2) +
  " | presentation/manual QC only; not annotation, mask, bbox, geometry, or benchmark evidence.";
window.__trajectoryPreviewDebug = {
  scene,
  camera,
  renderer,
  groups: {trajectoryGroup, waypointGroup, nodeGroup, roiGroup, gridGroup, axisGroup},
  root,
  zRoot
};
window.__trajectoryPreviewRender = render;
setZScale(zScale);
resize();
window.__trajectoryPreviewRender();
requestAnimationFrame(render);
setTimeout(render, 0);
setTimeout(render, 120);
renderer.setAnimationLoop(render);
</script>
</body>
</html>
"""
    return template.replace("__THREE_SRC__", three_src).replace("__DATA_JSON__", data_json)


def _build_html(svg: str, trajectories: list[dict[str, Any]]) -> str:
    controls = [
        '<label><input type="checkbox" data-toggle=".layer-nodes" checked> camera nodes</label>',
        '<label><input type="checkbox" data-toggle=".layer-roi" checked> local ROI / scale</label>',
        '<label><input type="checkbox" data-toggle=".layer-waypoints" checked> waypoints</label>',
        '<label><input type="checkbox" data-toggle=".layer-profile" checked> height profile</label>',
    ]
    for trajectory in trajectories:
        controls.append(
            f'<label><input type="checkbox" data-toggle=".{trajectory["class_name"]}" checked> '
            f'{html.escape(str(trajectory["trajectory_id"]))}</label>'
        )
    return (
        "<!doctype html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        "<title>CARLA-Air trajectory spatial preview</title>\n"
        "<style>\n"
        "body{margin:0;font-family:Arial,sans-serif;background:#f7f6f1;color:#20252b}"
        ".toolbar{position:sticky;top:0;z-index:2;display:flex;gap:14px;flex-wrap:wrap;padding:10px 14px;background:#20252b;color:#fff}"
        ".toolbar label{font-size:13px;white-space:nowrap}.stage{padding:14px}.stage svg{max-width:100%;height:auto;border:1px solid #d4dae2;background:#f7f6f1}"
        "</style>\n"
        "</head>\n"
        "<body>\n"
        f'<div class="toolbar">{"".join(controls)}</div>\n'
        f'<div class="stage">{svg}</div>\n'
        "<script>\n"
        "function setVisible(selector, visible){document.querySelectorAll(selector).forEach(function(el){el.style.display=visible?'':'none';});}\n"
        "document.querySelectorAll('[data-toggle]').forEach(function(input){input.addEventListener('change',function(){setVisible(input.dataset.toggle,input.checked);});});\n"
        "</script>\n"
        "</body>\n"
        "</html>\n"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export offline CARLA-Air trajectory/node spatial preview assets.")
    parser.add_argument("--trajectory-config", default=str(DEFAULT_TRAJECTORY_CONFIG), help="Trajectory config JSON.")
    parser.add_argument("--node-config", default=str(DEFAULT_NODE_CONFIG), help="Camera node config JSON.")
    parser.add_argument("--trajectory-id", action="append", help="Trajectory id to include. Repeat for multiple ids.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory.")
    parser.add_argument("--map-scale-mode", choices=["local", "runtime", "both"], default="local")
    parser.add_argument("--view-mode", choices=["2d", "3d", "both"], default="both")
    parser.add_argument("--z-scale", type=float, default=1.0, help="Static/initial 3D vertical scale; 1.0 preserves metric scale.")
    parser.add_argument("--three-js-source", default=str(DEFAULT_THREE_JS_SOURCE), help="Vendored Three.js module source.")
    parser.add_argument("--width", type=int, default=1400)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--carla-host", default="127.0.0.1")
    parser.add_argument("--carla-port", type=int, default=2000)
    parser.add_argument("--runtime-timeout", type=float, default=1.0)
    return parser


def _run(args: argparse.Namespace) -> int:
    trajectory_config_path = Path(args.trajectory_config).resolve()
    node_config_path = Path(args.node_config).resolve()
    output_dir = Path(args.output_dir).resolve()
    view_mode = str(args.view_mode)
    write_2d = view_mode in {"2d", "both"}
    write_3d = view_mode in {"3d", "both"}
    z_scale = float(args.z_scale)
    if z_scale <= 0:
        raise SystemExit("--z-scale must be > 0")
    three_js_source = Path(args.three_js_source).resolve()
    if write_3d and not three_js_source.exists():
        raise SystemExit(f"Three.js source not found: {three_js_source}")
    selected_ids = args.trajectory_id or list(DEFAULT_TRAJECTORY_IDS)
    trajectory_config, trajectories = _load_trajectories(trajectory_config_path, selected_ids)
    node_config, nodes = _load_nodes(node_config_path)
    selected_points = _all_points(trajectories, nodes)
    raw_bounds = _bounds(selected_points)
    padded_bounds = _pad_bounds(raw_bounds)
    runtime_extent: dict[str, Any] = {
        "available": False,
        "attempted": False,
        "reason": "not_requested",
    }
    if str(args.map_scale_mode) in {"runtime", "both"}:
        runtime_extent = _runtime_map_extent(str(args.carla_host), int(args.carla_port), float(args.runtime_timeout))

    scale_bar_m = _nice_length(
        min(
            float(padded_bounds["max_x"]) - float(padded_bounds["min_x"]),
            float(padded_bounds["max_y"]) - float(padded_bounds["min_y"]),
        )
        * 0.18
    )
    svg = ""
    if write_2d:
        svg, scale_bar_m = _build_svg(
            width=int(args.width),
            height=int(args.height),
            trajectory_config=trajectory_config,
            node_config=node_config,
            trajectories=trajectories,
            nodes=nodes,
            raw_bounds=raw_bounds,
            padded_bounds=padded_bounds,
            runtime_extent=runtime_extent,
        )

    static_3d_svg = ""
    if write_3d:
        static_3d_svg = _build_static_3d_svg(
            width=int(args.width),
            height=int(args.height),
            trajectory_config=trajectory_config,
            node_config=node_config,
            trajectories=trajectories,
            nodes=nodes,
            raw_bounds=raw_bounds,
            padded_bounds=padded_bounds,
            z_scale=z_scale,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    svg_path = output_dir / "trajectory_spatial_preview.svg"
    png_path = output_dir / "trajectory_spatial_preview.png"
    html_path = output_dir / "trajectory_spatial_preview.html"
    svg_3d_path = output_dir / "trajectory_spatial_preview_3d.svg"
    png_3d_path = output_dir / "trajectory_spatial_preview_3d.png"
    html_3d_path = output_dir / "trajectory_spatial_preview_3d.html"
    three_js_output_path = output_dir / "vendor" / "three.min.js"
    manifest_path = output_dir / "manifest.json"

    output_records: dict[str, Any] = {}
    output_groups: dict[str, Any] = {}
    view_modes: list[str] = []
    if write_2d:
        _write_text(svg_path, svg)
        _write_text(html_path, _build_html(svg, trajectories))
        _draw_png(
            path=png_path,
            width=int(args.width),
            height=int(args.height),
            trajectory_config=trajectory_config,
            node_config=node_config,
            trajectories=trajectories,
            nodes=nodes,
            raw_bounds=raw_bounds,
            padded_bounds=padded_bounds,
            runtime_extent=runtime_extent,
            scale_bar_m=scale_bar_m,
        )
        output_records.update(
            {
                "svg": {"path": str(svg_path), "size_bytes": svg_path.stat().st_size, "view_mode": "2d"},
                "png": {"path": str(png_path), "size_bytes": png_path.stat().st_size, "view_mode": "2d"},
                "html": {"path": str(html_path), "size_bytes": html_path.stat().st_size, "view_mode": "2d"},
            }
        )
        output_groups["2d"] = {
            "svg": str(svg_path),
            "png": str(png_path),
            "html": str(html_path),
        }
        view_modes.append("2d")

    three_js_vendor_record: dict[str, Any] | None = None
    if write_3d:
        _write_text(svg_3d_path, static_3d_svg)
        _draw_static_3d_png(
            path=png_3d_path,
            width=int(args.width),
            height=int(args.height),
            trajectory_config=trajectory_config,
            node_config=node_config,
            trajectories=trajectories,
            nodes=nodes,
            raw_bounds=raw_bounds,
            padded_bounds=padded_bounds,
            z_scale=z_scale,
        )
        _copy_file(three_js_source, three_js_output_path)
        three_js_vendor_record = {
            "source_path": str(three_js_source),
            "output_path": str(three_js_output_path),
            "relative_path": "vendor/three.min.js",
            "version": THREE_JS_VERSION,
            "sha256": _sha256(three_js_output_path),
            "size_bytes": three_js_output_path.stat().st_size,
        }
        _write_text(
            html_3d_path,
            _build_interactive_3d_html(
                trajectory_config=trajectory_config,
                node_config=node_config,
                trajectories=trajectories,
                nodes=nodes,
                raw_bounds=raw_bounds,
                z_scale=z_scale,
                three_js_relative_path="vendor/three.min.js",
            ),
        )
        output_records.update(
            {
                "static_3d_svg": {"path": str(svg_3d_path), "size_bytes": svg_3d_path.stat().st_size, "view_mode": "3d_static"},
                "static_3d_png": {"path": str(png_3d_path), "size_bytes": png_3d_path.stat().st_size, "view_mode": "3d_static"},
                "interactive_3d_html": {"path": str(html_3d_path), "size_bytes": html_3d_path.stat().st_size, "view_mode": "3d_interactive"},
                "three_js_vendor": three_js_vendor_record,
            }
        )
        output_groups["3d_static"] = {
            "svg": str(svg_3d_path),
            "png": str(png_3d_path),
        }
        output_groups["3d_interactive"] = {
            "html": str(html_3d_path),
            "three_js_vendor": str(three_js_output_path),
        }
        view_modes.extend(["3d_static", "3d_interactive"])

    manifest = {
        "schema_version": "carla_air_trajectory_spatial_preview_manifest_v2",
        "generated_at": _utc_now_iso(),
        "created_by": CREATED_BY,
        "presentation_only": True,
        "manual_qc_only": True,
        "not_formal_dataset_input": True,
        "not_annotation_evidence": True,
        "not_mask_evidence": True,
        "not_bbox_evidence": True,
        "not_geometry_evidence": True,
        "not_benchmark_evidence": True,
        "view_mode_cli": view_mode,
        "view_modes": view_modes,
        "z_scale_default": z_scale,
        "z_scale_is_true_metric": abs(z_scale - 1.0) < 1e-9,
        "three_js_vendor": three_js_vendor_record,
        "map_scale_mode": str(args.map_scale_mode),
        "map_extent_available": bool(runtime_extent.get("available")),
        "runtime_map_extent": runtime_extent,
        "inputs": {
            "trajectory_config": {
                "path": str(trajectory_config_path),
                "sha256": _sha256(trajectory_config_path),
                "schema_version": trajectory_config.get("schema_version"),
                "map": trajectory_config.get("map"),
            },
            "node_config": {
                "path": str(node_config_path),
                "sha256": _sha256(node_config_path),
                "schema_version": node_config.get("schema_version"),
                "map": node_config.get("map"),
                "carla_map_name": node_config.get("carla_map_name"),
            },
        },
        "trajectory_count": len(trajectories),
        "node_count": len(nodes),
        "selected_trajectory_ids": selected_ids,
        "trajectories": [_trajectory_summary(trajectory) for trajectory in trajectories],
        "nodes": [_node_summary(node) for node in nodes],
        "local_roi": {
            "bounds": _bounds_summary(raw_bounds),
            "padded_view_bounds": _bounds_summary(padded_bounds),
            "covers_all_selected_trajectory_waypoints_and_node_anchors": _covers(raw_bounds, selected_points),
            "scale_bar_m": scale_bar_m,
        },
        "output_groups": output_groups,
        "outputs": output_records,
        "notes": [
            "Offline spatial preview for presentation, manual QC, and waypoint tuning only.",
            "Polyline rendering matches the current runner's piecewise-linear waypoint interpolation semantics.",
            "3D static and interactive views are spatial QC aids; z-scale is recorded whenever it differs from 1.0.",
            "Do not use these assets as training input, annotation, mask, bbox, geometry, or benchmark evidence.",
        ],
    }
    output_records["manifest"] = {"path": str(manifest_path)}
    _write_json(manifest_path, manifest)
    output_records["manifest"]["size_bytes"] = manifest_path.stat().st_size
    _write_json(manifest_path, manifest)
    print(f"Wrote trajectory spatial preview to: {output_dir}")
    if write_2d:
        print(f"2D SVG: {svg_path}")
        print(f"2D PNG: {png_path}")
        print(f"2D HTML: {html_path}")
    if write_3d:
        print(f"3D SVG: {svg_3d_path}")
        print(f"3D PNG: {png_3d_path}")
        print(f"3D HTML: {html_3d_path}")
        print(f"Three.js vendor: {three_js_output_path}")
    print(f"Manifest: {manifest_path}")
    return 0


def main() -> None:
    raise SystemExit(_run(build_parser().parse_args()))


if __name__ == "__main__":
    main()
