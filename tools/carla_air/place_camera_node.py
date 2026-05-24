#!/usr/bin/env python3
"""Interactively place one CARLA-Air ground-to-air tri-camera node.

The tool previews only the active node's cam0/cam1/cam2 sensors. Saved node
poses are written to local runtime config and can be replayed by later capture
scripts.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import signal
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RIG = REPO_ROOT / "configs" / "camera_rigs" / "node_tri_cam_parallel_v1.json"
DEFAULT_OUTPUT = REPO_ROOT / "local" / "carla_air" / "camera_nodes" / "Town10HD_ground_to_air_nodes_v1.json"
PREVIEW_ROLE_PREFIX = "ground_to_air_node_preview"


@dataclass
class TransformSpec:
    x: float
    y: float
    z: float
    pitch: float
    yaw: float
    roll: float


@dataclass
class CameraSpec:
    cam_id: str
    image_size: tuple[int, int]
    k: list[list[float]]
    distortion_model: str
    dist: list[float]
    t_node_from_cam: list[list[float]]
    fovy_deg: float | None
    fov_x_deg: float
    rel_location: tuple[float, float, float]
    rel_rotation: tuple[float, float, float]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def _natural_camera_key(cam_id: str) -> tuple[str, int]:
    m = re.fullmatch(r"([A-Za-z_]+)(\d+)", cam_id)
    if not m:
        return cam_id, -1
    return m.group(1), int(m.group(2))


def _next_node_id(node_id: str) -> str:
    m = re.fullmatch(r"([A-Za-z_]*?)(\d+)", node_id)
    if not m:
        return f"{node_id}_next"
    prefix, digits = m.groups()
    return f"{prefix}{int(digits) + 1:0{len(digits)}d}"


def _prev_node_id(node_id: str) -> str | None:
    m = re.fullmatch(r"([A-Za-z_]*?)(\d+)", node_id)
    if not m:
        return None
    prefix, digits = m.groups()
    value = int(digits)
    if value <= 1:
        return None
    return f"{prefix}{value - 1:0{len(digits)}d}"


def _node_id_parts(node_id: str) -> tuple[str, int, int] | None:
    m = re.fullmatch(r"([A-Za-z_]*?)(\d+)", node_id)
    if not m:
        return None
    prefix, digits = m.groups()
    return prefix, int(digits), len(digits)


def _resume_next_node_id(output_path: Path, fallback_node_id: str) -> str:
    if not output_path.exists():
        return fallback_node_id
    data = _load_json(output_path)
    fallback_parts = _node_id_parts(fallback_node_id)
    prefix_filter = fallback_parts[0] if fallback_parts else None
    width = fallback_parts[2] if fallback_parts else 2
    best_prefix = prefix_filter or "node"
    best_value: int | None = None

    for node in data.get("nodes", []):
        node_id = str(node.get("node_id", ""))
        parts = _node_id_parts(node_id)
        if parts is None:
            continue
        prefix, value, node_width = parts
        if prefix_filter is not None and prefix != prefix_filter:
            continue
        if best_value is None or value > best_value:
            best_prefix = prefix
            best_value = value
            width = max(width, node_width)

    if best_value is None:
        return fallback_node_id
    return f"{best_prefix}{best_value + 1:0{width}d}"


def _transform_to_dict(tf: TransformSpec) -> dict[str, float]:
    return {
        "x": float(tf.x),
        "y": float(tf.y),
        "z": float(tf.z),
        "pitch": float(tf.pitch),
        "yaw": float(tf.yaw),
        "roll": float(tf.roll),
    }


def _dict_to_transform(data: dict[str, Any]) -> TransformSpec:
    return TransformSpec(
        x=float(data.get("x", 0.0)),
        y=float(data.get("y", 0.0)),
        z=float(data.get("z", 20.0)),
        pitch=float(data.get("pitch", 0.0)),
        yaw=float(data.get("yaw", 0.0)),
        roll=float(data.get("roll", 0.0)),
    )


def _rotation_matrix_from_carla(pitch: float, yaw: float, roll: float) -> list[list[float]]:
    p = math.radians(pitch)
    y = math.radians(yaw)
    r = math.radians(roll)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    cr, sr = math.cos(r), math.sin(r)

    forward = [cp * cy, cp * sy, sp]
    right = [cy * sp * sr - sy * cr, sy * sp * sr + cy * cr, -cp * sr]
    up = [-cy * sp * cr - sy * sr, -sy * sp * cr + cy * sr, cp * cr]
    return [
        [forward[0], right[0], up[0]],
        [forward[1], right[1], up[1]],
        [forward[2], right[2], up[2]],
    ]


def _carla_rotation_from_matrix(rm: list[list[float]]) -> tuple[float, float, float]:
    sp = max(-1.0, min(1.0, float(rm[2][0])))
    pitch = math.degrees(math.asin(sp))
    cp = math.cos(math.radians(pitch))
    if abs(cp) < 1e-6:
        yaw = math.degrees(math.atan2(-float(rm[0][1]), float(rm[1][1])))
        roll = 0.0
    else:
        yaw = math.degrees(math.atan2(float(rm[1][0]), float(rm[0][0])))
        roll = math.degrees(math.atan2(-float(rm[2][1]), float(rm[2][2])))
    return pitch, yaw, roll


def _matmul3(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def _matvec3(a: list[list[float]], v: list[float]) -> list[float]:
    return [sum(a[i][k] * v[k] for k in range(3)) for i in range(3)]


def _transpose3(a: list[list[float]]) -> list[list[float]]:
    return [[a[j][i] for j in range(3)] for i in range(3)]


def _fov_x_from_k(image_size: tuple[int, int], k: list[list[float]]) -> float:
    width = float(image_size[0])
    fx = float(k[0][0])
    if width <= 0 or fx <= 0:
        raise ValueError(f"Invalid image width/fx for FOV conversion: width={width}, fx={fx}")
    return math.degrees(2.0 * math.atan(width / (2.0 * fx)))


def _load_camera_specs(rig_path: Path) -> tuple[dict[str, Any], list[CameraSpec]]:
    rig = _load_json(rig_path)
    cameras = rig.get("cameras")
    if not isinstance(cameras, dict) or not cameras:
        raise SystemExit(f"Rig has no cameras: {rig_path}")

    # Rig node -> CARLA local node: rig x -> CARLA y, rig y -> CARLA x, rig z -> CARLA z.
    rig_node_to_carla_node = [
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    # CV camera -> CARLA camera actor local: x_right,y_down,z_forward -> y_right,-z_up,x_forward.
    cv_cam_to_carla_cam = [
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ]
    carla_cam_to_cv_cam = _transpose3(cv_cam_to_carla_cam)

    specs: list[CameraSpec] = []
    for cam_id in sorted(cameras, key=_natural_camera_key):
        entry = cameras[cam_id]
        image_size_raw = entry.get("image_size")
        k_raw = entry.get("K")
        t_raw = entry.get("T_node_from_cam")
        if not (
            isinstance(image_size_raw, list)
            and len(image_size_raw) == 2
            and isinstance(k_raw, list)
            and len(k_raw) == 3
            and isinstance(t_raw, list)
            and len(t_raw) == 4
        ):
            raise SystemExit(f"Invalid camera entry for {cam_id} in {rig_path}")

        image_size = (int(image_size_raw[0]), int(image_size_raw[1]))
        k = [[float(v) for v in row] for row in k_raw]
        t_node_from_cam = [[float(v) for v in row] for row in t_raw]
        r_rig_from_cv = [row[:3] for row in t_node_from_cam[:3]]
        t_rig = [row[3] for row in t_node_from_cam[:3]]
        r_carla_node_from_carla_cam = _matmul3(
            rig_node_to_carla_node,
            _matmul3(r_rig_from_cv, carla_cam_to_cv_cam),
        )
        t_carla_node = _matvec3(rig_node_to_carla_node, t_rig)
        rel_pitch, rel_yaw, rel_roll = _carla_rotation_from_matrix(r_carla_node_from_carla_cam)
        specs.append(
            CameraSpec(
                cam_id=cam_id,
                image_size=image_size,
                k=k,
                distortion_model=str(entry.get("distortion_model", "none")),
                dist=[float(v) for v in entry.get("dist", [])],
                t_node_from_cam=t_node_from_cam,
                fovy_deg=float(entry["fovy_deg"]) if entry.get("fovy_deg") is not None else None,
                fov_x_deg=_fov_x_from_k(image_size, k),
                rel_location=(float(t_carla_node[0]), float(t_carla_node[1]), float(t_carla_node[2])),
                rel_rotation=(float(rel_pitch), float(rel_yaw), float(rel_roll)),
            )
        )
    return rig, specs


def _compose_transform(anchor: TransformSpec, rel: CameraSpec) -> TransformSpec:
    r_anchor = _rotation_matrix_from_carla(anchor.pitch, anchor.yaw, anchor.roll)
    r_rel = _rotation_matrix_from_carla(*rel.rel_rotation)
    r_world = _matmul3(r_anchor, r_rel)
    t_rel = list(rel.rel_location)
    t_world_offset = _matvec3(r_anchor, t_rel)
    pitch, yaw, roll = _carla_rotation_from_matrix(r_world)
    return TransformSpec(
        x=anchor.x + t_world_offset[0],
        y=anchor.y + t_world_offset[1],
        z=anchor.z + t_world_offset[2],
        pitch=pitch,
        yaw=yaw,
        roll=roll,
    )


def _initial_output(rig: dict[str, Any], map_name: str, full_map_name: str) -> dict[str, Any]:
    return {
        "schema_version": "carla_air_ground_to_air_camera_nodes_v1",
        "map": map_name,
        "carla_map_name": full_map_name,
        "layout_id": rig.get("layout_id", "node_tri_cam_parallel_v1"),
        "coordinate_convention": {
            "world": "CARLA/UE: x forward, y right, z up; pitch/yaw/roll are CARLA Rotation degrees",
            "node_anchor": "CARLA transform for the node optical axis; cameras are fixed relative to this anchor",
            "source_rig_camera": "CV camera convention: x right, y down, z forward",
            "source_T_node_from_cam": "Maps CV camera-frame points into the source rig node frame",
            "rig_node_to_carla_node": "source rig x -> CARLA local y, source rig y -> CARLA local x, source rig z -> CARLA local z",
        },
        "nodes": [],
    }


def _find_node(data: dict[str, Any], node_id: str) -> dict[str, Any] | None:
    for node in data.get("nodes", []):
        if node.get("node_id") == node_id:
            return node
    return None


def _node_entry(
    *,
    node_id: str,
    anchor: TransformSpec,
    camera_specs: list[CameraSpec],
    map_name: str,
    full_map_name: str,
) -> dict[str, Any]:
    cameras: dict[str, Any] = {}
    for spec in camera_specs:
        world_tf = _compose_transform(anchor, spec)
        cameras[spec.cam_id] = {
            "camera_id": spec.cam_id,
            "camera_name": f"{node_id}_{spec.cam_id}",
            "image_size": [int(spec.image_size[0]), int(spec.image_size[1])],
            "K": spec.k,
            "distortion_model": spec.distortion_model,
            "dist": spec.dist,
            "T_node_from_cam": spec.t_node_from_cam,
            "fovy_deg": spec.fovy_deg,
            "fov_x_deg": spec.fov_x_deg,
            "carla_relative_transform": {
                "x": spec.rel_location[0],
                "y": spec.rel_location[1],
                "z": spec.rel_location[2],
                "pitch": spec.rel_rotation[0],
                "yaw": spec.rel_rotation[1],
                "roll": spec.rel_rotation[2],
            },
            "carla_world_transform": _transform_to_dict(world_tf),
        }
    return {
        "node_id": node_id,
        "map": map_name,
        "carla_map_name": full_map_name,
        "anchor_transform": _transform_to_dict(anchor),
        "camera_order": [spec.cam_id for spec in camera_specs],
        "cameras": cameras,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def _save_node(
    *,
    output_path: Path,
    rig: dict[str, Any],
    node_id: str,
    anchor: TransformSpec,
    camera_specs: list[CameraSpec],
    map_name: str,
    full_map_name: str,
) -> None:
    if output_path.exists():
        data = _load_json(output_path)
    else:
        data = _initial_output(rig, map_name, full_map_name)
    data.setdefault("nodes", [])
    data["map"] = map_name
    data["carla_map_name"] = full_map_name
    data["layout_id"] = rig.get("layout_id", data.get("layout_id", "node_tri_cam_parallel_v1"))

    replacement = _node_entry(
        node_id=node_id,
        anchor=anchor,
        camera_specs=camera_specs,
        map_name=map_name,
        full_map_name=full_map_name,
    )
    nodes = [node for node in data["nodes"] if node.get("node_id") != node_id]
    nodes.append(replacement)
    nodes.sort(key=lambda n: _natural_camera_key(str(n.get("node_id", ""))))
    data["nodes"] = nodes
    _write_json(output_path, data)
    print(f"Saved {node_id}: {output_path}")


def _load_saved_anchor(output_path: Path, node_id: str) -> TransformSpec | None:
    if not output_path.exists():
        return None
    node = _find_node(_load_json(output_path), node_id)
    if not node:
        return None
    anchor = node.get("anchor_transform")
    if not isinstance(anchor, dict):
        return None
    return _dict_to_transform(anchor)


def _carla_transform(carla: Any, tf: TransformSpec) -> Any:
    return carla.Transform(
        carla.Location(x=tf.x, y=tf.y, z=tf.z),
        carla.Rotation(pitch=tf.pitch, yaw=tf.yaw, roll=tf.roll),
    )


def _transform_from_carla_tf(tf: Any) -> TransformSpec:
    return TransformSpec(
        x=float(tf.location.x),
        y=float(tf.location.y),
        z=float(tf.location.z),
        pitch=float(tf.rotation.pitch),
        yaw=float(tf.rotation.yaw),
        roll=float(tf.rotation.roll),
    )


def _spawn_preview_sensors(carla: Any, world: Any, specs: list[CameraSpec], anchor: TransformSpec) -> list[Any]:
    bp_lib = world.get_blueprint_library()
    sensors = []
    for spec in specs:
        bp = bp_lib.find("sensor.camera.rgb")
        bp.set_attribute("image_size_x", str(spec.image_size[0]))
        bp.set_attribute("image_size_y", str(spec.image_size[1]))
        bp.set_attribute("fov", f"{spec.fov_x_deg:.8f}")
        bp.set_attribute("sensor_tick", "0.033333")
        if bp.has_attribute("role_name"):
            bp.set_attribute("role_name", f"{PREVIEW_ROLE_PREFIX}.{spec.cam_id}")
        sensor = world.spawn_actor(bp, _carla_transform(carla, _compose_transform(anchor, spec)))
        sensors.append(sensor)
    return sensors


def _destroy_sensors(sensors: list[Any]) -> None:
    for sensor in sensors:
        try:
            sensor.stop()
        except Exception:
            pass
    for sensor in sensors:
        try:
            sensor.destroy()
        except Exception:
            pass
    sensors.clear()


def _cleanup_stale_preview_sensors(world: Any) -> int:
    stale = []
    for actor in world.get_actors().filter("sensor.camera.rgb"):
        role_name = str(actor.attributes.get("role_name", ""))
        if role_name.startswith(PREVIEW_ROLE_PREFIX):
            stale.append(actor)
    for actor in stale:
        try:
            actor.stop()
        except Exception:
            pass
    for actor in stale:
        try:
            actor.destroy()
        except Exception:
            pass
    return len(stale)


def _render_panel(pygame: Any, surface: Any, frame: Any, rect: Any, label: str, font: Any) -> None:
    if frame is None:
        pygame.draw.rect(surface, (20, 24, 28), rect)
        pygame.draw.rect(surface, (70, 80, 90), rect, 1)
        text = font.render(f"{label} waiting for frame", True, (230, 230, 230))
        surface.blit(text, (rect.x + 12, rect.y + 12))
        return

    img_h, img_w = frame.shape[:2]
    img_surface = pygame.image.frombuffer(frame.tobytes(), (img_w, img_h), "RGB")
    if img_w != rect.w or img_h != rect.h:
        img_surface = pygame.transform.smoothscale(img_surface, (rect.w, rect.h))
    surface.blit(img_surface, (rect.x, rect.y))
    pygame.draw.rect(surface, (255, 255, 255), rect, 1)
    text = font.render(label, True, (255, 255, 255))
    bg = pygame.Surface((text.get_width() + 10, text.get_height() + 6), pygame.SRCALPHA)
    bg.fill((0, 0, 0, 150))
    surface.blit(bg, (rect.x + 6, rect.y + 6))
    surface.blit(text, (rect.x + 11, rect.y + 9))


def _print_pose(node_id: str, anchor: TransformSpec, specs: list[CameraSpec]) -> None:
    print(json.dumps({"node_id": node_id, "anchor_transform": _transform_to_dict(anchor)}, indent=2))
    for spec in specs:
        print(json.dumps({spec.cam_id: _transform_to_dict(_compose_transform(anchor, spec))}, indent=2))


def _is_port_open(host: str, port: int, timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _wait_for_carla_port(host: str, port: int, wait_seconds: float) -> None:
    if wait_seconds <= 0:
        return
    deadline = time.monotonic() + wait_seconds
    while time.monotonic() < deadline:
        if _is_port_open(host, port):
            return
        time.sleep(1.0)


def _connection_help(host: str, port: int) -> str:
    return (
        f"Cannot connect to CARLA at {host}:{port}.\n\n"
        "Start CARLA-Air first in another terminal, for example:\n"
        "  cd /home/grasp/data/3d-reid/local/carla_air/simulators/CarlaAir-v0.1.7\n"
        "  conda activate carlaAir\n"
        "  ./CarlaAir.sh Town10HD --res 1280x720 --quality Low --fg\n\n"
        "Then run this placement tool from the repo root:\n"
        "  cd /home/grasp/data/3d-reid\n"
        "  conda activate carlaAir\n"
        "  python tools/carla_air/place_camera_node.py\n\n"
        "If CARLA-Air is already starting, pass --wait-seconds 120."
    )


def _run_interactive(args: argparse.Namespace) -> int:
    import numpy as np
    import pygame
    import carla

    rig, camera_specs = _load_camera_specs(Path(args.rig).resolve())
    if len(camera_specs) != 3:
        raise SystemExit(f"Expected exactly 3 cameras in rig, got {len(camera_specs)}")

    host = str(args.host)
    port = int(args.port)
    _wait_for_carla_port(host, port, float(args.wait_seconds))
    if not _is_port_open(host, port):
        raise SystemExit(_connection_help(host, port))

    client = carla.Client(host, port)
    client.set_timeout(float(args.timeout))
    try:
        world = client.get_world()
    except RuntimeError as exc:
        raise SystemExit(_connection_help(host, port)) from exc
    stale_count = _cleanup_stale_preview_sensors(world)
    if stale_count:
        print(f"Cleaned up {stale_count} stale preview sensor(s)")
    full_map_name = str(world.get_map().name)
    map_name = full_map_name.split("/")[-1]

    output_path = Path(args.output).resolve()
    node_id = str(args.node_id)
    if bool(args.resume_next):
        resumed_node_id = _resume_next_node_id(output_path, node_id)
        if resumed_node_id != node_id:
            print(f"Resume next: {node_id} -> {resumed_node_id}")
        else:
            print(f"Resume next: no saved sequential node found, using {node_id}")
        node_id = resumed_node_id

    saved_anchor = _load_saved_anchor(output_path, node_id)
    if saved_anchor is not None:
        anchor = saved_anchor
        print(f"Loaded existing {node_id} from {output_path}")
    else:
        anchor = _transform_from_carla_tf(world.get_spectator().get_transform())
        print(f"Initialized {node_id} from spectator transform")

    frames: dict[str, Any] = {spec.cam_id: None for spec in camera_specs}
    sensors: list[Any] = []

    def respawn_sensors() -> None:
        nonlocal sensors, frames
        _destroy_sensors(sensors)
        frames = {spec.cam_id: None for spec in camera_specs}
        sensors = _spawn_preview_sensors(carla, world, camera_specs, anchor)
        for spec, sensor in zip(camera_specs, sensors):
            def on_image(image: Any, cam_id: str = spec.cam_id) -> None:
                arr = np.frombuffer(image.raw_data, dtype=np.uint8)
                arr = arr.reshape((image.height, image.width, 4))[:, :, :3][:, :, ::-1]
                frames[cam_id] = arr.copy()

            sensor.listen(on_image)
        print(f"Previewing {node_id}: {', '.join(spec.cam_id for spec in camera_specs)}")

    def save_current() -> None:
        _save_node(
            output_path=output_path,
            rig=rig,
            node_id=node_id,
            anchor=anchor,
            camera_specs=camera_specs,
            map_name=map_name,
            full_map_name=full_map_name,
        )

    def switch_node(next_id: str) -> None:
        nonlocal node_id, anchor
        save_current()
        node_id = next_id
        saved = _load_saved_anchor(output_path, node_id)
        if saved is not None:
            anchor = saved
            print(f"Loaded existing {node_id}")
        else:
            print(f"Created {node_id} from current pose")
        respawn_sensors()

    pygame.init()
    panel_w = int(args.panel_width)
    panel_h = int(args.panel_height)
    hud_h = 64
    display = pygame.display.set_mode((panel_w * 3, panel_h + hud_h))
    pygame.display.set_caption("CARLA-Air Camera Node Placement")
    pygame.event.set_grab(False)
    pygame.mouse.set_visible(True)
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 14)

    speed = float(args.speed)
    look_speed = float(args.look_speed)
    running = True
    last_update = 0.0

    def request_stop(signum: int, _frame: Any) -> None:
        nonlocal running
        running = False
        print(f"Received signal {signum}; cleaning up preview sensors...", flush=True)

    previous_sigint = signal.getsignal(signal.SIGINT)
    previous_sigterm = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    print(
        "Controls: W/S forward/back, A/D left/right, Q/E down/up, arrow keys yaw/pitch, "
        "Z/X roll, [/ ] speed, Space/Ctrl+S/F5 save, N next, B previous, P print, ESC quit"
    )

    try:
        respawn_sensors()
        while running:
            dt = max(0.001, clock.tick(30) / 1000.0)
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    running = False
                elif ev.type == pygame.KEYDOWN:
                    mods = pygame.key.get_mods()
                    if ev.key == pygame.K_ESCAPE:
                        running = False
                    elif ev.key in (pygame.K_SPACE, pygame.K_F5) or (ev.key == pygame.K_s and (mods & pygame.KMOD_CTRL)):
                        save_current()
                    elif ev.key == pygame.K_n:
                        switch_node(_next_node_id(node_id))
                    elif ev.key == pygame.K_b:
                        prev_id = _prev_node_id(node_id)
                        if prev_id is None:
                            print(f"No previous node before {node_id}")
                        else:
                            switch_node(prev_id)
                    elif ev.key == pygame.K_p:
                        _print_pose(node_id, anchor, camera_specs)
                    elif ev.key == pygame.K_LEFTBRACKET:
                        speed = max(0.1, round(speed / 1.5, 3))
                        print(f"Speed: {speed}")
                    elif ev.key == pygame.K_RIGHTBRACKET:
                        speed = min(100.0, round(speed * 1.5, 3))
                        print(f"Speed: {speed}")
            keys = pygame.key.get_pressed()
            fwd = float(keys[pygame.K_w]) - float(keys[pygame.K_s])
            right = float(keys[pygame.K_d]) - float(keys[pygame.K_a])
            up = float(keys[pygame.K_e]) - float(keys[pygame.K_q])
            roll_delta = float(keys[pygame.K_x]) - float(keys[pygame.K_z])
            yaw_delta = float(keys[pygame.K_RIGHT]) - float(keys[pygame.K_LEFT])
            pitch_delta = float(keys[pygame.K_UP]) - float(keys[pygame.K_DOWN])
            anchor.yaw += yaw_delta * look_speed * dt
            anchor.pitch = max(-89.0, min(89.0, anchor.pitch + pitch_delta * look_speed * dt))

            rm = _rotation_matrix_from_carla(anchor.pitch, anchor.yaw, anchor.roll)
            forward_vec = [rm[0][0], rm[1][0], rm[2][0]]
            right_vec = [rm[0][1], rm[1][1], rm[2][1]]
            anchor.x += (fwd * forward_vec[0] + right * right_vec[0]) * speed * dt
            anchor.y += (fwd * forward_vec[1] + right * right_vec[1]) * speed * dt
            anchor.z += (fwd * forward_vec[2] + right * right_vec[2]) * speed * dt + up * speed * dt
            anchor.roll += roll_delta * speed * 10.0 * dt

            now = time.monotonic()
            if now - last_update > 1.0 / 30.0:
                for spec, sensor in zip(camera_specs, sensors):
                    sensor.set_transform(_carla_transform(carla, _compose_transform(anchor, spec)))
                world.get_spectator().set_transform(_carla_transform(carla, anchor))
                last_update = now

            try:
                world.tick()
            except Exception:
                pass

            display.fill((18, 20, 24))
            for i, spec in enumerate(camera_specs):
                rect = pygame.Rect(i * panel_w, 0, panel_w, panel_h)
                _render_panel(
                    pygame,
                    display,
                    frames.get(spec.cam_id),
                    rect,
                    f"{node_id}/{spec.cam_id} fov={spec.fov_x_deg:.1f}",
                    font,
                )

            hud_lines = [
                f"{node_id}  x={anchor.x:.2f} y={anchor.y:.2f} z={anchor.z:.2f} "
                f"pitch={anchor.pitch:.1f} yaw={anchor.yaw:.1f} roll={anchor.roll:.1f}  speed={speed:.2f}",
                "W/S A/D Q/E move | arrows yaw/pitch | Z/X roll | Space save | N next | B previous | P print | ESC quit",
            ]
            for j, line in enumerate(hud_lines):
                text = font.render(line, True, (230, 230, 230))
                display.blit(text, (10, panel_h + 8 + j * 22))
            pygame.display.flip()
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)
        _destroy_sensors(sensors)
        pygame.quit()
        print("Preview sensors cleaned up.", flush=True)

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Interactively place one CARLA-Air tri-camera node and save reproducible node poses.",
    )
    parser.add_argument("--host", default="localhost", help="CARLA host. Default: localhost")
    parser.add_argument("--port", type=int, default=2000, help="CARLA RPC port. Default: 2000")
    parser.add_argument("--timeout", type=float, default=15.0, help="CARLA client timeout seconds. Default: 15")
    parser.add_argument("--wait-seconds", type=float, default=0.0, help="Wait for the CARLA port before connecting. Default: 0")
    parser.add_argument("--rig", default=str(DEFAULT_RIG), help=f"Camera rig layout JSON. Default: {DEFAULT_RIG}")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help=f"Saved node placement JSON. Default: {DEFAULT_OUTPUT}")
    parser.add_argument("--node-id", default="node01", help="Initial active node id. Default: node01")
    parser.add_argument("--resume-next", action="store_true", help="Start from the next sequential node after the largest saved node matching --node-id prefix.")
    parser.add_argument("--panel-width", type=int, default=640, help="Preview panel width per camera. Default: 640")
    parser.add_argument("--panel-height", type=int, default=360, help="Preview panel height per camera. Default: 360")
    parser.add_argument("--speed", type=float, default=5.0, help="Initial movement speed in meters/sec. Default: 5")
    parser.add_argument("--look-speed", type=float, default=60.0, help="Keyboard look speed in degrees/sec. Default: 60")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return _run_interactive(args)


if __name__ == "__main__":
    raise SystemExit(main())
