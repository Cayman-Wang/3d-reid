#!/usr/bin/env python3
"""CARLA-Air smoke capture for saved ground-to-air camera nodes.

The default sync mode temporarily drives CARLA synchronous_mode and writes
groups where cam0/cam1/cam2 share the world.tick() frame. Passive mode remains
available for debugging sensor reception without modifying world settings.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import signal
import socket
import struct
import sys
import time
import zlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "local" / "carla_air" / "camera_nodes" / "Town10HD_ground_to_air_nodes_v1.json"
DEFAULT_OUT_ROOT = REPO_ROOT / "local" / "carla_air" / "captures"
CAPTURE_ROLE_PREFIX = "ground_to_air_capture_smoke"


@dataclass(frozen=True)
class TransformSpec:
    x: float
    y: float
    z: float
    pitch: float
    yaw: float
    roll: float


@dataclass(frozen=True)
class CameraConfig:
    cam_id: str
    image_size: tuple[int, int]
    k: list[list[float]]
    distortion_model: str
    dist: list[float]
    t_node_from_cam: list[list[float]]
    fov_x_deg: float
    fovy_deg: float | None
    carla_world_transform: TransformSpec
    raw_entry: dict[str, Any]


@dataclass(frozen=True)
class NodeConfig:
    node_id: str
    camera_order: list[str]
    cameras: dict[str, CameraConfig]
    raw_entry: dict[str, Any]


@dataclass(frozen=True)
class FramePacket:
    frame: int
    timestamp: float
    platform_timestamp: float
    width: int
    height: int
    raw_data: bytes


@dataclass
class NodeCaptureState:
    node: NodeConfig
    node_dir: Path
    expected_cams: set[str]
    frame_buffers: dict[int, dict[str, FramePacket]] = field(default_factory=dict)
    written_frames: set[int] = field(default_factory=set)
    discarded_frames: set[int] = field(default_factory=set)
    incomplete_dropped_frames: int = 0
    images_written: int = 0
    groups_written: int = 0
    first_frame: int | None = None
    last_frame: int | None = None
    csv_file: Any = None
    csv_writer: Any = None


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _make_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _path_basename(value: str) -> str:
    return value.replace("\\", "/").rstrip("/").split("/")[-1]


def _configured_map_basenames(config_data: dict[str, Any]) -> list[str]:
    basenames: list[str] = []
    for key in ("carla_map_name", "map"):
        value = config_data.get(key)
        if value is None:
            continue
        basename = _path_basename(str(value))
        if basename and basename not in basenames:
            basenames.append(basename)
    return basenames


def _transform_from_dict(data: dict[str, Any]) -> TransformSpec:
    return TransformSpec(
        x=float(data["x"]),
        y=float(data["y"]),
        z=float(data["z"]),
        pitch=float(data["pitch"]),
        yaw=float(data["yaw"]),
        roll=float(data["roll"]),
    )


def _load_capture_config(path: Path) -> tuple[dict[str, Any], dict[str, NodeConfig]]:
    data = _load_json(path)
    nodes: dict[str, NodeConfig] = {}
    for node_entry in data.get("nodes", []):
        node_id = str(node_entry.get("node_id", ""))
        if not node_id:
            continue
        camera_order = [str(v) for v in node_entry.get("camera_order", [])]
        camera_entries = node_entry.get("cameras", {})
        if not isinstance(camera_entries, dict):
            raise SystemExit(f"Invalid cameras entry for {node_id} in {path}")
        if not camera_order:
            camera_order = sorted(camera_entries)

        cameras: dict[str, CameraConfig] = {}
        for cam_id in camera_order:
            raw_cam = camera_entries.get(cam_id)
            if not isinstance(raw_cam, dict):
                raise SystemExit(f"Missing camera {node_id}/{cam_id} in {path}")
            world_tf = raw_cam.get("carla_world_transform")
            if not isinstance(world_tf, dict):
                raise SystemExit(f"Missing carla_world_transform for {node_id}/{cam_id} in {path}")
            image_size_raw = raw_cam.get("image_size")
            if not isinstance(image_size_raw, list) or len(image_size_raw) != 2:
                raise SystemExit(f"Invalid image_size for {node_id}/{cam_id} in {path}")

            cameras[cam_id] = CameraConfig(
                cam_id=cam_id,
                image_size=(int(image_size_raw[0]), int(image_size_raw[1])),
                k=[[float(v) for v in row] for row in raw_cam.get("K", [])],
                distortion_model=str(raw_cam.get("distortion_model", "none")),
                dist=[float(v) for v in raw_cam.get("dist", [])],
                t_node_from_cam=[[float(v) for v in row] for row in raw_cam.get("T_node_from_cam", [])],
                fov_x_deg=float(raw_cam["fov_x_deg"]),
                fovy_deg=float(raw_cam["fovy_deg"]) if raw_cam.get("fovy_deg") is not None else None,
                carla_world_transform=_transform_from_dict(world_tf),
                raw_entry=raw_cam,
            )
        nodes[node_id] = NodeConfig(
            node_id=node_id,
            camera_order=camera_order,
            cameras=cameras,
            raw_entry=node_entry,
        )
    if not nodes:
        raise SystemExit(f"No nodes found in {path}")
    return data, nodes


def _parse_node_args(raw_nodes: list[str] | None, *, all_nodes: bool, available: dict[str, NodeConfig]) -> list[str]:
    if all_nodes:
        if raw_nodes:
            raise SystemExit("Use either --all-nodes or --nodes, not both.")
        return sorted(available)

    tokens = raw_nodes or ["node01"]
    node_ids: list[str] = []
    for token in tokens:
        node_ids.extend(part.strip() for part in token.split(",") if part.strip())
    missing = [node_id for node_id in node_ids if node_id not in available]
    if missing:
        raise SystemExit(f"Node(s) not found in config: {', '.join(missing)}")
    return node_ids


def _carla_transform(carla: Any, tf: TransformSpec) -> Any:
    return carla.Transform(
        carla.Location(x=tf.x, y=tf.y, z=tf.z),
        carla.Rotation(pitch=tf.pitch, yaw=tf.yaw, roll=tf.roll),
    )


def _is_port_open(host: str, port: int, timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _wait_for_carla_port(host: str, port: int, wait_seconds: float) -> None:
    deadline = time.monotonic() + max(0.0, wait_seconds)
    while time.monotonic() < deadline:
        if _is_port_open(host, port):
            return
        time.sleep(1.0)


def _connection_help(host: str, port: int) -> str:
    return (
        f"Cannot connect to CARLA-Air at {host}:{port}.\n\n"
        "Start CARLA-Air in another terminal first, for example:\n"
        "  cd /home/grasp/data/3d-reid/local/carla_air/simulators/CarlaAir-v0.1.7\n"
        "  ./CarlaAir.sh Town10HD --res 1280x720 --quality Low --fg\n\n"
        "Then run this capture tool from the repo root with the carlaAir Python:\n"
        "  PYTHONNOUSERSITE=1 /home/grasp/miniconda3/envs/carlaAir/bin/python "
        "tools/carla_air/capture_camera_nodes_smoke.py --nodes node01 --seconds 10 --fps 10 --wait-seconds 120"
    )


def _cleanup_stale_capture_sensors(world: Any) -> int:
    stale = []
    for actor in world.get_actors().filter("sensor.camera.rgb"):
        role_name = str(actor.attributes.get("role_name", ""))
        if role_name.startswith(CAPTURE_ROLE_PREFIX):
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


def _sensor_failure_record(sensor: Any, phase: str, exc: Exception) -> dict[str, Any]:
    try:
        actor_id = int(sensor.id)
    except Exception:
        actor_id = None
    try:
        role_name = str(sensor.attributes.get("role_name", ""))
    except Exception:
        role_name = ""
    return {
        "phase": phase,
        "actor_id": actor_id,
        "role_name": role_name,
        "error": f"{type(exc).__name__}: {exc}",
    }


def _stop_sensors(sensors: list[Any]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for sensor in sensors:
        try:
            sensor.stop()
        except Exception as exc:
            failures.append(_sensor_failure_record(sensor, "stop", exc))
    return failures


def _destroy_sensors(sensors: list[Any]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for sensor in sensors:
        try:
            sensor.destroy()
        except Exception as exc:
            failures.append(_sensor_failure_record(sensor, "destroy", exc))
    sensors.clear()
    return failures


def _png_chunk(tag: bytes, data: bytes) -> bytes:
    return struct.pack("!I", len(data)) + tag + data + struct.pack("!I", zlib.crc32(tag + data) & 0xFFFFFFFF)


def _write_png_rgb(path: Path, rgb: Any) -> None:
    height, width = rgb.shape[:2]
    rgb = rgb.reshape((height, width, 3))
    scanlines = b"".join(b"\x00" + rgb[row].tobytes() for row in range(height))
    png = b"".join(
        [
            b"\x89PNG\r\n\x1a\n",
            _png_chunk(b"IHDR", struct.pack("!IIBBBBB", width, height, 8, 2, 0, 0, 0)),
            _png_chunk(b"IDAT", zlib.compress(scanlines, level=3)),
            _png_chunk(b"IEND", b""),
        ]
    )
    path.write_bytes(png)


def _packet_to_rgb(packet: FramePacket) -> Any:
    import numpy as np

    bgra = np.frombuffer(packet.raw_data, dtype=np.uint8).reshape((packet.height, packet.width, 4))
    return np.ascontiguousarray(bgra[:, :, :3][:, :, ::-1])


def _camera_calib_entry(camera: CameraConfig) -> dict[str, Any]:
    return {
        "camera_id": camera.cam_id,
        "K": camera.k,
        "distortion_model": camera.distortion_model,
        "dist": camera.dist,
        "image_size": [camera.image_size[0], camera.image_size[1]],
        "T_node_from_cam": camera.t_node_from_cam,
        "carla_world_transform": {
            "x": camera.carla_world_transform.x,
            "y": camera.carla_world_transform.y,
            "z": camera.carla_world_transform.z,
            "pitch": camera.carla_world_transform.pitch,
            "yaw": camera.carla_world_transform.yaw,
            "roll": camera.carla_world_transform.roll,
        },
        "fov_x_deg": camera.fov_x_deg,
        "fovy_deg": camera.fovy_deg,
    }


def _prepare_node_output(run_dir: Path, node: NodeConfig) -> NodeCaptureState:
    node_dir = run_dir / "nodes" / node.node_id
    for cam_id in node.camera_order:
        (node_dir / "cams" / cam_id / "frames").mkdir(parents=True, exist_ok=True)
    (node_dir / "calib").mkdir(parents=True, exist_ok=True)

    rig = {
        "schema_version": "carla_air_ground_to_air_capture_rig_v1",
        "node_id": node.node_id,
        "camera_order": node.camera_order,
        "anchor_transform": node.raw_entry.get("anchor_transform"),
        "cameras": {cam_id: _camera_calib_entry(node.cameras[cam_id]) for cam_id in node.camera_order},
        "source_node_config": node.raw_entry,
    }
    _write_json(node_dir / "calib" / "rig.json", rig)

    csv_file = (node_dir / "frame_times.csv").open("w", encoding="utf-8", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["ts_us", "carla_frame", "carla_timestamp", "cam_id", "filename"])
    csv_file.flush()

    return NodeCaptureState(
        node=node,
        node_dir=node_dir,
        expected_cams=set(node.camera_order),
        csv_file=csv_file,
        csv_writer=csv_writer,
    )


def _close_node_outputs(states: dict[str, NodeCaptureState]) -> None:
    for state in states.values():
        if state.csv_file is not None:
            state.csv_file.flush()
            state.csv_file.close()
            state.csv_file = None
            state.csv_writer = None


def _write_complete_group(state: NodeCaptureState, frame: int, packets: dict[str, FramePacket]) -> None:
    ordered_packets = {cam_id: packets[cam_id] for cam_id in state.node.camera_order}
    reference = ordered_packets[state.node.camera_order[0]]
    ts_us = int(round(reference.timestamp * 1_000_000.0))
    filename = f"{ts_us:016d}.png"

    for cam_id, packet in ordered_packets.items():
        frame_path = state.node_dir / "cams" / cam_id / "frames" / filename
        _write_png_rgb(frame_path, _packet_to_rgb(packet))
        rel_filename = str(Path("cams") / cam_id / "frames" / filename)
        state.csv_writer.writerow([ts_us, packet.frame, f"{packet.timestamp:.9f}", cam_id, rel_filename])
        state.images_written += 1

    state.csv_file.flush()
    state.groups_written += 1
    state.written_frames.add(frame)
    state.first_frame = frame if state.first_frame is None else min(state.first_frame, frame)
    state.last_frame = frame if state.last_frame is None else max(state.last_frame, frame)


def _ready_node_ids_for_frame(states: dict[str, NodeCaptureState], frame: int) -> list[str]:
    ready_node_ids: list[str] = []
    for node_id, state in states.items():
        packets = state.frame_buffers.get(frame)
        if packets is not None and state.expected_cams.issubset(packets.keys()):
            ready_node_ids.append(node_id)
    return ready_node_ids


def _write_target_frame_for_nodes(
    states: dict[str, NodeCaptureState],
    frame: int,
    node_ids: list[str],
    writer: Any = _write_complete_group,
) -> None:
    for node_id in node_ids:
        state = states[node_id]
        packets = state.frame_buffers.pop(frame)
        writer(state, frame, packets)


def _discard_frame_for_unwritten_nodes(
    states: dict[str, NodeCaptureState],
    frame: int,
    *,
    count_incomplete: bool,
) -> None:
    for state in states.values():
        if frame in state.written_frames or frame in state.discarded_frames:
            continue
        state.frame_buffers.pop(frame, None)
        if count_incomplete:
            state.incomplete_dropped_frames += 1
        state.discarded_frames.add(frame)


def _discard_remaining_buffers(states: dict[str, NodeCaptureState], *, count_incomplete: bool) -> None:
    for state in states.values():
        buffered_frames = list(state.frame_buffers)
        if count_incomplete:
            state.incomplete_dropped_frames += len(buffered_frames)
        state.discarded_frames.update(buffered_frames)
        state.frame_buffers.clear()


def _flush_buffers(states: dict[str, NodeCaptureState], *, prune_frame_lag: int, final: bool = False) -> None:
    for state in states.values():
        complete_frames = [
            frame
            for frame, packets in state.frame_buffers.items()
            if state.expected_cams.issubset(packets.keys()) and frame not in state.written_frames
        ]
        for frame in sorted(complete_frames):
            packets = state.frame_buffers.pop(frame)
            _write_complete_group(state, frame, packets)

        if not state.frame_buffers:
            continue

        if final:
            dropped = len(state.frame_buffers)
            state.incomplete_dropped_frames += dropped
            state.discarded_frames.update(state.frame_buffers)
            state.frame_buffers.clear()
            continue

        newest_frame = max(state.frame_buffers)
        stale_frames = [frame for frame in state.frame_buffers if frame <= newest_frame - prune_frame_lag]
        for frame in stale_frames:
            state.frame_buffers.pop(frame, None)
            state.discarded_frames.add(frame)
            state.incomplete_dropped_frames += 1


def _spawn_sensors(
    *,
    carla: Any,
    world: Any,
    run_id: str,
    selected_nodes: list[NodeConfig],
    states: dict[str, NodeCaptureState],
    fps: float,
    lock: Any,
) -> list[Any]:
    bp_lib = world.get_blueprint_library()
    sensors = []
    sensor_tick = 1.0 / fps
    for node in selected_nodes:
        for cam_id in node.camera_order:
            camera = node.cameras[cam_id]
            bp = bp_lib.find("sensor.camera.rgb")
            bp.set_attribute("image_size_x", str(camera.image_size[0]))
            bp.set_attribute("image_size_y", str(camera.image_size[1]))
            bp.set_attribute("fov", f"{camera.fov_x_deg:.8f}")
            bp.set_attribute("sensor_tick", f"{sensor_tick:.8f}")
            if bp.has_attribute("role_name"):
                bp.set_attribute("role_name", f"{CAPTURE_ROLE_PREFIX}.{run_id}.{node.node_id}.{cam_id}")
            sensor = world.spawn_actor(bp, _carla_transform(carla, camera.carla_world_transform))
            sensors.append(sensor)

            def on_image(image: Any, node_id: str = node.node_id, camera_id: str = cam_id) -> None:
                packet = FramePacket(
                    frame=int(image.frame),
                    timestamp=float(image.timestamp),
                    platform_timestamp=float(getattr(image, "platform_timestamp", 0.0)),
                    width=int(image.width),
                    height=int(image.height),
                    raw_data=bytes(image.raw_data),
                )
                with lock:
                    state = states[node_id]
                    if packet.frame in state.written_frames or packet.frame in state.discarded_frames:
                        return
                    state.frame_buffers.setdefault(packet.frame, {})[camera_id] = packet

            sensor.listen(on_image)
    return sensors


def _state_summary(state: NodeCaptureState) -> dict[str, Any]:
    return {
        "camera_order": state.node.camera_order,
        "synchronized_frame_groups": state.groups_written,
        "images_written": state.images_written,
        "dropped_or_incomplete_frames": state.incomplete_dropped_frames,
        "first_carla_frame": state.first_frame,
        "last_carla_frame": state.last_frame,
        "node_dir": str(state.node_dir),
    }


def _min_group_failure_reasons(states: dict[str, NodeCaptureState], min_groups: int) -> list[str]:
    return [
        f"{node_id}: synchronized_frame_groups={state.groups_written} < min_groups={min_groups}"
        for node_id, state in states.items()
        if state.groups_written < min_groups
    ]


def _progress_snapshot(states: dict[str, NodeCaptureState]) -> list[tuple[str, int, int]]:
    return [
        (node_id, state.groups_written, state.incomplete_dropped_frames)
        for node_id, state in states.items()
    ]


def _format_node_progress(snapshot: list[tuple[str, int, int]]) -> str:
    return " | ".join(f"{node_id} ok={ok} drop={drop}" for node_id, ok, drop in snapshot)


def _print_progress(line: str) -> None:
    print(line, end="\r", flush=True)


def _print_progress_newline(progress_enabled: bool) -> None:
    if progress_enabled:
        print("", flush=True)


def _wait_for_frame_callbacks(
    *,
    states: dict[str, NodeCaptureState],
    frame: int,
    timeout_seconds: float,
    lock: Any,
) -> list[str]:
    deadline = time.monotonic() + timeout_seconds
    ready_node_ids: list[str] = []
    while time.monotonic() <= deadline:
        with lock:
            ready_node_ids = _ready_node_ids_for_frame(states, frame)
            if len(ready_node_ids) == len(states):
                return ready_node_ids
        time.sleep(0.002)
    with lock:
        return _ready_node_ids_for_frame(states, frame)


def _run_passive_loop(
    *,
    args: argparse.Namespace,
    states: dict[str, NodeCaptureState],
    stop_requested: Any,
    lock: Any,
) -> None:
    start = time.monotonic()
    deadline = time.monotonic() + float(args.seconds)
    next_progress_at = start
    while not stop_requested() and time.monotonic() < deadline:
        with lock:
            _flush_buffers(states, prune_frame_lag=int(args.prune_frame_lag), final=False)
            snapshot = _progress_snapshot(states)
        now = time.monotonic()
        if not args.no_progress and (now >= next_progress_at or now >= deadline):
            elapsed = min(float(args.seconds), now - start)
            percent = min(100.0, elapsed / float(args.seconds) * 100.0)
            _print_progress(
                f"[passive] {percent:5.1f}% | wall {elapsed:.1f}/{float(args.seconds):.1f}s | "
                f"{_format_node_progress(snapshot)}"
            )
            next_progress_at = now + float(args.progress_interval)
        time.sleep(0.05)


def _run_sync_loop(
    *,
    world: Any,
    args: argparse.Namespace,
    states: dict[str, NodeCaptureState],
    stop_requested: Any,
    lock: Any,
) -> dict[str, Any]:
    warmup_frames = int(args.warmup_frames)
    for _ in range(warmup_frames):
        if stop_requested():
            break
        frame = int(world.tick())
        _wait_for_frame_callbacks(
            states=states,
            frame=frame,
            timeout_seconds=float(args.frame_timeout),
            lock=lock,
        )
        with lock:
            _discard_frame_for_unwritten_nodes(states, frame, count_incomplete=False)

    target_frame_groups = max(1, int(math.ceil(float(args.seconds) * float(args.fps))))
    attempted_frames = 0
    consecutive_timeouts = 0
    max_consecutive_timeouts_seen = 0
    stopped_due_to_timeouts = False
    next_progress_at = time.monotonic()
    while not stop_requested() and attempted_frames < target_frame_groups:
        frame = int(world.tick())
        attempted_frames += 1
        ready_node_ids = _wait_for_frame_callbacks(
            states=states,
            frame=frame,
            timeout_seconds=float(args.frame_timeout),
            lock=lock,
        )
        with lock:
            _write_target_frame_for_nodes(states, frame, ready_node_ids)
            _discard_frame_for_unwritten_nodes(states, frame, count_incomplete=True)
            snapshot = _progress_snapshot(states)
        if ready_node_ids:
            consecutive_timeouts = 0
        else:
            consecutive_timeouts += 1
            max_consecutive_timeouts_seen = max(max_consecutive_timeouts_seen, consecutive_timeouts)
        now = time.monotonic()
        if not args.no_progress and (now >= next_progress_at or attempted_frames >= target_frame_groups):
            percent = attempted_frames / target_frame_groups * 100.0
            sim_elapsed = attempted_frames / float(args.fps)
            _print_progress(
                f"[sync] {percent:5.1f}% | groups {attempted_frames}/{target_frame_groups} | "
                f"sim {sim_elapsed:.1f}/{float(args.seconds):.1f}s | {_format_node_progress(snapshot)}"
            )
            next_progress_at = now + float(args.progress_interval)
        if int(args.max_consecutive_timeouts) > 0 and consecutive_timeouts >= int(args.max_consecutive_timeouts):
            stopped_due_to_timeouts = True
            break
    return {
        "target_frame_groups": target_frame_groups,
        "attempted_frame_groups": attempted_frames,
        "consecutive_timeouts": consecutive_timeouts,
        "max_consecutive_timeouts_seen": max_consecutive_timeouts_seen,
        "stopped_due_to_timeouts": stopped_due_to_timeouts,
    }


def _run_capture(args: argparse.Namespace) -> int:
    if float(args.fps) <= 0:
        raise SystemExit("--fps must be positive.")
    if float(args.seconds) <= 0:
        raise SystemExit("--seconds must be positive.")
    if int(args.min_groups) < 0:
        raise SystemExit("--min-groups must be greater than or equal to 0.")
    if float(args.frame_timeout) <= 0:
        raise SystemExit("--frame-timeout must be positive.")
    if int(args.warmup_frames) < 0:
        raise SystemExit("--warmup-frames must be greater than or equal to 0.")
    if float(args.progress_interval) <= 0:
        raise SystemExit("--progress-interval must be positive.")
    if float(args.sensor_stop_grace) < 0:
        raise SystemExit("--sensor-stop-grace must be greater than or equal to 0.")
    if int(args.max_consecutive_timeouts) < 0:
        raise SystemExit("--max-consecutive-timeouts must be greater than or equal to 0.")

    import threading
    import carla

    config_path = Path(args.config).resolve()
    out_root = Path(args.out_root).resolve()
    config_data, available_nodes = _load_capture_config(config_path)
    node_ids = _parse_node_args(args.nodes, all_nodes=bool(args.all_nodes), available=available_nodes)
    selected_nodes = [available_nodes[node_id] for node_id in node_ids]

    run_id = str(args.run_id) if args.run_id else _make_run_id()
    run_dir = out_root / run_id

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

    stale_count = _cleanup_stale_capture_sensors(world)
    if stale_count:
        print(f"Cleaned up {stale_count} stale capture sensor(s)")

    actual_map = str(world.get_map().name)
    actual_map_basename = _path_basename(actual_map)
    configured_map = str(config_data.get("carla_map_name") or config_data.get("map") or "")
    configured_map_basenames = _configured_map_basenames(config_data)
    if not configured_map_basenames:
        raise SystemExit(
            f"Camera node config does not define carla_map_name or map: {config_path}\n"
            "Use a node placement config that records the intended CARLA map."
        )
    map_matches = actual_map_basename in configured_map_basenames
    if not map_matches:
        message = (
            "CARLA map mismatch.\n"
            f"  Active world map: {actual_map} (basename: {actual_map_basename})\n"
            f"  Config map basenames: {', '.join(configured_map_basenames)}\n"
            f"  Config file: {config_path}\n"
            "Use the camera node config saved for the active map, or restart CARLA-Air with the matching map."
        )
        if bool(args.allow_map_mismatch):
            print(f"Warning: {message}")
        else:
            raise SystemExit(message + "\nPass --allow-map-mismatch only for debugging.")

    if run_dir.exists() and any(run_dir.iterdir()):
        raise SystemExit(f"Output run directory already exists and is not empty: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    states = {node.node_id: _prepare_node_output(run_dir, node) for node in selected_nodes}
    lock = threading.Lock()
    sensors: list[Any] = []
    stop_requested = False
    original_settings: Any = None
    sync_settings_applied = False
    world_settings_restored = False
    fixed_delta_seconds = 1.0 / float(args.fps)
    target_frame_groups = 0
    attempted_frame_groups = 0
    consecutive_timeouts = 0
    max_consecutive_timeouts_seen = 0
    stopped_due_to_timeouts = False
    sensor_stop_failures: list[dict[str, Any]] = []
    sensor_destroy_failures: list[dict[str, Any]] = []
    runtime_exception: str | None = None
    interrupted = False

    def request_stop(signum: int, _frame: Any) -> None:
        nonlocal interrupted, stop_requested
        interrupted = True
        stop_requested = True
        print(f"Received signal {signum}; stopping capture...", flush=True)

    previous_sigint = signal.getsignal(signal.SIGINT)
    previous_sigterm = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    capture_meta: dict[str, Any] = {
        "schema_version": "carla_air_ground_to_air_capture_smoke_v1",
        "run_id": run_id,
        "mode": args.mode,
        "started_at": _utc_now_iso(),
        "finished_at": None,
        "duration_seconds_requested": float(args.seconds),
        "fps_requested": float(args.fps),
        "wall_time_seconds": None,
        "progress_interval_seconds": float(args.progress_interval),
        "progress_enabled": not bool(args.no_progress),
        "host": host,
        "port": port,
        "config": str(config_path),
        "out_dir": str(run_dir),
        "configured_map": configured_map,
        "configured_map_basenames": configured_map_basenames,
        "actual_carla_map": actual_map,
        "actual_carla_map_basename": actual_map_basename,
        "allow_map_mismatch": bool(args.allow_map_mismatch),
        "map_match": map_matches,
        "nodes_requested": node_ids,
        "nodes": {},
        "dropped_or_incomplete_frames": 0,
        "min_groups_required": int(args.min_groups),
        "synchronous_mode_applied": False,
        "fixed_delta_seconds": None,
        "warmup_frames": int(args.warmup_frames),
        "frame_timeout_seconds": float(args.frame_timeout),
        "target_frame_groups": None,
        "attempted_frame_groups": 0,
        "consecutive_timeouts": 0,
        "max_consecutive_timeouts_seen": 0,
        "max_consecutive_timeouts_allowed": int(args.max_consecutive_timeouts),
        "stopped_due_to_timeouts": False,
        "world_settings_restored": False,
        "interrupted": False,
        "runtime_exception": None,
        "sensor_stop_grace_seconds": float(args.sensor_stop_grace),
        "sensor_stop_failures": [],
        "sensor_stop_failure_count": 0,
        "sensor_destroy_failures": [],
        "sensor_destroy_failure_count": 0,
        "success": False,
        "failure_reasons": [],
        "notes": [
            "Sync mode temporarily applies CARLA synchronous_mode and fixed_delta_seconds, then drives world.tick().",
            "Passive mode is for debugging sensor reception and does not modify world settings or call world.tick().",
            "A synchronized frame group is written only when every camera in a node receives the same CARLA image.frame.",
        ],
    }
    failure_reasons: list[str] = []
    capture_started_monotonic = time.monotonic()

    try:
        if args.mode == "sync":
            original_settings = world.get_settings()
            sync_settings = world.get_settings()
            sync_settings.synchronous_mode = True
            sync_settings.fixed_delta_seconds = fixed_delta_seconds
            world.apply_settings(sync_settings)
            sync_settings_applied = True
            capture_meta["synchronous_mode_applied"] = True
            capture_meta["fixed_delta_seconds"] = fixed_delta_seconds

        sensors = _spawn_sensors(
            carla=carla,
            world=world,
            run_id=run_id,
            selected_nodes=selected_nodes,
            states=states,
            fps=float(args.fps),
            lock=lock,
        )
        print(
            f"Capturing {', '.join(node_ids)} in {args.mode} mode for {float(args.seconds):.1f}s "
            f"at {float(args.fps):.2f} fps "
            f"into {run_dir}"
        )
        if args.mode == "sync":
            sync_stats = _run_sync_loop(
                world=world,
                args=args,
                states=states,
                stop_requested=lambda: stop_requested,
                lock=lock,
            )
            target_frame_groups = int(sync_stats["target_frame_groups"])
            attempted_frame_groups = int(sync_stats["attempted_frame_groups"])
            consecutive_timeouts = int(sync_stats["consecutive_timeouts"])
            max_consecutive_timeouts_seen = int(sync_stats["max_consecutive_timeouts_seen"])
            stopped_due_to_timeouts = bool(sync_stats["stopped_due_to_timeouts"])
            if stopped_due_to_timeouts:
                failure_reasons.append(
                    "Stopped early after "
                    f"{consecutive_timeouts} consecutive sync ticks with no complete node frames "
                    f"(limit={int(args.max_consecutive_timeouts)})"
                )
        else:
            target_frame_groups = max(1, int(math.ceil(float(args.seconds) * float(args.fps))))
            _run_passive_loop(
                args=args,
                states=states,
                stop_requested=lambda: stop_requested,
                lock=lock,
            )
    except Exception as exc:
        runtime_exception = f"{type(exc).__name__}: {exc}"
        failure_reasons.append(f"Runtime exception during capture: {runtime_exception}")
    finally:
        sensor_stop_failures = _stop_sensors(sensors)
        if sensor_stop_failures:
            failure_reasons.append(f"{len(sensor_stop_failures)} sensor stop failure(s)")
        if sync_settings_applied and original_settings is not None:
            try:
                world.apply_settings(original_settings)
                world_settings_restored = True
            except Exception as exc:
                failure_reasons.append(f"Failed to restore original CARLA world settings: {exc}")
        if sensors and float(args.sensor_stop_grace) > 0:
            time.sleep(float(args.sensor_stop_grace))
        sensor_destroy_failures = _destroy_sensors(sensors)
        if sensor_destroy_failures:
            failure_reasons.append(f"{len(sensor_destroy_failures)} sensor destroy failure(s)")
        with lock:
            if args.mode == "passive":
                _flush_buffers(states, prune_frame_lag=int(args.prune_frame_lag), final=True)
            else:
                _discard_remaining_buffers(states, count_incomplete=True)
            final_snapshot = _progress_snapshot(states)
        if not args.no_progress:
            if args.mode == "sync" and target_frame_groups > 0:
                percent = attempted_frame_groups / target_frame_groups * 100.0
                sim_elapsed = attempted_frame_groups / float(args.fps)
                _print_progress(
                    f"[sync] {percent:5.1f}% | groups {attempted_frame_groups}/{target_frame_groups} | "
                    f"sim {sim_elapsed:.1f}/{float(args.seconds):.1f}s | "
                    f"{_format_node_progress(final_snapshot)}"
                )
            elif args.mode == "passive":
                elapsed = min(float(args.seconds), time.monotonic() - capture_started_monotonic)
                percent = min(100.0, elapsed / float(args.seconds) * 100.0)
                _print_progress(
                    f"[passive] {percent:5.1f}% | wall {elapsed:.1f}/{float(args.seconds):.1f}s | "
                    f"{_format_node_progress(final_snapshot)}"
                )
            _print_progress_newline(True)
        capture_meta["finished_at"] = _utc_now_iso()
        capture_meta["wall_time_seconds"] = time.monotonic() - capture_started_monotonic
        capture_meta["nodes"] = {node_id: _state_summary(state) for node_id, state in states.items()}
        capture_meta["dropped_or_incomplete_frames"] = sum(
            state.incomplete_dropped_frames for state in states.values()
        )
        capture_meta["target_frame_groups"] = target_frame_groups
        capture_meta["attempted_frame_groups"] = attempted_frame_groups
        capture_meta["consecutive_timeouts"] = consecutive_timeouts
        capture_meta["max_consecutive_timeouts_seen"] = max_consecutive_timeouts_seen
        capture_meta["stopped_due_to_timeouts"] = stopped_due_to_timeouts
        capture_meta["world_settings_restored"] = world_settings_restored
        capture_meta["interrupted"] = interrupted
        capture_meta["runtime_exception"] = runtime_exception
        capture_meta["sensor_stop_failures"] = sensor_stop_failures
        capture_meta["sensor_stop_failure_count"] = len(sensor_stop_failures)
        capture_meta["sensor_destroy_failures"] = sensor_destroy_failures
        capture_meta["sensor_destroy_failure_count"] = len(sensor_destroy_failures)
        if interrupted:
            failure_reasons.append("Capture interrupted by signal")
        failure_reasons.extend(_min_group_failure_reasons(states, int(args.min_groups)))
        capture_meta["success"] = not failure_reasons
        capture_meta["failure_reasons"] = failure_reasons
        _write_json(run_dir / "capture_meta.json", capture_meta)
        _close_node_outputs(states)
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)
        print(f"Capture sensors cleaned up. Metadata: {run_dir / 'capture_meta.json'}", flush=True)

    if failure_reasons:
        print("Capture smoke failed:")
        for reason in failure_reasons:
            print(f"  - {reason}")
        return 2
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Synchronous RGB smoke capture for saved CARLA-Air ground-to-air camera nodes.",
    )
    parser.add_argument("--host", default="localhost", help="CARLA host. Default: localhost")
    parser.add_argument("--port", type=int, default=2000, help="CARLA RPC port. Default: 2000")
    parser.add_argument("--timeout", type=float, default=15.0, help="CARLA client timeout seconds. Default: 15")
    parser.add_argument("--wait-seconds", type=float, default=120.0, help="Wait for the CARLA port before connecting. Default: 120")
    parser.add_argument("--mode", choices=["sync", "passive"], default="sync", help="Capture mode. Default: sync")
    parser.add_argument("--nodes", nargs="+", help="Node ids to capture, for example: --nodes node01 node02. Default: node01")
    parser.add_argument("--all-nodes", action="store_true", help="Capture all nodes in the placement config.")
    parser.add_argument("--allow-map-mismatch", action="store_true", help="Allow capture to continue when the active CARLA map does not match the node config. Debug only.")
    parser.add_argument("--seconds", type=float, default=10.0, help="Capture duration in seconds. Default: 10")
    parser.add_argument("--fps", type=float, default=10.0, help="Sensor FPS via sensor_tick=1/fps. Default: 10")
    parser.add_argument("--frame-timeout", type=float, default=2.0, help="Max seconds to wait for all callbacks after each sync tick. Default: 2.0")
    parser.add_argument("--warmup-frames", type=int, default=3, help="Sync ticks to run after spawning sensors before writing frames. Default: 3")
    parser.add_argument("--min-groups", type=int, default=1, help="Minimum synchronized frame groups required per requested node. Default: 1")
    parser.add_argument("--progress-interval", type=float, default=1.0, help="Seconds between terminal progress refreshes. Default: 1.0")
    parser.add_argument("--no-progress", action="store_true", help="Disable terminal progress output.")
    parser.add_argument("--sensor-stop-grace", type=float, default=0.2, help="Seconds to wait between sensor.stop() and sensor.destroy(). Default: 0.2")
    parser.add_argument(
        "--max-consecutive-timeouts",
        type=int,
        default=20,
        help="Stop sync capture after this many consecutive ticks with no complete node frames. Use 0 to disable. Default: 20",
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help=f"Camera node placement JSON. Default: {DEFAULT_CONFIG}")
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT), help=f"Capture output root. Default: {DEFAULT_OUT_ROOT}")
    parser.add_argument("--run-id", help="Optional run id. Default: current local timestamp")
    parser.add_argument(
        "--prune-frame-lag",
        type=int,
        default=120,
        help="Drop incomplete buffered CARLA frames after this frame-number lag. Default: 120",
    )
    return parser


def _run_internal_self_test() -> int:
    node = NodeConfig(node_id="node01", camera_order=["cam0", "cam1", "cam2"], cameras={}, raw_entry={})
    state = NodeCaptureState(node=node, node_dir=Path("_dummy"), expected_cams={"cam0", "cam1", "cam2"})
    failures = _min_group_failure_reasons({"node01": state}, min_groups=1)
    if not failures:
        print("Expected 0 synchronized frame groups to fail min-groups=1")
        return 1
    state.groups_written = 1
    failures = _min_group_failure_reasons({"node01": state}, min_groups=1)
    if failures:
        print("Expected 1 synchronized frame group to pass min-groups=1")
        return 1
    if max(1, int(math.ceil(5.0 * 5.0))) != 25:
        print("Expected 5 seconds at 5 fps to target 25 frame groups")
        return 1

    node2 = NodeConfig(node_id="node02", camera_order=["cam0", "cam1", "cam2"], cameras={}, raw_entry={})
    state1 = NodeCaptureState(node=node, node_dir=Path("_dummy/node01"), expected_cams={"cam0", "cam1", "cam2"})
    state2 = NodeCaptureState(node=node2, node_dir=Path("_dummy/node02"), expected_cams={"cam0", "cam1", "cam2"})
    frame = 1234
    dummy = FramePacket(frame=frame, timestamp=1.234, platform_timestamp=0.0, width=1, height=1, raw_data=b"")
    state1.frame_buffers[frame] = {"cam0": dummy, "cam1": dummy, "cam2": dummy}
    state2.frame_buffers[frame] = {"cam0": dummy, "cam1": dummy}
    states = {"node01": state1, "node02": state2}
    ready = _ready_node_ids_for_frame(states, frame)
    if ready != ["node01"]:
        print(f"Expected only node01 to be ready, got {ready}")
        return 1

    def fake_writer(target_state: NodeCaptureState, target_frame: int, _packets: dict[str, FramePacket]) -> None:
        target_state.groups_written += 1
        target_state.written_frames.add(target_frame)

    _write_target_frame_for_nodes(states, frame, ready, writer=fake_writer)
    _discard_frame_for_unwritten_nodes(states, frame, count_incomplete=True)
    if state1.groups_written != 1 or state2.groups_written != 0 or state2.incomplete_dropped_frames != 1:
        print("Expected per-node sync write to keep node01 and drop only node02")
        return 1
    progress = _format_node_progress(_progress_snapshot(states))
    if "node01 ok=1 drop=0" not in progress or "node02 ok=0 drop=1" not in progress:
        print(f"Unexpected progress format: {progress}")
        return 1

    class FailingSensor:
        id = 42
        attributes = {"role_name": "ground_to_air_capture_smoke.test.node01.cam0"}

        def stop(self) -> None:
            raise RuntimeError("stop failed")

        def destroy(self) -> None:
            raise RuntimeError("destroy failed")

    failing_sensors: list[Any] = [FailingSensor()]
    stop_failures = _stop_sensors(failing_sensors)
    destroy_failures = _destroy_sensors(failing_sensors)
    if len(stop_failures) != 1 or stop_failures[0].get("actor_id") != 42:
        print(f"Expected one stop failure with actor id 42, got {stop_failures}")
        return 1
    if len(destroy_failures) != 1 or destroy_failures[0].get("actor_id") != 42 or failing_sensors:
        print(f"Expected one destroy failure and cleared sensor list, got {destroy_failures}")
        return 1
    print("internal self-test ok")
    return 0


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if argv == ["--_internal-self-test"]:
        return _run_internal_self_test()
    parser = build_parser()
    args = parser.parse_args(argv)
    return _run_capture(args)


if __name__ == "__main__":
    raise SystemExit(main())
