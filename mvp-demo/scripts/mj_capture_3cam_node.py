from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _now_scene_id(prefix: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{ts}"


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _as_list_xyz(value: np.ndarray) -> List[float]:
    return [float(value[0]), float(value[1]), float(value[2])]


def _K_from_fovy_deg(fovy_deg: float, w: int, h: int) -> List[List[float]]:
    # Pin-hole intrinsics from vertical FOV (MuJoCo `fovy` is vertical). Assume square pixels => fx == fy.
    fovy_rad = math.radians(float(fovy_deg))
    fy = (float(h) * 0.5) / math.tan(fovy_rad * 0.5)
    fx = fy
    cx = float(w) * 0.5
    cy = float(h) * 0.5
    return [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]


def _T_from_Rt(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R.astype(np.float32)
    T[:3, 3] = t.astype(np.float32)
    return T


def _invert_T(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=np.float32)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def _mj_znear_zfar_m(model) -> Tuple[float, float]:
    # MuJoCo stores znear/zfar as fractions of model.stat.extent.
    extent = float(model.stat.extent)
    znear = float(model.vis.map.znear) * extent
    zfar = float(model.vis.map.zfar) * extent
    return znear, zfar


def _rot_x_deg(angle_deg: float) -> np.ndarray:
    a = math.radians(float(angle_deg))
    c = math.cos(a)
    s = math.sin(a)
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, c, -s],
            [0.0, s, c],
        ],
        dtype=np.float32,
    )


def _rot_z_deg(angle_deg: float) -> np.ndarray:
    a = math.radians(float(angle_deg))
    c = math.cos(a)
    s = math.sin(a)
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def _quat_wxyz_from_R(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64)
    trace = float(np.trace(R))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(max(1e-12, 1.0 + float(R[0, 0]) - float(R[1, 1]) - float(R[2, 2]))) * 2.0
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(max(1e-12, 1.0 + float(R[1, 1]) - float(R[0, 0]) - float(R[2, 2]))) * 2.0
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(max(1e-12, 1.0 + float(R[2, 2]) - float(R[0, 0]) - float(R[1, 1]))) * 2.0
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    quat = np.array([qw, qx, qy, qz], dtype=np.float32)
    quat /= max(1e-12, float(np.linalg.norm(quat)))
    return quat


def _spin_quat_wxyz(
    t_sec: float,
    *,
    capture_seconds: float,
    yaw_start_deg: float,
    yaw_end_deg: float,
    pitch_amp_deg: float,
    pitch_period_sec: float,
) -> tuple[np.ndarray, float, float]:
    alpha = 0.0 if capture_seconds <= 1e-6 else min(max(t_sec / capture_seconds, 0.0), 1.0)
    yaw_deg = float(yaw_start_deg) + alpha * (float(yaw_end_deg) - float(yaw_start_deg))
    pitch_period = max(1e-6, float(pitch_period_sec))
    pitch_deg = float(pitch_amp_deg) * math.sin(2.0 * math.pi * (t_sec / pitch_period))
    R = _rot_z_deg(yaw_deg) @ _rot_x_deg(pitch_deg)
    return _quat_wxyz_from_R(R), yaw_deg, pitch_deg


def _render_depth(renderer) -> np.ndarray:
    # MuJoCo Python APIs differ slightly across versions.
    def _to_2d(d: np.ndarray) -> np.ndarray:
        d = np.asarray(d)
        if d.ndim == 3:
            d = d[..., 0]
        return d

    if hasattr(renderer, "enable_depth_rendering"):
        try:
            renderer.enable_depth_rendering()
            depth = renderer.render()
            return _to_2d(np.asarray(depth))
        finally:
            try:
                renderer.disable_depth_rendering()
            except Exception:
                pass

    # Older mujoco bindings: render(depth=True)
    depth = renderer.render(depth=True)  # type: ignore[call-arg]
    return _to_2d(np.asarray(depth))


def _render_rgb(renderer) -> np.ndarray:
    try:
        renderer.disable_depth_rendering()
    except Exception:
        pass
    try:
        renderer.disable_segmentation_rendering()
    except Exception:
        pass
    return np.asarray(renderer.render())


def _update_scene_with_catmask(renderer, model, data, *, camera: str, catmask: int, scene_option=None) -> None:
    """
    Renderer.update_scene hardcodes catmask=mjCAT_ALL; for masks we often want mjCAT_DYNAMIC only.
    Use mujoco.mjv_updateScene directly with a custom catmask.
    """
    import mujoco  # type: ignore

    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
    if cam_id < 0:
        raise ValueError(f'Camera "{camera}" does not exist in the model.')

    cam = mujoco.MjvCamera()
    cam.fixedcamid = int(cam_id)
    cam.type = mujoco.mjtCamera.mjCAMERA_FIXED

    opt = scene_option
    if opt is None:
        # Prefer renderer's default option if available.
        opt = getattr(renderer, "_scene_option", None)
    if opt is None:
        opt = mujoco.MjvOption()

    mujoco.mjv_updateScene(model, data, opt, None, cam, int(catmask), renderer._scene)  # type: ignore[attr-defined]


@dataclass(frozen=True)
class _CamOut:
    cam_id: str  # e.g. "cam0" (output folder id)
    camera_name: str  # mjcf camera name, e.g. "node01_cam0"


def main() -> None:
    ap = argparse.ArgumentParser(description="Capture a 3-camera MuJoCo node (parallel optical axes) to disk.")
    ap.add_argument(
        "--mjcf",
        default=str(Path(__file__).resolve().parents[1] / "assets" / "scene" / "mujoco_3cam_node_parallel_j10.xml"),
        type=str,
        help="MJCF with a node body and 3 cameras.",
    )
    ap.add_argument(
        "--out_root",
        default=str(Path(__file__).resolve().parents[1] / "data" / "nodes"),
        type=str,
        help="Output root. Scene is written to <out_root>/<node_id>/scenes/<scene_id>/ ...",
    )
    ap.add_argument("--node_id", default="node01", type=str, help="Logical node id (also used in default camera names).")
    ap.add_argument(
        "--node_body",
        default="",
        type=str,
        help='Body name used as node frame. Default: same as --node_id (e.g. "node01").',
    )
    ap.add_argument(
        "--scene_id",
        default="",
        type=str,
        help="Optional explicit scene id. Default: auto timestamp.",
    )
    ap.add_argument("--width", default=1280, type=int)
    ap.add_argument("--height", default=720, type=int)
    ap.add_argument("--fps", default=30.0, type=float, help="Capture FPS (timestamps are derived from this).")
    ap.add_argument("--seconds", default=3.0, type=float, help="Capture duration in seconds.")
    ap.add_argument(
        "--camera_names",
        default="",
        type=str,
        help='Comma-separated MJCF camera names. Default: "<node_id>_cam0,<node_id>_cam1,<node_id>_cam2".',
    )
    ap.add_argument(
        "--save_depth",
        action="store_true",
        help="Also save full-scene depth GT as .npy under cams/<cam>/<depth_subdir>/ (can be large!).",
    )
    ap.add_argument(
        "--save_masks_gt",
        action="store_true",
        help="Save MuJoCo dynamic-only GT masks under cams/<cam>/<mask_subdir>/.",
    )
    ap.add_argument(
        "--mask_subdir",
        default="masks_gt",
        type=str,
        help='GT mask folder name under each camera. Default: "masks_gt".',
    )
    ap.add_argument(
        "--depth_subdir",
        default="depth_gt",
        type=str,
        help='GT depth folder name under each camera. Default: "depth_gt".',
    )
    ap.add_argument(
        "--target_body",
        default="target",
        type=str,
        help='Name of the moving target body (must have a freejoint for the built-in trajectories).',
    )
    ap.add_argument(
        "--identity_id",
        default="",
        type=str,
        help="Optional identity label for this capture (used by downstream retrieval evaluation).",
    )
    ap.add_argument(
        "--traj",
        default="circle_xz",
        choices=[
            "static",
            "circle_xz",
            "line_x",
            "line_y",
            "line_nodes",
            "static_spin_yaw_pitch",
            "circle_xz_spin_yaw_pitch",
        ],
        help="Simple kinematic trajectory for the target body.",
    )
    ap.add_argument(
        "--traj_from_body",
        default="node01",
        type=str,
        help='For --traj line_nodes: start body name (world positions define the segment).',
    )
    ap.add_argument(
        "--traj_to_body",
        default="node02",
        type=str,
        help='For --traj line_nodes: end body name (world positions define the segment).',
    )
    ap.add_argument(
        "--mid_y",
        default=None,
        type=float,
        help="For --traj line_nodes: optionally translate the segment midpoint to this world Y.",
    )
    ap.add_argument(
        "--mid_z",
        default=None,
        type=float,
        help="For --traj line_nodes: optionally translate the segment midpoint to this world Z.",
    )
    ap.add_argument("--traj_center", default="0 6 2", type=str, help='Trajectory center "x y z" in world/node frame.')
    ap.add_argument("--traj_radius", default=1.0, type=float, help="Trajectory radius (meters).")
    ap.add_argument("--traj_period", default=12.0, type=float, help="Trajectory period (seconds) for circle_xz.")
    ap.add_argument(
        "--yaw_start_deg",
        default=-60.0,
        type=float,
        help="For spin trajectories: start yaw angle in degrees. The yaw sweep spans the capture duration.",
    )
    ap.add_argument(
        "--yaw_end_deg",
        default=60.0,
        type=float,
        help="For spin trajectories: end yaw angle in degrees. The yaw sweep spans the capture duration.",
    )
    ap.add_argument(
        "--pitch_amp_deg",
        default=10.0,
        type=float,
        help="For spin trajectories: pitch sine amplitude in degrees.",
    )
    ap.add_argument(
        "--pitch_period",
        default=8.0,
        type=float,
        help="For spin trajectories: pitch sine period in seconds.",
    )
    args = ap.parse_args()

    try:
        import cv2  # type: ignore
        import mujoco  # type: ignore
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            "Missing deps. Install (suggested): mujoco, numpy, opencv-python.\n"
            "Headless tip: set MUJOCO_GL=osmesa or MUJOCO_GL=egl.\n"
            f"Import error: {e!r}"
        )

    # Keep lexical paths on Windows so junction-based ASCII paths are not
    # canonicalized back to a non-ASCII real path that MuJoCo fails to open.
    mjcf_path = Path(args.mjcf)
    if not mjcf_path.is_absolute():
        mjcf_path = Path.cwd() / mjcf_path
    mjcf_path = mjcf_path.absolute()
    if not mjcf_path.exists():
        raise SystemExit(f"--mjcf not found: {mjcf_path}")

    out_root = Path(args.out_root)
    if not out_root.is_absolute():
        out_root = Path.cwd() / out_root
    out_root = out_root.absolute()
    node_id = str(args.node_id).strip()
    node_body = str(args.node_body).strip() or node_id
    scene_id = str(args.scene_id).strip() or _now_scene_id(f"mj_{node_id}")

    # Output dirs.
    scene_dir = out_root / node_id / "scenes" / scene_id
    cams_root = scene_dir / "cams"
    calib_dir = scene_dir / "calib"
    _ensure_dir(cams_root)
    _ensure_dir(calib_dir)

    # Cameras.
    if args.camera_names.strip():
        cam_names = [t.strip() for t in args.camera_names.split(",") if t.strip()]
    else:
        cam_names = [f"{node_id}_cam0", f"{node_id}_cam1", f"{node_id}_cam2"]

    cams: List[_CamOut] = []
    for i, name in enumerate(cam_names):
        cams.append(_CamOut(cam_id=f"cam{i}", camera_name=name))

    # Per-cam output dirs.
    cam_dirs: Dict[str, Dict[str, Path]] = {}
    for c in cams:
        base = cams_root / c.cam_id
        frames_dir = base / "frames"
        _ensure_dir(frames_dir)
        masks_dir = base / str(args.mask_subdir) if args.save_masks_gt else None
        if masks_dir is not None:
            _ensure_dir(masks_dir)
        depth_dir = base / str(args.depth_subdir) if args.save_depth else None
        if depth_dir is not None:
            _ensure_dir(depth_dir)
        cam_dirs[c.cam_id] = {
            "frames": frames_dir,
            **({"masks_gt": masks_dir} if masks_dir is not None else {}),
            **({"depth": depth_dir} if depth_dir is not None else {}),
        }

    # Load model.
    model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    # Validate node body + cameras.
    node_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, node_body)
    if node_body_id < 0:
        raise SystemExit(f'Node body "{node_body}" not found in MJCF (use --node_body).')

    cam_name_to_id: Dict[str, int] = {}
    for c in cams:
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, c.camera_name)
        if cam_id < 0:
            raise SystemExit(f'Camera "{c.camera_name}" not found in MJCF.')
        cam_name_to_id[c.camera_name] = int(cam_id)

    target_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, str(args.target_body))
    if target_body_id < 0:
        raise SystemExit(f'Target body "{args.target_body}" not found in MJCF.')

    spin_trajs = {"static_spin_yaw_pitch", "circle_xz_spin_yaw_pitch"}
    spin_enabled = str(args.traj) in spin_trajs

    # Optional: resolve node-line endpoints for the line_nodes trajectory.
    line_from_id: Optional[int] = None
    line_to_id: Optional[int] = None
    line_a: Optional[np.ndarray] = None
    line_b: Optional[np.ndarray] = None
    if str(args.traj) == "line_nodes":
        from_body = str(args.traj_from_body).strip()
        to_body = str(args.traj_to_body).strip()
        if not from_body or not to_body:
            raise SystemExit("--traj_from_body/--traj_to_body must be non-empty for --traj line_nodes.")
        bid0 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, from_body)
        bid1 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, to_body)
        if bid0 < 0:
            raise SystemExit(f'--traj_from_body "{from_body}" not found in MJCF.')
        if bid1 < 0:
            raise SystemExit(f'--traj_to_body "{to_body}" not found in MJCF.')
        line_from_id = int(bid0)
        line_to_id = int(bid1)
        line_a = np.asarray(data.xpos[line_from_id], dtype=np.float32).copy()
        line_b = np.asarray(data.xpos[line_to_id], dtype=np.float32).copy()
        if args.mid_y is not None or args.mid_z is not None:
            mid0 = (line_a + line_b) * 0.5
            target_mid_y = float(args.mid_y) if args.mid_y is not None else float(mid0[1])
            target_mid_z = float(args.mid_z) if args.mid_z is not None else float(mid0[2])
            line_shift = np.array([0.0, target_mid_y - float(mid0[1]), target_mid_z - float(mid0[2])], dtype=np.float32)
            line_a = (line_a + line_shift).astype(np.float32)
            line_b = (line_b + line_shift).astype(np.float32)

    # Find freejoint qpos address (7 DoF) for kinematic trajectories.
    jadr = int(model.body_jntadr[target_body_id])
    jnum = int(model.body_jntnum[target_body_id])
    qpos_adr: Optional[int] = None
    if jnum >= 1:
        jtype = int(model.jnt_type[jadr])
        if jtype == int(mujoco.mjtJoint.mjJNT_FREE):
            qpos_adr = int(model.jnt_qposadr[jadr])
    if args.traj != "static" and qpos_adr is None:
        raise SystemExit(
            f'--traj={args.traj} requires target body "{args.target_body}" to have a <freejoint/>.'
        )

    # Parse traj center.
    traj_center_tokens = [t for t in str(args.traj_center).strip().split() if t]
    if len(traj_center_tokens) != 3:
        raise SystemExit('--traj_center must be 3 floats: "x y z"')
    traj_center = np.array([float(traj_center_tokens[0]), float(traj_center_tokens[1]), float(traj_center_tokens[2])])

    # Renderer.
    renderer = mujoco.Renderer(model, width=int(args.width), height=int(args.height))

    # Camera calibration export.
    znear_m, zfar_m = _mj_znear_zfar_m(model)
    far_eps = max(1e-3, 1e-6 * float(zfar_m))
    C_mj_from_cv = np.diag([1.0, -1.0, -1.0]).astype(np.float32)  # cv -> mujoco camera axes

    # Node pose (world <- node).
    node_pos_w = np.asarray(data.xpos[node_body_id], dtype=np.float32).copy()
    node_R_w_n = np.asarray(data.xmat[node_body_id], dtype=np.float32).reshape(3, 3).copy()
    T_w_n = _T_from_Rt(node_R_w_n, node_pos_w)
    T_n_w = _invert_T(T_w_n)

    rig: Dict[str, Any] = {"node_id": node_id, "world_frame": "node", "cameras": {}}
    cam_positions_node: Dict[str, np.ndarray] = {}
    for c in cams:
        cid = cam_name_to_id[c.camera_name]
        fovy_deg = float(model.cam_fovy[cid])
        K = _K_from_fovy_deg(fovy_deg, int(args.width), int(args.height))

        cam_pos_w = np.asarray(data.cam_xpos[cid], dtype=np.float32).copy()
        R_w_c_mj = np.asarray(data.cam_xmat[cid], dtype=np.float32).reshape(3, 3).copy()
        # Export in a CV-friendly camera convention: x right, y down, z forward.
        R_w_c_cv = (R_w_c_mj @ C_mj_from_cv).astype(np.float32)
        T_w_c = _T_from_Rt(R_w_c_cv, cam_pos_w)
        T_n_c = (T_n_w @ T_w_c).astype(np.float32)
        cam_positions_node[c.cam_id] = T_n_c[:3, 3].copy()

        rig["cameras"][c.cam_id] = {
            "image_size": [int(args.width), int(args.height)],
            "K": K,
            "distortion_model": "none",
            "dist": [],
            "T_node_from_cam": T_n_c.tolist(),
            "mjcf_camera_name": c.camera_name,
            "fovy_deg": fovy_deg,
        }

    (calib_dir / "rig.json").write_text(json.dumps(rig, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    # Capture meta.
    baselines_m: Dict[str, float] = {}
    for i in range(len(cams)):
        for j in range(i + 1, len(cams)):
            a = cams[i].cam_id
            b = cams[j].cam_id
            baselines_m[f"{a}-{b}"] = float(np.linalg.norm(cam_positions_node[a] - cam_positions_node[b]))

    capture_meta: Dict[str, Any] = {
        "node_id": node_id,
        "scene_id": scene_id,
        "created_at_iso": datetime.now().isoformat(timespec="seconds"),
        "simulator": "mujoco",
        "mujoco_version": getattr(mujoco, "__version__", "unknown"),
        "mjcf": str(mjcf_path),
        "render": {
            "width": int(args.width),
            "height": int(args.height),
            "fps": float(args.fps),
            "seconds": float(args.seconds),
            "znear_m": float(znear_m),
            "zfar_m": float(zfar_m),
        },
        "cameras": {c.cam_id: {"mjcf_camera_name": c.camera_name} for c in cams},
        "baselines_m": baselines_m,
        "ground_truth_exports": {
            "save_masks_gt": bool(args.save_masks_gt),
            "mask_subdir": str(args.mask_subdir),
            "save_depth_gt": bool(args.save_depth),
            "depth_subdir": str(args.depth_subdir),
        },
        "mask_gt_strategy": {
            "type": "mjCAT_DYNAMIC_depth",
            "note": "GT mask is derived by rendering depth with catmask=mjCAT_DYNAMIC and thresholding < zfar.",
        },
        "target": {
            "body": str(args.target_body),
            "identity_id": str(args.identity_id).strip() or None,
            "traj": str(args.traj),
            **(
                {
                    "traj_from_body": str(args.traj_from_body),
                    "traj_to_body": str(args.traj_to_body),
                    "mid_y": float(args.mid_y) if args.mid_y is not None else None,
                    "mid_z": float(args.mid_z) if args.mid_z is not None else None,
                }
                if str(args.traj) == "line_nodes"
                else {}
            ),
            "traj_center": [float(x) for x in traj_center.tolist()],
            "traj_radius": float(args.traj_radius),
            "traj_period": float(args.traj_period),
            **(
                {
                    "spin_axes": ["yaw", "pitch"],
                    "yaw_start_deg": float(args.yaw_start_deg),
                    "yaw_end_deg": float(args.yaw_end_deg),
                    "yaw_profile": "linear_across_capture_duration",
                    "pitch_amp_deg": float(args.pitch_amp_deg),
                    "pitch_period": float(args.pitch_period),
                    "pitch_profile": "sine",
                    "roll_amp_deg": 0.0,
                }
                if spin_enabled
                else {}
            ),
        },
    }
    (scene_dir / "capture_meta.json").write_text(
        json.dumps(capture_meta, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )

    # Prepare frame_times.csv
    frame_times_path = scene_dir / "frame_times.csv"
    with frame_times_path.open("w", newline="", encoding="utf-8") as f_csv:
        w_csv = csv.writer(f_csv)
        w_csv.writerow(["ts_us", "cam_id", "filename"])

        num_frames = max(1, int(round(float(args.fps) * float(args.seconds))))
        fps = float(args.fps)
        dt = 1.0 / fps

        for frame_idx in range(num_frames):
            t = float(frame_idx) * dt
            ts_us = int(round(t * 1e6))

            # Kinematic target pose (optional).
            if qpos_adr is not None:
                if args.traj == "static":
                    pos = traj_center
                elif args.traj == "static_spin_yaw_pitch":
                    pos = traj_center
                elif args.traj == "circle_xz":
                    theta = 2.0 * math.pi * (t / max(1e-6, float(args.traj_period)))
                    pos = traj_center + np.array(
                        [float(args.traj_radius) * math.cos(theta), 0.0, float(args.traj_radius) * math.sin(theta)],
                        dtype=np.float32,
                    )
                elif args.traj == "circle_xz_spin_yaw_pitch":
                    theta = 2.0 * math.pi * (t / max(1e-6, float(args.traj_period)))
                    pos = traj_center + np.array(
                        [float(args.traj_radius) * math.cos(theta), 0.0, float(args.traj_radius) * math.sin(theta)],
                        dtype=np.float32,
                    )
                elif args.traj == "line_x":
                    # Back-and-forth line in X.
                    s = math.sin(2.0 * math.pi * (t / max(1e-6, float(args.traj_period))))
                    pos = traj_center + np.array([float(args.traj_radius) * s, 0.0, 0.0], dtype=np.float32)
                elif args.traj == "line_y":
                    # Back-and-forth line in Y (distance changes).
                    s = math.sin(2.0 * math.pi * (t / max(1e-6, float(args.traj_period))))
                    pos = traj_center + np.array([0.0, float(args.traj_radius) * s, 0.0], dtype=np.float32)
                elif args.traj == "line_nodes":
                    assert line_a is not None and line_b is not None
                    period = max(1e-6, float(args.traj_period))
                    u = (t / period) % 1.0
                    s = (2.0 * u) if u < 0.5 else (2.0 * (1.0 - u))  # 0->1->0 triangular wave
                    pos = ((1.0 - s) * line_a + s * line_b).astype(np.float32)
                else:
                    pos = traj_center

                data.qpos[qpos_adr + 0 : qpos_adr + 3] = pos  # type: ignore[index]
                if spin_enabled:
                    quat_wxyz, _yaw_deg, _pitch_deg = _spin_quat_wxyz(
                        t,
                        capture_seconds=float(args.seconds),
                        yaw_start_deg=float(args.yaw_start_deg),
                        yaw_end_deg=float(args.yaw_end_deg),
                        pitch_amp_deg=float(args.pitch_amp_deg),
                        pitch_period_sec=float(args.pitch_period),
                    )
                else:
                    quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
                data.qpos[qpos_adr + 3 : qpos_adr + 7] = quat_wxyz
                data.qvel[:] = 0.0
                data.time = t

            mujoco.mj_forward(model, data)

            stem = f"{ts_us:012d}"

            for c in cams:
                cam_id = c.cam_id
                cam_name = c.camera_name
                paths = cam_dirs[cam_id]

                # RGB (full scene).
                renderer.update_scene(data, camera=cam_name)
                rgb = _render_rgb(renderer)
                rgb_u8 = np.asarray(rgb, dtype=np.uint8)
                frame_path = paths["frames"] / f"{stem}.jpg"
                ok = cv2.imwrite(str(frame_path), rgb_u8[..., ::-1])  # RGB->BGR
                if not ok:
                    raise RuntimeError(f"Failed to write image: {frame_path}")

                # Depth (optional, full scene).
                if args.save_depth:
                    renderer.update_scene(data, camera=cam_name)
                    depth = _render_depth(renderer)
                    depth = np.asarray(depth, dtype=np.float32)
                    depth_path = paths["depth"] / f"{stem}.npy"  # type: ignore[index]
                    np.save(str(depth_path), depth)

                if args.save_masks_gt:
                    # GT mask: render dynamic-only depth and threshold.
                    _update_scene_with_catmask(
                        renderer,
                        model,
                        data,
                        camera=cam_name,
                        catmask=mujoco.mjtCatBit.mjCAT_DYNAMIC.value,
                    )
                    depth_dyn = np.asarray(_render_depth(renderer), dtype=np.float32)

                    has = np.isfinite(depth_dyn) & (depth_dyn > 0) & (depth_dyn < float(zfar_m) - far_eps)
                    mask_u8 = (has.astype(np.uint8) * 255)
                    mask_path = paths["masks_gt"] / f"{stem}.png"  # type: ignore[index]
                    ok = cv2.imwrite(str(mask_path), mask_u8)
                    if not ok:
                        raise RuntimeError(f"Failed to write mask: {mask_path}")

                # CSV index row (relative path inside scene).
                rel_frame = frame_path.relative_to(scene_dir).as_posix()
                w_csv.writerow([ts_us, cam_id, rel_frame])


if __name__ == "__main__":
    main()
