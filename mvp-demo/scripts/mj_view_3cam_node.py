from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np


def _require_body_id(model, mujoco, name: str) -> int:
    name = str(name).strip()
    if not name:
        raise SystemExit("Body name must be non-empty.")
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if bid < 0:
        raise SystemExit(f'Body "{name}" not found in MJCF.')
    return int(bid)


def _freejoint_qpos_adr(model, mujoco, body_id: int, *, body_name: str) -> int:
    jadr = int(model.body_jntadr[body_id])
    jnum = int(model.body_jntnum[body_id])
    if jnum < 1 or int(model.jnt_type[jadr]) != int(mujoco.mjtJoint.mjJNT_FREE):
        raise SystemExit(f'Target body "{body_name}" must have a <freejoint/>.')
    return int(model.jnt_qposadr[jadr])


def _triangular_wave_0_1_0(t: float, period_s: float) -> float:
    if period_s <= 1e-6:
        return 0.5
    u = (t / period_s) % 1.0
    return float(1.0 - abs(1.0 - 2.0 * u))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Open a MuJoCo viewer and animate a target along a straight segment between two bodies "
            "(back-and-forth)."
        )
    )
    ap.add_argument(
        "--mjcf",
        default=str(Path(__file__).resolve().parents[1] / "assets" / "mujoco_humanoid_3cam_node_parallel.xml"),
        type=str,
        help="MJCF to load.",
    )
    ap.add_argument("--from_body", default="node01", type=str, help="Start body name for the motion segment.")
    ap.add_argument("--to_body", default="node02", type=str, help="End body name for the motion segment.")
    ap.add_argument("--target_body", default="target", type=str, help="Moving target body name (must have <freejoint/>).")
    ap.add_argument("--mid_y", default=6.0, type=float, help="Translate the segment so its midpoint has this world Y.")
    ap.add_argument("--mid_z", default=2.0, type=float, help="Translate the segment so its midpoint has this world Z.")
    ap.add_argument("--period_s", default=12.0, type=float, help="Back-and-forth period in seconds (<=0 => static).")
    ap.add_argument(
        "--roll_dps",
        default=0.0,
        type=float,
        help="Optional roll rate around the segment axis (degrees/sec). 0 disables roll.",
    )
    args = ap.parse_args()

    try:
        import mujoco  # type: ignore
        import mujoco.viewer  # type: ignore
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            "Missing deps. Install (suggested): mujoco.\n"
            "GUI tip: you need an OpenGL context (desktop or X11 forwarding). Try MUJOCO_GL=glfw.\n"
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

    model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    from_body_id = _require_body_id(model, mujoco, str(args.from_body))
    to_body_id = _require_body_id(model, mujoco, str(args.to_body))

    target_body = str(args.target_body).strip()
    target_body_id = _require_body_id(model, mujoco, target_body)
    qpos_adr = _freejoint_qpos_adr(model, mujoco, int(target_body_id), body_name=target_body)

    a0 = np.asarray(data.xpos[from_body_id], dtype=np.float32)
    b0 = np.asarray(data.xpos[to_body_id], dtype=np.float32)
    mid0 = (a0 + b0) * 0.5
    shift = np.array([0.0, float(args.mid_y) - float(mid0[1]), float(args.mid_z) - float(mid0[2])], dtype=np.float32)
    a = (a0 + shift).astype(np.float32)
    b = (b0 + shift).astype(np.float32)
    axis = (b - a).astype(np.float64)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm > 1e-9:
        axis = axis / axis_norm
    else:
        axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    quat_identity = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    quat_roll = np.empty(4, dtype=np.float64)
    roll_rate_rad_s = math.radians(float(args.roll_dps))

    dt = float(model.opt.timestep)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Make it easier to find the motion segment in free camera mode.
        try:
            viewer.cam.lookat[:] = ((a + b) * 0.5)  # type: ignore[index]
        except Exception:
            pass

        while viewer.is_running():
            t = float(data.time)
            s = _triangular_wave_0_1_0(t, float(args.period_s))
            pos = ((1.0 - s) * a + s * b).astype(np.float32)

            data.qpos[qpos_adr + 0 : qpos_adr + 3] = pos  # type: ignore[index]
            if abs(roll_rate_rad_s) > 0.0:
                roll_rad = (roll_rate_rad_s * t) % (2.0 * math.pi)
                mujoco.mju_axisAngle2Quat(quat_roll, axis, float(roll_rad))
                data.qpos[qpos_adr + 3 : qpos_adr + 7] = quat_roll  # type: ignore[index]
            else:
                data.qpos[qpos_adr + 3 : qpos_adr + 7] = quat_identity  # type: ignore[index]
            data.qvel[:] = 0.0

            mujoco.mj_forward(model, data)
            viewer.sync()

            # Advance time so the trajectory animates even if everything else is kinematic.
            data.time = t + dt


if __name__ == "__main__":
    main()
