from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np


def _normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        return v * 0.0
    return v / n


def _mj_freecam_w2c_approx(lookat: np.ndarray, distance: float, azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    """
    Approximate MuJoCo free-camera to a COLMAP/3DGS-style W2C matrix.

    Camera convention used for 3DGS (COLMAP-like):
    - x: right
    - y: down
    - z: forward

    NOTE: MuJoCo's internal orbit conventions can differ by sign/axis.
    If the background looks mirrored/rotated, tweak the azimuth/elevation signs
    or swap axes here.
    """
    az = math.radians(float(azimuth_deg))
    el = math.radians(float(elevation_deg))

    # Orbit around lookat.
    cam_pos = lookat + float(distance) * np.array(
        [math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)], dtype=np.float32
    )

    # Build axes so that +z points forward (towards lookat), +y points down.
    z_fwd = _normalize(lookat - cam_pos)
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    x_right = _normalize(np.cross(z_fwd, world_up))
    y_down = np.cross(z_fwd, x_right)

    R_wc = np.column_stack([x_right, y_down, z_fwd]).astype(np.float32)  # cam->world
    R_cw = R_wc.T  # world->cam
    t = (-R_cw @ cam_pos).astype(np.float32)

    w2c = np.eye(4, dtype=np.float32)
    w2c[:3, :3] = R_cw
    w2c[:3, 3] = t
    return w2c


def _fovx_from_fovy(fovy_rad: float, w: int, h: int) -> float:
    return 2.0 * math.atan(math.tan(fovy_rad * 0.5) * (float(w) / float(h)))


def _linearize_depth_from_gl(depthbuf: np.ndarray, znear: float, zfar: float) -> np.ndarray:
    # OpenGL depth buffer (0..1) -> linear depth along camera Z.
    z_ndc = 2.0 * depthbuf - 1.0
    return (2.0 * znear * zfar) / (zfar + znear - z_ndc * (zfar - znear))


def main() -> None:
    ap = argparse.ArgumentParser(description="MuJoCo freecam + 3DGS background compositing (with occlusion).")
    ap.add_argument(
        "--mjcf",
        default=str(Path(__file__).resolve().parents[1] / "assets" / "mujoco_minimal_foreground.xml"),
        type=str,
        help="Foreground-only MJCF.",
    )
    ap.add_argument("--connect", default="tcp://127.0.0.1:5555", type=str, help="ZMQ connect address (REQ socket).")
    ap.add_argument("--width", default=1280, type=int)
    ap.add_argument("--height", default=720, type=int)
    ap.add_argument("--steps_per_frame", default=1, type=int)
    ap.add_argument("--eps_invdepth", default=1e-3, type=float, help="Invdepth threshold to reduce z-fighting.")
    ap.add_argument("--show_debug", action="store_true", help="Show occlusion mask and depth debug windows.")
    args = ap.parse_args()

    try:
        import cv2  # type: ignore
        import mujoco  # type: ignore
        import mujoco.viewer  # type: ignore
        import zmq  # type: ignore
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            "Missing deps. Create a separate env and install: mujoco, opencv-python, numpy, pyzmq.\n"
            f"Import error: {e!r}"
        )

    mjcf_path = Path(args.mjcf).resolve()
    if not mjcf_path.exists():
        raise SystemExit(f"--mjcf not found: {mjcf_path}")

    model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    data = mujoco.MjData(model)
    # Ensure derived fields (geom poses, etc.) are initialized before first render.
    mujoco.mj_forward(model, data)

    # Read render parameters from the model (keep explicit in MJCF for reproducibility).
    # Field names differ slightly across mujoco-python versions; fall back to defaults.
    fovy_deg = 60.0
    try:
        fovy_deg = float(model.vis.global_.fovy)
    except Exception:
        try:
            # Some bindings might expose `model.vis.global` (keyword in Python),
            # so use getattr to avoid syntax errors.
            fovy_deg = float(getattr(model.vis, "global").fovy)  # type: ignore[attr-defined]
        except Exception:
            pass

    znear = 0.2
    zfar = 200.0
    try:
        # MuJoCo stores znear/zfar as fractions of model.stat.extent; convert to metric.
        extent = float(model.stat.extent)
        znear = float(model.vis.map.znear) * extent
        zfar = float(model.vis.map.zfar) * extent
    except Exception:
        pass

    fovy_rad = math.radians(fovy_deg)
    fovx_rad = _fovx_from_fovy(fovy_rad, args.width, args.height)

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REQ)
    sock.connect(args.connect)
    print(f"[mj_freecam] connected to 3DGS server at {args.connect}")

    renderer = mujoco.Renderer(model, width=args.width, height=args.height)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("[mj_freecam] drag in MuJoCo viewer to move the free camera; output shown in OpenCV window.")
        last = time.time()
        while viewer.is_running():
            for _ in range(max(1, int(args.steps_per_frame))):
                mujoco.mj_step(model, data)

            viewer.sync()

            # Read MuJoCo free camera state.
            cam = viewer.cam
            lookat = np.array(cam.lookat, dtype=np.float32)
            distance = float(cam.distance)
            azimuth = float(cam.azimuth)
            elevation = float(cam.elevation)

            # Approximate W2C for 3DGS renderer.
            w2c = _mj_freecam_w2c_approx(lookat, distance, azimuth, elevation)

            # Request background render.
            req = {
                "w": int(args.width),
                "h": int(args.height),
                "fovx": float(fovx_rad),
                "fovy": float(fovy_rad),
                "znear": float(znear),
                "zfar": float(zfar),
            }
            sock.send_multipart([json.dumps(req).encode("utf-8"), w2c.astype(np.float32).tobytes(order="C")])
            resp_parts = sock.recv_multipart()
            if len(resp_parts) != 3:
                raise RuntimeError(f"Bad response from server: expected 3 parts, got {len(resp_parts)}")
            _meta = json.loads(resp_parts[0].decode("utf-8"))
            bg_rgb = np.frombuffer(resp_parts[1], dtype=np.uint8).reshape(args.height, args.width, 3)
            bg_invdepth = np.frombuffer(resp_parts[2], dtype=np.float32).reshape(args.height, args.width)

            # Render MuJoCo foreground from the same free camera.
            # Many mujoco-python versions accept camera=cam (mjvCamera instance).
            try:
                renderer.update_scene(data, camera=cam)
            except TypeError:
                # Fallback: render default camera (won't match freecam; use only for debugging).
                renderer.update_scene(data)

            # Render RGB.
            try:
                renderer.disable_depth_rendering()
            except Exception:
                pass
            try:
                renderer.disable_segmentation_rendering()
            except Exception:
                pass
            fg_rgb = renderer.render()

            # Render depth: MuJoCo APIs differ across versions.
            try:
                fg_depth = renderer.render(depth=True)  # type: ignore[call-arg]
            except TypeError:
                # MuJoCo>=3 uses a mode switch instead of a render(depth=...) kwarg.
                if not hasattr(renderer, "enable_depth_rendering"):
                    raise
                renderer.enable_depth_rendering()
                fg_depth = renderer.render()
                renderer.disable_depth_rendering()

            # Convert fg_depth to linear z and invdepth.
            fg_depth = np.asarray(fg_depth)
            if fg_depth.ndim != 2:
                fg_depth = fg_depth[..., 0]

            if float(np.nanmax(fg_depth)) <= 1.0 + 1e-3:
                # Likely OpenGL depth buffer (0..1).
                fg_depthbuf = fg_depth.astype(np.float32)
                has_fg = fg_depthbuf < 1.0 - 1e-6
                fg_z = _linearize_depth_from_gl(fg_depthbuf, znear, zfar)
            else:
                # Likely already linear depth (distance). Invalid pixels may be inf/0.
                fg_z = fg_depth.astype(np.float32)
                # In MuJoCo>=3 depth rendering, background pixels are typically set to the far plane.
                far_eps = max(1e-3, 1e-6 * float(zfar))
                has_fg = np.isfinite(fg_z) & (fg_z > 0) & (fg_z < float(zfar) - far_eps)

            fg_invdepth = np.zeros_like(fg_z, dtype=np.float32)
            fg_invdepth[has_fg] = 1.0 / np.maximum(fg_z[has_fg], 1e-6)

            bg_has = bg_invdepth > 0
            vis_fg = has_fg & (~bg_has | (fg_invdepth > bg_invdepth + float(args.eps_invdepth)))
            occ_fg = has_fg & ~vis_fg

            out = bg_rgb.copy()
            out[vis_fg] = fg_rgb[vis_fg]

            cv2.imshow("mj+3dgs composite", out[..., ::-1])  # RGB->BGR
            if args.show_debug:
                occ_u8 = (occ_fg.astype(np.uint8) * 255)
                cv2.imshow("occ_fg (white=occluded by 3DGS)", occ_u8)

                # Depth debug (normalized for display).
                disp_bg = bg_invdepth.copy()
                disp_bg = np.clip(disp_bg, 0, np.percentile(disp_bg, 99) if disp_bg.max() > 0 else 1.0)
                disp_bg = (disp_bg / (disp_bg.max() + 1e-6) * 255).astype(np.uint8)
                cv2.imshow("bg_invdepth (3DGS)", disp_bg)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

            # Basic frame pacing (optional).
            now = time.time()
            last = now

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
