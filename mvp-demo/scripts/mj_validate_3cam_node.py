from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def _parse_vec3(s: str) -> np.ndarray:
    toks = [t for t in str(s).strip().split() if t]
    if len(toks) != 3:
        raise ValueError('Expected 3 floats, e.g. "0 1 0".')
    return np.array([float(toks[0]), float(toks[1]), float(toks[2])], dtype=np.float32)


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= 0:
        raise ValueError("Zero-length vector.")
    return (v / n).astype(np.float32)


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    a = _normalize(a)
    b = _normalize(b)
    d = float(np.clip(np.dot(a, b), -1.0, 1.0))
    return float(math.degrees(math.acos(d)))


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


def _camera_T_node_from_cam_cv(model, data, *, node_body_id: int, cam_id: int) -> np.ndarray:
    # MuJoCo camera axes are OpenGL-like: x right, y up, and it "looks" along -Z.
    # Export in a CV-friendly camera convention: x right, y down, z forward.
    C_mj_from_cv = np.diag([1.0, -1.0, -1.0]).astype(np.float32)  # cv -> mujoco camera axes

    node_pos_w = np.asarray(data.xpos[node_body_id], dtype=np.float32).copy()
    node_R_w_n = np.asarray(data.xmat[node_body_id], dtype=np.float32).reshape(3, 3).copy()
    T_w_n = _T_from_Rt(node_R_w_n, node_pos_w)
    T_n_w = _invert_T(T_w_n)

    cam_pos_w = np.asarray(data.cam_xpos[cam_id], dtype=np.float32).copy()
    R_w_c_mj = np.asarray(data.cam_xmat[cam_id], dtype=np.float32).reshape(3, 3).copy()
    R_w_c_cv = (R_w_c_mj @ C_mj_from_cv).astype(np.float32)
    T_w_c = _T_from_Rt(R_w_c_cv, cam_pos_w)
    return (T_n_w @ T_w_c).astype(np.float32)


def _default_camera_names(node_body: str) -> List[str]:
    return [f"{node_body}_cam0", f"{node_body}_cam1", f"{node_body}_cam2"]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Validate a MuJoCo 3-camera node rig (parallel optical axes): baselines, forward vectors, and layout."
        )
    )
    ap.add_argument(
        "--mjcf",
        default=str(Path(__file__).resolve().parents[1] / "assets" / "mujoco_3cam_node_parallel.xml"),
        type=str,
        help="MJCF with a node body and cameras.",
    )
    ap.add_argument("--node_body", default="node01", type=str, help="Body name used as node frame.")
    ap.add_argument(
        "--camera_names",
        default="",
        type=str,
        help='Comma-separated MJCF camera names. Default: "<node_body>_cam0,<node_body>_cam1,<node_body>_cam2".',
    )
    ap.add_argument(
        "--expected_fwd_node",
        default="0 1 0",
        type=str,
        help='Expected camera forward direction in node frame (CV convention), e.g. "0 1 0".',
    )
    ap.add_argument("--max_baseline_m", default=1.0, type=float, help="Max allowed baseline between any two cameras.")
    ap.add_argument("--fwd_tol_deg", default=1.0, type=float, help="Tolerance to expected forward direction.")
    ap.add_argument("--parallel_tol_deg", default=0.5, type=float, help="Tolerance between any two camera forwards.")
    ap.add_argument(
        "--plane",
        default="xz",
        choices=["xz", "xy", "yz", "none"],
        help="Optional layout check: cameras should lie in the specified plane in node frame.",
    )
    ap.add_argument("--plane_tol_m", default=1e-6, type=float, help="Tolerance for the plane check (meters).")
    ap.add_argument("--json", action="store_true", help="Also print a machine-readable JSON report to stdout.")
    args = ap.parse_args()

    try:
        import mujoco  # type: ignore
    except Exception as e:  # pragma: no cover
        raise SystemExit(f"Missing deps. Install (suggested): mujoco, numpy.\nImport error: {e!r}")

    mjcf_path = Path(args.mjcf).resolve()
    if not mjcf_path.exists():
        raise SystemExit(f"--mjcf not found: {mjcf_path}")

    node_body = str(args.node_body).strip()
    if not node_body:
        raise SystemExit("--node_body must be non-empty.")

    if str(args.camera_names).strip():
        cam_names = [t.strip() for t in str(args.camera_names).split(",") if t.strip()]
    else:
        cam_names = _default_camera_names(node_body)

    model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    node_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, node_body)
    if node_body_id < 0:
        raise SystemExit(f'Node body "{node_body}" not found in MJCF.')

    cams: List[Tuple[str, int]] = []
    for name in cam_names:
        cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
        if cid < 0:
            raise SystemExit(f'Camera "{name}" not found in MJCF.')
        cams.append((name, int(cid)))

    expected_fwd = _normalize(_parse_vec3(str(args.expected_fwd_node)))

    report: Dict[str, object] = {
        "mjcf": str(mjcf_path),
        "node_body": node_body,
        "expected_fwd_node": [float(x) for x in expected_fwd.tolist()],
        "cameras": {},
        "baselines_m": {},
        "angles_deg": {"to_expected": {}, "pairwise_parallel": {}},
        "checks": {},
    }

    # Per-camera pose in node frame (CV camera convention).
    cam_pos: Dict[str, np.ndarray] = {}
    cam_fwd: Dict[str, np.ndarray] = {}
    cam_R: Dict[str, np.ndarray] = {}
    for name, cid in cams:
        T_n_c = _camera_T_node_from_cam_cv(model, data, node_body_id=int(node_body_id), cam_id=int(cid))
        pos = T_n_c[:3, 3].copy()
        fwd = _normalize(T_n_c[:3, :3] @ np.array([0.0, 0.0, 1.0], dtype=np.float32))
        cam_pos[name] = pos
        cam_fwd[name] = fwd
        cam_R[name] = T_n_c[:3, :3].copy()

        report["cameras"][name] = {  # type: ignore[index]
            "pos_node_m": [float(x) for x in pos.tolist()],
            "fwd_node": [float(x) for x in fwd.tolist()],
            "T_node_from_cam": T_n_c.tolist(),
        }
        report["angles_deg"]["to_expected"][name] = _angle_deg(fwd, expected_fwd)  # type: ignore[index]

    # Baselines + parallel angles.
    max_baseline = 0.0
    max_parallel_angle = 0.0
    for i in range(len(cams)):
        for j in range(i + 1, len(cams)):
            ni = cams[i][0]
            nj = cams[j][0]
            b = float(np.linalg.norm(cam_pos[ni] - cam_pos[nj]))
            report["baselines_m"][f"{ni}-{nj}"] = b  # type: ignore[index]
            max_baseline = max(max_baseline, b)

            ang = _angle_deg(cam_fwd[ni], cam_fwd[nj])
            report["angles_deg"]["pairwise_parallel"][f"{ni}-{nj}"] = ang  # type: ignore[index]
            max_parallel_angle = max(max_parallel_angle, ang)

    # Optional: layout plane check (node frame).
    plane_ok = True
    plane_axis = {"xz": 1, "xy": 2, "yz": 0}.get(str(args.plane), None)
    if plane_axis is not None:
        tol = float(args.plane_tol_m)
        for name, _ in cams:
            if abs(float(cam_pos[name][plane_axis])) > tol:
                plane_ok = False
                break

    # Check forward vs expected + pairwise parallel.
    fwd_ok = True
    fwd_tol = float(args.fwd_tol_deg)
    for name, _ in cams:
        if float(report["angles_deg"]["to_expected"][name]) > fwd_tol:  # type: ignore[index]
            fwd_ok = False
            break

    parallel_ok = max_parallel_angle <= float(args.parallel_tol_deg)
    baseline_ok = max_baseline <= float(args.max_baseline_m)

    report["checks"] = {
        "baseline_ok": bool(baseline_ok),
        "max_baseline_m": float(max_baseline),
        "parallel_ok": bool(parallel_ok),
        "max_parallel_angle_deg": float(max_parallel_angle),
        "fwd_ok": bool(fwd_ok),
        "plane_ok": bool(plane_ok),
        "result": bool(baseline_ok and parallel_ok and fwd_ok and plane_ok),
    }

    # Human-readable report.
    print(f"mjcf: {mjcf_path}")
    print(f"node_body: {node_body}")
    print(f"expected_fwd_node: {expected_fwd.round(6).tolist()}")
    print("")
    print("cameras:")
    for name, _ in cams:
        pos = cam_pos[name]
        fwd = cam_fwd[name]
        ang = float(report["angles_deg"]["to_expected"][name])  # type: ignore[index]
        print(f"  - {name}: pos_node={pos.round(6).tolist()} fwd_node={fwd.round(6).tolist()} ang_to_expected={ang:.3f}deg")
    print("")
    print("baselines_m:")
    for k, v in sorted(report["baselines_m"].items()):  # type: ignore[union-attr]
        print(f"  - {k}: {float(v):.6f}")
    print(f"max_baseline_m: {max_baseline:.6f} (limit {float(args.max_baseline_m):.6f})")
    print("")
    print("pairwise_parallel_angles_deg:")
    for k, v in sorted(report["angles_deg"]["pairwise_parallel"].items()):  # type: ignore[union-attr]
        print(f"  - {k}: {float(v):.6f}")
    print(f"max_parallel_angle_deg: {max_parallel_angle:.6f} (limit {float(args.parallel_tol_deg):.6f})")
    print("")
    if str(args.plane) != "none":
        axis_name = {0: "x", 1: "y", 2: "z"}[int(plane_axis)]  # type: ignore[arg-type]
        print(f"plane_check: {args.plane} (|{axis_name}|<= {float(args.plane_tol_m):.6g}) -> {plane_ok}")
        print("")
    print(f"result: {bool(report['checks']['result'])}")  # type: ignore[index]

    if args.json:
        print("")
        print(json.dumps(report, indent=2, ensure_ascii=True))

    if not bool(report["checks"]["result"]):  # type: ignore[index]
        raise SystemExit(2)


if __name__ == "__main__":
    main()

