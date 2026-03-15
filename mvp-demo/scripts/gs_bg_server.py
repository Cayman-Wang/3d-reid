from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _add_gs_repo_to_syspath(gs_repo: Path) -> None:
    gs_repo = gs_repo.resolve()
    if not gs_repo.exists():
        raise SystemExit(f"--gs_repo does not exist: {gs_repo}")
    sys.path.insert(0, str(gs_repo))


def _find_latest_iteration(point_cloud_dir: Path) -> int:
    iters: list[int] = []
    if not point_cloud_dir.exists():
        raise SystemExit(f"point_cloud dir not found: {point_cloud_dir}")
    for p in point_cloud_dir.iterdir():
        if not p.is_dir():
            continue
        name = p.name
        if not name.startswith("iteration_"):
            continue
        try:
            iters.append(int(name.split("_", 1)[1]))
        except Exception:
            continue
    if not iters:
        raise SystemExit(f"No iteration_* folders under: {point_cloud_dir}")
    return max(iters)


def _parse_w2c(mat_bytes: bytes) -> np.ndarray:
    m = np.frombuffer(mat_bytes, dtype=np.float32)
    if m.size != 16:
        raise ValueError(f"W2C must have 16 float32 values, got {m.size}")
    return m.reshape(4, 4)


@dataclass
class _Pipe:
    convert_SHs_python: bool = False
    compute_cov3D_python: bool = False
    debug: bool = False
    antialiasing: bool = False


class _MiniCam:
    # Minimal camera wrapper matching gaussian_renderer.render expectations.
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        import torch

        self.image_width = int(width)
        self.image_height = int(height)
        self.FoVy = float(fovy)
        self.FoVx = float(fovx)
        self.znear = float(znear)
        self.zfar = float(zfar)
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]


def main() -> None:
    default_gs_repo = Path(__file__).resolve().parents[1] / "third_party" / "gaussian-splatting"

    ap = argparse.ArgumentParser(description="3DGS background render server (returns RGB + invdepth) via ZMQ.")
    ap.add_argument("--gs_repo", default=str(default_gs_repo), type=str)
    ap.add_argument(
        "--model_dir",
        default=str(Path(__file__).resolve().parents[1] / "assets" / "tiyuzhongxin"),
        type=str,
        help="3DGS model dir (contains point_cloud/iteration_*/point_cloud.ply).",
    )
    ap.add_argument("--iteration", default=-1, type=int, help="-1=latest")
    ap.add_argument("--bind", default="tcp://127.0.0.1:5555", type=str, help="ZMQ bind address (REP socket).")
    ap.add_argument("--sh_degree", default=3, type=int)
    ap.add_argument("--white_bg", action="store_true", help="Use white background instead of black.")
    ap.add_argument("--antialiasing", action="store_true", help="Enable rasterizer antialiasing.")
    args = ap.parse_args()

    gs_repo = Path(args.gs_repo).resolve()
    _add_gs_repo_to_syspath(gs_repo)

    import torch  # noqa: E402
    import zmq  # noqa: E402
    from gaussian_renderer import GaussianModel, render  # noqa: E402
    from utils.graphics_utils import getProjectionMatrix  # noqa: E402

    model_dir = Path(args.model_dir).resolve()
    point_cloud_dir = model_dir / "point_cloud"
    iteration = int(args.iteration)
    if iteration == -1:
        iteration = _find_latest_iteration(point_cloud_dir)
    ply_path = point_cloud_dir / f"iteration_{iteration}" / "point_cloud.ply"
    if not ply_path.exists():
        raise SystemExit(f"point_cloud.ply not found: {ply_path}")

    print(f"[gs_bg_server] loading gaussians: {ply_path}")
    gaussians = GaussianModel(int(args.sh_degree))
    gaussians.load_ply(str(ply_path), use_train_test_exp=False)

    pipe = _Pipe(antialiasing=bool(args.antialiasing))
    bg_color = [1.0, 1.0, 1.0] if args.white_bg else [0.0, 0.0, 0.0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REP)
    sock.bind(args.bind)
    print(f"[gs_bg_server] listening on {args.bind}")

    while True:
        parts = sock.recv_multipart()
        if len(parts) != 2:
            sock.send_string("ERR: expected 2-part message (json, w2c)")
            continue

        t0 = time.time()
        try:
            meta = json.loads(parts[0].decode("utf-8"))
            w2c = _parse_w2c(parts[1])
            w = int(meta["w"])
            h = int(meta["h"])
            fovx = float(meta["fovx"])
            fovy = float(meta["fovy"])
            znear = float(meta["znear"])
            zfar = float(meta["zfar"])
        except Exception as e:
            sock.send_string(f"ERR: bad request: {e!r}")
            continue

        with torch.no_grad():
            # Follow graphdeco convention: pass transposed matrices to match column-major CUDA kernels.
            world_view = torch.tensor(w2c, dtype=torch.float32, device="cuda").transpose(0, 1)
            proj = getProjectionMatrix(znear=znear, zfar=zfar, fovX=fovx, fovY=fovy).transpose(0, 1).cuda()
            full = world_view @ proj
            cam = _MiniCam(
                width=w,
                height=h,
                fovy=fovy,
                fovx=fovx,
                znear=znear,
                zfar=zfar,
                world_view_transform=world_view,
                full_proj_transform=full,
            )

            out = render(cam, gaussians, pipe, background)
            rgb_t = out["render"]  # (3,H,W) float
            invdepth_t = out["depth"]  # (1,H,W) float, expected invdepth

            rgb_u8 = (rgb_t.clamp(0, 1) * 255.0).to(torch.uint8).permute(1, 2, 0).contiguous().cpu().numpy()
            if invdepth_t.ndim == 3 and invdepth_t.shape[0] == 1:
                invdepth_np = invdepth_t[0].contiguous().cpu().numpy().astype(np.float32, copy=False)
            else:
                invdepth_np = invdepth_t.contiguous().cpu().numpy().astype(np.float32, copy=False)

        resp_meta = {
            "w": w,
            "h": h,
            "rgb_shape": [h, w, 3],
            "rgb_dtype": "uint8",
            "invdepth_shape": [h, w],
            "invdepth_dtype": "float32",
            "iteration": iteration,
            "render_ms": (time.time() - t0) * 1000.0,
        }
        sock.send_multipart(
            [
                json.dumps(resp_meta).encode("utf-8"),
                rgb_u8.tobytes(order="C"),
                invdepth_np.tobytes(order="C"),
            ]
        )


if __name__ == "__main__":
    main()

