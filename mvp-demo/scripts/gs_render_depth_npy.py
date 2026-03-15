from __future__ import annotations

import csv
import json
import os
import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore[assignment]


def _add_gs_repo_to_syspath(gs_repo: Path) -> None:
    gs_repo = gs_repo.resolve()
    if not gs_repo.exists():
        raise SystemExit(f"--gs_repo does not exist: {gs_repo}")
    sys.path.insert(0, str(gs_repo))


def _safe_stem(name: str, fallback: str) -> str:
    try:
        return Path(str(name)).stem
    except Exception:
        return fallback


def _iter_views(views, desc: str):
    if tqdm is None:
        return views
    return tqdm(views, desc=desc)


def render_depth_set(
    model_path,
    name,
    iteration,
    views,
    gaussians,
    pipeline,
    background,
    train_test_exp,
    separate_sh,
    render_func,
    out_dir: Path | None,
    index_rows: list[dict],
):
    if out_dir is not None:
        depth_dir = out_dir
    else:
        depth_dir = Path(model_path) / name / f"ours_{iteration}" / "depth_npy"
    depth_dir.mkdir(parents=True, exist_ok=True)

    for idx, view in enumerate(_iter_views(views, desc=f"Depth rendering ({name})")):
        out = render_func(view, gaussians, pipeline, background, use_trained_exp=train_test_exp, separate_sh=separate_sh)
        depth = out["depth"].detach().cpu().numpy().astype(np.float32)
        if depth.ndim == 3 and depth.shape[0] == 1:
            depth = depth[0]

        image_name = (
            getattr(view, "image_name", None)
            or getattr(view, "image_path", None)
            or getattr(view, "name", None)
        )
        stem = _safe_stem(str(image_name), fallback=f"{idx:05d}")
        out_path = depth_dir / f"{stem}.npy"
        np.save(str(out_path), depth)
        index_rows.append(
            {
                "stem": stem,
                "split": name,
                "filename": out_path.name,
            }
        )


def main() -> None:
    # Parse only --gs_repo here; the rest of args follow the upstream 3DGS style (e.g. `-m <model_dir>`).
    pre = ArgumentParser(add_help=False)
    pre.add_argument("--gs_repo", required=True, type=str, help="Path to graphdeco/gaussian-splatting repo")
    known, remaining = pre.parse_known_args()

    gs_repo = Path(known.gs_repo)
    _add_gs_repo_to_syspath(gs_repo)

    from arguments import ModelParams, PipelineParams, get_combined_args  # noqa: E402
    from gaussian_renderer import GaussianModel, render  # noqa: E402
    from scene import Scene  # noqa: E402
    from utils.general_utils import safe_state  # noqa: E402

    parser = ArgumentParser(description="(internal) 3DGS renderer args")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int, help="Which iteration to load; -1=latest")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--out_dir",
        default="",
        type=str,
        help="Optional: write all depth maps to this folder (aligned by image stem). When set, train/test are merged.",
    )

    old_argv = sys.argv[:]
    try:
        sys.argv = [old_argv[0]] + remaining
        args = get_combined_args(parser)
    finally:
        sys.argv = old_argv

    safe_state(args.quiet)

    dataset = model.extract(args)
    pipe = pipeline.extract(args)

    try:
        from diff_gaussian_rasterization import SparseGaussianAdam  # noqa: F401

        sparse_adam_available = True
    except Exception:
        sparse_adam_available = False

    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        out_dir = Path(args.out_dir).resolve() if str(getattr(args, "out_dir", "")).strip() else None
        index_rows: list[dict] = []

        if not args.skip_train:
            render_depth_set(
                dataset.model_path,
                "train",
                scene.loaded_iter,
                scene.getTrainCameras(),
                gaussians,
                pipe,
                background,
                dataset.train_test_exp,
                sparse_adam_available,
                render,
                out_dir,
                index_rows,
            )
        if not args.skip_test:
            render_depth_set(
                dataset.model_path,
                "test",
                scene.loaded_iter,
                scene.getTestCameras(),
                gaussians,
                pipe,
                background,
                dataset.train_test_exp,
                sparse_adam_available,
                render,
                out_dir,
                index_rows,
            )

        if out_dir is not None:
            # Write small metadata files so downstream can reliably consume aligned depths.
            index_rows_sorted = sorted(index_rows, key=lambda r: (r.get("stem", ""), r.get("split", "")))
            meta = {
                "iteration": int(scene.loaded_iter),
                "model_dir": str(Path(dataset.model_path).resolve()),
                "out_dir": str(out_dir),
                "count": int(len(index_rows_sorted)),
            }
            (out_dir / "depth_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            with (out_dir / "depth_index.csv").open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["stem", "split", "filename"])
                w.writeheader()
                w.writerows(index_rows_sorted)


if __name__ == "__main__":
    main()
