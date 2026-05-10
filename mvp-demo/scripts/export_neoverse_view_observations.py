from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from neoverse_fused_utils import (
    RuntimeConfig,
    alpha_to_frames,
    bundle_camera_arrays,
    build_gaussians,
    colorize_depth_sequence,
    load_bundle,
    load_reconstructor,
    parse_torch_dtype,
    repo_root,
    resolve_repo_relative_path,
    move_gaussians_to_device,
    render_sequence,
    rgb_to_frames,
    save_video_exact_size,
    to_rel,
    write_json,
)


DEFAULT_CAMS = ["cam0", "cam1", "cam2"]


def _save_uint8_png(path: Path, frame: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(frame, dtype=np.uint8)).save(path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Render per-camera NeoVerse observations from isolated reconstruction bundles.")
    ap.add_argument("--scene_dir", required=True, type=str)
    ap.add_argument("--cams", default="cam0,cam1,cam2", type=str)
    ap.add_argument("--bundle_root", default="mvp-demo/output/neoverse_fused", type=str)
    ap.add_argument("--neoverse_repo", default="third_party/NeoVerse", type=str)
    ap.add_argument("--reconstructor_path", default="third_party/NeoVerse/models/NeoVerse/reconstructor.ckpt", type=str)
    ap.add_argument("--device", default="cuda", type=str)
    ap.add_argument("--torch_dtype", default="float16", type=str)
    ap.add_argument("--camera_source", default="rendered", choices=["rendered", "predicted_camera"])
    ap.add_argument("--fps", default=16, type=int)
    ap.add_argument("--out_root", default="mvp-demo/output/neoverse_fused", type=str)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    repo = repo_root()
    scene_dir = Path(str(args.scene_dir)).resolve()
    if not scene_dir.exists():
        raise SystemExit(f"Missing --scene_dir: {scene_dir}")

    cams = [c.strip() for c in str(args.cams).split(",") if c.strip()]
    if cams != DEFAULT_CAMS:
        raise SystemExit(f"This first version only supports cams={DEFAULT_CAMS}. Got: {cams}.")

    bundle_root = resolve_repo_relative_path(str(args.bundle_root), repo)
    out_root = resolve_repo_relative_path(str(args.out_root), repo)
    neoverse_repo = resolve_repo_relative_path(str(args.neoverse_repo), repo)
    reconstructor_path = resolve_repo_relative_path(str(args.reconstructor_path), repo)

    if not neoverse_repo.exists():
        raise SystemExit(f"Missing --neoverse_repo: {neoverse_repo}")
    if not reconstructor_path.exists():
        raise SystemExit(f"Missing --reconstructor_path: {reconstructor_path}")

    scene_id = scene_dir.name
    scene_out_root = out_root / scene_id / "observations"
    scene_out_root.mkdir(parents=True, exist_ok=True)

    report_rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []

    reconstructor = load_reconstructor(
        neoverse_repo=neoverse_repo,
        reconstructor_path=reconstructor_path,
        device=str(args.device),
        torch_dtype=parse_torch_dtype(str(args.torch_dtype)),
    )
    rasterizer = reconstructor.gs_renderer.rasterizer

    for cam_id in cams:
        bundle_path = bundle_root / scene_id / "per_camera" / cam_id / "reconstruction_bundle.pt"
        if not bundle_path.exists():
            failures.append({"cam_id": cam_id, "status": "missing_bundle", "bundle_path": bundle_path.as_posix()})
            continue

        cam_out = scene_out_root / cam_id
        if cam_out.exists() and bool(args.overwrite):
            shutil.rmtree(cam_out)
        cam_out.mkdir(parents=True, exist_ok=True)

        try:
            bundle = load_bundle(bundle_path)
            manifest = bundle.get("source_manifest") or {}
            views = list(manifest.get("views") or bundle.get("source_views_meta") or [])
            if not views:
                raise SystemExit(f"Bundle missing source views: {bundle_path}")

            _, camera_poses, camera_intrs, timestamps = bundle_camera_arrays(bundle, str(args.camera_source))
            camera_c2w = camera_poses[0].detach().cpu().float()
            camera_w2c = torch.linalg.inv(camera_c2w)
            gaussians = build_gaussians(bundle, neoverse_repo)
            gaussians = move_gaussians_to_device(gaussians, str(args.device), parse_torch_dtype(str(args.torch_dtype)))
            rgb, depth, alpha = render_sequence(
                rasterizer=rasterizer,
                gaussians=gaussians,
                render_viewmats=camera_w2c,
                render_Ks=camera_intrs[0],
                render_timestamps=timestamps[0],
                width=int(manifest.get("input_resolution", {}).get("width", 280)),
                height=int(manifest.get("input_resolution", {}).get("height", 168)),
                device=str(args.device),
                torch_dtype=parse_torch_dtype(str(args.torch_dtype)),
            )

            rgb_frames = rgb_to_frames(rgb)
            depth_frames = colorize_depth_sequence(depth)
            alpha_frames = alpha_to_frames(alpha)

            rgb_dir = cam_out / "rgb"
            depth_dir = cam_out / "depth"
            alpha_dir = cam_out / "alpha"
            rgb_dir.mkdir(parents=True, exist_ok=True)
            depth_dir.mkdir(parents=True, exist_ok=True)
            alpha_dir.mkdir(parents=True, exist_ok=True)

            index_rows: list[dict[str, object]] = []
            for view_idx, item in enumerate(views):
                scene_stem = str(item.get("scene_stem") or f"{view_idx:06d}")
                logical_t_idx = int(item.get("logical_t_idx", view_idx))
                rgb_path = rgb_dir / f"{scene_stem}.png"
                depth_path = depth_dir / f"{scene_stem}.npy"
                alpha_path = alpha_dir / f"{scene_stem}.npy"

                _save_uint8_png(rgb_path, rgb_frames[view_idx])
                np.save(depth_path, np.asarray(depth[0, view_idx].detach().cpu().numpy(), dtype=np.float32))
                np.save(alpha_path, np.asarray(alpha[0, view_idx].detach().cpu().numpy(), dtype=np.float32))

                row = {
                    "cam_id": cam_id,
                    "scene_stem": scene_stem,
                    "logical_t_idx": logical_t_idx,
                    "rgb_path": to_rel(rgb_path, scene_out_root),
                    "depth_path": to_rel(depth_path, scene_out_root),
                    "alpha_path": to_rel(alpha_path, scene_out_root),
                    "input_variant": str(item.get("input_variant") or manifest.get("input_variant") or "full_frame"),
                    "crop_applied": int(bool(item.get("crop_applied", False))),
                    "crop_box_xyxy": json.dumps(item.get("crop_box_xyxy"), ensure_ascii=False),
                    "crop_mask_source": str(item.get("crop_mask_source") or ""),
                }
                index_rows.append(row)
                report_rows.append(row)

            index_path = cam_out / "index.csv"
            with index_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "cam_id",
                        "scene_stem",
                        "logical_t_idx",
                        "rgb_path",
                        "depth_path",
                        "alpha_path",
                        "render_K",
                        "render_c2w",
                        "render_w2c",
                        "width",
                        "height",
                        "resize_mode",
                        "camera_source",
                        "input_variant",
                        "crop_applied",
                        "crop_box_xyxy",
                        "crop_mask_source",
                    ],
                )
                writer.writeheader()
                for view_idx, row in enumerate(index_rows):
                    row["render_K"] = json.dumps(np.asarray(camera_intrs[0, view_idx].detach().cpu().numpy(), dtype=np.float32).tolist(), ensure_ascii=False)
                    row["render_c2w"] = json.dumps(np.asarray(camera_c2w[view_idx].numpy(), dtype=np.float32).tolist(), ensure_ascii=False)
                    row["render_w2c"] = json.dumps(np.asarray(camera_w2c[view_idx].numpy(), dtype=np.float32).tolist(), ensure_ascii=False)
                    row["width"] = int(manifest.get("input_resolution", {}).get("width", 280))
                    row["height"] = int(manifest.get("input_resolution", {}).get("height", 168))
                    row["resize_mode"] = str(manifest.get("resize_mode") or "center_crop")
                    row["camera_source"] = str(args.camera_source)
                    writer.writerow(row)

            cam_meta = {
                "scene_id": scene_id,
                "cam_id": cam_id,
                "bundle": bundle_path.as_posix(),
                "camera_source": str(args.camera_source),
                "width": int(manifest.get("input_resolution", {}).get("width", 280)),
                "height": int(manifest.get("input_resolution", {}).get("height", 168)),
                "num_frames": len(index_rows),
                "outputs": {
                    "index_csv": to_rel(index_path, scene_out_root),
                    "rgb_dir": to_rel(rgb_dir, scene_out_root),
                    "depth_dir": to_rel(depth_dir, scene_out_root),
                    "alpha_dir": to_rel(alpha_dir, scene_out_root),
                },
                "source_manifest": manifest,
            }
            write_json(cam_out / "observation_meta.json", cam_meta)

            # Optional lightweight preview for debugging.
            save_video_exact_size(rgb_frames, cam_out / "rgb_preview.mp4", fps=int(args.fps))
            save_video_exact_size(depth_frames, cam_out / "depth_preview.mp4", fps=int(args.fps))
            save_video_exact_size(alpha_frames, cam_out / "alpha_preview.mp4", fps=int(args.fps))
        except Exception as exc:
            failures.append({"cam_id": cam_id, "status": "error", "error_type": type(exc).__name__, "error": str(exc)})

    report = {
        "scene_id": scene_id,
        "scene_dir": scene_dir.as_posix(),
        "bundle_root": bundle_root.as_posix(),
        "num_rows": len(report_rows),
        "num_failures": len(failures),
        "failures": failures,
    }
    write_json(scene_out_root / "observations_report.json", report)

    index_path = scene_out_root / "index.csv"
    with index_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "cam_id",
                "scene_stem",
                "logical_t_idx",
                "rgb_path",
                "depth_path",
                "alpha_path",
                "render_K",
                "render_c2w",
                "render_w2c",
                "width",
                        "height",
                        "resize_mode",
                        "camera_source",
                        "input_variant",
                        "crop_applied",
                        "crop_box_xyxy",
                        "crop_mask_source",
                    ],
                )
        writer.writeheader()
        for row in report_rows:
            writer.writerow(row)

    print(f"Wrote observations to: {scene_out_root}")
    print(f"Wrote: {index_path}")
    print(f"Wrote: {scene_out_root / 'observations_report.json'}")

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
