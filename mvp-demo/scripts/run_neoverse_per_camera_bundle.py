from __future__ import annotations

import argparse
import shutil
from datetime import datetime, timezone
from pathlib import Path

import torch

from neoverse_fused_utils import (
    RuntimeConfig,
    build_scene_glb,
    load_json,
    parse_torch_dtype,
    repo_root,
    resolve_repo_relative_path,
    run_reconstruction,
    serialize_splats,
    to_rel,
    write_json,
)


DEFAULT_CAMS = ["cam0", "cam1", "cam2"]


def main() -> None:
    ap = argparse.ArgumentParser(description="Run NeoVerse reconstructor separately for each camera and save isolated bundles.")
    ap.add_argument("--scene_dir", required=True, type=str)
    ap.add_argument("--cams", default="cam0,cam1,cam2", type=str)
    ap.add_argument("--neoverse_repo", default="third_party/NeoVerse", type=str)
    ap.add_argument("--reconstructor_path", default="third_party/NeoVerse/models/NeoVerse/reconstructor.ckpt", type=str)
    ap.add_argument("--device", default="cuda", type=str)
    ap.add_argument("--torch_dtype", default="float16", type=str)
    ap.add_argument("--enable_vram_management", action="store_true")
    ap.add_argument("--scene_mode", default="general", choices=["general", "static"], type=str)
    ap.add_argument("--height", default=168, type=int)
    ap.add_argument("--width", default=280, type=int)
    ap.add_argument("--num_frames", default=81, type=int, help="Number of frames sampled per camera; <=0 means using all frames")
    ap.add_argument("--resize_mode", choices=["center_crop", "resize"], default="center_crop", type=str)
    ap.add_argument("--input_variant", choices=["full_frame", "object_crop"], default="full_frame", type=str)
    ap.add_argument("--crop_padding", default=0.25, type=float)
    ap.add_argument("--crop_mask_source", choices=["auto", "masks_gt", "masks"], default="auto", type=str)
    ap.add_argument("--out_root", default="mvp-demo/output/neoverse_fused", type=str)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    repo = repo_root()
    scene_dir = Path(str(args.scene_dir)).resolve()
    if not scene_dir.exists():
        raise SystemExit(f"Missing --scene_dir: {scene_dir}")

    cams = [c.strip() for c in str(args.cams).split(",") if c.strip()]
    if not cams or any(cam_id not in DEFAULT_CAMS for cam_id in cams):
        raise SystemExit(f"Unsupported --cams: {cams}. Expected a non-empty subset of {DEFAULT_CAMS}.")

    neoverse_repo = resolve_repo_relative_path(str(args.neoverse_repo), repo)
    if not neoverse_repo.exists():
        raise SystemExit(f"Missing --neoverse_repo: {neoverse_repo}")

    reconstructor_path = resolve_repo_relative_path(str(args.reconstructor_path), repo)
    if not reconstructor_path.exists():
        raise SystemExit(f"Missing --reconstructor_path: {reconstructor_path}")

    out_root = resolve_repo_relative_path(str(args.out_root), repo)
    scene_id = scene_dir.name
    scene_root = out_root / scene_id / "per_camera"
    scene_root.mkdir(parents=True, exist_ok=True)

    failures: list[dict[str, object]] = []
    cam_summaries: list[dict[str, object]] = []

    for cam_id in cams:
        cam_root = scene_root / cam_id
        if cam_root.exists() and bool(args.overwrite):
            shutil.rmtree(cam_root)
        cam_root.mkdir(parents=True, exist_ok=True)

        started_at = datetime.now(timezone.utc).isoformat()
        try:
            runtime_stats, predictions, views_meta = run_reconstruction(
                scene_dir=scene_dir,
                cam_id=cam_id,
                neoverse_repo=neoverse_repo,
                reconstructor_path=reconstructor_path,
                cfg=RuntimeConfig(
                    device=str(args.device),
                    torch_dtype=parse_torch_dtype(str(args.torch_dtype)),
                    enable_vram_management=bool(args.enable_vram_management),
                ),
                target_width=int(args.width),
                target_height=int(args.height),
                resize_mode=str(args.resize_mode),
                scene_mode=str(args.scene_mode),
                num_frames=int(args.num_frames),
                input_variant=str(args.input_variant),
                crop_padding=float(args.crop_padding),
                crop_mask_source=str(args.crop_mask_source),
            )
            finished_at = datetime.now(timezone.utc).isoformat()
            sampling_meta = runtime_stats.get("sampling_meta") or {}

            bundle = {
                "schema_version": "neoverse_fused_per_camera_bundle_v1",
                "scene_id": scene_id,
                "cam_id": cam_id,
                "created_at_utc": finished_at,
                "splats_serialized": serialize_splats(predictions),
                "predicted_camera_intrinsics": predictions["camera_intrs"][0].detach().cpu(),
                "predicted_camera_cam2world": predictions["camera_poses"][0].detach().cpu(),
                "rendered_intrinsics": predictions["rendered_intrinsics"][0].detach().cpu(),
                "rendered_cam2world": predictions["rendered_extrinsics"][0].detach().cpu(),
                "rendered_timestamps": predictions["rendered_timestamps"][0].detach().cpu(),
                "source_manifest": {
                    "schema_version": "neoverse_fused_per_camera_manifest_v1",
                    "scene_id": scene_id,
                    "scene_dir": scene_dir.as_posix(),
                    "cam_id": cam_id,
                    "cams": [cam_id],
                    "rig_json": (scene_dir / "calib" / "rig.json").as_posix(),
                     "scene_mode": str(args.scene_mode),
                     "input_resolution": {"width": int(args.width), "height": int(args.height)},
                     "resize_mode": str(args.resize_mode),
                     "input_variant": str(args.input_variant),
                     "crop_padding": float(args.crop_padding),
                     "crop_mask_source": str(args.crop_mask_source),
                     "num_frames_requested": int(sampling_meta.get("num_frames_requested", args.num_frames)),
                     "num_frames_selected": int(sampling_meta.get("num_frames_selected", len(views_meta))),
                     "selected_frame_stems": sampling_meta.get("selected_frame_stems", [v.get("scene_stem") for v in views_meta]),
                    "frame_sampling_rule": sampling_meta.get("frame_sampling_rule", "uniform"),
                    "num_views": int(len(views_meta)),
                    "num_sync_steps": int(len(views_meta)),
                    "views": views_meta,
                },
                "source_views_meta": views_meta,
            }

            bundle_path = cam_root / "reconstruction_bundle.pt"
            torch.save(bundle, bundle_path)

            probe_meta = {
                "scene_id": scene_id,
                "cam_id": cam_id,
                "scene_dir": scene_dir.as_posix(),
                "reconstructor_path": reconstructor_path.as_posix(),
                "conditioning_mode": "rig_camera_priors",
                "geometry_anchor_mode": "rig_gtcamera",
                "model_loading_mode": "reconstructor_only",
                 "scene_mode": str(args.scene_mode),
                 "input_resolution": {"width": int(args.width), "height": int(args.height)},
                 "resize_mode": str(args.resize_mode),
                 "input_variant": str(args.input_variant),
                 "crop_padding": float(args.crop_padding),
                 "crop_mask_source": str(args.crop_mask_source),
                 "num_frames_requested": int(sampling_meta.get("num_frames_requested", args.num_frames)),
                 "num_frames_selected": int(sampling_meta.get("num_frames_selected", len(views_meta))),
                 "selected_frame_stems": sampling_meta.get("selected_frame_stems", [v.get("scene_stem") for v in views_meta]),
                "frame_sampling_rule": sampling_meta.get("frame_sampling_rule", "uniform"),
                "num_views": int(len(views_meta)),
                "started_at_utc": started_at,
                "finished_at_utc": finished_at,
                "runtime_sec": float(runtime_stats["runtime_sec"]),
                "device": str(args.device),
                "torch_dtype": str(parse_torch_dtype(str(args.torch_dtype))),
                "outputs": {
                    "reconstruction_bundle_pt": to_rel(bundle_path, repo),
                    "scene_glb": to_rel(cam_root / "scene.glb", repo),
                },
                "source_manifest": bundle["source_manifest"],
            }
            probe_meta_path = cam_root / "probe_meta.json"
            write_json(probe_meta_path, probe_meta)

            glb_path = build_scene_glb(predictions=predictions, out_dir=cam_root, neoverse_repo=neoverse_repo)
            cam_summary = {
                "cam_id": cam_id,
                "status": "ok",
                "bundle": bundle_path.as_posix(),
                "probe_meta": probe_meta_path.as_posix(),
                "scene_glb": glb_path.as_posix(),
                "runtime_sec": float(runtime_stats["runtime_sec"]),
            }
            cam_summaries.append(cam_summary)
            print(f"[ok] cam={cam_id} wrote {bundle_path}")
        except Exception as exc:
            failures.append({"cam_id": cam_id, "status": "error", "error_type": type(exc).__name__, "error": str(exc)})
            print(f"[warn] cam failed: {cam_id} error_type={type(exc).__name__} error={exc}")

    report_path = scene_root / "per_camera_run_report.json"
    write_json(
        report_path,
        {
            "scene_id": scene_id,
            "scene_dir": scene_dir.as_posix(),
            "cams": cams,
            "num_ok": len(cam_summaries),
            "num_failures": len(failures),
            "failures": failures,
            "results": cam_summaries,
        },
    )

    if failures:
        raise SystemExit(1)

    print(f"Wrote per-camera bundles to: {scene_root}")
    print(f"Wrote: {report_path}")


if __name__ == "__main__":
    main()
