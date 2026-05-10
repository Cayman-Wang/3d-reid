from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"Failed to read json: {path}\nError: {exc!r}")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _to_rel(path: Path, base: Path) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except Exception:
        return path.resolve().as_posix()


def _resolve_path(repo_root: Path, value: str) -> Path:
    p = Path(str(value))
    if not p.is_absolute():
        p = repo_root / p
    return p.resolve()


def _prepare_imports(dggt_repo: Path) -> None:
    p = str(dggt_repo.resolve())
    if p not in sys.path:
        sys.path.insert(0, p)


def _parse_torch_dtype(value: str, torch_module: Any):
    v = str(value).strip().lower()
    if v in {"float32", "fp32"}:
        return torch_module.float32
    if v in {"float16", "fp16"}:
        return torch_module.float16
    if v in {"bfloat16", "bf16"}:
        return torch_module.bfloat16
    raise SystemExit(f"Unsupported --torch_dtype: {value}")


def _pose_center_w2c(extr_w2c_3x4):
    import numpy as np

    R = extr_w2c_3x4[:, :3]
    t = extr_w2c_3x4[:, 3]
    center = -np.linalg.inv(R) @ t
    return center


def _build_pose_alignment_report(
    views: list[dict[str, Any]],
    pred_extr_w2c,
    input_extr_w2c,
) -> dict[str, Any]:
    import numpy as np

    report: dict[str, Any] = {
        "schema_version": "dggt_pose_alignment_report_v1",
        "num_views": int(len(views)),
        "per_camera": {},
    }

    cams = sorted({str(v["cam_id"]) for v in views})
    for cam in cams:
        idxs = [i for i, v in enumerate(views) if str(v["cam_id"]) == cam]
        pred_centers = np.stack([_pose_center_w2c(pred_extr_w2c[i]) for i in idxs], axis=0)
        input_centers = np.stack([_pose_center_w2c(input_extr_w2c[i]) for i in idxs], axis=0)

        center_rmse = float(np.sqrt(((pred_centers - input_centers) ** 2).sum(axis=1).mean()))
        center_mae = float(np.abs(pred_centers - input_centers).mean())

        report["per_camera"][cam] = {
            "count": int(len(idxs)),
            "pred_center_mean": pred_centers.mean(axis=0).round(6).tolist(),
            "input_center_mean": input_centers.mean(axis=0).round(6).tolist(),
            "center_rmse": center_rmse,
            "center_mae": center_mae,
        }

    return report


def _npz_save(path: Path, payload: dict[str, Any]) -> None:
    import numpy as np

    np.savez_compressed(path, **payload)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run DGGT tri-camera joint multiview inference from manifest.json.")
    ap.add_argument("--manifest", required=True, type=str)
    ap.add_argument("--dggt_repo", default="third_party/dggt", type=str)
    ap.add_argument("--ckpt_path", required=True, type=str)
    ap.add_argument("--device", default="cuda", type=str)
    ap.add_argument("--torch_dtype", default="float32", type=str)
    ap.add_argument("--use_input_calib", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--out_root", default="mvp-demo/output/dggt_multiview", type=str)
    ap.add_argument("--preprocess_mode", default="crop", choices=["crop", "pad"], type=str)
    args = ap.parse_args()

    try:
        import numpy as np
        import torch
    except Exception as exc:
        raise SystemExit(
            "Torch is required to run DGGT joint inference. Please use a DGGT runtime environment with torch installed."
            f"\nImport error: {exc!r}"
        )

    repo_root = _repo_root()
    manifest_path = _resolve_path(repo_root, str(args.manifest))
    if not manifest_path.exists():
        raise SystemExit(f"Missing --manifest: {manifest_path}")

    dggt_repo = _resolve_path(repo_root, str(args.dggt_repo))
    if not dggt_repo.exists():
        raise SystemExit(f"Missing --dggt_repo: {dggt_repo}")

    ckpt_path = _resolve_path(repo_root, str(args.ckpt_path))
    if not ckpt_path.exists():
        raise SystemExit(f"Missing --ckpt_path: {ckpt_path}")

    out_root = _resolve_path(repo_root, str(args.out_root))

    _prepare_imports(dggt_repo)
    from dggt.utils.geometry import unproject_depth_map_to_point_map
    from dggt.utils.inference_adapter import (
        apply_preprocess_to_mask_with_meta,
        build_model_from_checkpoint,
        load_and_prepare_images_with_intrinsics,
        meta_to_dict,
        run_forward_with_optional_input_calib,
    )

    manifest = _load_json(manifest_path)
    scene_id = str(manifest.get("scene_id") or "unknown_scene")
    scene_dir = Path(str(manifest.get("scene_dir") or ""))
    if not scene_dir.is_absolute():
        scene_dir = repo_root / scene_dir
    scene_dir = scene_dir.resolve()
    if not scene_dir.exists():
        raise SystemExit(f"Manifest scene_dir not found: {scene_dir}")

    views = list(manifest.get("views") or [])
    if not views:
        raise SystemExit("manifest.views is empty")

    frame_paths: list[str] = []
    intrinsics_list: list[np.ndarray] = []
    input_extrinsics_w2c: list[np.ndarray] = []
    logical_t_indices: list[int] = []
    view_cam_ids: list[str] = []
    scene_stems_per_view: list[str] = []
    mask_source_used: list[str] = []

    for item in views:
        frame_rel = str(item["frame_rel"])
        frame_abs = scene_dir / frame_rel
        if not frame_abs.exists():
            raise SystemExit(f"Frame not found: {frame_abs}")

        K = np.asarray(item["camera_intrinsic_3x3"], dtype=np.float32)
        extr = np.asarray(item["camera_extrinsic_w2c_3x4"], dtype=np.float32)
        if K.shape != (3, 3):
            raise SystemExit(f"Invalid K shape: {K.shape}")
        if extr.shape != (3, 4):
            raise SystemExit(f"Invalid extrinsic shape: {extr.shape}")

        frame_paths.append(str(frame_abs.as_posix()))
        intrinsics_list.append(K)
        input_extrinsics_w2c.append(extr)
        logical_t_indices.append(int(item["logical_t_idx"]))
        view_cam_ids.append(str(item["cam_id"]))
        scene_stems_per_view.append(str(item["scene_stem"]))
        mask_source_used.append(str(item.get("mask_source_used") or "dynamic_conf"))

    device = str(args.device)
    torch_dtype = _parse_torch_dtype(str(args.torch_dtype), torch)

    started_at = datetime.now(timezone.utc).isoformat()
    t0 = time.perf_counter()

    images_schw, prepared_intrinsics, preprocess_metas = load_and_prepare_images_with_intrinsics(
        image_paths=frame_paths,
        intrinsic_list_3x3=intrinsics_list,
        mode=str(args.preprocess_mode),
        target_size=518,
    )

    images_schw = images_schw.to(device=device, dtype=torch_dtype)
    model = build_model_from_checkpoint(str(ckpt_path), device=device, strict=True)

    forward_result = run_forward_with_optional_input_calib(
        model=model,
        images_bshw=images_schw,
        prepared_intrinsics_bx3x3=torch.from_numpy(prepared_intrinsics),
        input_extrinsics_w2c_bx3x4=torch.from_numpy(np.stack(input_extrinsics_w2c, axis=0)),
        use_input_calib=bool(args.use_input_calib),
    )

    predictions = forward_result.predictions

    pred_depth = predictions["depth"].detach().cpu().float().numpy()[0]
    pred_depth_conf = predictions["depth_conf"].detach().cpu().float().numpy()[0]
    pred_dynamic_conf = predictions["dynamic_conf"].detach().cpu().float().numpy()[0]
    pred_gs_map = predictions["gs_map"].detach().cpu().float().numpy()[0]
    pred_gs_conf = predictions["gs_conf"].detach().cpu().float().numpy()[0]
    pred_pose_enc = predictions["pose_enc"].detach().cpu().float().numpy()[0]

    pred_extrinsics_w2c = forward_result.pred_extrinsics_w2c.detach().cpu().float().numpy()[0]
    pred_intrinsics = None
    if forward_result.pred_intrinsics is not None:
        pred_intrinsics = forward_result.pred_intrinsics.detach().cpu().float().numpy()[0]

    geometry_extr_w2c = forward_result.geometry_extrinsics_w2c.detach().cpu().float().numpy()[0]
    geometry_intrinsics = forward_result.geometry_intrinsics.detach().cpu().float().numpy()[0]
    geometry_source = "input_calib" if bool(args.use_input_calib) else "pred_pose"

    world_points_geometry = unproject_depth_map_to_point_map(
        depth_map=pred_depth,
        extrinsics_cam=geometry_extr_w2c,
        intrinsics_cam=geometry_intrinsics,
    ).astype(np.float32)

    runtime_sec = float(time.perf_counter() - t0)
    finished_at = datetime.now(timezone.utc).isoformat()

    out_dir = out_root / scene_id / "run_full_frame_joint"
    out_dir.mkdir(parents=True, exist_ok=True)

    scene_stems_sync = [str(s["scene_stem"]) for s in list(manifest.get("sync_steps") or [])]

    bundle_payload: dict[str, Any] = {
        "schema_version": np.array("dggt_multiview_bundle_v1"),
        "scene_id": np.array(scene_id),
        "geometry_source": np.array(geometry_source),
        "use_input_calib": np.array(bool(args.use_input_calib)),
        "scene_stems": np.asarray(scene_stems_sync, dtype=object),
        "logical_t_indices": np.asarray(logical_t_indices, dtype=np.int32),
        "view_cam_ids": np.asarray(view_cam_ids, dtype=object),
        "input_extrinsics_w2c": np.stack(input_extrinsics_w2c, axis=0).astype(np.float32),
        "input_intrinsics": np.stack(intrinsics_list, axis=0).astype(np.float32),
        "prepared_intrinsics": prepared_intrinsics.astype(np.float32),
        "pred_depth": pred_depth.astype(np.float32),
        "pred_depth_conf": pred_depth_conf.astype(np.float32),
        "pred_dynamic_conf": pred_dynamic_conf.astype(np.float32),
        "pred_gs_map": pred_gs_map.astype(np.float32),
        "pred_gs_conf": pred_gs_conf.astype(np.float32),
        "pred_pose_enc": pred_pose_enc.astype(np.float32),
        "pred_extrinsics_w2c": pred_extrinsics_w2c.astype(np.float32),
        "pred_intrinsics": (pred_intrinsics.astype(np.float32) if pred_intrinsics is not None else np.zeros((0, 3, 3), dtype=np.float32)),
        "geometry_extrinsics_w2c": geometry_extr_w2c.astype(np.float32),
        "geometry_intrinsics": geometry_intrinsics.astype(np.float32),
        "world_points_geometry": world_points_geometry.astype(np.float32),
        "frame_paths": np.asarray(frame_paths, dtype=object),
        "mask_source_used": np.asarray(mask_source_used, dtype=object),
        # Backward-compatible alias for earlier consumers.
        "world_points_input_calib": world_points_geometry.astype(np.float32),
        "scene_stem_per_view": np.asarray(scene_stems_per_view, dtype=object),
    }

    bundle_path = out_dir / "reconstruction_bundle.npz"
    _npz_save(bundle_path, bundle_payload)

    pose_report = _build_pose_alignment_report(
        views=views,
        pred_extr_w2c=pred_extrinsics_w2c,
        input_extr_w2c=np.stack(input_extrinsics_w2c, axis=0),
    )
    pose_report["use_input_calib"] = bool(args.use_input_calib)
    pose_report_path = out_dir / "pose_alignment_report.json"
    _write_json(pose_report_path, pose_report)

    preprocess_meta_list = [meta_to_dict(meta) for meta in preprocess_metas]

    mask_debug_rows: list[dict[str, Any]] = []
    for view, meta in zip(views, preprocess_metas):
        mask_rel = view.get("mask_rel")
        if not mask_rel:
            continue
        mask_abs = (scene_dir / str(mask_rel)).resolve()
        if not mask_abs.exists():
            continue
        prepared_mask = apply_preprocess_to_mask_with_meta(str(mask_abs), meta)
        mask_debug_rows.append(
            {
                "view_idx": int(view["view_idx"]),
                "cam_id": str(view["cam_id"]),
                "logical_t_idx": int(view["logical_t_idx"]),
                "mask_rel": str(mask_rel),
                "prepared_mask_shape": [int(prepared_mask.shape[0]), int(prepared_mask.shape[1])],
            }
        )

    probe_meta = {
        "schema_version": "dggt_probe_meta_v1",
        "scene_id": scene_id,
        "scene_dir": scene_dir.as_posix(),
        "manifest": _to_rel(manifest_path, repo_root),
        "cams": list(manifest.get("cams") or []),
        "num_sync_steps": int(manifest.get("num_sync_steps") or 0),
        "num_views": int(manifest.get("num_views") or len(views)),
        "use_input_calib": bool(args.use_input_calib),
        "geometry_source": geometry_source,
        "ckpt_path": str(ckpt_path.as_posix()),
        "device": device,
        "torch_dtype": str(torch_dtype),
        "preprocess_mode": str(args.preprocess_mode),
        "inference_resolution": {
            "height": int(images_schw.shape[-2]),
            "width": int(images_schw.shape[-1]),
        },
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
        "runtime_sec": runtime_sec,
        "outputs": {
            "reconstruction_bundle_npz": _to_rel(bundle_path, repo_root),
            "pose_alignment_report_json": _to_rel(pose_report_path, repo_root),
            "probe_meta_json": _to_rel(out_dir / "probe_meta.json", repo_root),
        },
        "bundle_geometry_keys": {
            "geometry_source": "geometry_source",
            "use_input_calib": "use_input_calib",
            "geometry_extrinsics_w2c": "geometry_extrinsics_w2c",
            "geometry_intrinsics": "geometry_intrinsics",
            "world_points_geometry": "world_points_geometry",
            "legacy_world_points_input_calib": "world_points_input_calib",
        },
        "preprocess_meta": preprocess_meta_list,
        "mask_debug_rows": mask_debug_rows,
    }

    probe_meta_path = out_dir / "probe_meta.json"
    _write_json(probe_meta_path, probe_meta)

    print(f"Wrote: {bundle_path}")
    print(f"Wrote: {pose_report_path}")
    print(f"Wrote: {probe_meta_path}")


if __name__ == "__main__":
    main()
