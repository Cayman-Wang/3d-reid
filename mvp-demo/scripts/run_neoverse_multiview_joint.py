from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torchvision.transforms import functional as F


@dataclass
class RuntimeConfig:
    device: str
    torch_dtype: torch.dtype
    enable_vram_management: bool


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"Failed to read json: {path}\nError: {exc!r}")


def _to_rel(path: Path, base: Path) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except Exception:
        return path.resolve().as_posix()


def _prepare_imports(neoverse_repo: Path) -> None:
    p = str(neoverse_repo.resolve())
    if p not in sys.path:
        sys.path.insert(0, p)


def _pose_center(pose4x4: np.ndarray) -> np.ndarray:
    return np.asarray(pose4x4, dtype=np.float64)[:3, 3]


def _intrinsics_signature(K: np.ndarray) -> np.ndarray:
    K = np.asarray(K, dtype=np.float64)
    return np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]], dtype=np.float64)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _apply_rig_anchor_geometry_patch(neoverse_repo: Path) -> None:
    _prepare_imports(neoverse_repo)
    raster_mod = importlib.import_module("diffsynth.auxiliary_models.worldmirror.models.models.rasterization")
    GaussianSplatRenderer = getattr(raster_mod, "GaussianSplatRenderer", None)
    if GaussianSplatRenderer is None:
        raise SystemExit("Missing GaussianSplatRenderer in NeoVerse rasterization module")
    if getattr(GaussianSplatRenderer, "_neoverse_rig_anchor_patched", False):
        return

    original_render = getattr(GaussianSplatRenderer, "render", None)
    if original_render is None or not callable(original_render):
        raise SystemExit("GaussianSplatRenderer.render is not patchable")

    def rig_anchor_render(
        self,
        gs_feats: torch.Tensor,
        images: torch.Tensor,
        predictions: dict[str, torch.Tensor],
        views: dict[str, torch.Tensor],
        context_predictions: dict[str, torch.Tensor],
        is_inference: bool = True,
    ) -> dict[str, torch.Tensor]:
        if self.training or not is_inference:
            return original_render(
                self,
                gs_feats,
                images,
                predictions,
                views,
                context_predictions,
                is_inference=is_inference,
            )

        B = images.shape[0]
        S = gs_feats.shape[1]
        H = gs_feats.shape[3]
        W = gs_feats.shape[4]

        gs_feats_reshape = gs_feats.reshape(B * S, gs_feats.shape[2], H, W)
        gs_params_static = self.gs_head(gs_feats_reshape)
        if self.is_4dgs:
            gs_params_dynamic = self.gs_head_dynamic(gs_feats_reshape)
            is_static = views["is_static"][:, :S].reshape(-1)
            gs_params = torch.where(is_static[:, None, None, None], gs_params_static, gs_params_dynamic)
        else:
            gs_params = gs_params_static

        splats = self.prepare_splats(
            views,
            predictions,
            images,
            gs_params,
            S,
            position_from="gsdepth+gtcamera",
        )

        predictions["splats"] = splats
        predictions["rendered_extrinsics"] = views["camera_poses"]
        predictions["rendered_intrinsics"] = views["camera_intrs"]
        predictions["rendered_timestamps"] = views["timestamp"]
        return predictions

    GaussianSplatRenderer.render = rig_anchor_render
    GaussianSplatRenderer._neoverse_rig_anchor_patched = True


def _build_camera_prior_alignment_report(
    manifest: dict[str, Any],
    views_meta: list[dict[str, Any]],
    predictions: dict[str, Any],
) -> dict[str, Any]:
    if "camera_poses" not in predictions or "camera_intrs" not in predictions:
        raise SystemExit("predictions missing camera_poses/camera_intrs for alignment report")
    if "rendered_extrinsics" not in predictions or "rendered_intrinsics" not in predictions:
        raise SystemExit("predictions missing rendered_extrinsics/rendered_intrinsics for alignment report")

    rig_poses = np.asarray([item["camera_pose_c2w"] for item in views_meta], dtype=np.float64)
    pred_poses = predictions["camera_poses"][0].detach().cpu().numpy()
    rendered_poses = predictions["rendered_extrinsics"][0].detach().cpu().numpy()
    pred_intrs = predictions["camera_intrs"][0].detach().cpu().numpy()
    rendered_intrs = predictions["rendered_intrinsics"][0].detach().cpu().numpy()

    report: dict[str, Any] = {
        "scene_id": str(manifest.get("scene_id") or "unknown_scene"),
        "geometry_anchor_mode": "rig_gtcamera",
        "render_camera_source": "rig_input",
        "splat_camera_source": "rig_input",
        "num_views": int(len(views_meta)),
        "per_camera": {},
    }

    cams = sorted({str(item["cam_id"]) for item in views_meta})
    for cam in cams:
        idxs = [i for i, item in enumerate(views_meta) if str(item["cam_id"]) == cam]
        rig_centers = np.stack([_pose_center(rig_poses[i]) for i in idxs], axis=0)
        pred_centers = np.stack([_pose_center(pred_poses[i]) for i in idxs], axis=0)
        rendered_centers = np.stack([_pose_center(rendered_poses[i]) for i in idxs], axis=0)

        rig_intr_sig = np.stack(
            [_intrinsics_signature(views_meta[i].get("prepared_camera_K", views_meta[i]["camera_K"])) for i in idxs],
            axis=0,
        )
        pred_intr_sig = np.stack([_intrinsics_signature(pred_intrs[i]) for i in idxs], axis=0)
        rendered_intr_sig = np.stack([_intrinsics_signature(rendered_intrs[i]) for i in idxs], axis=0)

        report["per_camera"][cam] = {
            "count": int(len(idxs)),
            "rig_center_mean": rig_centers.mean(axis=0).round(6).tolist(),
            "predicted_center_mean": pred_centers.mean(axis=0).round(6).tolist(),
            "rendered_center_mean": rendered_centers.mean(axis=0).round(6).tolist(),
            "rig_vs_pred_center_rmse": float(np.sqrt(((rig_centers - pred_centers) ** 2).sum(axis=1).mean())),
            "rig_vs_rendered_center_rmse": float(np.sqrt(((rig_centers - rendered_centers) ** 2).sum(axis=1).mean())),
            "rig_vs_pred_intrinsics_mae": float(np.abs(rig_intr_sig - pred_intr_sig).mean()),
            "rig_vs_rendered_intrinsics_mae": float(np.abs(rig_intr_sig - rendered_intr_sig).mean()),
        }

    return report


def _parse_torch_dtype(name: str) -> torch.dtype:
    value = str(name).strip().lower()
    if value in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if value in {"fp16", "float16"}:
        return torch.float16
    if value in {"fp32", "float32"}:
        return torch.float32
    raise SystemExit(f"Unsupported --torch_dtype: {name}")


def _read_rgb_image(path: Path):
    from PIL import Image

    try:
        img = Image.open(path).convert("RGB")
    except Exception as exc:
        raise SystemExit(f"Failed to read image: {path}; error={exc!r}")
    return img


def _resize_with_intrinsics(img, K: np.ndarray, target_width: int, target_height: int):
    from PIL import Image

    src_width, src_height = img.size
    scale_x = float(target_width) / float(src_width)
    scale_y = float(target_height) / float(src_height)
    out = img.resize((target_width, target_height), resample=Image.LANCZOS)

    K_out = np.asarray(K, dtype=np.float32).copy()
    K_out[0, 0] *= scale_x
    K_out[1, 1] *= scale_y
    K_out[0, 2] *= scale_x
    K_out[1, 2] *= scale_y
    return out, K_out


def _center_crop_with_intrinsics(img, K: np.ndarray, target_width: int, target_height: int):
    from PIL import Image

    src_width, src_height = img.size
    scale = max(float(target_width) / float(src_width), float(target_height) / float(src_height))
    scaled_width = int(round(src_width * scale))
    scaled_height = int(round(src_height * scale))
    scaled = img.resize((scaled_width, scaled_height), resample=Image.LANCZOS)

    left = (scaled_width - target_width) // 2
    top = (scaled_height - target_height) // 2
    out = scaled.crop((left, top, left + target_width, top + target_height))

    K_out = np.asarray(K, dtype=np.float32).copy()
    K_out[0, 0] *= scale
    K_out[1, 1] *= scale
    K_out[0, 2] = K_out[0, 2] * scale - float(left)
    K_out[1, 2] = K_out[1, 2] * scale - float(top)
    return out, K_out


def _prepare_image_and_intrinsics(
    img,
    K: np.ndarray,
    target_width: int,
    target_height: int,
    resize_mode: str,
):
    if resize_mode == "resize":
        return _resize_with_intrinsics(img, K, target_width, target_height)
    if resize_mode == "center_crop":
        return _center_crop_with_intrinsics(img, K, target_width, target_height)
    raise SystemExit(f"Unsupported --resize_mode: {resize_mode}")


def _build_views(
    manifest: dict[str, Any],
    scene_dir: Path,
    device: str,
    target_width: int,
    target_height: int,
    resize_mode: str,
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
    views_meta = [dict(item) for item in list(manifest.get("views") or [])]
    if not views_meta:
        raise SystemExit("manifest.views is empty")

    tensors = []
    timestamps = []
    camera_intrs = []
    camera_poses = []
    for item in views_meta:
        frame_rel = str(item["frame_rel"])
        frame_abs = scene_dir / frame_rel
        if not frame_abs.exists():
            raise SystemExit(f"Missing frame: {frame_abs}")
        img = _read_rgb_image(frame_abs)
        timestamps.append(int(item["logical_t_idx"]))
        K = item.get("camera_K")
        pose = item.get("camera_pose_c2w")
        if K is None or pose is None:
            raise SystemExit("manifest.views requires camera_K and camera_pose_c2w for rig-conditioned reconstruction")
        img, K_prepared = _prepare_image_and_intrinsics(
            img,
            np.asarray(K, dtype=np.float32),
            target_width=target_width,
            target_height=target_height,
            resize_mode=resize_mode,
        )
        item["prepared_camera_K"] = K_prepared.tolist()
        tensors.append(F.to_tensor(img)[None])
        camera_intrs.append(torch.tensor(K_prepared, dtype=torch.float32))
        camera_poses.append(torch.tensor(pose, dtype=torch.float32))

    img_tensor = torch.stack(tensors, dim=1).to(device)
    S = img_tensor.shape[1]

    views = {
        "img": img_tensor,
        "is_target": torch.zeros((1, S), dtype=torch.bool, device=device),
        "is_static": torch.zeros((1, S), dtype=torch.bool, device=device),
        "timestamp": torch.tensor(timestamps, dtype=torch.int64, device=device).unsqueeze(0),
        "camera_intrs": torch.stack(camera_intrs, dim=0).to(device).unsqueeze(0),
        "camera_poses": torch.stack(camera_poses, dim=0).to(device).unsqueeze(0),
    }
    return views, views_meta


def _serialize_tensor(x: Any) -> Any:
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.detach().cpu()
    return x


def _serialize_splats(predictions: dict[str, Any]) -> list[dict[str, Any]]:
    splats_batch = predictions["splats"][0]
    serialized: list[dict[str, Any]] = []
    for gs in splats_batch:
        serialized.append(
            {
                "means": _serialize_tensor(gs.means),
                "harmonics": _serialize_tensor(gs.harmonics),
                "opacities": _serialize_tensor(gs.opacities),
                "scales": _serialize_tensor(gs.scales),
                "rotations": _serialize_tensor(gs.rotations),
                "confidences": _serialize_tensor(getattr(gs, "confidences", None)),
                "timestamp": int(getattr(gs, "timestamp", -1)),
                "life_span": _serialize_tensor(getattr(gs, "life_span", 1.0)),
                "life_span_gamma": float(getattr(gs, "life_span_gamma", 0.0)),
                "forward_timestamp": getattr(gs, "forward_timestamp", None),
                "forward_vel": _serialize_tensor(getattr(gs, "forward_vel", None)),
                "forward_scales": _serialize_tensor(getattr(gs, "forward_scales", None)),
                "forward_rotations": _serialize_tensor(getattr(gs, "forward_rotations", None)),
                "backward_timestamp": getattr(gs, "backward_timestamp", None),
                "backward_vel": _serialize_tensor(getattr(gs, "backward_vel", None)),
                "backward_scales": _serialize_tensor(getattr(gs, "backward_scales", None)),
                "backward_rotations": _serialize_tensor(getattr(gs, "backward_rotations", None)),
            }
        )
    return serialized


def _build_pose_cluster_report(
    cam2world: np.ndarray,
    views_meta: list[dict[str, Any]],
    pose_source: str,
) -> dict[str, Any]:
    centers = cam2world[:, :3, 3]
    by_cam: dict[str, list[np.ndarray]] = {}
    for i, item in enumerate(views_meta):
        cam = str(item["cam_id"])
        by_cam.setdefault(cam, []).append(centers[i])

    report: dict[str, Any] = {
        "num_views": int(cam2world.shape[0]),
        "pose_source": str(pose_source),
        "camera_center_clusters": {},
    }
    for cam in sorted(by_cam.keys()):
        arr = np.stack(by_cam[cam], axis=0)
        report["camera_center_clusters"][cam] = {
            "count": int(arr.shape[0]),
            "mean": arr.mean(axis=0).round(6).tolist(),
            "std": arr.std(axis=0).round(6).tolist(),
            "min": arr.min(axis=0).round(6).tolist(),
            "max": arr.max(axis=0).round(6).tolist(),
        }
    return report


def _make_scene_glb(predictions: dict[str, Any], out_dir: Path) -> Path:
    app_mod = importlib.import_module("diffsynth.utils.app")
    build_scene_glb = getattr(app_mod, "build_scene_glb")
    extract_point_cloud = getattr(app_mod, "extract_point_cloud")

    points, colors, frame_indices = extract_point_cloud(predictions)
    cam2world = predictions["rendered_extrinsics"][0].detach().cpu().numpy()
    scene = build_scene_glb(points, colors, frame_indices, cam2world)
    glb_path = out_dir / "scene.glb"
    scene.export(str(glb_path))
    return glb_path


def _run_reconstruction(
    manifest_path: Path,
    neoverse_repo: Path,
    reconstructor_path: str,
    cfg: RuntimeConfig,
    target_width: int,
    target_height: int,
    resize_mode: str,
) -> tuple[dict[str, Any], dict[str, torch.Tensor], list[dict[str, Any]]]:
    _apply_rig_anchor_geometry_patch(neoverse_repo)
    models_mod = importlib.import_module("diffsynth.models")
    ModelManager = getattr(models_mod, "ModelManager")

    manifest = _load_json(manifest_path)
    scene_dir = Path(str(manifest.get("scene_dir", "")))
    if not scene_dir.exists():
        raise SystemExit(f"manifest.scene_dir not found: {scene_dir}")

    views, views_meta = _build_views(
        manifest=manifest,
        scene_dir=scene_dir,
        device=cfg.device,
        target_width=target_width,
        target_height=target_height,
        resize_mode=resize_mode,
    )

    reconstructor_device = "cpu" if cfg.enable_vram_management else cfg.device
    model_manager = ModelManager(torch_dtype=cfg.torch_dtype, device=reconstructor_device)
    model_manager.load_model(
        reconstructor_path,
        device=reconstructor_device,
        torch_dtype=cfg.torch_dtype,
    )
    reconstructor = model_manager.fetch_model("reconstructor")

    if cfg.enable_vram_management:
        reconstructor.to(cfg.device)

    use_amp = str(cfg.device).startswith("cuda")
    t0 = time.perf_counter()
    with torch.no_grad():
        if use_amp:
            with torch.amp.autocast("cuda", dtype=cfg.torch_dtype):
                predictions = reconstructor(
                    views,
                    cond_flags=[0, 1, 1],
                    is_inference=True,
                    use_motion=False,
                )
        else:
            predictions = reconstructor(
                views,
                cond_flags=[0, 1, 1],
                is_inference=True,
                use_motion=False,
            )
    dt = time.perf_counter() - t0

    if cfg.enable_vram_management:
        reconstructor.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    stats = {
        "runtime_sec": float(dt),
    }
    return stats, predictions, views_meta


def main() -> None:
    ap = argparse.ArgumentParser(description="Run NeoVerse tri-camera joint multiview reconstruction from manifest.json.")
    ap.add_argument("--manifest", required=True, type=str)
    ap.add_argument("--neoverse_repo", default="third_party/NeoVerse", type=str)
    ap.add_argument("--model_path", default="models", type=str)
    ap.add_argument("--reconstructor_path", default="models/NeoVerse/reconstructor.ckpt", type=str)
    ap.add_argument("--device", default="cuda", type=str)
    ap.add_argument("--torch_dtype", default="bfloat16", type=str)
    ap.add_argument("--enable_vram_management", action="store_true")
    ap.add_argument("--out_root", default="mvp-demo/output/neoverse_multiview", type=str)
    ap.add_argument("--height", default=336, type=int)
    ap.add_argument("--width", default=560, type=int)
    ap.add_argument("--resize_mode", choices=["center_crop", "resize"], default="center_crop", type=str)
    args = ap.parse_args()

    repo_root = _repo_root()
    manifest_path = Path(str(args.manifest))
    if not manifest_path.is_absolute():
        manifest_path = repo_root / manifest_path
    manifest_path = manifest_path.resolve()
    if not manifest_path.exists():
        raise SystemExit(f"Missing --manifest: {manifest_path}")

    manifest = _load_json(manifest_path)
    scene_id = str(manifest.get("scene_id") or "unknown_scene")

    neoverse_repo = Path(str(args.neoverse_repo))
    if not neoverse_repo.is_absolute():
        neoverse_repo = repo_root / neoverse_repo
    neoverse_repo = neoverse_repo.resolve()
    if not neoverse_repo.exists():
        raise SystemExit(f"Missing --neoverse_repo: {neoverse_repo}")

    out_root = Path(str(args.out_root))
    if not out_root.is_absolute():
        out_root = repo_root / out_root
    out_dir = out_root / scene_id / "run_full_frame_joint"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = RuntimeConfig(
        device=str(args.device),
        torch_dtype=_parse_torch_dtype(str(args.torch_dtype)),
        enable_vram_management=bool(args.enable_vram_management),
    )

    started_at = datetime.now(timezone.utc).isoformat()
    runtime_stats, predictions, views_meta = _run_reconstruction(
        manifest_path=manifest_path,
        neoverse_repo=neoverse_repo,
        reconstructor_path=str(args.reconstructor_path),
        cfg=cfg,
        target_width=int(args.width),
        target_height=int(args.height),
        resize_mode=str(args.resize_mode),
    )
    finished_at = datetime.now(timezone.utc).isoformat()

    predicted_camera_intrinsics = predictions["camera_intrs"][0].detach().cpu()
    predicted_camera_cam2world = predictions["camera_poses"][0].detach().cpu()
    rendered_intrinsics = predictions["rendered_intrinsics"][0].detach().cpu()
    rendered_cam2world = predictions["rendered_extrinsics"][0].detach().cpu()
    rendered_timestamps = predictions["rendered_timestamps"][0].detach().cpu()

    bundle = {
        "schema_version": "neoverse_multiview_bundle_v2",
        "scene_id": scene_id,
        "created_at_utc": finished_at,
        "splats_serialized": _serialize_splats(predictions),
        "predicted_camera_intrinsics": predicted_camera_intrinsics,
        "predicted_camera_cam2world": predicted_camera_cam2world,
        "rendered_intrinsics": rendered_intrinsics,
        "rendered_cam2world": rendered_cam2world,
        "rendered_timestamps": rendered_timestamps,
        "source_manifest": manifest,
    }

    bundle_path = out_dir / "reconstruction_bundle.pt"
    torch.save(bundle, bundle_path)

    pose_report = _build_pose_cluster_report(
        predicted_camera_cam2world.numpy(),
        views_meta,
        pose_source="predicted_camera_poses",
    )
    pose_report_path = out_dir / "pose_cluster_report.json"
    pose_report_path.write_text(json.dumps(pose_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    glb_path = _make_scene_glb(predictions=predictions, out_dir=out_dir)

    scene_stems = [str(s.get("scene_stem")) for s in manifest.get("sync_steps", [])]
    probe_meta = {
        "scene_id": scene_id,
        "manifest": _to_rel(manifest_path, repo_root),
        "conditioning_mode": "rig_camera_priors",
        "geometry_anchor_mode": "rig_gtcamera",
        "model_loading_mode": "reconstructor_only",
        "cond_flags": [0, 1, 1],
        "camera_prior_source": "rig.json",
        "render_camera_source": "rig_input",
        "splat_camera_source": "rig_input",
        "input_resolution": {"width": int(args.width), "height": int(args.height)},
        "resize_mode": str(args.resize_mode),
        "num_views": int(manifest.get("num_views", 0)),
        "num_sync_steps": int(manifest.get("num_sync_steps", 0)),
        "cams": list(manifest.get("cams", [])),
        "scene_stems": scene_stems,
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
        "runtime_sec": float(runtime_stats["runtime_sec"]),
        "device": cfg.device,
        "torch_dtype": str(cfg.torch_dtype),
        "outputs": {
            "reconstruction_bundle_pt": _to_rel(bundle_path, repo_root),
            "pose_cluster_report_json": _to_rel(pose_report_path, repo_root),
            "camera_prior_alignment_json": _to_rel(out_dir / "camera_prior_alignment.json", repo_root),
            "scene_glb": _to_rel(glb_path, repo_root),
        },
    }
    probe_meta_path = out_dir / "probe_meta.json"
    _write_json(probe_meta_path, probe_meta)

    alignment_report = _build_camera_prior_alignment_report(manifest, views_meta, predictions)
    alignment_path = out_dir / "camera_prior_alignment.json"
    _write_json(alignment_path, alignment_report)

    print(f"Wrote: {bundle_path}")
    print(f"Wrote: {probe_meta_path}")
    print(f"Wrote: {pose_report_path}")
    print(f"Wrote: {alignment_path}")
    print(f"Wrote: {glb_path}")


if __name__ == "__main__":
    main()
