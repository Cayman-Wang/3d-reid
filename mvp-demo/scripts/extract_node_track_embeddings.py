from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def l2norm(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(x))
    return (x / (n + eps)).astype(np.float32)


def _load_mask(path: Path) -> np.ndarray:
    try:
        import cv2  # type: ignore

        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(path)
        return np.asarray(mask) > 0
    except Exception:
        from PIL import Image  # type: ignore

        return np.asarray(Image.open(path).convert("L")) > 0


def _clip_bbox_xyxy(bbox: list[float] | list[int], w: int, h: int) -> list[int] | None:
    x1, y1, x2, y2 = bbox
    x1 = int(round(float(x1)))
    y1 = int(round(float(y1)))
    x2 = int(round(float(x2)))
    y2 = int(round(float(y2)))
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _pick_indices(n: int, max_items: int) -> list[int]:
    if max_items <= 0 or n <= max_items:
        return list(range(n))
    idx = np.linspace(0, n - 1, num=max_items, dtype=int).tolist()
    return sorted(set(int(i) for i in idx))


def _sample_points(points_xyz: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    pts = np.asarray(points_xyz, dtype=np.float32)
    if n <= 0 or pts.shape[0] <= n:
        return pts
    idx = rng.choice(pts.shape[0], size=int(n), replace=False)
    return pts[idx]


def _normalize_unit_sphere(points_xyz: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    pts = np.asarray(points_xyz, dtype=np.float32)
    if pts.shape[0] == 0:
        return pts
    center = pts.mean(axis=0, keepdims=True)
    pts0 = pts - center
    radius = float(np.linalg.norm(pts0, axis=1).max())
    return (pts0 / (radius + eps)).astype(np.float32)


def _rgb_hist_desc(bgr: np.ndarray, h_bins: int = 16, s_bins: int = 8) -> np.ndarray:
    import cv2  # type: ignore

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [int(h_bins), int(s_bins)], [0, 180, 0, 256])
    vec = hist.flatten().astype(np.float32)
    return vec / (float(vec.sum()) + 1e-6)


def _radial_hist_desc(points_xyz: np.ndarray, bins: int = 33) -> np.ndarray:
    if points_xyz.shape[0] < 10:
        return np.zeros((int(bins),), dtype=np.float32)
    radii = np.linalg.norm(points_xyz.astype(np.float32), axis=1)
    hist, _ = np.histogram(radii, bins=int(bins), range=(0.0, 1.0))
    vec = hist.astype(np.float32)
    return vec / (float(vec.sum()) + 1e-6)


def _try_build_clip(device: str, model_name: str, pretrained: str):
    try:
        import os
        import sys

        if sys.platform.startswith("linux") and hasattr(sys, "setdlopenflags") and hasattr(os, "RTLD_LAZY"):
            try:
                sys.setdlopenflags(os.RTLD_GLOBAL | os.RTLD_LAZY)
            except Exception:
                pass

        import torch  # type: ignore
        import open_clip  # type: ignore
        from PIL import Image  # type: ignore
    except Exception as e:  # pragma: no cover
        return None, None, None, None, e

    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model = model.to(device).eval()
    return model, preprocess, torch, Image, None


def _clip_embed(model, preprocess, torch, Image, device: str, bgr_crop: np.ndarray) -> np.ndarray:
    import cv2  # type: ignore

    rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
    x = preprocess(Image.fromarray(rgb)).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model.encode_image(x)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.detach().cpu().numpy().astype(np.float32).squeeze(0)


def _try_import_open3d():
    try:
        import open3d as o3d  # type: ignore
    except Exception as e:  # pragma: no cover
        return None, e
    return o3d, None


def _fpfh_global(o3d, points_xyz: np.ndarray) -> np.ndarray:
    if points_xyz.shape[0] < 100:
        return np.zeros((33,), dtype=np.float32)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_xyz.astype(np.float64))
    pcd = pcd.voxel_down_sample(voxel_size=0.02)
    if np.asarray(pcd.points).shape[0] < 50:
        return np.zeros((33,), dtype=np.float32)

    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    feat = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=100),
    )
    data = np.asarray(feat.data).astype(np.float32)
    if data.size == 0:
        return np.zeros((33,), dtype=np.float32)
    return data.mean(axis=1)


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract node-level track embeddings from multi-camera tracks.")
    ap.add_argument("--scene_dir", required=True, type=str)
    ap.add_argument("--tracklets", default="tracks/tracklets.json", type=str)
    ap.add_argument("--out_dir", default="embeddings", type=str)
    ap.add_argument("--max_timestamps_per_track", default=30, type=int)
    ap.add_argument("--max_points_per_timestamp", default=5000, type=int)
    ap.add_argument("--seed", default=0, type=int)
    ap.add_argument("--apply_mask_to_rgb", action="store_true")
    ap.add_argument("--rgb_backend", default="auto", choices=["auto", "clip", "hist"], type=str)
    ap.add_argument("--geo_backend", default="auto", choices=["auto", "open3d_fpfh", "radial_hist", "none"], type=str)
    ap.add_argument("--geo_bins", default=33, type=int)
    ap.add_argument("--device", default="auto", type=str, help="auto|cpu|cuda")
    ap.add_argument("--clip_model", default="ViT-B-32", type=str)
    ap.add_argument("--clip_pretrained", default="laion2b_s34b_b79k", type=str)
    args = ap.parse_args()

    scene_dir = Path(args.scene_dir).resolve()
    tracklets_path = scene_dir / args.tracklets
    if not tracklets_path.exists():
        raise SystemExit(f"tracklets not found: {tracklets_path}")

    try:
        import cv2  # type: ignore
    except Exception as e:  # pragma: no cover
        raise SystemExit(f"Missing opencv-python dependency. Original import error: {e!r}")

    tracklets = json.loads(tracklets_path.read_text(encoding="utf-8"))
    if not isinstance(tracklets, list):
        raise SystemExit(f"Expected a JSON list in: {tracklets_path}")

    rng = np.random.default_rng(int(args.seed))

    device = str(args.device)
    if device == "auto":
        device = "cpu"
        try:
            import torch  # type: ignore

            if torch.cuda.is_available():
                device = "cuda"
        except Exception:
            device = "cpu"

    rgb_backend = str(args.rgb_backend)
    clip_model = None
    clip_preprocess = None
    clip_torch = None
    clip_Image = None
    clip_err = None
    if rgb_backend in {"auto", "clip"}:
        clip_model, clip_preprocess, clip_torch, clip_Image, clip_err = _try_build_clip(
            device=device,
            model_name=str(args.clip_model),
            pretrained=str(args.clip_pretrained),
        )
        if clip_model is None and rgb_backend == "clip":
            raise SystemExit(f"--rgb_backend clip requested but open_clip/torch not available: {clip_err}")
        if clip_model is not None and rgb_backend == "auto":
            rgb_backend = "clip"
        if clip_model is None and rgb_backend == "auto":
            rgb_backend = "hist"

    geo_backend = str(args.geo_backend)
    o3d = None
    o3d_err = None
    if geo_backend in {"auto", "open3d_fpfh"}:
        o3d, o3d_err = _try_import_open3d()
        if o3d is None and geo_backend == "open3d_fpfh":
            raise SystemExit(f"--geo_backend open3d_fpfh requested but open3d not available: {o3d_err}")
        if o3d is not None and geo_backend == "auto":
            geo_backend = "open3d_fpfh"
        if o3d is None and geo_backend == "auto":
            geo_backend = "radial_hist"

    print(f"[cfg] rgb_backend={rgb_backend} geo_backend={geo_backend} device={device}")

    track_embs: list[np.ndarray] = []
    track_meta: list[dict] = []
    geo_dim = 0 if geo_backend == "none" else int(args.geo_bins)

    for track in tracklets:
        if not isinstance(track, dict):
            continue
        timestamp_stems = list(track.get("timestamp_stems") or [])
        per_camera = track.get("per_camera") or {}
        fused_points_paths = list(track.get("fused_points_paths") or [])
        if not timestamp_stems or not isinstance(per_camera, dict):
            continue

        indices = _pick_indices(len(timestamp_stems), int(args.max_timestamps_per_track))
        per_timestamp_embs: list[np.ndarray] = []
        used_timestamp_stems: list[str] = []

        for idx in indices:
            stem = str(timestamp_stems[idx])
            rgb_embs: list[np.ndarray] = []

            for cam_id, cam_entry in per_camera.items():
                if not isinstance(cam_entry, dict):
                    continue

                try:
                    frame_rel = str(cam_entry["frame_paths"][idx])
                    mask_rel = str(cam_entry["mask_paths"][idx])
                    bbox = cam_entry["bboxes_xyxy"][idx]
                except Exception:
                    continue

                img_path = scene_dir / frame_rel
                mask_path = scene_dir / mask_rel
                if not img_path.exists() or not mask_path.exists():
                    continue

                img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
                if img is None:
                    continue
                h, w = img.shape[:2]
                bbox_i = _clip_bbox_xyxy(bbox, w=w, h=h)
                if bbox_i is None:
                    continue

                x1, y1, x2, y2 = bbox_i
                crop = img[y1:y2, x1:x2].copy()
                if crop.size == 0:
                    continue

                if args.apply_mask_to_rgb:
                    mask = _load_mask(mask_path)
                    if mask.shape == (h, w):
                        crop_mask = mask[y1:y2, x1:x2]
                        crop[~crop_mask] = 0

                if rgb_backend == "clip":
                    rgb_emb = _clip_embed(clip_model, clip_preprocess, clip_torch, clip_Image, device, crop)
                else:
                    rgb_emb = _rgb_hist_desc(crop)
                rgb_embs.append(l2norm(rgb_emb))

            if not rgb_embs:
                continue

            rgb_track = l2norm(np.stack(rgb_embs, axis=0).mean(axis=0).astype(np.float32))

            if geo_backend == "none":
                geo_emb = np.zeros((0,), dtype=np.float32)
            else:
                geo_emb = np.zeros((geo_dim,), dtype=np.float32)
                if idx < len(fused_points_paths):
                    points_rel = fused_points_paths[idx]
                    if points_rel:
                        points_path = scene_dir / str(points_rel)
                        if points_path.exists():
                            pts = np.load(str(points_path))
                            pts = np.asarray(pts, dtype=np.float32)
                            if pts.ndim == 2 and pts.shape[1] == 3:
                                pts = _sample_points(pts, int(args.max_points_per_timestamp), rng)
                                pts = _normalize_unit_sphere(pts)
                                if geo_backend == "open3d_fpfh":
                                    geo_emb = _fpfh_global(o3d, pts)
                                else:
                                    geo_emb = _radial_hist_desc(pts, bins=int(args.geo_bins))
                geo_emb = l2norm(geo_emb)

            fused = np.concatenate([rgb_track, geo_emb], axis=0) if geo_emb.size else rgb_track
            per_timestamp_embs.append(l2norm(fused.astype(np.float32)))
            used_timestamp_stems.append(stem)

        if not per_timestamp_embs:
            continue

        emb = l2norm(np.stack(per_timestamp_embs, axis=0).mean(axis=0).astype(np.float32))
        track_embs.append(emb)
        track_meta.append(
            {
                "track_id": str(track.get("track_id", "")),
                "identity_id": track.get("identity_id"),
                "node_id": track.get("node_id"),
                "scene_dir": str(scene_dir),
                "n_timestamps_total": int(len(timestamp_stems)),
                "n_timestamps_used": int(len(per_timestamp_embs)),
                "used_timestamp_stems": used_timestamp_stems,
                "rgb_backend": rgb_backend,
                "geo_backend": geo_backend,
                "dim": int(emb.shape[0]),
            }
        )

    out_dir = scene_dir / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    embs = np.stack(track_embs, axis=0).astype(np.float32) if track_embs else np.zeros((0, 0), dtype=np.float32)
    np.save(str(out_dir / "tracks.npy"), embs)
    (out_dir / "tracks_meta.json").write_text(json.dumps(track_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote embeddings: {out_dir / 'tracks.npy'} shape={tuple(embs.shape)}")
    print(f"Wrote metadata:   {out_dir / 'tracks_meta.json'} tracks={len(track_meta)}")


if __name__ == "__main__":
    main()
