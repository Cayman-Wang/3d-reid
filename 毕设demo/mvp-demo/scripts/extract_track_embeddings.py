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

        m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise FileNotFoundError(path)
        return (m > 0)
    except Exception:
        from PIL import Image  # type: ignore

        m = Image.open(path).convert("L")
        return (np.array(m) > 0)


def _mask_bbox_xyxy(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max()) + 1
    y2 = int(ys.max()) + 1
    return [x1, y1, x2, y2]


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


def _bbox_to_mask(bbox_xyxy: list[int], h: int, w: int) -> np.ndarray:
    x1, y1, x2, y2 = bbox_xyxy
    m = np.zeros((h, w), dtype=bool)
    m[y1:y2, x1:x2] = True
    return m


def _depth_to_points(depth: np.ndarray, K: dict, mask: np.ndarray) -> np.ndarray:
    fx = float(K["fx"])
    fy = float(K["fy"])
    cx = float(K["cx"])
    cy = float(K["cy"])

    z = depth.astype(np.float32)
    valid = np.isfinite(z) & (z > 0) & mask
    if int(valid.sum()) < 50:
        return np.zeros((0, 3), dtype=np.float32)

    ys, xs = np.where(valid)
    zs = z[ys, xs]
    xs = xs.astype(np.float32)
    ys = ys.astype(np.float32)

    X = (xs - cx) / fx * zs
    Y = (ys - cy) / fy * zs
    pts = np.stack([X, Y, zs], axis=1).astype(np.float32)
    return pts


def _normalize_unit_sphere(pts: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    if pts.shape[0] == 0:
        return pts
    c = pts.mean(axis=0, keepdims=True)
    pts0 = pts - c
    r = float(np.linalg.norm(pts0, axis=1).max())
    return (pts0 / (r + eps)).astype(np.float32)


def _sample_points(pts: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    if pts.shape[0] <= n:
        return pts
    idx = rng.choice(pts.shape[0], size=int(n), replace=False)
    return pts[idx]


def _rgb_hist_desc(bgr: np.ndarray, h_bins: int = 16, s_bins: int = 8) -> np.ndarray:
    import cv2  # type: ignore

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [int(h_bins), int(s_bins)], [0, 180, 0, 256])
    v = hist.flatten().astype(np.float32)
    v = v / (float(v.sum()) + 1e-6)
    return v


def _radial_hist_desc(pts_unit: np.ndarray, bins: int = 33) -> np.ndarray:
    if pts_unit.shape[0] < 10:
        return np.zeros((int(bins),), dtype=np.float32)
    r = np.linalg.norm(pts_unit.astype(np.float32), axis=1)
    hist, _ = np.histogram(r, bins=int(bins), range=(0.0, 1.0))
    v = hist.astype(np.float32)
    v = v / (float(v.sum()) + 1e-6)
    return v


def _try_build_clip(device: str, model_name: str, pretrained: str):
    try:
        import os
        import sys

        # Same PyTorch iJIT workaround as in gated_capture_yolo.py (safe no-op for most envs).
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
    pil = Image.fromarray(rgb)
    x = preprocess(pil).unsqueeze(0).to(device)
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


def _fpfh_global(o3d, pts_unit: np.ndarray) -> np.ndarray:
    if pts_unit.shape[0] < 100:
        return np.zeros((33,), dtype=np.float32)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_unit.astype(np.float64))
    pcd = pcd.voxel_down_sample(voxel_size=0.02)
    if np.asarray(pcd.points).shape[0] < 50:
        return np.zeros((33,), dtype=np.float32)

    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    feat = o3d.pipelines.registration.compute_fpfh_feature(
        pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=100)
    )
    f = np.asarray(feat.data).astype(np.float32)  # (33, npts)
    if f.size == 0:
        return np.zeros((33,), dtype=np.float32)
    return f.mean(axis=1)


def _pick_frame_indices(n: int, max_frames: int) -> list[int]:
    if max_frames <= 0 or n <= max_frames:
        return list(range(n))
    idx = np.linspace(0, n - 1, num=max_frames, dtype=int).tolist()
    # keep stable and unique
    return sorted(set(int(i) for i in idx))


def _get_intrinsics_for_stem(intr: dict, stem: str) -> dict:
    if stem in intr:
        return intr[stem]
    if "__default__" in intr:
        return intr["__default__"]
    if len(intr) == 1:
        return next(iter(intr.values()))
    raise KeyError(f"Missing intrinsics for stem={stem}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract track embeddings from images+depth (+ optional masks).")
    ap.add_argument("--scene_dir", required=True, type=str)
    ap.add_argument("--tracklets", default="tracklets.json", type=str)
    ap.add_argument("--images_dir", default="images", type=str)
    ap.add_argument("--depth_dir", default="depth_npy", type=str)
    ap.add_argument("--intrinsics", default="intrinsics.json", type=str)
    ap.add_argument("--out_dir", default="embeddings", type=str)
    ap.add_argument("--max_frames_per_track", default=30, type=int)
    ap.add_argument("--n_points", default=5000, type=int, help="Max points sampled per frame for geometry")
    ap.add_argument("--seed", default=0, type=int)

    ap.add_argument("--rgb_backend", default="auto", choices=["auto", "clip", "hist"], type=str)
    ap.add_argument("--geo_backend", default="auto", choices=["auto", "open3d_fpfh", "radial_hist", "none"], type=str)
    ap.add_argument("--geo_bins", default=33, type=int, help="Bins for radial_hist geometry descriptor")

    ap.add_argument("--device", default="auto", type=str, help="For CLIP: auto|cpu|cuda")
    ap.add_argument("--clip_model", default="ViT-B-32", type=str)
    ap.add_argument("--clip_pretrained", default="laion2b_s34b_b79k", type=str)
    args = ap.parse_args()

    scene_dir = Path(args.scene_dir).resolve()
    tracklets_path = scene_dir / args.tracklets
    images_dir = scene_dir / args.images_dir
    depth_dir = scene_dir / args.depth_dir
    intrinsics_path = scene_dir / args.intrinsics
    out_dir = scene_dir / args.out_dir

    if not tracklets_path.exists():
        raise SystemExit(f"tracklets not found: {tracklets_path}")
    if not images_dir.exists():
        raise SystemExit(f"images dir not found: {images_dir}")
    if not depth_dir.exists():
        raise SystemExit(f"depth dir not found: {depth_dir}")

    tracklets = json.loads(tracklets_path.read_text(encoding="utf-8"))
    if not isinstance(tracklets, list):
        raise SystemExit(f"Expected a JSON list in: {tracklets_path}")

    intr: dict = {}
    if args.geo_backend != "none":
        if not intrinsics_path.exists():
            raise SystemExit(
                f"intrinsics not found: {intrinsics_path}. "
                f"Run scripts/export_colmap_intrinsics_json.py first, or use --geo_backend none."
            )
        intr = json.loads(intrinsics_path.read_text(encoding="utf-8"))
        if not isinstance(intr, dict) or not intr:
            raise SystemExit(f"Invalid intrinsics.json: {intrinsics_path}")

    rng = np.random.default_rng(int(args.seed))

    # Decide backends (auto -> best available).
    rgb_backend = str(args.rgb_backend)
    clip_model = None
    clip_preprocess = None
    clip_torch = None
    clip_Image = None
    clip_err = None

    device = str(args.device)
    if device == "auto":
        device = "cpu"
        try:
            import torch  # type: ignore

            if torch.cuda.is_available():
                device = "cuda"
        except Exception:
            device = "cpu"

    if rgb_backend in {"auto", "clip"}:
        clip_model, clip_preprocess, clip_torch, clip_Image, clip_err = _try_build_clip(
            device=device, model_name=str(args.clip_model), pretrained=str(args.clip_pretrained)
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

    if geo_backend == "none":
        intr = {}

    print(f"[cfg] rgb_backend={rgb_backend} geo_backend={geo_backend} device={device}")

    track_embs: list[np.ndarray] = []
    track_meta: list[dict] = []

    import cv2  # type: ignore  # noqa: E402

    for t in tracklets:
        if not isinstance(t, dict):
            continue
        track_id = str(t.get("track_id", ""))
        frame_names = list(t.get("frame_names") or [])
        if not frame_names:
            continue

        mask_paths = t.get("mask_paths", None)
        bboxes_xyxy = t.get("bboxes_xyxy", None)

        if mask_paths is not None and len(mask_paths) != len(frame_names):
            raise SystemExit(f"mask_paths length mismatch in track_id={track_id}")
        if bboxes_xyxy is not None and len(bboxes_xyxy) != len(frame_names):
            raise SystemExit(f"bboxes_xyxy length mismatch in track_id={track_id}")

        frame_indices = _pick_frame_indices(len(frame_names), int(args.max_frames_per_track))

        per_frame: list[np.ndarray] = []
        used_frames: list[str] = []

        for i in frame_indices:
            frame_name = str(frame_names[i])
            stem = Path(frame_name).stem

            img_path = images_dir / frame_name
            if not img_path.exists():
                # tolerate different extensions if caller stored .jpg but images are .png (or vice versa)
                cand = list(images_dir.glob(stem + ".*"))
                if not cand:
                    continue
                img_path = cand[0]

            depth_path = depth_dir / f"{stem}.npy"
            if not depth_path.exists():
                continue

            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is None:
                continue
            depth = np.load(str(depth_path))
            if depth.ndim != 2:
                continue

            h, w = img.shape[0], img.shape[1]
            if depth.shape[0] != h or depth.shape[1] != w:
                # Depth should be aligned to images/; if not, skip this frame.
                continue

            mask = None
            if mask_paths is not None:
                mask = _load_mask(scene_dir / str(mask_paths[i]))
                if mask.shape != (h, w):
                    mask = None

            bbox = None
            if mask is not None:
                bbox = _mask_bbox_xyxy(mask)
            if bbox is None and bboxes_xyxy is not None:
                bbox = _clip_bbox_xyxy(bboxes_xyxy[i], w=w, h=h)
            if bbox is None:
                continue

            x1, y1, x2, y2 = bbox
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            # RGB embedding.
            if rgb_backend == "clip":
                rgb_emb = _clip_embed(clip_model, clip_preprocess, clip_torch, clip_Image, device, crop)
            else:
                rgb_emb = _rgb_hist_desc(crop)
            rgb_emb = l2norm(rgb_emb)

            # Geometry embedding.
            if geo_backend == "none":
                geo_emb = np.zeros((0,), dtype=np.float32)
            else:
                if mask is None:
                    mask = _bbox_to_mask(bbox, h=h, w=w)
                K = _get_intrinsics_for_stem(intr, stem)
                pts = _depth_to_points(depth, K=K, mask=mask)
                pts = _sample_points(pts, n=int(args.n_points), rng=rng)
                pts = _normalize_unit_sphere(pts)

                if geo_backend == "open3d_fpfh":
                    geo_emb = _fpfh_global(o3d, pts)
                else:
                    geo_emb = _radial_hist_desc(pts, bins=int(args.geo_bins))
                geo_emb = l2norm(geo_emb)

            fused = np.concatenate([rgb_emb, geo_emb], axis=0) if geo_emb.size else rgb_emb
            per_frame.append(fused.astype(np.float32))
            used_frames.append(frame_name)

        if not per_frame:
            continue

        emb = np.stack(per_frame, axis=0).mean(axis=0).astype(np.float32)
        emb = l2norm(emb)

        track_embs.append(emb)
        track_meta.append(
            {
                "track_id": track_id,
                "object_id": t.get("object_id", None),
                "n_frames_total": int(len(frame_names)),
                "n_frames_used": int(len(per_frame)),
                "used_frames": used_frames,
                "rgb_backend": rgb_backend,
                "geo_backend": geo_backend,
                "dim": int(emb.shape[0]),
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    if track_embs:
        embs = np.stack(track_embs, axis=0).astype(np.float32)
    else:
        embs = np.zeros((0, 0), dtype=np.float32)

    np.save(str(out_dir / "tracks.npy"), embs)
    (out_dir / "tracks_meta.json").write_text(json.dumps(track_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote embeddings: {out_dir / 'tracks.npy'} shape={tuple(embs.shape)}")
    print(f"Wrote metadata:   {out_dir / 'tracks_meta.json'} tracks={len(track_meta)}")


if __name__ == "__main__":
    main()
