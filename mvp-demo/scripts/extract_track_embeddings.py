from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def l2norm(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(x))
    return (x / (n + eps)).astype(np.float32)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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


def _lineage_lookup_keys(frame_name: str) -> list[str]:
    p = Path(str(frame_name))
    keys = [str(frame_name), p.name, p.stem]
    out: list[str] = []
    seen: set[str] = set()
    for key in keys:
        if key and key not in seen:
            out.append(key)
            seen.add(key)
    return out


def _index_tracklet_lineage(rows: object, track_id: str, require: bool) -> dict[str, dict | None]:
    if rows is None:
        if require:
            raise SystemExit(f"tracklet input_lineage missing in track_id={track_id}")
        return {}
    if not isinstance(rows, list):
        raise SystemExit(f"tracklet input_lineage must be a list in track_id={track_id}")

    index: dict[str, dict | None] = {}
    for row_i, row in enumerate(rows):
        if not isinstance(row, dict):
            raise SystemExit(f"tracklet input_lineage[{row_i}] must be an object in track_id={track_id}")

        row_keys: list[str] = []
        frame_name = row.get("frame_name", "")
        frame_path = row.get("frame_path", "")
        if frame_name:
            row_keys.extend(_lineage_lookup_keys(str(frame_name)))
        if frame_path:
            row_keys.extend(_lineage_lookup_keys(Path(str(frame_path)).name))
            row_keys.extend(_lineage_lookup_keys(str(frame_path)))

        if not row_keys:
            raise SystemExit(f"tracklet input_lineage[{row_i}] missing frame_name/frame_path in track_id={track_id}")

        for key in row_keys:
            if key in index:
                existing = index[key]
                if existing is not row:
                    index[key] = None
                continue
            index[key] = row

    if require and not index:
        raise SystemExit(f"tracklet input_lineage empty in track_id={track_id}")
    return index


def _find_tracklet_lineage(index: dict[str, dict | None], frame_name: str, track_id: str) -> dict | None:
    for key in _lineage_lookup_keys(frame_name):
        if key not in index:
            continue
        row = index[key]
        if row is None:
            raise SystemExit(f"Ambiguous tracklet input_lineage key={key!r} in track_id={track_id}")
        return row
    return None


def _normalize_bbox_for_compare(bbox: object) -> list[int] | None:
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    try:
        return [int(round(float(v))) for v in bbox]
    except Exception:
        return None


def _validate_tracklet_lineage_hashes(
    *,
    track_id: str,
    frame_name: str,
    lineage: dict,
    frame_sha256: str,
    mask_sha256: str | None,
    bbox_xyxy: list[int] | None,
) -> None:
    expected_frame_sha256 = lineage.get("frame_sha256")
    if not expected_frame_sha256:
        raise SystemExit(f"tracklet lineage missing frame_sha256 in track_id={track_id} frame_name={frame_name}")
    if str(expected_frame_sha256) != frame_sha256:
        raise SystemExit(
            f"tracklet lineage frame_sha256 mismatch in track_id={track_id} frame_name={frame_name}: "
            f"expected={expected_frame_sha256} actual={frame_sha256}"
        )

    expected_mask_sha256 = lineage.get("mask_sha256")
    if mask_sha256 is not None and not expected_mask_sha256:
        raise SystemExit(f"tracklet lineage missing mask_sha256 in track_id={track_id} frame_name={frame_name}")
    if expected_mask_sha256:
        if mask_sha256 is None:
            raise SystemExit(
                f"tracklet lineage has mask_sha256 but no mask was used in track_id={track_id} "
                f"frame_name={frame_name}"
            )
        if str(expected_mask_sha256) != mask_sha256:
            raise SystemExit(
                f"tracklet lineage mask_sha256 mismatch in track_id={track_id} frame_name={frame_name}: "
                f"expected={expected_mask_sha256} actual={mask_sha256}"
            )

    expected_bbox = _normalize_bbox_for_compare(lineage.get("bbox_xyxy"))
    if expected_bbox is not None and bbox_xyxy is not None and expected_bbox != bbox_xyxy:
        raise SystemExit(
            f"tracklet lineage bbox mismatch in track_id={track_id} frame_name={frame_name}: "
            f"expected={expected_bbox} actual={bbox_xyxy}"
        )


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
    ap.add_argument(
        "--require-tracklet-lineage",
        action="store_true",
        help="Require tracklet input_lineage and fail if current frame/mask hashes no longer match it.",
    )
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
    tracklets_sha256 = _sha256_file(tracklets_path)

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

        tracklet_lineage_rows = t.get("input_lineage", None)
        tracklet_lineage_index = _index_tracklet_lineage(
            tracklet_lineage_rows,
            track_id=track_id,
            require=bool(args.require_tracklet_lineage),
        )
        tracklet_lineage_count = len(tracklet_lineage_rows) if isinstance(tracklet_lineage_rows, list) else 0

        frame_indices = _pick_frame_indices(len(frame_names), int(args.max_frames_per_track))

        per_frame: list[np.ndarray] = []
        used_frames: list[str] = []
        input_lineage: list[dict] = []

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
            frame_sha256 = _sha256_file(img_path)
            depth_sha256 = _sha256_file(depth_path)

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
            mask_rel = None
            mask_sha256 = None
            if mask_paths is not None:
                mask_rel = str(mask_paths[i])
                mask_path = scene_dir / mask_rel
                if not mask_path.exists():
                    continue
                mask_sha256 = _sha256_file(mask_path)
                mask = _load_mask(mask_path)
                if mask.shape != (h, w):
                    mask = None

            bbox = None
            if mask is not None:
                bbox = _mask_bbox_xyxy(mask)
            if bbox is None and bboxes_xyxy is not None:
                bbox = _clip_bbox_xyxy(bboxes_xyxy[i], w=w, h=h)
            if bbox is None:
                continue

            tracklet_lineage_row = _find_tracklet_lineage(tracklet_lineage_index, frame_name, track_id=track_id)
            if args.require_tracklet_lineage:
                if tracklet_lineage_row is None:
                    raise SystemExit(f"tracklet lineage missing for track_id={track_id} frame_name={frame_name}")
                _validate_tracklet_lineage_hashes(
                    track_id=track_id,
                    frame_name=frame_name,
                    lineage=tracklet_lineage_row,
                    frame_sha256=frame_sha256,
                    mask_sha256=mask_sha256,
                    bbox_xyxy=bbox,
                )

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
            input_lineage.append(
                {
                    "schema_version": "track_embedding_input_lineage_v1",
                    "frame_name": frame_name,
                    "frame_path": str(img_path.relative_to(scene_dir)) if img_path.is_relative_to(scene_dir) else str(img_path),
                    "frame_sha256": frame_sha256,
                    "depth_path": str(depth_path.relative_to(scene_dir)) if depth_path.is_relative_to(scene_dir) else str(depth_path),
                    "depth_sha256": depth_sha256,
                    "mask_path": mask_rel,
                    "mask_sha256": mask_sha256,
                    "bbox_xyxy": bbox,
                    "tracklet_lineage_present": tracklet_lineage_row is not None,
                    "tracklet_lineage_verified": bool(args.require_tracklet_lineage),
                }
            )

        if not per_frame:
            continue

        emb = np.stack(per_frame, axis=0).mean(axis=0).astype(np.float32)
        emb = l2norm(emb)

        track_embs.append(emb)
        track_meta.append(
            {
                "track_id": track_id,
                "schema_version": "track_embedding_meta_v2",
                "object_id": t.get("object_id", None),
                "identity_id": t.get("identity_id", None),
                "scene_dir": str(scene_dir),
                "tracklets_path": str(tracklets_path),
                "tracklets_sha256": tracklets_sha256,
                "tracklet_schema_version": t.get("schema_version", ""),
                "tracklet_source": t.get("source", ""),
                "tracklet_diagnostic_only": bool(t.get("diagnostic_only", False)),
                "tracklet_identity_proof": bool(t.get("identity_proof", False)),
                "tracklet_pixel_accurate": bool(t.get("pixel_accurate", False)),
                "tracklet_formal_synthetic_annotation_ready": bool(
                    t.get("formal_synthetic_annotation_ready", False)
                ),
                "tracklet_lineage_required": bool(args.require_tracklet_lineage),
                "tracklet_input_lineage_count": int(tracklet_lineage_count),
                "n_frames_total": int(len(frame_names)),
                "n_frames_used": int(len(per_frame)),
                "used_frames": used_frames,
                "input_lineage": input_lineage,
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
