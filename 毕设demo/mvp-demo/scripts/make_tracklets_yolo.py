from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path


def _parse_classes_arg(model, classes: str) -> list[int] | None:
    """
    Parse `--classes` which may contain comma-separated class ids or names.
    Returns None when empty (meaning "all classes").
    """
    classes = (classes or "").strip()
    if not classes:
        return None

    raw = [c.strip() for c in classes.split(",") if c.strip()]
    names = getattr(model, "names", None)
    if not names:
        # If we can't resolve names, only accept ints.
        out: list[int] = []
        for t in raw:
            out.append(int(t))
        return sorted(set(out))

    if isinstance(names, dict):
        id_to_name = {int(k): str(v) for k, v in names.items()}
    else:
        id_to_name = {i: str(v) for i, v in enumerate(list(names))}
    name_to_id = {v: k for k, v in id_to_name.items()}

    out: list[int] = []
    for t in raw:
        if t.isdigit() or (t.startswith("-") and t[1:].isdigit()):
            out.append(int(t))
            continue
        if t not in name_to_id:
            known = ", ".join(sorted(name_to_id.keys())[:30])
            raise SystemExit(f"Unknown class name in --classes: {t}. Known (subset): {known}")
        out.append(int(name_to_id[t]))
    return sorted(set(out))


def main() -> None:
    ap = argparse.ArgumentParser(description="Run YOLO+tracker on SCENE_DIR/images and export tracklets.json (bbox-only).")
    ap.add_argument("--scene_dir", required=True, type=str)
    ap.add_argument("--images_dir", default="images", type=str)
    ap.add_argument("--out", default="tracklets.json", type=str)
    ap.add_argument("--model", default="yolov8n.pt", type=str, help="YOLO weights path or model name")
    ap.add_argument("--tracker", default="bytetrack.yaml", type=str, help="Ultralytics tracker config (bytetrack/botsort)")
    ap.add_argument("--conf", default=0.25, type=float)
    ap.add_argument("--imgsz", default=640, type=int)
    ap.add_argument("--device", default="", type=str, help="''=auto, or 'cpu', '0', '0,1', ...")
    ap.add_argument("--classes", default="", type=str, help="Comma-separated class ids or names; empty means all")
    ap.add_argument("--min_frames", default=3, type=int, help="Drop tracks shorter than this")
    ap.add_argument("--save_vis", action="store_true", help="Save visualization frames with track ids")
    ap.add_argument("--vis_dir", default="tracks_vis", type=str, help="Relative dir under scene_dir (when --save_vis)")
    args = ap.parse_args()

    scene_dir = Path(args.scene_dir).resolve()
    images_dir = scene_dir / args.images_dir
    out_path = scene_dir / args.out

    if not images_dir.exists():
        raise SystemExit(f"images dir not found: {images_dir}")

    # Workaround: some PyTorch builds reference optional iJIT symbols (Intel JIT profiling).
    # Using RTLD_LAZY defers symbol resolution so `import torch` can succeed.
    if sys.platform.startswith("linux") and hasattr(sys, "setdlopenflags") and hasattr(os, "RTLD_LAZY"):
        try:
            sys.setdlopenflags(os.RTLD_GLOBAL | os.RTLD_LAZY)
        except Exception:
            pass

    from ultralytics import YOLO  # noqa: E402

    model = YOLO(args.model)
    classes = _parse_classes_arg(model, args.classes)

    results = model.track(
        source=str(images_dir),
        stream=True,
        persist=True,
        tracker=str(args.tracker),
        conf=float(args.conf),
        imgsz=int(args.imgsz),
        device=str(args.device) if str(args.device).strip() else None,
        classes=classes,
        verbose=False,
    )

    # track_id -> lists aligned by frame order
    tracks: dict[int, dict] = defaultdict(lambda: {"frame_names": [], "bboxes_xyxy": [], "class_ids": [], "confs": []})

    vis_dir = scene_dir / args.vis_dir
    if args.save_vis:
        vis_dir.mkdir(parents=True, exist_ok=True)
        import cv2  # noqa: E402

    for res in results:
        frame_path = Path(getattr(res, "path", ""))
        frame_name = frame_path.name if frame_path.name else "frame.jpg"

        boxes = getattr(res, "boxes", None)
        if boxes is None:
            continue
        if getattr(boxes, "id", None) is None:
            continue

        ids = boxes.id.detach().cpu().numpy().astype(int)
        xyxy = boxes.xyxy.detach().cpu().numpy()
        cls = boxes.cls.detach().cpu().numpy().astype(int) if getattr(boxes, "cls", None) is not None else None
        conf = boxes.conf.detach().cpu().numpy() if getattr(boxes, "conf", None) is not None else None

        for i, tid in enumerate(ids.tolist()):
            bbox = [float(v) for v in xyxy[i].tolist()]
            tracks[tid]["frame_names"].append(frame_name)
            tracks[tid]["bboxes_xyxy"].append(bbox)
            tracks[tid]["class_ids"].append(int(cls[i]) if cls is not None else -1)
            tracks[tid]["confs"].append(float(conf[i]) if conf is not None else -1.0)

        if args.save_vis:
            vis = res.plot()
            cv2.imwrite(str(vis_dir / frame_name), vis)

    tracklets: list[dict] = []
    for tid, t in sorted(tracks.items(), key=lambda kv: kv[0]):
        if len(t["frame_names"]) < int(args.min_frames):
            continue
        tracklets.append(
            {
                "track_id": f"{scene_dir.name}_track_{tid:04d}",
                "scene_dir": str(scene_dir),
                "object_id": tid,
                "frame_names": t["frame_names"],
                "bboxes_xyxy": t["bboxes_xyxy"],
                "class_ids": t["class_ids"],
                "confs": t["confs"],
                "tracker": str(args.tracker),
                "det_model": str(args.model),
            }
        )

    out_path.write_text(json.dumps(tracklets, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(tracklets)} tracklets to: {out_path}")
    if args.save_vis:
        print(f"Wrote visualizations to: {vis_dir}")


if __name__ == "__main__":
    main()
