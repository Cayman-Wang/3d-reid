from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


DEFAULT_PROMPT = "aircraft . airplane . fighter jet . drone . UAV . unmanned aerial vehicle ."


def _sorted_frames(images_dir: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png"}
    frames = [p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]

    def _key(path: Path):
        try:
            return (0, int(path.stem))
        except Exception:
            return (1, path.stem)

    return sorted(frames, key=_key)


def _resolve_device(device: str, torch) -> str:
    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _sample_probe_indices(total: int) -> list[int]:
    if total <= 0:
        return []
    picks = [0, total // 2, total - 1]
    out: list[int] = []
    seen: set[int] = set()
    for idx in picks:
        idx = max(0, min(total - 1, int(idx)))
        if idx not in seen:
            out.append(idx)
            seen.add(idx)
    return out


def _box_area_ratio(box_xyxy: list[float], width: int, height: int) -> float:
    x0, y0, x1, y1 = [float(v) for v in box_xyxy]
    area = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    denom = max(float(width * height), 1.0)
    return float(area / denom)


def _normalize_detection_box(box: Any) -> list[float]:
    if isinstance(box, dict):
        return [float(box["xmin"]), float(box["ymin"]), float(box["xmax"]), float(box["ymax"])]
    if isinstance(box, (list, tuple)) and len(box) == 4:
        return [float(v) for v in box]
    raise SystemExit(f"Unsupported detector box payload: {box!r}")


def _choose_best_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    if not candidates:
        raise SystemExit("Detector produced no candidates to choose from")
    reasonable = [item for item in candidates if bool(item["area_reasonable"])]
    pool = reasonable if reasonable else candidates
    pool.sort(
        key=lambda item: (
            float(item["score"]),
            -abs(float(item["area_ratio"]) - float(item["target_area_ratio"])),
            -float(item["area_ratio"]),
        ),
        reverse=True,
    )
    return pool[0]


def _draw_visualization(
    image: Image.Image,
    detections: list[dict[str, Any]],
    selected_box: list[float] | None,
    out_path: Path,
) -> None:
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    for item in detections:
        box = item["bbox_xyxy"]
        score = float(item["score"])
        color = "lime" if selected_box is not None and [round(v, 3) for v in box] == [round(v, 3) for v in selected_box] else "yellow"
        draw.rectangle(box, outline=color, width=3)
        draw.text((box[0] + 2, max(0.0, box[1] - 14.0)), f"{score:.3f}", fill=color)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Detect per-camera text-prompt boxes for node scenes using GroundingDINO.")
    ap.add_argument("--scene_dir", required=True, type=str)
    ap.add_argument("--cams", default="cam0,cam1,cam2", type=str)
    ap.add_argument("--frames_subdir", default="frames", type=str)
    ap.add_argument("--out_json", default="cams/auto_prompt_boxes_text_prompt_v1.json", type=str)
    ap.add_argument("--vis_subdir", default="auto_prompt_boxes_text_prompt_v1", type=str)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT, type=str)
    ap.add_argument("--model_id", default="IDEA-Research/grounding-dino-tiny", type=str)
    ap.add_argument("--device", default="auto", type=str)
    ap.add_argument("--box_threshold", default=0.20, type=float)
    ap.add_argument("--text_threshold", default=0.20, type=float)
    ap.add_argument("--min_area_ratio", default=0.0005, type=float)
    ap.add_argument("--max_area_ratio", default=0.60, type=float)
    ap.add_argument("--target_area_ratio", default=0.03, type=float)
    args = ap.parse_args()

    scene_dir = Path(str(args.scene_dir)).resolve()
    if not scene_dir.exists():
        raise SystemExit(f"--scene_dir not found: {scene_dir}")

    try:
        import torch  # type: ignore
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor  # type: ignore
    except Exception as exc:
        raise SystemExit(
            "GroundingDINO dependencies are not available. Install transformers + torch before running this script. "
            f"Original import error: {exc!r}"
        )

    device = _resolve_device(str(args.device), torch)
    processor = AutoProcessor.from_pretrained(str(args.model_id), use_fast=False)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(str(args.model_id)).to(device).eval()

    cams = [c.strip() for c in str(args.cams).split(",") if c.strip()]
    if not cams:
        raise SystemExit("--cams is empty")

    detections_out: list[dict[str, Any]] = []
    probe_summary: dict[str, list[dict[str, Any]]] = {}

    for cam_id in cams:
        frames_dir = scene_dir / "cams" / cam_id / str(args.frames_subdir)
        if not frames_dir.exists():
            raise SystemExit(f"frames dir not found for {cam_id}: {frames_dir}")
        frames = _sorted_frames(frames_dir)
        if not frames:
            raise SystemExit(f"no frames found for {cam_id}: {frames_dir}")

        probe_indices = _sample_probe_indices(len(frames))
        cam_candidates: list[dict[str, Any]] = []
        cam_probe_rows: list[dict[str, Any]] = []

        for frame_idx in probe_indices:
            frame_path = frames[frame_idx]
            image = Image.open(frame_path).convert("RGB")
            inputs = processor(images=image, text=str(args.prompt), return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)

            results = processor.post_process_grounded_object_detection(
                outputs,
                inputs["input_ids"],
                threshold=float(args.box_threshold),
                text_threshold=float(args.text_threshold),
                target_sizes=[(image.height, image.width)],
            )
            result = results[0]
            frame_detections: list[dict[str, Any]] = []
            boxes = result.get("boxes")
            scores = result.get("scores")
            labels = result.get("labels")
            if boxes is None or scores is None or labels is None:
                boxes = []
                scores = []
                labels = []
            for box, score, label in zip(boxes, scores, labels):
                bbox_xyxy = _normalize_detection_box(box.tolist() if hasattr(box, "tolist") else box)
                area_ratio = _box_area_ratio(bbox_xyxy, image.width, image.height)
                candidate = {
                    "cam_id": cam_id,
                    "frame_idx": int(frame_idx),
                    "frame_stem": frame_path.stem,
                    "bbox_xyxy": bbox_xyxy,
                    "score": float(score.item() if hasattr(score, "item") else score),
                    "label": str(label),
                    "prompt": str(args.prompt),
                    "detector_backend": "transformers_groundingdino",
                    "model_id": str(args.model_id),
                    "area_ratio": float(area_ratio),
                    "area_reasonable": bool(float(args.min_area_ratio) <= area_ratio <= float(args.max_area_ratio)),
                    "target_area_ratio": float(args.target_area_ratio),
                }
                frame_detections.append(candidate)
                cam_candidates.append(candidate)

            _draw_visualization(
                image=image,
                detections=frame_detections,
                selected_box=None,
                out_path=scene_dir / "cams" / cam_id / str(args.vis_subdir) / f"{frame_path.stem}.png",
            )
            cam_probe_rows.append(
                {
                    "frame_idx": int(frame_idx),
                    "frame_stem": frame_path.stem,
                    "num_detections": int(len(frame_detections)),
                    "detections": frame_detections,
                }
            )

        if not cam_candidates:
            raise SystemExit(f"Detector produced no boxes for {cam_id} with prompt={args.prompt!r}")

        best = _choose_best_candidate(cam_candidates)
        best_record = {
            "cam_id": cam_id,
            "init_frame": int(best["frame_idx"]),
            "frame_stem": str(best["frame_stem"]),
            "bbox_xyxy": [int(round(v)) for v in best["bbox_xyxy"]],
            "score": float(best["score"]),
            "prompt": str(args.prompt),
            "detector_backend": str(best["detector_backend"]),
            "model_id": str(best["model_id"]),
            "area_ratio": float(best["area_ratio"]),
            "area_reasonable": bool(best["area_reasonable"]),
        }
        detections_out.append(best_record)
        probe_summary[cam_id] = cam_probe_rows

        best_image = Image.open(frames[best_record["init_frame"]]).convert("RGB")
        matching_frame_detections = [
            item for item in cam_candidates if int(item["frame_idx"]) == int(best_record["init_frame"])
        ]
        _draw_visualization(
            image=best_image,
            detections=matching_frame_detections,
            selected_box=[float(v) for v in best_record["bbox_xyxy"]],
            out_path=scene_dir / "cams" / cam_id / str(args.vis_subdir) / f"{best_record['frame_stem']}_selected.png",
        )

    out_path = scene_dir / str(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "node_text_prompt_boxes_v1",
        "scene_id": scene_dir.name,
        "scene_dir": scene_dir.as_posix(),
        "prompt": str(args.prompt),
        "detector_backend": "transformers_groundingdino",
        "model_id": str(args.model_id),
        "detections": detections_out,
        "probe_summary": probe_summary,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
