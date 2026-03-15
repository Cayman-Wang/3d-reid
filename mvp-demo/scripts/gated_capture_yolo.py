from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Deque, List, Optional, Tuple


def _parse_csv_tokens(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [t.strip() for t in value.split(",") if t.strip()]


def _try_int(value: str) -> Optional[int]:
    try:
        return int(value)
    except Exception:
        return None


@dataclass
class GateConfig:
    model: str
    gate_classes: List[str]
    conf_th: float
    imgsz: int
    device: str
    detect_fps: float
    k_on: int
    k_off: int


@dataclass
class CaptureConfig:
    source: str
    mode: str
    scene_root: str
    scene_prefix: str
    pre_seconds: float
    fps_save: float
    max_seconds: float
    min_record_seconds: float
    save_max_side: int
    jpeg_quality: int
    multi_scene: bool
    display: bool


@dataclass
class SceneSummary:
    scene_id: str
    scene_dir: str
    input_dir: str
    started_at_iso: str
    stopped_at_iso: str
    start_ts_sec: float
    stop_ts_sec: float
    buffered_frames_flushed: int
    recorded_frames_after_trigger: int
    total_saved_frames: int
    stop_reason: str


def _now_id(prefix: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{ts}"


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _resize_max_side(frame, max_side: int):
    if max_side <= 0:
        return frame
    h, w = frame.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return frame
    scale = max_side / float(m)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    import cv2

    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _open_capture(source: str):
    import cv2

    camera_index = _try_int(source)
    if camera_index is not None and str(camera_index) == source.strip():
        # Use a platform-appropriate backend. CAP_DSHOW is Windows-only; on Linux prefer V4L2.
        if sys.platform.startswith("win"):
            cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
            if not cap.isOpened():  # Fallback for environments without DirectShow.
                cap = cv2.VideoCapture(camera_index)
        elif sys.platform.startswith("linux") and hasattr(cv2, "CAP_V4L2"):
            cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
            if not cap.isOpened():  # Fallback when V4L2 is unavailable/misconfigured.
                cap = cv2.VideoCapture(camera_index)
        else:
            cap = cv2.VideoCapture(camera_index)
        return cap, "camera"
    cap = cv2.VideoCapture(source)
    return cap, "file_or_stream"


def _get_source_fps(cap, default_fps: float = 30.0) -> float:
    import cv2

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1e-3:
        return default_fps
    return float(fps)


def _resolve_gate_class_ids(model, gate_classes: List[str]) -> Optional[List[int]]:
    if not gate_classes:
        return None

    names = getattr(model, "names", None)
    if not names:
        raise SystemExit("YOLO model has no `.names`; cannot resolve `--gate_classes`.")

    # Ultralytics: names can be dict[int,str] or list[str].
    if isinstance(names, dict):
        id_to_name = {int(k): str(v) for k, v in names.items()}
    else:
        id_to_name = {i: str(v) for i, v in enumerate(list(names))}

    name_to_id = {v: k for k, v in id_to_name.items()}

    resolved: List[int] = []
    for token in gate_classes:
        as_int = _try_int(token)
        if as_int is not None:
            if as_int not in id_to_name:
                raise SystemExit(f"Invalid class id in --gate_classes: {as_int}")
            resolved.append(as_int)
            continue
        if token not in name_to_id:
            known = ", ".join(sorted(name_to_id.keys())[:30])
            raise SystemExit(f"Unknown class name in --gate_classes: {token}. Known (subset): {known}")
        resolved.append(name_to_id[token])

    return sorted(set(resolved))


def _should_run(ts_sec: float, last_ts_sec: float, target_fps: float) -> bool:
    if target_fps <= 0:
        return True
    if last_ts_sec < 0:
        return True
    return (ts_sec - last_ts_sec) >= (1.0 / target_fps)


def _write_frame_jpg(path: Path, frame, jpeg_quality: int) -> None:
    import cv2

    ok = cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
    if not ok:
        raise RuntimeError(f"Failed to write image: {path}")


def _launch_3dgs_pipeline(
    *,
    scene_dir: Path,
    scene_id: str,
    gs_env: str,
    gs_repo: Path,
    model_root: Path,
    gs_resize: bool,
    gs_max_iter: str,
    gs_mode: str,
) -> None:
    """
    Launch 3DGS (convert.py -> train.py -> depth) using the gaussian_splatting conda env.

    This script is intended to run in the lightweight capture env; we use `conda run`
    so that 3DGS dependencies live in a separate env.
    """

    run_script = Path(__file__).with_name("run_3dgs_scene.py").resolve()
    model_dir = (model_root / scene_id).resolve()
    model_dir.mkdir(parents=True, exist_ok=True)
    log_path = model_dir / "run_3dgs.log"

    if not run_script.exists():
        print(f"[3dgs] skip (run_3dgs_scene.py not found): {run_script}")
        return
    if not gs_repo.exists():
        print(f"[3dgs] skip (--gs_repo not found): {gs_repo}")
        return

    cmd = [
        "conda",
        "run",
        "-n",
        gs_env,
        "python",
        str(run_script),
        "--gs_repo",
        str(gs_repo),
        "--scene_dir",
        str(scene_dir),
        "--model_dir",
        str(model_dir),
    ]
    if gs_resize:
        cmd.append("--resize")
    if gs_max_iter:
        cmd.extend(["--max_iter", gs_max_iter])

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")

    if gs_mode == "blocking":
        print(f"[3dgs] RUN (blocking) scene={scene_id} model_dir={model_dir}")
        subprocess.run(cmd, check=True, env=env)
        print(f"[3dgs] DONE (blocking) scene={scene_id} model_dir={model_dir}")
        return

    # Default: background run; keep the capture loop responsive (esp. --multi_scene).
    with log_path.open("a", encoding="utf-8") as log_f:
        p = subprocess.Popen(
            cmd,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,  # detach from current terminal session
        )
    print(f"[3dgs] RUN (background) scene={scene_id} pid={p.pid} log={log_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="YOLO gating → capture frames to a 3DGS scene folder (input/)")
    ap.add_argument("--source", required=True, type=str, help="Video path/URL or camera index (e.g., 0)")
    ap.add_argument("--mode", default="auto", choices=["auto", "offline", "realtime"])
    default_scene_root = Path(__file__).resolve().parents[1] / "data" / "scenes"
    ap.add_argument("--scene_root", default=str(default_scene_root), type=str)
    ap.add_argument("--scene_prefix", default="cam1", type=str)

    ap.add_argument("--model", default="yolov8n.pt", type=str)
    ap.add_argument("--gate_classes", default="person", type=str, help="Comma-separated class names/ids; empty = any")
    ap.add_argument("--conf_th", default=0.5, type=float)
    ap.add_argument("--imgsz", default=640, type=int)
    ap.add_argument("--device", default="", type=str, help="Ultralytics device string, e.g. '0', 'cpu'")
    ap.add_argument("--detect_fps", default=5.0, type=float)
    ap.add_argument("--k_on", default=3, type=int)
    ap.add_argument("--k_off", default=15, type=int)

    ap.add_argument("--pre_seconds", default=3.0, type=float)
    ap.add_argument("--fps_save", default=3.0, type=float)
    ap.add_argument("--max_seconds", default=30.0, type=float)
    ap.add_argument("--min_record_seconds", default=5.0, type=float)

    ap.add_argument("--save_max_side", default=0, type=int, help="Downscale saved frames if max(H,W) > this value; 0=keep")
    ap.add_argument("--jpeg_quality", default=95, type=int)

    ap.add_argument("--multi_scene", action="store_true", help="Keep running and record multiple scenes")
    ap.add_argument("--display", action="store_true", help="Show a preview window with state overlay")

    # Optional: auto-trigger 3DGS after a scene is recorded.
    default_gs_repo = Path(__file__).resolve().parents[1] / "third_party" / "gaussian-splatting"
    default_model_root = Path(__file__).resolve().parents[1] / "output"
    ap.add_argument("--auto_3dgs", action="store_true", help="Auto-run 3DGS when a scene stops recording")
    ap.add_argument("--gs_env", default="gaussian_splatting", type=str, help="Conda env name for 3DGS (conda run -n)")
    ap.add_argument("--gs_repo", default=str(default_gs_repo), type=str, help="Path to gaussian-splatting repo")
    ap.add_argument("--model_root", default=str(default_model_root), type=str, help="Where to create output/<scene_id>/")
    ap.add_argument("--gs_resize", action="store_true", help="Forward --resize to 3DGS convert.py")
    ap.add_argument(
        "--gs_max_iter",
        default="",
        type=str,
        help="Forward extra args to 3DGS train.py via run_3dgs_scene.py --max_iter (e.g. '--iterations 7000')",
    )
    ap.add_argument(
        "--gs_mode",
        default="background",
        choices=["background", "blocking"],
        help="How to run 3DGS after capture (background recommended for --multi_scene)",
    )
    args = ap.parse_args()

    # Workaround: some PyTorch builds reference optional iJIT symbols (Intel JIT profiling).
    # Using RTLD_LAZY defers symbol resolution so `import torch` can succeed.
    if sys.platform.startswith("linux") and hasattr(sys, "setdlopenflags") and hasattr(os, "RTLD_LAZY"):
        try:
            sys.setdlopenflags(os.RTLD_GLOBAL | os.RTLD_LAZY)
        except Exception:
            pass

    import cv2
    from ultralytics import YOLO

    gate = GateConfig(
        model=args.model,
        gate_classes=_parse_csv_tokens(args.gate_classes),
        conf_th=float(args.conf_th),
        imgsz=int(args.imgsz),
        device=str(args.device),
        detect_fps=float(args.detect_fps),
        k_on=int(args.k_on),
        k_off=int(args.k_off),
    )
    capture = CaptureConfig(
        source=args.source,
        mode=args.mode,
        scene_root=args.scene_root,
        scene_prefix=args.scene_prefix,
        pre_seconds=float(args.pre_seconds),
        fps_save=float(args.fps_save),
        max_seconds=float(args.max_seconds),
        min_record_seconds=float(args.min_record_seconds),
        save_max_side=int(args.save_max_side),
        jpeg_quality=int(args.jpeg_quality),
        multi_scene=bool(args.multi_scene),
        display=bool(args.display),
    )

    cap, source_type = _open_capture(capture.source)
    if not cap.isOpened():
        raise SystemExit(f"Failed to open source: {capture.source}")

    source_fps = _get_source_fps(cap)
    if capture.mode == "auto":
        # If it's a local file path, offline timestamps are more meaningful; otherwise treat as realtime stream.
        mode = "offline" if Path(capture.source).exists() else "realtime"
    else:
        mode = capture.mode

    model = YOLO(gate.model)
    gate_class_ids = _resolve_gate_class_ids(model, gate.gate_classes)

    pre_max_frames = max(1, int(round(capture.pre_seconds * max(1.0, capture.fps_save))))
    buffer: Deque[Tuple[float, Any]] = deque(maxlen=pre_max_frames)

    state = "IDLE"  # IDLE | RECORD
    detect_on = 0
    detect_off = 0

    last_detect_ts = -1.0
    last_sample_ts = -1.0
    last_save_ts = -1.0

    frame_idx = -1
    real_start = time.monotonic()

    current_scene_dir: Optional[Path] = None
    current_input_dir: Optional[Path] = None
    scene_id: Optional[str] = None
    scene_started_at: Optional[datetime] = None
    scene_stopped_at: Optional[datetime] = None
    record_start_ts: Optional[float] = None
    stop_reason = ""

    buffered_flushed = 0
    recorded_after_trigger = 0
    save_index = 0
    frame_times: List[Tuple[str, float]] = []

    def ts_sec_for_frame(frame_index: int) -> float:
        if mode == "offline":
            return float(frame_index) / float(source_fps)
        return time.monotonic() - real_start

    def start_scene(ts_sec: float) -> None:
        nonlocal state, current_scene_dir, current_input_dir, scene_id, scene_started_at
        nonlocal record_start_ts, save_index, buffered_flushed, recorded_after_trigger, frame_times, last_save_ts

        state = "RECORD"
        record_start_ts = ts_sec
        scene_started_at = datetime.now()
        scene_id = _now_id(capture.scene_prefix)
        current_scene_dir = Path(capture.scene_root) / scene_id
        current_input_dir = current_scene_dir / "input"
        _ensure_dir(current_input_dir)
        save_index = 0
        recorded_after_trigger = 0
        frame_times = []

        # Flush buffered frames first.
        buffered_flushed = 0
        while buffer:
            t_buf, f_buf = buffer.popleft()
            out_name = f"{save_index:06d}.jpg"
            out_path = current_input_dir / out_name
            _write_frame_jpg(out_path, f_buf, capture.jpeg_quality)
            frame_times.append((out_name, t_buf))
            save_index += 1
            buffered_flushed += 1

        last_save_ts = ts_sec
        print(f"[gate] START scene={scene_id} dir={current_scene_dir}")

    def stop_scene(ts_sec: float, reason: str) -> SceneSummary:
        nonlocal state, current_scene_dir, current_input_dir, scene_stopped_at, stop_reason
        nonlocal record_start_ts, recorded_after_trigger, frame_times

        state = "IDLE"
        stop_reason = reason
        scene_stopped_at = datetime.now()

        assert current_scene_dir is not None
        assert current_input_dir is not None
        assert scene_id is not None
        assert record_start_ts is not None

        meta = {
            "scene_id": scene_id,
            "source": capture.source,
            "source_type": source_type,
            "mode": mode,
            "source_fps": source_fps,
            "gate": asdict(gate),
            "capture": asdict(capture),
            "record": {
                "start_ts_sec": record_start_ts,
                "stop_ts_sec": ts_sec,
                "buffered_frames_flushed": buffered_flushed,
                "recorded_frames_after_trigger": recorded_after_trigger,
                "total_saved_frames": buffered_flushed + recorded_after_trigger,
                "stop_reason": reason,
                "started_at_iso": scene_started_at.isoformat() if scene_started_at else "",
                "stopped_at_iso": scene_stopped_at.isoformat() if scene_stopped_at else "",
            },
        }
        (current_scene_dir / "capture_meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        with (current_scene_dir / "frame_times.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["filename", "timestamp_sec"])
            for name, t in frame_times:
                w.writerow([name, f"{t:.6f}"])

        summary = SceneSummary(
            scene_id=scene_id,
            scene_dir=str(current_scene_dir),
            input_dir=str(current_input_dir),
            started_at_iso=scene_started_at.isoformat() if scene_started_at else "",
            stopped_at_iso=scene_stopped_at.isoformat() if scene_stopped_at else "",
            start_ts_sec=float(record_start_ts),
            stop_ts_sec=float(ts_sec),
            buffered_frames_flushed=int(buffered_flushed),
            recorded_frames_after_trigger=int(recorded_after_trigger),
            total_saved_frames=int(buffered_flushed + recorded_after_trigger),
            stop_reason=reason,
        )

        print(f"[gate] STOP  scene={scene_id} reason={reason} frames={summary.total_saved_frames}")

        # Reset scene locals.
        current_scene_dir = None
        current_input_dir = None
        record_start_ts = None
        recorded_after_trigger = 0
        frame_times = []
        return summary

    summaries: List[SceneSummary] = []
    gs_repo = Path(args.gs_repo).resolve()
    model_root = Path(args.model_root).resolve()

    def on_scene_stopped(summary: SceneSummary) -> None:
        summaries.append(summary)
        if not args.auto_3dgs:
            return
        try:
            _launch_3dgs_pipeline(
                scene_dir=Path(summary.scene_dir),
                scene_id=summary.scene_id,
                gs_env=str(args.gs_env),
                gs_repo=gs_repo,
                model_root=model_root,
                gs_resize=bool(args.gs_resize),
                gs_max_iter=str(args.gs_max_iter),
                gs_mode=str(args.gs_mode),
            )
        except Exception as e:
            print(f"[3dgs] FAILED to launch scene={summary.scene_id}: {e}")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                if state == "RECORD" and record_start_ts is not None:
                    on_scene_stopped(stop_scene(ts_sec_for_frame(frame_idx + 1), reason="eof"))
                break

            frame_idx += 1
            ts_sec = ts_sec_for_frame(frame_idx)

            # Sample frames for buffering/recording at fps_save.
            if _should_run(ts_sec, last_sample_ts, capture.fps_save):
                last_sample_ts = ts_sec
                frame_s = _resize_max_side(frame, capture.save_max_side)
                if state == "IDLE":
                    buffer.append((ts_sec, frame_s.copy()))
                elif state == "RECORD":
                    assert current_input_dir is not None
                    if _should_run(ts_sec, last_save_ts, capture.fps_save):
                        out_name = f"{save_index:06d}.jpg"
                        out_path = current_input_dir / out_name
                        _write_frame_jpg(out_path, frame_s, capture.jpeg_quality)
                        frame_times.append((out_name, ts_sec))
                        save_index += 1
                        recorded_after_trigger += 1
                        last_save_ts = ts_sec

            # Run YOLO gating at detect_fps.
            ran_detection = False
            detected = False
            best_conf = 0.0
            det_count = 0
            annotated = None
            if _should_run(ts_sec, last_detect_ts, gate.detect_fps):
                ran_detection = True
                last_detect_ts = ts_sec
                results = model.predict(
                    frame,
                    conf=gate.conf_th,
                    classes=gate_class_ids,
                    imgsz=gate.imgsz,
                    device=gate.device if gate.device else None,
                    verbose=False,
                )
                r0 = results[0]
                boxes = getattr(r0, "boxes", None)
                if boxes is not None and len(boxes) > 0:
                    detected = True
                    det_count = int(len(boxes))
                    try:
                        best_conf = float(boxes.conf.max().item())
                    except Exception:
                        best_conf = 0.0
                if capture.display:
                    try:
                        annotated = r0.plot()
                    except Exception:
                        annotated = frame

                if detected:
                    detect_on += 1
                    detect_off = 0
                else:
                    detect_off += 1
                    detect_on = 0

                if state == "IDLE" and detect_on >= gate.k_on:
                    start_scene(ts_sec)
                elif state == "RECORD" and record_start_ts is not None:
                    duration = ts_sec - record_start_ts
                    if duration >= capture.max_seconds:
                        on_scene_stopped(stop_scene(ts_sec, reason="max_seconds"))
                        if not capture.multi_scene:
                            break
                    elif detect_off >= gate.k_off and duration >= capture.min_record_seconds:
                        on_scene_stopped(stop_scene(ts_sec, reason="lost_target"))
                        if not capture.multi_scene:
                            break

            if capture.display:
                view = annotated if annotated is not None else frame
                text = f"{state} t={ts_sec:6.1f}s on={detect_on} off={detect_off}"
                if ran_detection:
                    text += f" det={det_count} conf={best_conf:.2f}"
                cv2.putText(view, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.imshow("gated_capture_yolo", view)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    if state == "RECORD" and record_start_ts is not None:
                        on_scene_stopped(stop_scene(ts_sec, reason="user_quit"))
                    break

    finally:
        cap.release()
        if capture.display:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

    if summaries:
        last = summaries[-1]
        print(f"[done] last_scene_dir={last.scene_dir}")
    else:
        print("[done] no scene recorded (gate never triggered)")


if __name__ == "__main__":
    main()
