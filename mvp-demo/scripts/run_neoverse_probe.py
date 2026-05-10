from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _normalize_python(path_str: str) -> Path:
    p = Path(path_str).expanduser().resolve()
    if not p.exists():
        raise SystemExit(f"--neoverse_python does not exist: {p}")
    return p


def _pick_video(input_dir: Path, variant: str) -> Path:
    if variant == "object_crop":
        path = input_dir / "object_crop.mp4"
    elif variant == "full_frame":
        path = input_dir / "full_frame.mp4"
    else:
        raise SystemExit(f"Unsupported --video_variant: {variant}")
    if not path.exists():
        raise SystemExit(
            f"Input video not found: {path}\n"
            "Run scripts/export_neoverse_probe_input.py first to prepare input videos."
        )
    return path


def _build_cmd(
    neoverse_python: Path,
    inference_path: Path,
    input_video: Path,
    output_video: Path,
    trajectory: str,
    angle: float,
    orbit_radius: float,
    vis_rendering: bool,
    low_vram: bool,
) -> list[str]:
    cmd = [
        str(neoverse_python),
        str(inference_path),
        "--input_path",
        str(input_video),
        "--output_path",
        str(output_video),
        "--trajectory",
        str(trajectory),
        "--angle",
        str(angle),
        "--orbit_radius",
        str(orbit_radius),
    ]
    if vis_rendering:
        cmd.append("--vis_rendering")
    if low_vram:
        cmd.append("--low_vram")
    return cmd


def _run(cmd: list[str], cwd: Path) -> tuple[int, float]:
    env = dict(os.environ)
    env.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    pretty = " ".join(f'"{x}"' if " " in x else x for x in cmd)
    print(f"[run] {pretty}")
    t0 = time.perf_counter()
    completed = subprocess.run(cmd, cwd=str(cwd), env=env)
    dt = time.perf_counter() - t0
    return int(completed.returncode), float(dt)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run NeoVerse inference as a reproducible probe backend.")
    ap.add_argument("--neoverse_python", required=True, type=str)
    ap.add_argument("--neoverse_repo", default="third_party/NeoVerse", type=str)
    ap.add_argument("--scene_dir", required=True, type=str)
    ap.add_argument("--cam_id", default="cam0", type=str)
    ap.add_argument("--video_variant", default="object_crop", choices=["object_crop", "full_frame"])
    ap.add_argument("--trajectory", default="orbit_left", type=str)
    ap.add_argument("--angle", default=12.0, type=float)
    ap.add_argument("--orbit_radius", default=0.08, type=float)
    ap.add_argument("--vis_rendering", action="store_true")
    ap.add_argument("--low_vram", action="store_true")
    args = ap.parse_args()

    repo_root = _repo_root()
    scene_dir = Path(args.scene_dir).resolve()
    if not scene_dir.exists():
        raise SystemExit(f"Missing --scene_dir: {scene_dir}")

    scene_id = scene_dir.name
    cam_id = str(args.cam_id).strip() or "cam0"

    neoverse_python = _normalize_python(str(args.neoverse_python))
    neoverse_repo = Path(str(args.neoverse_repo))
    if not neoverse_repo.is_absolute():
        neoverse_repo = repo_root / neoverse_repo
    neoverse_repo = neoverse_repo.resolve()
    if not neoverse_repo.exists():
        raise SystemExit(f"--neoverse_repo not found: {neoverse_repo}")

    inference_path = neoverse_repo / "inference.py"
    if not inference_path.exists():
        raise SystemExit(f"NeoVerse inference script not found: {inference_path}")

    probe_root = repo_root / "mvp-demo" / "output" / "neoverse_probe" / scene_id / cam_id
    input_dir = probe_root / "input"
    run_dir = probe_root / f"run_{args.video_variant}"
    run_dir.mkdir(parents=True, exist_ok=True)

    input_variant_video = _pick_video(input_dir=input_dir, variant=str(args.video_variant))
    input_video_for_run = run_dir / "input_video.mp4"
    shutil.copy2(str(input_variant_video), str(input_video_for_run))

    output_video = run_dir / "output.mp4"
    cmd = _build_cmd(
        neoverse_python=neoverse_python,
        inference_path=inference_path,
        input_video=input_video_for_run,
        output_video=output_video,
        trajectory=str(args.trajectory),
        angle=float(args.angle),
        orbit_radius=float(args.orbit_radius),
        vis_rendering=bool(args.vis_rendering),
        low_vram=bool(args.low_vram),
    )

    started_at = datetime.now(timezone.utc).isoformat()
    return_code, elapsed_sec = _run(cmd, cwd=neoverse_repo)
    finished_at = datetime.now(timezone.utc).isoformat()

    vis_dir = run_dir / "vis_rendering"
    raw_vis_dir = run_dir / "output"
    if bool(args.vis_rendering) and raw_vis_dir.exists():
        if vis_dir.exists():
            shutil.rmtree(vis_dir)
        shutil.move(str(raw_vis_dir), str(vis_dir))

    probe_meta = {
        "scene_id": scene_id,
        "cam_id": cam_id,
        "video_variant": str(args.video_variant),
        "trajectory": str(args.trajectory),
        "angle": float(args.angle),
        "orbit_radius": float(args.orbit_radius),
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
        "runtime_sec": elapsed_sec,
        "low_vram": bool(args.low_vram),
        "vis_rendering": bool(args.vis_rendering),
        "return_code": return_code,
        "neoverse_repo": str(neoverse_repo),
        "neoverse_python": str(neoverse_python),
        "command": cmd,
        "outputs": {
            "input_video_mp4": str(input_video_for_run),
            "output_mp4": str(output_video),
            "vis_rendering_dir": str(vis_dir) if vis_dir.exists() else None,
        },
    }
    (run_dir / "probe_meta.json").write_text(
        json.dumps(probe_meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    if return_code != 0:
        raise SystemExit(return_code)

    print(f"Wrote probe outputs to: {run_dir}")
    print(
        "[summary] "
        f"scene={scene_id} cam={cam_id} variant={args.video_variant} "
        f"trajectory={args.trajectory} runtime_sec={elapsed_sec:.2f}"
    )


if __name__ == "__main__":
    main()
