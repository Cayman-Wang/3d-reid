from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str], cwd: Path) -> None:
    printable = " ".join([str(c) for c in cmd])
    print(f"[run] {printable}")
    subprocess.run(cmd, cwd=str(cwd), check=True)


def main() -> None:
    default_gs_repo = Path(__file__).resolve().parents[1] / "third_party" / "gaussian-splatting"

    ap = argparse.ArgumentParser(description="Run COLMAP→3DGS→depth for a captured scene (expects external 3DGS repo)")
    ap.add_argument(
        "--gs_repo",
        default=str(default_gs_repo),
        type=str,
        help="Path to graphdeco/gaussian-splatting repo (default: mvp-demo/third_party/gaussian-splatting)",
    )
    ap.add_argument("--scene_dir", required=True, type=str, help="Captured scene dir (contains input/)")
    ap.add_argument("--model_dir", required=True, type=str, help="Output model dir (train.py -m)")
    ap.add_argument("--resize", action="store_true", help="Pass --resize to convert.py (optional)")

    ap.add_argument("--skip_convert", action="store_true")
    ap.add_argument("--skip_train", action="store_true")
    ap.add_argument("--skip_depth", action="store_true")
    ap.add_argument("--max_iter", default="", type=str, help="Optional: forward extra args to train.py (e.g. '--iterations 7000')")
    ap.add_argument(
        "--depth_out_dir",
        default="",
        type=str,
        help="Optional: write merged depth_npy aligned with SCENE_DIR/images (default: <scene_dir>/depth_npy).",
    )
    args = ap.parse_args()

    gs_repo = Path(args.gs_repo).resolve()
    scene_dir = Path(args.scene_dir).resolve()
    model_dir = Path(args.model_dir).resolve()

    if not gs_repo.exists():
        raise SystemExit(f"--gs_repo not found: {gs_repo}")
    if not scene_dir.exists():
        raise SystemExit(f"--scene_dir not found: {scene_dir}")

    py = sys.executable

    if not args.skip_convert:
        cmd = [py, "convert.py", "-s", str(scene_dir)]
        if args.resize:
            cmd.append("--resize")
        _run(cmd, cwd=gs_repo)

    if not args.skip_train:
        cmd = [py, "train.py", "-s", str(scene_dir), "-m", str(model_dir), "--eval"]
        if args.max_iter:
            cmd.extend(args.max_iter.split())
        _run(cmd, cwd=gs_repo)

    if not args.skip_depth:
        depth_script = Path(__file__).with_name("gs_render_depth_npy.py").resolve()
        depth_out_dir = Path(args.depth_out_dir).resolve() if args.depth_out_dir else (scene_dir / "depth_npy")
        cmd = [py, str(depth_script), "--gs_repo", str(gs_repo), "-m", str(model_dir), "--out_dir", str(depth_out_dir)]
        _run(cmd, cwd=gs_repo)
        # Quick alignment check: SCENE_DIR/images/*.jpg should have a matching depth_npy/<stem>.npy.
        images_dir = scene_dir / "images"
        if images_dir.exists():
            img_stems = {p.stem for p in images_dir.glob("*.jpg")}
            depth_stems = {p.stem for p in depth_out_dir.glob("*.npy")}
            missing = sorted(img_stems - depth_stems)
            if missing:
                preview = ", ".join(missing[:20])
                print(f"[warn] depth_npy missing {len(missing)} / {len(img_stems)} (sample: {preview})")
            else:
                print(f"[ok] depth_npy aligned: {len(depth_stems)} depths for {len(img_stems)} images at {depth_out_dir}")


if __name__ == "__main__":
    main()
