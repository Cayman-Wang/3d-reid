from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"Failed to read json: {path}\nError: {exc!r}")


def _windows_runtime_root() -> str:
    return "D:/node01_spin_runtime_ascii"


def _linux_runtime_root(repo_root: Path) -> Path:
    value = os.environ.get("REID_NODE01_RUNTIME_ROOT", "").strip()
    if value:
        return Path(value).expanduser().resolve()
    return repo_root


def _resolve_path_text(repo_root: Path, path_text: str) -> Path:
    text = str(path_text).replace("\\", "/")
    win_root = _windows_runtime_root()
    if text.lower().startswith(win_root.lower() + "/"):
        suffix = text[len(win_root) + 1 :]
        return (_linux_runtime_root(repo_root) / suffix).resolve()
    path = Path(text)
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def _run(cmd: list[str], cwd: Path) -> None:
    printable = " ".join(f'"{x}"' if " " in x else x for x in cmd)
    print(f"[run] {printable}")
    env = dict(os.environ)
    env.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    subprocess.run(cmd, cwd=str(cwd), check=True, env=env)


def _write_failures(path: Path, manifest_path: Path, failures: list[dict[str, Any]]) -> None:
    payload = {
        "manifest": str(manifest_path),
        "num_failures": len(failures),
        "failures": failures,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _resolve_posix(path: Path) -> str:
    return path.resolve().as_posix()


def _normalize_filter_config(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "opacity_thresh": float(payload["opacity_thresh"]),
        "confidence_thresh": float(payload["confidence_thresh"]),
        "voxel_size_m": float(payload["voxel_size_m"]),
        "max_points": int(payload["max_points"]),
        "min_points": int(payload["min_points"]),
    }


def _export_filter_config(args: argparse.Namespace) -> dict[str, Any]:
    return _normalize_filter_config(
        {
            "opacity_thresh": args.opacity_thresh,
            "confidence_thresh": args.confidence_thresh,
            "voxel_size_m": args.voxel_size_m,
            "max_points": args.max_points,
            "min_points": args.min_points,
        }
    )


def _points_cache_matches(
    points_dir: Path,
    source_bundle: Path,
    points_subdir: str,
    expected_filters: dict[str, Any],
) -> bool:
    meta_path = points_dir / "meta.json"
    index_path = points_dir / "points_index.csv"
    if not meta_path.exists() or not index_path.exists():
        return False

    npy_files = sorted(points_dir.glob("*.npy"))
    if not npy_files:
        return False

    try:
        meta = _load_json(meta_path)
        meta_filters = _normalize_filter_config(dict(meta.get("filters") or {}))
    except Exception:
        return False

    if str(meta.get("schema_version")) != "points_neoverse_multiview_v2":
        return False
    if str(meta.get("out_subdir")) != str(points_subdir):
        return False
    if str(meta.get("source_bundle")) != _resolve_posix(source_bundle):
        return False
    if meta_filters != expected_filters:
        return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description="Precompute NeoVerse multiview geometry for static benchmark manifest.")
    ap.add_argument(
        "--manifest",
        default="research/plans/tri_camera_node_3d_aware_reid/benchmarks/node01_neoverse_multiview_static_v1.json",
        type=str,
    )
    ap.add_argument("--neoverse_python", required=True, type=str)
    ap.add_argument("--neoverse_repo", default="third_party/NeoVerse", type=str)
    ap.add_argument("--num_sync_steps", default=27, type=int)
    ap.add_argument("--skip_if_points_exist", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--skip_missing_scene", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--continue_on_error", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--model_path", default="models", type=str)
    ap.add_argument("--reconstructor_path", default="models/NeoVerse/reconstructor.ckpt", type=str)
    ap.add_argument("--device", default="cuda", type=str)
    ap.add_argument("--torch_dtype", default="bfloat16", type=str)
    ap.add_argument("--height", default=336, type=int)
    ap.add_argument("--width", default=560, type=int)
    ap.add_argument("--resize_mode", choices=["center_crop", "resize"], default="center_crop", type=str)
    ap.add_argument("--opacity_thresh", default=0.05, type=float)
    ap.add_argument("--confidence_thresh", default=0.0, type=float)
    ap.add_argument("--voxel_size_m", default=0.02, type=float)
    ap.add_argument("--max_points", default=50000, type=int)
    ap.add_argument("--min_points", default=32, type=int)
    args = ap.parse_args()

    repo_root = _repo_root()
    scripts_dir = repo_root / "mvp-demo" / "scripts"

    manifest_path = _resolve_path_text(repo_root, str(args.manifest))
    if not manifest_path.exists():
        raise SystemExit(f"Missing manifest: {manifest_path}")

    manifest = _load_json(manifest_path)
    entries = list(manifest.get("entries") or [])
    if not entries:
        raise SystemExit(f"No entries in manifest: {manifest_path}")

    branch_cfg = dict((manifest.get("branch_configs") or {}).get("rgb_neoverse_multiview_geometry") or {})
    points_subdir = str(branch_cfg.get("points_subdir", "recon/points_neoverse_multiview"))
    expected_filters = _export_filter_config(args)
    failures: list[dict[str, Any]] = []
    failure_path = repo_root / "mvp-demo" / "output" / "neoverse_multiview" / "precompute_failures.json"
    failure_path.parent.mkdir(parents=True, exist_ok=True)

    neoverse_python = Path(str(args.neoverse_python)).resolve()
    if not neoverse_python.exists():
        raise SystemExit(f"Missing --neoverse_python: {neoverse_python}")

    for entry in entries:
        scene_dir = _resolve_path_text(repo_root, str(entry["scene_dir"]))
        scene_id = str(entry["scene_id"])
        bundle_path = (
            repo_root
            / "mvp-demo"
            / "output"
            / "neoverse_multiview"
            / scene_id
            / "run_full_frame_joint"
            / "reconstruction_bundle.pt"
        )
        if not scene_dir.exists():
            if bool(args.skip_missing_scene):
                failures.append({"scene_id": scene_id, "scene_dir": str(scene_dir), "status": "missing_scene"})
                print(f"[skip] missing scene: {scene_id} -> {scene_dir}")
                continue
            raise SystemExit(f"Missing scene_dir for {scene_id}: {scene_dir}")

        points_dir = scene_dir / points_subdir
        if bool(args.skip_if_points_exist) and points_dir.exists():
            if _points_cache_matches(
                points_dir=points_dir,
                source_bundle=bundle_path,
                points_subdir=points_subdir,
                expected_filters=expected_filters,
            ):
                print(f"[skip] scene={scene_id} points cache already valid: {points_dir}")
                continue
            print(f"[stale] scene={scene_id} points cache mismatch, recomputing: {points_dir}")

        manifest_out = repo_root / "mvp-demo" / "output" / "neoverse_multiview" / scene_id / "input" / "manifest.json"

        try:
            _run(
                [
                    sys.executable,
                    str(scripts_dir / "prepare_neoverse_multiview_manifest.py"),
                    "--scene_dir",
                    str(scene_dir),
                    "--num_sync_steps",
                    str(args.num_sync_steps),
                ],
                cwd=repo_root,
            )

            _run(
                [
                    str(neoverse_python),
                    str(scripts_dir / "run_neoverse_multiview_joint.py"),
                    "--manifest",
                    str(manifest_out),
                    "--neoverse_repo",
                    str(args.neoverse_repo),
                    "--model_path",
                    str(args.model_path),
                    "--reconstructor_path",
                    str(args.reconstructor_path),
                    "--device",
                    str(args.device),
                    "--torch_dtype",
                    str(args.torch_dtype),
                    "--height",
                    str(args.height),
                    "--width",
                    str(args.width),
                    "--resize_mode",
                    str(args.resize_mode),
                ],
                cwd=repo_root,
            )

            _run(
                [
                    str(neoverse_python),
                    str(scripts_dir / "export_neoverse_multiview_points.py"),
                    "--bundle",
                    str(bundle_path),
                    "--scene_dir",
                    str(scene_dir),
                    "--neoverse_repo",
                    str(args.neoverse_repo),
                    "--out_subdir",
                    points_subdir,
                    "--opacity_thresh",
                    str(args.opacity_thresh),
                    "--confidence_thresh",
                    str(args.confidence_thresh),
                    "--voxel_size_m",
                    str(args.voxel_size_m),
                    "--max_points",
                    str(args.max_points),
                    "--min_points",
                    str(args.min_points),
                ],
                cwd=repo_root,
            )
        except Exception as exc:
            failure_item = {
                "scene_id": scene_id,
                "scene_dir": str(scene_dir),
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            if isinstance(exc, subprocess.CalledProcessError):
                failure_item["returncode"] = int(exc.returncode)
            failures.append(failure_item)
            if not bool(args.continue_on_error):
                _write_failures(failure_path, manifest_path, failures)
                raise
            print(f"[warn] scene failed but continuing: {scene_id} error_type={type(exc).__name__}")

    _write_failures(failure_path, manifest_path, failures)


if __name__ == "__main__":
    main()
