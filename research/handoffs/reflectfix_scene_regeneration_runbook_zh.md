# `reflectfix` 场景与 `SAM2` 诊断重建 runbook

本文档用于替代旧的临时脚本入口，回答两个问题：

1. 如果以后要从正式 scene 重新补建 `reflectfix` 资产，应该走什么入口？
2. 如果以后还要重建某个 paired `SAM2` 诊断图，应该如何从正式 scene 直接生成？

当前固定稳定入口：

- formal manifest：
  - `research/plans/tri_camera_node_3d_aware_reid/benchmarks/iciscae_node01_uav_v3_clean_reflectfix.json`
- diagnostics 根目录：
  - `mvp-demo/output/diagnostics/clean_vs_reflectfix/`

## 1. 重建单个 `reflectfix` scene 的标准链路

以下命令以 `mj_node01_j10_clean_reflectfix_circle_xz_b` 为例。

其余 `reflectfix` scene 不允许只替换 `scene_id`。统一规则是：先查 formal manifest，再同步替换 `mjcf`、`scene_id`、`identity_id`、`traj` 和对应轨迹参数，`sam2_camera_boxes` 也以 manifest 为准。

| scene_id | mjcf | traj | 额外参数 |
| --- | --- | --- | --- |
| `mj_node01_j10_clean_reflectfix_line_nodes_a` | `assets/scene/mujoco_3cam_node_parallel_j10.xml` | `line_nodes` | `traj_from_body=node01`, `traj_to_body=node02`, `mid_y=6`, `mid_z=2`, `traj_period=8` |
| `mj_node01_j10_clean_reflectfix_circle_xz_b` | `assets/scene/mujoco_3cam_node_parallel_j10.xml` | `circle_xz` | `traj_center="0 6 2"`, `traj_radius=1.0`, `traj_period=6` |
| `mj_node01_uav1_clean_reflectfix_line_nodes_a` | `assets/scene/mujoco_uav1_3cam_node_parallel_v2.xml` | `line_nodes` | `traj_from_body=node01`, `traj_to_body=node02`, `mid_y=6`, `mid_z=2`, `traj_period=8` |
| `mj_node01_uav1_clean_reflectfix_circle_xz_b` | `assets/scene/mujoco_uav1_3cam_node_parallel_v2.xml` | `circle_xz` | `traj_center="0 6 2"`, `traj_radius=1.0`, `traj_period=6` |
| `mj_node01_su34_clean_reflectfix_line_nodes_a` | `assets/scene/mujoco_su34_3cam_node_parallel.xml` | `line_nodes` | `traj_from_body=node01`, `traj_to_body=node02`, `mid_y=6`, `mid_z=2`, `traj_period=8` |
| `mj_node01_su34_clean_reflectfix_circle_xz_b` | `assets/scene/mujoco_su34_3cam_node_parallel.xml` | `circle_xz` | `traj_center="0 6 2"`, `traj_radius=1.0`, `traj_period=6` |

```powershell
cd d:\研究生\grad_project\mvp-demo
$env:HF_ENDPOINT = "https://hf-mirror.com"

python scripts/mj_capture_3cam_node.py `
  --mjcf assets/scene/mujoco_3cam_node_parallel_j10.xml `
  --out_root data/nodes `
  --node_id node01 `
  --scene_id mj_node01_j10_clean_reflectfix_circle_xz_b `
  --identity_id j10 `
  --target_body target `
  --traj circle_xz `
  --traj_center "0 6 2" `
  --traj_radius 1.0 `
  --traj_period 6.0 `
  --fps 30 `
  --seconds 3 `
  --save_depth `
  --save_masks_gt

python scripts/run_node_depth_anything_v2.py `
  --scene_dir data/nodes/node01/scenes/mj_node01_j10_clean_reflectfix_circle_xz_b `
  --overwrite

python scripts/run_node_sam2_masks.py `
  --scene_dir data/nodes/node01/scenes/mj_node01_j10_clean_reflectfix_circle_xz_b `
  --checkpoint third_party/sam2/checkpoints/sam2.1_hiera_tiny.pt `
  --model_cfg third_party/sam2/sam2/configs/sam2.1/sam2.1_hiera_t.yaml `
  --camera_box cam0=490,300,660,720 `
  --camera_box cam1=600,330,800,720 `
  --camera_box cam2=600,270,800,640 `
  --overwrite

python scripts/flatten_node_sam2_masks.py `
  --scene_dir data/nodes/node01/scenes/mj_node01_j10_clean_reflectfix_circle_xz_b `
  --overwrite
```

说明：

- `camera_box` 以 manifest 中记录的 `sam2_camera_boxes` 为准。
- 如果只是补建若干 scene，不需要恢复任何已删除的 `tmp` 原料目录。
- 如果要复现实验总表，统一使用 formal manifest 重新跑 branch eval，而不是恢复旧的临时脚本。

## 2. 重建一个代表性 paired `SAM2` 诊断图

下面的命令会从正式 scene 直接导出：

- `sam2_mask_comparison_mid.png`
- `sam2_mask_comparison_mid.json`

它只依赖正式 scene 内已有的：

- `cams/cam*/frames/`
- `cams/cam*/masks/`
- `cams/cam*/masks_gt/`

### 2.1 clean 侧示例

```powershell
cd d:\研究生\grad_project
$env:SCENE_DIR = "mvp-demo/data/nodes/node01/scenes/mj_node01_j10_clean_circle_xz_b"
$env:OUT_DIR = "mvp-demo/output/diagnostics/clean_vs_reflectfix/rebuilt_cases/diag_clean_j10_circle_xz_b"
$env:STEM = "000001500000"
@'
from pathlib import Path
import os
import json
import numpy as np
import cv2

scene_dir = Path(os.environ["SCENE_DIR"])
out_dir = Path(os.environ["OUT_DIR"])
stem = os.environ["STEM"]
cams = ["cam0", "cam1", "cam2"]
out_dir.mkdir(parents=True, exist_ok=True)

def read_color(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)

def read_gray(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)

def iou(pred, gt):
    pred = pred > 0
    gt = gt > 0
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return 0.0 if union == 0 else float(inter / union)

def label(img, text):
    out = img.copy()
    cv2.putText(out, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(out, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    return out

tiles = []
ious = {}
for cam in cams:
    rgb_path = scene_dir / "cams" / cam / "frames" / f"{stem}.jpg"
    if not rgb_path.exists():
        rgb_path = scene_dir / "cams" / cam / "frames" / f"{stem}.png"
    pred = read_gray(scene_dir / "cams" / cam / "masks" / f"{stem}.png")
    gt = read_gray(scene_dir / "cams" / cam / "masks_gt" / f"{stem}.png")
    rgb = read_color(rgb_path)
    pred_panel = rgb.copy()
    gt_panel = rgb.copy()
    pred_panel[pred > 0] = pred_panel[pred > 0] * 0.55 + np.array([0, 96, 255]) * 0.45
    gt_panel[gt > 0] = gt_panel[gt > 0] * 0.55 + np.array([0, 255, 180]) * 0.45
    panel = np.hstack([
        label(rgb, f"{cam} RGB"),
        label(pred_panel.astype(np.uint8), f"{cam} SAM2"),
        label(gt_panel.astype(np.uint8), f"{cam} GT"),
    ])
    tiles.append(panel)
    ious[cam] = iou(pred, gt)

canvas = np.vstack(tiles)
png_path = out_dir / "sam2_mask_comparison_mid.png"
ok = cv2.imwrite(str(png_path), canvas)
if not ok:
    cv2.imencode(".png", canvas)[1].tofile(str(png_path))

(out_dir / "sam2_mask_comparison_mid.json").write_text(
    json.dumps({"scene_dir": str(scene_dir), "stem": stem, "ious": ious, "file": str(png_path)}, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
print(json.dumps({"scene_dir": str(scene_dir), "stem": stem, "ious": ious, "file": str(png_path)}, ensure_ascii=False, indent=2))
'@ | python -
```

如果要换其他 clean case，只改 `SCENE_DIR / OUT_DIR / STEM` 三个环境变量即可。

### 2.2 reflectfix 侧示例

```powershell
cd d:\研究生\grad_project
$env:SCENE_DIR = "mvp-demo/data/nodes/node01/scenes/mj_node01_j10_clean_reflectfix_circle_xz_b"
$env:OUT_DIR = "mvp-demo/output/diagnostics/clean_vs_reflectfix/rebuilt_cases/diag_reflectfix_j10_circle_xz_b"
$env:STEM = "000001500000"
@'
from pathlib import Path
import os
import json
import numpy as np
import cv2

scene_dir = Path(os.environ["SCENE_DIR"])
out_dir = Path(os.environ["OUT_DIR"])
stem = os.environ["STEM"]
cams = ["cam0", "cam1", "cam2"]
out_dir.mkdir(parents=True, exist_ok=True)

def read_color(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)

def read_gray(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)

def iou(pred, gt):
    pred = pred > 0
    gt = gt > 0
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return 0.0 if union == 0 else float(inter / union)

def label(img, text):
    out = img.copy()
    cv2.putText(out, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(out, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    return out

tiles = []
ious = {}
for cam in cams:
    rgb_path = scene_dir / "cams" / cam / "frames" / f"{stem}.jpg"
    if not rgb_path.exists():
        rgb_path = scene_dir / "cams" / cam / "frames" / f"{stem}.png"
    pred = read_gray(scene_dir / "cams" / cam / "masks" / f"{stem}.png")
    gt = read_gray(scene_dir / "cams" / cam / "masks_gt" / f"{stem}.png")
    rgb = read_color(rgb_path)
    pred_panel = rgb.copy()
    gt_panel = rgb.copy()
    pred_panel[pred > 0] = pred_panel[pred > 0] * 0.55 + np.array([0, 96, 255]) * 0.45
    gt_panel[gt > 0] = gt_panel[gt > 0] * 0.55 + np.array([0, 255, 180]) * 0.45
    panel = np.hstack([
        label(rgb, f"{cam} RGB"),
        label(pred_panel.astype(np.uint8), f"{cam} SAM2"),
        label(gt_panel.astype(np.uint8), f"{cam} GT"),
    ])
    tiles.append(panel)
    ious[cam] = iou(pred, gt)

canvas = np.vstack(tiles)
png_path = out_dir / "sam2_mask_comparison_mid.png"
ok = cv2.imwrite(str(png_path), canvas)
if not ok:
    cv2.imencode(".png", canvas)[1].tofile(str(png_path))

(out_dir / "sam2_mask_comparison_mid.json").write_text(
    json.dumps({"scene_dir": str(scene_dir), "stem": stem, "ious": ious, "file": str(png_path)}, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
print(json.dumps({"scene_dir": str(scene_dir), "stem": stem, "ious": ious, "file": str(png_path)}, ensure_ascii=False, indent=2))
'@ | python -
```

如果要换其他 reflectfix case，也只改 `SCENE_DIR / OUT_DIR / STEM`。不需要恢复任何已删除的 `tmp` 原料目录。

## 3. 若要重跑 reflectfix benchmark，总是使用 formal manifest

```powershell
python mvp-demo/scripts/run_iciscae_branch_eval.py --manifest research/plans/tri_camera_node_3d_aware_reid/benchmarks/iciscae_node01_uav_v3_clean_reflectfix.json --branch rgb_only
python mvp-demo/scripts/run_iciscae_branch_eval.py --manifest research/plans/tri_camera_node_3d_aware_reid/benchmarks/iciscae_node01_uav_v3_clean_reflectfix.json --branch rgb_predicted_depth_geometry
python mvp-demo/scripts/run_iciscae_branch_eval.py --manifest research/plans/tri_camera_node_3d_aware_reid/benchmarks/iciscae_node01_uav_v3_clean_reflectfix.json --branch rgb_fused_geometry
python mvp-demo/scripts/run_iciscae_branch_eval.py --manifest research/plans/tri_camera_node_3d_aware_reid/benchmarks/iciscae_node01_uav_v3_clean_reflectfix.json --branch gt_upper_bound
python mvp-demo/scripts/summarize_iciscae_branch_comparison.py --benchmark_id iciscae_node01_uav_v3_clean_reflectfix
python mvp-demo/scripts/analyze_iciscae_failure_modes.py --benchmark_id iciscae_node01_uav_v3_clean_reflectfix
```

## 4. 当前边界

- 旧的 `tmp/reflection_fix_ab/run_reflectfix_capture.py` 与 `tmp/reflection_fix_ab/run_reflectfix_depth_sam2.py` 已退出正式入口。
- `mvp-demo/output/evals/iciscae_node01_uav_v3_clean_reflectfix/` 继续作为现有全量 reflectfix 结果根目录。
- `mvp-demo/output/diagnostics/clean_vs_reflectfix/` 继续作为 clean-vs-reflectfix 诊断资产根目录。
