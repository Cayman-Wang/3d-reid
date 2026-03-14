# Handoff / Progress Snapshot

## Goal

实现 MVP：**YOLO 目标检测只做门控触发** → 触发后保存一段**全帧**到 `SCENE_DIR/input/` → 后台/离线跑 **COLMAP → 3DGS → 渲染每帧 depth_npy**（与 `images/` 对齐）。前期用手机/相机视频文件验证，后期把 `--source` 换成摄像头实时输入。

## Key Decisions

- 3DGS 源码固定放在：`mvp-demo/third_party/gaussian-splatting/`
- 门控只决定“何时开始采集/何时停止”，**不裁剪**再重建（避免破坏 COLMAP/相机内参一致性）。
- 采集输出固定为：`mvp-demo/data/scenes/<scene_id>/input/%06d.jpg`（并写 `capture_meta.json`、`frame_times.csv`）。
- 下游对齐以 `SCENE_DIR/images/` 为准；深度固定输出到 `SCENE_DIR/depth_npy/`，文件名与 `images/*.jpg` 同 stem 对齐（并写 `depth_meta.json`、`depth_index.csv`）。

## What’s Implemented

- 门控采集脚本：`mvp-demo/scripts/gated_capture_yolo.py`
- 串联 3DGS（可选）：`mvp-demo/scripts/run_3dgs_scene.py`（默认 `--gs_repo` 指向 `mvp-demo/third_party/gaussian-splatting`）
- 3DGS depth 导出脚本：`mvp-demo/scripts/gs_render_depth_npy.py`
- ReID 下游脚本（MVP 基线）：
  - `mvp-demo/scripts/run_sam2_video_masks.py`（SAM2：`images/` → `masks/obj_*/<stem>.png`）
  - `mvp-demo/scripts/export_colmap_intrinsics_json.py`（导出 `intrinsics.json`）
  - `mvp-demo/scripts/make_tracklets_from_masks.py`（mask → tracklets.json）
  - `mvp-demo/scripts/make_tracklets_yolo.py`（YOLO+ByteTrack → tracklets.json，bbox-only）
  - `mvp-demo/scripts/extract_track_embeddings.py`（track embedding：RGB+Depth/几何）
  - `mvp-demo/scripts/search_track_embeddings.py`（numpy top‑K 检索 demo）
- 依赖与忽略：`mvp-demo/requirements.txt`、`mvp-demo/.gitignore`
- 文档：`mvp-demo/README.md`、`3D重建-3Dreid/3dgs_gated_mvp_pipeline.md`

## Quick Start (video file)

```bash
# From repo root:
cd mvp-demo

# 1) 门控采集环境（YOLO + OpenCV）
conda create -n mvp_capture python=3.10 -y
conda activate mvp_capture
python -m pip install -U pip
# 可选：给 YOLO 装 GPU 版 PyTorch（更快；没有也能跑，只是会走 CPU）
conda install -c pytorch -c nvidia pytorch torchvision torchaudio pytorch-cuda=11.8 -y
pip install -r requirements.txt

# 2) 拉取 3DGS 到固定路径
mkdir -p third_party
git clone --recursive https://github.com/graphdeco-inria/gaussian-splatting.git third_party/gaussian-splatting

# 3) 3DGS 环境（按官方 environment.yml）
cd third_party/gaussian-splatting
conda env create -f environment.yml
conda activate gaussian_splatting
cd ../..

# 4) COLMAP（convert.py 会调用 colmap；建议装到 gaussian_splatting env）
conda install -n gaussian_splatting -c conda-forge colmap -y
colmap -h
```

采集（YOLO 门控触发）：

```bash
conda activate mvp_capture
python scripts/gated_capture_yolo.py \
  --source ./demo.mp4 \
  --scene_root data/scenes \
  --gate_classes person
```

自动触发 3DGS（可选）：

```bash
conda activate mvp_capture
python scripts/gated_capture_yolo.py \
  --source ./demo.mp4 \
  --scene_root data/scenes \
  --gate_classes person \
  --auto_3dgs \
  --gs_env gaussian_splatting \
  --gs_resize
```

重建（需你已装好 COLMAP，并已配置好 3DGS conda 环境）：

```bash
conda activate gaussian_splatting
python scripts/run_3dgs_scene.py \
  --scene_dir data/scenes/<scene_id> \
  --model_dir output/<scene_id> \
  --resize
```

期望产物（对齐口径）：

- `data/scenes/<scene_id>/images/*.jpg`（undistorted）
- `data/scenes/<scene_id>/depth_npy/<stem>.npy`（与 `images/` 同 stem）
- `data/scenes/<scene_id>/depth_npy/depth_meta.json`、`data/scenes/<scene_id>/depth_npy/depth_index.csv`

## What’s Next

- 确认 `colmap` 可用（在 PATH 中），并实际跑通一次 `convert.py → train.py → depth_npy`，检查 `images/` 与 `depth_npy/` 数量一致。
- 根据目标类别修改 `--gate_classes` 或换成你自己的 YOLO 权重（`--model`）。
- 继续做 ReID：先准备 `masks/obj_*`（SAM2）或用 `make_tracklets_yolo.py` 生成 bbox tracklets，然后：
  1) `export_colmap_intrinsics_json.py` 导出 `intrinsics.json`
  2) `extract_track_embeddings.py` 生成 `embeddings/`
  3) `search_track_embeddings.py` 跨 scene 做 top‑K 检索
