# MVP Demo（YOLO 门控 + 3DGS 重建）

本目录用于把“**检测到目标才启动重建**”这条 MVP 跑通：YOLO 只负责**门控触发**与采集控制；3DGS/COLMAP 仍对**全帧序列**做重建与深度渲染。

## 运行环境（本机已验证）

- OS：Ubuntu 22.04.5 LTS
- GPU：NVIDIA GeForce RTX 3090 x2（Driver 560.28.03；CUDA Driver 12.6）
- CUDA Toolkit（编译用）：`nvcc 11.8`
- Conda：`conda 25.5.1`（base 是 Python 3.13；不建议直接用 base 跑，建议新建 env）

## 目录结构（约定）

- `scripts/`: 采集/门控/辅助脚本
- `third_party/gaussian-splatting/`: 3DGS 代码（graphdeco-inria 官方实现，推荐用 submodule）
- `data/scenes/<scene_id>/input/`: 门控触发后落盘的原始帧（给 3DGS 的 `convert.py` 用）
- `data/scenes/<scene_id>/images/`: `convert.py` 输出的 undistorted images（后续对齐用它）
- `data/scenes/<scene_id>/depth_npy/`: 3DGS 渲染出的深度（与 `images/` 同 stem 对齐）
- `output/<scene_id>/`: 3DGS 输出目录（`train.py -m`）

## 环境与代码准备（第一次运行前）

说明：门控采集（Ultralytics YOLO）与 3DGS（graphdeco gaussian-splatting）建议用**两个 conda 环境**隔离依赖。

下面所有命令默认你已进入本目录：

```bash
# From repo root:
cd mvp-demo
```

1) 新建门控采集环境（YOLO + OpenCV）

```bash
conda create -n mvp_capture python=3.10 -y
conda activate mvp_capture
python -m pip install -U pip
# 可选：给 YOLO 装 GPU 版 PyTorch（更快；没有也能跑，只是会走 CPU）
conda install -c pytorch -c nvidia pytorch torchvision torchaudio pytorch-cuda=11.8 -y
pip install -r requirements.txt
```

2) 拉取 3DGS 代码到固定路径

```bash
mkdir -p third_party
# 如果仓库使用 submodule：
git submodule update --init --recursive
# 如果没有 submodule，再手动 clone：
# git clone --recursive https://github.com/graphdeco-inria/gaussian-splatting.git third_party/gaussian-splatting
```

3) 准备 3DGS 运行环境（按官方 `environment.yml`）

```bash
cd third_party/gaussian-splatting
conda env create -f environment.yml
conda activate gaussian_splatting
cd ../..
```

4) 安装/配置 `COLMAP`（`convert.py` 会调用 `colmap` 命令，要求在 PATH 中）

推荐装到 `gaussian_splatting` 环境里（免 sudo）：

```bash
conda install -n gaussian_splatting -c conda-forge colmap -y
colmap -h
```

可选：如果你要用 `convert.py --resize`，再装 ImageMagick（否则不要传 `--resize`）：

```bash
conda install -n gaussian_splatting -c conda-forge imagemagick -y
magick -version || convert -version
```

## 需要部署的算法/模块（按阶段）

**A. 实时侧（常驻）**

- 目标检测：YOLO（用于“是否出现目标”的判断）
- 门控状态机：`K_on/K_off` 去抖 + 滞回
- 环形缓冲：`pre_seconds` 秒触发前缓存（触发时一并写入，保证 COLMAP 视角/帧数）
- 采集与抽帧：按 `fps_save` 保存全帧到 `input/`

**B. 重建侧（触发后后台/离线跑）**

- COLMAP（SfM）：相机位姿 + 稀疏重建（通常由 3DGS 仓库的 `convert.py` 调用）
- 3D Gaussian Splatting：训练场景模型（`train.py`）
- Depth 渲染：把每个相机视角的深度导出为 `SCENE_DIR/depth_npy/*.npy`（与 undistorted `images/` 同 stem 对齐）

**C. 下游（后续做 ReID 才需要）**

- 实例分割：SAM2（推荐，用 mask 提升几何/深度鲁棒性）
- 多目标跟踪：ByteTrack / BoT-SORT / OCSort（生成 tracklet）
- 特征与检索：RGB(+Depth/3D) encoder + track 聚合 + FAISS 检索

## MuJoCo 节点级闭环（当前实验主线）

当前仓库已经补齐了一个适合先做仿真验证、再迁移真实环境的三相机节点级管线。建议直接使用 `mvp_demo` 环境：

```bash
conda activate mvp_demo
python -m pip install -r requirements_node_pipeline.txt
```

说明：

- 主链路只消费传感器口径数据：`cams/cam*/frames/`、`cams/cam*/depth/`、`cams/cam*/masks/`、`calib/rig.json`、`frame_times.csv`。
- MuJoCo 导出的 `depth_gt/`、`masks_gt/` 只用于调试和评估，不进入主检索特征。
- 当前默认假设是单目标单轨迹；`build_node_tracklets.py` 已兼容平铺 mask (`masks/<stem>.png`) 和 SAM2 输出 (`masks/obj_XXX/<stem>.png`)。

推荐命令链如下：

1) 采集 MuJoCo 三相机场景（可选导出 GT）

```bash
python scripts/mj_capture_3cam_node.py \
  --mjcf assets/scene/mujoco_humanoid_uav1_3cam_node_parallel.xml \
  --node_id node01 \
  --traj line_nodes \
  --save_depth \
  --save_masks_gt
```

2) 用单目深度模型给每个相机流补 `depth/`

```bash
python scripts/run_node_depth_anything_v2.py \
  --scene_dir data/nodes/node01/scenes/<scene_id>
```

3) 用 SAM2 给每个相机流补 `masks/`

```bash
python scripts/run_node_sam2_masks.py \
  --scene_dir data/nodes/node01/scenes/<scene_id> \
  --checkpoint third_party/sam2/checkpoints/sam2.1_hiera_tiny.pt \
  --model_cfg configs/sam2.1/sam2.1_hiera_t.yaml \
  --camera_box cam0=x1,y1,x2,y2 \
  --camera_box cam1=x1,y1,x2,y2 \
  --camera_box cam2=x1,y1,x2,y2
```

4) 构建节点级单轨 tracklet

```bash
python scripts/build_node_tracklets.py \
  --scene_dir data/nodes/node01/scenes/<scene_id> \
  --mask_subdir masks \
  --depth_subdir depth \
  --identity_id person_a
```

如果第 3 步直接使用 SAM2 的 `obj_000/` 输出，也可以这样指定：

```bash
python scripts/build_node_tracklets.py \
  --scene_dir data/nodes/node01/scenes/<scene_id> \
  --mask_subdir masks \
  --depth_subdir depth
```

5) 提取节点级 embedding

```bash
python scripts/extract_node_track_embeddings.py \
  --scene_dir data/nodes/node01/scenes/<scene_id> \
  --rgb_backend hist \
  --geo_backend radial_hist
```

6) 做 query-gallery 检索评估

```bash
python scripts/eval_node_track_retrieval.py \
  --query_scene_dir data/nodes/node01/scenes/<query_scene_id> \
  --gallery_scene_dir data/nodes/node01/scenes/<gallery_scene_id>
```

如果你已经有 GT depth 融合点云，也可以继续复用已有的 `scripts/recon_fuse_depth_points.py`，再把 `extract_node_track_embeddings.py` 的 `--geo_backend` 切到 `radial_hist` 或 `open3d_fpfh`。

## 快速开始（先用视频文件验证）

0) 准备一个视频文件（例如 `demo.mp4`）。

说明：示例视频不随仓库提供，请自行准备。

- 如果你把它放在 `mvp-demo/` 目录下：`--source ./demo.mp4`
- 如果你把它放在仓库根目录：`--source ../demo.mp4`

1) 门控采集：用视频文件模拟“视频流”，触发后采集到一个 scene

```bash
conda activate mvp_capture
python scripts/gated_capture_yolo.py \
  --source ./demo.mp4 \
  --scene_root data/scenes \
  --gate_classes person
```

脚本结束会打印：

```text
[done] last_scene_dir=...
```

2) 3DGS 重建：对生成的 `SCENE_DIR` 跑 `convert.py → train.py → depth_npy`

注意：`run_3dgs_scene.py` 会用“当前 python”去调用 3DGS 仓库脚本，所以务必在 `gaussian_splatting` 环境里运行。

```bash
conda activate gaussian_splatting
python scripts/run_3dgs_scene.py \
  --scene_dir data/scenes/<scene_id> \
  --model_dir output/<scene_id>
```

跑完后，期望产物：

- `data/scenes/<scene_id>/images/*.jpg`
- `data/scenes/<scene_id>/sparse/0/{cameras.bin,images.bin,points3D.bin}`
- `data/scenes/<scene_id>/depth_npy/*.npy`（以及 `depth_meta.json`、`depth_index.csv`）

如果你装了 ImageMagick 并希望多尺度 `--resize`：

```bash
python scripts/run_3dgs_scene.py \
  --scene_dir data/scenes/<scene_id> \
  --model_dir output/<scene_id> \
  --resize
```

## 自动触发 3DGS（可选）

如果你不想手动跑第 2 步，可以在门控采集时加 `--auto_3dgs`：录制结束后自动启动 `convert.py → train.py → depth_npy`。

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

- 默认是后台运行（不阻塞采集）；日志：`output/<scene_id>/run_3dgs.log`
- 深度输出：`data/scenes/<scene_id>/depth_npy/*.npy`（与 `images/` 同 stem 对齐）
- 如果你想等待 3DGS 跑完再退出：加 `--gs_mode blocking`

## 摄像头实时输入（后期切换）

把 `--source` 换成摄像头索引即可（其余逻辑不变）：

```bash
conda activate mvp_capture
python scripts/gated_capture_yolo.py \
  --source 0 \
  --scene_root data/scenes \
  --gate_classes person \
  --display
```

如果你在无 GUI/远程 SSH 环境，去掉 `--display`（否则 OpenCV 可能打不开窗口）。

## 下游：3D ReID（MVP 从这里接）

前提：你已经得到一个 `SCENE_DIR`，并且 **`images/` 与 `depth_npy/` 同 stem 一一对应**（本仓库的 `run_3dgs_scene.py` 已按这个口径输出）。

下面命令默认你已进入本目录：

```bash
cd mvp-demo
```

### 1) 导出相机内参（COLMAP → intrinsics.json）

说明：这个脚本只读取 `SCENE_DIR/sparse/0/cameras.bin` 并写 `intrinsics.json`，不依赖 `colmap` 命令；建议在 **Python ≥ 3.9** 的环境里运行（例如 `mvp_capture` / `reid`）。如果你的 `gaussian_splatting` 环境还是 Python 3.7，不要在那里面跑。

```bash
python scripts/export_colmap_intrinsics_json.py \
  --scene_dir data/scenes/<scene_id>
```

产物：`data/scenes/<scene_id>/intrinsics.json`（按 stem 映射 `fx,fy,cx,cy`）。

### 2) 生成 tracklets（两种方式二选一）

**A. 有 mask（推荐）**：把 SAM2 的视频分割输出保存成 `SCENE_DIR/masks/obj_*/<stem>.png`（0/255），然后：

（第一次用 SAM2）配置示例（建议独立 `sam2` 环境；输入必须用 `SCENE_DIR/images/`，不要用 `input/`）：

```bash
conda create -n sam2 python=3.10 -y
conda activate sam2

# 装 PyTorch（GPU 优先；如果你只能用 CPU，就换成 cpu index-url）
conda install -c pytorch -c nvidia pytorch torchvision torchaudio pytorch-cuda=12.1 -y

pip install opencv-python pillow tqdm

mkdir -p third_party
# 如果仓库使用 submodule：
git submodule update --init --recursive
# 如果没有 submodule，再手动 clone：
# git clone https://github.com/facebookresearch/sam2.git third_party/sam2
pip install -e third_party/sam2

# 下载权重（按 SAM2 官方脚本）
(cd third_party/sam2/checkpoints && bash ./download_ckpts.sh)
```

生成 masks（需要你给一个首帧的 init box；坐标系就是 `images/*.jpg` 的像素坐标）：

如果你有 GUI，可以用 OpenCV 快速圈一个框拿到坐标（示例读第 0 帧）：

```bash
python - <<'PY'
import cv2
img = cv2.imread("data/scenes/<scene_id>/images/000000.jpg")
x, y, w, h = cv2.selectROI("init_box", img, fromCenter=False, showCrosshair=True)
print(f"{x},{y},{x+w},{y+h}")
PY
```

```bash
conda activate sam2
python scripts/run_sam2_video_masks.py \
  --scene_dir data/scenes/<scene_id> \
  --init_frame 0 \
  --init_box "x1,y1,x2,y2" \
  --checkpoint third_party/sam2/checkpoints/<your_ckpt>.pt \
  --model_cfg configs/sam2.1/<your_cfg>.yaml
```

本仓库当前下载到的 checkpoint/config 对应关系（任选其一）：

```text
checkpoint: sam2.1_hiera_large.pt      -> model_cfg: configs/sam2.1/sam2.1_hiera_l.yaml
checkpoint: sam2.1_hiera_base_plus.pt  -> model_cfg: configs/sam2.1/sam2.1_hiera_b+.yaml
checkpoint: sam2.1_hiera_small.pt      -> model_cfg: configs/sam2.1/sam2.1_hiera_s.yaml
checkpoint: sam2.1_hiera_tiny.pt       -> model_cfg: configs/sam2.1/sam2.1_hiera_t.yaml
```

期望产物：`data/scenes/<scene_id>/masks/obj_000/<stem>.png`

```bash
python scripts/make_tracklets_from_masks.py \
  --scene_dir data/scenes/<scene_id>
```

**B. 没 mask（bbox-only baseline）**：用 YOLO + ByteTrack 在 `images/` 上直接跟踪：

说明：`yolov8n.pt` 不随仓库提供，Ultralytics 会自动下载，或手动放到本目录再指定路径。

```bash
conda activate mvp_capture
python scripts/make_tracklets_yolo.py \
  --scene_dir data/scenes/<scene_id> \
  --model yolov8n.pt \
  --tracker bytetrack.yaml
```

注：Ultralytics 的 tracker 可能依赖额外包（例如 `lap`）；如果提示缺失，先在同一环境里 `pip install lap` 再重跑。

产物：`data/scenes/<scene_id>/tracklets.json`

### 3) 提特征（track embedding）并做检索

建议单独建一个 `reid` 环境（避免和 3DGS/YOLO 的 torch 版本互相干扰）。`extract_track_embeddings.py` 支持：
- `open_clip_torch + torch` 可用时：用 CLIP 做 RGB embedding；否则自动回退到颜色直方图
- `open3d` 可用时：用 FPFH 做几何 embedding；否则自动回退到简单的径向直方图

（可选）最小 `reid` 环境（CPU 版 torch，够你先跑通流程）：

```bash
conda create -n reid python=3.10 -y
conda activate reid
pip install numpy opencv-python tqdm
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install open_clip_torch open3d
```

提取 embedding：

```bash
python scripts/extract_track_embeddings.py \
  --scene_dir data/scenes/<scene_id>
```

产物：`data/scenes/<scene_id>/embeddings/tracks.npy`、`data/scenes/<scene_id>/embeddings/tracks_meta.json`

跨 scene 做 top‑K 检索（numpy baseline）：

```bash
python scripts/search_track_embeddings.py \
  --query_scene_dir data/scenes/<sceneA> \
  --gallery_scene_dir data/scenes/<sceneB> \
  --topk 5
```
