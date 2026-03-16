# 三相机节点工程交接（当前有效）

本文档只回答当前主线的工程问题：用什么资产、走什么脚本链、产出什么目录、如何做 smoke check。

研究边界、论文口径和 benchmark 冻结结论请以：

- `research/plans/ACTIVE_PLAN.md`
- `research/plans/tri_camera_node_3d_aware_reid/master_plan_zh.md`
- `research/plans/tri_camera_node_3d_aware_reid/benchmarks/iciscae_node01_uav_v1.json`

为准。

## 1. 当前工程边界

当前只收口：

- `MuJoCo`
- `node01`
- `UAV/aircraft`
- `single-node, cross-scene, track-level retrieval`

当前主链路只消费：

- `cams/cam*/frames/`
- `cams/cam*/masks/`
- `cams/cam*/depth/`
- `calib/rig.json`
- `frame_times.csv`

调试和 upper-bound 可额外使用：

- `cams/cam*/masks_gt/`
- `cams/cam*/depth_gt/`

但它们不进入当前小论文主结果。

## 2. 当前工程入口

当前节点级主链脚本如下：

1. `mvp-demo/scripts/mj_capture_3cam_node.py`
2. `mvp-demo/scripts/run_node_depth_anything_v2.py`
3. `mvp-demo/scripts/run_node_sam2_masks.py`
4. `mvp-demo/scripts/build_node_tracklets.py`
5. `mvp-demo/scripts/recon_fuse_depth_points.py`
6. `mvp-demo/scripts/extract_node_track_embeddings.py`
7. `mvp-demo/scripts/eval_node_track_retrieval.py`

以下脚本仍保留在仓库中，但不再作为当前研究执行入口：

- `mvp-demo/scripts/gated_capture_yolo.py`
- `mvp-demo/scripts/run_3dgs_scene.py`
- `mvp-demo/scripts/gs_render_depth_npy.py`

## 3. 当前可用 MuJoCo 资产

### 3.1 节点结构

- `mvp-demo/assets/mujoco_3cam_node_parallel.xml`
- `mvp-demo/assets/mujoco_humanoid_3cam_node_parallel.xml`

### 3.2 当前目标场景

- `mvp-demo/assets/scene/mujoco_humanoid_3cam_node_parallel_j10.xml`
- `mvp-demo/assets/scene/mujoco_humanoid_uav1_3cam_node_parallel.xml`
- `mvp-demo/assets/scene/mujoco_humanoid_dji_mavic_3cam_node_parallel.xml`

当前小论文正式身份集合固定为：

- `j10`
- `uav1`
- `dji_mavic`

## 4. 数据契约

每个正式 scene 目录应满足：

```text
mvp-demo/data/nodes/<node_id>/scenes/<scene_id>/
  capture_meta.json
  frame_times.csv
  calib/
    rig.json
  cams/
    cam0/
      frames/
      masks/
      depth/
      masks_gt/
      depth_gt/
    cam1/
    cam2/
  tracks/
    tracklets.json
  embeddings/
    tracks.npy
    tracks_meta.json
```

固定约定：

- `capture_meta.target.identity_id` 是正式 benchmark 的身份权威来源。
- `frame_times.csv` 是三相机同步时间轴权威来源。
- 正式 benchmark 只接受平铺 mask：`cams/cam*/masks/<ts>.png`。
- `obj_000/<ts>.png` 只允许作为中间产物，不直接进入正式 benchmark。

## 5. 最小 smoke run

先进入目录并安装节点级依赖：

```bash
cd mvp-demo
conda activate mvp_demo
python -m pip install -r requirements_node_pipeline.txt
```

### 5.1 采集一个正式风格 scene

```bash
python scripts/mj_capture_3cam_node.py \
  --mjcf assets/scene/mujoco_humanoid_3cam_node_parallel_j10.xml \
  --node_id node01 \
  --scene_id mj_node01_j10_line_nodes_a \
  --identity_id j10 \
  --traj line_nodes \
  --save_depth \
  --save_masks_gt
```

最低成功标准：

- 成功生成 `data/nodes/node01/scenes/<scene_id>/`
- 三路 `frames/` 存在
- `calib/rig.json` 存在
- `capture_meta.json` 与 `frame_times.csv` 存在

### 5.2 补预测深度和预测 masks

```bash
python scripts/run_node_depth_anything_v2.py \
  --scene_dir data/nodes/node01/scenes/<scene_id>
```

```bash
python scripts/run_node_sam2_masks.py \
  --scene_dir data/nodes/node01/scenes/<scene_id> \
  --checkpoint third_party/sam2/checkpoints/sam2.1_hiera_tiny.pt \
  --model_cfg configs/sam2.1/sam2.1_hiera_t.yaml \
  --camera_box cam0=x1,y1,x2,y2 \
  --camera_box cam1=x1,y1,x2,y2 \
  --camera_box cam2=x1,y1,x2,y2
```

### 5.3 构建 tracklet 与 embedding

```bash
python scripts/build_node_tracklets.py \
  --scene_dir data/nodes/node01/scenes/<scene_id> \
  --mask_subdir masks \
  --depth_subdir depth \
  --min_timestamps 5
```

`RGB-only`：

```bash
python scripts/extract_node_track_embeddings.py \
  --scene_dir data/nodes/node01/scenes/<scene_id> \
  --rgb_backend clip \
  --geo_backend none
```

如果已经完成多相机点云融合：

```bash
python scripts/recon_fuse_depth_points.py \
  --scene_dir data/nodes/node01/scenes/<scene_id> \
  --depth_subdir depth \
  --mask_subdir masks
```

```bash
python scripts/extract_node_track_embeddings.py \
  --scene_dir data/nodes/node01/scenes/<scene_id> \
  --rgb_backend clip \
  --geo_backend radial_hist
```

### 5.4 做一次 retrieval

```bash
python scripts/eval_node_track_retrieval.py \
  --query_scene_dir data/nodes/node01/scenes/<query_scene_id> \
  --gallery_scene_dir data/nodes/node01/scenes/<gallery_scene_id> \
  --exclude_same_track_id \
  --exclude_same_scene \
  --out output/evals/<benchmark_id>/<run_name>.json
```

## 6. 节点结构验证

### 6.1 静态检查

打开 `mvp-demo/assets/mujoco_3cam_node_parallel.xml`，确认：

- 存在 `body name="node01"`
- `node01` 下存在 3 个相机：`node01_cam0/1/2`
- 三个相机 `xyaxes` 一致
- 三相机位置构成小圈，直径不超过 `1m`

### 6.2 viewer 检查

```bash
MUJOCO_GL=glfw python scripts/mj_view_3cam_node.py
```

### 6.3 离屏采集检查

```bash
MUJOCO_GL=osmesa python scripts/mj_capture_3cam_node.py --seconds 1 --fps 5 --width 320 --height 240
```

## 7. 当前完成定义

进入当前主线下一阶段前，至少需要满足：

1. `node01` 的正式 scene 可稳定导出三路 `frames/`、`rig.json` 和 `frame_times.csv`
2. 预测 `depth/` 与预测 `masks/` 已补齐
3. `tracklets.json`、`tracks.npy`、`tracks_meta.json` 可以稳定生成
4. 跨 scene 检索能生成有效 summary，而不是只停留在单 scene 的 proof-of-pipeline
