# mvp-demo 运行入口

本目录存放当前项目的运行脚本、MuJoCo 资产和输出目录约定，不再负责定义研究边界。

当前权威文档请优先阅读：

- `research/plans/ACTIVE_PLAN.md`
- `research/plans/tri_camera_node_3d_aware_reid/master_plan_zh.md`
- `research/handoffs/tri_camera_node_engineering_handoff_zh.md`

## 当前主线

当前正式主线固定为：

- `MuJoCo`
- `node01`
- `UAV/aircraft`
- `single-node, cross-scene, track-level retrieval`

主链路只消费：

- `cams/cam*/frames/`
- `cams/cam*/masks/`
- `cams/cam*/depth/`
- `calib/rig.json`
- `frame_times.csv`

MuJoCo 导出的 `masks_gt/`、`depth_gt/` 只用于排错和 upper-bound，不进入正式小论文主结果。

## 当前推荐命令链

先安装节点级管线依赖：

```bash
cd mvp-demo
conda activate mvp_demo
python -m pip install -r requirements_node_pipeline.txt
```

### 1. 采集一个三相机场景

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

### 2. 用图像侧模型补 `depth/`

```bash
python scripts/run_node_depth_anything_v2.py \
  --scene_dir data/nodes/node01/scenes/<scene_id>
```

### 3. 用 SAM2 补 `masks/`

```bash
python scripts/run_node_sam2_masks.py \
  --scene_dir data/nodes/node01/scenes/<scene_id> \
  --checkpoint third_party/sam2/checkpoints/sam2.1_hiera_tiny.pt \
  --model_cfg configs/sam2.1/sam2.1_hiera_t.yaml \
  --camera_box cam0=x1,y1,x2,y2 \
  --camera_box cam1=x1,y1,x2,y2 \
  --camera_box cam2=x1,y1,x2,y2
```

正式 benchmark 需要平铺后的 `cams/cam*/masks/<ts>.png`。如果当前还是 `obj_000/<ts>.png` 形式，先在进入 benchmark 前展平。

### 4. 构建 tracklet

```bash
python scripts/build_node_tracklets.py \
  --scene_dir data/nodes/node01/scenes/<scene_id> \
  --mask_subdir masks \
  --depth_subdir depth \
  --min_timestamps 5
```

说明：

- 正式 benchmark 默认从 `capture_meta.target.identity_id` 读取身份标签。
- `--identity_id` 只用于补历史 scene，不应作为正式 scene 的常规写法。

### 5. 可选：融合多相机点云

```bash
python scripts/recon_fuse_depth_points.py \
  --scene_dir data/nodes/node01/scenes/<scene_id> \
  --depth_subdir depth \
  --mask_subdir masks
```

### 6. 提取 embedding

`RGB-only`：

```bash
python scripts/extract_node_track_embeddings.py \
  --scene_dir data/nodes/node01/scenes/<scene_id> \
  --rgb_backend clip \
  --geo_backend none
```

`RGB + fused geometry`：

```bash
python scripts/extract_node_track_embeddings.py \
  --scene_dir data/nodes/node01/scenes/<scene_id> \
  --rgb_backend clip \
  --geo_backend radial_hist
```

### 7. 做检索评测

```bash
python scripts/eval_node_track_retrieval.py \
  --query_scene_dir data/nodes/node01/scenes/<query_scene_id> \
  --gallery_scene_dir data/nodes/node01/scenes/<gallery_scene_id> \
  --exclude_same_track_id \
  --exclude_same_scene \
  --out output/evals/<benchmark_id>/<run_name>.json
```

## 当前可用场景

- `assets/scene/mujoco_humanoid_3cam_node_parallel_j10.xml`
- `assets/scene/mujoco_humanoid_uav1_3cam_node_parallel.xml`
- `assets/scene/mujoco_humanoid_dji_mavic_3cam_node_parallel.xml`

## 历史脚本说明

以下脚本仍保留在目录中，但只作为辅助 demo，不属于当前里程碑：

- `scripts/gated_capture_yolo.py`
- `scripts/run_3dgs_scene.py`
- `scripts/gs_render_depth_npy.py`

如果后续需要查看节点结构验证、数据契约、viewer 检查方法，请回到：

- `research/handoffs/tri_camera_node_engineering_handoff_zh.md`
