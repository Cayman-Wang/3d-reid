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

当前激活 benchmark 为 `iciscae_node01_uav_v3_clean`，身份集合固定为：

- `j10`
- `uav1`
- `su34`

历史 `iciscae_node01_uav_v1 / v2` 结果只保留为归档，不再作为当前主线。

主链路只消费：

- `cams/cam*/frames/`
- `cams/cam*/masks/`
- `cams/cam*/depth/`
- `calib/rig.json`
- `frame_times.csv`

MuJoCo 导出的 `masks_gt/`、`depth_gt/` 只用于排错和 upper-bound，不进入正式小论文主结果。

## 当前推荐命令链

当前默认运行环境固定为 `conda` 环境 `mvp_demo`。未特别说明时，本目录中的主线脚本命令均默认在该环境中执行；涉及 MuJoCo scene 加载时，建议从 ASCII 工作目录运行。

先安装节点级管线依赖：

```bash
cd mvp-demo
conda activate mvp_demo
python -m pip install -r requirements_node_pipeline.txt
```

如果是第一次使用 `uav1_v2 / su34` 的 clean 场景，先在本地生成 split meshes：

```bash
python scripts/prepare_su34_png_textures.py

python scripts/split_obj_by_usemtl.py \
  --obj assets/models/uav1_ascii/cgaxis_models_117_01_obj_drone.obj \
  --out-dir assets/models/uav1_ascii/meshes \
  --prefix uav1_mtl_

python scripts/split_obj_by_usemtl.py \
  --obj assets/models/su34/Sukhoi-34.obj \
  --out-dir assets/models/su34/meshes \
  --prefix su34_mtl_
```

### 1. 采集一个三相机场景

```bash
python scripts/mj_capture_3cam_node.py \
  --mjcf assets/scene/mujoco_su34_3cam_node_parallel.xml \
  --node_id node01 \
  --scene_id mj_node01_su34_clean_line_nodes_a \
  --identity_id su34 \
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
  --geo_backend open3d_fpfh
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

- `assets/scene/mujoco_3cam_node_parallel.xml`
- `assets/scene/mujoco_3cam_node_parallel_j10.xml`
- `assets/scene/mujoco_uav1_3cam_node_parallel_v2.xml`
- `assets/scene/mujoco_su34_3cam_node_parallel.xml`

历史归档场景：

- `assets/scene/legacy/v1/`
- `assets/scene/legacy/humanoid/`

说明：

- `assets/scene/` 根目录只保留当前 clean 主线场景。
- 所有 `mujoco_humanoid_*.xml` 已归档到 `assets/scene/legacy/`，不再作为当前推荐入口。

## 当前推荐 benchmark 入口

`v3_clean` 结果线建议显式传入 manifest：

```bash
python scripts/run_iciscae_branch_eval.py \
  --manifest research/plans/tri_camera_node_3d_aware_reid/benchmarks/iciscae_node01_uav_v3_clean.json \
  --branch rgb_only
```

全部 branch 跑完后，统一结果与失败分析可一起生成：

```bash
python scripts/summarize_iciscae_branch_comparison.py \
  --benchmark_id iciscae_node01_uav_v3_clean
```

该命令当前会同时输出：

- `output/evals/iciscae_node01_uav_v3_clean/branch_comparison_summary.json`
- `output/evals/iciscae_node01_uav_v3_clean/branch_comparison_summary.md`
- `output/evals/iciscae_node01_uav_v3_clean/query_failure_analysis.json`
- `output/evals/iciscae_node01_uav_v3_clean/query_failure_analysis.md`

如果只想单独重跑逐 query 失败分析，也可以直接执行：

```bash
python scripts/analyze_iciscae_failure_modes.py \
  --benchmark_id iciscae_node01_uav_v3_clean
```

## 历史脚本说明

以下脚本仍保留在目录中，但只作为辅助 demo，不属于当前里程碑：

- `scripts/gated_capture_yolo.py`
- `scripts/run_3dgs_scene.py`
- `scripts/gs_render_depth_npy.py`

如果后续需要查看节点结构验证、数据契约、viewer 检查方法，请回到：

- `research/handoffs/tri_camera_node_engineering_handoff_zh.md`
