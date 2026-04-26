# 三相机节点工程交接（当前有效）

本文档只回答当前主线的工程问题：用什么资产、走什么脚本链、产出什么目录、如何做 smoke check。

研究边界、论文口径和 benchmark 冻结结论请以：

- `research/plans/ACTIVE_PLAN.md`
- `research/plans/tri_camera_node_3d_aware_reid/master_plan_zh.md`
- `research/plans/tri_camera_node_3d_aware_reid/benchmarks/iciscae_node01_uav_v3_clean.json`

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

默认运行环境固定为 `conda` 环境 `mvp_demo`。未特别说明时，本文档中的主链命令默认都在该环境中执行；涉及 MuJoCo scene 加载时，仍优先从 ASCII 工作目录运行。

当前节点级主链脚本如下：

1. `mvp-demo/scripts/mj_capture_3cam_node.py`
2. `mvp-demo/scripts/run_node_depth_anything_v2.py`
3. `mvp-demo/scripts/run_node_sam2_masks.py`
4. `mvp-demo/scripts/build_node_tracklets.py`
5. `mvp-demo/scripts/recon_fuse_depth_points.py`
6. `mvp-demo/scripts/extract_node_track_embeddings.py`
7. `mvp-demo/scripts/eval_node_track_retrieval.py`

当前 NeoVerse fused 4D / spin 重建线的默认采集轨迹固定为：

- `traj=static_spin_yaw_pitch`
- `yaw_start_deg=-45`
- `yaw_end_deg=45`
- `pitch_amp_deg=20`
- `pitch_period=8`
- `seconds=8`
- `fps=30`

除专门做对照实验外，不再回退到旧的低俯仰版本。

以下脚本仍保留在仓库中，但不再作为当前研究执行入口：

- `mvp-demo/scripts/gated_capture_yolo.py`
- `mvp-demo/scripts/run_3dgs_scene.py`
- `mvp-demo/scripts/gs_render_depth_npy.py`

## 3. 当前可用 MuJoCo 资产

### 3.1 节点结构

- `mvp-demo/assets/mujoco_3cam_node_parallel.xml`

### 3.2 当前目标场景

- `mvp-demo/assets/scene/mujoco_3cam_node_parallel_j10.xml`
- `mvp-demo/assets/scene/mujoco_uav1_3cam_node_parallel_v2.xml`
- `mvp-demo/assets/scene/mujoco_su34_3cam_node_parallel.xml`

当前小论文正式身份集合固定为：

- `j10`
- `uav1`
- `su34`

历史 `v1` 归档场景统一收口到：

- `mvp-demo/assets/scene/legacy/v1/mujoco_humanoid_uav1_3cam_node_parallel.xml`
- `mvp-demo/assets/scene/legacy/v1/mujoco_humanoid_dji_mavic_3cam_node_parallel.xml`
- `mvp-demo/assets/scene/legacy/humanoid/mujoco_humanoid_3cam_node_parallel_j10.xml`
- `mvp-demo/assets/scene/legacy/humanoid/mujoco_humanoid_uav1_3cam_node_parallel_v2.xml`
- `mvp-demo/assets/scene/legacy/humanoid/mujoco_humanoid_su34_3cam_node_parallel.xml`

其中 `dji_mavic` 只保留为 `v1` 历史身份与结果复现，不再作为当前主线目标；当前主线不再默认使用任何 humanoid 场景。

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

先进入目录并安装节点级依赖（默认环境：`mvp_demo`）：

```bash
cd mvp-demo
conda activate mvp_demo
python -m pip install -r requirements_node_pipeline.txt
```

### 5.1 采集一个正式风格 scene

```bash
python scripts/mj_capture_3cam_node.py \
  --mjcf assets/scene/mujoco_3cam_node_parallel_j10.xml \
  --node_id node01 \
  --scene_id mj_node01_j10_clean_line_nodes_a \
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

如果要做 branch-specific 几何分支，先把点云写到独立子目录，避免覆盖正式 `RGB-only`：

```bash
python scripts/recon_fuse_depth_points.py \
  --scene_dir data/nodes/node01/scenes/<scene_id> \
  --depth_subdir depth \
  --mask_subdir masks \
  --out_subdir recon/points_fused
```

`RGB + predicted-depth geometry`（当前冻结为 `cam0 + open3d_fpfh`）：

```bash
python scripts/recon_fuse_depth_points.py \
  --scene_dir data/nodes/node01/scenes/<scene_id> \
  --cams cam0 \
  --depth_subdir depth \
  --mask_subdir masks \
  --out_subdir recon/points_depth_cam0
```

```bash
python scripts/build_node_tracklets.py \
  --scene_dir data/nodes/node01/scenes/<scene_id> \
  --mask_subdir masks \
  --depth_subdir depth \
  --points_subdir recon/points_depth_cam0 \
  --out tracks_rgb_predicted_depth_geometry/tracklets.json \
  --min_timestamps 5
```

```bash
python scripts/extract_node_track_embeddings.py \
  --scene_dir data/nodes/node01/scenes/<scene_id> \
  --tracklets tracks_rgb_predicted_depth_geometry/tracklets.json \
  --out_dir embeddings_rgb_predicted_depth_geometry \
  --rgb_backend clip \
  --geo_backend open3d_fpfh
```

`RGB + fused geometry`（当前冻结为三相机融合 + `open3d_fpfh`）：

```bash
python scripts/build_node_tracklets.py \
  --scene_dir data/nodes/node01/scenes/<scene_id> \
  --mask_subdir masks \
  --depth_subdir depth \
  --points_subdir recon/points_fused \
  --out tracks_rgb_fused_geometry/tracklets.json \
  --min_timestamps 5
```

```bash
python scripts/extract_node_track_embeddings.py \
  --scene_dir data/nodes/node01/scenes/<scene_id> \
  --tracklets tracks_rgb_fused_geometry/tracklets.json \
  --out_dir embeddings_rgb_fused_geometry \
  --rgb_backend clip \
  --geo_backend open3d_fpfh
```

### 5.4 做一次 retrieval

```bash
python scripts/eval_node_track_retrieval.py \
  --query_scene_dir data/nodes/node01/scenes/<query_scene_id> \
  --gallery_scene_dir data/nodes/node01/scenes/<gallery_scene_id> \
  --exclude_same_track_id \
  --exclude_same_scene \
  --embeddings_subdir embeddings_rgb_fused_geometry \
  --out output/evals/<benchmark_id>/<run_name>.json
```

如果要批量复现实验矩阵，可直接使用：

```bash
python scripts/run_iciscae_branch_eval.py --manifest research/plans/tri_camera_node_3d_aware_reid/benchmarks/iciscae_node01_uav_v3_clean.json --branch rgb_predicted_depth_geometry
python scripts/run_iciscae_branch_eval.py --manifest research/plans/tri_camera_node_3d_aware_reid/benchmarks/iciscae_node01_uav_v3_clean.json --branch rgb_fused_geometry
python scripts/run_iciscae_branch_eval.py --manifest research/plans/tri_camera_node_3d_aware_reid/benchmarks/iciscae_node01_uav_v3_clean.json --branch gt_upper_bound
python scripts/summarize_iciscae_branch_comparison.py --benchmark_id iciscae_node01_uav_v3_clean
```

`summarize_iciscae_branch_comparison.py` 当前会同时生成：

- `branch_comparison_summary.json/.md`
- `query_failure_analysis.json/.md`

如果只想在既有结果上重跑逐 query 失败分析，可直接执行：

```bash
python scripts/analyze_iciscae_failure_modes.py --benchmark_id iciscae_node01_uav_v3_clean
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

## 7. NeoVerse 4D 点云接入 ReID 实验分支

新增实验分支如下：

- 分支名：`rgb_neoverse_fused_4d_geometry`
- 几何输入：`../../../../../output/neoverse_fused/<scene_id>/points_by_timestamp`
- 输入契约：`index.csv + meta.json + *.npy`

该分支是当前阶段的新增实验线，用于验证 `NeoVerse 三相机 4D 动态点云` 对检索的可用性。

边界说明：

- 它不替代旧主链：`RGB-only / predicted-depth / fused geometry / recon_spin`。
- 它当前只代表工程闭环 proof-of-pipeline，不代表正式多身份 benchmark 结果。
- `fused_scene.glb` 只用于静态查看，不作为 ReID 输入。
- 当前权威点云产物位于 `mvp-demo/output/neoverse_fused/<scene_id>/points_by_timestamp/`；若后续需要简化调用，可单独新增同步脚本把它镜像到 `scene_dir/recon/points_by_timestamp`，但这不是当前既有契约。

最小 smoke 验收（当前固定）：

1. `points_by_timestamp/index.csv` 至少记录 `81` 个时间戳点云
2. 可生成 `tracks/tracklets_points_by_timestamp_smoke.json`，且至少含 `1` 条 tracklet
3. 可生成 `embeddings_points_by_timestamp_smoke/tracks.npy`
4. `tracks.npy` 形状为 `(1,161)`，并且无 NaN
5. 当前 smoke backend 与既有产物一致：`rgb_backend=hist`、`geo_backend=radial_hist`，且 `n_timestamps_used=2`

注：若任一项失败，先回到覆盖率、动态约束和点云厚度诊断，不进入正式 ReID 对比结论。

## 8. 当前完成定义

进入当前主线下一阶段前，至少需要满足：

1. `node01` 的正式 scene 可稳定导出三路 `frames/`、`rig.json` 和 `frame_times.csv`
2. 预测 `depth/` 与预测 `masks/` 已补齐
3. `tracklets.json`、`tracks.npy`、`tracks_meta.json` 可以稳定生成
4. 跨 scene 检索能生成有效 summary，而不是只停留在单 scene 的 proof-of-pipeline
