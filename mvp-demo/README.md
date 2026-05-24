# mvp-demo 运行入口

本目录存放当前项目的运行脚本、MuJoCo 资产和输出目录约定，不再负责定义研究边界。

当前权威文档请优先阅读：

- `research/plans/ACTIVE_PLAN.md`
- `research/plans/tri_camera_node_3d_aware_reid/master_plan_zh.md`
- `research/handoffs/tri_camera_node_engineering_handoff_zh.md`

## 当前主线

当前 M8 正式主线固定为：

- `MuJoCo` clean UAV/aircraft 三相机节点级 track-level 3D/4D-aware retrieval
- `NeoVerse fused 4D points_by_timestamp -> tracklet -> embedding -> ReID`
- `node01` 单节点 `3 identities x 2 scenes` NeoVerse fused 4D 可评测矩阵已闭环
- 下一步并行推进 `node02 / cross-node smoke` 与 ReID 表征补强 prototype

当前 4D 几何主输入契约为：

- `points_by_timestamp/index.csv`
- `points_by_timestamp/meta.json`
- `points_by_timestamp/*.npy`
- `meta.json.schema_version = neoverse_points_by_timestamp_v1`

当前已闭环的 `node01` NeoVerse fused 4D matrix 身份集合为：

- `j10`
- `uav1`
- `su34`

`iciscae_node01_uav_v3_clean` 以及历史 `iciscae_node01_uav_v1 / v2` 结果只保留为 ICISCAE/历史保底线，不再作为当前 M8 NeoVerse 4D 主入口。

节点侧基础数据仍消费：

- `cams/cam*/frames/`
- `cams/cam*/masks/`
- `cams/cam*/depth/`
- `calib/rig.json`
- `frame_times.csv`

当前 `node01_neoverse_fused_4d_eval_matrix_v1` bootstrap 使用 `masks_gt/depth_gt` 隔离几何接入问题。该矩阵只能写成 NeoVerse 4D geometry 已接入并可评测，且当前与 RGB-only 持平；不能写成 geometry、三相机或 own-depth 已带来 Rank/mAP 提升。

## 当前推荐命令链

当前 Linux 适配阶段统一使用本机 `neoverse` 环境推进，不再依赖尚未补齐的 `mvp_demo` 环境。该环境当前覆盖 MuJoCo、OpenCV、PyTorch、Pillow、Open3D、Transformers 与 NeoVerse 基础依赖。

```bash
conda activate neoverse
```

若从仓库根目录直接调用，可显式使用环境解释器：

```bash
/home/grasp/miniconda3/envs/neoverse/bin/python mvp-demo/scripts/mj_capture_3cam_node.py --help
```

当前 `neoverse` 环境暂未安装 `open_clip`，Linux smoke 优先使用直方图 RGB 表征：

```bash
export REID_LINUX_RGB_BACKEND=hist
```

若要复用历史 Windows manifest 中的 `D:/node01_spin_runtime_ascii/...` 路径，在 Linux 上设置运行时根目录映射：

```bash
export REID_NODE01_RUNTIME_ROOT=/home/grasp/data/3d-reid
```

说明：映射只影响以 `D:/node01_spin_runtime_ascii/` 开头的历史绝对路径；普通相对路径仍按仓库根目录或 scene 目录解析。

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
- `assets/scene/mujoco_3cam_node_parallel_proxy.xml`（Linux smoke 优先入口，不依赖外部模型资源）

历史归档场景：

- `assets/scene/legacy/v1/`
- `assets/scene/legacy/humanoid/`

说明：

- `assets/scene/` 根目录只保留当前 clean 主线场景。
- 所有 `mujoco_humanoid_*.xml` 已归档到 `assets/scene/legacy/`，不再作为当前推荐入口。
- 当前 Linux 迁移 smoke 先使用 proxy 场景；J10/UAV1/SU34 真实模型场景需要补齐 `mvp-demo/assets/models/` 资源后再验证。

## Linux smoke（neoverse 环境）

从仓库根目录执行：

```bash
export PYTHON=/home/grasp/miniconda3/envs/neoverse/bin/python
export MUJOCO_GL=egl

$PYTHON mvp-demo/scripts/mj_capture_3cam_node.py \
  --mjcf mvp-demo/assets/scene/mujoco_3cam_node_parallel_proxy.xml \
  --node_id node01 \
  --scene_id linux_proxy_smoke \
  --identity_id proxy \
  --seconds 0.2 \
  --fps 5 \
  --save_depth \
  --save_masks_gt

$PYTHON mvp-demo/scripts/build_node_tracklets.py \
  --scene_dir mvp-demo/data/nodes/node01/scenes/linux_proxy_smoke \
  --mask_subdir masks_gt \
  --depth_subdir depth_gt \
  --min_timestamps 1

$PYTHON mvp-demo/scripts/extract_node_track_embeddings.py \
  --scene_dir mvp-demo/data/nodes/node01/scenes/linux_proxy_smoke \
  --rgb_backend hist \
  --geo_backend none
```

几何 smoke 可在已有 depth/mask 的 scene 上继续执行：

```bash
$PYTHON mvp-demo/scripts/recon_fuse_depth_points.py \
  --scene_dir mvp-demo/data/nodes/node01/scenes/linux_proxy_smoke \
  --depth_subdir depth_gt \
  --mask_subdir masks_gt

$PYTHON mvp-demo/scripts/extract_node_track_embeddings.py \
  --scene_dir mvp-demo/data/nodes/node01/scenes/linux_proxy_smoke \
  --rgb_backend hist \
  --geo_backend open3d_fpfh
```

## 当前推荐 benchmark 入口

当前 M8 NeoVerse fused 4D 主入口建议显式传入 manifest：

```bash
python scripts/run_iciscae_branch_eval.py \
  --manifest research/plans/tri_camera_node_3d_aware_reid/benchmarks/node01_neoverse_fused_4d_eval_matrix_v1.json \
  --branch rgb_only_clip_gtmask_eval_v1
```

NeoVerse 4D geometry 对照分支：

```bash
python scripts/run_iciscae_branch_eval.py \
  --manifest research/plans/tri_camera_node_3d_aware_reid/benchmarks/node01_neoverse_fused_4d_eval_matrix_v1.json \
  --branch rgb_neoverse_fused_4d_clip_fpfh_eval_v1
```

当前两个正式对照 branch 为：

- `rgb_only_clip_gtmask_eval_v1`
- `rgb_neoverse_fused_4d_clip_fpfh_eval_v1`

当前 `node01` 结果口径：

- `metric_queries = 6`
- 两个 branch 都是 `mAP = 1.0`、`Rank-1 = 1.0`、`Rank-5 = 1.0`
- 结论是接入并可评测，不能写成 NeoVerse 4D geometry 已经带来指标提升

历史/保底 `iciscae_node01_uav_v3_clean` 入口保留如下，不作为当前 M8 主入口：

```bash
python scripts/run_iciscae_branch_eval.py \
  --manifest research/plans/tri_camera_node_3d_aware_reid/benchmarks/iciscae_node01_uav_v3_clean.json \
  --branch rgb_only
```

历史结果线全部 branch 跑完后，统一结果与失败分析可一起生成：

```bash
python scripts/summarize_iciscae_branch_comparison.py \
  --benchmark_id iciscae_node01_uav_v3_clean
```

## 历史脚本说明

以下脚本仍保留在目录中，但只作为辅助 demo，不属于当前里程碑：

- `scripts/gated_capture_yolo.py`
- `scripts/run_3dgs_scene.py`
- `scripts/gs_render_depth_npy.py`

如果后续需要查看节点结构验证、数据契约、viewer 检查方法，请回到：

- `research/handoffs/tri_camera_node_engineering_handoff_zh.md`

## NeoVerse 并行演示后端（probe）

该入口只用于并行探索展示，不替换当前主线，不改 benchmark 契约，也不接入 retrieval。

### 1) 导出 NeoVerse 输入视频（固定 cam0）

```bash
python scripts/export_neoverse_probe_input.py \
  --scene_dir data/nodes/node01/scenes/mj_node01_j10_spin_static_yp_a \
  --cam_id cam0
```

默认输出：

- `mvp-demo/output/neoverse_probe/mj_node01_j10_spin_static_yp_a/cam0/input/full_frame.mp4`
- `mvp-demo/output/neoverse_probe/mj_node01_j10_spin_static_yp_a/cam0/input/object_crop.mp4`

说明：

- 输入帧固定使用 `cams/cam0/frames`。
- `object_crop.mp4` 使用 `masks` 或 `masks_gt` 做目标中心稳定裁剪，默认 padding 为 20%。
- 默认优先喂 `object_crop.mp4`，仅当裁剪明显破坏时序稳定性时再回退 `full_frame.mp4`。

### 2) 运行 NeoVerse 推理并归档产物

```bash
python scripts/run_neoverse_probe.py \
  --neoverse_python /home/grasp/miniconda3/envs/neoverse/bin/python \
  --neoverse_repo third_party/NeoVerse \
  --scene_dir data/nodes/node01/scenes/mj_node01_j10_spin_static_yp_a \
  --cam_id cam0 \
  --video_variant object_crop \
  --trajectory orbit_left \
  --angle 12 \
  --orbit_radius 0.08 \
  --vis_rendering
```

默认不启用 `--static_scene`（当前是固定相机 + 动态目标，不属于完全静态视频）。

输出目录：

- `mvp-demo/output/neoverse_probe/<scene_id>/cam0/run_<variant>/`

最小对齐产物：

- `input_video.mp4`
- `output.mp4`
- `vis_rendering/`（当启用 `--vis_rendering`）
- `probe_meta.json`（记录 scene/cam/参数/命令/运行时长/输出路径）

## NeoVerse 三机联合重建（静态实验分支）

该入口用于 node01 静态 spin 场景的实验线，不替换 M8 NeoVerse fused 4D 主线。
当前语义是“rig-anchored multiview geometry for retrieval”，不是全时序 dense 4D geometry。

### 1) 准备三机联合输入 manifest

```bash
python scripts/prepare_neoverse_multiview_manifest.py \
  --scene_dir data/nodes/node01/scenes/mj_node01_j10_spin_static_yp_a \
  --num_sync_steps 27
```

输出：

- `mvp-demo/output/neoverse_multiview/<scene_id>/input/manifest.json`

说明：

- manifest 会写入每个 view 的 `camera_K` 与 `camera_pose_c2w`（来自 `rig.json`）。
- `mask_rel` 为可空元信息，不再是联合重建的硬前置条件。

### 2) 运行三机联合 NeoVerse 重建

```bash
python scripts/run_neoverse_multiview_joint.py \
  --manifest mvp-demo/output/neoverse_multiview/mj_node01_j10_spin_static_yp_a/input/manifest.json \
  --neoverse_repo third_party/NeoVerse \
  --device cuda \
  --torch_dtype bfloat16
```

输出目录：

- `mvp-demo/output/neoverse_multiview/<scene_id>/run_full_frame_joint/`

最小产物：

- `reconstruction_bundle.pt`
- `probe_meta.json`
- `pose_cluster_report.json`
- `camera_prior_alignment.json`
- `scene.glb`

说明：

- 联合重建固定使用 rig camera priors，内部 `cond_flags=[0,1,1]`，`use_motion=False`。
- 当前 wrapper 会把 `rendered_cam2world` / `rendered_intrinsics` 直接锚定到 `rig.json`，并在 bundle 中同时保留 `predicted_camera_cam2world` / `predicted_camera_intrinsics`。
- `pose_cluster_report.json` 当前统计的是 predicted camera poses 的中心聚类；`camera_prior_alignment.json` 负责对比 rig / predicted / rendered 三者差异。

### 3) 导出 retrieval 可消费点云

```bash
python scripts/export_neoverse_multiview_points.py \
  --bundle mvp-demo/output/neoverse_multiview/mj_node01_j10_spin_static_yp_a/run_full_frame_joint/reconstruction_bundle.pt \
  --scene_dir data/nodes/node01/scenes/mj_node01_j10_spin_static_yp_a \
  --neoverse_repo third_party/NeoVerse
```

输出目录：

- `scene_dir/recon/points_neoverse_multiview/`

附带：

- `meta.json`
- `points_index.csv`

### 4) 静态实验分支预计算 + retrieval 评测

先批量预计算两条静态场景几何：

```bash
python scripts/precompute_neoverse_multiview_static_geometry.py \
  --manifest research/plans/tri_camera_node_3d_aware_reid/benchmarks/node01_neoverse_multiview_static_v1.json \
  --neoverse_python /home/grasp/miniconda3/envs/neoverse/bin/python \
  --neoverse_repo third_party/NeoVerse
```

再复用 retrieval runner：

```bash
python scripts/run_iciscae_branch_eval.py \
  --manifest research/plans/tri_camera_node_3d_aware_reid/benchmarks/node01_neoverse_multiview_static_v1.json \
  --branch rgb_neoverse_multiview_geometry
```

说明：

- NeoVerse branch 配置 `require_points_per_timestamp=true` 且 `min_points_per_timestamp=32`，仅消费满足最小点数门槛的时间步，避免“RGB 全时序 + 空几何/极小几何”错位。
- `precompute_neoverse_multiview_static_geometry.py --skip_if_points_exist` 会校验 `meta.json`、`points_index.csv`、source bundle、导出 filter 参数和 schema version；只有缓存完全匹配时才跳过。

### 5) 从 bundle 做轻量渲染预览

该脚本只做 Gaussian rasterization，不加载 diffusion / T5 / VAE / LoRA。

```bash
/home/grasp/miniconda3/envs/neoverse/bin/python mvp-demo/scripts/render_neoverse_multiview_preview.py \
  --bundle mvp-demo/output/neoverse_multiview/mj_node01_j10_spin_static_yp_a/run_full_frame_joint/reconstruction_bundle.pt \
  --neoverse_repo third_party/NeoVerse \
  --reconstructor_path third_party/NeoVerse/models/NeoVerse/reconstructor.ckpt \
  --device cuda \
  --torch_dtype float16 \
  --preview_mode both \
  --camera_source rendered
```

默认输出：

- `mvp-demo/output/neoverse_multiview/<scene_id>/run_full_frame_joint/render_preview/original/`
- `mvp-demo/output/neoverse_multiview/<scene_id>/run_full_frame_joint/render_preview/original_compare/`
- `mvp-demo/output/neoverse_multiview/<scene_id>/run_full_frame_joint/render_preview/orbit/`
- `mvp-demo/output/neoverse_multiview/<scene_id>/run_full_frame_joint/render_preview/preview_meta.json`

说明：

- `camera_source=rendered` 时优先复核 rig-anchored 输出。
- `camera_source=predicted_camera` 时允许和 rig 结果产生偏移，用于诊断相机锚点是否真正生效。
- 默认分辨率优先读取同目录 `probe_meta.json`，缺失时回退到 `280x168`。
