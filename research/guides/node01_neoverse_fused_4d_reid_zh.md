# Node01 NeoVerse Fused 4D 点云接入 3D-ReID 指南

## 目标与边界

本指南用于固定当前实验分支：

- 目标：`NeoVerse 三相机 4D 动态点云 -> points_by_timestamp -> tracklet -> embedding -> ReID`
- 范围：`mvp-demo` 下的三相机 NeoVerse fused 分支
- 非目标：不覆盖 `third_party/NeoVerse` 的单目动静分离文档

当前口径必须区分两件事：

- 已完成：`node01` 单节点 `3 identities x 2 scenes` 的工程闭环与可评测矩阵 bootstrap；`RGB-only` 与 `RGB + NeoVerse 4D geometry` 两条分支都已得到 `metric_queries=6`
- 未完成：cross-node benchmark；当前 `node01` bootstrap 已有 Rank/mAP，但尚未观察到几何分支带来稳定增益

当前默认采集轨迹也已冻结：

- `traj=static_spin_yaw_pitch`
- `yaw_start_deg=-45`、`yaw_end_deg=45`
- `pitch_amp_deg=20`、`pitch_period=8`
- `seconds=8`、`fps=30`

除专门做消融对照外，后续 NeoVerse fused 4D 运行都默认沿用这组带俯仰幅度的轨迹。

## 环境分工

当前项目允许同时使用 `mvp_demo` 与 `neoverse` 两个 `conda` 环境，但职责已经固定：

- `mvp_demo`：MuJoCo 采集、`run_node_depth_anything_v2.py`、`run_node_sam2_masks.py`、`build_node_tracklets.py`、`extract_node_track_embeddings.py`、`eval_node_track_retrieval.py` 等常规节点主线脚本
- `neoverse`：NeoVerse fused 4D 专用脚本，包括 `run_neoverse_per_camera_bundle.py`、`export_neoverse_view_observations.py`、`backproject_neoverse_observations.py`、`fuse_neoverse_multiview_world_points.py`、`constrain_neoverse_multiview_dynamic.py`、`render_fused_world_preview.py`、`analyze_fused_multiview_quality.py`、`preview_points_by_timestamp_slider.py`

执行口径：

- 未特别说明时，项目默认环境仍是 `mvp_demo`
- 只要进入 NeoVerse fused 4D 七步链或其预览/分析脚本，就应显式切到 `neoverse`
- 当前本机实践已经证明这两个环境都可用，因此后续文档与命令应继续按此分工记录，而不是要求单环境统一承载所有脚本

### 运行注意事项（当前机器）

当前这台 Windows 机器有两个已经复现过的稳定坑，后续默认按下面的规避方式执行：

1. 不要优先用 `conda run -n ... python ...` 包 NeoVerse fused 4D 脚本。
   原因：当前控制台会出现 `UnicodeEncodeError: 'gbk' codec can't encode character ...`，脚本可能已经执行但 `conda` 在打印 stdout 时崩掉，容易误判步骤是否成功。
   当前机器默认改成直接调用环境内解释器：

```powershell
D:\ML\anaconda3\envs\neoverse\python.exe ...
D:\ML\anaconda3\envs\mvp_demo\python.exe ...
```

2. `export_neoverse_view_observations.py` 在本机不要用 `float16`。
   原因：当前会触发：

```text
"projection_ewa_3dgs_fused_fwd_kernel" not implemented for 'Half'
```

   直接后果是 `observations/index.csv` 为空，下游会继续报 `No observations found ...`。
   当前机器固定写成：

```powershell
--torch_dtype float32
```

补充说明：

- `run_neoverse_per_camera_bundle.py` 当前仍可沿用默认 `float16`
- 但 `export_neoverse_view_observations.py` 必须显式切到 `float32`
- 若 `observations_report.json` 中 `num_rows = 0`，优先先检查是否误用了 `float16`

## 1. 当前链路

当前建议的脚本链分两段：

1. 全链路产物生成：
   `scripts/run_neoverse_per_camera_bundle.py -> scripts/export_neoverse_view_observations.py -> scripts/backproject_neoverse_observations.py -> scripts/fuse_neoverse_multiview_world_points.py -> scripts/constrain_neoverse_multiview_dynamic.py -> scripts/analyze_fused_multiview_quality.py / scripts/render_fused_world_preview.py`
2. 基于已有产物接 ReID：
   `scripts/build_node_tracklets.py -> scripts/extract_node_track_embeddings.py -> scripts/eval_node_track_retrieval.py`

可用的分析/导出辅助脚本包括：

- `scripts/analyze_fused_multiview_quality.py`
- `scripts/analyze_iciscae_score_fusion.py`
- `scripts/preview_points_by_timestamp_slider.py`
- `scripts/prepare_neoverse_multiview_manifest.py`
- `scripts/export_neoverse_multiview_points.py`

其中 `prepare_neoverse_multiview_manifest.py` 与 `export_neoverse_multiview_points.py` 更偏向历史 multiview/static 分支辅助脚本，不属于当前 fused 4D 主线入口。

### 1.1 当前机器推荐执行口径

在当前机器上，推荐把 NeoVerse fused 4D 主链固定为以下口径：

- `run_neoverse_per_camera_bundle.py`：
  - `width = 280`
  - `height = 168`
  - `resize_mode = resize`
  - `input_variant = object_crop`
  - `crop_padding = 0.25`
  - `crop_mask_source = auto`
  - `num_frames = 81`
- `export_neoverse_view_observations.py`：
  - `torch_dtype = float32`
  - `camera_source = rendered`
- `backproject_neoverse_observations.py`：
  - `fg_alpha_thresh = 0.01`
  - `bg_alpha_thresh = 0.02`
  - `fg_voxel_size_m = 0.005`
  - `bg_voxel_size_m = 0.02`
  - `mask_dilate_px = 3`
- `fuse_neoverse_multiview_world_points.py`：
  - `bg_voxel_size_m = 0.02`
  - `dynamic_voxel_size_m = 0.01`
  - `min_bg_cam_support = 2`
  - `dynamic_track_radius_m = 0.40`
  - `dynamic_merge_radius_m = 0.08`
  - `dynamic_min_component_points = 12`
- `constrain_neoverse_multiview_dynamic.py`：
  - `hull_voxel_size_m = 0.02`
  - `output_voxel_size_m = 0.01`
  - `roi_padding_m = 0.12`
  - `min_mask_cam_support = 2`
  - `point_support_radius_m = 0.03`
  - `depth_trim_radius_m = 0.04`
  - `min_trimmed_points = 40`
  - `scale_guard_ratio = 0.25`
  - `min_depth_mask_pixels = 24`
  - `depth_support_source = aligned_fg_points`
  - `max_roi_voxels = 400000`

## 2. 输出契约（当前冻结）

当前 4D 动态点云正式下游接口固定为：

- `mvp-demo/output/neoverse_fused/<scene_id>/points_by_timestamp/index.csv`
- `mvp-demo/output/neoverse_fused/<scene_id>/points_by_timestamp/meta.json`
- `mvp-demo/output/neoverse_fused/<scene_id>/points_by_timestamp/*.npy`

契约要求：

- `meta.json.schema_version` 必须为 `neoverse_points_by_timestamp_v1`
- `index.csv` 每行对应一个可消费时间戳点云
- `*.npy` 为每个时间戳的点云文件

说明：`fused_scene.glb` 只用于静态汇总查看，不作为 4D/ReID 输入文件。
当前权威点云产物位于 `mvp-demo/output/neoverse_fused/<scene_id>/points_by_timestamp/`；若后续希望简化 ReID 调用，可单独新增同步脚本把它镜像到 `scene_dir/recon/points_by_timestamp`，但这不是当前既有契约。

## 3. ReID smoke（当前最小验收）

当前最小 smoke 产物固定为：

- `tracklets_points_by_timestamp_smoke.json`
- `embeddings_points_by_timestamp_smoke/tracks.npy`

参考命令（在 `mvp-demo` 目录执行）：

```bash
python scripts/build_node_tracklets.py \
  --scene_dir data/nodes/node01/scenes/mj_node01_j10_spin_static_yp20_a \
  --mask_subdir masks_gt \
  --depth_subdir depth_gt \
  --points_subdir ../../../../../output/neoverse_fused/mj_node01_j10_spin_static_yp20_a/points_by_timestamp \
  --out tracks/tracklets_points_by_timestamp_smoke.json \
  --min_timestamps 5
```

```bash
python scripts/extract_node_track_embeddings.py \
  --scene_dir data/nodes/node01/scenes/mj_node01_j10_spin_static_yp20_a \
  --tracklets tracks/tracklets_points_by_timestamp_smoke.json \
  --out_dir embeddings_points_by_timestamp_smoke \
  --rgb_backend hist \
  --geo_backend radial_hist
```

当前已知 smoke 证据：

- `points_by_timestamp` 为 `81` 个时间戳
- `tracklets_points_by_timestamp_smoke.json` 至少含 `1` 条 tracklet
- `tracks.npy` 形状为 `(1,161)` 且无 NaN
- `tracks_meta.json` 当前记录 `rgb_backend=hist`、`geo_backend=radial_hist`
- `tracks_meta.json` 当前记录 `n_timestamps_total=81`、`n_timestamps_used=2`
- 这组 smoke 只证明“接口可跑通”，不证明“全时间戳特征聚合已完成”

## 3.1 Node01 eval matrix（当前阶段正式验收）

在 smoke 之外，当前已完成 `node01` 单节点 `3 identities x 2 scenes` 的正式 bootstrap 矩阵验收：

- `manifest`：`research/plans/tri_camera_node_3d_aware_reid/benchmarks/node01_neoverse_fused_4d_eval_matrix_v1.json`
- `6` 个 `scene` 的 `points_by_timestamp` 都满足：
  - `meta.json.schema_version = neoverse_points_by_timestamp_v1`
  - `index.csv = 81 rows`
  - `*.npy = 81`
- `rgb_only_clip_gtmask_eval_v1`：
  - `metric_queries = 6`
  - `mAP = 1.0`
  - `recall_at_1 = 1.0`
  - `recall_at_5 = 1.0`
  - 每个 `scene` 的 `tracks.npy` 形状为 `(1, 512)`
- `rgb_neoverse_fused_4d_clip_fpfh_eval_v1`：
  - `metric_queries = 6`
  - `mAP = 1.0`
  - `recall_at_1 = 1.0`
  - `recall_at_5 = 1.0`
  - 每个 `scene` 的 `tracks.npy` 形状为 `(1, 545)`
  - `tracks_meta.json` 记录 `rgb_backend=clip`、`geo_backend=open3d_fpfh`、`rgb_weight=1.0`、`geo_weight=0.35`

当前阶段结论必须保守表述：

- 已完成接入与可评测对比
- 当前 `masks_gt/depth_gt` bootstrap 下，`RGB-only` 与 `RGB + NeoVerse 4D geometry` 持平
- 不能写成几何分支已经带来指标提升

## 4. 预览与可视化解释

当前建议按以下分工看结果：

- `original_overlay`：看二维对齐和掩码是否贴合目标
- `fused_scene.glb`：看静态汇总几何分布
- `preview_points_by_timestamp_slider.py`：看 4D 动态点云随时间变化

请勿把 `fused_scene.glb` 解释为可播放的 4D 动画文件。
逐帧查看时，应优先消费 `points_by_timestamp/index.csv + *.npy`，而不是直接把 `fused_scene.glb` 当作单帧几何。

## 5. 风险与失败判断

出现以下情况时应视为失败或高风险：

- 三路覆盖率持续偏低（例如某路长期 < 0.6）
- `hist` 或 `radial_hist` fallback 非 0 且持续升高
- `depth_trim_status` 未介入或长期无效
- 点云厚度过大（动态点云呈明显“糊厚层”）
- embedding 只消费了极少数时间戳

当前建议：先修复覆盖率和几何厚度，再推进多身份矩阵评测，避免把 smoke 结果误当正式主结果。

## 6. 切换高性能机器后的优化顺序

当前本地笔记本的主要限制不是“链路不能跑”，而是 NeoVerse 输入分辨率和后续点云密度上限偏低。当前能够稳定运行的配置是 `280x168`，这会直接限制单路细节和最终点云稠密度。

后续切换到高性能机器时，优化顺序固定如下：

1. 先只提高 `per-camera bundle` 输入分辨率，不同时改几何约束参数。
2. 合法候选分辨率优先尝试 `336x336`、`448x448`、`560x336`。
3. 分辨率提高后，先复用当前稳定的 `yp20_r02params` 约束参数做一轮对照，确认增益来自输入细节，而不是参数漂移。
4. 若单帧点云仍显得稀疏，再单独把 `output_voxel_size_m` 从 `0.01` 收紧到 `0.005`，并同步检查 `max_dynamic_points` 是否需要上调。
5. 在单帧厚度已经可接受的前提下，不优先继续收紧 `depth_trim_radius_m`；后续重点应从“压厚度”转向“提细节”和“提密度”。

### 6.1 本机后处理点云增强进展

当前已完成一轮只在本机后处理层做的点云增密尝试：

- `run id`: `2026-04-26_j10_yp20_dense_points_r01`
- 基线：`2026-04-26_j10_yp20_r02params_r01`
- 唯一参数变化：`output_voxel_size_m: 0.01 -> 0.005`
- 未改变项：输入分辨率仍为 `280x168`，未重跑 NeoVerse `per-camera bundle / observations export / backproject`

本轮结果说明：

- `points_by_timestamp/index.csv` 仍为 `81` 行，`multiview_supported_frames=81`，`trim_applied_frames=81`
- `fused_dynamic_points` 没有高于基线，仍为 `204680`
- 三路 `mean_coverage` 与基线一致，未出现额外投影退化，但也没有密度增益
- `depth_support_ratio` 统计也与基线一致，说明仅靠本机后处理把 `output_voxel_size_m` 改到 `0.005`，当前没有带来可见的有效增强

当前定位应保持清晰：

- 这是“本机后处理增强”尝试，只是在已有点云上调输出采样密度，不代表高分辨率源头增强
- 若目标是明显更稠密的单帧点云，优先级仍应放在高性能机器上提高合法输入分辨率，而不是继续在本机只调 `output_voxel_size_m`

补充约束：

- NeoVerse 输入宽高必须是 `14` 的倍数，`320x320` 不是合法尺寸。
- 若目标是更细的点云，不应同时改分辨率、trim 半径和 ROI 参数，否则无法归因。
- 即使换到更强机器，当前产物仍然是点云而不是连续表面；若后续需要更完整外观，仍要单独补表面重建或 Gaussian/mesh 渲染导出。

## 7. 与旧分支关系

本分支是新增实验线：

- 分支名：`rgb_neoverse_fused_4d_geometry`
- 输入几何：`points_by_timestamp`（动态）
- 当前 `points_subdir`：`../../../../../output/neoverse_fused/<scene_id>/points_by_timestamp`

它不替代历史分支：

- `rgb_only`
- `rgb_predicted_depth_geometry`
- `rgb_fused_geometry`
- `rgb_recon_enhanced_geometry`

旧 `v3_clean / recon_spin` 文档继续保留为历史主线证据；本指南只用于当前 NeoVerse 4D 接入阶段。
