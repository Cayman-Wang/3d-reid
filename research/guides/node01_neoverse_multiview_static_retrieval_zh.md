# Node01 NeoVerse 三机联合静态检索接入说明

## 目标

本指南说明如何将 NeoVerse 三相机联合重建产物接入节点级 retrieval 实验分支。

## 1. manifest.json 语义

输入索引固定为：

- mvp-demo/output/neoverse_multiview/<scene_id>/input/manifest.json

关键字段：

- schema_version: 当前固定 neoverse_multiview_manifest_v1。
- scene_id, scene_dir: 场景标识与绝对路径。
- cams: 首版固定 [cam0, cam1, cam2]。
- num_sync_steps: 采样后的同步时间步数，默认 27。
- views: 按固定顺序展开的视图列表，顺序必须是 cam0_t0, cam1_t0, cam2_t0, cam0_t1...。
- views.camera_K: 每个 view 对应 rig.json 的相机内参 K。
- views.camera_pose_c2w: 每个 view 对应 rig.json 的 T_node_from_cam（直接作为 c2w 先验）。
- views.mask_rel: 可空字段；存在时记录路径，不存在时为 null。
- timestamps: 逻辑时间戳序列，规则固定为 [0,0,0,1,1,1,...]。
- sync_steps: 每个 logical_t_idx 对应的 ts_us 与 scene_stem。

说明：masks_gt 仍用于 retrieval 侧 tracklet/框裁剪，但不是 multiview manifest 的硬前置条件。

## 2. reconstruction_bundle.pt 内容

联合重建输出目录：

- mvp-demo/output/neoverse_multiview/<scene_id>/run_full_frame_joint/

最小产物：

- reconstruction_bundle.pt
- probe_meta.json
- pose_cluster_report.json
- camera_prior_alignment.json
- scene.glb

bundle 关键字段：

- splats_serialized: CPU 可重载高斯序列化结果。
- predicted_camera_intrinsics: 模型预测相机内参张量。
- predicted_camera_cam2world: 模型预测相机位姿张量。
- rendered_intrinsics: 实际用于渲染/锚定几何的内参张量（当前固定对齐 rig 输入）。
- rendered_cam2world: 实际用于渲染/锚定几何的位姿张量（当前固定对齐 rig 输入）。
- rendered_timestamps: 实际用于渲染的时间戳张量。
- source_manifest: 完整输入 manifest 副本。

probe_meta 关键语义：

- conditioning_mode: rig_camera_priors。
- geometry_anchor_mode: rig_gtcamera。
- cond_flags: [0,1,1]（固定启用 rays + camera priors，不启用 depth prior）。
- camera_prior_source: rig.json。
- render_camera_source: rig_input。
- splat_camera_source: rig_input。
- pose_cluster_report.json: 当前统计的是 predicted camera poses 的中心聚类，不再混同于 rig-anchored rendered poses。
- camera_prior_alignment.json: 记录 rig / predicted / rendered 的中心和内参差异。

## 3. points_neoverse_multiview 如何进入 retrieval

导出脚本会把 bundle 转为每时间戳点云：

- 输出目录: scene_dir/recon/points_neoverse_multiview/
- 命名规则: 使用 manifest 中 logical_t_idx 对应的 scene_stem。
- 几何覆盖: 仅覆盖采样的同步时刻（默认 27 个），不是全时序 dense 4D geometry。
- 过滤规则: opacity >= 0.05，若存在 confidence 则要求 > 0.0。
- 后处理: 默认体素下采样 voxel_size_m=0.02，最多 50000 点。
- 最小点数: 默认 min_points=32，小于阈值的时间步不写出。
- 索引文件: meta.json 和 points_index.csv；当前 schema_version 固定为 points_neoverse_multiview_v2。

在 benchmark 中启用分支：

- branch: rgb_neoverse_multiview_geometry
- manifest: research/plans/tri_camera_node_3d_aware_reid/benchmarks/node01_neoverse_multiview_static_v1.json
- branch config: require_points_per_timestamp=true，确保 tracklet 时间步与几何时间步严格对齐。
- branch config: min_points_per_timestamp=32，确保空几何和极小几何不会进入 retrieval。

运行顺序：

1. 先执行预计算脚本，批量生成 recon/points_neoverse_multiview。
2. 再执行 run_iciscae_branch_eval.py，复用既有 retrieval 主逻辑。

## 4. 从 bundle 做轻量渲染预览

如果只想核对 4D bundle 的几何效果，可以直接运行：

```bash
D:\ML\anaconda3\envs\neoverse\python.exe mvp-demo/scripts/render_neoverse_multiview_preview.py \
  --bundle mvp-demo/output/neoverse_multiview/mj_node01_j10_spin_static_yp_a/run_full_frame_joint/reconstruction_bundle.pt \
  --preview_mode both \
  --camera_source rendered
```

预览输出分三类：

- `original/`：cam0/cam1/cam2 原视角回放。
- `original_compare/`：输入帧、渲染帧、alpha mask 的对比视频。
- `orbit/`：基于 cam0 的 novel-view orbit 预览。

默认分辨率优先读同目录 `probe_meta.json`，否则回退到 `280x168`。

补充说明：

- `precompute_neoverse_multiview_static_geometry.py --skip_if_points_exist` 不再只看有没有 `*.npy`，而是会同时校验 `meta.json` 中的 source bundle、out_subdir、filter 参数与 schema_version。
- 只有缓存和当前导出配置完全一致时才会跳过；否则会视为 stale cache 并重新导出。
