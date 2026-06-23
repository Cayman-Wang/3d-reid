# Legacy goal note

This file preserves the previous active default goal as a historical record only. It is no longer the active default.

# CARLA-Air / 4D-ReID Long Task Goal

请接续 `/home/grasp/data/3d-reid` 的 CARLA-Air / 4D-ReID 长任务。

## 总目标

把当前 CARLA-Air pipeline 中的 proxy annotation / proxy points 替换为更真实的 synthetic annotation 和真 4D geometry，然后扩展为多 identity benchmark。

## 必读文件

1. `AGENTS.md`
2. `docs/goal.md`
3. `research/CURRENT_STATUS.md`
4. `research/plans/ACTIVE_PLAN.md`
5. `research/handoffs/carla_air_ground_to_air_collection_handoff_zh.md`
6. `research/reports/carla_air_actor_to_pixel_contract_downstream_gate_integration_2026_05_29_zh.md`
7. `research/reports/carla_air_objective_completion_audit_2026_05_29_zh.md`

## 按需追溯报告

以下报告仅在需要追溯 artifact lineage、blocker tracing、command reproduction 或 regression history 时读取：

- `research/reports/carla_air_target_traceability_gap_audit_2026_05_29_zh.md`
- `research/reports/carla_air_actor_to_pixel_evidence_contract_2026_05_29_zh.md`
- `research/reports/carla_air_traceability_and_ue_import_guard_hardening_2026_05_29_zh.md`
- `research/reports/carla_air_world_semantic_lidar_geometry_audit_2026_05_29_zh.md`
- `research/reports/carla_air_guarded_live_calibration_runner_2026_05_29_zh.md`
- `research/reports/carla_air_post_guarded_traceability_audits_2026_05_29_zh.md`
- `research/reports/carla_air_existing_actor_roi_sweep_target_readiness_integration_2026_05_29_zh.md`
- `research/reports/carla_air_existing_actor_roi_decode_summary_integration_2026_05_29_zh.md`
- `research/reports/carla_air_guarded_existing_actor_sweep_passthrough_2026_05_29_zh.md`
- `research/reports/carla_air_multi_identity_local_poc_current_gate_2026_05_29_zh.md`
- `research/reports/carla_air_research_docs_soft_archive_2026_05_29_zh.md`
- `research/reports/carla_air_scene_status_target_selection_guard_2026_05_29_zh.md`
- `research/reports/carla_air_target_selection_controlled_decode_guard_2026_05_29_zh.md`
- `research/reports/carla_air_neoverse_geometry_readiness_2026_05_29_zh.md`

## 当前状态

- Town10HD `node01-node05` 固定三相机节点已完成。
- POC02/POC03/POC04 live synchronized capture 已跑通。
- POC03/POC04 已 gate pass，并已导出 node 级 `scene_dir` skeleton。
- 4B 数据契约已冻结，正式 capture 输入是：
  - `capture_meta.json`
  - `trajectory_capture_manifest.json`
  - `trajectory_frame_groups.csv`
  - `frame_times.csv`
  - `calib/rig.json`
  - `cams/cam*/frames/*.png`
- `presence_gate.json`、`review_summary.csv`、`review_overlays/`、mp4 只允许作为 QC，不允许作为正式 pipeline 输入。
- 当前已完成“最小正式样本”：
  - POC03 node03/node04
  - 使用 `build_minimal_formal_sample.py` 从 recorded drone pose + fixed camera rig + fixed proxy dimensions 生成 proxy mask / bbox / object pose / proxy points
  - 已生成 `tracks/tracklets.json`
  - 已生成 proxy `points_by_timestamp`
  - 已跑通 `scene_dir -> tracklets -> points_by_timestamp -> embeddings -> retrieval`
  - node03 <-> node04 retrieval smoke 已通过
- 当前 annotation / points 仍是 proxy，不是最终真实 synthetic annotation，也不是 NeoVerse 真 4D geometry。

## 重要约束

- 不修改 `tools/carla_air/capture_camera_nodes_smoke.py`，除非先说明必要性。
- 不把 `presence_gate.json`、`review_summary.csv`、`review_overlays/`、mp4 当正式输入。
- 不伪造 normalized aircraft asset。
- 不把 proxy annotation 写成最终真标注。
- 当前主线继续使用 known fixed camera intrinsics + known fixed camera pose。
- 若需要运行 Python，优先使用：

```bash
PYTHONNOUSERSITE=1 /home/grasp/miniconda3/envs/carlaAir/bin/python
```

- 若需要下载 Hugging Face 资产，先配置：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

- runtime artifacts 保持在 ignored 路径下。
- 决策记录、报告、blocker 写入 `research/reports/`。
- `research/reports/` 下旧报告已归入“按需追溯报告”；仅在追溯具体 artifact、命令、sha256 lineage 或 blocker 时按需读取，不默认全文阅读。
- 如果调整数据落点，优先尊重用户原意：原始采集和派生数据放在 `local/carla_air/`；`mvp-demo/scripts` 可作为工具继续消费显式路径。

## 推荐阶段

### 阶段 1：状态与契约核查

目标：

- 只读核查当前 capture、scene_dir、tracklets、points_by_timestamp、eval 输出。
- 确认哪些 scene 已 gate-pass，哪些已有 proxy sample，哪些还只是 skeleton。

输出：

- 当前可用 scene 列表；
- 每个 scene 状态：
  - raw capture
  - scene skeleton
  - proxy annotation
  - real annotation
  - proxy points
  - real points
  - eval result

禁止：

- 不修改文件。

### 阶段 2：确认真实 synthetic annotation 来源

目标：

只读检查 CARLA-Air / AirSim / CARLA Python API 当前能否导出以下任一真实标注来源：

- depth image
- semantic segmentation
- instance segmentation
- object / actor pose
- object / actor bounding box
- object mesh
- AirSim segmentation image
- CARLA depth / semantic camera blueprint

重点检查：

- CARLA Python API 是否能 spawn depth / semantic segmentation camera；
- AirSim 是否能返回 segmentation / depth；
- 当前 drone actor 是否能稳定拿到 actor id / transform；
- 是否能将 drone 设置为单独 instance / segmentation id；
- 当前 CARLA-Air runtime 是否需要重新启动；
- 是否有可复用 example 或 script。

输出报告到 `research/reports/`：

- 可用 annotation source；
- 不可用 source；
- blocker；
- 推荐实现路径；
- 证据路径；
- 明确说明不允许把 presence gate bbox 当正式 bbox。

### 阶段 3：替换 proxy annotation

触发条件：

- 阶段 2 确认至少一种可实现 annotation source。

目标：

实现最小真实或更真实 synthetic annotation 生成器。

输入：

- 一个已 gate-passed 的 `scene_dir`

输出：

- `cams/<cam>/masks_gt/*.png` 或更合适的 `masks_synth/*.png`
- `annotations/object_pose_by_timestamp.csv`
- `annotations/annotation_meta.json`
- `tracks/tracklets.json`
- 状态文件更新：
  - `pipeline_contract.json`
  - `tracks/tracklets_status.json`

要求：

- `annotation_meta.json` 必须写明 source，例如：
  - `carla_semantic_segmentation_camera`
  - `airsim_segmentation`
  - `carla_actor_pose_projected_mesh`
  - `carla_actor_pose_projected_bbox`
- 若只能拿到 pose 而不能拿到 pixel-accurate mask，不能声称 pixel-accurate。
- 状态必须区分：
  - `minimal_formal_proxy_exported`
  - `synthetic_annotation_exported`
  - `pixel_accurate_synthetic_annotation_exported`
- 不覆盖已有 proxy 结果，除非显式 `--overwrite`。

### 阶段 4：替换 proxy points_by_timestamp 为真 4D geometry

目标：

基于真实 annotation 或 masks 接入 NeoVerse / 等价 4D pipeline。

优先复用：

- `prepare_neoverse_multiview_manifest.py`
- `run_neoverse_per_camera_bundle.py`
- `export_neoverse_view_observations.py`
- `backproject_neoverse_observations.py`
- `fuse_neoverse_multiview_world_points.py`
- `constrain_neoverse_multiview_dynamic.py`

输出必须满足：

- `points_by_timestamp/index.csv`
- `points_by_timestamp/meta.json`
- `points_by_timestamp/*.npy`
- `meta.json.schema_version = neoverse_points_by_timestamp_v1`

要求：

- 不覆盖已有 proxy points；
- 新输出目录必须区分 proxy 和 real geometry；
- `pipeline_contract.json` 必须明确 points source；
- 如果 NeoVerse 跑不通，写 blocker report，不伪造真 geometry。

### 阶段 5：复跑最小 ReID 闭环

至少使用 POC03 node03/node04：

1. 生成真实 annotation。
2. 生成真实 points_by_timestamp。
3. 运行：
   - `build_node_tracklets.py`
   - `extract_node_track_embeddings.py`
   - `eval_node_track_retrieval.py`
4. 输出 eval JSON 到：
   - `mvp-demo/output/evals/`
