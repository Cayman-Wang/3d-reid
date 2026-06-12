# CARLA-Air Dataset Generation Pipeline V1 Sidecar Quality Standalone Artifact 报告

## 结论

当前主线仍是离线 `Dataset Generation Pipeline V1` 的 verifier / schema / index 工作；`UE/CARLA` import/readback 已外部化，完整 6-identity live dataset 仍处于阻塞状态。

本次完成的是 sidecar quality 的 standalone artifact 归档与校验，不涉及 runtime、ports、UE、CARLA、AirSim 启动。

## 改动

- 改动脚本：
  - `tools/carla_air/build_dataset_training_index_v1.py`
  - `tools/carla_air/run_dataset_generation_v1.py`
  - `tools/carla_air/verify_dataset_generation_run_v1.py`
- 新增 artifact：
  - `sidecar_quality_manifest.json`
  - schema: `carla_air_sidecar_quality_manifest_v1`
- 运行目录：
  - `local/carla_air/dataset_runs/dataset_v1_sidecar_quality_artifact_main_review_20260531/`
- 验证输出：
  - `local/carla_air/tmp/dataset_v1_sidecar_quality_artifact_main_review_verification_20260531.json`

## 验证

- verifier 结果：`ok=true`
- `failure_count=0`
- `warning_count=1`
- 唯一 warning：`identity_model_switch_mismatch_observed_scene_passthrough`
- 计数摘要：
  - `sample_count=2520`
  - `scene_count=7`
  - `capture_task_count=540`
  - split 分布：`train=1440 / val_in_domain=360 / test_cross_layout=720`
  - sidecar complete：`60/2520`
  - `complete_fraction=0.023809523809523808`
  - 缺失 `depth/semantic/instance`：`2460` each
  - `mask_gt_available_count=0`
  - `no_mask_sample_count=2520`
  - observed identity：`default_airsim_drone`
  - planned identity count：`6`
- artifact accounting：
  - `artifact_manifest artifact_count=18`
  - `run_contract contract artifact count=19`
  - 差异原因为 `artifact_manifest` 自引用
  - `sidecar_quality_manifest_json` 同时存在于 `run_contract` 与 `artifact_manifest`
- 稳定 hashes：
  - `overall`
  - `by_split`
  - `by_scene`
  - `by_scene_split`
  - `manifest_payload_digest_without_manifest_digest`

## 边界

- 不设置 `ue_import_verified` / `carla_import_verified`。
- 不宣称完整 6-identity live dataset 已完成。
- 不把 `default_airsim_drone` 视为目标身份完成证据。
- 不把 proxy / candidate / pseudo / legacy 的 `mask_gt`、actor-bbox、semantic-lidar 候选提升为受信 `mask_gt`、正式标注或真实 4D 几何。
