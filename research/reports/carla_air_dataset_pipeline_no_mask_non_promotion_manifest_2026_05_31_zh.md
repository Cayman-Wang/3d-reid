# CARLA-Air Dataset Generation Pipeline V1 No-Mask / Non-Promotion Manifest 报告

## 结论

本次完成的是离线 `Dataset Generation Pipeline V1` 的 standalone `no_mask_non_promotion_manifest.json` 硬化，schema 为 `carla_air_no_mask_non_promotion_manifest_v1`。这属于 V1 的 contract hardening，不涉及 runtime、ports、UE、CARLA 或 AirSim 启动。

## 改动

- 改动脚本：
  - `tools/carla_air/build_dataset_training_index_v1.py`
  - `tools/carla_air/run_dataset_generation_v1.py`
  - `tools/carla_air/verify_dataset_generation_run_v1.py`
- 新增 artifact：
  - `no_mask_non_promotion_manifest.json`
  - schema: `carla_air_no_mask_non_promotion_manifest_v1`
- 运行目录：
  - `local/carla_air/dataset_runs/dataset_v1_no_mask_non_promotion_manifest_main_review_20260531/`
- 验证输出：
  - `local/carla_air/tmp/dataset_v1_no_mask_non_promotion_manifest_main_review_verification_20260531.json`

## 验证

- verifier 结果：`ok=true`
- `failure_count=0`
- `warning_count=1`
- 唯一 warning：`identity_model_switch_mismatch_observed_scene_passthrough`
- 关键计数：
  - `sample_count=2520`
  - `scene_count=7`
  - `capture_task_count=540`
  - split：`train=1440 / val_in_domain=360 / test_cross_layout=720`
  - `mask_gt_available_count=0`
  - `no_mask_sample_count=2520`
  - `sidecar_complete=60/2520`
  - missing `depth/semantic/instance`：`2460` each
  - observed identity：`default_airsim_drone`
  - planned identities：`6`
  - `identity_mismatch_count=2520`
- 样本抽查：
  - `2520` rows
  - `available=0`
  - `unavailable=2520`
  - `is_mask_gt_true=0`
  - `missing_policy_flag=0`
  - `legacy_flag_missing=0`
- artifact accounting：
  - `artifact_count=20`
  - `contract_artifact_count=21`
  - self-reference gap 由 `artifact_manifest_json` 解释
  - `no_mask_non_promotion_manifest_json` 已进入 `run_contract.artifacts` 和 `artifact_manifest`
- 稳定 hashes：
  - `core_digest_sha256=e5cf131b44a3b3737bdfec7eded5f1809015c90b9e67c866b3e0e5679e716c4c`
  - `manifest_digest_sha256=a405b2591c610fe5d7c6ef4c537133fcaa98c6131f837f75e0254a6aaec2d9de`

## 策略标记

- `no_mask_samples_allowed_in_index=true`
- `mask_gt_availability_unavailable_is_not_candidate_proxy_pseudo_promotion=true`
- `proxy_candidate_pseudo_legacy_not_promoted_to_mask_gt=true`
- `legacy_masks_gt_directory_is_not_formal_mask_gt_without_evidence=true`
- `trusted_mask_gt_requires_explicit_formal_evidence=true`

## 守卫

- `starts_runtime=false`
- `writes_scene_outputs=false`
- `non_promotion=true`
- `full_v1_live_dataset_ready=false`

## 边界

- 不启动 runtime。
- 不检查端口。
- 不构建或配置 UE / CARLA。
- 不设置 `ue_import_verified` / `carla_import_verified`。
- 不宣称 full 6-identity live dataset complete。
- 不把 `default_airsim_drone` 视为可信目标身份完成证据。
- 不把 proxy / candidate / pseudo / legacy 的 `mask_gt`、actor-bbox、semantic-lidar 候选提升为可信 `mask_gt`、正式标注或真实 4D 几何。

## 风险

- 目前 warning 仍指向 observed scene passthrough identity mismatch，说明离线 index / manifest 已硬化，但 observed identity 与 planned identity 之间仍存在非一致性，需要继续保持 non-promotion 语义。
- `mask_gt_available_count=0` 仍意味着当前链路只能做 no-mask contract hardening，不能向 formal mask_gt 或真实几何前推。
