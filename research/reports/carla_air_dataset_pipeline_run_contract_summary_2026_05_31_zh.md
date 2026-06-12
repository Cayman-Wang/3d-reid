# CARLA-Air Dataset Pipeline V1 Run Contract Summary

date: 2026-05-31

## 结论

本轮为离线 `Dataset Generation Pipeline V1` 增加 `dataset_run_contract_summary`，把 run/index 的关键语义统计显式写入并由 verifier 交叉校验。该字段现在同时出现在 `dataset_manifest.json`、`dataset_index_manifest.json`、`run_contract.json`、`batch_run_manifest.json`，用于证明这些 artifact 描述的是同一批样本、同一套 split、同一组 scene、同一批 capture task 和同一组 non-promotion 边界。

本轮没有启动 runtime，没有检查或启动 CARLA-Air / AirSim 端口，没有执行 UE / Unreal 构建或 import，也没有把任何 proxy / candidate / pseudo / legacy mask 提升为可信 `mask_gt`。

## 改动范围

- `tools/carla_air/build_dataset_training_index_v1.py`
  - 新增 `dataset_run_contract_summary` 生成逻辑。
  - 写入 `dataset_manifest.json` 与 `dataset_index_manifest.json`。
- `tools/carla_air/run_dataset_generation_v1.py`
  - 从 index/manifest 读取 `dataset_run_contract_summary`。
  - 写入 `run_contract.json` 与 `batch_run_manifest.json`。
- `tools/carla_air/verify_dataset_generation_run_v1.py`
  - 从四个 artifact 读取 summary。
  - 全缺失时 legacy-compatible warning；部分存在或跨 artifact 不一致时失败。
  - 校验 summary 与 verifier 重算的 sample、split、scene、capture task、sidecar、mask、identity 和 offline guard 事实一致。

## 新 run

```text
local/carla_air/dataset_runs/dataset_v1_run_contract_summary_main_review_20260531/
```

主 verifier 输出：

```text
local/carla_air/tmp/dataset_v1_run_contract_summary_main_review_verification_20260531.json
```

兼容验证输出：

```text
local/carla_air/tmp/dataset_v1_split_policy_digest_compat_after_run_contract_summary_20260531.json
```

## 验证结果

主 verifier：

- `ok=true`
- `failure_count=0`
- `warning_count=1`
- warning 仍为 `identity_model_switch_mismatch_observed_scene_passthrough`
- `sample_count=2520`
- `split_distribution=train=1440, val_in_domain=360, test_cross_layout=720`
- `dataset_run_contract_summary.observed_source_count=4`
- `dataset_run_contract_summary.digest=591d4dae64beea8f5ee0ab13f43a86c10295a8518a6bdd820e634b5cc77907e7`

四个 summary source digest 一致：

- `dataset_manifest`
- `dataset_index_manifest`
- `run_contract`
- `batch_run_manifest`

兼容验证：

- 旧 run `dataset_v1_split_policy_digest_main_review_20260531` 仍 `ok=true`
- `failure_count=0`
- 新增 `dataset_run_contract_summary_missing_legacy_compatible` warning，说明旧 artifact 缺该字段时不会被误判为坏 run

## Summary 字段事实

新 summary 记录：

- `scene_count=7`
- `capture_task_count=540`
- `sample_with_capture_task_count=0`
- `sample_without_capture_task_count=2520`
- `strict_matrix_entry_sample_count=0`
- `legacy_or_observed_scene_passthrough_count=2520`
- `mask_gt_available_count=0`
- `no_mask_sample_count=2520`
- `sidecar_complete_count=60`
- `sidecar_complete_fraction=0.023809523809523808`
- `sidecar_missing_count_by_modality.depth=2460`
- `sidecar_missing_count_by_modality.semantic=2460`
- `sidecar_missing_count_by_modality.instance=2460`
- `planned_identity_ids` 为 6 个 V1 identity
- `observed_identity_ids=["default_airsim_drone"]`
- `identity_mismatch_count=2520`
- `strict_planned_identity_sample_count=0`

## Non-promotion 边界

summary 与 verifier 均保持：

- `starts_runtime=false`
- `writes_scene_outputs=false`
- `non_promotion=true`
- `full_v1_live_dataset_ready=false`

因此，本轮只增强离线 batch run/index 的语义一致性闭环；它不证明 full 6-identity live dataset 完成，不设置 `ue_import_verified=true` 或 `carla_import_verified=true`，也不把 `default_airsim_drone` 旧 scene、proxy/candidate/pseudo/legacy masks、actor-bbox candidate 或 semantic-lidar actor-relative points 当作可信 `mask_gt`、formal annotation 或 real 4D geometry。
