# CARLA-Air Dataset Pipeline V1 Capture Queue Manifest 报告

## 结论

本轮完成的是离线 `Dataset Generation Pipeline V1` 的 standalone `capture_queue_manifest.json` 硬化，schema 为 `carla_air_capture_queue_manifest_v1`。这仍然是纯离线契约与清单，不启动 runtime，不检查端口，不构建 UE，也不改变任何 UE/CARLA import/readback 状态。

`capture_queue_manifest` 由 `dataset_manifest.outputs`、`run_contract.artifacts`、`batch_run_manifest.artifact_paths` 与 `artifact_manifest` 显式声明。当前结果只说明 capture queue 已被规范化、可审计、可阻断，不说明 live readiness 已完成。

## 证据位置

run 目录：

```text
local/carla_air/dataset_runs/dataset_v1_capture_queue_manifest_main_review_20260531/
```

verifier 输出：

```text
local/carla_air/tmp/dataset_v1_capture_queue_manifest_main_review_verification_20260531.json
```

## Verifier 结果

主 verifier 结果为：

- `ok=true`
- `failure_count=0`
- `warning_count=1`
- 唯一 warning 为 `identity_model_switch_mismatch_observed_scene_passthrough`

## 队列摘要

- `capture_queue_item_count=540`
- `blocked_capture_queue_item_count=540`
- `queued_capture_queue_item_count=0`
- `state_counts={blocked:540}`
- `block_reason_counts={await_ue_carla_import_readback_evidence:540}`
- `scene_group_count=180`
- `capture_task_count=540`
- `sample_count=2520`
- `scene_count=7`
- split: `train=1440` / `val_in_domain=360` / `test_cross_layout=720`
- `mask_gt_available_count=0`
- `no_mask_sample_count=2520`
- `sidecar_complete=60/2520`
- missing `depth/semantic/instance=2460` each
- observed identity=`default_airsim_drone`
- planned identities=`6`
- `identity_mismatch_count=2520`

## 关键 Hash 与 计数

- `capture_task_id_order_sha256=33e73472a65f50b12da14e4e28bd8bda2a3df566633777440e59a4cec78d3b77`
- `expected_scene_root_order_sha256=fe40470dfe7d96a174bd82c4bc4def36af75754437ebe20a41f808af04f444ae`
- `artifact_count=24`
- `contract_artifact_count=25`
- self-reference gap 由 `artifact_manifest_json` 不纳入自身 hashed artifact entries 解释

## 边界声明

- 不启动 runtime。
- 不检查端口。
- 不构建或配置 UE / CARLA。
- 不设置 `ue_import_verified` / `carla_import_verified`。
- 不宣称 full 6-identity live dataset complete。
- 不把 `default_airsim_drone` 视为可信完成证据。
- 不把 proxy / candidate / pseudo / legacy 的 `mask_gt`、actor-bbox、semantic-lidar 候选提升为可信 `mask_gt`、formal annotation 或 real 4D geometry。

## 验证建议

```bash
python -m py_compile tools/carla_air/run_dataset_generation_v1.py tools/carla_air/verify_dataset_generation_run_v1.py
python tools/carla_air/verify_dataset_generation_run_v1.py --run-dir local/carla_air/dataset_runs/dataset_v1_capture_queue_manifest_main_review_20260531 --require-samples --require-run-contract --out local/carla_air/tmp/dataset_v1_capture_queue_manifest_main_review_verification_20260531.json
git diff --check -- research/plans/ACTIVE_PLAN.md research/reports/carla_air_dataset_pipeline_capture_queue_manifest_2026_05_31_zh.md
```

## 风险点

- 当前 540 个 queue item 全部 `blocked`，blocker 仍是 `await_ue_carla_import_readback_evidence`，因此这份 manifest 只能证明离线阻断契约，不代表 live 采集已可执行。
- `identity_model_switch_mismatch_observed_scene_passthrough` 仍在，说明 observed identity 与 planned identities 不一致，不能把 passthrough 结果上提为正式 live dataset 完成。
- `mask_gt_available_count=0` 且 sidecar 仅 `60/2520` 完整，当前链路仍不支持把 no-mask / proxy / candidate / pseudo / legacy 结果当作可信正式标注。
