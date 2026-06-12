# CARLA-Air Dataset Generation Pipeline V1 Offline Closure 报告

## 结论

本次完成的是离线 `Dataset Generation Pipeline V1` 的最终 closure，目标边界已收敛为“离线 Dataset Generation Pipeline V1 完成”。这是一轮纯离线收束，不启动 runtime，不检查端口，不构建 UE，也不改变任何 UE/CARLA import/readback 状态。

这次 closure 证明的是离线产物、契约、清单与 verifier 已经收敛到可接受状态；它不把 full 6-identity live dataset、UE/CARLA import/readback、可信 `mask_gt`、formal annotation、NeoVerse 或 real 4D geometry 作为当前完成 blocker。

## 改动

- 新增报告：
  - `research/reports/carla_air_dataset_pipeline_offline_v1_closure_2026_05_31_zh.md`
- 更新计划：
  - `research/plans/ACTIVE_PLAN.md`
- 运行目录：
  - `local/carla_air/dataset_runs/dataset_v1_offline_v1_closure_main_review_20260531/`
- 验证输出：
  - `local/carla_air/tmp/dataset_v1_offline_v1_closure_main_review_verification_20260531.json`

## 本阶段新增能力

- 离线 V1 可以稳定生成 `dataset_plan`、`capture_queue`、`run_contract`、training index、deployment split / episodes、batch run manifest 与 artifact manifest。
- 固定入口已收敛为 `Town10HD` + `node01-node05`，并能表达 6 个 planned identities 的 `trajectory x node camera x identity` capture matrix / queue。
- existing-scene no-mask samples 可进入统一 index，且 `mask_gt.availability=unavailable` 不会阻塞离线索引。
- verifier 已覆盖核心 artifact、schema、split policy、sidecar quality、no-mask non-promotion、capture queue、scene/sample membership、dataset gap manifest 与 artifact accounting。

## 核心生成输出

本轮确认存在的核心 artifact 为：

- `dataset_plan.json`
- `capture_queue.jsonl`
- `run_contract.json`
- `dataset_manifest.json`
- `dataset_samples.jsonl`
- `dataset_splits.json`
- `deployment_episodes.json`
- `batch_run_manifest.json`
- `artifact_manifest.json`

补充证据：

- `dataset_gap_manifest` 存在，schema 为 `carla_air_dataset_gap_manifest_v1`
- `mask_gt_available_count=0`
- `no_mask_sample_count=2520`
- `sidecar_complete=60`
- `sidecar_incomplete=2460`
- missing `depth/semantic/instance=2460` each
- sample schema required fields 缺失计数均为 `0`，包括 `sample_id`、`scene_id`、`scene_key`、`identity_id`、`trajectory_id`、`node_id`、`camera_id`、`timestamp`、`split`、`rgb`、`pose`、`calib`、`depth`、`semantic`、`instance`、`mask_gt`
- `capture_queue` 处于 `blocked=540`，block reason 为 `await_ue_carla_import_readback_evidence`

## 验证结果

主 verifier 结果为：

- `ok=true`
- `failure_count=0`
- `warning_count=1`
- 唯一 warning 为 `identity_model_switch_mismatch_observed_scene_passthrough`

关键计数：

- `Town10HD`
- `node01-node05`
- `planned identity_count=6`
- `capture_task_count=540`
- `capture_queue_item_count=540`
- `scene_output_count=180`
- `sample_count=2520`
- `scene_count=7`
- split: `train=1440` / `val_in_domain=360` / `test_cross_layout=720`
- `artifact_count=25`
- `contract_artifact_count=26`

已核对的观察值：

- planned identities: `dji_drone_fbx_obj_local_poc`, `f22_fbx_obj_local_poc`, `j20_fbx_obj_local_poc`, `procedural_canard_uav_v1`, `procedural_delta_uav_v1`, `procedural_twinboom_uav_v1`
- observed identity: `default_airsim_drone`
- `identity_mismatch_count=2520`

## Non-Promotion 边界

本次明确不做以下上提：

- 不把 full 6-identity live dataset 视为当前完成项。
- 不把 UE/CARLA import/readback 视为已完成。
- 不把 `default_airsim_drone` 视为可信完成证据。
- 不把 `mask_gt_available_count=0` 的结果提升为可信 `mask_gt`。
- 不把 proxy / candidate / pseudo / legacy 的 `mask_gt`、actor-bbox、semantic-lidar 候选提升为可信 `mask_gt`、formal annotation 或 real 4D geometry。

`identity_model_switch_mismatch_observed_scene_passthrough` 只作为后续 live 阶段 blocker 记录，不阻止本次离线 V1 完成。

## 后续阶段

后续 live 6-identity dataset 仍需要 UE/CARLA import/readback evidence 才能继续推进；这些证据对 live 阶段重要，但不是当前离线 V1 完成的 blocker。

## 验证命令

```bash
python tools/carla_air/verify_dataset_generation_run_v1.py --run-dir local/carla_air/dataset_runs/dataset_v1_offline_v1_closure_main_review_20260531 --require-samples --require-run-contract --out local/carla_air/tmp/dataset_v1_offline_v1_closure_main_review_verification_20260531.json
git diff --check -- research/plans/ACTIVE_PLAN.md research/reports/carla_air_dataset_pipeline_offline_v1_closure_2026_05_31_zh.md
```

## 风险说明

- 当前 verifier 仍报告 1 条 warning，说明 observed scene passthrough 与 planned identity 之间存在不一致；它不影响离线 V1 关闭，但会进入后续 live 阶段审计。
- `capture_queue` 仍为全量 blocked，说明这轮产物是离线收束，不是 live 执行完成。
- `mask_gt_available_count=0` 和 sidecar 不完整状态都说明当前链路仍应维持 non-promotion 语义，不能向正式标注或真实几何前推。
