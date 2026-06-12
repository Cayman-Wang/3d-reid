# CARLA-Air Dataset Pipeline V1 sample schema coverage manifest 结果

date: 2026-05-31

## 结论

本轮在离线 `Dataset Generation Pipeline V1` 上新增并硬化了独立的 `sample_schema_coverage_manifest.json`，schema 为 `carla_air_sample_schema_coverage_manifest_v1`。该 artifact 把样本 required fields 覆盖情况从 `dataset_manifest.sample_schema_coverage_summary` 扩展为可由 verifier 单独检查的 run artifact。

本轮不启动 CARLA/AirSim runtime，不检查端口，不构建 UE/CARLA，也不改变任何 UE/CARLA import/readback 状态。

## 证据位置

run 目录：

```text
local/carla_air/dataset_runs/dataset_v1_sample_schema_coverage_manifest_main_review_20260531/
```

verifier 输出：

```text
local/carla_air/tmp/dataset_v1_sample_schema_coverage_manifest_main_review_verification_20260531.json
```

standalone manifest：

```text
local/carla_air/dataset_runs/dataset_v1_sample_schema_coverage_manifest_main_review_20260531/sample_schema_coverage_manifest.json
```

## Verifier 结果

主 verifier 结果为：

- `ok=true`
- `failure_count=0`
- `warning_count=1`
- 唯一 warning 为 expected `identity_model_switch_mismatch_observed_scene_passthrough`

## 新增能力

- `sample_schema_coverage_manifest.json` 由 build/index 阶段写出，并由 `dataset_manifest.outputs`、`run_contract.artifacts`、`batch_run_manifest.artifact_paths` 与 `artifact_manifest` 显式声明。
- verifier 对新 run 执行 hard-check：schema、run_id、offline guard flags、required fields、field present/missing counts、stable hash，以及 standalone manifest 与 `dataset_manifest.sample_schema_coverage_summary` / verifier recomputed summary 的一致性。
- 旧 run 未声明该 artifact 时保持 legacy-compatible warning；声明后缺失或不一致会失败。

## 关键计数

- `sample_count=2520`
- `scene_count=7`
- `capture_task_count=540`
- split: `train=1440` / `val_in_domain=360` / `test_cross_layout=720`
- `mask_gt_available_count=0`
- `no_mask_sample_count=2520`
- sidecar complete: `60/2520`
- missing `depth/semantic/instance=2460` each
- observed identity=`default_airsim_drone`
- planned identities=`6`
- `identity_mismatch_count=2520`

sample schema coverage 的 required fields 全部为 `field_missing_count=0`：

```text
sample_id, scene_id, scene_key, identity_id, trajectory_id, node_id,
camera_id, timestamp, split, rgb, pose, calib, depth, semantic,
instance, mask_gt
```

注意：`depth` / `semantic` / `instance` 字段存在不表示对应 sidecar 完整；当前仍有 2460 个样本缺对应 sidecar。该 manifest 只证明字段位和 unavailable policy 可审计，不证明真实 `mask_gt` 或 formal annotation。

## Artifact Accounting

- `artifact_count=23`
- `contract_artifact_count=24`
- self-reference gap 仍由 `artifact_manifest_json` 被排除在自身 hashed artifact entries 之外解释

## Stable Hash

```text
canonical_payload_sha256=9979c0a96171560d3c3c88caade4be9f3797aa0331debf81e6486f3b539b422a
```

## 边界声明

- 不启动 runtime
- 不检查端口
- 不构建 UE/CARLA
- 不设置 `ue_import_verified=true`
- 不设置 `carla_import_verified=true`
- 不宣称 full 6-identity live dataset complete
- 不把 `default_airsim_drone`、proxy/candidate/pseudo/legacy masks、actor-bbox、semantic-lidar actor-relative points 提升为可信 `mask_gt`、formal annotation 或 real 4D geometry

## 验证命令

```bash
python -m py_compile tools/carla_air/build_dataset_training_index_v1.py tools/carla_air/run_dataset_generation_v1.py tools/carla_air/verify_dataset_generation_run_v1.py
python tools/carla_air/run_dataset_generation_v1.py --run-id dataset_v1_sample_schema_coverage_manifest_main_review_20260531 --allow-fail
python tools/carla_air/verify_dataset_generation_run_v1.py --run-dir local/carla_air/dataset_runs/dataset_v1_sample_schema_coverage_manifest_main_review_20260531 --require-samples --require-run-contract --out local/carla_air/tmp/dataset_v1_sample_schema_coverage_manifest_main_review_verification_20260531.json
```
