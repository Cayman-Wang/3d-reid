# CARLA-Air Dataset Pipeline V1 identity switch artifact bridge report

date: 2026-05-31

## 结论

本次只做了离线 V1 schema / index / verifier hardening，没有启动 runtime，没有构建或配置 UE，也没有写入 scene outputs。run 路径为 `local/carla_air/dataset_runs/dataset_v1_identity_switch_artifact_build_bridge_main_review_20260531/`，verifier 路径为 `local/carla_air/tmp/dataset_v1_identity_switch_artifact_build_bridge_main_review_verification_20260531.json`。

build/index 阶段现在已经产出 standalone `identity_model_switch_manifest.json`，并由 `run_contract` 显式声明；verifier 侧对 standalone manifest 与 `dataset_manifest.identity_model_switch_contract` 做了 cross-check。当前结果是 `ok=true`、`failure_count=0`、`warning_count=1`，warning 仅是 observed scene passthrough identity mismatch。

## 修改点

- 补齐了 V1 的 run contract / dataset manifest / identity model switch manifest 之间的引用与 accounting，确保 index 产物可独立追踪。
- 将 `identity_model_switch_manifest.json` 提升为独立产物，记录 planned identity、switch method、readback 状态与 blocker，不再只依赖嵌入式 contract。
- verifier 增加对 standalone manifest 与 embedded contract 的交叉核对，确认它们在 identity 计划、样本归属和 non-promotion 语义上保持一致。
- 保持 `starts_runtime=false`、`writes_scene_outputs=false`、`non_promotion=true`，因此这次只是离线 schema/index/verifier hardening。

## 新产物

- `local/carla_air/dataset_runs/dataset_v1_identity_switch_artifact_build_bridge_main_review_20260531/dataset_manifest.json`
- `local/carla_air/dataset_runs/dataset_v1_identity_switch_artifact_build_bridge_main_review_20260531/run_contract.json`
- `local/carla_air/dataset_runs/dataset_v1_identity_switch_artifact_build_bridge_main_review_20260531/identity_model_switch_manifest.json`
- `local/carla_air/tmp/dataset_v1_identity_switch_artifact_build_bridge_main_review_verification_20260531.json`

关键计数如下：

- `sample_count=2520`
- `scene_root_count=7` / `scene_count=7`
- `capture_task_count=540`
- `mask_gt_available_count=0`
- `identity_mismatch_count=2520`
- observed identity=`default_airsim_drone`
- `sidecar_complete=60/2520`
- missing `depth/semantic/instance=2460`

## 验证结果

verifier 结果为 `ok=true`、`failure_count=0`、`warning_count=1`。warning 的具体含义是：observed sample identity 与 planned identities 不一致，但该 passthrough 被保留用于 no-mask index，且 full live readiness 仍然 blocked。

run contract 也一致表明：`sample_count=2520`、`scene_count=7`、`split_distribution=train 1440 / val_in_domain 360 / test_cross_layout 720`、`capture_task_count=540`、`no_mask_sample_count=2520`、`sidecar_complete_count=60`、`mask_gt_available_count=0`。同时，`identity_model_switch_manifest.json` 继续要求 UE/CARLA import readback，且 `blocked_capture_task_count=540`、`full_v1_live_dataset_ready=false`。

## Non-promotion

- 不启动 runtime。
- 不构建、不配置 UE。
- 不设置 `ue_import_verified=true` 或 `carla_import_verified=true`。
- 不宣称 full 6-identity live dataset complete。
- 不把 `default_airsim_drone` 当作 planned identity 的完成替代。
- 不把 proxy / candidate / pseudo / legacy masks 当作可信 `mask_gt`、formal annotation 或 real 4D geometry。

## 剩余 blocker

- 6 个 planned identity 仍缺少真实 UE/CARLA import readback 证据，因此对应 capture tasks 仍 blocked。
- `mask_gt_available_count=0`，当前仍是 no-mask index 路径，不应晋升为 formal mask_gt pipeline。
- full V1 live dataset readiness 仍为 `false`，后续仍需要外部 UE/CARLA import/readback 证据来解除 blocker。
