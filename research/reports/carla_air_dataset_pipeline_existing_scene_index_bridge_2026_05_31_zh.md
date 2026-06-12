# CARLA-Air Dataset Pipeline V1 existing scene index bridge 报告

date: 2026-05-31

## 结论

本次完成离线 existing scene index bridge 的加固与验收记录，未启动 runtime，未构建或配置 UE，也未写入 scene outputs。run 路径为 `local/carla_air/dataset_runs/dataset_v1_existing_scene_index_bridge_main_review_20260531/`，verifier 路径为 `local/carla_air/tmp/dataset_v1_existing_scene_index_bridge_main_review_verification_20260531.json`。

本次新增 `existing_scene_index_bridge_manifest.json`，schema 为 `carla_air_existing_scene_index_bridge_manifest_v1`。它把 existing scene root / index reuse 从原先隐式汇总，提升为独立可验收的 artifact / invariant：bridge 的 scene 发现、索引归属、split membership、sample accounting 和 non-promotion 现在都可以由单独产物直接核对，而不是只依赖嵌入在 `dataset_manifest` / `run_contract` 里的汇总结论。

## 修改点

- 新增独立 bridge manifest，用于显式记录 existing scene root/index reuse 的输入范围、scene 计数、样本计数和 split membership hash。
- 将 bridge 产物纳入 artifact accounting，并在 `run_contract` 中显式声明，避免 existing scene reuse 只作为隐式派生结果存在。
- 维持 `starts_runtime=false`、`writes_scene_outputs=false`、`non_promotion=true`，因此这次仅是离线索引桥接与验收硬化。

## 新产物

- `local/carla_air/dataset_runs/dataset_v1_existing_scene_index_bridge_main_review_20260531/existing_scene_index_bridge_manifest.json`
- `local/carla_air/dataset_runs/dataset_v1_existing_scene_index_bridge_main_review_20260531/run_contract.json`
- `local/carla_air/dataset_runs/dataset_v1_existing_scene_index_bridge_main_review_20260531/artifact_manifest.json`
- `local/carla_air/tmp/dataset_v1_existing_scene_index_bridge_main_review_verification_20260531.json`

关键 bridge 计数如下：

- `scene_root_count=7`
- `indexed_scene_count=7`
- `sample_count=2520`
- `mask_gt_available_count=0`
- `no_mask_sample_count=2520`
- `scene_entries=7`
- `scene_split_membership_hash=6bebccfe5f586ca8ee7c43560e5074bd71f615f5d192083af0ea79dc772a04be`

## 验证结果

verifier 结果为 `ok=true`、`failure_count=0`、`warning_count=1`。warning 的内容是 expected observed scene passthrough identity mismatch，也就是 observed sample identity 与 planned identities 不一致，但该 passthrough 仍保留给 no-mask index，full live readiness 仍 blocked。

artifact accounting 结果为 `artifact_count=17`、`contract_artifact_count=18`，且 `existing_scene_index_bridge_manifest` 的状态满足 `present=true` / `contract_declared=true` / `contract_required=true`。这说明 bridge manifest 已经被纳入 run contract 的正式核算，而不是游离在外的附属文件。

此外，保持的关键事实仍是：

- `capture_task_count=540`
- `identity_mismatch_count=2520`
- observed identity=`default_airsim_drone`
- `sidecar_complete=60/2520`
- missing `depth/semantic/instance=2460`

## Non-promotion

- 不启动 runtime。
- 不构建、不配置 UE。
- 不设置 `ue_import_verified=true` 或 `carla_import_verified=true`。
- 不宣称 full 6-identity live dataset complete。
- 不把 `default_airsim_drone` 当作 planned identity 的完成替代。
- 不把 `proxy` / `candidate` / `pseudo` / `legacy` masks 当作可信 `mask_gt`、formal annotation 或 real 4D geometry。

## 剩余 blocker

- 6 个 planned identity 仍缺少真实 UE/CARLA import readback 证据，因此对应 capture tasks 仍 blocked。
- `mask_gt_available_count=0`，当前仍是 no-mask index 路径，不应晋升为 formal `mask_gt` pipeline。
- full V1 live dataset readiness 仍为 `false`，后续仍需要外部 UE/CARLA import/readback 证据来解除 blocker。
