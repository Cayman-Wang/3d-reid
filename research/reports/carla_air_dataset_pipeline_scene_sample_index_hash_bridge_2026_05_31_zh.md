# CARLA-Air Dataset Pipeline Scene-Sample Index Hash Bridge

## 结论

本轮完成离线 Dataset Generation Pipeline V1 的 cross-manifest hash bridge：`dataset_index_manifest` 已写出 `scene_sample_index_hash`，verifier 也对 `dataset_index_manifest` 与 `scene_sample_index_manifest` 的 `scene_sample_index_hash` / `scene_count` / `scene_keys_sorted_hash` 做了交叉核对。

本轮未启动 runtime，未检查端口，未启动或构建 UE / CARLA，也未做任何 live probe。

## 变更

- run 路径：`local/carla_air/dataset_runs/dataset_v1_scene_sample_index_hash_bridge_main_review_20260531/`
- verifier 路径：`local/carla_air/tmp/dataset_v1_scene_sample_index_hash_bridge_main_review_verification_20260531.json`
- bridge 目标：把 `dataset_index_manifest.scene_sample_index_hash` 与 `scene_sample_index_manifest.scene_sample_index_hash` 统一到同一离线证据链，并让 verifier 复核两份 manifest 的 scene 级哈希、scene 数量与排序后的 scene keys 哈希一致性
- self-reference gap：`artifact_count=19`、`contract_artifact_count=20`，预期由 `artifact_manifest_json` 解释

## 验证

- verifier 结果：`ok=true`、`failure_count=0`、`warning_count=1`
- 唯一 warning：`identity_model_switch_mismatch_observed_scene_passthrough`
- 核心计数：`sample_count=2520`、`scene_count=7`、`capture_task_count=540`
- split 分布：`train=1440`、`val_in_domain=360`、`test_cross_layout=720`
- mask / sidecar 状态：`mask_gt_available_count=0`、`no_mask_sample_count=2520`、`sidecar_complete=60/2520`、missing depth/semantic/instance=`2460` each
- identity 状态：observed identity=`default_airsim_drone`、planned identities=`6`、`identity_mismatch_count=2520`
- hashes：`dataset_index_manifest.scene_sample_index_hash=24ba7e9b26e923e062dfff815c774fe64abbe3d8bb1b7b7431f2171f752ac6c2`、`scene_sample_index_manifest.scene_sample_index_hash=24ba7e9b26e923e062dfff815c774fe64abbe3d8bb1b7b7431f2171f752ac6c2`、`scene_split_membership_hash=6bebccfe5f586ca8ee7c43560e5074bd71f615f5d192083af0ea79dc772a04be`、`scene_keys_sorted_hash=22e137d8ff006e7b9759228eb8a25f03b69a4b0d5638b66d027da2d0814bb35f`

## 边界

- 不设置 `ue_import_verified` / `carla_import_verified`
- 不宣称 full 6-identity live dataset complete
- 不把 `default_airsim_drone` 视为完整 6-identity live dataset
- 不把 proxy / candidate / pseudo / legacy `mask_gt`、actor-bbox candidate、semantic-lidar candidate 晋升为可信 `mask_gt`、formal annotation 或 real 4D geometry
- 本轮仅记录离线 V1 hash bridge 的 manifest 交叉核对结果，没有额外 runtime 证据
