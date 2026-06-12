# CARLA-Air Dataset Pipeline Scene-Sample Index Manifest Hardened

## 结论

本轮完成离线 Dataset Generation Pipeline V1 的 scene-sample index manifest 加固。当前主线仍是 verifier / schema / index work，UE/CARLA import/readback 已外部化，完整 6-identity live dataset 仍被阻塞。

本轮未进行任何 runtime、端口、UE、CARLA 或 AirSim 启动与 probe。

## 改动

- 改动脚本：`tools/carla_air/build_dataset_training_index_v1.py`、`tools/carla_air/run_dataset_generation_v1.py`、`tools/carla_air/verify_dataset_generation_run_v1.py`
- 新增 artifact：`scene_sample_index_manifest.json`
- schema：`carla_air_scene_sample_index_manifest_v1`
- 目标是把 `scene -> sample -> split/identity/camera/timestamp` membership 固化为一个可由 verifier 独立重算的 invariant
- run 路径：`local/carla_air/dataset_runs/dataset_v1_scene_sample_index_manifest_main_review_20260531/`

manifest 的 scene entry 覆盖字段包括：`scene_key`、`scene_id`、`scene_dir/source_scene_root`、`identity_id`、`trajectory_id`、`node_id`、`split_names`、`camera_ids`、`sample_count`、`timestamp_count`、`first/last sample_id`、`first/last timestamp_us`、`sample/timestamp hashes`、`mask/no-mask counts`、`sidecar_complete_count`。

## 验证

- verifier 输出：`local/carla_air/tmp/dataset_v1_scene_sample_index_manifest_main_review_verification_20260531.json`
- 结果：`ok=true`、`failure_count=0`、`warning_count=1`
- 唯一 warning：`identity_model_switch_mismatch_observed_scene_passthrough`
- 计数：`sample_count=2520`、`scene_count=7`、`capture_task_count=540`
- split 分布：`train=1440`、`val_in_domain=360`、`test_cross_layout=720`
- mask / sidecar 状态：`mask_gt_available_count=0`、`no_mask_sample_count=2520`、`sidecar_complete=60/2520`、`missing depth/semantic/instance=2460` each
- identity 状态：observed identity=`default_airsim_drone`、planned identity count=`6`
- 新 manifest summary：`artifact_count=19`、`contract_artifact_count=20`（self-reference 已解释）、`scene_split_membership_hash=6bebccfe5f586ca8ee7c43560e5074bd71f615f5d192083af0ea79dc772a04be`、`scene_sample_index_hash=24ba7e9b26e923e062dfff815c774fe64abbe3d8bb1b7b7431f2171f752ac6c2`、`scene_keys_sorted_hash=22e137d8ff006e7b9759228eb8a25f03b69a4b0d5638b66d027da2d0814bb35f`

## 边界

- 不设置 `ue_import_verified` / `carla_import_verified`
- 不宣称 full 6-identity live dataset complete
- 不把 `default_airsim_drone` 视为完整 6-identity live dataset
- 不把 proxy / candidate / pseudo / legacy `masks_gt`、actor-bbox candidate、semantic-lidar candidate 晋升为可信 `mask_gt`、formal annotation 或 real 4D geometry
- 本轮只完成离线 index manifest 加固与 verifier 可重算闭环，没有额外 runtime 证据
