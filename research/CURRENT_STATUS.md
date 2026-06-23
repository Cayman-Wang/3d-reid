# CARLA-Air / 4D-ReID Current Status

Last updated: 2026-06-12

2026-06-23 note: DGGT + MapAnything is now planned as a diagnostic sidecar path for CARLA-Air 4D reconstruction optimization. The intended sequence is sidecar POC -> unified geometry contract -> trajectory recovery comparison -> 3D/4D-ReID smoke, with outputs under `local/carla_air/geometry_4d/<capture_id>/<method>/` for `dggt`, `mapanything`, and `dggt_mapanything_aligned`. This does not replace the current triangulation/oracle baselines and does not change `real_4d_geometry_ready=false`, `formal_neoverse_ready=false`, `formal_annotation_ready=false`, or any benchmark-promotion gate.

## Read this first

This file is the default entry point for future CARLA-Air / 4D-ReID work. It is a soft archive index: older reports keep their original paths for provenance, but future agents should not read the full historical report set by default.

Future agents should default-read `docs/goal.md` first, then this file and `research/plans/ACTIVE_PLAN.md`.

Default core reading set:

```text
docs/goal.md
research/CURRENT_STATUS.md
research/plans/ACTIVE_PLAN.md
research/guides/carla_air_runtime_self_start_zh.md
research/handoffs/carla_air_ground_to_air_collection_handoff_zh.md
research/reports/carla_air_actor_to_pixel_contract_downstream_gate_integration_2026_05_29_zh.md
```

Read older reports only when tracing a specific lineage, command, output path, or blocker. Useful on-demand groups:

- Runtime/live probe lineage: `carla_air_runtime_*`, `carla_air_fg_opengl_low_*`, `carla_air_guarded_live_*`.
- Target-selection and annotation lineage: `carla_air_scene_status_*`, `carla_air_target_selection_*`, `carla_air_formal_annotation_*`.
- Geometry lineage: `carla_air_neoverse_*`, `carla_air_semantic_lidar_*`, `carla_air_world_semantic_lidar_*`.
- Multi-identity lineage: `carla_air_identity_*`, `carla_air_aircraft_identity_*`, `carla_air_procedural_*`, `carla_air_ue_*`.
- On-demand current lineage reports:
  - `research/reports/carla_air_target_traceability_gap_audit_2026_05_29_zh.md`
  - `research/reports/carla_air_actor_to_pixel_evidence_contract_2026_05_29_zh.md`
  - `research/reports/carla_air_world_semantic_lidar_geometry_audit_2026_05_29_zh.md`
  - `research/reports/carla_air_guarded_live_calibration_runner_2026_05_29_zh.md`
  - `research/reports/carla_air_post_guarded_traceability_audits_2026_05_29_zh.md`
  - `research/reports/carla_air_existing_actor_roi_sweep_target_readiness_integration_2026_05_29_zh.md`
  - `research/reports/carla_air_guarded_existing_actor_sweep_passthrough_2026_05_29_zh.md`
  - `research/reports/carla_air_multi_identity_local_poc_current_gate_2026_05_29_zh.md`
  - `research/reports/carla_air_local_poc_multi_identity_guard_2026_05_29_zh.md`
  - `research/reports/carla_air_private_model_policy_split_2026_05_30_zh.md`

## Current formal status

The long task is still active and incomplete.

Active stage reset: CARLA-Air Dataset Generation Pipeline V1.

Current stage facts:

- Initial config is fixed to Town10HD + `node01-node05`.
- Offline Dataset Generation Pipeline V1 is closed as a minimal offline loop as of 2026-05-31: planner / runner / training index builder / verifier / trajectory config and the selected offline evidence reports are present, and the closure verifier records `ok=true`, `failure_count=0`.
- The only expected offline-closure warning is `identity_model_switch_mismatch_observed_scene_passthrough`; it is a future live 6-identity blocker, not a failure of the offline V1 loop.
- V1 targets 6 normalized identities, but full live 6-identity dataset generation still requires true UE/CARLA import and runtime readback first.
- Final output should be a unified training index with a deployment-oriented split for the future two-camera-node deployment.
- `mask_gt` availability has been carried through the offline no-mask contract: current offline V1 evidence keeps `mask_gt_available_count=0`, `no_mask_sample_count=2520`, and no sample may set `is_mask_gt=true`.
- Proxy / candidate / pseudo masks must not be labeled `mask_gt`.
- Historical blockers remain unchanged: proxy annotation / proxy points are not replaced yet, `target_selection_ready=false`, `formal_neoverse_ready=false`, formal annotation is not ready, NeoVerse / real 4D geometry is not complete, and UE/CARLA import/readback is still missing.

Current gates:

| Route | Current status |
| --- | --- |
| Real synthetic annotation replacement | blocked |
| True / final 4D geometry or NeoVerse reconstruction | blocked |
| Formal multi-identity benchmark | blocked |
| Weak bbox + semantic-lidar diagnostic route | guarded diagnostic only |
| Local technical multi-identity POC | ready for isolated local POC only |
| Private local multi-identity benchmark roster | ready for private local use only |

Do not claim completion from historical smoke outputs. Do not treat the 2026-05-31 offline closure as completion of live 6-identity capture, trusted `mask_gt`, formal annotation, UE/CARLA import/readback, NeoVerse, or real 4D geometry.

Fresh current gate evidence:

- `research/reports/carla_air_dataset_pipeline_offline_v1_closure_2026_05_31_zh.md` records the current offline V1 closure: `local/carla_air/dataset_runs/dataset_v1_offline_v1_closure_main_review_20260531/` has the core offline artifacts, and `local/carla_air/tmp/dataset_v1_offline_v1_closure_main_review_verification_20260531.json` is `ok=true`, `failure_count=0`, `warning_count=1`; the warning is the expected observed-scene identity passthrough mismatch. This closes only the offline data-generation loop.
- `research/reports/carla_air_dataset_pipeline_no_mask_non_promotion_manifest_2026_05_31_zh.md` records the no-mask contract: `sample_count=2520`, `mask_gt_available_count=0`, `no_mask_sample_count=2520`, `is_mask_gt_true=0`, and no legacy / proxy / candidate / pseudo mask is promoted as trusted `mask_gt`.
- `research/reports/carla_air_dataset_pipeline_capture_queue_manifest_2026_05_31_zh.md` records `capture_queue_item_count=540`, all blocked with `await_ue_carla_import_readback_evidence`; this keeps full live 6-identity capture outside the offline closure.
- `research/reports/carla_air_dataset_pipeline_run_contract_summary_2026_05_31_zh.md` and `research/reports/carla_air_dataset_pipeline_sample_schema_coverage_manifest_2026_05_31_zh.md` record the run-contract / sample-schema coverage artifacts for the offline loop. Required sample fields are present, but field presence does not imply sidecar completeness or trusted `mask_gt`.
- `research/reports/carla_air_dataset_generation_pipeline_v1_plan_index_scaffold_2026_05_30_zh.md` records the V1 plan/index scaffold: the no-scene-root path remains `plan_only=true`, `sample_count=0`, split names `train` / `val_in_domain` / `test_cross_layout`, and `deployment_episode_count=4`; no live runtime was started and no pseudo / candidate / proxy was promoted as `mask_gt`.
- `research/reports/carla_air_dataset_pipeline_mask_gt_decoupling_2026_05_30_zh.md` records the follow-up implementation: `tools/carla_air/build_dataset_training_index_v1.py` can now materialize an existing-scene no-mask index. With the node04 POC03 scene root, `local/carla_air/dataset_runs/main_review_existing_scene_after_mask_decouple/dataset_v1_main_review_20260530/dataset_manifest.json` is `ok=true`, `plan_only=false`, `sample_count=360`, while `mask_gt_availability_summary.available_count=0` and every sample keeps `mask_gt.availability=unavailable` / `is_mask_gt=false`. This is an index/materialization smoke only; full 6-identity live capture still requires UE/CARLA import/readback evidence.
- `research/reports/carla_air_target_traceability_gap_audit_2026_05_29_zh.md` reports `ok=true`, `missing_inputs=0`, `same_frame_world_geometry_evidence=true`, `actor_to_pixel_traceability_proven=false`, `formal_annotation_replacement_allowed=false`, `formal_points_replacement_allowed=false`, `can_replace_proxy_annotation_or_points_now=false`, `goal_complete=false`.
- `research/reports/carla_air_actor_to_pixel_evidence_contract_2026_05_29_zh.md` reports `ok=true`, `target_selection_ready=false`, `trusted_target_id_or_tag_available=false`, `ready_for_formal_conversion=false`, `formalization_ready=false`, `goal_complete=false`, `required_actor_to_pixel_evidence.ready_for_target_selection=false`, `required_actor_to_pixel_evidence.satisfied=false`, and `can_replace_proxy_annotation_or_points_now=false`.
- `research/reports/carla_air_world_semantic_lidar_geometry_audit_2026_05_29_zh.md` reports `pair_count=8`, `shared_timestamp_total_across_pairs=60`, all compared roots are `carla_world_xyz`, and `coordinate_frames_match=true`; this is diagnostic only.
- `research/reports/carla_air_guarded_live_calibration_runner_2026_05_29_zh.md` reports `diagnostic_only=true`, `starts_runtime=false`, `stops_runtime=false`, `gate_passed=false`, `runtime_stable_for_window=false`, `all_required_apis_responsive=false`, and probes skipped; no-runtime review only.
- `research/reports/carla_air_post_guarded_traceability_audits_2026_05_29_zh.md` reports that post-guarded audits now pass guarded-live runtime audit paths into target-selection readiness, while preserving `can_replace_proxy_annotation_or_points_now=false` and `goal_complete=false`.
- `research/reports/carla_air_existing_actor_roi_sweep_target_readiness_integration_2026_05_29_zh.md` reports that existing actor `24` ROI sweep evidence is now consumed by target-selection readiness; 6 views / 12 captures still show no actor id `24` in projected ROI under packed or single-channel decodes.
- `research/reports/carla_air_existing_actor_roi_decode_summary_integration_2026_05_29_zh.md` reports that `tools/carla_air/summarize_existing_actor_roi_decodes.py` now feeds a diagnostic-only ROI decode summary into target readiness and the post-guarded wrapper; `actor_id=24`, `observation_count=12`, `capture_ok_count=12`, `decode_count=10`, and `any_decode_roi_actor_id_positive=false` still leave `target_selection_ready=false`, `formalization_ready=false`, `can_replace_proxy_annotation_or_points_now=false`, and `goal_complete=false`.
- `research/reports/carla_air_guarded_existing_actor_sweep_passthrough_2026_05_29_zh.md` reports that the guarded runner now accepts existing-actor sweep passthrough args and the post-guarded wrapper routes `existing_actor` reports to `--existing-actor-roi-sweep`; this is still no-runtime smoke only, with `can_replace_proxy_annotation_or_points_now=false` and `goal_complete=false`.
- `research/reports/carla_air_traceability_and_ue_import_guard_hardening_2026_05_29_zh.md` reports that `audit_target_traceability_gap.py` now accepts `--existing-actor-roi-decode-summary` and records a compact `evidence_summary`, while `verify_aircraft_identity_ue_import_smoke_plan.py` now enforces identity-scoped planned_report paths and exact `build_status.json` / patch alignment; `local/carla_air/tmp/target_traceability_gap_existing_actor_decode_summary_main_review_20260529.json` and `local/carla_air/tmp/aircraft_identity_ue_import_smoke_plan_verification_path_contract_20260529.json` both keep `can_replace_proxy_annotation_or_points_now=false` / `formal_benchmark_ready=false` / `goal_complete=false`.
- `research/reports/carla_air_synthetic_annotation_evidence_contract_verifier_2026_05_29_zh.md` reports that `tools/carla_air/verify_synthetic_annotation_evidence_contract.py` is a read-only future evidence contract verifier. `local/carla_air/tmp/synthetic_annotation_evidence_contract_main_review_blocked_20260529.json` remains `ok=false`, `formal_synthetic_annotation_ready=false`, `can_replace_proxy_annotation_now=false`, `goal_complete=false`, with blocker `manifest_unavailable`; the strict positive fixture stays fixture-only and the candidate / research-manifest negatives are rejected. This does not replace proxy annotation, does not create real synthetic annotation, does not write scene outputs/contracts, does not run CARLA / AirSim / UE / NeoVerse, and does not complete benchmark.
- `research/reports/carla_air_objective_completion_audit_2026_05_29_zh.md` reports that `tools/carla_air/verify_carla_air_objective_completion.py` is a read-only objective-completion auditor. `local/carla_air/tmp/carla_air_objective_completion_audit_blocked_20260529.json` is `ok=true` only as audit integrity, with `objective_complete=false`, `goal_complete=false`, all five requirements `false`, and `non_promotion_verified=true`; `--require-complete` returns `ok=false` with blocker `objective_incomplete_required`. This does not start runtime, CARLA / AirSim / UE / NeoVerse, and does not write scenes, assets, contracts, or promotion evidence.
- `research/reports/carla_air_traceability_gap_default_decode_summary_gate_2026_05_29_zh.md` reports that `tools/carla_air/audit_target_traceability_gap.py` now defaults to requiring `local/carla_air/tmp/existing_actor_roi_decode_summary_main_review_20260529.json`; the positive default output `local/carla_air/tmp/target_traceability_gap_default_decode_summary_required_main_review_20260529.json` is `ok=true` but still has `can_replace_proxy_annotation_or_points_now=false` / `goal_complete=false` with blocker `existing_actor_roi_decode_summary_zero_positive_actor_id_decode_hits`, while the missing-summary negative output `local/carla_air/tmp/target_traceability_gap_missing_decode_summary_negative_main_review_20260529.json` is `ok=false` with blocker `existing_actor_roi_decode_summary_missing`. This is gate hardening only and does not放行 proxy replacement / real geometry / benchmark.
- `research/reports/carla_air_proxy_replaceability_final_geometry_gate_integration_2026_05_29_zh.md` reports that `tools/carla_air/audit_target_traceability_gap.py` now reads `--final-geometry-gate` and requires `formal_final_geometry_ready=true` before `formal_points_replacement_allowed` can become true. The default output `local/carla_air/tmp/target_traceability_gap_final_geometry_gate_main_review_20260529.json` is `ok=true`, `formal_annotation_replacement_allowed=false`, `formal_points_replacement_allowed=false`, `can_replace_proxy_annotation_or_points_now=false`, `goal_complete=false`, with blocker `final_geometry_gate_not_ready`; a missing-gate negative output is `ok=false` with blocker `final_geometry_gate_missing`, and a semantics fixture confirms `formal_points_allowed_now=true` still leaves `formal_points_replacement_allowed=false` when `formal_final_geometry_ready=false`.
- `research/reports/carla_air_final_neoverse_geometry_gate_2026_05_29_zh.md` reports that `tools/carla_air/audit_final_neoverse_geometry.py` now provides a strict read-only final geometry gate. The default output `local/carla_air/tmp/final_neoverse_geometry_main_review_20260529.json` is `ok=true` but `formal_final_geometry_ready=false`, `formal_final_root_count=0`, `goal_complete=false`, with blockers `no_formal_final_neoverse_geometry_root` and `proxy_or_baseline_or_diagnostic_roots_only`; a positive fixture passes only when `neoverse_reconstruction=true` and `final_real_4d_geometry=true`, while a missing-flags negative fixture stays false. This is a future final-root verifier only and does not run NeoVerse or promote current proxy / depth / diagnostic points.
- `research/reports/carla_air_neoverse_checkpoint_recheck_2026_05_30_zh.md` records that `third_party/NeoVerse/models/NeoVerse/reconstructor.ckpt` is now a symlink to `/mnt/windows_data2/4dreid_neoverse_weights/NeoVerse/reconstructor.ckpt`, resolving to a `6039277930` byte regular file. `local/carla_air/tmp/neoverse_geometry_readiness_checkpoint_recheck_20260530.json` is still `formal_neoverse_ready=false` / `goal_complete=false`; the checkpoint blocker is removed, but remaining blockers are no formal NeoVerse points root, target selection not ready, candidate formalization not ready, semantic-lidar / weak diagnostic geometry not formal, and multi-identity not ready.
- `research/reports/carla_air_final_geometry_evidence_contract_verifier_2026_05_29_zh.md` reports that `tools/carla_air/verify_final_geometry_evidence_contract.py` is a read-only future evidence contract verifier: it does not run NeoVerse/runtime, does not create final geometry, does not replace proxy points, and does not complete benchmark. `local/carla_air/tmp/final_geometry_evidence_contract_main_review_blocked_20260529.json` stays `ok=false`, `formal_final_geometry_ready=false`, `can_replace_proxy_points_now=false`, `goal_complete=false` with blocker `manifest_unavailable`; `local/carla_air/tmp/final_geometry_evidence_contract_positive_fixture_verification_20260529.json` is positive-fixture-only and still `goal_complete=false`. Diagnostic and research-manifest negative verifications continue to reject promotion.
- `research/reports/carla_air_goal_matrix_final_geometry_gate_integration_2026_05_29_zh.md` reports that `tools/carla_air/audit_goal_execution_matrix.py` now reads the final geometry gate through `--final-geometry-gate`. The default output `local/carla_air/tmp/goal_execution_matrix_final_geometry_gate_main_review_20260529.json` is `ok=true`, `formal_neoverse_ready=false`, `formal_final_geometry_ready=false`, `formal_final_root_count=0`, `can_run_formal_neoverse_now=false`, `goal_complete=false`; a semantics fixture confirms `can_run_formal_neoverse_now` remains controlled by `formal_neoverse_ready`, while final geometry blockers are exposed separately with prefix `Final geometry gate:`. This is downstream visibility only and does not promote current geometry.
- `research/reports/carla_air_goal_matrix_proxy_replaceability_integration_2026_05_29_zh.md` reports that `tools/carla_air/audit_goal_execution_matrix.py` now reads the proxy replacement gate through `--proxy-replaceability` and exposes an independent `proxy_replacement_readiness` route. The default output `local/carla_air/tmp/goal_execution_matrix_proxy_replaceability_main_review_20260529.json` is `ok=true`, `route_count=5`, `can_replace_proxy_annotation_or_points_now=false`, `formal_annotation_replacement_allowed=false`, `formal_points_replacement_allowed=false`, `can_execute_formal_conversion_now=false`, `can_run_formal_neoverse_now=false`, `goal_complete=false`; a semantics fixture confirms proxy replacement readiness can be toggled independently without changing the formal conversion or NeoVerse run booleans. This is matrix visibility only and does not replace current proxy outputs.
- `research/reports/carla_air_actor_to_pixel_contract_downstream_gate_integration_2026_05_29_zh.md` reports that `audit_formal_annotation_path_preflight.py` and `audit_goal_execution_matrix.py` now both consume `required_actor_to_pixel_evidence` from target-selection JSON. The new default target-selection input is `local/carla_air/tmp/poc03_target_selection_readiness_actor_to_pixel_contract_20260529.json`; formal preflight stays `ok=true` but `route_status.formal_conversion_allowed_now=false`, `route_status.formal_points_allowed_now=false`, `goal_complete=false`, and the summary still says `present=true` / `ready_for_target_selection=false` / `satisfied=false`. The goal matrix stays `ok=true` with `can_execute_formal_conversion_now=false`, `can_run_formal_neoverse_now=false`, `can_plan_multi_identity_roster_now=false`, `goal_complete=false`, and the target route blocked. This is downstream gate integration only and does not replace proxy annotation / proxy points, create real synthetic annotation, create true final 4D geometry, run NeoVerse, or change multi-identity benchmark readiness.

Recent controlled-vehicle calibration rerun (2026-05-29) confirmed only a packed-decode diagnostic for `vehicle.tesla.model3` actor `25` under `Town10HD --opengl --quality Low --fg`: `green_plus_256_blue` saw the actor in ROI, while the formal `contract_red_plus_256_green` did not. This remains non-promotion evidence only; it does not change `target_selection_ready=false`, `ready_for_formal_conversion=false`, or `goal_complete=false`.

2026-05-29 existing-actor ROI sweep for AirSim drone actor `24` under `Town10HD --opengl --quality Low --fg` stayed diagnostic only: `--target-actor-id` existing actor mode and `--camera-distance-m-list` / `--camera-yaw-offset-deg-list` now work, but 12/12 captures still yielded `roi_actor_id_pixel_count=0` for both `contract_red_plus_256_green` and `green_plus_256_blue`. `cleanup_remaining_after.count=0`, but the CarlaAir session exited with `Signal 11`; keep this as a runtime cleanup caveat, not a promotion signal.

2026-05-29 guarded existing-actor sweep passthrough smoke confirmed the guarded runner accepts existing-actor sweep passthrough args and the post-guarded wrapper routes `existing_actor` reports to `--existing-actor-roi-sweep`. This is no-runtime smoke only; `can_replace_proxy_annotation_or_points_now=false`, `goal_complete=false`, and there is no formal replacement yet.

Current hard booleans:

```text
offline_dataset_generation_v1_closed=true
offline_dataset_generation_v1_verifier_ok=true
offline_dataset_generation_v1_failure_count=0
mask_gt_available_count=0
formal_mask_gt_ready=false
pixel_accurate_mask_gt_ready=false
full_live_6_identity_dataset_complete=false
ue_carla_import_readback_complete=false
formal_annotation_ready=false
real_4d_geometry_ready=false
target_selection_ready=false
trusted_target_id_or_tag_available=false
ready_for_formal_conversion=false
can_execute_formal_conversion_now=false
formal_neoverse_ready=false
can_run_formal_neoverse_now=false
benchmark_eligible_count=0
private_benchmark_eligible_count=6
private_carla_benchmark_eligible_count=0
benchmark_roster_count=0
private_benchmark_roster_count=6
can_plan_multi_identity_roster_now=false
goal_complete=false
```

2026-05-30 latest update: dataset pipeline strategy is now explicitly split. `Dataset Generation Pipeline V1` remains the mainline, while `mask_gt` is treated as a separate limited-scope audit / pluggable label source rather than a default promotion target. The node-camera probe confirmed CARLA `RGB` / `semantic` / `instance` / `depth` availability on `node03` / `node04`, but packed instance decode still did not observe `actor_id=24`; AirSim onboard camera remains excluded from dataset `mask_gt` sourcing. Proxy / candidate / pseudo outputs still must not be written as `mask_gt`.

Foreground OpenGL / Low live-window update:

- `local/carla_air/tmp/runtime_stability_audit_fg_opengl_low_api_probe_window_20260529.json` 记录的这一个 live window 中，`runtime_stable_for_window=true`，`all_required_ports_open_sample_count=5`，`all_required_ports_open_all_samples=true`。
- `local/carla_air/tmp/runtime_api_responsiveness_audit_fg_opengl_low_ports_open_20260529.json` 记录的这一个 live window 中，`carla_api_responsive=true`、`airsim_api_responsive=true`、`all_required_apis_responsive=true`。
- `local/carla_air/tmp/target_actor_binding_probe_fg_opengl_low_api_responsive_20260529.json` 记录 actor `24` / `airsim.drone` 与 AirSim `SimpleFlight` 的 pose 绑定可复现，但 `airsim_segmentation_set_effective_by_get_id=false`，`sufficient_for_identity_proof=false`，`sufficient_for_pixel_accuracy_proof=false`。
- `local/carla_air/tmp/carla_annotation_api_surface_probe_fg_opengl_low_20260529.json` 继续显示公开 CARLA Python API surface 没有 actor/world instance-id、segmentation-id 或 custom-stencil setter。
- 因此该 live window 只移除了 foreground OpenGL / Low 启动模式下的即时 runtime TCP/API blocker；`target_selection_ready`、`ready_for_formal_conversion`、`formalization_ready` 与 `goal_complete` 仍保持 false。
- 2026-05-29 runtime evidence reconciliation 已记录到 `research/reports/carla_air_runtime_evidence_reconciliation_2026_05_29_zh.md`。`local/carla_air/tmp/poc03_target_selection_readiness_runtime_evidence_reconcile_20260529.json` 与 `local/carla_air/tmp/goal_execution_matrix_runtime_evidence_reconcile2_20260529.json` 明确区分当前端口离线和 evidence-window runtime gate：带 foreground OpenGL / Low evidence 时 runtime gates 为 `not_blocking`，但 `can_execute_formal_conversion_now=false`，target route blockers 仍是 actor-to-pixel / target identity 缺口。
- 2026-05-29 annotation source bridge live probe 已记录到 `research/reports/carla_air_annotation_source_bridge_live_probe_2026_05_29_zh.md`。`local/carla_air/tmp/annotation_source_probe_offline_bridge_check_20260529.json` 维持 `raw_carla_depth_semantic_instance_camera_available="unknown"`；`local/carla_air/tmp/annotation_source_probe_live_bridge_20260529.json` 仅证明 CARLA depth / semantic / instance camera 与 AirSim Segmentation / DepthPlanar 的 source availability。`converter_present=true`，supported converter sources 为 `instance` 和 `semantic`，但 `formal_conversion_still_requires_target_selection=true`、`ready_for_formal_conversion=false`、`formal_neoverse_ready=false`、`goal_complete=false`。cleanup caveat 仍在：probe 自身 `cleanup_probe_sensors_after=1` 且 runtime session 以 Signal 11 退出。

## Current authoritative outputs

Use these fresh outputs for current-state claims:

```text
local/carla_air/tmp/post_guarded_traceability_audits_main_review_20260529.json
local/carla_air/tmp/post_guarded_main_review/poc03_target_selection_readiness.json
local/carla_air/tmp/post_guarded_main_review/candidate_formalization_readiness.json
local/carla_air/tmp/post_guarded_main_review/target_traceability_gap.json
local/carla_air/tmp/poc03_target_selection_readiness_existing_actor_sweep_main_review_20260529.json
local/carla_air/tmp/poc03_target_selection_readiness_existing_actor_decode_summary_main_review_20260529.json
local/carla_air/tmp/candidate_formalization_readiness_existing_actor_sweep_main_review_20260529.json
local/carla_air/tmp/target_traceability_gap_existing_actor_sweep_main_review_20260529.json
local/carla_air/tmp/existing_actor_roi_decode_summary_main_review_20260529.json
local/carla_air/tmp/post_guarded_existing_sweep_wrapper_fixture_20260529.json
local/carla_air/tmp/post_guarded_existing_sweep_wrapper_fixture_20260529/existing_actor_roi_decode_summary.json
local/carla_air/tmp/poc03_target_selection_readiness_controlled_decode_guard_20260529.json
local/carla_air/tmp/scene_pipeline_status_audit_target_selection_guard_20260529.json
local/carla_air/tmp/neoverse_geometry_readiness_20260529.json
local/carla_air/tmp/goal_execution_matrix_current_defaults_20260529.json
local/carla_air/tmp/dataset_v1_offline_v1_closure_main_review_verification_20260531.json
local/carla_air/tmp/dataset_v1_identity_switch_artifact_build_bridge_main_review_verification_20260531.json
local/carla_air/tmp/dataset_v1_sidecar_quality_artifact_main_review_verification_20260531.json
local/carla_air/tmp/dataset_v1_no_mask_non_promotion_manifest_main_review_verification_20260531.json
local/carla_air/tmp/dataset_v1_capture_queue_manifest_main_review_verification_20260531.json
local/carla_air/tmp/dataset_v1_existing_scene_index_bridge_main_review_verification_20260531.json
local/carla_air/tmp/dataset_v1_scene_sample_index_manifest_main_review_verification_20260531.json
local/carla_air/tmp/dataset_v1_scene_sample_index_hash_bridge_main_review_verification_20260531.json
local/carla_air/tmp/dataset_v1_run_contract_summary_main_review_verification_20260531.json
local/carla_air/tmp/dataset_v1_sample_schema_coverage_manifest_main_review_verification_20260531.json
local/carla_air/tmp/goal_execution_matrix_runtime_ports_candidate_defaults_final_20260529.json
local/carla_air/tmp/candidate_formalization_readiness_current_defaults_final_20260529.json
local/carla_air/tmp/runtime_stability_audit_runtime_restart_attempt_20260529.json
local/carla_air/tmp/runtime_stability_audit_fg_opengl_low_api_probe_window_20260529.json
local/carla_air/tmp/goal_execution_matrix_runtime_stability_gate_main_review_20260529.json
local/carla_air/tmp/runtime_api_responsiveness_audit_fg_opengl_low_ports_open_20260529.json
local/carla_air/tmp/poc03_target_selection_readiness_runtime_restart_attempt_20260529.json
local/carla_air/tmp/poc03_target_selection_readiness_fg_opengl_low_api_responsive_20260529.json
local/carla_air/tmp/poc03_target_selection_readiness_runtime_evidence_reconcile_20260529.json
local/carla_air/tmp/candidate_formalization_readiness_runtime_restart_attempt_20260529.json
local/carla_air/tmp/candidate_formalization_readiness_fg_opengl_low_api_actor_binding_20260529.json
local/carla_air/tmp/goal_execution_matrix_runtime_evidence_reconcile2_20260529.json
local/carla_air/tmp/annotation_source_probe_offline_bridge_check_20260529.json
local/carla_air/tmp/annotation_source_probe_live_bridge_20260529.json
local/carla_air/tmp/neoverse_geometry_readiness_after_annotation_source_bridge_20260529.json
local/carla_air/tmp/goal_execution_matrix_after_annotation_source_bridge_rerun_20260529.json
local/carla_air/tmp/goal_execution_matrix_current_multi_local_poc_20260529.json
local/carla_air/tmp/aircraft_identity_readiness_current_multi_local_poc_20260529.json
local/carla_air/tmp/aircraft_identity_roster_plan_current_multi_local_poc_20260529.json
local/carla_air/tmp/aircraft_identity_readiness_private_policy_final_20260530.json
local/carla_air/tmp/aircraft_identity_roster_plan_private_policy_final_20260530.json
local/carla_air/tmp/goal_execution_matrix_private_policy_final_20260530.json
local/carla_air/tmp/aircraft_identity_import_smoke_current_multi_local_poc_20260529.json
local/carla_air/tmp/aircraft_identity_ue_carla_import_readiness_current_multi_local_poc_20260529.json
local/carla_air/tmp/local_poc_multi_identity_guard_20260529.json
local/carla_air/tmp/aircraft_identity_readiness_procedural_permission_poc_20260529.json
local/carla_air/tmp/aircraft_identity_permission_evidence_procedural_permission_poc_20260529.json
local/carla_air/tmp/aircraft_identity_roster_plan_procedural_permission_poc_20260529.json
local/carla_air/tmp/aircraft_identity_import_smoke_procedural_permission_poc_20260529.json
local/carla_air/tmp/aircraft_identity_ue_carla_import_readiness_procedural_permission_poc_20260529.json
local/carla_air/tmp/local_poc_multi_identity_guard_procedural_permission_poc_20260529.json
local/carla_air/tmp/goal_execution_matrix_procedural_permission_poc_20260529.json
local/carla_air/tmp/aircraft_identity_readiness_procedural_twinboom_poc_20260529.json
local/carla_air/tmp/aircraft_identity_permission_evidence_procedural_twinboom_poc_20260529.json
local/carla_air/tmp/aircraft_identity_roster_plan_procedural_twinboom_poc_20260529.json
local/carla_air/tmp/aircraft_identity_import_smoke_procedural_twinboom_poc_20260529.json
local/carla_air/tmp/aircraft_identity_ue_carla_import_readiness_procedural_twinboom_poc_20260529.json
local/carla_air/tmp/local_poc_multi_identity_guard_procedural_twinboom_poc_20260529.json
local/carla_air/tmp/goal_execution_matrix_procedural_twinboom_poc_20260529.json
local/carla_air/tmp/aircraft_identity_readiness_procedural_canard_poc_20260529.json
local/carla_air/tmp/aircraft_identity_permission_evidence_procedural_canard_poc_20260529.json
local/carla_air/tmp/aircraft_identity_roster_plan_procedural_canard_poc_20260529.json
local/carla_air/tmp/aircraft_identity_import_smoke_procedural_canard_poc_20260529.json
local/carla_air/tmp/aircraft_identity_ue_carla_import_readiness_procedural_canard_poc_20260529.json
local/carla_air/tmp/aircraft_identity_ue_import_smoke_plan_procedural_canard_poc_20260529.json
local/carla_air/tmp/aircraft_identity_ue_import_smoke_plan_verification_procedural_canard_poc_20260529.json
local/carla_air/tmp/aircraft_identity_ue_import_smoke_plan_recheck_20260529.json
local/carla_air/tmp/aircraft_identity_ue_import_smoke_plan_verification_recheck_20260529.json
local/carla_air/tmp/local_poc_multi_identity_guard_procedural_canard_poc_20260529.json
local/carla_air/tmp/goal_execution_matrix_procedural_canard_poc_20260529.json
local/carla_air/tmp/semantic_lidar_actor_points_candidate_verification_scene_lineage_exit_20260529.json
local/carla_air/tmp/semantic_lidar_actor_local_geometry_derivation_20260529.json
local/carla_air/tmp/semantic_lidar_actor_local_geometry_verification_20260529.json
local/carla_air/tmp/goal_execution_matrix_after_actor_local_geometry_20260529.json
local/carla_air/tmp/goal_execution_matrix_semantic_lidar_scene_lineage_guard_20260529.json
local/carla_air/tmp/goal_execution_matrix_mask_tracklet_lineage_required_20260529.json
local/carla_air/tmp/local_annotation_hook_surface_audit_20260529.json
local/carla_air/tmp/instance_id_calibration_actor_probe_fg_opengl_low_rerun_20260529/report.json
local/carla_air/tmp/existing_actor24_decode_roi_sweep_fg_opengl_low_20260529/report.json
local/carla_air/tmp/poc03_target_selection_readiness_after_fg_opengl_low_calibration_rerun_20260529.json
local/carla_air/tmp/poc03_target_selection_readiness_after_actor24_existing_sweep_20260529.json
local/carla_air/tmp/goal_execution_matrix_after_actor24_existing_sweep_20260529.json
```

Runtime snapshot from the latest runtime stability gate report:

```text
runtime_stable_for_window=false
sample_count=5
all_required_ports_open_sample_count=0
matching_process_seen_any_sample=false
CARLA 127.0.0.1:2000 not stable/open for the window
AirSim 127.0.0.1:41451 not stable/open for the window
```

## Non-promotion rules

These artifacts are not formal evidence:

- `5387`, `44800`, `61871`, `35093`, close ROI, bbox rectangles, raycast/GBuffer diagnostics, and alternate decode values are not target identity proof.
- Actor-pose projected bbox masks are not pixel-accurate target masks.
- Semantic-lidar actor-relative points are not fixed-camera capture geometry, not NeoVerse reconstruction, and not final 4D geometry.
- Semantic-lidar actor-local geometry candidates are useful actor-level geometry consistency evidence, but still not fixed-camera capture geometry, not NeoVerse reconstruction, and not final 4D geometry.
- Current `mvp-demo/output/neoverse_fused/...` POC03 roots remain proxy-source outputs.
- Current `mvp-demo/output/carla_air_depth_points/...` POC03 roots remain `carla_depth_synth_mask_backprojection_v1` baseline, not NeoVerse.
- Weak diagnostic embedding/eval outputs are not formal benchmark metrics.
- Assimp import/readback is not UE/CARLA import evidence and is not benchmark permission evidence.
- Proxy / candidate / pseudo masks are not `mask_gt`.
- The UE/CARLA import smoke plan is a future evidence checklist only; it is not import evidence, does not patch `build_status.json`, and does not make any identity benchmark-eligible.
- The procedural identity POCs improve three identities' permission evidence, but they are not UE/CARLA import evidence and do not make any identity benchmark-eligible yet.
- The local text/config/source hook audit found read surfaces and render-support references only; it found no actionable actor-level instance/segmentation/stencil hook, so current formal flags remain `false` and the audit does not replace proxy annotation / proxy points or imply final geometry.
- The 2026-05-29 controlled-vehicle calibration rerun is a packed-decode diagnostic only; it does not bind the POC03 drone target actor `24`, does not authorize formal conversion, and does not justify moving old reports on disk.

## Soft archive policy

Older reports under `research/reports/` are preserved in place for source attribution and regression history. They should be treated as one of:

- `historical`: useful context, not current gate truth.
- `superseded`: replaced by a later guard or audit.
- `smoke-only`: proves a tool path or fixture once worked, not formal readiness.
- `lineage evidence`: read only when verifying a referenced artifact path or sha256 lineage.

Do not move old reports without updating all references in `ACTIVE_PLAN.md`, handoffs, and dependent reports.

## Next work

1. Resolve POC03 target-to-pixel evidence so `target_selection_ready=true` can be proven without relying on candidate-only ids.
2. Produce a non-proxy formal NeoVerse / final 4D geometry output root from trusted target-selection / formal annotation inputs. The NeoVerse checkpoint path now exists via symlink, but no formal NeoVerse points root exists yet.
3. For private local benchmark work, downloaded/purchased `assets/models/` identities may be used without blocking on public license / redistribution evidence before open-source release. For public/formal benchmark release, auditable source/license/benchmark permission evidence and UE/CARLA import evidence are still required. Current gate split is `private_benchmark_eligible_count=6`, `private_carla_benchmark_eligible_count=0`, `benchmark_eligible_count=0`; no `UE4Editor` / `UnrealEditor` import toolchain is available.
4. Only after the relevant gates pass, run formal conversion / formal geometry / formal multi-identity benchmark.
5. The local hook audit only confirms read surfaces / render support in text, config, and source; it does not expose an actionable actor-level instance/segmentation/stencil setter, so the current false gates stay in force.

Latest UE/CARLA import toolchain gate:

- `research/reports/carla_air_ue_import_explicit_editor_gate_2026_05_29_zh.md` records explicit `--editor-cmd` support for UE import readiness and smoke planning.
- `/does/not/exist` and `/bin/true` negative checks remain blocked; `/bin/true` is executable but rejected because it is not a recognized Unreal editor command name.
- Current benchmark status remains `benchmark_eligible_count=0`, `verified_identity_count=0`, `can_run_ue_import_smoke_now=false`, `formal_benchmark_ready=false`.
- 2026-05-30 private/public model policy split is recorded in `research/reports/carla_air_private_model_policy_split_2026_05_30_zh.md`. Current downloaded/purchased and procedural normalized identities are private-local eligible: `private_benchmark_eligible_count=6`, `private_benchmark_roster_count=6`, `private_benchmark_multi_identity_ready=true`. Public/formal benchmark remains blocked: `benchmark_eligible_count=0`, `benchmark_roster_count=0`, `formal_benchmark_ready=false`; private CARLA benchmark also remains blocked by missing UE/CARLA import evidence: `private_carla_benchmark_eligible_count=0`.
- 2026-05-29 新增 `tools/carla_air/apply_ue_import_smoke_evidence.py` 与报告 `research/reports/carla_air_ue_import_smoke_evidence_applier_2026_05_29_zh.md`；它们只做 UE import smoke evidence 的受控应用与核对，不进入默认 must_read。当前 multi-identity / UE import gate 仍阻塞：`normalized_identity_count=6`、`benchmark_permission_ready_count=3`、`ue_carla_verified_identity_count=0`、`benchmark_eligible_count=0`，且 annotation / proxy / final geometry 状态不变。验证结果保持一致：readonly positive fixture `ok=true` / `patch_applied=false`，dry-run apply negative `ok=false` / `patch_applied=false`，`research/` report negative 拒绝 `report_under_research_forbidden` / `report_outside_local_forbidden`。
- 2026-05-29 UE import smoke plan -> applier contract bridge 已同步到 `research/reports/carla_air_ue_import_smoke_plan_applier_contract_bridge_2026_05_29_zh.md`：plan contract bridge `ok=false`、`runnable_now_count=0`、`blocked_now_count=6`、`formal_benchmark_ready=false`；verifier `ok=true`、`failure_count=0`、`verified_identity_plan_count=6`、`non_promotion_verified=true`；applier template-copied negative rejects `template_contract_marker_present` / `editor_command_not_recognized`；benchmark 仍为 `benchmark_eligible_count=0`、`ue_carla_verified_identity_count=0`，annotation / proxy / final geometry unchanged。
- 2026-05-29 新增 `tools/carla_air/verify_proxy_replacement_offline_regression.py` 与报告 `research/reports/carla_air_proxy_replacement_contract_integration_2026_05_29_zh.md`；该 verifier 现在对齐 synthetic annotation contract 与 final geometry contract，主审 `local/carla_air/tmp/proxy_replacement_offline_regression_contract_integration_main_review_blocked_20260529.json` 仍为 `ok=true`、`failure_count=0`、`expect_blocked=true`、`any_upstream_positive=false`，且 contract blockers 仍是 `manifest_unavailable` / `manifest_unavailable`；route-positive fixture 仅证明 `--no-expect-blocked` 下的一致性，annotation / geometry 不可晋升，benchmark 仍未完成。
- 2026-05-29 新增 `tools/carla_air/verify_proxy_replacement_offline_regression.py` 与报告 `research/reports/carla_air_proxy_replacement_offline_regression_gate_2026_05_29_zh.md`；它只读 target traceability、final geometry gate 与 goal execution matrix，默认验证当前 blocked invariant。主审输出 `local/carla_air/tmp/proxy_replacement_offline_regression_main_review_20260529.json` 为 `ok=true`、`failure_count=0`、`any_upstream_positive=false`、`expect_blocked=true`；负例可抓住 inconsistent true replacement，future route-positive fixture 在 `--no-expect-blocked` 下可通过。当前仍不能替换 proxy annotation / points，不能声明 final 4D geometry、NeoVerse/runtime 或 benchmark readiness。
- 2026-05-29 新增 `tools/carla_air/verify_benchmark_promotion_offline_regression.py` 与报告 `research/reports/carla_air_benchmark_promotion_offline_regression_2026_05_29_zh.md`；主审 `local/carla_air/tmp/benchmark_promotion_offline_regression_main_review_blocked_20260529.json` 为 `ok=true`、`failure_count=0`、`expect_blocked=true`、`formal_benchmark_ready=false`、`can_plan_multi_identity_roster_now=false`、`goal_complete=false`、`non_promotion_verified=true`。route-positive 仅是 fixture-only，不能据此宣称真实 benchmark ready；当前 benchmark 仍是 `benchmark_eligible_count=0`、`ue_carla_verified_identity_count=0`。
- 2026-05-29 新增 `research/reports/carla_air_goal_execution_matrix_regression_visibility_2026_05_29_zh.md`；`tools/carla_air/audit_goal_execution_matrix.py` 的 regression visibility 主审输出 `local/carla_air/tmp/goal_execution_matrix_regression_visibility_main_review_20260529.json` 为 `ok=true`、`route_count=5`、`blocked_route_ids=['target_to_pixel_binding','formal_neoverse_geometry','multi_identity_benchmark','proxy_replacement_readiness','weak_diagnostic_guarded_route']`、`goal_complete=false`，并保持 `can_execute_formal_conversion_now=false`、`can_run_formal_neoverse_now=false`、`can_plan_multi_identity_roster_now=false`、`can_replace_proxy_annotation_or_points_now=false`、`benchmark_eligible_count=0`；这只是 route-level regression 的诊断/可见性同步，不改变阻塞态。
