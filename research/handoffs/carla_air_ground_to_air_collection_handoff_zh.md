# CARLA-Air 地对空采集工程交接

日期：2026-05-25

## 1. 当前状态

当前项目路径固定为：

```text
/home/grasp/data/3d-reid
```

后续 AI 默认先读：

```text
docs/goal.md
research/CURRENT_STATUS.md
research/plans/ACTIVE_PLAN.md
research/handoffs/carla_air_ground_to_air_collection_handoff_zh.md
```

2026-05-30 起，活跃阶段目标已重置为 CARLA-Air Dataset Generation Pipeline V1：先用 Town10HD `node01-node05` 跑通支持 6 个 normalized identity 的数据集生成、统一 training index、真实模型替换 gate 与 `mask_gt` 可得性审计。旧的 immediate proxy / NeoVerse replacement 目标已归档到 `docs/history/goal_2026_05_30_proxy_replacement_legacy.md`，只作为历史追溯入口。

`research/reports/` 下旧报告保留原路径作为 provenance / regression history，不默认全文阅读；只有追溯具体 artifact、命令、sha256 lineage 或 blocker 时再按需打开。不要物理移动旧报告，除非同步更新所有引用。

CARLA-Air v0.1.7 runtime 已部署在：

```text
local/carla_air/simulators/CarlaAir-v0.1.7/
```

当前已完成：

- CARLA-Air simulator 与 Python API smoke；
- `carlaAir` conda 环境；
- 交互式地对空三相机节点布设脚本；
- `Town10HD` 中 `node01-node05` 的固定相机节点配置；
- 基于已保存节点配置的 CARLA-Air RGB capture smoke；
- `node01-node05` 多节点同步 RGB、相机内参、相机位姿和 timestamp 导出验证；
- 组会展示用三路相机并排视频导出验证；
- 默认 AirSim drone 轨迹 smoke 工具与离线 presence gate 工具实现；
- live 默认 AirSim drone 轨迹闭环样例：`traj_poc_01_west_corridor + node01 node02 node05`，`capture_run_id=20260525_082900`，presence gate pass；
- 4 条 live smoke baseline 已跑通，均生成 `presence_gate.json`、`review_summary.csv` 和 `review_overlays/`；
- `tools/carla_air/run_drone_trajectory_smoke_suite.py` 已加入，支持按轨迹配置串行 orchestration，目前只完成 dry-run 命令计划检查，尚未执行 live suite 全量自动化。
- `tools/carla_air/capture_drone_trajectory_nodes.py` 已加入，进入阶段 4A 同步采集器骨架，后续轨迹规划将升级为 coverage-first trajectory generator。
- `traj_poc_02_short_overlap + node01 node03 node05` 4A live 同步采集已跑通，`capture_success=true`、`gate_pass=true`，`trajectory_frame_groups.csv` 行数为 `120 * 3 = 360`。
- 4B 正式数据契约已冻结，权威输入为 `capture_meta.json`、`trajectory_capture_manifest.json`、`trajectory_frame_groups.csv`、`nodes/<node_id>/frame_times.csv`、`nodes/<node_id>/calib/rig.json`、`nodes/<node_id>/cams/cam*/frames/*.png`。
- coverage-first trajectory generator 已加入，`town10hd_coverage_first_v1.json` 可被 trajectory runner 和同步 capture 入口 dry-run 消费。
- `tools/carla_air/export_capture_scene_dirs.py` 已加入，用于把已通过 gate 的 capture 转换为 node 级 `scene_dir` skeleton，并显式落 `tracklets` / `points_by_timestamp` 的 pending 契约。
- 代表样例 `20260525_155751_traj_poc_02_short_overlap` 已导出 node01/node03/node05 三个 scene skeleton，三路相机各 120 帧，未混入 smoke/QC/video 产物。
- coverage-first 补强 `traj_poc_03_southeast_pass` 已完成 live capture / gate / scene skeleton 导出，`run_id=20260525_180505_traj_poc_03_southeast_pass`，nodes=`node03 node04`，每节点三路相机各 120 帧，`dropped_or_incomplete_frames=0`，`gate_pass=true`。
- coverage-first 补强 `traj_poc_04_north_bridge` 已完成 live capture / gate / scene skeleton 导出，`run_id=20260525_180648_traj_poc_04_north_bridge`，nodes=`node02 node05`，每节点三路相机各 120 帧，`dropped_or_incomplete_frames=0`，`gate_pass=true`。
- POC03 node03/node04 已完成最小正式样本验证：从 `trajectory_frame_groups.csv` 的 recorded drone pose 与 `calib/rig.json` 的 fixed camera rig 生成 formal proxy annotations、`tracks/tracklets.json`、proxy `points_by_timestamp`、`embeddings_minimal_formal/`，并完成 node03 <-> node04 cross-node retrieval smoke。
- 已完成 annotation source / geometry readiness 核查：CARLA / AirSim Python API 可见 depth、semantic、instance、segmentation、pose 路径；2026-05-28 当前会话按 runtime 自启动规则启动 CARLA-Air 后，CARLA RPC `2000` 与 AirSim RPC `41451` 已在线并完成 live synthetic sensor export / inspect / verify smoke。`tools/carla_air/probe_annotation_sources.py` 与 `tools/carla_air/export_live_synthetic_annotations.py` 保留为后续可复跑入口。
- 2026-05-29 foreground OpenGL / Low live window 进一步把 runtime / API / actor-binding 证据串起来：`local/carla_air/tmp/runtime_stability_audit_fg_opengl_low_api_probe_window_20260529.json` 记录 `sample_count=5`、`all_required_ports_open_sample_count=5`、`all_required_ports_open_all_samples=true`、`matching_process_seen_any_sample=true`、`runtime_stable_for_window=true`；`local/carla_air/tmp/runtime_api_responsiveness_audit_fg_opengl_low_ports_open_20260529.json` 记录 `carla_api_responsive=true`、`airsim_api_responsive=true`、`all_required_apis_responsive=true`。`local/carla_air/tmp/target_actor_binding_probe_fg_opengl_low_api_responsive_20260529.json` 显示 actor `24` / `airsim.drone` 与 AirSim `SimpleFlight` 的 pose 绑定可复现，但 `airsim_segmentation_set_effective_by_get_id=false`、`sufficient_for_identity_proof=false`、`sufficient_for_pixel_accuracy_proof=false`；`local/carla_air/tmp/carla_annotation_api_surface_probe_fg_opengl_low_20260529.json` 继续确认公开 CARLA Python API 没有 actor/world instance-id、segmentation-id 或 custom-stencil setter。该 live window 只移除了 foreground OpenGL / Low 启动模式下的即时 runtime TCP/API blocker，不改变 `target_selection_ready=false`、`ready_for_formal_conversion=false`、`formalization_ready=false` 或 `goal_complete=false`。
- 2026-05-29 runtime evidence reconciliation 已记录到 `research/reports/carla_air_runtime_evidence_reconciliation_2026_05_29_zh.md`。`tools/carla_air/audit_poc03_target_selection_readiness.py` 现在可选读取 `--runtime-stability` 与 `--runtime-api-responsiveness`，`tools/carla_air/audit_goal_execution_matrix.py` 的 target route 也区分 current live ports 和 supplied runtime evidence gates。带 foreground OpenGL / Low 证据的 `local/carla_air/tmp/poc03_target_selection_readiness_runtime_evidence_reconcile_20260529.json` 显示 `runtime_evidence_window_not_blocking=true`，`local/carla_air/tmp/goal_execution_matrix_runtime_evidence_reconcile2_20260529.json` 显示 runtime gates 均 `not_blocking` 但 `can_execute_formal_conversion_now=false`；当前端口离线只作为不能立即刷新 live probe 的 note，不再误写成该 evidence window 的 runtime blocker。默认 offline/restart evidence 仍保持 blocked。
- 已补齐 synthetic sensor 到下游契约的离线承接工具：`tools/carla_air/convert_synthetic_sensor_masks.py` 负责 target `masks_synth` / `tracklets_synth`，`tools/carla_air/backproject_synthetic_depth_points.py` 负责 `depth_synth + masks_synth + fixed rig` 到非 proxy `points_by_timestamp`；已用 ignored 临时 scene 完成 miniature smoke，未写入正式 POC03 scene。
- 已新增 read-only synthetic sensor id/tag 候选检查工具：`tools/carla_air/inspect_synthetic_sensor_ids.py`，只输出 candidate statistics，不写 `masks_synth`、`tracklets_synth`、annotation status 或 pipeline contract；已用 ignored 临时 scene 验证 BGRA 解码。
- 已新增 target-candidate validation gate：`tools/carla_air/validate_synthetic_target_candidate.py`，只读核查 candidate JSON 的 scene/node/identity/camera/source/threshold 一致性，输出 `target_candidate_gate_passed_not_identity_proof` 或失败报告，不写 mask/tracklet/status，也不声称 identity proof 或 pixel accuracy。
- `tools/carla_air/convert_synthetic_sensor_masks.py` 现已收紧为正式 conversion 必须携带通过的 `--target-validation-report`，并把 validation evidence 写入 `tracklets_synth` / `annotation_meta_synth`；`tools/carla_air/backproject_synthetic_depth_points.py` 也必须携带同一 report 并把 evidence 写入 points `meta.json`；`--allow-unvalidated-target` 只可用于 debug，`--mark-pixel-accurate` 必须另给 `--pixel-accuracy-evidence`。
- 已新增 POC03 node03/node04 synthetic ReID 编排入口：`tools/carla_air/run_poc03_synthetic_reid_pipeline.py`，默认只生成 plan，`--execute` 可串联 live probe、synthetic sensor export、candidate id inspection、target-candidate validation、target mask conversion、depth-backprojection points、embedding、cross-node eval 与 optional verify；进入 conversion 阶段必须显式给出 target `instance_id` 或 `semantic_tag`，执行模式不允许跳过 target validation；plan-only 输出已包含 `readiness` 区块，会显式报告 CARLA/AirSim runtime blocker、target id/tag 是否已选择、当前 synthetic 输出是否存在、下一条 inspect 执行命令、inspect 后 verify 执行模板和 inspect 后必须进入 target validation 的规则。plan-only 可用 `<instance_id>` / `<semantic_tag>` 占位展示完整链路，含占位符的 step 会标记 `template_requires_user_values=true` / `template_placeholders=[...]`，并在顶层 `readiness.templated_steps_require_user_values` / `readiness.templated_steps` 汇总；执行模式仍强制真实 target 且拒绝含占位符计划。
- 已新增 post-run anti-proxy verifier：`tools/carla_air/verify_poc03_synthetic_reid_outputs.py`，只读核查 `tracklets_synth`、`annotation_meta_synth`、`masks_synth`、depth-backproject points、embeddings 与双向 eval；会拒绝 proxy source、`masks_gt`、`neoverse_fused` proxy root、scene 内 QC artifacts，以及 track/meta/points meta 缺失 target-validation evidence 的 synthetic 输出。2026-05-28 live smoke 后，官方 POC03 node03/node04 separate synthetic variant 已通过该 verifier；早先缺少 synthetic 输出导致的负例仅作为历史回归证据保留。
- 2026-05-27 已收紧 synthetic lineage 与状态契约：raw sensor export、candidate validation、conversion、points 和 verifier 都要求同一个 `sensor_export_id`；formal synthetic mask / points 状态只写 additive variants，不覆盖最小 proxy 主字段；re-export / re-convert / re-backproject 会把已有 derived synthetic variants 标为 pending / not-ready，避免 stale points 继续被消费。
- 2026-05-27 已用 ignored fixture `local/carla_air/tmp/synthetic_contract_e2e_20260527_1902` 跑通 lineage-aware contract E2E：`verifier_report_lineage_final.json` 为 `ok=true`、`failure_count=0`，node03/node04 均为 1 track、2 timestamps、6 masks、2 point files、embedding shape `[1,161]`，双向 eval `mAP=1.0` / `R@1=1.0`。该 fixture 是 contract smoke，不是 live runtime evidence。
- 2026-05-27 verifier 进一步收紧 depth-backproject points 对应关系：`track.timestamp_stems`、`fused_points_paths`、`points_root/index.csv`、`meta.json.count` 与 points root 下 `.npy` stem 必须逐 timestamp 对齐；positive fixture `verifier_report_point_correspondence.json` 为 `ok=true`，stale-point 负例 `local/carla_air/tmp/synthetic_contract_point_lineage_negative_20260527/evidence/verifier_report_point_lineage_negative.json` 为 `ok=false`、`failure_count=7`。
- 2026-05-27 formal depth-backprojection points 进一步改为以 `tracks/tracklets_synth.json` 为 timestamp 来源，逐项核对 `timestamp_stems`、`mask_paths`、`depth_paths` 与 `frame_times.csv`；track-driven positive fixture `local/carla_air/tmp/synthetic_contract_track_driven_points_20260527/evidence/verifier_report_track_driven_points.json` 为 `ok=true`，stale depth path 负例会在写 points 前拒绝。
- 2026-05-27 POC03 synthetic 编排入口默认终点从 `eval` 改为 `verify`；显式 `--stop-after eval` 仍可用于诊断计划，但默认 plan/execute 会包含 anti-proxy verifier。
- 2026-05-27 live smoke 前官方 POC03 verifier 历史负例：`local/carla_air/tmp/official_poc03_synthetic_verification_20260527_track_driven_points.json` 中 `ok=false`、`failure_count=22`，缺少 `tracklets_synth.json`、`annotation_meta_synth.json`、raw synthetic sensor meta/status、depth-backproject points、synthetic embeddings 与双向 eval。
- 2026-05-28 已收紧 embedding / eval stale provenance：`extract_node_track_embeddings.py` 写 `node_track_embedding_meta_v2`，记录 `tracklets_synth` hash 与 used timestamp 的 frame/mask/points hash；`eval_node_track_retrieval.py` 写 `node_track_retrieval_eval_v2`，记录 query/gallery embedding 文件 hash；verifier 要求 embedding meta 与当前 `tracklets_synth`、当前 frame/mask/points 文件、当前 eval query/gallery 引用逐项一致。positive fixture `local/carla_air/tmp/synthetic_contract_track_driven_points_20260527/evidence/verifier_report_embedding_eval_provenance_restored.json` 为 `ok=true`，targeted stale embedding / eval 负例分别为 `ok=false`。
- 2026-05-28 早先官方 POC03 verifier 负例 `local/carla_air/tmp/official_poc03_synthetic_verification_20260528_embedding_eval_provenance.json` 为历史状态；同日 live runtime smoke 已覆盖该状态，当前权威 verifier 为 `local/carla_air/pipeline_runs/20260528_210808_poc03_synth_depth_reid/synthetic_output_verification.json`，`ok=true`、`failure_count=0`。
- 2026-05-28 已新增 multi-identity aircraft readiness gate：`tools/carla_air/check_aircraft_identity_readiness.py`。该工具只读扫描 raw / normalized assets，不生成或伪造 `normalized.fbx`；当前 latest gate `local/carla_air/tmp/aircraft_identity_readiness_after_target_gate_20260529.json` 为 `ok=false`、`benchmark_ready=false`、raw identity 11、unsupported raw identity 16、normalized identity 0、benchmark eligible 0。对应报告：`research/reports/carla_air_aircraft_identity_readiness_gate_2026_05_28_zh.md`。
- 2026-05-29 已新增 raw aircraft normalization prioritization 支线：`tools/carla_air/prioritize_raw_aircraft_normalization.py` 与报告 `research/reports/carla_air_raw_aircraft_normalization_prioritization_2026_05_29_zh.md`。正例输出 `local/carla_air/tmp/raw_aircraft_normalization_prioritization_20260529.json` 为 `ok=true`、`benchmark_ready=false`、`normalization_ready=false`、`formalization_ready=false`、`goal_complete=false`、`writes_normalized_assets=false`、`writes_scene_outputs=false`；top candidates 为 `dji_drone_fbx_abc`、`dji_inspire2_fbx`、`dji_tech_drone_gltf`。该工具只做 raw candidate 排序，不生成 `normalized.fbx` / `preview.png` / `asset_meta.json`，不把 raw assets 升级为 benchmark identities。
- 2026-05-28 已继续收紧 raw synthetic sensor lineage：`convert_synthetic_sensor_masks.py` 在 `tracklets_synth.json` / `annotation_meta_synth.json` 中记录每个 timestamp/camera 的 raw source sensor、RGB frame、mask、depth path 与 sha256；`backproject_synthetic_depth_points.py` 在 points `meta.json` 中记录 depth/mask input sha256 与输出 point file sha256，并在 formal geometry 写入前校验当前 mask/depth hash；verifier 逐项核对 raw source sensor、mask、depth、points 当前文件 hash，拒绝同路径 in-place 替换。positive fixture `local/carla_air/tmp/synthetic_contract_track_driven_points_20260527/evidence/verifier_report_raw_sensor_lineage.json` 为 `ok=true`、`failure_count=0`；raw `instance_synth` 负例 `verifier_report_raw_sensor_lineage_negative.json` 为 `ok=false`、`failure_count=2`；`depth_synth` 负例 `verifier_report_depth_sensor_lineage_negative.json` 为 `ok=false`、`failure_count=3`。
- 2026-05-28 官方 POC03 raw-lineage 负例 `local/carla_air/tmp/official_poc03_synthetic_verification_20260528_raw_sensor_lineage.json` 为 live smoke 前的历史状态；当前 live run 已生成 raw sensor lineage、target validation、mask/depth/points hash lineage，并通过 `synthetic_output_verification.json`。
- 2026-05-28 已新增官方 exported scene 状态审计工具：`tools/carla_air/audit_scene_pipeline_status.py`。live smoke 前 `local/carla_air/tmp/scene_pipeline_status_audit_20260528.json` 显示 2 个 POC03 scene 仍为 proxy ready；live smoke 后 `local/carla_air/tmp/scene_pipeline_status_audit_20260528_after_live_synth.json` 显示官方 scene 共 7 个，formal inputs 全完整，5 个仍为 `skeleton_pending_formal_annotations`，2 个 POC03 scene 为 `synthetic_annotation_and_depth_geometry_ready`，`synthetic_annotation_ready_count=2`、`synthetic_depth_points_ready_count=2`，scene 内 QC artifact count 为 0。
- 2026-05-28 已继续收紧 candidate inspection / target-validation input lineage：`inspect_synthetic_sensor_ids.py` 在 candidate JSON 中记录每个 raw `instance_synth` / `semantic_synth` PNG 的 `sensor_path`、`sensor_sha256`、timestamp、camera 与 image shape；`validate_synthetic_target_candidate.py` 要求 candidate `input_lineage` 覆盖 requested source / required cams，并逐项核对当前 scene raw PNG sha256；`convert_synthetic_sensor_masks.py` 与 `backproject_synthetic_depth_points.py` 会在消费 target-validation report 前再次校验该 lineage，verifier 也会检查 report 内 candidate raw PNG hash。positive fixture `local/carla_air/tmp/synthetic_contract_track_driven_points_20260527/evidence/verifier_report_candidate_input_lineage.json` 为 `ok=true`、`failure_count=0`；stale candidate raw PNG 负例 `node03_target_validation_stale_input_lineage_negative.json` 为 `ok=false`、`failure_count=1`，failure 为 `candidate_input_lineage_sha256_matches`。
- 2026-05-28 官方 POC03 candidate-lineage 负例 `local/carla_air/tmp/official_poc03_synthetic_verification_20260528_candidate_input_lineage.json` 为 live smoke 前的历史状态；当前 live run 已用 `instance_id=5387` 通过两侧 target-candidate validation。该 id 只证明候选稳定并通过 lineage gate，不证明目标身份或 pixel accuracy。
- 2026-05-28 live synthetic depth-backprojection ReID smoke 已完成：`run_id=20260528_210808_poc03_synth_depth_reid`，命令为 `run_poc03_synthetic_reid_pipeline.py --execute --stop-after verify --limit 10 --overwrite --run-id 20260528_210808_poc03_synth_depth_reid --source instance --instance-id 5387`。node03/node04 均写出 `tracks/tracklets_synth.json`、`annotations/annotation_meta_synth.json`、`cams/<cam>/masks_synth/*.png`、`mvp-demo/output/carla_air_depth_points/<scene_id>/points_by_timestamp/`、`embeddings_synth_depth_backproject/` 与双向 eval；verifier `ok=true`、`failure_count=0`。该结果可记为 validated synthetic annotation / non-proxy depth-backproject geometry smoke，不可记为 pixel-accurate identity proof 或 NeoVerse 真重建。
- 2026-05-28 已新增只读 synthetic target mask quality diagnostic audit：`tools/carla_air/audit_synthetic_target_mask_quality.py` 输出 `local/carla_air/tmp/synthetic_target_mask_quality_audit_20260528.json`，对应报告路径预期为 `research/reports/carla_air_synthetic_target_mask_quality_audit_2026_05_28_zh.md`。该审计不是 pixel-accuracy proof；结果为 `scene_count=2`、`observation_count=60`、两个 POC03 scene 都触发风险，全局风险包括 `large_mask_area_majority`、`edge_touch_majority`、`near_full_width_bbox_majority`，node03 还触发 `low_proxy_bbox_overlap_majority`。因此当前 verifier 仍可支持 validated synthetic annotation / non-proxy depth-backprojection smoke，但新增强 blocker：不能把 `instance_id=5387` 说成 identity proof 或 pixel-accurate mask，也不能说 NeoVerse 真重建 / real 4D geometry 已完成。
- 2026-05-28 已新增 QC-only synthetic candidate overlay 工具：`tools/carla_air/render_synthetic_candidate_qc_overlays.py`。代表输出包括 `local/carla_air/tmp/qc_overlay_node03_instance_5387_with_proxy_ref_20260528.json`、`local/carla_air/tmp/qc_overlay_node03_semantic_14_with_proxy_ref_20260528.json`、`local/carla_air/tmp/qc_overlay_node04_instance_5387_20260528.json` 等；overlay 只用于人工/诊断复核，不进入正式 pipeline。视觉复核显示 `instance_id=5387` 覆盖大片天空/背景；小像素 semantic tag 候选更像场景元素或局部小物体，且跨 node 不一致，不能直接替换为 drone target。
- 2026-05-28 已新增 live target actor binding probe：`tools/carla_air/probe_target_actor_binding.py`，对应报告 `research/reports/carla_air_target_actor_binding_probe_2026_05_28_zh.md`。v4 证据 `local/carla_air/tmp/target_actor_binding_probe_replay_seg_20260528_goal_resume_v4.json` 显示 replay POC03 node03 row 0 后，CARLA actor `id=24` / `type_id=airsim.drone` 距 recorded CARLA pose `0.002423m`，AirSim `SimpleFlight` 距 recorded NED `0.002422m`；runtime restore 验证通过。但 AirSim segmentation id mutation 后 `SimpleFlight` 仍查询为 `-1`，且没有证据把 actor/object 绑定到既有 `instance_id=5387` mask，因此这只是 actor/pose binding evidence，不是 identity proof 或 pixel-accurate mask proof。
- 2026-05-28 已新增 actor bbox instance candidate audit：`tools/carla_air/audit_actor_bbox_instance_candidates.py`，只读对比 recorded pose + fixed rig + CARLA actor bbox projection 与 raw `instance_synth` PNG 候选。`local/carla_air/tmp/actor_bbox_instance_candidate_audit_20260528_limit10_v2.json` 显示 node03 `44800` 是当前几何 sanity 最强候选，`mean_candidate_bbox_iou=0.162635`、`mean_candidate_area_ratio_to_projection=0.162635`、`mean_candidate_center_distance_px=0.459163`；既有 `5387` 为大面积背景型候选，`mean_candidate_bbox_iou=0.000695`、`mean_candidate_area_ratio_to_projection=1525.454545`。`44800` 仍只是候选，不是 actor-id binding proof，也不能直接进入正式 conversion。
- 2026-05-28 已新增 actor-pose projected bbox candidate 工具：`tools/carla_air/export_actor_pose_projected_bbox_annotations.py`，对应报告 `research/reports/carla_air_actor_pose_projected_bbox_candidate_2026_05_28_zh.md`。该工具把 recorded pose + actor bbox + fixed rig 投影为独立 `masks_actor_bbox` / `tracklets_actor_bbox.json`，source 写为 `carla_actor_pose_projected_bbox`，默认拒绝写 `masks_gt`、`masks_synth`、`tracklets.json`、`tracklets_synth.json`。node03 当前 10 帧可导出 candidate annotation、debug depth-backproject points 与 node-only embedding smoke；node04 当前窗口 projected bbox outside。2026-05-29 该工具新增按 `trajectory_frame_groups.csv` 基准的 `--start-index`，并可在隔离副本中导出 node04 后段 bbox candidate。该输出只用于 bbox-region diagnostic，不是 pixel-accurate target mask、identity proof、正式 synthetic target mask 或 final 4D geometry。
- 2026-05-29 已新增 actor-id instance encoding 审计：`tools/carla_air/audit_actor_id_instance_encoding.py`，对应报告 `research/reports/carla_air_actor_id_instance_mapping_audit_2026_05_29_zh.md`。`local/carla_air/tmp/actor_id_instance_encoding_audit_20260529.json` 显示在 POC03 node03/node04 前 10 帧、三路相机、6 种 packed instance-id 编码假设下，CARLA actor id `24` 没有出现在 `instance_synth` PNG 中；node03 projected bbox ROI 里按当前 contract 的 top values 仍是 `65027`、`59907`、`44800`、`5387`、`7433`。单通道分量值 `24` 只作为 channel diagnostic，不是 actor-id mapping proof。该工具只写 local/tmp JSON/CSV，不修改 scene_dir。
- 2026-05-29 已新增 controlled actor instance-id calibration probe：`tools/carla_air/probe_instance_id_calibration_actor.py`，对应报告 `research/reports/carla_air_controlled_instance_id_calibration_probe_2026_05_29_zh.md`。正例输出 `local/carla_air/tmp/instance_id_calibration_actor_probe_20260529/report.json`：受控 `vehicle.tesla.model3` actor `id=345` 在 3 帧 ROI 中可由 `green_plus_256_blue` 解码恢复，`roi_actor_id_pixel_count=9532`，但当前 formal conversion contract `contract_red_plus_256_green` 下为 0；probe 运行后 `cleanup_remaining_after.count=0`。随后复跑 POC03 close actor-relative probe `local/carla_air/tmp/actor_instance_camera_binding_probe_node03_row0_alt_decode_20260529/report.json`，actor `24` 的 ROI 在替代解码下主要为 `61871` / `35093` 而非 `24`；fixed-camera 复核 `local/carla_air/tmp/actor_id_instance_encoding_audit_after_calibration_20260529.json` 仍为 `status=actor_id_not_observed_in_tested_instance_png_encodings`。该结果只提供 runtime instance PNG 编码线索，不证明 `44800` / `5387` 是 target identity，不是 pixel-accurate mask、正式 synthetic annotation 或 real 4D geometry。
- 2026-05-29 已新增 CARLA blueprint inventory 诊断：`tools/carla_air/probe_carla_blueprint_inventory.py`，对应报告 `research/reports/carla_air_airsim_drone_blueprint_spawn_probe_2026_05_29_zh.md`。当前 runtime 端口 `2000/41451` 在线，但 `probe_instance_id_calibration_actor.py --actor-blueprint-filter 'airsim.*' --preferred-actor-blueprint airsim.drone` 因 `No CARLA actor blueprints matched: airsim.*` 失败；`local/carla_air/tmp/carla_blueprint_inventory_probe_airsim_drone_20260529.json` 显示 `blueprint_count=220`、`exact_blueprint_present["airsim.drone"]=false`、`matched_blueprints_by_filter["airsim.*"]=[]`、`drone/uav/quad/simple/flight/pawn` token 均无命中。因此 AirSim drone live actor 虽可通过 world actor / pose probe 绑定，但当前不作为可 spawn CARLA blueprint 暴露，不能用普通 controlled actor calibration 直接覆盖 POC03 drone `actor_id=24`。该诊断只写 local JSON，不 spawn actor、不改 scene/contract/mask/tracklet/points/eval，也不是 identity proof、pixel-accurate annotation、formal synthetic annotation 或 real/final geometry。
- 2026-05-29 已新增 POC03 target-selection readiness gate：`tools/carla_air/audit_poc03_target_selection_readiness.py`，对应报告 `research/reports/carla_air_poc03_target_selection_readiness_gate_2026_05_29_zh.md`。该工具只读聚合 actor/pose binding、actor-id instance encoding、close actor-relative ROI、mask-quality、CARLA blueprint inventory 与 POC03 plan-only readiness；输出 `local/carla_air/tmp/poc03_target_selection_readiness_audit_20260529.json` 为 `ok=true`、`target_selection_ready=false`、`trusted_target_id_or_tag_available=false`、`ready_for_formal_conversion=false`、`formalization_ready=false`、`goal_complete=false`。`5387` 与 `44800` 均被标记为 `trusted_for_formal_conversion=false`、`identity_proof=false`、`pixel_accurate=false`。该 gate 不选择 target、不执行 conversion、不写 scene/contract/mask/tracklet/points/embedding/eval，只把“当前无可信 target id/tag”变成可复跑 blocker。
- 2026-05-29 已新增 semantic-lidar actor-index 诊断：`tools/carla_air/probe_semantic_lidar_actor_idx.py`，对应报告 `research/reports/carla_air_semantic_lidar_actor_idx_probe_2026_05_29_zh.md`。`local/carla_air/tmp/semantic_lidar_actor_idx_probe_node03_row0_range120_20260529.json` 显示固定三相机 long-range probe 没有 `object_idx=24` 命中，但目标上方/下方 2m close-offset probe 分别命中 6567 / 6722 个 `object_idx=24` 点，`object_tag=42`。该结果说明 CARLA semantic lidar 可提供 actor-level 3D hit evidence，但不是 camera pixel mask、NeoVerse reconstruction 或正式 `points_by_timestamp`。
- 2026-05-29 已新增 semantic-lidar actor points isolated export：`tools/carla_air/export_semantic_lidar_actor_points.py`，对应报告 `research/reports/carla_air_semantic_lidar_actor_points_export_2026_05_29_zh.md`。该工具 replay recorded AirSim pose，在 actor-relative offset 位置临时 spawn `sensor.lidar.ray_cast_semantic`，只导出 `object_idx == actor_id` 的点到 caller 指定 `points_by_timestamp` root，source 写为 `carla_semantic_lidar_actor_idx_v1`；默认输出必须在仓库 `local/` 下，非 local 输出需要显式 `--allow-nonlocal-output`。当前 POC03 node03/node04 前 10 帧分别导出 10 个 `.npy`：node03 总点数 54474、单帧 5213-5535 点，node04 总点数 54094、单帧 5180-5559 点；输出均在 `local/carla_air/tmp/semantic_lidar_actor_points_*_20260529/`，`diagnostic_only=true`、`formal_scene_outputs_modified=false`、`updates_pipeline_contract=false`。官方 verifier `local/carla_air/tmp/poc03_synthetic_verification_after_semantic_lidar_actor_points_pair_20260529.json` 仍 `ok=true`、`failure_count=0`，scene audit 仍为 7 scenes / 2 synthetic depth geometry ready / 5 skeleton pending / QC artifact count 0，说明正式 scene outputs 未被该候选污染。
- 2026-05-29 已新增 semantic-lidar actor points candidate verifier：`tools/carla_air/verify_semantic_lidar_actor_points_candidate.py`，对应报告 `research/reports/carla_air_semantic_lidar_actor_points_candidate_verifier_2026_05_29_zh.md`。该工具只读核查 isolated candidate root 的 `meta.json`、`index.csv`、`points_index.csv`、`.npy` sha256/shape/点数和 diagnostic guard，拒绝 `synthetic_depth_backproject`、`masks_synth`、`tracklets_synth`、camera pixel mask、NeoVerse reconstruction 或 formal scene update 混用。当前输出 `local/carla_air/tmp/semantic_lidar_actor_points_candidate_verification_20260529.json` 为 `ok=true`、`failure_count=0`、`point_root_count=2`；node03 root 为 10 个 timestamp / 54474 点，node04 root 为 10 个 timestamp / 54094 点。该结果只证明 candidate root 自洽，不把 semantic-lidar actor-relative points 升级为正式 scene input、official verifier 输入或 final real geometry。
- 2026-05-29 semantic-lidar actor points verifier 已补 scene lineage guard，报告为 `research/reports/carla_air_semantic_lidar_scene_lineage_guard_2026_05_29_zh.md`。`tools/carla_air/verify_semantic_lidar_actor_points_candidate.py` 现在反查 candidate `meta.scene_dir` 下的 `capture_meta.json` 与 `trajectory_frame_groups.csv`，要求 scene/node/identity/timestamp/node_dir、observation planned frame 与由 recorded pose + sensor placement 复算的 sensor pose 一致；默认正例 `local/carla_air/tmp/semantic_lidar_actor_points_candidate_verification_scene_lineage_exit_20260529.json` 为 `ok=true`、`failure_count=0`，timestamp 篡改负例 `local/carla_air/tmp/semantic_lidar_actor_points_lineage_negative_20260529/report_exit.json` 为 `ok=false`、`failure_count=5` 且默认 `RC=1`。该 guard 只防 candidate root 与 scene lineage 错配，不把 actor-relative semantic-lidar points 升级为 fixed-camera geometry、NeoVerse reconstruction 或 final real 4D geometry。
- 2026-05-29 已新增 close actor-relative instance/semantic camera binding probe：`tools/carla_air/probe_actor_instance_camera_binding.py`，对应报告 `research/reports/carla_air_actor_instance_camera_binding_probe_2026_05_29_zh.md`。该工具 replay recorded AirSim pose，在目标 actor 近处临时 spawn `sensor.camera.instance_segmentation` 与 `sensor.camera.semantic_segmentation`，只输出 `local/carla_air/tmp/actor_instance_camera_binding_probe_*` 诊断 PNG / JSON，不写 masks、tracklets、pipeline contract、embeddings 或 eval。当前 node03 row0 上下 probe 为 `observation_count=2`、`capture_ok_count=2`；node03 below limit10 ROI 中 `5387=297499` px、`44800=67892` px、semantic `11=297217` px；node04 below limit10 ROI 中 `5387=298083` px、`44800=68077` px、semantic `11=298034` px。该结果增强 `44800` 在 close actor ROI 内稳定出现的几何候选证据，但仍不是 actor id `24` 到 `44800` / `5387` 的映射证明，不是 fixed-camera capture geometry、pixel-accurate mask、synthetic annotation 或 real 4D geometry。
- 2026-05-29 已新增 CARLA camera GBuffer actor-to-pixel 诊断：`tools/carla_air/probe_actor_gbuffer_binding.py`，对应报告 `research/reports/carla_air_actor_gbuffer_binding_probe_2026_05_29_zh.md`。该工具 replay 指定 `trajectory_frame_groups.csv` 行，在 fixed camera 或 actor-relative 临时 RGB camera 上尝试通过 `listen_to_gbuffer` 捕获 `SceneStencil`、`CustomDepth`、`CustomStencil` 与可选 `GBufferA`；同日新增 `--include-basic-gbuffers`，可额外请求 `SceneColor` / `SceneDepth` 区分 GBuffer callback 整体不可用和仅 actor-binding buffer 不可用。所有输出只写 `local/carla_air/tmp/actor_gbuffer_binding_probe_*` report/诊断 artifacts，不写 masks、tracklets、pipeline contract、embeddings 或 eval。当前 node03 row0 actor-relative 与 fixed-cam cam0 basic-buffer probe 均为 runtime 端口在线、row replay 成功、actor `24` 与 recorded pose 毫米级对齐，但 `captured_buffer_names=[]`、`basic_gbuffer_capture_ready=false`、`actor_binding_gbuffer_capture_ready=false`；因此当前 CARLA-Air runtime/Python GBuffer 路径不能作为 actor-to-pixel evidence、identity proof 或 pixel-accurate mask。
- 2026-05-29 已新增 CARLA raycast/project_point visibility 诊断：`tools/carla_air/probe_actor_raycast_visibility.py`，对应报告 `research/reports/carla_air_actor_raycast_visibility_probe_2026_05_29_zh.md`。live 输出 `local/carla_air/tmp/actor_raycast_visibility_probe_node03_row0_20260529.json` 在端口在线时 replay POC03 node03 row0，actor `24` / `airsim.drone` 距 recorded pose 约 `0.000318677m`；`world_cast_ray=true`、`world_project_point=true`，三路 fixed camera 均可投影 bbox，24 个 sample 的 `cast_ray` / `project_point` 调用均成功，返回 label 为 `Buildings` / `None`。本地 API 只给 `LabelledPoint` semantic label / location，不给 actor id，因此该工具只能做 geometry/occlusion sanity，不关闭 actor-to-pixel blocker，不写 masks、tracklets、points、status、contracts、embeddings 或 eval，也不能把 `44800`、`5387` 或 bbox 升级为 identity proof、pixel-accurate mask、formal annotation 或 real/final geometry。
- 2026-05-29 已新增 CARLA annotation API surface 只读 live probe：`tools/carla_air/probe_carla_annotation_api_surface.py`，对应报告 `research/reports/carla_air_annotation_api_surface_probe_2026_05_29_zh.md`。在 CARLA `2000` 与 AirSim `41451` 已在线时输出 `local/carla_air/tmp/carla_annotation_api_surface_probe_20260529.json`：actor `24` / `airsim.drone` 可见，但 actor 侧只暴露 `semantic_tags` 且无 annotation setter；world 侧只暴露 `set_annotations_traverse_translucency`；`sensor.camera.instance_segmentation`、`sensor.camera.semantic_segmentation`、`sensor.camera.depth` 与 `sensor.lidar.ray_cast_semantic` blueprint 无 instance / segmentation / custom / stencil 相关可设 attribute。该证据已接入 `local/carla_air/tmp/poc03_target_selection_readiness_after_api_surface_20260529.json`，gate 仍 `target_selection_ready=false`、`trusted_target_id_or_tag_available=false`、`ready_for_formal_conversion=false`，并新增 blocker：公开 CARLA Python API surface 没有 actor/world instance-id、segmentation-id 或 custom-stencil setter；`local/carla_air/tmp/goal_execution_matrix_after_api_surface_20260529.json` 仍 `ok=true` 且三条正式路线不可执行。
- 2026-05-29 actor-id -> instance-PNG mapping / AirSim segmentation controllability 核查继续收紧 blocker：本地 CARLA Python API 文档和 `carla/libcarla.pyi` 未发现将 actor id 映射或设置为 `sensor.camera.instance_segmentation` PNG encoded value 的接口；AirSim `simSetSegmentationObjectID` / `simGetSegmentationObjectID` 只是 0-255 object/mesh segmentation ID。`local/carla_air/tmp/target_actor_binding_probe_replay_scan_seg_20260529.json` 显示 replay 后 actor 附近 `SimpleFlight`、`WeatherActor_C_*`、多个 `BP_PIPCamera_C_*`、`ExternalCamera` 的 segmentation id 均为 `-1`；`local/carla_air/tmp/airsim_segmentation_candidate_setget_20260529_clean.json` 对 9 个近邻候选 object 临时 set/get 后 `effective_count=0`。因此当前没有可用的 AirSim object segmentation 控制路径来解释或设置 `44800` / `5387`。
- 2026-05-29 runtime guarded defaults recheck 已记录到 `research/reports/carla_air_runtime_guarded_defaults_recheck_2026_05_29_zh.md`。fresh outputs `local/carla_air/tmp/goal_execution_matrix_runtime_ports_candidate_defaults_final_20260529.json` 与 `local/carla_air/tmp/candidate_formalization_readiness_current_defaults_final_20260529.json` 共同确认：CARLA `2000` open、AirSim `41451` closed，`all_required_ports_open=false`，`can_execute_formal_conversion_now=false`，`can_run_formal_neoverse_now=false`，`can_plan_multi_identity_roster_now=false`，且 `formalization_ready=false` / `goal_complete=false` / `official_synthetic_smoke.ready=false`。这只是 recheck，不改变 `5387`、`44800`、bbox rectangle、semantic-lidar actor-relative points、depth-backprojection baseline、GBuffer / raycast diagnostics 或 weak route 的非正式边界。
- 2026-05-29 node04 fixed-camera visibility 复核：前 10 帧本地投影复算显示 actor 在 node04 三路相机前方但位于画面左侧之外，camera-forward depth 约 `22.56-27.00m`，projected x 范围约 `cam0=[-781,-645]`、`cam1/cam2=[-762,-630]`。`local/carla_air/tmp/actor_bbox_instance_candidate_audit_20260529_limit120.json` 中 node04 `observation_count=0`、`candidate_count=0`、`skipped_count=287`、`missing_input_count=73`，说明当前 raw synthetic window 也不足以支持 node04 bbox/mesh projection formalization。node04 close-camera `44800` 证据不能直接迁移为 fixed-camera formal mask。
- 2026-05-29 已新增保守 row-window / append-window 诊断能力：`export_live_synthetic_annotations.py`、`inspect_synthetic_sensor_ids.py`、`convert_synthetic_sensor_masks.py` 支持 `--start-index`，窗口基准统一为 `trajectory_frame_groups.csv`；`run_poc03_synthetic_reid_pipeline.py` 可透传该窗口。`--append-window` 只用于 `local/` 隔离 scene 副本中的 raw sensor gap filling / inspect / validation 诊断，拒绝进入 formal convert / points / eval / verify，避免污染官方 scene 的已验证 synthetic lineage。
- 2026-05-29 已在普通副本 `local/carla_air/tmp/node04_later_window_scene_fullcopy_20260529/` 上补跑 node04 后段 fixed-camera synthetic sensor：`start_index=93`、`limit=20` 覆盖 `84921711-86821711`；export meta 为 `sensor_export_id=20260529_node04_later_window_isolated_export`、`rows_requested=20`、`rows_written=180`、`failures=[]`。`local/carla_air/tmp/node04_later_window_scene_fullcopy_20260529_candidates.json` 为 `sensor_images_read=120`、`missing_inputs_count=0`；`local/carla_air/tmp/node04_later_window_actor_bbox_instance_candidate_audit_20260529.json` 显示 `observation_count=52`、`candidate_count=8`，top candidate `44800` 的 `observation_count=46`、`timestamp_count=18`、`cam_count=3`、`mean_candidate_bbox_iou=0.412969`、`mean_candidate_area_ratio_to_projection=0.412969`、`mean_candidate_center_distance_px=3.074031`。该证据只说明 node04 后段 fixed-camera window 中 `44800` 是强几何候选，不证明 actor id `24` 映射到 `44800`，不是 pixel-accurate mask、正式 synthetic annotation 或 real geometry。官方 verifier `local/carla_air/tmp/poc03_synthetic_verification_after_isolated_node04_later_export_20260529.json` 仍 `ok=true`、`failure_count=0`；scene audit `local/carla_air/tmp/scene_pipeline_status_audit_after_isolated_node04_later_export_20260529.json` 仍为 7 scenes / 2 synthetic ready / 5 skeleton pending / QC artifact count 0。
- 2026-05-29 已在同一 node04 后段隔离副本导出 actor-bbox candidate：`local/carla_air/tmp/node04_later_window_actor_bbox_annotation_export_20260529.json` 显示 `start_index=93`、`limit=20`、timestamp range `84921711-86821711`、`valid_timestamps=15`、45 张 `masks_actor_bbox_later_window`、`tracklets=tracks/tracklets_actor_bbox_later_window.json`、`source=carla_actor_pose_projected_bbox`、`pixel_accurate=false`、`identity_proof=false`、`updates_contract=false`、`writes_formal_target_masks=false`。对应报告 `research/reports/carla_air_node04_later_window_actor_bbox_candidate_2026_05_29_zh.md`。该结果不是正式 `masks_synth`、不是 identity proof、不是 pixel-accurate annotation、不是 real 4D geometry，也未污染官方 scene：`local/carla_air/tmp/poc03_synthetic_verification_after_node04_later_bbox_candidate_20260529.json` 为 `ok=true`、`failure_count=0`，`local/carla_air/tmp/scene_pipeline_status_audit_after_node04_later_bbox_candidate_20260529.json` 仍为 7 scenes / 2 synthetic ready / 5 skeleton pending / QC artifact count 0。
- 2026-05-29 已冻结候选标注/几何正式化边界：`research/reports/carla_air_candidate_geometry_formalization_boundaries_2026_05_29_zh.md`。当前 contract 可复用 `variants/source/status` 模式，但 bbox rectangle 与 semantic-lidar actor points 必须使用独立 variant/source/status；在没有 actor-to-pixel binding 或 pixel-accuracy evidence 前，必须保留 `pixel_accurate=false`、`identity_proof=false`，不得覆盖 `synthetic_target_masks`、`synthetic_depth_backproject`、`masks_synth` 或 `tracklets_synth`。
- 2026-05-29 `audit_scene_pipeline_status.py` 已新增 candidate artifact guard：报告 `candidate_artifact_policy.checked_not_consumed_as_formal_inputs=true` / `formal_inputs_consumed=false`，并识别 `masks_actor_bbox*`、`tracklets_actor_bbox*`、`embeddings_actor_bbox*` 等候选/调试产物。`local/carla_air/tmp/scene_pipeline_status_audit_with_candidate_guard_20260529.json` 中官方 scene 状态仍为 7 scenes / 2 synthetic ready / 5 skeleton pending / QC artifact count 0；候选产物统计只用于可见性，不改变正式 readiness。
- 2026-05-29 继续执行状态已按 `docs/goal.md` 最新规则刷新并记录到 `research/reports/carla_air_goal_continue_status_2026_05_29_zh.md`。本轮 fresh audits 显示 `local/carla_air/tmp/poc03_target_selection_readiness_goal_continue_20260529.json` 仍 `target_selection_ready=false`、`trusted_target_id_or_tag_available=false`、`ready_for_formal_conversion=false`；`local/carla_air/tmp/goal_execution_matrix_goal_continue_20260529.json` 与 `local/carla_air/tmp/goal_execution_matrix_readonly_rerun_20260529.json` 仍 `can_execute_formal_conversion_now=false`、`can_run_formal_neoverse_now=false`、`can_plan_multi_identity_roster_now=false`；`local/carla_air/tmp/neoverse_geometry_readiness_goal_continue_20260529.json` 仍 `formal_neoverse_ready=false`、`formal_neoverse_output_count=0`，现有 `neoverse_fused` roots 是 proxy source、`carla_air_depth_points` roots 是 depth-backprojection baseline；`local/carla_air/tmp/aircraft_identity_readiness_goal_continue_20260529.json` 仍 `benchmark_ready=false`、`normalized_inventory.benchmark_eligible_count=0`。`local/carla_air/tmp/poc03_synthetic_verification_goal_continue_latest_20260529.json` 虽仍 `ok=true`、`failure_count=0`，但只证明 POC03 separate synthetic depth-backprojection smoke 自洽，不是 pixel-accurate identity proof、NeoVerse reconstruction、final real geometry 或 goal completion。
- 2026-05-29 当前会话 active audit 已记录到 `research/reports/carla_air_goal_active_audit_2026_05_29_zh.md`。本轮从当前 worktree 重新取证，fresh outputs 为 `local/carla_air/tmp/*_goal_active_20260529.json`：`poc03_target_selection_readiness` 仍 `target_selection_ready=false`、`trusted_target_id_or_tag_available=false`；`goal_execution_matrix` 仍 `can_execute_formal_conversion_now=false`、`can_run_formal_neoverse_now=false`、`can_plan_multi_identity_roster_now=false`；`neoverse_geometry_readiness` 仍 `formal_neoverse_ready=false`、`formal_neoverse_output_count=0`；`aircraft_identity_readiness` 仍 `benchmark_ready=false`、`normalized_inventory.benchmark_eligible_count=0`。POC03 synthetic verifier 仍 `ok=true`、`failure_count=0`，但只证明 synthetic depth-backprojection smoke，不是 pixel-accurate identity proof、NeoVerse reconstruction、final geometry 或 goal completion。本轮没有启动 simulator、没有执行 formal conversion、没有改写官方 scene raw synthetic outputs。
- 2026-05-29 继续执行状态 2 已按 `docs/goal.md` 最新规则记录到 `research/reports/carla_air_goal_continue2_status_2026_05_29_zh.md`。本轮没有启动 simulator、没有执行 live export / capture / inspect、没有改写官方 scene raw synthetic outputs；只读 explorer 复核未发现能推进正式路线的新证据，追加 actor-to-pixel 本地线索 explorer 也未发现 actor id -> instance PNG value、segmentation id setter、custom stencil / material route 或 AirSim drone annotation binding。fresh outputs 为 `local/carla_air/tmp/*_goal_continue2_20260529.json`：`poc03_target_selection_readiness` 仍 `target_selection_ready=false`、`trusted_target_id_or_tag_available=false`、`ready_for_formal_conversion=false`；`poc03_synthetic_verification` 仍 `ok=true`、`failure_count=0`，但只证明 separate synthetic depth-backprojection smoke 自洽；`scene_pipeline_status_audit` 仍为 7 scenes / 2 synthetic smoke ready / 5 skeleton pending，candidate artifacts 不作为 formal inputs；`candidate_formalization_readiness` 仍 `formalization_ready=false`、`goal_complete=false`；修复后的 `goal_execution_matrix_goal_continue2_20260529.json` 为 `ok=true`、`missing_or_unreadable=[]`，三条正式路线仍分别为 `can_execute_formal_conversion_now=false`、`can_run_formal_neoverse_now=false`、`can_plan_multi_identity_roster_now=false`。该刷新不改变边界：`5387`、`44800`、bbox rectangle、semantic-lidar actor-relative points、depth-backprojection baseline、GBuffer / raycast diagnostics 与 weak route 都不能升级为 identity proof、pixel-accurate mask、formal annotation、NeoVerse reconstruction、final geometry 或 benchmark completion。
- 2026-05-29 新增 actor-bbox formalization preflight：`tools/carla_air/audit_actor_bbox_formalization_preflight.py`，对应报告 `research/reports/carla_air_actor_bbox_formalization_preflight_2026_05_29_zh.md`，输出 `local/carla_air/tmp/actor_bbox_formalization_preflight_20260529.json` 为 `ok=true`、`can_continue_as_independent_bbox_annotation_variant=true`，但 `formal_annotation_replacement_ready=false`、`ready_for_masks_synth_or_tracklets_synth=false`、`identity_proof=false`、`pixel_accurate=false`、`real_or_final_geometry=false`、`goal_complete=false`。该 preflight 说明 actor-bbox route 可以继续作为独立非 pixel-accurate bbox annotation variant 工程化，但不得写入 `masks_synth` / `tracklets_synth`，不得称为 identity proof、pixel-accurate mask、NeoVerse reconstruction、final geometry 或 benchmark evidence。同轮 multi-identity explorer 复核确认 `assets/models/aircraft_normalized/` 仍无真实 identity，`benchmark_eligible_count=0`；下一步只能先生成真实 normalized asset 后重跑 readiness gate，不能用 raw assets 或占位文件替代。
- 2026-05-29 当前 worktree re-audit 已记录到 `research/reports/carla_air_current_worktree_reaudit_2026_05_29_zh.md`。本轮按仓库级 subagent 策略并行调用两个只读 explorer，分别核查 research/docs 侧与 `mvp-demo` / `tools/carla_air` / runtime artifacts 侧；主 agent 复跑 current-session fresh audits 后确认 POC03 separate synthetic depth-backprojection smoke 仍 `ok=true` / `failure_count=0`，但 target selection 仍 `ready_for_formal_conversion=false`，NeoVerse readiness 仍 `formal_neoverse_ready=false`，aircraft identity readiness 仍 `benchmark_ready=false` / `benchmark_eligible_count=0`。最终串行重跑的 `local/carla_air/tmp/goal_execution_matrix_current_session_20260529.json` 为 `ok=true`、`missing_or_unreadable=[]`，三条正式路线仍为 `can_execute_formal_conversion_now=false`、`can_run_formal_neoverse_now=false`、`can_plan_multi_identity_roster_now=false`。本轮没有启动 simulator、没有 live export / capture / inspect、没有改写官方 scene raw synthetic outputs，也没有执行 formal conversion / NeoVerse / benchmark 写入。
- 2026-05-29 multi-identity 支线新增 local normalized identity POC：`tools/carla_air/build_local_normalized_aircraft_identity.py` 从 `assets/models/aircraft_raw/dji_drone_fbx_obj/extracted/modelNew.obj` 生成 `assets/models/aircraft_normalized/dji_drone_fbx_obj_local_poc/normalized.fbx`、`preview.png`、`asset_meta.json` 与 `build_status.json`。报告为 `research/reports/carla_air_local_normalized_identity_poc_2026_05_29_zh.md`。该 POC 真实写出 normalized asset attempt，但 MeshLab readback 对生成 FBX segmentation fault，且 source/license benchmark permission 与 UE import 均未验证；因此 `check_aircraft_identity_readiness.py` 已收紧 optional `build_status.json` readback/import guard，最新 `local/carla_air/tmp/aircraft_identity_readiness_after_local_normalized_readback_guard_20260529.json` 仍 `ok=false`、`benchmark_ready=false`，normalized inventory 为 `identity_count=1`、`technical_ready_count=0`、`local_poc_eligible_count=0`、`benchmark_eligible_count=0`。该结果不关闭 multi-identity blocker，只把“有 raw assets”推进到“有可审计但失败的 normalized identity attempt”。
- 2026-05-29 current follow-up readiness 已记录到 `research/reports/carla_air_current_followup_readiness_2026_05_29_zh.md`。本轮未新增 subagent，因为前一轮已完成 research/docs 与 `mvp-demo` / `tools/carla_air` / outputs 两侧只读 explorer 核查，本轮只做聚焦 gate 重跑和工具链探针。fresh outputs 为 `local/carla_air/tmp/*_current_followup_20260529.json`；`goal_execution_matrix_current_followup_20260529.json` 仍 `ok=true`、`missing_or_unreadable=[]`、`can_execute_formal_conversion_now=false`、`can_run_formal_neoverse_now=false`、`can_plan_multi_identity_roster_now=false`。Meshlab 新探针显示 raw FBX `dji_drone_fbx_obj/extracted/modelNew.fbx` 与 `dji_drone_fbx_abc/.../Drone_fb.FBX` 可读取并导出 OBJ，但 raw OBJ importer abort，Meshlab FBX re-export/roundtrip 不能保存可回读 FBX；本机仍缺 Blender / assimp / fbx2gltf / UE / bpy。该结果只增加 raw-source inspect evidence，不能把 local normalized POC 或 raw assets 升级为 benchmark identities；`aircraft_identity_readiness_current_followup_20260529.json` 仍 `benchmark_ready=false`、`technical_ready_count=0`、`benchmark_eligible_count=0`。本轮未启动 simulator、未执行 formal conversion / NeoVerse / benchmark 写入。
- 2026-05-29 已新增 actor-bbox candidate annotation verifier：`tools/carla_air/verify_actor_bbox_candidate_annotations.py`，对应报告 `research/reports/carla_air_actor_bbox_candidate_annotation_verifier_2026_05_29_zh.md`。默认核查 node03 前段 official scene candidate 与 node04 后段 `local/` 隔离副本 candidate，输出 `local/carla_air/tmp/actor_bbox_candidate_annotation_verification_20260529.json` 为 `ok=true`、`failure_count=0`、`candidate_count=2`；node03 为 10 个 timestamp / 30 张 mask，node04 为 15 个 timestamp / 45 张 mask，均为 `source=carla_actor_pose_projected_bbox`、`pixel_accurate=false`、`identity_proof=false`、`updates_contract=false`、`writes_formal_target_masks=false`。负例 `local/carla_air/tmp/actor_bbox_candidate_annotation_verification_negative_formal_tracklets_20260529.json` 使用 `tracks/tracklets_synth.json` 并按预期 `ok=false`。该 verifier 只证明 candidate 自洽并防误用，不把 bbox rectangle 升级为正式 synthetic annotation、identity proof、pixel-accurate mask 或 final real geometry。
- 2026-05-29 已新增 candidate formalization readiness 聚合审计：`tools/carla_air/audit_candidate_formalization_readiness.py`，对应报告 `research/reports/carla_air_candidate_formalization_readiness_audit_2026_05_29_zh.md`。输出 `local/carla_air/tmp/candidate_formalization_readiness_audit_20260529.json` 为 `ok=true`、`formalization_ready=false`、`goal_complete=false`；官方 POC03 synthetic smoke 与 scene candidate policy 自洽，且已消费 weak diagnostic ReID smoke verifier 与 weak writer dry-run。weak writer dry-run 为 `ok=true`、`writes_scene_outputs=false`、`modifies_pipeline_contract=false`、`writer_stage=dry_run_plan_only`，但仍被记录为 blocker 而不是 real writer / official verifier / promotion evidence；actor-bbox 缺 actor-to-pixel / pixel-accuracy evidence，semantic-lidar actor points 仍是 actor-relative diagnostic geometry，multi-identity benchmark `benchmark_eligible_count=0`，POC03 plan-only 缺 trusted target id/tag。该审计不修改 scene、contract、mask、tracklet、points、embedding 或 eval，也不把 candidate 升级为正式输入。
- 2026-05-29 已新增 weak variant contract plan：`research/reports/carla_air_weak_variant_contract_plan_2026_05_29_zh.md`。该 plan 只定义 bbox / semantic-lidar 独立 weak variant 的字段、guard 与 promotion 条件，当前不修改 scene、contract、mask、tracklet、points、embedding 或 eval。结论保持 `formalization_ready=false`、`goal_complete=false`；禁止复用 `masks_synth`、`tracklets_synth`、`annotation_meta_synth`、`synthetic_target_masks` 或 `synthetic_depth_backproject`。下一步如果继续该支线，应先做 plan-only / dry-run contract linter，在 `local/` 生成 proposed patch JSON，由主 agent review 后再决定是否进入 writer 阶段。
- 2026-05-29 已新增 weak variant contract patch linter：`tools/carla_air/plan_weak_variant_contract_patch.py`，对应报告 `research/reports/carla_air_weak_variant_contract_patch_linter_2026_05_29_zh.md`。正例输出 `local/carla_air/tmp/weak_variant_contract_patch_plan_20260529.json` 为 `ok=true`、`writes_scene_outputs=false`、`formalization_ready=false`、`goal_complete=false`；bbox weak proposed variant 汇总 2 个 candidate、25 个 timestamp、75 张 mask，semantic-lidar weak proposed variant 汇总 2 个 root、20 个 timestamp、108568 点。缺输入负例 `ok=false` 并非零退出，非 `local/` 输出默认被拒绝。该工具不是 writer，不读取或修改 `pipeline_contract.json`。
- 2026-05-29 已新增 weak variant writer dry-run：`tools/carla_air/plan_weak_variant_writer_dry_run.py`，对应报告 `research/reports/carla_air_weak_variant_writer_dry_run_2026_05_29_zh.md`。正例输出 `local/carla_air/tmp/weak_variant_writer_dry_run_plan_20260529.json` 为 `ok=true`、`writes_scene_outputs=false`、`modifies_pipeline_contract=false`、`formalization_ready=false`、`goal_complete=false`；它只基于 patch linter 输出生成 node03/node04 additive weak diagnostic variant 的待审计划、official verifier 拒绝规则与 downstream opt-in 参数。负例确认非 `local/` 输出默认拒绝，输入 plan 若把 `formalization_ready=true` 则失败。该工具不是 real writer，不读取或修改 `pipeline_contract.json`，不写 masks / tracklets / points / embeddings / eval，也不把 bbox / semantic-lidar candidate 升级为 formal synthetic annotation、identity proof、pixel-accurate mask、real/final geometry、NeoVerse reconstruction 或 benchmark evidence。
- 2026-05-29 已新增 weak variant official-readiness verifier：`tools/carla_air/verify_weak_variant_official_readiness.py`，对应报告 `research/reports/carla_air_weak_variant_official_readiness_verifier_2026_05_29_zh.md`。正例输出 `local/carla_air/tmp/weak_variant_official_readiness_verification_20260529.json` 为 `ok=true`、`failure_count=0`、`verifier_scope=pre_writer_readiness_only`、`writes_scene_outputs=false`、`modifies_pipeline_contract=false`、`formalization_ready=false`、`goal_complete=false`、`real_writer_implemented=false`、`promotion_evidence_satisfied=false`。该工具只交叉核查 weak writer dry-run、weak diagnostic ReID smoke verifier 与 candidate readiness audit 的 non-promotion guard；负例确认非 `local/` 输出、dry-run 篡改为修改 contract、readiness 缺失 weak writer summary 都会失败。该 verifier 不是 real writer，不修改 contract，不证明 identity / pixel accuracy / final geometry，也不解除 multi-identity blocker。
- 2026-05-29 已完成 local-only weak diagnostic ReID smoke：报告为 `research/reports/carla_air_weak_variant_diag_reid_smoke_2026_05_29_zh.md`，只读 verifier 为 `tools/carla_air/verify_weak_variant_diag_reid_smoke.py`，正例输出 `local/carla_air/tmp/weak_variant_diag_reid_smoke_verification_20260529.json` 为 `ok=true`、`failure_count=0`、`formalization_ready=false`、`goal_complete=false`。本次只在 `local/` 隔离副本中把 actor-bbox candidate masks 与 `carla_semantic_lidar_actor_idx_v1` actor-level points 组合为 `tracklets_actor_bbox_semantic_lidar_diag.json`，node03 为 10 个 timestamp / 30 张 bbox mask / 10 个 fused points 文件，node04 later-window 为 15 个 timestamp / 45 张 bbox mask / 15 个 fused points 文件；两侧 `embeddings_actor_bbox_semantic_lidar_diag/tracks.npy` shape 均为 `[1,161]`，双向 eval `mAP=1.0`、`recall_at_1/5/10=1.0`、`num_queries=1`、`num_gallery=1`。该 smoke 只证明 isolated weak-route operability，不更新 official contract，不替代 `masks_synth` / `tracklets_synth`，不是 identity proof、pixel-accurate mask、final real geometry、NeoVerse reconstruction、正式 benchmark 或 goal completion 证据。
- 2026-05-29 已新增 downstream weak diagnostic opt-in guard：公共 guard 为 `mvp-demo/scripts/carla_air_weak_variant_guard.py`，并接入 `mvp-demo/scripts/extract_node_track_embeddings.py` 与 `mvp-demo/scripts/eval_node_track_retrieval.py`。任何下游消费 `actor_bbox_semantic_lidar_diag`、`masks_actor_bbox`、`semantic_lidar_actor_points` 或 `carla_semantic_lidar_actor_idx_v1` 相关路径/JSON，必须显式传 `--weak-variant actor_bbox_semantic_lidar_diag` 与 `--weak-variant-readiness local/carla_air/tmp/weak_variant_official_readiness_verification_20260529.json`；eval 还要求 weak embeddings 带新版 `weak_variant` metadata，且 weak eval 输出留在 repo `local/`。正例生成 `embeddings_actor_bbox_semantic_lidar_diag_guarded_20260529` 与双向 guarded eval，负例确认无 flag、无 readiness、坏 readiness、旧 weak metadata、非 local eval 输出均拒绝。报告为 `research/reports/carla_air_weak_variant_downstream_guard_2026_05_29_zh.md`。该 guard 只防止默认误用，不是 real writer、official contract update、promotion evidence 或正式 benchmark。
- 2026-05-29 已新增 weak variant contract-update verifier：`tools/carla_air/verify_weak_variant_contract_update.py`，对应报告 `research/reports/carla_air_weak_variant_contract_update_verifier_2026_05_29_zh.md`。正例输出 `local/carla_air/tmp/weak_variant_contract_update_verification_20260529.json` 为 `ok=true`、`failure_count=0`、`scene_count=7`、`registered_weak_variant_count=0`、`weak_marker_scene_count=0`，确认当前官方 `pipeline_contract.json` 未登记或污染 weak diagnostic variant；同时核查 weak writer dry-run 仍是 `dry_run_plan_only`，official readiness 仍为 `real_writer_implemented=false` / `promotion_evidence_satisfied=false`。负例确认 dry-run 篡改为 `formalization_ready=true` 会失败，weak marker 写入 `synthetic_target_masks` / `synthetic_annotations.target_masks` 会失败。该 verifier 只读，不写 scene outputs，不修改 `pipeline_contract.json`，不实现 real writer，不提供 promotion evidence，也不解除 actor-to-pixel / pixel accuracy / final geometry / multi-identity blocker。
- 2026-05-29 已新增 protected weak contract writer dry-run：`tools/carla_air/apply_weak_variant_contract_update.py`，对应报告 `research/reports/carla_air_weak_variant_contract_update_writer_dry_run_2026_05_29_zh.md`。默认只读取 weak writer dry-run、contract-update verifier 与官方 contracts，生成待审 `weak_diagnostic_variants` diff，不修改 `pipeline_contract.json`；正例 `local/carla_air/tmp/weak_variant_contract_update_writer_dry_run_20260529.json` 为 `ok=true`、`dry_run=true`、`planned_update_count=2`、`written_update_count=0`，只计划 POC03 node03/node04 的 `actor_pose_projected_bbox_weak` 与 `semantic_lidar_actor_idx_weak`。真实写入必须同时传 `--execute`、`--enable-official-contract-update` 与当前 verifier sha256；负例 `local/carla_air/tmp/weak_variant_contract_update_writer_execute_blocked_20260529.json` 显示单独 `--execute` 被拒绝、`written_update_count=0`，写后复核 `registered_weak_variant_count=0`。该 writer scaffold 不是 promotion evidence，未执行 official contract integration，也不解除 `formalization_ready=false` / `goal_complete=false`。
- 2026-05-29 已新增 inspect instance decode 诊断增强：`tools/carla_air/inspect_synthetic_sensor_ids.py` 新增默认关闭的 `--include-instance-decode-diagnostics`，只在 `instance_decode_diagnostics` 下输出 alternate packed decode 与 single-channel component summaries；正式 `candidates.instance` 仍使用 `contract_red_plus_256_green` / `red + 256 * green`。报告为 `research/reports/carla_air_instance_decode_diagnostic_inspect_enhancement_2026_05_29_zh.md`；默认输出 `local/carla_air/tmp/inspect_instance_default_no_decode_diag_node03_20260529.json` 不含诊断字段，显式输出 `local/carla_air/tmp/inspect_instance_decode_diagnostics_node03_20260529.json` 含 10 个 decode summaries，且两者 `candidates.instance` 完全一致。负例 `local/carla_air/tmp/validate_decode_diagnostic_not_formal_candidate_negative_20260529.json` 为 `ok=false`、`failure_count=1`、失败项 `target_candidate_present`，确认 alternate decode value 不会被 formal validation 消费；正例 `local/carla_air/tmp/validate_decode_diagnostic_formal_candidate_positive_20260529.json` 为 `ok=true`，确认新增诊断字段不破坏既有 validation。`run_poc03_synthetic_reid_pipeline.py` 同步新增 plan/inspect passthrough `--inspect-instance-decode-diagnostics`，但不改变 convert/points/eval/verify 语义。该增强只让诊断证据可见，不证明 actor `24` 已绑定，也不能升级 `44800`、`5387`、`61871`、`35093` 或单通道 component 值。
- 2026-05-29 已新增 tracklet input contract 审计：`tools/carla_air/audit_tracklet_input_contract.py` 与报告 `research/reports/carla_air_tracklet_input_contract_audit_2026_05_29_zh.md`。默认官方扫描输出 `local/carla_air/tmp/tracklet_input_contract_audit_20260529.json` 为 `ok=true`、`formalization_ready=false`、`goal_complete=false`、`tracklet_count=5`，分类为 2 个 `proxy_minimal_formal`、2 个 `synthetic_depth_backproject_smoke`、1 个 `actor_bbox_candidate`；加入本地 weak diagnostic tracklet 后输出 `local/carla_air/tmp/tracklet_input_contract_audit_with_weak_diag_20260529.json` 为 `tracklet_count=6`，新增 1 个 `weak_actor_bbox_semantic_lidar_diagnostic`。该工具只读，不生成或修改 tracklet，不写 scene outputs，不进行 formal promotion；所有 tracklet 仍为 `ready_for_final_annotation=false` / `ready_for_final_geometry=false`，只能作为下游消费前的防误用 preflight。
- 2026-05-29 已新增 instance PNG mapping 本地 API 审计：报告为 `research/reports/carla_air_instance_png_mapping_local_api_audit_2026_05_29_zh.md`。本地 `sensor_gallery.py` 明确写明 `object_id = R + 256*G (BGRA)`、`B = semantic tag`，支持当前正式 decode contract；但 `python_api.md`、`libcarla.pyi` 与 AirSim segmentation set/get 证据均未提供 actor id -> instance PNG value 的 setter / lookup，因此仍不能把 actor `24` 绑定到 `44800` 或 `5387`。
- 2026-05-29 续跑 readiness 复核已记录到 `research/reports/carla_air_goal_resume_readiness_rerun_2026_05_29_zh.md`。本轮不重置长任务，按“积极但保守”策略委派 actor-to-pixel 本地源码/API 只读 explorer；结论仍是本地只见 GBuffer/CustomStencil 与 AirSim segmentation setter/getter 入口，没有 actor id -> instance segmentation PNG value 的映射、setter 或 lookup。只读/plan 输出显示 `local/carla_air/tmp/poc03_target_selection_readiness_rerun_20260529.json` 仍 `target_selection_ready=false`、`trusted_target_id_or_tag_available=false`，`local/carla_air/tmp/candidate_formalization_readiness_rerun_20260529.json` 仍 `formalization_ready=false`，`local/carla_air/tmp/aircraft_identity_readiness_rerun_20260529.json` 仍 `benchmark_ready=false` / `benchmark_eligible_count=0`；`local/carla_air/tmp/poc03_synth_plan_inspect_decode_diag_rerun_20260529.json` 只是 plan-only inspect 诊断，端口在线不等于 target validation 或 formal conversion 就绪。没有新 target evidence 时，不应仅因端口 open 就执行 `--execute --stop-after inspect --overwrite` 改写官方 scene raw synthetic outputs。
- 2026-05-29 已执行受保护 weak diagnostic contract update，报告为 `research/reports/carla_air_weak_variant_contract_update_execute_2026_05_29_zh.md`。执行前 `local/carla_air/tmp/weak_variant_contract_update_verification_rerun_pre_execute_20260529.json` 为 `ok=true`、`registered_weak_variant_count=0`，dry-run diff 经主 agent review 后只新增 `weak_diagnostic_variants` 且所有 guard 保持 false；真实执行输出 `local/carla_air/tmp/weak_variant_contract_update_writer_execute_20260529.json` 为 `ok=true`、`written_update_count=2`、`formalization_ready=false`、`goal_complete=false`、`not_identity_proof=true`、`not_pixel_accurate_mask_evidence=true`、`not_real_or_final_geometry=true`。写后 verifier `local/carla_air/tmp/weak_variant_contract_update_verification_post_execute_20260529.json` 为 `ok=true`、`failure_count=0`、`registered_weak_variant_count=4`、`weak_marker_scene_count=2`；只在 POC03 node03/node04 的 `pipeline_contract.json` 中登记 `actor_pose_projected_bbox_weak` 与 `semantic_lidar_actor_idx_weak`，不写 `masks_synth` / `tracklets_synth`，不解除 actor-to-pixel、pixel accuracy、final geometry 或 multi-identity blocker。
- 2026-05-29 已新增 weak variant downstream contract guard：报告为 `research/reports/carla_air_weak_variant_downstream_contract_guard_2026_05_29_zh.md`。weak diagnostic embedding/eval 现在必须额外传 `--weak-contract-verification local/carla_air/tmp/weak_variant_contract_update_verification_post_execute_20260529.json`，并把 post-write verifier evidence 写入 `weak_variant.contract_verification` metadata；`verify_weak_variant_diag_reid_smoke.py` 也可用 `--weak-contract-verification` 与 `--embeddings-subdir` 核验该 evidence。正例 `local/carla_air/tmp/weak_contract_guard_diag_reid_verification_20260529.json` 为 `ok=true`、`failure_count=0`，旧 weak outputs 负例 `local/carla_air/tmp/weak_contract_guard_diag_reid_negative_old_outputs_20260529.json` 为 `ok=false`、`failure_count=8`。该 guard 只是 provenance / non-promotion check，不是 formal benchmark、promotion evidence、identity proof、pixel-accurate mask 或 real/final geometry。
- 2026-05-29 已新增 NeoVerse geometry readiness audit：`tools/carla_air/audit_neoverse_geometry_readiness.py` 与报告 `research/reports/carla_air_neoverse_geometry_readiness_2026_05_29_zh.md`。输出 `local/carla_air/tmp/neoverse_geometry_readiness_20260529.json` 为 `ok=true`、`formal_neoverse_ready=false`、`goal_complete=false`、`writes_scene_outputs=false`、`updates_pipeline_contract=false`、`depth_backprojection_baseline_ready=true`、`depth_backprojection_is_neoverse_reconstruction=false`、`formal_neoverse_output_count=0`。审计确认 `mvp-demo/output/neoverse_fused` 当前两个 POC03 root 仍是 `carla_air_minimal_formal_proxy_box_surface_v1` proxy points，`mvp-demo/output/carla_air_depth_points` 两个 root 是 `carla_depth_synth_mask_backprojection_v1` baseline。2026-05-30 复核确认 `third_party/NeoVerse/models/NeoVerse/reconstructor.ckpt` 已通过软链接接入且目标文件大小为 `6039277930` bytes；checkpoint 缺失 blocker 已移除，但仍没有 formal NeoVerse points root，target-selection / candidate formalization 仍未 ready。该工具只读，不运行 runtime，不写 scene outputs，不把 proxy points、depth-backprojection baseline、semantic-lidar actor-relative diagnostic points 或 weak diagnostic eval 升级为 NeoVerse / final real geometry。
- 2026-05-29 已新增 long-task goal execution matrix：`tools/carla_air/audit_goal_execution_matrix.py` 与报告 `research/reports/carla_air_goal_execution_matrix_2026_05_29_zh.md`。输出 `local/carla_air/tmp/goal_execution_matrix_20260529.json` 为 `ok=true`、`formalization_ready=false`、`goal_complete=false`、`writes_scene_outputs=false`、`updates_pipeline_contract=false`、`starts_runtime=false`、`can_execute_formal_conversion_now=false`、`can_run_formal_neoverse_now=false`、`can_plan_multi_identity_roster_now=false`。该矩阵聚合 target selection、NeoVerse readiness、identity readiness、weak diagnostic contract 与 tracklet audit，明确 target-to-pixel、formal NeoVerse、multi-identity 三条正式路线仍 blocked，weak diagnostic 只能 guarded diagnostic；只给下一步命令与禁止动作，不执行 runtime、conversion、NeoVerse 或 benchmark。
- 2026-05-29 默认 goal execution matrix 已对齐当前权威 gate：报告为 `research/reports/carla_air_goal_execution_matrix_current_defaults_2026_05_29_zh.md`，`tools/carla_air/audit_goal_execution_matrix.py` 默认 target-selection 输入改为 `local/carla_air/tmp/poc03_target_selection_readiness_controlled_decode_guard_20260529.json`，默认 identity 输入改为 `local/carla_air/tmp/aircraft_identity_readiness_current_multi_local_poc_20260529.json`，默认输出改为 `local/carla_air/tmp/goal_execution_matrix_current_defaults_20260529.json`。新默认矩阵仍 `ok=true`、`target_selection_ready=false`、`trusted_target_id_or_tag_available=false`、`ready_for_formal_conversion=false`、`formal_neoverse_ready=false`、`benchmark_ready=false`、`benchmark_eligible_count=0`、`can_execute_formal_conversion_now=false`、`can_run_formal_neoverse_now=false`、`can_plan_multi_identity_roster_now=false`、`goal_complete=false`；这只是避免默认复跑读取较早 rerun JSON，不放行正式 conversion / NeoVerse / benchmark。
- 2026-05-29 resume 执行状态已记录到 `research/reports/carla_air_goal_resume_execution_status_2026_05_29_zh.md`。fresh audits 仍显示 `local/carla_air/tmp/goal_execution_matrix_resume_20260529.json` 三条正式路线不可执行，`local/carla_air/tmp/poc03_target_selection_readiness_resume_20260529.json` 为 `target_selection_ready=false` / `trusted_target_id_or_tag_available=false`，`local/carla_air/tmp/neoverse_geometry_readiness_resume_20260529.json` 为 `formal_neoverse_ready=false`，`local/carla_air/tmp/aircraft_identity_readiness_resume_20260529.json` 为 `benchmark_ready=false` / normalized identity 0。runtime 端口在线后按规则执行最小 GBuffer live diagnostic `local/carla_air/tmp/actor_gbuffer_binding_probe_node03_row0_actor_relative_resume_20260529/report.json`，row replay 成功且 actor `24` 对齐 recorded pose，但 `SceneStencil` / `CustomDepth` / `CustomStencil` / `SceneColor` / `SceneDepth` / `GBufferA` 全部无 callback；随后 `local/carla_air/tmp/poc03_target_selection_readiness_after_resume_gbuffer_20260529.json` 与 `local/carla_air/tmp/goal_execution_matrix_after_resume_gbuffer_20260529.json` 仍 blocked。本轮没有启动新 simulator、没有执行 formal conversion、没有改写官方 scene raw synthetic outputs。
- 2026-05-29 session re-audit 已记录到 `research/reports/carla_air_session_reaudit_2026_05_29_zh.md`。本轮并行调用 research/docs 侧与 `mvp-demo` / `tools/carla_air` / runtime outputs 侧只读 explorer，并以主 agent fresh JSON 复核为准。正式三条路线仍 blocked：`local/carla_air/tmp/goal_execution_matrix_session_reaudit_20260529.json` 为 `can_execute_formal_conversion_now=false`、`can_run_formal_neoverse_now=false`、`can_plan_multi_identity_roster_now=false`；`poc03_synthetic_verification_session_reaudit_20260529.json` 为 `ok=true`、`failure_count=0`，但只证明 POC03 synthetic depth-backprojection smoke lineage 自洽，不是 pixel-accurate identity proof、NeoVerse reconstruction 或 final geometry。multi-identity 支线当前已有 3 个 local technical POC normalized identity 且 `aircraft_identity_import_smoke_session_reaudit_20260529.json` 显示 Assimp import smoke 全通过；但 `aircraft_identity_roster_plan_session_reaudit_20260529.json` 仍为 `formal_benchmark_ready=false`、`benchmark_roster_count=0`，原因是缺 source/license/benchmark permission 与 UE/CARLA import evidence。最终 runtime 快照只见 CARLA `:2000` listening，未见 AirSim `:41451`，本轮未执行 AirSim-dependent live export/capture/inspect。
- 2026-05-29 identity permission evidence guard 已记录到 `research/reports/carla_air_identity_permission_evidence_guard_2026_05_29_zh.md`。新增 `tools/carla_air/audit_aircraft_identity_permission_evidence.py` 并收紧 `tools/carla_air/check_aircraft_identity_readiness.py`：formal benchmark eligibility 现在不仅要求 metadata permission flags，还要求可审计 license evidence 与 benchmark/public/redistribution permission evidence。`local/carla_air/tmp/aircraft_identity_permission_evidence_audit_20260529.json` 为 `ok=false`、`benchmark_permission_ready_count=0`；`local/carla_air/tmp/aircraft_identity_readiness_permission_evidence_guard_20260529.json` 仍为 3 个 technical-ready local POC、0 个 benchmark eligible，且每个 local POC 新增 `asset_meta_license_evidence_present` / `asset_meta_benchmark_permission_evidence_present` blockers。`local/carla_air/tmp/aircraft_identity_roster_plan_permission_evidence_guard_20260529.json` 仍 `formal_benchmark_ready=false`、`benchmark_roster_count=0`；Assimp import smoke 仍通过但不是 license/benchmark permission。该 guard 不修改 asset metadata，不把 Aigei promotional notes 当作授权证据，不关闭 multi-identity benchmark blocker。
- 2026-05-30 private/public model policy split 已记录到 `research/reports/carla_air_private_model_policy_split_2026_05_30_zh.md`。用户确认 `assets/models/` 中已下载/购买模型在开源前用于本地私有 POC / private local benchmark，不应被公开许可证据阻塞；因此 `check_aircraft_identity_readiness.py` 与 `plan_aircraft_identity_roster.py` 现区分 `private_benchmark_eligible` 与严格 public/formal `benchmark_eligible`。主审输出为 `private_benchmark_eligible_count=6`、`private_benchmark_roster_count=6`、`private_benchmark_multi_identity_ready=true`，但 `private_carla_benchmark_eligible_count=0`、`benchmark_eligible_count=0`、`benchmark_roster_count=0`、`formal_benchmark_ready=false`。后续可以推进 private local benchmark；开源发布、公开数据集、可再分发 benchmark 与 formal public benchmark 仍必须补 source/license/redistribution/UE-CARLA import evidence。
- 2026-05-29 UE/CARLA import readiness guard 已记录到 `research/reports/carla_air_identity_ue_carla_import_readiness_2026_05_29_zh.md`。新增只读 `tools/carla_air/audit_aircraft_identity_ue_carla_import_readiness.py`，并进一步收紧 `tools/carla_air/check_aircraft_identity_readiness.py` / `tools/carla_air/plan_aircraft_identity_roster.py`：formal benchmark identity 还必须有 UE/CARLA import evidence。`local/carla_air/tmp/aircraft_identity_ue_carla_import_readiness_20260529.json` 为 `ok=false`、`verified_identity_count=0`、`can_run_ue_import_smoke_now=false`，blockers 为 `unreal_editor_missing` 与 `identity_ue_carla_import_evidence_missing`。当前 packaged `CarlaUE4.uproject` 与 `CarlaUE4-Linux-Shipping` 存在，但没有 `UE4Editor` / `UnrealEditor`，shipping runtime 不算 import toolchain。重跑 `local/carla_air/tmp/aircraft_identity_readiness_ue_carla_import_guard_20260529.json` 后 3 个 local POC 仍 technical ready / local POC eligible，但 benchmark eligible 仍为 0，并新增 `build_status_ue_carla_import_verified` blocker；`local/carla_air/tmp/goal_execution_matrix_ue_carla_import_guard_20260529.json` 仍三条正式路线不可执行。
- 2026-05-29 multi-identity current gate 已记录到 `research/reports/carla_air_multi_identity_local_poc_current_gate_2026_05_29_zh.md`。本轮按仓库级策略并行调用 research/docs 侧与 implementation/output 侧只读 explorer；两者均确认当前 annotation / geometry / benchmark 仍未正式完成。fresh outputs 显示 `local/carla_air/tmp/aircraft_identity_readiness_current_multi_local_poc_20260529.json` 为 `ok=false`、`benchmark_ready=false`，但 `normalized_inventory.identity_count=3`、`technical_ready_count=3`、`local_poc_eligible_count=3`、`benchmark_eligible_count=0`；`local/carla_air/tmp/aircraft_identity_roster_plan_current_multi_local_poc_20260529.json` 为 `local_poc_multi_identity_ready=true`、`formal_benchmark_ready=false`、`local_poc_roster_count=3`、`benchmark_roster_count=0`；`local/carla_air/tmp/aircraft_identity_import_smoke_current_multi_local_poc_20260529.json` 为 `ok=true`、`passed_identity_count=3`，但仍 `not_ue_or_carla_import=true` / `not_license_or_benchmark_permission=true`；`local/carla_air/tmp/aircraft_identity_ue_carla_import_readiness_current_multi_local_poc_20260529.json` 为 `ok=false`、`verified_identity_count=0`、`can_run_ue_import_smoke_now=false`。`local/carla_air/tmp/goal_execution_matrix_current_multi_local_poc_20260529.json` 仍为 `can_execute_formal_conversion_now=false`、`can_run_formal_neoverse_now=false`、`can_plan_multi_identity_roster_now=false`。因此可称为 3-identity local technical POC gate pass，但不能称为 formal multi-identity benchmark。
- 2026-05-29 procedural identity permission POC 已记录到 `research/reports/carla_air_procedural_identity_permission_poc_2026_05_29_zh.md`。新增 project-owned `procedural_delta_uav_v1` 后，fresh outputs 显示 `local/carla_air/tmp/aircraft_identity_readiness_procedural_permission_poc_20260529.json` 为 `identity_count=4`、`technical_ready_count=4`、`local_poc_eligible_count=4`、`benchmark_eligible_count=0`；`local/carla_air/tmp/aircraft_identity_permission_evidence_procedural_permission_poc_20260529.json` 为 `benchmark_permission_ready_count=1`；`local/carla_air/tmp/aircraft_identity_import_smoke_procedural_permission_poc_20260529.json` 为 `ok=true`、`passed_identity_count=4`；`local/carla_air/tmp/local_poc_multi_identity_guard_procedural_permission_poc_20260529.json` 为 `ok=true`、`local_poc_multi_identity_ready=true`、`formal_benchmark_ready=false`。该 POC 只把 local technical roster 推进到 4 identities，并补了 1 个 permission-ready identity；UE/CARLA import verified 仍为 0，`can_plan_multi_identity_roster_now=false`，不能称为 formal benchmark ready。
- 2026-05-29 procedural twinboom identity POC 已记录到 `research/reports/carla_air_procedural_twinboom_identity_poc_2026_05_29_zh.md`。新增 project-owned `procedural_twinboom_uav_v1` 后，fresh outputs 显示 `local/carla_air/tmp/aircraft_identity_readiness_procedural_twinboom_poc_20260529.json` 为 `identity_count=5`、`technical_ready_count=5`、`local_poc_eligible_count=5`、`benchmark_eligible_count=0`；`local/carla_air/tmp/aircraft_identity_permission_evidence_procedural_twinboom_poc_20260529.json` 为 `benchmark_permission_ready_count=2`；`local/carla_air/tmp/aircraft_identity_import_smoke_procedural_twinboom_poc_20260529.json` 为 `ok=true`、`passed_identity_count=5`；`local/carla_air/tmp/local_poc_multi_identity_guard_procedural_twinboom_poc_20260529.json` 为 `ok=true`、`local_poc_multi_identity_ready=true`、`formal_benchmark_ready=false`。该 POC 只把 local technical roster 推进到 5 identities，并补了第 2 个 permission-ready procedural identity；UE/CARLA import verified 仍为 0，`can_plan_multi_identity_roster_now=false`，不能称为 formal benchmark ready。
- 2026-05-29 procedural canard identity POC 已记录到 `research/reports/carla_air_procedural_canard_identity_poc_2026_05_29_zh.md`。新增 project-owned `procedural_canard_uav_v1` 后，fresh outputs 显示 `local/carla_air/tmp/aircraft_identity_readiness_procedural_canard_poc_20260529.json` 为 `identity_count=6`、`technical_ready_count=6`、`local_poc_eligible_count=6`、`benchmark_eligible_count=0`；`local/carla_air/tmp/aircraft_identity_permission_evidence_procedural_canard_poc_20260529.json` 为 `benchmark_permission_ready_count=3`；`local/carla_air/tmp/aircraft_identity_import_smoke_procedural_canard_poc_20260529.json` 为 `ok=true`、`passed_identity_count=6`；`local/carla_air/tmp/local_poc_multi_identity_guard_procedural_canard_poc_20260529.json` 为 `ok=true`、`local_poc_multi_identity_ready=true`、`formal_benchmark_ready=false`。该 POC 只把 local technical roster 推进到 6 identities，并补了第 3 个 permission-ready procedural identity；UE/CARLA import verified 仍为 0，`can_plan_multi_identity_roster_now=false`，不能称为 formal benchmark ready。
- 2026-05-29 UE/CARLA import smoke plan 已记录到 `research/reports/carla_air_identity_ue_import_smoke_plan_2026_05_29_zh.md`。新增只读 `tools/carla_air/plan_aircraft_identity_ue_import_smoke.py` 与 `tools/carla_air/verify_aircraft_identity_ue_import_smoke_plan.py`；输出 `local/carla_air/tmp/aircraft_identity_ue_import_smoke_plan_procedural_canard_poc_20260529.json` 为 `identity_count=6`、`runnable_now_count=0`、`blocked_now_count=6`、`formal_benchmark_ready=false`，blocker 是没有 `UE4Editor` / `UnrealEditor` import toolchain；verifier `local/carla_air/tmp/aircraft_identity_ue_import_smoke_plan_verification_procedural_canard_poc_20260529.json` 为 `ok=true`、`failure_count=0`、`non_promotion_verified=true`。该 plan 只是未来 per-identity editor import smoke checklist，不运行 Unreal、不写 asset metadata、不 patch `build_status.json`、不把任何 identity 升级为 benchmark-eligible。
- 2026-05-29 UE/CARLA import explicit editor gate 已记录到 `research/reports/carla_air_ue_import_explicit_editor_gate_2026_05_29_zh.md`。`tools/carla_air/audit_aircraft_identity_ue_carla_import_readiness.py` 与 `tools/carla_air/plan_aircraft_identity_ue_import_smoke.py` 现在支持 `--editor-cmd`，但显式路径必须是存在、可执行且 basename 为 `UnrealEditor-Cmd` / `UnrealEditor` / `UE4Editor-Cmd` / `UE4Editor`；`/does/not/exist` 与 `/bin/true` 负例均被 `explicit_unreal_editor_invalid` 拒绝。该 gate 只让未来真实 Unreal Editor command 可被 readiness/plan 消费，不运行 import、不写资产、不 patch `build_status.json`，当前仍为 `verified_identity_count=0` / `benchmark_eligible_count=0`。
- 2026-05-29 formal target-selection guard 已记录到 `research/reports/carla_air_formal_target_selection_guard_2026_05_29_zh.md`。`tools/carla_air/convert_synthetic_sensor_masks.py` 与 `tools/carla_air/backproject_synthetic_depth_points.py` 现在都要求 formal path 提供 `--target-selection-readiness`，且 readiness report 与对应 candidate assessment 必须证明 trusted actor-to-pixel target、identity proof、pixel accuracy 与 formal synthetic annotation readiness；`target_candidate_gate_passed_not_identity_proof` 不再足以写 formal `masks_synth` / `tracklets_synth` 或 formal depth-backprojection points。`tools/carla_air/run_poc03_synthetic_reid_pipeline.py` 新增 passthrough，并在 execute 到 convert 及之后时缺 readiness 直接拒绝。新输出 `local/carla_air/tmp/poc03_target_selection_readiness_formal_guard_plan_20260529.json` 仍 `target_selection_ready=false`、`trusted_target_id_or_tag_available=false`、`ready_for_formal_conversion=false`，`local/carla_air/tmp/goal_execution_matrix_formal_guard_plan_20260529.json` 仍三条正式路线不可执行。负向 smoke 证明这些 guard 在写入前拒绝 current false readiness，official node03 scene contract / annotation / tracklets / points meta sha256 未变化。
- 2026-05-29 synthetic verifier target-selection guard 已记录到 `research/reports/carla_air_synthetic_verifier_target_selection_guard_2026_05_29_zh.md`。`tools/carla_air/verify_poc03_synthetic_reid_outputs.py` 现在要求 track / annotation meta / points meta 的 target validation 都内嵌 `target_selection_readiness`，且 report 与 candidate assessment 均证明 trusted actor-to-pixel target、identity proof、pixel accuracy 与 formal synthetic annotation readiness。旧 POC03 output 负向 verifier `local/carla_air/tmp/poc03_synthetic_verification_target_selection_guard_negative_20260529.json` 为 `ok=false`、`failure_count=84`，全部 failure 都是 `target_selection_readiness*`；默认无 `--allow-fail` 时返回 `RC=1`。因此旧 candidate-only synthetic smoke 不能再作为 formal verifier pass。
- 2026-05-29 scene status target-selection guard 已记录到 `research/reports/carla_air_scene_status_target_selection_guard_2026_05_29_zh.md`。`tools/carla_air/audit_scene_pipeline_status.py` 现在对 synthetic annotation / synthetic depth-backprojection variant 也要求 target-selection readiness evidence；旧 POC03 candidate-only synthetic smoke 不再计为 `synthetic_annotation_and_depth_geometry_ready`。最新输出 `local/carla_air/tmp/scene_pipeline_status_audit_target_selection_guard_20260529.json` / `.csv` 显示 7 个官方 scene 中 5 个为 `skeleton_pending_formal_annotations`，2 个 POC03 scene 为 `synthetic_candidate_smoke_needs_target_selection`；`synthetic_annotation_ready_count=0`、`synthetic_depth_points_ready_count=0`、`synthetic_candidate_smoke_count=2`、`synthetic_target_selection_ready_count=0`。早期 scene audit 中的 `2 synthetic ready` 只能作为 historical smoke 快照，不能作为当前 formal readiness。
- 2026-05-29 target-selection controlled-decode guard 已记录到 `research/reports/carla_air_target_selection_controlled_decode_guard_2026_05_29_zh.md`。`tools/carla_air/audit_poc03_target_selection_readiness.py` 现在读取受控 actor instance-id calibration、close-view alternate decode probe，并实时探测 runtime 端口。最新输出 `local/carla_air/tmp/poc03_target_selection_readiness_controlled_decode_guard_20260529.json` 为 `ok=true` 但 `target_selection_ready=false`、`trusted_target_id_or_tag_available=false`、`ready_for_formal_conversion=false`；ordinary controlled actor `345` 可由 `green_plus_256_blue` 解码命中 9532 个 ROI 像素，但 POC03 AirSim drone actor `24` 在同一 close-view alt decode 中 `actor_id_seen_in_close_roi=false` / `actor_id_pixel_count=0`。实时端口复核为 CARLA `2000` open、AirSim `41451` closed，不能刷新 AirSim-dependent live target evidence；执行矩阵 `local/carla_air/tmp/goal_execution_matrix_controlled_decode_guard_20260529.json` 仍显示三条正式路线不可执行。
- 2026-05-29 semantic-lidar actor-local geometry candidate 已记录到 `research/reports/carla_air_semantic_lidar_actor_local_geometry_candidate_2026_05_29_zh.md`。新增 `tools/carla_air/derive_semantic_lidar_actor_local_geometry.py` 与 `tools/carla_air/verify_semantic_lidar_actor_local_geometry.py`，把已通过 scene-lineage verifier 的 actor-index world points 离线转成 `actor_local_xyz` candidate roots：node03 为 10 timestamps / 54474 points，node04 为 10 timestamps / 54094 points，roundtrip max error 均为 `0.0m`；verifier `local/carla_air/tmp/semantic_lidar_actor_local_geometry_verification_20260529.json` 为 `ok=true`、`failure_count=0`、`final_real_4d_geometry=false`。该结果只增强 actor-level geometry consistency evidence，不写 scene outputs、不更新 contract、不替换 official `points_by_timestamp`，也不是 fixed-camera geometry、camera pixel mask、NeoVerse reconstruction 或 final real 4D geometry。
- 2026-05-29 mask tracklet embedding provenance 已记录到 `research/reports/carla_air_mask_tracklet_embedding_provenance_2026_05_29_zh.md`，并继续收紧为 `research/reports/carla_air_mask_tracklet_lineage_required_guard_2026_05_29_zh.md`。`mvp-demo/scripts/make_tracklets_from_masks.py` 现在输出 `mask_tracklet_v2`，每帧记录 frame/mask sha256 与 bbox lineage，并显式标注 `diagnostic_only`、`identity_proof=false`、`pixel_accurate=false`、`formal_synthetic_annotation_ready=false`；`mvp-demo/scripts/extract_track_embeddings.py` 现在输出 `track_embedding_meta_v2`，记录 `tracklets_sha256`、tracklet schema/source/diagnostic flags，以及 used frame 的 frame/mask/depth sha256。新增 `--require-tracklet-lineage` 后，embedding 提取会强制核对 tracklet lineage 中的 frame/mask hash 与 bbox；正例 `local/carla_air/tmp/mask_tracklet_provenance_smoke_20260529/embeddings_lineage_required/` 通过，负例 `local/carla_air/tmp/mask_tracklet_provenance_negative_20260529/` 修改 1 个 mask 像素后以 `tracklet lineage mask_sha256 mismatch` 拒绝。该结果只让 mask-derived tracklet / embedding 可追溯并防 stale 输入，不是 formal annotation、identity proof、pixel-accurate mask、true 4D geometry 或 benchmark evidence，不能绕过 CARLA-Air formal writer/verifier。

当前尚未完成：

- 自定义飞行器模型导入；
- 多轨迹批量正式采集；
- pixel-accurate depth/mask/bbox/object pose/trajectory 正式导出；
- target identity / pixel-accuracy evidence 补强或更可信 target id/tag 重新 inspect / validate / conversion / backprojection / verify；
- pixel-level actor/segmentation id 绑定，或让目标 actor 拥有可追踪、可解释的 segmentation/instance id；当前 actor/pose binding 已有证据，node03/node04 close-view `44800` 是几何候选，但 segmentation id / instance PNG 映射仍未解决，本地 Python API 和 AirSim object segmentation set/get 都未给出可用映射或控制入口；
- CARLA camera GBuffer 支线已实测但暂不可用：node03 row0 的 actor-relative 与 fixed-cam probe 均无法捕获 `SceneStencil`、`CustomDepth`、`CustomStencil` 或 `GBufferA` callback；追加 `SceneColor` / `SceneDepth` basic-buffer probe 后仍没有任何 GBuffer callback。除非有 CARLA-Air/UE 侧修复或更底层支持证据，否则不要把 GBuffer 当作可用 actor-to-pixel 路线；
- CARLA raycast / project_point 支线已实测可调用，但只返回 semantic label / location，不返回 actor id；node03 row0 fixed-camera samples 返回 `Buildings` / `None`，只能作为遮挡/几何 sanity，不能作为 actor-to-pixel evidence；
- CARLA annotation API surface 支线已实测公开 Python API 缺口：actor `24` 只暴露 `semantic_tags`，无 actor-level setter；world 只暴露 `set_annotations_traverse_translucency`；相关 sensor blueprint 没有 instance / segmentation / custom / stencil 可设 attribute，不能通过公开 API 直接给 POC03 drone 设置或查询 instance PNG value / segmentation id / custom stencil；
- actor id `24` 到 current `instance_synth` packed PNG id 的直接映射未找到；controlled actor calibration 证明普通 CARLA actor id 可在当前 runtime 的 instance camera 中由 `green + 256 * blue` 解码恢复，但 POC03 drone actor `24` 在 close-view 与 fixed-camera audit 中仍未按该 decode 出现；close actor-relative camera probe 与 node04 later-window fixed-camera ROI 审计只增强 `44800` 的几何候选地位，当前证据仍不能把 `44800` 或 `5387` 升级为 identity proof；
- 本地 `sensor_gallery.py` 支持 `red + 256 * green` 作为 CARLA instance PNG decode contract，但这只是 PNG value 解码规则，不是 actor id 映射表；仍没有 API 或本地表能把 actor `24` 设置/查询为 `44800`、`5387` 或其他 fixed-camera candidate value；
- CARLA blueprint library 当前不暴露 `airsim.drone` 或 `airsim.*` spawn candidate；因此无法用 `probe_instance_id_calibration_actor.py` 直接 spawn 受控 AirSim drone actor 做 calibration，普通 `vehicle.*` actor calibration 不能外推为 POC03 drone identity proof；
- POC03 target-selection readiness gate 当前为 `target_selection_ready=false`、`trusted_target_id_or_tag_available=false`；在该 gate 没有基于新 actor-to-pixel evidence 变为 ready 前，不应把 `5387`、`44800` 或任何 close ROI 候选用于 formal conversion；
- formal conversion / formal points 现在还要求显式 `--target-selection-readiness`；candidate validation report 通过只能说明候选统计与 lineage 自洽，不是 identity proof、pixel-accuracy proof 或 formal output 写入许可；
- `verify_poc03_synthetic_reid_outputs.py` 现在也会拒绝缺 `target_selection_readiness` 的旧 POC03 synthetic smoke；后续不能引用旧 verifier `ok=true` 作为正式输出证据；
- inspect / POC03 plan-only 入口现在可显式输出 alternate instance decode diagnostics，但该字段是 read-only diagnostic，不进入 `candidates.instance`、validation formal candidate、conversion contract 或 downstream pipeline；因此 actor-to-pixel blocker 仍未解除；
- actor-pose projected bbox 已在 node03 当前窗口与 node04 后段隔离 fixed-camera window 形成候选；node04 后段候选只写 `masks_actor_bbox_later_window` 与 `tracklets_actor_bbox_later_window.json`，未更新正式 contract；该候选不得替换正式 `masks_synth` 或 `tracklets_synth`；
- semantic-lidar actor-index close-offset / actor-relative export 可命中 actor `24` 并导出 POC03 node03/node04 isolated points candidate；actor-local 派生与 verifier 已能提供更强 actor-level geometry consistency evidence，但固定三相机位置没有命中，actor-relative placement 不等于 fixed-camera capture geometry，因此仍不能直接当正式 ReID geometry；
- bbox / semantic-lidar 若后续继续正式化，必须按 `carla_air_candidate_geometry_formalization_boundaries_2026_05_29_zh.md` 使用独立 variant/source/status；semantic-lidar actor points 当前已有 isolated candidate verifier，weak diagnostic variants 已在受保护 writer 与 post-write verifier 下登记到 `weak_diagnostic_variants` namespace，但这仍不是 formal synthetic target mask、final geometry 或 benchmark evidence，不得混入当前 synthetic target mask 或 depth-backproject variant；
- candidate formalization readiness 聚合 gate 当前为 `formalization_ready=false`、`goal_complete=false`；在 actor-bbox / semantic-lidar / identity readiness / POC03 target selection blocker 未解除前，不启动候选正式化或 multi-identity benchmark；
- weak variant contract 当前只是 plan，不是 promotion；bbox / semantic-lidar writer 前必须先有独立 dry-run linter、独立 namespace、独立 verifier 与 downstream opt-in；
- weak variant contract patch linter 已可生成 proposed patch JSON，但仍不是 writer；即使 linter `ok=true`，也不能关闭 actor-to-pixel evidence、semantic-lidar placement 或 official writer/verifier blocker；
- weak variant official-readiness verifier 与 contract-update verifier 已可检查 dry-run / diagnostic / readiness 的 non-promotion guard，以及 official contract / future update 的 weak 污染风险；protected writer 已在 post-write verifier 保护下登记 weak diagnostic variants，但这只证明允许 namespace 内的 non-promotion contract evidence，不关闭 promotion evidence missing、actor-to-pixel evidence missing、pixel accuracy missing、final geometry missing、multi-identity missing、`formalization_ready=false` 或 `goal_complete=false`；
- downstream weak diagnostic opt-in guard 已阻止默认消费 weak outputs，并进一步要求 post-write contract verification evidence；但它只增强下游安全，不关闭 promotion evidence / actor-to-pixel / pixel accuracy / final geometry / multi-identity blockers；
- tracklet input contract audit 已能分类 proxy / synthetic smoke / actor-bbox candidate / weak diagnostic tracklet，但它只是 preflight 安全门，不表示 formalization ready、goal complete、pixel-accurate annotation、final geometry、NeoVerse reconstruction 或 benchmark ready；
- weak diagnostic ReID smoke 已跑通但仅为 1 query vs 1 gallery 的本地诊断闭环，且依赖 bbox rectangle candidate 与 actor-relative semantic-lidar candidate；即使带 contract verification evidence，也不能关闭 actor-to-pixel evidence missing、pixel-accuracy missing、fixed-camera/final geometry missing 或 multi-identity missing blocker；
- NeoVerse readiness audit 当前为 `formal_neoverse_ready=false`：现有 `neoverse_fused` POC03 points root 是 proxy source，现有 `carla_air_depth_points` root 是 depth-backprojection baseline；NeoVerse reconstructor checkpoint 已接入默认路径，但仍缺 formal NeoVerse points root、target-ready / formal annotation 输入，不得把任一现有 root 写成 formal NeoVerse / final real geometry；
- goal execution matrix 当前明确三条正式路线均不可执行：`can_execute_formal_conversion_now=false`、`can_run_formal_neoverse_now=false`、`can_plan_multi_identity_roster_now=false`；后续应先让对应 readiness gate 变 true，而不是直接跑 conversion / NeoVerse / benchmark；
- multi-identity benchmark 现在同时被 permission evidence completeness 与 UE/CARLA import evidence 阻断：6 个 local POC 的 Assimp import smoke 通过，其中 `procedural_delta_uav_v1`、`procedural_twinboom_uav_v1` 与 `procedural_canard_uav_v1` 具备 project-owned benchmark/public/redistribution permission evidence，但另外 3 个 identity 仍缺 source/license/benchmark permission evidence，且 6 个 identity 都没有 UE/CARLA import verified evidence；当前环境没有 `UE4Editor` / `UnrealEditor`，不能把 packaged shipping runtime 当作导入工具；
- UE/CARLA import smoke plan 只把上述 import blocker 拆成 future checklist；当前没有实际 `local/carla_air/tmp/ue_import_smoke/<identity_id>/import_smoke_report.json` 成功证据，不能据此更新 `build_status.json` 或 formal benchmark roster；
- NeoVerse 或等价真 4D geometry 导出；
- 多身份、多轨迹的正式 ReID benchmark 批量生成。

## 2. 关键文件

布设脚本：

```text
tools/carla_air/place_camera_node.py
```

采集 smoke 脚本：

```text
tools/carla_air/capture_camera_nodes_smoke.py
```

默认三相机 rig：

```text
configs/camera_rigs/node_tri_cam_parallel_v1.json
```

已保存相机节点：

```text
local/carla_air/camera_nodes/Town10HD_ground_to_air_nodes_v1.json
```

阶段汇总：

```text
research/reports/carla_air_ground_to_air_camera_nodes_milestone_2026_05_24_zh.md
research/reports/carla_air_capture_smoke_milestone_2026_05_25_zh.md
research/reports/carla_air_aircraft_asset_registry_2026_05_25_zh.md
research/reports/carla_air_aircraft_normalization_poc_2026_05_25_zh.md
research/reports/carla_air_aircraft_identity_readiness_gate_2026_05_28_zh.md
research/reports/carla_air_raw_aircraft_normalization_prioritization_2026_05_29_zh.md
research/reports/carla_air_default_drone_trajectory_presence_gate_smoke_2026_05_25_zh.md
research/reports/carla_air_synchronized_trajectory_capture_poc_2026_05_25_zh.md
research/reports/carla_air_synchronized_trajectory_capture_live_smoke_2026_05_25_zh.md
research/reports/carla_air_formal_data_contract_4b_2026_05_25_zh.md
research/reports/carla_air_scene_skeleton_export_4b_2026_05_25_zh.md
research/reports/carla_air_coverage_first_trajectory_generator_2026_05_25_zh.md
research/reports/carla_air_poc03_poc04_coverage_readiness_2026_05_25_zh.md
research/reports/carla_air_poc03_poc04_live_capture_gate_scene_export_2026_05_25_zh.md
research/reports/carla_air_minimal_formal_sample_validation_2026_05_25_zh.md
research/reports/carla_air_annotation_source_geometry_readiness_2026_05_25_zh.md
research/reports/carla_air_synthetic_annotation_geometry_bridge_2026_05_25_zh.md
research/reports/carla_air_poc03_synthetic_reid_orchestrator_2026_05_25_zh.md
research/reports/carla_air_synthetic_target_mask_quality_audit_2026_05_28_zh.md
research/reports/carla_air_actor_pose_projected_bbox_candidate_2026_05_28_zh.md
research/reports/carla_air_actor_id_instance_mapping_audit_2026_05_29_zh.md
research/reports/carla_air_semantic_lidar_actor_idx_probe_2026_05_29_zh.md
research/reports/carla_air_semantic_lidar_actor_points_export_2026_05_29_zh.md
research/reports/carla_air_semantic_lidar_actor_points_candidate_verifier_2026_05_29_zh.md
research/reports/carla_air_actor_instance_camera_binding_probe_2026_05_29_zh.md
research/reports/carla_air_actor_gbuffer_binding_probe_2026_05_29_zh.md
research/reports/carla_air_node04_later_window_instance_candidate_2026_05_29_zh.md
research/reports/carla_air_candidate_formalization_readiness_audit_2026_05_29_zh.md
research/reports/carla_air_weak_variant_contract_plan_2026_05_29_zh.md
research/reports/carla_air_weak_variant_contract_patch_linter_2026_05_29_zh.md
research/reports/carla_air_weak_variant_diag_reid_smoke_2026_05_29_zh.md
research/reports/carla_air_weak_variant_downstream_guard_2026_05_29_zh.md
research/reports/carla_air_weak_variant_contract_update_verifier_2026_05_29_zh.md
research/reports/carla_air_weak_variant_contract_update_writer_dry_run_2026_05_29_zh.md
research/reports/carla_air_weak_variant_contract_update_execute_2026_05_29_zh.md
research/reports/carla_air_weak_variant_downstream_contract_guard_2026_05_29_zh.md
research/reports/carla_air_tracklet_input_contract_audit_2026_05_29_zh.md
research/reports/carla_air_instance_png_mapping_local_api_audit_2026_05_29_zh.md
research/reports/carla_air_goal_resume_readiness_rerun_2026_05_29_zh.md
research/reports/carla_air_goal_resume_execution_status_2026_05_29_zh.md
research/reports/carla_air_goal_continue2_status_2026_05_29_zh.md
research/reports/carla_air_actor_bbox_formalization_preflight_2026_05_29_zh.md
research/reports/carla_air_current_worktree_reaudit_2026_05_29_zh.md
research/reports/carla_air_local_normalized_identity_poc_2026_05_29_zh.md
research/reports/carla_air_identity_ue_carla_import_readiness_2026_05_29_zh.md
research/reports/carla_air_identity_ue_import_smoke_plan_2026_05_29_zh.md
research/reports/carla_air_ue_import_smoke_plan_status_sync_2026_05_29_zh.md
research/reports/carla_air_instance_decode_diagnostic_inspect_enhancement_2026_05_29_zh.md
research/reports/carla_air_airsim_drone_blueprint_spawn_probe_2026_05_29_zh.md
research/reports/carla_air_poc03_target_selection_readiness_gate_2026_05_29_zh.md
research/reports/carla_air_annotation_api_surface_probe_2026_05_29_zh.md
research/reports/carla_air_current_followup_readiness_2026_05_29_zh.md
research/reports/carla_air_formal_target_selection_guard_2026_05_29_zh.md
research/reports/carla_air_synthetic_verifier_target_selection_guard_2026_05_29_zh.md
research/reports/carla_air_scene_status_target_selection_guard_2026_05_29_zh.md
research/reports/carla_air_target_selection_controlled_decode_guard_2026_05_29_zh.md
research/reports/carla_air_mask_tracklet_embedding_provenance_2026_05_29_zh.md
research/reports/carla_air_mask_tracklet_lineage_required_guard_2026_05_29_zh.md
research/reports/carla_air_semantic_lidar_scene_lineage_guard_2026_05_29_zh.md
research/reports/carla_air_semantic_lidar_actor_local_geometry_candidate_2026_05_29_zh.md
```

轨迹与 gate 工具：

```text
configs/carla_air/trajectories/town10hd_node_visibility_poc_v1.json
tools/carla_air/run_drone_trajectory_smoke.py
tools/carla_air/check_capture_presence_gate.py
tools/carla_air/run_drone_trajectory_smoke_suite.py
tools/carla_air/capture_drone_trajectory_nodes.py
tools/carla_air/generate_coverage_trajectory.py
tools/carla_air/export_capture_scene_dirs.py
tools/carla_air/build_minimal_formal_sample.py
tools/carla_air/probe_annotation_sources.py
tools/carla_air/export_live_synthetic_annotations.py
tools/carla_air/inspect_synthetic_sensor_ids.py
tools/carla_air/convert_synthetic_sensor_masks.py
tools/carla_air/backproject_synthetic_depth_points.py
tools/carla_air/run_poc03_synthetic_reid_pipeline.py
tools/carla_air/audit_scene_pipeline_status.py
tools/carla_air/audit_synthetic_target_mask_quality.py
tools/carla_air/probe_target_actor_binding.py
tools/carla_air/audit_actor_bbox_instance_candidates.py
tools/carla_air/export_actor_pose_projected_bbox_annotations.py
tools/carla_air/verify_actor_bbox_candidate_annotations.py
tools/carla_air/audit_candidate_formalization_readiness.py
tools/carla_air/audit_actor_id_instance_encoding.py
tools/carla_air/probe_semantic_lidar_actor_idx.py
tools/carla_air/export_semantic_lidar_actor_points.py
tools/carla_air/verify_semantic_lidar_actor_points_candidate.py
tools/carla_air/derive_semantic_lidar_actor_local_geometry.py
tools/carla_air/verify_semantic_lidar_actor_local_geometry.py
tools/carla_air/probe_actor_instance_camera_binding.py
tools/carla_air/probe_actor_gbuffer_binding.py
tools/carla_air/probe_carla_annotation_api_surface.py
tools/carla_air/probe_carla_blueprint_inventory.py
tools/carla_air/audit_poc03_target_selection_readiness.py
tools/carla_air/check_aircraft_identity_readiness.py
tools/carla_air/audit_aircraft_identity_ue_carla_import_readiness.py
tools/carla_air/plan_aircraft_identity_ue_import_smoke.py
tools/carla_air/verify_aircraft_identity_ue_import_smoke_plan.py
tools/carla_air/prioritize_raw_aircraft_normalization.py
tools/carla_air/plan_weak_variant_contract_patch.py
tools/carla_air/plan_weak_variant_writer_dry_run.py
tools/carla_air/verify_weak_variant_diag_reid_smoke.py
tools/carla_air/verify_weak_variant_official_readiness.py
tools/carla_air/verify_weak_variant_contract_update.py
tools/carla_air/apply_weak_variant_contract_update.py
tools/carla_air/audit_tracklet_input_contract.py
tools/carla_air/audit_neoverse_geometry_readiness.py
tools/carla_air/audit_goal_execution_matrix.py
```

采集路线文档：

```text
数据集采集/ground_to_air_synthetic_uav_4d_reid_collection_guide_zh.md
```

## 3. 启动方式

终端 1：启动 CARLA-Air。不要在该终端激活 `carlaAir` conda 环境。

```bash
conda deactivate
cd /home/grasp/data/3d-reid/local/carla_air/simulators/CarlaAir-v0.1.7
./CarlaAir.sh Town10HD --res 1280x720 --quality Low --fg
```

终端 2：运行 Python 工具。

```bash
cd /home/grasp/data/3d-reid
conda activate carlaAir
python tools/carla_air/place_camera_node.py --wait-seconds 120
```

继续从已有最大节点的下一个节点开始布设：

```bash
python tools/carla_air/place_camera_node.py --wait-seconds 120 --resume-next
```

手动指定某个节点：

```bash
python tools/carla_air/place_camera_node.py --node-id node03 --wait-seconds 120
```

## 4. 布设工具控制

```text
W/S        前进/后退
A/D        左右移动
Q/E        下/上移动
←/→        调整 yaw
↑/↓        调整 pitch
Z/X        roll
[/]        调整移动速度
Space      保存
Ctrl+S/F5  保存备选
N          保存并切到下一个 node
B          保存并切到上一个 node
P          打印位姿
ESC        退出
```

## 5. 当前节点摘要

当前配置文件：

```text
local/carla_air/camera_nodes/Town10HD_ground_to_air_nodes_v1.json
```

摘要：

```text
schema_version: carla_air_ground_to_air_camera_nodes_v1
map: Town10HD
layout_id: node_tri_cam_parallel_v1
node_count: 5
node_ids: node01, node02, node03, node04, node05
```

每个 node 都有：

```text
cam0 / cam1 / cam2
K
image_size
T_node_from_cam
fov_x_deg
carla_relative_transform
carla_world_transform
```

## 6. 快速检查命令

检查 CARLA-Air 是否正在运行：

```bash
ps -eo pid,cmd | rg 'CarlaUE4|CarlaAir|auto_traffic'
ss -tlnp | rg '2000|41451'
```

检查布设脚本：

```bash
PYTHONNOUSERSITE=1 /home/grasp/miniconda3/envs/carlaAir/bin/python -m py_compile tools/carla_air/place_camera_node.py
PYTHONNOUSERSITE=1 /home/grasp/miniconda3/envs/carlaAir/bin/python tools/carla_air/place_camera_node.py --help
```

检查已保存节点：

```bash
python - <<'PY'
import json
from pathlib import Path
p = Path('local/carla_air/camera_nodes/Town10HD_ground_to_air_nodes_v1.json')
data = json.loads(p.read_text())
print(data.get('schema_version'), data.get('map'), data.get('layout_id'))
nodes = data.get('nodes', [])
print(len(nodes), [n.get('node_id') for n in nodes])
for n in nodes:
    print(n.get('node_id'), sorted(n.get('cameras', {})))
PY
```

## 7. 下一步任务

当前 `capture smoke` 已完成，已验证：

```text
sample_run: local/carla_air/captures/20260525_010855
nodes: node01, node02
fps: 10
duration: 10 seconds
per_node_groups: 100
per_node_images: 300
dropped_or_incomplete_frames: 0
world_settings_restored: true
success: true
```

当前默认 drone live trajectory smoke 已完成，已验证：

```text
trajectory_run_id: local/carla_air/trajectory_runs/20260525_082858_traj_poc_01_west_corridor
capture_run_id: local/carla_air/captures/20260525_082900
trajectory_id: traj_poc_01_west_corridor
identity_id: default_airsim_drone
nodes: node01, node02, node05
fps: 10
duration: 14 seconds
per_node_groups: 140
per_node_images: 420
dropped_or_incomplete_frames: 0
capture_success: true
presence_gate: true
passing_nodes: node01, node02, node05
```

当前 4 轨迹 smoke baseline 已完成，已验证：

```text
traj_poc_01_west_corridor -> capture_run_id=20260525_082900
traj_poc_02_short_overlap -> capture_run_id=20260525_stage3b_cap02_short_overlap
traj_poc_03_southeast_pass -> capture_run_id=20260525_stage3b_cap03_southeast_pass
traj_poc_04_north_bridge -> capture_run_id=20260525_stage3b_cap04_north_bridge
```

统一结果：

```text
capture_success: true
dropped_or_incomplete_frames: 0
presence_gate: true
review_summary_csv: present
review_overlays: present
```

注意：旧 baseline `20260525_stage3b_cap04_north_bridge` 中 `traj_poc_04_north_bridge` 是整体 gate pass，但 `node02` 只有 `cam2` 通过，不能记为该旧 run 的 node02 三路稳定可见。新 live run `20260525_180648_traj_poc_04_north_bridge` 的 `node02` 与 `node05` 三路 camera 均通过 presence gate，应按 run_id 区分证据。`traj_poc_01_west_corridor` 的 `node05` 也只应作为 smoke 证据，人工复核更偏弱。

当前 `capture_camera_nodes_smoke.py` 能力：

- 读取 `Town10HD_ground_to_air_nodes_v1.json`；
- spawn 指定节点或全部节点的 `cam0/cam1/cam2`；
- 默认 `sync` 模式临时启用 CARLA `synchronous_mode` 和 `fixed_delta_seconds`；
- 导出同步 RGB、`frame_times.csv`、`calib/rig.json`、相机内参、相机位姿和 `capture_meta.json`；
- 记录进度、world settings restore 状态、sensor stop/destroy failure、timeout 和 interrupted 状态；
- 输出到 `local/carla_air/captures/<run_id>/`。

视频导出状态：

```text
local/carla_air/captures/20260525_010855/node01_triptych_1920x360_preview.mp4
local/carla_air/captures/20260525_010855/node02_triptych_1920x360_preview.mp4
```

视频只作为组会展示和人工质检产物，不进入正式 pipeline。后续 pipeline 继续消费：

```text
capture_meta.json
trajectory_capture_manifest.json
trajectory_frame_groups.csv
cams/cam*/frames/*.png
frame_times.csv
calib/rig.json
```

推荐下一步：

1. 对后续 gate-passed capture 继续用 `tools/carla_air/export_capture_scene_dirs.py` 导出为 `mvp-demo/data/carla_air/nodes/<node_id>/scenes/<scene_id>/`；
2. 若需复跑 POC03 synthetic smoke，使用同一编排入口并保持 target validation；进入 `convert` / `points` / `verify` formal path 时还必须提供已通过的 `--target-selection-readiness <report.json>`。当前没有 ready report，因此不要直接用 `instance_id=5387`、`44800` 或 close ROI 重新写 `masks_synth` / `tracklets_synth` / formal points；不得用 `--skip-target-validation` 生成正式输出；
3. 优先补强 target identity / pixel-accuracy evidence；动手前先复跑 `tools/carla_air/audit_goal_execution_matrix.py`，确认正式路线仍被 gate 住。当前 mask-quality diagnostic audit 与 QC overlay 对两个 POC03 scene 都触发强风险，`instance_id=5387` 已通过两侧 target-candidate gate 但视觉上更像 sky/background segmentation candidate，仍不能声称 identity proof 或 pixel-accurate target mask；最新 actor binding probe 已证明 recorded pose 可绑定到 CARLA `airsim.drone` actor / AirSim `SimpleFlight`，但 AirSim segmentation id 设置仍不可查询，且对 actor 近邻 9 个 object 的 set/get 也全部未生效。actor bbox candidate audit 显示 node03 `44800` 更几何合理；close actor-relative camera probe 进一步显示 node03/node04 actor ROI 内都稳定出现 `44800` 小区域；node04 后段 isolated fixed-camera window 也显示 `44800` 是 bbox ROI 最强几何候选，且已导出 isolated `carla_actor_pose_projected_bbox` rectangle candidate。但这些仍只是 candidate evidence，不是 actor-id mapping proof。2026-05-29 actor-id encoding audit 没有在当前 `instance_synth` packed 编码中找到 actor id `24`；controlled actor calibration 证明普通 CARLA actor 可由 `green + 256 * blue` 解码恢复 actor id，但同一线索对 POC03 drone actor `24` 仍不成立，且 current CARLA blueprint library 不暴露 `airsim.drone` / `airsim.*` 可 spawn blueprint，不能直接改正式 conversion 或升级 `44800` / `5387`。本地 CARLA Python API 也未暴露 actor id -> instance PNG value 映射或 setter；GBuffer `SceneStencil` / `CustomDepth` / `CustomStencil` / `GBufferA` 与 basic `SceneColor` / `SceneDepth` 支线在 node03 row0 actor-relative 与 fixed-cam probe 中没有任何 callback，暂不能作为 actor-to-pixel 路线；raycast / project_point 虽可调用，但只返回 label/location，不返回 actor id，也不能作为 actor-to-pixel proof。semantic-lidar actor-relative export 可为 POC03 node03/node04 产出 `carla_semantic_lidar_actor_idx_v1` actor-level points candidate，但固定三相机位置不命中，且它不是 camera mask、fixed-camera capture geometry、NeoVerse reconstruction 或正式 ReID geometry。下一步应调查 CARLA-Air / CARLA 是否有 UE 内部 actor id 到 instance PNG id 的映射表，调整 live synthetic export 让目标 actor 拥有可追踪、可解释的 segmentation/instance id，或设计明确 source 的 bbox/mesh/semantic-lidar geometry 正式化路线；只有 `audit_poc03_target_selection_readiness.py` 基于新证据变为 ready 后，才重新 inspect / validate / conversion / backprojection / verify 选择更可信 target id/tag；
- 2026-05-29 runtime restart attempt + stability gate blocker：`local/carla_air/tmp/runtime_stability_audit_runtime_restart_attempt_20260529.json` 显示 `runtime_stable_for_window=false`、`sample_count=5`、`all_required_ports_open_sample_count=0`、`matching_process_seen_any_sample=false`；`local/carla_air/tmp/goal_execution_matrix_runtime_stability_gate_main_review_20260529.json` 仍为 `blocked`。这说明 runtime 窗口未稳定开放，不能据此做 promotion、formal conversion 或 benchmark。
4. 下游消费 tracklet 前先运行 `tools/carla_air/audit_tracklet_input_contract.py` 或复用最新审计 JSON，确认 proxy、synthetic smoke、actor-bbox candidate 与 weak diagnostic tracklet 被正确分类；weak diagnostic 输入仍必须经过显式 weak opt-in 和 readiness evidence；
5. 编排入口生成的 synthetic 输出应落到 `tracks/tracklets_synth.json`、`embeddings_synth_depth_backproject/`、`mvp-demo/output/carla_air_depth_points/<scene_id>/points_by_timestamp/` 和 `mvp-demo/output/evals/carla_air_synthetic_depth_backproject_*.json`，不要覆盖 minimal proxy 输出；
6. 每次正式 smoke 后运行 `tools/carla_air/verify_poc03_synthetic_reid_outputs.py` 或 orchestrator 的 `--stop-after verify`；只有 verifier `ok=true` 后，才能把该 run 记为 synthetic annotation / non-proxy depth-backprojection geometry smoke 通过；
7. 每次汇总官方 scene readiness 时复跑 `tools/carla_air/audit_scene_pipeline_status.py`，并以 target-selection guard 后的 `synthetic_candidate_smoke_needs_target_selection` / `synthetic_*_ready_count=0` 语义为准；不要把旧 POC03 candidate-only synthetic smoke 或早期 `2 synthetic ready` scene audit 作为当前 formal annotation / formal geometry 证据；
8. target-selection readiness 复核应使用最新 `tools/carla_air/audit_poc03_target_selection_readiness.py`，并检查 `controlled_instance_calibration_summary`、`alt_decode_close_instance_probe_summary` 与 `current_runtime_ports`；ordinary controlled actor 可解码不等于 POC03 actor-to-pixel proof，只有 POC03 actor `24` 被可信映射到候选像素且 pixel-accuracy evidence 成立后，才允许重新推进 formal conversion；
9. 若使用通用 `make_tracklets_from_masks.py` / `extract_track_embeddings.py` 做 isolated mask-derived smoke，必须保留 `mask_tracklet_v2` / `track_embedding_meta_v2` provenance；下游 embedding 建议显式加 `--require-tracklet-lineage`，让当前 frame/mask hash 与 tracklet lineage 不一致时直接失败。该链路只用于验证下游消费可追溯，不能直接写 official `tracklets.json` / `tracklets_synth.json` 或替代 CARLA-Air formal writer/verifier；
10. 若需要 NeoVerse reconstruction，先复跑 `tools/carla_air/audit_neoverse_geometry_readiness.py` 确认 readiness，再基于 validated masks/depth/observations 接入 NeoVerse 链，替换 depth-backprojection baseline 或输出单独 real-neoverse points root；若继续 bbox / semantic-lidar geometry，按候选正式化边界报告保持 isolated source/namespace；semantic-lidar actor points 已有 candidate verifier 与 scene lineage guard，weak diagnostic smoke 已证明 isolated route 可运行，weak diagnostic variants 已在受保护 writer 与 post-write verifier 下登记，downstream guard 已阻止默认消费并要求 post-write contract evidence；后续不得直接接入 `tracklets_synth` 或 synthetic depth-backproject official route，应优先补 actor-to-pixel / pixel-accuracy / fixed-camera geometry evidence 或设计更强正式化 gate；
11. 后续新增 coverage-first 轨迹时，沿用 POC03/POC04 已验证流程：live capture -> presence gate -> `export_capture_scene_dirs.py`，只补证据，不改方向；
12. 自定义模型替换仍按 `aircraft_normalization_poc` 与 UE 导入/recook 路线独立推进，不阻塞默认 drone 轨迹和 gate smoke。

## 8. 注意事项

- 不要把 `local/carla_air/` 下的 runtime、capture 或 camera_nodes 提交进 git。
- 启动 UE/CARLA-Air 二进制时不要激活 `carlaAir` conda；Python 脚本再激活。
- 当前节点 pitch 多为小幅负值，是否满足地对空仰视需要结合实际预览画面确认。
- 当前工具只调 node anchor，不调单个相机相对布局；如需非平行光轴或不同 FOV，应新增新的 `layout_id`，不要覆盖 `node_tri_cam_parallel_v1`。
- 毕设仿真主线暂定为已知固定相机参数与已知固定相机位姿；真实弱标定或 NeoVerse predicted camera 可作为后续消融/扩展，不作为当前正式采集前置条件。
