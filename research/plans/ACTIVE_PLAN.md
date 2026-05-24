# ACTIVE_PLAN

goal: 以 NeoVerse fused 4D `points_by_timestamp` 作为当前 4D 几何主输入，在 node01 已闭环的 single-node、cross-scene、track-level 3D-aware retrieval 基础上，推进 cross-node smoke、learned point encoder、4D motion rerank 与端到端 4D-aware ReID prototype；毕业论文继续沿同一数据契约扩展到 cross-node、真实三相机迁移和系统性误差分析。
current_milestone: M8 NeoVerse 4D 主线正式化与 ReID 表征补强阶段：`node01` 单节点 `3 identities x 2 scenes` NeoVerse fused 4D eval matrix、view-count ablation 和 own-depth no-depthGT 评测均已完成；当前结论仍是 NeoVerse 4D geometry 已接入并可评测，但在 `masks_gt/depth_gt` 或 `masks_gt/own-depth` bootstrap 下与 RGB-only 持平，不能写成 geometry 已带来指标提升。数据建设侧已完成 CARLA-Air v0.1.7 runtime/Python API smoke、地对空相机节点布设工具和 `Town10HD node01-node05` 固定三相机节点配置；下一步并行推进 CARLA-Air capture smoke、`cross-node smoke` 与 `learned point encoder / end-to-end 4D-aware ReID prototype`。
must_read:
  - research/plans/tri_camera_node_3d_aware_reid/master_plan_zh.md
  - research/plans/tri_camera_node_3d_aware_reid/neoverse_4d_reid_strengthening_plan_zh.md
  - research/plans/tri_camera_node_3d_aware_reid/synthetic_uav_4d_reid_dataset_plan_zh.md
  - research/plans/tri_camera_node_3d_aware_reid/benchmarks/node01_neoverse_fused_4d_eval_matrix_v1.json
  - research/plans/tri_camera_node_3d_aware_reid/benchmarks/node01_neoverse_fused_4d_view_ablation_v1.json
  - research/plans/tri_camera_node_3d_aware_reid/benchmarks/node01_neoverse_own_depth_4d_eval_v1.json
  - research/guides/node01_neoverse_fused_4d_reid_zh.md
  - research/reports/node01_neoverse_fused_4d_eval_matrix_v1_group_update_zh.md
  - research/reports/node01_neoverse_fused_4d_view_ablation_v1_zh.md
  - research/reports/node01_neoverse_own_depth_4d_eval_v1_zh.md
  - research/reports/carla_air_ground_to_air_camera_nodes_milestone_2026_05_24_zh.md
  - research/handoffs/tri_camera_node_engineering_handoff_zh.md
  - research/handoffs/carla_air_ground_to_air_collection_handoff_zh.md
locked_decisions:
  - 当前 4D 几何研究主线正式转为 `NeoVerse fused 4D points_by_timestamp -> tracklet -> embedding -> ReID`；`points_by_timestamp/index.csv + meta.json + *.npy` 是当前 4D ReID 的权威几何输入契约，`meta.json.schema_version` 固定为 `neoverse_points_by_timestamp_v1`。
  - `recon_spin_points.py`、`RGB + predicted-depth geometry`、旧 `RGB + fused geometry`、`GT upper-bound` 保留为历史 baseline、诊断线或对照线，不再作为当前 4D 主线。
  - 当前项目允许两个 `conda` 环境分工：`mvp_demo` 负责 MuJoCo、图像侧、tracklet、embedding、benchmark 等常规脚本；`neoverse` 负责 `run_neoverse_per_camera_bundle.py` 到 `analyze_fused_multiview_quality.py` 的 NeoVerse fused 4D 链及预览/分析脚本。
  - 研究任务仍固定为“三相机节点级 track-level 3D/4D-aware retrieval”；YOLO 门控 + 3DGS 只保留为辅助 demo，dynamic 3DGS 不作为当前主线方法。
  - ICISCAE/近期保底结果仍限于 `node01` 的 single-node、cross-scene、track-level retrieval；论文叙事必须写成仿真节点检索验证，不能宣称 cross-node 已完成。
  - 毕业论文继续沿同一数据契约扩展到 cross-node retrieval、真实三相机节点接入、系统性误差分析和端到端 4D-aware ReID。
  - synthetic UAV/aircraft 4D-ReID benchmark 纳入近期数据建设路线：优先复用现成城市仿真场景，自定义多节点三相机部署、飞行器 identity、轨迹和标注导出，用于支撑 cross-node、更高难度 hard negative 与 4D-aware ReID 训练/评测。
  - CARLA-Air v0.1.7 作为 synthetic UAV/aircraft POC 的当前第一落地平台；已保存的 `local/carla_air/camera_nodes/Town10HD_ground_to_air_nodes_v1.json` 是当前 Town10HD 地对空固定相机节点配置来源。
  - CARLA-Air/UE 二进制启动与 Python API 脚本环境分离：启动 simulator 时不激活 `carlaAir` conda，运行 `tools/carla_air/place_camera_node.py` 等 Python 工具时再激活 `carlaAir`。
  - 当前 ReID 结果矩阵必须保守表述：`node01_neoverse_fused_4d_eval_matrix_v1`、view-count ablation、own-depth 评测都只能证明“接入与可评测完成”，不能写成 geometry、三相机或 own-depth 已提升 Rank/mAP。
  - 近期 ReID 补强路线固定为：`CLIP + FPFH baseline -> learned point encoder -> 4D temporal/motion modeling -> object gallery + rerank -> end-to-end 4D-aware ReID prototype`。
  - 端到端 ReID 纳入近期规划不等于已有端到端训练结果；文档和汇报必须写成 prototype / next-stage method。
  - 正式 benchmark 的 mask 输入布局仍固定为平铺 `cams/cam*/masks/<ts>.png`；SAM2 的 `obj_XXX` 只允许作为中间产物，不直接进入正式 benchmark。
  - `identity_id` 的权威来源固定为 `capture_meta.target.identity_id`；`build_node_tracklets.py --identity_id` 只用于补历史 scene。
  - 当前默认采集、默认 viewer 和默认 benchmark 均为无 humanoid 的 clean 飞行器场景；ICISCAE 小论文目标域固定为 UAV/aircraft，不再做人形 benchmark。
  - 当前本机 NeoVerse fused 4D 结果主要用于链路验证；切换到高性能机器后，优化顺序固定为“先提高合法输入分辨率，再视结果收紧 `output_voxel_size_m`”。
  - `hist` 和 `radial_hist` 只作为 smoke fallback，不作为正式 ReID 主结果线命名或结论依据。
  - 权威研究文档统一收口到 `research/`；`mvp-demo/` 仅保留运行入口与资产说明。
next_action: 并行推进三条近期任务：1) 基于已保存的 `Town10HD node01-node05` 相机节点配置实现 CARLA-Air capture smoke，先导出短序列 RGB、camera intrinsics、camera world pose 与 timestamps；2) 在不改写当前 `node01` 结论的前提下补 `node02 / cross-node smoke`，继续保留 RGB-only 与 RGB+NeoVerse 4D geometry 的同口径对比；3) 按 `neoverse_4d_reid_strengthening_plan_zh.md` 启动 ReID 表征补强 prototype，优先实现 learned point encoder 替换 FPFH 的最小接口，再扩展 4D motion rerank 与端到端 4D-aware ReID。
out_of_scope:
  - 多目标关联与多实例同时检索不作为当前 node01 bootstrap 成功条件。
  - dynamic 3DGS 或连续表面重建不作为当前主线方法。
  - 完整大规模真实场景端到端训练不作为当前 node01 bootstrap 成功条件；近期只规划 prototype。
  - 大规模真实数据采集不作为当前小论文成功条件。
  - 不把 `obj_XXX` 嵌套 mask 直接作为正式 benchmark 输入。
  - 不把 cross-node retrieval 或真实节点迁移作为当前小论文成功条件。
latest_retrospective: none
last_updated: 2026-05-24
