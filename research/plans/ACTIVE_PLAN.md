# ACTIVE_PLAN

goal: 先用 MuJoCo 在 node01 上完成 UAV/aircraft 的 single-node、cross-scene、track-level 3D-aware retrieval benchmark，作为 ICISCAE 小论文；再在同一数据契约上扩展到 cross-node 与真实三相机迁移，作为毕业论文主线。
current_milestone: M6 node01 `spin recon` 第一轮收口：6 条正式 spin scene 的 GT / predicted recon 已打通，`j10 + su34` 子集上 `rgb_recon_enhanced_geometry` 已明显超过 `rgb_fused_geometry`，6-scene 全量结果也已小幅超过 fused 几何，当前待继续收口 static scene 的剩余漂移与 `uav1_static` 的 canonical 饱和问题
must_read:
  - research/plans/tri_camera_node_3d_aware_reid/master_plan_zh.md
  - research/plans/tri_camera_node_3d_aware_reid/benchmarks/node01_recon_spin_v1.json
  - research/handoffs/tri_camera_node_engineering_handoff_zh.md
  - research/guides/node01_recon_spin_points_v1_zh.md
  - research/guides/node01_spin_gt_validation_zh.md
locked_decisions:
  - 当前项目默认运行环境固定为 `conda` 环境 `mvp_demo`；未特别说明时，项目脚本、smoke check 和主线命令均默认在该环境中执行。
  - 研究主线固定为“三相机节点级 3D-aware track retrieval”，YOLO 门控 + 3DGS 仅保留为辅助 demo。
  - ICISCAE 小论文目标域固定为 UAV/aircraft；当前不再做人形 benchmark。
  - ICISCAE 的正式范围固定为 node01 的 single-node、cross-scene、track-level retrieval；允许 MuJoCo-only 仿真结果，但论文叙事必须写成仿真节点检索验证，不能宣称 cross-node 已完成。
  - 毕业论文继续沿同一数据契约扩展到 cross-node retrieval、真实三相机节点接入和系统性误差分析。
  - 主链路只消费 frames、masks、depth、rig.json、frame_times.csv；MuJoCo GT 仅用于排错和上界评测。
  - 正式 benchmark 的 mask 输入布局固定为平铺 cams/cam*/masks/<ts>.png；SAM2 的 obj_XXX 只允许作为中间产物，不直接进入正式 benchmark。
  - identity_id 的权威来源固定为 capture_meta.target.identity_id；build_node_tracklets.py --identity_id 只用于补历史 scene。
  - 正式 benchmark 的历史 `v1` 身份集合保留为 `j10 / uav1 / dji_mavic`；历史 `v2` 身份集合保留为 `j10 / uav1 / su34`；当前激活 benchmark 固定为 clean 主线 `v3_clean`，身份集合仍为 `j10 / uav1 / su34`。
  - Windows 下包含中文资产目录的 MuJoCo scene 可能无法被直接加载；当前激活的 `uav1_v2` 与 `su34` 都固定使用 ASCII-safe 资产路径，从 `D:\grad_project_ascii\mvp-demo` 运行。
  - 当前激活的正式 scene 协议固定为每个 identity 两条轨迹：`line_nodes` 和 `circle_xz`；`v3_clean` 的正式 scene_id 固定为 `mj_node01_{j10|uav1|su34}_clean_{line_nodes_a|circle_xz_b}`，`split_role` 统一为 `both`。
  - 正式检索评测默认开启 exclude_same_track_id 和 exclude_same_scene；研究目标矩阵保持 RGB-only / RGB + predicted-depth geometry / RGB + fused geometry，但工程顺序先跑 RGB-only，再补两条几何支路。
  - 当前小论文结果不得把 MuJoCo GT 作为主链输入；现有基于 masks_gt/depth_gt 跑通的 scene 仅作为 proof-of-pipeline 和 upper-bound 证据。
  - `RGB + predicted-depth geometry` 的当前冻结实现固定为 `cam0` predicted `depth + mask` 回投点云，`points_subdir = recon/points_depth_cam0`，`geo_backend = open3d_fpfh`。
  - `RGB + fused geometry` 的当前冻结实现固定为三相机 predicted `depth + mask` 融合点云，`points_subdir = recon/points_fused`，`geo_backend = open3d_fpfh`。
  - `GT upper-bound` 只作为分析线，固定为 `masks_gt + depth_gt + fused geometry`，当前激活输出目录为 `recon/points_fused_gt` 与 `mvp-demo/output/evals/iciscae_node01_uav_v3_clean/gt_upper_bound/`。
  - 当前激活的重建主线固定为 `recon_spin_points.py`：先把多时刻点云变到目标 canonical 坐标系，再做 canonical 支撑过滤与逐时刻回投；预测分支的默认支撑策略固定为 `static: 3 -> 2`、`circle: 4 -> 3 -> 2` 的自动回退，GT 分支固定为 `1`。
  - 当前默认采集、默认 viewer 和默认 benchmark 都切换到无 humanoid 的 clean 场景；`mvp-demo/assets/scene/` 根目录不再保留任何 `mujoco_humanoid_*.xml`。
  - 所有 humanoid 场景统一归档到 `mvp-demo/assets/scene/legacy/`：`legacy/v1/` 用于历史 `v1` 复现，`legacy/humanoid/` 用于从主线下线的 humanoid scene。
  - 权威研究文档统一收口到 research；mvp-demo 仅保留运行入口与资产说明。
next_action: 继续围绕 `j10_static`、`su34_static` 与 `uav1_static` 做第二轮 predicted recon 清理，优先定位 static scene 中的 mask/depth 漏检与 `uav1_static` 的 canonical 饱和来源；在此基础上重跑 `node01_recon_spin_v1` 的 `rgb_recon_enhanced_geometry` 与分支对比汇总，并整理新的组会/导师汇报口径。
out_of_scope:
  - 多目标关联与多实例同时检索
  - dynamic 3DGS 作为当前主线方法
  - 训练新的端到端 3D encoder
  - 大规模真实数据采集
  - 不把 obj_XXX 嵌套 mask 直接作为正式 benchmark 输入
  - 不把 cross-node retrieval 或真实节点迁移作为当前小论文成功条件
  - 不把组会阶段文档并入主线 research 文档
latest_retrospective: none
last_updated: 2026-03-23
