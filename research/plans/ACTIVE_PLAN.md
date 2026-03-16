# ACTIVE_PLAN

goal: 先用 MuJoCo 在 node01 上完成 UAV/aircraft 的 single-node、cross-scene、track-level 3D-aware retrieval benchmark，作为 ICISCAE 小论文；再在同一数据契约上扩展到 cross-node 与真实三相机迁移，作为毕业论文主线。
current_milestone: M1 第三个飞行目标落地与 node01 正式 3x2 benchmark 采集
must_read:
  - research/plans/tri_camera_node_3d_aware_reid/master_plan_zh.md
  - research/plans/tri_camera_node_3d_aware_reid/benchmarks/iciscae_node01_uav_v1.json
  - 3D重建-3Dreid/sim_to_real_3cam_node_reid_research.md
  - mvp-demo/README.md
  - mvp-demo/todo/mujoco_3cam_node_handoff.md
locked_decisions:
  - 研究主线固定为“三相机节点级 3D-aware track retrieval”，YOLO 门控 + 3DGS 仅保留为辅助 demo。
  - ICISCAE 小论文目标域固定为 UAV/aircraft；当前不再做人形 benchmark。
  - ICISCAE 的正式范围固定为 node01 的 single-node、cross-scene、track-level retrieval；允许 MuJoCo-only 仿真结果，但论文叙事必须写成仿真节点检索验证，不能宣称 cross-node 已完成。
  - 毕业论文继续沿同一数据契约扩展到 cross-node retrieval、真实三相机节点接入和系统性误差分析。
  - 主链路只消费 frames、masks、depth、rig.json、frame_times.csv；MuJoCo GT 仅用于排错和上界评测。
  - 正式 benchmark 的 mask 输入布局固定为平铺 cams/cam*/masks/<ts>.png；SAM2 的 obj_XXX 只允许作为中间产物，不直接进入正式 benchmark。
  - identity_id 的权威来源固定为 capture_meta.target.identity_id；build_node_tracklets.py --identity_id 只用于补历史 scene。
  - 正式 benchmark 规模固定为 3 identities x 2 scenes，默认身份集合为 j10 / uav1 / dji_mavic；若 DJI Mavic 资产无法稳定导入 MuJoCo，则 fallback 为 大疆无人机 的纯几何或手工材质版本。
  - Windows 下包含中文资产目录的 MuJoCo scene 可能无法被直接加载；当前已验证 dji_mavic 可通过 ASCII 仓库别名加载，uav1 若在 Windows 直跑失败则需改用 ASCII 资产别名或 Linux/WSL 环境。
  - 正式 scene 协议固定为每个 identity 两条轨迹：line_nodes 和 circle_xz；split_role 统一为 both。
  - 正式检索评测默认开启 exclude_same_track_id 和 exclude_same_scene；研究目标矩阵保持 RGB-only / RGB + predicted-depth geometry / RGB + fused geometry，但工程顺序先跑 RGB-only，再补两条几何支路。
  - 当前小论文结果不得把 MuJoCo GT 作为主链输入；现有基于 masks_gt/depth_gt 跑通的 scene 仅作为 proof-of-pipeline 和 upper-bound 证据。
next_action: 按 manifest 验证 dji_mavic 场景可加载，并为 j10 / uav1 / dji_mavic 采集 node01 的 6 个正式 scene（显式 scene_id、显式 identity_id），随后补齐 predicted depth、predicted masks 和 RGB-only 首轮评测结果。
out_of_scope:
  - 多目标关联与多实例同时检索
  - dynamic 3DGS 作为当前主线方法
  - 训练新的端到端 3D encoder
  - 大规模真实数据采集
  - 不把 obj_XXX 嵌套 mask 直接作为正式 benchmark 输入
  - 不把 cross-node retrieval 或真实节点迁移作为当前小论文成功条件
  - 不把组会阶段文档并入主线 research 文档
latest_retrospective: none
last_updated: 2026-03-16
