# ACTIVE_PLAN

goal: 在 MuJoCo 三相机节点上完成严格传感器口径的 3D-aware track retrieval 验证，并在相同数据契约下迁移到真实三相机节点。
current_milestone: M0 协议冻结与最小实验协议落地
must_read:
  - research/plans/tri_camera_node_3d_aware_reid/master_plan_zh.md
  - 3D重建-3Dreid/sim_to_real_3cam_node_reid_research.md
  - mvp-demo/README.md
locked_decisions:
  - 研究主线固定为“三相机节点级 3D-aware track retrieval”，YOLO 门控 + 3DGS 仅保留为辅助 demo。
  - 主链路只消费 frames、masks、depth、rig.json、frame_times.csv；MuJoCo GT 仅用于排错和上界评测。
  - 当前实验单位固定为单目标、单轨迹、track-level retrieval。
  - 近期评测矩阵固定为 RGB-only、RGB+predicted depth、RGB+fused geometry。
next_action: 冻结最小实验协议（2-3 个 identity、至少 2 个 scenes、明确 query/gallery 划分），并补齐这批 scene 的 predicted depth 与 SAM2 masks。
out_of_scope:
  - 多目标关联与多实例同时检索
  - dynamic 3DGS 作为当前主线方法
  - 训练新的端到端 3D encoder
  - 大规模真实数据采集
latest_retrospective: none
last_updated: 2026-03-15
