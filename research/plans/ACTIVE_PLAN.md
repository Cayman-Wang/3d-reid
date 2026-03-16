# ACTIVE_PLAN

goal: 在 MuJoCo 三相机节点上完成严格传感器口径的 3D-aware track retrieval 验证，并在相同数据契约下迁移到真实三相机节点。
current_milestone: M0 协议冻结与 benchmark manifest 落地
must_read:
  - research/plans/tri_camera_node_3d_aware_reid/master_plan_zh.md
  - 3D重建-3Dreid/sim_to_real_3cam_node_reid_research.md
  - mvp-demo/README.md
  - mvp-demo/todo/mujoco_3cam_node_handoff.md
locked_decisions:
  - 研究主线固定为“三相机节点级 3D-aware track retrieval”，YOLO 门控 + 3DGS 仅保留为辅助 demo。
  - 近阶段正式 benchmark 先做 node01 的多 scene、cross-scene、track-level retrieval；cross-node retrieval 放到 M5 之后再启动。
  - 主链路只消费 frames、masks、depth、rig.json、frame_times.csv；MuJoCo GT 仅用于排错和上界评测。
  - 官方 benchmark 的 mask 输入布局固定为平铺 cams/cam*/masks/<ts>.png；SAM2 的 obj_XXX 只允许作为中间产物，不直接进入正式 benchmark。
  - identity_id 的权威来源固定为 capture_meta.target.identity_id；build_node_tracklets.py --identity_id 只用于补历史 scene。
  - 正式检索评测默认开启 exclude_same_track_id 和 exclude_same_scene；研究目标矩阵保持 RGB-only / RGB + predicted-depth geometry / RGB + fused geometry，但工程顺序先跑 RGB-only，再补两条几何支路。
next_action: 生成最小 benchmark manifest，覆盖 2-3 个 identity、每个 identity 至少 2 个 scene，并记录 scene_dir / identity_id / split_role / mask_source / depth_source / mask_layout / sam2_camera_boxes / min_valid_timestamps / eval_out_json。
out_of_scope:
  - 多目标关联与多实例同时检索
  - dynamic 3DGS 作为当前主线方法
  - 训练新的端到端 3D encoder
  - 大规模真实数据采集
  - 不把 obj_XXX 嵌套 mask 直接作为正式 benchmark 输入
latest_retrospective: none
last_updated: 2026-03-15
