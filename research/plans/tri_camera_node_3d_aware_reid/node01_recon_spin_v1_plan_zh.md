# node01_recon_spin_v1 冻结规划

- 状态：已冻结，等待按主线分支逐步落地
- 适用分支：`feat/node01_recon_3daware_retrieval`
- 不影响分支：`paper/node01_v3_clean_manuscript`

## 1. 目标

- 在固定三相机、已知相机位姿条件下，引入轻度目标自转，提升 object-centric geometry 的可见面覆盖。
- 第一阶段不直接上完整 `4D reconstruction`，优先做 `known-camera-pose + object-centric reconstruction`。
- 新几何产物必须继续复用当前 `tracklets / embeddings / eval` 契约，不改下游输入格式。

## 2. 协议冻结

- benchmark id：`node01_recon_spin_v1`
- identity：`j10 / uav1 / su34`
- 每个 identity 两条 scene：
  - `static_spin_yaw_pitch_a`
  - `circle_xz_spin_yaw_pitch_b`
- 固定采集参数：
  - `seconds = 8`
  - `fps = 30`
  - `num_frames = 240`
  - `traj_center = (0, 6, 2)`
  - `traj_radius = 1.0` for `circle_xz_spin_yaw_pitch`
- 固定自转参数：
  - `spin_axes = [yaw, pitch]`
  - `yaw: -60° -> +60°`
  - `pitch_amp_deg = 10`
  - `pitch_period = 8`
  - `roll = disabled`

## 3. 执行顺序

1. 先完成 spin 采集协议与 manifest。
2. 先用 `masks_gt + depth_gt` 验证自转协议是否带来更完整的几何。
3. GT 跑通后，再切到 `SAM2 masks + predicted depth`。
4. 输出统一回写为逐时间戳几何：
   - `recon/points_recon_spin_gt`
   - `recon/points_recon_spin`
5. 下游沿用：
   - `build_node_tracklets.py`
   - `extract_node_track_embeddings.py`
   - `eval_node_track_retrieval.py`

## 4. 验证口径

- 采集成功：
  - 240 个同步时间戳完整落盘
  - `capture_meta.json` 含完整 spin 参数
  - `static_spin` 可见明确侧面与部分后侧面
- 几何成功：
  - 与 `recon/points_fused_gt` 相比，可见面覆盖增加
  - 几何连续性和稳定性改善
  - 新几何与 `frame_times.csv` 一一对齐
- 检索成功：
  - `gt_recon_enhanced_geometry` 优于 `gt_upper_bound` 或至少提供明确可视化改善
  - `rgb_recon_enhanced_geometry` 优于 `rgb_fused_geometry`
- 归因规则：
  - GT 提升、预测不提升：瓶颈在 `SAM2/depth`
  - GT 也不提升：瓶颈在 spin 协议或重建后端
  - 几何变好但检索不涨：瓶颈转到 geometry descriptor / fusion

## 5. 组会素材

- 采集推进过程中的可视化可直接用于组会，但口径只允许写成：
  - 可见面覆盖是否增加
  - 三视图输入质量是否稳定
  - 新旧几何是否更完整
- 每个 spin scene 建议至少导出：
  - `presentation_assets/triview_video.mp4`
  - 3 张 `presentation_assets/overview_<stem>.png`
  - 1 组旧几何 vs 新几何对比图
  - 1 组失败例或边界例

## 6. 当前不做

- 不覆盖 `v3_clean` 的 benchmark id 和 scene id
- 不把 `4D reconstruction` 作为第一阶段主线
- 不在第一版启用 `roll`
- 不先改 retrieval 契约
