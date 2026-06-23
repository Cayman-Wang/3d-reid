# CARLA-Air 航迹恢复实现指导方案

## 目标

本方案用于落实导师提出的“一种飞机条件下，当前 3D-ReID 项目是否能够复现出航迹”的要求。这里的“复现出航迹”不是传统 ReID 排名任务本身，而是一个 ReID / tracklet 之后的多视角 3D 航迹恢复任务：

```text
多相机图像 / ReID tracklet 结果
-> 多视角 2D 观测聚合
-> CARLA world 坐标系下的时序 3D 航迹 t,x,y,z
-> 与仿真真值 drone_carla_pose_json 对比误差
```

v1 采用两层设计：

- 行为边界：只恢复单一飞机、单一身份的时序 3D 航迹。
- 数据结构：从第一版开始保留 `identity_id`、`model_id`、`track_id`、`trajectory_id` 等字段，结构上支持后续不同模型 / 多身份导入。

因此 v1 是“结构上支持多身份，行为上只启用单身份”。多身份数据关联、不同飞行器模型导入、track 分裂 / 合并、遮挡冲突处理都不进入 v1 验收。

如果 v1 输入中检测到多个 `identity_id`，或同一时间窗口存在多个 active `track_id`，默认必须报错并提示进入后续多身份阶段，不得静默混合轨迹。

## 当前可用输入

默认先使用已有离线 capture，不启动 CARLA-Air / AirSim runtime：

```text
local/carla_air/captures/view_traj_cov_01_default_drone/
local/carla_air/captures/view_traj_cov_02_default_drone/
```

核心时间同步与真值来自：

```text
trajectory_frame_groups.csv
```

关键字段：

- `trajectory_id`：轨迹配置 ID。
- `identity_id`：当前为 `default_airsim_drone`，后续多身份阶段继续沿用。
- `node_id`：地面相机节点，例如 `node01`。
- `planned_frame_index` / `planned_t_sec`：规划时间索引。
- `ts_us`：采集时间戳，建议作为跨节点 / 跨相机对齐主键。
- `drone_carla_pose_json`：仿真真值航迹，字段包含 `x,y,z,pitch,yaw,roll`。
- `camera_meta_json`：每个相机的 `K`、`T_node_from_cam`、`carla_world_transform`、`image_size`。
- `image_files_json`：每个相机对应图像路径。
- `node_dir`：节点目录。

相机标定也可从各节点读取：

```text
nodes/<node_id>/calib/rig.json
```

v1 后续观测统一按以下键组织，即使当前只有一个身份：

```text
identity_id
model_id
track_id
trajectory_id
ts_us
node_id
camera_id
```

其中 `model_id` 在当前 default drone 阶段可填 `default_airsim_drone_model` 或 `unknown_default_drone_model`，但字段必须保留。

## 推荐实现路线

建议新增独立脚本，不混入 ReID 训练或 embedding 提取代码：

```text
tools/carla_air/recover_trajectory_from_multiview_tracklets.py
```

建议命令接口：

```bash
python tools/carla_air/recover_trajectory_from_multiview_tracklets.py \
  --capture-root local/carla_air/captures/view_traj_cov_01_default_drone \
  --observation-mode oracle_projection \
  --identity-policy single_strict \
  --output-dir local/carla_air/trajectory_recovery/view_traj_cov_01_oracle_projection
```

参数约定：

- `--capture-root`：输入 capture 根目录。
- `--tracklets`：可选 ReID / detector tracklet JSON；oracle 阶段可不传。
- `--observation-mode oracle_gt_pose|oracle_projection|tracklet_bbox|tracklet_keypoint`：观测来源。
- `--identity-policy single_strict`：v1 唯一支持策略。
- `--output-dir`：输出目录。

保留但 v1 不启用的未来策略：

```text
--identity-policy multi_identity_associated
```

该策略只作为 reserved / future，后续多身份阶段再实现。

### M0 数据审计

读取 `trajectory_frame_groups.csv`，按 `trajectory_id, identity_id, ts_us, node_id, camera_id` 建立观测索引，检查：

- 每个 `ts_us` 是否有多个节点 / 相机可用。
- `drone_carla_pose_json` 是否存在且可解析。
- `camera_meta_json` 中 `K`、`carla_world_transform`、`image_size` 是否齐全。
- `identity_id` 是否只有一个取值。

若 `--identity-policy single_strict` 下出现多个 `identity_id` 或多个 active `track_id`，直接失败。

### M1 真值投影 sanity check

用 `drone_carla_pose_json` 中的 CARLA world 位置作为 3D 点，结合每个相机的内外参投影到图像平面，输出每帧每相机的真值投影点。

该阶段用于验证：

- CARLA world -> camera -> pixel 的矩阵方向是否正确。
- 相机 `carla_world_transform` 是否可直接构造 world-from-camera / camera-from-world。
- 投影点是否落在 `image_size` 范围附近。

M1 不恢复航迹，只确认几何链路没有坐标系错误。

### M2 oracle 三角化

M2 是 v1 的首个核心闭环，只处理一个 `identity_id`：

```text
drone_carla_pose_json 真值 3D 点
-> 投影生成 oracle 2D observation
-> 多相机射线最小二乘三角化
-> recovered 3D point
-> 与原始 drone_carla_pose_json 对比误差
```

如果 M2 误差明显偏大，应优先排查：

- world-from-camera / camera-from-world 是否反了。
- CARLA 坐标轴和相机坐标轴定义是否混用。
- 同一 `ts_us` 下跨节点 / 跨相机是否对齐。
- 投影点是否来自同一个真值时间戳。

M2 的理想误差应接近数值误差或小量浮点误差。它是后续接入真实 ReID tracklet 前的几何必经门槛。

### M3 ReID tracklet 接入

M3 只允许一个目标 track，使用 ReID / detector / mask tracklet 结果作为 2D 观测来源：

- `tracklet_bbox`：使用 bbox center 作为 2D 点。
- `tracklet_keypoint`：使用 mask centroid、目标中心点或可见关键点作为 2D 点。

同一 `ts_us` 下需要至少两个有效相机观测才能三角化。少于两个观测的帧记为 `recovery_status=insufficient_observations`，不得伪造恢复点。

M3 不实现多身份 association。若 tracklet 输入中存在多个 active `track_id`，v1 必须失败并提示后续进入多身份阶段。

### M4 误差报告与可视化

将 recovered trajectory 与 `drone_carla_pose_json` 真值按 `ts_us` 对齐，输出误差表、摘要 JSON 和可视化。

可复用现有轨迹空间预览思路，但必须明确区别：

- `docs/carla_air_trajectory_spatial_visualization_plan_zh.md` 是轨迹展示 / QC。
- 本方案是从观测反推航迹并做误差评估。

### M5 后续多身份扩展

M5 才进入不同模型 / 多身份能力：

- 多身份 ReID association。
- 多个 `identity_id` / `model_id` 的 tracklet-to-identity 绑定。
- track 分裂 / 合并。
- 遮挡与冲突处理。
- 分身份、分 track、分模型的航迹恢复误差评测。

M5 不改变 v1 输出字段，只扩展 `by_identity` / `by_track` / `by_model` 的条目数量和数据关联逻辑。

## 输出契约

默认输出目录：

```text
local/carla_air/trajectory_recovery/<run_name>/
```

建议生成：

```text
recovered_trajectory.csv
trajectory_error_summary.json
trajectory_observations.csv
trajectory_compare.svg
trajectory_compare.png
trajectory_compare.html
manifest.json
```

`recovered_trajectory.csv` 必须包含：

```text
identity_id
model_id
track_id
trajectory_id
ts_us
planned_t_sec
x
y
z
source_observation_count
recovery_status
```

其中：

- `x,y,z` 是恢复出的 CARLA world 坐标。
- `source_observation_count` 是该时间戳参与三角化的相机观测数量。
- `recovery_status` 建议使用 `ok|insufficient_observations|triangulation_failed|filtered_out`。

`trajectory_observations.csv` 建议包含：

```text
identity_id
model_id
track_id
trajectory_id
ts_us
planned_t_sec
node_id
camera_id
u
v
observation_mode
observation_status
```

`trajectory_error_summary.json` 必须预留分组结构：

```json
{
  "schema_version": "carla_air_trajectory_recovery_error_v1",
  "identity_policy": "single_strict",
  "overall": {},
  "by_identity": {},
  "by_track": {}
}
```

v1 中 `by_identity` 和 `by_track` 只有一个条目，但结构必须存在。

建议 `overall` / `by_identity` / `by_track` 至少报告：

- `frame_count`
- `triangulated_frame_count`
- `coverage_ratio`
- `rmse_3d_m`
- `mae_3d_m`
- `median_3d_m`
- `p95_3d_m`
- `max_3d_m`
- `rmse_xy_m`
- `rmse_z_m`

`manifest.json` 必须记录：

- 输入 capture root。
- 观测模式。
- identity policy。
- 唯一 `identity_id`、`track_id`、`model_id`。
- 输出文件清单和 hash。
- 是否使用 runtime：v1 默认 `starts_runtime=false`。
- non-promotion 边界。

## 参考项目借鉴

已下载参考索引：

```text
参考工作/papers/trajectory_recovery_refs/README.md
```

可借鉴关系：

- OpenPTrack：参考多相机标定、同步、3D 轨迹输出的系统划分。
- 3D-Visual-MOT：参考多视角视觉观测到 3D MOT 输出的工程组织方式。
- MVTracker：参考 observation / tracklet / trajectory 的接口组织。
- ADA-Track：参考 3D tracking 中的数据关联和 ablation 写法，主要服务后续 M5。
- TrackEval：参考 HOTA / MOTA / IDF1 等 tracking 评测组织方式，v1 仍以 3D 航迹误差为主。
- evo：参考 APE / RPE 轨迹误差指标和可视化，必要时将本项目 `t,x,y,z` 转换为 evo 支持格式。

这些项目只作为实现和论文写法参考，不直接替换当前 CARLA-Air / 3D-ReID 主线。

## 验收标准

文档级检查：

```bash
test -s docs/carla_air_trajectory_recovery_implementation_plan_zh.md
git diff --check
```

后续脚本实现后的基础检查：

```bash
python -m py_compile tools/carla_air/recover_trajectory_from_multiview_tracklets.py
```

oracle smoke 示例：

```bash
python tools/carla_air/recover_trajectory_from_multiview_tracklets.py \
  --capture-root local/carla_air/captures/view_traj_cov_01_default_drone \
  --observation-mode oracle_projection \
  --identity-policy single_strict \
  --output-dir local/carla_air/trajectory_recovery/view_traj_cov_01_oracle_projection
```

验收口径：

- 单身份输入正常输出 `recovered_trajectory.csv` 和 `trajectory_error_summary.json`。
- `trajectory_error_summary.json` 必须包含 `overall`、`by_identity`、`by_track`。
- 即使只有一个身份，`by_identity` 和 `by_track` 也必须存在且各含一个条目。
- `trajectory_error_summary.json` 必须包含 `rmse_3d_m`、`mae_3d_m`、`coverage_ratio`、`triangulated_frame_count`。
- 人工构造两个 `identity_id` 的输入时，v1 必须失败并给出明确错误，不得静默混合轨迹。
- 人工构造多个 active `track_id` 的输入时，v1 必须失败并提示 `multi_identity_associated` 属于后续阶段。

## 实现边界

本方案不改变当前项目完成状态：

- 不推进不同飞行器模型导入。
- 不启动 CARLA-Air / AirSim runtime。
- 不生成 formal annotation。
- 不把 proxy / candidate / pseudo 当作 `mask_gt`。
- 不声明 `formal_neoverse_ready=true`。
- 不声明 `real_4d_geometry_ready=true`。
- 不声明多身份 benchmark 已完成。

航迹恢复 v1 的意义是优先回答导师当前问题：在一种飞机、已有多相机 capture 和仿真真值条件下，项目是否能从观测反推出时序 3D 航迹，并给出可量化误差。
