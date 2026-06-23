# CARLA-Air 4D 几何 Sidecar 契约

本文档只定义 `local/carla_air/geometry_4d/<capture_id>/<method>/` 的旁路几何产物契约。
它是诊断性 sidecar 约定，不是正式几何标准，也不替代现有 oracle / 三角化基线。

## 适用范围

固定 `method` 名称：

```text
dggt
mapanything
dggt_mapanything_aligned
```

统一目录结构：

```text
local/carla_air/geometry_4d/<capture_id>/<method>/
  manifest.json
  input_manifest.json
  depth_by_frame/
  points_by_timestamp/
  camera_alignment.json
  quality_summary.json
```

sidecar readiness 总览输出固定为：

```text
local/carla_air/geometry_4d/readiness/readiness_summary.json
```

## 强约束

所有 sidecar 产物都必须显式标记：

```text
diagnostic_only=true
non_promotion=true
not_formal_geometry=true
```

这意味着：

- 只能用于诊断、对比、回溯和烟雾检查。
- 不能被解释为正式 synthetic annotation。
- 不能被解释为最终真实 4D geometry。
- 不能替代 `oracle_projection`。
- 不能替代 bbox triangulation baseline，包含现有 `tracklet_bbox_yolo`、`tracklet_bbox_diff`、`tracklet_bbox_depth` 等对照链路。

## 文件语义

`manifest.json`
: 记录 capture、method、模型/权重、输入摘要、hash、许可信息、non-promotion 边界，以及上面的三项布尔约束。

`input_manifest.json`
: 记录本次 sidecar 使用的原始输入来源、时间戳范围、相机、同步关系和预处理参数。

`local/carla_air/geometry_4d/readiness/readiness_summary.json`
: 记录当前 sidecar 的本地 readiness 检查结果，只覆盖 repo、权重、软链接和 Python import 可用性。它不运行 DGGT 或 MapAnything 推理，不启动 CARLA-Air / AirSim runtime，不代表 geometry 已生成。

`depth_by_frame/`
: 保存按 frame / camera / timestamp 切分的深度结果，供后续 ROI 反投影、深度采样和异常帧排查。

`points_by_timestamp/`
: 保存按 timestamp 组织的点数据。默认应以 world-frame `xyz` 为主，可附加 `rgb`、`confidence`、`source_model`、`node_id`、`camera_id`、`ts_us` 等字段。

`camera_alignment.json`
: 记录 DGGT 或 MapAnything 侧路相机姿态与 CARLA 参考姿态的对齐误差、偏差方向和统计摘要。

`quality_summary.json`
: 记录点数、有效深度比例、ROI 命中数、异常帧、缺失帧、对齐质量等汇总指标。

placeholder exporter 只允许闭合 metadata 契约：

- 可以写 `manifest.json`、`camera_alignment.json`、`quality_summary.json`。
- 可以创建空的 `depth_by_frame/`、`points_by_timestamp/` 目录。
- 不允许伪造 depth、points、alignment metrics 或质量统计。
- 在 placeholder 阶段，`geometry_ready=false`、`depth_ready=false`、`points_ready=false`、`inference_executed=false` 必须保持为 `false`。
- `formal_annotation_ready=false`、`final_4d_geometry_ready=false`、`benchmark_ready=false` 必须显式保留。

## `tracklet_bbox_depth` 的消费方式

`tracklet_bbox_depth` 只消费 sidecar 的局部几何，不直接把整场景几何当作真值。典型流程如下：

1. 取 tracklet bbox。
2. 在对应 frame / timestamp 中读取 `depth_by_frame/` 和 `points_by_timestamp/`。
3. 将 bbox 内像素或局部区域做深度反投影，筛掉低置信度点和明显离群点。
4. 按相机或多相机做融合，必要时再做 RANSAC / 稳健聚合。
5. 输出每帧目标 3D center、恢复轨迹和误差汇总。

因此，`tracklet_bbox_depth` 依赖的是 sidecar 的深度与点数据，但它的结果仍是诊断/对比产物，不自动升级为正式几何。

## 与基线的关系

这份 sidecar 契约只服务于旁路诊断，不改动主链口径：

- `oracle_projection` 继续作为最直接的参考基线。
- bbox triangulation baseline 继续保留为主对照。
- `dggt`、`mapanything`、`dggt_mapanything_aligned` 只是在同一输入上提供可比的局部几何输出。

任何报告都必须保持这个边界，不得把 sidecar 结果写成“已完成正式几何”或“已替代基线”。

## 4D-ReID 口径

4D-ReID 这里只能做单身份 sanity check。

允许的说法：

- single-identity 4D-ReID sanity
- 局部几何特征可用性检查
- 跨捕获或跨视角的检索烟雾测试

不允许的说法：

- 多身份 benchmark 已完成
- 正式 4D-ReID benchmark 已建立
- sidecar 几何已经可作为最终训练真值

## Input Manifest Dry-Run 验收

`input_manifest.json` 的 dry-run 验收只检查输入可追溯性，不要求已经产出真实几何：

- `input_manifest.json` 存在。
- 目录位于 `local/carla_air/geometry_4d/<capture_id>/<method>/`。
- `method` 允许为 `dggt`、`mapanything`、`dggt_mapanything_aligned`。
- 明确写出 `diagnostic_only=true`、`non_promotion=true`、`not_formal_geometry=true`。
- 至少包含 `capture_id`、`identity_id`、`trajectory_id`、`selected_ts_us` 和 `views`。
- `views` 至少记录 `node_id`、`camera_id`、`frame_path`、`K`、`camera_pose_c2w`、`camera_extrinsic_w2c`、`drone_gt_pose`。
- dry-run 不复制图片、不写大体积点云、不写正式 `depth_by_frame/` 或 `points_by_timestamp/` 结果。
- `local/carla_air/geometry_4d/readiness/readiness_summary.json` 可作为额外准备信号，但它只说明本地环境/文件 readiness，不说明 sidecar geometry 已可用。
- placeholder exporter 生成的 metadata 也不属于 dry-run 输入验收的必需项；它只是把目录契约补齐到 metadata 层。

## Full Sidecar Geometry Output 验收

任一 `<capture_id>/<method>/` 目录至少应满足：

- `manifest.json` 存在，且明确写出 `diagnostic_only=true`、`non_promotion=true`、`not_formal_geometry=true`。
- `input_manifest.json` 存在。
- `depth_by_frame/` 与 `points_by_timestamp/` 至少其一非空，且与 capture 时间范围一致。
- `camera_alignment.json` 与 `quality_summary.json` 存在。
- 输出未被 downstream 当作 oracle、baseline replacement 或 formal geometry 使用。

额外边界：

- placeholder exporter 只闭合 metadata 契约，仍然不满足 Full Sidecar Geometry Output 验收。
- 仅当后续真实模型推理完成，并且实际写入有效 `depth_by_frame/` / `points_by_timestamp/` 后，才允许把 `depth_ready` / `points_ready` 改为 `true`。
- 仅当真实对齐计算已执行后，才允许把 `alignment_ready` 改为 `true`。
