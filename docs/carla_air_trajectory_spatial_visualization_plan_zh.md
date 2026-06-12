# CARLA-Air 轨迹空间可视化规划

## 目标

为当前已实现的轨迹生成功能增加一个离线空间预览能力，用于展示生成航迹、相机节点位置与局部场景尺度之间的相对关系。默认不推进不同飞行器模型导入，不启动 CARLA-Air / AirSim runtime，不改变正式数据集、标注或几何管线。

该能力面向组会展示、人工质检和 waypoint 调参前的空间审查。核心视图应能回答：

- 轨迹在 CARLA world XY 平面上的位置和走向。
- `node01-node05` 相机节点相对轨迹的位置。
- 轨迹 waypoint、高度范围、局部 ROI 尺度和米制比例尺。
- 可选 runtime 地图尺度信息是否可读，以及它与局部 ROI 的关系。

## 默认输入

默认读取现有离线配置：

```text
configs/carla_air/trajectories/town10hd_coverage_first_v1.json
local/carla_air/camera_nodes/Town10HD_ground_to_air_nodes_v1.json
```

轨迹优先使用 `waypoints_carla`；若缺失，则回退到兼容字段 `waypoints`。相机节点使用节点配置中的 `anchor_transform`，并用 yaw / optical-axis 方向生成节点方向提示。

## 默认输出

建议新增独立导出脚本：

```text
tools/carla_air/export_trajectory_spatial_preview.py
```

默认输出目录按日期写入 `组会思路/`，例如：

```text
组会思路/carla_air_trajectory_spatial_preview_2026_06_12/
```

默认生成以下文件：

```text
trajectory_spatial_preview.svg
trajectory_spatial_preview.png
trajectory_spatial_preview.html
trajectory_spatial_preview_3d.svg
trajectory_spatial_preview_3d.png
trajectory_spatial_preview_3d.html
vendor/three.min.js
manifest.json
```

其中：

- `trajectory_spatial_preview.svg/png/html`：保留 2D 顶视图和高度 profile，用于精确读图。
- `trajectory_spatial_preview_3d.svg/png`：静态 3D 等距视图，用于组会材料和快速人工检查。
- `trajectory_spatial_preview_3d.html`：基于 vendored Three.js 的离线交互 3D 视图，支持鼠标旋转、滚轮缩放、图层开关和 Z scale 调整。
- `vendor/three.min.js`：随输出目录复制的 Three.js classic/global 文件，HTML 使用相对路径引用，不依赖 CDN，支持离线 file-open。
- `manifest.json`：记录输入、输出、轨迹计数、节点计数、局部尺度、hash、`view_modes`、`z_scale_default`、Three.js vendor 信息与 non-promotion 边界。

## 可视化内容

默认画布包含：

- XY 顶视图：用折线连接每条 trajectory 的 waypoint。
- Waypoint 标记：显示 waypoint 顺序和关键点。
- 相机节点：显示 `node01-node05` anchor 位置，并用短箭头表示节点朝向。
- 推荐节点：对 trajectory 的 `recommended_nodes` / `target_nodes` 做高亮。
- 局部 ROI：用轨迹点和节点点位的联合包围盒生成米制局部范围。
- 比例尺：显示固定米制 scale bar，例如 `20 m`。
- 高度 profile：展示 trajectory `z` 随路径距离变化，单位为米。

当前 runner 对 waypoint 做分段线性插值，因此默认可视化应使用折线表达真实执行语义。如需更平滑的曲线，只能作为视觉辅助，不能替代 waypoint / runner 语义。

3D 视图额外包含：

- 静态 3D：轨迹折线、waypoint、camera node、node yaw arrow、地面米制网格、XYZ 轴和局部 ROI 3D bounding cage。
- 交互 3D：Three.js 场景中的彩色 3D polyline、小球 waypoint、节点方向箭头、透明 ROI 线框盒和米制地面网格。
- Z 轴默认真实米制比例 `1.0`。静态输出可用 `--z-scale` 指定垂直缩放；交互 HTML 可用滑条调整。非 `1.0` 时必须在图例和 manifest 中记录为垂直夸张显示。

## 场景尺度策略

默认采用两层尺度策略：

```text
local
runtime
```

`local` 为默认模式：只使用轨迹 waypoint 和相机节点 anchor 计算局部 ROI，不依赖 runtime。它必须始终可用。

`runtime` 为可选补充：当 CARLA-Air / AirSim runtime 已按 runbook 可用时，可尝试读取 Town10HD map extent 或可替代的地图尺度信息，并写入 manifest。若读取失败，不应阻断离线图生成，只记录：

```text
map_extent_available=false
```

默认实现不得自动启动 runtime。若后续确实需要 live map extent，必须遵守 `research/guides/carla_air_runtime_self_start_zh.md` 的端口检查、日志、PID 和失败报告规则。

## 实现边界

该可视化只用于展示、人工质检和轨迹调参，不作为正式 pipeline evidence。

manifest 中必须明确记录：

```text
presentation_only=true
manual_qc_only=true
not_formal_dataset_input=true
not_annotation_evidence=true
not_mask_evidence=true
not_bbox_evidence=true
not_geometry_evidence=true
not_benchmark_evidence=true
```

不得因为该图存在而改变以下状态：

- `mask_gt_available_count=0`
- `formal_mask_gt_ready=false`
- `full_live_6_identity_dataset_complete=false`
- `ue_carla_import_readback_complete=false`
- `formal_annotation_ready=false`
- `real_4d_geometry_ready=false`
- `formal_neoverse_ready=false`
- `goal_complete=false`

## 建议命令接口

示例命令：

```bash
python tools/carla_air/export_trajectory_spatial_preview.py \
  --trajectory-config configs/carla_air/trajectories/town10hd_coverage_first_v1.json \
  --node-config local/carla_air/camera_nodes/Town10HD_ground_to_air_nodes_v1.json \
  --trajectory-id traj_cov_01_all_nodes_sweep_return \
  --trajectory-id traj_cov_02_all_nodes_reverse_s_curve \
  --output-dir 组会思路/carla_air_trajectory_spatial_preview_2026_06_12 \
  --map-scale-mode local \
  --view-mode both \
  --z-scale 1.0
```

建议参数：

- `--trajectory-config`：轨迹配置 JSON。
- `--node-config`：相机节点配置 JSON。
- `--trajectory-id`：可重复指定；不指定时默认选两条 coverage 主轨迹。
- `--output-dir`：输出目录。
- `--map-scale-mode local|runtime|both`：默认 `local`。
- `--view-mode 2d|3d|both`：默认 `both`，生成 2D、静态 3D 和交互 3D。
- `--z-scale FLOAT`：默认 `1.0`，保持 Z 轴真实米制比例；大于或小于 `1.0` 只用于视觉质检。
- `--three-js-source PATH`：覆盖默认 vendored Three.js 来源；默认使用 `tools/carla_air/vendor/three.min.js`。
- `--width` / `--height`：静态图尺寸。

## 测试方案

基础检查：

```bash
python -m py_compile tools/carla_air/export_trajectory_spatial_preview.py
```

离线 smoke：

```bash
python tools/carla_air/export_trajectory_spatial_preview.py \
  --trajectory-config configs/carla_air/trajectories/town10hd_coverage_first_v1.json \
  --node-config local/carla_air/camera_nodes/Town10HD_ground_to_air_nodes_v1.json \
  --trajectory-id traj_cov_01_all_nodes_sweep_return \
  --trajectory-id traj_cov_02_all_nodes_reverse_s_curve \
  --output-dir local/carla_air/tmp/trajectory_spatial_preview_smoke \
  --map-scale-mode local
```

验收标准：

- 2D `SVG/PNG/HTML`、3D `SVG/PNG/HTML`、`vendor/three.min.js`、`manifest.json` 均存在且非空。
- `manifest.json` 中 `trajectory_count=2`。
- `manifest.json` 中 `node_count=5`。
- `manifest.json` 中 `view_modes=["2d","3d_static","3d_interactive"]`。
- `manifest.json` 中 `z_scale_default=1.0`。
- `manifest.json` 中 Three.js vendor 记录包含 `relative_path`、`sha256` 和 `size_bytes`。
- `manifest.json` 中 `map_scale_mode=local`。
- 局部 ROI 覆盖所有 selected trajectory waypoints 和 node anchors。
- 2D PNG 和 3D PNG 均可由 Pillow 打开。
- Playwright 可用时，打开 `trajectory_spatial_preview_3d.html` 检查 canvas 非空、桌面/移动视口可渲染、拖拽和 Z scale 控件会改变画面。
- `git diff --check` 通过。

## 当前落盘状态

本规划已落地为：

```text
tools/carla_air/export_trajectory_spatial_preview.py
```

当前实现已升级为默认输出 2D + 静态 3D + 交互 3D：

```text
trajectory_spatial_preview.svg
trajectory_spatial_preview.png
trajectory_spatial_preview.html
trajectory_spatial_preview_3d.svg
trajectory_spatial_preview_3d.png
trajectory_spatial_preview_3d.html
vendor/three.min.js
manifest.json
```

实现仍保持本文边界：不推进飞行器模型导入，不默认启动 CARLA-Air / AirSim runtime，不修改 `docs/goal.md`，不修改 `research/` 当前状态入口。可视化产物只用于展示、人工质检和 waypoint 调参，不作为 annotation、mask、bbox、geometry 或 benchmark evidence。
