# 地对空 Synthetic UAV/Aircraft 4D-ReID 数据集采集指导

更新时间：2026-05-24

本文档用于固定“地面固定多相机节点对空拍摄飞行器”的 synthetic UAV/aircraft 4D-ReID 数据集采集思路、推荐平台和第一阶段落地协议。它是工程选型与采集协议指南，不替代 `research/plans/tri_camera_node_3d_aware_reid/synthetic_uav_4d_reid_dataset_plan_zh.md`。

## 1. 任务定义

当前采集任务不是无人机第一视角导航，也不是飞行器拍摄地面。固定任务定义为：

```text
真实或高保真城市/园区场景
-> 地面固定多相机节点 node01 / node02 / ... / nodeN
-> 每个 node 内 3 个同步相机 cam0 / cam1 / cam2
-> 相机以监控摄像头方式从地面或建筑立面向空域观测
-> 飞行器 identity 沿预设轨迹穿过多个节点视野
-> 导出 RGB / depth / mask / bbox / pose / trajectory
-> 转换为 tracklet + points_by_timestamp + ReID eval 契约
```

该数据集服务于当前主线：

```text
NeoVerse fused 4D points_by_timestamp
-> build_node_tracklets.py
-> extract_node_track_embeddings.py
-> eval_node_track_retrieval.py
```

第一阶段成功标准是数据协议闭环和 `cross-node smoke`，不是端到端 ReID 指标提升。

## 2. 推荐平台与场景优先级

### 2.1 第一优先级：CARLA / CARLA-Air

推荐用途：

- 第一阶段 POC；
- 可控批量采集；
- 固定多相机节点、天气、时间、交通背景和传感器同步；
- 地对空 UAV/aircraft 采集协议验证。

推荐理由：

- CARLA 是开源城市自动驾驶仿真器，传感器、天气、地图和 Python API 较成熟；
- CARLA 支持 RGB、depth、semantic segmentation 等传感器；
- CARLA-Air 将 CARLA 城市场景与 AirSim 飞行能力整合在同一 Unreal Engine 进程，更贴近“城市 + 飞行器 + 多传感器”的需求。

主要风险：

- CARLA 原生任务更偏地面交通，飞行器 identity、飞行动力学和地对空监控视角需要额外工程；
- CARLA-Air 仍需验证与当前采集协议、标注导出和长期维护的匹配度；
- 如果目标是高真实感论文展示图，CARLA 城市场景质感可能不如 UE City Sample。

结论：

```text
CARLA / CARLA-Air 适合作为第一主线。
先用它跑通 POC，再考虑迁移到更高保真 UE 场景。
```

参考链接：

- CARLA GitHub：https://github.com/carla-simulator/carla
- CARLA sensors reference：https://carla.readthedocs.io/en/latest/ref_sensors/
- CARLA-Air GitHub：https://github.com/louiszengCN/CarlaAir
- CARLA-Air project：https://www.carla-air.com/

### 2.2 第二优先级：UE City Sample / Matrix City Sample

推荐用途：

- 高保真城市背景；
- 论文展示图；
- 后续更复杂遮挡、尺度变化、远距小目标和建筑背景。

推荐理由：

- City Sample 是 Epic 发布的 UE5 城市样例工程，城市规模和视觉质量高；
- 适合展示“真实城市监控式地对空观测”场景；
- 可在 UE 中自定义 CameraActor、飞行器 Actor、轨迹和采集插件。

主要风险：

- 不是严格意义上的开源数据集，资产许可需要按 Epic/Fab/Unreal Engine 生态处理；
- 工程体积大、硬件要求高；
- 需要自己补固定相机采集器、实例标注、mask/depth/bbox 导出和轨迹记录。

结论：

```text
UE City Sample 适合作为第二阶段高保真补充。
不建议在第一天就把它作为唯一数据生产主线。
```

参考链接：

- Epic City Sample 文档：https://dev.epicgames.com/documentation/en-us/unreal-engine/city-sample-project-unreal-engine-demonstration
- Fab City Sample：https://www.fab.com/listings/4898e707-7855-404b-af0e-a505ee690e68

### 2.3 第三优先级：EmbodiedCity

推荐用途：

- 借鉴城市级 UE/AirSim 场景组织方式；
- 借鉴 embodied city benchmark 的场景构建、数据组织和 simulator 发布方式；
- 作为后续候选城市环境或对照平台。

主要风险：

- EmbodiedCity 的默认任务是城市 embodied intelligence / agent benchmark，不是固定监控相机网络；
- 默认采集视角偏 agent/drone egocentric，不等价于本项目的地面固定相机地对空采集；
- 需要确认是否能拿到可编辑 Unreal 工程，而不是只能运行 packaged simulator；
- 若不能编辑场景、放置固定相机、加载飞行器 identity 和导出自定义标注，就不适合作为第一主线。

结论：

```text
EmbodiedCity 适合作参考，不作为当前唯一主线。
只有在确认可编辑性和自定义标注导出后，才进入正式采集候选。
```

参考链接：

- EmbodiedCity overview：https://embodiedcity.github.io/overview/
- EmbodiedCity GitHub：https://github.com/tsinghua-fib-lab/EmbodiedCity
- EmbodiedCity Hugging Face：https://huggingface.co/EmbodiedCity

### 2.4 参考/备选：UrbanScene3D

推荐用途：

- 借鉴 UE4 + AirSim 的大规模城市场景仿真组织；
- 借鉴真实重建城市、深度图、2D/3D bbox、点云和实例标注的数据生产方式；
- 作为后续城市环境补充或论文 related benchmark 参考。

主要风险：

- 资产和 mesh 导出存在版权或许可限制时，不适合直接作为可再发布数据生产工程；
- 默认任务与本项目的固定多相机地对空 ReID 仍有差异；
- 需要额外验证能否部署地面固定节点、加载飞行器 identity、导出同步多相机标注。

结论：

```text
UrbanScene3D 适合作参考和候选验证。
不建议作为第一版 POC 的唯一依赖。
```

参考链接：

- UrbanScene3D GitHub：https://github.com/yilinliu77/UrbanScene3D
- UrbanScene3D paper：https://arxiv.org/abs/2107.04286

### 2.5 参考/备选：Cesium for Unreal + OSM / 3D Tiles

推荐用途：

- 快速加载真实地理城市结构；
- 构建真实城市布局和地理尺度背景；
- 后续做真实场景外观或 sim-to-real 视觉差异分析。

主要风险：

- Cesium for Unreal 插件本身开源，但在线 3D Tiles / OSM Buildings / photogrammetry 数据需要按 Cesium ion 或对应数据源许可使用；
- 流式 geospatial tiles 不天然提供 ReID 所需的稳定 instance id、语义标签、mask、bbox 和可控遮挡标注；
- 深度、分割、实例标注和离线复现实验需要大量额外工程。

结论：

```text
Cesium 适合作真实城市背景增强。
不适合作第一阶段标注闭环主线。
```

参考链接：

- Cesium for Unreal：https://cesium.com/learn/unreal/
- Cesium Unreal FAQ：https://cesium.com/learn/unreal/unreal-faq/
- Cesium for Unreal GitHub：https://github.com/CesiumGS/cesium-unreal

### 2.6 参考/备选：AirSim 示例环境

推荐用途：

- 快速验证无人机控制、相机 API、depth/segmentation 导出；
- 作为 CARLA-Air 或自定义 UE 工程的接口参考。

主要风险：

- Microsoft AirSim 原仓库已进入归档/停止维护状态；
- 示例环境规模和真实感不足以支撑正式 UAV/aircraft 4D-ReID 数据集；
- 默认任务形态仍偏无人机第一视角或移动平台。

结论：

```text
AirSim 可以作为 API 和飞行控制参考。
不建议作为正式数据集主场景来源。
```

参考链接：

- AirSim GitHub：https://github.com/microsoft/AirSim
- AirSim docs：https://microsoft.github.io/AirSim/

## 3. 第一阶段 POC 采集协议

第一阶段目标是验证完整协议，不追求大规模和端到端训练。

固定规模：

```text
nodes: 2
cameras_per_node: 3
identities: 5-10
trajectories_per_identity: >= 2 cross-node trajectories
```

固定节点结构：

```text
node01/
  cam0
  cam1
  cam2
node02/
  cam0
  cam1
  cam2
```

固定采集视角：

- 相机布设在地面、楼顶边缘、建筑立面或杆塔位置；
- 相机朝向空域或飞行走廊；
- 相机固定不随飞行器运动；
- 每个 node 内三相机需要时间同步；
- node 间允许 viewpoint、distance、background、illumination 差异。

固定导出字段：

```text
scene_id
node_id
camera_id
timestamp
track_id
identity_id
rgb
bbox
mask
depth
camera intrinsics
camera extrinsics
object pose
3D centroid
3D bbox
velocity / motion stats
trajectory_id
```

当前项目对接目标：

```text
scene_dir
tracks/tracklets.json
points_by_timestamp/index.csv
points_by_timestamp/meta.json
points_by_timestamp/*.npy
```

## 4. 实施路线

### 4.1 平台选型验证

先用 `CARLA / CARLA-Air` 做最小验证：

1. 打开一个可稳定加载的城市地图。
2. 放置 `node01` 和 `node02`。
3. 每个 node 放置 3 个固定相机。
4. 验证每个相机能同步导出 RGB。
5. 验证同一 timestamp 下能导出 depth / segmentation 或 mask。
6. 验证可记录相机内外参。

通过后再接入飞行器 identity 和轨迹控制。

### 4.2 飞行器 identity 设计

第一阶段不要求 100+ identity，但要避免过于简单：

- 至少包含固定翼、四旋翼/多旋翼、轻小型飞行器；
- 每个 identity 有稳定 `identity_id`；
- 不只依赖大面积可读编号或强颜色差异；
- 建议加入轻微尺度、挂载、局部纹理、颜色和姿态差异；
- hard negative 应包含外形接近但身份不同的飞行器。

### 4.3 轨迹设计

每个 identity 至少采集两类跨节点轨迹：

- 直线穿越：从 node01 观测区域进入 node02 观测区域；
- 盘旋/转向：在节点附近产生姿态和视角变化；
- 可选遮挡：经过建筑边缘、杆塔或复杂背景；
- 可选高度变化：同一 identity 在不同高度和距离下被观测。

轨迹必须记录 `trajectory_id`。后续训练/评测划分不能让同一条轨迹同时进入 query 和 gallery 的可泄漏对照中。

### 4.4 固定相机配置

每个相机至少记录：

```text
camera_id
node_id
image_width
image_height
fov
focal length or intrinsics K
camera pose in world
mount height
look-at target or yaw/pitch/roll
```

地对空采集的关键不是相机越广越好，而是目标在图像中可用于 ReID：

- FOV 过大时目标像素太小；
- FOV 过小时容易出视野；
- 建议先用少量轨迹做 bbox 尺寸统计，再确定焦距和分辨率；
- 对每个 scene 输出目标 bbox 面积、可见帧数和遮挡率统计。

### 4.5 标注和同步导出

采集器需要保证：

- `cam0/cam1/cam2` 同一 timestamp 对齐；
- RGB、depth、mask、bbox 使用同一 timestamp；
- bbox 从 mask 或 actor projection 生成，避免手写偏移；
- object pose、camera pose、trajectory state 与帧时间一致；
- 目标完全不可见时必须显式记录，不要伪造 mask；
- 多目标扩展前，第一阶段优先保证单目标 tracklet 干净。

### 4.6 转换到当前 ReID pipeline

转换脚本后续应生成与当前项目一致的结构：

```text
data/nodes/<node_id>/scenes/<scene_id>/
  cams/cam0/frames/<ts>.png
  cams/cam0/depth/<ts>.png or .npy
  cams/cam0/masks/<ts>.png
  cams/cam1/...
  cams/cam2/...
  calib/rig.json
  frame_times.csv
  capture_meta.json
  tracks/tracklets.json
```

NeoVerse 4D 分支最终需要：

```text
output/neoverse_fused_runs/<benchmark_id>/<scene_id>/points_by_timestamp/
  index.csv
  meta.json
  *.npy
```

`meta.json.schema_version` 应保持：

```text
neoverse_points_by_timestamp_v1
```

## 5. 评测协议

第一阶段至少保留三条同口径结果：

```text
1. RGB-only
2. RGB + FPFH handcrafted geometry
3. RGB + NeoVerse fused 4D geometry
```

若启动 learned prototype，再加入：

```text
4. RGB + learned point geometry
5. RGB + learned geometry + motion rerank
```

第一阶段不把完整端到端 4D-aware ReID 作为成功条件。端到端训练需要更大规模 identity 和 tracklet，否则只能作为 overfit-prone prototype 报告。

推荐 split：

- query 和 gallery 至少跨 node；
- query 和 gallery 不应来自同一条 trajectory；
- train / val / test 按 identity 和 trajectory 双重隔离；
- 所有方法必须共享相同 query/gallery 划分；
- 需要保留 RGB-only baseline，防止几何分支在低难度数据上被满分掩盖。

指标：

```text
mAP
Recall@1
Recall@5
Recall@10
top-K qualitative cases
failure cases
```

## 6. 质量控制与风险

### 6.1 小目标与成像质量

地对空监控式采集的核心风险是飞行器太小、太远或曝光不稳定。每个 scene 应统计：

- 每帧 bbox 宽高；
- bbox 面积占图像比例；
- 可见帧数；
- mask 非零像素数；
- 遮挡比例；
- 每个 camera 的有效 timestamp 数；
- 每个 identity 的跨 node 可见性。

若大量帧中目标小到无法分辨外形，应先调整相机焦距、分辨率、节点距离或轨迹高度，而不是直接训练 ReID。

### 6.2 标注一致性

必须避免：

- RGB 与 mask/depth 时间错位；
- `identity_id` 来自文件名猜测；
- 同一 actor 在不同 node 被写成不同 identity；
- 同一 trajectory 泄漏到 train/test 两侧；
- segmentation 类别 mask 替代 instance mask 但没有区分目标实例。

### 6.3 平台与许可

必须单独记录每个场景和模型的来源：

```text
asset_name
source_url
license_or_terms
allowed_use
redistribution_allowed
notes
```

对于 City Sample、EmbodiedCity、UrbanScene3D、Cesium ion 数据和第三方飞行器模型，不能默认写成完全开源可再发布资产。论文和数据发布前需要逐项核查许可。

## 7. 阶段性验收

### 7.1 POC 验收

POC 通过条件：

- 至少 `2 nodes x 3 cameras`；
- 至少 `5` 个 identity；
- 每个 identity 至少 `2` 条跨节点轨迹；
- 每条轨迹能导出 RGB / depth / mask / bbox / pose / trajectory；
- 能转换为当前 `scene_dir + tracklets + points_by_timestamp` 契约；
- 能跑通 RGB-only 与 RGB+geometry 的同口径 retrieval eval；
- 生成一份失败案例和目标尺寸统计。

### 7.2 主实验数据集验收

进入主实验前，目标规模应扩到：

```text
100-300 identities
1k-9k tracklets
multi-node / multi-viewpoint / multi-distance / multi-trajectory
```

此阶段才适合系统比较：

- `RGB-only`
- `RGB + FPFH`
- `RGB + learned point geometry`
- `RGB + motion rerank`

### 7.3 端到端训练验收

完整端到端 4D-aware ReID 需要更大规模：

```text
500+ identities
10000+ tracklets
更丰富的 scene / node / viewpoint / distance / occlusion
```

在达到该规模前，端到端网络只能写成 prototype，不能写成稳定方法结论。

## 8. 当前建议

当前最稳的执行顺序：

```text
1. CARLA / CARLA-Air 固定相机 POC
2. 地对空飞行器轨迹和 identity 导出
3. 2-node cross-node smoke
4. 转换到 3d-reid 数据契约
5. RGB-only / RGB+FPFH / RGB+NeoVerse 4D 同口径评测
6. 再扩到 learned geometry 和 motion rerank
7. 最后才做端到端 4D-aware ReID
```

不建议当前直接从 EmbodiedCity 或 City Sample 开始做大规模数据集，除非已经确认：

- UE 工程可编辑；
- 固定相机节点可自由放置；
- 飞行器模型和轨迹可控；
- 实例 mask、depth、bbox、pose、trajectory 可同步导出；
- 资产许可允许当前论文和数据使用方式。

## 9. 来源入口

工具和场景入口：

- CARLA：https://github.com/carla-simulator/carla
- CARLA sensors：https://carla.readthedocs.io/en/latest/ref_sensors/
- CARLA-Air：https://github.com/louiszengCN/CarlaAir
- CARLA-Air project：https://www.carla-air.com/
- Epic City Sample：https://dev.epicgames.com/documentation/en-us/unreal-engine/city-sample-project-unreal-engine-demonstration
- Fab City Sample：https://www.fab.com/listings/4898e707-7855-404b-af0e-a505ee690e68
- EmbodiedCity：https://embodiedcity.github.io/overview/
- EmbodiedCity GitHub：https://github.com/tsinghua-fib-lab/EmbodiedCity
- EmbodiedCity Hugging Face：https://huggingface.co/EmbodiedCity
- UrbanScene3D：https://github.com/yilinliu77/UrbanScene3D
- UrbanScene3D paper：https://arxiv.org/abs/2107.04286
- Cesium for Unreal：https://cesium.com/learn/unreal/
- Cesium Unreal FAQ：https://cesium.com/learn/unreal/unreal-faq/
- Cesium for Unreal GitHub：https://github.com/CesiumGS/cesium-unreal
- AirSim：https://github.com/microsoft/AirSim
- AirSim docs：https://microsoft.github.io/AirSim/

