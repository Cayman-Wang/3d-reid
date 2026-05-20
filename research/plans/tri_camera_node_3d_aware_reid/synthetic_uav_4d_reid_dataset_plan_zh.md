# 仿真飞行器 4D-ReID 数据集路线

更新时间：2026-05-19

本文档用于固定 synthetic UAV/aircraft 4D-ReID benchmark 的建设路线。它是当前 `NeoVerse fused 4D -> tracklet -> ReID` 主线的上游数据规划，不代表该数据集已经完成，也不代表真实场景泛化问题已经解决。

## 1. 背景与定位

当前 `node01` 的 single-node、cross-scene、track-level ReID 闭环已经具备：

```text
NeoVerse fused 4D points_by_timestamp
-> build_node_tracklets.py
-> extract_node_track_embeddings.py
-> eval_node_track_retrieval.py
```

但当前 `node01` GT-mask bootstrap 难度较低，仍不足以充分拉开：

- `RGB-only`
- `RGB + FPFH handcrafted geometry`
- `RGB + learned point geometry`
- `RGB + learned geometry + motion rerank`
- `end-to-end 4D-aware ReID`

之间的差异。

同时，真实飞行器的多节点、多相机、跨视角 ReID 数据在短期内难以规模采集、稳定标注和反复重跑，因此当前毕业论文阶段建议采用：

```text
现成城市场景
+ 自定义多节点三相机部署
+ 飞行器 identity / trajectory / 标注导出
-> synthetic UAV/aircraft 4D-ReID benchmark
```

该 benchmark 的目标不是替代真实数据，而是作为：

- 当前 4D-aware ReID 方法训练与消融的主数据来源；
- cross-node / harder negative / motion-aware ReID 的可控验证平台；
- 后续少量真实三相机迁移实验的前置训练与接口验证集。

## 2. 总体路线

数据集主线固定为：

```text
现成城市仿真场景
-> 部署 node01 / node02 / ... / nodeN
-> 每个 node 内布设 3 个同步相机
-> 放置飞行器 identity 与多条飞行轨迹
-> 导出 RGB / depth / mask / bbox / pose / trajectory
-> 转换为 NeoVerse 4D-ReID 上游输入
-> 生成 points_by_timestamp / tracklets / identity labels
-> 训练与评测 4D-aware ReID
```

固定原则如下：

- 不从零自建整座城市环境，优先复用现成城市仿真场景。
- 研究重点不放在场景建模，而放在飞行器 identity、多节点布局、轨迹生成、标注导出和 ReID 协议设计。
- synthetic benchmark 必须与当前 `tracklet + points_by_timestamp + retrieval/eval` 契约兼容。

## 3. 场景与平台选择

推荐顺序固定为：

### 3.1 第一优先级：CARLA / 可控城市仿真

用途：

- 快速部署多节点三相机；
- 获得稳定 RGB / depth / segmentation / camera pose；
- 快速构建多场景、跨节点、跨天气 benchmark。

优点：

- 工程闭环稳定；
- 传感器与标注导出成熟；
- 便于快速批量采集。

### 3.2 第二优先级：UE City Sample / 高真实感城市

用途：

- 构建高真实感城市背景；
- 增加遮挡、尺度变化、远距小目标、复杂建筑背景；
- 作为论文展示图和高难度 synthetic split 的主要来源。

固定说明：

- `UE City Sample` 更偏高真实感展示场景；
- 许可证应按 Epic/UE 生态要求处理，不能直接写成完全开源资产。

### 3.3 第三优先级：Isaac Sim / Omniverse synthetic data

用途：

- 自动批量导出合成标注；
- 统一输出 detection / segmentation / pose / synthetic annotations；
- 后续如果需要系统性训练集生成，可作为数据生产平台。

固定说明：

- 若近期目标是尽快构建 benchmark，可先不把 Isaac Sim 作为唯一依赖；
- 若目标转向大量训练数据与程序化标注导出，可逐步把 Isaac Sim 提升为主数据生产平台。

### 3.4 参考线：UrbanScene3D / AirSim

用途：

- 借鉴城市级场景组织、航拍视角和导出协议；
- 作为替代场景来源或辅助数据来源。

固定说明：

- AirSim 只作为参考，不作为近期唯一主线；
- UrbanScene3D 更适合借鉴场景与航拍组织方式，而不是直接等价替代当前 benchmark 协议。

## 4. benchmark 设计

### 4.1 节点布局

固定采用多节点结构：

```text
node01
node02
node03
...
```

每个节点内固定为：

```text
cam0
cam1
cam2
```

固定要求：

- 节点内 3 相机时间同步；
- 节点间允许 viewpoint、distance、background 和 illumination 差异；
- 节点位置应形成跨节点 ReID 难度，而不只是同场景重复观察。

### 4.2 飞行器 identity 设计

目标域固定为：

```text
UAV / aircraft
```

identity 设计应同时覆盖：

- 固定翼；
- 四旋翼或多旋翼无人机；
- 航模或轻小型飞行器；
- 外形接近但 identity 不同的 hard negatives。

固定要求：

- 每个 identity 必须具备稳定 `identity_id`；
- 同一 identity 可在不同节点、不同 scene、不同 trajectory 中重复出现；
- 不能只依赖大面积可读编号或极强纹理差异区分身份，否则会削弱 3D/4D-aware ReID 的研究价值。

建议可控变化：

- 机身尺度微扰；
- 挂载差异；
- 局部纹理差异；
- 颜色/磨损变化；
- 飞行高度、速度、转向模式；
- 遮挡与背景复杂度。

### 4.3 轨迹设计

每个 identity 至少需要：

- 节点内轨迹；
- 跨节点轨迹；
- 直线、盘旋、悬停、转向、穿越遮挡等多种运动模式。

轨迹固定目标：

- 为 `motion rerank` 和 `4D temporal modeling` 提供有效时序差异；
- 不让 track-level ReID 简化为单帧外观分类。

### 4.4 标注与导出字段

每个 scene 必须可导出：

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
```

与当前项目对接时，最终应转换为：

```text
scene_dir
tracks/tracklets.json
mvp-demo/output/neoverse_fused/<scene_id>/points_by_timestamp/index.csv
mvp-demo/output/neoverse_fused/<scene_id>/points_by_timestamp/meta.json
mvp-demo/output/neoverse_fused/<scene_id>/points_by_timestamp/*.npy
```

## 5. 数据规模规划

### 5.1 POC / smoke benchmark

用于验证接口和方法可训练性：

```text
2 个 node
每个 node 3 个相机
5-10 个 identity
每个 identity 至少 2 条跨节点轨迹
```

目标：

- 打通 synthetic scene -> tracklet -> points_by_timestamp -> ReID eval；
- 避免继续只在 `node01` 低难度 bootstrap 上做方法比较。

### 5.2 主实验 benchmark

用于 learned geometry、motion rerank 与 track-level encoder：

```text
100-300 个 identity
1k-9k 条 tracklet
多天气 / 多时间 / 多节点 / 多轨迹模式
```

目标：

- 支撑 `CLIP + learned point geometry`；
- 支撑 `geometry + motion rerank`；
- 明显提升 hard negative 和 cross-node 检索难度。

### 5.3 端到端 4D-aware ReID 数据规模

若后续要稳定训练端到端 ReID：

```text
500+ identity
1万+ tracklet
更丰富的 scene / node / viewpoint / distance / occlusion
```

固定说明：

- 近期允许先做 prototype；
- 不能把小规模 synthetic 训练结果直接写成最终真实场景结论。

## 6. 训练与评测对接

该数据集路线必须服务于当前五级方法线：

```text
1. RGB-only
2. RGB + FPFH handcrafted geometry
3. RGB + learned point geometry
4. RGB + learned geometry + motion rerank
5. end-to-end 4D-aware ReID
```

固定验收：

- 数据能进入现有 `build_node_tracklets.py -> extract_node_track_embeddings.py -> eval_node_track_retrieval.py` 流程；
- 后续训练型 ReID 也必须输出与当前检索评测兼容的 `track embedding`；
- query/gallery 划分必须避免同轨迹泄漏；
- 对比实验必须和 `RGB-only`、`CLIP + FPFH` 保持同口径。

## 7. 与真实数据的关系

当前 synthetic benchmark 的角色固定为：

- 方法开发主数据；
- 训练数据；
- 复杂 hard negative 和 cross-node 差异的受控验证平台。

真实数据近期只承担：

- 少量 sanity check；
- 接口稳定性验证；
- sim-to-real 误差归因。

固定表述：

- 可以写成 `synthetic UAV/aircraft 4D-ReID benchmark`；
- 不能写成“已解决真实飞行器 cross-node ReID 泛化”；
- 不能把 synthetic 满分或提升直接外推到真实域。

## 8. 落盘建议

该路线属于当前 research 主线的近期数据建设方案，建议：

- 在 `research/plans/tri_camera_node_3d_aware_reid/master_plan_zh.md` 中作为毕业论文阶段的数据来源路线引用；
- 在 `research/plans/ACTIVE_PLAN.md` 的 `must_read` 中加入本文档；
- 文档和汇报统一写成 planned benchmark / next-stage dataset plan，而不是已完成成果。
