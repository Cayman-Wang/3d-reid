# 3D/4D-ReID 端到端补强思路与参考项目映射

更新时间：2026-05-18

本文档整理本轮对话形成的 `3D-aware / 4D-aware ReID` 补强路线。它的定位不是宣称已经复现或集成外部系统，而是把外部 3D ReID、RGB-D object ReID、gallery-based tracking 和运动学 ReID 工作中的可迁移模块，压缩并适配到当前项目的 `NeoVerse 4D reconstruction -> track-level ReID` 管线中。

当前准确表述应为：

> 本项目不直接复现某一个外部 ReID 系统，而是将已有 3D ReID 工作中的可学习点云表征、RGB-D object memory、track-level gallery 组织和飞行器运动几何匹配思想，逐步适配到当前 single-node、track-level、RGB-depth-pointcloud 的 3D/4D-aware ReID 管线中。

## 1. 当前 ReID 实现口径

当前项目中的 ReID 属于 `track-level 检索式 ReID`，不是端到端训练型 ReID 网络。

### 1.1 当前输入

```text
RGB / texture input:
scene/cams/cam*/frames/*.png
scene/cams/cam*/masks_gt or masks/*.png
mask-derived bbox

Geometry input:
recon/points_fused/*.npy
or NeoVerse points_by_timestamp/*.npy

Track input:
tracks/tracklets.json
```

### 1.2 当前输出

```text
embeddings/tracks.npy
embeddings/tracks_meta.json
retrieval / eval json
```

### 1.3 当前流程

```text
RGB crop -> CLIP 512D
point cloud -> FPFH 33D
CLIP + FPFH -> weighted concat
timestamp mean pooling -> track embedding
cosine retrieval -> mAP / Recall@K
```

可写成：

```text
E_track = mean_t( normalize([w_rgb * CLIP(crop_t), w_geo * FPFH(P_t)]) )
score(q, g) = cosine(E_track_q, E_track_g)
```

当前方案优点是简单、可复现、已经形成闭环；短板是：

- `FPFH` 是手工几何描述子，细粒度身份区分能力有限。
- NeoVerse 4D 点云目前主要通过多帧平均进入 ReID，时序信息没有被真正建模。
- RGB 分支是 frame-level CLIP 后简单均值池化，没有 track-level visual encoder。
- Gallery 只是 `tracks.npy + tracks_meta.json`，还不是 object memory / multi-modal gallery。
- 当前没有端到端 metric learning，没有针对本项目目标身份学习判别特征。

## 2. 总体补强路线

建议按四级递进，不要一步跳到重型端到端系统。

```text
Baseline 1:
RGB-only
CLIP crop -> track pooling -> cosine retrieval

Baseline 2:
RGB + handcrafted geometry
CLIP + FPFH -> weighted fusion -> cosine retrieval

Proposed 1:
RGB + learned point-cloud geometry
CLIP + point-cloud encoder -> weighted fusion -> cosine retrieval

Proposed 2:
RGB + learned geometry + 4D motion rerank
embedding retrieval -> 2D/3D trajectory geometry rerank

Proposed 3:
End-to-end 4D-aware ReID network
RGB sequence + point sequence + motion sequence -> fusion network -> metric learning
```

这里 `Proposed 1/2` 仍然可以保持检索式结构，风险较低；`Proposed 3` 才是端到端训练型 ReID 网络，性能上限更高，但需要更多身份、更多跨视角/跨场景样本和更稳定的标注。

## 3. 四个模块级补强点

### 3.1 把 FPFH 换成 learned point encoder

这一点解决的是：`单个 timestamp 的点云 P_t 怎么提取更强几何特征`。

当前：

```text
P_t -> Open3D FPFH -> 33D geometry feature
```

建议：

```text
P_t -> sample / normalize -> PointNet or DGCNN or PointTransformer -> E_geo_t
```

第一版可以新增几何后端：

```text
geo_backend=open3d_fpfh       # 当前 baseline
geo_backend=learned_pointnet  # 第一版 learned geometry
geo_backend=learned_dgcnn     # 第二版
geo_backend=learned_calm_6c   # 后续增强
```

#### 借鉴工作

| 项目 | 文献 | 借鉴点 | 本项目落点 |
| --- | --- | --- | --- |
| `point-cloud-reid` | Object Re-Identification from Point Clouds, WACV 2024 | object point-cloud ReID 范式；点云采样、归一化、PointNet/PointTransformer backbone、Siamese/metric matching | 替换当前 `FPFH` 几何分支 |
| `CALM-Net` | CALM-Net: Curvature-Aware LiDAR Point Cloud-based Multi-Branch Neural Network for Vehicle Re-Identification, arXiv 2025 | DGCNN、PointTransformer、cross/local attention、局部几何/eigenvalue/曲率感知增强 | 第二阶段几何 encoder 增强 |
| `AIC2025_Track1_ZV` | Multi-Camera 3D Object Tracking via 3D Point Clouds and Re-Identification, ICCV Workshop 2025 | voxel-based 3D feature extractor、normalized embedding | 后续可作为 voxel/sparse-conv 路线参考，不作为第一版 point encoder |

汇报口径：

> 当前几何分支采用 Open3D FPFH，是可复现的手工几何 baseline。后续借鉴 Object Re-ID from Point Clouds 和 CALM-Net，将每帧点云编码从手工统计特征升级为可学习点云几何表征。

### 3.2 把 NeoVerse 4D 信息真正用起来

这一点解决的是：`一整条 4D 轨迹怎么用`。

当前虽然输入是：

```text
P_1, P_2, ..., P_T
```

但 ReID 中基本是：

```text
每帧点云 -> FPFH
多帧特征 -> mean pooling
```

这样会把 4D 信息压成平均几何，丢掉时间顺序、速度、方向、尺度变化和姿态变化。

建议：

```text
P_t -> learned point encoder -> E_geo_t
{E_geo_t}_{t=1..T} -> temporal pooling / temporal attention -> E_4d_geo_track

同时：
P_t -> centroid_t, bbox3d_t, extent_t, point_count_t
centroid sequence -> velocity_t, direction_t, trajectory length, motion smoothness
```

最终输出可以拆成：

```text
E_geo_track
E_motion_track
motion_stats.json
```

#### 借鉴工作

| 项目 | 文献 | 借鉴点 | 本项目落点 |
| --- | --- | --- | --- |
| `AIC2025_Track1_ZV` | Multi-Camera 3D Object Tracking via 3D Point Clouds and Re-Identification | trajectory aggregation、short-term tracking、long-term trajectory linking、spatial distance + ReID similarity | 轨迹级聚合与跨时间关联 |
| 2D/3D Kinematics aerial target ReID | Multiple Aerial Targets Re-Identification by 2D- and 3D-Kinematics-Based Matching, Journal of Imaging 2022 | 速度向量、运动方向、2D/3D 运动学匹配 | `motion_geometry_rerank` |
| Multi-drone trajectory ReID | Multi-camera Multi-drone Detection, Tracking and Localization with Trajectory-based Re-identification, ICA-SYMP 2021 | 多相机多无人机轨迹关联、trajectory-based ReID | 后续 multi-camera / cross-node 参考 |

第一点和第二点的区别：

| 补强点 | 处理对象 | 解决问题 | 输出 |
| --- | --- | --- | --- |
| learned point encoder | 单帧点云 `P_t` | 每一帧几何特征不够强 | `E_geo_t` |
| 4D 信息建模 | 点云序列 `{P_t}` 和轨迹 | 多帧时序/运动没有被利用 | `E_4d_track` 或 rerank score |

汇报口径：

> Learned point encoder 解决每帧点云如何提特征；4D 建模解决整条轨迹如何利用时间、运动和姿态变化。当前系统已经接入 NeoVerse 4D 点云，但还没有充分使用 4D 时序信息。

### 3.3 纹理分支改进成 track-level visual encoder

当前 RGB 分支是：

```text
per timestamp:
multi-camera crop -> CLIP -> average

per track:
multi-timestamp feature -> average
```

这个方式稳定，但不能区分清晰视角、遮挡视角和弱纹理视角，也没有学习跨视角一致性。

建议：

```text
multi-view RGB crops over time
-> CLIP / vehicle-pretrained visual encoder
-> view-quality weighting
-> temporal attention pooling
-> E_rgb_track
```

可加入质量权重：

```text
mask area
bbox size
crop sharpness
point coverage
view coverage
occlusion ratio
```

#### 借鉴工作

| 项目 | 文献 | 借鉴点 | 本项目落点 |
| --- | --- | --- | --- |
| `DATOR / instance-based-loc` | Towards Global Localization using Multi-Modal Object-Instance Re-Identification, AIR 2025 / arXiv 2024 | RGB-D object-instance descriptor、dual-path RGB/depth 表征、多模态 object embedding | track-level visual/depth descriptor 组织 |
| `VehicleMAE` | Structural Information Guided Multimodal Pre-training for Vehicle-centric Perception, AAAI 2024 | 车辆中心视觉预训练、结构信息引导外观表征 | RGB 分支或 RGB-only baseline 增强 |
| `CVNet` | CVNet: Lightweight Cross-View Vehicle ReID with Multi-Scale Localization, Sensors 2025 | 跨视角车辆外观 ReID、多尺度 localization | 跨视角纹理 ReID baseline |
| `3D-LENS` | 3D-LENS: 3D Lifting-based Elevated Novel-view Synthesis for Single-View Aerial-Ground Re-Identification, arXiv 2026 | 3D lifting、novel-view synthesis | 远期视角增强，不进当前 MVP |
| `GSAlign` | GSAlign: Geometric and Semantic Alignment Network for Aerial-Ground Person Re-Identification, NeurIPS 2025 | 几何/语义对齐、跨视角可见区域对齐 | 远期跨域对齐，不进当前 MVP |

汇报口径：

> 当前 CLIP 是 frame-level appearance baseline。后续将 RGB 分支升级为 track-level visual encoder，让模型利用多视角、多时间的稳定纹理信息，而不是简单均值。

### 3.4 借鉴 DATOR / AIC2025 做 gallery 和 rerank

当前 gallery 是：

```text
tracks.npy
tracks_meta.json
cosine retrieval
```

建议升级为 object gallery：

```text
track_id
identity_id
E_rgb
E_geo
E_motion
E_fused
centroid_seq
bbox3d_seq
point_quality
view_coverage
timestamp_range
source_scene
```

检索流程：

```text
Step 1: cosine(E_fused_q, E_fused_g) -> top-K
Step 2: geometry / motion rerank on top-K
Step 3: final ranking
```

非训练版 rerank 可以先写成：

```text
score_final =
  alpha * cosine(E_fused_q, E_fused_g)
+ beta  * cosine(E_geo_q, E_geo_g)
+ gamma * motion_similarity(q, g)
+ delta * shape_similarity(q, g)
```

可用的 motion / shape 项：

```text
velocity direction cosine
trajectory length ratio
centroid displacement consistency
scale change consistency
3D bbox extent similarity
point cloud eigenvalue similarity
normalized shape extent similarity
```

#### 借鉴工作

| 项目 | 文献 | 借鉴点 | 本项目落点 |
| --- | --- | --- | --- |
| `instance-based-loc / DATOR` | Towards Global Localization using Multi-Modal Object-Instance Re-Identification | object memory、embedding-to-object 管理、RGB-D object descriptor、top-K object matching | 将 `tracks_meta.json` 升级为 object gallery |
| `AIC2025_Track1_ZV` | Multi-Camera 3D Object Tracking via 3D Point Clouds and Re-Identification | gallery embeddings、Embedding Similarity Classifier、trajectory association | gallery refinement 和 long-term association |
| 2D/3D Kinematics aerial target ReID | Multiple Aerial Targets Re-Identification by 2D- and 3D-Kinematics-Based Matching | 运动几何匹配 | top-K 后的 motion rerank |

汇报口径：

> 当前 ReID gallery 只是向量文件和 metadata。后续借鉴 DATOR 的 object memory 和 AIC2025 的 gallery-based refinement，把每条 track 作为对象实例，维护 RGB、几何、运动和质量信息，并在初检 top-K 后做几何/运动重排序。

## 4. 端到端训练型 ReID 网络方案

当场景复杂度提高、目标种类和身份数量增加后，端到端训练型 ReID 的性能上限通常高于当前检索式方案。但前提是有足够训练数据、可靠 identity label、跨视角/跨场景划分和稳定的 3D/4D 输入。

### 4.1 检索式 ReID 与端到端 ReID 的差别

| 对比项 | 当前 track-level 检索式 ReID | 端到端训练型 ReID |
| --- | --- | --- |
| 是否训练新模型 | 通常不训练或只调权重 | 训练 RGB/point/temporal/fusion 网络 |
| 输入单位 | track 的多帧、多视角数据 | 图像序列、点云序列、track pair/triplet/batch |
| 输出 | track embedding + cosine score | learned embedding 或 same/different score |
| 数据需求 | 低 | 高 |
| 工程难度 | 低 | 高 |
| 性能上限 | 中等 | 更高 |
| 4D 信息利用 | 多帧平均或规则 rerank | 可学习时序、运动、姿态变化 |
| 适合阶段 | 当前 MVP / baseline | 后续 proposed method |

### 4.2 端到端 4D-aware ReID 结构

建议结构：

```text
RGB crop sequence -----> visual encoder --------\
                                                  -> fusion encoder -> E_track
point cloud sequence --> point encoder ----------/
motion sequence -------> temporal/motion encoder-/

E_track -> ID classification loss
E_track -> triplet / contrastive loss
track pair -> same/different matching loss
```

可以拆成三个 encoder：

```text
E_rgb_t    = VisualEncoder(crop_t)
E_geo_t    = PointEncoder(P_t)
E_motion   = MotionEncoder({centroid_t, velocity_t, bbox3d_t})
E_track    = Fusion({E_rgb_t}, {E_geo_t}, E_motion)
```

训练目标：

```text
identity classification loss
triplet loss
contrastive loss
same/different matching loss
cross-modal consistency loss
```

### 4.3 推荐阶段划分

```text
Stage 0: 当前已完成
CLIP + FPFH -> track-level retrieval

Stage 1: 几何分支学习化
CLIP + learned point encoder -> track-level retrieval

Stage 2: 4D 信息显式利用
CLIP + learned geometry + temporal pooling + motion rerank

Stage 3: gallery/object memory 化
multi-modal object gallery + top-K rerank

Stage 4: 端到端 4D-aware ReID
RGB sequence + point sequence + motion sequence -> fusion network -> metric learning
```

### 4.4 真实场景迁移预期

当场景更复杂、目标数量更多、hard negative 更多时：

```text
小数据 / 少身份 / 简单场景:
检索式 ReID 更稳

多身份 / 多视角 / 多场景 / 真实标注充足:
端到端训练型 ReID 更可能优于检索式 ReID
```

原因是端到端方法可以学习：

- 哪些纹理差异对飞行器身份有用。
- 哪些 3D 局部结构能区分相似型号。
- 哪些 4D 运动模式应作为身份约束。
- 什么时候更信 RGB，什么时候更信 geometry。
- 如何把 hard negative 拉开。

但如果训练数据不足，端到端模型容易过拟合，可能不如 `CLIP + FPFH` 这种检索式 baseline 稳定。

## 5. 参考项目与文献索引

| 项目/工作 | 文献 | 类型 | 主要借鉴点 |
| --- | --- | --- | --- |
| `point-cloud-reid` | Object Re-Identification from Point Clouds, WACV 2024 | 主实现参考 | learned point-cloud encoder、Siamese/metric matching |
| `CALM-Net` | CALM-Net: Curvature-Aware LiDAR Point Cloud-based Multi-Branch Neural Network for Vehicle Re-Identification, arXiv 2025 | 模块增强参考 | curvature/local geometry、EdgeConv、attention、多分支几何编码 |
| `instance-based-loc / DATOR` | Towards Global Localization using Multi-Modal Object-Instance Re-Identification, AIR 2025 / arXiv 2024 | gallery/object memory 参考 | RGB-D object descriptor、object memory、multi-modal embedding |
| `AIC2025_Track1_ZV` | Multi-Camera 3D Object Tracking via 3D Point Clouds and Re-Identification, ICCV Workshop 2025 | 系统组织参考 | 3D embedding gallery、ESC、tracking + ReID 二阶段关联 |
| `VehicleMAE` | Structural Information Guided Multimodal Pre-training for Vehicle-centric Perception, AAAI 2024 | RGB 分支参考 | 结构化视觉预训练、vehicle-centric representation |
| `CVNet` | CVNet: Lightweight Cross-View Vehicle ReID with Multi-Scale Localization, Sensors 2025 | RGB baseline 参考 | 跨视角车辆 ReID、多尺度视觉定位 |
| `3D-LENS` | 3D-LENS: 3D Lifting-based Elevated Novel-view Synthesis for Single-View Aerial-Ground Re-Identification, arXiv 2026 | 远期视角增强 | 3D lifting、novel-view synthesis |
| `GSAlign` | GSAlign: Geometric and Semantic Alignment Network for Aerial-Ground Person Re-Identification, NeurIPS 2025 | 远期跨视角对齐 | geometric / semantic alignment |
| 2D/3D Kinematics aerial target ReID | Multiple Aerial Targets Re-Identification by 2D- and 3D-Kinematics-Based Matching, Journal of Imaging 2022 | 运动重排序参考 | 速度方向、瞬时速度向量、2D/3D 运动学匹配 |
| Multi-drone trajectory ReID | Multi-camera Multi-drone Detection, Tracking and Localization with Trajectory-based Re-identification, ICA-SYMP 2021 | 轨迹关联参考 | 多无人机轨迹 ReID、trajectory association |

## 6. 组会汇报推荐口径

可以按 5 页讲：

1. **当前实现**
   - `CLIP + FPFH`
   - track-level pooling
   - cosine retrieval
   - 已完成 `NeoVerse points_by_timestamp -> ReID` 接入

2. **当前问题**
   - FPFH 是手工几何特征
   - 4D 信息被均值池化压平
   - RGB 分支不是 track-level visual encoder
   - Gallery 不是 object memory

3. **参考工作映射**
   - point-cloud-reid / CALM-Net -> learned geometry
   - DATOR -> RGB-D descriptor + object memory
   - AIC2025 -> gallery refinement + tracking/ReID
   - 2D/3D kinematics -> motion rerank

4. **拟实现方法**

```text
RGB crop sequence -> track-level visual encoder ----\
4D point sequence -> learned point encoder ----------> fusion embedding -> top-K retrieval -> geometry/motion rerank
4D trajectory ----> motion geometry stats ----------/
```

5. **实验设计**

```text
RGB-only
RGB + FPFH
RGB + learned point geometry
RGB + learned point geometry + motion rerank
End-to-end 4D-aware ReID
```

一句话总结：

> 当前系统已经证明 NeoVerse 4D 点云可以被 ReID 管线消费，但几何侧仍是 FPFH，4D 时序也主要被均值池化压平。后续将借鉴 point-cloud-reid 和 CALM-Net，把几何分支升级为 learned point encoder；借鉴 DATOR 和 AIC2025，把 track embedding 扩展为多模态 object gallery；再引入飞行器 2D/3D 运动几何 rerank，最终过渡到端到端训练型 4D-aware ReID 网络。
