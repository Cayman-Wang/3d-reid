# 3D ReID 相关工作与可复现 Baseline 组合

更新时间：2026-03-16

本文档面向当前仓库的主线任务整理相关工作，不按传统“行人重识别”口径泛泛综述，而是按你当前已经冻结的研究目标来筛选：

- 任务口径：`single-node, cross-scene, track-level 3D-aware retrieval`
- 当前目标域：`UAV / aircraft`
- 当前输入契约：`RGB + depth + mask + rig.json + timestamps`
- 当前阶段不做：新的端到端 3D encoder 训练、cross-node 正式结果、dynamic 3DGS 主链

下面的“级别”采用计算机视觉领域常见口径：

- `顶会`：CVPR / ICCV / ECCV / NeurIPS / IJCAI 等
- `顶刊`：TIP / TNNLS / Pattern Recognition 等
- `强会`：WACV / ACM MM 等
- `中高水平期刊`：CVIU / KBS 等
- `Workshop`：顶会 workshop
- `预印本`：arXiv 等未正式发表版本

## 1. 相关工作总表

| 工作 | 时间 | Venue | 级别 | 任务类型 | 开源情况 | 代码仓库 |
| --- | --- | --- | --- | --- | --- | --- |
| Robust Depth-based Person Re-identification | 2017 | IEEE TIP | 顶刊 | depth-only person ReID | 无公开 GitHub，提供项目页 | https://www.isee-ai.cn/project/DepthReID.htm |
| Cross-modal distillation for RGB-depth person re-identification | 2022（arXiv 2018） | CVIU | 中高水平期刊 | RGB-D person ReID | 已开源 | https://github.com/frhf/cross-modal-distillation-reidentification |
| Self-Supervised Gait Encoding with Locality-Aware Attention for Person Re-Identification | 2020 | IJCAI | 顶会 | skeleton / gait ReID | 已开源 | https://github.com/Kali-Hac/SGE-LA |
| SM-SGE: A Self-Supervised Multi-Scale Skeleton Graph Encoding Framework for Person Re-Identification | 2021 | ACM MM | 强会 | skeleton / gait ReID | 已开源 | https://github.com/Kali-Hac/SM-SGE |
| Parameter-Efficient Person Re-Identification in the 3D Space | 2022 early access / 2024 print | IEEE TNNLS | 顶刊 | pseudo-3D / shape-aware person ReID | 已开源 | https://github.com/layumi/person-reid-3d |
| LiDAR-based Person Re-identification | 2024 | CVPR | 顶会 | LiDAR / point-cloud person ReID | 已开源 | https://github.com/GWxuan/ReID3D |
| Object Re-Identification from Point Clouds | 2024 | WACV | 强会 | arbitrary object point-cloud ReID | 已开源 | https://github.com/bentherien/point-cloud-reid |
| 3D Semantic MapNet: Building Maps for Multi-Object Re-Identification in 3D | 2024 | arXiv | 预印本 | multi-object 3D map-based ReID | 项目页，未见公开代码 | https://vincentcartillier.github.io/3d_smnet.html |
| Multi-Camera 3D Object Tracking via 3D Point Clouds and Re-Identification | 2025 | ICCV 2025 Workshop (AI City) | Workshop | 3D MOT + ReID | 已开源 | https://github.com/ZIOVISION/AIC2025_Track1_ZV |

## 2. 按当前项目划分的三类工作

### 2.1 可直接复用

这类工作与当前 `三相机节点级 3D-aware track retrieval` 的数据形态最接近，或者可以较低改动接入你的主链。

| 工作 | 为什么可直接复用 | 当前建议用途 |
| --- | --- | --- |
| Cross-modal distillation for RGB-depth person re-identification | 现成双模态输入口径，天然适合 `RGB + predicted depth`；训练与评测框架比从零搭更省时间 | 作为 `RGB + Depth` baseline 的直接参考骨架 |
| Object Re-Identification from Point Clouds | 面向任意物体，不依赖人体骨架或行人先验；更接近 UAV/aircraft 的 object-centric 检索 | 作为 `Geometry-only` baseline 的首选外部参考 |

结论：

- 如果你要最快形成可对比基线，优先复用这两条线。
- 一个负责 `RGB+Depth`，一个负责 `point-cloud geometry`，正好对应你已经冻结的两条几何支路。

### 2.2 可借鉴思想

这类工作不是当前 MVP 的直接模板，但里面有值得迁移的建模思路。

| 工作 | 借鉴点 | 不直接复用的原因 |
| --- | --- | --- |
| LiDAR-based Person Re-identification | 点云采样、几何增强、metric learning、3D person embedding 设计 | 目标域是 LiDAR 行人，不是 arbitrary object；数据先验不同 |
| Parameter-Efficient Person Re-Identification in the 3D Space | 外观与形状解耦、参数高效的 3D-aware 建模思路 | 强依赖 person ReID 设定，不适合直接迁移到 UAV/aircraft |
| Self-Supervised Gait Encoding with Locality-Aware Attention | track-level 时序聚合、自监督表征 | 依赖 3D skeleton / gait，当前目标域不是人体 |
| SM-SGE | 多尺度时序建模、自监督 skeleton graph 编码 | 依赖骨架输入，不适用于当前 object-centric MVP |
| Robust Depth-based Person Re-identification | 深度分支与形状描述子设计 | 早期 depth-only person ReID，代码与工程形态不适合直接落地 |
| 3D Semantic MapNet | 3D map-level association，适合后续 `cross-node` 或更强 object-centric 3D 组织 | 方案偏重，当前 single-node cross-scene MVP 不需要上图级建图 |

### 2.3 不适合当前 MVP

这类工作可以留作远期参考，但不应该进入当前最小闭环的 baseline 范围。

| 工作 | 不适合原因 | 当前阶段建议 |
| --- | --- | --- |
| Multi-Camera 3D Object Tracking via 3D Point Clouds and Re-Identification | 更像比赛型 `3D MOT + ReID` 系统，系统假设比你当前 benchmark 更重 | 仅作未来 cross-node / multi-camera 系统参考 |
| 3D Semantic MapNet | 当前单节点跨 scene 检索不需要 3D semantic map 级建模 | 等你进入 `cross-node` 后再回看 |
| skeleton / gait 系列（SGE-LA、SM-SGE） | 输入依赖人体骨架，不适合 UAV/aircraft | 只借鉴 `track-level aggregation` 思路，不进入当前实验矩阵 |

## 3. 面向当前项目的可复现 Baseline 组合

这里不把 baseline 写成“理想论文方法”，而是按你当前仓库能尽快跑通、能形成正式结果矩阵的方式来排。

### Baseline A：RGB-only Track Retrieval

定义：

- 输入：`RGB crops / masked RGB crops`
- 表征：现有 `CLIP` 或你仓库已经能跑通的视觉 embedding
- 聚合：track pooling
- 检索：cosine / FAISS

适配当前主线的原因：

- 这是你正式结果矩阵里的第一条主基线
- 不依赖 predicted depth 质量，最稳，最适合作为 benchmark 起点

与外部工作的关系：

- 不直接依赖本次拉取的任何一个仓库
- 但后续如果你想把训练和评测写得更规范，可借用 `cross-modal-distillation-reidentification` 的数据组织和评测骨架

当前建议：

- 先把 `RGB-only` 做成最先收口的正式结果
- 后面的几何支路都和它做严格对照

### Baseline B：RGB + Depth 双分支检索

定义：

- 输入：`RGB + predicted depth`
- 组织方式：双分支或两模态融合
- 输出：track-level embedding

首选参考：

- `cross-modal-distillation-reidentification`

为什么它是当前最可落地的双模态 baseline：

- 现成处理 `RGB / depth` 两模态的 person ReID 骨架
- 虽然目标域仍是 person，但模态结构与训练/测试流程最接近你当前需要的 `RGB + predicted depth geometry`

当前改造建议：

- 把身份定义从 `person ID` 改成 `identity_id`
- 把图像单元从单帧改成 `track` 或 `tracklet` 聚合
- 把 BIWI 等 RGB-D 数据集接口替换成你当前的 `frames + masks + depth + frame_times`

落地难度判断：

- 中等
- 比从零新写一套 RGB-D 检索训练代码明显更省时间

### Baseline C：Geometry-only Point-Cloud Retrieval

定义：

- 输入：`fused point cloud` 或每帧 / 每 track 的 object point cloud
- 只使用几何特征做检索

首选参考：

- `point-cloud-reid`

次选参考：

- `ReID3D`

推荐顺序：

1. 先看 `point-cloud-reid`
2. 再看 `ReID3D`

原因：

- `point-cloud-reid` 面向任意物体，更契合 UAV/aircraft
- `ReID3D` 虽然是更新的顶会工作，但 person/LiDAR 先验更强，适合借鉴而不是直接套

当前改造建议：

- 输入优先用你节点内融合后的 point cloud，而不是 LiDAR 原始点
- 先做 `track-level average / pooling`
- 不急着训练新 encoder，先验证 geometry branch 是否真的提供增益

落地难度判断：

- 中等偏高
- 但最符合“真正 3D-aware”这个叙事

### Baseline D：RGB + Geometry Late Fusion

定义：

- `RGB-only` embedding 与 `Geometry-only` embedding 各自独立提取
- 用 late fusion 方式在 track-level 做拼接或加权融合

推荐实现：

- RGB 分支：沿用你当前本地的 CLIP / 视觉 embedding
- Geometry 分支：参考 `point-cloud-reid`
- 若后续需要规范的 metric learning 训练骨架，再借用 `cross-modal-distillation-reidentification` 或 `ReID3D`

为什么这是当前最合理的最终 baseline：

- 与你冻结的正式矩阵完全一致
- 允许分别观察 `RGB-only`、`geometry-only`、`RGB+geometry` 的增益
- 风险比端到端联合训练更低

当前建议：

- 先做最简单的 `L2 normalize + concat`
- 不在当前 MVP 阶段引入复杂 attention fusion 或 cross-modal transformer

## 4. 推荐的落地顺序

按你当前仓库状态，最稳的工程顺序是：

1. `RGB-only`
2. `RGB + predicted-depth geometry`
3. `Geometry-only`
4. `RGB + geometry late fusion`

不建议当前阶段优先投入的方向：

- skeleton / gait 系列
- LiDAR person 专用方法的直接复现
- map-level 3D semantic association
- 比赛型多相机 3D MOT 系统

## 5. 已拉取的开源仓库镜像

说明：

- `完整浅克隆`：已拉到工作树，可直接浏览 README 和主要源码
- `稀疏浅克隆`：已拉到工作树，但刻意跳过大权重或非当前必需目录，以避免大文件导致 checkout 卡死
- `未递归子模块`：仓库本体已拉取，但不会自动补拉其外部子模块

| 仓库 | 当前分类 | 本地路径 | commit | 镜像类型 |
| --- | --- | --- | --- | --- |
| point-cloud-reid | 可直接复用 | `参考工作/repos/point-cloud-reid` | `3fdb42599ce5853b772dad03319c150e62b6693e` | 完整浅克隆 |
| cross-modal-distillation-reidentification | 可直接复用 | `参考工作/repos/cross-modal-distillation-reidentification` | `740e01394b677154e86f8a5cf588f2835f4ed645` | 完整浅克隆 |
| ReID3D | 可借鉴思想 | `参考工作/repos/ReID3D` | `e79b3b12339a882b5aa42f7d259c0ff2af411d23` | 完整浅克隆 |
| person-reid-3d | 可借鉴思想 | `参考工作/repos/person-reid-3d` | `3ec32fc41f8fd8f6a9f03de329aa1109816053c7` | 稀疏浅克隆 |
| SGE-LA | 可借鉴思想 | `参考工作/repos/SGE-LA` | `742e691c0a3e47c2bac133e55225b05cbfd60058` | 稀疏浅克隆 |
| SM-SGE | 可借鉴思想 | `参考工作/repos/SM-SGE` | `e30b99401b385995c691c95369cb448ed9151d25` | 完整浅克隆 |
| AIC2025_Track1_ZV | 不适合当前 MVP | `参考工作/repos/AIC2025_Track1_ZV` | `71c93e25a29d29f1a783bbd64c1a6a145944ea83` | 完整浅克隆（未递归子模块） |

## 6. 我对当前主线的具体建议

如果目标是先把 `ICISCAE` 小论文和当前 `node01` benchmark 稳定收口，最值得优先投入的是两条：

1. `cross-modal-distillation-reidentification`
   - 用来吸收 `RGB + depth` 的双模态训练与评测骨架
2. `point-cloud-reid`
   - 用来吸收 object-centric point-cloud retrieval 的组织方式

`ReID3D` 的价值更偏“研究方法参考”，不是当前最快的复现入口。

换句话说，当前阶段最实用的 baseline 组合不是“去完整复现一篇顶会”，而是：

- 用你已有工程链路产出 `tracklets / depth / fused point clouds`
- 以 `RGB-only` 先收口
- 用 `cross-modal-distillation-reidentification` 补 `RGB + depth`
- 用 `point-cloud-reid` 补 `geometry-only`
- 最后做 `RGB + geometry late fusion`

## 7. 外部链接索引

- Robust Depth-based Person Re-identification
  - https://www.isee-ai.cn/project/DepthReID.htm
- Cross-modal distillation for RGB-depth person re-identification
  - https://www.sciencedirect.com/science/article/abs/pii/S1077314221001806
  - https://github.com/frhf/cross-modal-distillation-reidentification
- Self-Supervised Gait Encoding with Locality-Aware Attention for Person Re-Identification
  - https://www.ijcai.org/proceedings/2020/125
  - https://github.com/Kali-Hac/SGE-LA
- SM-SGE
  - https://arxiv.org/abs/2107.01903
  - https://github.com/Kali-Hac/SM-SGE
- Parameter-Efficient Person Re-Identification in the 3D Space
  - https://pubmed.ncbi.nlm.nih.gov/36315532/
  - https://github.com/layumi/person-reid-3d
- LiDAR-based Person Re-identification
  - https://openaccess.thecvf.com/content/CVPR2024/html/Guo_LiDAR-based_Person_Re-identification_CVPR_2024_paper.html
  - https://github.com/GWxuan/ReID3D
- Object Re-Identification from Point Clouds
  - https://arxiv.org/abs/2305.10210
  - https://github.com/bentherien/point-cloud-reid
- 3D Semantic MapNet
  - https://vincentcartillier.github.io/3d_smnet.html
- Multi-Camera 3D Object Tracking via 3D Point Clouds and Re-Identification
  - https://openaccess.thecvf.com/content/ICCV2025W/AICity/html/Lee_Multi-Camera_3D_Object_Tracking_via_3D_Point_Clouds_and_Re-Identification_ICCVW_2025_paper.html
  - https://github.com/ZIOVISION/AIC2025_Track1_ZV
