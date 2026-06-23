# NeoVerse 4D-ReID 表征补强规划

更新时间：2026-05-18

本文档把 `参考工作/3d_4d_reid_end_to_end_strengthening_plan_zh.md` 中的参考思路转成当前 research 主线的可执行规划。它不是实验结果报告，也不宣称端到端 4D-aware ReID 已完成；它用于固定近期 prototype 的方向、接口和验收口径。

## 1. 当前基线与主线契约

当前 4D 几何主线正式固定为：

```text
NeoVerse fused 4D points_by_timestamp
-> build_node_tracklets.py
-> extract_node_track_embeddings.py
-> eval_node_track_retrieval.py
```

权威输入契约：

```text
mvp-demo/output/neoverse_fused/<scene_id>/points_by_timestamp/index.csv
mvp-demo/output/neoverse_fused/<scene_id>/points_by_timestamp/meta.json
mvp-demo/output/neoverse_fused/<scene_id>/points_by_timestamp/*.npy
```

当前 ReID baseline：

```text
RGB crop -> CLIP 512D
point cloud -> Open3D FPFH 33D
CLIP + FPFH -> weighted concat
timestamp mean pooling -> track embedding
cosine retrieval -> mAP / Recall@K
```

保守结论固定为：

- NeoVerse 4D geometry 已接入并可评测。
- 当前 `node01` GT-mask bootstrap 下，RGB-only 与 RGB+NeoVerse 4D geometry 持平。
- 不能写成 geometry、三相机或 own-depth 已带来 Rank/mAP 提升。

## 2. 近期方法路线

近期补强路线按五级推进：

```text
Baseline 1:
RGB-only
CLIP crop -> track pooling -> cosine retrieval

Baseline 2:
RGB + FPFH handcrafted geometry
CLIP + FPFH -> weighted fusion -> cosine retrieval

Prototype 1:
RGB + learned point geometry
CLIP + point encoder -> weighted fusion -> cosine retrieval

Prototype 2:
RGB + learned geometry + 4D motion rerank
embedding retrieval -> geometry / motion rerank

Prototype 3:
End-to-end 4D-aware ReID
RGB sequence + point sequence + motion sequence -> fusion network -> metric learning
```

`Prototype 1/2` 仍可保持检索式结构，用于快速替换弱几何分支和验证 4D 信息增益；`Prototype 3` 才是端到端训练型 ReID 网络。

## 3. 模块设计

### 3.1 learned point encoder 替换 FPFH

目标：把每个 timestamp 的点云几何描述从手工 FPFH 升级为可学习点云 embedding。

当前：

```text
P_t -> FPFH -> 33D
```

近期 prototype：

```text
P_t -> sample / normalize -> PointNet or DGCNN -> E_geo_t
```

最小接口建议：

```text
extract_node_track_embeddings.py
  --geo_backend learned_pointnet
  --point_encoder_checkpoint <path>
  --point_encoder_dim 128|256|512
```

借鉴工作：

- `point-cloud-reid`：Object Re-Identification from Point Clouds, WACV 2024；借鉴点云采样、归一化、point encoder、Siamese/metric matching。
- `CALM-Net`：CALM-Net: Curvature-Aware LiDAR Point Cloud-based Multi-Branch Neural Network for Vehicle Re-Identification, arXiv 2025；借鉴 DGCNN、PointTransformer、局部几何/eigenvalue、cross/local attention。

近期不直接移植完整 CALM-Net；先做轻量 point encoder，确认现有 NeoVerse 点云质量能否支撑 learned geometry。

### 3.2 NeoVerse 4D temporal / motion 信息利用

目标：避免把 4D 点云序列简单平均成静态几何，显式利用时序和运动信息。

当前：

```text
P_1, P_2, ..., P_T
-> per-frame FPFH
-> mean pooling
```

近期 prototype：

```text
P_t -> point encoder -> E_geo_t
{E_geo_t}_{t=1..T} -> temporal pooling / quality weighting -> E_geo_track

P_t -> centroid_t, bbox3d_t, extent_t
centroid sequence -> velocity_t, direction_t, trajectory length, motion smoothness
```

输出建议：

```text
E_geo_track
E_motion_track or motion_stats
```

借鉴工作：

- `AIC2025_Track1_ZV`：借鉴 trajectory aggregation、short-term tracking、long-term trajectory linking、spatial distance + ReID similarity。
- `Multiple Aerial Targets Re-Identification by 2D- and 3D-Kinematics-Based Matching`：借鉴速度向量、运动方向和 2D/3D 运动学匹配。

### 3.3 track-level visual encoder

目标：把当前 frame-level CLIP 均值池化升级为 track-level RGB 表征。

当前：

```text
multi-camera crop_t -> CLIP -> average
multi-timestamp -> average
```

近期 prototype：

```text
multi-view RGB crops over time
-> CLIP / vehicle-pretrained visual encoder
-> view-quality weighting
-> temporal pooling
-> E_rgb_track
```

质量权重可先使用非训练规则：

```text
mask area
bbox size
crop sharpness
point coverage
view coverage
```

借鉴工作：

- `DATOR / instance-based-loc`：借鉴 RGB-D object-instance descriptor 和多模态 object embedding 组织方式。
- `VehicleMAE / CVNet`：作为跨视角外观 ReID 和 vehicle-centric visual representation 的参考，不作为近期必须复现项。

### 3.4 object gallery + rerank

目标：把当前 `tracks.npy + tracks_meta.json` 升级为可维护多模态信息的 object gallery，并在 top-K 后加入几何/运动重排序。

当前：

```text
tracks.npy + tracks_meta.json
-> cosine retrieval
```

近期 prototype：

```text
track_gallery:
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
  source_scene
```

rerank 先采用非训练版：

```text
score_final =
  alpha * cosine(E_fused_q, E_fused_g)
+ beta  * cosine(E_geo_q, E_geo_g)
+ gamma * motion_similarity(q, g)
+ delta * shape_similarity(q, g)
```

借鉴工作：

- `instance-based-loc / DATOR`：object memory、embedding-to-object 管理、top-K object matching。
- `AIC2025_Track1_ZV`：gallery embeddings、Embedding Similarity Classifier、trajectory association。

## 4. 端到端 4D-aware ReID prototype

端到端 ReID 纳入近期规划，但只作为 prototype，不作为当前 node01 bootstrap 的完成条件。

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

训练目标候选：

```text
identity classification loss
triplet loss
contrastive loss
same/different matching loss
cross-modal consistency loss
```

近期 prototype 的成功标准不是直接追求最终指标，而是：

- 能用现有 `tracklets + points_by_timestamp + identity_id` 生成训练样本。
- 能输出与当前 `tracks.npy` 兼容的 track embedding。
- 能与 `RGB-only`、`CLIP + FPFH`、`CLIP + learned point encoder` 同口径评测。
- 不把小数据过拟合结果写成真实复杂场景结论。

## 5. 实验与验收口径

近期实验矩阵建议：

```text
1. RGB-only
2. RGB + FPFH
3. RGB + learned point geometry
4. RGB + learned point geometry + motion rerank
5. End-to-end 4D-aware ReID prototype
```

验收指标：

```text
mAP
Recall@1
Recall@5
Recall@10
top-K qualitative cases
failure cases
```

固定写法：

- 若指标持平，写成“链路可评测，当前数据难度不足或几何分支尚未释放价值”。
- 若 learned geometry 有提升，必须同时报告 RGB-only、FPFH baseline 和相同 query/gallery 划分。
- 若端到端 prototype 有提升，必须说明数据规模、训练/测试划分和过拟合风险。

### 5.1 DGGT + MapAnything sidecar 优化提案

这是一条 `旁路 POC -> 统一几何契约 -> 航迹恢复对比 -> 3D/4D-ReID smoke` 路线，不直接替换现有三角化主链，也不提前把推测性几何增益写成结论。

角色分工：

- `DGGT`：动态 4D 主干，负责 3DGS、dynamic motion、depth-like geometry、动态一致性。
- `MapAnything`：度量几何基线/补强，利用 CARLA-Air 已知内外参输出 metric depth / point maps。
- `Adapter`：把两条分支统一成下游可消费的 `points_by_timestamp`、`depth_by_frame` 和 `geometry_manifest`。

分支结构：

```text
DGGT branch:
  RGB sync frames -> DGGT -> 3DGS / dynamic map / motion / depth / points

MapAnything branch:
  RGB sync frames + CARLA K/extrinsics -> MapAnything -> metric depth / point maps

Adapter:
  DGGT / MapAnything outputs -> CARLA world-frame geometry contract
```

`method` 名称固定为：

```text
dggt
mapanything
dggt_mapanything_aligned
```

统一输出目录固定为：

```text
local/carla_air/geometry_4d/<capture_id>/<method>/
  manifest.json
  points_by_timestamp/
    index.csv
    *.npy
  depth_by_frame/
  camera_alignment.json
  quality_summary.json
```

几何契约：

- `points_by_timestamp/*.npy` 默认保存 world-frame `xyz`，优先扩展 `rgb`、`confidence`、`source_model`、`node_id`、`camera_id`、`ts_us`。
- `depth_by_frame/` 保存 per node / camera / timestamp 深度图。
- `camera_alignment.json` 记录 DGGT inferred pose 与 CARLA pose 的对齐误差。
- `quality_summary.json` 记录点数、有效深度比例、动态置信度、目标 ROI 点数、异常帧。
- 所有 sidecar 产物必须标记 `diagnostic_only=true`、`non_promotion=true`、`not_formal_geometry=true`。

航迹恢复新增几何辅助模式：

```text
tracklet_bbox_depth
tracklet_geometry_fusion
tracklet_mask_depth
```

v1 优先实现 `tracklet_bbox_depth`：

```text
tracklet bbox + depth/point map
-> ROI 内反投影/筛点
-> 多相机融合或 RANSAC
-> 每帧飞机 3D center
-> recovered_trajectory.csv + error summary
```

保留现有基线对照：

```text
oracle_projection
tracklet_bbox_yolo
tracklet_bbox_diff
tracklet_bbox_depth_dggt
tracklet_bbox_depth_mapanything
```

4D-ReID smoke 不使用整场景点云做 ReID，只从 bbox / mask 对应的目标局部区域提取：

```text
RGB crop embedding
local depth statistics
local point cloud descriptor
temporal motion descriptor
```

报告口径仍只能是 `single-identity 4D-ReID sanity`，不能声明多身份 benchmark。

里程碑：

- `M0 Readiness`：补齐 `third_party/dggt`、DGGT checkpoint、DGGT Python 环境；准备 MapAnything repo/weights，优先使用 Apache 权重；如使用非商用权重，manifest 必须明确标注；按仓库 Hugging Face mirror 规则下载权重。
- `M1 Sidecar POC`：cov01 / cov02 各抽 10-20 个同步 timestamp；跑 DGGT 和 MapAnything 两条旁路分支；输出 `geometry_4d/<capture_id>/<method>/`，不改主链产物。
- `M2 Adapter`：新增 adapter，把 DGGT / MapAnything 输出转成统一 `points_by_timestamp` 和 `depth_by_frame`；所有产物标记 non-promotion 边界。
- `M3 Trajectory Recovery`：实现 `tracklet_bbox_depth`；在 cov01 / cov02 上分别跑 DGGT、MapAnything geometry-assisted recovery；与 oracle、YOLO bbox、diff bbox 同表对比 RMSE / MAE / P95 / coverage / jitter。
- `M4 4D-ReID Smoke`：从目标局部几何提取 geometry embedding；复用现有 cov01<->cov02 retrieval sanity；输出双向 retrieval JSON，但报告中明确只有单身份 sanity。

静态检查：

```text
python -m py_compile <新增或修改脚本>
git diff --check
```

几何 POC 验收：

- `manifest.json` 非空，记录模型、checkpoint、license、输入 capture、hash、non-promotion 边界。
- `points_by_timestamp/index.csv` 非空。
- `depth_by_frame/` 至少覆盖抽样 timestamp 的有效相机。
- `quality_summary.json` 包含有效点数、有效深度比例、ROI 点数、失败帧。

航迹恢复验收输出沿用现有恢复产物：

```text
recovered_trajectory.csv
trajectory_observations.csv
trajectory_error_summary.json
trajectory_compare.html/svg/png
manifest.json
```

`trajectory_error_summary.json` 至少包含：

```text
rmse_3d_m
mae_3d_m
p95_3d_m
coverage_ratio
geometry_frame_count
jitter_m
```

ReID smoke 验收：

- cov01->cov02、cov02->cov01 都有输出。
- `num_queries > 0`。
- `num_gallery > 0`。
- `metric_queries > 0`。
- 报告标注 `single_identity_sanity_only=true`。

前置假设：

- 本阶段不启动 CARLA-Air / AirSim runtime，只使用已有 cov01 / cov02 capture。
- DGGT / MapAnything 输出先作为旁路 diagnostic artifact，不升级为 formal annotation、`mask_gt` 或 final 4D geometry。
- 目标 gating v1 使用现有 bbox / diff / YOLO tracklet；可靠 mask 出现后再启用 `tracklet_mask_depth`。
- 当前三角化链路和 `oracle_projection` 必须保留，作为判断 4D 几何是否真正提升的基线。

## 6. 与历史分支关系

- `recon_spin`：保留为历史重建与诊断线。
- `RGB + predicted-depth geometry`：保留为弱几何 baseline。
- 旧 `RGB + fused geometry`：保留为多相机 depth/mask 融合对照线。
- `GT upper-bound`：保留为感知误差诊断线。
- `NeoVerse fused 4D points_by_timestamp`：当前 4D 几何主输入。

后续 planning、汇报和新对话应优先读取：

```text
research/plans/ACTIVE_PLAN.md
research/plans/tri_camera_node_3d_aware_reid/master_plan_zh.md
research/plans/tri_camera_node_3d_aware_reid/neoverse_4d_reid_strengthening_plan_zh.md
research/guides/node01_neoverse_fused_4d_reid_zh.md
```
