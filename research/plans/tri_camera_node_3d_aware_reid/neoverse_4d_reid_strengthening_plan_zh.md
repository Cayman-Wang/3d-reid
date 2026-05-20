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
