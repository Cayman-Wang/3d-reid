# Node01 NeoVerse Fused 4D Eval Matrix Handoff

## 当前定位

当前阶段是 `M7 多相机 NeoVerse 4D 动态点云接入 3D-ReID 正式化阶段`。已完成 `node01/j10` 单 scene 工程 smoke：`points_by_timestamp -> tracklet -> CLIP RGB crop + FPFH -> ReID`。该 smoke 证明接口闭环，但不产生正式 Rank/mAP 结论。

本次新增的矩阵入口是：

```text
research/plans/tri_camera_node_3d_aware_reid/benchmarks/node01_neoverse_fused_4d_eval_matrix_v1.json
```

## 本地 readiness

本地 `D:/node01_spin_runtime_ascii` 目前有 6 个 spin scene，均具备 `frames / masks_gt / depth_gt / rig.json / frame_times.csv`：

```text
mj_node01_j10_spin_static_yp20_a
mj_node01_j10_spin_circle_yp_b
mj_node01_uav1_spin_static_yp_a
mj_node01_uav1_spin_circle_yp_b
mj_node01_su34_spin_static_yp_a
mj_node01_su34_spin_circle_yp_b
```

但当前只有 1 个 scene 具备 NeoVerse fused 4D `points_by_timestamp`：

```text
D:/node01_spin_runtime_ascii/mvp-demo/output/neoverse_fused_runs/2026-04-26_j10_yp20_r02params_r01/mj_node01_j10_spin_static_yp20_a/points_by_timestamp
```

该目录已验证：

```text
schema_version = neoverse_points_by_timestamp_v1
num_timestamps = 81
npy files = 81
```

因此当前不能直接跑完整 `RGB + NeoVerse 4D geometry` 6-scene branch；缺失的 5 个 scene 必须先生成 `points_by_timestamp`。

## 已实现能力

`run_iciscae_branch_eval.py` 已支持 branch config 中使用 entry 字段模板，例如：

```json
"points_subdir": "{neoverse_points_subdir}"
```

这样同一个 branch 可以在不同 scene 上消费不同的 `points_by_timestamp` 目录。旧 manifest 不受影响。

新 manifest 提供两个分支：

```text
rgb_only_clip_gtmask_eval_v1
rgb_neoverse_fused_4d_clip_fpfh_eval_v1
```

`rgb_only_clip_gtmask_eval_v1` 已运行通过，用于确认 6-scene retrieval/eval surface 正常并产生 `metric_queries > 0`。

`rgb_neoverse_fused_4d_clip_fpfh_eval_v1` 当前是正式矩阵的目标分支，等待缺失 scene 的 `points_by_timestamp` 补齐后运行。

## 固定参数

NeoVerse 4D ReID 分支保持当前 smoke 参数：

```text
rgb_backend = clip
geo_backend = open3d_fpfh
rgb_weight = 1.0
geo_weight = 0.35
max_timestamps_per_track = 30
max_points_per_timestamp = 5000
points_contract = neoverse_points_by_timestamp_v1
apply_mask_to_rgb = false
```

当前矩阵 bootstrap 使用 `masks_gt/depth_gt`，目的是让 RGB-only 与 NeoVerse 4D geometry 分支共享同一 crop/bbox 来源，先隔离几何分支接入问题。

## 建议下一步

1. 为缺失的 5 个 spin scene 生成 NeoVerse fused 4D `points_by_timestamp`。
2. 对每个新目录检查 `meta.json.schema_version == neoverse_points_by_timestamp_v1`，并确认 `index.csv` 和 `*.npy` 数量匹配。
3. 运行 `rgb_neoverse_fused_4d_clip_fpfh_eval_v1`。
4. 对比两个分支的 `all_queries_vs_all_scenes.json`，再判断 NeoVerse 4D geometry 是否带来提升。

已运行的 RGB-only baseline 命令：

```powershell
conda run -n mvp_demo python mvp-demo/scripts/run_iciscae_branch_eval.py `
  --manifest research/plans/tri_camera_node_3d_aware_reid/benchmarks/node01_neoverse_fused_4d_eval_matrix_v1.json `
  --branch rgb_only_clip_gtmask_eval_v1 `
  --topk 5
```

结果文件：

```text
mvp-demo/output/evals/node01_neoverse_fused_4d_eval_matrix_v1/rgb_only_clip_gtmask_eval_v1/all_queries_vs_all_scenes.json
```

当前结果：

```text
num_queries = 6
num_gallery = 6
metric_queries = 6
mAP = 1.0
recall_at_1 = 1.0
recall_at_5 = 1.0
recall_at_10 = 1.0
tracks.npy shape = (1, 512) for each scene
```

完整点云补齐后再运行：

```powershell
conda run -n mvp_demo python mvp-demo/scripts/run_iciscae_branch_eval.py `
  --manifest research/plans/tri_camera_node_3d_aware_reid/benchmarks/node01_neoverse_fused_4d_eval_matrix_v1.json `
  --branch rgb_neoverse_fused_4d_clip_fpfh_eval_v1 `
  --topk 5
```

## 不要误写的结论

当前不能写“NeoVerse 4D geometry 已经获得正式 Rank/mAP 提升”。正确表述是：

```text
NeoVerse fused 4D 已完成 ReID 接口级闭环；当前正在从单 scene smoke 扩展到多 scene、多 identity 的可评测矩阵。
```
