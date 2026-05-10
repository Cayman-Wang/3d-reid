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

初始状态下只有 1 个 scene 具备 NeoVerse fused 4D `points_by_timestamp`：

```text
D:/node01_spin_runtime_ascii/mvp-demo/output/neoverse_fused_runs/2026-04-26_j10_yp20_r02params_r01/mj_node01_j10_spin_static_yp20_a/points_by_timestamp
```

该目录已验证：

```text
schema_version = neoverse_points_by_timestamp_v1
num_timestamps = 81
npy files = 81
```

当前已补齐剩余 5 个 scene；6 个 scene 现在都具备可直接消费的 `points_by_timestamp`：

```text
D:/node01_spin_runtime_ascii/mvp-demo/output/neoverse_fused_runs/2026-04-26_j10_yp20_r02params_r01/mj_node01_j10_spin_static_yp20_a/points_by_timestamp
D:/node01_spin_runtime_ascii/mvp-demo/output/neoverse_fused_runs/node01_neoverse_fused_4d_eval_matrix_v1/mj_node01_j10_spin_circle_yp_b/points_by_timestamp
D:/node01_spin_runtime_ascii/mvp-demo/output/neoverse_fused_runs/node01_neoverse_fused_4d_eval_matrix_v1/mj_node01_uav1_spin_static_yp_a/points_by_timestamp
D:/node01_spin_runtime_ascii/mvp-demo/output/neoverse_fused_runs/node01_neoverse_fused_4d_eval_matrix_v1/mj_node01_uav1_spin_circle_yp_b/points_by_timestamp
D:/node01_spin_runtime_ascii/mvp-demo/output/neoverse_fused_runs/node01_neoverse_fused_4d_eval_matrix_v1/mj_node01_su34_spin_static_yp_a/points_by_timestamp
D:/node01_spin_runtime_ascii/mvp-demo/output/neoverse_fused_runs/node01_neoverse_fused_4d_eval_matrix_v1/mj_node01_su34_spin_circle_yp_b/points_by_timestamp
```

每个目录当前都已验证：

```text
schema_version = neoverse_points_by_timestamp_v1
index.csv rows = 81
npy files = 81
```

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

`rgb_neoverse_fused_4d_clip_fpfh_eval_v1` 现已可直接运行。

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

## 本机运行问题与规避

### 1. `conda run` 在当前 Windows 控制台上会炸编码

现象：

```text
UnicodeEncodeError: 'gbk' codec can't encode character ...
```

触发点：

- 主要出现在 `conda run -n neoverse python ...` 包装 NeoVerse fused 4D 脚本时
- 脚本本身可能已经执行成功，但 `conda` 在回写 stdout 时崩掉，导致难以判断哪些步骤真正完成

当前机器的规避方式：

```powershell
D:\ML\anaconda3\envs\neoverse\python.exe ...
D:\ML\anaconda3\envs\mvp_demo\python.exe ...
```

结论：在这台机器上，NeoVerse fused 4D 主链和 eval 主链都优先直接调用环境内 `python.exe`，不要继续依赖 `conda run`。

### 2. `export_neoverse_view_observations.py` 在本机不能用 `float16`

现象：

```text
"projection_ewa_3dgs_fused_fwd_kernel" not implemented for 'Half'
```

直接后果：

- `observations/observations_report.json` 会出现 `num_rows = 0`
- `observations/index.csv` 只有表头或为空
- 下游会继续报：
  - `No observations found in ... observations/index.csv`
  - `Missing bg points index ... points_per_view/points_index.csv`
  - `Missing dynamic_index.csv ... fused/dynamic_index.csv`

当前机器的规避方式：

- `run_neoverse_per_camera_bundle.py` 保持当前默认 `float16` 可以正常跑
- 但 `export_neoverse_view_observations.py` 必须显式改成：

```powershell
--torch_dtype float32
```

结论：本机 NeoVerse fused 4D 链条里，`export_neoverse_view_observations.py` 的 dtype 固定写成 `float32`，不要再尝试 `float16`。

## 当前结果

已运行 RGB-only baseline：

- `rgb_only_clip_gtmask_eval_v1`
- `metric_queries = 6`
- `mAP = 1.0`
- `recall_at_1 = 1.0`
- `tracks.npy shape = (1, 512)` for each scene

已运行 RGB + NeoVerse 4D geometry：

- `rgb_neoverse_fused_4d_clip_fpfh_eval_v1`
- `metric_queries = 6`
- `mAP = 1.0`
- `recall_at_1 = 1.0`
- `recall_at_5 = 1.0`
- `recall_at_10 = 1.0`
- `tracks.npy shape = (1, 545)` for each scene
- `tracks_meta.json` 记录：
  - `rgb_backend = clip`
  - `geo_backend = open3d_fpfh`
  - `rgb_weight = 1.0`
  - `geo_weight = 0.35`

## 建议下一步

1. 后续在这台机器上复跑 NeoVerse fused 4D 时，直接用环境内 `python.exe`，避免 `conda run` 编码崩溃。
2. 固定把 `export_neoverse_view_observations.py` 的 `--torch_dtype` 写成 `float32`。
3. 若后续扩展更多 scene，继续把新 `points_by_timestamp` 写入 `D:/node01_spin_runtime_ascii/mvp-demo/output/neoverse_fused_runs/node01_neoverse_fused_4d_eval_matrix_v1/<scene_id>/points_by_timestamp`。
4. 当前 6-scene 指标仍与 RGB-only 持平；后续若要追求提升，应优先从更高输入分辨率或更强机器开始，而不是重复本机链路级排障。

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

当前完整分支命令：

```powershell
D:\ML\anaconda3\envs\mvp_demo\python.exe mvp-demo/scripts/run_iciscae_branch_eval.py `
  --manifest research/plans/tri_camera_node_3d_aware_reid/benchmarks/node01_neoverse_fused_4d_eval_matrix_v1.json `
  --branch rgb_neoverse_fused_4d_clip_fpfh_eval_v1 `
  --topk 5
```

## 不要误写的结论

当前不能写“NeoVerse 4D geometry 已经获得正式 Rank/mAP 提升”。正确表述是：

```text
NeoVerse fused 4D 已完成 node01 单节点 3 identities x 2 scenes 可评测矩阵 bootstrap；当前已完成接入与可评测对比，下一阶段转向 cross-node smoke。
```
