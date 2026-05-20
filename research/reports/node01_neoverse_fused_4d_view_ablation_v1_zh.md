# Node01 NeoVerse Fused 4D View-Count Ablation V1

日期：2026-05-11

## 1. 结论

本次新增 `node01_neoverse_fused_4d_view_ablation_v1`，用于比较 `cam0/cam1/cam2` 单相机与既有 tri-cam 三相机在 3D-ReID bootstrap 上的差异。

结论必须保守：

- 这是 view-count ablation，不替代三相机主线。
- 单相机 NeoVerse 几何只用于对照，不作为最终系统目标。
- 当前主线仍是三相机节点级 3D-aware track retrieval，后续再扩展到 cross-node。
- 当前 `node01` GT-mask bootstrap 难度不足以体现视角数量差异：单相机与三相机结果均为 `mAP=1.0, R@1=1.0, R@5=1.0`。
- 不得写成“三相机已有指标提升”，也不得写成“geometry 已带来提升”。

本次没有删除任何产物，没有覆盖当前稳定 NeoVerse run，没有覆盖当前 eval matrix，没有覆盖已有 preview 目录。

## 2. Manifest 与输出目录

- 新 manifest：
  - `research/plans/tri_camera_node_3d_aware_reid/benchmarks/node01_neoverse_fused_4d_view_ablation_v1.json`
- 既有 tri-cam baseline manifest：
  - `research/plans/tri_camera_node_3d_aware_reid/benchmarks/node01_neoverse_fused_4d_eval_matrix_v1.json`
- 新单相机几何输出根目录：
  - `D:\node01_spin_runtime_ascii\mvp-demo\output\neoverse_fused_runs\node01_neoverse_fused_4d_view_ablation_v1\<cam>\<scene_id>\points_by_timestamp`

单相机几何没有重跑 `run_neoverse_per_camera_bundle.py`。每个 `cam/scene` 都复用已有矩阵的 `per_camera / observations / points_per_view`，再派生对应单相机 `fused/` 与 `points_by_timestamp/`。

## 3. 脚本修复

本次最小修改了 3 个脚本：

| 脚本 | 修改 |
| --- | --- |
| `mvp-demo/scripts/run_iciscae_branch_eval.py` | `build_node_tracklets.py` 调用新增传入 branch config 的 `--cams`，保证单相机 RGB-only branch 真正只消费指定相机。 |
| `mvp-demo/scripts/fuse_neoverse_multiview_world_points.py` | `--cams` 现在会过滤 `points_index.csv` 中的 `cam_id`，单相机运行可配合 `--min_bg_cam_support 1` 生成真单相机融合产物。 |
| `mvp-demo/scripts/constrain_neoverse_multiview_dynamic.py` | 放开原三相机固定检查，支持 `--cams <cam> --min_mask_cam_support 1` 的单相机动态约束；保留三相机主线参数口径。 |

同时更新 `.gitignore`：

- 新增忽略 `third_party/Track4World/`
- 不删除 `third_party/Track4World/`

## 4. 几何契约验证

已完成 `cam0/cam1/cam2 x 6 scenes = 18` 个单相机 `points_by_timestamp` 目录。

全部目录满足：

- `meta.json.schema_version = neoverse_points_by_timestamp_v1`
- `index.csv = 81 rows`
- `*.npy = 81`
- 每帧点数均高于 ReID branch 的 `min_points_per_timestamp=32`

按相机汇总的最小帧点数：

| camera | scenes | min frame points |
| --- | ---: | ---: |
| `cam0` | 6 | 1710 |
| `cam1` | 6 | 281 |
| `cam2` | 6 | 1985 |

## 5. 单相机 Branch 结果

| branch | metric_queries | mAP | Rank-1 | Rank-5 | tracks.npy |
| --- | ---: | ---: | ---: | ---: | --- |
| `rgb_only_clip_gtmask_cam0_eval_v1` | 6 | 1.0 | 1.0 | 1.0 | `(1,512)` |
| `rgb_only_clip_gtmask_cam1_eval_v1` | 6 | 1.0 | 1.0 | 1.0 | `(1,512)` |
| `rgb_only_clip_gtmask_cam2_eval_v1` | 6 | 1.0 | 1.0 | 1.0 | `(1,512)` |
| `rgb_neoverse_single_cam0_4d_clip_fpfh_eval_v1` | 6 | 1.0 | 1.0 | 1.0 | `(1,545)` |
| `rgb_neoverse_single_cam1_4d_clip_fpfh_eval_v1` | 6 | 1.0 | 1.0 | 1.0 | `(1,545)` |
| `rgb_neoverse_single_cam2_4d_clip_fpfh_eval_v1` | 6 | 1.0 | 1.0 | 1.0 | `(1,545)` |

验收结果：

- `all_queries_vs_all_scenes.json` 中 `summary.metric_queries = 6`
- RGB-only 单相机 `tracks.npy` 形状均为 `(1,512)`
- RGB+4D 单相机 `tracks.npy` 形状均为 `(1,545)`
- `tracks.npy` 无 NaN
- RGB+4D `tracks_meta.json` 记录：
  - `rgb_backend=clip`
  - `geo_backend=open3d_fpfh`
  - `rgb_weight=1.0`
  - `geo_weight=0.35`

## 6. View-Count 对比

既有 tri-cam baseline 直接引用 `node01_neoverse_fused_4d_eval_matrix_v1`，本次不重跑三相机 branch。

| view count | RGB-only mAP / R@1 / R@5 | RGB+4D mAP / R@1 / R@5 | metric_queries | 口径 |
| --- | --- | --- | ---: | --- |
| `cam0` | `1.0 / 1.0 / 1.0` | `1.0 / 1.0 / 1.0` | 6 | 单相机对照 |
| `cam1` | `1.0 / 1.0 / 1.0` | `1.0 / 1.0 / 1.0` | 6 | 单相机对照 |
| `cam2` | `1.0 / 1.0 / 1.0` | `1.0 / 1.0 / 1.0` | 6 | 单相机对照 |
| `tri-cam` | `1.0 / 1.0 / 1.0` | `1.0 / 1.0 / 1.0` | 6 | 主线 baseline，引用既有 eval matrix |

解释：

- 当前结果说明 `node01` GT-mask bootstrap 对 CLIP 外观检索已经过于容易。
- RGB-only、单相机 RGB+4D、三相机 RGB+4D 都满分，因此本消融不能支持“视角越多指标越高”的结论。
- 当前可写入论文或组会的结论是：view-count ablation 已完成，同口径可评测；但需要更难的 cross-node、真实 mask/depth 或更细粒度身份/姿态干扰来体现视角数量和几何的价值。

## 7. 运行与不变项

本次运行遵守当前机器约定，优先直接调用解释器：

```powershell
D:\ML\anaconda3\envs\mvp_demo\python.exe
D:\ML\anaconda3\envs\neoverse\python.exe
```

未执行的操作：

- 未删除任何产物。
- 未重跑 `run_neoverse_per_camera_bundle.py`。
- 未覆盖 `2026-04-26_j10_yp20_r02params_r01`。
- 未覆盖 `node01_neoverse_fused_4d_eval_matrix_v1` 的现有三相机结果。
- 未覆盖已有 preview 目录。
