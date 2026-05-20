# Node01 NeoVerse Own-Depth 4D Eval V1

日期：2026-05-11

## 结论

本实验用于验证：在 `node01` 三相机 NeoVerse 4D 几何链中，保留 `masks_gt` 和 `rig.json`，去掉 `scene depth_gt` 锚深度之后，NeoVerse own-depth 4D geometry 是否仍可被 ReID 消费。

本报告的正确口径是：

- 去掉 `depth_gt` 后，NeoVerse own-depth 4D geometry 已可评测。
- 这不是“完全无真值”实验，因为第一版仍使用 `masks_gt` 和 `rig.json`。
- 不得写成 geometry 已提升，除非指标确实提升。
- 如果指标仍为 `1.0`，结论应写成：当前 `node01` GT-mask bootstrap 难度不足以体现 depth source 差异。

## 产物边界

- 稳定 run 不覆盖：
  - `D:/node01_spin_runtime_ascii/mvp-demo/output/neoverse_fused_runs/2026-04-26_j10_yp20_r02params_r01`
- 当前 eval matrix、view ablation、已有 preview 不覆盖。
- 不修改：
  - `D:/研究生/grad_project_recon/组会思路/26-05-11_node01_neoverse_4d_group_meeting_visuals_and_metrics.md`
- 新实验输出根目录：
  - `D:/node01_spin_runtime_ascii/mvp-demo/output/neoverse_fused_runs/node01_neoverse_own_depth_4d_eval_v1/<scene_id>/points_by_timestamp`

本次实际未执行的动作：

- 未删除任何产物。
- 未覆盖稳定 run：`2026-04-26_j10_yp20_r02params_r01`
- 未覆盖当前 eval matrix：`node01_neoverse_fused_4d_eval_matrix_v1`
- 未覆盖当前 view ablation：`node01_neoverse_fused_4d_view_ablation_v1`
- 未覆盖已有 preview
- 未修改 `组会思路/26-05-11_node01_neoverse_4d_group_meeting_visuals_and_metrics.md`

## 代码修改

本次最小改动如下：

| 文件 | 修改 |
| --- | --- |
| `mvp-demo/scripts/constrain_neoverse_multiview_dynamic.py` | 新增 `--anchor_depth_source {scene_depth_gt_then_multiview,multiview_rays_only}`；默认保持旧行为。使用 `multiview_rays_only` 时不再读取 `scene_dir/cams/<cam>/depth_gt/*.npy`，只用三相机 mask bbox center rays 三角化得到 `mask_anchor_world` 和 `anchor_depths`；仍保留 observation depth 用于 `local_depth_median`、depth scale、aligned fg points。 |
| `mvp-demo/scripts/build_node_tracklets.py` | 新增 `--allow_missing_depth`；启用后即使 `depth_subdir` 文件不存在也不跳过 timestamp，`depth_paths` 写空字符串，其他字段保持正常。 |
| `mvp-demo/scripts/run_iciscae_branch_eval.py` | 支持 branch config 的 `allow_missing_depth` 并透传给 `build_node_tracklets.py`；旧 branch 默认行为不变。 |
| `research/plans/tri_camera_node_3d_aware_reid/benchmarks/node01_neoverse_own_depth_4d_eval_v1.json` | 新增 own-depth 评测 manifest。 |

## 几何生成口径

6 个 scene 的 own-depth `points_by_timestamp` 都复用已有 `observations` 和 `points_per_view`，不重跑 NeoVerse reconstructor。

生成命令口径：

- `fuse_neoverse_multiview_world_points.py`：
  - `D:\ML\anaconda3\envs\neoverse\python.exe`
  - `--cams cam0,cam1,cam2`
  - `--min_bg_cam_support 2`
- `constrain_neoverse_multiview_dynamic.py`：
  - `D:\ML\anaconda3\envs\neoverse\python.exe`
  - `--cams cam0,cam1,cam2`
  - `--min_mask_cam_support 2`
  - `--anchor_depth_source multiview_rays_only`

own-depth 输出目录：

- `D:/node01_spin_runtime_ascii/mvp-demo/output/neoverse_fused_runs/node01_neoverse_own_depth_4d_eval_v1/<scene_id>/points_by_timestamp`

## 几何契约验收

6 个 scene 全部满足：

- `points_by_timestamp/meta.json.schema_version = neoverse_points_by_timestamp_v1`
- `points_by_timestamp/index.csv = 81 rows`
- `points_by_timestamp/*.npy = 81`
- `fused/dynamic_constraint_meta.json.uses_scene_depth_gt = false`
- `fused/dynamic_constraint_meta.json.anchor_depth_source = multiview_rays_only`
- `points_by_timestamp/meta.json` 也记录：
  - `uses_scene_depth_gt = false`
  - `anchor_depth_source = multiview_rays_only`
  - `render_depth_unit = neoverse_local_metric_like`
  - `cams = ["cam0", "cam1", "cam2"]`

## ReID 结果

新 branch：

- `rgb_neoverse_own_depth_4d_clip_fpfh_eval_v1`

结果：

| branch | metric_queries | mAP | Rank-1 | Rank-5 |
| --- | ---: | ---: | ---: | ---: |
| `rgb_neoverse_own_depth_4d_clip_fpfh_eval_v1` | 6 | 1.0 | 1.0 | 1.0 |

embedding 验收：

- 每个 scene 的 `tracks.npy` 形状均为 `(1,545)`
- `tracks.npy` 无 NaN
- `tracks_meta.json` 记录：
  - `rgb_backend=clip`
  - `geo_backend=open3d_fpfh`
  - `rgb_weight=1.0`
  - `geo_weight=0.35`

## 三线对比

| 实验线 | mAP | Rank-1 | Rank-5 | metric_queries |
| --- | ---: | ---: | ---: | ---: |
| `RGB-only` | 1.0 | 1.0 | 1.0 | 6 |
| `旧 RGB+NeoVerse depth_gt bootstrap` | 1.0 | 1.0 | 1.0 | 6 |
| `新 RGB+NeoVerse own-depth no-depthGT` | 1.0 | 1.0 | 1.0 | 6 |

## 结论口径

当前应写成：

- 去掉 `scene depth_gt` 锚深度后，NeoVerse own-depth 4D geometry 已可评测。
- 这仍不是“完全无真值”实验，因为第一版保留了 `masks_gt` 和 `rig.json`。
- 当前 `node01` GT-mask bootstrap 难度不足以体现 depth source 差异。
- 本次结果不能支持“geometry 已提升”或“own-depth 优于 depth_gt bootstrap”的结论。
