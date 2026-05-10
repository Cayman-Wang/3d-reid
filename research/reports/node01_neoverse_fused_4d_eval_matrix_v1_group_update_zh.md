# Node01 NeoVerse Fused 4D Eval Matrix V1 阶段汇总

日期：2026-05-10

## 1. 阶段结论

- `node01` 单节点 `3 identities x 2 scenes` 的 NeoVerse fused 4D 接入 3D-ReID 可评测矩阵已经闭环完成。
- 6 个 `scene` 的 `points_by_timestamp` 均满足下游契约：`schema_version=neoverse_points_by_timestamp_v1`、`index.csv=81 rows`、`*.npy=81`。
- `rgb_only_clip_gtmask_eval_v1` 与 `rgb_neoverse_fused_4d_clip_fpfh_eval_v1` 都已跑出 `metric_queries=6`。
- 当前不能写成 geometry 带来提升。两条分支在当前 `masks_gt/depth_gt` bootstrap 下持平：`mAP=1.0`、`Rank-1=1.0`、`Rank-5=1.0`。

## 2. 矩阵盘点

当前矩阵覆盖 `j10 / uav1 / su34` 各 2 个 `spin scene`：

| identity_id | scene_id | scene_dir | points_by_timestamp | timestamps | status |
| --- | --- | --- | --- | ---: | --- |
| `j10` | `mj_node01_j10_spin_static_yp20_a` | `.../scenes/mj_node01_j10_spin_static_yp20_a` | `.../2026-04-26_j10_yp20_r02params_r01/mj_node01_j10_spin_static_yp20_a/points_by_timestamp` | `81` | `ready` |
| `j10` | `mj_node01_j10_spin_circle_yp_b` | `.../scenes/mj_node01_j10_spin_circle_yp_b` | `.../node01_neoverse_fused_4d_eval_matrix_v1/mj_node01_j10_spin_circle_yp_b/points_by_timestamp` | `81` | `ready` |
| `uav1` | `mj_node01_uav1_spin_static_yp_a` | `.../scenes/mj_node01_uav1_spin_static_yp_a` | `.../node01_neoverse_fused_4d_eval_matrix_v1/mj_node01_uav1_spin_static_yp_a/points_by_timestamp` | `81` | `ready` |
| `uav1` | `mj_node01_uav1_spin_circle_yp_b` | `.../scenes/mj_node01_uav1_spin_circle_yp_b` | `.../node01_neoverse_fused_4d_eval_matrix_v1/mj_node01_uav1_spin_circle_yp_b/points_by_timestamp` | `81` | `ready` |
| `su34` | `mj_node01_su34_spin_static_yp_a` | `.../scenes/mj_node01_su34_spin_static_yp_a` | `.../node01_neoverse_fused_4d_eval_matrix_v1/mj_node01_su34_spin_static_yp_a/points_by_timestamp` | `81` | `ready` |
| `su34` | `mj_node01_su34_spin_circle_yp_b` | `.../scenes/mj_node01_su34_spin_circle_yp_b` | `.../node01_neoverse_fused_4d_eval_matrix_v1/mj_node01_su34_spin_circle_yp_b/points_by_timestamp` | `81` | `ready` |

补充说明：

- `manifest` 路径：`research/plans/tri_camera_node_3d_aware_reid/benchmarks/node01_neoverse_fused_4d_eval_matrix_v1.json`
- `j10 static` 沿用稳定最佳 run：`2026-04-26_j10_yp20_r02params_r01`
- 其余 5 个 scene 的新产物位于：`D:\node01_spin_runtime_ascii\mvp-demo\output\neoverse_fused_runs\node01_neoverse_fused_4d_eval_matrix_v1\<scene_id>\points_by_timestamp`

## 3. Manifest 与 Contract 固化

当前 `manifest` 已满足以下状态：

- 6 个 `entry.neoverse_points_status` 均为 `ready`
- `rgb_neoverse_fused_4d_clip_fpfh_eval_v1` 使用 `points_subdir: "{neoverse_points_subdir}"`
- `points_contract` 固定为 `neoverse_points_by_timestamp_v1`

合同复核结果：

- 每个 `meta.json` 均记录 `schema_version=neoverse_points_by_timestamp_v1`
- 每个 `index.csv` 均为 `81 rows`
- 每个目录下 `*.npy=81`，与 `index.csv` 一一对应

## 4. RGB-only vs RGB+NeoVerse 4D Geometry

| branch | input | embedding dim | metric_queries | mAP | Rank-1 | Rank-5 | conclusion |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `rgb_only_clip_gtmask_eval_v1` | `RGB crop only + masks_gt/depth_gt bootstrap` | `512` | `6` | `1.0` | `1.0` | `1.0` | `Node01 3x2 matrix baseline 已可稳定评测。` |
| `rgb_neoverse_fused_4d_clip_fpfh_eval_v1` | `RGB crop + points_by_timestamp + masks_gt/depth_gt bootstrap` | `545` | `6` | `1.0` | `1.0` | `1.0` | `NeoVerse 4D geometry branch 已接入并可评测，但当前 GT-mask bootstrap 上与 RGB-only 持平，未观察到指标提升。` |

补充核对：

- RGB-only 每个 `scene` 的 `tracks.npy` 形状均为 `(1, 512)`，且无 `NaN`
- RGB+NeoVerse 4D 每个 `scene` 的 `tracks.npy` 形状均为 `(1, 545)`，且无 `NaN`
- `545 = 512 CLIP + 33 Open3D FPFH`
- `tracks_meta.json` 已记录：
  - `rgb_backend=clip`
  - `geo_backend=open3d_fpfh`
  - `rgb_weight=1.0`
  - `geo_weight=0.35`

## 5. 组会展示口径

建议组会里把两栏图解释为：

- 左栏：`RGB crop`，保留外观纹理与颜色线索
- 右栏：`fused 4D points` 的投影叠加，不是重新渲染的纹理图

对于 `soft preview` 看起来发糊，建议固定解释为以下 4 个原因：

- NeoVerse 当前输入分辨率只有 `280x168`
- 目标在原图中占比小
- 单帧点云仍然偏稀疏
- `soft splat` 会做颜色平均，不负责恢复纹理细节

注意事项：

- 不覆盖既有展示目录：`preview/local_target_compare_v1`、`preview/local_target_rgb_points_compare_v1`、`gaussian_style_compare_fixed_alpha`
- 若补新图，只写入新的 `preview` 子目录

## 6. 当前机器运行注意事项

为避免后续重复踩坑，当前机器固定记录两条：

1. ReID 与 NeoVerse fused 脚本都优先直接调用环境解释器，不优先使用 `conda run`

```powershell
D:\ML\anaconda3\envs\mvp_demo\python.exe ...
D:\ML\anaconda3\envs\neoverse\python.exe ...
```

原因：`conda run` 在当前 Windows 控制台上会触发 `UnicodeEncodeError`，容易造成脚本已执行但终端报错的误判。

2. `export_neoverse_view_observations.py` 在本机必须显式使用：

```powershell
--torch_dtype float32
```

原因：若误用 `float16`，会报 `"projection_ewa_3dgs_fused_fwd_kernel" not implemented for 'Half'"`，并导致 `observations/index.csv` 为空。

更完整的机器说明已同步到：

- `research/guides/node01_neoverse_fused_4d_reid_zh.md`
- `research/handoffs/node01_neoverse_fused_4d_eval_matrix_handoff_zh.md`

## 7. 下一阶段路线

当前阶段不再优先继续调 preview，也不再把重点放在“补齐 node01 点云”。下一步应切到 `node02 / cross-node smoke`：

1. 先复刻 `j10 / uav1 / su34`，每个 `identity` 至少补 `1 scene` 做 `node02` 冒烟闭环。
2. 在 `node01 + node02` 上形成最小跨节点矩阵：`3 identities x 2 nodes x 2 scenes`
3. 继续保留 `RGB-only` 作为基线，与 `RGB + NeoVerse 4D geometry` 做同口径对比
4. 在仍使用 `masks_gt / depth_gt` 的前提下，先隔离 cross-node 评测链路问题，再决定是否切到更真实的 mask/depth 来源

当前阶段的正确表述应是：

- `node01 单节点、多身份、多 scene、NeoVerse 4D 接入 3D-ReID 的可评测矩阵闭环完成`
- `当前已完成接入与可评测对比，RGB-only 与 RGB+NeoVerse 4D geometry 持平`
- `下一阶段转向 cross-node smoke，而不是继续优化展示图`
