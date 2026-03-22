# ICISCAE 周收口包（2026-03-29）

## 1. 收口结论

- 本周冻结的执行计划已经全部完成：
  - `RGB + predicted-depth geometry` 已跑通并独立落盘；
  - `RGB + fused geometry` 已跑通并独立落盘；
  - `GT upper-bound` 已按 `masks_gt + depth_gt + fused geometry` 独立落盘；
  - 小论文正文草稿与周汇总文档已补齐。
- 当前统一对照结论已经固化到：
  - `mvp-demo/output/evals/iciscae_node01_uav_v1/branch_comparison_summary.json`
  - `mvp-demo/output/evals/iciscae_node01_uav_v1/branch_comparison_summary.md`
- 本周最核心的研究判断不是“geometry 已经带来收益”，而是：
  - geometry 能修复个别 query（如 `j10_line`）；
  - 但当前实现没有在整体指标上超过 `RGB-only`；
  - `GT upper-bound` 也没有整体超过 `RGB-only`，所以问题不只是感知输入噪声。

## 2. 本周新增工程能力

### 2.1 新增或更新的脚本能力

- `mvp-demo/scripts/recon_fuse_depth_points.py`
  - 新增 `--out_subdir`，支持把弱几何、强几何、GT 点云写到不同目录。
  - 已补强中文路径下的 mask 读取。
- `mvp-demo/scripts/eval_node_track_retrieval.py`
  - 新增 `--embeddings_subdir`，支持 branch-specific 检索。
- 新增批量执行脚本：
  - `mvp-demo/scripts/run_iciscae_branch_eval.py`
- 新增对照汇总脚本：
  - `mvp-demo/scripts/summarize_iciscae_branch_comparison.py`

### 2.2 分支输出布局已冻结

- `rgb_only`
  - `tracks/tracklets.json`
  - `embeddings/`
  - `mvp-demo/output/evals/iciscae_node01_uav_v1/rgb_only/`
- `rgb_predicted_depth_geometry`
  - `recon/points_depth_cam0/`
  - `tracks_rgb_predicted_depth_geometry/tracklets.json`
  - `embeddings_rgb_predicted_depth_geometry/`
  - `mvp-demo/output/evals/iciscae_node01_uav_v1/rgb_predicted_depth_geometry/`
- `rgb_fused_geometry`
  - `recon/points_fused/`
  - `tracks_rgb_fused_geometry/tracklets.json`
  - `embeddings_rgb_fused_geometry/`
  - `mvp-demo/output/evals/iciscae_node01_uav_v1/rgb_fused_geometry/`
- `gt_upper_bound`
  - `recon/points_fused_gt/`
  - `tracks_gt_upper_bound/tracklets.json`
  - `embeddings_gt_upper_bound/`
  - `mvp-demo/output/evals/iciscae_node01_uav_v1/gt_upper_bound/`

## 3. 验收检查

### 3.1 scene 级检查

`6` 个正式 scene 当前全部满足：

- `identity_id` 仍与 manifest 一致；
- `min_valid_timestamps >= 5`；
- `tracks_rgb_predicted_depth_geometry/tracklets.json`、`tracks_rgb_fused_geometry/tracklets.json`、`tracks_gt_upper_bound/tracklets.json` 均已生成；
- 每条 branch-specific tracklet 都保留 `90` 个 `timestamp_stems` 与 `90` 个非空 `fused_points_paths`；
- `recon/points_depth_cam0/*.npy`、`recon/points_fused/*.npy`、`recon/points_fused_gt/*.npy` 各 `90` 个。

### 3.2 点云规模检查

| points_subdir | 平均点数范围 | 备注 |
| --- | --- | --- |
| `recon/points_depth_cam0` | 约 `6996` 到 `7290` | 单相机弱几何 |
| `recon/points_fused` | 约 `19320` 到 `19742` | 三相机预测几何 |
| `recon/points_fused_gt` | 约 `2155` 到 `3437` | GT mask/depth 的上界分析线 |

## 4. 结果总表

| branch | mAP | recall@1 | recall@5 | recall@10 | 结论 |
| --- | --- | --- | --- | --- | --- |
| `rgb_only` | `0.8333` | `0.6667` | `1.0000` | `1.0000` | 当前最好主基线 |
| `rgb_predicted_depth_geometry` | `0.6389` | `0.3333` | `1.0000` | `1.0000` | 修复 `j10_line`，但整体退化 |
| `rgb_fused_geometry` | `0.6667` | `0.3333` | `1.0000` | `1.0000` | 比弱几何略好，但仍低于 `rgb_only` |
| `gt_upper_bound` | `0.6111` | `0.3333` | `1.0000` | `1.0000` | 说明瓶颈不只在 predicted 输入 |

## 5. 已知关键样例对照

### 5.1 `j10_line` 被修复

| branch | top1 | top1_relevant | AP |
| --- | --- | --- | --- |
| `rgb_only` | `node01_mj_node01_uav1_line_nodes_a` | `NO` | `0.5000` |
| `rgb_predicted_depth_geometry` | `node01_mj_node01_j10_circle_xz_b` | `YES` | `1.0000` |
| `rgb_fused_geometry` | `node01_mj_node01_j10_circle_xz_b` | `YES` | `1.0000` |
| `gt_upper_bound` | `node01_mj_node01_uav1_line_nodes_a` | `NO` | `0.5000` |

判断：geometry 对 `j10` 的个别 case 有帮助，但这种帮助并不稳定。

### 5.2 `uav1_circle` 仍未修复

| branch | top1 | top1_relevant | AP |
| --- | --- | --- | --- |
| `rgb_only` | `node01_mj_node01_dji_mavic_circle_xz_b` | `NO` | `0.5000` |
| `rgb_predicted_depth_geometry` | `node01_mj_node01_dji_mavic_circle_xz_b` | `NO` | `0.5000` |
| `rgb_fused_geometry` | `node01_mj_node01_dji_mavic_circle_xz_b` | `NO` | `0.5000` |
| `gt_upper_bound` | `node01_mj_node01_j10_circle_xz_b` | `NO` | `0.5000` |

判断：当前 geometry 分支没有修复 `uav1 / dji_mavic` 的近形态混淆，GT 线只是把错误目标换成了另一种错误。

### 5.3 几何分支带来的新退化

- `uav1_line` 在 `rgb_only` 中原本 `top1` 正确，但在两条 geometry 分支里都被 `dji_mavic_line` 抢走。
- `dji_mavic_line` 与 `dji_mavic_circle` 在 `rgb_only` 中原本都正确，但在 geometry 分支里被 `uav1` 抢走。
- 因此，当前结论不能写成“geometry 普遍提升了检索性能”。

## 6. 结果解释

- 当前的 geometry 特征采用 `open3d_fpfh` 全局描述子，再与 `CLIP` embedding 直接拼接。
- 从结果看，问题更可能出在表示与融合本身，而不只是输入感知噪声：
  - 预测几何与融合几何都没有超过 `rgb_only`；
  - `GT upper-bound` 也只在 `dji_mavic` 上局部恢复，没有带来整体提升。
- 一个更谨慎的论文表述应当是：
  - 当前三相机节点 benchmark 已经足够证明 `RGB-only` 可作为稳定基线；
  - geometry branch 在当前实现下尚未形成可靠增益；
  - 这恰好为后续毕业论文继续研究 `descriptor / fusion / cross-node` 提供了问题入口。

## 7. 下周建议

1. 直接以 `rgb_only` 作为小论文主结果线，把 geometry 两条分支写成“未形成稳定增益的对照实验”。
2. 在正文里保留 `j10_line` 的修复例子，但不能把它夸大成整体趋势。
3. 若继续优化 geometry，优先检查：
   - `open3d_fpfh` 是否适合当前小型 aircraft / UAV 目标；
   - `CLIP` 与 geometry 的拼接尺度是否失衡；
   - 点云归一化、时间聚合和采样策略是否抹平了细粒度差异。
4. 当前不建议在小论文阶段继续扩 scene、扩身份或切到 `cross-node`；先把文稿收口并把现有失败模式讲清楚。
