# ICISCAE `v3_clean` 失败分析（2026-03-23）

## 1. 结论速览

- 当前权威 benchmark 已切到 `iciscae_node01_uav_v3_clean`，scene 只使用无 humanoid 的 clean 场景。
- `v3_clean` 的四条结果线已经全部落盘：
  - `rgb_only`: `mAP = 0.5750`, `recall@1 = 0.3333`
  - `rgb_predicted_depth_geometry`: `0.4222`, `0.1667`
  - `rgb_fused_geometry`: `0.4833`, `0.1667`
  - `gt_upper_bound`: `0.8333`, `0.6667`
- clean 场景移除了 humanoid 干扰后，预测 geometry 分支依旧没有超过 `rgb_only`；但 `GT upper-bound` 明显更好，说明当前主瓶颈更偏向 `SAM2/depth` 感知质量，而不是 humanoid 遮挡本身。

## 2. 当前最重要的诊断

统一逐 query 差分表已生成到：

- `mvp-demo/output/evals/iciscae_node01_uav_v3_clean/query_failure_analysis.json`
- `mvp-demo/output/evals/iciscae_node01_uav_v3_clean/query_failure_analysis.md`

当前最值得保留的三类案例如下：

### 2.1 `GT` 能恢复、预测分支不能恢复

- `mj_node01_j10_clean_circle_xz_b`
- `mj_node01_uav1_clean_line_nodes_a`
- `mj_node01_uav1_clean_circle_xz_b`

这三条 query 在 `rgb_only` 与两条预测 geometry 分支里都没有得到正确 `top1`，但在 `gt_upper_bound` 中都恢复为正确匹配。说明：

- geometry 信息并不是无用；
- 一旦 `mask/depth` 质量足够好，当前 pipeline 能把同身份 track 拉近；
- 因而 clean 主线下应优先解释预测感知误差，而不是继续把锅归给 humanoid 遮挡。

### 2.2 clean 后仍持续存在的混淆

- `j10_circle -> uav1_circle`
- `uav1_line -> su34_line`
- `uav1_circle -> j10_circle`
- `j10_line -> su34_line`

这些错误说明：即使移除了 humanoid 背景干扰，`j10/uav1/su34` 之间的细粒度外形与视角变化仍然足以让预测 geometry 失败。

### 2.3 geometry 额外带来的退化

- `mj_node01_su34_clean_circle_xz_b` 在 `rgb_only` 中 `top1` 正确，但在 `rgb_predicted_depth_geometry` 中退化为错配 `j10_circle`。
- `mj_node01_su34_clean_line_nodes_a` 在 `rgb_only` 中 `top1` 正确，但在 `rgb_fused_geometry` 中退化为错配 `uav1_line`。

这说明当前的 `open3d_fpfh + CLIP 直接拼接` 还不稳定，预测 geometry 不是“加上就更好”。

## 3. 论文应落下的解释

建议把 clean 主线的结果解释固定为三句话：

1. clean 场景排除了 humanoid 干扰，但预测 geometry 分支仍未超过 `rgb_only`。
2. `GT upper-bound` 明显优于预测分支，说明当前差距主要来自 `SAM2 mask / predicted depth` 的感知质量。
3. `GT upper-bound` 仍未做到全 query 完美，说明后续毕业论文仍需要继续研究 geometry descriptor 与 fusion，而不是只靠更干净的输入。

## 4. 立即下一步

- 把 `v3_clean` 指标与上述三类案例写回论文主结果、失败分析与结论段。
- 图表优先保留：
  - `GT` 能恢复但预测分支不能恢复的 1 组 query；
  - geometry 额外退化的 1 组 query；
  - `rgb_only / predicted / fused / GT` 的统一总表。
- 若继续做小范围方法复核，先查 `SAM2/depth` 在 `j10_circle`、`uav1_line`、`uav1_circle` 上的输入质量，再讨论 descriptor 替换。
