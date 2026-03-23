# 当前进展看板（`v3_clean`, 2026-03-23）

## 1. 结论速览

| 检查项 | 状态 | 结论 |
| --- | --- | --- |
| clean `6 scene` 采集 | GREEN | `j10 / uav1 / su34` 的两条正式轨迹都已落盘 |
| 三条正式结果线 | GREEN | `rgb_only / rgb_predicted_depth_geometry / rgb_fused_geometry` 已全部完成 |
| `GT upper-bound` | GREEN | `masks_gt + depth_gt + fused geometry` 已独立落盘 |
| branch comparison | GREEN | `branch_comparison_summary` 与 `query_failure_analysis` 已生成 |
| 论文收口 | YELLOW | 失败分析与 clean 版草稿已补齐，待整理图表和导师汇报口径 |

一句话判断：

`当前 clean 主线已经完成结果齐套，下一步不是补跑 benchmark，而是把“为什么 clean 后预测 geometry 仍未优于 rgb_only”讲清楚。`

## 2. 当前正式结果

| branch | mAP | recall@1 | recall@5 | recall@10 |
| --- | --- | --- | --- | --- |
| `rgb_only` | `0.5750` | `0.3333` | `1.0000` | `1.0000` |
| `rgb_predicted_depth_geometry` | `0.4222` | `0.1667` | `1.0000` | `1.0000` |
| `rgb_fused_geometry` | `0.4833` | `0.1667` | `1.0000` | `1.0000` |
| `gt_upper_bound` | `0.8333` | `0.6667` | `1.0000` | `1.0000` |

## 3. 当前最重要的实验判断

- clean 场景移除了 humanoid 后，预测 geometry 两条分支仍未超过 `rgb_only`。
- `GT upper-bound` 明显优于预测分支，说明当前最大差距主要在 `SAM2/depth` 感知质量，而不是遮挡背景。
- `GT upper-bound` 仍未完全解决 `j10_line` 与 `su34_line` 的混淆，说明 descriptor/fusion 也仍有改进空间。

## 4. 当前必须能回答的问题

- clean 后问题有没有自动消失？
  - 没有。`rgb_only` 仍优于两条预测 geometry 分支。
- geometry 是否完全无效？
  - 不是。`GT upper-bound` 证明只要输入质量够好，geometry 分支可以显著变好。
- 现在下一步该优先改什么？
  - 先做 `SAM2/depth` 误差归因，再决定是否继续改 geometry descriptor/fusion。

## 5. 立即下一步

1. 把 `v3_clean` 总表和 failure cases 写回论文正文。
2. 补一版导师汇报图表：总表 + 2 个关键 query 案例。
3. 若继续实验，只做小范围输入质量诊断，不扩 scene、不切 `cross-node`。
