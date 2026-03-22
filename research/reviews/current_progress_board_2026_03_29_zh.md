# 当前进展看板（2026-03-29）

## 1. 结论速览

| 检查项 | 状态 | 结论 |
| --- | --- | --- |
| `M4` 几何两条分支 | GREEN | `RGB + predicted-depth geometry` 与 `RGB + fused geometry` 已全部落盘 |
| `GT upper-bound` | GREEN | `masks_gt + depth_gt + fused geometry` 已完成并独立落盘 |
| 分支隔离与可复现性 | GREEN | 新增 branch-specific `tracklets` / `embeddings` / eval 目录，不覆盖现有 `rgb_only` |
| 小论文草稿 | GREEN | 已形成可交导师的文稿草稿与周收口文档 |
| geometry 是否优于 `RGB-only` | RED | 当前两条 geometry 分支都未超过 `RGB-only` 主基线 |

一句话判断：

`2026-03-23` 到 `2026-03-29` 这一周的计划性交付已经全部完成，但实验结论不是“geometry 修复了混淆”，而是“当前 geometry 实现没有超过 RGB-only，需要转向误差归因与方法重审”。`

## 2. 当前里程碑对照

- 当前主线已从 `M1` 正式 scene 采集切换到 `M5 ICISCAE 收口`。
- 当前正式结果矩阵已经齐全：
  - `rgb_only`
  - `rgb_predicted_depth_geometry`
  - `rgb_fused_geometry`
  - `gt_upper_bound`
- 当前统一对照文件已经生成：
  - `mvp-demo/output/evals/iciscae_node01_uav_v1/branch_comparison_summary.json`
  - `mvp-demo/output/evals/iciscae_node01_uav_v1/branch_comparison_summary.md`

## 3. 本周新增能力与产物

- 已为 `recon_fuse_depth_points.py` 增加 `--out_subdir`，支持弱几何、强几何和 GT 点云并行落盘。
- 已为 `eval_node_track_retrieval.py` 增加 `--embeddings_subdir`，支持 branch-specific 检索而不覆盖 `rgb_only`。
- 已新增：
  - `mvp-demo/scripts/run_iciscae_branch_eval.py`
  - `mvp-demo/scripts/summarize_iciscae_branch_comparison.py`
- `6` 个正式 scene 当前都具备：
  - `recon/points_depth_cam0/*.npy` 共 `90` 个
  - `recon/points_fused/*.npy` 共 `90` 个
  - `recon/points_fused_gt/*.npy` 共 `90` 个
  - branch-specific `tracklets.json` 与 `tracks.npy`

## 4. 当前正式结果

| branch | mAP | recall@1 | recall@5 | recall@10 |
| --- | --- | --- | --- | --- |
| `rgb_only` | `0.8333` | `0.6667` | `1.0000` | `1.0000` |
| `rgb_predicted_depth_geometry` | `0.6389` | `0.3333` | `1.0000` | `1.0000` |
| `rgb_fused_geometry` | `0.6667` | `0.3333` | `1.0000` | `1.0000` |
| `gt_upper_bound` | `0.6111` | `0.3333` | `1.0000` | `1.0000` |

## 5. 当前最重要的实验判断

- `j10_line` 的 top1 混淆在两条 geometry 分支里都被修复，说明几何信息不是完全无效。
- 但 `uav1 / dji_mavic` 的混淆没有被修复，反而在多条 query 上比 `rgb_only` 更差。
- `GT upper-bound` 也没有整体超过 `rgb_only`，说明当前瓶颈不只是 predicted `mask/depth` 的感知误差，更可能来自：
  - 点云描述子选择
  - RGB 与 geometry 的简单拼接方式
  - 当前 point normalization / pooling 对 aircraft-like 细粒度差异不敏感

## 6. 建议的下一步

1. 先用 `research/reviews/iciscae_week_closure_2026_03_29_zh.md` 和 `research/reviews/iciscae_paper_draft_2026_03_29_zh.md` 收口文稿。
2. 如果继续做方法实验，优先重审 `descriptor / fusion`，而不是扩 scene 或提前切到 `cross-node`。
3. 若继续保留 geometry 主线，下一轮应重点检查 `uav1 / dji_mavic` 的点云质量、归一化和融合权重。
