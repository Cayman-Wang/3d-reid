# `v3_clean_reflectfix` 全量实验核验记录（2026-03-29）

## 1. 目的

本记录用于回答一个工程问题：

> 本地现有 `v3_clean_reflectfix` 结果，是否已经满足 “6 个正式 scene × 4 条 branch 全量完成，且可直接复用”？

当前结论是：

> **满足。当前本地 `v3_clean_reflectfix` 已通过全量核验，不需要覆盖式完整重跑。**

## 2. 核验对象

- manifest：
  - `research/plans/tri_camera_node_3d_aware_reid/benchmarks/iciscae_node01_uav_v3_clean_reflectfix.json`
- 输出根目录：
  - `mvp-demo/output/evals/iciscae_node01_uav_v3_clean_reflectfix/`
- 四条 branch：
  - `rgb_only`
  - `rgb_predicted_depth_geometry`
  - `rgb_fused_geometry`
  - `gt_upper_bound`

## 3. 核验结果

### 3.1 manifest 与 scene 集合

- reflectfix manifest 存在。
- manifest 中列出了 `6` 个正式 reflectfix scenes：
  - `mj_node01_j10_clean_reflectfix_line_nodes_a`
  - `mj_node01_j10_clean_reflectfix_circle_xz_b`
  - `mj_node01_uav1_clean_reflectfix_line_nodes_a`
  - `mj_node01_uav1_clean_reflectfix_circle_xz_b`
  - `mj_node01_su34_clean_reflectfix_line_nodes_a`
  - `mj_node01_su34_clean_reflectfix_circle_xz_b`

### 3.2 branch 产物完整性

四条 branch 都满足：

- `run_meta.json` 存在
- `all_queries_vs_all_scenes.json` 存在
- `6` 个 per-scene JSON 存在
- `benchmark_id` 一致为 `iciscae_node01_uav_v3_clean_reflectfix`
- `scene_ids` 与 reflectfix manifest 完全一致
- `manifest` 都指回同一个 reflectfix manifest

### 3.3 顶层汇总产物

以下文件均存在：

- `branch_comparison_summary.md`
- `branch_comparison_summary.json`
- `query_failure_analysis.md`
- `query_failure_analysis.json`

### 3.4 时间一致性

当前时间顺序是合理的：

- `run_iciscae_branch_eval.py`、`summarize_iciscae_branch_comparison.py`、`analyze_iciscae_failure_modes.py` 的时间戳早于 reflectfix manifest
- reflectfix manifest 早于四条 branch 的 `run_meta.json`
- 四条 branch 的 `run_meta.json` 早于顶层 `branch_comparison_summary.md` 与 `query_failure_analysis.md`

因此，当前没有发现“产物早于 manifest / 脚本版本，导致口径失效”的问题。

## 4. 判定

本次核验没有触发以下任一重跑条件：

- 缺 branch
- 缺 per-scene JSON
- `run_meta` 与 manifest 不匹配
- summary 或 failure analysis 缺失
- 产物版本明显落后于当前应使用的 manifest / 脚本

因此，本地现有 `v3_clean_reflectfix` **可直接作为全量鲁棒性变体结果复用**。

## 5. 对当前论文与汇报的影响

- `v3_clean` 仍是正式主表 benchmark
- `v3_clean_reflectfix` 可被视为**已核验通过的 6-scene 全量鲁棒性变体**
- 当前不需要为了“补齐 reflectfix 全量 run”再做一次覆盖式完整重跑
- 后续写作可直接复用现有 reflectfix 的：
  - branch comparison summary
  - query failure analysis
  - paired `SAM2` nuisance 证据

## 6. 当前建议动作

1. 继续沿用现有 `v3_clean_reflectfix` 结果，不重跑。
2. 在写作与导师沟通中明确：
   - 地面反光会显著伤害 `SAM2` 分割
   - reflectfix 已有全量 run 支撑
   - 但 reflectfix 尚未形成稳定的 end-to-end retrieval 提升
3. 只有当你后续主动修改 reflectfix manifest、branch 配置或下游评测脚本时，才重新考虑覆盖式重跑。
