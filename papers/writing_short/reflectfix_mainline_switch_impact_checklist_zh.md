# `reflectfix` 切主线影响清单

## 1. 用途

这份清单服务于一个具体动作：

> 如果后续决定把当前“默认主线”从 `v3_clean` 切到 `v3_clean_reflectfix`，哪些材料必须改，哪些建议改，哪些暂时不要动？

当前默认前提是：

- `v3_clean` 先保留为历史主线
- `v3_clean_reflectfix` 升格为新的候选官方主线
- **不覆盖旧 `v3_clean` benchmark_id**

## 2. 必改

这些文件一旦正式切主线，就必须同步更新，否则仓库会出现“当前激活 benchmark”与“论文主表”冲突。

### 2.1 研究主线与 active benchmark

- `research/plans/ACTIVE_PLAN.md`
  - 要把“当前激活 benchmark = `v3_clean`”改成 reflectfix 版本
  - `next_action` 也要一起改，避免继续引用旧主表路径
- `research/plans/tri_camera_node_3d_aware_reid/master_plan_zh.md`
  - 要明确新激活 benchmark 是 reflectfix
  - 原 `v3_clean` 要改写成历史主线，而不是仍写成当前主线

### 2.2 默认运行入口

- `research/handoffs/tri_camera_node_engineering_handoff_zh.md`
  - 默认 branch eval / summary / failure-analysis 命令要改指向 reflectfix
- `mvp-demo/README.md`
  - 当前推荐 benchmark 入口必须改成 reflectfix manifest
  - 默认 summary 输出路径也要改成 reflectfix benchmark_id

### 2.3 写作主稿口径

- `papers/writing_short/draft_zh.md`
  - 主结果段要从 `v3_clean` 切到 reflectfix 结果
  - failure analysis 段要重新审，确认关键 case 与结论仍对应
  - limitation 段与 conclusion 段也要复核
- `papers/writing_short/figures_manifest.md`
  - 主表与案例图的来源要切到 reflectfix 或明确保留旧图的理由

## 3. 建议改

这些文件不改不一定会马上出错，但会造成“仓库仍默认旧主线”的阅读错觉。

### 3.1 研究评审与状态板

- `research/reviews/current_progress_board_verified_2026_03_28_zh.md`
- `research/reviews/iciscae_v3_clean_failure_analysis_2026_03_23_zh.md`
- `research/reviews/iciscae_paper_draft_v3_clean_2026_03_23_zh.md`

建议做法不是硬覆盖旧内容，而是：

- 新增一版 reflectfix 主线 review / progress board
- 或在文件开头明确“此文件对应旧主线 `v3_clean`”

### 3.2 写作工作区索引

- `papers/writing_short/README.md`
  - 要明确当前工作区的“当前权威依据”是否改成 reflectfix
- `papers/writing_short/worklog.md`
  - 要记录“何时决定把 reflectfix 升格为候选主线”

### 3.3 附录与补充材料

- `papers/writing_short/appendix_reflectfix_robustness_zh.md`
  - 如果 reflectfix 成为主线，这个文件不能再继续被写成“附录鲁棒性证据”
  - 需要改成：
    - 历史迁移说明
    - 或 `v3_clean` vs reflectfix 的主线切换说明
- `papers/writing_short/v3_clean_vs_v3_clean_reflectfix_comparison_zh.md`
  - 仍然有用，但角色会从“附录支持材料”变成“主线切换说明文档”

## 4. 暂不改

这些内容在“只是升格 reflectfix 为候选主线”阶段不建议动。

### 4.1 历史结果与历史输出目录

- `mvp-demo/output/evals/iciscae_node01_uav_v3_clean/`
- `research/plans/tri_camera_node_3d_aware_reid/benchmarks/iciscae_node01_uav_v3_clean.json`

建议保留，作为历史主线对照，不要删除，不要偷偷覆盖。

### 4.2 旧结果的 benchmark_id

- 不建议把 reflectfix 重命名成旧 `iciscae_node01_uav_v3_clean`
- 否则历史追溯会变得很混乱

### 4.3 与切主线无关的增强路线

- `3DGS` 辅助 PoC
- `late-fusion` 补充分析
- `cross-node` / 真实节点迁移

这些都不是这次切主线的必要前置条件，不要和主线切换混做一件事。

## 5. 推荐迁移顺序

如果你后续真要切，建议按下面顺序做：

1. 先更新研究层：
   - `ACTIVE_PLAN`
   - `master_plan_zh`
2. 再更新工程默认入口：
   - handoff
   - `mvp-demo/README.md`
3. 再更新写作主稿与图表引用：
   - `draft_zh.md`
   - `figures_manifest.md`
4. 最后补工作区索引与说明：
   - `papers/writing_short/README.md`
   - `papers/writing_short/worklog.md`
   - reflectfix 相关附录角色调整

## 6. 最小验收标准

如果后续正式切主线，至少要满足：

- `ACTIVE_PLAN` 与 `master_plan_zh` 都明确写 reflectfix 为当前激活 benchmark
- 默认运行命令都指向 reflectfix manifest
- `draft_zh.md` 的主表数值不再引用旧 `v3_clean`
- 旧 `v3_clean` 被显式降级为历史主线，而不是无声消失
- 仓库内不再出现“当前主线是 `v3_clean`”与“当前主表是 reflectfix”并存的冲突写法
