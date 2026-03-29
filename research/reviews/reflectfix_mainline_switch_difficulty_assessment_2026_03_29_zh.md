# `reflectfix` 升格为主线的难度评估（2026-03-29）

## 1. 目的

本评估只回答一个问题：

> 如果后续决定把当前主线从 `v3_clean` 切换到 `v3_clean_reflectfix`，改动难度到底有多大？

当前结论是：

> **只改工程默认入口，难度中低；连论文主线、主表口径与 active plan 一起切，难度中高。**

## 2. 当前事实基础

- 当前激活 benchmark 仍是 `v3_clean`，并已写入：
  - `research/plans/ACTIVE_PLAN.md`
  - `research/plans/tri_camera_node_3d_aware_reid/master_plan_zh.md`
  - `research/handoffs/tri_camera_node_engineering_handoff_zh.md`
  - `mvp-demo/README.md`
- `v3_clean_reflectfix` 已不是待接入候选，而是一套已完成核验的独立 benchmark：
  - 有独立 manifest
  - 有 `6` 个正式 reflectfix scenes
  - 有 `4` 条 branch 全量结果
  - 有 `branch_comparison_summary` 与 `query_failure_analysis`
  - 已通过全量 run 核验

因此，当前难点不在“能不能跑”，而在“要不要重写已经冻结下来的主线口径”。

## 3. 难度为什么不是很低

### 3.1 工程入口并不难改

如果只把默认运行入口切到 reflectfix，改动相对集中：

- 默认 manifest 指向
- README 默认命令
- handoff 默认命令
- 默认 benchmark_id 相关入口

这类改动的本质是“把当前默认入口从 `v3_clean` 改为 `v3_clean_reflectfix`”，而不是开发新链路，所以难度是 **中低**。

### 3.2 真正重的是研究口径迁移

如果把 reflectfix 升格为新的官方主线，就不只是换命令，而是要同步回答以下问题：

- 当前正式主表是否改成 reflectfix 总表
- `v3_clean` 是否降级为历史结果
- “clean scene” 的含义是否改成“去 humanoid + 去反光”
- 现有 failure analysis 图、结论段、局限段是否仍能直接沿用
- 现有 review / progress board / handoff 中所有“当前主线 = v3_clean”的表述是否要改

这些问题都不是脚本问题，而是**冻结研究口径**的问题，所以一旦连论文与 active plan 一起切，难度会上升到 **中高**。

## 4. 推荐切法

当前推荐策略不是覆盖旧主线，而是：

- `v3_clean` 保留为历史正式主线 / 旧主表
- `v3_clean_reflectfix` 升格为新的**候选官方主线**
- 先完成影响清单与口径迁移，再决定是否真正切主稿

不建议采用下面这种做法：

- 直接把 reflectfix 覆盖成旧 `v3_clean` 身份

原因是：

- 历史结果会变得不可追溯
- 旧 review / progress board / handoff 会和新结果混线
- 很难解释“为什么之前的正式主表和现在的正式主表不是同一批 scene”

## 5. 难度评级

| 改动范围 | 难度 | 说明 |
| --- | --- | --- |
| 仅工程默认入口切换 | `中低` | 主要是 manifest、README、handoff 默认命令改指向 |
| 工程 + 文稿 + review/handoff 一起切 | `中高` | 主要成本在研究口径与主表叙事迁移 |
| 工程 + 文稿 + 历史口径完全重命名替代旧主线 | `高` | 风险最高，容易损坏历史可追溯性 |

## 6. 推荐的后续动作

如果你后续真要推动 reflectfix 升格主线，建议按这个顺序推进：

1. 先做一份“切主线影响清单”
2. 明确哪些文件属于：
   - 必改
   - 建议改
   - 暂不改
3. 明确新旧主线并存期间的口径：
   - `v3_clean` 是历史主线
   - `v3_clean_reflectfix` 是候选新主线
4. 最后才决定是否把论文主表正式切到 reflectfix

## 7. 当前建议结论

截至当前仓库状态，最稳妥的说法是：

- reflectfix **已经足够成熟，可以作为候选官方主线来评估**
- 但它**还没有低成本到可以“随手替换掉旧主线”**
- 若只做工程默认切换，难度不大
- 若连论文主线、主表、review 与 active plan 一起切，难度明确属于 **中高**
