# 26-03-23 ICISCAE 小论文本周执行日程

> 存档说明：本文档用于归档 `2026-03-23` 到 `2026-03-29` 这一周的执行版日程，只服务于本周 `ICISCAE` 小论文收口，不替代 `research/` 主线计划。唯一口径以 `research/plans/ACTIVE_PLAN.md`、`research/plans/tri_camera_node_3d_aware_reid/master_plan_zh.md`、`research/plans/tri_camera_node_3d_aware_reid/benchmarks/iciscae_node01_uav_v1.json` 为准。

## 1. 本周成功定义

本周结束时，以下 4 条必须成立：

- `RGB-only`、`RGB + predicted-depth geometry`、`RGB + fused geometry` 三组正式结果已经全部落盘。
- `GT upper-bound` 已独立落盘到 `mvp-demo/output/evals/iciscae_node01_uav_v1/gt_upper_bound/`，且不混入正式主结果命名。
- 已形成一份统一对照文件，能够直接回答“geometry 是否修复了 `RGB-only` 的混淆”。
- 小论文正文最少完成：任务定义、benchmark protocol、实验设置、related work、主结果表、失败分析、结论段。

本周固定说明：

- 公共接口或类型变更：只允许脚本级非破坏性参数扩展，不做数据契约改动。
- 本周不再新增正式 scene，不改 benchmark 身份集合。
- 本周不做 `cross-node`。
- 本周不训练新的端到端 encoder。
- 本周不回到 `YOLO + 3DGS` demo 主链。

## 2. 本周目标

本周 4 个硬目标：

- 冻结两条 geometry 分支的实现口径，并把 branch-specific 输出目录隔离开，不覆盖现有 `rgb_only`。
- 完成 `rgb_predicted_depth_geometry`、`rgb_fused_geometry`、`gt_upper_bound` 的批量评测与全量 summary。
- 形成一份 branch comparison 对照表，明确每条 query 在四条结果线上的 top1 变化。
- 完成一版可直接交导师的小论文草稿与周收口文档。

本周 4 个明确不做：

- 不扩充 `6-scene` benchmark。
- 不更换 `CLIP` backbone。
- 不复现 `cross-modal-distillation-reidentification` 或 `point-cloud-reid` 的完整训练链。
- 不把 `GT upper-bound` 写成正式主结果。

## 3. 每日安排

### 2026-03-23

目标：冻结 geometry 分支口径并排掉工程覆盖风险。

任务：

- 确认本周分支矩阵：
  - `rgb_predicted_depth_geometry`
  - `rgb_fused_geometry`
  - `gt_upper_bound`
- 设计 branch-specific 目录布局，确保不覆盖现有：
  - `tracks/`
  - `embeddings/`
  - `mvp-demo/output/evals/iciscae_node01_uav_v1/rgb_only/`
- 给 `recon_fuse_depth_points.py` 增加 `--out_subdir` 的最小改造方案。
- 给 `eval_node_track_retrieval.py` 增加 `--embeddings_subdir` 的最小改造方案。
- 用 `1` 个正式 scene 先做弱几何 `cam0` dry-run 与三相机融合 dry-run。

输出物：

- 一份本周 geometry 分支执行口径。
- 一份 branch-specific 目录布局。
- 一份最小脚本改造清单。

收尾检查：

- `rgb_only` 目录不被覆盖。
- `points_depth_cam0`、`points_fused`、`points_fused_gt` 三类点云路径已被明确区分。
- `open3d` 在 `mvp_demo` 环境下可用。

执行记录（按本周收口结果回填，`2026-03-23` 任务已完成）：

- 已确认 `mvp_demo` 环境中：
  - `open3d 0.19.0`
  - `torch 2.10.0+cu128`
  - `open_clip 3.3.0`
  均可正常导入。
- 已冻结本周三条新增分析线：
  - `rgb_predicted_depth_geometry = cam0 predicted depth + mask + open3d_fpfh`
  - `rgb_fused_geometry = 3-camera predicted depth + mask fusion + open3d_fpfh`
  - `gt_upper_bound = masks_gt + depth_gt + fused geometry`
- 已确定本周必须新增两个非破坏性参数：
  - `recon_fuse_depth_points.py --out_subdir`
  - `eval_node_track_retrieval.py --embeddings_subdir`
- 已在单 scene dry-run 中确认：
  - 点云落盘链路可通
  - `tracklets -> embeddings -> retrieval` 链路可通
  - 真正需要解决的问题是 branch-specific 输出隔离，而不是主链不可运行

本周固定批量执行命令口径：

```powershell
conda run -n mvp_demo python scripts/run_iciscae_branch_eval.py `
  --branch rgb_predicted_depth_geometry
```

```powershell
conda run -n mvp_demo python scripts/run_iciscae_branch_eval.py `
  --branch rgb_fused_geometry
```

```powershell
conda run -n mvp_demo python scripts/run_iciscae_branch_eval.py `
  --branch gt_upper_bound
```

```powershell
conda run -n mvp_demo python scripts/summarize_iciscae_branch_comparison.py
```

### 2026-03-24

目标：完成 `rgb_predicted_depth_geometry` 全量跑通。

任务：

- 对 `6` 个正式 scene 统一生成 `recon/points_depth_cam0/`。
- 生成：
  - `tracks_rgb_predicted_depth_geometry/tracklets.json`
  - `embeddings_rgb_predicted_depth_geometry/tracks.npy`
  - `mvp-demo/output/evals/iciscae_node01_uav_v1/rgb_predicted_depth_geometry/*.json`
- 汇总全量 summary。
- 重点检查两个已知错误样例：
  - `mj_node01_j10_line_nodes_a`
  - `mj_node01_uav1_circle_xz_b`

输出物：

- `rgb_predicted_depth_geometry` 全量结果目录。
- 一份弱几何分支首轮结果判断。

收尾检查：

- `6` 个 scene 都存在 `recon/points_depth_cam0/*.npy`
- 每个 scene 都有 branch-specific `tracklets` 和 `embeddings`
- `all_queries_vs_all_scenes.json` 已生成

执行记录（`2026-03-24` 任务已完成）：

- `6` 个正式 scene 已全部生成 `recon/points_depth_cam0/*.npy`，每个 scene 各 `90` 个点云文件。
- `rgb_predicted_depth_geometry` 已完整落盘到：
  - `mvp-demo/output/evals/iciscae_node01_uav_v1/rgb_predicted_depth_geometry/`
- 当前全量 summary 为：
  - `mAP = 0.6389`
  - `recall@1 = 0.3333`
  - `recall@5 = 1.0000`
  - `recall@10 = 1.0000`
- 当前最关键现象：
  - `mj_node01_j10_line_nodes_a` 已被修复为正确 `top1`
  - `uav1 / dji_mavic` 的相互混淆没有被修复

### 2026-03-25

目标：冻结弱几何分支判断，并明确是否触发 fallback。

任务：

- 汇总 `rgb_predicted_depth_geometry` 与 `rgb_only` 的逐 query 对照。
- 判断 `open3d_fpfh` 是否稳定。
- 若不稳定，才允许退回 `radial_hist`；若稳定，则本周不再切换 descriptor。
- 输出一版“弱几何是否值得继续”的中间判断。

输出物：

- 一张 `rgb_only vs rgb_predicted_depth_geometry` 逐 query 对照表。
- 一条“是否触发 fallback”的明确结论。

收尾检查：

- 不允许在本周同时混用 `open3d_fpfh` 和 `radial_hist` 产生正式结果。
- 若不触发 fallback，需要在文稿中明确写死当前实现。

执行记录（`2026-03-25` 任务已完成）：

- 本周没有触发 `radial_hist` fallback；`open3d_fpfh` 可稳定完成全量计算。
- 弱几何分支的结论已冻结为：
  - 个别 query 可被修复
  - 但整体上不优于 `rgb_only`
- 因此，本周后半段继续沿 `open3d_fpfh` 保持口径一致，不再切换 descriptor。

### 2026-03-26

目标：完成 `rgb_fused_geometry` 全量跑通。

任务：

- 对 `6` 个正式 scene 统一生成 `recon/points_fused/`。
- 生成：
  - `tracks_rgb_fused_geometry/tracklets.json`
  - `embeddings_rgb_fused_geometry/tracks.npy`
  - `mvp-demo/output/evals/iciscae_node01_uav_v1/rgb_fused_geometry/*.json`
- 把结果直接和：
  - `rgb_only`
  - `rgb_predicted_depth_geometry`
  做对比。

输出物：

- `rgb_fused_geometry` 全量结果目录。
- 一版强几何分支相对弱几何和 `rgb_only` 的判断。

收尾检查：

- `6` 个 scene 都存在 `recon/points_fused/*.npy`
- 强几何分支的 `tracklets / embeddings / eval` 都已完整生成

执行记录（`2026-03-26` 任务已完成）：

- `6` 个正式 scene 已全部生成 `recon/points_fused/*.npy`，每个 scene 各 `90` 个点云文件。
- `rgb_fused_geometry` 已完整落盘到：
  - `mvp-demo/output/evals/iciscae_node01_uav_v1/rgb_fused_geometry/`
- 当前全量 summary 为：
  - `mAP = 0.6667`
  - `recall@1 = 0.3333`
  - `recall@5 = 1.0000`
  - `recall@10 = 1.0000`
- 结论：
  - 强几何略好于弱几何
  - 但仍未超过 `rgb_only`

### 2026-03-27

目标：完成 `GT upper-bound` 与统一结果对照。

任务：

- 用 `masks_gt + depth_gt` 重新生成 `recon/points_fused_gt/`。
- 生成：
  - `tracks_gt_upper_bound/tracklets.json`
  - `embeddings_gt_upper_bound/tracks.npy`
  - `mvp-demo/output/evals/iciscae_node01_uav_v1/gt_upper_bound/*.json`
- 输出至少包含四行的统一结果表：
  - `rgb_only`
  - `rgb_predicted_depth_geometry`
  - `rgb_fused_geometry`
  - `gt_upper_bound`

输出物：

- `gt_upper_bound` 全量结果目录。
- 一张统一对照表。

收尾检查：

- `GT upper-bound` 单独落盘，不覆盖正式主结果。
- 能明确回答“问题主要来自 perception 还是表示本身”。

执行记录（`2026-03-27` 任务已完成）：

- `6` 个正式 scene 已全部生成 `recon/points_fused_gt/*.npy`，每个 scene 各 `90` 个点云文件。
- `gt_upper_bound` 已完整落盘到：
  - `mvp-demo/output/evals/iciscae_node01_uav_v1/gt_upper_bound/`
- 当前全量 summary 为：
  - `mAP = 0.6111`
  - `recall@1 = 0.3333`
  - `recall@5 = 1.0000`
  - `recall@10 = 1.0000`
- 最关键判断已经明确：
  - `GT upper-bound` 也没有整体超过 `rgb_only`
  - 因此当前瓶颈不只在 predicted `mask / depth`，更可能在几何描述子和融合方式本身

### 2026-03-28

目标：把实验结果转换成论文正文。

任务：

- 写出：
  - 任务定义
  - benchmark protocol
  - 实验设置
  - related work
  - 主结果表
  - 失败分析
  - 结论段
- 固定失败分析主图：
  - `j10_line` 被修复的例子
  - `uav1_circle` 仍未修复的例子
- 把 geometry 分支写成“当前实现未形成稳定增益”的对照实验，而不是主结论。

输出物：

- 一份完整论文草稿。
- 一份图表与案例清单。

收尾检查：

- 论文草稿不再停留在零散段落。
- 已能直接给导师阅读。

执行记录（`2026-03-28` 任务已完成）：

- 已生成论文草稿：
  - `research/reviews/iciscae_paper_draft_2026_03_29_zh.md`
- 当前草稿已覆盖：
  - 任务定义
  - benchmark protocol
  - 实验设置
  - related work
  - 主结果表
  - 失败分析
  - 结论段
- 当前正文结论口径已经固定为：
  - `rgb_only` 是当前最稳定的正式 baseline
  - geometry 分支在当前实现下尚未形成可靠增益

### 2026-03-29

目标：做本周验收并输出新的周收口文档。

任务：

- 检查三组正式结果线是否全部可复现。
- 检查 `GT upper-bound` 是否已独立落盘。
- 生成统一 branch comparison 对照文件。
- 更新：
  - 当前进展看板
  - 周收口文档
  - `ACTIVE_PLAN`

输出物：

- 一份新的周收口文档。
- 一份新的当前进展看板。
- 更新后的 `ACTIVE_PLAN`。

收尾检查：

- 可以直接回答“geometry 是否修复了 `RGB-only` 的混淆”。
- 可以直接回答“下周应不应该继续扩 benchmark 或切到 `cross-node`”。

执行记录（`2026-03-29` 任务已完成）：

- 已生成统一对照文件：
  - `mvp-demo/output/evals/iciscae_node01_uav_v1/branch_comparison_summary.json`
  - `mvp-demo/output/evals/iciscae_node01_uav_v1/branch_comparison_summary.md`
- 已生成本周收口文档：
  - `research/reviews/iciscae_week_closure_2026_03_29_zh.md`
- 已更新当前进展看板：
  - `research/reviews/current_progress_board_2026_03_29_zh.md`
- 已更新主线计划：
  - `research/plans/ACTIVE_PLAN.md`
- 本周最终结论已经冻结：
  - geometry 分支只修复了个别 case
  - 当前整体最优仍是 `rgb_only`
  - `GT upper-bound` 也未显著优于 `rgb_only`
  - 下周不应继续扩 scene 或提前切到 `cross-node`，而应优先解释并重审 `descriptor / fusion`

## 4. 周验收标准

文档层面必须出现以下内容：

- 绝对日期 `2026-03-23` 到 `2026-03-29`，不用“周一到周日”这种相对表述。
- 四条结果线全部写清：
  - `rgb_only`
  - `rgb_predicted_depth_geometry`
  - `rgb_fused_geometry`
  - `gt_upper_bound`
- 文档明确写出本周结束时应看到的目录与结果位置：三条 geometry/GT eval 目录、统一 comparison 文件、论文草稿、周收口文档。

周末验收按 4 条判断：

- `rgb_predicted_depth_geometry`、`rgb_fused_geometry`、`gt_upper_bound` 都已落盘。
- 已形成统一 comparison 文件。
- 已能明确回答“geometry 是否优于 `rgb_only`”。
- 论文草稿已经具备可读的结果和结论段，而不只是日志和截图。

建议在周末检查以下目录或产物：

- `mvp-demo/output/evals/iciscae_node01_uav_v1/rgb_only/`
- `mvp-demo/output/evals/iciscae_node01_uav_v1/rgb_predicted_depth_geometry/`
- `mvp-demo/output/evals/iciscae_node01_uav_v1/rgb_fused_geometry/`
- `mvp-demo/output/evals/iciscae_node01_uav_v1/gt_upper_bound/`
- `mvp-demo/output/evals/iciscae_node01_uav_v1/branch_comparison_summary.md`
- `research/reviews/iciscae_week_closure_2026_03_29_zh.md`
- `research/reviews/iciscae_paper_draft_2026_03_29_zh.md`

## 5. 风险与回退

### 5.1 geometry 分支覆盖 `rgb_only` 结果

- 风险：若继续复用默认 `recon/points_fused`、`tracks/`、`embeddings/`，会污染现有主结果。
- 回退：本周统一使用 branch-specific 子目录，禁止覆盖现有 `rgb_only`。

### 5.2 `open3d_fpfh` 不稳定

- 风险：若 `open3d_fpfh` 在当前点云规模下不稳定，geometry 分支可能无法形成可复现结果。
- 回退：本周唯一允许的 fallback 是 `radial_hist`，但必须明确记入文稿；若未触发 fallback，则整个本周保持 `open3d_fpfh` 不变。

### 5.3 geometry 没有优于 `rgb_only`

- 风险：本周可能得到“geometry 没有带来提升”的负结果。
- 回退：不回避负结果，直接把它写成失败分析和误差归因证据。

### 5.4 `GT upper-bound` 也不提升

- 风险：即使切到 `masks_gt + depth_gt`，结果也可能没有显著变好。
- 回退：把结论转向“问题不只在感知输入，而在表示与融合设计”，为后续毕业论文方法改进留下清晰入口。

## 6. 本周结束时应该看到什么

如果本周按计划完成，周末应该同时看到：

- 三条新增结果线全部落盘：
  - `rgb_predicted_depth_geometry`
  - `rgb_fused_geometry`
  - `gt_upper_bound`
- 一份统一 branch comparison 对照文件。
- 一份新的周收口文档。
- 一份新的当前进展看板。
- 一份可以直接给导师的论文草稿。

一句话总结本周目标：

`这一周的重点不是继续扩 benchmark，而是把 geometry 两条分支、GT upper-bound 和论文结论一次性收口，并明确当前方法问题到底出在哪。`
