# [historical v3_clean] 当前项目进展梳理（已核对，2026-03-28）

> Historical `v3_clean` note:
> 本文档基于 `2026-03-28` 时点的 `v3_clean` 主线实物核对结果撰写。
> 当前激活主线已切到 `v3_clean_reflectfix`，本文件不再代表现行主线真值，只保留为历史盘点记录。

## 1. 用途与权威依据

本文档用于项目内部推进和导师沟通时的“真状态盘点”，不是答辩版包装材料。

当前结论只以本地仓库中已经核对过的实物为准，优先依据：

- `research/plans/ACTIVE_PLAN.md`
- `research/plans/tri_camera_node_3d_aware_reid/master_plan_zh.md`
- `research/plans/tri_camera_node_3d_aware_reid/benchmarks/iciscae_node01_uav_v3_clean.json`
- `research/handoffs/tri_camera_node_engineering_handoff_zh.md`
- `mvp-demo/output/evals/iciscae_node01_uav_v3_clean/branch_comparison_summary.md`
- `mvp-demo/output/evals/iciscae_node01_uav_v3_clean/query_failure_analysis.md`
- `mvp-demo/output/evals/iciscae_node01_uav_v3_clean/rgb_only/run_meta.json`

若其他 review 或草稿文档与以上实物不一致，以本文件和上述产物为准。

## 2. 摘要

- 项目已从“搭主链”进入 `M5.5 ICISCAE 论文收口` 阶段，不是早期原型状态。
- 当前冻结主线是：`MuJoCo + node01 + UAV/aircraft + single-node cross-scene + track-level retrieval`。
- `v3_clean` 已完成 `3 identities x 2 scenes = 6` 个正式 scene，四条结果线和失败分析产物都已落盘。
- 当前最可信结论是：`rgb_only` 仍是正式主基线；预测几何两条分支都未超过它；`gt_upper_bound` 明显更强，说明几何分支有潜力，但当前实现瓶颈仍在输入质量与 `descriptor / fusion`。

## 3. 当前完成度

### 3.1 文档层面

- 研究边界、benchmark 协议和工程 handoff 已冻结到 `research/`。
- 当前主线不再做人形 benchmark，也不把 `cross-node` 作为当前小论文成功条件。
- 当前研究口径固定为：先完成 `node01` 的单节点跨 scene 检索验证，再把 `cross-node` 和真实三相机迁移留给毕业论文主线。

### 3.2 工程层面

- 节点级主链已经齐全，可走通：
  - `mj_capture_3cam_node.py`
  - `run_node_depth_anything_v2.py`
  - `run_node_sam2_masks.py`
  - `build_node_tracklets.py`
  - `recon_fuse_depth_points.py`
  - `extract_node_track_embeddings.py`
  - `eval_node_track_retrieval.py`
- 当前 branch-specific 输出布局已经具备，可分离 `rgb_only`、预测几何和 `gt_upper_bound` 的 `tracklets / embeddings / eval` 结果，避免相互覆盖。

### 3.3 数据层面

- `j10 / uav1 / su34` 的 clean 正式 scene 已齐备。
- `v3_clean manifest` 中的 `6` 个 scene 与本地 `mvp-demo/data/nodes/node01/scenes/` 下的 clean scene 目录一致：
  - `mj_node01_j10_clean_line_nodes_a`
  - `mj_node01_j10_clean_circle_xz_b`
  - `mj_node01_uav1_clean_line_nodes_a`
  - `mj_node01_uav1_clean_circle_xz_b`
  - `mj_node01_su34_clean_line_nodes_a`
  - `mj_node01_su34_clean_circle_xz_b`
- `run_meta.json` 已确认当前 `rgb_only` 正式结果对应的 benchmark 是 `iciscae_node01_uav_v3_clean`，scene 列表与 manifest 一致。

### 3.4 实验层面

- `rgb_only / rgb_predicted_depth_geometry / rgb_fused_geometry / gt_upper_bound` 四条线已生成统一汇总。
- `branch_comparison_summary` 与 `query_failure_analysis` 都已在 `mvp-demo/output/evals/iciscae_node01_uav_v3_clean/` 落盘。
- 当前可复现实验入口已经齐备：
  - `run_iciscae_branch_eval.py`
  - `summarize_iciscae_branch_comparison.py`
  - `analyze_iciscae_failure_modes.py`

## 4. 当前正式结果

以下数值以 `mvp-demo/output/evals/iciscae_node01_uav_v3_clean/branch_comparison_summary.md` 为准：

| branch | mAP | R@1 | R@5 | R@10 | 结论 |
| --- | --- | --- | --- | --- | --- |
| `rgb_only` | `0.5750` | `0.3333` | `1.0000` | `1.0000` | 当前正式主基线 |
| `rgb_predicted_depth_geometry` | `0.4222` | `0.1667` | `1.0000` | `1.0000` | 弱于 `rgb_only` |
| `rgb_fused_geometry` | `0.4833` | `0.1667` | `1.0000` | `1.0000` | 仍弱于 `rgb_only` |
| `gt_upper_bound` | `0.8333` | `0.6667` | `1.0000` | `1.0000` | 证明高质量几何输入有价值 |

当前最重要的实验判断：

- clean 场景移除 humanoid 后，预测 geometry 两条分支仍未超过 `rgb_only`。
- `gt_upper_bound` 明显强于两条预测 geometry 分支，说明问题不只在场景遮挡，输入质量仍是主要瓶颈。
- `gt_upper_bound` 仍未完全消除全部混淆，说明 `descriptor / fusion` 本身也仍有继续优化空间。

## 5. 当前冻结接口与验证口径

- 当前无新增 public API。
- 当前冻结的实验接口是：`v3_clean manifest + scene 目录数据契约 + branch-specific 输出布局`。
- 主链正式输入固定为：
  - `cams/cam*/frames/`
  - `cams/cam*/masks/`
  - `cams/cam*/depth/`
  - `calib/rig.json`
  - `frame_times.csv`
- `masks_gt / depth_gt` 只用于 `gt_upper_bound` 和误差分析，不进入当前小论文正式主结果。
- 仓库暂无自动化测试；当前主要验证方式仍是：
  - smoke check
  - scene 目录完整性核对
  - `branch_comparison_summary` / `query_failure_analysis` 等已落盘产物核对

## 6. 风险、缺口与下一步

### 6.1 当前风险

- 当前最大风险不是“没做完”，而是“文档口径漂移”。
- 仓库中存在标注为 `2026-03-29` 的进展板与周收口草稿，但其结果数值与 `v3_clean` 实际汇总不一致。
- 截至 `2026-03-28`，当前应以 `v3_clean` 实际输出以及 `2026-03-23` 的 clean 进展板为准，不应直接把 later-dated 草稿视为最终冻结结论。
- 当前 worktree 中仍有未提交的 `research/` 文稿修改，说明论文文本仍在收口中；这些草稿可参考，但不应自动当作最终口径。

### 6.2 近期最合理的下一步

1. 锁定论文总表、关键 failure case 图，以及“主结果段 / 失败分析段 / 结论段”的最终口径。
2. 不再扩 scene，不切 `cross-node`，不新增正式 benchmark 分支。
3. 若还补实验，只建议做 `j10_circle`、`uav1_line`、`uav1_circle` 的小范围输入质量诊断。
4. 待论文图表和草稿锁定后，再从 `2026-03-30` 起选择 `1-3` 个代表性 scene 启动 3DGS 离线 PoC。

## 7. 默认假设

- 本文档默认服务于内部推进、导师沟通和阶段对齐，而不是对外宣传。
- 本文档默认“当前项目进展”应以本地仓库实物、manifest、eval 汇总和 handoff 为准，而不是以未来日期或草稿性 review 文档为准。
