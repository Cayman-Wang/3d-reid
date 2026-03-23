# ICISCAE 小论文草稿（`v3_clean`, 2026-03-23）

## 题目建议

英文暂定：`Single-node Cross-scene 3D-aware Track Retrieval for UAV/Aircraft Targets with a Clean Tri-camera MuJoCo Benchmark`

中文暂定：`基于无干扰三相机 MuJoCo 基准的无人机/飞行器单节点跨场景轨迹检索`

## 摘要草稿

针对低空监测场景中飞行器目标跨视角检索难、真实多节点数据采集成本高的问题，本文在 MuJoCo 中构建了一个无 humanoid 干扰的三相机节点级 `single-node, cross-scene, track-level retrieval` 基准，用于验证 `RGB + depth + mask + rig + timestamps` 数据契约是否足以支持 3D-aware 目标检索。我们固定 `node01` 三相机构型，并围绕 `j10`、`uav1`、`su34` 三个身份构建 `3 identities x 2 scenes` 的 `v3_clean` benchmark。基于该基准，本文比较了 `RGB-only`、`RGB + predicted-depth geometry`、`RGB + fused geometry` 三条正式结果线，并额外给出 `masks_gt + depth_gt` 的 `GT upper-bound` 诊断线。实验表明：`rgb_only` 目前在预测输入条件下仍是最稳定基线（`mAP = 0.5750`, `recall@1 = 0.3333`）；两条预测 geometry 分支均未超过该基线；但 `GT upper-bound` 可达到 `mAP = 0.8333`, `recall@1 = 0.6667`。这一结果说明，在 clean 场景下移除 humanoid 干扰后，当前主瓶颈更偏向 `SAM2 mask / predicted depth` 的感知质量，而不只是场景遮挡；同时，几何表征与融合设计仍有进一步优化空间。该 benchmark 与失败案例为后续 `cross-node 3D re-ID` 研究提供了可复现的诊断起点。

## 1. 任务定义

### 1.1 研究问题

本文研究的是三相机节点级的 `track-level retrieval`：

- 输入：同一目标在一个采集窗口中的三相机多帧观测；
- 中间表示：以 `tracklet` 为基本单元组织 `frames / masks / depth / rig / timestamps`；
- 输出：为每个 `tracklet` 生成稳定的 `track embedding`，并在其他 scene 中检索同身份目标。

### 1.2 为什么当前阶段仍然属于 ReID

虽然本文当前只在 `node01` 中评测，但评测时显式排除了：

- 同一 `track_id`
- 同一 `scene`

因此 query 需要在其他 scene 中找回同一 `identity_id` 的 track，这仍然是标准的 re-identification / retrieval 问题，只是暂时去掉了 `cross-node` 变量，用于先验证表征是否成立。

### 1.3 本文范围

- 本文只报告 `node01` 的 `single-node, cross-scene, track-level retrieval`。
- 本文不宣称 `cross-node` 已完成。
- 本文不把 MuJoCo GT 输入作为正式主链，只把 GT 结果作为 upper-bound 诊断线。

## 2. Benchmark Protocol

### 2.1 正式身份与 clean scenes

正式 benchmark 固定为 `3 identities x 2 scenes`：

| identity_id | scene A | scene B |
| --- | --- | --- |
| `j10` | `mj_node01_j10_clean_line_nodes_a` | `mj_node01_j10_clean_circle_xz_b` |
| `uav1` | `mj_node01_uav1_clean_line_nodes_a` | `mj_node01_uav1_clean_circle_xz_b` |
| `su34` | `mj_node01_su34_clean_line_nodes_a` | `mj_node01_su34_clean_circle_xz_b` |

### 2.2 输入契约

正式主链固定消费：

- `frames`
- predicted `masks`
- predicted `depth`
- `calib/rig.json`
- `frame_times.csv`

形式约束：

- `mask_layout = flat`
- `min_valid_timestamps = 5`
- 评测统一开启 `exclude_same_track_id` 与 `exclude_same_scene`

### 2.3 分支定义

| branch | 定义 | 当前实现 |
| --- | --- | --- |
| `rgb_only` | 仅使用 RGB 外观特征 | `CLIP + no geometry` |
| `rgb_predicted_depth_geometry` | RGB + 单相机弱几何 | `cam0 predicted depth + mask + open3d_fpfh` |
| `rgb_fused_geometry` | RGB + 三相机融合强几何 | `predicted depth + mask fusion + open3d_fpfh` |
| `gt_upper_bound` | 上界分析线 | `masks_gt + depth_gt + fused geometry` |

## 3. 实验设置

### 3.1 环境与执行链路

- 环境：`mvp_demo`
- 核心依赖：`torch 2.10.0+cu128`、`mujoco 3.6.0`、`open_clip 3.3.0`、`open3d 0.19.0`
- predicted depth：`run_node_depth_anything_v2.py`
- predicted masks：`run_node_sam2_masks.py`
- 几何重建：`recon_fuse_depth_points.py`
- tracklet 构建：`build_node_tracklets.py`
- embedding 提取：`extract_node_track_embeddings.py`
- retrieval 评测：`eval_node_track_retrieval.py`

### 3.2 可复现脚本与结果位置

- 批量执行：`mvp-demo/scripts/run_iciscae_branch_eval.py`
- branch 汇总：`mvp-demo/scripts/summarize_iciscae_branch_comparison.py`
- 失败分析：`mvp-demo/scripts/analyze_iciscae_failure_modes.py`

当前主结果文件：

- `mvp-demo/output/evals/iciscae_node01_uav_v3_clean/rgb_only/all_queries_vs_all_scenes.json`
- `mvp-demo/output/evals/iciscae_node01_uav_v3_clean/rgb_predicted_depth_geometry/all_queries_vs_all_scenes.json`
- `mvp-demo/output/evals/iciscae_node01_uav_v3_clean/rgb_fused_geometry/all_queries_vs_all_scenes.json`
- `mvp-demo/output/evals/iciscae_node01_uav_v3_clean/gt_upper_bound/all_queries_vs_all_scenes.json`
- `mvp-demo/output/evals/iciscae_node01_uav_v3_clean/branch_comparison_summary.md`
- `mvp-demo/output/evals/iciscae_node01_uav_v3_clean/query_failure_analysis.md`

## 4. Related Work 草稿

### 4.1 RGB-D / 3D-aware ReID

现有 3D-aware ReID 大多围绕行人、骨架或 LiDAR 点云展开，任务先验与本文的 object-centric `UAV / aircraft` 检索存在差异。与此相比，本文更关注三相机节点级数据契约在飞行器小目标检索中的可行性，而不是立即训练新的端到端 3D encoder。

### 4.2 点云目标检索

面向任意物体的 point-cloud retrieval 为本文的几何分支设计提供了参考，尤其是“以 object point cloud 作为检索输入”的思路。但本文当前阶段没有完整复现外部网络，而是先用可快速落地的 `open3d_fpfh` 全局几何描述子做 feasibility check。

### 4.3 多相机与跨节点跟踪

多相机跟踪工作通常将 ReID 与数据关联耦合，直接服务于更重的 `MOT / cross-camera tracking` 系统。本文有意把小论文范围收缩到 `single-node, cross-scene, track-level retrieval`，先验证表征是否可行，再把 `cross-node` 与真实迁移留给毕业论文。

## 5. 主结果表

### 5.1 总体指标

| branch | mAP | recall@1 | recall@5 | recall@10 |
| --- | --- | --- | --- | --- |
| `rgb_only` | `0.5750` | `0.3333` | `1.0000` | `1.0000` |
| `rgb_predicted_depth_geometry` | `0.4222` | `0.1667` | `1.0000` | `1.0000` |
| `rgb_fused_geometry` | `0.4833` | `0.1667` | `1.0000` | `1.0000` |
| `gt_upper_bound` | `0.8333` | `0.6667` | `1.0000` | `1.0000` |

### 5.2 结果解读

- 在预测输入条件下，`rgb_only` 仍是当前最强正式基线。
- `rgb_predicted_depth_geometry` 与 `rgb_fused_geometry` 都没有在总体指标上超过 `rgb_only`。
- `rgb_fused_geometry` 略好于单相机弱几何，但不足以支撑“几何信息已稳定提升检索”的结论。
- `gt_upper_bound` 明显优于两条预测 geometry 分支，也高于 `rgb_only`，说明 clean 主线下的主要差距已经更集中地指向 `mask/depth` 预测质量。

## 6. 失败分析草稿

### 6.1 clean 后问题没有自动消失

clean 场景移除了 humanoid 干扰，但 `j10`、`uav1`、`su34` 之间的混淆并未自动消失：

- `mj_node01_j10_clean_circle_xz_b` 在三条预测分支中都错配到 `uav1_clean_circle_xz_b`；
- `mj_node01_uav1_clean_line_nodes_a` 在三条预测分支中都被 `su34_clean_line_nodes_a` 吸走；
- `mj_node01_uav1_clean_circle_xz_b` 在三条预测分支中都更接近 `j10_clean_circle_xz_b`。

因此，当前负结果不能再简单归因为 humanoid 遮挡背景。

### 6.2 `GT upper-bound` 说明了什么

`GT upper-bound` 恢复了三条此前失败的 query：

- `mj_node01_j10_clean_circle_xz_b`
- `mj_node01_uav1_clean_line_nodes_a`
- `mj_node01_uav1_clean_circle_xz_b`

这说明：

- 只要 `mask/depth` 质量足够好，geometry 分支是有潜力的；
- 当前预测 geometry 的主要问题并不是“几何无信息”，而是“输入质量不足以稳定支撑几何检索”。

### 6.3 `GT upper-bound` 仍未完全解决的问题

`GT upper-bound` 并没有让所有 query 都变成完美：

- `mj_node01_j10_clean_line_nodes_a` 仍然把 `su34_clean_line_nodes_a` 排在第一；
- `mj_node01_su34_clean_line_nodes_a` 仍然把 `j10_clean_line_nodes_a` 排在第一。

因此，clean 主线下最稳妥的结论应是：

- 当前第一瓶颈是预测 `mask/depth` 的感知误差；
- 但 geometry descriptor 与 fusion 也不是完全解决的。

### 6.4 geometry 额外带来的退化

- `mj_node01_su34_clean_circle_xz_b` 在 `rgb_only` 中 `top1` 正确，但在 `rgb_predicted_depth_geometry` 中退化。
- `mj_node01_su34_clean_line_nodes_a` 在 `rgb_only` 中 `top1` 正确，但在 `rgb_fused_geometry` 中退化。

这说明当前的 `open3d_fpfh + CLIP` 直接拼接仍然不稳定，预测 geometry 分支还没有学会“只在有用时帮助，而不是在困难 query 上进一步扰动排序”。

## 7. 结论段草稿

本文在 MuJoCo 三相机节点上构建了一个无 humanoid 干扰的 `single-node, cross-scene, track-level retrieval` benchmark，并在统一数据契约下比较了 `rgb_only`、`rgb_predicted_depth_geometry`、`rgb_fused_geometry` 与 `gt_upper_bound` 四条结果线。实验表明：clean 场景并没有让预测 geometry 分支自动超过 `rgb_only`；但 `GT upper-bound` 的显著提升说明当前主瓶颈主要来自 `SAM2 mask / predicted depth` 的感知质量，而不只是背景遮挡。与此同时，`GT upper-bound` 也未完全消除 `j10` 与 `su34` 的 line 轨迹混淆，说明 geometry descriptor 与融合策略仍需进一步研究。因此，当前阶段最合理的结论不是“geometry 已经稳定有效”，而是“在 clean benchmark 下，geometry 的潜力已经被看见，但要把这种潜力转化为稳定增益，首先需要解决预测输入质量与几何融合稳定性问题”。这一结果为后续毕业论文继续推进 `cross-node` 与系统性误差分析提供了清晰入口。

## 8. 给导师汇报时的说法建议

- 可以明确说：`clean benchmark 已完成，四条结果线已齐。`
- 可以明确说：`移除 humanoid 后，预测 geometry 仍没跑赢 rgb_only，所以问题不只是遮挡。`
- 更建议说：`GT upper-bound 明显更强，说明后续最该优先做的是感知误差诊断，而不是继续扩 benchmark。`
- 不建议说：`geometry 已经稳定有效。`
