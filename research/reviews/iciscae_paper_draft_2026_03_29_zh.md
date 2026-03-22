# ICISCAE 小论文草稿（2026-03-29）

## 题目建议

英文暂定：`Single-node Cross-scene 3D-aware Track Retrieval for UAV/Aircraft Targets Using a Tri-camera MuJoCo Benchmark`

中文暂定：`面向无人机/飞行器目标的三相机节点级 3D-aware 轨迹检索基准与初步实验`

## 摘要草稿

针对低空监测场景中飞行器目标跨视角再识别难、真实多节点数据采集成本高的问题，本文先在 MuJoCo 中构建一个三相机节点级的 `single-node, cross-scene, track-level retrieval` 基准，用于验证 `RGB + depth + mask + rig + timestamps` 数据契约是否足以支持 3D-aware 目标检索。我们固定 `node01` 的三相机节点，并围绕 `j10`、`uav1`、`dji_mavic` 三个身份构建 `3 identities x 2 scenes` 的正式 benchmark。基于该基准，本文比较了 `RGB-only`、`RGB + predicted-depth geometry`、`RGB + fused geometry` 三条结果线，并额外给出 `masks_gt + depth_gt` 的 `GT upper-bound` 对照。实验表明，`RGB-only` 目前取得最佳总体结果（`mAP = 0.8333`, `recall@1 = 0.6667`）；geometry 分支能够修复个别 query 的误检索，但尚未形成稳定的整体增益；`GT upper-bound` 也未显著超过 `RGB-only`，说明当前瓶颈不只在感知输入噪声，更可能来自几何描述子与融合策略本身。该 benchmark 和失败案例为后续面向 `cross-node 3D ReID` 的研究提供了可复现起点与误差归因依据。

## 1. 任务定义

### 1.1 研究问题

本文研究的不是传统单帧分类，而是节点级三相机观测下的 `track-level retrieval`：

- 输入：同一目标在一个采集窗口中的三相机多帧观测；
- 中间表示：以 `tracklet` 为基本单元组织 `frames / masks / depth / rig / timestamps`；
- 输出：为每个 `tracklet` 生成 `track embedding`，并在其他 scene 中做检索。

### 1.2 为什么当前阶段仍然属于 ReID

虽然当前实验固定在 `node01` 内进行，但评测时已经显式排除：

- 同一 `track_id`
- 同一 `scene`

因此 query 需要在其他 scene 中找回同一 `identity_id` 的 track，这仍然是标准的 re-identification / retrieval 问题，只是先去掉了 `cross-node` 变量，用于验证表征本身是否成立。

### 1.3 本文范围

- 本文只报告 `node01` 的 `single-node, cross-scene, track-level retrieval`。
- 本文不宣称 `cross-node` 已完成。
- 本文不把 MuJoCo GT 输入作为正式主链，只把 GT 结果作为 upper-bound 分析线。

## 2. Benchmark Protocol

### 2.1 正式身份与 scene

正式 benchmark 固定为 `3 identities x 2 scenes`：

| identity_id | scene A | scene B |
| --- | --- | --- |
| `j10` | `mj_node01_j10_line_nodes_a` | `mj_node01_j10_circle_xz_b` |
| `uav1` | `mj_node01_uav1_line_nodes_a` | `mj_node01_uav1_circle_xz_b` |
| `dji_mavic` | `mj_node01_dji_mavic_line_nodes_a` | `mj_node01_dji_mavic_circle_xz_b` |

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
- 弱/强几何重建：`recon_fuse_depth_points.py`
- tracklet 构建：`build_node_tracklets.py`
- embedding 提取：`extract_node_track_embeddings.py`
- retrieval 评测：`eval_node_track_retrieval.py`

### 3.2 本周新增的可复现脚本

- `mvp-demo/scripts/run_iciscae_branch_eval.py`
  - 用于批量执行 `rgb_predicted_depth_geometry`、`rgb_fused_geometry`、`gt_upper_bound`
- `mvp-demo/scripts/summarize_iciscae_branch_comparison.py`
  - 用于生成 branch 级结果对照文件

### 3.3 结果文件位置

- `rgb_only`：`mvp-demo/output/evals/iciscae_node01_uav_v1/rgb_only/all_queries_vs_all_scenes.json`
- `rgb_predicted_depth_geometry`：`mvp-demo/output/evals/iciscae_node01_uav_v1/rgb_predicted_depth_geometry/all_queries_vs_all_scenes.json`
- `rgb_fused_geometry`：`mvp-demo/output/evals/iciscae_node01_uav_v1/rgb_fused_geometry/all_queries_vs_all_scenes.json`
- `gt_upper_bound`：`mvp-demo/output/evals/iciscae_node01_uav_v1/gt_upper_bound/all_queries_vs_all_scenes.json`
- 统一对照：`mvp-demo/output/evals/iciscae_node01_uav_v1/branch_comparison_summary.md`

## 4. Related Work 草稿

### 4.1 RGB-D / 3D-aware ReID

现有 3D-aware ReID 大多围绕行人、骨架或 LiDAR 点云展开，任务先验与本文的 object-centric `UAV / aircraft` 目标存在差异。与这类工作相比，本文更关注三相机节点级数据契约在飞行器小目标检索中的可行性，而不是立即训练新的端到端 3D encoder。

### 4.2 点云目标检索

面向任意物体的 point-cloud re-identification 工作为本文提供了几何分支设计参考，尤其是“以 object point cloud 作为检索输入”的思路。但本文当前阶段没有完整复现外部网络，而是先使用可快速落地的 `open3d_fpfh` 全局几何描述子进行验证。

### 4.3 多相机与跨节点跟踪

面向多相机跟踪的工作通常将 ReID 与数据关联耦合起来，并直接服务于更重的 `MOT / cross-camera tracking` 系统。本文当前小论文范围有意收缩到 `single-node, cross-scene, track-level retrieval`，目的在于先验证表征，再把 `cross-node` 和真实迁移留给毕业论文主线。

## 5. 主结果表

### 5.1 总体指标

| branch | mAP | recall@1 | recall@5 | recall@10 |
| --- | --- | --- | --- | --- |
| `rgb_only` | `0.8333` | `0.6667` | `1.0000` | `1.0000` |
| `rgb_predicted_depth_geometry` | `0.6389` | `0.3333` | `1.0000` | `1.0000` |
| `rgb_fused_geometry` | `0.6667` | `0.3333` | `1.0000` | `1.0000` |
| `gt_upper_bound` | `0.6111` | `0.3333` | `1.0000` | `1.0000` |

### 5.2 结果解读

- `rgb_only` 当前仍是最强总体基线。
- `rgb_predicted_depth_geometry` 与 `rgb_fused_geometry` 都没有在总体指标上超过 `rgb_only`。
- `rgb_fused_geometry` 比弱几何分支略好，但增益不足以支撑“geometry 明显提升”的结论。
- `gt_upper_bound` 也没有超过 `rgb_only`，说明当前瓶颈并不只是 predicted `mask / depth` 的误差。

## 6. 失败分析草稿

### 6.1 geometry 修复了什么

`mj_node01_j10_line_nodes_a` 是当前最值得保留的正例：

- 在 `rgb_only` 中，它的 `top1` 被 `uav1_line` 抢走；
- 在 `rgb_predicted_depth_geometry` 与 `rgb_fused_geometry` 中，它都被修复为正确匹配 `j10_circle`。

这说明 geometry 分支不是完全无效，而是对个别形态差异更明显的 case 有帮助。

### 6.2 geometry 没修复什么

`mj_node01_uav1_circle_xz_b` 是当前最关键的失败例：

- `rgb_only` 的 `top1` 已经错到 `dji_mavic_circle`；
- 两条 geometry 分支没有修复这个错误；
- `gt_upper_bound` 甚至把错误目标换成了 `j10_circle`。

因此，当前的核心难点仍是 `uav1 / dji_mavic` 之间的高相似度混淆。

### 6.3 geometry 带来的新退化

更值得注意的是，一些原本在 `rgb_only` 中正确的 query 在 geometry 分支里变差了：

- `uav1_line` 从 `top1` 正确变成被 `dji_mavic_line` 抢走；
- `dji_mavic_line` 与 `dji_mavic_circle` 也都在 geometry 分支里受到 `uav1` 干扰。

这说明“把几何特征直接拼接到 RGB embedding 后”并不能自动得到更好的细粒度区分能力。

### 6.4 当前最可能的问题来源

结合实验结果，当前更可能的问题包括：

- `open3d_fpfh` 对当前小型 aircraft / UAV 的区分能力不足；
- RGB 与 geometry 直接拼接后，两个模态的尺度与判别性不匹配；
- 点云归一化、时间池化与采样过程抹平了细粒度几何差异。

## 7. 结论段草稿

本文首先在 MuJoCo 三相机节点上构建了一个面向 `UAV / aircraft` 目标的 `single-node, cross-scene, track-level retrieval` benchmark，并在统一数据契约下比较了 `rgb_only`、`rgb_predicted_depth_geometry`、`rgb_fused_geometry` 以及 `gt_upper_bound` 四条结果线。实验表明，当前 `CLIP + no geometry` 仍是最稳定的正式基线；引入几何信息能够修复个别误检索，但尚未形成稳定的整体收益；即便使用 `masks_gt + depth_gt` 的上界分析线，也没有整体超过 `rgb_only`。因此，当前阶段最合理的结论不是“geometry 已经有效解决了目标混淆”，而是“geometry 在当前表示与融合实现下尚未形成可靠增益”。这一结果为后续毕业论文继续推进 `descriptor / fusion / cross-node` 提供了明确问题入口。

## 8. 给导师汇报时的说法建议

- 可以明确说：`benchmark 已完成、三组主结果已齐、GT upper-bound 也已补齐`。
- 也可以明确说：`当前 geometry 没有跑赢 rgb_only，这不是坏事，反而说明下一步应该把方法问题讲清楚，而不是继续堆实验。`
- 不建议说：`geometry 已经证明有效`。
- 更建议说：`geometry 对个别样例有帮助，但目前尚未形成稳定增益，下一步要针对 descriptor 与 fusion 继续做误差归因。`
