# [historical v3_clean] 论文收口与 3DGS/VGGT 接入窗口（2026-03-23）

> Historical `v3_clean` note:
> 本文档基于旧主线 `v3_clean` 的阶段判断撰写，用于保留当时的论文收口与 3DGS/VGGT 接入时机分析。
> 当前激活主线已切到 `v3_clean_reflectfix`，本文件只保留为历史决策参考，不再代表现行主线判断。

## 1. 当前阶段判断

- `v3_clean` 已完成 `6 scene` 正式 benchmark 与四条结果线，当前最重要的稳定事实已经固定：
  - `rgb_only` 是当前最稳的正式基线；
  - 两条预测 geometry 分支均未超过 `rgb_only`；
  - `gt_upper_bound` 明显更强。
- 因此从现在开始，项目优先级应切到**论文收口**，而不是继续扩 benchmark、扩 scene 或并行开新主线。
- 当前项目其实已经具备一条可用的轻量三维重建入口：`mvp-demo/scripts/recon_fuse_depth_points.py` 可直接生成 `recon/points_depth_cam0`、`recon/points_fused`、`recon/points_fused_gt`，用于当前误差归因和 geometry 分支分析。

## 2. 论文收口的直接任务

- 正式主结果固定只保留 `rgb_only / rgb_predicted_depth_geometry / rgb_fused_geometry / gt_upper_bound` 四行，不再补新的正式结果线。
- 当前必须尽快锁定的正文内容只有三块：
  - 主结果段：clean 后预测 geometry 仍未超过 `rgb_only`
  - 失败分析段：当前主瓶颈更偏向 `SAM2 mask / predicted depth`
  - 结论段：geometry 有潜力，但当前尚未形成稳定增益
- 当前图表优先级固定为：
  - 1 张四条结果线的统一总表
  - 1 组“GT 能恢复、预测不能恢复”的关键案例图
  - 1 组“geometry 反而带来退化”的关键案例图
- 若还做补充实验，只允许做小范围输入质量诊断，优先对象固定为：
  - `mj_node01_j10_clean_circle_xz_b`
  - `mj_node01_uav1_clean_line_nodes_a`
  - `mj_node01_uav1_clean_circle_xz_b`

## 3. 3DGS / VGGT 的接入时机

- **现在就能开始的部分**：继续基于 `recon_fuse_depth_points.py` 做三维重建误差归因；这仍属于当前论文主线内部工作。
- **最早可开始 3DGS 小规模接入的时间点**：放在本轮论文草稿和图表锁定之后，建议从 `2026-03-30` 这一周开始，先做 `1-3` 个代表性 scene 的离线验证。
- **3DGS 优先于 VGGT**：仓库里已经保留 `run_3dgs_scene.py`、`gs_render_depth_npy.py`、`mj_freecam_gs_composite.py` 等 3DGS 相关脚手架，因此它更适合作为第一条工程 PoC 分支。
- **VGGT 不建议现在并行开**：当前仓库没有稳定的 VGGT 主链入口，它更适合放到小论文收口之后，作为毕业论文阶段的重建增强预研。

## 4. 接入方式与边界

- 3DGS/VGGT 的第一接入点不要直接改 retrieval 下游，而应先替换或增强当前 `recon/points_*` 这层几何产物。
- 第一阶段目标固定为：让 3DGS/VGGT 产出与当前点云分支可比较、可诊断的几何结果，再复用既有的 `tracklets / embeddings / eval` 链。
- 第一阶段不把 3DGS/VGGT 写成新的正式 benchmark 分支，只把它当作**辅助重建分支**，用来回答“它能否提供比 predicted depth fusion 更稳定的几何输入”。

## 5. 阶段化验收

- 论文收口完成的标志：
  - 正文已有摘要、任务定义、实验设置、主结果、失败分析、结论
  - 图表与文字口径一致
  - 能明确回答“为什么 clean 后仍未超过 `rgb_only`”
- 3DGS PoC 完成的标志：
  - 能在代表性 scene 上稳定重建
  - 输出可与当前 `recon/points_fused` 同口径对比
  - 至少在可视化质量、几何完整性或后续检索排序上体现一种明确收益
- VGGT 启动前提：
  - 小论文正文已经基本定稿
  - 3DGS PoC 已经说明当前问题仍需要更强多视图重建，而不是仅靠调 `SAM2/depth` 即可解决
