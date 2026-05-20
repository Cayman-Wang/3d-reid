# Node01 NeoVerse（新视界）4D 阶段汇报材料

日期：2026-05-11

## 1. 研究目的

本阶段研究目标是验证：多相机 NeoVerse（新视界）4D 动态点云能否作为 3D-ReID（3D 重识别）的几何分支，稳定接入现有 ReID pipeline（重识别流程链路），并形成可复现实验矩阵。

具体问题分为三层：

- 工程闭环：NeoVerse 4D 输出能否转换为 ReID 可消费的 `points_by_timestamp`（按时间戳组织的点云结果）。
- 几何分支有效性：`RGB crop（RGB 裁剪图） + 4D geometry（4D 几何）` 是否能与 `RGB-only（仅 RGB）` 在同一评测矩阵下对比。
- 真实化推进：在去掉 `scene（场景） depth_gt（深度真值）` 后，NeoVerse own-depth（自有深度）是否仍能支撑三相机 4D geometry（4D 几何）评测。

当前阶段不是做高质量纹理重建，也不是证明 3DGS（3D Gaussian Splatting，三维高斯泼溅）视觉效果，而是证明 `NeoVerse 4D points（4D 点云） -> tracklet（轨迹片段） -> CLIP RGB + FPFH geometry（几何特征） -> ReID retrieval（重识别检索）` 这条链路可运行、可评测、可扩展。

本阶段一句话结论：

```text
node01 单节点 3 identities（身份） x 2 scenes（场景） 的 NeoVerse 4D ReID 可评测矩阵已经闭环；view-count ablation（视角数量消融）和 own-depth no-depthGT（自有深度、无深度真值）第一版实验也已完成。当前所有分支指标持平，说明链路已可评测，但现有 GT-mask bootstrap（真值掩码引导的起步评测设置）难度不足以体现几何、视角数量或深度来源差异。
```

## 2. 研究方法

### 2.1 总体技术路线

核心链路如下：

```text
三相机 RGB / masks_gt（掩码真值） / NeoVerse observation depth（观测深度）
-> per-camera NeoVerse observations（逐相机观测结果）
-> points_per_view（逐视角点云）
-> fused 4D points_by_timestamp（融合后的逐时间戳 4D 点云）
-> node-level tracklet（节点级轨迹片段）
-> CLIP RGB crop（CLIP 的 RGB 裁剪特征） + Open3D FPFH geometry（Open3D 的 FPFH 几何特征）
-> cross-scene ReID retrieval（跨场景重识别检索）
```

其中：

- RGB 分支使用 `CLIP 512D`，负责目标外观。
- 几何分支使用 `Open3D FPFH 33D`，消费每个 timestamp（时间戳）的 3D 点云。
- 融合后 embedding（嵌入特征）为 `545D = 512D CLIP + 33D FPFH`。
- 下游权威几何输入固定为 `points_by_timestamp/index.csv + meta.json + *.npy`。

### 2.2 矩阵设计

当前矩阵覆盖 `node01` 上的 3 个 identity（身份），每个 identity 2 个 scene（场景）：

| identity（身份） | scene（场景）数 | timestamps（时间戳） | status（状态） |
| --- | ---: | ---: | --- |
| `j10` | 2 | 81/scene | ready |
| `uav1` | 2 | 81/scene | ready |
| `su34` | 2 | 81/scene | ready |

每个 scene 都满足：

- `points_by_timestamp/meta.json.schema_version = neoverse_points_by_timestamp_v1`
- `points_by_timestamp/index.csv = 81 rows`
- `points_by_timestamp/*.npy = 81 files`

### 2.3 对照实验设计

本阶段设计了三类对照：

| 对照 | 目的 | 当前状态 |
| --- | --- | --- |
| `RGB-only（仅 RGB）` vs `RGB + NeoVerse 4D` | 验证几何分支接入和同口径评测 | 已完成 |
| `cam0/cam1/cam2` vs `tri-cam（三相机）` | 验证单相机与三相机 view-count（视角数量）差异 | 已完成 |
| `depth_gt bootstrap（深度真值起步设置）` vs `own-depth no-depthGT（自有深度、无深度真值）` | 验证去掉 `scene depth_gt` 后是否仍可评测 | 已完成 |

需要明确边界：

- `own-depth no-depthGT` 去掉的是 `scene_dir/cams/<cam>/depth_gt/*.npy` 动态锚深度。
- 第一版仍使用 `masks_gt`（掩码真值）和 `rig.json`（相机刚体标定文件），因此不能写成“完全无真值”。
- 当前所有分支指标持平，不能写成 geometry（几何）、三相机或 own-depth（自有深度）已带来 Rank/mAP 提升。

## 3. 实验手段

### 3.1 实验分支

主线三线对比：

| 实验线 | 输入 | embedding | eval branch |
| --- | --- | ---: | --- |
| `RGB-only（仅 RGB）` | RGB crop（RGB 裁剪图） | 512D | `rgb_only_clip_gtmask_eval_v1` |
| `RGB + NeoVerse 4D depth_gt bootstrap` | RGB crop + 4D points（4D 点云） | 545D | `rgb_neoverse_fused_4d_clip_fpfh_eval_v1` |
| `RGB + NeoVerse own-depth no-depthGT` | RGB crop + own-depth 4D points（自有深度 4D 点云） | 545D | `rgb_neoverse_own_depth_4d_clip_fpfh_eval_v1` |

View-count（视角数量）消融：

| view（视角） | RGB-only branch（仅 RGB 分支） | RGB+4D branch（RGB+4D 分支） |
| --- | --- | --- |
| `cam0` | `rgb_only_clip_gtmask_cam0_eval_v1` | `rgb_neoverse_single_cam0_4d_clip_fpfh_eval_v1` |
| `cam1` | `rgb_only_clip_gtmask_cam1_eval_v1` | `rgb_neoverse_single_cam1_4d_clip_fpfh_eval_v1` |
| `cam2` | `rgb_only_clip_gtmask_cam2_eval_v1` | `rgb_neoverse_single_cam2_4d_clip_fpfh_eval_v1` |
| `tri-cam` | `rgb_only_clip_gtmask_eval_v1` | `rgb_neoverse_fused_4d_clip_fpfh_eval_v1` |

### 3.2 可视化展示材料

主图建议使用稳定 run（稳定运行结果目录）的 preview（预览图） ，而不是 own-depth 或 view-count 目录。own-depth 和 view-count 本轮主要产出是评测结果与点云 contract（结果契约），不是新 preview 图。

稳定 preview 根目录：

```text
D:\node01_spin_runtime_ascii\mvp-demo\output\neoverse_fused_runs\2026-04-26_j10_yp20_r02params_r01\mj_node01_j10_spin_static_yp20_a\preview
```

推荐主图：两栏局部目标对比

```text
D:\node01_spin_runtime_ascii\mvp-demo\output\neoverse_fused_runs\2026-04-26_j10_yp20_r02params_r01\mj_node01_j10_spin_static_yp20_a\preview\local_target_rgb_points_compare_v1\cam0_local_compare_first.png
D:\node01_spin_runtime_ascii\mvp-demo\output\neoverse_fused_runs\2026-04-26_j10_yp20_r02params_r01\mj_node01_j10_spin_static_yp20_a\preview\local_target_rgb_points_compare_v1\cam1_local_compare_first.png
D:\node01_spin_runtime_ascii\mvp-demo\output\neoverse_fused_runs\2026-04-26_j10_yp20_r02params_r01\mj_node01_j10_spin_static_yp20_a\preview\local_target_rgb_points_compare_v1\cam2_local_compare_first.png
```

可选播放视频：

```text
D:\node01_spin_runtime_ascii\mvp-demo\output\neoverse_fused_runs\2026-04-26_j10_yp20_r02params_r01\mj_node01_j10_spin_static_yp20_a\preview\local_target_rgb_points_compare_v1\cam0_local_compare.mp4
D:\node01_spin_runtime_ascii\mvp-demo\output\neoverse_fused_runs\2026-04-26_j10_yp20_r02params_r01\mj_node01_j10_spin_static_yp20_a\preview\local_target_rgb_points_compare_v1\cam1_local_compare.mp4
D:\node01_spin_runtime_ascii\mvp-demo\output\neoverse_fused_runs\2026-04-26_j10_yp20_r02params_r01\mj_node01_j10_spin_static_yp20_a\preview\local_target_rgb_points_compare_v1\cam2_local_compare.mp4
```

三路相机同步输入 / overlay（叠加显示）：

```text
D:\node01_spin_runtime_ascii\mvp-demo\output\neoverse_fused_runs\2026-04-26_j10_yp20_r02params_r01\mj_node01_j10_spin_static_yp20_a\preview\original_overlay\cam0_overlay.mp4
D:\node01_spin_runtime_ascii\mvp-demo\output\neoverse_fused_runs\2026-04-26_j10_yp20_r02params_r01\mj_node01_j10_spin_static_yp20_a\preview\original_overlay\cam1_overlay.mp4
D:\node01_spin_runtime_ascii\mvp-demo\output\neoverse_fused_runs\2026-04-26_j10_yp20_r02params_r01\mj_node01_j10_spin_static_yp20_a\preview\original_overlay\cam2_overlay.mp4
```

三路相机原始视角对比：

```text
D:\node01_spin_runtime_ascii\mvp-demo\output\neoverse_fused_runs\2026-04-26_j10_yp20_r02params_r01\mj_node01_j10_spin_static_yp20_a\preview\original_compare\cam0_compare.mp4
D:\node01_spin_runtime_ascii\mvp-demo\output\neoverse_fused_runs\2026-04-26_j10_yp20_r02params_r01\mj_node01_j10_spin_static_yp20_a\preview\original_compare\cam1_compare.mp4
D:\node01_spin_runtime_ascii\mvp-demo\output\neoverse_fused_runs\2026-04-26_j10_yp20_r02params_r01\mj_node01_j10_spin_static_yp20_a\preview\original_compare\cam2_compare.mp4
```

Soft preview（软渲染预览）解释图：

```text
D:\node01_spin_runtime_ascii\mvp-demo\output\neoverse_fused_runs\2026-04-26_j10_yp20_r02params_r01\mj_node01_j10_spin_static_yp20_a\preview\gaussian_style_compare_fixed_alpha\sample_rgb_view_mask\cam0_overlay_gaussian_first.png
D:\node01_spin_runtime_ascii\mvp-demo\output\neoverse_fused_runs\2026-04-26_j10_yp20_r02params_r01\mj_node01_j10_spin_static_yp20_a\preview\gaussian_style_compare_fixed_alpha\sample_rgb_view_mask\cam1_overlay_gaussian_first.png
D:\node01_spin_runtime_ascii\mvp-demo\output\neoverse_fused_runs\2026-04-26_j10_yp20_r02params_r01\mj_node01_j10_spin_static_yp20_a\preview\gaussian_style_compare_fixed_alpha\sample_rgb_view_mask\cam2_overlay_gaussian_first.png
```

讲解口径：

- 两栏图左侧是 RGB crop（RGB 裁剪图），右侧是 fused 4D points overlay（融合 4D 点云叠加图）。
- 右侧不是重渲染图，不负责恢复纹理。
- Soft preview 看起来糊，主要因为 NeoVerse 输入分辨率为 `280x168`、目标小、点云稀疏、soft splat（软泼溅渲染）会平均颜色。

### 3.3 结果产物路径

三相机 depth_gt bootstrap 几何输出：

```text
D:\node01_spin_runtime_ascii\mvp-demo\output\neoverse_fused_runs\node01_neoverse_fused_4d_eval_matrix_v1\<scene_id>\points_by_timestamp
```

j10 static 稳定 run：

```text
D:\node01_spin_runtime_ascii\mvp-demo\output\neoverse_fused_runs\2026-04-26_j10_yp20_r02params_r01\mj_node01_j10_spin_static_yp20_a\points_by_timestamp
```

view-count ablation 几何输出：

```text
D:\node01_spin_runtime_ascii\mvp-demo\output\neoverse_fused_runs\node01_neoverse_fused_4d_view_ablation_v1\<cam>\<scene_id>\points_by_timestamp
```

own-depth no-depthGT 几何输出：

```text
D:\node01_spin_runtime_ascii\mvp-demo\output\neoverse_fused_runs\node01_neoverse_own_depth_4d_eval_v1\<scene_id>\points_by_timestamp
D:\node01_spin_runtime_ascii\mvp-demo\output\neoverse_fused_runs\node01_neoverse_own_depth_4d_eval_v1\<scene_id>\fused\dynamic_constraint_meta.json
```

ReID eval（重识别评测）结果：

```text
D:\研究生\grad_project_recon\mvp-demo\output\evals\node01_neoverse_fused_4d_eval_matrix_v1\rgb_only_clip_gtmask_eval_v1\all_queries_vs_all_scenes.json
D:\研究生\grad_project_recon\mvp-demo\output\evals\node01_neoverse_fused_4d_eval_matrix_v1\rgb_neoverse_fused_4d_clip_fpfh_eval_v1\all_queries_vs_all_scenes.json
D:\研究生\grad_project_recon\mvp-demo\output\evals\node01_neoverse_fused_4d_view_ablation_v1
D:\研究生\grad_project_recon\mvp-demo\output\evals\node01_neoverse_own_depth_4d_eval_v1\rgb_neoverse_own_depth_4d_clip_fpfh_eval_v1\all_queries_vs_all_scenes.json
```

如果讲 own-depth（自有深度），建议展示：

```text
dynamic_constraint_meta.json:
uses_scene_depth_gt = false
anchor_depth_source = multiview_rays_only
```

## 4. 结果分析

### 4.1 三线主结果

| 实验线 | embedding dim（嵌入维度） | metric_queries（有效查询数） | mAP | Rank-1 | Rank-5 | 结论 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `RGB-only（仅 RGB）` | 512 | 6 | 1.0 | 1.0 | 1.0 | RGB baseline（基线）可稳定评测 |
| `RGB + NeoVerse depth_gt bootstrap` | 545 | 6 | 1.0 | 1.0 | 1.0 | 几何分支已接入 |
| `RGB + NeoVerse own-depth no-depthGT` | 545 | 6 | 1.0 | 1.0 | 1.0 | 去掉 `scene depth_gt` 后仍可评测 |

分析：

- `metric_queries=6` 表示当前已经不是单 scene（场景） smoke（冒烟测试），而是跨 scene retrieval（跨场景检索）。
- 三条线均为满分，说明当前 `node01` GT-mask bootstrap 对 CLIP 外观检索过于容易。
- 当前结果不能支持“geometry 提升指标”或“own-depth 优于 depth_gt bootstrap”。
- 当前结果能支持“NeoVerse 4D geometry（4D 几何）已经可被 ReID pipeline 稳定消费和评测”。

### 4.2 View-count ablation（视角数量消融）

| view count（视角数量） | RGB-only mAP / R@1 / R@5 | RGB+4D mAP / R@1 / R@5 | metric_queries | 结论 |
| --- | --- | --- | ---: | --- |
| `cam0` | `1.0 / 1.0 / 1.0` | `1.0 / 1.0 / 1.0` | 6 | 单相机可评测 |
| `cam1` | `1.0 / 1.0 / 1.0` | `1.0 / 1.0 / 1.0` | 6 | 单相机可评测 |
| `cam2` | `1.0 / 1.0 / 1.0` | `1.0 / 1.0 / 1.0` | 6 | 单相机可评测 |
| `tri-cam（三相机）` | `1.0 / 1.0 / 1.0` | `1.0 / 1.0 / 1.0` | 6 | 主线 baseline（基线） |

分析：

- 单相机与三相机消融已经完成。
- 当前结果不能支持“三相机已经优于单相机”。
- 该消融的意义是补齐对照基线，说明当前数据难度不足，需要更难场景验证多视角价值。

### 4.3 Own-depth no-depthGT（自有深度、无深度真值）验收

| 检查项 | 结果 |
| --- | --- |
| `points_by_timestamp/meta.json.schema_version` | `neoverse_points_by_timestamp_v1` |
| `points_by_timestamp/index.csv` | `81 rows/scene` |
| `points_by_timestamp/*.npy` | `81 files/scene` |
| `dynamic_constraint_meta.json.uses_scene_depth_gt` | `false` |
| `dynamic_constraint_meta.json.anchor_depth_source` | `multiview_rays_only` |
| `tracks.npy` | `(1,545)` |
| `all_queries_vs_all_scenes.json` | `metric_queries=6, mAP=1.0, Rank-1=1.0, Rank-5=1.0` |

分析：

- 去掉 `scene depth_gt` 后，NeoVerse own-depth 4D geometry 仍能形成合法 `points_by_timestamp`。
- ReID embedding（重识别嵌入特征）维度和元数据保持正常，说明下游消费链路没有被破坏。
- 第一版仍使用 `masks_gt` 和 `rig.json`，因此它是 no-depthGT 实验，不是全自动真实部署实验。

### 4.4 阶段结论边界

可以写：

- 已完成 `node01` 单节点、多身份、多 scene 的 NeoVerse 4D 接入 3D-ReID 可评测矩阵。
- 已完成 RGB-only、depth_gt bootstrap geometry、own-depth no-depthGT geometry 的三线对比。
- 已完成 `cam0/cam1/cam2/tri-cam` view-count ablation（视角数量消融）。
- 去掉 `scene depth_gt` 后，NeoVerse own-depth 4D geometry 仍可被 ReID pipeline 消费和评测。

不建议写：

- 不写“NeoVerse 4D geometry 带来了 Rank/mAP 指标增益”。
- 不写“三相机已经优于单相机”。
- 不写“own-depth 优于 depth_gt bootstrap”。
- 不写“已经完全无真值”，因为第一版仍使用 `masks_gt` 和 `rig.json`。
- 不写“cross-node（跨节点） 3D-ReID 已经完成”。
- 不写“Gaussian-style preview（高斯风格预览）是正式 Gaussian reconstruction（高斯重建）结果”。

建议口头表述：

```text
当前阶段我们已经把 NeoVerse 4D 的 points_by_timestamp 接入到了 3D-ReID pipeline 中，并从单 scene smoke 扩展到了 node01 的 3 identities x 2 scenes 可评测矩阵。在此基础上，我们补齐了两个关键消融：一是 cam0/cam1/cam2 与 tri-cam 的 view-count 对比，二是去掉 scene depth_gt 后使用 NeoVerse own-depth 的三相机几何分支。结果上，RGB-only、旧 depth_gt bootstrap 几何、新 own-depth 几何，以及单相机/三相机分支在当前 GT-mask bootstrap 下都达到 mAP 和 Rank-1 满分。因此本阶段结论不是“几何已经提升指标”，而是“链路已经闭环，去 depth_gt 后仍可评测；下一步需要更真实的 mask（掩码）、cross-node（跨节点）和更难的数据来验证几何与多视角的实际收益”。
```

## 5. 下一步计划

### 5.1 从 GT mask（真值掩码）迁移到 predicted mask（预测掩码）

当前最主要的真值依赖已经从 `depth_gt` 缩小到 `masks_gt + rig.json`。下一步建议先做单变量实验：

```text
masks_gt + own-depth
-> predicted mask + own-depth
```

目标是判断分割误差会如何影响：

- RGB crop 质量
- bbox center ray triangulation
- `points_by_timestamp` 点数和稳定性
- ReID mAP / Rank-1 / Rank-5

### 5.2 启动 node02 / cross-node smoke（跨节点冒烟测试）

当前 node01 单节点跨 scene 已经过于容易，下一步应扩展到跨节点：

- 先在 `node02` 复刻 `j10 / uav1 / su34`，每个 identity 至少 1 个 scene。
- 做 `node01 -> node02` cross-node smoke。
- 通过后扩展到 `3 identities x 2 nodes x 2 scenes` 正式矩阵。

### 5.3 增加评测难度

如果继续出现所有分支满分，应优先增加任务难度，而不是继续优化 preview：

- 增加更多 identity。
- 增加更接近的外观类别。
- 增加姿态、距离、遮挡和视角变化。
- 引入预测 mask 和真实 depth/own-depth 不确定性。
- 在 cross-node 条件下比较 RGB-only、single-cam（单相机）、tri-cam（三相机）、RGB+4D geometry。

### 5.4 保留当前展示边界

后续组会和文档仍保持以下边界：

- Preview（预览图）只用于解释点云和三相机视角，不作为核心指标。
- `fused_scene.glb` 是静态运动包络，不是 4D 动画。
- Gaussian-style preview（高斯风格预览）不是正式 Gaussian reconstruction（高斯重建）。
- 当前阶段的正式贡献是可评测闭环、消融基线和去 depth_gt 可评测验证。
