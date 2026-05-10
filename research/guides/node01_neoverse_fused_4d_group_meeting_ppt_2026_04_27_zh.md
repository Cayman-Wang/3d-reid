# Node01 NeoVerse Fused 4D（NeoVerse 融合 4D）接入 ReID（重识别）组会 PPT 提纲

## 汇报主线

本轮汇报重点不是展示高质量纹理重建，而是说明多相机 NeoVerse fused 4D（NeoVerse 融合 4D）点云已经接入 3D-ReID（3D 重识别）的工程链路。当前策略是外观与几何解耦：原始 RGB crop（RGB 裁剪图）负责纹理外观，`points_by_timestamp`（按时间戳组织的点云）负责 3D 几何信息。

## 第 1 页：阶段目标

标题：多相机 NeoVerse fused 4D（NeoVerse 融合 4D）点云接入 3D-ReID（3D 重识别）

建议文字：

- 当前目标不是生成高质量 textured reconstruction（带纹理重建）。
- 当前目标是把多相机 4D 几何作为 ReID（重识别）的几何分支。
- 下游权威输入固定为 `points_by_timestamp/index.csv + *.npy + meta.json`。
- 当前 `node01` 侧已完成 `3 identities（身份） x 2 scenes（场景）` 的可评测矩阵 bootstrap（自举闭环）。

## 第 2 页：当前链路

建议放流程图：

```text
三相机 RGB / mask / depth
-> NeoVerse fused 4D（NeoVerse 融合 4D）
-> points_by_timestamp（按时间戳组织的点云）
-> tracklet（轨迹片段）
-> RGB crop（RGB 裁剪图） + geometry embedding（几何嵌入特征）
-> ReID（重识别）
```

建议标注：

- RGB crop（RGB 裁剪图）提供外观纹理。
- `points_by_timestamp`（按时间戳组织的点云）提供逐时间戳 3D 点云。
- `fused_scene.glb`（融合场景静态模型）只用于静态汇总查看，不作为 ReID（重识别）输入。

## 第 3 页：当前结果

建议放关键指标：

```text
6 scenes ready for points_by_timestamp contract
schema_version = neoverse_points_by_timestamp_v1
index.csv = 81 rows per scene
*.npy = 81 per scene
metric_queries = 6
RGB-only: mAP = 1.0 / R@1 = 1.0 / R@5 = 1.0
RGB + NeoVerse 4D: mAP = 1.0 / R@1 = 1.0 / R@5 = 1.0
```

建议说明：

- 当前阶段的结果是“可评测矩阵闭环完成”，不是“geometry 指标已提升”。
- `fused_scene.glb` 看起来厚，是 81 个时间戳一起静态显示成运动包络。

## 第 4 页：为什么全局 soft preview（软预览）看起来糊

建议放 `gaussian_style_compare_fixed_alpha` 的画面。

建议文字：

- 当前预览是 Gaussian-style soft point preview（高斯风格软点预览），不是真正 Gaussian（高斯）重建。
- 本机 NeoVerse 实际输入分辨率为 `280x168`，目标在画面中很小。
- 单帧点云稀疏，soft splat（软点扩散投影）会平均颜色，因此无法呈现纹理级细节。

## 第 5 页：外观与几何分工

建议放两栏图：

```text
Original RGB crop（原始 RGB 裁剪图） | RGB + fused 4D points（RGB + 融合 4D 点云）
```

建议文字：

- 外观细节来自原始 RGB crop（RGB 裁剪图）的 CLIP（图文预训练视觉特征）特征。
- `points_by_timestamp`（按时间戳组织的点云）提供 3D 形状、尺度、姿态和跨视角几何约束。
- 点云不是用于重渲染纹理，而是用于几何增强。

本页推荐使用本轮新输出：

```text
D:\node01_spin_runtime_ascii\mvp-demo\output\neoverse_fused_runs\2026-04-26_j10_yp20_r02params_r01\mj_node01_j10_spin_static_yp20_a\preview\local_target_rgb_points_compare_v1
```

## 第 6 页：ReID Smoke（冒烟验证）进展

建议改成“Node01 Eval Matrix（Node01 评测矩阵）结果”：

```text
RGB-only（仅 RGB）:
tracks.npy shape = (1, 512)
metric_queries = 6
mAP = 1.0 / R@1 = 1.0 / R@5 = 1.0

RGB + NeoVerse 4D geometry（RGB + NeoVerse 4D 几何）:
tracks.npy shape = (1, 545)
512 = CLIP, 33 = FPFH
rgb_weight = 1.0
geo_weight = 0.35
metric_queries = 6
mAP = 1.0 / R@1 = 1.0 / R@5 = 1.0
```

建议说明：

- 当前已经不是单 scene（单场景） smoke，而是 `3 identities（身份） x 2 scenes（场景）` 的正式 bootstrap 矩阵。
- 结论应写成“NeoVerse 4D geometry 分支已接入并可评测，但在当前 GT-mask bootstrap 上与 RGB-only 持平”。

## 第 7 页：下一步计划

建议写三步：

```text
Step 1: 启动 node02（节点 02）/ cross-node smoke（跨节点冒烟验证），先复刻 j10 / uav1 / su34 每个 identity 至少 1 个 scene。
Step 2: 扩展到 3 identities（身份） x 2 nodes（节点） x 2 scenes（场景）的跨节点正式矩阵。
Step 3: 继续保留 RGB-only（仅 RGB）基线，与 RGB + NeoVerse 4D geometry（RGB + NeoVerse 4D 几何）做同口径对比。
```

如需单独增加一页“Next ReID Steps（ReID 下一步）”，建议写：

- `node01` 侧 `3 identities（身份） x 2 scenes（场景）` 已完成，下一步不再写“先补齐点云”，而是转向 `node02 / cross-node smoke（跨节点冒烟验证）`。
- 继续使用当前融合策略：`RGB crop（RGB 裁剪图） -> CLIP（图文预训练视觉特征）`，`points_by_timestamp`（按时间戳组织的点云）` -> open3d_fpfh（Open3D 快速点特征直方图）`，融合权重保持 `rgb_weight=1.0 / geo_weight=0.35`。
- `CLIP RGB-only（仅 RGB）` 对照分支已经补齐，当前结果与 `NeoVerse` 几何分支持平，不能写成“已有提升”。
- `Gaussian-style preview（高斯风格预览）` 暂时不要作为正式 `ReID`（重识别）输入；它只用于展示和诊断。

## 建议措辞

可以写：

- 当前阶段的策略是外观与几何解耦：RGB crop（RGB 裁剪图）保留纹理外观，4D 点云提供空间几何。
- Gaussian-style preview（高斯风格预览）仅用于可视化 `points_by_timestamp`（按时间戳组织的点云），不代表真正 Gaussian（高斯）重建质量。
- 当前已完成 `node01` 单节点 `3 identities（身份） x 2 scenes（场景）` 评测矩阵 bootstrap（自举闭环），下一步转向 `cross-node`（跨节点）评测。

不建议写：

- 已经完成真正 Gaussian 重建。
- 点云可以恢复目标纹理。
- NeoVerse 4D geometry 已经带来 Rank/mAP（排序准确率/平均精度均值）提升。
