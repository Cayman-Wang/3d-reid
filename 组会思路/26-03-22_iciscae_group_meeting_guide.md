# ICISCAE 组会汇报包使用说明（2026-03-22）

> 存档说明：本文档用于把 `组会思路/26-03-16_iciscae_week_execution_schedule.md` 的执行版周计划，转换成可直接用于组会汇报的口径、素材清单和重导命令；实验唯一口径仍以 `research/reviews/iciscae_week_closure_2026_03_22_zh.md` 和 `mvp-demo/output/evals/iciscae_node01_uav_v1/rgb_only/all_formal_queries.json` 为准。

## 1. 核验结论

- 经仓库目录与结果文件复核，`2026-03-16` 到 `2026-03-22` 的周执行会话都已完成，可以整体关单。
- 当前最适合组会强调的结论不是“系统已经做完”，而是“正式 benchmark、正式 baseline 和正式汇报材料已经一次性收口”。
- 当前正式 `RGB-only` baseline 的全量 summary 为：
  - `mAP = 0.6389`
  - `recall@1 = 0.3333`
  - `recall@5 = 1.0000`
  - `recall@10 = 1.0000`

| 日期 | 周计划目标 | 当前状态 | 仓库内证据 |
| --- | --- | --- | --- |
| `2026-03-16` | 冻结执行口径、排掉环境风险 | 已完成 | 正式采集命令、`CLIP + no geometry` 评测口径、ASCII 路径风险收敛都已写入周计划文档 |
| `2026-03-17` | 完成 `j10` 两条正式 scene | 已完成 | `mvp-demo/data/nodes/node01/scenes/mj_node01_j10_line_nodes_a/` 与 `mvp-demo/data/nodes/node01/scenes/mj_node01_j10_circle_xz_b/` 已落盘 |
| `2026-03-18` | 完成 `uav1` 两条正式 scene | 已完成 | `mvp-demo/data/nodes/node01/scenes/mj_node01_uav1_line_nodes_a/` 与 `mvp-demo/data/nodes/node01/scenes/mj_node01_uav1_circle_xz_b/` 已落盘 |
| `2026-03-19` | 完成 `dji_mavic` 两条正式 scene | 已完成 | `mvp-demo/data/nodes/node01/scenes/mj_node01_dji_mavic_line_nodes_a/` 与 `mvp-demo/data/nodes/node01/scenes/mj_node01_dji_mavic_circle_xz_b/` 已落盘 |
| `2026-03-20` | 补齐 predicted `masks` 与 predicted `depth` | 已完成 | `6` 个正式 scene 都已有 `cams/cam*/masks/*.png` 和 `cams/cam*/depth/*.npy`；汇报时按每路 `90` 张 predicted depth 计，`depth/` 目录中额外的 `depth_meta.json` 不计入帧数 |
| `2026-03-21` | 跑出正式 `RGB-only` 结果 | 已完成 | `mvp-demo/output/evals/iciscae_node01_uav_v1/rgb_only/` 已生成单 query JSON 与 `all_formal_queries.json` |
| `2026-03-22` | 转成论文与组会材料 | 已完成 | `6` 个正式 scene 都已有 `presentation_assets/`，且周收口文档与进展看板已生成 |

补充核验口径：

- 当前 `6` 个正式 scene 都同时具备：
  - 三路 `cams/cam*/frames/`，每路 `90` 帧；
  - `calib/rig.json`；
  - `frame_times.csv`；
  - predicted `cams/cam*/masks/*.png`；
  - predicted `cams/cam*/depth/*.npy`；
  - `tracks/tracklets.json`；
  - `embeddings/tracks.npy` 与 `embeddings/tracks_meta.json`；
  - `presentation_assets/`。
- 因此，这周已经不是“只有命令和日志”，而是形成了完整的“数据-结果-图像-视频-分析”闭环。

## 2. 组会主讲口径

建议把整场组会的主标题讲成：

`ICISCAE 小论文周收口：正式 benchmark、RGB-only baseline 与几何分支切入点`

建议开场直接说清 4 句话：

1. 这周的重点不是继续扩展系统，而是把 `node01` 的正式 benchmark、正式 baseline 和论文写作骨架一次性收口。
2. 当前正式 benchmark 已冻结为 `single-node, cross-scene, track-level retrieval`，身份集合固定为 `j10 / uav1 / dji_mavic`，共 `6` 个正式 scene。
3. 当前 `RGB-only (CLIP + no geometry)` 已经形成正式 baseline，但只能证明“链路可复现、结果可写入论文”，还不能证明“检索质量已经足够好”。
4. 最主要的失败模式集中在 `uav1 / dji_mavic` 之间，这正好说明下周优先补几何分支是合理且必要的。

组会里建议主动避免两种说法：

- 不要说“3D-aware ReID 系统已经完成”；
- 不要说“当前结果已经足够支撑最终论文结论”。

更稳妥的表达是：

`本周已经把正式 benchmark 和正式 baseline 固定下来，并且把后续几何分支所需的证据链和汇报材料都准备齐了。`

## 3. 推荐的 6 页汇报结构

| 页码 | 标题 | 必放内容 | 建议讲法 |
| --- | --- | --- | --- |
| `Slide 1` | 本周目标与收口结论 | `6-scene`、predicted `masks/depth`、`RGB-only` baseline、汇报材料四项硬交付 | 强调“本周是收口，不是扩系统” |
| `Slide 2` | 正式 benchmark 已落盘 | `6` 个正式 scene 名、身份集合、评测协议 | 讲清 benchmark 已冻结，后续所有支路都共用这套口径 |
| `Slide 3` | 主链输入与三视图证据 | `overview_*.png` 或 `triview_video.mp4` | 让老师先直观看到三视图 `RGB / Mask / Depth` 已齐全 |
| `Slide 4` | `RGB-only` 正式结果表 | `mAP / recall@1 / recall@5 / recall@10` | 把它讲成正式 baseline，而不是最终最好结果 |
| `Slide 5` | 失败案例与原因判断 | `uav1 / dji_mavic` 对比图、top1 混淆说明 | 结论要落到“近形态目标需要几何信息” |
| `Slide 6` | 下周最短路径 | `rgb_predicted_depth_geometry`、`rgb_fused_geometry` | 只承诺补几何支路，不重复 `CLIP + no geometry` 复跑 |

逐页建议如下：

### 3.1 `Slide 1`：本周目标与收口结论

- 一句话总结：
  `这周已经把正式 benchmark、正式 baseline 和可汇报材料一起收口。`
- 建议列出四项已完成内容：
  - 正式 `6-scene` 数据已落盘；
  - predicted `masks/depth` 已补齐；
  - `RGB-only` 正式结果已落盘；
  - 图表、案例图和视频素材已可直接复用。

### 3.2 `Slide 2`：正式 benchmark 已落盘

- 这里放 `6` 个正式 scene 的清单，不要只说“3 identities x 2 scenes”。
- 必须强调：
  - 任务是 `single-node, cross-scene, track-level retrieval`；
  - 主链输入是 `frames + masks + depth + rig.json + frame_times.csv`；
  - 本轮正式结果固定为 `CLIP + no geometry`。

### 3.3 `Slide 3`：主链输入与三视图证据

- 这一页建议先放一张 `overview_000001500000.png`。
- 如果现场允许播放视频，再补一个 `triview_video.mp4`。
- 建议讲法：
  `现在不是只有目录结构，而是每个正式 scene 都已经有三视图 RGB、flat mask、predicted depth 和统一时间戳，可以直接作为后续几何分支的输入。`

### 3.4 `Slide 4`：`RGB-only` 正式结果表

- 推荐直接放全量 summary：
  - `mAP = 0.6389`
  - `recall@1 = 0.3333`
  - `recall@5 = 1.0000`
  - `recall@10 = 1.0000`
- 同时列一张每个 query 的结果表，突出：
  - `j10` 的两条 query 都是 `rank1` 正确；
  - `uav1 / dji_mavic` 四条 query 都发生 top1 混淆。

### 3.5 `Slide 5`：失败案例与原因判断

- 这一页最好做成“三联图”：
  - 左：`uav1` query 的 `overview_*.png`
  - 中：误检索到的 `dji_mavic` gallery 的 `overview_*.png`
  - 右：对应 JSON 中的 `top1` 与 `rank3` 结果摘要
- 建议结论只讲到这里：
  `RGB-only` 已经足够成为正式 baseline，但对近形态飞行器目标仍有明显混淆，因此几何支路不是锦上添花，而是下一步必须验证的方向。

### 3.6 `Slide 6`：下周最短路径

- 建议只保留两项：
  - `rgb_predicted_depth_geometry`
  - `rgb_fused_geometry`
- 不建议在这一页承诺：
  - 新训练 backbone；
  - `cross-node` 扩展；
  - 重新重复 `CLIP + no geometry` 多轮复跑。

## 4. 现成可用的图像、视频与表格

如果只是为了组会展示，当前不需要重跑实验，直接复用下面这些现成文件即可。

| 类型 | 推荐文件 | 用途 |
| --- | --- | --- |
| 开场视频 | `mvp-demo/data/nodes/node01/scenes/mj_node01_j10_circle_xz_b/presentation_assets/triview_video.mp4` | 用最直观的方式展示三视图同步采集 |
| 主链输入总览图 | `mvp-demo/data/nodes/node01/scenes/mj_node01_j10_line_nodes_a/presentation_assets/overview_000001500000.png` | 一页展示 `RGB / Mask / Depth` 三行三列证据 |
| 失败案例图一 | `mvp-demo/data/nodes/node01/scenes/mj_node01_uav1_circle_xz_b/presentation_assets/overview_000001500000.png` | 作为 `uav1` query 展示图 |
| 失败案例图二 | `mvp-demo/data/nodes/node01/scenes/mj_node01_dji_mavic_circle_xz_b/presentation_assets/overview_000001500000.png` | 作为 `dji_mavic` 错配 gallery 展示图 |
| 结果表来源 | `mvp-demo/output/evals/iciscae_node01_uav_v1/rgb_only/all_formal_queries.json` | 提取 summary 与每条 query 的检索结果 |
| 汇报文字底稿 | `research/reviews/iciscae_week_closure_2026_03_22_zh.md` | 直接复用 benchmark、实验设置、失败分析的表述 |

关于 `presentation_assets/` 目录，建议记住 3 个事实：

1. 每个正式 scene 当前都有：
   - `3` 张 `overview_*.png`
   - `1` 个 `triview_video.mp4`
   - `1` 个 `manifest.json`
2. `overview_*.png` 是 `cam0 / cam1 / cam2` 三路视角的 `RGB / Mask / Depth` 九宫格总览图。
3. `manifest.json` 会记录：
   - 当前 scene 路径；
   - 导出的关键时间戳；
   - 使用的 `mask_subdir` 与 `depth_subdir`；
   - 视频帧数与 `fps`。

## 5. 如何获取或重导可视化素材

### 5.1 直接使用现成素材

如果只是准备汇报，优先直接打开各个 scene 下的 `presentation_assets/`，不必重新导出。

推荐优先检查这几个目录：

- `mvp-demo/data/nodes/node01/scenes/mj_node01_j10_line_nodes_a/presentation_assets/`
- `mvp-demo/data/nodes/node01/scenes/mj_node01_j10_circle_xz_b/presentation_assets/`
- `mvp-demo/data/nodes/node01/scenes/mj_node01_uav1_circle_xz_b/presentation_assets/`
- `mvp-demo/data/nodes/node01/scenes/mj_node01_dji_mavic_circle_xz_b/presentation_assets/`

### 5.2 单个 scene 重导命令

在仓库根目录下执行：

```powershell
conda run -n mvp_demo python mvp-demo/scripts/export_node_presentation_assets.py `
  --scene_dir mvp-demo/data/nodes/node01/scenes/mj_node01_uav1_line_nodes_a
```

默认行为：

- 输出目录默认为该 scene 下的 `presentation_assets/`；
- 默认相机顺序为 `cam0,cam1,cam2`；
- 默认会从所有同步时间戳里选择 `首帧 / 中帧 / 末帧` 三个关键时间戳；
- 默认画布尺寸为 `1024 x 768`。

### 5.3 一次性重导 `6` 个正式 scene

```powershell
$scenes = @(
  'mj_node01_j10_line_nodes_a',
  'mj_node01_j10_circle_xz_b',
  'mj_node01_uav1_line_nodes_a',
  'mj_node01_uav1_circle_xz_b',
  'mj_node01_dji_mavic_line_nodes_a',
  'mj_node01_dji_mavic_circle_xz_b'
)

foreach ($scene in $scenes) {
  conda run -n mvp_demo python mvp-demo/scripts/export_node_presentation_assets.py `
    --scene_dir "mvp-demo/data/nodes/node01/scenes/$scene"
}
```

### 5.4 指定关键帧或导出更大画布

如果需要更适合 PPT 的大图，可以手动指定关键帧和画布尺寸：

```powershell
conda run -n mvp_demo python mvp-demo/scripts/export_node_presentation_assets.py `
  --scene_dir mvp-demo/data/nodes/node01/scenes/mj_node01_dji_mavic_circle_xz_b `
  --key_stems 000000000000,000001500000,000002966667 `
  --canvas_width 1600 `
  --canvas_height 1200
```

### 5.5 导出脚本的使用注意

- 该脚本只依赖已有的 `frames`、`masks`、`depth` 与 `capture_meta.json`，不会重新跑模型。
- 若某个 scene 没有 predicted `masks/depth`，脚本会自动尝试回退到 `masks_gt/depth_gt`。
- 组会口径里建议继续优先展示正式链路的 predicted `masks/depth`，这样和当前 benchmark 主链一致。

### 5.6 重导后的 smoke check

每个 scene 重导成功后，至少应看到：

- `presentation_assets/overview_*.png` 共 `3` 张；
- `presentation_assets/triview_video.mp4` 共 `1` 个；
- `presentation_assets/manifest.json` 共 `1` 个。

如果组会前只来得及做一次快速检查，优先确认：

1. `triview_video.mp4` 能正常播放；
2. `overview_000001500000.png` 能清楚看到三路 `RGB / Mask / Depth`；
3. `manifest.json` 中的 `generated_files`、`num_video_frames` 与 `fps` 合理。

## 6. 组会前 5 分钟检查清单

- 打开 `mvp-demo/output/evals/iciscae_node01_uav_v1/rgb_only/all_formal_queries.json`，确认 `mAP = 0.6389` 与 `recall@1 = 0.3333`。
- 打开一段 `triview_video.mp4`，确认视频可正常播放。
- 打开一张 `overview_000001500000.png`，确认 `RGB / Mask / Depth` 面板显示正常。
- 准备一页 `uav1 / dji_mavic` 的对比图，避免只报 summary 不解释失败模式。
- 汇报时统一口径：
  - `RGB-only` 是正式 baseline；
  - 不是最终最好结果；
  - 下周优先补几何分支，不重复同口径复跑。
