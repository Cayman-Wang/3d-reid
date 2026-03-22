# 当前进展看板（2026-03-22）

## 1. 结论速览

| 检查项 | 状态 | 结论 |
| --- | --- | --- |
| `M0` 文档与协议冻结 | GREEN | 已完成 |
| `2026-03-16` 准备任务 | GREEN | 已完成，可关单 |
| 正式 `6-scene` 数据落盘 | GREEN | `j10 / uav1 / dji_mavic` 的 `6` 个正式 scene 已全部生成 |
| predicted `depth` / flat `masks` | GREEN | `6` 个正式 scene 已补齐 `cams/cam*/depth/` 与 `cams/cam*/masks/` |
| `RGB-only (CLIP + no geometry)` 正式评测 | GREEN | 正式 JSON 已落盘到 `mvp-demo/output/evals/iciscae_node01_uav_v1/rgb_only/` |
| `RGB-only` 首轮结果质量 | YELLOW | 已形成正式基线，但 `all_formal_queries.json` 仅达 `mAP=0.6389`、`recall@1=0.3333` |
| 当前仓库内可展示汇报材料 | GREEN | `6` 个正式 scene 均已生成 `presentation_assets/`，周收口文档已补齐 |

一句话判断：

`2026-03-16` 到 `2026-03-22` 这周的硬交付已经全部落盘，但当前 `RGB-only` 只能视为正式 baseline，不能直接作为“检索效果已经足够好”的结论；真正的误检索集中在 `uav1 / dji_mavic` 之间，下一步应优先补几何分支而不是继续重复同口径复跑。

## 2. 当前里程碑对照

- 当前权威 benchmark 仍是 `node01` 的 `single-node, cross-scene, track-level retrieval`。
- 正式身份集合固定为 `j10 / uav1 / dji_mavic`，正式 scene 已全部落盘：
  - `mj_node01_j10_line_nodes_a`
  - `mj_node01_j10_circle_xz_b`
  - `mj_node01_uav1_line_nodes_a`
  - `mj_node01_uav1_circle_xz_b`
  - `mj_node01_dji_mavic_line_nodes_a`
  - `mj_node01_dji_mavic_circle_xz_b`
- 每个正式 scene 当前都同时具备：
  - 三路 `cams/cam*/frames/`，每路 `90` 帧
  - `calib/rig.json`
  - `frame_times.csv`
  - `capture_meta.target.identity_id`
  - predicted `cams/cam*/depth/*.npy`
  - flat `cams/cam*/masks/*.png`
  - `tracks/tracklets.json`
  - `embeddings/tracks.npy`
  - `presentation_assets/`

## 3. `2026-03-18` 到 `2026-03-20` 的执行结论

### 3.1 正式采集已封闭

- 已在 `D:\grad_project_ascii\mvp-demo`、`mvp_demo` 环境下补完 `uav1` 与 `dji_mavic` 的 `4` 条正式采集命令。
- 截至 `2026-03-22`，正式 `6-scene` 已全部通过采集后检查：
  - `capture_meta.target.identity_id` 正确；
  - 三路 `frames/` 齐全且每路 `90` 帧；
  - `rig.json` 与 `frame_times.csv` 齐全；
  - scene 名与 manifest 一致。
- 正式采集阶段已关闭，不再存在“`6-scene` 是否落齐”的未决问题。

### 3.2 predicted 输入已补齐

- `Depth Anything V2 Small` 已通过 `HF_ENDPOINT=https://hf-mirror.com` 成功下载并运行，`6` 个正式 scene 的三路相机各生成 `90` 张 predicted depth。
- `SAM2.1 hiera tiny` checkpoint 已下载到 `mvp-demo/third_party/sam2/checkpoints/sam2.1_hiera_tiny.pt`。
- 首轮 prompt box 因为框到了错误的小目标而被废弃；当前正式链路统一冻结以下三路 prompt box，并已回填到 benchmark manifest：
  - `cam0 = [490, 300, 660, 720]`
  - `cam1 = [600, 330, 800, 720]`
  - `cam2 = [600, 270, 800, 640]`
- 为满足正式 benchmark 的 flat mask 约束，当前保留 `mvp-demo/scripts/flatten_node_sam2_masks.py`，把 `obj_000/<ts>.png` 展平为 `cams/cam*/masks/<ts>.png`。
- 当前 `6` 个正式 scene 的每路相机均已具备：
  - `depth = 90/90`
  - `flat masks = 90/90`
  - `nonempty masks = 90/90`
- 当前正式主链已不再依赖 `masks_gt / depth_gt`；GT 目录只保留为 upper-bound 与排错证据。

### 3.3 Windows 路径兼容性已收口

- 已补强以下脚本的 Windows Unicode 路径兼容性：
  - `mvp-demo/scripts/build_node_tracklets.py`
  - `mvp-demo/scripts/extract_node_track_embeddings.py`
  - `mvp-demo/scripts/export_node_presentation_assets.py`
  - `mvp-demo/scripts/run_node_sam2_masks.py`
- 当前正式链路在 ASCII 工作目录下可稳定执行，不需要切到 WSL。

## 4. `2026-03-21` 正式评测结果

### 4.1 全量 summary

- 正式评测文件已落盘：
  - 单 query 文件：`mvp-demo/output/evals/iciscae_node01_uav_v1/rgb_only/mj_node01_*.json`
  - 全量汇总：`mvp-demo/output/evals/iciscae_node01_uav_v1/rgb_only/all_formal_queries.json`
- `all_formal_queries.json` 的 summary 为：
  - `num_queries = 6`
  - `num_gallery = 6`
  - `metric_queries = 6`
  - `mAP = 0.6389`
  - `recall@1 = 0.3333`
  - `recall@5 = 1.0000`
  - `recall@10 = 1.0000`

### 4.2 每条正式 query 的结果

| query_scene | identity_id | traj | mAP | recall@1 | top1_relevant | top1_track |
| --- | --- | --- | --- | --- | --- | --- |
| `mj_node01_j10_line_nodes_a` | `j10` | `line_nodes` | `1.0000` | `1.0000` | `YES` | `node01_mj_node01_j10_circle_xz_b` |
| `mj_node01_j10_circle_xz_b` | `j10` | `circle_xz` | `1.0000` | `1.0000` | `YES` | `node01_mj_node01_j10_line_nodes_a` |
| `mj_node01_uav1_line_nodes_a` | `uav1` | `line_nodes` | `0.5000` | `0.0000` | `NO` | `node01_mj_node01_dji_mavic_line_nodes_a` |
| `mj_node01_uav1_circle_xz_b` | `uav1` | `circle_xz` | `0.5000` | `0.0000` | `NO` | `node01_mj_node01_dji_mavic_circle_xz_b` |
| `mj_node01_dji_mavic_line_nodes_a` | `dji_mavic` | `line_nodes` | `0.5000` | `0.0000` | `NO` | `node01_mj_node01_uav1_line_nodes_a` |
| `mj_node01_dji_mavic_circle_xz_b` | `dji_mavic` | `circle_xz` | `0.3333` | `0.0000` | `NO` | `node01_mj_node01_uav1_circle_xz_b` |

### 4.3 结果解释

- 当前 `RGB-only` 正式链路已经是可复现实验，不再是 smoke 样例。
- `j10` 的两条 query 都能在跨 scene 检索中稳定回到同身份正样本，说明 baseline 至少具备最基本的跨轨迹识别能力。
- 误检索集中在 `uav1 / dji_mavic` 之间，并且主要发生在相同轨迹形态下：
  - `uav1_line_nodes_a -> dji_mavic_line_nodes_a`
  - `uav1_circle_xz_b -> dji_mavic_circle_xz_b`
  - `dji_mavic_line_nodes_a -> uav1_line_nodes_a`
  - `dji_mavic_circle_xz_b -> uav1_circle_xz_b`（正确正样本已退到 `rank3`）
- 这说明 `CLIP + no geometry` 能形成正式 baseline，但对外形接近的飞行器目标仍有明显混淆；下一轮实验应优先验证几何信息是否能够打破这类近邻混淆。

## 5. `2026-03-22` 汇报材料状态

- `6` 个正式 scene 已全部导出 `presentation_assets/`，每个 scene 当前都有：
  - `3` 张 `overview_*.png`
  - `triview_video.mp4`
  - `manifest.json`
- 本周收口文档已生成：
  - `research/reviews/iciscae_week_closure_2026_03_22_zh.md`
- 当前可以直接用于导师汇报的材料包括：
  - 正式 `6-scene` 清单
  - predicted 输入完成证据
  - `RGB-only` 正式结果表
  - 案例图与三视图视频
  - `uav1 / dji_mavic` 混淆的失败分析

## 6. 下一步建议

当前最短路径已经从“补齐正式 scene 与正式结果”切换为“冻结现有 baseline，并尽快补几何支路”。建议顺序如下：

1. 直接使用 `research/reviews/iciscae_week_closure_2026_03_22_zh.md` 写 benchmark protocol、实验设置和结果分析。
2. 在现有 `6-scene` benchmark 上优先推进 `rgb_predicted_depth_geometry`，验证 predicted depth 是否能缓解 `uav1 / dji_mavic` 的近形态混淆。
3. 再进入 `rgb_fused_geometry`，把当前 `presentation_assets` 和失败案例直接复用到论文图表。
4. 不建议继续重复当前 `CLIP + no geometry` 口径复跑，除非更换 backbone 或显著扩充 scene 多样性。
