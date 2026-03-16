# 26-03-16 ICISCAE 小论文本周执行日程

> 存档说明：本文档用于归档 `2026-03-16` 这一周的执行版日程，只服务于本周 `ICISCAE` 小论文收口，不替代 `research/` 主线计划。唯一口径以 `research/plans/ACTIVE_PLAN.md`、`research/plans/tri_camera_node_3d_aware_reid/master_plan_zh.md` 和 `research/plans/tri_camera_node_3d_aware_reid/benchmarks/iciscae_node01_uav_v1.json` 为准。

## 1. 本周成功定义

本周结束时，以下 4 条必须成立：

- `node01` 的正式 `3 identities x 2 scenes` 数据集已经落盘完成。
- `RGB-only` 正式结果已经跑通，并落到 `mvp-demo/output/evals/iciscae_node01_uav_v1/rgb_only/`。
- 小论文正文最少完成：任务定义、benchmark protocol、实验设置、related work、首版结果表。
- `RGB + predicted-depth geometry` 和 `RGB + fused geometry` 只作为加分项，不作为本周硬完成条件。

本周固定说明：

- 公共接口或类型变更：无。
- 本周不做代码架构扩展。
- 本周不做 `cross-node`。
- 本周不训练新的端到端 encoder。
- 本周不继续投入 `YOLO + 3DGS` demo。

## 2. 本周目标

本周 4 个硬目标：

- 完成正式 `6` 个 scene 的采集与命名对齐：`j10 / uav1 / dji_mavic` 各两条轨迹。
- 完成正式 `6` 个 scene 的 predicted `masks` 与 predicted `depth`。
- 完成 `RGB-only` 首轮正式评测，主基线固定为 `CLIP`。
- 完成一版可直接进入小论文的文稿骨架、结果表和案例图清单。

本周 3 个明确不做：

- 不把 `hist / radial_hist` 作为小论文主结果命名。
- 不把 `masks_gt / depth_gt` 作为正式 benchmark 主链输入。
- 不在本周展开 `cross-modal-distillation-reidentification` 或 `point-cloud-reid` 的完整复现。

## 3. 每日安排

### 2026-03-16

目标：冻结本周执行口径并排掉环境风险。

任务：

- 确认 `j10 / uav1 / dji_mavic` 三个资产的可加载路径。
- 确认 `mvp-demo/scripts/extract_node_track_embeddings.py` 本周统一使用 `--rgb_backend clip --geo_backend none` 跑 `RGB-only`。
- 整理 `6` 条正式 scene 采集命令，固定 `scene_id`、`identity_id`、轨迹参数。
- 对照 manifest 检查正式 scene 名是否全部可用：
  - `mj_node01_j10_line_nodes_a`
  - `mj_node01_j10_circle_xz_b`
  - `mj_node01_uav1_line_nodes_a`
  - `mj_node01_uav1_circle_xz_b`
  - `mj_node01_dji_mavic_line_nodes_a`
  - `mj_node01_dji_mavic_circle_xz_b`

输出物：

- `6` 条正式采集命令。
- 一份风险清单。
- 明确 `uav1` 是否需要 ASCII alias 或 WSL。

收尾检查：

- 正式身份不再写成 `person_a`。
- 正式 scene 名与 manifest 完全一致。
- 本周评测命令口径固定为 `CLIP + no geometry`。

### 2026-03-17

目标：完成 `j10` 的两条正式 scene。

任务：

- 采集 `mj_node01_j10_line_nodes_a`。
- 采集 `mj_node01_j10_circle_xz_b`。
- 检查三路 `frames/`、`frame_times.csv`、`calib/rig.json` 是否齐全。
- 检查 `capture_meta.target.identity_id == j10`。

输出物：

- `mvp-demo/data/nodes/node01/scenes/mj_node01_j10_line_nodes_a/`
- `mvp-demo/data/nodes/node01/scenes/mj_node01_j10_circle_xz_b/`

收尾检查：

- 每个 scene 都不是测试名或临时名。
- 每个 scene 都能被后续 `build_node_tracklets.py` 和 `extract_node_track_embeddings.py` 正常识别。

### 2026-03-18

目标：完成 `uav1` 的两条正式 scene。

任务：

- 优先在当前 Windows 路径下尝试采集。
- 若中文资产路径阻塞，当天切换 ASCII alias 或 WSL，不继续拖延。
- 采集 `mj_node01_uav1_line_nodes_a`。
- 采集 `mj_node01_uav1_circle_xz_b`。

输出物：

- `mvp-demo/data/nodes/node01/scenes/mj_node01_uav1_line_nodes_a/`
- `mvp-demo/data/nodes/node01/scenes/mj_node01_uav1_circle_xz_b/`

收尾检查：

- `capture_meta.target.identity_id == uav1`。
- 两个 scene 的命名完全匹配 manifest。
- 若切换到 ASCII alias 或 WSL，命令和路径要被记录下来，后续可复现。

### 2026-03-19

目标：完成 `dji_mavic` 的两条正式 scene，并封闭正式采集阶段。

任务：

- 采集 `mj_node01_dji_mavic_line_nodes_a`。
- 采集 `mj_node01_dji_mavic_circle_xz_b`。
- 若 `dji_mavic` 连续失败，按冻结计划切换到 `big_dji` fallback，但当天必须做出决定，不留悬而未决状态。
- 汇总正式 `6` 个 scene 的目录清单。

输出物：

- `mvp-demo/data/nodes/node01/scenes/mj_node01_dji_mavic_line_nodes_a/`
- `mvp-demo/data/nodes/node01/scenes/mj_node01_dji_mavic_circle_xz_b/`
- 一份正式 `6-scene` 清单。

收尾检查：

- 正式 `6` 个 scene 全部存在。
- 每个 scene 都有三路 `frames/`、`rig.json`、`frame_times.csv`。
- 当天关闭“正式采集是否完成”这个问题，不把采集拖到后半周。

### 2026-03-20

目标：把 `6` 个正式 scene 补成论文主链输入。

任务：

- 对 `6` 个 scene 统一运行 predicted depth。
- 对 `6` 个 scene 统一运行 predicted masks。
- 将任何 `obj_XXX` 结果展平成正式 `cams/cam*/masks/<ts>.png`。
- 做覆盖率检查，保证每个 scene 的有效时间戳不少于 `5`。

输出物：

- `6` 个 scene 的 `cams/cam*/depth/`
- `6` 个 scene 的 `cams/cam*/masks/`

收尾检查：

- 正式 benchmark 主链不再依赖 `masks_gt/` 和 `depth_gt/`。
- 覆盖率不足的 scene 当天标记重跑。
- `mask_layout` 保持为 `flat`。

### 2026-03-21

目标：跑出 `RGB-only` 首轮正式结果。

任务：

- 对 `6` 个正式 scene 统一运行 `build_node_tracklets.py`。
- 用 `extract_node_track_embeddings.py --rgb_backend clip --geo_backend none` 生成正式 `RGB-only` embedding。
- 用 `eval_node_track_retrieval.py` 开启 `--exclude_same_track_id` 和 `--exclude_same_scene` 做正式评测。
- 汇总 `mAP / recall@1 / recall@5 / recall@10`，形成首版结果表。

输出物：

- 每个 scene 的 `tracks/tracklets.json`
- 每个 scene 的 `embeddings/tracks.npy`
- 每个 scene 的 `embeddings/tracks_meta.json`
- `mvp-demo/output/evals/iciscae_node01_uav_v1/rgb_only/*.json`
- 一张首版结果表

收尾检查：

- 每个 query 至少有 `1` 个跨 scene 正样本。
- 结果命名不再使用 `hist + radial_hist` smoke 口径。
- 结果目录与 manifest 的 `eval_out_json` 保持一致。

### 2026-03-22

目标：把本周实验转换成小论文可写材料。

任务：

- 写出 benchmark protocol 初稿。
- 写出实验设置初稿。
- 写出 `RGB-only` 结果分析初稿。
- 写出 related work 初稿。
- 整理 `1` 张正式结果表、`2` 到 `3` 张案例图、`1` 段失败分析。
- 把 `cross-modal-distillation-reidentification` 和 `point-cloud-reid` 仅作为下周几何支路参考写入 related work，不在本周展开复现。

输出物：

- 小论文文稿骨架
- 图表清单
- 下周待办

收尾检查：

- 形成一个可以给导师汇报的“本周已完成 / 下周要补”的包。
- 不再只剩零散日志、命令输出和中间截图。

## 4. 周验收标准

文档层面必须出现以下内容：

- 绝对日期 `2026-03-16` 到 `2026-03-22`，不用“周一到周日”这种相对表述。
- `6` 个正式 scene 名全部写全：`j10 / uav1 / dji_mavic` 各两条。
- `RGB-only` 明确指定 `CLIP`。
- `cross-modal-distillation-reidentification` 和 `point-cloud-reid` 被定位为后续几何基线参考，而不是本周硬交付。
- 文档明确写出本周结束时应看到的目录与结果位置：正式 scene 目录、`rgb_only` eval 目录、首版结果表、文稿草稿。

周末验收按 4 条判断：

- 正式 `6` 个 scene 已存在并可复现。
- `6` 个 scene 都已补齐 predicted `masks` 与 predicted `depth`。
- `RGB-only` 正式 JSON 已落盘。
- 论文正文最少 `4` 个核心小节已经有可读草稿。

建议在周末检查以下目录或产物：

- `mvp-demo/data/nodes/node01/scenes/mj_node01_j10_line_nodes_a/`
- `mvp-demo/data/nodes/node01/scenes/mj_node01_j10_circle_xz_b/`
- `mvp-demo/data/nodes/node01/scenes/mj_node01_uav1_line_nodes_a/`
- `mvp-demo/data/nodes/node01/scenes/mj_node01_uav1_circle_xz_b/`
- `mvp-demo/data/nodes/node01/scenes/mj_node01_dji_mavic_line_nodes_a/`
- `mvp-demo/data/nodes/node01/scenes/mj_node01_dji_mavic_circle_xz_b/`
- `mvp-demo/output/evals/iciscae_node01_uav_v1/rgb_only/`
- 一张正式结果表
- 一份小论文文稿草稿

## 5. 风险与回退

### 5.1 `uav1` 中文路径风险

- 风险：Windows 下中文资产目录可能导致 MuJoCo 无法直接加载。
- 回退：当天切换 ASCII alias 或 WSL，不把问题拖到后半周。

### 5.2 `dji_mavic` 加载稳定性风险

- 风险：`dji_mavic` 可能在加载、渲染或采集时不稳定。
- 回退：若在 `2026-03-19` 仍无法稳定加载，当天启用 `big_dji` fallback，不继续拖延。

### 5.3 SAM2 / depth 覆盖率不足

- 风险：预测 `masks` 或 `depth` 覆盖率过低，会直接影响 tracklet 与正式评测。
- 回退：先做覆盖率检查，不达标的 scene 当天标记重跑；`M4` 两条几何分支延后到下周，不影响本周 `RGB-only` 收口。

### 5.4 `CLIP` 环境缺失

- 风险：`open_clip / torch` 不可用时，脚本会回退到 `hist`，导致结果口径不符合小论文主基线。
- 回退：本周必须优先保证 `CLIP` 环境可用；若 `CLIP` 无法运行，本周 `RGB-only` 结果不能视为正式主结果，只能记为 smoke fallback。

## 6. 本周结束时应该看到什么

如果本周按计划完成，周末应该同时看到：

- 正式 `6` 个 scene 目录全部落盘。
- 每个 scene 都具备 `frames + masks + depth + rig.json + frame_times.csv`。
- `mvp-demo/output/evals/iciscae_node01_uav_v1/rgb_only/` 下已有正式 JSON。
- 一张首版 `RGB-only` 结果表。
- 小论文正文至少已有以下 5 个部分的可读草稿：
  - 任务定义
  - benchmark protocol
  - 实验设置
  - related work
  - 首版结果表与结果分析

一句话总结本周目标：

`这一周的重点不是继续扩展系统，而是把 node01 的正式 benchmark、RGB-only 主结果和论文写作骨架一次性收口。`
