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

当日执行记录（`2026-03-16` 已完成）：

- `mvp_demo` 环境已验证可用：`torch 2.10.0+cu128`、`open_clip 3.3.0`、`mujoco 3.6.0` 可正常导入。
- `mvp-demo/scripts/extract_node_track_embeddings.py` 已核对参数口径：
  - `--rgb_backend` 支持 `clip`
  - `--geo_backend` 支持 `none`
  - 本周 `RGB-only` 正式命令固定为 `CLIP + no geometry`，不再把 `hist / radial_hist` 当论文主结果。
- manifest 中的 `6` 个正式 scene 名已逐项核对，命名冻结如下，且当前正式目录仍未落盘：
  - `mj_node01_j10_line_nodes_a`
  - `mj_node01_j10_circle_xz_b`
  - `mj_node01_uav1_line_nodes_a`
  - `mj_node01_uav1_circle_xz_b`
  - `mj_node01_dji_mavic_line_nodes_a`
  - `mj_node01_dji_mavic_circle_xz_b`
- `j10 / uav1 / dji_mavic` 三个 MJCF 已在 `mvp_demo` 环境下通过 MuJoCo 加载测试，但该结论仅在 ASCII junction 路径 `D:\grad_project_ascii\mvp-demo` 下成立；直接使用中文仓库根路径 `D:\研究生\grad_project` 仍会触发 MuJoCo `ParseXML` 打开失败。
- `uav1` 的中文资产路径问题已做本地修复：
  - 新增 ASCII-safe 资产别名目录：`mvp-demo/assets/models/uav1_ascii/`
  - `mvp-demo/assets/scene/mujoco_humanoid_uav1_3cam_node_parallel.xml` 已改为引用该别名目录
  - `uav1` 最小 smoke capture 已跑通，产物位于 `mvp-demo/tmp/smoke_capture/node01/scenes/smoke_uav1_ascii/`
- 结论：`uav1` 本周不需要切到 WSL；在 Windows 下使用 `D:\grad_project_ascii\mvp-demo + mvp_demo` 即可继续正式采集。

本日冻结的正式采集命令：

说明：以下命令统一在 `D:\grad_project_ascii\mvp-demo` 工作目录下执行，环境统一为 `mvp_demo`。

```powershell
conda run -n mvp_demo python scripts/mj_capture_3cam_node.py `
  --mjcf assets/scene/mujoco_humanoid_3cam_node_parallel_j10.xml `
  --out_root data/nodes `
  --node_id node01 `
  --scene_id mj_node01_j10_line_nodes_a `
  --identity_id j10 `
  --traj line_nodes `
  --traj_from_body node01 `
  --traj_to_body node02 `
  --mid_y 6.0 `
  --mid_z 2.0 `
  --traj_period 8.0 `
  --save_depth `
  --save_masks_gt
```

```powershell
conda run -n mvp_demo python scripts/mj_capture_3cam_node.py `
  --mjcf assets/scene/mujoco_humanoid_3cam_node_parallel_j10.xml `
  --out_root data/nodes `
  --node_id node01 `
  --scene_id mj_node01_j10_circle_xz_b `
  --identity_id j10 `
  --traj circle_xz `
  --traj_center "0 6 2" `
  --traj_radius 1.0 `
  --traj_period 6.0 `
  --save_depth `
  --save_masks_gt
```

```powershell
conda run -n mvp_demo python scripts/mj_capture_3cam_node.py `
  --mjcf assets/scene/mujoco_humanoid_uav1_3cam_node_parallel.xml `
  --out_root data/nodes `
  --node_id node01 `
  --scene_id mj_node01_uav1_line_nodes_a `
  --identity_id uav1 `
  --traj line_nodes `
  --traj_from_body node01 `
  --traj_to_body node02 `
  --mid_y 6.0 `
  --mid_z 2.0 `
  --traj_period 8.0 `
  --save_depth `
  --save_masks_gt
```

```powershell
conda run -n mvp_demo python scripts/mj_capture_3cam_node.py `
  --mjcf assets/scene/mujoco_humanoid_uav1_3cam_node_parallel.xml `
  --out_root data/nodes `
  --node_id node01 `
  --scene_id mj_node01_uav1_circle_xz_b `
  --identity_id uav1 `
  --traj circle_xz `
  --traj_center "0 6 2" `
  --traj_radius 1.0 `
  --traj_period 6.0 `
  --save_depth `
  --save_masks_gt
```

```powershell
conda run -n mvp_demo python scripts/mj_capture_3cam_node.py `
  --mjcf assets/scene/mujoco_humanoid_dji_mavic_3cam_node_parallel.xml `
  --out_root data/nodes `
  --node_id node01 `
  --scene_id mj_node01_dji_mavic_line_nodes_a `
  --identity_id dji_mavic `
  --traj line_nodes `
  --traj_from_body node01 `
  --traj_to_body node02 `
  --mid_y 6.0 `
  --mid_z 2.0 `
  --traj_period 8.0 `
  --save_depth `
  --save_masks_gt
```

```powershell
conda run -n mvp_demo python scripts/mj_capture_3cam_node.py `
  --mjcf assets/scene/mujoco_humanoid_dji_mavic_3cam_node_parallel.xml `
  --out_root data/nodes `
  --node_id node01 `
  --scene_id mj_node01_dji_mavic_circle_xz_b `
  --identity_id dji_mavic `
  --traj circle_xz `
  --traj_center "0 6 2" `
  --traj_radius 1.0 `
  --traj_period 6.0 `
  --save_depth `
  --save_masks_gt
```

本周固定评测命令口径：

```powershell
conda run -n mvp_demo python scripts/extract_node_track_embeddings.py `
  --scene_dir data/nodes/node01/scenes/<scene_id> `
  --rgb_backend clip `
  --geo_backend none
```

```powershell
conda run -n mvp_demo python scripts/eval_node_track_retrieval.py `
  --query_scene_dir data/nodes/node01/scenes/<query_scene_id> `
  --gallery_scene_dir data/nodes/node01/scenes/<gallery_scene_id> `
  --exclude_same_track_id `
  --exclude_same_scene `
  --out output/evals/iciscae_node01_uav_v1/rgb_only/<run_name>.json
```

本日风险清单与当前状态：

- `uav1` 路径风险：已从“内部模型路径含中文”收敛为“执行 MuJoCo 命令时必须使用 ASCII junction 工作目录”；当前不需要 WSL。
- `dji_mavic` 加载风险：当前加载测试已通过，但正式采集稳定性仍要在 `2026-03-19` 当天复核。
- SAM2 / depth 覆盖率风险：今天尚未进入正式 `6-scene` 预测阶段，风险未解除。
- CLIP 环境缺失风险：今天已解除，`mvp_demo` 环境中 `open_clip` 可正常导入。

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

执行记录（于 `2026-03-22` 补执行并补记，`2026-03-17` 任务已完成）：

- 已在 `D:\grad_project_ascii\mvp-demo` 工作目录、`mvp_demo` 环境下执行冻结的两条 `j10` 正式采集命令。
- 已生成正式目录：
  - `mvp-demo/data/nodes/node01/scenes/mj_node01_j10_line_nodes_a/`
  - `mvp-demo/data/nodes/node01/scenes/mj_node01_j10_circle_xz_b/`
- 两个 scene 均已通过采集后检查：
  - 三路 `cams/cam*/frames/` 已生成，当前每路相机各 `90` 帧。
  - `frame_times.csv` 已生成。
  - `calib/rig.json` 已生成。
  - `capture_meta.target.identity_id == j10`。
- 本次执行仍遵守本周冻结口径：
  - 工作目录使用 ASCII 路径，未在中文仓库路径下直接执行 MuJoCo。
  - scene 名使用 manifest 冻结的正式命名，未复用历史临时 scene。
- 结论：`2026-03-17` 的 `j10` 正式采集任务可以关单，下一步可推进 `2026-03-18` 的 `uav1` 两条正式 scene 采集。

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

执行记录（于 `2026-03-22` 补执行并补记，`2026-03-18` 任务已完成）：

- 已在 `D:\grad_project_ascii\mvp-demo` 工作目录、`mvp_demo` 环境下执行冻结的两条 `uav1` 正式采集命令，生成：
  - `mvp-demo/data/nodes/node01/scenes/mj_node01_uav1_line_nodes_a/`
  - `mvp-demo/data/nodes/node01/scenes/mj_node01_uav1_circle_xz_b/`
- 两个 scene 均已通过采集后检查：
  - `capture_meta.target.identity_id == uav1`
  - 三路 `cams/cam*/frames/` 各 `90` 帧
  - `frame_times.csv` 已生成
  - `calib/rig.json` 已生成
- 本次执行未切到 `WSL`；继续沿用 `ASCII alias + ASCII junction` 的冻结口径即可稳定运行。
- 结论：`2026-03-18` 的 `uav1` 正式采集任务可以关单，下一步推进 `2026-03-19` 的 `dji_mavic` 两条正式 scene。

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

执行记录（于 `2026-03-22` 补执行并补记，`2026-03-19` 任务已完成）：

- 已在 `D:\grad_project_ascii\mvp-demo` 工作目录、`mvp_demo` 环境下执行冻结的两条 `dji_mavic` 正式采集命令，生成：
  - `mvp-demo/data/nodes/node01/scenes/mj_node01_dji_mavic_line_nodes_a/`
  - `mvp-demo/data/nodes/node01/scenes/mj_node01_dji_mavic_circle_xz_b/`
- `dji_mavic` 本轮未触发 `big_dji` fallback；当前纯材质版 MJCF 可稳定完成正式采集。
- 截至 `2026-03-22`，正式 `6-scene` 清单已全部落齐：
  - `mj_node01_j10_line_nodes_a`
  - `mj_node01_j10_circle_xz_b`
  - `mj_node01_uav1_line_nodes_a`
  - `mj_node01_uav1_circle_xz_b`
  - `mj_node01_dji_mavic_line_nodes_a`
  - `mj_node01_dji_mavic_circle_xz_b`
- 结论：`2026-03-19` 的正式采集阶段已封闭，后续不再存在“正式 scene 是否落齐”的未决问题。

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

执行记录（于 `2026-03-22` 补执行并补记，`2026-03-20` 任务已完成）：

- 已对正式 `6` 个 scene 全部补齐 predicted `depth` 与 predicted `masks`：
  - predicted depth 使用 `Depth Anything V2 Small`
  - predicted masks 使用 `SAM2.1 hiera tiny`
- `Depth Anything V2` 的下载通过 `HF_ENDPOINT=https://hf-mirror.com` 成功解决 Hugging Face 直连超时问题。
- `SAM2.1` checkpoint 已补到 `mvp-demo/third_party/sam2/checkpoints/sam2.1_hiera_tiny.pt`；首轮错误 prompt box 已废弃，当前正式 `6-scene` 统一冻结以下三路 prompt box，并已回填到 `research/plans/tri_camera_node_3d_aware_reid/benchmarks/iciscae_node01_uav_v1.json`：
  - `cam0 = [490, 300, 660, 720]`
  - `cam1 = [600, 330, 800, 720]`
  - `cam2 = [600, 270, 800, 640]`
- 为满足正式 `flat masks` 协议，当前使用：
  - `mvp-demo/scripts/flatten_node_sam2_masks.py`
- 当前 `6` 个 scene 的每路相机均已具备：
  - `cams/cam*/depth/` 下 `90` 张 `.npy`
  - `cams/cam*/masks/` 下 `90` 张 flat `.png`
  - `nonempty masks = 90/90`
- 当前主链已切换为 `frames + masks + depth + rig.json + frame_times.csv`；`masks_gt / depth_gt` 只保留为 upper-bound 与排错用途。


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

执行记录（于 `2026-03-22` 补执行并补记，`2026-03-21` 任务已完成）：

- 已对正式 `6` 个 scene 统一执行：
  - `build_node_tracklets.py --mask_subdir masks --depth_subdir depth --min_timestamps 5`
  - `extract_node_track_embeddings.py --rgb_backend clip --geo_backend none`
  - `eval_node_track_retrieval.py --exclude_same_track_id --exclude_same_scene`
- 当前每个正式 scene 均已生成：
  - `tracks/tracklets.json`
  - `embeddings/tracks.npy`
  - `embeddings/tracks_meta.json`
- 正式 `RGB-only` 结果目录已落盘到：
  - `mvp-demo/output/evals/iciscae_node01_uav_v1/rgb_only/`
- 全量汇总文件：
  - `mvp-demo/output/evals/iciscae_node01_uav_v1/rgb_only/all_formal_queries.json`
- 当前首轮正式 summary 为：
  - `mAP = 0.6389`
  - `recall@1 = 0.3333`
  - `recall@5 = 1.0000`
  - `recall@10 = 1.0000`
- `j10` 的两条 query 当前均能正确回到同身份跨 scene 正样本。
- 当前最主要的误检索集中在 `uav1 / dji_mavic` 之间：
  - `mj_node01_uav1_line_nodes_a -> top1 = node01_mj_node01_dji_mavic_line_nodes_a`
  - `mj_node01_uav1_circle_xz_b -> top1 = node01_mj_node01_dji_mavic_circle_xz_b`
  - `mj_node01_dji_mavic_line_nodes_a -> top1 = node01_mj_node01_uav1_line_nodes_a`
  - `mj_node01_dji_mavic_circle_xz_b -> top1 = node01_mj_node01_uav1_circle_xz_b`（正确正样本已退到 `rank3`）
- 结论：`2026-03-21` 的正式 `RGB-only` 结果已经形成，但它只能作为正式 baseline，不能直接当作“检索质量已经足够好”的结论；下一步必须用论文文字和几何分支去解释并缓解当前混淆。


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

执行记录（`2026-03-22` 当天已完成）：

- 已为正式 `6` 个 scene 全部导出 `presentation_assets/`：
  - 每个 scene 各有 `3` 张 `overview_*.png`
  - 每个 scene 各有 `triview_video.mp4`
  - 每个 scene 各有 `manifest.json`
- 已生成周收口文档：
  - `research/reviews/iciscae_week_closure_2026_03_22_zh.md`
- 已更新本周状态板：
  - `research/reviews/current_progress_board_2026_03_22_zh.md`
- 当前可直接用于导师汇报的包已经具备：
  - 正式 `6-scene` 清单
  - predicted `masks/depth` 完成证据
  - `RGB-only` 首轮结果表
  - `presentation_assets` 案例图与三视图视频
  - `uav1 / dji_mavic` 混淆的失败分析与后续改进方向
- 结论：本周目标已经从“执行版周计划”收口成“可汇报、可写论文、可继续补几何支路”的完整包。

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
