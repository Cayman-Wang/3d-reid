# 当前进展看板（2026-03-22）

## 1. 结论速览

| 检查项 | 状态 | 结论 |
| --- | --- | --- |
| `M0` 文档与协议冻结 | GREEN | 已完成 |
| `2026-03-16` 准备任务 | GREEN | 已完成，可关单 |
| `2026-03-17` `j10` 正式采集与下游 smoke | GREEN | 已于 `2026-03-22` 完成，两个正式 scene 的 `capture + tracklets + embeddings smoke` 均已通过 |
| 正式 `6-scene` 数据落盘 | YELLOW | 已完成 `j10` 的 `2/6`，其余 `uav1 / dji_mavic` 仍未完成 |
| predicted `depth` / `masks` | RED | 尚未开始正式批量产出 |
| `RGB-only (CLIP + no geometry)` 正式评测 | RED | 尚未产出正式结果目录 |
| 当前仓库内可展示原型 | YELLOW | 已有 proof-of-pipeline，但不能当正式 benchmark 结果 |

一句话判断：

`2026-03-16` 的准备性任务已经完成，`2026-03-17` 的 `j10` 两条正式 scene 已于 `2026-03-22` 在 ASCII 工作目录下完成 `capture + tracklets + embeddings smoke` 全链路验证；当前可以继续推进 `2026-03-18` 的 `uav1` 两条正式 scene 采集。

## 2. 当前里程碑对照

- 当前权威里程碑仍是 `M1 第三个飞行目标落地与 node01 正式 3x2 benchmark 采集`。
- 正式 benchmark 已冻结为 `3 identities x 2 scenes`，身份集合固定为 `j10 / uav1 / dji_mavic`。
- 截至 `2026-03-22`，manifest 中定义的 `6` 个正式 scene 目录已实际落盘 `2` 个，分别是：
  - `mj_node01_j10_line_nodes_a`
  - `mj_node01_j10_circle_xz_b`
- 仓库内已有若干历史 scene，可证明链路跑通，但它们仍带有以下问题：
  - scene 名不是正式命名；
  - `identity_id` 仍出现 `person_a`；
  - 已有 embedding 口径仍包含 `hist + radial_hist`；
  - 有的 scene 仍依赖 `depth_gt / masks_gt`。

## 3. `2026-03-16` 任务检查

| `2026-03-16` 任务 | 期望结果 | 实际检查 | 状态 |
| --- | --- | --- | --- |
| 确认 `j10 / uav1 / dji_mavic` 三个资产可加载路径 | 三个 MJCF 都有稳定可用的运行路径 | 本次复验中，三个 MJCF 在 `D:\\grad_project_ascii\\mvp-demo` 下均可被 `MuJoCo 3.6.0` 成功加载；`uav1` 场景已改用 `uav1_ascii` 资产别名 | GREEN |
| 确认本周 `RGB-only` 命令口径 | `extract_node_track_embeddings.py` 支持 `--rgb_backend clip --geo_backend none`，并且环境可导入 `open_clip` | 脚本参数已支持 `clip / none`；`mvp_demo` 环境已复验可导入 `torch 2.10.0+cu128`、`mujoco 3.6.0`、`open_clip 3.3.0` | GREEN |
| 冻结 `6` 条正式 scene 命名与采集命令 | `scene_id`、`identity_id`、轨迹参数全部固定 | 周计划文档已经写全 `6` 条命令；scene 名与 manifest 一致 | GREEN |
| 输出风险清单 | 至少明确路径、资产稳定性、SAM2/depth、CLIP 环境风险 | 周计划文档已记录风险与回退策略 | GREEN |
| 明确 `uav1` 是否需要 ASCII alias 或 WSL | 给出唯一执行口径 | 结论已固定为：使用 `ASCII alias + ASCII junction`，当前不需要切到 `WSL` | GREEN |

### 3.1 关键证据

- 权威主计划和下一步动作已冻结在 `research/plans/ACTIVE_PLAN.md`。
- 正式 `6-scene` manifest 已冻结在 `research/plans/tri_camera_node_3d_aware_reid/benchmarks/iciscae_node01_uav_v1.json`。
- `uav1` ASCII alias 已写入 `mvp-demo/assets/scene/mujoco_humanoid_uav1_3cam_node_parallel.xml`。
- `uav1` smoke capture 已存在于 `mvp-demo/tmp/smoke_capture/node01/scenes/smoke_uav1_ascii/`，其 `capture_meta.json` 中 `identity_id = uav1`。
- `2026-03-16` 的执行记录、风险清单和 `6` 条正式命令已收口到 `组会思路/26-03-16_iciscae_week_execution_schedule.md`。

### 3.2 对 `2026-03-16` 的结论

- 从“准备任务是否完成”这个角度看：`已完成`。
- 从“正式数据是否已经开始落盘”这个角度看：`已开始`，当前已完成 `j10` 的 `2` 条正式 scene。

## 4. `2026-03-17` 任务执行结论

结论：`已完成，且已达到完整通过标准`。

### 4.1 本次执行结果

- 已在 `D:\\grad_project_ascii\\mvp-demo` 工作目录、`mvp_demo` 环境下执行冻结命令，生成：
  - `mvp-demo/data/nodes/node01/scenes/mj_node01_j10_line_nodes_a/`
  - `mvp-demo/data/nodes/node01/scenes/mj_node01_j10_circle_xz_b/`
- 两个 scene 的 `capture_meta.target.identity_id` 均为 `j10`。
- 两个 scene 的 `calib/rig.json` 与 `frame_times.csv` 均已生成。
- 两个 scene 三路相机 `cams/cam*/frames/` 均已生成，当前每路相机各有 `90` 帧。
- 使用 `masks_gt + depth_gt` 做 GT smoke 时，`build_node_tracklets.py` 已成功为两个 scene 各生成 `1` 条 track，`identity_id` 均为 `j10`，`timestamp_stems` 数量均为 `90`。
- `extract_node_track_embeddings.py --rgb_backend clip --geo_backend none` 也已在两个 scene 上成功跑通，产出 `embeddings/tracks.npy` 与 `embeddings/tracks_meta.json`，并确认元信息为 `rgb_backend = clip`、`geo_backend = none`。

### 4.2 仍需保持的执行约束

1. 工作目录必须使用 `D:\\grad_project_ascii\\mvp-demo`，不要直接在中文仓库路径下跑 MuJoCo 正式采集。
2. 后续 `uav1 / dji_mavic` 正式采集仍只能使用 manifest 冻结的正式 scene 名。
3. `capture_meta.target.identity_id` 必须继续作为正式 benchmark 的身份权威来源，不能回退到 `person_a`。
4. 后续正式结果仍不能用历史临时 scene 替代。
5. 每次采集完成后仍需立刻检查：
   - 三路 `frames/`
   - `frame_times.csv`
   - `calib/rig.json`
   - `capture_meta.target.identity_id`

## 5. 当前不能误判为“已完成”的事项

以下事项截至 `2026-03-22` 仍不能算完成：

- manifest 中 `6` 个正式 scene 目录仍未全部落齐；当前只完成 `j10` 的 `2` 个 scene。
- `mvp-demo/output/evals/iciscae_node01_uav_v1/rgb_only/` 尚未形成正式结果目录。
- 当前仓库内检索结果仍是历史样例，不是正式 `RGB-only (CLIP + no geometry)` 结果。
- 当前仓库内已有样例 scene 仍存在 `identity_id = person_a` 的旧口径。
- 当前仓库内已有 embedding 结果仍可见 `rgb_backend = hist`、`geo_backend = radial_hist` 的旧实验配置。

## 6. 建议的下一步

最短路径已经从“执行 `2026-03-17` 的两条 `j10` 正式采集命令”切换为“继续推进 `2026-03-18` 的两条 `uav1` 正式采集命令”。下一步仍应沿同一检查模板收尾：

1. `mj_node01_uav1_line_nodes_a/` 是否完整生成
2. `mj_node01_uav1_circle_xz_b/` 是否完整生成
3. 两个 scene 的 `capture_meta.target.identity_id` 是否都是 `uav1`
4. 两个 scene 是否都能被后续 `build_node_tracklets.py` 正常识别
5. 两个 scene 是否都能被 `extract_node_track_embeddings.py --rgb_backend clip --geo_backend none` 正常消费

如果这五项都成立，就可以继续推进 `2026-03-19` 的 `dji_mavic` 两条正式 scene 采集。

## 7. `2026-03-17` 实测验证结果

本轮按 `2026-03-17` 验证方案对 `j10` 的两条正式 scene 做了实测复核，结果如下：

| scene_id | identity_id | frames_per_cam | unique_timestamps | tracklets_ok | embeddings_ok | notes |
| --- | --- | --- | --- | --- | --- | --- |
| `mj_node01_j10_line_nodes_a` | `j10` | `90 / 90 / 90` | `90` | YES | YES | `rgb_backend=clip`, `geo_backend=none` |
| `mj_node01_j10_circle_xz_b` | `j10` | `90 / 90 / 90` | `90` | YES | YES | `rgb_backend=clip`, `geo_backend=none` |

补充结论：

- 两个正式 scene 的目录名、`scene_id`、`target.identity_id`、轨迹类型都符合 `2026-03-17` 计划要求。
- 两个 scene 都具备 `frames/`、`frame_times.csv`、`calib/rig.json`，且 `frame_times.csv` 都包含 `270` 行、`90` 个唯一时间戳、`3` 个相机。
- 使用 `masks_gt + depth_gt` 做 GT smoke 时，`build_node_tracklets.py` 已成功为两个 scene 各生成 `1` 条 track，`identity_id` 均为 `j10`，`timestamp_stems` 数量均为 `90`。
- `extract_node_track_embeddings.py --rgb_backend clip --geo_backend none` 已在两个 scene 上成功跑通；本次通过预先下载本地 CLIP 权重文件并显式传入 `--clip_pretrained`，规避了在线拉取 Hugging Face 权重的超时问题。

因此：

- 按“最低通过”标准，`2026-03-17` 已通过。
- 按“完整通过”标准，`2026-03-17` 现也已通过。
