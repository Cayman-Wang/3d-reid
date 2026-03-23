# tri_camera_node_3d_aware_reid 主计划（中文）

- 创建日期：2026-03-15
- 最近冻结：2026-03-23
- 状态：ICISCAE 范围固定为 `node01 + UAV/aircraft + single-node cross-scene`，当前主线为 `v3_clean benchmark: j10 / uav1 / su34`，已完成 clean `6 scene` 采集与四条结果线

## 一、冻结结论

- 研究主线固定为“三相机节点级 3D-aware track retrieval”，不再把 YOLO 门控 + 3DGS 作为主方法。
- `ICISCAE 小论文` 固定做 `node01` 下的 `single-node, cross-scene, track-level retrieval`。
- `毕业论文主线` 固定做 `cross-node 3D re-ID` 与真实三相机迁移，但不与当前保底投稿绑定。
- 小论文目标域固定为 `UAV/aircraft`，不再做人形 benchmark。
- 小论文允许只做 MuJoCo 仿真，但标题、摘要和结论只能写成“仿真三相机节点检索验证”，不能提前宣称跨节点已完成。
- 研究口径、工程 handoff 和 benchmark 统一收口到 `research/`；`mvp-demo/` 仅保留运行入口与资产说明。

## 二、研究命题与当前主问题

本项目研究的不是传统行人闭集识别，而是一个更一般的实例级跨视角检索问题：

- 输入：每个节点由 3 台同步相机构成，连续采集目标在视野中的运动过程。
- 中间表示：将同一目标在一个采集窗口中的多帧、多相机观测组织为一个 `tracklet`。
- 输出：为每个 `tracklet` 生成稳定的 `track embedding`，并在其他 scene 或其他节点中做检索。

当前阶段的主问题是：

> 在不使用 MuJoCo GT 参与主检索计算的前提下，`RGB + depth + mask + rig + timestamps` 是否足以支持稳定的 3D-aware track retrieval。

### 2.1 为什么只做 `node01` 仍然是 Re-ID

`single-node` 并不等于“不是 re-ID”。

当前阶段的检索任务被拆成了一个更小但更干净的问题：

- query：来自 `scene A` 的一个 `track`
- gallery：来自 `scene B/C/...` 的其他 `track`
- 正样本：其他 scene 中与 query 拥有相同 `identity_id` 的 `track`
- 约束：评测时排除同一 `track`，同时排除同一 `scene`

因此当前验证的是：

> 同一目标在不同 scene、不同时间窗、不同三相机观测组合下，能否被重新找回。

这仍然是标准的 re-ID / retrieval 问题，只是先把“节点差异”这个变量拿掉，用来验证表征本身是否成立。

## 三、两层论文对照表

| 维度 | ICISCAE 小论文 | 毕业论文主线 |
| --- | --- | --- |
| 核心任务 | `node01` 的 `single-node, cross-scene, track-level retrieval` | `cross-node 3D re-ID` + 真实节点迁移 |
| 目标域 | `UAV/aircraft` | 延续 `UAV/aircraft`，后续可扩到更丰富目标 |
| 数据来源 | MuJoCo-only 可接受 | MuJoCo + 至少一部分真实三相机数据 |
| benchmark 最小规模 | `3 identities x 2 scenes` | 至少 `3 identities x 2 nodes x 多 scene` |
| 研究问题 | 几何信息是否提升单节点跨 scene 检索 | 表征能否抗节点差异和真实域偏移 |
| 主结果 | `RGB-only`、`RGB + predicted-depth geometry`、`RGB + fused geometry` | 在小论文三组基础上补 `cross-node` 主结果、真实迁移结果和系统误差分析 |
| 允许缺失 | 可以没有 `node02` 和真实数据 | 不能长期停留在单节点仿真 |
| 成功定义 | benchmark、三组结果、失败分析和 GT 上界完整 | 在上述基础上补齐跨节点和真实迁移证据 |

## 四、当前方法收口

当前主线只保留与正式 benchmark 直接相关的三条结果线：

| 结果线 | 输入契约 | 当前口径 |
| --- | --- | --- |
| `RGB-only` | `frames + tracklets` | 当前第一优先级，作为正式基线 |
| `RGB + predicted-depth geometry` | `frames + masks + predicted depth + rig` | 作为弱几何分支，验证预测深度本身是否带来增益 |
| `RGB + fused geometry` | `frames + masks + predicted depth + rig + fused points` | 作为强几何分支，验证多相机融合后的几何收益 |

固定说明如下：

- RGB 主基线固定为 `CLIP`。
- `hist` 和 `radial_hist` 只保留为 smoke fallback，不作为小论文主结果命名。
- 当前工程里已经直接具备 `RGB-only` 与 `fused geometry` 的入口；`predicted-depth geometry` 继续沿同一数据契约补齐，不改变 benchmark 定义。
- `YOLO 门控 + 3DGS` 相关脚本只保留为辅助 demo，不再参与当前主线里程碑。

## 五、当前工程能力与证据现状

### 5.1 已具备的主链

- 当前仓库已经具备节点级主链脚本：`mj_capture_3cam_node.py -> run_node_depth_anything_v2.py -> run_node_sam2_masks.py -> build_node_tracklets.py -> extract_node_track_embeddings.py -> eval_node_track_retrieval.py`。
- 当前仓库已经具备几何增强入口：`recon_fuse_depth_points.py` 可用于多相机 `depth + mask` 融合点云。
- 当前仓库同时保留一条 `YOLO 门控 + 3DGS` 静态场景 demo 链，但该链只保留为辅助演示，不纳入主线里程碑。

### 5.2 当前本地证据

- 当前 `v3_clean` 的 `6` 条正式 scene 已全部落盘，`tracks/`、`embeddings/`、单 scene eval JSON 与全量 summary 已齐备。
- 当前 `v3_clean` 四条结果线已完成：`rgb_only (mAP=0.5750, R@1=0.3333)`、`rgb_predicted_depth_geometry (0.4222, 0.1667)`、`rgb_fused_geometry (0.4833, 0.1667)`、`gt_upper_bound (0.8333, 0.6667)`。
- clean 场景下移除 humanoid 后，geometry 两条分支仍未超过 `rgb_only`，说明当前主瓶颈仍更偏向 `SAM2/depth` 感知误差，而不只是场景遮挡。

### 5.3 为什么当前不再做人形

- 仓库中现成的人形目标只有 `ikun` 一类，无法自然支撑 `3 identities x 2 scenes` 的人形 benchmark。
- 当前已有正式候选结果实际上更接近飞行目标域，而不是人形域。
- 与其在近期补 2 个新的人形外观，不如先利用现有飞行模型收口 ICISCAE，再把跨节点和真实迁移留给毕业论文。

## 六、目标资产与 MuJoCo 场景策略

| identity_id | 资产来源 | 当前状态 | 对应 MJCF |
| --- | --- | --- | --- |
| `j10` | `assets/models/J10` | 当前激活 clean 场景，去除 humanoid 干扰 | `mvp-demo/assets/scene/mujoco_3cam_node_parallel_j10.xml` |
| `uav1` | `assets/models/uav1_ascii` | 当前激活的 clean `v2` split-material 场景，按 J10 式显式导入 | `mvp-demo/assets/scene/mujoco_uav1_3cam_node_parallel_v2.xml` |
| `su34` | `assets/models/su34` | 当前激活的 clean 第三身份场景，按 J10 式显式导入 | `mvp-demo/assets/scene/mujoco_su34_3cam_node_parallel.xml` |
| `dji_mavic` | `assets/models/DJI Mavic Air Drone` | 仅保留为 `v1` 历史身份与结果归档，不再继续修复 | `mvp-demo/assets/scene/legacy/v1/mujoco_humanoid_dji_mavic_3cam_node_parallel.xml` |

冻结结论如下：

- 历史 `v1` benchmark 保留 `j10 / uav1 / dji_mavic`，但不再作为当前主线。
- 历史 `v2` benchmark 保留 `j10 / uav1 / su34`，但当前主线已切到 clean `v3_clean`。
- 历史 `v1` 的 `uav1 / dji_mavic` 场景 canonical 路径统一收口到 `mvp-demo/assets/scene/legacy/v1/`；历史 humanoid 主线场景统一收口到 `mvp-demo/assets/scene/legacy/humanoid/`。
- 活跃 `mvp-demo/assets/scene/` 根目录只保留 clean 场景，不再保留任何 `mujoco_humanoid_*.xml`。
- `dji_mavic` 不再继续修复；若后续确实需要替补第三身份，优先沿 `su34` 这条显式材质链继续扩展，不回退到 `ikun`。
- `C919 / A380 / 3D白色大疆` 目前缺少可直接被 MuJoCo 使用的 `obj/stl + 材质` 组合，不纳入近阶段主线。

## 七、ICISCAE 正式 benchmark 协议

### 7.1 权威 manifest

- 正式 benchmark 不再写死在计划正文中。
- 权威文件固定为：
  - 当前激活：`research/plans/tri_camera_node_3d_aware_reid/benchmarks/iciscae_node01_uav_v3_clean.json`
  - 历史归档：`research/plans/tri_camera_node_3d_aware_reid/benchmarks/iciscae_node01_uav_v2.json`
  - 历史归档：`research/plans/tri_camera_node_3d_aware_reid/benchmarks/iciscae_node01_uav_v1.json`

### 7.2 固定协议

- `benchmark_id`：`iciscae_node01_uav_v3_clean`
- `node_id`：`node01`
- `task`：`single-node, cross-scene, track-level retrieval`
- `split_role`：所有正式 scene 均为 `both`
- `mask_source`：`sam2_pred`
- `depth_source`：`depth_anything_v2`
- `mask_layout`：`flat`
- `min_valid_timestamps`：`5`
- 评测默认开启：
  - `exclude_same_track_id`
  - `exclude_same_scene`

### 7.3 正式 3x2 scene 设计

每个 identity 固定两条 scene：

1. `line_nodes`
   - `traj=line_nodes`
   - `traj_from_body=node01`
   - `traj_to_body=node02`
   - `mid_y=6`
   - `mid_z=2`
   - `traj_period=8`
2. `circle_xz`
   - `traj=circle_xz`
   - `traj_center=0 6 2`
   - `traj_radius=1`
   - `traj_period=6`

当前激活的正式 scene_id 固定如下：

- `mj_node01_j10_clean_line_nodes_a`
- `mj_node01_j10_clean_circle_xz_b`
- `mj_node01_uav1_clean_line_nodes_a`
- `mj_node01_uav1_clean_circle_xz_b`
- `mj_node01_su34_clean_line_nodes_a`
- `mj_node01_su34_clean_circle_xz_b`

### 7.4 主结果矩阵

- `RGB-only`
- `RGB + predicted-depth geometry`
- `RGB + fused geometry`

固定说明：

- RGB 主基线固定为 `CLIP`。
- `hist` 和 `radial_hist` 只保留为 smoke fallback，不作为小论文主结果命名。
- MuJoCo 的 `masks_gt/` 和 `depth_gt/` 只用于 GT upper-bound 和误差分析。

## 八、执行顺序与验收

### M0 文档与协议冻结

- 交付物：
  - 冻结研究口径、论文拆分和 benchmark manifest。
- 完成定义：
  - 后续执行不再依赖组会稿或口头约定。

### M1 clean 资产与正式 scene 采集

- 交付物：
  - `j10 / uav1 / su34` 对应的 clean MuJoCo 场景可被加载。
  - 按 `v3_clean manifest` 为 `j10 / uav1 / su34` 落地 `6` 个正式 clean scene。
- 完成定义：
  - 每个 scene 都带显式 `scene_id` 和显式 `identity_id`。
  - 每个 scene 都能导出三路 `frames/`、`rig.json` 和 `frame_times.csv`。

### M2 图像侧 depth 与 masks

- 交付物：
  - 全部正式 scene 补齐 `cams/cam*/depth/` 与 `cams/cam*/masks/`。
- 完成定义：
  - 每个正式 scene 的有效同步时间戳不少于 `5`。
  - `obj_XXX` 嵌套结果若存在，必须先展平成 `flat masks` 再进入正式 benchmark。

### M3 `RGB-only` 首轮正式结果

- 交付物：
  - `tracklets.json`
  - `tracks.npy`
  - `tracks_meta.json`
  - `RGB-only` 结果 JSON
- 完成定义：
  - 每个 query 都至少存在 `1` 个跨 scene 正样本。
  - 正式评测统一使用 `exclude_same_track_id + exclude_same_scene`。

### M4 几何两条分支

- 交付物：
  - `RGB + predicted-depth geometry`
  - `RGB + fused geometry`
- 完成定义：
  - 三组主结果全部落盘到 `mvp-demo/output/evals/<benchmark_id>/...`
  - 可以清晰比较 geometry 是否优于 `RGB-only`

### M5 ICISCAE 收口

- 交付物：
  - 表格
  - 案例图
  - GT upper-bound 对比
  - 失败分析
- 完成定义：
  - 形成可直接写进小论文的结果和结论段落。

### M6 毕业论文扩展

- 交付物：
  - `node02` 或真实节点数据接入
  - `cross-node` 主结果
  - 系统性误差分析
- 完成定义：
  - 在不改下游数据契约的前提下跑通跨节点与真实迁移。

## 九、当前验收标准

### 9.1 小论文验收

- `node01` 的正式 benchmark 已固定并可复现。
- `3 identities x 2 scenes` 全部满足主链要求。
- 三组主结果齐全。
- GT upper-bound 和失败分析齐全。

### 9.2 毕业论文验收

- 在小论文基础上，补齐 `node02` 或真实节点的正式 benchmark。
- 给出 `cross-node` 主结果，而不是只停留在 `node01`。
- 给出仿真到真实的接口稳定性与误差归因。

## 十、风险与应对

- `identity_id` 缺失会直接导致评测指标无效。
  - 对策：统一以 `capture_meta.target.identity_id` 为权威来源。
- Windows 下 MuJoCo 可能无法直接加载中文资产目录。
  - 对策：当前激活的 `uav1_v2` 与 `su34` 都固定从 ASCII-safe 路径运行；不再把中文资产目录作为主线入口。
- 多材质 OBJ 在 MuJoCo 中可能因为单材质压缩、JPG 纹理或薄片 mesh 失败。
  - 对策：统一改成 J10 式显式 `texture/material/mesh` 图，`uav1` 使用 PNG 纹理，`su34` 对 split mesh 固定使用 `inertia="shell"`。
- SAM2 和 depth 覆盖率可能不足。
  - 对策：先做覆盖率检查，再进入 embedding 提取。
- 小 baseline 下 predicted depth 噪声大。
  - 对策：先把 `predicted-depth geometry` 视为弱几何分支，再用 `fused geometry` 判断几何是否带来真正收益。
- clean 场景下若 geometry 仍弱于 `RGB-only`，说明问题不只来自 humanoid 遮挡。
  - 对策：把 `GT upper-bound` 作为感知误差诊断线，重点检查 `SAM2` mask 与 depth 回投质量。

## 十一、当前不做与文档入口

- 当前 ICISCAE 不做人形 benchmark。
- 当前 ICISCAE 不把 `cross-node retrieval` 作为成功条件。
- 当前 ICISCAE 不把真实三相机结果作为主结果。
- 当前阶段不训练新的端到端 3D encoder。
- 组会思路目录下的归档文档不并入主线 research 文档。

当前应优先阅读：

- `research/plans/ACTIVE_PLAN.md`
- `research/plans/tri_camera_node_3d_aware_reid/benchmarks/iciscae_node01_uav_v3_clean.json`
- `research/handoffs/tri_camera_node_engineering_handoff_zh.md`
