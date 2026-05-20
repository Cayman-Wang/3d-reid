# tri_camera_node_3d_aware_reid 主计划（中文）

- 创建日期：2026-03-15
- 最近冻结：2026-05-18
- 状态：当前 4D 几何主线正式转为 `NeoVerse fused 4D points_by_timestamp -> tracklet -> embedding -> ReID`；`node01` 单节点 `3 identities x 2 scenes` eval matrix、`cam0/cam1/cam2` view-count ablation 和 own-depth no-depthGT 评测均已完成。现有结果只能证明 NeoVerse 4D geometry 已接入并可评测，当前 bootstrap 下不能证明 geometry、三相机或 own-depth 已带来指标提升。下一阶段转向 `cross-node smoke` 与 `learned point encoder / end-to-end 4D-aware ReID prototype`。

## 一、冻结结论

- 研究主线固定为“三相机节点级 3D/4D-aware track retrieval”，当前 4D 几何主输入固定为 NeoVerse fused 4D `points_by_timestamp`，不再把 YOLO 门控 + 3DGS 或 recon_spin 作为主方法。
- `ICISCAE 小论文` 固定做 `node01` 下的 `single-node, cross-scene, track-level retrieval`。
- `毕业论文主线` 固定做 `cross-node 3D/4D re-ID`、真实三相机迁移和端到端 4D-aware ReID 表征补强，但不与当前保底投稿绑定。
- `毕业论文数据路线` 固定优先建设 `synthetic UAV/aircraft 4D-ReID benchmark`：复用现成城市仿真场景，自定义多节点三相机、飞行器 identity、轨迹与标注导出，再用少量真实数据做 sanity check 与 sim-to-real 误差归因。
- 小论文目标域固定为 `UAV/aircraft`，不再做人形 benchmark。
- 小论文允许只做 MuJoCo 仿真，但标题、摘要和结论只能写成“仿真三相机节点检索验证”，不能提前宣称跨节点已完成。
- 研究口径、工程 handoff 和 benchmark 统一收口到 `research/`；`mvp-demo/` 仅保留运行入口与资产说明。

## 二、研究命题与当前主问题

本项目研究的不是传统行人闭集识别，而是一个更一般的实例级跨视角检索问题：

- 输入：每个节点由 3 台同步相机构成，连续采集目标在视野中的运动过程。
- 中间表示：将同一目标在一个采集窗口中的多帧、多相机观测组织为一个 `tracklet`。
- 输出：为每个 `tracklet` 生成稳定的 `track embedding`，并在其他 scene 或其他节点中做检索。

当前阶段的主问题是：

> 在不使用 MuJoCo GT 参与主检索计算的前提下，`RGB + masks + rig + timestamps + NeoVerse 4D points_by_timestamp` 是否足以支持稳定的 3D/4D-aware track retrieval，并能否通过 learned geometry、motion rerank 与端到端 ReID prototype 提升复杂场景下的身份检索能力。

当前阶段的关键问题补充如下：

- 当前 `fused_scene.glb` 仍只是静态汇总点云查看产物，不是 4D 回放文件，也不作为 ReID 几何输入契约。
- 当前 NeoVerse fused 4D 点云已可被 ReID 消费，但本机输入分辨率与点云稠密度仍限制细节表达。
- 当前 `node01` 已有多身份、多 scene Rank/mAP bootstrap 指标，但数据难度不足以体现几何、三相机或 depth source 差异；后续判断重点应转向 cross-node、更高难度数据和 learned 3D/4D 表征。

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
| 核心任务 | `node01` 的 `single-node, cross-scene, track-level 3D/4D retrieval` | `cross-node 3D/4D re-ID` + 真实节点迁移 |
| 目标域 | `UAV/aircraft` | 延续 `UAV/aircraft`，后续可扩到更丰富目标 |
| 数据来源 | MuJoCo-only 可接受 | MuJoCo + 至少一部分真实三相机数据 |
| benchmark 最小规模 | `3 identities x 2 scenes` | 至少 `3 identities x 2 nodes x 多 scene` |
| 研究问题 | NeoVerse 4D geometry 是否可稳定接入 ReID，并为 learned geometry / motion rerank 提供输入 | 表征能否抗节点差异、真实域偏移和复杂 hard negative |
| 主结果 | `RGB-only`、`RGB + FPFH handcrafted geometry`、`RGB + NeoVerse 4D geometry` | 在上述基础上补 `learned point geometry`、`motion rerank`、端到端 4D-aware ReID、cross-node 与真实迁移结果 |
| 允许缺失 | 可以没有 `node02` 和真实数据 | 不能长期停留在单节点仿真 |
| 成功定义 | benchmark、三组结果、失败分析和 GT 上界完整 | 在上述基础上补齐跨节点和真实迁移证据 |

补充说明：

- 毕业论文阶段允许以 synthetic benchmark 作为主要训练数据来源；
- 真实三相机数据近期可只承担 sanity check、接口稳定性验证和 sim-to-real 误差归因，不要求先完成大规模真实采集。

## 四、当前方法收口

当前主线保留以下结果线，并明确区分当前 4D 主线、历史 baseline 与近期 prototype：

| 结果线 | 输入契约 | 当前口径 |
| --- | --- | --- |
| `RGB-only` | `frames + tracklets` | 当前正式外观基线 |
| `RGB + FPFH handcrafted geometry` | `frames + tracklets + points_by_timestamp` | 当前 NeoVerse 4D 几何 baseline，几何后端为 `open3d_fpfh` |
| `RGB + learned point geometry` | `frames + tracklets + points_by_timestamp + point encoder` | 近期 prototype，目标是替代 FPFH |
| `RGB + learned geometry + motion rerank` | `points_by_timestamp + trajectory/motion stats` | 近期 prototype，目标是显式利用 NeoVerse 4D 时序信息 |
| `end-to-end 4D-aware ReID` | `RGB sequence + point sequence + motion sequence` | 近期 prototype / 毕业论文方法候选，不是当前已有结果 |
| `RGB + predicted-depth geometry` | `frames + masks + predicted depth + rig` | 历史弱几何 baseline / 诊断线 |
| `RGB + fused geometry` | `frames + masks + predicted depth + rig + fused points` | 历史多相机 depth/mask 融合对照线 |
| `GT upper-bound` | `masks_gt + depth_gt + fused geometry` | 感知误差诊断线 |

固定说明如下：

- RGB 主基线固定为 `CLIP`。
- `hist` 和 `radial_hist` 只保留为 smoke fallback，不作为小论文主结果命名。
- `RGB + NeoVerse 4D geometry` 来自 `per-camera + backproject + fuse + dynamic constraint` 的外部融合方案，不是早期 joint multiview static bundle 导出路线；当前权威几何目录为 `mvp-demo/output/neoverse_fused/<scene_id>/points_by_timestamp/`，ReID 接入使用相对 `scene_dir` 的路径 `../../../../../output/neoverse_fused/<scene_id>/points_by_timestamp`。
- 旧 `RGB + predicted-depth geometry` 与 `RGB + fused geometry` 不再作为当前 4D 主线，仅用于历史基线、诊断或对照。
- `node01_neoverse_fused_4d_view_ablation_v1` 仅用于 view-count ablation：单相机 `cam0/cam1/cam2` 几何从已有 `per_camera / observations / points_per_view` 派生，并通过 `--cams <cam> --min_bg_cam_support 1 --min_mask_cam_support 1` 生成单相机 `points_by_timestamp`；它不重跑 NeoVerse reconstructor，也不替代三相机主线。
- 当前 NeoVerse fused 4D 与 spin 重建线的默认采集轨迹固定为高俯仰 `static_spin_yaw_pitch`：`yaw_start_deg=-45`、`yaw_end_deg=45`、`pitch_amp_deg=20`、`pitch_period=8`、`seconds=8`、`fps=30`。除消融外，后续新运行默认沿用该配置。
- 当前本地笔记本更偏向“链路验证机”而不是“高质量重建机”；后续切换到高性能机器时，NeoVerse fused 4D 的优先优化方向固定为：先提输入分辨率，再看是否需要减小 `output_voxel_size_m`，而不是先继续收紧 trim 参数。
- 当前工程已具备 `RGB-only`、`CLIP + FPFH`、NeoVerse 4D `points_by_timestamp` 接入与评测入口；learned point encoder、motion rerank 和端到端 4D-aware ReID 属于下一阶段 prototype。
- `YOLO 门控 + 3DGS` 相关脚本只保留为辅助 demo，不再参与当前主线里程碑。

## 五、当前工程能力与证据现状

### 5.1 已具备的主链

- 当前仓库已经具备节点级主链脚本：`mj_capture_3cam_node.py -> run_node_depth_anything_v2.py -> run_node_sam2_masks.py -> build_node_tracklets.py -> extract_node_track_embeddings.py -> eval_node_track_retrieval.py`。
- 当前仓库已经具备 NeoVerse fused 4D 主链脚本：`run_neoverse_per_camera_bundle.py -> export_neoverse_view_observations.py -> backproject_neoverse_observations.py -> fuse_neoverse_multiview_world_points.py -> constrain_neoverse_multiview_dynamic.py`，并固定输出 `points_by_timestamp` 供 ReID 消费。
- 当前仓库仍保留历史几何增强入口：`recon_fuse_depth_points.py` 可用于多相机 `depth + mask` 融合点云，但它不再是当前 4D 主线。
- 当前仓库同时保留一条 `YOLO 门控 + 3DGS` 静态场景 demo 链，但该链只保留为辅助演示，不纳入主线里程碑。

### 5.2 当前本地证据

- 历史 `v3_clean` 的 `6` 条正式 scene 已全部落盘，`tracks/`、`embeddings/`、单 scene eval JSON 与全量 summary 已齐备；该分支保留为 early baseline 与感知误差诊断证据。
- 历史 `v3_clean` 四条结果线已完成：`rgb_only (mAP=0.5750, R@1=0.3333)`、`rgb_predicted_depth_geometry (0.4222, 0.1667)`、`rgb_fused_geometry (0.4833, 0.1667)`、`gt_upper_bound (0.8333, 0.6667)`。
- clean 场景下移除 humanoid 后，历史 geometry 两条分支仍未超过 `rgb_only`，说明当时主瓶颈更偏向 `SAM2/depth` 感知误差，而不只是场景遮挡。
- 早期 `points_by_timestamp` 单 scene smoke 已完成，现仅作为接口接通的历史证据。
- 早期 `node01/j10` smoke 只证明“工程闭环已打通”；当前已经补齐 `node01` 多身份、多 scene 的正式 bootstrap 指标，后续判断重点应转向更高难度数据、cross-node 或真实 mask/depth，而不是继续把 `node01` bootstrap 当作未完成项。
- 当前 `node01_neoverse_fused_4d_eval_matrix_v1` 已完成 `j10/uav1/su34 x static/circle` 共 `6` 个 spin scene：`rgb_only_clip_gtmask_eval_v1` 与 `rgb_neoverse_fused_4d_clip_fpfh_eval_v1` 均为 `metric_queries=6, mAP=1.0, R@1=1.0, R@5=1.0`。RGB-only embedding 为 `(1,512)`，RGB+4D embedding 为 `(1,545)`。
- 当前 `node01_neoverse_fused_4d_view_ablation_v1` 已完成单相机 view-count ablation：`cam0/cam1/cam2` 的 RGB-only 与 RGB+4D 六个 branch 均为 `metric_queries=6, mAP=1.0, R@1=1.0, R@5=1.0`。这说明当前 `node01` GT-mask bootstrap 难度不足以体现视角数量差异，不应强行写成三相机提升。
- 当前 `node01_neoverse_own_depth_4d_eval_v1` 已完成 own-depth no-depthGT 分支：去掉 `scene depth_gt` 锚深度后，NeoVerse own-depth 4D geometry 仍可被 ReID 消费；指标仍为满分时，只能说明当前 bootstrap 难度不足以体现 depth source 差异。
- 当前单帧逐时间戳查看下，动态点云厚度已基本可接受；下一阶段更突出的问题是点云还不够稠密，细节仍偏粗，这与本地只能稳定跑 `280x168` 有直接关系。
- 已完成本机后处理增密消融：`2026-04-26_j10_yp20_dense_points_r01` 基于 `2026-04-26_j10_yp20_r02params_r01`，只把 `output_voxel_size_m` 从 `0.01` 改到 `0.005`，其余保持不变；结果 `fused_dynamic_points`、三路 coverage 和 `depth_support_ratio` 与基线一致，说明当前本机只改输出体素大小没有产生有效增密。

### 5.3 切换高性能机器后的固定优化顺序

后续切到高性能机器时，NeoVerse fused 4D 分支按以下顺序优化：

1. 先提高 `per-camera bundle` 输入分辨率，优先尝试合法的 `14` 倍数尺寸：`336x336`、`448x448`、`560x336`。
2. 分辨率提升阶段保持 `yp20_r02params` 的现有动态约束参数不变，确保结果可归因。
3. 若分辨率提高后单帧点云仍偏稀疏，再单独把 `output_voxel_size_m` 从 `0.01` 收紧到 `0.005`，并视情况提高 `max_dynamic_points`。
4. 当前单帧厚度已可接受时，不优先继续压 `depth_trim_radius_m`；该参数后续只做温和收紧，不作为首要优化轴。
5. 若后续目标从“更密点云”升级为“连续表面”，则应单独新增表面重建或 Gaussian/mesh 导出，不把这一需求混入当前点云参数调优。

补充结论：

- `2026-04-26_j10_yp20_dense_points_r01` 已证明：在输入分辨率仍为 `280x168`、且不重跑 NeoVerse 源头重建的前提下，只做本机后处理的 `output_voxel_size_m 0.01 -> 0.005` 不足以带来实际点云增密收益。
- 因此下一步仍应优先在高性能机器上提高合法输入分辨率，再判断是否需要继续调点云密度；不建议把当前本机 dense_points run 升级为默认结果。

### 5.4 为什么当前不再做人形

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
- `RGB + FPFH handcrafted geometry`
- `RGB + NeoVerse 4D geometry`
- `RGB + learned point geometry`（近期 prototype）
- `RGB + learned geometry + motion rerank`（近期 prototype）
- `end-to-end 4D-aware ReID`（近期 prototype / 毕业论文方法候选）

固定说明：

- RGB 主基线固定为 `CLIP`。
- `hist` 和 `radial_hist` 只保留为 smoke fallback，不作为小论文主结果命名。
- MuJoCo 的 `masks_gt/` 和 `depth_gt/` 只用于 GT upper-bound 和误差分析。
- `RGB + predicted-depth geometry` 与旧 `RGB + fused geometry` 保留为历史 baseline / 诊断线，不再作为当前 4D 主结果矩阵的默认命名。

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

### M4 历史几何 baseline 与诊断线

- 交付物：
  - `RGB + predicted-depth geometry`
  - `RGB + fused geometry`
  - `GT upper-bound`
- 完成定义：
  - 三组主结果全部落盘到 `mvp-demo/output/evals/<benchmark_id>/...`
  - 可以清晰分析预测深度、mask 和多相机融合误差，不再把这两条作为当前 4D 主线。

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

### M7 NeoVerse 4D 接入正式化

- 交付物：
  - `node01_neoverse_fused_4d_eval_matrix_v1`
  - `node01_neoverse_fused_4d_view_ablation_v1`
  - `node01_neoverse_own_depth_4d_eval_v1`
- 完成定义：
  - `NeoVerse fused 4D points_by_timestamp -> tracklet -> CLIP + FPFH -> ReID` 闭环可复现。
  - 结论保守写成“接入与可评测完成，当前 bootstrap 下与 RGB-only 持平”。

### M8 NeoVerse 4D 主线与 ReID 表征补强

- 交付物：
  - `node02 / cross-node smoke`
  - `CLIP + learned point geometry` prototype
  - `geometry / motion rerank` prototype
  - `end-to-end 4D-aware ReID` prototype
- 完成定义：
  - learned geometry 分支能与 `RGB-only`、`CLIP + FPFH` 同口径输出 `tracks.npy / tracks_meta.json` 并参与 mAP/Recall 评测。
  - motion rerank 能在 top-K 上输出可解释的 geometry / trajectory 调整分数。
  - end-to-end prototype 能生成与现有检索评测兼容的 track embedding，但不把小数据训练结果写成最终性能结论。

## 九、当前验收标准

### 9.1 小论文验收

- `node01` 的正式 benchmark 已固定并可复现。
- `3 identities x 2 scenes` 全部满足主链要求。
- 三组主结果齐全。
- GT upper-bound 和失败分析齐全。

### 9.2 毕业论文验收

- 在小论文基础上，补齐 `node02` 或真实节点的正式 benchmark。
- 给出 `cross-node` 主结果，而不是只停留在 `node01`。
- 给出 learned point geometry、4D motion rerank 或端到端 4D-aware ReID 的至少一条方法线，并与 `RGB-only`、`CLIP + FPFH` 做同口径消融。
- 给出仿真到真实的接口稳定性与误差归因。

### 9.3 synthetic UAV/aircraft 4D-ReID benchmark 路线

- 不从零自建整座城市环境，优先复用现成城市仿真场景。
- 研究重点放在多节点三相机部署、飞行器 identity、跨节点轨迹、标注导出与 ReID 协议设计。
- synthetic benchmark 应服务于 `cross-node`、更高难度 hard negative 和 `end-to-end 4D-aware ReID` 训练。
- 近期权威规划文档固定为 `research/plans/tri_camera_node_3d_aware_reid/synthetic_uav_4d_reid_dataset_plan_zh.md`。

## 十、风险与应对

- `identity_id` 缺失会直接导致评测指标无效。
  - 对策：统一以 `capture_meta.target.identity_id` 为权威来源。
- Windows 下 MuJoCo 可能无法直接加载中文资产目录。
  - 对策：当前激活的 `uav1_v2` 与 `su34` 都固定从 ASCII-safe 路径运行；不再把中文资产目录作为主线入口。
- 多材质 OBJ 在 MuJoCo 中可能因为单材质压缩、JPG 纹理或薄片 mesh 失败。
  - 对策：统一改成 J10 式显式 `texture/material/mesh` 图，`uav1` 使用 PNG 纹理，`su34` 对 split mesh 固定使用 `inertia="shell"`。
- SAM2 和 depth 覆盖率可能不足。
  - 对策：先做覆盖率检查，再进入 embedding 提取。
- 小 baseline 下 predicted depth 和旧 fused geometry 噪声大。
  - 对策：把它们视为历史弱几何 baseline / 诊断线，当前 4D 主线优先使用 NeoVerse `points_by_timestamp`。
- 当前 bootstrap 过于容易，RGB-only、RGB+NeoVerse 4D、own-depth 和 view-count ablation 都可能满分。
  - 对策：不强行写成 geometry 提升；下一步转向 cross-node、更高难度数据、learned geometry 和 motion rerank。
- 小数据端到端训练容易过拟合。
  - 对策：端到端 4D-aware ReID 近期只作为 prototype，必须与 `RGB-only`、`CLIP + FPFH`、`learned point geometry` 同口径比较，并明确训练/测试划分。

## 十一、当前不做与文档入口

- 当前 ICISCAE 不做人形 benchmark。
- 当前 ICISCAE 不把 `cross-node retrieval` 作为成功条件。
- 当前 ICISCAE 不把真实三相机结果作为主结果。
- 当前阶段允许规划和实现端到端 4D-aware ReID prototype，但不把完整大规模真实场景端到端训练作为 node01 bootstrap 成功条件，也不能把 prototype 写成已有最终成果。
- 组会思路目录下的归档文档不并入主线 research 文档。

当前应优先阅读：

- `research/plans/ACTIVE_PLAN.md`
- `research/plans/tri_camera_node_3d_aware_reid/neoverse_4d_reid_strengthening_plan_zh.md`
- `research/plans/tri_camera_node_3d_aware_reid/synthetic_uav_4d_reid_dataset_plan_zh.md`
- `research/plans/tri_camera_node_3d_aware_reid/benchmarks/node01_neoverse_fused_4d_eval_matrix_v1.json`
- `research/handoffs/tri_camera_node_engineering_handoff_zh.md`
