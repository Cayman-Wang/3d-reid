# tri_camera_node_3d_aware_reid 主计划（中文）

- 创建日期：2026-03-15
- 最近冻结：2026-03-16
- 状态：ICISCAE 范围已收紧到 `node01 + UAV/aircraft + single-node cross-scene`，当前进入 M1 资产落地与正式 scene 采集

## 一、冻结结论

- 研究主线固定为“三相机节点级 3D-aware track retrieval”，不再把 YOLO 门控 + 3DGS 作为主方法。
- `ICISCAE 小论文` 固定做 `node01` 下的 `single-node, cross-scene, track-level retrieval`。
- `毕业论文主线` 固定做 `cross-node 3D re-ID` 与真实三相机迁移，但不与当前保底投稿绑定。
- 小论文目标域固定为 `UAV/aircraft`，不再做人形 benchmark。
- 小论文允许只做 MuJoCo 仿真，但标题、摘要和结论只能写成“仿真三相机节点检索验证”，不能提前宣称跨节点已完成。

## 二、两层论文对照表

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

## 三、当前工程能力与证据现状

### 3.1 已具备的主链

- 当前仓库已经具备节点级主链脚本：`mj_capture_3cam_node.py -> run_node_depth_anything_v2.py -> run_node_sam2_masks.py -> build_node_tracklets.py -> extract_node_track_embeddings.py -> eval_node_track_retrieval.py`。
- 当前仓库已经具备几何增强入口：`recon_fuse_depth_points.py` 可用于多相机 `depth + mask` 融合点云。
- 当前仓库同时保留一条 `YOLO 门控 + 3DGS` 静态场景 demo 链，但该链只保留为辅助演示，不纳入主线里程碑。

### 3.2 当前本地证据

- 已经存在少量完成到 `tracks/`、`embeddings/` 与 retrieval JSON 的 `node01` scenes，证明 pipeline 已经从“想法”进入“可运行原型”。
- 当前正式候选 scene 仍主要集中在单一 `identity_id`，其中可直接参考的结果主要来自 `j10` 场景，尚不足以形成正式 benchmark。
- 现有使用 `masks_gt/` 或 `depth_gt/` 跑通的结果只保留为 proof-of-pipeline 和 upper-bound，不进入小论文主结果。

### 3.3 为什么当前不再做人形

- 仓库中现成的人形目标只有 `ikun` 一类，无法自然支撑 `3 identities x 2 scenes` 的人形 benchmark。
- 当前已有正式候选结果实际上更接近飞行目标域，而不是人形域。
- 与其在近期补 2 个新的人形外观，不如先利用现有飞行模型收口 ICISCAE，再把跨节点和真实迁移留给毕业论文。

## 四、目标资产与 MuJoCo 场景策略

| identity_id | 资产来源 | 当前状态 | 对应 MJCF |
| --- | --- | --- | --- |
| `j10` | `assets/models/J10` | 已有现成场景与主结果证据 | `mvp-demo/assets/scene/mujoco_humanoid_3cam_node_parallel_j10.xml` |
| `uav1` | `assets/models/无人机1` | 已有现成场景与纹理链，但 Windows 直跑受中文资产路径限制 | `mvp-demo/assets/scene/mujoco_humanoid_uav1_3cam_node_parallel.xml` |
| `dji_mavic` | `assets/models/DJI Mavic Air Drone` | 本轮新增 MuJoCo 场景，默认纯材质版本 | `mvp-demo/assets/scene/mujoco_humanoid_dji_mavic_3cam_node_parallel.xml` |
| `big_dji` | `assets/models/大疆无人机` | 只作为 `dji_mavic` 失败时的 fallback | 待需要时再补对应 MJCF |

冻结结论如下：

- 小论文正式 benchmark 默认身份集合固定为 `j10 / uav1 / dji_mavic`。
- 若 `dji_mavic` 在实际采集或渲染中仍不稳定，再回退到 `big_dji`，但不回退到 `ikun`。
- `C919 / A380 / 3D白色大疆` 目前缺少可直接被 MuJoCo 使用的 `obj/stl + 材质` 组合，不纳入近阶段主线。

## 五、ICISCAE 正式 benchmark 协议

### 5.1 权威 manifest

- 正式 benchmark 不再写死在计划正文中。
- 权威文件固定为：
  - `research/plans/tri_camera_node_3d_aware_reid/benchmarks/iciscae_node01_uav_v1.json`

### 5.2 固定协议

- `benchmark_id`：`iciscae_node01_uav_v1`
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

### 5.3 正式 3x2 scene 设计

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

计划中的正式 scene_id 固定如下：

- `mj_node01_j10_line_nodes_a`
- `mj_node01_j10_circle_xz_b`
- `mj_node01_uav1_line_nodes_a`
- `mj_node01_uav1_circle_xz_b`
- `mj_node01_dji_mavic_line_nodes_a`
- `mj_node01_dji_mavic_circle_xz_b`

### 5.4 主结果矩阵

- `RGB-only`
- `RGB + predicted-depth geometry`
- `RGB + fused geometry`

固定说明：

- RGB 主基线固定为 `CLIP`。
- `hist` 和 `radial_hist` 只保留为 smoke fallback，不作为小论文主结果命名。
- MuJoCo 的 `masks_gt/` 和 `depth_gt/` 只用于 GT upper-bound 和误差分析。

## 六、执行顺序与验收

### M0 文档与协议冻结

- 交付物：
  - 冻结研究口径、论文拆分和 benchmark manifest。
- 完成定义：
  - 后续执行不再依赖组会稿或口头约定。

### M1 第三个飞行目标落地与正式 scene 采集

- 交付物：
  - `dji_mavic` 对应的 MuJoCo 场景可被加载。
  - 按 manifest 为 `j10 / uav1 / dji_mavic` 采集 `6` 个正式 scene。
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

## 七、当前验收标准

### 7.1 小论文验收

- `node01` 的正式 benchmark 已固定并可复现。
- `3 identities x 2 scenes` 全部满足主链要求。
- 三组主结果齐全。
- GT upper-bound 和失败分析齐全。

### 7.2 毕业论文验收

- 在小论文基础上，补齐 `node02` 或真实节点的正式 benchmark。
- 给出 `cross-node` 主结果，而不是只停留在 `node01`。
- 给出仿真到真实的接口稳定性与误差归因。

## 八、风险与应对

- `identity_id` 缺失会直接导致评测指标无效。
  - 对策：统一以 `capture_meta.target.identity_id` 为权威来源。
- Windows 下 MuJoCo 可能无法直接加载中文资产目录。
  - 对策：当前 `dji_mavic` 使用 ASCII 资产目录；`uav1` 若在 Windows 直跑失败，则改用 ASCII 资产别名或 Linux/WSL 环境执行采集。
- `dji_mavic` 可能缺少完整材质链。
  - 对策：当前场景先采用纯材质版本，先保证可导入、可采集、可检索。
- SAM2 和 depth 覆盖率可能不足。
  - 对策：先做覆盖率检查，再进入 embedding 提取。
- 小 baseline 下 predicted depth 噪声大。
  - 对策：先把 `predicted-depth geometry` 视为弱几何分支，再用 `fused geometry` 判断几何是否带来真正收益。

## 九、当前明确不做

- 当前 ICISCAE 不做人形 benchmark。
- 当前 ICISCAE 不把 `cross-node retrieval` 作为成功条件。
- 当前 ICISCAE 不把真实三相机结果作为主结果。
- 当前阶段不训练新的端到端 3D encoder。
- 组会思路目录下的归档文档不并入主线 research 文档。
