# 三相机节点 3D-aware ReID：冻结版研究说明

本文档用于解释当前研究主线为什么要先停在 `node01`，以及为什么近期应当先用 MuJoCo 的 `UAV/aircraft` benchmark 收口 ICISCAE，再扩展到毕业论文所需的 `cross-node 3D re-ID`。

## 1. 研究命题

本项目研究的不是传统行人闭集识别，而是一个更一般的 **实例级跨视角检索** 问题：

- 输入：每个节点由 3 台同步相机构成，连续采集目标在视野中的运动过程。
- 中间表示：将同一目标在一个采集窗口中的多帧、多相机观测组织为一个 `tracklet`。
- 输出：为每个 `tracklet` 生成稳定的 `track embedding`，并在其他 scene 或其他节点中做检索。

当前阶段的主问题是：

> 在不使用 MuJoCo GT 参与主检索计算的前提下，`RGB + depth + mask + rig + timestamps` 是否足以支持稳定的 3D-aware track retrieval。

## 2. 为什么只做 `node01` 仍然是 Re-ID

`single-node` 并不等于“不是 re-ID”。

在当前阶段，重识别任务被拆成了一个更小但更干净的问题：

- query：来自 `scene A` 的一个 `track`
- gallery：来自 `scene B/C/...` 的其他 `track`
- 正样本：其他 scene 中与 query 拥有相同 `identity_id` 的 `track`
- 约束：评测时排除同一 `track`，同时排除同一 `scene`

因此当前验证的是：

> 同一目标在不同 scene、不同时间窗、不同三相机观测组合下，能否被重新找回。

这仍然是标准的 re-ID / retrieval 问题，只是先把“节点差异”这个变量拿掉，用来验证表征本身是否成立。

## 3. 小论文与大论文的固定拆分

| 维度 | ICISCAE 小论文 | 毕业论文主线 |
| --- | --- | --- |
| 目标 | 先证明单节点跨 scene 的 3D-aware 检索成立 | 再证明该表征可以跨节点、跨真实域迁移 |
| 节点范围 | `node01` | `node01 + node02`，后续接真实节点 |
| 数据来源 | MuJoCo-only 即可 | MuJoCo + 真实三相机 |
| 目标域 | `UAV/aircraft` | 先延续 `UAV/aircraft`，后续再扩展 |
| 结论口径 | 仿真三相机节点检索验证 | 跨节点 3D 重识别与仿真到真实迁移 |

冻结结论如下：

- `ICISCAE 小论文`：允许只做 MuJoCo，但题目和摘要只能写成“仿真三相机节点上的 3D-aware 检索验证”。
- `毕业论文`：不能停留在单节点仿真，必须补齐 `cross-node` 或真实节点证据。

## 4. 为什么当前小论文改做 `UAV/aircraft`

当前仓库中的有效证据决定了近期路线：

- 人形方向当前只有 `ikun` 一类现成目标，无法自然支撑 `3 identities x 2 scenes` 的正式 benchmark。
- 当前已经完成到 `tracks/embeddings` 的正式候选 scene，主要来自 `j10` 场景，而不是人形场景。
- 仓库里现成可被 MuJoCo 使用或较低成本接入的飞行模型更多，近期更适合先做 `UAV/aircraft` 的 benchmark 收口。

因此当前正式目标域冻结为：

- `j10`
- `uav1`
- `dji_mavic`

若 `dji_mavic` 仍不稳定，再回退到 `大疆无人机` 的无贴图或手工材质版本，但不回退到人形。

## 5. 当前资产可用性结论

### 5.1 已经有 MuJoCo 场景证据的资产

- `J10`
  - 现成场景：`mvp-demo/assets/scene/mujoco_humanoid_3cam_node_parallel_j10.xml`
- `无人机1`
  - 现成场景：`mvp-demo/assets/scene/mujoco_humanoid_uav1_3cam_node_parallel.xml`
  - 备注：Windows 下当前中文资产目录可能导致 MuJoCo 直载失败，需要 ASCII 资产别名或 Linux/WSL
- `ikun`
  - 现成场景：`mvp-demo/assets/scene/mujoco_humanoid_3cam_node_parallel.xml`
  - 但不再纳入当前小论文目标域

### 5.2 本轮新增接入策略

- `DJI Mavic Air Drone`
  - 资产目录：`mvp-demo/assets/models/DJI Mavic Air Drone`
  - 当前新增场景：`mvp-demo/assets/scene/mujoco_humanoid_dji_mavic_3cam_node_parallel.xml`
  - 策略：先采用纯材质 visual geom，不依赖缺失的原始 `mtl` 贴图链；已在 ASCII 仓库别名路径下完成 MuJoCo 直载验证

### 5.3 暂不纳入主线的资产

- `大疆无人机`
  - 只有 `obj`，但原始 `mtl` 与纹理不完整，当前只做 fallback
- `C919`
  - 当前主要是 `fbx/c4d`，需要先转 `obj/stl`
- `A380`
  - 当前主要是 `max/tga`，需要先转 `obj/stl`
- `3D白色大疆`
  - 当前主要是 `max/jpg`，需要先转 `obj/stl`

## 6. 冻结后的 benchmark

正式 benchmark 不再写在正文里，而是写入独立 manifest：

- `research/plans/tri_camera_node_3d_aware_reid/benchmarks/iciscae_node01_uav_v1.json`

其固定含义为：

- `benchmark_id`：`iciscae_node01_uav_v1`
- `node_id`：`node01`
- `task`：`single-node, cross-scene, track-level retrieval`
- `scale`：`3 identities x 2 scenes`
- `split_role`：全部 `both`
- `mask_source`：`sam2_pred`
- `depth_source`：`depth_anything_v2`
- `mask_layout`：`flat`
- `min_valid_timestamps`：`5`

每个 identity 固定采两条 scene：

1. `line_nodes`
2. `circle_xz`

固定 scene_id 为：

- `mj_node01_j10_line_nodes_a`
- `mj_node01_j10_circle_xz_b`
- `mj_node01_uav1_line_nodes_a`
- `mj_node01_uav1_circle_xz_b`
- `mj_node01_dji_mavic_line_nodes_a`
- `mj_node01_dji_mavic_circle_xz_b`

## 7. 正式结果矩阵

小论文必须交付三组结果：

1. `RGB-only`
2. `RGB + predicted-depth geometry`
3. `RGB + fused geometry`

并固定以下边界：

- RGB 主基线固定为 `CLIP`
- `hist` 和 `radial_hist` 只做 smoke fallback
- `masks_gt / depth_gt` 只用于 GT upper-bound 与误差分析

## 8. 只做仿真是否足够发 ICISCAE

结论是：

> 够作为当前保底投稿方案，但不够作为毕业论文终版。

前提是论文叙事必须收紧：

- 可以写：
  - MuJoCo 三相机节点仿真
  - 单节点跨 scene 检索
  - 3D-aware 特征对比
  - benchmark、ablation 和 failure analysis
- 不能写：
  - 跨节点已完成
  - 仿真到真实已完成
  - 真实部署已验证

因此当前最稳的策略是：

1. 先用 MuJoCo 的 `node01 UAV/aircraft benchmark` 把 ICISCAE 发出去
2. 再把 `node02` 和真实节点作为毕业论文的第二阶段

## 9. 从小论文到毕业论文的扩展路径

当前阶段先回答：

> 表征本身是否已经具备跨 scene 重识别能力。

后续阶段再回答：

> 该表征是否还能抗节点差异、视角域偏移和真实采集误差。

因此扩展顺序固定为：

1. `node01 single-node cross-scene`
2. `node01 + node02 cross-node`
3. `MuJoCo -> real-node`

这三步之间共享同一套下游契约：

- `frames`
- `masks`
- `depth`
- `rig.json`
- `frame_times.csv`
- `tracklets.json`
- `tracks.npy`
- `tracks_meta.json`

## 10. 当前的完成定义

只要满足以下条件，就可以认为 ICISCAE 小论文主线已经成立：

1. `node01` 的 `3 identities x 2 scenes` benchmark 固定并可复现。
2. 三组主结果都已落盘为 JSON。
3. 每个 query 至少有一个跨 scene 正样本。
4. GT upper-bound、失败案例和结论草稿齐全。

在此之后，项目才进入真正的毕业论文扩展阶段，而不是继续在小论文和大论文之间来回混线。
