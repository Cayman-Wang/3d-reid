# 三相机节点 3D-aware ReID：仿真到真实研究方案

本文档用于统一当前项目的研究主线、工程边界和实验计划，服务于后续的实现推进、阶段验收和毕业设计写作。文档定位不是论文终稿，也不是单纯的工程 TODO，而是一份面向后续工作的 research 初稿。

核心目标固定为：

> 在 MuJoCo 三相机节点中先完成严格传感器口径的 3D-aware track retrieval 验证，再迁移到真实三相机节点，最终实现跨节点实例检索。

---

## 1. 问题定义与研究目标

### 1.1 任务定义

本项目关注的不是传统封闭类别的行人重识别，而是更一般的 **实例级跨视角检索** 问题：

- 输入：每个节点由 3 台同步相机构成，连续采集目标在视野中的运动过程。
- 中间表示：将同一目标在一个采集窗口中的多帧、多相机观测组织为一个 `tracklet`。
- 输出：为每个 `tracklet` 生成一个稳定的 `track embedding`，并在跨 scene、跨节点的 gallery 中做 top-K 检索。

### 1.2 为什么需要 3D-aware

纯 RGB 表征在以下情况下容易失效：

- 视角变化较大时，同一目标的纹理外观差异明显。
- 背景、光照、遮挡会对裁剪图像造成强干扰。
- 对于无人机、人物、细长目标等实例，轮廓和几何结构往往比局部纹理更稳定。

因此，本项目采用 **RGB + 几何** 的混合表征路线。这里的几何并不追求一开始就做到高质量动态三维重建，而是先以“足以支持检索”的几何线索为目标，通过 depth、点云和轻量几何描述子补充跨视角不变性。

### 1.3 研究目标

本文档冻结的阶段性研究目标为：

1. 在 MuJoCo 三相机节点中，以严格传感器口径跑通节点级检索闭环。
2. 证明当前数据契约可以支撑从仿真迁移到真实三相机节点，而不需要重写下游 tracklet、embedding 和 retrieval 模块。
3. 给出一套可执行、可验收的实验路线，用于回答“几何信息是否真的对跨节点检索有帮助”。

### 1.4 研究问题

围绕上述目标，本文档聚焦以下三个问题：

- **RQ1**：在不使用仿真 GT 参与主检索计算的条件下，仅使用 `RGB + calibration + timestamps`，是否能稳定完成节点级 track retrieval。
- **RQ2**：相较于 RGB-only，加入 depth/几何表征后，跨 scene / 跨节点检索性能是否存在可测量提升。
- **RQ3**：当前 MuJoCo-first 的数据契约是否足够稳定，能否直接迁移到真实三相机采集系统。

---

## 2. 当前进展与项目基础

### 2.1 已有仿真场景与节点定义

项目中已经具备可直接使用的 MuJoCo 三相机场景，典型文件包括：

- `mvp-demo/assets/scene/mujoco_humanoid_3cam_node_parallel.xml`
- `mvp-demo/assets/scene/mujoco_humanoid_uav1_3cam_node_parallel.xml`

这些场景已经满足当前阶段的关键约束：

- 每个节点包含 3 台相机。
- 三相机为紧凑刚性 rig。
- 场景中已经存在可运动的目标模型。
- 节点布局可用于后续扩展到多节点场景。

### 2.2 已有节点级脚本链

围绕三相机节点，项目已经形成了一条从仿真采集到检索评估的脚本链：

- `scripts/mj_capture_3cam_node.py`：MuJoCo 三相机节点采集，导出 `frames`、`rig.json`、`frame_times.csv`，并支持可选 GT 导出。
- `scripts/run_node_depth_anything_v2.py`：对每路相机流生成 `depth/`。
- `scripts/run_node_sam2_masks.py`：对每路相机流生成 `masks/`。
- `scripts/build_node_tracklets.py`：将三相机同步观测组织为节点级 `tracklets.json`。
- `scripts/extract_node_track_embeddings.py`：从节点级 tracklets 中提取 RGB+Geometry 融合 embedding。
- `scripts/eval_node_track_retrieval.py`：执行 query-gallery 检索并计算 `mAP`、`Recall@K`。

这意味着项目当前已经从“路线设计阶段”进入“节点级可运行原型阶段”。

### 2.3 当前最重要的冻结结论

结合现有代码、文档和实验验证，当前阶段最重要的结论有三点：

1. **MuJoCo 可以作为前期验证平台**。  
   它已经足以验证三相机 rig、同步数据流、下游 tracklet 构建与 retrieval 接口。

2. **主链路必须采用严格传感器口径**。  
   也就是主检索链只消费图像侧生成的 `masks/` 与 `depth/`，而不是直接吃 MuJoCo 的 GT。

3. **当前最合理的主线不是继续扩散方法分支，而是围绕仿真验证、评测协议和真实迁移收敛。**

---

## 3. 研究假设、边界与默认决策

### 3.1 主链路与 GT 的边界

本项目明确区分“主链路输入”和“调试/评测辅助信息”。

主链路允许消费的数据只有：

- `cams/cam*/frames/`
- `cams/cam*/masks/`
- `cams/cam*/depth/`
- `calib/rig.json`
- `frame_times.csv`

仿真器生成的：

- `masks_gt/`
- `depth_gt/`

只允许用于：

- 误差分析
- 排错
- 上界评测

**它们不参与最终检索 embedding 的主计算路径。**

### 3.2 当前阶段的任务约束

为了优先验证链路，当前阶段冻结如下设定：

- 单目标、单轨迹是默认实验单位。
- 首个成功目标是 **cross-node retrieval 跑通**，不是先追求 end-to-end 训练。
- 节点级 `track` 是检索的基本单位，不以单帧为单位做最终检索。
- `SAM2 + 人工初始化框` 是当前默认 mask 路线。

### 3.3 几何路线的选择理由

当前三相机节点采用紧凑、小基线的刚性布局，因此本文档将 **tri-camera depth fusion** 冻结为节点内几何主路线，而不把以下方法作为当前主线：

- dynamic 3DGS
- visual hull
- 依赖静态场景假设的 COLMAP + 3DGS 目标重建

原因是：

- 动态目标下，静态重建假设容易失效。
- 小基线平行光轴布局更适合“depth -> 反投影 -> 融合”的思路。
- 当前阶段需要的是“能支撑检索的几何线索”，而不是高保真动态网格。

### 3.4 当前阶段不纳入主线的内容

以下内容可以作为后续扩展，但不属于本文档的当前主交付：

- 多目标同时跟踪与多实例关联
- 端到端学习新的 3D encoder
- 大规模真实数据集采集
- 针对开放类别的大规模检测器比较
- 正式论文级别的相关工作综述与方法创新章节

---

## 4. 数据契约与系统总流程

### 4.1 数据组织约定

当前节点级数据按以下结构组织：

```text
mvp-demo/data/nodes/<node_id>/scenes/<scene_id>/
  capture_meta.json
  frame_times.csv
  calib/
    rig.json
  cams/
    cam0/
      frames/
      masks/
      depth/
      masks_gt/     # optional
      depth_gt/     # optional
    cam1/
    cam2/
  recon/
    points_fused/   # optional
  tracks/
    tracklets.json
  embeddings/
    tracks.npy
    tracks_meta.json
```

这套目录结构是后续迁移到真实系统时必须保留的核心接口。

### 4.2 关键接口定义

为了避免后续实现过程再次漂移，本文档冻结以下接口语义：

- **scene_dir**：一次连续采集窗口的根目录。
- **rig.json**：三相机内外参与 node 坐标系定义。
- **frame_times.csv**：三相机共享的同步时间轴。
- **tracklets.json**：节点级 `track` 表示，包含 `track_id`、`identity_id`、`timestamp_stems`、`per_camera` 和可选 `fused_points_paths`。
- **tracks.npy / tracks_meta.json**：用于检索与评估的 track-level embedding。

### 4.3 主流程

本文档将主流程冻结为如下形式：

```text
MuJoCo node capture / Real node capture
  -> synchronized frames + rig + timestamps
  -> image-side masks
  -> image-side depth
  -> optional depth fusion / fused points
  -> node-level tracklet construction
  -> RGB + geometry embedding extraction
  -> track-level retrieval evaluation
```

也就是说，无论前端是 MuJoCo 还是现实相机，只要产物满足相同的数据契约，下游 pipeline 不应该改变。

### 4.4 当前推荐命令链

后续执行工作建议统一使用 `mvp_demo` 环境，并按以下顺序推进：

```bash
conda activate mvp_demo

# 1) 采集或复用一个 node scene
python mvp-demo/scripts/mj_capture_3cam_node.py \
  --mjcf mvp-demo/assets/scene/mujoco_humanoid_uav1_3cam_node_parallel.xml \
  --node_id node01 \
  --traj line_nodes

# 2) 生成图像侧 depth
python mvp-demo/scripts/run_node_depth_anything_v2.py \
  --scene_dir mvp-demo/data/nodes/node01/scenes/<scene_id>

# 3) 生成图像侧 masks
python mvp-demo/scripts/run_node_sam2_masks.py \
  --scene_dir mvp-demo/data/nodes/node01/scenes/<scene_id> \
  --checkpoint <sam2_checkpoint> \
  --model_cfg configs/sam2.1/sam2.1_hiera_t.yaml \
  --camera_box cam0=x1,y1,x2,y2 \
  --camera_box cam1=x1,y1,x2,y2 \
  --camera_box cam2=x1,y1,x2,y2

# 4) 构建节点级 tracklets
python mvp-demo/scripts/build_node_tracklets.py \
  --scene_dir mvp-demo/data/nodes/node01/scenes/<scene_id> \
  --mask_subdir masks \
  --depth_subdir depth \
  --identity_id <identity_id>

# 5) 提取节点级 embeddings
python mvp-demo/scripts/extract_node_track_embeddings.py \
  --scene_dir mvp-demo/data/nodes/node01/scenes/<scene_id> \
  --rgb_backend hist \
  --geo_backend radial_hist

# 6) 执行 retrieval
python mvp-demo/scripts/eval_node_track_retrieval.py \
  --query_scene_dir <query_scene_dir> \
  --gallery_scene_dir <gallery_scene_dir>
```

这条命令链既是当前工程闭环，也是本文档后续实验设计的默认基础。

---

## 5. 方法设计

### 5.1 节点级 tracklet 构建

在本项目中，单帧观测不直接作为最终检索对象。我们以时间同步后的多相机观测序列为基础，构建节点级 `tracklet`：

- 每个时间戳对应 3 个相机的同步观测。
- 每个观测包含 `frame`、`mask`、`depth` 和可选的 `bbox`。
- 一个 `tracklet` 可以看作同一目标在一个 scene 中的多相机、多时间步观测集合。

这种设计的直接好处是：

- 可以通过多时间步 pooling 降低单帧噪声。
- 可以自然地支持后续从单帧几何过渡到 track-level 几何聚合。
- 更符合真实检索需求，因为跨节点检索本质上是“轨迹对轨迹”的匹配问题。

### 5.2 RGB 分支

当前阶段，RGB 分支不追求复杂网络结构，而强调两点：

- 先形成稳定可比较的 baseline。
- 保持实现轻量，避免早期被模型训练拖住。

因此当前推荐路线是：

- baseline：颜色直方图或现成视觉编码器
- embedding 生成后做 `L2 normalization`
- 多帧内做 `mean pooling`

这条路线的意义在于建立“几何是否有增益”的参照组。

### 5.3 Geometry 分支

Geometry 分支当前以可解释、轻量、可快速复现实验为原则，采用：

- 由 `depth + mask + rig` 构建对象几何线索
- 若已有 `points_fused`，则直接基于融合点云提取描述子
- 当前优先使用：
  - `radial_hist`
  - `FPFH`

几何分支的角色不是单独替代 RGB，而是提供：

- 对视角变化更稳定的结构信息
- 对背景干扰更低的辅助信号
- 对“同一目标不同纹理状态”的补偿能力

### 5.4 多模态融合与聚合

当前阶段的融合策略保持简单：

- 每个时间步先分别提取 RGB 与 Geometry embedding
- 二者做归一化后拼接或联合归一化
- 对 track 内多个时间步做 `mean pooling`

这样做的理由是：

- 可以尽快验证几何分支是否有真实贡献
- 便于做消融实验
- 避免复杂融合模块带来的额外变量

### 5.5 为什么当前方法适合作为研究主线

这条方法路线的价值不在于“最终性能已经最强”，而在于它满足研究推进中的四个关键条件：

1. **接口清晰**：从仿真迁移到真实环境时，下游模块不需要推倒重来。
2. **实验可控**：可以单独替换 depth、mask 或 geometry descriptor。
3. **结果可解释**：便于分析几何分支到底带来了什么。
4. **扩展性好**：后续可以平滑升级到更强的视觉编码器、多模态训练或点云网络。

---

## 6. 实验设计与阶段验收

本项目后续实验不再按“想到什么做什么”的方式推进，而是按四个阶段组织。

### 6.1 阶段 A：MuJoCo 节点正确性验证

**目标**

- 验证三相机 rig、同步和标定导出是否正确。

**需要观察的内容**

- 三相机是否能稳定输出同步帧。
- 目标在 3 台相机中是否具有足够重叠可见性。
- `rig.json` 是否能支持正确的几何投影关系。

**验收标准**

- 能稳定产出三相机 `frames/`
- 能导出 `rig.json` 和 `frame_times.csv`
- 同一 scene 内大多数时间戳下三相机都能观测到目标

### 6.2 阶段 B：严格传感器路径闭环

**目标**

- 不依赖 GT，跑通节点级 track retrieval 主链路。

**实验流程**

1. 使用图像侧方法生成 `depth/`
2. 使用图像侧方法生成 `masks/`
3. 构建 `tracklets.json`
4. 提取 `track embeddings`
5. 执行 query-gallery 检索

**验收标准**

- 成功生成 `tracks/tracklets.json`
- 成功生成 `embeddings/tracks.npy`
- 成功运行 `mAP`、`Recall@K` 评估
- 至少形成一组可重复 smoke-level 成功案例

### 6.3 阶段 C：多 scene / 多 identity 评测

**目标**

- 从“能跑通”过渡到“可比较、可分析”。

**数据组织原则**

- 每次运行单目标单轨迹
- 多次运行形成多个 `scene`
- `identity_id` 作为检索标签
- query 与 gallery 来自不同 scene

**核心对比实验**

- RGB-only
- Geometry-only
- RGB + Geometry
- GT depth vs predicted depth

**关键指标**

- `mAP`
- `Recall@1`
- `Recall@5`
- `Recall@10`

### 6.4 阶段 D：仿真到真实迁移

**目标**

- 保持下游 pipeline 不变，只替换采集侧。

**真实系统需要满足的最小条件**

- 三相机标定
- 时间同步
- 与仿真一致的目录结构
- 与仿真一致的 `rig.json + frame_times.csv` 接口

**迁移时允许改变的部分**

- 采集端脚本
- depth 来源
- mask 来源

**迁移时不应改变的部分**

- `build_node_tracklets.py` 的输入契约
- 节点级 embedding 抽取逻辑
- query-gallery 检索与评估逻辑

---

## 7. 评测指标与观察维度

为了避免后续实验只盯着单一 mAP，本项目将从三层指标观察系统表现。

### 7.1 模块层指标

- mask 可用率：目标是否被稳定分割
- depth 有效率：目标区域是否有足够有效深度
- tracklet 连续性：时间戳覆盖是否稳定

### 7.2 检索层指标

- `mAP`
- `Recall@1/5/10`
- top-K 结果的人工可解释性

### 7.3 迁移层指标

- 仿真和真实环境是否共用相同数据契约
- 更换采集端后，下游是否无需改动即可跑通
- 真实数据上的失败模式是否能对应到仿真阶段已知问题

---

## 8. 风险分析

### 8.1 深度质量风险

当前图像侧 depth 可能是相对深度或低精度深度，这会带来：

- 几何分支噪声大
- 点云形状不稳
- 不同 scene 之间尺度不一致

对应策略：

- 保留 GT depth 仅用于误差分析
- 在几何描述子阶段采用对尺度更稳健的归一化
- 先把 Geometry 作为辅助分支，而不是唯一分支

### 8.2 SAM2 依赖人工初始化

当前 mask 路线依赖首帧人工框，意味着：

- 自动化程度有限
- 多目标场景扩展成本较高

但在当前阶段，这是一个合理折中，因为：

- 目标是优先验证检索链路
- 手工初始化可以显著降低分割变量带来的干扰

### 8.3 仿真到真实存在 domain gap

真实数据可能出现：

- 光照变化更强
- 噪声更大
- 标定不完美
- 时间同步误差

因此 MuJoCo 的价值并不是“完全代表真实环境”，而是用于提前冻结接口、验证数据流和定位模块问题。

### 8.4 当前评测规模较小

在当前阶段，scene 数量和 identity 数量都有限，可能导致：

- 指标波动大
- 难以支撑正式结论

因此阶段 C 的重点不是追求大样本统计显著性，而是先构建最小但规范的评测协议。

---

## 9. 后续工作计划

### 9.1 近期目标（短周期）

1. 扩充 MuJoCo scene 数量与目标 identity 数量。
2. 建立标准化 query-gallery 划分。
3. 固化 RGB-only、Geometry-only、RGB+Geometry 三种 baseline。
4. 补齐 GT depth 与 predicted depth 的误差分析。

### 9.2 中期目标（中周期）

1. 将当前节点级 pipeline 稳定运行在多 scene 数据上。
2. 建立统一实验记录模板，包括配置、scene、指标与失败案例。
3. 形成一版可放入毕设正文的方法章节与实验章节草稿。

### 9.3 后期目标（真实部署前）

1. 设计并实现真实三相机节点采集脚本。
2. 复用相同目录结构和标定文件格式。
3. 先在真实环境中跑通最小闭环，再考虑更强模型或更复杂多目标设置。

---

## 10. 本阶段完成定义

如果后续工作满足以下条件，则可认为“MuJoCo-first research 主线”已经成立：

1. 能在严格传感器口径下完成节点级检索闭环。
2. 能在多个 scene 间稳定输出可比较的 retrieval 指标。
3. 能明确量化几何分支相对于 RGB-only 的价值。
4. 能给出迁移到真实三相机节点时必须保持不变的数据契约。

达到这一点后，项目就从“想法验证”进入“系统化研究与迁移准备”阶段。

---

## 11. 结论

基于当前项目进展，MuJoCo 三相机节点已经不再只是一个演示环境，而是后续研究工作的主验证平台。当前最重要的任务不是继续扩展方法分支，而是围绕统一的数据契约、可复现的节点级检索闭环和仿真到真实迁移路径，把已有原型收敛为一条稳定的研究主线。

因此，本文档给出的最终判断是：

- **MuJoCo-first 是可行的。**
- **严格传感器口径是必要的。**
- **tri-camera depth fusion + SAM2 + track-level retrieval 是当前阶段最合理的研究主线。**
- **后续工作应优先围绕实验规范化、评测协议和真实迁移接口展开。**
