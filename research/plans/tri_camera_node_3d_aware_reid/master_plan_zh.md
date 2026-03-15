# tri_camera_node_3d_aware_reid 主计划（中文）

- 创建日期：2026-03-15
- 状态：方向已冻结，进入执行阶段

## 一、目标
- 在 MuJoCo 三相机节点上完成严格传感器口径的 `3D-aware track retrieval` 闭环验证。
- 证明当前数据契约可以支撑从仿真节点平滑迁移到真实三相机节点，而不需要重写下游 tracklet、embedding 和 retrieval 模块。
- 用可复现的对比实验回答一个核心问题：几何信息是否真的能提升跨视角、跨 scene、跨节点实例检索。

## 二、项目基础与问题定义
- 当前仓库已经具备节点级主链脚本：`mj_capture_3cam_node.py -> run_node_depth_anything_v2.py -> run_node_sam2_masks.py -> build_node_tracklets.py -> extract_node_track_embeddings.py -> eval_node_track_retrieval.py`。
- 当前仓库已经具备几何增强入口：`recon_fuse_depth_points.py` 可用于多相机 depth+mask 融合点云。
- 当前仓库同时保留一条 `YOLO 门控 + 3DGS` 静态场景 demo 链，但该链不再作为研究主线，只保留为辅助演示和参考实现。
- 当前阶段关注的是节点级 `track-level retrieval`，不是单帧分类，也不是多目标大系统。

## 三、方案总览
- 数据契约固定为：`frames + masks + depth + rig.json + frame_times.csv`。
- 主链路坚持严格传感器口径：主检索特征只来自图像侧生成的 `masks/` 和 `depth/`，MuJoCo 的 `masks_gt/`、`depth_gt/` 只用于评测和排错。
- 检索单位固定为 `track`，每个 `track` 由一个 scene 中同一 identity 的多帧、多相机同步观测组成。
- 近阶段默认采用轻量混合表征：
  - 外观分支：`RGB-only` baseline 或 `RGB + mask` 裁剪后的外观 embedding。
  - 几何分支：`predicted depth` 或 `fused geometry` 生成轻量描述子。
  - 检索层：track mean pooling + cosine retrieval + `mAP / Recall@K`。

## 四、里程碑
### M0 协议冻结与最小实验协议落地
- 交付物：
  - 冻结研究主线、输入输出契约、评测矩阵和非目标。
  - 冻结最小实验集定义：`2-3` 个 identity、至少 `2` 个 scenes、明确 query/gallery 划分。
- 完成定义：
  - 任何后续实验都能落到同一 scene 契约和同一评测脚本上。
  - 每个 scene 都能明确写出 `identity_id` 和其在评测中的角色。

### M1 三相机节点采集与几何口径验证
- 交付物：
  - 一批合规 MuJoCo scene，包含三路同步 `frames/`、`frame_times.csv`、`calib/rig.json`。
  - 对 node rig 的 baseline、forward 方向、平行光轴约束做数值校验。
- 完成定义：
  - 每个 `ts_us` 都存在三路 frame。
  - `mj_validate_3cam_node.py` 对目标节点返回通过结果。

### M2 图像侧 depth 与 masks 基线
- 交付物：
  - 对选定 scenes 跑 `run_node_depth_anything_v2.py`，补齐 `cams/cam*/depth/`。
  - 对选定 scenes 跑 `run_node_sam2_masks.py`，补齐 `cams/cam*/masks/`。
- 完成定义：
  - depth 和 masks 对时间戳的覆盖率足以支撑后续 tracklet 构建。
  - 任何缺失或失败 scene 都能定位到具体相机和时间戳。

### M3 节点级检索闭环跑通
- 交付物：
  - `build_node_tracklets.py` 生成节点级 `tracks/tracklets.json`。
  - `extract_node_track_embeddings.py` 生成 `embeddings/tracks.npy` 与 `tracks_meta.json`。
  - `eval_node_track_retrieval.py` 输出 query-gallery 检索结果。
- 完成定义：
  - 至少跑通一轮 `RGB-only` baseline。
  - 至少跑通一轮 `RGB + predicted depth` baseline。
  - 每个 query 都存在明确正样本定义。

### M4 几何分支强化
- 交付物：
  - 将 `recon_fuse_depth_points.py` 纳入默认实验链，生成 `recon/points_fused/<ts>.npy`。
  - 对比 `radial_hist` 与 `open3d_fpfh` 两类几何描述子。
- 完成定义：
  - 跑出 `RGB + fused geometry` 结果。
  - 形成三组对比：`RGB-only`、`RGB + predicted depth`、`RGB + fused geometry`。

### M5 系统性误差分析与消融
- 交付物：
  - predicted masks/depth 与 GT masks/depth 的上界对比。
  - geometry backend、scene 设置、query/gallery 划分的失败案例分析。
- 完成定义：
  - 能明确说明几何增益来自哪里，或者失败主要来自哪里。
  - 形成可直接用于汇报和论文写作的表格、案例和结论草稿。

### M6 真实三相机节点迁移准备
- 交付物：
  - 一组真实三相机小规模采集数据，目录契约与 MuJoCo scene 保持一致。
  - 用最少 glue code 将真实数据接入现有 `tracklets -> embeddings -> retrieval` 链。
- 完成定义：
  - 真实数据能复用当前评测入口。
  - 能验证“仿真到真实”的接口稳定性。

## 五、近期工作规划
- 第 1 周：
  - 完成 M0。
  - 选定最小实验集并整理 scene 清单。
  - 对这批 scene 补齐 predicted depth 和 SAM2 masks。
- 第 2 周：
  - 完成 M3 的 `RGB-only` 与 `RGB + predicted depth` 首轮结果。
  - 排查 tracklet 覆盖率、identity 标注、query/gallery 划分中的缺口。
- 第 3 周：
  - 接入 `recon_fuse_depth_points.py`，完成 `RGB + fused geometry` 对比。
  - 输出首版消融结果与失败案例。
- 第 4 周及以后：
  - 根据结果决定是优先做真实节点小规模迁移，还是先继续加固 MuJoCo 侧几何分支。

## 六、验收标准
- 数据完整性：
  - `frame_times.csv` 与磁盘文件一一对应。
  - 每个 scene 的三路 frame、depth、mask 命名和时间戳一致。
- 几何口径：
  - rig 验证通过，baseline 和 forward 方向满足约束。
  - depth、mask、frame 的分辨率和时间戳对齐。
- 检索有效性：
  - `tracks.npy` 与 `tracks_meta.json` 数量一致、维度一致。
  - `mAP`、`Recall@1/5/10` 可稳定输出。
  - 每个 query 至少存在一个 gallery 正样本。
- 研究结论：
  - 必须给出 `RGB-only`、`RGB + predicted depth`、`RGB + fused geometry` 三组对比。
  - 如果 geometry 没有超过 `RGB-only`，也必须给出失败分析，而不是只保留数值。

## 七、风险与对策
- `identity_id` 缺失会直接导致评测指标不成立。
  - 对策：最小实验集先人工确认 identity 与 query/gallery 划分。
- SAM2 或 depth 覆盖率不足会导致 tracklet 无法构建。
  - 对策：先做覆盖率检查，再进入 embedding 提取。
- 小 baseline 下 predicted depth 噪声大，可能导致 geometry 分支不稳定。
  - 对策：先把 `RGB + predicted depth` 视作弱几何基线，再用 fused points 判断几何是否真正带来收益。
- `YOLO 门控 + 3DGS` 链与主线混用会分散资源。
  - 对策：明确将其降级为辅助 demo，不纳入近阶段里程碑。

## 八、默认假设
- 默认中文文档
- 默认按“讨论 -> 冻结 -> 生成”执行
- 当前默认单目标、单轨迹、track-level retrieval
- 近阶段不处理多目标关联与端到端 3D encoder 训练
- MuJoCo GT 只用于评测和排错，不进入主检索特征
