# tri_camera_node_3d_aware_reid 主计划（中文）

- 创建日期：2026-03-15
- 状态：方向已冻结，当前进入 benchmark 协议收口阶段

## 一、目标
- 在 MuJoCo 三相机节点上完成严格传感器口径的 `3D-aware track retrieval` 闭环验证。
- 证明当前数据契约可以支撑从仿真节点平滑迁移到真实三相机节点，而不需要重写下游 tracklet、embedding 和 retrieval 模块。
- 用可复现的对比实验回答一个核心问题：几何信息是否真的能提升跨视角、跨 scene、跨节点实例检索。
- `M0-M4` 的正式验收目标固定为 `single-node, cross-scene, track-level retrieval`。
- `cross-node retrieval` 是 `M5` 之后的扩展目标，不作为当前 benchmark 成功条件。

## 二、当前工程能力与问题定义
### 2.1 当前已具备的工程能力
- 当前仓库已经具备节点级主链脚本：`mj_capture_3cam_node.py -> run_node_depth_anything_v2.py -> run_node_sam2_masks.py -> build_node_tracklets.py -> extract_node_track_embeddings.py -> eval_node_track_retrieval.py`。
- 当前仓库已经具备几何增强入口：`recon_fuse_depth_points.py` 可用于多相机 depth+mask 融合点云。
- 当前仓库同时保留一条 `YOLO 门控 + 3DGS` 静态场景 demo 链，但该链不再作为研究主线，只保留为辅助演示和参考实现。
- 当前仓库已经存在可复用的 `node01` 多个 MuJoCo scene，可用于单节点、跨 scene 的 benchmark 起步。

### 2.2 当前仍需补齐的关键缺口
- 当前正式 benchmark 还没有冻结 scene 清单、identity 标注、query/gallery 角色和评测结果落盘位置。
- 当前 `extract_node_track_embeddings.py` 的 geometry 分支只直接消费 `fused_points_paths`，尚未实现“直接从 predicted depth 构造几何描述子”的适配层。
- 当前 `run_node_sam2_masks.py` 默认输出 `masks/obj_XXX/<ts>.png`，而 `recon_fuse_depth_points.py` 默认读取平铺 `masks/<ts>.png`；若不统一 mask 布局，`M4` 会断链。
- 当前仓库还没有成型的 `node02` benchmark scene，因此 `cross-node retrieval` 不能作为近阶段正式验收目标。

### 2.3 当前阶段的问题定义
- 当前阶段关注的是节点级 `track-level retrieval`，不是单帧分类，也不是多目标大系统。
- 近阶段的正式 benchmark 固定为：同一 `node01` 下、多 scene、跨 scene 的 `track` 检索。
- `cross-node retrieval`、真实三相机迁移和更大规模数据采集属于后续扩展，而不是当前 benchmark 的成功条件。

## 三、方案总览
- 数据契约固定为：`frames + masks + depth + rig.json + frame_times.csv`。
- 主链路坚持严格传感器口径：主检索特征只来自图像侧生成的 `masks/` 和 `depth/`，MuJoCo 的 `masks_gt/`、`depth_gt/` 只用于评测和排错。
- 检索单位固定为 `track`，每个 `track` 由一个 scene 中同一 identity 的多帧、多相机同步观测组成。
- 研究目标矩阵保持三组对比：`RGB-only`、`RGB + predicted-depth geometry`、`RGB + fused geometry`。
- 近阶段的工程执行顺序固定为：
  - 先完成 `RGB-only` baseline。
  - 再补 `predicted-depth geometry adapter`。
  - 最后跑通 `fused geometry` 对比与消融。
- 检索层保持轻量实现：track mean pooling + cosine retrieval + `mAP / Recall@K`。

## 四、里程碑
### M0 协议冻结与最小实验协议落地
- 交付物：
  - 冻结研究主线、输入输出契约、评测矩阵和非目标。
  - 冻结最小 benchmark manifest：`2-3` 个 identity、每个 identity 至少 `2` 个 scene，明确 query/gallery 划分。
- 完成定义：
  - 任何后续实验都能落到同一 scene 契约和同一评测脚本上。
  - 每个 scene 都能明确写出 `identity_id`、`split_role`、`mask_source`、`depth_source` 和 `eval_out_json`。
  - 当前正式 benchmark 只使用 `node01` scene 做单节点、跨 scene 检索。

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
  - 对每个候选 scene 做覆盖率检查，确认 `frames + masks + depth` 三路对齐后的有效同步时间戳数量。
- 完成定义：
  - 每个正式 benchmark scene 的有效同步时间戳不少于 `5`。
  - 若一个 scene 在 `frames + masks + depth` 三路对齐后少于 `5` 个有效时间戳，则该 scene 不进入正式评测。
  - 任何缺失或失败 scene 都能定位到具体相机和时间戳。

### M3 节点级检索闭环跑通
- 交付物：
  - `build_node_tracklets.py` 生成节点级 `tracks/tracklets.json`。
  - `extract_node_track_embeddings.py` 生成 `embeddings/tracks.npy` 与 `tracks_meta.json`。
  - `eval_node_track_retrieval.py` 输出 `RGB-only` query-gallery 检索结果，并落盘评测 JSON。
- 完成定义：
  - 至少跑通一轮 `RGB-only` baseline。
  - 正式检索评测默认开启 `exclude_same_track_id` 和 `exclude_same_scene`。
  - 每个 query 都存在至少 `1` 个 gallery 正样本。
  - 评测结果必须以 JSON 形式落盘，而不是只看终端打印。

### M4 几何分支强化
- 交付物：
  - 实现 `predicted-depth geometry adapter`，把 predicted depth 真正转换成可用几何描述子。
  - 将 `recon_fuse_depth_points.py` 纳入默认实验链，生成 `recon/points_fused/<ts>.npy`。
  - 对比 `radial_hist` 与 `open3d_fpfh` 两类几何描述子。
- 完成定义：
  - 跑出 `RGB + predicted-depth geometry` 结果。
  - 跑出 `RGB + fused geometry` 结果。
  - 每个 query 都存在明确正样本定义。
  - 形成三组对比：`RGB-only`、`RGB + predicted depth`、`RGB + fused geometry`。

### M5 系统性误差分析与 single-node benchmark 收口
- 交付物：
  - predicted masks/depth 与 GT masks/depth 的上界对比。
  - geometry backend、scene 设置、query/gallery 划分的失败案例分析。
- 完成定义：
  - 能明确说明几何增益来自哪里，或者失败主要来自哪里。
  - 形成可直接用于汇报和论文写作的表格、案例和结论草稿。
  - 完成单节点、跨 scene benchmark 的收口后，才启动跨节点扩展。

### M6 跨节点与真实三相机节点迁移准备
- 交付物：
  - 一组跨节点或真实三相机小规模采集数据，目录契约与 MuJoCo scene 保持一致。
  - 用最少 glue code 将真实数据接入现有 `tracklets -> embeddings -> retrieval` 链。
- 完成定义：
  - 跨节点或真实数据能复用当前评测入口。
  - 能验证“仿真到真实”的接口稳定性。

## 五、近期工作规划
- 第 1 周：
  - 完成 M0。
  - 先完成 benchmark manifest 和 scene 清单。
  - 再按 manifest 补齐 depth、mask、identity 和评测结果落盘位置。
- 第 2 周：
  - 完成 M3 的 `RGB-only` 首轮结果。
  - 排查 tracklet 覆盖率、identity 标注、query/gallery 划分中的缺口。
- 第 3 周：
  - 实现 `predicted-depth geometry adapter`。
  - 接入 `recon_fuse_depth_points.py`，完成 `RGB + fused geometry` 对比。
  - 输出首版消融结果与失败案例。
- 第 4 周及以后：
  - 先完成 M5 的 single-node benchmark 收口。
  - 再根据结果决定是优先做跨节点扩展，还是先推进真实节点小规模迁移。

## 六、验收标准
- 数据完整性：
  - `frame_times.csv` 与磁盘文件一一对应。
  - 每个 scene 的三路 frame、depth、mask 命名和时间戳一致。
  - 正式 benchmark 的 mask 输入布局固定为平铺 `cams/cam*/masks/<ts>.png`。
  - 每个正式 benchmark scene 的有效同步时间戳不少于 `5`。
- 几何口径：
  - rig 验证通过，baseline 和 forward 方向满足约束。
  - depth、mask、frame 的分辨率和时间戳对齐。
- 检索有效性：
  - `tracks.npy` 与 `tracks_meta.json` 数量一致、维度一致。
  - `mAP`、`Recall@1/5/10` 可稳定输出。
  - 每个 query 至少存在一个 gallery 正样本。
  - 正式评测结果必须落到 `mvp-demo/output/evals/<benchmark_id>/...` 下的 JSON 文件中。
- 研究结论：
  - 必须给出 `RGB-only`、`RGB + predicted depth`、`RGB + fused geometry` 三组对比。
  - 如果 geometry 没有超过 `RGB-only`，也必须给出失败分析，而不是只保留数值。

## 七、风险与对策
- `identity_id` 缺失会直接导致评测指标不成立。
  - 对策：在 capture 阶段把 `identity_id` 写入 `capture_meta.target.identity_id`，并在 benchmark manifest 中二次核对。
- SAM2 或 depth 覆盖率不足会导致 tracklet 无法构建。
  - 对策：先做覆盖率检查，再进入 embedding 提取。
- 小 baseline 下 predicted depth 噪声大，可能导致 geometry 分支不稳定。
  - 对策：先把 `RGB + predicted depth` 视作弱几何基线，再用 fused points 判断几何是否真正带来收益。
- `SAM2 camera_box` 的手工初始化会直接影响复现性。
  - 对策：把每个 scene 的 `camera_box` 固定进 benchmark manifest，并作为复现实验的必填字段。
- SAM2 的 `obj_XXX` 输出与几何融合脚本默认读取的平铺 mask 布局不一致，会导致 `M4` 断链。
  - 对策：正式 benchmark 输入统一采用平铺 `masks/<ts>.png`，嵌套结构仅作为原始中间结果保存。
- `YOLO 门控 + 3DGS` 链与主线混用会分散资源。
  - 对策：明确将其降级为辅助 demo，不纳入近阶段里程碑。

## 八、默认假设
- 默认中文文档
- 默认按“讨论 -> 冻结 -> 生成”执行
- 当前默认单目标、单轨迹、track-level retrieval
- 近阶段不处理多目标关联与端到端 3D encoder 训练
- MuJoCo GT 只用于评测和排错，不进入主检索特征
- benchmark manifest 先写入本主计划文本，不额外新增 planning 文件
- 正式评测结果默认保存在 `mvp-demo/output/evals/<benchmark_id>/...`，不写入 `research/`

## 九、最小 benchmark manifest
正式 benchmark 先不单独新增文件，但必须在执行时维护一份字段完整的 manifest。推荐至少包含以下字段：

```text
scene_dir
node_id
scene_id
identity_id
split_role            # query | gallery | both
mask_source           # sam2_pred | masks_gt_debug
depth_source          # depth_anything_v2 | depth_gt_debug
mask_layout           # 固定为 flat
sam2_camera_boxes
min_valid_timestamps
eval_out_json
```

字段约束如下：
- `scene_dir / node_id / scene_id` 用于唯一定位 scene。
- `identity_id` 必须与 `capture_meta.target.identity_id` 一致，不能依赖 scene 名自动回填。
- `split_role` 只能取 `query`、`gallery` 或 `both`，并且必须保证每个 query 至少存在一个 gallery 正样本。
- `mask_source`、`depth_source` 必须显式记录主链路到底使用 predicted 结果还是 GT debug 结果。
- `mask_layout` 在正式 benchmark 中固定为 `flat`；若原始 SAM2 结果是 `obj_XXX`，必须先转换成平铺布局再进入评测。
- `sam2_camera_boxes` 需要按 `cam0/cam1/cam2` 记录初始化框，防止同一 scene 重跑时结果漂移。
- `min_valid_timestamps` 的默认门槛是 `5`；低于该值的 scene 不进入正式 benchmark。
- `eval_out_json` 默认落到 `mvp-demo/output/evals/<benchmark_id>/<experiment_name>.json`。
