# ICISCAE 本周收口包（2026-03-22）

## 1. 收口结论

- 本周冻结的硬交付已经全部完成：
  - `node01` 的正式 `3 identities x 2 scenes` 已落盘；
  - `6` 个正式 scene 已补齐 predicted `depth` 与 flat `masks`；
  - `RGB-only (CLIP + no geometry)` 正式结果已落盘；
  - 可用于汇报的结果表、案例图和失败分析已形成。
- 当前正式 baseline 的全量 summary 位于 `mvp-demo/output/evals/iciscae_node01_uav_v1/rgb_only/all_formal_queries.json`：
  - `mAP = 0.6389`
  - `recall@1 = 0.3333`
  - `recall@5 = 1.0000`
  - `recall@10 = 1.0000`
- 本周工作已经从“补场景和补链路”切换为“冻结正式 baseline、写论文并补几何分支”。

## 2. Benchmark Protocol

- 任务：`node01` 的 `single-node, cross-scene, track-level retrieval`
- 身份集合：`j10 / uav1 / dji_mavic`
- 正式 scene：
  - `mj_node01_j10_line_nodes_a`
  - `mj_node01_j10_circle_xz_b`
  - `mj_node01_uav1_line_nodes_a`
  - `mj_node01_uav1_circle_xz_b`
  - `mj_node01_dji_mavic_line_nodes_a`
  - `mj_node01_dji_mavic_circle_xz_b`
- 主链输入固定为：
  - `frames`
  - predicted `masks`
  - predicted `depth`
  - `rig.json`
  - `frame_times.csv`
- 主链约束：
  - `mask_layout = flat`
  - `min_valid_timestamps = 5`
  - 评测统一开启 `exclude_same_track_id` 与 `exclude_same_scene`
- 本轮正式结果固定为 `RGB-only`：
  - `extract_node_track_embeddings.py --rgb_backend clip --geo_backend none`

## 3. 实验设置与执行细节

- 执行日期：`2026-03-22`
- 工作目录：`D:\grad_project_ascii\mvp-demo`
- 环境：`mvp_demo`
- 核心依赖：
  - `torch 2.10.0+cu128`
  - `mujoco 3.6.0`
  - `open_clip 3.3.0`
- predicted depth：
  - 脚本：`mvp-demo/scripts/run_node_depth_anything_v2.py`
  - 模型：`depth-anything/Depth-Anything-V2-Small-hf`
  - 下载方式：`HF_ENDPOINT=https://hf-mirror.com`
- predicted masks：
  - 脚本：`mvp-demo/scripts/run_node_sam2_masks.py`
  - checkpoint：`mvp-demo/third_party/sam2/checkpoints/sam2.1_hiera_tiny.pt`
  - config：`mvp-demo/third_party/sam2/sam2/configs/sam2.1/sam2.1_hiera_t.yaml`
  - 冻结 prompt box：
    - `cam0 = [490, 300, 660, 720]`
    - `cam1 = [600, 330, 800, 720]`
    - `cam2 = [600, 270, 800, 640]`
  - flat 化辅助脚本：`mvp-demo/scripts/flatten_node_sam2_masks.py`
- Windows 兼容性补丁：
  - `build_node_tracklets.py`、`extract_node_track_embeddings.py`、`export_node_presentation_assets.py` 已补强 Unicode 路径读取
  - `run_node_sam2_masks.py` 已改为 `cv2.imencode(...).tofile(...)` 写 PNG，避免中文路径下 `cv2.imwrite` 失败

## 4. RGB-only 正式结果表

| query_scene | identity_id | mAP | recall@1 | top1_relevant | top1_track | top1_score |
| --- | --- | --- | --- | --- | --- | --- |
| `mj_node01_j10_line_nodes_a` | `j10` | `1.0000` | `1.0000` | `YES` | `node01_mj_node01_j10_circle_xz_b` | `0.9902` |
| `mj_node01_j10_circle_xz_b` | `j10` | `1.0000` | `1.0000` | `YES` | `node01_mj_node01_j10_line_nodes_a` | `0.9902` |
| `mj_node01_uav1_line_nodes_a` | `uav1` | `0.5000` | `0.0000` | `NO` | `node01_mj_node01_dji_mavic_line_nodes_a` | `0.9999` |
| `mj_node01_uav1_circle_xz_b` | `uav1` | `0.5000` | `0.0000` | `NO` | `node01_mj_node01_dji_mavic_circle_xz_b` | `0.9995` |
| `mj_node01_dji_mavic_line_nodes_a` | `dji_mavic` | `0.5000` | `0.0000` | `NO` | `node01_mj_node01_uav1_line_nodes_a` | `0.9999` |
| `mj_node01_dji_mavic_circle_xz_b` | `dji_mavic` | `0.3333` | `0.0000` | `NO` | `node01_mj_node01_uav1_circle_xz_b` | `0.9995` |

全量汇总：

| metric | value |
| --- | --- |
| `num_queries` | `6` |
| `num_gallery` | `6` |
| `metric_queries` | `6` |
| `mAP` | `0.6389` |
| `recall@1` | `0.3333` |
| `recall@5` | `1.0000` |
| `recall@10` | `1.0000` |

## 5. 结果解释与失败分析

- `j10` 两条 query 都能把同身份的另一条 scene 召回到 `rank1`，说明当前 baseline 至少能稳定分开 `j10` 与其余 aircraft-like 目标。
- `uav1 / dji_mavic` 的四条 query 全部发生 top1 混淆，且相似度非常接近：
  - `uav1_line_nodes_a` 与 `dji_mavic_line_nodes_a`
  - `uav1_circle_xz_b` 与 `dji_mavic_circle_xz_b`
  - `dji_mavic_line_nodes_a` 与 `uav1_line_nodes_a`
  - `dji_mavic_circle_xz_b` 与 `uav1_circle_xz_b`
- 其中 `mj_node01_dji_mavic_circle_xz_b` 最差，正确正样本已经退到 `rank3`，对应 `mAP = 0.3333`。
- 因此，当前最合理的技术判断不是“RGB-only 已经够用”，而是“RGB-only 已经足够作为正式 baseline，但必须靠几何分支去处理近形态目标混淆”。

## 6. 案例图与可视化素材

推荐直接引用以下正式素材：

- `mvp-demo/data/nodes/node01/scenes/mj_node01_j10_line_nodes_a/presentation_assets/overview_000001500000.png`
- `mvp-demo/data/nodes/node01/scenes/mj_node01_j10_circle_xz_b/presentation_assets/triview_video.mp4`
- `mvp-demo/data/nodes/node01/scenes/mj_node01_uav1_circle_xz_b/presentation_assets/overview_000001500000.png`
- `mvp-demo/data/nodes/node01/scenes/mj_node01_dji_mavic_circle_xz_b/presentation_assets/overview_000001500000.png`
- `mvp-demo/data/nodes/node01/scenes/mj_node01_dji_mavic_line_nodes_a/presentation_assets/triview_video.mp4`

## 7. 下周建议顺序

1. 先冻结当前 `RGB-only` 正式结果，把 `mAP=0.6389`、`recall@1=0.3333` 作为 baseline 写进文稿。
2. 立即在同一 `6-scene` benchmark 上补 `rgb_predicted_depth_geometry`，验证 predicted depth 是否能修复 `uav1 / dji_mavic` 的 top1 混淆。
3. 再补 `rgb_fused_geometry`，把当前 `presentation_assets` 复用为论文图表和导师汇报素材。
4. 若后续还要优化 `RGB-only`，前提应当是更换 backbone 或扩充 scene 多样性，而不是继续重复当前 `CLIP + no geometry` 配置。
