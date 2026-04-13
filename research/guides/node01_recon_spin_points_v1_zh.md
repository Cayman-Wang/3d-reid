# node01 `recon_spin_points.py` 落地说明（v1）

## 1. 目标

- `recon_spin_points.py` 的职责不是“单时刻三相机融合”，而是“利用整条 spin scene 的多时刻观测做目标中心三维重建”。
- 第一版先做 `GT`：
  - 输入：`masks_gt + depth_gt + calib/rig.json + frame_times.csv + capture_meta.json`
  - 输出：`recon/points_recon_spin_gt/<ts>.npy`
- 下游契约保持不变，继续复用：
  - `build_node_tracklets.py`
  - `extract_node_track_embeddings.py`
  - `eval_node_track_retrieval.py`

## 2. 方法定义

- 先按每个时间戳，把三相机 `depth + mask` 回投影到 `node frame`，得到 `P_node(t)`。
- 再按 `capture_meta.json` 中冻结的 spin 协议回放目标位姿，构造 `T_target_from_node(t)`。
- 把每个时间戳的点云变到目标 canonical 坐标系：
  - `P_target(t) = T_target_from_node(t) * P_node(t)`
- 聚合整条 scene 的 `P_target(*)`，做一次 canonical voxel 下采样，并可再做一次点数上限截断，得到更完整且可控大小的 `P_target_canonical`。
- 最后对每个时间戳再投回 node 坐标系：
  - `P_node_recon(t) = T_node_from_target(t) * P_target_canonical`
- 逐时间戳落盘到 `recon/points_recon_spin_gt/<ts>.npy`。

第一版是整段 rigid 聚合，不做：

- 非刚体建模
- 可见性裁剪
- 颜色融合
- mesh 重建

## 3. 当前推荐命令

先在单 scene 上做 smoke check：

```powershell
python mvp-demo/scripts/recon_spin_points.py `
  --scene_dir mvp-demo/data/nodes/node01/scenes/mj_node01_j10_spin_circle_yp_b `
  --mask_subdir masks_gt `
  --depth_subdir depth_gt `
  --out_subdir recon/points_recon_spin_gt `
  --write_ply
```

如果后续切到预测分支，保持同一脚本，只替换输入输出子目录：

```powershell
python mvp-demo/scripts/recon_spin_points.py `
  --scene_dir mvp-demo/data/nodes/node01/scenes/mj_node01_j10_spin_circle_yp_b `
  --mask_subdir masks `
  --depth_subdir depth `
  --out_subdir recon/points_recon_spin
```

当前脚本默认还会按 scene 类型自动选择 canonical 支撑阈值：

- `GT`：固定 `1`
- `predicted static`：优先 `3`，过小则回退到 `2`
- `predicted circle`：优先 `4`，必要时再回退到 `3 / 2`

这一步的目的不是追求更复杂的重建器，而是先把“跨时刻低支撑噪声”在轻量后端里压下去。

完成 `gt` 几何后，评测入口仍走：

```powershell
python mvp-demo/scripts/run_iciscae_branch_eval.py `
  --manifest research/plans/tri_camera_node_3d_aware_reid/benchmarks/node01_recon_spin_v1.json `
  --branch gt_recon_enhanced_geometry
```

## 4. 成功标准

- `recon/points_recon_spin_gt/` 下有与 `frame_times.csv` 对齐的逐时间戳 `.npy`
- `meta.json` 与 `points_index.csv` 成功写出
- `build_node_tracklets.py --points_subdir recon/points_recon_spin_gt` 能直接跑通
- 与 `recon/points_fused_gt` 相比，目标可见面覆盖更完整，侧后方区域补全更明显

## 4.1 当前推荐诊断入口

如果想先看 predicted recon 失败在什么地方，优先跑：

```powershell
python mvp-demo/scripts/analyze_spin_recon_quality.py `
  --scene_dir mvp-demo/data/nodes/node01/scenes/mj_node01_j10_spin_circle_yp_b
```

该脚本会输出：

- `canonical_compare.png`
- `counts_compare.png`
- `summary.md`
- `summary.json`

重点先看三件事：

- predicted 输入点数相对 GT 是明显偏多还是偏少
- canonical 点云是否出现体积膨胀或漂移
- 支撑过滤后保留下来的体素比例是否过低

## 5. 后续如何接入 3DGS

当前 3DGS 不是主线重建器，只作为后续可替换的 canonical geometry 生成后端。

推荐接入顺序固定为：

1. 先稳定 `recon/points_recon_spin_gt`
2. 再稳定 `recon/points_recon_spin`
3. 最后再尝试用 3DGS 替换“canonical 几何生成”这一步

接入原则：

- 不改 `tracklets -> embeddings -> eval` 契约
- 不让下游直接消费高斯参数
- 3DGS 最终仍需导出逐时间戳点云到：
  - `recon/points_recon_spin_gt`
  - `recon/points_recon_spin`

建议做法：

- 先把一条 spin scene 整理成 object-centric 的 3DGS 训练输入
- 在 canonical 目标坐标系中训练或拟合高斯表示
- 再按每个时间戳的 `T_node_from_target(t)` 渲染或采样当前时刻的目标几何
- 最终继续落盘成与 `frame_times.csv` 对齐的 `.npy` 点云

因此，3DGS 后续替换的是“canonical 几何的生成方式”，而不是替换当前检索主链接口。
