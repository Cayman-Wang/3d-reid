# Node01 Spin GT 验证执行指南

适用范围：`node01_recon_spin_v1` 的 GT 验证阶段。  
目标：先完成真实 `J10` 两条 spin scene 的采集/导出/`points_fused_gt` 验证，再决定是否扩到 `uav1` / `su34`，最后才进入 `points_recon_spin_*`。

## 0. 先搭建外部 ASCII runtime

运行目录固定为：`D:\node01_spin_runtime_ascii`

- `scripts` / `research` / `assets/scene` 来自 `grad_project_recon`
- `assets/models` 来自 `grad_project`
- `data` / `output` 只写在 runtime 本地

```powershell
powershell -ExecutionPolicy Bypass -File mvp-demo/scripts/setup_node01_spin_runtime.ps1
cd D:\node01_spin_runtime_ascii
```

后续本指南中的所有命令，都默认在 `D:\node01_spin_runtime_ascii` 下执行。

## 1. 先做 MJCF / 资产前置检查

在 Windows 上，MuJoCo 应从 ASCII-safe 工作路径运行；真实 `J10` 还要求 `mvp-demo/assets/models/J10/...` 资源齐全。

```powershell
python mvp-demo/scripts/validate_node_spin_scene.py `
  --mjcf mvp-demo/assets/scene/mujoco_3cam_node_parallel_j10.xml
```

通过标准：

- `mjcf_preflight_ok: True`
- 没有 `missing_refs`
- 没有非 ASCII 路径告警

如果这里失败，不进入采集。

## 2. 采真实 `J10 static_spin`

冻结参数来源：`research/plans/tri_camera_node_3d_aware_reid/benchmarks/node01_recon_spin_v1.json`

```powershell
python mvp-demo/scripts/mj_capture_3cam_node.py `
  --mjcf mvp-demo/assets/scene/mujoco_3cam_node_parallel_j10.xml `
  --out_root mvp-demo/data/nodes `
  --node_id node01 `
  --scene_id mj_node01_j10_spin_static_yp20_a `
  --target_body target `
  --identity_id j10 `
  --traj static_spin_yaw_pitch `
  --traj_center "0 6 2" `
  --traj_period 8 `
  --fps 30 `
  --seconds 8 `
  --save_depth `
  --save_masks_gt `
  --mask_subdir masks_gt `
  --depth_subdir depth_gt `
  --yaw_start_deg -45 `
  --yaw_end_deg 45 `
  --pitch_amp_deg 20 `
  --pitch_period 8
```

## 3. 导出组会素材并生成 `points_fused_gt`

```powershell
python mvp-demo/scripts/export_node_presentation_assets.py `
  --scene_dir mvp-demo/data/nodes/node01/scenes/mj_node01_j10_spin_static_yp20_a `
  --mask_subdir masks_gt `
  --depth_subdir depth_gt
```

```powershell
python mvp-demo/scripts/recon_fuse_depth_points.py `
  --scene_dir mvp-demo/data/nodes/node01/scenes/mj_node01_j10_spin_static_yp20_a `
  --cams cam0,cam1,cam2 `
  --depth_subdir depth_gt `
  --mask_subdir masks_gt `
  --out_subdir recon/points_fused_gt
```

## 4. 用统一验证脚本做 gate

```powershell
python mvp-demo/scripts/validate_node_spin_scene.py `
  --scene_dir mvp-demo/data/nodes/node01/scenes/mj_node01_j10_spin_static_yp20_a `
  --scene_id mj_node01_j10_spin_static_yp20_a
```

通过标准：

- `240` 个唯一时间戳
- 三路 `frames/masks_gt/depth_gt` 齐全
- `capture_meta.json` 与冻结协议一致
- `node_id / scene_id / identity_id / target_body` 必须与 manifest 对齐
- `traj / traj_center / traj_radius / traj_period` 必须与 manifest 对齐
- `spin_axes / yaw_start_deg / yaw_end_deg / yaw_profile / pitch_amp_deg / pitch_period / pitch_profile / roll_amp_deg` 必须与 manifest 对齐
- `presentation_assets/triview_video.mp4` 存在，且有至少 3 张 `overview_*.png`
- `recon/points_fused_gt` 数量与时间戳对齐，空点云数为 `0`

注意：旧的 `mj_node01_j10_spin_static_yp_a` 只是早期 proxy/低俯仰样本；当前正式验证应以 `mj_node01_j10_spin_static_yp20_a` 及其冻结参数为准。

## 5. `static_spin` 通过后再采 `J10 circle`

只有真实 `J10 static_spin` 通过后，才继续：

```powershell
python mvp-demo/scripts/mj_capture_3cam_node.py `
  --mjcf mvp-demo/assets/scene/mujoco_3cam_node_parallel_j10.xml `
  --out_root mvp-demo/data/nodes `
  --node_id node01 `
  --scene_id mj_node01_j10_spin_circle_yp_b `
  --target_body target `
  --identity_id j10 `
  --traj circle_xz_spin_yaw_pitch `
  --traj_center "0 6 2" `
  --traj_radius 1 `
  --traj_period 8 `
  --fps 30 `
  --seconds 8 `
  --save_depth `
  --save_masks_gt `
  --mask_subdir masks_gt `
  --depth_subdir depth_gt `
  --yaw_start_deg -45 `
  --yaw_end_deg 45 `
  --pitch_amp_deg 20 `
  --pitch_period 8
```

随后重复相同的导出、重建、验证链：

- `export_node_presentation_assets.py`
- `recon_fuse_depth_points.py`
- `validate_node_spin_scene.py --scene_id mj_node01_j10_spin_circle_yp_b`

## 6. 后续门槛

- 只有 `J10 static + J10 circle` 都通过，才扩到 `uav1` / `su34`
- 只有六条 GT spin scene 都通过，才开始实现：
  - `recon/points_recon_spin_gt`
  - `recon/points_recon_spin`
