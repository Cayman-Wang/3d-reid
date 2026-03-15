# MuJoCo 三相机节点（平行光轴）已完成内容与验证方法（Handoff）

目的：只聚焦“相机节点（每节点 3 相机）”在 MuJoCo 中是否已经正确创建，并提供可重复的验证步骤，方便你开启新对话后继续推进。

---

## 1. 已完成的文件/脚本

**相机节点 MJCF（节点定义）**
- `mvp-demo/assets/mujoco_3cam_node_parallel.xml`
  - `body name="node01"` 下挂 3 个 `<camera>`：`node01_cam0/1/2`
  - 三相机在 `x-z` 平面围成小圈：当前直径 `0.9m`（半径 `0.45m`，满足 <= 1m）
  - 三相机光轴平行：三者 `xyaxes` 完全一致
  - 约定：`x` 右、`y` 前（look direction）、`z` 上
  - 说明：MuJoCo 相机看向本地 `-Z`，因此用 `xyaxes="1 0 0  0 0 1"` 让相机朝世界/节点 `+Y`

**相机节点 MJCF（Humanoid 场景版本，可用于更复杂背景验证）**
- `mvp-demo/assets/mujoco_humanoid_3cam_node_parallel.xml`
  - 基于 `mvp-demo/third_party/mujoco-3.4.0/model/humanoid/humanoid.xml` 场景
  - 同样包含 `node01_cam0/1/2`（平行光轴、小圈三目）
  - `node01` 放在 `pos="0 -3 1.6"`，相机默认朝 `+Y`，方便把 humanoid（在 `y≈0`）放进视野
  - 额外保留 `body name="target"`，这样采集/Viewer 脚本可保持默认参数直接运行

**离屏采集（验证：能否正确输出三路同步数据 + 标定文件）**
- `mvp-demo/scripts/mj_capture_3cam_node.py`
  - 输出目录：`mvp-demo/data/nodes/<node_id>/scenes/<scene_id>/...`
  - 每路相机输出：`frames/*.jpg`、`masks/*.png`
  - 可选输出深度：`--save_depth` -> `depth/*.npy`
  - 导出标定：`calib/rig.json`（含 `K` + `T_node_from_cam`，相机坐标采用 CV 口径：x 右、y 下、z 前）

**GUI 查看（验证：场景里能否看到相机节点 + 目标沿节点间直线运动）**
- `mvp-demo/scripts/mj_view_3cam_node.py`
  - 默认加载 Humanoid 场景：`mvp-demo/assets/mujoco_humanoid_3cam_node_parallel.xml`
  - 打开 MuJoCo viewer 后，`target` 会在 `node01` 与 `node02` 的连线方向上做往返直线运动（并整体平移到更靠前的 Y/Z，避免贴近相机平面）
  - 用自由相机模式观察 `node01/node02` 的三相机 marker（浅蓝半透明小球）与 `target` 是否在视野前方

---

## 2. 环境要求（很重要）

建议用 **Python 3.10** 运行以上脚本（不要用 base 的 Python 3.13）。

本机已验证可用的依赖组合：
- `mujoco==3.1.3`
- `numpy`
- `opencv-python`（采集脚本写 jpg/png 用；viewer 脚本不强依赖 opencv）

渲染模式：
- 有桌面/显示器：`MUJOCO_GL=glfw`（可开窗口）
- 无 GUI/服务器：`MUJOCO_GL=osmesa`（离屏渲染；用导出的图片验证）

---

## 3. 如何验证“相机节点已构建成功”

### 3.1 静态检查（MJCF）
打开 `mvp-demo/assets/mujoco_3cam_node_parallel.xml`，确认：
- 存在 `body name="node01"`
- `node01` 下存在 3 个相机：`node01_cam0/1/2`
- 3 个相机 `xyaxes` 一致（= 光轴平行）
- 三相机 `pos` 构成小圈（当前直径 0.9m）

这是“结构正确”的最低要求。

### 3.2 GUI 里查看（推荐，有界面就用）
在 `mvp-demo/` 目录运行：
```bash
MUJOCO_GL=glfw python3.10 scripts/mj_view_3cam_node.py
```
你应能看到：
- `node01/node02` 附近 3 个浅蓝色半透明小球（相机位置 marker）
- `target` 在一条直线上往返运动（默认端点：`node01` 与 `node02`）

如果你想调整轨迹（端点/位置/速度），示例：
```bash
MUJOCO_GL=glfw python3.10 scripts/mj_view_3cam_node.py --from_body node01 --to_body node02 --mid_y 6 --mid_z 2 --period_s 12
```

如果你想让目标在直线运动时做滚转（绕连线方向转），示例：
```bash
MUJOCO_GL=glfw python3.10 scripts/mj_view_3cam_node.py --roll_dps 90
```

### 3.3 无 GUI 时用离屏采集验证（最稳、可回归）
在 `mvp-demo/` 目录运行：
```bash
MUJOCO_GL=osmesa python3.10 scripts/mj_capture_3cam_node.py --seconds 1 --fps 5 --width 320 --height 240
```
成功标准（必须同时满足）：
- 生成一个新目录：`data/nodes/node01/scenes/mj_node01_YYYYmmdd_HHMMSS/`
- 目录内存在：
  - `cams/cam0/frames/*.jpg`、`cams/cam1/frames/*.jpg`、`cams/cam2/frames/*.jpg`
  - `cams/cam*/masks/*.png`
  - `calib/rig.json`
  - `capture_meta.json`、`frame_times.csv`

如果你想进一步验证“深度是否能导出”（可选）：
```bash
MUJOCO_GL=osmesa python3.10 scripts/mj_capture_3cam_node.py --seconds 1 --fps 5 --width 320 --height 240 --save_depth
```
期望多出：
- `cams/cam*/depth/*.npy`

### 3.4 数值验证（光轴平行 + baseline < 1m）
你可以用两种方式做数值验证：

**方式 A：直接验证 MJCF（推荐：不依赖采集输出）**
在 `mvp-demo/` 目录运行：
```bash
python3.10 scripts/mj_validate_3cam_node.py
```
期望：
- `result: True`
- `max_baseline_m <= 1.0`
- 三路 `ang_to_expected` 近似 0（默认期望 forward 为 node 的 `[0, 1, 0]`）

**方式 B：基于采集输出的 `rig.json` 验证（和后续 3D 链路更一致）**
离屏采集后，在 `mvp-demo/` 目录运行（会自动取最新 scene）：
```bash
python3.10 - <<'PY'
import json, numpy as np
from pathlib import Path
scene = sorted(Path("data/nodes/node01/scenes").glob("mj_node01_*"))[-1]
rig = json.loads((scene/"calib/rig.json").read_text())
cams = rig["cameras"]

for cid in ["cam0","cam1","cam2"]:
    T = np.array(cams[cid]["T_node_from_cam"], float)
    # CV 相机坐标：前向是 +Z；变到 node：fwd_node = R * [0,0,1]
    fwd = T[:3,:3] @ np.array([0,0,1.0])
    fwd = fwd/np.linalg.norm(fwd)
    print(cid, "pos", T[:3,3].round(3).tolist(), "fwd", fwd.round(3).tolist())

pos = {cid: np.array(cams[cid]["T_node_from_cam"], float)[:3,3] for cid in ["cam0","cam1","cam2"]}
import itertools
for a,b in itertools.combinations(pos,2):
    print(a,b,"baseline_m", float(np.linalg.norm(pos[a]-pos[b])))
print("scene_dir=", scene)
PY
```
期望：
- 三个 `fwd` 都接近 `[0, 1, 0]`（都朝 node 的 +Y，证明“光轴平行且方向正确”）
- 任意两相机 `baseline_m < 1.0`（满足“圈直径不超过 1m”的硬约束）

---

## 4. 已知坑/排错提示

**(1) segmentation 渲染在本环境不稳定**
- 现象：`Renderer.enable_segmentation_rendering()` 在 MuJoCo 3.1.3 + 当前环境下可能触发 `IndexError`。
- 处理：采集脚本的 mask 不依赖 segmentation，而是用 `mjCAT_DYNAMIC` 的 depth 阈值生成 silhouette（你会在 `capture_meta.json` 看到 `mask_strategy`）。

**(2) viewer 打不开窗口**
- 处理：
  - 确认在有桌面环境，或配置了 X11 forwarding
  - 使用 `MUJOCO_GL=glfw`
  - 无 GUI 就走离屏采集（`MUJOCO_GL=osmesa`）+ 看导出图片

---

## 5. 如何改相机节点布局（只改 MJCF 即可）

你现在的布局是：
- 三相机在 `x-z` 平面绕圈
- 看向 `+y`（圆面法线方向）

常改参数：
- **圈直径**：改 `mvp-demo/assets/mujoco_3cam_node_parallel.xml` 里三相机 `pos` 的数值（半径=直径/2）
- **看向方向**：改 3 个相机的 `xyaxes`（保持一致才是“平行光轴”）
- **改 node_id**：如果你把 body 改名（比如 `node02`），同时也要改相机名（`node02_cam0/1/2`），并在采集脚本里传 `--node_id node02` 或 `--camera_names ...`

---

## 6. 本阶段的“完成定义”（DoD）

当你满足以下三条，就可以认为“相机节点已构建成功，可进入下一阶段”：
1) GUI viewer 能看到 node01 的相机 marker（或至少离屏采集不报错）
2) 离屏采集产出三路 `frames/` + `masks/` + `calib/rig.json`
3) 数值验证：三路 `fwd≈[0,1,0]` 且 baselines < 1m
