# MuJoCo 双节点 6 路相机：RGB+标定“传出”到 3D ReID 流程（思路文档）

场景基础：`mvp-demo/assets/mujoco_humanoid_3cam_node_parallel.xml` 已包含两个节点：
- `node01`：`node01_cam0 / node01_cam1 / node01_cam2`
- `node02`：`node02_cam0 / node02_cam1 / node02_cam2`

目标：把 MuJoCo 中 6 路相机的 **RGB 图像 + 相机标定 + 时间戳** 可靠“传出”，供后续 **3D 重识别（ReID）** 使用。

约束（当前决定）：
- 先不新增脚本；仅给出后续工作路线与接口口径。
- 输出口径以“真实传感器”为目标：**RGB + 标定** 为主；GT mask/depth 仅作为可选评测。
- 传出方式希望同时支持：**落盘** + **实时流式**（后续再实现）。

---

## 1) 关键原则：把“仿真”当“相机系统”来对待

为了让后续 ReID/3D 管线可迁移到真实相机，建议把 MuJoCo 输出设计成“传感器数据集”：

1. **只依赖 RGB + 标定 + 时间戳**  
   - 下游的 mask（分割/检测）来自图像算法（YOLO/SAM2 等）。
   - 下游的 depth 来自立体匹配/学习深度（或先用 GT depth 做误差评估，但不要把 GT 作为默认依赖）。

2. **时间轴是第一等公民**  
   - 3D（深度/三角化/多视融合）对同步敏感。必须保证：同一个 `ts_us` 能定位到 6 路同一时刻的帧。
   - 对于 MuJoCo：最简单可控的是“固定 FPS 采样”，每帧 `t = i / fps`，再把 `ts_us = round(t * 1e6)` 作为文件名。

3. **标定文件要可直接被 CV 管线消费**  
   - 内参 `K`：由 `fovy + width/height` 推出 pinhole（fx=fy，cx=w/2，cy=h/2）。
   - 外参 `T_node_from_cam`：采用 CV 相机坐标（x 右、y 下、z 前），以便直接用 pinhole 反投影/投影。

4. **跨节点的坐标语义要清晰**  
   - 每个 node 的 `T_node_from_cam` 是在各自 node 坐标系下，便于“节点内 3D”。
   - 如果要做跨节点几何（可选），还需要导出 `T_world_from_node` 与 `T_world_from_cam`（或等价信息）用于拼到同一 world frame。

5. **多节点阵列的同步层级要分级管理**  
   - 仅做节点内 3D / 节点内 ReID：只需节点内三相机严格同步。
   - 需要跨节点 ReID / 轨迹关联：建议做跨节点粗同步（统一时间轴 + 时间窗匹配）。
   - 需要跨节点几何融合 / 同时刻 3D：才需要接近帧级的跨节点严格同步。

---

## 2) 推荐的数据契约（落盘）——最小可用、可扩展

沿用 repo 已存在的 node 目录结构（`data/nodes/<node_id>/scenes/<scene_id>/...`），并建议让 `node01` 与 `node02` 使用同一个 `scene_id`（便于对齐）。

每个节点一份目录：

```text
data/nodes/<node_id>/scenes/<scene_id>/
  capture_meta.json
  frame_times.csv
  calib/
    rig.json
  cams/
    cam0/
      frames/<ts_us>.jpg
    cam1/
      frames/<ts_us>.jpg
    cam2/
      frames/<ts_us>.jpg
```

建议新增一个“跨节点索引”（后续可加，便于一次 run 对齐两节点）：

```text
data/nodes/multinode_runs/<scene_id>.json
```

内容类似：

```json
{
  "scene_id": "mj_multinode_YYYYmmdd_HHMMSS",
  "nodes": {
    "node01": "data/nodes/node01/scenes/<scene_id>",
    "node02": "data/nodes/node02/scenes/<scene_id>"
  }
}
```

### 2.1 `frame_times.csv`（必须）

目标：提供一个“共享时间轴”，可用来按 `ts_us` 聚合 6 路帧。

```csv
ts_us,node_id,cam_id,filename
000000033333,node01,cam0,cams/cam0/frames/000000033333.jpg
...
000000033333,node02,cam2,cams/cam2/frames/000000033333.jpg
```

要求：
- 每个 `ts_us` 应当有 6 行（node01 x3 + node02 x3）。

### 2.2 `calib/rig.json`（必须）

建议沿用现有 3cam 节点的口径（参考 `scripts/mj_capture_3cam_node.py` 的导出结构）：

```json
{
  "node_id": "node01",
  "world_frame": "node",
  "cameras": {
    "cam0": {
      "image_size": [1280, 720],
      "K": [[fx,0,cx],[0,fy,cy],[0,0,1]],
      "distortion_model": "none",
      "dist": [],
      "T_node_from_cam": [[...4x4...]],
      "mjcf_camera_name": "node01_cam0",
      "fovy_deg": 60.0
    }
  }
}
```

为后续跨节点几何留扩展字段（可选）：
- `T_world_from_node`：4x4
- `T_world_from_cam`：4x4

---

## 3) “实时流式传出”的设计思路（后续实现）

因为你希望“落盘 + 实时”两者都要，建议把“数据契约”固定在上一节，然后把实时流式当成“同一数据的另一种输运层”。

### 3.1 推流载荷最小字段（推荐）

每条帧消息至少携带：
- `scene_id`
- `ts_us`
- `node_id`
- `cam_id`（cam0/cam1/cam2）
- `mjcf_camera_name`（用于排错）
- `width/height`
- `encoding`（例如 jpeg）
- `color`（RGB 或 BGR，必须写清楚）

推流时序建议：
1) 启动后先发一次 `calib`（包含两个 node 的 rig.json 或其路径/内容）。  
2) 按帧发送 `frame` 消息；同一个 `ts_us` 会有 6 条帧消息。

### 3.2 背压与丢帧策略（必须提前定）

ReID 下游往往有重计算（分割/跟踪/特征），订阅端跟不上时不能阻塞采集。建议策略：
- 默认允许丢帧（例如只保留最新的 N 帧，或按 cam 各自最新）。
- 保留“关键帧/触发窗口”时才要求全帧不丢（如果后续你要加门控/缓存）。

### 3.3 输运方案候选（按实现成本）

- ZeroMQ PUB/SUB：轻量、易集成、适合原型验证。
- ROS2：生态好（可视化/录包/时间戳），但依赖与集成成本更高。
- 共享内存 + 元数据队列：最高性能，但工程量大（同步/生命周期/兼容性）。

---

## 4) 从“RGB+标定”到“3D ReID”的后续步骤（节点内 + 跨节点）

这里给一个“从简单到可交付”的推进路径，优先保证每一步都能独立验收。

### 4.1 里程碑 A：6 路数据对齐（先做口径验证）

验收要点：
- 同一个 `ts_us` 下，6 路图片都存在。
- 两个节点的 `rig.json` 可加载，三相机光轴方向一致、baseline 合理（可用 `scripts/mj_validate_3cam_node.py` 对 node01/node02 分别验证）。
- 随机抽取 10 个 `ts_us`，肉眼确认 6 路画面时间一致（目标位置不会错位离谱）。

### 4.2 里程碑 B：2D 侧“目标实例”稳定（mask/tracklet）

节点内 3D/跨节点 ReID 的前提是“对象级别”的时序片段：

1) 每路相机跑检测/分割：
   - 选择：YOLO (bbox) / YOLO-seg / SAM2（mask 更好）。
2) 节点内做多视一致性关联（可先用简单规则）：
   - 先做“单相机 tracklet”（ByteTrack/BoT-SORT）。
   - 再把 3 路相机的 tracklet 在同一 `ts_us` 上做关联（基于外参的极线约束/重投影门限/外观相似度）。

最低可交付：先只做“单相机 tracklet + 外观 embedding”，跨节点直接做 2D ReID（作为 baseline），为 3D ReID 提供对照组。

### 4.3 里程碑 C：节点内 3D 表示（深度/点云/体素）

对你的布局（小 baseline + 光轴平行），建议路线：

- 主线：**深度 -> 反投影 -> 多相机点云融合**  
  - 深度来源：
    1) 双目立体：对 (cam0,cam1)、(cam0,cam2)、(cam1,cam2) 做 stereo。
    2) 学习深度：单目深度只能提供相对尺度，需额外尺度对齐；更推荐 stereo/MVS。
  - 反投影：用 `K` 与深度把像素变成 cam 坐标 3D 点。
  - 变换到 node：用 `T_node_from_cam` 把点云变到同一 node frame。
  - 融合/下采样：体素栅格下采样（稳定、便于后续特征）。

备注：repo 里已有 `scripts/recon_fuse_depth_points.py`（可用于“多相机 depth+mask 融合点云”的目标），但当前你决定“先不加脚本”，因此这里只把它当成“未来可复用的方向”。

### 4.4 里程碑 D：3D ReID 表征与检索（track-level）

推荐在“track 级别”生成 embedding（比逐帧更稳）：

- 外观 embedding：CLIP / ReID 网络（对 mask crop 做池化/时序聚合）。
- 几何 embedding：
  - 轻量 baseline：点云归一化后做径向直方图（尺度/位姿不敏感）。
  - 更强但重：FPFH / PointNet++ / DGCNN / transformer（需要训练与数据）。

跨节点匹配：
- 同一场景 MuJoCo 可在 world frame 有一致的几何参考，但真实部署中跨节点可能没有共享坐标系。  
  因此建议 embedding 尽量做 **object-centric 归一化**（中心化、尺度归一）减少对绝对坐标的依赖。

---

## 5) 风险点与排错清单（优先级从高到低）

1) **时间戳不一致**：`ts_us` 聚合后某些相机缺帧/错位 → 下游 3D 会直接崩。
2) **坐标系方向搞反**：尤其是相机 forward 与 `T_node_from_cam` 的定义 → 会导致投影/反投影错位。
3) **小 baseline 导致深度不稳**：目标距离稍远时 stereo 深度噪声大 → 需要更强的 mask/先验或调布局/调距离范围。
4) **编码色彩通道不一致**：RGB/BGR 混用 → embedding 质量会受影响；必须在契约里写清楚并保持一致。

---

## 6) 推荐的推进顺序（你下一步可以做什么）

在不新增脚本的前提下，建议把工作拆成“可验收的小步”：

1) 先用现有 `scripts/mj_capture_3cam_node.py` 分别对 node01 与 node02 落盘，确保两次采集参数一致（fps/seconds/分辨率），并手动用同一 `scene_id` 管理两节点输出（后续再自动化）。
2) 用 `scripts/mj_validate_3cam_node.py` 分别验证 node01 与 node02 的三相机几何（baseline/平行光轴/方向）。
3) 在下游先做 2D ReID baseline（只用 RGB + mask crop），验证跨节点检索的基本可行性。
4) 再引入“节点内 3D”（深度/点云融合），把 embedding 升级为 2D+3D 的 track-level 表征。
