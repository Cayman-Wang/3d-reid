# 动态无人机 3D（每节点 3 相机）+ 跨节点 ReID：分步实现计划

目标：每个节点有 **3 台完成标定 + 时间同步的相机**。节点内做动态目标（例如飞行中的无人机）的 **object-centric 三维重建**；节点间做 **track-level 向量检索（ReID/实例检索）**。

说明：节点内“动态目标重建”不建议走 `COLMAP -> 3DGS`。SfM/3DGS 基本假设是（大体）静态一致的场景/观测，目标在高速运动时会破坏特征匹配与位姿估计，从而导致重建失败或深度不稳定。3DGS 可以保留为独立的“静态场景/深度来源”demo，但不作为动态无人机的节点内 3D 核心。

---

## 0) 相机节点几何：你选择的是“平行光轴 + 小圈三目”（tri-stereo）

你给的约束是：
- 3 相机尽量靠近、围成一个圆（圈直径 <= 1m）。
- 不是看向圆心，而是 **三相机都看向圆面法线方向（同一方向）**，即 **光轴平行**。

这等价于一个紧凑的 **tri-stereo rig**：三相机在同一“安装面”上分布（例如 `x-z` 平面），共同朝向该平面的法线（例如 `+y`）。

### 重要后果（直接决定节点内 3D 路线）

- **visual hull/体素雕刻不适合作为该 rig 的 MVP**：  
  你的相机“很近 + 光轴平行”，当目标距离稍远时，三个视点的“角差”会变得很小，hull 容易沿视线方向拉伸/塌陷，形状不稳（看起来像一条胖条/一团雾）。
- 节点内 MVP 更稳的路线是：**深度（stereo/learning stereo）-> 反投影 -> 多相机点云融合**。
  - 需要足够 baseline 和重叠视场；无人机距离远、像素占比小的话，stereo 深度会变脆弱（易飘/易断）。
  - 提升 2（重但研究味强）：**4D/动态 NeRF/动态 3DGS**。
    - 对 3 相机固定视角的数据要求高、训练重、调参深；不建议作为第一条可交付路线。

后续计划默认按 **平行光轴 + 深度->点云融合** 展开；visual hull 作为“更大角差布局”时的备选，不作为当前布局主线。

---

## 0.5) 定稿：严格传感器模式（不依赖 MuJoCo 的 GT mask/GT depth）

目标：即使你用 MuJoCo 生成图像，也按“真实相机”思路走全链路——**只消费 RGB 画面 + 标定 + 时间戳**，mask 与 depth 都由图像侧算法产出。

约定：
- `cams/*/masks/`：来自图像侧分割（YOLO-seg / SAM2 / MaskRCNN 等），不是仿真器导出。
- `cams/*/depth/`：来自图像侧深度（stereo/learning stereo），不是仿真器 depth buffer。
- 如需在仿真阶段做误差评测，可额外保存 `cams/*/masks_gt/`、`cams/*/depth_gt/`，但下游融合/跟踪/检索**默认只读 masks/depth**。

---

## 1) 数据约定（目录结构 + 命名 + 对齐口径）

沿用本 repo 的约定：一次“触发/录制窗口”就是一个 `scene`，但把它扩展成三相机版本：

```text
mvp-demo/data/nodes/<node_id>/scenes/<scene_id>/
  capture_meta.json                # 节点信息 + 相机列表 + fps + 同步方式 + 门控阈值等
  frame_times.csv                  # 时间戳索引表（见下方 schema）

  cams/
    cam0/
      frames/                      # 原始帧（BGR/RGB）
        <ts>.jpg
      masks/                       # （主线）图像侧分割结果：与 frames 像素对齐的二值 mask（silhouette）
        <ts>.png
      depth/                       # （主线）图像侧深度（stereo/learning stereo），单位：米
        <ts>.npy
      masks_gt/                    # （可选）仿真器导出的 GT mask，仅用于评测/排错
        <ts>.png
      depth_gt/                    # （可选）仿真器导出的 GT depth，仅用于评测/排错
        <ts>.npy
    cam1/...
    cam2/...

  calib/
    rig.json                       # cam0/1/2 的内参 + 外参（见下方 schema）

  recon/
    points_fused/                  # （主线）深度/立体匹配 -> 反投影 -> 多相机点云融合
      <ts>.npy                     # (N,3) float32，node 坐标系
      meta.json                    # 深度来源、下采样 voxel size、mask 处理等

    visual_hull/                   # （备选）更大角差布局时再用
      points/
        <ts>.npy                   # (N,3) float32，node 坐标系
      meta.json                    # voxel size、bounds 等参数

  tracks/
    tracklets.json                 # tracklet 元信息（2D + 3D）

  embeddings/
    tracks.npy
    tracks_meta.json
```

### 时间戳命名

使用稳定的整数时间戳 key，例如 **微秒（microseconds）**：
- `<ts>` 用 `int64` 表示：从 scene 开始计时的微秒，或 UNIX epoch 微秒（两者选一个，全程一致）。
- 示例：`000000123456.jpg`（123.456 ms）。

### `frame_times.csv`（最小 schema）

你需要一个“三相机共享的时间轴”：

```csv
ts_us,cam_id,filename
123456,cam0,cams/cam0/frames/000000123456.jpg
123456,cam1,cams/cam1/frames/000000123456.jpg
123456,cam2,cams/cam2/frames/000000123456.jpg
...
```

---

## 2) 标定（硬性前置条件）

你需要：
- 每台相机的内参：`K`（最好包含畸变参数）。
- 每台相机的外参：`T_node_from_cam`（或 `T_cam_from_node`，但必须全程统一坐标定义）。

### `calib/rig.json`（建议 schema）

```json
{
  "node_id": "node01",
  "world_frame": "node",
  "cameras": {
    "cam0": {
      "image_size": [1920, 1080],
      "K": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
      "distortion_model": "opencv",
      "dist": [k1, k2, p1, p2, k3],
      "T_node_from_cam": [[...4x4...]]
    },
    "cam1": { "...": "..." },
    "cam2": { "...": "..." }
  }
}
```

验收（smoke check）：
- 做一个小的重投影检查：把少量已知 3D 点（或标定板角点）投回每个相机，像素误差应明显小于目标尺寸。
- 畸变明显时：要么先把 frame/mask/深度 去畸变后再做反投影/融合，要么投影器必须正确处理畸变模型。

---

## 2.5) MuJoCo：如何构建“平行光轴三相机节点”（可直接跑通）

你现在要先用 MuJoCo 把“相机节点”跑通，建议把目标拆成两步：

1) **先把 node rig（3 个相机）定义对**：位置满足圈直径 <= 1m，三相机光轴平行（看向圆面法线方向）。  
2) **再把数据口径输出对**：三路同步 RGB + 时间戳 + `calib/rig.json`。  
   严格传感器模式下，`masks/` 与 `depth/` 应由图像侧算法产出；如需评测，可额外保存 `*_gt`（但下游不依赖）。

本 repo 已给你一个可直接用的最小实现：
- MJCF：`mvp-demo/assets/mujoco_3cam_node_parallel.xml`
  - 坐标系：`x` 右、`y` 前、`z` 上
  - 三相机在 `x-z` 平面围成一圈（直径 0.9m），共同朝向 `+y`（圆面法线）
- 采集脚本：`mvp-demo/scripts/mj_capture_3cam_node.py`
  - 输出到：`mvp-demo/data/nodes/<node_id>/scenes/<scene_id>/...`
  - 当前实现的 mask/（可选）depth 属于“仿真器渲染 GT”（depth buffer）。为了严格传感器模式，建议后续把它们改成输出到 `masks_gt/`、`depth_gt/`，并支持完全关闭。

实现细节提示（方便你后续自己改 MJCF）：
- 光轴“平行且看向 `+y`”在 MJCF 里靠 **相同的朝向**实现：例如 `xyaxes="1 0 0  0 0 1"`（MuJoCo 相机看向本地 `-z`，所以这会让相机看向世界 `+y`）。
- 脚本导出的 `T_node_from_cam` 使用 **CV 口径相机坐标**（x 右、y 下、z 前），方便你直接用 pinhole 反投影公式做点云。

推荐你先做一次 smoke run（从 `mvp-demo/` 目录运行）：
```bash
# 无 GUI/服务器环境推荐：
MUJOCO_GL=osmesa python scripts/mj_capture_3cam_node.py --seconds 3 --fps 30
```

你跑完后，至少应看到（严格传感器模式最低要求）：
- `.../cams/cam0/frames/*.jpg`、`.../cams/cam1/frames/*.jpg`、`.../cams/cam2/frames/*.jpg`
- `.../calib/rig.json`（脚本会导出 `K` + `T_node_from_cam`）
- `capture_meta.json`、`frame_times.csv`

说明：
- 当前脚本会额外生成 `masks/*.png`（由 GT depth 阈值化得到）。如果你要严格模拟真实世界，请在后续里程碑 B 里用图像侧分割结果覆盖到 `masks/`，并把仿真 GT 移到 `masks_gt/`（或直接不保存）。
- 如需 GT depth 评测，可临时用 `--save_depth` 生成 `depth/*.npy`，但建议后续同样迁移为 `depth_gt/`。

校验要点（别跳过，后面所有 3D 都靠它）：
- 三相机图像里目标应同时可见（强重叠）。
- `rig.json` 的外参要能“重投影对齐”（把 node 里的 3D 点投回 3 个相机，落在目标上）。

### MuJoCo：导入外部模型文件（mesh/URDF）

MuJoCo 的“导入”通常分两层：
1) **导入整套模型描述**：MJCF（MuJoCo XML）或 URDF（XML）。  
2) **在模型描述里引用外部资源**：mesh/纹理/高度图等（用于外观与/或碰撞）。

#### 1) 可导入的“模型描述”类型

- **MJCF**：`*.xml`（根节点 `<mujoco>`），MuJoCo 原生格式。
- **URDF**：`*.urdf` / `*.xml`（根节点 `<robot>`），MuJoCo 3.x 支持直接解析（功能覆盖不如 MJCF 完整）。

加载方式（Python，与你当前脚本一致）：
```python
import mujoco
model = mujoco.MjModel.from_xml_path("path/to/model.xml")   # MJCF
model = mujoco.MjModel.from_xml_path("path/to/robot.urdf")  # URDF
```

#### 2) 可导入的“外部资源”类型（常用）

- **三角网格（mesh）**：常用 `*.obj`、`*.stl`（如果你手里是 `fbx/gltf/dae` 等，一般先用 Blender/MeshLab 转成 obj/stl）。
- **纹理贴图（texture）**：常用 `*.png`、`*.jpg`。
- **高度图地形（heightfield）**：常用灰度 `*.png`（MuJoCo 会把像素转成高度场）。

#### 3) MJCF 里引用 mesh 的最小模板

建议做法：mesh 只做**可视化**，碰撞用 box/capsule 等原生几何（更稳、更快）。

```xml
<mujoco model="demo_with_mesh">
  <!-- 可选：集中管理资源目录（相对当前 xml 文件） -->
  <compiler meshdir="meshes" texturedir="textures"/>

  <asset>
    <!-- 1) 注册 mesh（name 可省略；省略时常用文件名当作 mesh 名） -->
    <mesh name="uav" file="uav.obj" scale="0.001 0.001 0.001"/>

    <!-- 2) 可选：纹理/材质 -->
    <texture name="uav_tex" type="2d" file="uav.png"/>
    <material name="uav_mat" texture="uav_tex"/>
  </asset>

  <worldbody>
    <body name="target" pos="0 6 2">
      <freejoint/>

      <!-- 仅外观（不参与碰撞） -->
      <geom type="mesh" mesh="uav" material="uav_mat"
            contype="0" conaffinity="0" group="1"/>

      <!-- 单独做一个简化碰撞体（示例） -->
      <geom type="box" size="0.35 0.18 0.10" rgba="1 0 0 0.2" group="3"/>
    </body>
  </worldbody>
</mujoco>
```

常见坑（导入失败/比例不对时优先排查）：
- **单位**：MuJoCo 以“米”为单位；很多 mesh 是“毫米/厘米”，需要 `scale`。
- **朝向**：mesh 自身坐标系可能不是你期望的 `x/y/z`；用 `<geom euler="...">` / `quat="..."` 调整。
- **路径**：`file="..."` 默认相对当前 MJCF；也可以用 `<compiler meshdir="...">` 统一管理。

## 3) 里程碑 A：三相机采集 + 时间同步

交付物：生成一个 `<scene_id>` 目录，并且每个 `ts` 都有 **三相机同步三元组**。

实现建议：
- 先用最朴素的方法跑通：3 个采集线程/进程各自抓帧，但最终写盘时按统一 `ts_us` 对齐成三元组。
- 同步目标：时间误差要远小于无人机每帧运动量；高速目标建议 < 5-10ms 量级。
- 门控（可选）：可以沿用你现有的“检测触发-录制窗口”的 scene 思路，但触发后要 **同时录制三路**，并且 `pre_seconds` 缓冲也要按三路一起 flush（复用 `mvp-demo/scripts/gated_capture_yolo.py` 的状态机思想即可）。

必须产出：
- `cams/cam*/frames/<ts>.jpg`
- `frame_times.csv` 每个 `<ts>` 有 3 行（cam0/cam1/cam2）。
- `capture_meta.json` 至少包含 fps、分辨率、曝光/快门/增益（能拿到就写，后面排错很有用）。
- `calib/rig.json`（内参 + 外参，且与 frames 分辨率一致）

可选产出（仅用于仿真阶段评测/排错，不作为下游依赖）：
- `cams/cam*/masks_gt/<ts>.png`
- `cams/cam*/depth_gt/<ts>.npy`

验收（smoke check）：
- 随机抽 10 个 `<ts>`，确认三路 frame 都存在，并且肉眼看起来是同一时刻。

---

## 4) 里程碑 B：每相机无人机 mask（silhouette）

即便你走“深度 -> 点云融合”，也**强烈建议保留 silhouette（二值 mask）**（ROI/去背景/融合门控/下游外观特征裁剪都靠它）。对无人机这种“小目标/高速/纹理弱”的场景，mask 质量几乎决定成败。

两条可落地路线：
1) 专用无人机分割模型（如果你有数据，这是最稳的）。
2) 检测（YOLO）-> 裁剪 ROI -> 分割（SAM2 或同类）-> 贴回全分辨率。

必须产出：
- `cams/cam*/masks/<ts>.png`（0/255），分辨率与对应 frame 完全一致。

验收（smoke check）：
- 连续抽 50 帧把 mask 边缘叠加在 RGB 上：抖动要小、背景泄漏要少（泄漏会把点云/重建污染成一大片“背景墙/胖团”）。
- 仿真阶段可选：与 `masks_gt/` 做 IoU/边界误差统计（只做评测，不作为下游依赖）。

---

## 5) 里程碑 C：节点内 3D（MVP：深度 -> 点云融合）

### 深度来源（严格传感器模式：stereo / learning stereo）

推荐 MVP：把 tri-stereo 拆成两对双目（`cam0-cam1`、`cam0-cam2`），用图像侧立体匹配得到深度，再进入“反投影 -> 融合”。

- 输入：三路同步 `frames/<ts>.jpg` + `calib/rig.json`（`K/dist/T_node_from_cam`）+ `masks/<ts>.png`（可选但强烈推荐做 ROI）。
- 输出：`cams/cam*/depth/<ts>.npy`（float32，单位米；建议用 **z-depth（相机坐标 z）** 口径，便于直接反投影）。
- 最小实现路径（先跑通再优化）：OpenCV `stereoRectify` + `StereoSGBM`（可选再加 WLS filter）。
- 升级路径：learning stereo（RAFT-Stereo/IGEV 等）在“小目标/弱纹理”时通常更稳，但工程依赖更重。
- 兜底方案：如果 dense depth 在无人机小目标上不稳定，先用多视角 bbox center/关键点做三角化，得到每帧 3D centroid/距离，先把节点内 3D 跟踪跑通，再回头增强成深度图/点云。

深度口径定稿（建议在 `capture_meta.json` / `recon/points_fused/meta.json` 里明确写死）：
- `depth_mode=z`：`depth[u,v]` 表示该像素反投影到相机坐标后的 `z`（前向）距离，单位米。
- `invalid=0`：无效/不可用像素写 0（融合时统一按 `depth>0` 过滤）；如你更偏好 `nan` 也行，但要全链路一致。
- ROI 策略：深度估计只在 `mask` 的 bbox ROI 内做（先跑通），再逐步扩到全图/更大 ROI（节省算力且更稳）。

验收（smoke check）：
- 先只看 ROI：把 `depth` 伪彩色叠回目标区域，深度随距离变化应连续（不要大片 NaN/0 或随机噪点）。
- 仿真阶段可用 `depth_gt/` 做评测：统计 mask 内的 `abs error / rel error`，快速定位口径问题（z-depth vs range、单位、外参方向）。

实现要点（OpenCV stereo 的“坑”提前约定，避免后面返工）：
- 立体匹配通常在 rectified 图上做；你有两种一致性选择：
  1) depth 也以 rectified 相机为坐标系输出：同时落盘一个 `calib/rig_rectified.json`（含 rectified 的 `K'` 与 `T_node_from_cam_rect`），后续反投影用这套标定。
  2) depth 输出回原始相机坐标系：需要把 rectified depth/点云 warp 回原图（实现更绕，但最终目录结构更“直观”）。
- 推荐 MVP 选项：先走 (1)（实现最直接、误差更可控），把“rectified 标定”当成一个新的虚拟相机模型进入后续点云融合。

### 核心思想
对每个时间 `ts`，三相机同时观测目标。对每一路相机：
1) 用 mask 选出前景像素（silhouette 内）。  
2) 用深度 `d(u,v)` + 内参 `K` 把像素反投影成相机坐标点云 `P_cam`。  
3) 用外参 `T_node_from_cam` 把点变换到 node 坐标系：`P_node = T_node_from_cam * P_cam`。  
4) 把 3 路点云 merge，并做一次轻量下采样/滤波，得到 `P_node(ts)`。

反投影公式（推荐使用 CV 口径：x 右、y 下、z 前）：
```text
x = (u - cx) / fx * d
y = (v - cy) / fy * d
z = d
```

必须产出：
- `cams/cam*/depth/<ts>.npy`：来自图像侧深度估计的深度图（单位米；与 frames 像素对齐）。
- `recon/points_fused/<ts>.npy`：融合点云 `(N,3)` float32（node 坐标系）。
- `recon/points_fused/meta.json`：记录深度来源（stereo/learning）、下采样 voxel size、mask 处理等参数，保证可复现。

验收（关键 smoke check）：
- 可视化一小段序列点云：应该看到一个跟着目标运动的“点云团”，相机切换不应导致整体跳变。
- 如果点云发散/断裂：优先排查外参方向（坐标系约定）、深度单位/near-far、mask 与 RGB/深度是否对齐。

工程上更稳的小技巧：
- 融合时允许“2/3 相机有效像素”即可（某一路偶发坏深度/坏 mask 不至于全空）。
- 对 mask 做轻微腐蚀（erode）减少背景泄漏；泄漏会把点云污染成一大片“背景墙”。

### 最小实现（脚本级 smoke check）

本 repo 提供了一个最小“深度 + mask -> 融合点云”的脚本：
- `mvp-demo/scripts/recon_fuse_depth_points.py`

用法示例（把 `<scene_dir>` 换成你刚采集出来的目录）：
```bash
python scripts/recon_fuse_depth_points.py \
  --scene_dir data/nodes/node01/scenes/<scene_id> \
  --voxel_size_m 0.02 \
  --write_ply
```

期望产物：
- `.../recon/points_fused/<ts>.npy`
- `.../recon/points_fused/meta.json`
- `.../recon/points_fused/points_index.csv`
- （可选）`.../recon/points_fused_ply/<ts>.ply`（可用 CloudCompare/MeshLab 打开）

如果你发现点云“沿视线方向被拉伸/尺度不对”，优先尝试切换深度口径：
```bash
python scripts/recon_fuse_depth_points.py --scene_dir <scene_dir> --depth_mode range
```

---

## 6) 里程碑 D：3D 跟踪（节点内 tracklets）

目标：在一个节点内，为每个无人机实例生成一条 tracklet（轨迹片段）。

最简 MVP：
- 每个 `<ts>` 点云计算 3D centroid + 尺寸统计（例如点云半径/体素数）。
- 用 3D Kalman filter 做时序平滑与关联（或者更简单：用 centroid 最近邻贪心关联先跑通）。

必须产出：
- `tracks/tracklets.json`（沿用你现有的 tracklets 思路，但扩展为 3D + 多视角路径）。

建议最小 schema：
```json
[
  {
    "track_id": "node01_scene123_track0001",
    "node_id": "node01",
    "scene_id": "scene123",
    "ts_us": [123456, 156789, "..."],
    "points_paths": ["recon/points_fused/000000123456.npy", "..."],
    "centroids_xyz": [[x,y,z], "..."],
    "per_view": {
      "cam0": {"frame_paths": ["cams/cam0/frames/<ts>.jpg", "..."], "mask_paths": ["cams/cam0/masks/<ts>.png", "..."]},
      "cam1": {"frame_paths": ["..."], "mask_paths": ["..."]},
      "cam2": {"frame_paths": ["..."], "mask_paths": ["..."]}
    }
  }
]
```

验收（smoke check）：
- centroid 轨迹要平滑且物理上合理（速度/加速度不应乱跳）。

---

## 7) 里程碑 E：节点级 track embedding（RGB + Geometry）

检索框架保持不变：仍然是 "track embedding + retrieval"。几何分支输入统一成 **node 坐标系点云**，
来源可以是 “深度/立体匹配 -> 反投影融合点云（主线）” 或 “visual hull 点云（备选）”。

### RGB 分支（复用现有思路）
- 每个 `ts`、每个相机：用 mask 做 crop（或 masked crop）-> CLIP（或先用 hist baseline）。
- 聚合方式：
  - 同一 `ts` 的多视角聚合：3 路特征取平均（或加权平均）。
  - 时间聚合：对采样的若干时间点取 mean pooling。

### Geometry 分支（点云描述子）
- 输入直接就是 node 坐标系点云：例如 `recon/points_fused/<ts>.npy`。
- 归一化（避免尺度/距离把 ID 泄漏掉）：
  - per-frame 或 per-track 做中心化 + unit-sphere 归一化（可参考 `mvp-demo/scripts/extract_track_embeddings.py` 的思路）。
- MVP 描述子：
  - radial histogram（半径直方图，最快、最易跑通）
  - 或 Open3D FPFH（在点云质量不错时更强）

必须产出：
- `embeddings/tracks.npy`
- `embeddings/tracks_meta.json`

验收（smoke check）：
- 同一 track 内的余弦相似度应明显高于不同 track（至少在节点内能拉开差距）。

---

## 8) 里程碑 F：跨节点检索（ReID）

跨节点检索可以直接沿用你现在的 top-K 检索范式：
- 每个节点把自己的 scene 产出 `embeddings/`（或者把 embedding + meta 上报到中心服务）。
- 中心脚本/服务加载多个 gallery scenes，做 cosine search（参考 `mvp-demo/scripts/search_track_embeddings.py` 的输出形式）。

强烈建议加一个“便宜但很有效”的约束：
- 只在 **相邻/下游节点** + **合理到达时间窗** 内做候选比较（由节点间距 + 无人机速度上限给出）。这会显著减少误匹配。

必须产出：
- 每条 query track 的 top-K 匹配结果（格式可复用现有脚本的 json/打印风格）。

验收（smoke check）：
- 受控实验（单架无人机按顺序经过节点）：正确节点应稳定出现在 top-K 里。

---

## 9) 建议实现顺序（一步一步交付）

1) 先把 `calib/rig.json` 跑通，并做一个最小重投影 sanity check。
2) 三相机采集 + 同步落盘 -> 同一 `ts` 有 3 路 frame + `frame_times.csv`。
3) 严格传感器模式落地：下游只依赖 `frames/ + rig.json + frame_times.csv`；GT（`*_gt/`）只用于评测/排错。
4) 三相机 UAV masks（哪怕一开始很粗，也要先把接口打通）。
5) 先做“稀疏几何兜底”：bbox center/关键点三角化 -> 3D centroid（把节点内 3D tracking 跑通）。
6) 再做 dense depth：stereo/learning stereo（建议先走 rectified 口径）-> 反投影 -> 多相机点云融合（逐 `ts`）-> `recon/points_fused/<ts>.npy`。
7) 3D tracking -> `tracks/tracklets.json`。
8) track embeddings（RGB + geometry）-> `embeddings/`。
9) 跨节点检索 +（可选）相邻节点/时间窗门控。

每一步都保留一个“10 秒 smoke check”（脚本或手工 checklist 均可）。这些中间产物不稳定时，不要急着上训练很重/很复杂的方法。

---

## 常见失败模式（排错清单）

- 同步偏差 20-50ms：多相机点云融合会出现撕裂/重影（像“拖影”）。
- 外参错误（rig.json 不对）：点云融合后整体“分叉/错位”，或重投影回每个相机时明显对不上 mask。
- 深度口径不一致：单位/near-far/无效像素处理不同，会导致点云尺度错/远处发散。
- mask 背景泄漏：点云被背景墙/地面污染（先腐蚀 mask；再提升分割质量）。
