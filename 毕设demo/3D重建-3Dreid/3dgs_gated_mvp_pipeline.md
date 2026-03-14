# 3DGS 路线（MVP）：目标检测门控 → 采集 → COLMAP → 3DGS → 渲染 Depth

你的需求是“**相机视窗检测到目标出现才启动重建**”。这里把“检测”作为**门控/触发器（gating）**：它只决定**何时开始采集/启动重建任务**，但**不建议用检测框裁剪后再做 3DGS 重建**（裁剪会破坏 SfM 特征与相机内参一致性，导致 COLMAP/3DGS 更容易跑不通）。

本文给出一条可先跑通的 MVP：**触发采集一段短序列 → 用 graphdeco 3DGS 跑 COLMAP+训练 → 为每帧渲染 depth_npy**，并约定好后续与分割/跟踪/检索的对齐方式。

---

## 0. MVP 范围与验收

**输入：** 摄像头实时画面（或视频流）。  
**输出（验收标准）：**

1. 触发后生成一个 `SCENE_DIR/input/*.jpg`（连续编号）。  
2. `python convert.py -s SCENE_DIR` 后得到 `SCENE_DIR/images/` 与 `SCENE_DIR/sparse/0/*`。  
3. 训练后渲染出 `MODEL_DIR/train/ours_<iter>/depth_npy/*.npy`，数量与 `SCENE_DIR/images/*` 对齐（同 stem 或可建立映射）。

---

## 1. 推荐目录结构（先固定，后面少踩坑）

建议在仓库根目录新建（你也可以放到别处，关键是结构一致）：

```text
data/
  scenes/
    <scene_id>/
      input/                 # 门控触发后保存的原始帧（供 convert.py）
      images/                # convert.py 输出的 undistorted images（后续一律用它做对齐）
      sparse/0/              # COLMAP 输出（cameras.bin/images.bin/points3D.bin）
      capture_meta.json      # 采集参数（fps、分辨率、触发帧、阈值等）
output/
  <scene_id>/                # 3DGS 模型输出（-m 指向这里）
```

`scene_id` 推荐用时间戳：`cam1_20260115_1507`。

---

## 2. 门控（检测）怎么做才“像门控”

门控的目标不是精确框，而是稳定触发：**少误触发、少抖动、触发后能留出足够视差/帧数给 COLMAP/3DGS。**

### 2.1 状态机（建议照抄这个逻辑）

参数（MVP 推荐值）：

- `conf_th=0.5`：检测置信度阈值
- `K_on=3~5`：连续检测到 K 帧才触发（去抖）
- `K_off=10~20`：连续丢失 K 帧才停止（滞回，避免频繁开关）
- `pre_seconds=2~5`：触发前环形缓冲（把触发前的帧也写入 input，COLMAP 更稳）
- `fps_save=2~5`：保存帧率（3DGS 不需要把 30fps 全存；MVP 存稀一点更快）
- `max_seconds=20~60`：单次采集上限（避免无限增长）

状态机：

```text
IDLE（只检测+缓冲）
  └─(检测满足 K_on)→ RECORD（落盘：先flush缓冲，再持续写帧）
RECORD（持续写帧）
  ├─(丢失满足 K_off)→ STOP
  └─(达到 max_seconds)→ STOP
STOP（关闭采集；触发一个“重建任务”）
```

关键点：

- **不要裁剪保存**：保存全帧到 `SCENE_DIR/input/`，后续 `convert.py` 需要全帧特征做位姿。
- **触发后人为配合**：触发后请做一个“绕目标的小扫视”（产生视差），否则 COLMAP 可能 sparse 很少或直接失败。

### 2.2 检测器选择（MVP 优先）

只为门控的话，优先选“好装、跑得动”的：

- YOLOv8/YOLOv5（Ultralytics）：门控只用 `class + conf`，甚至可以只检测每 N 帧。
- 如果你的“目标”不是 COCO 类别：用“颜色/ArUco/AprilTag/简单模板”做门控也行，门控不追求泛化。

---

## 3. 触发后：从 input/ 跑到 depth_npy（3DGS）

> 这一步建议直接复用 `rgbd_3d_reid_pipeline_routes.md` 的 `10.2 3DGS` 命令模板；下面只给“门控采集接上 3DGS”的关键对齐点与最小命令序列。

### 3.1 COLMAP + 去畸变（convert.py）

要求：`SCENE_DIR/input/%06d.jpg` 已存在。

```bash
python convert.py -s "$SCENE_DIR" --resize
```

验收：

- `SCENE_DIR/images/` 有 undistorted 图片
- `SCENE_DIR/sparse/0/` 有 `{cameras.bin,images.bin,points3D.bin}`

**对齐规则（非常重要）：**

- 后续所有分割/跟踪/特征提取都尽量在 `SCENE_DIR/images/` 上做，确保与 3DGS 渲染的 depth 对齐。
- `input/` 只作为“原始采集入口”，不要混用。

### 3.2 训练 3DGS

```bash
python train.py -s "$SCENE_DIR" -m "$MODEL_DIR" --eval
```

### 3.3 渲染 depth_npy

按 `rgbd_3d_reid_pipeline_routes.md` 的 10.2 小脚本（`render_depth_npy.py`）渲染：

```bash
python render_depth_npy.py -m "$MODEL_DIR" --skip_test
```

验收：

- `MODEL_DIR/train/ours_<iter>/depth_npy/*.npy` 存在
- 深度文件名与 `SCENE_DIR/images/*.jpg` 的 stem 能对上（或你能用 `image_list.txt` 建映射）

---

## 4. depth 有了以后怎么“先跑通 ReID（最简）”

为了先跑通，你可以暂时不做复杂 3D 描述子：

1. 在 `SCENE_DIR/images/` 上做 **检测/分割 + 跟踪** 得到 tracklet（建议有 mask）。
2. 对每帧：`RGB crop + depth crop (+ mask)` → 做归一化（见 `rgbd_3d_reid_pipeline_routes.md` 的 `normalize_depth`）→ 提特征。
3. track 内 `mean pooling` 得到 track embedding → FAISS 做检索。

如果你用 SAM2：建议同样输入 `SCENE_DIR/images/`，这样 mask 与 depth 更容易像素对齐。

---

## 5. 常见“跑不通”原因（优先排查）

- **视差不够**：触发后相机没移动/只平移很小 → COLMAP sparse 点少、位姿不稳。
- **动态占比太大**：大面积运动目标/背景 → COLMAP/3DGS 深度不可靠（MVP 尽量选静态场景/静态目标）。
- **对齐用错目录**：在 `input/` 上做检测/分割，但 depth 是按 `images/` 渲染 → 一定错位。
- **帧太多**：MVP 先用 50~200 帧跑通；太多会让 COLMAP/训练变慢、排错成本高。

