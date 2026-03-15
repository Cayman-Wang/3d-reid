# 任意物体实例 ReID（普通相机视频）+ 3DGS/VGGT 生成 RGBD：推荐路线与最小可跑通方案

目标：用普通相机视频流，通过 `3DGS` 或 `VGGT(单目深度/几何)` 生成每帧对齐的 **RGB + Depth**，再接一个 **实例重识别/实例检索（instance ReID）** 模块，形成可工作的 pipeline。

你现在最需要的是：**先跑通（MVP）**，让系统能在同一段/不同段视频里，把“同一物体实例”检索出来；之后再逐步引入更强的 3D 表达与跨场景鲁棒性。

## 0. 总览：从视频到实例 embedding 的完整 pipeline

```text
Video frames (RGB)
   │
   ├─(A) 3DGS / VGGT → per-frame Depth (对齐到RGB)
   │
   ├─(B) Detector / Segmenter → instance mask/box
   │
   ├─(C) Tracker (ByteTrack/OCSort/...) → tracklet
   │
   ├─(D) Crop+Align: (RGB crop, Depth crop, mask)
   │
   └─(E) ReID Encoder (推荐三条路线之一)
          → frame embedding
          → track embedding (mean/attention pooling)
          → retrieval / matching
```

**关键建议（影响成败）：**
- 尽量用 **mask** 而不是纯 bbox（几何/深度对背景很敏感）。
- 深度不一定是绝对尺度；MVP 阶段建议采用 **尺度不敏感** 的深度归一化（见后文）。
- ReID 训练最简单的数据构造：把 tracker 产出的 **tracklet 当作伪ID**（Tracklet-as-ID）。

## 1) 推荐路线 A：双流 Late-Fusion（最稳的 MVP 基线）

**一句话：** 用成熟的 2D ReID 框架（FastReID/torchreid 风格），把 `Depth` 当成第二模态做一个并行分支，最后融合即可。

### A.1 结构（建议从简单开始）
- RGB encoder：`ResNet50`（或后续替换 `ViT/TransReID` 风格）
- Depth encoder：同结构 `ResNet50`（输入是深度的 3 通道版本）
- Fusion：`concat([f_rgb, f_d]) → FC → BNNeck → embedding`
- 训练损失：
  - `ID softmax`：把每条 tracklet 当一个类别（伪标签）
  - `Triplet (batch-hard)`：让同一 tracklet 更近、不同 tracklet 更远

### A.2 为什么它适合“任意物体实例 + 生成深度”
- **不依赖类别**：不需要“行人/车辆”先验；只要 tracker 能给稳定 tracklet，就能训练。
- **对深度质量更宽容**：深度只作为辅助模态，哪怕是相对深度/有噪声，也能先提升一定的跨视角一致性。
- **可直接复用 ReID 工具链**：采样策略、Triplet、BNNeck、测试度量都很成熟。

### A.3 Depth 输入怎么做最省事
- 最省事：把 `Depth (H×W)` 复制成 `3×H×W` 当作“伪RGB”喂给 Depth encoder。
- 更稳一点：对 depth 做 `clamp + 归一化`（例如逆深度/分位数归一化），尽量去掉尺度差异。
- 如果你有 mask：在裁剪前就把背景深度置 0 或 NaN（并在归一化时忽略无效像素）。

```python
import numpy as np

def normalize_depth(
    depth: np.ndarray,
    valid_mask: np.ndarray | None = None,
    clip_percentile: tuple[float, float] = (1.0, 99.0),
    eps: float = 1e-6,
) -> np.ndarray:
    """对单张 depth 做尺度不敏感的归一化。

    适合：单目/3DGS 渲染深度（尺度可能漂移），以及实例裁剪后的深度。
    输出：float32，约在 [0, 1]。
    """
    depth = depth.astype(np.float32)
    if valid_mask is None:
        valid_mask = np.isfinite(depth) & (depth > 0)
    if valid_mask.sum() < 10:
        return np.zeros_like(depth, dtype=np.float32)

    v = depth[valid_mask]
    lo, hi = np.percentile(v, clip_percentile)
    v = np.clip(depth, lo, hi)
    v = (v - lo) / (hi - lo + eps)
    v[~valid_mask] = 0.0
    return v.astype(np.float32)

def depth_to_3ch(depth01: np.ndarray) -> np.ndarray:
    """把归一化后的单通道深度变成 3 通道，直接喂给 CNN backbone。"""
    return np.repeat(depth01[None, ...], 3, axis=0).astype(np.float32)

# 示例：
# depth01 = normalize_depth(depth_crop, valid_mask=mask_crop)
# depth3 = depth_to_3ch(depth01)  # shape: (3, H, W)
```

## 2) 推荐路线 B：多帧 Set/Track 聚合（对视频更友好）

**一句话：** 单帧特征会抖（遮挡、视角、运动模糊），把一个 tracklet 里抽 K 帧当作“多视角集合”，做聚合得到更稳定的实例表征。

### B.1 结构选项（从简单到强）
- `Mean/Max pooling`：每帧编码 → 直接平均/最大池化（最简单，MVP 首选）
- `Attention pooling`：学习每帧权重（对遮挡/糊帧更稳）
- `Set Transformer / Temporal Transformer`：对 K 帧 token 做 self-attention 聚合（更强但更重）

### B.2 训练方式
- 仍然可以用 Tracklet-as-ID：
  - 一个样本 = 同一 tracklet 的 K 帧
  - label = tracklet id
- 损失同路线 A：`ID softmax + Triplet`（只是把“输入单位”从单帧变成 K 帧集合）

### B.3 为什么它适合你的场景
- 普通相机视频对“任意物体”通常更依赖多视角信息；
- 3DGS/VGGT 的深度在某些帧可能不稳定，多帧聚合能显著平滑噪声；
- track 级 embedding 也更符合你后续做跨时间/跨片段匹配的需求。

```python
import numpy as np
from dataclasses import dataclass

@dataclass
class Tracklet:
    track_id: int
    frame_indices: list[int]
    boxes_xyxy: list[tuple[int, int, int, int]]  # per frame

def sample_k_frames(frame_indices: list[int], k: int) -> list[int]:
    """均匀采样 K 帧（也可以改成随机采样、或优先清晰帧）。"""
    if len(frame_indices) <= k:
        return frame_indices
    xs = np.linspace(0, len(frame_indices) - 1, num=k)
    xs = np.round(xs).astype(int)
    return [frame_indices[i] for i in xs]

def aggregate_embeddings(frame_embs: np.ndarray, mode: str = "mean") -> np.ndarray:
    """frame_embs: (K, D) → (D,)"""
    if mode == "mean":
        return frame_embs.mean(axis=0)
    if mode == "max":
        return frame_embs.max(axis=0)
    raise ValueError(f"unknown mode: {mode}")
```

## 3) 推荐路线 C：Depth→点云/局部几何 + 2D 特征融合（更“3D”，但仍可控）

**一句话：** 把深度反投影成点云（在实例 mask 内），用 3D encoder 提几何描述子，再和 RGB 外观特征融合。

### C.1 为什么要做点云分支
- 对“任意物体实例”来说，外观（纹理/颜色）在光照、反射、拍摄条件变化时会不稳定；几何形状往往更稳。
- 如果你使用 3DGS，往往还可以拿到相机位姿（或至少相对一致的深度），这使得几何信息更可用。

### C.2 MVP 版怎么做（建议先不追求很复杂）
- 单帧点云：从 depth + intrinsics 反投影，取 mask 内点 → 采样 N 个点 → `center + scale normalize`。
- Track 级点云（可选更强）：把同一 tracklet 的多个视角点云合并（如果有相机外参可变换到同一坐标系）。
- 3D encoder 选择：
  - **最易用**：PointNet++ / DGCNN（实现多）
  - **更强**：Point Transformer 系列（实现相对少但效果好）
  - **想少训练**：用现成的 3D 预训练 encoder（例如 OpenShape/ULIP 思路），先直接取特征做检索，再慢慢 finetune
- 融合：简单 `concat([f_rgb, f_3d])` 就能跑；后续再尝试 cross-attention。

### C.3 你需要注意的坑
- 单目深度/伪深度尺度不准：点云做 **中心化 + 单位球归一化**，避免把尺度当身份特征。
- mask 的边界抖动会影响点云：可以做形态学腐蚀、或只取高置信区域。
- 深度空洞：采样前做简单的 hole filling（或在采样时跳过无效点）。

```python
import numpy as np

def depth_to_point_cloud(
    depth: np.ndarray,
    K: np.ndarray,
    mask: np.ndarray | None = None,
    eps: float = 1e-6,
) -> np.ndarray:
    """把单帧 depth 反投影为点云。
    depth: (H, W) float
    K: (3, 3) intrinsics
    mask: (H, W) bool，True 表示保留（通常用实例 mask）
    return: (N, 3) float32
    """
    H, W = depth.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    ys, xs = np.indices((H, W))
    zs = depth.astype(np.float32)
    valid = np.isfinite(zs) & (zs > 0)
    if mask is not None:
        valid &= mask.astype(bool)

    xs = xs[valid].astype(np.float32)
    ys = ys[valid].astype(np.float32)
    zs = zs[valid]

    X = (xs - cx) / (fx + eps) * zs
    Y = (ys - cy) / (fy + eps) * zs
    pts = np.stack([X, Y, zs], axis=1)
    return pts.astype(np.float32)

def normalize_point_cloud_unit_sphere(pts: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """中心化 + 缩放到单位球；适合尺度不确定的深度。"""
    if pts.shape[0] == 0:
        return pts.astype(np.float32)
    c = pts.mean(axis=0, keepdims=True)
    pts0 = pts - c
    r = np.linalg.norm(pts0, axis=1).max()
    return (pts0 / (r + eps)).astype(np.float32)

# 示例：
# pts = depth_to_point_cloud(depth_crop, K, mask=mask_crop)
# pts = normalize_point_cloud_unit_sphere(pts)
```

## 4) 我建议你怎么“最小代价跑通”

### 4.1 MVP 阶段（建议顺序）
1. **检测/分割 + 跟踪**：先让 tracklet 质量过关（这是 ReID 训练数据的来源）。
2. **3DGS/VGGT 出深度**：确保每帧 depth 和 RGB 对齐（同分辨率/同裁剪变换）。
3. **路线 A 单帧双流**：先让训练跑起来，能输出 per-frame embedding。
4. **track 级 pooling**：把同一 track 平均池化做检索；先在同一视频内做“重现检索”验证。

### 4.2 稍微提升稳定性（推荐）
- 上路线 B：训练输入从“单帧”升级为“同一 track 抽 K 帧”；
- 用 mask 抑制背景；
- 深度用分位数归一化（本文件的 `normalize_depth`）。

### 4.3 想更像“3D ReID”（下一步）
- 上路线 C：加点云/几何分支；先 `concat` 融合即可。
- 如果你有相机外参：尝试 track 内多视角点云对齐后再提特征（几何更稳定）。

## 5) 常见失败点与排查清单（很重要）

- **Depth 对齐错误**：RGB crop 和 depth crop 不是同一套几何变换（resize/letterbox/裁剪偏移）。
- **背景主导**：用 bbox 时，背景深度变化会“比物体本身更显著”，导致 embedding 学到场景而不是实例。
- **尺度泄漏**：单目深度不同视频的尺度不一致；不要让网络把“绝对深度值”当 ID 线索。
- **Tracker 伪ID噪声太大**：track ID 频繁断裂/换 ID，会让训练监督崩掉；先调 tracker。
- **跨场景泛化差**：MVP 用 tracklet 伪ID 通常只保证“同视频内有效”，跨视频要靠更强正则/自监督/更干净数据。

## 6) 文献/关键词检索建议（围绕你这条路线）

- 检索方向 1（RGB-D ReID / Multi-modal ReID）：
  - `RGB-D person re-identification`、`BIWI RGBD-ID reid`
  - `visible-infrared re-identification two-stream BNNeck`（把 Depth 当第二模态复用思路）
- 检索方向 2（Instance retrieval / object re-identification / tracking-reid）：
  - `tracklet re-identification`、`object re-identification embedding`、`tracking by re-identification`
- 检索方向 3（Depth/Point cloud encoder for recognition）：
  - `point cloud retrieval transformer`、`OpenShape`、`ULIP`、`point transformer`、`Minkowski sparse conv`
- 检索方向 4（3DGS depth rendering + downstream）：
  - `3D Gaussian Splatting depth rendering`、`gaussian splatting downstream recognition`

## 7) 跨不同视频/场景：推荐模块清单（3DGS/VGGT）

跨视频/跨场景的关键变化：**不再能把 tracklet 当真 ID**；你需要更强的“实例不变性”表示（外观 + 几何）与跨视频的无监督/弱监督对齐。

### 7.1 输入与几何（Depth / Pose / Intrinsics）
- 相机内参 `K`：优先从元数据/标定获取；没有就用 SfM/SLAM 估计或近似（后续可再校正）。
- Depth/几何生成二选一：
  - **3DGS 路线（重但一致性好）**：`COLMAP/SfM → 3DGS 训练 → 渲染每帧 depth`（可加法向/不确定度）。
  - **VGGT/单目路线（轻且实时友好）**：`Video depth` 输出 per-frame depth；需要的话再配 `SLAM/SfM` 给位姿。

### 7.2 实例发现（检测/分割）
- 类别未知/任意物体：`GroundingDINO(or OWL-ViT) + SAM2` 得到 mask。
- 类别已知或追求速度：`YOLOv8/YOLOv9 + (SAM2 可选精修)`。

### 7.3 跟踪（形成 tracklet）
- bbox 跟踪：`ByteTrack / BoT-SORT / OC-SORT`。
- mask 跟踪：`SAM2 video / XMem / DeAOT`（更适合形状差异大、遮挡多的实例）。

### 7.4 RGB-D 对齐与预处理（强烈建议做成独立模块）
- 记录每帧的 resize/letterbox/crop 变换，确保 RGB/mask/depth 同步。
- 深度处理：`clamp + 分位数归一化`；可输出额外通道：`inv-depth`、`valid mask`。
- 由 depth + mask 生成 **对象点云**：反投影→采样→单位球归一化（避免尺度泄漏）。

### 7.5 表征编码器（建议直接用 foundation model 起步）
- RGB 外观：`DINOv2 / CLIP(SigLIP)` 直接提特征（零训练也能先跑检索）。
- 深度/几何（二选一或都用）：
  - Depth-as-image：用与 RGB 同类的 ViT/CNN（可共享或独立权重）。
  - Point cloud：`PointNet++/DGCNN/Point Transformer`；想省训练可优先试 `OpenShape/ULIP/Uni3D` 类预训练。
- 融合：MVP 用 `L2-normalize 后 concat/加权和`；后续再上 cross-attention。

### 7.6 跨视频训练/对齐（从易到难）
- **零训练 baseline**：foundation embedding + track pooling + FAISS 检索。
- **弱监督/自举**：
  - 正样本：同一 track 多帧、多视角（3DGS 可渲染 novel view 作为额外正样本）。
  - 伪标签：跨视频 mutual-NN 或 DBSCAN 聚类 → 迭代训练（self-training）。
  - 约束/过滤：几何一致性（点云 Chamfer/ICP 残差）、mask 质量、时序一致性。

### 7.7 检索与后处理
- 向量索引：`FAISS`（track 级 embedding 建库）。
- Re-ranking：`k-reciprocal re-ranking`（ReID 经典做法，跨视频通常也有效）。

### 7.8 两套可落地组合（推荐）
- **组合 1（更学术/一致性强）**：`3DGS(depth+pose) + SAM2 + BoT-SORT + (RGB encoder + depth/point encoder) + concat fusion + FAISS + re-ranking`。
- **组合 2（更工程/轻量）**：`VGGT/VideoDepth + SAM2 video + (CLIP/DINOv2) + 深度归一化 + FAISS`（先零训练检索，再加自举训练）。

## 8) 跑通跨视角 3D ReID：模块选择建议与评价标准（先验证流程）

你当前目标是 **“先跑通、能稳定产出结果”**，所以每个模块都建议先选 **成熟、默认可用** 的方案；评价标准也以 **可视化 + 统计验收** 为主，必要时做一个很小的人工标注集算 ReID 指标。

### 8.1 一条最省事的 MVP 组合（推荐先用它跑通）
- Depth：`VGGT/VideoDepth`（逐帧推理得到 depth；先不追求绝对尺度）
- 实例 mask：`SAM2`（第一帧人工框/点一次，后续 video propagation）
- 跟踪：如果用 `SAM2 video`，可以把同一 mask 序列直接当 tracklet；否则 `YOLO + ByteTrack/BoT-SORT`
- 几何：`Depth + mask → 对象点云`（反投影 + 采样 + 单位球归一化）
- 表征：RGB 用 `DINOv2/CLIP`；3D 先用 `Open3D FPFH/ESF` 这类 **零训练描述子**（先验证流程）
- 融合：`L2norm(RGB_emb) ⊕ L2norm(3D_emb)`（concat）
- 聚合：tracklet 内 `mean pooling`
- 检索：`FAISS`（cosine/inner-product）+（可选）`k-reciprocal re-ranking`

### 8.2 每个模块的“验收标准/评价”

| 模块 | 产物 | 首选/备选 | 有 GT 时的指标 | 无 GT（流程验收）建议 |
|---|---|---|---|---|
| 视频解码 | RGB 帧序列 | OpenCV/FFmpeg | 掉帧率、时间戳误差 | 随机抽帧可视化，确认无错位/花屏 |
| 相机参数 | `K`（可选 pose） | 先近似 `K`；后续用 SfM/SLAM/3DGS | COLMAP 重投影误差 | 点云不应“严重拉伸”；同一物体形状在不同帧大致一致 |
| Depth 生成 | per-frame depth | VGGT/VideoDepth；后续 3DGS | AbsRel/RMSE/δ1 | (1) mask 内有效深度占比 (2) 边缘对齐（RGB 边缘 vs depth 梯度）(3) 时序抖动（深度分布随时间不应剧烈跳） |
| 实例分割/检测 | mask/box | SAM2；备选 YOLO-seg | mAP/IoU | mask 面积/形状随时间平滑；边界不过多包含背景；失败帧比例可统计 |
| 跟踪/关联 | tracklet | SAM2 video 或 ByteTrack/BoT-SORT | HOTA/IDF1/MOTA | track 平均长度、断裂率（同一物体被切成多个 track 的比例）；可人工抽样检查 ID switch |
| 对齐与裁剪 | RGB+depth+mask crop | 统一几何变换 | — | 叠加可视化（RGB+mask、depth colormap）；depth crop 与 mask 匹配、无明显偏移 |
| 点云生成 | object point cloud | depth 反投影 + 采样 | Chamfer/ICP error（若有 GT） | 点数 N 达标（如 1k/2k）；离群点比例低；同一 track 内 ICP 残差/Chamfer 不应爆炸 |
| 表征编码 | embedding | RGB: DINOv2/CLIP；3D: FPFH/ESF 或预训练点云模型 | ReID: mAP/CMC；检索: Recall@K | (1) 同一 track 的相似度显著高于不同 track (2) embedding 分布不塌缩（方差/均值检查） |
| 融合与聚合 | track embedding | concat + mean pooling | mAP/Recall@K 提升 | track pooling 后相似度更稳定（同一 track 方差下降）；融合后能看到“少量但稳定”的收益即可 |
| 检索与后处理 | top-K 列表 | FAISS + re-ranking | Recall@K / mAP | 延迟（ms/query）、top-K 可视化（query 与返回结果是否语义一致） |

### 8.3 最小 end-to-end 验证实验（强烈建议做一个小标注集）
- 录两段视频（或两相机）：同一批物体，视角/距离/光照有差异。
- 选 20–50 个“实例对”（同一物体在两个视频中的 tracklet），手工建立对应关系（很快）。
- 设定：`video A` 的 tracklets 做 query，`video B` 的 tracklets 做 gallery。
- 计算：`Recall@1/5/10` + `mAP`（够用）；并保存失败案例（mask 漏、depth 抖、track 断裂）。

## 9) 你需要拉取/安装哪些代码（按“先跑通”为目标的最小集合）

下面按模块给出 **最小必需** 和 **可选增强**。你可以先只拉最小集合，把数据流（RGB→depth→mask/track→embedding→检索）跑通。

### 9.1 Depth（两条路线二选一）
- **路线 A（更轻，先跑通推荐）**：任选一个“单目/视频深度”实现（你提到的 `VGGT` 也在这里）
  - 你需要：`depth inference` 代码 + 模型权重下载脚本
  - 备选（如果你暂时不确定 VGGT 用哪个实现）：用 `Depth-Anything-V2` 这类逐帧深度先顶上
- **路线 B（3DGS，更重但几何一致性好）**：
  - `COLMAP`（SfM/相机位姿/稀疏重建）
  - `3D Gaussian Splatting` 训练与渲染代码（渲染每帧 depth；可选渲染法向/不确定度）

### 9.2 实例分割/跟踪（强烈推荐优先用它避免 tracker 调参地狱）
- `SAM2`（视频分割/传播）：一旦 mask 序列稳定，你就天然有 tracklet。
- 可选自动发现：`GroundingDINO` / `OWL-ViT`（给 SAM2 提供初始化框/点，减少人工交互）。

### 9.3 表征（先零训练跑通）
- RGB embedding：`DINOv2` 或 `CLIP/SigLIP(open_clip)`
- 3D/几何 embedding（先不训练）：`Open3D`（内置 `FPFH/ESF` 等几何描述子，够你验证“几何分支数据流”）
- 可选更强（后续再上）：预训练点云模型（Point Transformer/Uni3D/OpenShape/ULIP 思路）

### 9.4 检索与评测
- 向量检索：`FAISS`（或先用 numpy 暴力余弦也行，数据少时够用）
- 评测脚本：`mAP + Recall@K`（自己写几十行即可，不一定要拉大 repo）
- 可选：`k-reciprocal re-ranking` 实现（提升跨视频检索稳定性）

### 9.5 工程依赖（通常直接 pip/conda 安装即可）
- `torch`/`torchvision`（GPU 建议）
- `opencv-python` / `ffmpeg`（视频读写）
- `numpy`、`scipy`、`tqdm`、`pyyaml`
- `open3d`（几何/点云）
- `faiss-cpu` 或 `faiss-gpu`（Windows 上更建议 conda 装）

如果你告诉我你最终想走 **VGGT** 还是 **3DGS**（以及你是否能跑 COLMAP），我可以把这一节进一步收敛成“明确的 repo + 目录结构 + 最小命令序列”。

## 10) 具体命令（Linux）：Depth-Anything-V2 路线 vs 3DGS 路线

下面给出两条“从视频生成深度（RGBD 中的 D）”的可执行命令模板，目标是**先跑通流程**。默认你在 Linux（Ubuntu 20.04/22.04）+ Conda 环境运行。

### 10.1 Depth-Anything-V2：视频 → depth（相对深度）

```bash
# 0) 基础依赖（有 sudo 用 apt；无 sudo 可用 conda-forge）
sudo apt-get update
sudo apt-get install -y ffmpeg git wget
# 无 sudo：conda install -c conda-forge ffmpeg git wget -y

# 1) 创建环境（示例：Python 3.10）
conda create -n da2 python=3.10 -y
conda activate da2

# 2) 安装 PyTorch（按你的 CUDA 版本选择；示例 cu121；没有 GPU 就装 cpu）
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 3) 拉取 Depth-Anything-V2 并安装依赖
git clone https://github.com/DepthAnything/Depth-Anything-V2.git
cd Depth-Anything-V2
pip install -r requirements.txt

# 4) 下载权重（vitl 更稳；vits 更快）
mkdir -p checkpoints
wget -O checkpoints/depth_anything_v2_vitl.pth \
  "https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true"

# 5A) 直接对视频跑（仓库自带脚本；输出通常是可视化/灰度图）
python run_video.py \
  --encoder vitl \
  --video-path /path/to/video.mp4 \
  --outdir /path/to/da2_out \
  --pred-only --grayscale

# 5B) 或者先抽帧再对图片目录跑（更方便和后续 mask/track 对齐）
mkdir -p /path/to/frames
ffmpeg -i /path/to/video.mp4 -vf fps=5 -q:v 2 /path/to/frames/%06d.jpg
python run.py \
  --encoder vitl \
  --img-path /path/to/frames \
  --outdir /path/to/da2_out \
  --pred-only --grayscale
```

### 10.2 3DGS：视频 → COLMAP → 3DGS 训练 → 渲染 depth

```bash
# 0) 基础依赖（convert.py 会调用 COLMAP；可选 ImageMagick 做多尺度 resize）
sudo apt-get update
sudo apt-get install -y ffmpeg git colmap imagemagick
# 无 sudo（或 apt 没有 colmap）：conda install -c conda-forge ffmpeg colmap imagemagick -y

# 1) 拉取 3DGS 官方实现（含 submodules）并创建环境
git clone https://github.com/graphdeco-inria/gaussian-splatting.git --recursive
cd gaussian-splatting
conda env create --file environment.yml
conda activate gaussian_splatting

# 2) 从视频抽帧，放到 <SCENE>/input/（3DGS 的 convert.py 约定）
SCENE=/abs/path/to/my_scene
mkdir -p "$SCENE/input"
ffmpeg -i /abs/path/to/video.mp4 -vf fps=2 -q:v 2 "$SCENE/input/%06d.jpg"

# 3) 运行 COLMAP + 去畸变/整理数据集结构（可加 --resize 生成 1/2,1/4,1/8 分辨率）
python convert.py -s "$SCENE" --resize
# 处理完成后，期望结构类似：
# $SCENE/images/...
# $SCENE/sparse/0/{cameras.bin,images.bin,points3D.bin}

# 4) 训练 3DGS（-s 指向 COLMAP 数据集；-m 指定输出目录）
MODEL_DIR=output/my_scene
python train.py -s "$SCENE" -m "$MODEL_DIR" --eval

# 5) 渲染 RGB（仓库自带）
python render.py -m "$MODEL_DIR" --skip_test

# 6) 渲染 depth：生成一个小脚本把 gaussian_renderer 的 depth 输出存成 .npy
cat > render_depth_npy.py <<'PY'
import os
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
from argparse import ArgumentParser

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import render, GaussianModel
from scene import Scene
from utils.general_utils import safe_state

try:
    from diff_gaussian_rasterization import SparseGaussianAdam  # noqa: F401
    SPARSE_ADAM_AVAILABLE = True
except Exception:
    SPARSE_ADAM_AVAILABLE = False

def render_depth_set(model_path, name, iteration, views, gaussians, pipeline, background, train_test_exp, separate_sh):
    depth_dir = os.path.join(model_path, name, f"ours_{iteration}", "depth_npy")
    os.makedirs(depth_dir, exist_ok=True)
    for idx, view in enumerate(tqdm(views, desc=f"Depth rendering ({name})")):
        out = render(view, gaussians, pipeline, background, use_trained_exp=train_test_exp, separate_sh=separate_sh)
        depth = out["depth"].detach().cpu().numpy().astype(np.float32)
        image_name = (
            getattr(view, "image_name", None)
            or getattr(view, "image_path", None)
            or getattr(view, "name", None)
        )
        stem = Path(str(image_name)).stem if image_name else f"{idx:05d}"
        np.save(os.path.join(depth_dir, f"{stem}.npy"), depth)

def main():
    parser = ArgumentParser(description="Render depth maps (.npy) for 3DGS cameras")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)

    safe_state(args.quiet)

    dataset = model.extract(args)
    pipe = pipeline.extract(args)

    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not args.skip_train:
            render_depth_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipe, background, dataset.train_test_exp, SPARSE_ADAM_AVAILABLE)
        if not args.skip_test:
            render_depth_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipe, background, dataset.train_test_exp, SPARSE_ADAM_AVAILABLE)

if __name__ == "__main__":
    main()
PY

python render_depth_npy.py -m "$MODEL_DIR" --skip_test
# depth 输出目录：$MODEL_DIR/train/ours_<iter>/depth_npy/*.npy
```

### 10.3 VGGT（Visual Geometry Grounded Transformer）：视频/图片 → depth（可选导出 COLMAP）

```bash
# 0) 基础依赖（有 sudo 用 apt；无 sudo 可用 conda-forge）
sudo apt-get update
sudo apt-get install -y ffmpeg git wget
# 无 sudo：conda install -c conda-forge ffmpeg git wget -y

# 1) 创建环境（示例：Python 3.10）
conda create -n vggt python=3.10 -y
conda activate vggt

# 2) 安装 PyTorch（按你的 CUDA 版本选择；VGGT 更推荐 GPU）
# pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
# pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu118
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cpu

# 3) 拉取 VGGT 并安装依赖
git clone https://github.com/facebookresearch/vggt.git
cd vggt
pip install -r requirements.txt

# 4) 准备 scene 目录：视频抽帧到 images/
SCENE=/abs/path/to/vggt_scene
mkdir -p "$SCENE/images"
# 推荐直接输出 518x518（VGGT 默认输入尺寸），保证 RGB/depth 对齐
ffmpeg -i /abs/path/to/video.mp4 \
  -vf "fps=2,scale=518:518:force_original_aspect_ratio=decrease,pad=518:518:(ow-iw)/2:(oh-ih)/2:black" \
  -q:v 2 "$SCENE/images/%06d.jpg"

# 5) 运行 VGGT：输出每帧 depth（.npy）+ 相机参数（可选）
cat > run_vggt_depth.py <<'PY'
from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images_square
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene_dir', required=True, type=str)
    ap.add_argument('--out_dir', required=True, type=str)
    ap.add_argument('--model_id', default='facebook/VGGT-1B', type=str)
    ap.add_argument('--img_size', default=518, type=int)
    ap.add_argument('--max_images', default=200, type=int)
    args = ap.parse_args()

    scene_dir = Path(args.scene_dir)
    image_dir = scene_dir / 'images'
    image_paths = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}])
    if not image_paths:
        raise SystemExit(f'No images found under: {image_dir}')
    if args.max_images > 0:
        image_paths = image_paths[: args.max_images]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = VGGT.from_pretrained(args.model_id).to(device).eval()

    images, _ = load_and_preprocess_images_square([str(p) for p in image_paths], target_size=args.img_size)
    images = images.to(device)
    images_b = images.unsqueeze(0)  # [B=1, S, 3, H, W]

    if device == 'cuda':
        major = torch.cuda.get_device_capability()[0]
        amp_dtype = torch.bfloat16 if major >= 8 else torch.float16
        amp_ctx = torch.cuda.amp.autocast(dtype=amp_dtype)
    else:
        amp_ctx = nullcontext()

    with torch.no_grad(), amp_ctx:
        tokens, ps_idx = model.aggregator(images_b)
        pose_enc = model.camera_head(tokens)[-1]
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images_b.shape[-2:])
        depth, depth_conf = model.depth_head(tokens, images_b, ps_idx)

    depth = depth.squeeze(0).squeeze(1).cpu().numpy().astype(np.float32)
    depth_conf = depth_conf.squeeze(0).squeeze(1).cpu().numpy().astype(np.float32)
    extrinsic = extrinsic.squeeze(0).cpu().numpy().astype(np.float32)
    intrinsic = intrinsic.squeeze(0).cpu().numpy().astype(np.float32)

    out_dir = Path(args.out_dir)
    depth_dir = out_dir / 'depth_npy'
    conf_dir = out_dir / 'depth_conf_npy'
    depth_dir.mkdir(parents=True, exist_ok=True)
    conf_dir.mkdir(parents=True, exist_ok=True)

    for i in range(depth.shape[0]):
        stem = image_paths[i].stem
        np.save(depth_dir / f'{stem}.npy', depth[i])
        np.save(conf_dir / f'{stem}.npy', depth_conf[i])

    np.save(out_dir / 'extrinsic.npy', extrinsic)
    np.save(out_dir / 'intrinsic.npy', intrinsic)
    (out_dir / 'image_list.txt').write_text('\n'.join([p.name for p in image_paths]) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
PY

python run_vggt_depth.py --scene_dir "$SCENE" --out_dir "$SCENE/vggt_out"
# 输出：
# - $SCENE/vggt_out/depth_npy/*.npy
# - $SCENE/vggt_out/depth_conf_npy/*.npy
# - $SCENE/vggt_out/{extrinsic.npy,intrinsic.npy,image_list.txt}

# 6) （可选）导出 COLMAP：生成 sparse 重建（需要额外依赖）
# 可能需要：sudo apt-get install -y build-essential cmake
pip install -r requirements_demo.txt
python demo_colmap.py --scene_dir="$SCENE"
# 如果要给 graphdeco 3DGS 用，通常需要 sparse/0：
if [ -f "$SCENE/sparse/cameras.bin" ]; then
  mkdir -p "$SCENE/sparse/0"
  mv "$SCENE/sparse/"{cameras.bin,images.bin,points3D.bin} "$SCENE/sparse/0/"
fi
```
