# Demo PPT Outline (5 Slides): 3D-aware Track Retrieval MVP

> 目标：用 3–5 分钟讲清楚“你这个 demo 在做什么、怎么做、跑通了什么、下一步怎么扩展到分布式多相机节点”。

---

## Slide 1 — Problem & Goal（要解决什么）

**标题建议**
- `3D-aware ReID MVP: From Video to Track Retrieval`

**一页必须讲清的 3 件事**
1) **任务定义（本 demo 解决的问题）**
   - 单类目标：`person / car / drone`（一次任务只做一种，不追求泛化）
   - 输入：普通相机视频（或摄像头流）
   - 输出：把“同一目标实例”在不同视频/不同 scene 中 **检索出来**（top‑K）

2) **为什么需要“3D-aware”**
   - 纯 RGB 容易受光照/背景/遮挡影响
   - 引入 Depth/几何特征，增强跨视角/跨场景鲁棒性（哪怕是粗几何也能提供约束）

3) **最小可交付（MVP 交付物）**
   - 每个 tracklet（轨迹片段）→ 一个向量 `track embedding`
   - Query track → 在多个 scene 的 gallery tracks 中 top‑K 检索

**建议配图/素材（放在右侧或底部）**
- 1 张“输入 vs 输出”的对比图：
  - 左：视频帧 + 目标（框或 mask）
  - 右：检索结果 top‑K（表格截图即可）

**讲稿提示（10–15 秒）**
- “我们先不做大而全的行人 ReID 训练，而是先跑通一个可复现的跨视频检索 MVP：视频 → mask + depth → track embedding → top‑K 检索。”

---

## Slide 2 — End-to-End Pipeline（总流程图 + 产物）

**标题建议**
- `Pipeline Overview (MVP)`

**必须包含的模块（按数据流顺序）**
1) **门控采集（可选）**：YOLO 检测触发录制，落盘一段 `scene`
2) **深度生成**：3DGS/COLMAP → 渲染 `depth_npy/`（与 `images/` 对齐）
3) **实例分割**：SAM2 视频分割 → `masks/obj_*/<stem>.png`
4) **Tracklets**：把每帧结果组织成轨迹片段 `tracklets.json`
5) **Embedding**：RGB embedding + 几何 embedding（depth→点云→descriptor）→ track mean pooling
6) **Retrieval**：跨 scene top‑K 检索（query vs gallery）

**流程图（推荐：ASCII 版，任何 Markdown 预览都能“显示”）**
```text
Video / Camera Stream
  |
  | (optional) YOLO gating capture
  v
SCENE_DIR/input/*.jpg  + capture_meta.json + frame_times.csv
  |
  v
COLMAP convert.py  ->  SCENE_DIR/images/*.jpg  (alignment anchor)
  |                      |
  |                      +--> SAM2 video masks -> SCENE_DIR/masks/obj_000/<stem>.png
  |                                      |
  |                                      +--> make tracklets.json
  |
  +--> cameras.bin -> export intrinsics.json
  |
  +--> 3DGS train.py -> render depth -> SCENE_DIR/depth_npy/<stem>.npy
                                 |
                                 v
                     RGB + Depth + Mask (+K)
                                 |
                                 v
                 frame embeddings -> mean pool -> track embedding
                                 |
                                 v
                     top-K retrieval across scenes
```

**流程图（Mermaid 版；如果你的 Markdown 预览不支持 Mermaid，请用上面的 ASCII 版）**
```mermaid
graph TD
  A[Video / camera stream] --> B[YOLO gating capture (optional)]
  B --> C[SCENE_DIR/input/*.jpg + meta]
  C --> D[COLMAP convert.py (undistort + sparse)]
  D --> E[SCENE_DIR/images/*.jpg (alignment anchor)]
  D --> F[SCENE_DIR/sparse/0/cameras.bin]
  F --> G[export intrinsics.json]
  D --> H[3DGS train.py]
  H --> I[render depth -> SCENE_DIR/depth_npy/*.npy]

  E --> J[SAM2 video masks (init box on frame 0)]
  J --> K[SCENE_DIR/masks/obj_000/<stem>.png]
  K --> L[make tracklets.json]

  E --> M[RGB crop per frame]
  I --> N[Depth per frame]
  G --> O[Intrinsics per frame]
  K --> P[Mask per frame]

  M --> Q[RGB embedding (CLIP or hist)]
  N --> R[Depth+K+Mask -> points -> geometry desc]
  Q --> S[fuse + normalize]
  R --> S
  S --> T[mean pool -> track embedding]
  T --> U[top-K retrieval across scenes]
```

**建议同时展示“目录产物”截图（非常加分）**
- `SCENE_DIR/images/`
- `SCENE_DIR/depth_npy/`
- `SCENE_DIR/masks/obj_000/`
- `SCENE_DIR/tracklets.json`
- `SCENE_DIR/embeddings/tracks.npy + tracks_meta.json`

**讲稿提示（20–30 秒）**
- “对齐口径以 `images/` 为准；深度与 mask 都对齐到这套帧。最终每条 track 只有一个向量，便于跨视频/跨节点检索。”

---

## Slide 3 — Key Design Choices（为什么这样设计）

**标题建议**
- `Why This Works as an MVP`

**建议分 4 块，每块 1–2 行（配小图标/短句）**
1) **mask 优先（SAM2）**
   - 深度/点云对背景极敏感；mask 能显著减少“学到背景/场景”的风险
   - 也能让 bbox 抖动对几何的影响变小

2) **对齐策略固定**
   - 一切以下游 `SCENE_DIR/images/` 作为像素坐标系基准
   - `depth_npy/<stem>.npy` 与 `images/<stem>.jpg` 一一对应，减少错位 bug

3) **track 级表征**
   - 单帧 embedding 易抖（遮挡/运动模糊）
   - track mean pooling 提升稳定性，检索单位更合理（track vs track）

4) **几何分支“先跑通再增强”**
   - MVP 用轻量几何描述子（radial hist / FPFH）就能形成“3D-aware”信号
   - 后续可替换为训练/预训练 3D encoder（PointNet++ / Transformer / OpenShape 等）

**建议放一张“mask + depth → point cloud”示意图**
- 左：RGB
- 中：mask overlay
- 右：depth colormap 或点云可视化截图（哪怕是稀疏点云）

---

## Slide 4 — Demo Evidence（跑通的证据：输入/中间件/结果）

**标题建议**
- `Demo Results (What You Can See)`

**推荐布局：上半部分“中间产物”，下半部分“检索结果”**

**上半部分：中间产物（建议 3 张图并排）**
1) `RGB frame`（原图）
2) `Mask overlay`（mask 叠加在 RGB 上）
3) `Depth visualization`（depth colormap 或与 mask 叠加的 depth）

**下半部分：检索结果（表格/终端截图都行）**
- 展示一条 query track：`query_track_id`
- 展示 top‑K：`rank / score / gallery_track_id / gallery_scene_dir`

**建议明确写出“你跑通了哪些脚本/产物”**
- SAM2 输出：`masks/obj_000/*.png`
- Tracklets：`tracklets.json`
- Embeddings：`embeddings/tracks.npy` + `embeddings/tracks_meta.json`
- Retrieval：top‑K 输出（屏幕截图）

**可选补充（如果你有两段/两 scene）**
- 展示跨 scene 检索：Scene A → Scene B（或多个 gallery scenes）

---

## Slide 5 — Limitations & Roadmap（扩展到“多节点×每节点3相机”）

**标题建议**
- `From MVP to Distributed Multi-Camera Nodes`

**现状边界（诚实但不否定成果）**
- 当前 depth 来源是 3DGS/COLMAP，依赖“视差/相机运动/静态一致性”，对“固定多相机 rig + 动态目标”不天然适配
- 当前是 MVP：以“跑通链路”为目标，未做大规模训练与标准评测

**下一步：你计划的仿真优先路线（IsaacLab）**
1) **节点内 3D：先用仿真 GT depth/pose 跑通**
   - 3 台相机同一时刻输出 RGB+Depth+Mask
   - 融合成节点内点云/占据体 → 几何 embedding（track pooling）
2) **跨节点 ReID：仍然 track embedding + top‑K 检索**
   - 多节点 = 多个 scene/gallery 的集合（检索层不变）
3) **从仿真到真实：只替换 depth 来源**
   - GT depth → stereo depth / visual hull / 其他多视角几何

**你可以在这一页放一个“未来架构小图”**
- Node 1（3 cameras）→ node-level track embeddings
- Node 2（3 cameras）→ node-level track embeddings
- Central retrieval service（或离线脚本）→ cross-node top‑K matches

**结束句（5 秒）**
- “MVP 已经证明：只要能稳定得到 mask +（哪怕粗）几何，就能做跨 scene/跨节点的 track 检索；接下来在 IsaacLab 把节点内 3D 融合跑通，再迁移到真实部署。”
