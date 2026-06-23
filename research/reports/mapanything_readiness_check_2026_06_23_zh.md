# MapAnything readiness check（只读）

- 对象：`third_party/map-anything`
- 日期：2026-06-23
- 结论：仓库内已有可直接复用的推理入口、离线权重加载路径、统一输出格式说明与 Apache 2.0 模型分支；本轮仅做只读审计，未运行推理，也未下载权重。

## 1. 建议环境

- Python：README 给出的起步环境是 `python=3.12`；`pyproject.toml` 允许 `>=3.10.0`。
- PyTorch / CUDA：README 明确未固定 PyTorch 或 CUDA 版本，要求按本机系统安装匹配版本。
- 运行习惯：脚本里对 CUDA 开了 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`，说明默认优先 CUDA；无 CUDA 时会回退到 CPU / 自动设备选择。

## 2. 安装方式

README 里给出的最小安装是：

```bash
conda create -n mapanything python=3.12 -y
conda activate mapanything
pip install -e .
```

如需完整可选依赖：

```bash
pip install -e ".[all]"
```

如果只补外部模型能力，README 还列了按需安装，例如 `.[gradio]`、`.[dust3r]`、`.[mast3r]`、`.[pi3]`、`.[pow3r]`、`.[must3r]`、`.[vggt-omega]`、`.[depth-anything-3]`。

## 3. Apache 模型与本地权重路径

- 在线加载时，README / demo 代码都支持 `facebook/map-anything-apache`。
- 建议把 Apache 本地权重统一放到 `local/models/mapanything-apache/`，避免直接把第三方模型文件混在 `third_party/map-anything/` 工作树里。
- 代码里的离线权重入口在 `scripts/demo_local_weight.py`，其 `LOCAL_CONFIG` 体现了本地配置约定：

```python
{
    "path": "configs/train.yaml",
    "model_str": "mapanything",
    "config_overrides": [
        "machine=aws",
        "model=mapanything",
        "model/task=images_only",
        "model.encoder.uses_torch_hub=false",
    ],
    "checkpoint_path": "ckpt/model.safetensors",
    "config_json_path": "config.json",
    "strict": False,
}
```

建议映射到本仓库本地路径后，最小目录可以整理成：

```text
local/models/mapanything-apache/
  config.json
  model.safetensors
```

如果沿用 `demo_local_weight.py` 的本地加载逻辑，则可把：

- `config_json_path` 对应到 `local/models/mapanything-apache/config.json`
- `checkpoint_path` 对应到 `local/models/mapanything-apache/model.safetensors`

- `mapanything.utils.hf_utils.initialize_mapanything_local()` 会先读 Hydra 配置，再按 `config_json_path` / `checkpoint_path` 加载本地资源。
- 训练/benchmark 侧的 checkpoint 转换脚本是 `scripts/convert_hf_to_benchmark_checkpoint.py`，导出的格式要求顶层包含 `model` 键，适合写成 `*.pth` 后再走 benchmark 配置。

## 4. 输入格式

README 和 `mapanything/utils/inference.py` 一致：每个 view 至少要有 `img` 和 `data_norm_type`，其中：

- `img`：RGB 图像张量，常见形状为 `(B, 3, H, W)`，原始示例里也接受按 `load_images()` 预处理后的视图。
- `intrinsics` 或 `ray_directions`：二者二选一，不能同时给。
- `depth_z`：需要配合 `intrinsics` 或 `ray_directions`。
- `camera_poses`：OpenCV 约定的 `cam2world`，即 `+X 右、+Y 下、+Z 前`，可用 `4x4` 矩阵，或 `(quats, trans)`。
- `is_metric_scale`：可选，表示几何输入是否已是米制尺度。

README 还明确了约束：

- 若某个 view 提供了 `depth_z`，必须同时提供 `intrinsics` 或 `ray_directions`。
- 若任一 view 带 `camera_poses`，第 0 个参考 view 也必须带 `camera_poses`。

## 5. 输出字段

README 的统一输出格式里，MapAnything 及其 wrapper 会稳定返回这些键：

- `pts3d`
- `pts3d_cam`
- `ray_directions`
- `depth_along_ray`
- `cam_trans`
- `cam_quats`
- `conf`

在 image-only / demo 脚本里，还能直接看到：

- `depth_z`
- `intrinsics`
- `camera_poses`
- `mask`
- `img_no_norm`

其中 `demo_images_only_inference.py` 和 `demo_local_weight.py` 都是用 `depth_z + intrinsics + camera_poses` 重建 `pts3d`，再配合 `mask` 做后处理或导出。

## 6. 如何适配到 `points_by_timestamp`

现有仓库没有把 MapAnything 直接接成 `points_by_timestamp` 的专用导出器，但适配路径是清楚的，且只需基于现有输出做轻量转换：

1. 按时间戳逐帧组织输入 view。
2. 对每帧取 `depth_z`、`intrinsics`、`camera_poses`，用现有几何函数或等价 backproject 逻辑把深度转成世界坐标点。
3. 用 `mask` / `conf` 过滤无效像素，再把每帧点云和对应时间戳写成 `points_by_timestamp/<timestamp>.npy`。
4. 另外写一个 `index.csv` 记录时间戳、相对路径、帧号等索引；这和仓库里现有 `points_by_timestamp` 目录约定一致。

换句话说，MapAnything 的输出已经包含了做 `points_by_timestamp` 所需的核心几何量，适配工作主要是“逐帧落盘 + 索引表”。

## 7. 本轮边界

- 未运行 `model.infer()`。
- 未下载 Hugging Face 权重。
- 未把 Apache / 研究版做功能性优劣结论，只记录仓库里明确写出的许可与入口。

## 8. 证据来源

- `third_party/map-anything/README.md`
- `third_party/map-anything/pyproject.toml`
- `third_party/map-anything/scripts/demo_local_weight.py`
- `third_party/map-anything/scripts/convert_hf_to_benchmark_checkpoint.py`
- `third_party/map-anything/mapanything/utils/inference.py`
