# DGGT readiness audit（只读证据版）

结论：`third_party/dggt` 在本仓库里已经具备可被包装调用的代码入口，但本轮没有执行重型推理，也没有下载任何权重。下面只基于仓库内 README、依赖文件和现有 wrapper 脚本做 readiness 归纳，不把 DGGT 输出表述为正式几何结论。

## 建议运行环境

- Python：`3.10`
- PyTorch：`torch==2.4.1`，配套 `torchvision==0.19.1`、`torchaudio==2.4.1`
- CUDA：建议使用与 PyTorch 2.4.1 匹配的 GPU 运行时，优先 `CUDA 11.8` 或 `12.1` 兼容环境
- 其他核心依赖：`third_party/dggt/requirements.txt` 中的 `open3d`、`gsplat`、`diffusers`、`transformers`、`rerun-sdk`、`opencv-python`、`huggingface_hub` 等
- 点云/数据侧依赖：`third_party/dggt/requirements_data.txt` 中还有 `waymo-open-dataset-tf-2-11-0`、`tensorflow-gpu==2.11.0`、`nuscenes-devkit` 等，是否安装取决于数据集分支

## 安装命令

```bash
conda create -n dggt python=3.10
conda activate dggt

pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1
pip install -r third_party/dggt/requirements.txt

cd third_party/dggt/third_party/pointops2
python setup.py install
cd /home/grasp/data/3d-reid
```

如果要跑数据处理分支，再按需补装 `third_party/dggt/requirements_data.txt` 里的数据集依赖。

## 权重与 checkpoint

仓库内 upstream README 明确列出的 checkpoint 是：

- `model_latest_waymo.pt`：DGGT 主推理模型下载文件名
- `model_difix.pkl`：diffusion refinement 模型下载文件名
- `tapip3d_final.pth`：TAPIP3D tracking 模型下载文件名
- `model.pt`：训练侧额外提到的 VGGT 预训练模型

建议本地路径统一放到 `local/models/dggt/`，便于 sidecar wrapper 与非仓库内第三方代码解耦：

- `local/models/dggt/model_latest_waymo.pt`
- `local/models/dggt/model_difix.pkl`
- `local/models/dggt/tapip3d_final.pth`
- `local/models/dggt/model.pt`

当前仓库的 `mvp-demo` wrapper 只要求一个 `--ckpt_path`，用于把单个 DGGT 主 checkpoint 喂给推理入口；`model_difix.pkl` 和 `tapip3d_final.pth` 只有在对应分支或后续扩展需要时才是必需项。

## 最小推理命令模板

原生 DGGT README 的最小推理入口是 `third_party/dggt/inference.py`。仓库内 wrapper 则是三段式：先生成 manifest，再跑 joint inference，再导出点云。推荐的最小模板如下：

```bash
python mvp-demo/scripts/prepare_dggt_multiview_manifest.py \
  --scene_dir mvp-demo/data/nodes/node01/scenes/<scene_id> \
  --cam_ids cam0,cam1,cam2 \
  --num_sync_steps 27 \
  --out_root mvp-demo/output/dggt_multiview

python mvp-demo/scripts/run_dggt_multiview_joint.py \
  --manifest mvp-demo/output/dggt_multiview/<scene_id>/input/manifest.json \
  --dggt_repo third_party/dggt \
  --ckpt_path local/models/dggt/model_latest_waymo.pt \
  --device cuda \
  --torch_dtype float32 \
  --use_input_calib

python mvp-demo/scripts/export_dggt_multiview_points.py \
  --bundle mvp-demo/output/dggt_multiview/<scene_id>/run_full_frame_joint/reconstruction_bundle.npz \
  --scene_dir mvp-demo/data/nodes/node01/scenes/<scene_id> \
  --dggt_repo third_party/dggt \
  --out_root mvp-demo/output/dggt_multiview
```

说明：上面是仓库 wrapper 的最小链路；upstream README 里的原生入口仍然是 `python inference.py --image_dir ... --ckpt_path ... --output_path ...`。两者目标不同，前者适配本仓库的 tri-camera scene manifest 和后处理结构，后者是 upstream 的通用单次推理接口。

## 预期输出结构

仓库 wrapper 的预期输出树是：

```text
mvp-demo/output/dggt_multiview/<scene_id>/
  input/manifest.json
  run_full_frame_joint/
    reconstruction_bundle.npz
    pose_alignment_report.json
    probe_meta.json
    points_export/
      meta.json
      points_index.csv
      points_by_timestamp/*.npy
      debug/raw_by_view/*.npy
      debug/fused_preview_ply/*.ply
```

其中 `reconstruction_bundle.npz` 是 wrapper 内部 bundle，不是 upstream 论文或官方 benchmark 的最终标准格式；`points_by_timestamp`、`points_index.csv`、`probe_meta.json` 都是仓库侧的诊断/导出产物。

## 与现有仓库 DGGT wrapper/scripts 的关系和差异

- `mvp-demo/scripts/prepare_dggt_multiview_manifest.py`：把本仓库的 `frame_times.csv` 和 `calib/rig.json` 变成 tri-camera `manifest.json`，只负责输入整理，不做模型推理。
- `mvp-demo/scripts/run_dggt_multiview_joint.py`：直接 import `third_party/dggt/utils/*`，读取 manifest 后做一次 joint forward，写出 `reconstruction_bundle.npz`、`pose_alignment_report.json` 和 `probe_meta.json`。
- `mvp-demo/scripts/export_dggt_multiview_points.py`：基于 bundle 做点云导出、体素下采样和阈值过滤，写出 `points_by_timestamp`、`points_index.csv` 和 debug 预览文件。
- `mvp-demo/scripts/precompute_dggt_multiview_static_geometry.py`：只是把上述三步串起来，并加了输出存在性检查；它是编排器，不是新的 DGGT 算法实现。
- 差异点：仓库 wrapper 面向本仓库的 tri-camera scene 和本地输出契约，upstream `third_party/dggt/inference.py` 是面向 DGGT 原生数据布局的通用入口；仓库 wrapper 还显式保留了 `use_input_calib`、mask 预处理、点云导出和诊断报告。

## 本轮状态

- 没有运行重型 DGGT inference
- 没有下载任何 checkpoint 或 diffusion/tracking 权重
- 没有把 DGGT 输出提升为 formal geometry 或 formal annotation
