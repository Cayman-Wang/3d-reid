# CARLA-Air 地对空相机节点布设里程碑

日期：2026-05-24

## 1. 阶段结论

- CARLA-Air v0.1.7 Linux runtime 已在当前项目内完成部署与 Python API smoke。
- Synthetic UAV/aircraft 数据集采集路线已从“方案讨论”进入“可交互布设固定相机节点”的工程阶段。
- 已实现 `tools/carla_air/place_camera_node.py`，用于在 CARLA-Air 场景中逐个布设地面固定三相机节点。
- 已在 `Town10HD` 中保存 `node01` 到 `node05` 共 5 个节点，每个节点包含 `cam0/cam1/cam2` 三路相机配置。
- 当前阶段只完成相机节点布设与配置固化；尚未完成正式 RGB/depth/mask/pose 序列采集，也尚未导入自定义飞行器模型或轨迹。

## 2. 已完成产物

CARLA-Air runtime：

```text
local/carla_air/simulators/CarlaAir-v0.1.7/
```

相机节点布设脚本：

```text
tools/carla_air/place_camera_node.py
```

默认节点内三相机布局：

```text
configs/camera_rigs/node_tri_cam_parallel_v1.json
```

已保存节点配置：

```text
local/carla_air/camera_nodes/Town10HD_ground_to_air_nodes_v1.json
```

数据集采集指导文档：

```text
数据集采集/ground_to_air_synthetic_uav_4d_reid_collection_guide_zh.md
```

## 3. 相机节点配置状态

当前已保存配置摘要：

```text
schema_version: carla_air_ground_to_air_camera_nodes_v1
map: Town10HD
layout_id: node_tri_cam_parallel_v1
node_count: 5
node_ids: node01, node02, node03, node04, node05
```

节点 anchor 位姿如下：

| node_id | x | y | z | pitch | yaw | roll |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `node01` | -91.684 | 190.147 | 10.605 | -0.649 | 237.728 | 0.000 |
| `node02` | -89.197 | 170.597 | 11.500 | -2.689 | -188.392 | 0.000 |
| `node03` | -82.552 | 158.592 | 10.933 | -2.689 | 256.448 | 0.000 |
| `node04` | -70.023 | 164.042 | 10.737 | -2.689 | 310.328 | 0.000 |
| `node05` | -81.250 | 106.707 | 11.242 | -8.089 | 117.008 | 0.000 |

每个节点均包含完整三路相机：

```text
cam0 / cam1 / cam2
image_size: 1280 x 720
fov_x_deg: 91.493
fields: K, distortion_model, dist, T_node_from_cam, carla_relative_transform, carla_world_transform
```

当前未发现重复 `node_id`，也未发现缺失相机的节点。

## 4. 布设工具行为

`place_camera_node.py` 的当前定位是交互式节点布设工具，不是正式采集脚本。

关键行为：

- 每次只显示 active node 的 `cam0/cam1/cam2` 三路画面；
- 使用 `node_tri_cam_parallel_v1` 作为节点内默认三相机布局；
- 第一版只移动/旋转整个 node anchor，不微调单个相机相对位姿；
- 保存时写入 `local/carla_air/camera_nodes/Town10HD_ground_to_air_nodes_v1.json`；
- 支持 `--resume-next`，可自动从已有最大顺序节点的下一个节点继续布设。

当前控制方式：

```text
W/S        前进/后退
A/D        左右移动
Q/E        下/上移动
←/→        调整 yaw
↑/↓        调整 pitch
Z/X        roll
[/]        调整移动速度
Space      保存
Ctrl+S/F5  保存备选
N          保存并切到下一个 node
B          保存并切到上一个 node
P          打印位姿
ESC        退出
```

## 5. 验证记录

已完成的静态检查：

```bash
PYTHONNOUSERSITE=1 /home/grasp/miniconda3/envs/carlaAir/bin/python -m py_compile tools/carla_air/place_camera_node.py
```

已完成的功能检查：

- CARLA-Air Python API smoke：`carla` 与 `airsim` 可导入并连接仿真器；
- `place_camera_node.py --help` 可正常输出参数；
- 布设脚本可连接 `Town10HD`，spawn active node 三路 preview camera；
- SIGINT/退出后 preview sensors 可清理；
- 已保存 `node01` 到 `node05` 的配置文件。

## 6. 运行注意事项

启动 CARLA-Air 仿真器时，不建议激活 `carlaAir` conda 环境。推荐分工是：

```text
CARLA-Air / UE 二进制：不激活 conda
Python API / 布设脚本：激活 carlaAir
```

推荐启动方式：

```bash
conda deactivate
cd /home/grasp/data/3d-reid/local/carla_air/simulators/CarlaAir-v0.1.7
./CarlaAir.sh Town10HD --res 1280x720 --quality Low --fg
```

另开终端运行布设脚本：

```bash
cd /home/grasp/data/3d-reid
conda activate carlaAir
python tools/carla_air/place_camera_node.py --wait-seconds 120
```

如果继续布设下一个节点：

```bash
python tools/carla_air/place_camera_node.py --wait-seconds 120 --resume-next
```

## 7. 当前限制与下一步

当前限制：

- 还没有正式采集脚本；
- 还没有导出 RGB/depth/mask/bbox/object pose/trajectory；
- 还没有接入自定义飞行器模型和 identity 轨迹；
- 当前 pitch 角是否满足“地对空”需要结合画面人工复核。

下一步建议：

1. 写 `capture smoke` 脚本，读取已保存的 `node01-node05`，spawn 全部或指定节点相机；
2. 先导出短序列 RGB、camera intrinsics、camera world pose、timestamps；
3. 再扩展 depth、semantic/instance mask、bbox、object pose；
4. 最后转换到当前项目主线需要的 `scene_dir + tracklets + points_by_timestamp` 数据契约。
