# CARLA-Air 地对空采集工程交接

日期：2026-05-24

## 1. 当前状态

当前项目路径固定为：

```text
/home/grasp/data/3d-reid
```

CARLA-Air v0.1.7 runtime 已部署在：

```text
local/carla_air/simulators/CarlaAir-v0.1.7/
```

当前已完成：

- CARLA-Air simulator 与 Python API smoke；
- `carlaAir` conda 环境；
- 交互式地对空三相机节点布设脚本；
- `Town10HD` 中 `node01-node05` 的固定相机节点配置。

当前尚未完成：

- 正式采集脚本；
- 自定义飞行器模型导入；
- 飞行轨迹控制；
- RGB/depth/mask/bbox/object pose/trajectory 导出；
- 到 `scene_dir + tracklets + points_by_timestamp` 的转换。

## 2. 关键文件

布设脚本：

```text
tools/carla_air/place_camera_node.py
```

默认三相机 rig：

```text
configs/camera_rigs/node_tri_cam_parallel_v1.json
```

已保存相机节点：

```text
local/carla_air/camera_nodes/Town10HD_ground_to_air_nodes_v1.json
```

阶段汇总：

```text
research/reports/carla_air_ground_to_air_camera_nodes_milestone_2026_05_24_zh.md
```

采集路线文档：

```text
数据集采集/ground_to_air_synthetic_uav_4d_reid_collection_guide_zh.md
```

## 3. 启动方式

终端 1：启动 CARLA-Air。不要在该终端激活 `carlaAir` conda 环境。

```bash
conda deactivate
cd /home/grasp/data/3d-reid/local/carla_air/simulators/CarlaAir-v0.1.7
./CarlaAir.sh Town10HD --res 1280x720 --quality Low --fg
```

终端 2：运行 Python 工具。

```bash
cd /home/grasp/data/3d-reid
conda activate carlaAir
python tools/carla_air/place_camera_node.py --wait-seconds 120
```

继续从已有最大节点的下一个节点开始布设：

```bash
python tools/carla_air/place_camera_node.py --wait-seconds 120 --resume-next
```

手动指定某个节点：

```bash
python tools/carla_air/place_camera_node.py --node-id node03 --wait-seconds 120
```

## 4. 布设工具控制

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

## 5. 当前节点摘要

当前配置文件：

```text
local/carla_air/camera_nodes/Town10HD_ground_to_air_nodes_v1.json
```

摘要：

```text
schema_version: carla_air_ground_to_air_camera_nodes_v1
map: Town10HD
layout_id: node_tri_cam_parallel_v1
node_count: 5
node_ids: node01, node02, node03, node04, node05
```

每个 node 都有：

```text
cam0 / cam1 / cam2
K
image_size
T_node_from_cam
fov_x_deg
carla_relative_transform
carla_world_transform
```

## 6. 快速检查命令

检查 CARLA-Air 是否正在运行：

```bash
ps -eo pid,cmd | rg 'CarlaUE4|CarlaAir|auto_traffic'
ss -tlnp | rg '2000|41451'
```

检查布设脚本：

```bash
PYTHONNOUSERSITE=1 /home/grasp/miniconda3/envs/carlaAir/bin/python -m py_compile tools/carla_air/place_camera_node.py
PYTHONNOUSERSITE=1 /home/grasp/miniconda3/envs/carlaAir/bin/python tools/carla_air/place_camera_node.py --help
```

检查已保存节点：

```bash
python - <<'PY'
import json
from pathlib import Path
p = Path('local/carla_air/camera_nodes/Town10HD_ground_to_air_nodes_v1.json')
data = json.loads(p.read_text())
print(data.get('schema_version'), data.get('map'), data.get('layout_id'))
nodes = data.get('nodes', [])
print(len(nodes), [n.get('node_id') for n in nodes])
for n in nodes:
    print(n.get('node_id'), sorted(n.get('cameras', {})))
PY
```

## 7. 下一步任务

推荐下一步做最小 `capture smoke`：

1. 新增 CARLA-Air 采集脚本，读取 `Town10HD_ground_to_air_nodes_v1.json`；
2. spawn 指定节点或全部节点的 `cam0/cam1/cam2`；
3. 导出短序列 RGB；
4. 同步保存 camera intrinsics、camera world transform、timestamps；
5. 输出到 `local/carla_air/captures/`；
6. 通过读取导出目录确认 frame count、camera count、pose 和 timestamp 对齐。

后续再扩展：

- depth；
- semantic / instance mask；
- bbox；
- object pose；
- aircraft identity；
- trajectory；
- `scene_dir + tracklets + points_by_timestamp` 转换。

## 8. 注意事项

- 不要把 `local/carla_air/` 下的 runtime、capture 或 camera_nodes 提交进 git。
- 启动 UE/CARLA-Air 二进制时不要激活 `carlaAir` conda；Python 脚本再激活。
- 当前节点 pitch 多为小幅负值，是否满足地对空仰视需要结合实际预览画面确认。
- 当前工具只调 node anchor，不调单个相机相对布局；如需非平行光轴或不同 FOV，应新增新的 `layout_id`，不要覆盖 `node_tri_cam_parallel_v1`。
