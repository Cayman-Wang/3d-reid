# CARLA-Air / AirSim Runtime 自启动 Runbook

## 目的与范围

当任务需要实时证据，而现有 CARLA-Air / AirSim runtime 尚未在线时，Codex 可以在本仓库内自启动该 runtime。

此文档只描述仓库内已观察到的默认流程；候选启动命令需要按现场情况确认，不代表跨环境通用真理。

## 端口检查

启动前先检查两项服务是否已经在线：

- CARLA: `127.0.0.1:2000`
- AirSim: `127.0.0.1:41451`

如果两者都已可达，不要再起新的 runtime。

## 启动器政策

只使用本文明确列出的启动方式，不要自行猜测路径或参数。

当前任务不再使用不稳定的非 `--fg` wrapper / traffic 启动路径。若用户没有手动启动 runtime，且任务确实需要 live evidence，Codex 可以且只能使用下面这条前台命令自启动：

```bash
cd /home/grasp/data/3d-reid/local/carla_air/simulators/CarlaAir-v0.1.7
./CarlaAir.sh Town10HD --res 1280x720 --quality Low --opengl --fg
```

`--fg` 会直接 `exec` UE 进程，不经过会自动注入 traffic 的 wrapper。当前任务只允许这条路径作为 Codex 自启动方式；不要再依赖非 `--fg` 的 wrapper / traffic 分支。

## 日志与 PID

- 运行日志写到 `local/carla_air/runtime_logs/`
- PID 记录写到 `local/carla_air/runtime_pids/`
- 只管理本次 Codex 自己启动的 PID
- 如果用户已经手动启动了 runtime，Codex 可以复用端口做 probe，但不得关闭用户窗口。
- 如果 Codex 自行启动 runtime，需要把外层命令日志写入 `local/carla_air/runtime_logs/`，PID 写入 `local/carla_air/runtime_pids/`，且只允许停止本轮 Codex 自己启动的 PID。

## 成功标准

满足以下条件后，才把 runtime 视为可用于 live probe：

- CARLA `127.0.0.1:2000` 可连通
- AirSim `127.0.0.1:41451` 可连通
- CARLA Python API 能完成一次真实 RPC 握手，例如 `get_world()` 成功返回 world/map。
- AirSim Python API 能完成一次真实连接握手，例如 `confirmConnection()` 或等价最小调用成功。
- 端口与 API 握手至少稳定一个短窗口；不能只凭 wrapper 打印的 `Ready` 判定成功。
- 启动日志无明显崩溃或卡死迹象。

当前 `carlaAir` Python 环境需要显式禁用 user site-packages，否则用户级 `msgpack` 可能覆盖环境依赖，导致 AirSim `confirmConnection()` 报 `Packer(... encoding=...)` 相关错误。握手命令使用：

```bash
PYTHONNOUSERSITE=1 /home/grasp/miniconda3/envs/carlaAir/bin/python
```

最小握手示例：

```bash
PYTHONNOUSERSITE=1 /home/grasp/miniconda3/envs/carlaAir/bin/python - <<'PY'
import carla
import airsim

carla_client = carla.Client("127.0.0.1", 2000)
carla_client.set_timeout(5.0)
world = carla_client.get_world()
print("CARLA:", world.get_map().name, len(world.get_actors()))

airsim_client = airsim.MultirotorClient(ip="127.0.0.1", port=41451)
airsim_client.confirmConnection()
print("AirSim:", airsim_client.listVehicles())
PY
```

## 停止策略

仅在本次运行中由 Codex 启动的 PID 才允许停止。不要尝试关闭用户或其他会话已经在运行的实例。

## 失败报告

如果 runtime 无法启动、端口无法就绪、或 live probe 失败，必须把阻塞原因写入 `research/reports/`。

不要补造截图、日志结论或探针结果。

## 示例探针流程

1. 先检查 `127.0.0.1:2000` 和 `127.0.0.1:41451`。
2. 如果两个端口已在线，优先复用现有 runtime，不启动第二个实例。
3. 如果端口未在线，且任务确实需要 live evidence，由 Codex 使用上面的 `--fg` 命令自启动。
4. 自启动后，按日志 / PID 规则记录；不要使用会自动注入 traffic 的非 `--fg` wrapper。
5. 等待两个端口、CARLA RPC、AirSim API 均就绪并稳定后，再执行最小 live probe。
6. 记录探针结果；如果失败，写 blocker 报告并保留原始日志路径。
