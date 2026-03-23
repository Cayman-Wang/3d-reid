# Humanoid Scene Archive

该目录存放从当前 clean 主线下线的 humanoid 场景：

- `mujoco_humanoid_3cam_node_parallel.xml`
- `mujoco_humanoid_3cam_node_parallel_j10.xml`
- `mujoco_humanoid_uav1_3cam_node_parallel_v2.xml`
- `mujoco_humanoid_su34_3cam_node_parallel.xml`

用途边界：

- 仅用于历史 `v2` 复现、对照分析或回放旧 `capture_meta.json` 中记录的路径。
- 不再作为默认 viewer、默认 capture 或当前 benchmark 的推荐入口。
- 当前主线 benchmark 为 `iciscae_node01_uav_v3_clean`，请改用 `mvp-demo/assets/scene/` 根目录下的 clean 场景。
