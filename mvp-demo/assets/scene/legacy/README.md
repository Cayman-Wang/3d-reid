# Scene Legacy Archive

该目录存放所有从当前主线下线的 MuJoCo 场景 canonical 归档。

- `v1/`：仅服务历史 `iciscae_node01_uav_v1` 复现。
- `humanoid/`：存放从当前 clean 主线下线的 humanoid 场景。

当前规则：

- `mvp-demo/assets/scene/` 根目录只允许保留 clean 主线场景。
- 任何 `mujoco_humanoid_*.xml` 都不再出现在活跃根目录。
- 若需复现 `v1 / v2` 历史结果，应从本目录中的 canonical 路径启动，而不是回退到旧 root 路径。
