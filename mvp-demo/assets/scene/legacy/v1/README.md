# `v1` Legacy Scene Archive

This directory stores the canonical archived MuJoCo scenes for the historical
`iciscae_node01_uav_v1` benchmark.

Current mainline work must not start new capture or new manifest edits from
these files. Use the clean scenes in `mvp-demo/assets/scene/` instead.

Canonical legacy scene paths:

- `mujoco_humanoid_uav1_3cam_node_parallel.xml`
- `mujoco_humanoid_dji_mavic_3cam_node_parallel.xml`
- `mujoco_uav1_3cam_node_parallel.xml`
- `mujoco_dji_mavic_3cam_node_parallel.xml`

Archive mapping:

- `legacy/v1/` stores the historical `uav1 / dji_mavic` canonical files for
  `iciscae_node01_uav_v1`.
- `legacy/humanoid/` stores the later humanoid scene variants that were removed
  from the active clean mainline.
- The active `mvp-demo/assets/scene/` root no longer keeps compatibility copies
  for `mujoco_humanoid_*.xml`.
