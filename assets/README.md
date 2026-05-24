# Assets

This directory is for local assets used by ground-to-air synthetic UAV/aircraft 4D-ReID data collection.

Large model files, scene packages, textures, converted engine assets, caches, and archives are local runtime assets and are ignored by git. Keep only lightweight metadata, notes, and directory placeholders under version control.

## Layout

```text
assets/
  models/
    aircraft_raw/
      <identity_id>/
        source_model.fbx|obj|gltf|glb|zip|rar
        textures/
        license_or_source.txt
        asset_meta.json
    aircraft_normalized/
      <identity_id>/
        normalized.fbx
        textures/
        preview.png
        asset_meta.json
  scenes/
    raw/
      <scene_source_or_name>/
        original_scene_files/
        license_or_source.txt
        scene_meta.json
    normalized/
      <scene_id>/
        engine_ready_scene_files/
        scene_meta.json
```

## Conventions

- `models/aircraft_raw/<identity_id>/` keeps original downloaded or prepared aircraft models unchanged.
- `models/aircraft_normalized/<identity_id>/` keeps Blender/UE-normalized assets with meter units, stable origin, known axes, and cleaned texture paths.
- `scenes/raw/<scene_source_or_name>/` keeps original downloaded or exported city/scene assets unchanged.
- `scenes/normalized/<scene_id>/` keeps project-ready scenes after scale, coordinate, lighting, collision, or engine import cleanup.
- Use stable lowercase English ids, for example `f15`, `j10`, `uav_quadrotor_01`, `carla_town10hd`, `city_sample_block_a`.
- Record source URL, license/terms, allowed use, and redistribution constraints before using third-party assets in experiments, papers, or released datasets.
- Do not commit large model or scene files to git.

