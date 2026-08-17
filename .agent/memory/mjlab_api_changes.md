---
name: mjlab 1.3 API changes
description: Breaking changes from mjlab 1.2 to 1.3 relevant to the stilt env
type: project
originSessionId: a6d0896f-1841-49f6-a2b9-837c74cd984b
---
Changes discovered when migrating from the PyPI 1.2 wheel to the local 1.3 submodule:

- **`update_assets` removed** from `mjlab.utils.os`. `MjSpec.from_file()` now loads mesh assets automatically from the filesystem (MuJoCo 3.7+). Remove any manual `spec.assets = ...` population.
- **`foot_swing_height` reward** no longer has `asset_cfg` in its params. It uses `sensor_name` (contact sensor subtree) + `height_sensor_name`. Only `foot_clearance` and `foot_slip` take `asset_cfg.site_names`.
- **`foot_height` critic obs** now uses `sensor_name: "foot_height_scan"` instead of `asset_cfg`. To redirect it to different foot sites, rewire the `foot_height_scan` sensor's `frame` to `ObjRef` tuples pointing at the desired sites.
- **`MjlabViserScene`** replaces `ViserMujocoScene` in viewer/viser/scene.py.

**How to apply:** When updating any env that customises foot rewards or observations, check whether it still uses the old asset_cfg pattern and switch to sensor-frame rewiring if so.
