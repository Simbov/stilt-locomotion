---
name: Viewer extension pattern
description: How to add task-specific GUI controls to the ViserPlayViewer Controls tab
type: project
originSessionId: a6d0896f-1841-49f6-a2b9-837c74cd984b
---
Three mjlab files were extended to support a `play_viewer_setup_fn` hook:

1. `mjlab/src/mjlab/viewer/viser/viewer.py` — `ViserPlayViewer.__init__` accepts `extra_gui_fn: Callable[[viser.ViserServer, EnvProtocol], None] | None`, called at end of Controls tab in `setup()`.
2. `mjlab/src/mjlab/tasks/registry.py` — `register_mjlab_task` accepts `play_viewer_setup_fn`, stored in `_TaskCfg`, loaded via `load_play_viewer_setup_fn(task_id)`.
3. `mjlab/src/mjlab/scripts/play.py` — loads the fn from registry and passes as `extra_gui_fn` to `ViserPlayViewer`.

The stilt mass slider in `envs/stilt_g1/__init__.py` is the reference implementation. It calls `dr.pseudo_inertia()` directly for immediate effect and also updates `term_cfg.params["alpha_range"]` so future resets use the same value.

**How to apply:** To add viewer controls for any new task, define `(server, env) -> None` and pass as `play_viewer_setup_fn` in `register_mjlab_task`.
