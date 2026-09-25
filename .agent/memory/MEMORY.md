# Memory Index

- [Project setup](project_setup.md) — uv editable install of local mjlab/, pyproject.toml sources, venv verification
- [Viewer extension pattern](viewer_extension.md) — How task-specific GUI is added to ViserPlayViewer (extra_gui_fn / play_viewer_setup_fn)
- [mjlab 1.3 API changes](mjlab_api_changes.md) — Breaking changes from 1.2→1.3 relevant to this project
- [mjlab upgrade 1.3→1.5](mjlab_upgrade_1_5.md) — 2026-06-30 upgrade to mjlab 1.5 / mujoco 3.10; breaking changes + memory-leak watch-item
- [Repo privacy](repo_privacy.md) — Repo is PUBLIC; keep HPC login / personal files out; no history rewrites
- [User preferences](user_preferences.md) — Working style and preferences observed this session
- [Stilt mass DR investigation](stilt_mass_dr_investigation.md) — Mass write confirmed landing but torques unresponsive; recompute_constants fix; CUDA graph hypothesis; open question for next session
- [Run 5 walking result](run5_walking_result.md) — Run 5 (2026-04-27_14-48-06) converged; robot walks, Phase 1 done, mass robust to 6kg/stilt; how to read tfevents metrics
- [Hardware deploy without sudo](hardware_deploy_no_sudo.md) — build unitree_rl_mjlab against ~/unitree_sdk2; never install into /usr/local on a lab robot
- [Hardware forward drift](hardware_forward_drift.md) — Run 8 walks forward at zero command; joystick and obs layout ruled out, base_lin_vel zero-fill suspected
- [Run 9 hardware result](run9_hardware_result.md) — 2026-09-14: Run 9 walks/turns stably on the G1, slight zero-command creep; sim creeps ~3 cm/s too; next visit ship --runtime for telemetry
- [Stilts first hardware test](stilts_first_hardware_test.md) — 2026-09-17 new G1, stilts on: diverged; real ankle moves ±0.6 rad vs ~0 in sim brace
- [Stilts second hardware test](stilts_second_hardware_test.md) — 2026-09-25: stilts-on 4 s run bounded (no divergence) but ankle still sags 0.1–0.2 rad, 30 cm left drift
