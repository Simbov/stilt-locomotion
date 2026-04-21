# Stilt Locomotion — Development Guide

## Project layout

```
stilt-locomotion/
├── mjlab/          # mjlab v1.3 — git submodule, installed editably
├── envs/
│   └── stilt_g1/   # Stilt G1 task (env config, curriculum, rewards, viewer GUI)
├── assets/mjcf/g1/ # Modified G1 MJCF with stilt bodies and tip sites
├── scripts/
│   ├── visualise.command   # macOS double-click launcher (1 env, viser viewer)
│   ├── play_stilt.py       # Play wrapper (registers stilt env then calls mjlab play)
│   └── train_stilt.py      # Training entry point
└── logs/rsl_rl/            # Training run outputs (checkpoints, W&B sync)
```

## Environment setup

The project uses **uv** with the local `mjlab/` submodule installed editably:

```sh
uv sync          # Install / update all deps (uses local mjlab/ — never PyPI mjlab)
uv run <cmd>     # Run anything inside the managed venv
```

`mjlab/` is pinned via `[tool.uv.sources]` in `pyproject.toml`, so `uv sync`
always installs the local source. **Never `pip install mjlab`** — it would
shadow the local source and changes to `mjlab/` would stop taking effect.

Verify the correct mjlab is active:

```sh
uv run python -c "import mjlab; print(mjlab.__file__)"
# Should print: .../stilt-locomotion/mjlab/src/mjlab/__init__.py
```

## Modifying mjlab

Edits to `mjlab/src/mjlab/` take effect immediately (editable install). Run
the mjlab checks before committing:

```sh
cd mjlab && make check && make test-fast && cd ..
```

**Keep mjlab changes minimal** — the submodule tracks stock v1.3.0 so upgrading
to v1.4 is a simple `cd mjlab && git pull`. Avoid patching mjlab source files;
extend behaviour from this project's code instead (see viewer pattern below).

### Viewer GUI pattern

Task-specific viewer controls are injected by subclassing `ViserPlayViewer` in
[`scripts/play_stilt.py`](scripts/play_stilt.py) and monkey-patching it into
`mjlab.scripts.play` before `main()` is called. No mjlab source files are
modified. The GUI function itself lives in
[`envs/stilt_g1/__init__.py`](envs/stilt_g1/__init__.py) as
`_stilt_mass_play_gui(server, env)`.

To add viewer controls for a new task, follow the same pattern:
1. Define a `(server, env) -> None` function in your env's `__init__.py`
2. Subclass `ViserPlayViewer` in your play script, call `super().setup()` then
   your function
3. Monkey-patch before calling `mjlab.scripts.play.main()`

## Running the viewer

Double-click `scripts/visualise.command` in Finder, or from a terminal:

```sh
source .venv/bin/activate
python scripts/play_stilt.py Mjlab-Velocity-Flat-Stilt-G1 \
    --checkpoint-file logs/rsl_rl/stilt_g1_velocity/<run>/model_<step>.pt \
    --num-envs 1 --viewer viser
```

The viewer opens at `http://localhost:8080`.

### Stilt Mass slider

The **Controls → Stilt Mass** folder lets you adjust per-stilt mass live:

- **Mass (kg) slider** — range 0–`MASS_PLAY_MAX_KG` kg, applies immediately to
  all envs by calling `dr.pseudo_inertia` directly (mass + inertia scale
  consistently). Also updates the event params so subsequent resets use the
  same value.
- **Randomize on reset** — when checked, uses the full trained alpha range
  (0.5–6.0 kg) so each reset samples a different mass.
- **sim mass readback** — label below the slider shows the actual value
  written to the Warp model and the matching `cinert[9]` (composite inertia
  mass component). Both should match, confirming the write landed and
  propagated through `set_const_0 → smooth.com_pos → _cinert`.

To change the slider ceiling, edit `MASS_PLAY_MAX_KG` at the top of
[`envs/stilt_g1/__init__.py`](envs/stilt_g1/__init__.py).

### Joint Torque Monitor

A live **Joint Torques** folder shows two metrics at 10 Hz for hip pitch,
knee, and ankle pitch joints:

**`qfrc_actuator` (PD output, clamped at `forcerange`):**
```
L hip pitch   +23.4Nm   27%  █████░░░░░░░░░░░░░░░
...
```
This is the force the actuator is applying. It can be masked by policy
adaptation: at 20 kg out-of-distribution the policy takes tiny steps
(small swing acceleration → small tracking error → small PD output).

**Gravity load (`qfrc_bias`):**
```
L hip pitch   -31.2Nm   35%  ███████░░░░░░░░░░░░░
...
```
This is the gravity + Coriolis force projected into joint space — computed
by the RNE algorithm from `cinert`, which reads `body_mass` every step.
`qfrc_bias` changes immediately and proportionally when the slider moves,
regardless of policy adaptation. Use this to verify the mass DR is active.
**Note (2026-04-21)**: This now updates even when the viewer is paused.

**Why `qfrc_actuator` alone is insufficient:** The mass change IS reaching
the physics every step (step → forward → fwd_position → com_pos → `_cinert`
reads `m.body_mass` directly). But `qfrc_actuator = kp*(q_des-q)` only
reflects tracking error, not gravitational load. A stiff PD controller with
a conservative policy can maintain similar tracking errors at 0.5 kg and
20 kg. `qfrc_bias` bypasses the policy entirely.

**Note on CUDA graphs:** In-place writes to existing Warp arrays (as done by
`pseudo_inertia`) ARE visible to captured CUDA graphs — the graph holds
pointers to the same GPU buffers, not copies. Only array *replacement*
(new allocation via `expand_model_fields`) would invalidate the graph.

**Note on `pseudo_inertia` in viewer context:** When called from the GUI
(not the event manager), `recompute_constants` must be called manually
afterward — the `@requires_model_fields` decorator only annotates the
function; it does not auto-trigger recomputation. This is already done in
`_apply()` in `__init__.py`.

## Stilt G1 environment (mjlab 1.3 API notes)

Key differences from the base G1 env that matter when updating:

- **`foot_height_scan` sensor frame** must be rewired to `left_stilt_tip` /
  `right_stilt_tip` sites. This drives the `foot_height` critic obs and all
  height-based rewards (`foot_clearance`, `foot_swing_height`).
- **`foot_clearance` and `foot_slip`** use `asset_cfg.site_names` → set to stilt
  tip sites.
- **`foot_swing_height`** uses a contact-sensor subtree (ankle_roll_link) so
  needs no site override — only `target_height` is set.
- **`MjSpec.from_file()`** loads mesh assets automatically in MuJoCo 3.7+; the
  old `update_assets` helper no longer exists in mjlab 1.3.

## Stilt mass curriculum

Defined in `envs/stilt_g1/curriculums.py` and wired in `env_cfgs.py`.
Uses `dr.pseudo_inertia` which scales mass and inertia consistently via the
pseudo-inertia matrix (Rucker & Wensing 2022). The alpha parameter is a
log-scale multiplier: `mass = 1.5 × e^(2α)`.

| Training iter | α range       | Mass range    | Purpose |
|---------------|---------------|---------------|---------|
| 0             | (0.0, 0.0)    | 1.5 kg fixed  | Solid baseline |
| 500           | (−0.2, 0.2)   | 1.0–2.2 kg    | Early variability |
| 1 000         | (−0.4, 0.4)   | 0.67–3.3 kg   | Wide stress testing |
| 2 000         | (−0.55, 0.69) | 0.5–6.0 kg    | Maximum stress range |

## Training

```sh
uv run python scripts/train_stilt.py
```

Checkpoints land in `logs/rsl_rl/stilt_g1_velocity/<timestamp>/`.

## Code style

Follow `mjlab/CLAUDE.md` for commit, PR, and style conventions. The short
version: `make check` in `mjlab/` must pass before any commit touching mjlab
source; run `uv run ruff format && uv run ruff check --fix` for project-level
Python files.
