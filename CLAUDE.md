# Stilt Locomotion — Development Guide

## Project layout

```
stilt-locomotion/
├── mjlab/          # mjlab v1.5 — git submodule (reference/dev), real dep is PyPI pin
├── envs/
│   └── stilt_g1/   # Stilt G1 task (env config, curriculum, rewards, viewer GUI)
├── assets/mjcf/g1/ # Modified G1 MJCF with stilt bodies and tip sites
├── scripts/
│   ├── visualise.command      # macOS double-click launcher (1 env, viser viewer)
│   ├── play_stilt.py          # Play wrapper (registers stilt env then calls mjlab play)
│   ├── train_stilt.py         # Training entry point
│   ├── build_stilt_meshes.py  # CAD → segment meshes + MJCF inertials (rerun on CAD change)
│   ├── solve_spawn_height.py  # Solves STILT_SPAWN_HEIGHT for the keyframe pose
│   └── check_height_dr.py     # Gate check: randomised stilt heights spawn on the floor
├── tests/                     # pytest suite — run before committing model changes
├── deploy/
│   ├── README.md                       # Build + deploy instructions
│   └── config/g1_stilt/deploy.yaml     # Deployment config (obs, gains, action scale)
└── logs/rsl_rl/            # Training run outputs (checkpoints, W&B sync)
```

## Environment setup

The project uses **uv**. Setup is identical locally and on HPC:

```sh
uv sync      # install all deps including mjlab==1.5.0 from PyPI
uv run <cmd> # run anything inside the managed venv
```

## Modifying mjlab

**Keep mjlab changes minimal** — the submodule tracks stock v1.5.0 so upgrading
is a simple `cd mjlab && git checkout <new-tag>` plus bumping the PyPI pin in
`pyproject.toml`. Avoid patching mjlab source files;
extend behaviour from this project's code instead (see viewer pattern below).

If you do need to test a mjlab change locally, install it editably as a one-off:

```sh
uv pip install -e mjlab/ --no-deps
```

Run the mjlab checks before committing any mjlab changes:

```sh
cd mjlab && make check && make test-fast && cd ..
```

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

### Stilt Mass sliders

The **Controls → Stilt Mass** folder edits stilt mass live, per segment, at any
point during a run:

- **One slider per segment** (mount, brace, post outer, post inner, plate),
  range 0–`MASS_PLAY_MAX_KG` kg. Each applies immediately to both stilts across
  all envs by calling `dr.pseudo_inertia` directly, so mass and inertia scale
  consistently.
- **master ×** — scales all five segments together, for quickly probing overall
  stilt weight without touching the distribution.
- **Randomize on reset** — hands control back to the reset event over the full
  trained alpha range (0.9–7.6 kg per stilt), so each reset samples a mass.
- **sim mass readback** — shows, per segment, the value written to the Warp
  model and the matching `cinert[9]` (composite inertia mass component), plus
  the per-stilt total. The two columns must match, confirming the write landed
  and propagated through `set_const_0 → smooth.com_pos → _cinert`.

To change the slider ceiling, edit `MASS_PLAY_MAX_KG` at the top of
[`envs/stilt_g1/__init__.py`](envs/stilt_g1/__init__.py).

### Stilt Loads and Ground Pressure

Two folders show where the hardware is actually loaded, at 10 Hz:

- **Stilt Loads** — axial, shear and bending at each of the five sections, per
  stilt, computed in [`envs/stilt_g1/loads.py`](envs/stilt_g1/loads.py).
- **Ground Pressure** — force in each of the 8 contact capsules per foot, so
  heel/toe and left/right distribution is directly visible.

Both read the `stilt_contact` `ContactSensor`. **mujoco_warp does not surface
per-contact geom ids through `get_data_into`** (they come back as zeros), so the
sensor is the only working source while the warp sim runs; the CPU-MuJoCo path
in the same module exists for the tests, which verify the statics closes against
a known static stand.

**The brace row is inertial load only — not the clamp reaction.** With the ankle
welded, shank and foot are one rigid body, so the split of the interface load
between the sole bolts and the shank clamp is statically indeterminate. The sim
gives the *total* wrench; the split needs a hand calc or FEA.

Bar scales live in `SECTION_LIMIT` and `CAPSULE_LIMIT_N` at the top of
`envs/stilt_g1/__init__.py`. They are display scales, not engineering limits.

### Joint Torque Monitor

A live **Joint Torques** folder shows two metrics at 10 Hz for hip pitch and
knee joints (the ankles no longer exist as joints):

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

## Stilt G1 environment (mjlab API notes — current as of v1.5)

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

## Stilt hardware model (2026-08-07 onwards)

The stilt is the telescoping shank-clamped design, 407.5 mm ground-to-mount,
2.8 kg per side. Five rigid segments per stilt, all **jointless** (zero added
DoFs), generated by [`scripts/build_stilt_meshes.py`](scripts/build_stilt_meshes.py)
— rerun that script whenever the CAD changes, and paste its `<inertial>` output
into `g1.xml`. Never hand-tune those numbers.

```
ankle_roll_link
└── *_stilt_mount        0.636 kg   mount plate
    ├── *_stilt_brace    0.999 kg   shank brace (clamps the calf)
    └── *_stilt_post_outer 0.313 kg
        └── *_stilt_post_inner 0.288 kg   ⟵ dr.body_pos randomises telescope height
            └── *_stilt_plate  0.565 kg   ground plate + 8 contact capsules + tip site
```

**The ankle joints are deleted.** The brace rigidly clamps the shank, so ankle
pitch and roll cannot move. The `ankle_pitch_link` / `ankle_roll_link` *bodies*
remain (the `foot_swing_height` contact subtree and every geom/site path depend
on them), but the joints and `G1_ACTUATOR_ANKLE` are gone: **action space
29 → 25**. Consequence: shank orientation is rigidly `hip_pitch + knee`, so the
stilt is upright only when `hip_pitch = −knee`, and there is no ankle balance
strategy at all. Expect training to be markedly harder than Run 5.

## Stilt mass and height curricula

Defined in `envs/stilt_g1/curriculums.py` and wired in `env_cfgs.py`.
Mass uses `dr.pseudo_inertia`, which scales mass and inertia consistently via
the pseudo-inertia matrix (Rucker & Wensing 2022). Alpha is a log-scale
multiplier applied to all ten stilt bodies: `mass = 2.8 × e^(2α)` per stilt.

| Training iter | α range       | Mass range   | Height offset | Stilt height |
|---------------|---------------|--------------|---------------|--------------|
| 0             | (0.0, 0.0)    | 2.8 kg fixed | (0, 0)        | 407.5 mm     |
| 500           | (−0.2, 0.2)   | 1.9–4.2 kg   | —             | —            |
| 750           | —             | —            | ±20 mm        | 387–427 mm   |
| 1 000         | (−0.4, 0.4)   | 1.3–6.2 kg   | —             | —            |
| 1 500         | —             | —            | ±50 mm        | 357–457 mm   |
| 2 000         | (−0.55, 0.5)  | 0.9–7.6 kg   | —             | —            |

Height DR moves `*_stilt_post_inner` with `shared_random=True` (both stilts are
set to the same length on real hardware). **`scene.reset()` applies the keyframe
*before* reset events run**, so the spawn height cannot see the sampled length;
`reset_stilt_spawn_height` in `envs/stilt_g1/events.py` runs after the height
term and corrects the root pose. That ordering is load-bearing — it depends on
`cfg.events` dict insertion order. Verify with:

```sh
uv run python scripts/check_height_dr.py
```

Do not widen the height curriculum past ±50 mm until the minimum safe tube
overlap is confirmed (assumed 80 mm, which would allow up to 522 mm).

## Tests

```sh
uv run pytest tests/ -q
```

Covers the mesh/inertia generator, the MJCF topology and geometry, the robot
config, the height-DR spawn correction, the env wiring, and the section-load
statics. Run these before any commit touching the stilt model — several of them
encode invariants that are easy to break silently (contact geom names, tip site
height, monotonic axial load, event ordering).

## Training

```sh
uv run python scripts/train_stilt.py
```

Checkpoints land in `logs/rsl_rl/stilt_g1_velocity/<timestamp>/`.

**The Run 5 checkpoint (2026-04-27_14-48-06) is invalid** for the current model:
different action space, mass, inertia, height and stance. A fresh run is
required, and `deploy/config/g1_stilt/deploy.yaml` must be regenerated from the
new ONNX metadata before any hardware deployment.

## Code style

Follow `mjlab/CLAUDE.md` for commit, PR, and style conventions. The short
version: `make check` in `mjlab/` must pass before any commit touching mjlab
source; run `uv run ruff format && uv run ruff check --fix` for project-level
Python files.
