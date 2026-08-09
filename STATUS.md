# Stilt Locomotion — Current Status
**Last updated: 2026-08-07**

---

## ⚠️ New stilt design landed (2026-08-07) — Run 5 superseded

The stilt was replaced with the real telescoping shank-clamped hardware
(`Assembled 40.7cm.STL`): 407.5 mm tall, **2.8 kg per side**, five rigid
segments with CAD-derived inertia, and a brace that **clamps the shank so the
ankle joints are deleted** (action space **29 → 25**). Telescope height is now
domain-randomised, and the viewer gained per-segment mass sliders plus section
load and ground-pressure readouts.

**Consequences:**
- **The Run 5 checkpoint is invalid.** Different action space, mass, inertia,
  height and stance. A fresh training run is required.
- **`deploy/config/g1_stilt/deploy.yaml` is superseded** and must be regenerated
  from the new ONNX metadata. Its arrays are still 29-DOF.
- **Expect slower convergence than Run 5.** With the ankle welded there is no
  ankle balance strategy at all — lateral balance is hip-roll only, and the
  brace puts 1.0 kg above the ankle. This is genuine peg-stilt walking.

**Trainability verified (2026-08-07).** A 64-env CPU smoke run reaches mean
episode length ~80 and holds, against ~53 for a matched stock-G1 control. An
earlier apparent training collapse was traced to a stale-read bug in
`reset_stilt_spawn_height`, not to the task — see spec §14. No reward shaping
is used. **Ready for an HPC run.**

Design and plan: `docs/superpowers/specs/2026-08-07-new-stilt-design.md`,
`docs/superpowers/plans/2026-08-07-new-stilt-design.md`.

---

## Run 6 result (2026-08-08_10-46-51) — balances, walks slowly, cannot turn

First run on the new telescoping stilt. 6000 iterations, 4096 envs, ~2h22m.
ONNX exported. **Balance is solved; velocity tracking is not.**

| | Run 5 (old stilt, free ankles) | Run 6 (new stilt, welded ankles) |
|---|---|---|
| Mean episode length | 984.6 | **983.2** (of 1000) |
| Mean reward | 33.4 | 32.1 |
| `fell_over` | 0.00 | **0.00** |
| `error_vel_xy` | 0.84 | 1.80 |

All curricula completed: mass 0.93–7.61 kg, height 358–458 mm, terminations
tightened to the stock 1.222 rad / 0.65 m, commands widened to ±2.0 m/s.

**What it does**, measured by driving the checkpoint with fixed commands:

| commanded vx | achieved vx |
|---|---|
| 0.25 | 0.06 |
| 0.40 | 0.33 |
| 0.50 | 0.36 |
| 0.60 | 0.36 |
| 0.75 | 0.05 |
| 1.00 | 0.00 |

- Stands still perfectly on zero command, and never falls.
- Walks forward at ~0.35 m/s for commands in 0.4–0.6, and strafes at ~0.38.
- **Freezes above ~0.7 m/s** — a discrete cutoff, not a gradual degradation.
- **Cannot turn**: yaw rate is ~0.005 rad/s for commands of 0.2, 0.35 and 0.7.
- Backward is weak (0.12 achieved for 0.5 commanded).

The aggregate `error_vel_xy` of 1.80 is worse than standing perfectly still
(1.09) purely because the command curriculum reached ±2.0 m/s while the policy
only covers a narrow low-speed band and freezes elsewhere.

**Next levers, in order:**

1. **Enable `air_time`** — still at weight 0.0 in `env_cfgs.py` ("disabled until
   the robot can walk"). It can now balance, so this is the obvious next step;
   nothing currently rewards taking a proper step.
2. **Cap the command curriculum near what is achievable** (~0.6–0.8 m/s) instead
   of 2.0. Training against commands it cannot meet likely drives the freeze.
3. **Yaw needs separate attention.** Run 5 tracked yaw poorly too, so this is a
   pre-existing weakness rather than a stilt regression, but with the ankle
   welded there is even less authority to turn with.

---

## Deployment Infrastructure (`deploy/`)

Hardware deployment config is ready for when Run 5 produces a good checkpoint.

| File | Purpose |
|---|---|
| `deploy/config/g1_stilt/deploy.yaml` | Full deployment config — obs space, PD gains, action scale, all from ONNX metadata |
| `deploy/README.md` | Step-by-step: sync ONNX from HPC, set up unitree_rl_mjlab, build, run on robot |

**Runtime:** [unitree_rl_mjlab](https://github.com/unitreerobotics/unitree_rl_mjlab) C++ binary (`g1_ctrl`) using ONNX Runtime 1.22.0 at 50 Hz. Launched via SSH — not app-based.

**`base_lin_vel` handling:** zeroed (policy still responds to joystick velocity commands normally; `base_lin_vel` is measured feedback, not commanded speed).

**To deploy a new checkpoint:** sync ONNX from HPC → copy to robot → run `./g1_ctrl`. No rebuild needed.

---

## What Has Been Built

### Robot Model (`assets/mjcf/g1/g1.xml`)
- Local copy of the G1 MJCF, safe to modify (mjlab original untouched in `.venv`)
- Stilts attached as rigid bodies to both `left_ankle_roll_link` and `right_ankle_roll_link`
- Stilt STL: 220×80×400 mm physical stilt; loaded with `scale="0.001 0.001 0.001"` and `refpos="70 40 435"` (pre-scale mm units) so the attachment face sits flush at the ankle origin
- Original G1 foot capsules commented out, replaced by 8 collision capsules per stilt (4 left block + 4 right block, modelled after G1's foot capsule style)
- Capsule positions: z = −0.425 m from ankle, radius = 0.01 m → bottom at −0.435 m
- `foot_capsule` default class explicitly sets `friction="1.0 0.005 0.0001" condim="3"` to guarantee ground contact regardless of CollisionCfg
- Pelvis spawn height in MJCF: 1.228 m (standing straight; training uses 1.16 m via keyframe)
- Stilt tip sites: `left_stilt_tip`, `right_stilt_tip` at `pos="0.04 0 -0.435"`
- Stilt inertial properties: **`mass="1.5"`**, **`diaginertia="0.024 0.024 0.003"`** (increased from 0.5 kg on 2026-04-21), COM at `pos="0.04 0 -0.2"`

### Stilt Environment (`envs/stilt_g1/`)
| File | Purpose |
|---|---|
| `env_cfgs.py` | Env config — overrides sites, geom names, reward targets, terminations, DR events, curricula |
| `curriculums.py` | Custom `stilt_mass_curriculum` class — widens stilt mass range over training |
| `stilt_robot.py` | Robot config — local MJCF path, spawn keyframe, CollisionCfg |
| `rl_cfg.py` | PPO hyperparameters (inherited from stock G1) |
| `__init__.py` | Registers `Mjlab-Velocity-Flat-Stilt-G1` + viewer GUI (mass slider, torque monitor) |

**Key environment settings vs stock G1:**
- `foot_height_scan` sensor rewired to stilt tip sites (drives foot_height obs + height rewards)
- `foot_clearance` / `foot_slip` use stilt tip sites via `asset_cfg.site_names`
- `foot_clearance` target height → 0.10 m
- `foot_swing_height` target height → 0.10 m (uses contact-sensor subtree, no site override needed)
- `air_time` weight → 0.0 (disabled until robot can walk)
- `torso_too_low` threshold → 0.65 m
- Friction randomisation targets stilt capsule geoms
- **Stilt mass curriculum active** — see curriculum section below

### Stilt Mass Curriculum
A four-stage curriculum progressively widens the stilt mass range during training.
Uses `dr.pseudo_inertia` (via `alpha_range`) so mass and inertia scale consistently —
physically correct for a density change. **Baseline stilt mass is 1.5 kg per stilt.**

| Iter | Step | alpha range | Approx mass range | Purpose |
|---|---|---|---|---|
| 0 | 0 | `(0.0, 0.0)` | fixed 1.5 kg | Solid baseline for heavy design |
| 500 | 12000 | `(-0.2, 0.2)` | 1.0–2.2 kg | Introduce variability early |
| 1000 | 24000 | `(-0.4, 0.4)` | 0.67–3.3 kg | Widen to stress testing levels |
| 2000 | 48000 | `(-0.55, 0.69)` | 0.5–6.0 kg | Maximum stress range (up to 4× baseline) |

`alpha` is a log-scale multiplier: mass = 1.5 × e^(2α). The curriculum logs
`Curriculum/stilt_mass/stilt_mass_min_kg` and `stilt_mass_max_kg` to W&B.

### Training Pipeline
- `scripts/train_stilt.py` — registers env and calls mjlab's `train` entry point
- `scripts/train_stilt.pbs` — PBS job: 1 node, 8 CPUs, 1×H100, 32 GB RAM, 8 hr walltime, 6000 max iterations
- `scripts/visualise.command` — double-click in Finder → file picker → viser browser viewer
- `scripts/play_stilt.py` — local viewer with mass slider + joint torque monitor GUI

### Package Versions
- **mjlab v1.5.3** from PyPI (local `mjlab/` submodule pinned to v1.5.3 for reference/dev)
- **mujoco 3.10.0** + **mujoco-warp 3.10.0.3** — driven by mjlab 1.5 (clean PyPI pins,
  no project-level pin). rsl-rl-lib 5.4.0, warp-lang 1.14.0.
- All deps managed via `uv` / `uv.lock` — run `uv sync` to install
- **⚠️ Memory-leak watch:** mujoco was previously pinned `<3.8` to dodge a 3.8.0
  ~670 MB/iter leak. 1.5 jumps to **3.10**, bypassing 3.8 — re-verify GPU memory
  over a long HPC run before trusting it (`nvidia-smi` / W&B system metrics).

---

## Git Tags (Revert Points)

| Tag | Commit | Description |
|---|---|---|
| `v0.2-mjlab-1.3` | `907c8ea` | Clean mjlab v1.3 upgrade, no other changes |
| `v0.3-stilt-mass-curriculum` | `b549513` | Stilt mass curriculum implemented and bug-fixed |

To revert to any tag: `git checkout <tag-name>`

---

## Training Runs

| Run | Date | Iters | Status | Notes |
|---|---|---|---|---|
| stilt run 1 | 2026-03-27_12-25-11 | short | abandoned | wrong foot site names |
| stilt run 2 | 2026-03-27_12-43-32 | ~2000 | slipping | stilt slipping, friction not applied |
| stilt run 3 | 2026-03-27_16-51-02 | short | abandoned | early test |
| stilt run 4 | 2026-03-27_20-32-07 | 1499 | **broken** | 13-step episodes — torso_too_low threshold too high (0.85 m) |
| stilt run 5 (false starts) | 2026-04-27_13-16→14-38 | 0 | aborted | several restarts, only `model_0` written |
| **stilt run 5** | **2026-04-27_14-48-06** | **6000** | **✅ complete — robot walks** | mjlab v1.3, stilt mass curriculum, 1.5 kg baseline |

**Run 5 setup:**
- 4096 envs, H100, ~3700 steps/sec, 32 GB RAM
- Warp kernels cached after first iteration (no recompile overhead)
- W&B: https://wandb.ai/<wandb-entity>/stilt-locomotion/runs/<run-id>

**Run 5 results (from `events.out.tfevents` — verified 2026-07-23):**

The full 6000-iteration run **converged to a stable walking gait.** Phase 1 (get the
robot walking) is complete.

| Metric | Start | ~iter 1k | ~iter 3k | Final (6k) |
|---|---|---|---|---|
| `Train/mean_episode_length` | 12.9 | 921 | 998 | **985** |
| `Episode_Reward/track_linear_velocity` | 0.00 | 1.04 | 1.47 | **1.31** |
| `Episode_Termination/fell_over` | 0 | 0.54 | 0 | **0** |
| `Episode_Termination/time_out` | 4.2 | 3.9 | 4.0 | **4.75** |
| `Episode_Termination/torso_too_low` | 0 | 0.29 | 0.04 | 0.21 |
| `Metrics/slip_velocity_mean` | 0.52 | 0.24 | 0.16 | **0.21** |
| `Curriculum/stilt_mass/…_max_kg` | 1.5 | 3.3 | 6.0 | **6.0** |

- Episode length jumped from the 13-step collapse (run 4) to ~985 steps — episodes now
  end almost entirely by `time_out`, i.e. the robot survives the full episode.
- Velocity tracking blew past the >0.5-by-3k target (hit 1.47).
- **Stilt mass robustness (answers Phase 2):** the policy stayed stable across the full
  curriculum sweep to **0.5–6.0 kg per stilt** without falls — up to 4× the 1.5 kg baseline.
- ⚠️ `Train/mean_reward` peaked ~49 near iter 3k and settled to ~33 by the end (mild
  late-training decline as the mass range widened); episode length held at ~985, so the
  gait is stable, not collapsing. Candidate for a shorter/gentler final curriculum stage.
- Deployable checkpoint: `model_5999.pt` + `2026-04-27_14-48-06.onnx`.

---

## Collision & Contact (Verified)

Verified via Python test (`assets/mjcf/g1/` directory):
- All 16 stilt capsules: `contype=1 conaffinity=1 condim=3` ✓
- Non-stilt geoms disabled by `CollisionCfg.disable_other_geoms=True` (expected behaviour)
- Stilt tip z at spawn: +0.003 m above floor (3 mm gap — normal, robot settles on first step)

---

## HPC Access
```bash
ssh <hpc-user>@aquarius02.hpc.qut.edu.au
cd ~/stilt-locomotion
git pull
qsub scripts/train_stilt.pbs
qstat -u $USER
```

The PBS script handles everything automatically: installs uv if missing, runs `uv sync`
to build the venv from `uv.lock` (mjlab from PyPI, no submodule needed).

Sync logs to Mac:
```bash
rsync -avz <hpc-user>@aquarius02.hpc.qut.edu.au:~/stilt-locomotion/logs/ ~/Desktop/stilt-locomotion/logs/
```

W&B: https://wandb.ai/<wandb-entity>/stilt-locomotion
