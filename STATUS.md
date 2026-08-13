# Stilt Locomotion — Current Status
**Last updated: 2026-08-13**

---

## ⚠️ Runs 6 and 7 are void — the ankles are never welded

Runs 6 and 7 trained a **25-DoF** policy for a robot whose ankle joints had been
deleted, on the reading that the shank brace rigidly clamps the calf. **That robot
does not exist.** The hardware is always the stock **29-DoF** G1 with its own
feet; the stilts bolt on and come off, and nothing is ever done to the real ankle
motors to hold them straight.

Every checkpoint before Run 8 is therefore invalid, and so are the load figures in
the structural report — they were sampled from the Run 7 policy. The report's
geometry and material sections came from the CAD and still stand; regenerate the
loads with `scripts/analyse_stilt_loads.py` once Run 8 has a checkpoint.

## Run 8 — one policy, stilts on and off

The task Run 8 trains is deliberately harder than anything before it: **a single
policy that walks with the stilts fitted and with them removed**, without being
told which. Each episode draws a morphology at 50/50, and the policy infers it
from 5 frames of observation history.

| | stilts ON | stilts OFF |
|---|---|---|
| Ground contact | 16 stilt capsules | the robot's own 14 foot capsules |
| Stilt mass | 2.8 kg/side, curriculum 0.9–7.6 kg | ×0.001 (removed) |
| Ankle | sprung 150–2000 Nm/rad by the brace | free, PD-controlled |
| Pelvis spawn | 1.1977 m | 0.7902 m |
| `torso_too_low` | 0.65 m | 0.45 m |
| Standing pose | shank vertical, `hip_pitch = −knee = −0.10`, ankle 0 | **the same** |

**One shared pose, not two.** The action offset and the `pose` reward both key on
the keyframe, so a zero action commands it. On stilts that pose is forced — the
post is clamped parallel to the shank, so the stilt only stands up at
`hip_pitch = −knee` with the ankle at the brace's neutral angle. Giving the bare
robot its own stock crouch instead makes the neutral action a falling pose in the
fitted envs; a smoke run with split poses collapsed from 35 to 16 mean episode
length while the return rose, which is a policy learning that ending the episode
is cheaper than standing.

The brace is modelled as **ankle joint stiffness applied at reset**, not as a
weld. A bolted clamp onto a rigid shank is close to rigid, so the physics is much
the same, but the DoF stays where the hardware has one and it disappears cleanly
when the stilts come off. The stiffness is randomised wide because the real clamp
has never been measured — that is the largest open modelling risk in Run 8.

**Reading the curves.** The aggregate metrics cannot distinguish "walks on stilts,
falls over without them" from "mediocre at both", so
`envs/stilt_g1/metrics.py` logs masked per-mode versions. Divide by
`stilts_fitted_fraction` (or `1 − fraction`) for the conditional mean:

- `Episode_Metrics/vel_error_stilts_{on,off}`
- `Episode_Metrics/upright_stilts_{on,off}`

**Two bugs the pre-flight caught, both worth remembering:**

- `CollisionCfg.disable_other_geoms` defaults to `True`, so listing only the
  stilt capsules in `geom_names_expr` disabled the robot's own feet. The bare
  envs were free-falling through the floor while still logging as upright — a
  falling torso is perfectly vertical. Fixed by enabling both sets; pinned by
  `test_both_contact_sets_can_actually_collide`.
- Calling `sim.expand_model_fields()` from inside a reset event recreates the
  CUDA graph *on every reset*. Invisible on CPU, ruinous on the H100. The
  supported path is the `@requires_model_fields` decorator, which expands once
  at construction.

**Pre-flight, all green (2026-08-13):**

- `uv run pytest tests/ -q` — 53 passed
- `uv run python scripts/check_height_dr.py` — both morphologies spawn resting on
  the floor across the full ±50 mm telescope range
- `uv run python scripts/solve_spawn_height.py` — both spawn heights solved, not
  hand-tuned
- Both morphologies stand passively at zero action for 30 steps — pelvis holds
  at 1.195 m and 0.781 m, and each mode's contact set is the one reporting
- CPU smoke run at 128 envs, benchmarked against a stock-G1 control at the same
  scale (see below)

**How to read a small smoke run — this cost an hour, so it is written down.**
At 128 envs the stilt config climbs to ~64 mean episode length by iteration 4,
holds to ~25, then declines while the return rises. That is the textbook "dying
is cheaper than trying" shape and it looks like a broken reward.

It isn't. **Stock G1 — mjlab's own task, none of this project's code — does the
same thing at 128 envs, and worse:** it peaks at 63 and is down to 9.9 by
iteration 33, against 29 at iteration 40 for the stilt config. With 128×24 =
3072 samples per update, PPO simply falls into the degenerate optimum. At 4096
envs the identical reward structure took Run 7 to 1000/1000 with zero falls.

| iteration | stock G1 control | Run 8 config |
|---|---|---|
| peak | 63 | 64 |
| 33 | 9.9 | ~47 |
| 40 | — | 29 |

So judge a smoke run against the stock control at the same env count, never
against a monotonic ideal.

Design and plan: `docs/superpowers/specs/2026-08-07-new-stilt-design.md`,
`docs/superpowers/plans/2026-08-07-new-stilt-design.md` — **both carry a
superseding banner on the ankle question**; everything else in them still holds.

---

## Run 6 result (2026-08-08_10-46-51) — VOID, kept for the lessons

> **Void:** trained with the ankle joints deleted (25-DoF). Do not deploy this
> checkpoint. The reward and curriculum findings below carried over into Run 8 and
> are why it is kept.

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

## Run 7 result (run7-airtime-cappedcmd) — VOID, but its fixes carried

> **Void:** same 25-DoF model as Run 6. Do not deploy. The two changes it tested
> both worked and are still in `env_cfgs.py` for Run 8.

Run 6 plus `air_time` at weight 0.5 (`command_threshold` 0.3) and a command
curriculum capped at 0.8 m/s instead of ramping to 2.0. Both landed:

| | Run 6 | Run 7 |
|---|---|---|
| Mean episode length | 983.2 | **1000 / 1000** |
| `fell_over` | 0.00 | 0.00 |
| Achieved vx | ~0.35, frozen above 0.7 | 0.4–0.6, saturating ~0.52 |
| Yaw | ~0.005 rad/s (none) | turns at a 0.6 command |
| `air_time` | — | 0.139 s |

The speed freeze was gone and yaw appeared, which is why both changes were kept.
Structural loads were sampled from this checkpoint, so the published report needs
regenerating against Run 8.

---

## Deployment Infrastructure (`deploy/`)

Hardware deployment config is in place, but `deploy.yaml` itself is stale — see
the banner in that file. Regenerate it from the Run 8 ONNX metadata.

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
- **Stock 29-DoF G1**: all four ankle joints present and actuated, all 14 of the
  robot's own foot capsules live
- Stilts attached as a 5-body jointless tree under each `*_ankle_roll_link`, from
  the telescoping shank-clamped CAD — 2.8 kg per side, 407.5 mm ground-to-mount,
  CAD-derived inertia from `scripts/build_stilt_meshes.py` (never hand-tuned)
- 16 stilt contact capsules (8 per side), bottom at −0.4425 m in the ankle frame;
  the robot's own foot capsules bottom out at −0.035 m
- `foot_capsule` default class explicitly sets `friction="1.0 0.005 0.0001" condim="3"`
  to guarantee ground contact regardless of CollisionCfg
- Stilt tip sites `left_stilt_tip` / `right_stilt_tip` at `pos="0.04 0 -0.4425"`;
  the reset event slides them up to the sole when the stilts come off

### Stilt Environment (`envs/stilt_g1/`)
| File | Purpose |
|---|---|
| `env_cfgs.py` | Env config — overrides sites, geom names, reward targets, terminations, DR events, curricula, metrics |
| `curriculums.py` | `stilt_mass_curriculum`, `stilt_height_curriculum`, `stilt_termination_curriculum` |
| `events.py` | `reset_stilts_fitted` (the whole on/off draw) and `reset_stilt_spawn_height` |
| `terminations.py` | `root_height_below_minimum` with a floor per morphology |
| `metrics.py` | Masked per-morphology tracking error and uprightness |
| `loads.py` | Section loads from the contact sensor, for the viewer and the statics tests |
| `stilt_robot.py` | Robot config — local MJCF path, both standing poses, CollisionCfg |
| `rl_cfg.py` | PPO hyperparameters (inherited from stock G1) |
| `__init__.py` | Registers `Mjlab-Velocity-Flat-Stilt-G1` + viewer GUI (mass sliders, loads, torque monitor) |

**Key environment settings vs stock G1:**
- Actor observation history 5 frames (495 inputs) — the policy's only way to tell
  the two morphologies apart
- `foot_height_scan` sensor rewired to stilt tip sites (drives foot_height obs + height rewards)
- `foot_clearance` / `foot_slip` use stilt tip sites via `asset_cfg.site_names`
- `foot_clearance` and `foot_swing_height` target height → 0.10 m
- `air_time` weight → 0.5, `command_threshold` → 0.3 (Run 7's fix for the frozen gait)
- Command curriculum capped at 0.8 m/s forward rather than ramping to 2.0
- `torso_too_low` → 0.45 m bare, 0.65 m on stilts
- Friction randomisation targets **both** contact sets
- Per-capsule `stilt_contact` ContactSensor feeding the viewer load panel

### Stilt Mass Curriculum
Four stages, progressively widening the stilt mass range. Uses `dr.pseudo_inertia`
(via `alpha_range`) so mass and inertia scale consistently — physically correct for
a density change. **Baseline stilt mass is 2.8 kg per stilt.**

| Iter | Step | alpha range | Approx mass range | Purpose |
|---|---|---|---|---|
| 0 | 0 | `(0.0, 0.0)` | fixed 2.8 kg | Solid baseline |
| 500 | 12000 | `(-0.2, 0.2)` | 1.9–4.2 kg | Introduce variability early |
| 1000 | 24000 | `(-0.4, 0.4)` | 1.3–6.2 kg | Widen to stress testing levels |
| 2000 | 48000 | `(-0.55, 0.5)` | 0.9–7.6 kg | Maximum stress range |

`alpha` is a log-scale multiplier: mass = 2.8 × e^(2α). The curriculum logs
`Curriculum/stilt_mass/stilt_mass_min_kg` and `stilt_mass_max_kg` to W&B. Note it
runs **before** the on/off draw, which then scales whatever it sampled — so in
stilts-off envs the sampled mass is multiplied away, as it should be.

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
| **stilt run 5** | **2026-04-27_14-48-06** | **6000** | complete — walked, but on the OLD box stilt | mjlab v1.3, stilt mass curriculum, 1.5 kg baseline |
| stilt run 6 | 2026-08-08_10-46-51 | 6000 | **void** | new stilt, but 25-DoF welded ankles — balanced, could not turn |
| stilt run 7 | run7-airtime-cappedcmd | 6000 | **void** | same 25-DoF error; air_time + capped commands both worked |
| **stilt run 8** | pending | 6000 | queued | first run on the real 29-DoF robot; stilts on AND off |

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
