# Stilt Locomotion — Current Status
**Last updated: 2026-04-28**

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
- **mjlab v1.3.0** from PyPI (local `mjlab/` submodule is for reference/dev only)
- **mujoco 3.7.0** + **mujoco-warp 3.7.0.1** — pinned; 3.8.0 has a memory leak (~670 MB/iter)
- All deps managed via `uv` / `uv.lock` — run `uv sync` to install

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
| **stilt run 5** | 2026-04-27_13-28-41 | 6000 | **running** | mjlab v1.3, stilt mass curriculum, 1.5 kg baseline |

**Run 5 setup:**
- 4096 envs, H100, ~3700 steps/sec, 32 GB RAM
- Warp kernels cached after first iteration (no recompile overhead)
- W&B: https://wandb.ai/simbov04-qut/stilt-locomotion/runs/ke9bopwf

---

## Collision & Contact (Verified)

Verified via Python test (`assets/mjcf/g1/` directory):
- All 16 stilt capsules: `contype=1 conaffinity=1 condim=3` ✓
- Non-stilt geoms disabled by `CollisionCfg.disable_other_geoms=True` (expected behaviour)
- Stilt tip z at spawn: +0.003 m above floor (3 mm gap — normal, robot settles on first step)

---

## HPC Access
```bash
ssh n11298111@aquarius02.hpc.qut.edu.au
cd ~/stilt-locomotion
git pull
qsub scripts/train_stilt.pbs
qstat -u $USER
```

The PBS script handles everything automatically: installs uv if missing, runs `uv sync`
to build the venv from `uv.lock` (mjlab from PyPI, no submodule needed).

Sync logs to Mac:
```bash
rsync -avz n11298111@aquarius02.hpc.qut.edu.au:~/stilt-locomotion/logs/ ~/Desktop/stilt-locomotion/logs/
```

W&B: https://wandb.ai/simbov04-qut/stilt-locomotion
