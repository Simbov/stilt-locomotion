# Stilt Locomotion — Current Status
**Last updated: 2026-04-18**

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
- Stilt inertial properties: `mass="0.5"`, `diaginertia="0.008 0.008 0.001"`, COM at `pos="0.04 0 -0.2"`

### Stilt Environment (`envs/stilt_g1/`)
| File | Purpose |
|---|---|
| `env_cfgs.py` | Env config — overrides sites, geom names, reward targets, terminations, DR events, curricula |
| `curriculums.py` | Custom `stilt_mass_curriculum` class — widens stilt mass range over training |
| `stilt_robot.py` | Robot config — local MJCF path, spawn keyframe, CollisionCfg |
| `rl_cfg.py` | PPO hyperparameters (inherited from stock G1) |
| `__init__.py` | Registers `Mjlab-Velocity-Flat-Stilt-G1` environment name |

**Key environment settings vs stock G1:**
- `site_names` → stilt tips (not foot sites)
- `foot_clearance` target height → 0.10 m (matches stock G1 — do NOT increase yet)
- `foot_swing_height` target height → 0.10 m
- `air_time` weight → 0.0 (disabled until robot can walk)
- `torso_too_low` threshold → 0.65 m (was wrongly set to 0.85 m — caused free-fall termination every 13 steps)
- Friction randomisation targets stilt capsule geoms
- **Stilt mass curriculum active** — see curriculum section below

### Stilt Mass Curriculum

A four-stage curriculum progressively widens the stilt mass range during training.
Uses `dr.pseudo_inertia` (via `alpha_range`) so mass and inertia scale consistently —
physically correct for a density change. Baseline stilt mass is 0.5 kg per stilt.

| Step | alpha range | Approx mass range | Purpose |
|---|---|---|---|
| 0 | `(0.0, 0.0)` | fixed 0.5 kg | Learn to stand/walk before adding variability |
| 1000 | `(-0.18, 0.18)` | 0.35–0.72 kg | Introduce modest variability |
| 2000 | `(-0.35, 0.35)` | 0.25–1.0 kg | Widen to half/double baseline |
| 4000 | `(-0.35, 0.69)` | 0.25–2.0 kg | Asymmetric upper push to stress heavier designs |

`alpha` is a log-scale multiplier: mass = 0.5 × e^(2α). The asymmetric upper end at step 4000
(up to ~2.0 kg) is intentional — it stress-tests heavier stilt designs to inform the mechanical
design spec. The curriculum logs `Curriculum/stilt_mass/stilt_mass_min_kg` and
`stilt_mass_max_kg` to TensorBoard/W&B.

### Training Pipeline
- `scripts/train_stilt.py` — registers env and calls mjlab's `train` entry point
- `scripts/train_stilt.pbs` — PBS job: 1 node, 8 CPUs, 1×H100, 8 GB RAM, 2 hr walltime, 6000 max iterations
- `scripts/visualise.command` — double-click in Finder → file picker → viser browser viewer
- `scripts/play_stilt.py` — Python launcher for stilt visualisation

### mjlab Version
- **v1.3.0** (upgraded from v1.2.0 on 2026-04-18)
- Installed via `requirements.txt` on HPC; submodule in repo for reference
- Key v1.3 additions relevant to this project: `termination_curriculum`, `RecorderManager`,
  `RelativeJointPositionAction`, `CollisionCfg` gains `margin`/`gap`/`solmix`, reward bar panel in Viser

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
| **stilt run 5** | pending | — | **ready to submit** | mjlab v1.3, stilt mass curriculum active |

**Run 4 diagnosis (logs confirmed):**
- `torso_too_low` fired on 100% of episodes (~315/iter)
- `mean_episode_length` = 13 steps = 0.26 s → pure free-fall time from 1.16 m to 0.85 m
- `track_linear_velocity` → 0.0008 (robot never moved)
- `foot_slip` → 0.0 (confirmed stilts DID make contact, just episodes too short to learn)
- Root cause: 0.85 m threshold too aggressive for a 1.16 m spawn height with bent knees

**Fixes applied before Run 5:**
1. `torso_too_low` lowered from 0.85 → 0.65 m
2. `foot_capsule` MJCF default gets explicit `friction` + `condim=3`
3. `CollisionCfg` narrowed to stilt geoms only
4. `foot_clearance` / `foot_swing_height` targets reduced 0.25 → 0.10 m
5. `air_time` weight set to 0.0
6. PBS walltime extended to 2 hr, max-iterations to 6000
7. mjlab upgraded to v1.3.0
8. Stilt mass curriculum added

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
git submodule update --init   # only needed if submodule not yet initialised
qsub scripts/train_stilt.pbs
qstat -u $USER
```

Note: if `git submodule update` fails due to network restrictions, it can be skipped —
the PBS script installs mjlab from `requirements.txt` (pip), not from the submodule.

Sync logs to Mac:
```bash
rsync -avz n11298111@aquarius02.hpc.qut.edu.au:~/stilt-locomotion/logs/ ~/Desktop/stilt-locomotion/logs/
```

W&B: https://wandb.ai/simbov04-qut/stilt-locomotion
