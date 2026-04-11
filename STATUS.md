# Stilt Locomotion — Current Status
**Last updated: 2026-04-11**

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

### Stilt Environment (`envs/stilt_g1/`)
| File | Purpose |
|---|---|
| `env_cfgs.py` | Env config — overrides sites, geom names, reward targets, terminations |
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

### Training Pipeline
- `scripts/train_stilt.py` — registers env and calls mjlab's `train` entry point
- `scripts/train_stilt.pbs` — PBS job: 1 node, 8 CPUs, 1×H100, 8 GB RAM, 2 hr walltime, 6000 max iterations
- `scripts/visualise.command` — double-click in Finder → file picker → viser browser viewer
- `scripts/play_stilt.py` — Python launcher for stilt visualisation

---

## Training Runs

| Run | Date | Iters | Status | Notes |
|---|---|---|---|---|
| stilt run 1 | 2026-03-27_12-25-11 | short | abandoned | wrong foot site names |
| stilt run 2 | 2026-03-27_12-43-32 | ~2000 | slipping | stilt slipping, friction not applied |
| stilt run 3 | 2026-03-27_16-51-02 | short | abandoned | early test |
| stilt run 4 | 2026-03-27_20-32-07 | 1499 | **broken** | 13-step episodes — torso_too_low threshold too high (0.85 m) |

**Run 4 diagnosis (logs confirmed):**
- `torso_too_low` fired on 100% of episodes (~315/iter)
- `mean_episode_length` = 13 steps = 0.26 s → pure free-fall time from 1.16 m to 0.85 m
- `track_linear_velocity` → 0.0008 (robot never moved)
- `foot_slip` → 0.0 (confirmed stilts DID make contact, just episodes too short to learn)
- Root cause: 0.85 m threshold too aggressive for a 1.16 m spawn height with bent knees

**Fixes applied before next run:**
1. `torso_too_low` lowered from 0.85 → 0.65 m (`envs/stilt_g1/env_cfgs.py`)
2. `foot_capsule` MJCF default gets explicit `friction` + `condim=3`
3. `CollisionCfg` narrowed to stilt geoms only (removes ambiguous catch-all)
4. `foot_clearance` / `foot_swing_height` targets reduced 0.25 → 0.10 m
5. `air_time` weight set to 0.0
6. PBS walltime extended to 2 hr, max-iterations to 6000

---

## Collision & Contact (Verified)

Verified via Python test (`assets/mjcf/g1/` directory):
- All 16 stilt capsules: `contype=1 conaffinity=1 condim=3` ✓
- Non-stilt geoms disabled by `CollisionCfg.disable_other_geoms=True` (expected behaviour)
- Stilt tip z at spawn: +0.003 m above floor (3 mm gap — normal, robot settles on first step)

---

## Latest Git Commits
```
d9532f7  chore(pbs): increase walltime to 2h and max-iterations to 6000
2e9fbfd  fix(stilt): lower torso_too_low threshold from 0.85m to 0.65m
1cfcade  feat(stilt): add torso height termination to end collapsed episodes early
2dafea0  fix(stilt): reduce reward targets and fix friction for early training
a534459  fix(stilt): symmetric inertia on both stilts, correct spawn height
6ae3318  fix(visualise): always use local Desktop project path for venv/scripts
820b5db  feat: add stilt G1 environment and training pipeline
```

---

## HPC Access
```bash
ssh n11298111@aquarius02.hpc.qut.edu.au
cd ~/stilt-locomotion
git pull
qsub scripts/train_stilt.pbs
qstat -u $USER
```

Sync logs to Mac:
```bash
rsync -avz n11298111@aquarius02.hpc.qut.edu.au:~/stilt-locomotion/logs/ ~/Desktop/stilt-locomotion/logs/
```

W&B: https://wandb.ai/simbov04-qut/stilt-locomotion
