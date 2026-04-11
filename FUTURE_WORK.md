# Stilt Locomotion — Future Work
**Last updated: 2026-04-11**

---

## Immediate (Next Run)

**Submit and monitor the current training job:**
```bash
# On HPC
git pull && qsub scripts/train_stilt.pbs
```

**What to look for in the logs / W&B after 500 iterations:**
- `mean_episode_length` should be increasing (>50 steps means robot is learning to balance)
- `torso_too_low` should NOT be the dominant termination (time_out should start appearing)
- `track_linear_velocity` should start increasing from near-zero
- If still stuck at 13-step episodes — lower `torso_too_low` threshold further to 0.50 m or remove it entirely

---

## Phase 1 — Get the Robot Standing and Walking (Short Stilts)

Priority order:

1. **Confirm current 0.435 m stilt training converges**
   - Target: mean_episode_length > 500 by iteration 1000
   - Target: track_linear_velocity > 0.5 by iteration 3000

2. **Once walking, tune reward weights for stilt dynamics**
   - Enable `air_time` reward (weight = 0.5–1.0) — stilts need a clear lift-and-plant rhythm
   - Increase `foot_clearance` target to 0.15–0.20 m once robot is stable
   - Consider increasing `soft_landing` weight (currently 1e-5) — stilt impact forces are large

3. **Tune pose reward standard deviations for stilt gait**
   - Stock G1 std values may be too tight/loose for stilt dynamics
   - Ankle pitch/roll stds may need loosening (stilts create larger ankle moments)
   - Monitor `Episode_Reward/pose` — if it stays near max, stds are too loose

4. **Analyse gait quality**
   - Check `Metrics/slip_velocity_mean` — should drop as robot learns to plant firmly
   - Check `Metrics/landing_force_mean` — stilt impact should be controlled
   - Visualise with `scripts/visualise.command` — look for smooth stepping, not shuffling

---

## Phase 2 — Stilt Length Curriculum

Once the robot walks stably at 0.435 m:

1. **Add stilt length as a domain randomisation parameter**
   - Randomise stilt length during training (e.g. 0.3–0.6 m range)
   - Requires parametric MJCF or runtime body position modification
   - Reference: `mjlab/src/mjlab/tasks/velocity/velocity_env_cfg.py` curriculum pattern

2. **Progressive length curriculum**
   - Stage 1: 0.3–0.4 m (current physical stilt length)
   - Stage 2: 0.4–0.5 m
   - Stage 3: 0.5–0.7 m
   - Tune spawn keyframe height per stage

3. **Coordinate with partner on physical stilt range**
   - What lengths are mechanically feasible?
   - Is there a quick-change mechanism? If so, curriculum should match hardware range.

---

## Phase 3 — Reward Engineering for Stilt Gait

Issues to revisit once basic walking works:

- **Foot clearance**: 0.10 m target was matched to stock G1. Stilts (longer pendulum) need higher clearance to avoid clipping on swing. Try 0.15–0.20 m after iter 2000.
- **Angular momentum penalty**: stilts shift COM upward → larger angular momentum swings. May need to reduce `angular_momentum` weight.
- **Body angular velocity penalty**: same concern — stilt walking naturally involves more upper body sway.
- **Termination threshold**: `torso_too_low = 0.65 m` — verify this doesn't fire during normal deep-knee bends. Consider moving to `0.60 m` if needed.

---

## Phase 4 — Rough Terrain (Optional)

If time permits and the robot walks well on flat ground:

1. Enable terrain curriculum in env config
2. Re-enable `height_scan` observations (currently removed for flat terrain)
3. Gradually increase terrain roughness using mjlab's terrain curriculum
4. This is important for real-world deployment but not required for thesis proof-of-concept

---

## Phase 5 — Sim-to-Real Transfer

### 5a. Policy Export
```python
# mjlab auto-exports .onnx at end of training
# Also manually:
from mjlab.utils.export import export_policy
export_policy(checkpoint_path="logs/.../model_6000.pt", output_path="stilt_policy.onnx")
```

### 5b. Observation Format
The policy expects at 50 Hz:
```
base_lin_vel      (3)   — from IMU or state estimator
base_ang_vel      (3)   — from IMU
projected_gravity (3)   — from IMU + orientation filter
joint_pos         (29)  — relative to default, from encoders
joint_vel         (29)  — from encoders
last_action       (29)  — previous joint position targets
velocity_command  (3)   — vx, vy, yaw from joystick/planner
```

### 5c. Deployment Stack
- Hardware: Unitree G1 onboard computer (or Jetson Orin)
- Interface: `unitree_sdk2py` for joint commands
- Policy runner: `onnxruntime` at 50 Hz
- PD controllers: run at 1 kHz between policy steps

### 5d. Safety Protocol
1. Start with robot on overhead gantry/harness
2. Test standing still (zero velocity command) before walking
3. Increase commanded velocity gradually
4. Emergency stop: set velocity command to (0, 0, 0)
5. First real-world test: short stilts (0.3 m), flat ground

---

## Thesis Contributions to Document

- [ ] Baseline G1 gait metrics (reward, episode length, gait frequency from Run 3)
- [ ] LIPM prediction: ω = √(g/h) → longer stilts slow natural gait frequency. Measure and compare.
- [ ] Reward convergence curves for stilt vs no-stilt training
- [ ] Ablation: what happens without `torso_too_low` termination? With different clearance targets?
- [ ] Collision geometry design decisions (8 capsule approach vs single sphere)
- [ ] Domain randomisation role (friction 0.3–1.2 range effect on transfer)

---

## Known Technical Debt

| Issue | File | Notes |
|---|---|---|
| `torso_too_low` threshold not validated for all joint configurations | `envs/stilt_g1/env_cfgs.py:65` | 0.65 m is a rough estimate; verify with kinematics at max squat |
| Stilt inertia is a rough estimate | `assets/mjcf/g1/g1.xml:129` | mass=0.5 kg, inertia=diag(0.008,0.008,0.001). Measure physical stilt. |
| `STILT_G1_KEYFRAME` spawn height not re-verified after joint angle changes | `envs/stilt_g1/stilt_robot.py:37` | 1.16 m is correct for current angles (verified by kinematics script) |
| Stilt body has no joint — cannot do length curriculum without MJCF regen | `assets/mjcf/g1/g1.xml:128` | Need to parameterise stilt body pos for curriculum |
| Air time reward disabled | `envs/stilt_g1/env_cfgs.py:49` | Re-enable at weight=0.5 once robot walks |
