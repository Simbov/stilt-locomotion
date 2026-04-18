# Stilt Locomotion — Future Work
**Last updated: 2026-04-18**

---

## Immediate (Run 5 — Ready to Submit)

```bash
# On HPC
git pull && qsub scripts/train_stilt.pbs
```

**What to look for in W&B after 500 iterations:**
- `mean_episode_length` increasing (>50 steps = robot learning to balance)
- `torso_too_low` NOT dominant — `time_out` should start appearing by iter 1000
- `track_linear_velocity` increasing from near-zero
- `Curriculum/stilt_mass/stilt_mass_min_kg` and `max_kg` stepping up at iters 1000/2000/4000

If still stuck at 13-step episodes — lower `torso_too_low` to 0.50 m or remove it entirely.

---

## Phase 1 — Get the Robot Walking (Short Stilts, Fixed Mass)

1. **Confirm 0.435 m stilt training converges**
   - Target: `mean_episode_length` > 500 by iteration 1000
   - Target: `track_linear_velocity` > 0.5 by iteration 3000

2. **Once walking, tune reward weights for stilt dynamics**
   - Enable `air_time` reward (weight = 0.5–1.0) — stilts need a clear lift-and-plant rhythm
   - Increase `foot_clearance` target to 0.15–0.20 m once robot is stable
   - Consider increasing `soft_landing` weight — stilt impact forces are large

3. **Tune pose reward standard deviations for stilt gait**
   - Ankle pitch/roll stds may need loosening (stilts create larger ankle moments)
   - Monitor `Episode_Reward/pose` — if it stays near max, stds are too loose

4. **Analyse gait quality**
   - `Metrics/slip_velocity_mean` should drop as robot learns to plant firmly
   - `Metrics/landing_force_mean` — stilt impact should be controlled
   - Visualise with `scripts/visualise.command`

---

## Phase 2 — Stilt Mass Curriculum (Active from Run 5)

The mass curriculum is already running. After Run 5 completes:

1. **Read the mass limit from training results**
   - If the policy converges well to iter 4000, the robot can handle ~0.25–2.0 kg
   - If training degrades at step 2000–4000, the upper bound is closer to 1.0 kg
   - Use `Curriculum/stilt_mass/stilt_mass_max_kg` in W&B to track the active ceiling

2. **Interpret for mechanical design**
   - The mass range the policy handles robustly = acceptable stilt weight range
   - Physical stilt is aluminium extrusion + attachment hardware — measure actual mass
   - If measured mass < curriculum upper bound → design has margin; if not → lighten the design

3. **Optionally extend the curriculum upper bound**
   - Edit stages in `env_cfgs.py` if results suggest >2.0 kg is achievable
   - `alpha = ln(mass / 0.5) / 2` gives the alpha value for any target mass

---

## Phase 3 — Stilt Length Curriculum

Once the robot walks stably at 0.435 m:

1. **Add stilt length as a domain randomisation parameter**
   - Requires runtime body position modification (`dr.body_pos` on `left_stilt`/`right_stilt`)
   - Also need to update stilt tip site positions (currently hardcoded in MJCF)
   - Spawn keyframe height must track stilt length changes

2. **Progressive length curriculum**
   - Stage 1: 0.3–0.4 m
   - Stage 2: 0.4–0.5 m
   - Stage 3: 0.5–0.7 m

3. **Coordinate with partner on physical stilt range**
   - What lengths are mechanically feasible?
   - Is there a quick-change mechanism?

---

## Phase 4 — Reward Engineering for Stilt Gait

- **Foot clearance**: stilts (longer pendulum) need higher clearance to avoid clipping on swing. Try 0.15–0.20 m after iter 2000.
- **Angular momentum penalty**: stilts shift COM upward → larger angular momentum swings. May need to reduce `angular_momentum` weight.
- **Termination threshold**: verify `torso_too_low = 0.65 m` doesn't fire during normal deep-knee bends.

---

## Phase 5 — Rough Terrain (Optional)

1. Enable terrain curriculum in env config
2. Re-enable `height_scan` observations (currently removed for flat terrain)
3. Gradually increase terrain roughness using mjlab's terrain curriculum

---

## Phase 6 — Sim-to-Real Transfer

### 6a. Policy Export
```python
# mjlab auto-exports .onnx at end of training
# Also manually:
from mjlab.utils.export import export_policy
export_policy(checkpoint_path="logs/.../model_6000.pt", output_path="stilt_policy.onnx")
```

### 6b. Observation Format
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

### 6c. Deployment Stack
- Hardware: Unitree G1 onboard computer (or Jetson Orin)
- Interface: `unitree_sdk2py` for joint commands
- Policy runner: `onnxruntime` at 50 Hz
- PD controllers: run at 1 kHz between policy steps

### 6d. Safety Protocol
1. Start with robot on overhead gantry/harness
2. Test standing still (zero velocity command) before walking
3. Increase commanded velocity gradually
4. Emergency stop: set velocity command to (0, 0, 0)
5. First real-world test: short stilts (0.3 m), flat ground

---

## Thesis Contributions to Document

- [ ] Baseline G1 gait metrics (reward, episode length, gait frequency)
- [ ] LIPM prediction: ω = √(g/h) → longer stilts slow natural gait frequency. Measure and compare.
- [ ] Reward convergence curves for stilt vs no-stilt training
- [ ] Stilt mass robustness range — what mass can the policy tolerate? (from Run 5 curriculum)
- [ ] Ablation: what happens without `torso_too_low` termination? With different clearance targets?
- [ ] Collision geometry design decisions (8 capsule approach vs single sphere)
- [ ] Domain randomisation role (friction + mass range effect on transfer)

---

## Known Technical Debt

| Issue | File | Notes |
|---|---|---|
| `torso_too_low` threshold not validated for all joint configurations | `envs/stilt_g1/env_cfgs.py:103` | 0.65 m is a rough estimate; verify with kinematics at max squat |
| Stilt inertia is a rough estimate | `assets/mjcf/g1/g1.xml:129` | mass=0.5 kg, inertia=diag(0.008,0.008,0.001) — measure physical stilt |
| Stilt tip sites hardcoded in MJCF | `assets/mjcf/g1/g1.xml:141` | Will need updating for length curriculum |
| Air time reward disabled | `envs/stilt_g1/env_cfgs.py:54` | Re-enable at weight=0.5 once robot walks |
| Mass curriculum upper bound (2.0 kg) unvalidated | `envs/stilt_g1/env_cfgs.py:91` | May need adjusting based on Run 5 results |
