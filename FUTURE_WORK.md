# Stilt Locomotion — Future Work
**Last updated: 2026-08-13**

---

## Now — Run 8, one policy for stilts on and off

See `STATUS.md` for the design. What is genuinely unknown going in, in the order
it is likely to bite:

1. **Can one policy do both?** The two morphologies differ by 44 cm of leg and
   2.8 kg per side. If it converges to one mode and abandons the other, the split
   metrics will show it early: watch `Episode_Metrics/upright_stilts_{on,off}`
   diverge. Fallbacks, cheapest first — raise the observation history above 5
   frames, then bias `fitted_probability` toward whichever mode is losing, then
   accept two policies and a switch.
2. **Is the brace stiffness range right?** 150–2000 Nm/rad is a guess. Nobody has
   measured the real clamp. If the hardware turns out stiffer than 2000, the
   ankles will do more static work than training predicted — watch their
   temperature on the first stilted runs.
3. **Minimum safe tube overlap** is still assumed at 80 mm, which caps the height
   curriculum at ±50 mm. Confirm it and the range could go to 522 mm.

After Run 8 has a checkpoint: regenerate the structural report
(`scripts/analyse_stilt_loads.py`) — the published one was sampled from the void
Run 7 policy — and regenerate `deploy/config/g1_stilt/deploy.yaml` from the new
ONNX metadata.

---

## ✅ DONE — Run 5 complete, robot walks (Phase 1 achieved)

Run `2026-04-27_14-48-06` (6000 iters, mjlab v1.3) converged to a stable walking gait.
See `STATUS.md` → *Run 5 results* for the full metrics. Headlines:
- `mean_episode_length` 13 → **985** (survives near-full episodes; ends by `time_out`, not falls)
- `track_linear_velocity` reward → **1.31** (target was >0.5 by iter 3k)
- `fell_over` termination ≈ **0**
- Stilt mass curriculum swept to the full **0.5–6.0 kg** range without collapse

**Next actions now that it walks** (roughly in priority order):
1. Tune the late-training reward decline (`mean_reward` 49→33 as mass range widened) — try a
   gentler final curriculum stage or a longer hold at each stage.
2. Move to Phase 4 reward engineering (re-enable `air_time`, raise `foot_clearance`).
3. Deploy the Run 5 stilt ONNX to hardware (Phase 6 — infra already in place).
4. Start the Phase 3 stilt-length curriculum.

---

## Phase 1 — Get the Robot Walking (Short Stilts, Fixed Mass) — ✅ COMPLETE

1. **Confirm 0.435 m stilt training converges** — ✅ done (run 5)
   - Target: `mean_episode_length` > 500 by iteration 1000 → **met (~921 at iter 1k)**
   - Target: `track_linear_velocity` > 0.5 by iteration 3000 → **met (~1.47 at iter 3k)**

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

## Phase 2 — Stilt Mass Curriculum — ✅ RESULT IN

Run 5 completed the full curriculum. **The policy stayed stable across the entire
sweep to 0.5–6.0 kg per stilt with `fell_over` ≈ 0** — up to 4× the 1.5 kg baseline.
So the acceptable stilt-weight range for mechanical design is generously wide; mass is
**not** the binding constraint. (Watch item: the mild late `mean_reward` decline coincides
with the widest mass stage — the gait holds but may be slightly conservative at the top end.)

Remaining Phase 2 follow-ups:

1. ~~Read the mass limit from training results~~ → **done: robust to ~6 kg/stilt.**
   - Use `Curriculum/stilt_mass/stilt_mass_max_kg` in the tfevents/W&B to track the active ceiling.

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

Deployment infrastructure is in place — see `deploy/` for all files and instructions.

### 6a. Runtime stack

Uses [unitree_rl_mjlab](https://github.com/unitreerobotics/unitree_rl_mjlab)'s C++ runtime:
- ONNX policy loaded by `OrtRunner` (ONNX Runtime 1.22.0)
- 50 Hz control loop via DDS (`rt/lowcmd` / `rt/lowstate`)
- PD gains, default pose, and action scale all embedded in the ONNX metadata and read by `deploy/config/g1_stilt/deploy.yaml`
- Launched via SSH: `./g1_ctrl --net eth0 --config deploy.yaml --policy policy.onnx`

There is no app-based policy selection — the custom policy runs as a standalone
binary alongside (not inside) Unitree's factory controllers.

### 6b. Observation format (99-dim, 50 Hz)

| Term | Dims | Source |
|---|---|---|
| `base_lin_vel` | 3 | **Zeroed** (no hardware sensor; policy is robust for non-speed-critical use) |
| `base_ang_vel` | 3 | IMU gyroscope (`rt/lowstate`) |
| `projected_gravity` | 3 | Computed from IMU quaternion |
| `joint_pos` | 29 | Encoder positions minus default pose |
| `joint_vel` | 29 | Encoder velocities |
| `last_action` | 29 | Previous policy output |
| `velocity_command` | 3 | vx, vy, yaw from joystick |

`base_lin_vel` is zeroed because the policy's goal is stable stilt walking, not
precise speed tracking. Variable speed still works — it is controlled via
`velocity_command` (joystick), which is unaffected. If precise speed tracking
becomes important, read from `rt/sportmodestate velocity[3]` and rotate to body
frame via IMU quaternion.

### 6c. Transfer steps

1. Sync ONNX from HPC: `rsync -avz <hpc>:~/stilt-locomotion/logs/ logs/`
2. Copy ONNX + config to robot: see `deploy/README.md`
3. Add `base_lin_vel` zero-fill stub in unitree_rl_mjlab observation manager (one-time)
4. Build: `cmake .. && make -j$(nproc)`
5. SSH to robot and run `./g1_ctrl`

For a new training run, only step 1–2 need repeating.

### 6d. Safety protocol
1. Robot on overhead gantry/harness — mandatory for all early tests
2. Short stilts (0.3 m) for first hardware tests
3. Test standing still (zero velocity command) before commanding motion
4. Increase commanded velocity gradually
5. Emergency stop: return joystick to centre (velocity command → 0,0,0)

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
| ~~Mass curriculum upper bound unvalidated~~ | `envs/stilt_g1/env_cfgs.py:91` | ✅ validated by run 5 — stable to 6.0 kg/stilt |
