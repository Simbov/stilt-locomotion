---
name: hardware-forward-drift
description: "Run 8 policy walks forward at zero command on the real G1; joystick and obs layout ruled out, base_lin_vel zero-fill is the prime suspect"
metadata: 
  node_type: memory
  type: project
  originSessionId: e47aed7e-46a3-4be7-b3b4-d8ee09eda3d2
  modified: 2026-08-31T01:46:00.691Z
---

First bare-robot hardware test of the Run 8 policy, 2026-08-31. The robot walks
forward at a **genuinely zero** velocity command, and settles into a persistent
forward torso pitch of ~12° (`projected_gravity_x` drifts from −0.06 to −0.22
over ~15 s under policy control and stays there).

Ruled out by measurement on hardware:

- **Joystick offset** — the command observation reads exactly `vx +0.0000` at
  rest, and responds correctly when pushed (−0.19 … −0.35 backwards).
- **Observation layout / joint ordering** — held in the FixStand pose, the
  probe read left-leg `joint_pos_rel` knee `+0.19` against a predicted `+0.20`
  and ankle `−0.26` against `−0.20`. The 495-dim term-major layout is correct
  on real hardware.

**CONFIRMED (not just suspected) — the `base_lin_vel` zero-fill.** Observation `[0:15]` is
hardcoded to zeros on hardware (no sensor). At a zero command the policy is
told "commanded 0, measured 0" — no error — no matter how fast it is actually
moving, so any drift that starts is never corrected. In sim that term was
ground truth and closed the loop.

Proved in sim by zeroing `obs["actor"][:, 0:15]` exactly as the hardware stub
does, bare morphology, 15 s:

| commanded vx | real `base_lin_vel` | zeroed (= hardware) |
|---|---|---|
| 0.0 | x +0.04 m net, wanders out to 0.18 and returns | x +1.05 m, monotonic, `vx_true` +0.065 steady |
| 0.4 | `vx_true` +0.38 (correct) | **`vx_true` +2.16 and still rising** |

**The stubbed numbers above are ONE sample each and do not repeat.** Re-runs
gave −0.83 m at zero command (vs +1.05 m) and +0.80 m/s at a 0.4 command (vs
+2.16). What *is* consistent is that every stubbed run fails to track the
command and never corrects its drift, with large episode-to-episode variance,
while every un-stubbed run is tight and repeatable. So the hazard on hardware is
**unpredictability**, not a guaranteed runaway — still reason enough not to run
walking gates. Average over several episodes before quoting any stubbed figure.
Testbed scripts: `sim_zero_cmd.py` / `sim_cmd_track.py` (scratchpad, 2026-08-31)
— rerun them against any new checkpoint before going to the robot.

**There is no body-velocity estimate to read.** The `unitree_hg` `LowState_`
carries only `imu_state_` (quaternion, gyro, accel) and motor states. Fixing
this properly means either leg odometry (stance-foot kinematics + IMU) or a
retrain with `base_lin_vel` dropped or noised. A command trim would hide the
symptom and leave the robot blind to its own drift.

**Remaining confound:** sim with the stub shows `grav_x` only −0.035 while the
robot showed −0.22, so the harness (carrying part of the weight on every run)
probably accounts for the rest of the lean. Worth a slack-harness run, but it
does not change the diagnosis or the fix.

**Preferred fix:** retrain with `base_lin_vel` dropped from the *actor*
observation and kept in the critic (asymmetric actor-critic — standard precisely
because the term is unavailable on hardware). ~2.5 h on the H100. Alternative is
leg odometry (stance-foot kinematics + IMU), which needs no retrain but is real
engineering.

See [[hardware-deploy-no-sudo]] for how the runtime was built on this robot.
