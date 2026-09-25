---
name: run9-hardware-result
description: "Run 9 on the real G1 (2026-09-14, bare feet) — stable walking/turning over the full command range, slight creep at zero command; sim also creeps"
metadata: 
  node_type: memory
  type: project
  originSessionId: fd27f27b-c8fd-409d-afd6-5eb91e7921a3
  modified: 2026-09-14T05:58:18.727Z
---

2026-09-14, QCR lab G1 (eth0 `3c:6d:66:a3:e5:73`), bare feet, Run 9
(`2026-08-31_13-35-11_run9-no-baselinvel`), tested cable-free via `setsid`.
User's observation: **walked on the spot with a slight creep at zero command,
but stable walking, turning and control across the whole commanded range.**
First policy that works on hardware — fixes the Run 8 zero-command runaway
([[hardware-forward-drift]]).

The run log (`logs/hardware/2026-09-14/run_1789343166.log`) had only FSM
transitions: one continuous 4 min 32 s Velocity stretch, no errors. No
quantitative data — that session predated the telemetry recorder.

**Sim creeps too.** `scripts/sim_telemetry.py` (bare, Run 9, one rollout) gave
2.5–3 cm/s drift at zero command with slow yaw (~−3°/s), true drift 13–25 cm per
5–7.5 s window. That contradicts the runbook's "within 4 cm over 15 s in sim"
claim — not yet reconciled (single rollout; earlier notes say per-episode
variance is large). So the hardware creep may be policy behaviour, not a
sim-to-real gap.

Leg odometry in `scripts/analyze_hardware_log.py` validated against sim truth:
velocities within 0.01 m/s, drift distance ~10% low.

**Next visit:** the robot's runtime has no telemetry yet — ship
`--runtime` once. See [[hardware-deploy-no-sudo]].
