---
name: stilts-first-hardware-test
description: 2026-09-17 first stilts-on Run 9 test on a third G1 (eth0 3c:6d:66:a3:e0:76) — diverged in 12 s; ankle moved ±0.6 rad where sim brace holds it near 0
metadata: 
  node_type: memory
  type: project
  originSessionId: 3757350a-055e-4435-9b7a-d853382431ac
  modified: 2026-09-17T05:38:21.398Z
---

2026-09-17, new G1 (eth0 `3c:6d:66:a3:e0:76`), stilts fitted, Run 9, zero command.
Setup: robot had no unitree_hg SDK (only go2/ros2 in `~/unitree_sdk2-main` and
`/usr/local`); copied laptop's `Desktop/Robot Training/unitree_sdk2` tarball to
`~/unitree_sdk2`, no sudo. First FixStand attempts failed with no FSM transition
logged — cause never pinned (raw remote bytes showed only taps, later holds worked
fine); after a power cycle L2+Up worked.

Result: two Velocity segments (1.8 s, 11.9 s), second diverged — leg actions to
±12, ankle pitch hitting the +0.54 rad limit, tilt 28°. Logs in
`logs/hardware/2026-09-17/`.

Key evidence vs `sim_telemetry.py --stilts`: in sim the ankle pitch stays within
−0.06..0 (brace stiffness 150–2000 Nm/rad); on hardware it sagged to −0.2..−0.35
immediately and later swung −0.64..+0.54, quasi-static tau/q ~25–65 Nm/rad. The
real brace is not constraining the ankle the way training assumed (loose clamp,
flex, or softer design). Stepping in place at zero command is normal for Run 9 in
sim too (L/R knee anti-phase, ~2 Hz sim vs 2.7 Hz HW).

**How to apply:** check the physical brace/clamp before another stilts test;
if it is genuinely that compliant, measure it and retrain with the stiffness range
lowered. Related: [[run9-hardware-result]], [[hardware-deploy-no-sudo]].
