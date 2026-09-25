---
name: stilts-second-hardware-test
description: 2026-09-25 Run 9 stilts-on 4 s run did not diverge (actions ≤4.5 vs 11–13 on 09-17) but ankle pitch still sags −0.08..−0.25 rad vs sim −0.01..−0.05; drifted 30 cm left
metadata:
  type: project
---

2026-09-25, same G1 as 09-17 (eth0 3c:6d:66:a3:e0:76), Run 9, stilts on, one 4.1 s zero-command
Velocity segment (`logs/hardware/2026-09-25/g1_20260925-141718.csv`, analysis `_stilts.md` beside it).
Earlier the same day (13:17–13:30) the robot ran Run 9 bare-feet for about 53 s: stable, tracked 0.32 m/s, forward creep 2–6 cm/s.

Stilts result vs 09-17: bounded. Max |action| 2.8–4.5 (sim 5, 09-17 11–13); ankle roll ±0.07 (was −0.3..+0.1).
Still off-distribution: ankle pitch sits at −0.08..−0.25 rad (sim −0.01..−0.05), so the brace is still
much softer than the 150 Nm/rad training floor. Drifted 30 cm LEFT in 3.6 s (8.5 cm/s), tilt max 10°.

D-pad Down → Passive was added to the robot config this day (backup `config.yaml.pre-quickdamp`).
Kill transitions logged fine in the 14:09 session. Earlier "kill didn't work" scare was a
frozen terminal tab, not the robot — read the robot's telemetry CSVs, not a stale terminal pane.

**How to apply:** before longer stilt runs, stiffen/shim the brace until the static ankle pitch sag ≈ 0;
if it can't be, measure it and retrain with a lower brace_stiffness_range. Related: [[stilts-first-hardware-test]], [[run9-hardware-result]].
