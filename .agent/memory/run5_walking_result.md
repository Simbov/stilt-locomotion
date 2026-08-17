---
name: run5-walking-result
description: "Run 5 (2026-04-27_14-48-06) converged — the stilt G1 walks; Phase 1 done, mass robust to 6kg/stilt"
metadata: 
  node_type: memory
  type: project
  originSessionId: 3b8d6f2e-420d-4498-9f51-6d5ca686dd9b
  modified: 2026-07-23T04:59:55.949Z
---

Stilt G1 **walks**. Run `logs/rsl_rl/stilt_g1_velocity/2026-04-27_14-48-06` (full
6000 iters, mjlab v1.3) converged. Verified 2026-07-23 by reading its
`events.out.tfevents` with tensorboard's EventAccumulator.

Key final metrics: `mean_episode_length` 13→985 (ends by `time_out`, not falls),
`track_linear_velocity` reward →1.31, `fell_over`≈0, `slip_velocity_mean` 0.52→0.21.
Stilt mass curriculum swept the full **0.5–6.0 kg/stilt** range with no falls → mass is
not the binding design constraint. Caveat: `mean_reward` peaked ~49 at iter 3k, settled
~33 by end (mild late decline as mass range widened) — gait stable, maybe conservative
at the top end.

Note: several 2026-04-27 run folders (13-16 … 14-38) are aborted false starts with only
`model_0`; the real completed run is **14-48-06**. Deployable artifacts: `model_5999.pt`
+ `2026-04-27_14-48-06.onnx`.

To read tfevents metrics: activate `.venv`, use
`tensorboard.backend.event_processing.event_accumulator.EventAccumulator`. rsl_rl logs to
tfevents (torch SummaryWriter) — no CSV/W&B file locally.

Supersedes the open question in [[stilt-mass-dr-investigation]]. Docs updated in
STATUS.md + FUTURE_WORK.md (Phase 1 marked complete, Phase 2 result recorded).

**How to apply:** Phase 1 is done — next work is reward tuning for the late-training
decline, Phase 4 reward engineering, hardware deploy (Phase 6), or Phase 3 length curriculum.
