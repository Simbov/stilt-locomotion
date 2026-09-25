# Training run results

One row per run, newest first. Numbers here are measured, not remembered — the
evaluation tables come from `scripts/eval_both_modes.py` (5 fresh episodes per
command, morphology pinned), and the training aggregates are tail means over
the last 50 iterations of the run's `events.out.tfevents`.

**Conditional means.** Every `*_stilts_on/off` metric in the training logs is
masked, so it must be divided by `stilts_fitted_fraction` (or `1 −` that) to
get the per-morphology mean. The raw aggregate cannot tell "walks on stilts,
falls over without them" from "mediocre at both". See `envs/stilt_g1/metrics.py`.

---

## Run 9 — `2026-08-31_13-35-11_run9-no-baselinvel` ✅ current, deployable

6000 iterations, 4096 envs, H100, 2h45m. Actor observation **480** (6 terms ×
5 frames). **The only change from Run 8: `base_lin_vel` moved to the critic.**

### Why it exists

Run 8 failed on hardware (see below). `base_lin_vel` is body linear velocity;
the G1 has no sensor for it, so the deploy runtime could only zero-fill it, and
Run 8 had learned to depend on it. Run 9 removes the dependency rather than
approximating the missing signal.

### Evaluation

| command (vx, vy, yaw) | stilts ON | stilts OFF |
|---|---|---|
| 0.0 | −0.033 | +0.021 |
| vx 0.2 | 0.181 | 0.192 |
| vx 0.4 | 0.281 | 0.343 |
| vx 0.6 | 0.555 | 0.331 |
| vx 0.8 | 0.777 | 0.673 |
| vx −0.4 | −0.272 | −0.312 |
| vy 0.4 | 0.337 | 0.318 |
| yaw 0.6 | 0.314 (sd 0.148) | 0.445 (sd 0.146) |
| **mean planar error** | **0.056 m/s** | **0.087 m/s** |
| **falls** | **0 / 40** | **0 / 40** |

Pelvis held 1.196–1.204 m fitted and 0.784–0.789 m bare across every command.

### Deployment behaviour (the test Run 8 failed)

`scripts/check_base_lin_vel_stub.py`, bare, 15 s, no velocity feedback available:

| commanded vx | result |
|---|---|
| 0.0 | position held within **4 cm**; `grav_x` oscillates ±0.07 about zero, no lean |
| 0.4 | `vx_true` **0.33–0.40, steady** — tracks, does not accelerate |

### Training aggregates vs Run 8

| metric | Run 8 | Run 9 |
|---|---|---|
| `Train/mean_episode_length` | 994.5 | 993.9 |
| `Train/mean_reward` | 63.4 | 59.5 |
| `Episode_Termination/fell_over` | 0.0017 | 0.0017 |
| `vel_error` stilts on *(conditional)* | 0.251 | 0.273 |
| `vel_error` stilts off *(conditional)* | 0.195 | 0.205 |
| `upright`, both modes *(conditional)* | 0.999 | 0.998–0.999 |

All curricula completed to the same endpoints: mass 7.61 kg/stilt, telescope
458 mm, command 0.8 m/s.

**Read these two tables together.** The training aggregate says Run 9 is 5–9%
worse; the behavioural evaluation says it is better. They measure different
things — the aggregate averages over the full randomised command and DR
distribution, the evaluation sweeps fixed commands. The fair summary is
"equivalent, with notably better yaw", not "improved".

### Still weak

- **Forward response is not monotonic.** 0.4 → 0.281 on stilts, 0.6 → 0.331
  bare. Present in every run since Run 5; worth a look in the viewer.
- **Yaw undershoots**, though far more consistently than Run 8 (sd 0.146 vs
  0.373 bare — Run 8's spread was wider than its own mean).

---

## Run 8 — `2026-08-13_20-35-42_run8-stilts-on-off` ⚠️ sim-only, NOT deployable

6000 iterations, 4096 envs, 2h36m. Actor observation 495 (7 terms × 5 frames,
including `base_lin_vel`).

**The two-morphology goal was met in sim** — one policy, never told which
configuration it is in, walking in both and never falling in either. This run
is still the reference for that result.

| command | stilts ON | stilts OFF |
|---|---|---|
| vx 0.2 | 0.235 | 0.210 |
| vx 0.4 | 0.413 | 0.386 |
| vx 0.6 | 0.445 | 0.465 |
| vx 0.8 | 0.785 | 0.557 |
| vx −0.4 | −0.247 | −0.265 |
| vy 0.4 | 0.303 | 0.253 |
| yaw 0.6 | 0.316 (sd 0.177) | 0.326 (sd 0.373) |
| **mean planar error** | 0.088 m/s | 0.094 m/s |
| **falls** | 0 / 40 | 0 / 40 |

### Why it is not deployable

Tested on the robot 2026-08-31 (bare, on the gantry). Everything in the
deployment path worked — it loaded, the observation layout was verified against
real encoders, the joystick read clean — and the policy **walked forward at a
genuinely zero command**, settling into a ~12° forward torso pitch
(`projected_gravity_x` −0.06 → −0.22 and holding).

Cause: the `base_lin_vel` zero-fill. Reproduced in sim by zeroing the same
slice — at zero command the drift goes uncorrected instead of self-cancelling
(~1 m in 15 s against 4 cm), and at a 0.4 command tracking becomes erratic
(+0.80 to +2.16 across runs, against a repeatable +0.38 un-stubbed). Every
stubbed run fails; every un-stubbed run is tight. The hazard is
unpredictability rather than a guaranteed runaway.

Logs: `logs/hardware/2026-08-31/`.

---

## Runs 6 and 7 — VOID

Both trained a **25-DoF** policy for a robot whose ankle joints had been
deleted, on the reading that the shank brace rigidly welds the ankle. That
robot does not exist: the hardware is always the stock 29-DoF G1. Kept only for
the two fixes that carried into Run 8 — `air_time` at weight 0.5 with
`command_threshold` 0.3, and capping the command curriculum at 0.8 m/s rather
than ramping to 2.0. Together those cured Run 6's frozen ~0.35 m/s gait.

## Run 5 — `2026-04-27_14-48-06`

First converged walk, but on the **old box stilt**, which the current CAD
replaced. Episode length 985/1000, zero falls, stable across 0.5–6.0 kg per
stilt. Superseded by the hardware change, not by a training problem.
