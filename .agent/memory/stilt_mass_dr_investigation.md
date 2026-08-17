---
name: Stilt mass DR viewer investigation
description: Resolution of why joint torques don't visibly change with stilt mass in the viewer, and the correct diagnostic metric
type: project
originSessionId: c70b4bd8-4444-4da3-a6df-655266d68398
---
**RESOLVED.** The mass change IS reaching the dynamics. The CUDA graph is NOT the problem.

## Why the mass reaches physics every step

`step_graph → mjwarp.step → forward → fwd_position → smooth.com_pos → _cinert kernel`

The `_cinert` kernel reads `m.body_mass` directly every step:
```python
mass = body_mass[worldid % body_mass.shape[0], bodyid]
```
`pseudo_inertia` writes to the same Warp array (in-place). CUDA graphs hold
pointers to GPU buffers — in-place writes are visible. Only array *replacement*
(new allocation via `expand_model_fields`) would invalidate a graph.

## Why `qfrc_actuator` didn't change

`qfrc_actuator = kp*(q_des - q)` — PD tracking error, not gravitational load.
At 20 kg (5-10× out-of-distribution), the policy adapts to a very slow/careful
gait (tiny swing accelerations → small tracking error → small PD output).
The physics are correct; the measurement was wrong.

## Correct metric: `qfrc_bias`

`qfrc_bias` = gravity + Coriolis, computed by RNE from `cinert` (which reads
`body_mass` every step). Changes immediately and proportionally with stilt mass,
regardless of policy adaptation.

**GUI change (2026-04-21):** Added `qfrc_bias` section to the Joint Torques
monitor in `envs/stilt_g1/__init__.py`. Also added `cinert[9]` (mass component)
to the mass readback label to confirm the write propagated through
`set_const_0 → com_pos → _cinert`.

## On `set_const`

`set_const` updates model constants (`dof_invweight0`, `body_subtreemass`,
`actuator_acc0`) used by the solver's preconditioning — not the main dynamics
path. The step already recomputes `cinert` from `body_mass` every step via
`com_pos`. `set_const` is still correct to call (it keeps solver weights
consistent with new mass), but it's not needed for dynamics correctness.

**How to apply:** When investigating why a DR variable seems to have no effect
in the viewer, check `qfrc_bias` not `qfrc_actuator`. Also read `cinert[body, 9]`
right after `_apply()` to confirm the write propagated to the composite inertia.
