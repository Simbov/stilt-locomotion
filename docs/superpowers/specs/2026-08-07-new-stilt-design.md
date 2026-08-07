# New Stilt Design — Telescoping Shank-Clamped Stilt

**Date:** 2026-08-07
**Status:** Approved design, pending implementation plan
**Source CAD:** `Assembled 40.7cm.STL` (66 solids, 99,170 triangles, mm units)

---

## 1. Motivation

The current stilt is a single solid box (220 × 80 × 400 mm, 1,008 triangles, 1.5 kg,
`diaginertia="0.024 0.024 0.003"`). It bolts to the sole of `ankle_roll_link` and nothing
else, so the ankle stays fully actuated.

The new hardware is different in three ways that the sim cannot express today:

1. **It clamps to the shank.** A brace rises 214 mm above the mounting plate and clamps the
   calf. Ankle pitch and roll can no longer move.
2. **It telescopes.** Two 250 mm square tubes (40 × 40 outer, 35 × 35 inner) slide, so
   height is adjustable rather than fixed.
3. **It is 2.8 kg, not 1.5 kg**, and the mass is distributed over a 621 mm tall structure.
   The current `diaginertia` is wrong by roughly an order of magnitude on two axes.

Goal: a model that matches the real hardware, instrumented so the internal loads are
visible during testing, followed by a fresh training run.

---

## 2. Measured geometry

Mesh units are mm. The mesh point `(114.1, 66.2, 667.3)` maps to the `ankle_roll_link`
origin, carrying forward the convention of the existing `refpos="70 40 435"`.

Confirmation that this convention is right: under it the ground plate footprint lands at
x −0.070…0.150, y ±0.040 in ankle frame, which matches the current collision capsule blocks
almost exactly.

| Feature | Mesh z (mm) | Ankle-frame z (m) |
|---|---|---|
| Brace clamp plate, top | 846.3 | +0.1790 |
| Mount face (bolts to sole) | 632.3 | −0.0350 |
| Ground plate, top | 259.8 | −0.4075 |
| Ground contact | 224.8 | −0.4425 |

Ground-to-mount-face height is **407.5 mm** — the "40.7 cm" in the filename. The current
stilt is 400 mm, so the robot stands 7.5 mm taller.

**Lateral symmetry.** COM sits exactly on the lateral mid-plane (y = 66.2 mm) and the
off-diagonal inertia terms are ~1e−5 kg·m². The design is laterally symmetric, so **one
mesh serves both feet — no mirroring.** (A naive mirrored-vertex test fails, but only
because round features such as bolt hex heads tessellate asymmetrically.)

**Mass.** 1,659 cm³ of material, 2.8 kg measured, mixed PLA and aluminium. All 66 solids
are individually watertight, so mass properties are computed from the real geometry at a
uniform effective density of 1,688 kg/m³ scaled to hit 2.8 kg.

> **Assumption:** uniform density. The true build mixes PLA plates with aluminium tube and
> steel fasteners, so per-segment COM carries some error — the aluminium tubes sit
> mid-height, the PLA plates at the extremes, so the real COM is likely slightly closer to
> mid-height than computed. If a per-part material assignment becomes available, the
> generator script (§7) recomputes everything from it.

---

## 3. Kinematics — ankle removal

Delete `left/right_ankle_pitch_joint` and `left/right_ankle_roll_joint` from
`assets/mjcf/g1/g1.xml`. **Keep the `ankle_pitch_link` and `ankle_roll_link` bodies.**

Jointless bodies are fused into the shank dynamically, so this is equivalent to reparenting
the stilt onto `knee_link`, but it leaves untouched:

- the `foot_swing_height` contact-sensor subtree rooted at `ankle_roll_link`
- the stilt's parent path, and therefore every geom and site name
- `_STILT_GEOM_NAMES`, the `foot_friction` DR target, and the `STILT_G1_COLLISION` regexes

A new `STILT_G1_ARTICULATION` reuses the G1 actuator set minus `G1_ACTUATOR_ANKLE`.

**Action space drops 29 → 25.**

### 3.1 Consequence: this is genuinely harder to learn

With the ankle welded, shank orientation is rigidly `hip_pitch + knee`. The stilt is
vertical only when `hip_pitch = −knee`. Two things follow:

- The existing keyframe (`hip_pitch −0.312`, `knee 0.669`, `ankle_pitch −0.363`) would spawn
  the stilts tilted ~20°. A new keyframe must satisfy `hip_pitch = −knee` with a shallower
  bend, and the spawn height recomputed from it.
- **There is no ankle strategy at all.** Lateral balance is hip-roll only; fore-aft balance
  is hip/knee only. This is true peg-stilt walking and should be expected to train
  substantially slower than Run 5.

This is a direct consequence of the shank clamp, not a modelling choice. It is accepted.

---

## 4. Body tree and mass properties

Five segments per stilt, split at the hardware's own boundaries. Every child body is
**jointless**, so this adds **zero DoFs** and costs essentially nothing to simulate.

```
ankle_roll_link
└── *_stilt_mount              mount plate + upper collars
    ├── *_stilt_brace          shank brace + clamp plate + cross bolt
    └── *_stilt_post_outer     40×40 outer tube + height-lock bolts
        └── *_stilt_post_inner 35×35 inner tube            ⟵ height DR moves this body
            └── *_stilt_plate  ground plate + brackets + 8 contact capsules + tip site
```

| Segment | n solids | Mass (kg) | COM in ankle frame (m) | `diaginertia` (kg·m²) | z span (m) |
|---|---|---|---|---|---|
| `*_stilt_mount` | 17 | 0.636 | (+0.0449, 0.0000, −0.0490) | 0.00044 0.00276 0.00299 | −0.116…−0.035 |
| `*_stilt_brace` | 16 | 0.999 | (−0.0802, 0.0000, +0.0391) | 0.00071 0.00944 0.00949 | −0.121…+0.179 |
| `*_stilt_post_outer` | 2 | 0.313 | (+0.0400, 0.0000, −0.2119) | 0.00157 0.00168 0.00311 | −0.336…−0.086 |
| `*_stilt_post_inner` | 6 | 0.288 | (+0.0400, −0.0001, −0.2675) | 0.00142 0.00146 0.00277 | −0.391…−0.141 |
| `*_stilt_plate` | 25 | 0.565 | (+0.0400, +0.0002, −0.4255) | 0.00041 0.00288 0.00304 | −0.443…−0.361 |
| **Total** | 66 | **2.800** | | | |

Compare to what this replaces: one body, 1.5 kg, `diaginertia="0.024 0.024 0.003"`.

**The brace is 36% of the stilt mass with its COM 39 mm above the ankle and 80 mm behind
it.** Nothing in the current model represents this, and it is the single largest change to
the leg's swing dynamics.

### 4.1 Visual meshes

`Assembled 40.7cm.STL` is partitioned into five meshes, one per segment, each re-origined
onto its body frame. The five meshes are decimated to ~15k triangles *in total per stilt*
(budget allocated by segment surface complexity) for viewer responsiveness — the raw 99k
× 2 feet makes viser crawl, and collision is capsule-based so physics is unaffected.

---

## 5. Collision

Eight capsules per stilt on `*_stilt_plate`, keeping the existing
`*_stilt_[lr][1-4]_collision` names so `_STILT_GEOM_NAMES`, the `foot_friction` DR target
and `STILT_G1_COLLISION` need no edits.

- axis along y, `fromto` y −0.030 … +0.030
- z = −0.4325, radius 0.01 → contact at −0.4425
- x stations: −0.060, −0.030, 0.000, 0.030, 0.060, 0.090, 0.120, 0.145

Posts, plates and brace are visual-only. They are rigid relative to the shank, so
self-collision against the leg is meaningless.

---

## 6. Height randomisation

The telescoping split is what makes this cheap: `dr.body_pos` on `*_stilt_post_inner` z
moves the inner tube, ground plate, contact capsules and tip site together as one, and the
visual telescopes correctly because the meshes are already split at that boundary.

**Mechanical range.** Nominal overlap is 195 mm of the 250 mm tube. Full insertion
(250 mm overlap) gives 352.5 mm; 80 mm overlap gives 522.5 mm.

> **Open item:** 80 mm is an assumed minimum safe overlap. Confirm the real usable range
> before widening the curriculum past ±50 mm.

A `stilt_height_curriculum` mirrors the existing mass curriculum: fixed 407.5 mm → ±20 mm
→ ±50 mm → full confirmed range.

**Spawn height (gate — investigated 2026-08-07, resolved).** The keyframe *cannot* see the
sampled height: `ManagerBasedRlEnv._reset_idx` calls `self.scene.reset(env_ids)` — which
applies `init_state` — **before** it applies `mode="reset"` events. So sampling the height
in an event happens after the robot has already been placed.

Resolution: a second reset event, `stilt_spawn_height`, registered *after* `stilt_height`
in the `cfg.events` dict. `EventManager` builds `_mode_term_cfgs` by iterating the config
dict, so dict insertion order is execution order, and both terms land after the base env's
`reset_base` (`reset_root_state_uniform`) which sets the nominal root pose.

`stilt_spawn_height` reads back the sampled `env.sim.model.body_pos[env_ids, bid, 2]` for
the `*_stilt_post_inner` bodies, differences it against the authored nominal (a constant we
own, since we author the MJCF), and raises the root by that amount via
`asset.write_root_link_pose_to_sim`. A taller stilt means a more negative body_pos z, so
the correction is `root_z -= delta`.

No mjlab source changes required.

---

## 7. Mesh/inertia generator script

`scripts/build_stilt_meshes.py` — single source of truth, rerun whenever the CAD changes:

1. Load the source STL, split into connected components
2. Assign each component to a segment by nearest-anchor-part centroid
3. Scale uniform density to the measured total mass (CLI arg, default 2.8 kg)
4. Emit five re-origined, decimated STLs into `assets/mjcf/g1/assets/`
5. Print the ready-to-paste MJCF `<inertial>` blocks and the segment table above

This keeps §4's numbers reproducible rather than hand-transcribed, and makes a per-part
material assignment a one-line change later.

---

## 8. Load instrumentation

The point is to see where the hardware is actually loaded during walking.

### 8.1 What is computed

- **Ground reaction distribution** — force in each of the 8 contact capsules per foot,
  giving heel/toe and left/right pressure distribution directly.
- **Section loads** — at each of the 5 segment interfaces: axial force, shear magnitude,
  bending moment magnitude, torsion.

### 8.2 How

Computed in Python inside the viewer GUI at 10 Hz on the single play env, from the contact
force array plus each segment's inertial terms. At one env and 10 Hz this is free.

**This deliberately does not depend on `mujoco_warp` implementing `force`/`torque`
sensors,** which is unverified. MJCF `<force>`/`<torque>` sensors on per-segment sites are
added **only if** they are confirmed to work under mjwarp — in which case they are a bonus
that could later feed training observations. The viewer readout is the deliverable.

### 8.3 Limitation: the sole/clamp split is not recoverable

With the ankle joints deleted, shank and foot are one rigid body. How the interface load
divides between the sole bolts and the shank clamp is **statically indeterminate** — the
sim reports the *total* wrench through the interface, not the split. Getting the split
requires either the rejected closed-loop equality-constraint model or a hand calc / FEA.

The `*_stilt_brace` sensor therefore reports the brace's own inertial load only, **not**
the real clamp reaction. This must be labelled as such in the GUI so the number is not
misread.

---

## 9. Viewer GUI

Extends the existing `_stilt_mass_play_gui(server, env)` pattern in
`envs/stilt_g1/__init__.py`. All folders update while paused, matching the existing torque
monitor behaviour.

| Folder | Contents |
|---|---|
| **Stilt Mass** | One slider per segment (5) + a master multiplier + the existing randomise-on-reset checkbox. Applied live via `dr.pseudo_inertia` followed by an explicit `recompute_constants` (required in viewer context — the `@requires_model_fields` decorator does not auto-trigger). Per-segment sim-mass readback. |
| **Stilt Loads** | Per section: axial / shear / bending / torsion, bar + % of a configurable limit. Brace row explicitly labelled "inertial only — not clamp reaction". |
| **Ground Pressure** | Per-capsule force, 8 per foot. |
| **Joint Torques** | Existing monitor, **retargeted** — it currently reads `left/right_ankle_pitch_joint`, which will no longer exist. Hip and knee only. |

Limits and slider ceilings live as module-level constants beside the existing
`MASS_PLAY_MAX_KG`.

---

## 10. Downstream changes

| File | Change |
|---|---|
| `assets/mjcf/g1/g1.xml` | Ankle joints deleted; stilt replaced by the 5-segment tree; new meshes, inertials, capsules, tip site |
| `assets/mjcf/g1/assets/` | 5 new segment STLs; old `stilt.STL` retired |
| `envs/stilt_g1/stilt_robot.py` | `STILT_G1_ARTICULATION` without the ankle actuator; keyframe with `hip_pitch = −knee` and no ankle entry; recomputed spawn height; new action-scale dict |
| `envs/stilt_g1/env_cfgs.py` | Mass curriculum targets all 10 stilt bodies, alpha rebased on 2.8 kg; height curriculum added; `torso_too_low` threshold revisited for the new stance |
| `envs/stilt_g1/curriculums.py` | New `stilt_height_curriculum` |
| `envs/stilt_g1/__init__.py` | GUI per §9; torque monitor retargeted off the deleted ankle joints |
| `scripts/build_stilt_meshes.py` | New (§7) |
| `deploy/config/g1_stilt/deploy.yaml` | 25-dim action list, ankles removed |
| `deploy/README.md` | The real ankle motors must be held in damping mode, not PD-tracked into the clamp |
| `CLAUDE.md`, `STATUS.md` | Updated to describe the new stilt |

**The Run 5 checkpoint becomes invalid.** Action space, mass, inertia, height and stance
all change. A fresh training run is required.

---

## 11. Verification

1. `MjSpec.from_file` compiles with no warnings; `mj_forward` runs clean.
2. Assertions: total robot mass increased by exactly 2 × (2.8 − 1.5) kg; `nq`/`nu` dropped
   by 4; both tip sites at z = −0.4425 in ankle frame; no ankle joint names resolve.
3. Spawn check: at the new keyframe, both stilt plates rest flat on the floor with zero
   initial penetration and near-zero initial contact impulse.
4. Height DR check (gate for §6): sample the full height range across envs and confirm
   every env spawns without floor intersection.
5. Viewer check at `--num-envs 1`: stilts render telescoped correctly; per-segment mass
   sliders move `body_mass` **and** `cinert[9]`; `qfrc_bias` at hip/knee responds
   proportionally to the mass sliders; ground-pressure bars sum to body weight in a static
   stand.
6. `uv run ruff format && uv run ruff check --fix` clean.

---

## 12. Decisions taken

| Decision | Choice | Rationale |
|---|---|---|
| Scope | Update sim, then retrain | Sim must match the hardware being built |
| Brace | Part of the stilt | Real hardware, clamps the shank |
| Ankle DoFs | Delete joints + actuator | Physically honest; keeping dead actuated joints would fight the clamp on hardware |
| Height | Randomise | One policy across the telescoping range |
| Forces | Read out, not apply | Structural visibility during testing |
| Sole/clamp split | Not modelled | Statically indeterminate in a rigid model |
| Mirroring | None | Design is laterally symmetric |

## 13. Open items

1. Confirmed minimum safe tube overlap (assumed 80 mm).
2. Per-part material assignment, to refine per-segment COM.
3. Whether mjwarp supports `force`/`torque` sensors — determines whether §8 sensors are
   added to the MJCF as a bonus.
4. Keyframe knee angle — chosen during implementation to balance stance height against
   ground clearance, under the `hip_pitch = −knee` constraint.
