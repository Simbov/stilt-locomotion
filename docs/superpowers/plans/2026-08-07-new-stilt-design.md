# New Stilt Design Implementation Plan

> ### ⚠️ Executed, and then partly reversed on 2026-08-13
>
> Every task here was carried out, but the ankle-removal work (Task 2 onward,
> wherever it deletes joints or drops the action space to 25) was **undone**: the
> robot is always the stock 29-DoF G1 and the stilts bolt on and off. The brace
> is now modelled as randomised ankle joint stiffness instead of a weld. See the
> banner on the matching spec, and `STATUS.md`, for the current design. Read this
> plan as a record of how the CAD partition, body tree, contact geometry and
> instrumentation were built — all of which still stand.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the single-box stilt with the real telescoping shank-clamped hardware — a 5-segment rigid body tree with correct mass properties, welded ankles, randomised height, and live load/mass instrumentation in the viewer.

**Architecture:** A generator script partitions the source STL into five segment meshes and emits their exact inertial blocks, so `g1.xml` is regenerated rather than hand-tuned. The MJCF gains a nested 5-body chain per stilt (all jointless — zero added DoFs) and loses the four ankle joints. Height randomisation moves one body via `dr.body_pos`, with a follow-up reset event correcting spawn height. Load readout is pure Python in the viewer, computed from contact forces and segment inertia, so it depends on no `mujoco_warp` sensor support.

**Tech Stack:** Python 3.12, uv, MuJoCo 3.10 / MjSpec, mjlab 1.5.0, mujoco_warp, trimesh, viser, PyTorch.

## Global Constraints

- Source CAD: `/Users/simonvollert/Downloads/Assembled 40.7cm.STL`, mm units, 66 watertight solids.
- Mesh point `(114.1, 66.2, 667.3)` maps to the `ankle_roll_link` origin.
- Total stilt mass **2.800 kg** per side; uniform effective density **1688 kg/m³**.
- Ground contact at ankle-frame z **−0.4425 m**; mount face at **−0.0350 m**.
- Contact geom names must stay `(left|right)_stilt_[lr][1-4]_collision` — `_STILT_GEOM_NAMES`, `foot_friction` DR and `STILT_G1_COLLISION` all match on them.
- Tip site names must stay `left_stilt_tip` / `right_stilt_tip`.
- Do **not** modify anything under `mjlab/` or `.venv/`. Extend from project code only (CLAUDE.md).
- Run `uv run ruff format && uv run ruff check --fix` before every commit.
- Segment body names, fixed across all tasks:
  `*_stilt_mount`, `*_stilt_brace`, `*_stilt_post_outer`, `*_stilt_post_inner`, `*_stilt_plate` (`*` ∈ {`left`, `right`}).
- Segment mesh names: `stilt_mount`, `stilt_brace`, `stilt_post_outer`, `stilt_post_inner`, `stilt_plate`.

---

### Task 0: Test harness

The repo has no test framework. Everything downstream needs one.

**Files:**
- Modify: `pyproject.toml`
- Create: `tests/conftest.py`

**Interfaces:**
- Produces: pytest available via `uv run pytest`; fixture `stilt_model` returning a compiled `(mujoco.MjModel, mujoco.MjData)` for `assets/mjcf/g1/g1.xml`.

- [ ] **Step 1: Add pytest as a dev dependency**

```bash
uv add --dev pytest
```

- [ ] **Step 2: Create the shared fixture**

```python
# tests/conftest.py
"""Shared fixtures for stilt-locomotion tests."""

from pathlib import Path

import mujoco
import pytest

REPO_ROOT = Path(__file__).parent.parent
G1_XML = REPO_ROOT / "assets" / "mjcf" / "g1" / "g1.xml"

SEGMENTS = (
  "stilt_mount",
  "stilt_brace",
  "stilt_post_outer",
  "stilt_post_inner",
  "stilt_plate",
)


@pytest.fixture(scope="session")
def stilt_model():
  """Compile the stilt G1 MJCF and run one forward pass."""
  spec = mujoco.MjSpec.from_file(str(G1_XML))
  model = spec.compile()
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  return model, data
```

- [ ] **Step 3: Verify the harness runs against the current model**

Run: `uv run pytest tests/ -v`
Expected: `no tests ran` (collection succeeds, zero tests). Any import or compile error here is a real problem — fix before continuing.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock tests/conftest.py
git commit -m "test: add pytest harness and stilt model fixture"
```

---

### Task 1: Mesh and inertia generator

Single source of truth for §4 and §7 of the spec. Rerun whenever the CAD changes.

**Files:**
- Create: `scripts/build_stilt_meshes.py`
- Create: `tests/test_build_stilt_meshes.py`
- Output: `assets/mjcf/g1/assets/stilt_{mount,brace,post_outer,post_inner,plate}.STL`

**Interfaces:**
- Produces:
  - `SEGMENTS: tuple[str, ...]` — the five names in the order above.
  - `MESH_ORIGIN_MM = (114.1, 66.2, 667.3)`
  - `partition(mesh: trimesh.Trimesh) -> dict[str, list[trimesh.Trimesh]]`
  - `segment_properties(parts: list[trimesh.Trimesh], density: float, origin_mm: tuple[float, float, float]) -> SegmentProps`
  - `@dataclass SegmentProps: name: str; mass: float; com: tuple[float,float,float]; diaginertia: tuple[float,float,float]; quat: tuple[float,float,float,float]; z_min: float; z_max: float`
  - CLI: `uv run python scripts/build_stilt_meshes.py --source <stl> --total-mass 2.8 --out assets/mjcf/g1/assets`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_build_stilt_meshes.py
"""The generator must reproduce the segment masses recorded in the spec."""

import trimesh
import pytest

from scripts.build_stilt_meshes import (
  MESH_ORIGIN_MM,
  SEGMENTS,
  SOURCE_STL,
  partition,
  segment_properties,
  uniform_density,
)

# From docs/superpowers/specs/2026-08-07-new-stilt-design.md §4.
EXPECTED_MASS_KG = {
  "stilt_mount": 0.636,
  "stilt_brace": 0.999,
  "stilt_post_outer": 0.313,
  "stilt_post_inner": 0.288,
  "stilt_plate": 0.565,
}
TOTAL_MASS_KG = 2.8


@pytest.fixture(scope="module")
def parts():
  mesh = trimesh.load(str(SOURCE_STL), force="mesh")
  return partition(mesh), uniform_density(mesh, TOTAL_MASS_KG)


def test_every_segment_is_populated(parts):
  groups, _ = parts
  assert set(groups) == set(SEGMENTS)
  for name in SEGMENTS:
    assert groups[name], f"segment {name} got no solids"


def test_partition_is_a_complete_disjoint_cover(parts):
  groups, _ = parts
  assigned = sum(len(v) for v in groups.values())
  assert assigned == 66, f"expected all 66 solids assigned, got {assigned}"


def test_segment_masses_match_spec(parts):
  groups, density = parts
  for name, expected in EXPECTED_MASS_KG.items():
    props = segment_properties(groups[name], density, MESH_ORIGIN_MM)
    assert props.mass == pytest.approx(expected, abs=0.002), name


def test_total_mass_is_conserved(parts):
  groups, density = parts
  total = sum(
    segment_properties(groups[n], density, MESH_ORIGIN_MM).mass for n in SEGMENTS
  )
  assert total == pytest.approx(TOTAL_MASS_KG, abs=0.005)


def test_ground_contact_is_at_minus_442_5_mm(parts):
  groups, density = parts
  plate = segment_properties(groups["stilt_plate"], density, MESH_ORIGIN_MM)
  assert plate.z_min == pytest.approx(-0.4425, abs=1e-4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_build_stilt_meshes.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.build_stilt_meshes'`

- [ ] **Step 3: Implement the generator**

Partition rule: five anchor part types are identified by their bounding-box extents (rounded to whole mm); every one of the 66 solids is then assigned to the segment whose anchor centroid is nearest in 3D. This is the rule that produced the spec's numbers.

```python
# scripts/build_stilt_meshes.py
"""Partition the assembled stilt STL into per-segment meshes and inertials.

Single source of truth for the stilt segment geometry in assets/mjcf/g1/g1.xml.
Rerun whenever the CAD changes:

    uv run python scripts/build_stilt_meshes.py

It writes one decimated STL per segment and prints ready-to-paste MJCF
<inertial> blocks.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh

REPO_ROOT = Path(__file__).parent.parent
SOURCE_STL = Path.home() / "Downloads" / "Assembled 40.7cm.STL"
DEFAULT_OUT = REPO_ROOT / "assets" / "mjcf" / "g1" / "assets"

SEGMENTS = (
  "stilt_mount",
  "stilt_brace",
  "stilt_post_outer",
  "stilt_post_inner",
  "stilt_plate",
)

# Mesh point (mm) that maps onto the ankle_roll_link origin.
MESH_ORIGIN_MM = (114.1, 66.2, 667.3)

# Triangle budget per segment, ~15k total per stilt (spec §4.1).
FACE_BUDGET = {
  "stilt_mount": 3000,
  "stilt_brace": 3000,
  "stilt_post_outer": 2000,
  "stilt_post_inner": 2000,
  "stilt_plate": 5000,
}

# Anchor parts, identified by bounding-box extents in mm (rounded).
_ANCHOR_EXTENTS: dict[tuple[float, float, float], str] = {
  (220.0, 110.0, 35.0): "stilt_mount",
  (30.0, 60.0, 200.0): "stilt_brace",
  (30.0, 60.0, 150.0): "stilt_brace",
  (45.0, 120.0, 30.0): "stilt_brace",
  (40.0, 40.0, 250.0): "stilt_post_outer",
  (35.0, 35.0, 250.0): "stilt_post_inner",
  (220.0, 80.0, 35.0): "stilt_plate",
}


@dataclass(frozen=True)
class SegmentProps:
  """Rigid-body properties of one stilt segment, in ankle-frame metres."""

  name: str
  mass: float
  com: tuple[float, float, float]
  diaginertia: tuple[float, float, float]
  quat: tuple[float, float, float, float]
  z_min: float
  z_max: float


def uniform_density(mesh: trimesh.Trimesh, total_mass_kg: float) -> float:
  """Density (kg/m^3) that makes the summed solid volume weigh total_mass_kg."""
  volume_mm3 = sum(abs(c.volume) for c in mesh.split(only_watertight=False))
  return total_mass_kg / (volume_mm3 * 1e-9)


def partition(mesh: trimesh.Trimesh) -> dict[str, list[trimesh.Trimesh]]:
  """Assign every connected solid to a segment by nearest anchor centroid."""
  components = mesh.split(only_watertight=False)

  anchors: list[tuple[np.ndarray, str]] = []
  for comp in components:
    key = tuple(np.round(comp.extents, 0))
    name = _ANCHOR_EXTENTS.get(key)  # type: ignore[arg-type]
    if name is not None:
      anchors.append((comp.center_mass, name))
  if len(anchors) != 12:
    raise RuntimeError(
      f"expected 12 anchor solids, found {len(anchors)} — CAD changed, "
      "update _ANCHOR_EXTENTS"
    )

  groups: dict[str, list[trimesh.Trimesh]] = {name: [] for name in SEGMENTS}
  for comp in components:
    centroid = comp.center_mass
    nearest = min(anchors, key=lambda a: float(np.linalg.norm(centroid - a[0])))
    groups[nearest[1]].append(comp)
  return groups


def segment_properties(
  parts: list[trimesh.Trimesh],
  density: float,
  origin_mm: tuple[float, float, float],
) -> SegmentProps:
  """Mass, COM and principal inertia of a segment, expressed in ankle frame."""
  origin_m = np.asarray(origin_mm) * 1e-3

  mass = 0.0
  weighted_com = np.zeros(3)
  for part in parts:
    part_mass = abs(part.volume) * 1e-9 * density
    mass += part_mass
    weighted_com += part_mass * part.center_mass * 1e-3
  com = weighted_com / mass

  inertia = np.zeros((3, 3))
  for part in parts:
    part_mass = abs(part.volume) * 1e-9 * density
    # trimesh reports inertia for unit density; rescale to this part's mass.
    part_inertia = part.moment_inertia / abs(part.volume) * 1e-6 * part_mass
    offset = part.center_mass * 1e-3 - com
    inertia += part_inertia + part_mass * (
      np.dot(offset, offset) * np.eye(3) - np.outer(offset, offset)
    )

  eigenvalues, eigenvectors = np.linalg.eigh(inertia)
  if np.linalg.det(eigenvectors) < 0:
    eigenvectors[:, 0] *= -1.0
  quat = trimesh.transformations.quaternion_from_matrix(
    np.pad(eigenvectors, ((0, 1), (0, 1)))[:4, :4] + np.diag([0, 0, 0, 1.0])
  )

  z_values = [p.bounds[:, 2] for p in parts]
  return SegmentProps(
    name="",
    mass=float(mass),
    com=tuple((com - origin_m).tolist()),
    diaginertia=tuple(eigenvalues.tolist()),
    quat=tuple(quat.tolist()),
    z_min=float(min(z[0] for z in z_values) * 1e-3 - origin_m[2]),
    z_max=float(max(z[1] for z in z_values) * 1e-3 - origin_m[2]),
  )


def _export(
  parts: list[trimesh.Trimesh],
  name: str,
  origin_mm: tuple[float, float, float],
  out_dir: Path,
) -> Path:
  """Merge, re-origin and decimate one segment, then write it as STL."""
  merged = trimesh.util.concatenate(parts)
  merged.apply_translation(-np.asarray(origin_mm))

  budget = FACE_BUDGET[name]
  if len(merged.faces) > budget:
    merged = merged.simplify_quadric_decimation(face_count=budget)

  path = out_dir / f"{name}.STL"
  merged.export(path)
  return path


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--source", type=Path, default=SOURCE_STL)
  parser.add_argument("--total-mass", type=float, default=2.8)
  parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
  args = parser.parse_args()

  mesh = trimesh.load(str(args.source), force="mesh")
  density = uniform_density(mesh, args.total_mass)
  groups = partition(mesh)

  args.out.mkdir(parents=True, exist_ok=True)
  print(f"effective uniform density: {density:.0f} kg/m^3\n")

  total = 0.0
  for name in SEGMENTS:
    props = segment_properties(groups[name], density, MESH_ORIGIN_MM)
    path = _export(groups[name], name, MESH_ORIGIN_MM, args.out)
    total += props.mass
    print(f"<!-- {name}: {len(groups[name])} solids, "
          f"z {props.z_min:+.3f}..{props.z_max:+.3f} m, {path.name} -->")
    print(
      f'<inertial pos="{props.com[0]:.5f} {props.com[1]:.5f} {props.com[2]:.5f}"\n'
      f'  quat="{props.quat[0]:.6f} {props.quat[1]:.6f} '
      f'{props.quat[2]:.6f} {props.quat[3]:.6f}"\n'
      f'  mass="{props.mass:.4f}"\n'
      f'  diaginertia="{props.diaginertia[0]:.6f} '
      f'{props.diaginertia[1]:.6f} {props.diaginertia[2]:.6f}"/>\n'
    )
  print(f"total mass: {total:.3f} kg")


if __name__ == "__main__":
  main()
```

- [ ] **Step 4: Make `scripts/` importable**

`tests/` imports `scripts.build_stilt_meshes`, which needs `scripts/__init__.py`:

```bash
touch scripts/__init__.py
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_build_stilt_meshes.py -v`
Expected: 5 passed.

If `test_partition_is_a_complete_disjoint_cover` fails with a count other than 66, the
source STL is not the file the spec was measured against — stop and confirm the CAD before
continuing.

- [ ] **Step 6: Generate the meshes**

Run: `uv run python scripts/build_stilt_meshes.py`
Expected: five `<inertial>` blocks printed, `total mass: 2.800 kg`, and five STLs written
into `assets/mjcf/g1/assets/`. **Keep this output** — Task 2 pastes it into the MJCF.

- [ ] **Step 7: Commit**

```bash
uv run ruff format && uv run ruff check --fix
git add scripts/build_stilt_meshes.py scripts/__init__.py tests/test_build_stilt_meshes.py assets/mjcf/g1/assets/stilt_*.STL
git commit -m "feat(stilt): add segment mesh/inertia generator and generated meshes"
```

---

### Task 2: MJCF — 5-segment tree and ankle removal

**Files:**
- Modify: `assets/mjcf/g1/g1.xml` (mesh assets ~line 46; left leg 110-144; right leg ~179-213)
- Create: `tests/test_stilt_mjcf.py`

**Interfaces:**
- Produces: bodies `(left|right)_stilt_{mount,brace,post_outer,post_inner,plate}`; geoms `(left|right)_stilt_[lr][1-4]_collision`; sites `(left|right)_stilt_tip`; **no** `.*_ankle_(pitch|roll)_joint` joints.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_stilt_mjcf.py
"""The stilt MJCF must match the geometry and topology in the spec."""

import mujoco
import pytest

SEGMENT_BODIES = (
  "stilt_mount",
  "stilt_brace",
  "stilt_post_outer",
  "stilt_post_inner",
  "stilt_plate",
)
SIDES = ("left", "right")


def _body_id(model, name):
  return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)


def test_ankle_joints_are_gone(stilt_model):
  model, _ = stilt_model
  names = {
    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    for i in range(model.njnt)
  }
  offenders = {n for n in names if n and "ankle" in n}
  assert not offenders, f"ankle joints still present: {offenders}"


def test_ankle_links_survive(stilt_model):
  """The bodies must stay — the foot_swing_height contact subtree depends on them."""
  model, _ = stilt_model
  for side in SIDES:
    assert _body_id(model, f"{side}_ankle_roll_link") >= 0


def test_all_segment_bodies_exist(stilt_model):
  model, _ = stilt_model
  for side in SIDES:
    for segment in SEGMENT_BODIES:
      assert _body_id(model, f"{side}_{segment}") >= 0, f"{side}_{segment}"


def test_segments_add_no_degrees_of_freedom(stilt_model):
  model, _ = stilt_model
  for side in SIDES:
    for segment in SEGMENT_BODIES:
      bid = _body_id(model, f"{side}_{segment}")
      assert model.body_dofnum[bid] == 0, f"{side}_{segment} has DoFs"


def test_segment_chain_is_nested_correctly(stilt_model):
  model, _ = stilt_model
  expected_parent = {
    "stilt_mount": "ankle_roll_link",
    "stilt_brace": "stilt_mount",
    "stilt_post_outer": "stilt_mount",
    "stilt_post_inner": "stilt_post_outer",
    "stilt_plate": "stilt_post_inner",
  }
  for side in SIDES:
    for child, parent in expected_parent.items():
      bid = _body_id(model, f"{side}_{child}")
      pid = model.body_parentid[bid]
      assert mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, pid) == (
        f"{side}_{parent}"
      ), f"{side}_{child}"


def test_stilt_mass_per_side(stilt_model):
  model, _ = stilt_model
  for side in SIDES:
    total = sum(
      model.body_mass[_body_id(model, f"{side}_{s}")] for s in SEGMENT_BODIES
    )
    assert total == pytest.approx(2.8, abs=0.01), side


def test_tip_sites_at_ground_contact(stilt_model):
  model, data = stilt_model
  for side in SIDES:
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_stilt_tip")
    assert sid >= 0
    ankle = _body_id(model, f"{side}_ankle_roll_link")
    relative_z = data.site_xpos[sid][2] - data.xpos[ankle][2]
    assert relative_z == pytest.approx(-0.4425, abs=1e-3), side


def test_contact_capsule_names_are_preserved(stilt_model):
  """_STILT_GEOM_NAMES, foot_friction DR and STILT_G1_COLLISION match on these."""
  model, _ = stilt_model
  names = {
    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) for i in range(model.ngeom)
  }
  for side in SIDES:
    for block in ("l", "r"):
      for i in range(1, 5):
        assert f"{side}_stilt_{block}{i}_collision" in names


def test_contact_capsules_reach_the_ground(stilt_model):
  model, data = stilt_model
  for side in SIDES:
    ankle = _body_id(model, f"{side}_ankle_roll_link")
    for block in ("l", "r"):
      for i in range(1, 5):
        gid = mujoco.mj_name2id(
          model, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_stilt_{block}{i}_collision"
        )
        bottom = data.geom_xpos[gid][2] - model.geom_size[gid][0]
        assert bottom - data.xpos[ankle][2] == pytest.approx(-0.4425, abs=1e-3)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_stilt_mjcf.py -v`
Expected: FAIL — the segment bodies do not exist and ankle joints are still present.

- [ ] **Step 3: Replace the mesh assets block**

In `assets/mjcf/g1/g1.xml`, replace the single stilt `<mesh>` (around line 46) with five
entries. The generator already re-origined the meshes onto the ankle frame, so **no
`refpos` and no `scale`** — they are exported in metres-ready mm and scaled here only:

```xml
<!-- Stilt segments — generated by scripts/build_stilt_meshes.py.
     Already re-origined so mesh (114.1, 66.2, 667.3) sits at the ankle_roll_link
     origin; only the mm→m scale remains. -->
<mesh name="stilt_mount" file="stilt_mount.STL" scale="0.001 0.001 0.001"/>
<mesh name="stilt_brace" file="stilt_brace.STL" scale="0.001 0.001 0.001"/>
<mesh name="stilt_post_outer" file="stilt_post_outer.STL" scale="0.001 0.001 0.001"/>
<mesh name="stilt_post_inner" file="stilt_post_inner.STL" scale="0.001 0.001 0.001"/>
<mesh name="stilt_plate" file="stilt_plate.STL" scale="0.001 0.001 0.001"/>
```

- [ ] **Step 4: Delete the ankle joints**

Remove these four lines (two per leg):

```xml
<joint name="left_ankle_pitch_joint" axis="0 1 0" range="-0.87267 0.5236"/>
<joint name="left_ankle_roll_joint" axis="1 0 0" range="-0.2618 0.2618"/>
<joint name="right_ankle_pitch_joint" axis="0 1 0" range="-0.87267 0.5236"/>
<joint name="right_ankle_roll_joint" axis="1 0 0" range="-0.2618 0.2618"/>
```

Leave the `left/right_ankle_pitch_link` and `left/right_ankle_roll_link` bodies in place.

- [ ] **Step 5: Replace the left stilt body**

Replace the whole `<body name="left_stilt" ...>...</body>` block (lines 129-143) with the
nested chain below. Paste the `<inertial>` blocks printed by Task 1 Step 6 in place of the
values shown here — these are the spec's values and should match to the digits shown:

```xml
<body name="left_stilt_mount" pos="0 0 0">
  <inertial pos="0.04490 0.00000 -0.04900" mass="0.6360"
    diaginertia="0.000440 0.002760 0.002990"/>
  <geom class="visual" material="silver" mesh="stilt_mount"/>

  <body name="left_stilt_brace" pos="0 0 0">
    <inertial pos="-0.08020 0.00000 0.03910" mass="0.9990"
      diaginertia="0.000710 0.009440 0.009490"/>
    <geom class="visual" material="silver" mesh="stilt_brace"/>
  </body>

  <body name="left_stilt_post_outer" pos="0 0 0">
    <inertial pos="0.04000 0.00000 -0.21190" mass="0.3130"
      diaginertia="0.001570 0.001680 0.003110"/>
    <geom class="visual" material="silver" mesh="stilt_post_outer"/>

    <!-- pos z is the telescope offset; dr.body_pos randomises it (spec §6).
         Nominal 0 == the assembled 407.5 mm configuration. -->
    <body name="left_stilt_post_inner" pos="0 0 0">
      <inertial pos="0.04000 -0.00010 -0.26750" mass="0.2880"
        diaginertia="0.001420 0.001460 0.002770"/>
      <geom class="visual" material="silver" mesh="stilt_post_inner"/>

      <body name="left_stilt_plate" pos="0 0 0">
        <inertial pos="0.04000 0.00020 -0.42550" mass="0.5650"
          diaginertia="0.000410 0.002880 0.003040"/>
        <geom class="visual" material="silver" mesh="stilt_plate"/>

        <!-- Ground contact: z -0.4325 with radius 0.01 → contact at -0.4425. -->
        <geom name="left_stilt_l1_collision" class="foot_capsule" size="0.01" rgba=".2 .6 .2 .5" fromto="-0.060 -0.030 -0.4325  -0.060 0.030 -0.4325"/>
        <geom name="left_stilt_l2_collision" class="foot_capsule" size="0.01" rgba=".2 .6 .2 .5" fromto="-0.030 -0.030 -0.4325  -0.030 0.030 -0.4325"/>
        <geom name="left_stilt_l3_collision" class="foot_capsule" size="0.01" rgba=".2 .6 .2 .5" fromto=" 0.000 -0.030 -0.4325   0.000 0.030 -0.4325"/>
        <geom name="left_stilt_l4_collision" class="foot_capsule" size="0.01" rgba=".2 .6 .2 .5" fromto=" 0.030 -0.030 -0.4325   0.030 0.030 -0.4325"/>
        <geom name="left_stilt_r1_collision" class="foot_capsule" size="0.01" rgba=".2 .6 .2 .5" fromto=" 0.060 -0.030 -0.4325   0.060 0.030 -0.4325"/>
        <geom name="left_stilt_r2_collision" class="foot_capsule" size="0.01" rgba=".2 .6 .2 .5" fromto=" 0.090 -0.030 -0.4325   0.090 0.030 -0.4325"/>
        <geom name="left_stilt_r3_collision" class="foot_capsule" size="0.01" rgba=".2 .6 .2 .5" fromto=" 0.120 -0.030 -0.4325   0.120 0.030 -0.4325"/>
        <geom name="left_stilt_r4_collision" class="foot_capsule" size="0.01" rgba=".2 .6 .2 .5" fromto=" 0.145 -0.030 -0.4325   0.145 0.030 -0.4325"/>

        <site name="left_stilt_tip" pos="0.04 0 -0.4425"/>
      </body>
    </body>
  </body>
</body>
```

- [ ] **Step 6: Replace the right stilt body**

Replace the `<body name="right_stilt" ...>` block identically, substituting `right_` for
`left_` in every body, geom and site name. The design is laterally symmetric (spec §2), so
**all numeric values are unchanged** — do not mirror any sign.

- [ ] **Step 7: Run tests to verify they pass**

Run: `uv run pytest tests/test_stilt_mjcf.py -v`
Expected: 9 passed.

- [ ] **Step 8: Commit**

```bash
git add assets/mjcf/g1/g1.xml tests/test_stilt_mjcf.py
git commit -m "feat(stilt): 5-segment stilt tree, weld ankles"
```

---

### Task 3: Robot config — articulation, keyframe, spawn height

**Files:**
- Modify: `envs/stilt_g1/stilt_robot.py`
- Create: `tests/test_stilt_robot_cfg.py`

**Interfaces:**
- Consumes: MJCF from Task 2.
- Produces:
  - `STILT_G1_ARTICULATION: EntityArticulationInfoCfg` — G1 actuators minus `G1_ACTUATOR_ANKLE`.
  - `STILT_G1_ACTION_SCALE: dict[str, float]` — rebuilt from `STILT_G1_ARTICULATION`.
  - `STILT_NOMINAL_POST_INNER_Z: float = 0.0` — nominal `body_pos` z of `*_stilt_post_inner`; Task 5 differences against it.
  - `STILT_KNEE_ANGLE: float`, `STILT_SPAWN_HEIGHT: float`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_stilt_robot_cfg.py
"""The stilt robot config must drop the ankle actuator and stand the stilts upright."""

import pytest

from envs.stilt_g1.stilt_robot import (
  STILT_G1_ACTION_SCALE,
  STILT_G1_ARTICULATION,
  STILT_G1_KEYFRAME,
  STILT_KNEE_ANGLE,
)


def test_ankle_actuator_is_absent():
  patterns = [
    p for a in STILT_G1_ARTICULATION.actuators for p in a.target_names_expr
  ]
  assert not [p for p in patterns if "ankle" in p]


def test_action_scale_has_no_ankle_entries():
  assert not [k for k in STILT_G1_ACTION_SCALE if "ankle" in k]


def test_keyframe_has_no_ankle_target():
  assert not [k for k in STILT_G1_KEYFRAME.joint_pos if "ankle" in k]


def test_keyframe_keeps_the_shank_vertical():
  """With welded ankles the stilt is upright only when hip_pitch == -knee."""
  hip = STILT_G1_KEYFRAME.joint_pos[".*_hip_pitch_joint"]
  knee = STILT_G1_KEYFRAME.joint_pos[".*_knee_joint"]
  assert hip == pytest.approx(-knee, abs=1e-9)
  assert knee == pytest.approx(STILT_KNEE_ANGLE, abs=1e-9)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_stilt_robot_cfg.py -v`
Expected: FAIL — `ImportError: cannot import name 'STILT_G1_ARTICULATION'`

- [ ] **Step 3: Rewrite the config**

Replace the keyframe, collision and articulation section of
`envs/stilt_g1/stilt_robot.py` with:

```python
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import (
  G1_ACTUATOR_4010,
  G1_ACTUATOR_5020,
  G1_ACTUATOR_7520_14,
  G1_ACTUATOR_7520_22,
  G1_ACTUATOR_WAIST,
)
from mjlab.entity import EntityArticulationInfoCfg

# The shank brace clamps the calf, so ankle pitch/roll cannot move and their
# joints are deleted from the MJCF. Drop G1_ACTUATOR_ANKLE to match:
# action space 29 -> 25.
STILT_G1_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    G1_ACTUATOR_5020,
    G1_ACTUATOR_7520_14,
    G1_ACTUATOR_7520_22,
    G1_ACTUATOR_4010,
    G1_ACTUATOR_WAIST,
  ),
  soft_joint_pos_limit_factor=0.9,
)

STILT_G1_ACTION_SCALE: dict[str, float] = {}
for _actuator in STILT_G1_ARTICULATION.actuators:
  assert isinstance(_actuator, BuiltinPositionActuatorCfg)
  assert _actuator.effort_limit is not None
  for _name in _actuator.target_names_expr:
    STILT_G1_ACTION_SCALE[_name] = (
      0.25 * _actuator.effort_limit / _actuator.stiffness
    )

# With the ankle welded, shank orientation is rigidly hip_pitch + knee, so the
# stilt is vertical only when hip_pitch == -knee (spec §3.1). A shallower bend
# than the old 0.669 keeps the stilt upright without crouching excessively.
STILT_KNEE_ANGLE = 0.30

# Ankle_roll_link sits 0.30001 + 0.017558 m below the knee origin; the stilt
# adds 0.4425 m below that. Pelvis height is solved in tests/verification and
# refined here.
STILT_SPAWN_HEIGHT = 1.20

# Nominal body_pos z of *_stilt_post_inner in the MJCF. dr.body_pos randomises
# this field; envs/stilt_g1/events.py differences against this value to correct
# the spawn height (spec §6).
STILT_NOMINAL_POST_INNER_Z = 0.0

STILT_G1_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, STILT_SPAWN_HEIGHT),
  joint_pos={
    ".*_hip_pitch_joint": -STILT_KNEE_ANGLE,
    ".*_knee_joint": STILT_KNEE_ANGLE,
    ".*_elbow_joint": 0.6,
    "left_shoulder_roll_joint": 0.2,
    "left_shoulder_pitch_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    "right_shoulder_pitch_joint": 0.2,
  },
  joint_vel={".*": 0.0},
)
```

Then change `get_stilt_g1_robot_cfg` to pass `articulation=STILT_G1_ARTICULATION`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_stilt_robot_cfg.py -v`
Expected: 4 passed.

- [ ] **Step 5: Solve the exact spawn height**

Run: `uv run python scripts/solve_spawn_height.py` (created below) and paste the printed
value into `STILT_SPAWN_HEIGHT`.

```python
# scripts/solve_spawn_height.py
"""Print the pelvis height that rests both stilt plates exactly on the floor."""

from pathlib import Path

import mujoco

from envs.stilt_g1.stilt_robot import STILT_KNEE_ANGLE

XML = Path(__file__).parent.parent / "assets" / "mjcf" / "g1" / "g1.xml"


def main() -> None:
  model = mujoco.MjSpec.from_file(str(XML)).compile()
  data = mujoco.MjData(model)

  for side in ("left", "right"):
    for joint, value in (
      ("hip_pitch", -STILT_KNEE_ANGLE),
      ("knee", STILT_KNEE_ANGLE),
    ):
      jid = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_{joint}_joint"
      )
      data.qpos[model.jnt_qposadr[jid]] = value

  data.qpos[2] = 1.0
  mujoco.mj_forward(model, data)

  lowest = min(
    data.geom_xpos[gid][2] - model.geom_size[gid][0]
    for gid in range(model.ngeom)
    if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or "").endswith(
      "_collision"
    )
    and "stilt" in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or "")
  )
  print(f"STILT_SPAWN_HEIGHT = {1.0 - lowest:.4f}")


if __name__ == "__main__":
  main()
```

- [ ] **Step 6: Commit**

```bash
uv run ruff format && uv run ruff check --fix
git add envs/stilt_g1/stilt_robot.py scripts/solve_spawn_height.py tests/test_stilt_robot_cfg.py
git commit -m "feat(stilt): drop ankle actuator, upright-shank keyframe"
```

---

### Task 4: Height randomisation events and curriculum

**Files:**
- Create: `envs/stilt_g1/events.py`
- Modify: `envs/stilt_g1/curriculums.py`
- Create: `tests/test_stilt_height_events.py`

**Interfaces:**
- Consumes: `STILT_NOMINAL_POST_INNER_Z` from Task 3.
- Produces:
  - `reset_stilt_spawn_height(env, env_ids, asset_cfg, nominal_z) -> None`
  - `class stilt_height_curriculum` — same shape as `stilt_mass_curriculum`, stages keyed `{"step": int, "offset_range": tuple[float, float]}`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_stilt_height_events.py
"""Spawn-height correction must exactly cancel the sampled telescope offset."""

import pytest

from envs.stilt_g1.events import spawn_height_correction


def test_taller_stilt_raises_the_robot():
  """A more negative body_pos z means a longer stilt, so the root must rise."""
  correction = spawn_height_correction(sampled_z=-0.05, nominal_z=0.0)
  assert correction == pytest.approx(+0.05)


def test_shorter_stilt_lowers_the_robot():
  correction = spawn_height_correction(sampled_z=+0.03, nominal_z=0.0)
  assert correction == pytest.approx(-0.03)


def test_nominal_height_needs_no_correction():
  assert spawn_height_correction(sampled_z=0.0, nominal_z=0.0) == pytest.approx(0.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_stilt_height_events.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'envs.stilt_g1.events'`

- [ ] **Step 3: Implement the events module**

```python
# envs/stilt_g1/events.py
"""Reset events specific to the telescoping stilt.

`ManagerBasedRlEnv._reset_idx` applies `scene.reset()` — and therefore the
keyframe — *before* it applies `mode="reset"` events. So the spawn height in
the keyframe cannot know the sampled stilt height. `reset_stilt_spawn_height`
runs after the height DR term and corrects the root pose to match.

Registration order matters: `EventManager` builds its term list by iterating
the config dict, so this term must be inserted into `cfg.events` *after* the
height randomisation term.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def spawn_height_correction(sampled_z: float, nominal_z: float) -> float:
  """Root height correction for a telescope offset.

  A longer stilt pushes `*_stilt_post_inner` further down (more negative
  `body_pos` z), so the root must rise by the same amount.
  """
  return nominal_z - sampled_z


def reset_stilt_spawn_height(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  asset_cfg: SceneEntityCfg,
  nominal_z: float,
) -> None:
  """Raise the root so randomised stilts still rest on the floor at spawn."""
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)

  asset = env.scene[asset_cfg.name]
  body_ids = asset_cfg.body_ids

  sampled_z = env.sim.model.body_pos[env_ids][:, body_ids, 2]
  correction = (nominal_z - sampled_z).mean(dim=1)

  pose = torch.cat(
    [
      asset.data.root_link_pos_w[env_ids].clone(),
      asset.data.root_link_quat_w[env_ids].clone(),
    ],
    dim=-1,
  )
  pose[:, 2] += correction
  asset.write_root_link_pose_to_sim(pose, env_ids=env_ids)
```

- [ ] **Step 4: Add the height curriculum**

Append to `envs/stilt_g1/curriculums.py`:

```python
class stilt_height_curriculum:
  """Widen the stilt telescope offset range over training.

  Stages define ``step`` thresholds and the target ``offset_range`` tuple, in
  metres of additional stilt length (negative = longer stilt, because the
  inner post moves down).
  """

  def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv):
    event_name: str = cfg.params["event_name"]
    self._stages: list[dict] = cfg.params["stages"]
    self._term_cfg = env.event_manager.get_term_cfg(event_name)

    steps = [s["step"] for s in self._stages]
    if steps != sorted(steps):
      raise ValueError(
        f"stilt_height_curriculum stages must be in nondecreasing step order, "
        f"got {steps}."
      )

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    event_name: str,
    stages: list[dict],
  ) -> dict[str, torch.Tensor]:
    del env_ids, event_name, stages

    active_range = self._term_cfg.params["ranges"]
    for stage in self._stages:
      if env.common_step_counter >= stage["step"]:
        active_range = stage["offset_range"]

    self._term_cfg.params["ranges"] = active_range

    lo, hi = active_range
    return {
      "stilt_height_min_m": torch.tensor(0.4075 - hi),
      "stilt_height_max_m": torch.tensor(0.4075 - lo),
    }
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_stilt_height_events.py -v`
Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
uv run ruff format && uv run ruff check --fix
git add envs/stilt_g1/events.py envs/stilt_g1/curriculums.py tests/test_stilt_height_events.py
git commit -m "feat(stilt): telescope height DR events and curriculum"
```

---

### Task 5: Env config wiring

**Files:**
- Modify: `envs/stilt_g1/env_cfgs.py`
- Create: `tests/test_stilt_env_cfg.py`

**Interfaces:**
- Consumes: Tasks 3 and 4.
- Produces: `cfg.events` containing `stilt_mass`, `stilt_height`, `stilt_spawn_height` in that order; `cfg.curriculum` containing `stilt_mass` and `stilt_height`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_stilt_env_cfg.py
"""Env wiring: DR targets, event ordering, and the rebased mass curriculum."""

from envs.stilt_g1.env_cfgs import _STILT_BODY_NAMES, stilt_g1_flat_env_cfg

SEGMENTS = (
  "stilt_mount",
  "stilt_brace",
  "stilt_post_outer",
  "stilt_post_inner",
  "stilt_plate",
)


def test_mass_dr_targets_every_segment():
  assert len(_STILT_BODY_NAMES) == 10
  for side in ("left", "right"):
    for segment in SEGMENTS:
      assert f"{side}_{segment}" in _STILT_BODY_NAMES


def test_spawn_correction_runs_after_height_sampling():
  """EventManager executes reset terms in config-dict order."""
  cfg = stilt_g1_flat_env_cfg()
  reset_terms = [k for k, v in cfg.events.items() if v.mode == "reset"]
  assert reset_terms.index("stilt_spawn_height") > reset_terms.index("stilt_height")


def test_spawn_correction_runs_after_reset_base():
  cfg = stilt_g1_flat_env_cfg()
  reset_terms = [k for k, v in cfg.events.items() if v.mode == "reset"]
  assert reset_terms.index("stilt_spawn_height") > reset_terms.index("reset_base")


def test_height_curriculum_starts_fixed():
  cfg = stilt_g1_flat_env_cfg()
  stages = cfg.curriculum["stilt_height"].params["stages"]
  assert stages[0]["offset_range"] == (0.0, 0.0)


def test_mass_curriculum_is_rebased_on_2_8_kg():
  cfg = stilt_g1_flat_env_cfg()
  assert cfg.curriculum["stilt_mass"].params["baseline_kg"] == 2.8


def test_play_config_has_no_curriculum():
  cfg = stilt_g1_flat_env_cfg(play=True)
  assert "stilt_height" not in cfg.curriculum
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_stilt_env_cfg.py -v`
Expected: FAIL — `ImportError: cannot import name '_STILT_BODY_NAMES'`

- [ ] **Step 3: Wire the config**

In `envs/stilt_g1/env_cfgs.py`:

```python
from .curriculums import stilt_height_curriculum, stilt_mass_curriculum
from .events import reset_stilt_spawn_height
from .stilt_robot import STILT_NOMINAL_POST_INNER_Z

_STILT_SEGMENTS = (
  "stilt_mount",
  "stilt_brace",
  "stilt_post_outer",
  "stilt_post_inner",
  "stilt_plate",
)

_STILT_BODY_NAMES = tuple(
  f"{side}_{segment}"
  for side in ("left", "right")
  for segment in _STILT_SEGMENTS
)

_STILT_INNER_POST_BODIES = ("left_stilt_post_inner", "right_stilt_post_inner")
```

Point the mass event at every segment:

```python
cfg.events["stilt_mass"] = EventTermCfg(
  func=dr.pseudo_inertia,
  mode="reset",
  params={
    "alpha_range": (0.0, 0.0),
    "asset_cfg": SceneEntityCfg("robot", body_names=list(_STILT_BODY_NAMES)),
  },
)
```

Then, **in this order**, the height terms:

```python
# Telescope offset. Negative = inner post pushed down = longer stilt.
cfg.events["stilt_height"] = EventTermCfg(
  func=dr.body_pos,
  mode="reset",
  params={
    "ranges": (0.0, 0.0),  # overwritten each step by the curriculum
    "axes": [2],
    "operation": "add",
    "asset_cfg": SceneEntityCfg(
      "robot", body_names=list(_STILT_INNER_POST_BODIES)
    ),
  },
)

# MUST follow stilt_height: EventManager runs reset terms in dict order, and
# scene.reset() (the keyframe) has already run by this point (spec §6).
cfg.events["stilt_spawn_height"] = EventTermCfg(
  func=reset_stilt_spawn_height,
  mode="reset",
  params={
    "asset_cfg": SceneEntityCfg(
      "robot", body_names=list(_STILT_INNER_POST_BODIES)
    ),
    "nominal_z": STILT_NOMINAL_POST_INNER_Z,
  },
)
```

Rebase the mass curriculum on 2.8 kg and add the height curriculum:

```python
if not play:
  cfg.curriculum["stilt_mass"] = CurriculumTermCfg(
    func=stilt_mass_curriculum,
    params={
      "event_name": "stilt_mass",
      "baseline_kg": 2.8,
      "stages": [
        {"step": 0, "alpha_range": (0.0, 0.0)},           # fixed 2.8 kg
        {"step": 500 * 24, "alpha_range": (-0.2, 0.2)},   # 1.9-4.2 kg
        {"step": 1000 * 24, "alpha_range": (-0.4, 0.4)},  # 1.3-6.2 kg
        {"step": 2000 * 24, "alpha_range": (-0.55, 0.5)}, # 0.9-7.6 kg
      ],
    },
  )

  cfg.curriculum["stilt_height"] = CurriculumTermCfg(
    func=stilt_height_curriculum,
    params={
      "event_name": "stilt_height",
      "stages": [
        {"step": 0, "offset_range": (0.0, 0.0)},              # fixed 407.5 mm
        {"step": 750 * 24, "offset_range": (-0.020, 0.020)},  # 387-427 mm
        {"step": 1500 * 24, "offset_range": (-0.050, 0.050)}, # 357-457 mm
        # Full mechanical range is gated on confirming minimum safe tube
        # overlap (spec §13 open item 1) — do not widen past ±50 mm until then.
      ],
    },
  )
```

Add `baseline_kg` to `stilt_mass_curriculum.__call__`, replacing the hardcoded `1.5`:

```python
  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    event_name: str,
    baseline_kg: float,
    stages: list[dict],
  ) -> dict[str, torch.Tensor]:
    del env_ids, event_name, stages
    ...
    return {
      "stilt_mass_min_kg": torch.tensor(baseline_kg * math.exp(2 * lo)),
      "stilt_mass_max_kg": torch.tensor(baseline_kg * math.exp(2 * hi)),
    }
```

Move `import math` to the top of `curriculums.py` while you are in there.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_stilt_env_cfg.py -v`
Expected: 6 passed.

- [ ] **Step 5: Verify the height DR gate end-to-end**

This is the spec's §6 gate. Run the environment and confirm no env spawns intersecting the
floor across the full sampled range:

```bash
uv run python scripts/check_height_dr.py
```

```python
# scripts/check_height_dr.py
"""Gate check: randomised stilt heights must all spawn resting on the floor."""

import torch

from mjlab.envs import ManagerBasedRlEnv

from envs.stilt_g1.env_cfgs import stilt_g1_flat_env_cfg


def main() -> None:
  cfg = stilt_g1_flat_env_cfg()
  cfg.scene.num_envs = 64
  cfg.events["stilt_height"].params["ranges"] = (-0.05, 0.05)

  env = ManagerBasedRlEnv(cfg)
  env.reset()

  tip_heights = []
  for side in ("left", "right"):
    site = env.scene["robot"].data.site_pos_w[
      :, env.scene["robot"].indexing.site_names.index(f"{side}_stilt_tip")
    ]
    tip_heights.append(site[:, 2] - env.scene.env_origins[:, 2])
  tips = torch.stack(tip_heights, dim=-1)

  print(f"tip height  min {tips.min():+.4f}  max {tips.max():+.4f}")
  assert tips.min() > -0.005, "envs spawning through the floor"
  assert tips.max() < 0.050, "envs spawning in the air"
  print("PASS: all 64 envs spawn resting on the floor")


if __name__ == "__main__":
  main()
```

Expected: `PASS: all 64 envs spawn resting on the floor`.

**If this fails**, the correction sign or the `body_pos` readback is wrong. Do not proceed
to Task 6 — fix it here.

- [ ] **Step 6: Commit**

```bash
uv run ruff format && uv run ruff check --fix
git add envs/stilt_g1/env_cfgs.py envs/stilt_g1/curriculums.py scripts/check_height_dr.py tests/test_stilt_env_cfg.py
git commit -m "feat(stilt): wire mass and height DR across all segments"
```

---

### Task 6: Load computation

Pure functions, no viser dependency, so they are testable in isolation.

**Files:**
- Create: `envs/stilt_g1/loads.py`
- Create: `tests/test_stilt_loads.py`

**Interfaces:**
- Produces:
  - `SECTIONS: tuple[str, ...]` — the five segment names, ground-up.
  - `contact_forces(model, data, side) -> dict[str, float]` — normal force per `*_stilt_[lr][1-4]_collision` geom.
  - `section_loads(model, data, side) -> dict[str, SectionLoad]`
  - `@dataclass SectionLoad: axial: float; shear: float; bending: float; torsion: float`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_stilt_loads.py
"""Section loads must satisfy statics in a known static stand."""

import mujoco
import pytest

from envs.stilt_g1.loads import SECTIONS, contact_forces, section_loads


@pytest.fixture(scope="module")
def standing(stilt_model):
  """Settle the robot into a static stand so contact forces are meaningful."""
  model, _ = stilt_model
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  for _ in range(500):
    mujoco.mj_step(model, data)
  return model, data


def test_all_sections_reported(standing):
  model, data = standing
  loads = section_loads(model, data, "left")
  assert set(loads) == set(SECTIONS)


def test_ground_reaction_supports_the_robot(standing):
  """Total vertical contact force must equal total weight in a static stand."""
  model, data = standing
  total = sum(contact_forces(model, data, s).values() for s in ("left", "right"))
  weight = model.body_mass.sum() * abs(model.opt.gravity[2])
  assert total == pytest.approx(weight, rel=0.10)


def test_axial_load_grows_toward_the_mount(standing):
  """Each section carries everything below it, so axial load is monotonic."""
  model, data = standing
  loads = section_loads(model, data, "left")
  axials = [loads[s].axial for s in SECTIONS]
  assert axials == sorted(axials), axials


def test_loads_are_finite(standing):
  model, data = standing
  for side in ("left", "right"):
    for load in section_loads(model, data, side).values():
      for value in (load.axial, load.shear, load.bending, load.torsion):
        assert value == value and abs(value) < 1e6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_stilt_loads.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'envs.stilt_g1.loads'`

- [ ] **Step 3: Implement**

Each section carries the ground reaction below it, minus the inertial reaction of the
segments below it. Sections are listed ground-up so `test_axial_load_grows_toward_the_mount`
holds.

```python
# envs/stilt_g1/loads.py
"""Internal load readout for the stilt segments.

Computed in Python from contact forces and segment inertia, so this does not
depend on mujoco_warp implementing force/torque sensors (spec §8.2).

Limitation (spec §8.3): with the ankle joints deleted, shank and foot are one
rigid body, so the split of the interface load between the sole bolts and the
shank clamp is statically indeterminate. `stilt_brace` therefore reports its
own inertial load only — NOT the real clamp reaction.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

# Ground-up: each section carries every section before it.
SECTIONS = (
  "stilt_plate",
  "stilt_post_inner",
  "stilt_post_outer",
  "stilt_mount",
  "stilt_brace",
)

# Segments whose weight passes through each section, ground-up.
_BELOW: dict[str, tuple[str, ...]] = {
  "stilt_plate": ("stilt_plate",),
  "stilt_post_inner": ("stilt_plate", "stilt_post_inner"),
  "stilt_post_outer": ("stilt_plate", "stilt_post_inner", "stilt_post_outer"),
  "stilt_mount": (
    "stilt_plate",
    "stilt_post_inner",
    "stilt_post_outer",
    "stilt_mount",
  ),
  "stilt_brace": ("stilt_brace",),  # leaf: inertial load only
}

_CAPSULES = tuple(
  f"{block}{i}" for block in ("l", "r") for i in range(1, 5)
)


@dataclass(frozen=True)
class SectionLoad:
  """Internal load at one section, in the world frame."""

  axial: float
  shear: float
  bending: float
  torsion: float


def contact_forces(
  mujoco_model, data, side: str
) -> dict[str, float]:
  """Normal contact force in each ground capsule of one stilt."""
  result = {name: 0.0 for name in _CAPSULES}
  wrench = np.zeros(6)

  ids = {}
  for name in _CAPSULES:
    gid = mujoco.mj_name2id(
      mujoco_model,
      mujoco.mjtObj.mjOBJ_GEOM,
      f"{side}_stilt_{name}_collision",
    )
    ids[gid] = name

  for i in range(data.ncon):
    contact = data.contact[i]
    name = ids.get(contact.geom1) or ids.get(contact.geom2)
    if name is None:
      continue
    mujoco.mj_contactForce(mujoco_model, data, i, wrench)
    result[name] += abs(float(wrench[0]))
  return result


def _segment_id(mujoco_model, side: str, segment: str) -> int:
  return mujoco.mj_name2id(
    mujoco_model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_{segment}"
  )


def section_loads(mujoco_model, data, side: str) -> dict[str, SectionLoad]:
  """Axial, shear, bending and torsion at each stilt section."""
  gravity = np.asarray(mujoco_model.opt.gravity)

  ground_force = np.zeros(3)
  ground_moment = np.zeros(3)
  wrench = np.zeros(6)
  capsule_ids = {}
  for name in _CAPSULES:
    gid = mujoco.mj_name2id(
      mujoco_model,
      mujoco.mjtObj.mjOBJ_GEOM,
      f"{side}_stilt_{name}_collision",
    )
    capsule_ids[gid] = gid

  for i in range(data.ncon):
    contact = data.contact[i]
    if contact.geom1 not in capsule_ids and contact.geom2 not in capsule_ids:
      continue
    mujoco.mj_contactForce(mujoco_model, data, i, wrench)
    frame = np.asarray(contact.frame).reshape(3, 3)
    force_world = frame.T @ wrench[:3]
    ground_force += force_world
    ground_moment += np.cross(np.asarray(contact.pos), force_world)

  loads: dict[str, SectionLoad] = {}
  for section in SECTIONS:
    force = np.zeros(3)
    moment = np.zeros(3)

    if section != "stilt_brace":
      force += ground_force
      moment += ground_moment

    for segment in _BELOW[section]:
      bid = _segment_id(mujoco_model, side, segment)
      mass = float(mujoco_model.body_mass[bid])
      com = np.asarray(data.xipos[bid])
      acceleration = np.asarray(data.cacc[bid][3:6]) if data.cacc is not None else np.zeros(3)
      inertial = mass * (acceleration - gravity)
      force -= inertial
      moment -= np.cross(com, inertial)

    origin = np.asarray(data.xipos[_segment_id(mujoco_model, side, section)])
    moment_at_section = moment - np.cross(origin, force)

    loads[section] = SectionLoad(
      axial=float(abs(force[2])),
      shear=float(np.linalg.norm(force[:2])),
      bending=float(np.linalg.norm(moment_at_section[:2])),
      torsion=float(abs(moment_at_section[2])),
    )
  return loads
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_stilt_loads.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
uv run ruff format && uv run ruff check --fix
git add envs/stilt_g1/loads.py tests/test_stilt_loads.py
git commit -m "feat(stilt): section load and ground pressure computation"
```

---

### Task 7: Viewer GUI

**Files:**
- Modify: `envs/stilt_g1/__init__.py`

**Interfaces:**
- Consumes: `envs.stilt_g1.loads.{SECTIONS, contact_forces, section_loads}`.

- [ ] **Step 1: Retarget the torque monitor**

`_JOINTS` (around line 42) lists `left/right_ankle_pitch_joint`, which no longer exist and
will raise on lookup. Remove both entries and the `"ankle"` key from `_LIMIT`, leaving hip
pitch and knee.

- [ ] **Step 2: Add per-segment mass sliders**

Replace the single mass slider with one per segment plus a master multiplier. Each applies
live through `dr.pseudo_inertia` followed by an explicit `recompute_constants` — the
`@requires_model_fields` decorator only annotates, it does not auto-trigger (CLAUDE.md).

```python
MASS_PLAY_MAX_KG = 8.0

_SEGMENT_BASELINE_KG = {
  "stilt_mount": 0.636,
  "stilt_brace": 0.999,
  "stilt_post_outer": 0.313,
  "stilt_post_inner": 0.288,
  "stilt_plate": 0.565,
}
```

Build one `server.gui.add_slider` per segment inside a `Stilt Mass` folder, each with
`min=0.0, max=MASS_PLAY_MAX_KG, initial_value=_SEGMENT_BASELINE_KG[name]`, wired to the
existing `_apply()` path with `body_names=[f"left_{name}", f"right_{name}"]`. Add a master
`server.gui.add_slider("master x", min=0.1, max=3.0, initial_value=1.0)` that scales all
five. Keep the existing "Randomize on reset" checkbox and per-segment sim-mass readback
labels.

- [ ] **Step 3: Add the Stilt Loads folder**

Inside a `Stilt Loads` folder, one label row per section per side, refreshed on the same
10 Hz timer as the torque monitor (which already updates while paused). Format matches the
existing torque monitor:

```
L plate       axial  245N   shear   31N   bend  12.4Nm   ███████░░░░░░
```

Add a `SECTION_LIMIT` dict beside `MASS_PLAY_MAX_KG` for the bar denominators. The
`stilt_brace` row must carry the label `inertial only — not clamp reaction` (spec §8.3).

- [ ] **Step 4: Add the Ground Pressure folder**

Eight rows per foot from `contact_forces`, bar-scaled to body weight.

- [ ] **Step 5: Verify in the viewer**

Run:

```bash
uv run python scripts/play_stilt.py Mjlab-Velocity-Flat-Stilt-G1 --num-envs 1 --viewer viser
```

Confirm at `http://localhost:8080`:
- stilts render as five telescoped segments, resting flat on the floor
- moving a segment mass slider changes that segment's readback **and** `qfrc_bias` at hip/knee
- ground pressure bars sum to roughly body weight in a static stand
- all folders keep updating while paused

- [ ] **Step 6: Commit**

```bash
uv run ruff format && uv run ruff check --fix
git add envs/stilt_g1/__init__.py
git commit -m "feat(stilt): per-segment mass sliders and load monitors in viewer"
```

---

### Task 8: Deploy config and documentation

**Files:**
- Modify: `deploy/config/g1_stilt/deploy.yaml`
- Modify: `deploy/README.md`
- Modify: `CLAUDE.md`
- Modify: `STATUS.md`

- [ ] **Step 1: Strip the ankles from the deploy joint list**

`deploy.yaml` documents a 23-joint ordering with `left_ankle_pitch` at index 4,
`left_ankle_roll` at 5, `right_ankle_pitch` at 10, `right_ankle_roll` at 11. Remove those
four and renumber. The policy now emits 25 actions.

- [ ] **Step 2: Document the hardware ankle handling**

Add to `deploy/README.md`:

```markdown
## Ankle handling (stilt hardware)

The stilt brace clamps the shank, so ankle pitch and roll are mechanically
locked and the policy does not command them — the action vector is 25, not 29.

**The four ankle motors must be left in damping mode, not PD position mode.**
PD-tracking a target into a rigid clamp makes the motors fight the structure
and overheat. Set them to zero stiffness with nonzero damping in the runtime
config before running `./g1_ctrl`.
```

- [ ] **Step 3: Update project docs**

In `CLAUDE.md`, replace the stilt description with the 5-segment tree, the 2.8 kg mass, the
407.5 mm height, and the welded ankles. In `STATUS.md`, mark the Run 5 checkpoint as
superseded and record that a fresh run is required.

- [ ] **Step 4: Full test run**

Run: `uv run pytest tests/ -v`
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add deploy/ CLAUDE.md STATUS.md
git commit -m "docs(stilt): update deploy config and project docs for new stilt"
```

---

## Self-Review

**Spec coverage:** §2 geometry → Task 1. §3 ankle removal → Tasks 2, 3. §4 bodies/mass →
Tasks 1, 2. §4.1 meshes → Task 1. §5 collision → Task 2. §6 height DR → Tasks 4, 5
(including the resolved gate, checked by `scripts/check_height_dr.py`). §7 generator →
Task 1. §8 instrumentation → Task 6. §9 GUI → Task 7. §10 downstream → Tasks 3, 5, 7, 8.
§11 verification → distributed across task tests, with §11.3 spawn check in Task 3 Step 5
and §11.4 in Task 5 Step 5.

**Known gap, deliberate:** §8.2's optional MJCF `<force>`/`<torque>` sensors are not built.
They are explicitly conditional on mjwarp support and the viewer readout in Task 6 is the
actual deliverable. Revisit only if the sensors are later needed as training observations.

**Open items carried from the spec:** minimum safe tube overlap (caps the Task 5 height
curriculum at ±50 mm until confirmed); per-part material assignment (a one-line change in
Task 1 if it arrives).
