"""Internal load readout for the stilt segments.

Computed in Python from contact forces and segment inertia, so this does not
depend on mujoco_warp implementing force/torque sensors.

Limitation: with the ankle joints deleted, shank and foot are one rigid body,
so the split of the interface load between the sole bolts and the shank clamp
is statically indeterminate. ``stilt_brace`` therefore reports its own inertial
load only — NOT the real clamp reaction. Label it as such wherever it is shown.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

# Ground-up: each structural section carries every section before it.
SECTIONS = (
  "stilt_plate",
  "stilt_post_inner",
  "stilt_post_outer",
  "stilt_mount",
  "stilt_brace",
)

# Segments whose weight passes through each section.
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
  # Leaf branch: carries no ground reaction, only its own inertial load.
  "stilt_brace": ("stilt_brace",),
}

_CAPSULES = tuple(f"{block}{i}" for block in ("l", "r") for i in range(1, 5))


@dataclass(frozen=True)
class SectionLoad:
  """Internal load at one section, resolved at that segment's COM."""

  axial: float
  shear: float
  bending: float
  torsion: float


def _capsule_ids(model, side: str) -> dict[int, str]:
  ids = {}
  for name in _CAPSULES:
    gid = mujoco.mj_name2id(
      model, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_stilt_{name}_collision"
    )
    if gid >= 0:
      ids[gid] = name
  return ids


def _body_id(model, side: str, segment: str) -> int:
  return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_{segment}")


def contact_forces(model, data, side: str) -> dict[str, float]:
  """Normal contact force magnitude in each ground capsule of one stilt."""
  result = {name: 0.0 for name in _CAPSULES}
  ids = _capsule_ids(model, side)
  wrench = np.zeros(6)

  for i in range(data.ncon):
    contact = data.contact[i]
    name = ids.get(contact.geom1) or ids.get(contact.geom2)
    if name is None:
      continue
    mujoco.mj_contactForce(model, data, i, wrench)
    result[name] += abs(float(wrench[0]))
  return result


def _ground_wrench(model, data, side: str) -> tuple[np.ndarray, np.ndarray]:
  """Total ground reaction on one stilt, as (force, moment about world origin)."""
  ids = _capsule_ids(model, side)
  force = np.zeros(3)
  moment = np.zeros(3)
  wrench = np.zeros(6)

  for i in range(data.ncon):
    contact = data.contact[i]
    if contact.geom1 not in ids and contact.geom2 not in ids:
      continue
    mujoco.mj_contactForce(model, data, i, wrench)
    # Contact frame rows are the normal and two tangents; the reported wrench
    # is in that frame, so rotate it back into world.
    frame = np.asarray(contact.frame).reshape(3, 3)
    force_world = frame.T @ wrench[:3]
    # geom1 is the first body in the pair, so flip when the stilt is geom2.
    if contact.geom2 in ids:
      force_world = -force_world
    force += force_world
    moment += np.cross(np.asarray(contact.pos), force_world)
  return force, moment


def section_loads(model, data, side: str) -> dict[str, SectionLoad]:
  """Axial, shear, bending and torsion carried at each stilt section."""
  gravity = np.asarray(model.opt.gravity)
  ground_force, ground_moment = _ground_wrench(model, data, side)

  loads: dict[str, SectionLoad] = {}
  for section in SECTIONS:
    force = np.zeros(3)
    moment = np.zeros(3)

    if section != "stilt_brace":
      force += ground_force
      moment += ground_moment

    for segment in _BELOW[section]:
      bid = _body_id(model, side, segment)
      mass = float(model.body_mass[bid])
      com = np.asarray(data.xipos[bid])
      # cacc is the body's spatial acceleration: [angular, linear].
      acceleration = np.asarray(data.cacc[bid][3:6])
      inertial = mass * (acceleration - gravity)
      force -= inertial
      moment -= np.cross(com, inertial)

    origin = np.asarray(data.xipos[_body_id(model, side, section)])
    moment_at_section = moment - np.cross(origin, force)

    loads[section] = SectionLoad(
      axial=float(abs(force[2])),
      shear=float(np.linalg.norm(force[:2])),
      bending=float(np.linalg.norm(moment_at_section[:2])),
      torsion=float(abs(moment_at_section[2])),
    )
  return loads
