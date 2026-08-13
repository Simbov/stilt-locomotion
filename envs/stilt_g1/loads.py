"""Internal load readout for the stilt segments.

Computed in Python from contact forces and segment inertia, so this does not
depend on mujoco_warp implementing force/torque sensors.

Limitation: the stilt bolts to the sole AND clamps the shank, so the interface
load has two parallel paths and the split between them is statically
indeterminate — the sim gives the total wrench only. ``stilt_brace`` therefore
reports its own inertial load, NOT the real clamp reaction. Label it as such
wherever it is shown. Sizing the brace needs a hand calculation or FEA.
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


def _capsule_ids(model, side: str, prefix: str) -> dict[int, str]:
  ids = {}
  for name in _CAPSULES:
    gid = mujoco.mj_name2id(
      model, mujoco.mjtObj.mjOBJ_GEOM, f"{prefix}{side}_stilt_{name}_collision"
    )
    if gid < 0:
      raise KeyError(
        f"geom '{prefix}{side}_stilt_{name}_collision' not found — in a scene "
        "model the names are namespaced, so pass prefix='robot/'"
      )
    ids[gid] = name
  return ids


def _body_id(model, side: str, segment: str, prefix: str) -> int:
  bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}{side}_{segment}")
  if bid < 0:
    raise KeyError(f"body '{prefix}{side}_{segment}' not found")
  return bid


def contact_forces(model, data, side: str, prefix: str = "") -> dict[str, float]:
  """Normal contact force magnitude in each ground capsule of one stilt.

  ``prefix`` namespaces the geom names. It is empty for the bare robot MJCF and
  ``"robot/"`` for a compiled mjlab scene.
  """
  result = {name: 0.0 for name in _CAPSULES}
  ids = _capsule_ids(model, side, prefix)
  wrench = np.zeros(6)

  for i in range(data.ncon):
    contact = data.contact[i]
    name = ids.get(contact.geom1) or ids.get(contact.geom2)
    if name is None:
      continue
    mujoco.mj_contactForce(model, data, i, wrench)
    result[name] += abs(float(wrench[0]))
  return result


def _ground_wrench(
  model, data, side: str, prefix: str
) -> tuple[np.ndarray, np.ndarray]:
  """Total ground reaction on one stilt, as (force, moment about world origin)."""
  ids = _capsule_ids(model, side, prefix)
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


def section_loads_core(
  ground_force: np.ndarray,
  ground_moment: np.ndarray,
  segments: dict[str, tuple[float, np.ndarray, np.ndarray]],
  gravity: np.ndarray,
) -> dict[str, SectionLoad]:
  """Statics for one stilt. This is the single implementation of the maths.

  Args:
    ground_force: total ground reaction on this stilt, world frame.
    ground_moment: its moment about the world origin.
    segments: ``name -> (mass, com_world, linear_acceleration_world)`` for all
      five segments.
    gravity: gravity vector, world frame.

  Both the CPU-MuJoCo path (:func:`section_loads`) and the mjlab sensor path
  (:func:`section_loads_from_sensor`) build these inputs and call this.
  """
  loads: dict[str, SectionLoad] = {}
  for section in SECTIONS:
    force = np.zeros(3)
    moment = np.zeros(3)

    if section != "stilt_brace":
      force += ground_force
      moment += ground_moment

    for segment in _BELOW[section]:
      mass, com, acceleration = segments[segment]
      inertial = mass * (acceleration - gravity)
      force -= inertial
      moment -= np.cross(com, inertial)

    origin = segments[section][1]
    moment_at_section = moment - np.cross(origin, force)

    loads[section] = SectionLoad(
      axial=float(abs(force[2])),
      shear=float(np.linalg.norm(force[:2])),
      bending=float(np.linalg.norm(moment_at_section[:2])),
      torsion=float(abs(moment_at_section[2])),
    )
  return loads


def section_loads(model, data, side: str, prefix: str = "") -> dict[str, SectionLoad]:
  """Section loads from CPU MuJoCo structures.

  ``prefix`` namespaces the body and geom names — see :func:`contact_forces`.
  """
  ground_force, ground_moment = _ground_wrench(model, data, side, prefix)

  segments = {}
  for segment in SECTIONS:
    bid = _body_id(model, side, segment, prefix)
    segments[segment] = (
      float(model.body_mass[bid]),
      np.asarray(data.xipos[bid], dtype=float),
      # cacc is the body's spatial acceleration: [angular, linear].
      np.asarray(data.cacc[bid][3:6], dtype=float),
    )

  return section_loads_core(
    ground_force, ground_moment, segments, np.asarray(model.opt.gravity)
  )


def sensor_capsule_forces(env, side: str, env_index: int = 0) -> dict[str, float]:
  """Per-capsule vertical ground reaction, read from the mjlab contact sensor.

  This is the live path used by the viewer. mujoco_warp does not surface
  per-contact geom ids through ``get_data_into`` — they come back as zeros — so
  the sensor is the only supported source while the warp sim is running.
  """
  from .env_cfgs import STILT_CONTACT_SENSOR

  sensor = env.scene.sensors[STILT_CONTACT_SENSOR]
  force = sensor.data.force[env_index]
  names = list(sensor.primary_names)

  result = {}
  for name in _CAPSULES:
    column = names.index(f"{side}_stilt_{name}_collision")
    result[name] = abs(float(force[column, 2]))
  return result


def section_loads_from_sensor(
  env, side: str, env_index: int = 0
) -> dict[str, SectionLoad]:
  """Section loads for the live warp sim, via the mjlab contact sensor."""
  from .env_cfgs import STILT_CONTACT_SENSOR

  sensor = env.scene.sensors[STILT_CONTACT_SENSOR]
  names = list(sensor.primary_names)
  force = sensor.data.force[env_index]
  pos = sensor.data.pos[env_index]

  ground_force = np.zeros(3)
  ground_moment = np.zeros(3)
  for name in _CAPSULES:
    column = names.index(f"{side}_stilt_{name}_collision")
    f = np.asarray(force[column].tolist(), dtype=float)
    p = np.asarray(pos[column].tolist(), dtype=float)
    ground_force += f
    ground_moment += np.cross(p, f)

  robot = env.scene["robot"]
  body_names = [b.name.split("/")[-1] for b in robot.indexing.bodies]
  gravity = np.asarray(env.sim.mj_model.opt.gravity, dtype=float)

  segments = {}
  for segment in SECTIONS:
    local = body_names.index(f"{side}_{segment}")
    global_id = int(robot.indexing.body_ids[local])
    mass = float(env.sim.model.body_mass[env_index, global_id])
    com = np.asarray(robot.data.body_link_pos_w[env_index, local].tolist(), dtype=float)
    # Segment acceleration is not exposed per body in the warp data; the stilt
    # segments are rigid and slow relative to the contact forces, so the
    # quasi-static approximation (zero acceleration) is used here. Gravity is
    # still applied, which is what dominates the section loads.
    segments[segment] = (mass, com, np.zeros(3))

  return section_loads_core(ground_force, ground_moment, segments, gravity)
