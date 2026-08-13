"""Section loads must satisfy statics in a known static stand."""

import mujoco
import pytest

from envs.stilt_g1.loads import SECTIONS, contact_forces, section_loads
from envs.stilt_g1.stilt_robot import (
  STILT_FITTED_SPAWN_HEIGHT,
  STILT_LEG_POSE,
)


@pytest.fixture(scope="module")
def standing():
  """Rest the robot vertically on a floor so the stilts carry its full weight.

  This is a statics rig, not a locomotion test. Three departures from the real
  model, all deliberate:

  * The project MJCF has no ground plane — mjlab supplies the terrain — so one
    is added here.
  * The pelvis free joint is replaced by a single vertical slide. The robot
    cannot balance passively without a controller and would simply topple,
    which tells us nothing about section loads. Constraining it to sink
    vertically means the ground reaction must equal total weight at rest, which
    is exactly the invariant these tests check.
  * Every joint gets a stiff position actuator, including the ankles. On the
    real hardware the brace does that job — the sim models it as ankle joint
    stiffness applied at reset (see reset_stilts_fitted); here a held actuator
    is the simpler equivalent and the load path through the stilt is the same.

  The pose is the fitted standing pose: shank vertical, ankle at the brace's
  neutral angle, which is the only pose the assembled stilt stands upright in.
  """
  from tests.conftest import G1_XML

  spec = mujoco.MjSpec.from_file(str(G1_XML))
  spec.worldbody.add_geom(
    name="test_floor",
    type=mujoco.mjtGeom.mjGEOM_PLANE,
    size=[10.0, 10.0, 0.1],
    pos=[0.0, 0.0, 0.0],
  )

  for joint in spec.joints:
    if joint.type == mujoco.mjtJoint.mjJNT_FREE:
      joint.type = mujoco.mjtJoint.mjJNT_SLIDE
      joint.axis = [0.0, 0.0, 1.0]
    else:
      # Hold the leg pose; without this the knees fold under load.
      actuator = spec.add_actuator(target=joint.name, trntype=mujoco.mjtTrn.mjTRN_JOINT)
      # Stiff enough to hold the pose, soft enough not to blow up on contact.
      actuator.gainprm[0] = 600.0
      actuator.biasprm[1] = -600.0
      actuator.biasprm[2] = -60.0

  model = spec.compile()
  data = mujoco.MjData(model)

  for side in ("left", "right"):
    for joint, value in STILT_LEG_POSE.items():
      jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_{joint}_joint")
      data.qpos[model.jnt_qposadr[jid]] = value
      aid = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{side}_{joint}_joint"
      )
      data.ctrl[aid] = value
  data.qpos[0] = STILT_FITTED_SPAWN_HEIGHT

  for _ in range(8000):
    mujoco.mj_step(model, data)

  # Only the vertical DoF matters for the weight-balance invariant; the arms
  # keep drifting slowly and are irrelevant to the stilt load path.
  assert abs(data.qvel[0]) < 0.01, f"rig did not settle: {data.qvel[0]:+.4f} m/s"
  return model, data


def test_all_sections_reported(standing):
  model, data = standing
  assert set(section_loads(model, data, "left")) == set(SECTIONS)


def test_every_capsule_is_reported(standing):
  model, data = standing
  forces = contact_forces(model, data, "left")
  assert len(forces) == 8


def test_ground_reaction_supports_the_robot(standing):
  """Total vertical contact force must roughly equal total weight when static."""
  model, data = standing
  total = sum(
    sum(contact_forces(model, data, side).values()) for side in ("left", "right")
  )
  weight = model.body_mass.sum() * abs(model.opt.gravity[2])
  assert total == pytest.approx(weight, rel=0.25)


def test_axial_load_grows_toward_the_mount(standing):
  """Each section carries everything below it, so axial load is monotonic."""
  model, data = standing
  loads = section_loads(model, data, "left")
  structural = [s for s in SECTIONS if s != "stilt_brace"]
  axials = [loads[s].axial for s in structural]
  assert axials == sorted(axials), axials


def test_loads_are_finite(standing):
  model, data = standing
  for side in ("left", "right"):
    for load in section_loads(model, data, side).values():
      for value in (load.axial, load.shear, load.bending, load.torsion):
        assert value == value and abs(value) < 1e6
