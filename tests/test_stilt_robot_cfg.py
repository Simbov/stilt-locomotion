"""The robot config keeps the stock 29-DoF G1 and solves both spawn heights."""

import mujoco
import pytest

from envs.stilt_g1.stilt_robot import (
  STILT_FITTED_SPAWN_HEIGHT,
  STILT_G1_ACTION_SCALE,
  STILT_G1_ARTICULATION,
  STILT_G1_KEYFRAME,
  STILT_LEG_POSE,
  STILT_SPAWN_HEIGHT,
  STILT_SPAWN_RISE,
)


def test_ankle_actuator_is_present():
  """The ankles are always actuated — see test_stilt_mjcf for why."""
  patterns = [p for a in STILT_G1_ARTICULATION.actuators for p in a.target_names_expr]
  assert [p for p in patterns if "ankle" in p]


def test_action_scale_covers_the_ankle():
  assert [k for k in STILT_G1_ACTION_SCALE if "ankle" in k]


def test_keyframe_sets_the_ankle():
  assert [k for k in STILT_G1_KEYFRAME.joint_pos if "ankle" in k]


def _lowest_contact(model, pose: dict[str, float], height: float, substring: str):
  data = mujoco.MjData(model)
  for side in ("left", "right"):
    for joint, value in pose.items():
      jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_{joint}_joint")
      data.qpos[model.jnt_qposadr[jid]] = value
  data.qpos[2] = height
  mujoco.mj_forward(model, data)

  def name(gid):
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""

  return min(
    data.geom_xpos[gid][2] - model.geom_size[gid][0]
    for gid in range(model.ngeom)
    if substring in name(gid) and "collision" in name(gid)
  )


def test_spawn_height_rests_the_FEET_on_the_floor(stilt_model):
  """Stilts off: the robot stands on its own foot capsules."""
  model, _ = stilt_model
  assert _lowest_contact(
    model, STILT_LEG_POSE, STILT_SPAWN_HEIGHT, "_foot"
  ) == pytest.approx(0.0, abs=0.005)


def test_fitted_spawn_height_rests_the_STILTS_on_the_floor(stilt_model):
  """Stilts on: the same pose, raised by reset_stilt_spawn_height."""
  model, _ = stilt_model
  assert _lowest_contact(
    model, STILT_LEG_POSE, STILT_FITTED_SPAWN_HEIGHT, "_stilt_"
  ) == pytest.approx(0.0, abs=0.005)


def test_spawn_rise_is_the_difference_between_the_two():
  assert STILT_SPAWN_RISE == pytest.approx(
    STILT_FITTED_SPAWN_HEIGHT - STILT_SPAWN_HEIGHT
  )


def test_the_pose_stands_the_shank_vertical():
  """The brace clamps the shank to a post perpendicular to the sole, so the
  stilt is upright only when hip_pitch + knee = 0 with the ankle at the brace's
  neutral angle."""
  assert STILT_LEG_POSE["hip_pitch"] + STILT_LEG_POSE["knee"] == pytest.approx(0.0)
  assert STILT_LEG_POSE["ankle_pitch"] == 0.0
  assert STILT_LEG_POSE["ankle_roll"] == 0.0


def test_the_keyframe_IS_that_pose(stilt_model):
  """The action offset and the `pose` reward both key on the keyframe. If it
  disagrees with the pose the stilts stand up in, the neutral action is a
  falling pose and training collapses — this is the bug that killed the first
  Run 8 smoke run."""
  for stem, value in STILT_LEG_POSE.items():
    if stem == "ankle_roll":
      continue  # not in the keyframe; MuJoCo defaults it to 0, which matches
    assert STILT_G1_KEYFRAME.joint_pos[f".*_{stem}_joint"] == pytest.approx(value)
