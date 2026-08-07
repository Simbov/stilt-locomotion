"""The stilt robot config must drop the ankle actuator and stand the stilts upright."""

import mujoco
import pytest

from envs.stilt_g1.stilt_robot import (
  STILT_G1_ACTION_SCALE,
  STILT_G1_ARTICULATION,
  STILT_G1_KEYFRAME,
  STILT_KNEE_ANGLE,
  STILT_SPAWN_HEIGHT,
)


def test_ankle_actuator_is_absent():
  patterns = [p for a in STILT_G1_ARTICULATION.actuators for p in a.target_names_expr]
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


def test_spawn_height_rests_the_stilts_on_the_floor(stilt_model):
  """At the keyframe pose the lowest stilt contact must sit at z=0."""
  model, _ = stilt_model
  data = mujoco.MjData(model)

  for side in ("left", "right"):
    for joint, value in (
      ("hip_pitch", -STILT_KNEE_ANGLE),
      ("knee", STILT_KNEE_ANGLE),
    ):
      jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_{joint}_joint")
      data.qpos[model.jnt_qposadr[jid]] = value
  data.qpos[2] = STILT_SPAWN_HEIGHT
  mujoco.mj_forward(model, data)

  lowest = min(
    data.geom_xpos[gid][2] - model.geom_size[gid][0]
    for gid in range(model.ngeom)
    if "stilt" in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or "")
  )
  assert lowest == pytest.approx(0.0, abs=1e-3)
