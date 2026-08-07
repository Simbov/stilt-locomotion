"""Stilt G1 robot configuration — points to the local modified MJCF."""

from pathlib import Path

import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import (
  G1_ACTUATOR_4010,
  G1_ACTUATOR_5020,
  G1_ACTUATOR_7520_14,
  G1_ACTUATOR_7520_22,
  G1_ACTUATOR_WAIST,
)
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

# Path to our local stilt MJCF (relative to this file: ../../assets/mjcf/g1/g1.xml)
STILT_G1_XML = Path(__file__).parent.parent.parent / "assets" / "mjcf" / "g1" / "g1.xml"
assert STILT_G1_XML.exists(), f"Stilt MJCF not found: {STILT_G1_XML}"


def get_stilt_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(STILT_G1_XML))


# The stilt's shank brace clamps the calf, so ankle pitch and roll are
# mechanically locked and their joints are deleted from the MJCF. Drop
# G1_ACTUATOR_ANKLE to match: action space 29 -> 25.
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

# Same 0.25 * effort_limit / stiffness rule the stock G1 uses, minus the ankle.
STILT_G1_ACTION_SCALE: dict[str, float] = {}
for _actuator in STILT_G1_ARTICULATION.actuators:
  assert isinstance(_actuator, BuiltinPositionActuatorCfg)
  assert _actuator.effort_limit is not None
  for _name in _actuator.target_names_expr:
    STILT_G1_ACTION_SCALE[_name] = 0.25 * _actuator.effort_limit / _actuator.stiffness

# With the ankle welded, shank orientation is rigidly hip_pitch + knee, so the
# stilt is vertical only when hip_pitch == -knee. The old keyframe relied on
# ankle_pitch=-0.363 to flatten the foot under a 0.669 knee bend; that is no
# longer available, so the bend is shallower and exactly cancelled at the hip.
#
# The angle is bounded by static stability, not comfort. Under the hip == -knee
# constraint, bending the knee swings the pelvis BACKWARD, and the stilt plate
# only reaches 70 mm behind the ankle. Whole-body COM x relative to the ankle:
#
#   knee 0.00 -> +0.009 m    knee 0.20 -> -0.032 m
#   knee 0.10 -> -0.012 m    knee 0.30 -> -0.052 m  (17 mm from the heel edge)
#   knee 0.15 -> -0.022 m    knee 0.40 -> -0.072 m  (outside the plate)
#
# 0.30 stood the robot on the back lip of its own support polygon: all ground
# reaction landed on the heel capsule and PPO learned to terminate on step 1
# rather than accumulate negative reward. 0.10 keeps ~58 mm of heel margin
# while retaining some knee compliance.
STILT_KNEE_ANGLE = 0.10

# Pelvis height that rests both stilt plates on the floor at the keyframe pose.
# Solved by scripts/solve_spawn_height.py — rerun it if STILT_KNEE_ANGLE or the
# stilt geometry changes.
STILT_SPAWN_HEIGHT = 1.1977

# Nominal body_pos z of *_stilt_post_inner in the MJCF. dr.body_pos randomises
# this field; envs/stilt_g1/events.py differences against this value to correct
# the spawn height.
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

# Stilt contact geoms explicitly get condim=3 (frictional) + friction=1.0.
# Applied in two passes so the stilt rule is never shadowed by the catch-all.
STILT_G1_COLLISION = CollisionCfg(
  geom_names_expr=(r"^(left|right)_stilt_[lr][1-4]_collision$",),
  condim={r"^(left|right)_stilt_[lr][1-4]_collision$": 3},
  priority={r"^(left|right)_stilt_[lr][1-4]_collision$": 1},
  friction={r"^(left|right)_stilt_[lr][1-4]_collision$": (1.0, 0.005, 0.0001)},
)


def get_stilt_g1_robot_cfg() -> EntityCfg:
  return EntityCfg(
    init_state=STILT_G1_KEYFRAME,
    collisions=(STILT_G1_COLLISION,),
    spec_fn=get_stilt_spec,
    articulation=STILT_G1_ARTICULATION,
  )
