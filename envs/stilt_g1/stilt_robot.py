"""Stilt G1 robot configuration — points to the local modified MJCF."""

from pathlib import Path

import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import (
  G1_ACTUATOR_4010,
  G1_ACTUATOR_5020,
  G1_ACTUATOR_7520_14,
  G1_ACTUATOR_7520_22,
  G1_ACTUATOR_ANKLE,
  G1_ACTUATOR_WAIST,
)
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

# Path to our local stilt MJCF (relative to this file: ../../assets/mjcf/g1/g1.xml)
STILT_G1_XML = Path(__file__).parent.parent.parent / "assets" / "mjcf" / "g1" / "g1.xml"
assert STILT_G1_XML.exists(), f"Stilt MJCF not found: {STILT_G1_XML}"


def get_stilt_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(STILT_G1_XML))


# The robot is ALWAYS the stock 29-DoF G1. The stilts bolt on and come off; the
# ankles are never removed and are always actuated. (Runs 6 and 7 deleted them on
# a misreading of the brace and are invalid for that reason.)
STILT_G1_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    G1_ACTUATOR_5020,
    G1_ACTUATOR_7520_14,
    G1_ACTUATOR_7520_22,
    G1_ACTUATOR_4010,
    G1_ACTUATOR_WAIST,
    G1_ACTUATOR_ANKLE,
  ),
  soft_joint_pos_limit_factor=0.9,
)

# Same 0.25 * effort_limit / stiffness rule the stock G1 uses.
STILT_G1_ACTION_SCALE: dict[str, float] = {}
for _actuator in STILT_G1_ARTICULATION.actuators:
  assert isinstance(_actuator, BuiltinPositionActuatorCfg)
  assert _actuator.effort_limit is not None
  for _name in _actuator.target_names_expr:
    STILT_G1_ACTION_SCALE[_name] = 0.25 * _actuator.effort_limit / _actuator.stiffness

# ── Standing pose — ONE pose, both morphologies ──────────────────────────────
#
# The stilts-fitted case has no freedom here. The mount bolts to the sole and the
# brace clamps the shank, so the post is perpendicular to the sole and parallel
# to the shank; for the stilt to stand upright the shank must be vertical, which
# means `hip_pitch = -knee` with the ankle at the brace's neutral angle. That
# neutral angle is ZERO, because the MJCF builds the mount and brace in the
# ankle-zero pose — the same reason the brace spring pulls toward zero.
#
# The bare robot then uses the SAME pose, even though the stock G1 crouch
# (hip -0.312, knee 0.669, ankle -0.363) would suit it better on its own. This is
# not cosmetic. `JointPositionActionCfg` offsets actions from the keyframe, so a
# zero action commands the keyframe pose — and the `pose` reward is keyed on it
# too. Two poses means the neutral action is wrong in one of the morphologies,
# and on stilts "wrong" means the stilts tilt ~20 degrees while the ankle PD
# fights the brace spring. A 60-iteration smoke run with split poses collapsed
# exactly that way: mean episode length fell 35 -> 16 while return rose, the
# signature of a policy learning that ending the episode is cheaper than
# standing, and the split metrics put the collapse squarely in the fitted envs.
#
# 0.10 rad rather than a deeper bend: the stilt plate only reaches 70 mm behind
# the ankle, and at knee 0.30 the pelvis sits 89 mm behind it — the COM falls off
# the back of the plate. Runs 6 and 7 both balanced from this stance.
STILT_KNEE_ANGLE = 0.10
STILT_HIP_PITCH = -STILT_KNEE_ANGLE
STILT_ANKLE_PITCH = 0.0

# Applied to both sides. Keyed by joint-name stem, as `{side}_{stem}_joint`.
STILT_LEG_POSE: dict[str, float] = {
  "hip_pitch": STILT_HIP_PITCH,
  "knee": STILT_KNEE_ANGLE,
  "ankle_pitch": STILT_ANKLE_PITCH,
  "ankle_roll": 0.0,
}

# Both solved by scripts/solve_spawn_height.py — do not hand-edit.
STILT_SPAWN_HEIGHT = 0.7902
STILT_FITTED_SPAWN_HEIGHT = 1.1977
# How far reset_stilt_spawn_height raises the root for a fitted env. With one
# shared pose this is exactly the stilt's ground clearance.
STILT_SPAWN_RISE = STILT_FITTED_SPAWN_HEIGHT - STILT_SPAWN_HEIGHT

# How far the stilt tip sites slide up when the stilts come off, to land on the
# robot's own sole: the tip site sits at -0.4425 in the ankle frame, the sole of
# the foot capsules at -0.035.
STILT_TIP_SITE_RISE = 0.4425 - 0.035

# Nominal body_pos z of *_stilt_post_inner in the MJCF. dr.body_pos randomises
# this field; envs/stilt_g1/events.py differences against this value to correct
# the spawn height.
STILT_NOMINAL_POST_INNER_Z = 0.0

STILT_G1_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, STILT_SPAWN_HEIGHT),
  joint_pos={
    ".*_hip_pitch_joint": STILT_HIP_PITCH,
    ".*_knee_joint": STILT_KNEE_ANGLE,
    ".*_ankle_pitch_joint": STILT_ANKLE_PITCH,
    ".*_elbow_joint": 0.6,
    "left_shoulder_roll_joint": 0.2,
    "left_shoulder_pitch_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    "right_shoulder_pitch_joint": 0.2,
  },
  joint_vel={".*": 0.0},
)

# Both ground-contact sets must be enabled, because both get used: the stilt
# capsules when the stilts are fitted, the robot's own foot capsules when they
# are not. `disable_other_geoms` defaults to True, so anything left out of
# `geom_names_expr` has its collisions switched off entirely — listing only the
# stilt geoms silently made the bare robot fall through the floor.
_STILT_CONTACT = r"^(left|right)_stilt_[lr][1-4]_collision$"
_FOOT_CONTACT = r"^(left|right)_foot[1-7]_collision$"

STILT_G1_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  # Frictional (condim 3) only where the robot touches the ground; everything
  # else is frictionless, matching the stock G1's FULL_COLLISION. Ordering
  # matters — the ground-contact rules must precede the catch-all or they are
  # shadowed by it.
  condim={_STILT_CONTACT: 3, _FOOT_CONTACT: 3, ".*_collision": 1},
  priority={_STILT_CONTACT: 1, _FOOT_CONTACT: 1},
  friction={_STILT_CONTACT: (1.0, 0.005, 0.0001), _FOOT_CONTACT: (0.6,)},
)


def get_stilt_g1_robot_cfg() -> EntityCfg:
  return EntityCfg(
    init_state=STILT_G1_KEYFRAME,
    collisions=(STILT_G1_COLLISION,),
    spec_fn=get_stilt_spec,
    articulation=STILT_G1_ARTICULATION,
  )
