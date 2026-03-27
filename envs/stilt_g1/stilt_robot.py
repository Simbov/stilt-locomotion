"""Stilt G1 robot configuration — points to the local modified MJCF."""

from pathlib import Path

import mujoco

from mjlab.asset_zoo.robots.unitree_g1.g1_constants import (
  G1_ACTION_SCALE,
  G1_ARTICULATION,
)
from mjlab.entity import EntityCfg
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg

# Path to our local stilt MJCF (relative to this file: ../../assets/mjcf/g1/g1.xml)
STILT_G1_XML = Path(__file__).parent.parent.parent / "assets" / "mjcf" / "g1" / "g1.xml"
assert STILT_G1_XML.exists(), f"Stilt MJCF not found: {STILT_G1_XML}"


def _get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, STILT_G1_XML.parent / "assets", meshdir)
  return assets


def get_stilt_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(STILT_G1_XML))
  spec.assets = _get_assets(spec.meshdir)
  return spec


# Spawn height: stock G1 knees-bent pelvis (0.76m) + extra stilt extension.
# Stock foot extends 0.035m below ankle_roll_link (capsule at -0.025, radius 0.01).
# Stilt capsule extends 0.435m below ankle_roll_link.
# Extra height needed = 0.435 - 0.035 = 0.400m → spawn at 0.76 + 0.40 = 1.16m.
STILT_G1_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 1.16),
  joint_pos={
    ".*_hip_pitch_joint": -0.312,
    ".*_knee_joint": 0.669,
    ".*_ankle_pitch_joint": -0.363,
    ".*_elbow_joint": 0.6,
    "left_shoulder_roll_joint": 0.2,
    "left_shoulder_pitch_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    "right_shoulder_pitch_joint": 0.2,
  },
  joint_vel={".*": 0.0},
)

# Stilt contact geoms get condim=3 (frictional) + higher priority; everything
# else gets condim=1 (frictionless, just for self-collision detection).
STILT_G1_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  condim={
    r"^(left|right)_stilt_[lr][1-4]_collision$": 3,
    ".*_collision": 1,
  },
  priority={r"^(left|right)_stilt_[lr][1-4]_collision$": 1},
  friction={r"^(left|right)_stilt_[lr][1-4]_collision$": (0.6,)},
)


def get_stilt_g1_robot_cfg() -> EntityCfg:
  return EntityCfg(
    init_state=STILT_G1_KEYFRAME,
    collisions=(STILT_G1_COLLISION,),
    spec_fn=get_stilt_spec,
    articulation=G1_ARTICULATION,
  )


# Actuators are identical to the stock G1 — reuse the same action scale.
STILT_G1_ACTION_SCALE = G1_ACTION_SCALE
