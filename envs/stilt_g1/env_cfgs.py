"""Stilt G1 velocity environment configuration.

Builds on the stock G1 flat env config, swapping in the stilt MJCF and
updating all reward/sensor parameters that reference foot sites or geoms.
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg

from .stilt_robot import STILT_G1_ACTION_SCALE, get_stilt_g1_robot_cfg

# Stilt contact geom names (match MJCF after _collision suffix rename)
_STILT_GEOM_NAMES = tuple(
  f"{side}_stilt_{block}{i}_collision"
  for side in ("left", "right")
  for block in ("l", "r")
  for i in range(1, 5)
)

_STILT_SITE_NAMES = ("left_stilt_tip", "right_stilt_tip")


def stilt_g1_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = unitree_g1_flat_env_cfg(play=play)

  # ── Robot ──────────────────────────────────────────────────────────────────
  cfg.scene.entities = {"robot": get_stilt_g1_robot_cfg()}

  # Action scale is identical to stock G1 (same actuators).
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = STILT_G1_ACTION_SCALE

  # ── Rewards ────────────────────────────────────────────────────────────────
  # Foot sites → stilt tips
  for name in ("foot_clearance", "foot_swing_height", "foot_slip"):
    cfg.rewards[name].params["asset_cfg"].site_names = _STILT_SITE_NAMES

  # Stilts need more clearance than G1 feet (longer pendulum, bigger step)
  cfg.rewards["foot_clearance"].params["target_height"] = 0.25
  cfg.rewards["foot_swing_height"].params["target_height"] = 0.25

  # Enable air-time reward — stilts need a clear lift-and-plant rhythm
  cfg.rewards["air_time"].weight = 1.0

  # ── Observations ───────────────────────────────────────────────────────────
  cfg.observations["critic"].terms["foot_height"].params[
    "asset_cfg"
  ].site_names = _STILT_SITE_NAMES

  # ── Domain randomisation ───────────────────────────────────────────────────
  cfg.events["foot_friction"].params["asset_cfg"].geom_names = _STILT_GEOM_NAMES

  return cfg
