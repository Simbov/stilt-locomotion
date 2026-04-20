"""Stilt G1 velocity environment configuration.

Builds on the stock G1 flat env config, swapping in the stilt MJCF and
updating all reward/sensor parameters that reference foot sites or geoms.
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import terminations as base_terminations
from mjlab.envs.mdp import dr
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ObjRef, TerrainHeightSensorCfg
from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg

from .curriculums import stilt_mass_curriculum
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

  # ── Sensors ────────────────────────────────────────────────────────────────
  # Rewire foot_height_scan to use stilt tip sites instead of stock foot sites.
  # This feeds both foot_height obs and height-based rewards (foot_clearance,
  # foot_swing_height) with the correct stilt tip positions.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "foot_height_scan":
      assert isinstance(sensor, TerrainHeightSensorCfg)
      sensor.frame = tuple(
        ObjRef(type="site", name=s, entity="robot") for s in _STILT_SITE_NAMES
      )

  # ── Rewards ────────────────────────────────────────────────────────────────
  # foot_clearance and foot_slip use asset_cfg.site_names; foot_swing_height
  # uses the contact sensor subtree (ankle_roll_link) so needs no change.
  for name in ("foot_clearance", "foot_slip"):
    cfg.rewards[name].params["asset_cfg"].site_names = _STILT_SITE_NAMES

  # Keep clearance targets same as stock G1 — robot must learn to balance first
  cfg.rewards["foot_clearance"].params["target_height"] = 0.10
  cfg.rewards["foot_swing_height"].params["target_height"] = 0.10

  # Keep air-time disabled initially — same as stock G1, enable once walking
  cfg.rewards["air_time"].weight = 0.0

  # ── Domain randomisation ───────────────────────────────────────────────────
  cfg.events["foot_friction"].params["asset_cfg"].geom_names = _STILT_GEOM_NAMES

  # Stilt mass curriculum: starts fixed at 0.5 kg, widens to ~0.25–2.0 kg.
  # alpha is a log-scale mass multiplier — mass = 0.5 * e^(2*alpha).
  # Inertia scales consistently via pseudo_inertia (not just body_mass).
  cfg.events["stilt_mass"] = EventTermCfg(
    func=dr.pseudo_inertia,
    mode="reset",
    params={
      "alpha_range": (0.0, 0.0),  # overwritten each step by the curriculum
      "asset_cfg": SceneEntityCfg(
        "robot", body_names=["left_stilt", "right_stilt"]
      ),
    },
  )

  # ── Curricula ──────────────────────────────────────────────────────────────
  if not play:
    cfg.curriculum["stilt_mass"] = CurriculumTermCfg(
      func=stilt_mass_curriculum,
      params={
        "event_name": "stilt_mass",
        # common_step_counter increments once per env step, not per training
        # iteration. With num_steps_per_env=24, multiply iter targets by 24.
        # Baseline = 1.5 kg.
        "stages": [
          # iter 0 → fixed 1.5 kg baseline
          {"step":       0, "alpha_range": (0.0,   0.0)},
          # iter 500 → ±~50%: 1.0–2.2 kg
          {"step":  500 * 24, "alpha_range": (-0.2,  0.2)},
          # iter 1000 → wider: 0.67–3.3 kg
          {"step": 1000 * 24, "alpha_range": (-0.4,  0.4)},
          # iter 2000 → aggressive upper push: 0.5–6.0 kg
          {"step": 2000 * 24, "alpha_range": (-0.55, 0.69)},
        ],
      },
    )

  # ── Terminations ───────────────────────────────────────────────────────────
  # Stilt G1 pelvis spawn height is ~1.16m (bent knees).
  # Kinematics check: ankle_roll_link sits at 0.44m, stilt tip at 0.004m.
  # A pelvis below 0.65m = stilts near horizontal = truly collapsed.
  # Previous threshold 0.85m was firing after just 13 steps (0.26s) because
  # any extra knee-bend during early training drops pelvis 0.31m — the robot
  # never had time to learn anything.
  cfg.terminations["torso_too_low"] = TerminationTermCfg(
    func=base_terminations.root_height_below_minimum,
    params={"minimum_height": 0.65},
  )

  return cfg
