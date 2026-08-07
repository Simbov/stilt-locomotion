"""Stilt G1 velocity environment configuration.

Builds on the stock G1 flat env config, swapping in the stilt MJCF and
updating all reward/sensor parameters that reference foot sites or geoms.
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp import terminations as base_terminations
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  ObjRef,
  TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg

from .curriculums import stilt_height_curriculum, stilt_mass_curriculum
from .events import reset_stilt_spawn_height
from .stilt_robot import (
  STILT_G1_ACTION_SCALE,
  STILT_NOMINAL_POST_INNER_Z,
  get_stilt_g1_robot_cfg,
)

# Stilt contact geom names (match MJCF after _collision suffix rename)
_STILT_GEOM_NAMES = tuple(
  f"{side}_stilt_{block}{i}_collision"
  for side in ("left", "right")
  for block in ("l", "r")
  for i in range(1, 5)
)

_STILT_SITE_NAMES = ("left_stilt_tip", "right_stilt_tip")

# Name of the per-capsule ground-reaction sensor, read by the viewer load panel.
STILT_CONTACT_SENSOR = "stilt_contact"

# The five rigid segments per stilt, from the MJCF.
_STILT_SEGMENTS = (
  "stilt_mount",
  "stilt_brace",
  "stilt_post_outer",
  "stilt_post_inner",
  "stilt_plate",
)

_STILT_BODY_NAMES = tuple(
  f"{side}_{segment}" for side in ("left", "right") for segment in _STILT_SEGMENTS
)

# Moving this body slides the inner tube, ground plate, contact capsules and
# tip site together — the whole telescoping lower assembly.
_STILT_INNER_POST_BODIES = ("left_stilt_post_inner", "right_stilt_post_inner")


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

  # Per-capsule ground reaction, for the load readout in the viewer. One column
  # per capsule, net wrench in the global frame. mujoco_warp does not expose
  # per-contact forces through get_data_into (geom ids come back as zeros), so
  # this sensor is the supported way to get them.
  cfg.scene.sensors = tuple(cfg.scene.sensors or ()) + (
    ContactSensorCfg(
      name=STILT_CONTACT_SENSOR,
      primary=ContactMatch(
        mode="geom",
        pattern=(r"^(left|right)_stilt_[lr][1-4]_collision$",),
        entity="robot",
      ),
      fields=("found", "force", "pos"),
      reduce="netforce",
      global_frame=True,
    ),
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

  # The stock G1 pose reward keys per-joint std on regexes that include the
  # ankles. Those joints no longer exist, and mjlab raises if a regex matches
  # nothing, so drop those entries.
  for std_key in ("std_standing", "std_walking", "std_running"):
    stds = cfg.rewards["pose"].params[std_key]
    for pattern in [k for k in stds if "ankle" in k]:
      del stds[pattern]

  # ── Domain randomisation ───────────────────────────────────────────────────
  cfg.events["foot_friction"].params["asset_cfg"].geom_names = _STILT_GEOM_NAMES

  # Stilt mass curriculum. alpha is a log-scale mass multiplier applied to every
  # segment: mass = 2.8 * e^(2*alpha) per stilt. Inertia scales consistently via
  # pseudo_inertia (not just body_mass).
  cfg.events["stilt_mass"] = EventTermCfg(
    func=dr.pseudo_inertia,
    mode="reset",
    params={
      "alpha_range": (0.0, 0.0),  # overwritten each step by the curriculum
      "asset_cfg": SceneEntityCfg("robot", body_names=list(_STILT_BODY_NAMES)),
    },
  )

  # Telescope offset on the inner post. Negative = post pushed down = longer
  # stilt. Nominal 0.0 is the assembled 407.5 mm configuration.
  cfg.events["stilt_height"] = EventTermCfg(
    func=dr.body_pos,
    mode="reset",
    params={
      "ranges": (0.0, 0.0),  # overwritten each step by the curriculum
      "axes": [2],
      "operation": "add",
      # Both stilts are physically set to the same length on the hardware.
      "shared_random": True,
      "asset_cfg": SceneEntityCfg("robot", body_names=list(_STILT_INNER_POST_BODIES)),
    },
  )

  # MUST stay after stilt_height: EventManager runs reset terms in dict order,
  # and scene.reset() (which applies the keyframe) has already run by now, so
  # the spawn height cannot know the sampled stilt length on its own.
  cfg.events["stilt_spawn_height"] = EventTermCfg(
    func=reset_stilt_spawn_height,
    mode="reset",
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=list(_STILT_INNER_POST_BODIES)),
      "nominal_z": STILT_NOMINAL_POST_INNER_Z,
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
        "baseline_kg": 2.8,
        "stages": [
          # iter 0 → fixed 2.8 kg baseline
          {"step": 0, "alpha_range": (0.0, 0.0)},
          # iter 500 → 1.9–4.2 kg
          {"step": 500 * 24, "alpha_range": (-0.2, 0.2)},
          # iter 1000 → 1.3–6.2 kg
          {"step": 1000 * 24, "alpha_range": (-0.4, 0.4)},
          # iter 2000 → 0.9–7.6 kg
          {"step": 2000 * 24, "alpha_range": (-0.55, 0.5)},
        ],
      },
    )

    cfg.curriculum["stilt_height"] = CurriculumTermCfg(
      func=stilt_height_curriculum,
      params={
        "event_name": "stilt_height",
        "stages": [
          # iter 0 → fixed 407.5 mm
          {"step": 0, "offset_range": (0.0, 0.0)},
          # iter 750 → 387–427 mm
          {"step": 750 * 24, "offset_range": (-0.020, 0.020)},
          # iter 1500 → 357–457 mm
          {"step": 1500 * 24, "offset_range": (-0.050, 0.050)},
          # The full mechanical range (352–522 mm) is gated on confirming the
          # minimum safe tube overlap — do not widen past ±50 mm until then.
        ],
      },
    )

  # ── Terminations ───────────────────────────────────────────────────────────
  # Stilt G1 pelvis spawn height is 1.1843 m (see STILT_SPAWN_HEIGHT). With the
  # ankle welded the stance is straighter than before — knee 0.30 rad cancelled
  # at the hip — so there is less pelvis drop available before the pose is
  # genuinely collapsed. A pelvis below 0.65 m means the stilts are near
  # horizontal. Keep this generous: an earlier 0.85 m threshold fired after 13
  # steps (0.26 s) on any extra knee bend, and the robot never learned anything.
  cfg.terminations["torso_too_low"] = TerminationTermCfg(
    func=base_terminations.root_height_below_minimum,
    params={"minimum_height": 0.65},
  )

  return cfg
