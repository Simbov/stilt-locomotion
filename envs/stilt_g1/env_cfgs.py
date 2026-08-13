"""Stilt G1 velocity environment configuration.

Builds on the stock G1 flat env config, swapping in the stilt MJCF and
updating all reward/sensor parameters that reference foot sites or geoms.
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  ObjRef,
  TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg

from . import metrics as stilt_metrics
from . import terminations as stilt_terminations
from .curriculums import (
  stilt_height_curriculum,
  stilt_mass_curriculum,
  stilt_termination_curriculum,
)
from .events import reset_stilt_spawn_height, reset_stilts_fitted
from .stilt_robot import (
  STILT_G1_ACTION_SCALE,
  STILT_NOMINAL_POST_INNER_Z,
  STILT_SPAWN_RISE,
  STILT_TIP_SITE_RISE,
  get_stilt_g1_robot_cfg,
)

# Stilt contact geom names (match MJCF after _collision suffix rename)
_STILT_GEOM_NAMES = tuple(
  f"{side}_stilt_{block}{i}_collision"
  for side in ("left", "right")
  for block in ("l", "r")
  for i in range(1, 5)
)

# The robot's own foot capsules, live whenever the stilts are off.
_FOOT_GEOM_NAMES = tuple(
  f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 8)
)

# One site per leg, at the surface that actually touches the ground. It sits at
# the stilt plate when the stilts are fitted and is slid up to the robot's own
# sole when they are not — see reset_stilts_fitted.
_STILT_SITE_NAMES = ("left_stilt_tip", "right_stilt_tip")

# Name of the per-capsule ground-reaction sensor, read by the viewer load panel.
STILT_CONTACT_SENSOR = "stilt_contact"

# Frames of observation history given to the actor so it can sense whether the
# stilts are fitted. See the observations block below.
STILT_OBS_HISTORY = 5

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

  # ── Observations ───────────────────────────────────────────────────────────
  # Memory. The stilts come on and off, and nothing in the observation says
  # which. The actor has to infer it from how the robot responds, which is
  # impossible from a single frame — the two morphologies look identical at one
  # instant and differ only in their dynamics. A short history makes the
  # inference available without telling the policy anything the robot cannot
  # sense on hardware.
  #
  # 5 frames x 99 = 495 inputs. Raise if the policy struggles to distinguish the
  # two modes; lower if the deploy runtime cannot buffer this much.
  cfg.observations["actor"].history_length = STILT_OBS_HISTORY
  cfg.observations["actor"].flatten_history_dim = True

  # ── Rewards ────────────────────────────────────────────────────────────────
  # foot_clearance and foot_slip use asset_cfg.site_names; foot_swing_height
  # uses the contact sensor subtree (ankle_roll_link) so needs no change.
  for name in ("foot_clearance", "foot_slip"):
    cfg.rewards[name].params["asset_cfg"].site_names = _STILT_SITE_NAMES

  # Keep clearance targets same as stock G1 — robot must learn to balance first
  cfg.rewards["foot_clearance"].params["target_height"] = 0.10
  cfg.rewards["foot_swing_height"].params["target_height"] = 0.10

  # Air time: pay for actually picking a stilt up. Off in stock G1 (weight 0.0)
  # and left off through Run 6, which balanced well but converged to a single
  # ~0.35 m/s shuffle — nothing in the reward paid for taking a step.
  #
  # feet_air_time counts feet whose air time is in [0.05, 0.5] s, so it maxes at
  # 2. Rewards are dt-scaled (0.02), giving 0.04 * weight per step; at 0.5 that
  # is half the 0.04/step available from track_linear_velocity (weight 2.0).
  # This is the main knob to tune if the gait comes out too hoppy or too flat.
  #
  # The sensor is a SUBTREE match on ankle_roll_link, so it picks up the stilt
  # capsules rather than the disabled original foot geoms — verified reporting
  # 10-16 contacts against terrain in the stilt env.
  cfg.rewards["air_time"].weight = 0.5
  # Default 0.5 would only reward stepping above half the capped command range;
  # 0.3 keeps it active across most of it.
  cfg.rewards["air_time"].params["command_threshold"] = 0.3

  # NOTE: no `alive` / `terminated` survival shaping here, deliberately, and
  # SMALL-SCALE SMOKE RUNS WILL LOOK LIKE THEY NEED IT. Don't be fooled.
  #
  # At 128 envs on CPU, episode length climbs to ~60 and then falls steadily
  # while the return rises — the classic "dying is cheaper than trying" shape,
  # because the per-step reward is net negative early and truncating the episode
  # truncates the penalty. It is tempting to read that as a broken reward.
  #
  # It is not. Stock G1 — mjlab's own task, none of this project's code — does
  # exactly the same thing at 128 envs, and worse: it peaks at 63 and is down to
  # 9.9 by iteration 33, against 29 at iteration 40 for this config. It is a
  # small-batch artifact of 128x24 = 3072 samples per update. At 4096 envs the
  # same reward structure took Run 7 to 1000/1000 with zero falls.
  #
  # So: judge a smoke run by whether it beats the stock-G1 control at the SAME
  # env count, not by whether episode length rises monotonically. And an `alive`
  # bonus large enough to matter (+0.04/step) would outweigh velocity tracking
  # (+0.012/step) and bias the policy toward standing still, so it stays out.

  # ── Domain randomisation ───────────────────────────────────────────────────
  # Both contact sets, because both are used: the stilt capsules when the stilts
  # are fitted and the robot's own foot capsules when they are not.
  cfg.events["foot_friction"].params["asset_cfg"].geom_names = (
    _STILT_GEOM_NAMES + _FOOT_GEOM_NAMES
  )

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

  # Telescope height. Moves the whole lower assembly — inner tube, ground plate,
  # contact capsules and tip site — so the stilt genuinely gets longer or
  # shorter. shared_random because both legs are set to the same length on real
  # hardware; the spawn-height term below relies on that.
  cfg.events["stilt_height"] = EventTermCfg(
    func=dr.body_pos,
    mode="reset",
    params={
      "ranges": (0.0, 0.0),  # overwritten each step by the curriculum
      "asset_cfg": SceneEntityCfg("robot", body_names=list(_STILT_INNER_POST_BODIES)),
      "axes": [2],
      "operation": "add",
      "shared_random": True,
    },
  )

  # Stilts on or off, drawn per environment. One event because mass, contact
  # geometry, tip sites and brace stiffness all have to agree.
  #
  # MUST follow stilt_mass: it scales whatever mass the curriculum sampled, so
  # running it first would let pseudo_inertia put the full mass back.
  cfg.events["stilts_fitted"] = EventTermCfg(
    func=reset_stilts_fitted,
    mode="reset",
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      "fitted_probability": 0.5,
      # A bolted clamp onto a rigid shank is close to rigid. The real value is
      # unmeasured, so randomise wide and make the policy cope with all of it.
      "brace_stiffness_range": (150.0, 2000.0),
      "tip_site_rise": STILT_TIP_SITE_RISE,
    },
  )

  # MUST follow stilts_fitted AND stilt_height: EventManager runs reset terms in
  # dict order, and scene.reset() (the keyframe) has already placed the robot at
  # its stilts-off height by this point.
  cfg.events["stilt_spawn_height"] = EventTermCfg(
    func=reset_stilt_spawn_height,
    mode="reset",
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=list(_STILT_INNER_POST_BODIES)),
      "spawn_rise": STILT_SPAWN_RISE,
      "nominal_z": STILT_NOMINAL_POST_INNER_Z,
    },
  )

  # ── Curricula ──────────────────────────────────────────────────────────────
  if not play:
    # Cap commanded velocity near what the hardware can actually do. Stock G1
    # ramps to lin_vel_x (-2.0, 3.0); Run 6 inherited (-1.5, 2.0) and spent most
    # of training on commands it could not meet, achieving ~0.35 m/s and simply
    # freezing above ~0.7. Capping at 0.8 keeps headroom above the current gait
    # without training against impossible targets.
    #
    # Yaw starts tighter than stock too: Run 6 produced ~0.005 rad/s for any yaw
    # command, so it needs an easier target to get any gradient at all.
    cfg.curriculum["command_vel"].params["velocity_stages"] = [
      {
        "step": 0,
        "lin_vel_x": (-0.3, 0.5),
        "lin_vel_y": (-0.3, 0.3),
        "ang_vel_z": (-0.3, 0.3),
      },
      {
        "step": 1000 * 24,
        "lin_vel_x": (-0.5, 0.7),
        "lin_vel_y": (-0.4, 0.4),
        "ang_vel_z": (-0.5, 0.5),
      },
      {
        "step": 3000 * 24,
        "lin_vel_x": (-0.6, 0.8),
        "lin_vel_y": (-0.5, 0.5),
        "ang_vel_z": (-0.6, 0.6),
      },
    ]

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

    # Telescope length. Held fixed while the policy learns to walk at all, then
    # widened. Capped at +/-50 mm until the minimum safe tube overlap is
    # confirmed — see CLAUDE.md.
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
        ],
      },
    )

    # Start permissive so early episodes are long enough to discover balance,
    # then tighten to the real limits. The final angle matches the stock G1
    # fell_over (1.2217 rad = 70°); the two height floors are per morphology,
    # roughly 60% of that morphology's standing height (see terminations.py).
    cfg.curriculum["stilt_termination"] = CurriculumTermCfg(
      func=stilt_termination_curriculum,
      params={
        "stages": [
          {"step": 0, "limit_angle": 1.5708, "height": 0.28, "fitted_height": 0.40},
          {
            "step": 300 * 24,
            "limit_angle": 1.4000,
            "height": 0.34,
            "fitted_height": 0.50,
          },
          {
            "step": 800 * 24,
            "limit_angle": 1.3000,
            "height": 0.40,
            "fitted_height": 0.58,
          },
          {
            "step": 1500 * 24,
            "limit_angle": 1.2217,
            "height": 0.45,
            "fitted_height": 0.65,
          },
        ],
      },
    )

  # ── Metrics ────────────────────────────────────────────────────────────────
  # Split by morphology. The aggregate cannot distinguish "walks on stilts,
  # falls over without them" from "mediocre at both" — see metrics.py for how to
  # read the masked means.
  cfg.metrics["stilts_fitted_fraction"] = MetricsTermCfg(
    func=stilt_metrics.stilts_fitted_fraction
  )
  cfg.metrics["vel_error_stilts_on"] = MetricsTermCfg(
    func=stilt_metrics.vel_error_stilts_on
  )
  cfg.metrics["vel_error_stilts_off"] = MetricsTermCfg(
    func=stilt_metrics.vel_error_stilts_off
  )
  cfg.metrics["upright_stilts_on"] = MetricsTermCfg(
    func=stilt_metrics.upright_stilts_on
  )
  cfg.metrics["upright_stilts_off"] = MetricsTermCfg(
    func=stilt_metrics.upright_stilts_off
  )

  # ── Terminations ───────────────────────────────────────────────────────────
  # Two floors, because the two morphologies stand 44 cm apart: 0.7565 m at the
  # pelvis without the stilts and 1.1977 m with them. One shared number cannot
  # work — 0.65 m is a collapsed stilt walker but a perfectly normal squat for
  # the bare robot.
  #
  # Both are deliberately generous. An earlier 0.85 m threshold on the fitted
  # case fired after 13 steps (0.26 s) on any extra knee bend and the robot never
  # learned anything. Below these the leg is near horizontal and unrecoverable.
  cfg.terminations["torso_too_low"] = TerminationTermCfg(
    func=stilt_terminations.root_height_below_minimum,
    params={"minimum_height": 0.45, "fitted_minimum_height": 0.65},
  )

  return cfg
