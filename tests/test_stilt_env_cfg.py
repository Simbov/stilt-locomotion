"""Env wiring: DR targets, event ordering, curricula and the two-mode split."""

from envs.stilt_g1.env_cfgs import (
  _FOOT_GEOM_NAMES,
  _STILT_BODY_NAMES,
  _STILT_GEOM_NAMES,
  stilt_g1_flat_env_cfg,
)
from envs.stilt_g1.stilt_robot import (
  STILT_FITTED_SPAWN_HEIGHT,
  STILT_SPAWN_HEIGHT,
)

SEGMENTS = (
  "stilt_mount",
  "stilt_brace",
  "stilt_post_outer",
  "stilt_post_inner",
  "stilt_plate",
)


def test_mass_dr_targets_every_segment():
  assert len(_STILT_BODY_NAMES) == 10
  for side in ("left", "right"):
    for segment in SEGMENTS:
      assert f"{side}_{segment}" in _STILT_BODY_NAMES


def test_spawn_correction_runs_after_height_sampling():
  """EventManager executes reset terms in config-dict order."""
  cfg = stilt_g1_flat_env_cfg()
  reset_terms = [k for k, v in cfg.events.items() if v.mode == "reset"]
  assert reset_terms.index("stilt_spawn_height") > reset_terms.index("stilt_height")


def test_spawn_correction_runs_after_reset_base():
  cfg = stilt_g1_flat_env_cfg()
  reset_terms = [k for k, v in cfg.events.items() if v.mode == "reset"]
  assert reset_terms.index("stilt_spawn_height") > reset_terms.index("reset_base")


def test_height_curriculum_starts_fixed():
  cfg = stilt_g1_flat_env_cfg()
  stages = cfg.curriculum["stilt_height"].params["stages"]
  assert stages[0]["offset_range"] == (0.0, 0.0)


def test_mass_curriculum_is_rebased_on_2_8_kg():
  cfg = stilt_g1_flat_env_cfg()
  assert cfg.curriculum["stilt_mass"].params["baseline_kg"] == 2.8


def test_play_config_has_no_curriculum():
  cfg = stilt_g1_flat_env_cfg(play=True)
  assert "stilt_height" not in cfg.curriculum


def test_stilts_fitted_runs_after_the_mass_curriculum():
  """It scales whatever mass was sampled; the other order silently undoes it."""
  cfg = stilt_g1_flat_env_cfg()
  reset_terms = [k for k, v in cfg.events.items() if v.mode == "reset"]
  assert reset_terms.index("stilts_fitted") > reset_terms.index("stilt_mass")


def test_spawn_correction_runs_after_the_morphology_is_drawn():
  cfg = stilt_g1_flat_env_cfg()
  reset_terms = [k for k, v in cfg.events.items() if v.mode == "reset"]
  assert reset_terms.index("stilt_spawn_height") > reset_terms.index("stilts_fitted")


def test_both_morphologies_are_actually_drawn():
  cfg = stilt_g1_flat_env_cfg()
  probability = cfg.events["stilts_fitted"].params["fitted_probability"]
  assert 0.0 < probability < 1.0


def test_friction_dr_covers_both_contact_sets():
  """Whichever set is live has to be randomised, or half the envs train on one
  fixed friction value."""
  cfg = stilt_g1_flat_env_cfg()
  geoms = set(cfg.events["foot_friction"].params["asset_cfg"].geom_names)
  assert set(_STILT_GEOM_NAMES) <= geoms
  assert set(_FOOT_GEOM_NAMES) <= geoms


def test_the_two_fall_floors_bracket_their_own_standing_heights():
  """One shared floor cannot serve morphologies 44 cm apart."""
  cfg = stilt_g1_flat_env_cfg()
  params = cfg.terminations["torso_too_low"].params
  bare, fitted = params["minimum_height"], params["fitted_minimum_height"]

  assert bare < STILT_SPAWN_HEIGHT, "the bare robot spawns already terminated"
  assert fitted < STILT_FITTED_SPAWN_HEIGHT, "the stilt robot spawns terminated"
  # The bare robot standing normally must not trip the fitted floor, which is
  # the exact failure a single shared threshold produces.
  assert fitted > STILT_SPAWN_HEIGHT - 0.15


def test_the_termination_curriculum_drives_both_floors():
  cfg = stilt_g1_flat_env_cfg()
  stages = cfg.curriculum["stilt_termination"].params["stages"]
  final = cfg.terminations["torso_too_low"].params
  for stage in stages:
    assert stage["height"] < stage["fitted_height"]
  assert stages[-1]["height"] == final["minimum_height"]
  assert stages[-1]["fitted_height"] == final["fitted_minimum_height"]
  # Permissive first, tightening after — the point of the curriculum.
  assert [s["height"] for s in stages] == sorted(s["height"] for s in stages)


def test_the_actor_has_memory():
  """Nothing tells the policy which morphology it is in, so it must infer it
  from the dynamics, which takes more than one frame."""
  cfg = stilt_g1_flat_env_cfg()
  assert cfg.observations["actor"].history_length > 1
  assert cfg.observations["actor"].flatten_history_dim


def test_the_policy_is_not_told_which_morphology_it_is_in():
  cfg = stilt_g1_flat_env_cfg()
  names = " ".join(cfg.observations["actor"].terms)
  assert "stilt" not in names and "fitted" not in names
