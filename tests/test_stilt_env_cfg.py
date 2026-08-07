"""Env wiring: DR targets, event ordering, and the rebased mass curriculum."""

from envs.stilt_g1.env_cfgs import _STILT_BODY_NAMES, stilt_g1_flat_env_cfg

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
