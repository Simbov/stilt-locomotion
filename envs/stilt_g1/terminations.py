"""Terminations that depend on which morphology the episode drew.

The two morphologies stand 44 cm apart — 0.76 m at the pelvis without the
stilts, 1.20 m with them — so a single "torso too low" floor cannot serve both.
A threshold tight enough to catch a collapsed stilt walker sits above the bare
robot's standing height; one loose enough for the bare robot lets a stilt walker
sink almost to the ground before the episode ends.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def root_height_below_minimum(
  env: ManagerBasedRlEnv,
  minimum_height: float,
  fitted_minimum_height: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Terminate when the pelvis drops below the floor for this morphology.

  Args:
    minimum_height: Floor for envs with the stilts off, relative to terrain.
    fitted_minimum_height: Floor for envs with the stilts on.
  """
  asset = env.scene[asset_cfg.name]
  fitted = getattr(env, "stilt_fitted", torch.ones(env.num_envs, device=env.device))
  threshold = fitted * fitted_minimum_height + (1.0 - fitted) * minimum_height
  height = asset.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
  return height < threshold
