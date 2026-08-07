"""Reset events specific to the telescoping stilt.

`ManagerBasedRlEnv._reset_idx` applies `scene.reset()` — and therefore the
keyframe — *before* it applies `mode="reset"` events. So the spawn height baked
into the keyframe cannot know the sampled stilt height.
`reset_stilt_spawn_height` runs after the height DR term and corrects the root
pose to match.

Registration order matters: `EventManager` builds its term list by iterating the
config dict, so this term must be inserted into `cfg.events` *after* the height
randomisation term (and after the base env's `reset_base`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def spawn_height_correction(sampled_z: float, nominal_z: float) -> float:
  """Root height correction for a telescope offset.

  A longer stilt pushes ``*_stilt_post_inner`` further down (more negative
  ``body_pos`` z), so the root must rise by the same amount.
  """
  return nominal_z - sampled_z


def reset_stilt_spawn_height(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  asset_cfg: SceneEntityCfg,
  nominal_z: float,
) -> None:
  """Raise the root so randomised stilts still rest on the floor at spawn."""
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)

  asset = env.scene[asset_cfg.name]

  # asset_cfg.body_ids are entity-local, but sim.model.body_pos is indexed by
  # global body id — map through the entity's indexing or the readback is zero.
  body_ids = asset.indexing.body_ids[asset_cfg.body_ids]

  # Correct for the LONGEST stilt (most negative z). With shared_random=True on
  # the height term both stilts match, but taking the max keeps the robot clear
  # of the floor even if they ever diverge.
  sampled_z = env.sim.model.body_pos[env_ids][:, body_ids, 2]
  correction = (nominal_z - sampled_z).max(dim=1).values

  pose = torch.cat(
    [
      asset.data.root_link_pos_w[env_ids].clone(),
      asset.data.root_link_quat_w[env_ids].clone(),
    ],
    dim=-1,
  )
  pose[:, 2] += correction
  asset.write_root_link_pose_to_sim(pose, env_ids=env_ids)
