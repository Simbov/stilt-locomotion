"""Reset events specific to the telescoping stilt.

The robot is always the stock 29-DoF G1. The stilts bolt on and come off, so
every episode draws one of two morphologies and the reset has to make the whole
model agree with the draw. Two events do that, in this order:

  1. `reset_stilts_fitted`      — mass, contact geometry, tip sites and the
                                  brace spring across the ankle
  2. `reset_stilt_spawn_height` — the root height that follows from 1, plus the
                                  sampled telescope length

Ordering is load-bearing twice over. `ManagerBasedRlEnv._reset_idx` applies
`scene.reset()` — and therefore the keyframe — *before* it runs any `mode="reset"`
event, so the keyframe cannot know either the morphology or the sampled stilt
height. And `EventManager` builds its term list by iterating the config dict, so
these two must be inserted into `cfg.events` after `reset_base`, after the mass
term and after the height term, in that order.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def spawn_height_correction(sampled_z: float, nominal_z: float) -> float:
  """Root height correction for a telescope offset.

  A longer stilt pushes ``*_stilt_post_inner`` further down (more negative
  ``body_pos`` z), so the root must rise by the same amount.
  """
  return nominal_z - sampled_z


# --- Stilts on / off -------------------------------------------------------
#
# Five things move together, which is why this is one custom event rather than
# five independent DR terms:
#
#   1. stilt segment mass      -> as sampled when fitted, ~0 when off
#   2. stilt contact capsules  -> parked out of the world when off, so the
#                                 robot's own foot geoms make ground contact
#   3. stilt tip sites         -> slid up to the robot's own sole when off, so
#                                 foot_height / foot_clearance / foot_slip keep
#                                 measuring the surface actually touching down
#   4. ankle spring stiffness  -> the brace clamps the shank when fitted
#   5. root spawn height       -> 44 cm higher when fitted (reset_stilt_spawn_height)
#
# The standing POSE is not among them: both morphologies use the same one, and
# STILT_LEG_POSE in stilt_robot.py explains why splitting it collapses training.
#
# The per-env choice is stored on the env as `stilt_fitted` so the spawn-height
# term can read it, and so the viewer can show it.

_STILT_SEGMENTS = (
  "stilt_mount",
  "stilt_brace",
  "stilt_post_outer",
  "stilt_post_inner",
  "stilt_plate",
)
# Where the stilt contact capsules go when the stilts are off: far enough from
# the robot that they touch nothing. They keep their names so every downstream
# regex, sensor and reward keeps resolving.
_PARK_OFFSET_M = (0.0, 0.0, 6.0)
# Leaving a trace of mass rather than exactly zero: a strictly zero-mass body in
# a kinematic chain is legal in MuJoCo but pointlessly close to singular, and
# 2.8 g is far below anything the policy can feel.
_REMOVED_MASS_FRACTION = 1e-3


# The decorator is how per-world storage for these fields gets allocated: the
# EventManager collects them at construction and expands ONCE. Do not call
# sim.expand_model_fields() from inside the event instead — it recreates the CUDA
# graph on every call, so on GPU that is a full graph recapture per reset. The
# decorator also drives the recompute, so no manual recompute_constants here.
@requires_model_fields(
  "body_mass",
  "body_inertia",
  "geom_pos",
  "site_pos",
  "jnt_stiffness",
  recompute=RecomputeLevel.set_const,
)
def reset_stilts_fitted(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  asset_cfg: SceneEntityCfg,
  fitted_probability: float,
  brace_stiffness_range: tuple[float, float],
  tip_site_rise: float,
) -> None:
  """Fit or remove the stilts, per environment."""
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)

  asset = env.scene[asset_cfg.name]
  sim = env.sim

  fitted = (torch.rand(len(env_ids), device=env.device) < fitted_probability).float()
  if not hasattr(env, "stilt_fitted"):
    env.stilt_fitted = torch.ones(env.num_envs, device=env.device)
  env.stilt_fitted[env_ids] = fitted

  body_names = [b.name.split("/")[-1] for b in asset.indexing.bodies]
  site_names = [s.name.split("/")[-1] for s in asset.indexing.sites]
  joint_names = list(asset.joint_names)

  # 1. Segment mass and inertia.
  #
  # Scaled off the CURRENT value, not the default: the `stilt_mass` curriculum
  # term runs before this one and writes the sampled mass through
  # dr.pseudo_inertia. Reading the default here would silently undo it in every
  # fitted env. pseudo_inertia itself always writes from the default, so the two
  # compose without compounding across resets.
  scale = fitted + (1.0 - fitted) * _REMOVED_MASS_FRACTION
  for segment in _STILT_SEGMENTS:
    for side in ("left", "right"):
      bid = int(asset.indexing.body_ids[body_names.index(f"{side}_{segment}")])
      sim.model.body_mass[env_ids, bid] *= scale
      sim.model.body_inertia[env_ids, bid] *= scale.unsqueeze(-1)

  # 2. Park every stilt geom when the stilts are off — the contact capsules so
  #    the robot's own feet take the ground, and the visual meshes so the render
  #    matches reality.
  #
  #    Selected by PARENT BODY, not by name. The visual meshes are unnamed in
  #    the MJCF (`<geom class="visual" mesh="stilt_plate"/>`), so a name filter
  #    silently skips all ten of them and the "stilts off" robot renders wearing
  #    ghost stilts sunk through the floor. Harmless physically — visual geoms
  #    do not collide — but it makes every video and viewer session misleading.
  park = torch.tensor(_PARK_OFFSET_M, device=env.device)
  removed = (1.0 - fitted).unsqueeze(-1)
  for i, geom in enumerate(asset.indexing.geoms):
    parent = geom.parent.name.split("/")[-1] if geom.parent is not None else ""
    if "stilt" not in parent:
      continue
    gid = int(asset.indexing.geom_ids[i])
    sim.model.geom_pos[env_ids, gid] = sim.get_default_field("geom_pos")[gid] + (
      removed * park
    )

  # 3. Slide the tip sites up to the robot's own sole when the stilts are off.
  #    foot_height_scan, foot_clearance and foot_slip all read these sites; left
  #    at the stilt plate they would report a sole 0.4 m underground and poison
  #    every height-based reward in half the envs.
  for i, sname in enumerate(site_names):
    if not sname.endswith("_stilt_tip"):
      continue
    sid = int(asset.indexing.site_ids[i])
    nominal = sim.get_default_field("site_pos")[sid]
    sim.model.site_pos[env_ids, sid] = nominal
    sim.model.site_pos[env_ids, sid, 2] = (
      nominal[2] + removed.squeeze(-1) * tip_site_rise
    )

  # 4. The brace across the ankle. A bolted clamp onto a rigid shank is close to
  #    rigid, but the stiffness is unmeasured, so it is randomised wide and the
  #    policy has to cope with the whole range. The spring pulls toward zero,
  #    which is the angle the MJCF assembles the mount and brace at.
  low, high = brace_stiffness_range
  k = low + torch.rand(len(env_ids), device=env.device) * (high - low)
  for i, jname in enumerate(joint_names):
    if "ankle" not in jname:
      continue
    jid = int(asset.indexing.joint_ids[i])
    sim.model.jnt_stiffness[env_ids, jid] = k * fitted

  # The standing pose is deliberately NOT switched here. Both morphologies use
  # the one keyframe pose — see STILT_LEG_POSE in stilt_robot.py for why a split
  # pose collapses training.


def reset_stilt_spawn_height(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  asset_cfg: SceneEntityCfg,
  spawn_rise: float,
  nominal_z: float,
) -> None:
  """Raise the root to match the morphology and the sampled telescope length.

  Must run after both `stilts_fitted` and the height randomisation.
  """
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
  telescope = (nominal_z - sampled_z).max(dim=1).values

  fitted = getattr(env, "stilt_fitted", torch.ones(env.num_envs, device=env.device))
  correction = (spawn_rise + telescope) * fitted[env_ids]

  # Apply the correction as an in-place delta on the authoritative sim state.
  #
  # Do NOT read asset.data.root_link_pos_w here and write it back: that view is
  # cached and still holds the PREVIOUS episode's pose at this point in the
  # reset, because `reset_base` wrote the fresh pose straight to the sim without
  # invalidating it. Round-tripping through it silently restores the fallen pose
  # and every episode after the first one spawns already collapsed.
  # free_joint_q_adr is the 7 addresses [x, y, z, qw, qx, qy, qz].
  root_z_adr = int(asset.indexing.free_joint_q_adr[2])
  env.sim.data.qpos[env_ids, root_z_adr] += correction
