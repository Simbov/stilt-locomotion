"""Per-morphology metrics.

One policy has to walk with the stilts on and with them off. The aggregate
numbers cannot tell those apart: a policy that walks well on stilts and falls
over without them logs the same mean tracking error as one that is mediocre at
both. These terms split it.

`MetricsManager` averages each term over all envs, so the split is done by
masking rather than by selecting — a masked mean is the conditional mean scaled
by that mode's share of the envs. To read the numbers, divide:

    E[error | stilts on]  = vel_error_stilts_on  / stilts_fitted_fraction
    E[error | stilts off] = vel_error_stilts_off / (1 - stilts_fitted_fraction)

With `fitted_probability` at 0.5 the denominators sit near 0.5, so the raw
curves are already directly comparable to each other — just not to the
unconditional error.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _fitted(env: ManagerBasedRlEnv) -> torch.Tensor:
  return getattr(env, "stilt_fitted", torch.ones(env.num_envs, device=env.device))


def _tracking_error(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Planar velocity tracking error, per env."""
  command = env.command_manager.get_command("twist")
  actual = env.scene["robot"].data.root_link_lin_vel_b[:, :2]
  return torch.linalg.norm(command[:, :2] - actual, dim=-1)


def stilts_fitted_fraction(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Share of envs running with the stilts on. Sanity check on the draw, and the
  denominator for the two masked errors below."""
  return _fitted(env)


def vel_error_stilts_on(env: ManagerBasedRlEnv) -> torch.Tensor:
  return _tracking_error(env) * _fitted(env)


def vel_error_stilts_off(env: ManagerBasedRlEnv) -> torch.Tensor:
  return _tracking_error(env) * (1.0 - _fitted(env))


def upright_stilts_on(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Masked uprightness, as a proxy for "is it still standing in this mode".

  projected_gravity z is -1 when perfectly upright and 0 on its side, so this
  rises toward the mode's env share as the robot stays up.
  """
  gravity_z = env.scene["robot"].data.projected_gravity_b[:, 2]
  return (-gravity_z).clamp(min=0.0) * _fitted(env)


def upright_stilts_off(env: ManagerBasedRlEnv) -> torch.Tensor:
  gravity_z = env.scene["robot"].data.projected_gravity_b[:, 2]
  return (-gravity_z).clamp(min=0.0) * (1.0 - _fitted(env))
