"""Curriculum terms for the stilt G1 environment."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.curriculum_manager import CurriculumTermCfg


class stilt_mass_curriculum:
  """Widen the stilt pseudo-inertia alpha_range over training.

  alpha encodes a log-scale mass multiplier: mass scales by e^(2*alpha).
  Stages define ``step`` thresholds and the target ``alpha_range`` tuple.

  Example::

    CurriculumTermCfg(
      func=stilt_mass_curriculum,
      params={
        "event_name": "stilt_mass",
        "stages": [
          {"step":    0, "alpha_range": (0.0,   0.0)},   # fixed 0.5 kg
          {"step": 1000, "alpha_range": (-0.18, 0.18)},  # ~0.35–0.72 kg
          {"step": 2000, "alpha_range": (-0.35, 0.35)},  # ~0.25–1.0 kg
          {"step": 4000, "alpha_range": (-0.35, 0.69)},  # ~0.25–1.97 kg
        ],
      },
    )
  """

  def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv):
    event_name: str = cfg.params["event_name"]
    self._stages: list[dict] = cfg.params["stages"]
    self._term_cfg = env.event_manager.get_term_cfg(event_name)

    steps = [s["step"] for s in self._stages]
    if steps != sorted(steps):
      raise ValueError(
        f"stilt_mass_curriculum stages must be in nondecreasing step order, got {steps}."
      )

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    event_name: str,
    baseline_kg: float,
    stages: list[dict],
  ) -> dict[str, torch.Tensor]:
    del env_ids, event_name, stages

    active_range = self._term_cfg.params["alpha_range"]
    for stage in self._stages:
      if env.common_step_counter >= stage["step"]:
        active_range = stage["alpha_range"]

    self._term_cfg.params["alpha_range"] = active_range

    lo, hi = active_range
    # Report actual kg bounds for easy monitoring in tensorboard/wandb.
    return {
      "stilt_mass_min_kg": torch.tensor(baseline_kg * math.exp(2 * lo)),
      "stilt_mass_max_kg": torch.tensor(baseline_kg * math.exp(2 * hi)),
    }


class stilt_height_curriculum:
  """Widen the stilt telescope offset range over training.

  Stages define ``step`` thresholds and the target ``offset_range`` tuple, in
  metres of vertical offset applied to the ``*_stilt_post_inner`` bodies.
  Negative offset pushes the inner post down, i.e. a longer stilt.

  Example::

    CurriculumTermCfg(
      func=stilt_height_curriculum,
      params={
        "event_name": "stilt_height",
        "stages": [
          {"step": 0, "offset_range": (0.0, 0.0)},
          {"step": 750 * 24, "offset_range": (-0.020, 0.020)},
        ],
      },
    )
  """

  # Ground-to-mount height of the assembled stilt, for reporting only.
  NOMINAL_HEIGHT_M = 0.4075

  def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv):
    event_name: str = cfg.params["event_name"]
    self._stages: list[dict] = cfg.params["stages"]
    self._term_cfg = env.event_manager.get_term_cfg(event_name)

    steps = [s["step"] for s in self._stages]
    if steps != sorted(steps):
      raise ValueError(
        f"stilt_height_curriculum stages must be in nondecreasing step order, "
        f"got {steps}."
      )

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    event_name: str,
    stages: list[dict],
  ) -> dict[str, torch.Tensor]:
    del env_ids, event_name, stages

    active_range = self._term_cfg.params["ranges"]
    for stage in self._stages:
      if env.common_step_counter >= stage["step"]:
        active_range = stage["offset_range"]

    self._term_cfg.params["ranges"] = active_range

    lo, hi = active_range
    # A negative offset lengthens the stilt, so the bounds swap.
    return {
      "stilt_height_min_m": torch.tensor(self.NOMINAL_HEIGHT_M - hi),
      "stilt_height_max_m": torch.tensor(self.NOMINAL_HEIGHT_M - lo),
    }


class stilt_termination_curriculum:
  """Loosen the fall terminations early, then tighten to their final values.

  With the ankle welded the robot cannot stand passively, so early in training
  every episode ends in a fall. Terminating at the final thresholds from step
  zero gives episodes too short to discover balance at all — the policy learns
  to fall immediately instead. Starting permissive buys the episode length
  needed to stumble and recover, then tightens back to the real limits.

  Stages define ``step`` thresholds plus the target ``limit_angle`` (rad, for
  ``fell_over``) and ``minimum_height`` (m, for ``torso_too_low``).
  """

  def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv):
    self._stages: list[dict] = cfg.params["stages"]
    self._fell_over = env.termination_manager.get_term_cfg("fell_over")
    self._torso_too_low = env.termination_manager.get_term_cfg("torso_too_low")

    steps = [s["step"] for s in self._stages]
    if steps != sorted(steps):
      raise ValueError(
        f"stilt_termination_curriculum stages must be in nondecreasing step "
        f"order, got {steps}."
      )

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    stages: list[dict],
  ) -> dict[str, torch.Tensor]:
    del env_ids, stages

    active = self._stages[0]
    for stage in self._stages:
      if env.common_step_counter >= stage["step"]:
        active = stage

    self._fell_over.params["limit_angle"] = active["limit_angle"]
    self._torso_too_low.params["minimum_height"] = active["minimum_height"]

    return {
      "fell_over_limit_angle_rad": torch.tensor(active["limit_angle"]),
      "torso_min_height_m": torch.tensor(active["minimum_height"]),
    }
