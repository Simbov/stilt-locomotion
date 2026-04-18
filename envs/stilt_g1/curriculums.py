"""Curriculum terms for the stilt G1 environment."""

from __future__ import annotations

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
    import math
    return {
      "stilt_mass_min_kg": torch.tensor(0.5 * math.exp(2 * lo)),
      "stilt_mass_max_kg": torch.tensor(0.5 * math.exp(2 * hi)),
    }
