#!/usr/bin/env python3
"""Play wrapper: registers the stilt G1 env, injects viewer GUI, then runs mjlab play."""

import os as _os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import mjlab.scripts.play as _play_mod  # noqa: E402

import envs.stilt_g1  # noqa: F401, E402
from envs.stilt_g1 import _stilt_mass_play_gui  # noqa: E402

# Optional: pin the morphology instead of drawing it 50/50 per episode.
#   STILT_FITTED_PROB=0.0  -> always bare (matches the hardware bring-up)
#   STILT_FITTED_PROB=1.0  -> always on stilts
# The registry deep-copies on load, so mutating it here is enough.
_prob = _os.environ.get("STILT_FITTED_PROB")
if _prob is not None:
  from mjlab.tasks.registry import _REGISTRY  # noqa: E402

  for _task in _REGISTRY.values():
    for _cfg in (_task.env_cfg, _task.play_env_cfg):
      _events = getattr(_cfg, "events", None)
      if _events and "stilts_fitted" in _events:
        _events["stilts_fitted"].params["fitted_probability"] = float(_prob)
  print(f"[play_stilt] fitted_probability pinned to {_prob}")

from mjlab.viewer.viser.viewer import ViserPlayViewer  # noqa: E402


class _StiltViserViewer(ViserPlayViewer):
  def setup(self) -> None:
    super().setup()
    # Expose sim primitives so the GUI callbacks can acquire the sim lock
    # and request a visual refresh safely.
    self.env.unwrapped.sim_lock = self._sim_lock
    self.env.unwrapped.sim_scene = self._scene
    _stilt_mass_play_gui(self._server, self.env)


_play_mod.ViserPlayViewer = _StiltViserViewer

_play_mod.main()
