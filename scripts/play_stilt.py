#!/usr/bin/env python3
"""Play wrapper: registers the stilt G1 env, injects viewer GUI, then runs mjlab play."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import envs.stilt_g1  # noqa: F401, E402
from envs.stilt_g1 import _stilt_mass_play_gui  # noqa: E402

import mjlab.scripts.play as _play_mod  # noqa: E402
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
