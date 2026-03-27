#!/usr/bin/env python3
"""Play wrapper that registers the stilt G1 environment then calls mjlab play."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import envs.stilt_g1  # noqa: F401, E402

from mjlab.scripts.play import main  # noqa: E402
main()
