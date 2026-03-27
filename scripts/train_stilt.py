#!/usr/bin/env python3
"""Train wrapper that registers the stilt G1 environment then calls mjlab train.

Usage (called by train_stilt.sh / train_stilt.pbs):
    python scripts/train_stilt.py <task_id> [mjlab train args...]
"""
import sys
from pathlib import Path

# Make the project root importable so `import envs.stilt_g1` works.
sys.path.insert(0, str(Path(__file__).parent.parent))

import envs.stilt_g1  # noqa: F401, E402 — registers Mjlab-Velocity-Flat-Stilt-G1

from mjlab.scripts.train import main  # noqa: E402

main()
