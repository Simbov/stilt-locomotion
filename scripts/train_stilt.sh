#!/usr/bin/env bash
# Local test training run for stilt G1 (Mac CPU).
# Usage: bash scripts/train_stilt.sh [extra args]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/../.venv/bin/activate"

python "$SCRIPT_DIR/train_stilt.py" Mjlab-Velocity-Flat-Stilt-G1 \
    --env.scene.num-envs 64 \
    --agent.max-iterations 200 \
    --agent.logger tensorboard \
    --gpu-ids None \
    "$@"
