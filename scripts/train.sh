#!/usr/bin/env bash
# Run a local test training run on Mac (CPU).
# Usage: bash scripts/train.sh [extra args]
#
# Example:
#   bash scripts/train.sh --agent.max-iterations 500

set -euo pipefail

source "$(dirname "$0")/../.venv/bin/activate"

train Mjlab-Velocity-Flat-Unitree-G1 \
    --env.scene.num-envs 64 \
    --agent.max-iterations 200 \
    --agent.logger tensorboard \
    --gpu-ids None \
    "$@"
