#!/usr/bin/env bash
# Convenience wrapper around mjlab's train command.
# Usage: bash scripts/train.sh [extra args]
#
# Example:
#   bash scripts/train.sh --env.scene.num-envs 4096

set -euo pipefail

source "$(dirname "$0")/../.venv/bin/activate"

uv run train Mjlab-Velocity-Flat-Unitree-G1 \
    --env.scene.num-envs 4096 \
    "$@"
