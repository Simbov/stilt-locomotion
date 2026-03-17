#!/usr/bin/env bash
# Watches for new checkpoints and reloads the viser viewer automatically.
# Opens http://localhost:8080 — does not affect TensorBoard on :6006
#
# Usage: bash scripts/watch_training.sh

set -euo pipefail

source "$(dirname "$0")/../.venv/bin/activate"

LOGS_DIR="logs/rsl_rl/g1_velocity"
TASK="Mjlab-Velocity-Flat-Unitree-G1"
NUM_ENVS=4

echo "Watching for new checkpoints in $LOGS_DIR ..."
echo "Open http://localhost:8080 to watch the robots"
echo ""

LAST_CHECKPOINT=""
PLAY_PID=""

while true; do
    # Find the latest checkpoint across all runs
    LATEST=$(find "$LOGS_DIR" -name "model_*.pt" | sort -t_ -k2 -V | tail -1)

    if [[ "$LATEST" != "$LAST_CHECKPOINT" && -n "$LATEST" ]]; then
        echo "New checkpoint: $LATEST"

        # Kill the previous viewer if running
        if [[ -n "$PLAY_PID" ]] && kill -0 "$PLAY_PID" 2>/dev/null; then
            kill "$PLAY_PID"
            sleep 1
        fi

        # Launch viewer with latest checkpoint
        play "$TASK" \
            --agent trained \
            --checkpoint-file "$LATEST" \
            --num-envs "$NUM_ENVS" \
            --viewer viser &
        PLAY_PID=$!
        LAST_CHECKPOINT="$LATEST"
    fi

    sleep 10  # Check for new checkpoint every 10 seconds
done
