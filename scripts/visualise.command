#!/bin/bash
# Double-click this file in Finder to open the checkpoint picker.
# It will ask which .pt checkpoint to visualise, then open the viewer in your browser.

# Always use the local Desktop copy — the venv only exists there, not on remote mounts.
PROJECT_DIR="$HOME/Desktop/stilt-locomotion"
LOGS_DIR="$PROJECT_DIR/logs/rsl_rl"

# ── Pick a checkpoint via native macOS file dialog ────────────────────────────
CHECKPOINT=$(osascript <<APPLESCRIPT
try
    set logsFolder to POSIX file "$LOGS_DIR" as alias
    set theFile to choose file ¬
        with prompt "Select a model checkpoint to visualise:" ¬
        default location logsFolder
    return POSIX path of theFile
on error
    return ""
end try
APPLESCRIPT
)

if [ -z "$CHECKPOINT" ]; then
    osascript -e 'display dialog "No checkpoint selected — exiting." buttons {"OK"} default button "OK" with icon caution'
    exit 0
fi

CHECKPOINT_NAME=$(basename "$CHECKPOINT")
echo "▶  Loading: $CHECKPOINT"

# ── Auto-detect env from log path ─────────────────────────────────────────────
if [[ "$CHECKPOINT" == *"stilt_g1"* ]]; then
    ENV_ID="Mjlab-Velocity-Flat-Stilt-G1"
    PLAY_SCRIPT="$PROJECT_DIR/scripts/play_stilt.py"
    echo "   Env: Stilt G1"
else
    ENV_ID="Mjlab-Velocity-Flat-Unitree-G1"
    PLAY_SCRIPT=""
    echo "   Env: Stock G1"
fi

# ── Kill any existing viewer ──────────────────────────────────────────────────
lsof -ti:8080 | xargs kill -9 2>/dev/null
sleep 1

# ── Launch viewer ─────────────────────────────────────────────────────────────
source "$PROJECT_DIR/.venv/bin/activate"

if [ -n "$PLAY_SCRIPT" ]; then
    python "$PLAY_SCRIPT" "$ENV_ID" \
        --agent trained \
        --checkpoint-file "$CHECKPOINT" \
        --num-envs 1 \
        --viewer viser &
else
    play "$ENV_ID" \
        --agent trained \
        --checkpoint-file "$CHECKPOINT" \
        --num-envs 4 \
        --viewer viser &
fi

VIEWER_PID=$!
echo "   Viewer PID: $VIEWER_PID"
echo "   Waiting for viewer to start..."

# ── Wait until port 8080 is ready, then open browser ─────────────────────────
for i in {1..30}; do
    sleep 1
    if curl -s http://localhost:8080 > /dev/null 2>&1; then
        echo "   Ready! Opening browser..."
        open http://localhost:8080
        osascript -e "display notification \"Viewing: $CHECKPOINT_NAME\" with title \"mjlab Visualiser\" sound name \"Ping\"" 2>/dev/null || true
        break
    fi
done

# Keep terminal open so viewer keeps running
echo ""
echo "   Press Ctrl+C to stop the viewer."
wait $VIEWER_PID
