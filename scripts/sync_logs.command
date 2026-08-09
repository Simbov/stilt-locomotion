#!/bin/bash
# Double-click this file in Finder to pull training logs down from the HPC.
# Wraps scripts/sync_logs.sh (which holds the host/user settings) so there is
# only one place to edit the connection details.

# Always use the local Desktop copy — matches the other .command launchers.
PROJECT_DIR="$HOME/Desktop/stilt-locomotion"
cd "$PROJECT_DIR" || {
    osascript -e 'display dialog "Could not find ~/Desktop/stilt-locomotion — exiting." buttons {"OK"} default button "OK" with icon stop'
    exit 1
}

HOST=$(grep -m1 '^HPC_HOST=' scripts/sync_logs.sh | cut -d'"' -f2)

echo "════════════════════════════════════════════════════════════"
echo "  Sync training logs from HPC"
echo "  host: $HOST"
echo "════════════════════════════════════════════════════════════"
echo ""

# ── Dry run first, so nothing is a surprise ──────────────────────────────────
echo "▶  Checking what would be transferred (dry run)…"
echo ""
if ! bash scripts/sync_logs.sh --dry-run; then
    echo ""
    echo "✗  Could not reach the HPC."
    echo "   Check the VPN, your SSH key, and HPC_HOST in scripts/sync_logs.sh"
    echo ""
    echo "Press any key to close…"
    read -n 1 -s
    exit 1
fi

echo ""
CHOICE=$(osascript -e 'display dialog "Dry run finished — see the Terminal window for what will be downloaded.\n\nDownload it now?" buttons {"Cancel", "Download"} default button "Download" with icon note' -e 'button returned of result' 2>/dev/null)

if [ "$CHOICE" != "Download" ]; then
    echo "Cancelled — nothing downloaded."
    echo ""
    echo "Press any key to close…"
    read -n 1 -s
    exit 0
fi

echo ""
echo "▶  Downloading…"
echo ""
bash scripts/sync_logs.sh
STATUS=$?

echo ""
if [ $STATUS -eq 0 ]; then
    LATEST=$(ls -1dt logs/rsl_rl/*/ 2>/dev/null | head -1)
    echo "✓  Done. Most recent run: ${LATEST:-none found}"
    if [ -n "$LATEST" ]; then
        echo "   Checkpoints: $(ls -1 "$LATEST"*.pt 2>/dev/null | wc -l | tr -d ' ')"
    fi
    echo ""
    echo "   Double-click scripts/visualise.command to view a checkpoint."
else
    echo "✗  rsync exited with status $STATUS"
fi

echo ""
echo "Press any key to close…"
read -n 1 -s
