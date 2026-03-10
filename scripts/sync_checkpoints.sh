#!/usr/bin/env bash
# Pull training checkpoints from a remote cloud instance.
# Usage: REMOTE=user@host bash scripts/sync_checkpoints.sh
#
# Requires REMOTE env var pointing to your cloud GPU instance.

set -euo pipefail

: "${REMOTE:?Set REMOTE=user@host before running this script}"

REMOTE_PATH="${REMOTE}:~/stilt-locomotion/runs/"
LOCAL_PATH="./runs/"

mkdir -p "$LOCAL_PATH"
rsync -avz --progress "$REMOTE_PATH" "$LOCAL_PATH"
echo "Checkpoints synced to $LOCAL_PATH"
