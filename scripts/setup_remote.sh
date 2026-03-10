#!/usr/bin/env bash
# Run this once on a fresh cloud GPU instance to set up the environment.
# Usage: bash scripts/setup_remote.sh

set -euo pipefail

git pull

if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi

source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "Remote setup complete. Activate with: source .venv/bin/activate"
