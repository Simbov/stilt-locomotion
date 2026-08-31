#!/usr/bin/env bash
# OPTIONAL. Makes FixStand hold the same pose the policy expects, so the
# R2+A handover is a no-op instead of a visible settle.
#
# FixStand's stock target is the G1 crouch (knee 0.3, ankle_pitch -0.2,
# shoulder_pitch 0.35, elbow 0.87). This policy's default_joint_pos is the
# shared stilt pose (knee 0.1, ankle 0, shoulder_pitch 0.2, elbow 0.6).
#
# Robot-side config only — the policy and deploy.yaml are untouched.
# Writes config.yaml.prepolicy as a backup; --revert puts it back.
set -euo pipefail

CFG=~/unitree_rl_mjlab/deploy/robots/g1/config/config.yaml
BAK="$CFG.prepolicy"

if [ "${1:-}" = "--revert" ]; then
    [ -f "$BAK" ] || { echo "no backup at $BAK"; exit 1; }
    cp "$BAK" "$CFG"; echo "reverted $CFG"; exit 0
fi

[ -f "$BAK" ] || cp "$CFG" "$BAK"

python3 - "$CFG" <<'PY'
import sys
p = sys.argv[1]
s = open(p).read()

edits = [
    ("0,0,0.3,-0.2,0",  "0,0,0.1,0,0",     2),  # both legs: knee + ankle_pitch
    ("0.35, 0.18,0,0.87", "0.2, 0.2,0,0.6", 1),  # left arm
    ("0.35,-0.18,0,0.87", "0.2,-0.2,0,0.6", 1),  # right arm
]
for old, new, want in edits:
    got = s.count(old)
    if got != want:
        print(f"ABORT: expected {want} occurrence(s) of {old!r}, found {got}.")
        print("The FixStand block is not in the shape this script knows.")
        print("Edit config.yaml by hand — copy default_joint_pos from deploy.yaml.")
        sys.exit(1)
    s = s.replace(old, new)

open(p, "w").write(s)
print("FixStand qs now:")
inside = False
for line in s.splitlines():
    if line.strip().startswith("qs:"): inside = True
    if inside:
        print("   ", line)
        if line.strip() == "]": break
PY

echo
echo "Backup at $BAK  (restore with: bash $0 --revert)"
echo "No rebuild needed — config.yaml is read at startup."
