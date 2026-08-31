#!/usr/bin/env bash
# Rewrites State_RLBase.cpp to add the base_lin_vel zero-fill registration.
#
# In unitree_rl_mjlab, per-robot observation registrations live in
#   deploy/robots/<robot>/src/State_RLBase.cpp
# NOT in a shared observation_manager.cpp (that file does not exist in this
# version of the repo).
#
# Usage (run from the unitree_rl_mjlab root, on the robot):
#   bash <path-to-this-repo>/deploy/patches/apply_base_lin_vel_stub.sh
#
# Safe to run multiple times: exits without change if stub already present.

set -euo pipefail

TARGET="deploy/robots/g1/src/State_RLBase.cpp"

if [ ! -f "$TARGET" ]; then
    echo "ERROR: $TARGET not found. Run this from the unitree_rl_mjlab root." >&2
    exit 1
fi

if grep -q "base_lin_vel" "$TARGET"; then
    echo "base_lin_vel already registered in $TARGET — nothing to do."
    exit 0
fi

# The registration must go inside the `namespace isaaclab { ... }` block.
# We insert it immediately after the keyboard_velocity_commands block.
# If that block is absent, fall back to inserting before the closing brace.

if ! grep -q "REGISTER_OBSERVATION(keyboard_velocity_commands)" "$TARGET"; then
    echo "ERROR: expected anchor (keyboard_velocity_commands) not found in $TARGET." >&2
    echo "Inspect the file and add the base_lin_vel block manually inside namespace isaaclab." >&2
    exit 1
fi

STUB='
REGISTER_OBSERVATION(base_lin_vel)
{
    // No direct body-velocity sensor on hardware — zero-fill.
    // Policy uses this as measured-speed feedback, not commanded speed.
    // Replace with a KF/EKF velocity estimate when available.
    return std::vector<float>{0.f, 0.f, 0.f};
}
'

# Find the closing brace of the keyboard_velocity_commands block and insert after it.
# The block ends with a lone `}` line after the `return cmd;` line.
ANCHOR_LINE=$(awk '/REGISTER_OBSERVATION\(keyboard_velocity_commands\)/{found=1} found && /^}$/{print NR; exit}' "$TARGET")

if [ -z "$ANCHOR_LINE" ]; then
    echo "ERROR: could not find closing brace of keyboard_velocity_commands block." >&2
    exit 1
fi

# Insert STUB after ANCHOR_LINE using sed.
sed -i "${ANCHOR_LINE}r /dev/stdin" "$TARGET" <<< "$STUB"

echo "Patched $TARGET (inserted base_lin_vel stub after line $ANCHOR_LINE)."
echo "Rebuild with: cd deploy/robots/g1/build && make -j\$(nproc)"
