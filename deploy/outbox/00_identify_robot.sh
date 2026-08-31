#!/usr/bin/env bash
# Answers two questions in one run:
#   1. Is this the same physical robot we deployed to on 2026-05-09?
#   2. Is the one-time unitree_rl_mjlab setup already done on it?
#
# Read-only. Safe to run any number of times.

RL=~/unitree_rl_mjlab
KNOWN_ETH0="3c:6d:66:2b:d1:f0"
KNOWN_WLAN0="fc:23:cd:8f:70:79"

green() { printf '  \033[32m%s\033[0m %s\n' "OK" "$1"; }
red()   { printf '  \033[31m%s\033[0m %s\n' "--" "$1"; }

echo "=============================================="
echo " WHICH ROBOT IS THIS?"
echo "=============================================="
ETH0=$(cat /sys/class/net/eth0/address 2>/dev/null)
WLAN0=$(cat /sys/class/net/wlan0/address 2>/dev/null)
echo "  hostname   : $(hostname)"
echo "  eth0  MAC  : ${ETH0:-<none>}    (2026-05 robot: $KNOWN_ETH0)"
echo "  wlan0 MAC  : ${WLAN0:-<none>}    (2026-05 robot: $KNOWN_WLAN0)"

if [ "$ETH0" = "$KNOWN_ETH0" ]; then
    echo
    echo "  >> SAME ROBOT as May 2026. MAC matches exactly."
    SAME=1
else
    echo
    echo "  >> DIFFERENT ROBOT (or a swapped compute module)."
    echo "     Expect to redo the one-time setup below."
    SAME=0
fi

echo
echo "  Home-directory fingerprint of the May robot was: FALCON, QCR_G1,"
echo "  g1plus_pc4_unitree_install, inspire_hand_ws, ws_livox, xr_teleoperate."
echo "  This robot has:"
for d in FALCON QCR_G1 g1plus_pc4_unitree_install inspire_hand_ws ws_livox xr_teleoperate walking_deployment; do
    [ -e "$HOME/$d" ] && green "$d" || red "$d (absent)"
done

echo
echo "=============================================="
echo " IS THE ONE-TIME SETUP DONE?"
echo "=============================================="
MISSING=""

[ -x "$RL/deploy/robots/g1/build/g1_ctrl" ] \
    && green "g1_ctrl binary built" \
    || { red "g1_ctrl NOT built"; MISSING="$MISSING build"; }

grep -q "REGISTER_OBSERVATION(base_lin_vel)" "$RL/deploy/robots/g1/src/State_RLBase.cpp" 2>/dev/null \
    && green "base_lin_vel zero-fill patch (2.4)" \
    || { red "base_lin_vel patch MISSING -> run apply_base_lin_vel_stub.sh"; MISSING="$MISSING 2.4"; }

grep -q "compat shim" "$RL/deploy/include/unitree_joystick_dsl.hpp" 2>/dev/null \
    && green "KeyBase shim (2.5)" \
    || { red "KeyBase shim MISSING -> README step 5b"; MISSING="$MISSING 2.5"; }

grep -qE "^[[:space:]]+fmt$" "$RL/deploy/robots/g1/CMakeLists.txt" 2>/dev/null \
    && { red "fmt still in CMakeLists (2.6) -> README step 5c"; MISSING="$MISSING 2.6"; } \
    || green "fmt removed from CMakeLists (2.6)"

ls /usr/local/lib/libunitree_hg_idl_cpp.a >/dev/null 2>&1 \
    && green "unitree_sdk2 hg IDL library (2.3)" \
    || { red "libunitree_hg_idl_cpp.a MISSING -> README step 4"; MISSING="$MISSING 2.3"; }

ls "$RL/deploy/thirdparty" 2>/dev/null | grep -qi onnxruntime \
    && green "onnxruntime in thirdparty (2.2)" \
    || { red "onnxruntime NOT in thirdparty -> README step 3"; MISSING="$MISSING 2.2"; }

echo
echo "  existing policies:"
ls "$RL/deploy/robots/g1/config/policy/velocity/" 2>/dev/null | sed 's/^/    /' || echo "    <none>"
echo "  Velocity policy_dir currently:"
grep -A4 "^  Velocity:" "$RL/deploy/robots/g1/config/config.yaml" 2>/dev/null | grep policy_dir | sed 's/^/    /'

echo
echo "=============================================="
if [ -z "$MISSING" ]; then
    echo " VERDICT: setup complete. Skip Part 2."
    echo "          Next: bash ~/run8/01_install_policy.sh"
else
    echo " VERDICT: missing ->$MISSING"
    echo "          Do those Part 2 steps first (deploy/README.md), then"
    echo "          bash ~/run8/01_install_policy.sh"
fi
[ "$SAME" = "0" ] && echo " NOTE:    this is NOT the May robot — re-check the IP and the feet too."
echo "=============================================="
