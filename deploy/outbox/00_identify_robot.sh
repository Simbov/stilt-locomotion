#!/usr/bin/env bash
# Which robot is this, and is it ready to run a policy?
# Read-only. Safe to run any number of times.

RL=~/unitree_rl_mjlab
SDK=~/unitree_sdk2

# Known robots — add a row when you meet a new one, and update the table in
# deploy/BRINGUP_CHECKLIST.md to match.
known_robot() {
    case "$1" in
        "3c:6d:66:2b:d1:f0") echo "seen 2026-05-09 (first G1)" ;;
        "3c:6d:66:a3:e5:73") echo "seen 2026-08-31 (QCR lab G1)" ;;
        "3c:6d:66:a3:e0:76") echo "seen 2026-09-17 (third G1; SDK copied to ~/unitree_sdk2)" ;;
        *) return 1 ;;
    esac
}

ok()  { printf '  \033[32mOK\033[0m %s\n' "$1"; }
bad() { printf '  \033[31m--\033[0m %s\n' "$1"; }

echo "=============================================="
echo " WHICH ROBOT IS THIS?"
echo "=============================================="
ETH0=$(cat /sys/class/net/eth0/address 2>/dev/null)
echo "  hostname  : $(hostname)   (always 'ubuntu' — not an identifier)"
echo "  eth0 MAC  : ${ETH0:-<none>}"
if WHEN=$(known_robot "$ETH0"); then
    echo "  >> KNOWN ROBOT — $WHEN"
else
    echo "  >> UNKNOWN ROBOT. Add its MAC to this script and to the table in"
    echo "     deploy/BRINGUP_CHECKLIST.md once you are done."
fi

echo
echo "=============================================="
echo " IS IT READY?"
echo "=============================================="
MISSING=""
CM="$RL/deploy/robots/g1/CMakeLists.txt"

[ -x "$RL/deploy/robots/g1/build/g1_ctrl" ] \
    && ok "g1_ctrl is built" \
    || { bad "g1_ctrl NOT built"; MISSING="$MISSING build"; }

# The runtime is patched on the laptop by scripts/prepare_runtime.py, so these
# three either all landed together or the tree was never shipped.
if [ -f "$CM" ]; then
    grep -q "UNITREE_SDK_ROOT" "$CM" \
        && ok "build points at ~/unitree_sdk2 (no /usr/local write)" \
        || { bad "CMakeLists NOT patched — ship with --runtime"; MISSING="$MISSING runtime"; }
    grep -qE "^[[:space:]]+fmt$" "$CM" \
        && { bad "unused fmt still in CMakeLists"; MISSING="$MISSING runtime"; } \
        || ok "fmt removed"
else
    bad "no runtime at $RL — ship with --runtime"; MISSING="$MISSING runtime"
fi

grep -q "compat shim" "$RL/deploy/include/unitree_joystick_dsl.hpp" 2>/dev/null \
    && ok "KeyBase shim" \
    || { bad "KeyBase shim missing — ship with --runtime"; MISSING="$MISSING runtime"; }

# The one real prerequisite we cannot ship: the home-directory SDK must carry
# the G1's unitree_hg DDS types. Do NOT fix this by installing over /usr/local.
[ -d "$SDK/include/unitree/idl/hg" ] \
    && ok "~/unitree_sdk2 has the unitree_hg IDL" \
    || { bad "NO unitree_hg IDL in $SDK — see deploy/README.md before installing anything"
         MISSING="$MISSING sdk"; }

ls "$RL/deploy/thirdparty" 2>/dev/null | grep -qi onnxruntime \
    && ok "onnxruntime present" \
    || { bad "onnxruntime missing — ship with --runtime"; MISSING="$MISSING runtime"; }

echo
echo "  installed policies:"
ls "$RL/deploy/robots/g1/config/policy/velocity/" 2>/dev/null | sed 's/^/    /' \
    || echo "    <none>"
echo "  Velocity FSM points at:"
grep -n "policy_dir" "$RL/deploy/robots/g1/config/config.yaml" 2>/dev/null \
    | head -1 | sed 's/^/    /' || echo "    <no config.yaml>"

echo
echo "=============================================="
if [ -z "$MISSING" ]; then
    echo " VERDICT: ready.  ->  bash ~/run8/01_install_policy.sh"
else
    case "$MISSING" in
        *sdk*) echo " VERDICT: this robot lacks the unitree_hg SDK. STOP and read"
               echo "          deploy/README.md — do not install one over /usr/local." ;;
        *)     echo " VERDICT: missing ->$MISSING"
               echo "          From the laptop:  ./scripts/ship_to_robot.sh --runtime"
               echo "          then:             bash ~/run8/01_install_policy.sh" ;;
    esac
fi
echo "=============================================="
