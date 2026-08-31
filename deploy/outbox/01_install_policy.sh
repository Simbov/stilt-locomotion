#!/usr/bin/env bash
# Installs the Run 8 policy as `stilt_run8` and points the FSM at it.
# Idempotent: safe to re-run. Verifies checksums before touching anything.
set -euo pipefail

BUNDLE="$(cd "$(dirname "$0")" && pwd)"
RL=~/unitree_rl_mjlab
DEST="$RL/deploy/robots/g1/config/policy/velocity/stilt_run8"
CFG="$RL/deploy/robots/g1/config/config.yaml"

echo "== 1/4  checksums of the shipped files"
cd "$BUNDLE"
if ! md5sum -c MANIFEST.txt --ignore-missing; then
    echo "ABORT: a shipped file does not match the manifest. Re-scp it." >&2
    exit 1
fi

echo "== 2/4  installing into $DEST"
mkdir -p "$DEST/params" "$DEST/exported"
cp "$BUNDLE/policy.onnx" "$DEST/exported/policy.onnx"
cp "$BUNDLE/deploy.yaml" "$DEST/params/deploy.yaml"
md5sum "$DEST/exported/policy.onnx" "$DEST/params/deploy.yaml"

echo "== 3/4  sanity-checking deploy.yaml against the runtime's expectations"
python3 - "$DEST/params/deploy.yaml" <<'PY'
import sys
try:
    import yaml
except ImportError:
    print("  SKIP: pyyaml not available on this python — checked on the laptop instead")
    sys.exit(0)
d = yaml.safe_load(open(sys.argv[1]))
fail = []
if "base_velocity" not in d.get("commands", {}):
    fail.append("commands: must be keyed base_velocity (velocity_commands hardcodes it)")
obs = d["observations"]
if "use_gym_history" in obs:
    fail.append("use_gym_history must be absent — it flips the history to frame-major")
for name, t in obs.items():
    if "params" not in t:
        fail.append(f"{name}: no params key — the block would parse as GROUPS and throw")
dim = sum(len(t["scale"]) * t["history_length"] for t in obs.values())
if dim != 495:
    fail.append(f"observation dims come to {dim}, policy takes 495")
for f in fail:
    print("  FAIL:", f)
if fail:
    sys.exit(1)
print(f"  ok: {len(obs)} terms, {dim} dims, commands keyed base_velocity")
PY

echo "== 4/4  pointing the Velocity FSM at stilt_run8"
python3 - "$CFG" <<'PY'
import re, sys
p = sys.argv[1]
s = open(p).read()
new, n = re.subn(r'policy_dir: config/policy/velocity(/\S+)?',
                 'policy_dir: config/policy/velocity/stilt_run8', s, count=1)
if n == 0:
    print("  FAIL: no Velocity policy_dir line found in config.yaml"); sys.exit(1)
open(p, 'w').write(new)
for l in new.splitlines():
    if 'policy_dir' in l:
        print("  ", l.strip())
PY

echo
echo "DONE. Next:"
echo "  cd $RL/deploy/robots/g1/build && cmake .. && make -j\$(nproc)"
echo "  then: ./g1_ctrl -n eth0 2>&1 | tee ~/run8_\$(date +%s).log"
echo "  Verify the startup banner says stilt_run8 BEFORE touching the joystick."
