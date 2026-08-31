#!/usr/bin/env bash
# One command to put the whole Run 8 bring-up bundle on the robot.
#
#   ./scripts/ship_to_robot.sh                      # default robot
#   ./scripts/ship_to_robot.sh unitree@10.0.0.5     # somewhere else
#
# Lands in ~/run8/ on the robot. Then, over ssh:
#   bash ~/run8/00_identify_robot.sh     <- same robot? setup done?
#   bash ~/run8/01_install_policy.sh     <- install + point the FSM at it
set -euo pipefail

ROBOT="${1:-unitree@192.168.123.164}"
RUN="2026-08-13_20-35-42_run8-stilts-on-off"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ONNX="$ROOT/logs/rsl_rl/stilt_g1_velocity/$RUN/$RUN.onnx"
YAML="$ROOT/deploy/config/g1_stilt/deploy.yaml"
OUT="$ROOT/deploy/outbox"

[ -f "$ONNX" ] || { echo "missing $ONNX" >&2; exit 1; }
[ -f "$YAML" ] || { echo "missing $YAML" >&2; exit 1; }

echo "== verifying the local files against the manifest"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
cp "$ONNX" "$STAGE/policy.onnx"
cp "$YAML" "$STAGE/deploy.yaml"
cp "$OUT"/*.sh "$OUT/MANIFEST.txt" "$STAGE/"
EXPECT_ONNX=$(awk '$2=="policy.onnx"{print $1}' "$OUT/MANIFEST.txt")
EXPECT_YAML=$(awk '$2=="deploy.yaml"{print $1}' "$OUT/MANIFEST.txt")
GOT_ONNX=$(md5 -q "$ONNX"); GOT_YAML=$(md5 -q "$YAML")
[ "$GOT_ONNX" = "$EXPECT_ONNX" ] || { echo "ONNX md5 $GOT_ONNX != manifest $EXPECT_ONNX" >&2; exit 1; }
[ "$GOT_YAML" = "$EXPECT_YAML" ] || {
    echo "deploy.yaml md5 $GOT_YAML != manifest $EXPECT_YAML" >&2
    echo "If you regenerated it deliberately, update deploy/outbox/MANIFEST.txt." >&2
    exit 1; }
echo "   policy.onnx  $GOT_ONNX"
echo "   deploy.yaml  $GOT_YAML"
echo "   both match the manifest."

echo "== shipping to $ROBOT:~/run8/  (password: 123)"
ssh -o ConnectTimeout=10 "$ROBOT" 'mkdir -p ~/run8'
scp -o ConnectTimeout=10 "$STAGE"/* "$ROBOT:~/run8/"

echo
echo "Done. On the robot, in order:"
echo "  ssh $ROBOT"
echo "  bash ~/run8/00_identify_robot.sh"
echo "  bash ~/run8/01_install_policy.sh"
echo "  cd ~/unitree_rl_mjlab/deploy/robots/g1/build && cmake .. && make -j\$(nproc)"
