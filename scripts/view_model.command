#!/bin/bash
# Double-click in Finder to open a MuJoCo model viewer.
# Defaults to the local G1 copy — pick a different XML via the file dialog.

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DEFAULT_MODEL="$SCRIPT_DIR/assets/mjcf/g1/g1.xml"

# ── Optionally pick a different XML ──────────────────────────────────────────
CHOICE=$(osascript <<APPLESCRIPT
try
    set defaultFile to POSIX file "$DEFAULT_MODEL" as alias
    button returned of (display dialog "Open which model?" ¬
        buttons {"Pick file…", "Open G1"} ¬
        default button "Open G1" ¬
        with title "MuJoCo Model Viewer")
on error
    return "cancel"
end try
APPLESCRIPT
)

if [ "$CHOICE" = "cancel" ] || [ -z "$CHOICE" ]; then
    exit 0
elif [ "$CHOICE" = "Pick file…" ]; then
    MODEL=$(osascript <<APPLESCRIPT
try
    set theFile to choose file ¬
        with prompt "Select a MuJoCo XML model:" ¬
        default location (POSIX file "$SCRIPT_DIR/assets/mjcf" as alias)
    return POSIX path of theFile
on error
    return ""
end try
APPLESCRIPT
    )
    if [ -z "$MODEL" ]; then exit 0; fi
else
    MODEL="$DEFAULT_MODEL"
fi

echo "Opening: $MODEL"

source "$SCRIPT_DIR/.venv/bin/activate"

python3 - "$MODEL" <<'PYTHON'
import sys
import mujoco
import mujoco.viewer

model_path = sys.argv[1]
m = mujoco.MjModel.from_xml_path(model_path)
m.opt.gravity[:] = 0   # disable gravity so robot holds its pose
d = mujoco.MjData(m)
mujoco.viewer.launch(m, d)
PYTHON
