"""Print the pelvis height that rests both stilt plates exactly on the floor.

Rerun whenever STILT_KNEE_ANGLE or the stilt geometry changes, then paste the
value into STILT_SPAWN_HEIGHT in envs/stilt_g1/stilt_robot.py.

    uv run python scripts/solve_spawn_height.py
"""

import sys
from pathlib import Path

import mujoco

if str(Path(__file__).parent.parent) not in sys.path:
  sys.path.insert(0, str(Path(__file__).parent.parent))

from envs.stilt_g1.stilt_robot import STILT_KNEE_ANGLE  # noqa: E402

XML = Path(__file__).parent.parent / "assets" / "mjcf" / "g1" / "g1.xml"

PROBE_HEIGHT = 1.0


def main() -> None:
  model = mujoco.MjSpec.from_file(str(XML)).compile()
  data = mujoco.MjData(model)

  for side in ("left", "right"):
    for joint, value in (
      ("hip_pitch", -STILT_KNEE_ANGLE),
      ("knee", STILT_KNEE_ANGLE),
    ):
      jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_{joint}_joint")
      data.qpos[model.jnt_qposadr[jid]] = value

  data.qpos[2] = PROBE_HEIGHT
  mujoco.mj_forward(model, data)

  lowest = min(
    data.geom_xpos[gid][2] - model.geom_size[gid][0]
    for gid in range(model.ngeom)
    if "stilt" in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or "")
  )
  print(f"knee angle           = {STILT_KNEE_ANGLE}")
  print(f"lowest stilt contact = {lowest:+.4f} m at pelvis {PROBE_HEIGHT} m")
  print(f"STILT_SPAWN_HEIGHT   = {PROBE_HEIGHT - lowest:.4f}")


if __name__ == "__main__":
  main()
