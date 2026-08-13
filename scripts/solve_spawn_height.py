"""Print the pelvis spawn height for BOTH morphologies.

The robot is the stock 29-DoF G1 and the stilts bolt on and come off. Both
morphologies stand in the SAME pose — see ``STILT_LEG_POSE`` for why — but at
different heights, because one rests on the stilt plates and the other on the
robot's own foot capsules.

Rerun whenever the pose or the stilt geometry changes, then paste the values
into ``envs/stilt_g1/stilt_robot.py``.

    uv run python scripts/solve_spawn_height.py
"""

import sys
from pathlib import Path

import mujoco

if str(Path(__file__).parent.parent) not in sys.path:
  sys.path.insert(0, str(Path(__file__).parent.parent))

from envs.stilt_g1.stilt_robot import STILT_LEG_POSE  # noqa: E402

XML = Path(__file__).parent.parent / "assets" / "mjcf" / "g1" / "g1.xml"

PROBE_HEIGHT = 1.5


def _solve(model, pose: dict[str, float], geom_substring: str) -> float:
  data = mujoco.MjData(model)
  for side in ("left", "right"):
    for joint, value in pose.items():
      jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_{joint}_joint")
      data.qpos[model.jnt_qposadr[jid]] = value
  data.qpos[2] = PROBE_HEIGHT
  mujoco.mj_forward(model, data)

  lowest = min(
    data.geom_xpos[gid][2] - model.geom_size[gid][0]
    for gid in range(model.ngeom)
    if geom_substring in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or "")
    and "collision" in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or "")
  )
  return PROBE_HEIGHT - lowest


def main() -> None:
  model = mujoco.MjSpec.from_file(str(XML)).compile()

  off = _solve(model, STILT_LEG_POSE, "_foot")
  on = _solve(model, STILT_LEG_POSE, "_stilt_")

  print(f"pose (both morphologies) {STILT_LEG_POSE}")
  print(f"  STILT_SPAWN_HEIGHT        = {off:.4f}   (on the robot's own feet)")
  print(f"  STILT_FITTED_SPAWN_HEIGHT = {on:.4f}   (on the stilt plates)")
  print(f"  STILT_SPAWN_RISE          = {on - off:.4f}")


if __name__ == "__main__":
  main()
