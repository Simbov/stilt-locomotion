"""Gate check: randomised stilt heights must all spawn resting on the floor.

scene.reset() applies the keyframe before reset events run, so the spawn height
correction in envs/stilt_g1/events.py is what keeps this true. Rerun after any
change to the height DR wiring.

    uv run python scripts/check_height_dr.py
"""

import sys
from pathlib import Path

import torch

if str(Path(__file__).parent.parent) not in sys.path:
  sys.path.insert(0, str(Path(__file__).parent.parent))

from mjlab.envs import ManagerBasedRlEnv  # noqa: E402

from envs.stilt_g1.env_cfgs import stilt_g1_flat_env_cfg  # noqa: E402

NUM_ENVS = 64
OFFSET_RANGE = (-0.05, 0.05)


def main() -> None:
  cfg = stilt_g1_flat_env_cfg(play=True)
  cfg.scene.num_envs = NUM_ENVS
  cfg.events["stilt_height"].params["ranges"] = OFFSET_RANGE

  env = ManagerBasedRlEnv(cfg, device="cpu")
  env.reset()

  robot = env.scene["robot"]
  # Site names are namespaced by entity, e.g. "robot/left_stilt_tip".
  site_names = [s.name.split("/")[-1] for s in robot.indexing.sites]
  tips = torch.stack(
    [
      robot.data.site_pos_w[:, site_names.index(f"{side}_stilt_tip"), 2]
      - env.scene.env_origins[:, 2]
      for side in ("left", "right")
    ],
    dim=-1,
  )

  # Guard against a trivial pass: confirm the DR actually varied stilt length.
  body_ids = robot.indexing.body_ids[
    [i for i, b in enumerate(robot.indexing.bodies) if "post_inner" in b.name]
  ]
  sampled = env.sim.model.body_pos[:, body_ids, 2]
  spread = float(sampled.max() - sampled.min())

  print(f"offset range {OFFSET_RANGE} over {NUM_ENVS} envs")
  print(f"sampled offset  min {sampled.min():+.4f}  max {sampled.max():+.4f}  m")
  print(f"tip height      min {tips.min():+.4f}  max {tips.max():+.4f}  m")

  assert spread > 0.05, f"height DR did nothing — offset spread only {spread:.4f} m"
  left_right = float((sampled[:, 0] - sampled[:, 1]).abs().max())
  assert left_right < 1e-6, f"stilts got different lengths: {left_right:.4f} m"

  # reset_base adds a deliberate 0.01-0.05 m drop-in clearance on top.
  assert tips.min() > 0.0, f"envs spawning through the floor: {tips.min():+.4f}"
  assert tips.max() < 0.060, f"envs spawning in the air: {tips.max():+.4f}"
  print(f"PASS: all {NUM_ENVS} envs spawn resting on the floor")


if __name__ == "__main__":
  main()
