"""Gate check: every env spawns resting on the floor, stilts on OR off.

`scene.reset()` applies the keyframe before reset events run, so the keyframe
knows neither which morphology was drawn nor how long the stilts are. Everything
that makes the spawn correct lives in `envs/stilt_g1/events.py`. Rerun after any
change to that wiring.

    uv run python scripts/check_height_dr.py
"""

import sys
from pathlib import Path

import torch

if str(Path(__file__).parent.parent) not in sys.path:
  sys.path.insert(0, str(Path(__file__).parent.parent))

from mjlab.envs import ManagerBasedRlEnv  # noqa: E402

from envs.stilt_g1.env_cfgs import stilt_g1_flat_env_cfg  # noqa: E402

NUM_ENVS = 128
OFFSET_RANGE = (-0.05, 0.05)
# reset_robot_joints perturbs the leg pose and reset_base adds a drop-in
# clearance, so the contact surface is never exactly on the floor. These bound
# how far either way is acceptable.
MAX_PENETRATION_M = 0.04
MAX_AIRBORNE_M = 0.10


def _lowest_contact(env, prefix: str) -> torch.Tensor:
  """Lowest point of any live contact capsule, per env, relative to terrain."""
  robot = env.scene["robot"]
  names = [g.name.split("/")[-1] for g in robot.indexing.geoms]
  ids = [
    int(robot.indexing.geom_ids[i])
    for i, n in enumerate(names)
    if n.startswith(("left_" + prefix, "right_" + prefix)) and n.endswith("_collision")
  ]
  assert ids, f"no contact geoms matched {prefix!r}"
  radius = env.sim.model.geom_size[:, ids, 0]
  bottom = env.sim.data.geom_xpos[:, ids, 2] - radius
  return bottom.min(dim=1).values - env.scene.env_origins[:, 2]


def main() -> None:
  cfg = stilt_g1_flat_env_cfg(play=True)
  cfg.scene.num_envs = NUM_ENVS
  cfg.events["stilt_height"].params["ranges"] = OFFSET_RANGE

  env = ManagerBasedRlEnv(cfg, device="cpu")
  env.reset()

  robot = env.scene["robot"]
  fitted = env.stilt_fitted.bool()
  assert fitted.any() and (~fitted).any(), (
    f"draw was degenerate: {int(fitted.sum())}/{NUM_ENVS} fitted"
  )

  # Guard against a trivial pass: confirm the height DR actually varied length.
  body_ids = robot.indexing.body_ids[
    [i for i, b in enumerate(robot.indexing.bodies) if "post_inner" in b.name]
  ]
  sampled = env.sim.model.body_pos[:, body_ids, 2]
  spread = float(sampled.max() - sampled.min())
  assert spread > 0.05, f"height DR did nothing — offset spread only {spread:.4f} m"
  left_right = float((sampled[:, 0] - sampled[:, 1]).abs().max())
  assert left_right < 1e-6, f"stilts got different lengths: {left_right:.4f} m"

  # Site names are namespaced by entity, e.g. "robot/left_stilt_tip".
  site_names = [s.name.split("/")[-1] for s in robot.indexing.sites]
  tips = torch.stack(
    [
      robot.data.site_pos_w[:, site_names.index(f"{side}_stilt_tip"), 2]
      - env.scene.env_origins[:, 2]
      for side in ("left", "right")
    ],
    dim=-1,
  ).min(dim=1)[0]

  print(f"offset range {OFFSET_RANGE} over {NUM_ENVS} envs")
  print(f"  fitted        {int(fitted.sum())} / {NUM_ENVS}")
  print(f"  telescope     min {sampled.min():+.4f}  max {sampled.max():+.4f}  m")

  for label, mask, prefix in (
    ("stilts ON ", fitted, "stilt_"),
    ("stilts OFF", ~fitted, "foot"),
  ):
    contact = _lowest_contact(env, prefix)[mask]
    tip = tips[mask]
    print(
      f"  {label}   contact min {contact.min():+.4f}  max {contact.max():+.4f}"
      f"   |  tip site min {tip.min():+.4f}  max {tip.max():+.4f}  m"
    )
    assert contact.min() > -MAX_PENETRATION_M, (
      f"{label}: spawning through the floor by {-contact.min():.4f} m"
    )
    assert contact.max() < MAX_AIRBORNE_M, (
      f"{label}: spawning {contact.max():.4f} m in the air"
    )
    # The tip site must track whichever surface is live, or every height-based
    # reward reads the wrong body in this morphology.
    assert abs(float(tip.mean() - contact[: len(tip)].mean())) < 0.05, (
      f"{label}: tip site is not on the live contact surface"
    )

  print(f"PASS: all {NUM_ENVS} envs spawn resting on the floor, both morphologies")


if __name__ == "__main__":
  main()
