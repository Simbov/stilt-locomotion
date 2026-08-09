"""Reset must restore the spawn pose on EVERY episode, not just the first.

Regression test for a stale-read bug in reset_stilt_spawn_height: it read the
root pose from `asset.data.root_link_pos_w` (a cached view still holding the
previous episode's pose) and wrote it back, silently undoing `reset_base`.
Every episode after the first then spawned in the fallen pose and terminated
immediately, which read as a training-collapse rather than a reset bug.
"""

import pytest
import torch

from envs.stilt_g1.env_cfgs import stilt_g1_flat_env_cfg
from envs.stilt_g1.stilt_robot import STILT_SPAWN_HEIGHT


@pytest.fixture(scope="module")
def env():
  from mjlab.envs import ManagerBasedRlEnv

  cfg = stilt_g1_flat_env_cfg()
  cfg.scene.num_envs = 8
  return ManagerBasedRlEnv(cfg, device="cpu")


def _pelvis_height(env):
  robot = env.scene["robot"]
  return (robot.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]).mean().item()


FALLEN_HEIGHT = 0.40


def test_reset_restores_spawn_height_from_an_arbitrary_pose(env):
  """Reset must recover the spawn pose from wherever the robot happened to be.

  The robot is *placed* in a fallen pose rather than knocked over with random
  actions. An earlier version did the latter and flaked: how far it falls in a
  fixed number of steps depends on the RNG state left by whichever tests ran
  first, and it sometimes stayed upright, making the test vacuous. Placing the
  pose directly tests the same invariant deterministically — the bug was that
  reset read a *cached* pose, so any non-spawn starting pose reproduces it.
  """
  env.reset()
  assert _pelvis_height(env) == pytest.approx(STILT_SPAWN_HEIGHT, abs=0.06)

  robot = env.scene["robot"]
  with torch.inference_mode():
    pose = torch.cat(
      [robot.data.root_link_pos_w.clone(), robot.data.root_link_quat_w.clone()],
      dim=-1,
    )
    pose[:, 2] = env.scene.env_origins[:, 2] + FALLEN_HEIGHT
    robot.write_root_link_pose_to_sim(pose)
    env.sim.forward()

    displaced = _pelvis_height(env)
    env.reset()
    after = _pelvis_height(env)

  assert displaced == pytest.approx(FALLEN_HEIGHT, abs=0.05), (
    f"setup failed, robot was not displaced: {displaced:.3f} m"
  )
  assert after == pytest.approx(STILT_SPAWN_HEIGHT, abs=0.06), (
    f"reset left the robot at {after:.3f} m, expected ~{STILT_SPAWN_HEIGHT:.3f} m"
  )
