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


def test_reset_restores_spawn_height_after_a_fall(env):
  env.reset()
  assert _pelvis_height(env) == pytest.approx(STILT_SPAWN_HEIGHT, abs=0.06)

  # Knock the robot over, confirm it actually fell, then reset again.
  dim = env.action_manager.total_action_dim
  with torch.inference_mode():
    for _ in range(120):
      env.step(torch.randn(env.num_envs, dim) * 2.0)
    fallen = _pelvis_height(env)
    env.reset()
    after = _pelvis_height(env)

  assert fallen < STILT_SPAWN_HEIGHT - 0.05, (
    f"robot did not fall, test is vacuous: {fallen:.3f} m"
  )
  assert after == pytest.approx(STILT_SPAWN_HEIGHT, abs=0.06), (
    f"reset left the robot at {after:.3f} m, expected ~{STILT_SPAWN_HEIGHT:.3f} m"
  )
