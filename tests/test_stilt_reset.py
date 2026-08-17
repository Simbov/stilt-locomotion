"""Reset must restore the spawn pose on EVERY episode, not just the first.

Regression test for a stale-read bug in reset_stilt_spawn_height: it read the
root pose from `asset.data.root_link_pos_w` (a cached view still holding the
previous episode's pose) and wrote it back, silently undoing `reset_base`.
Every episode after the first then spawned in the fallen pose and terminated
immediately, which read as a training-collapse rather than a reset bug.

The spawn height now depends on which morphology the episode drew, so these
assert per-mode rather than against one number.
"""

import pytest
import torch

from envs.stilt_g1.env_cfgs import stilt_g1_flat_env_cfg
from envs.stilt_g1.stilt_robot import (
  STILT_FITTED_SPAWN_HEIGHT,
  STILT_SPAWN_HEIGHT,
)

# reset_base adds a random 0.01-0.05 m drop-in clearance and reset_robot_joints
# perturbs the leg pose, so the spawn height is a band, not a point.
TOLERANCE_M = 0.08
FALLEN_HEIGHT = 0.40


@pytest.fixture(scope="module")
def env():
  from mjlab.envs import ManagerBasedRlEnv

  cfg = stilt_g1_flat_env_cfg()
  cfg.scene.num_envs = 32
  return ManagerBasedRlEnv(cfg, device="cpu")


def _pelvis_height(env) -> torch.Tensor:
  robot = env.scene["robot"]
  return robot.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]


def _expected(env) -> torch.Tensor:
  fitted = env.stilt_fitted
  return fitted * STILT_FITTED_SPAWN_HEIGHT + (1.0 - fitted) * STILT_SPAWN_HEIGHT


def _assert_at_spawn(env, when: str) -> None:
  error = (_pelvis_height(env) - _expected(env)).abs()
  assert float(error.max()) < TOLERANCE_M, (
    f"{when}: {int((error >= TOLERANCE_M).sum())} envs off their spawn height, "
    f"worst {float(error.max()):.3f} m"
  )


def test_both_morphologies_are_drawn(env):
  """A vacuous pass otherwise: every other test here would only cover one mode."""
  env.reset()
  fitted = env.stilt_fitted
  assert 0 < int(fitted.sum()) < env.num_envs, (
    f"draw was degenerate: {int(fitted.sum())}/{env.num_envs} fitted"
  )


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
  _assert_at_spawn(env, "first reset")

  # Deliberately NOT under torch.inference_mode(): writing sim state inside it
  # marks those buffers as inference tensors, and every later out-of-mode write
  # to them — including the next test's reset — then raises.
  robot = env.scene["robot"]
  with torch.no_grad():
    pose = torch.cat(
      [robot.data.root_link_pos_w.clone(), robot.data.root_link_quat_w.clone()],
      dim=-1,
    )
    pose[:, 2] = env.scene.env_origins[:, 2] + FALLEN_HEIGHT
    robot.write_root_link_pose_to_sim(pose)
    env.sim.forward()

  displaced = float(_pelvis_height(env).mean())
  env.reset()

  assert displaced == pytest.approx(FALLEN_HEIGHT, abs=0.05), (
    f"setup failed, robot was not displaced: {displaced:.3f} m"
  )
  _assert_at_spawn(env, "reset from a fallen pose")


def test_fitted_envs_spawn_higher_than_bare_ones(env):
  env.reset()
  height = _pelvis_height(env)
  fitted = env.stilt_fitted.bool()
  assert float(height[fitted].min()) > float(height[~fitted].max()), (
    "the two morphologies overlap in spawn height, which cannot be right — the "
    "stilts add 0.4 m"
  )


def test_removing_the_stilts_removes_their_mass(env):
  env.reset()
  robot = env.scene["robot"]
  names = [b.name.split("/")[-1] for b in robot.indexing.bodies]
  ids = [int(robot.indexing.body_ids[i]) for i, n in enumerate(names) if "_stilt_" in n]
  per_env = env.sim.model.body_mass[:, ids].sum(dim=1)
  fitted = env.stilt_fitted.bool()

  assert float(per_env[fitted].min()) > 1.0, "fitted stilts weigh nothing"
  assert float(per_env[~fitted].max()) < 0.05, "removed stilts still weigh something"


def test_the_brace_spring_exists_only_when_the_stilts_do(env):
  env.reset()
  robot = env.scene["robot"]
  names = list(robot.joint_names)
  ids = [int(robot.indexing.joint_ids[i]) for i, n in enumerate(names) if "ankle" in n]
  assert len(ids) == 4

  stiffness = env.sim.model.jnt_stiffness[:, ids]
  fitted = env.stilt_fitted.bool()
  assert float(stiffness[~fitted].abs().max()) == 0.0, (
    "the ankle is sprung with no stilt bolted to it"
  )
  assert float(stiffness[fitted].min()) > 0.0, "the brace applies no stiffness"


def test_both_contact_sets_can_actually_collide(env):
  """Geoms existing is not enough — they have to be collidable.

  `CollisionCfg.disable_other_geoms` defaults to True, so any geom left out of
  `geom_names_expr` has contype and conaffinity zeroed. Listing only the stilt
  capsules switched the robot's own feet off, and the bare-stilt envs fell
  silently through the floor while still logging as upright.
  """
  robot = env.scene["robot"]
  names = [g.name.split("/")[-1] for g in robot.indexing.geoms]
  model = env.sim.model

  for label, pattern in (("stilt", "_stilt_"), ("foot", "_foot")):
    ids = [
      int(robot.indexing.geom_ids[i])
      for i, n in enumerate(names)
      if pattern in n and n.endswith("_collision")
    ]
    assert ids, f"no {label} contact geoms found"
    assert int(model.geom_contype[ids].min()) > 0, (
      f"{label} capsules have contype 0 — they cannot collide with anything"
    )
    assert int(model.geom_conaffinity[ids].min()) > 0, (
      f"{label} capsules have conaffinity 0"
    )
    # Ground contact needs friction, so condim 3 rather than the frictionless 1.
    assert int(model.geom_condim[ids].min()) == 3, (
      f"{label} capsules are frictionless (condim 1) — the robot would skate"
    )


def test_removing_the_stilts_parks_the_VISUAL_meshes_too(env):
  """Not just the contact capsules — the meshes as well.

  The stilt visual geoms are UNNAMED in the MJCF, so a name-based filter skips
  all ten of them. Physically that is harmless (visual geoms do not collide),
  but the bare robot then renders wearing ghost stilts sunk through the floor,
  which makes every video and viewer session misleading. Select by parent body.
  """
  env.reset()
  robot = env.scene["robot"]
  ids, visual = [], 0
  for i, geom in enumerate(robot.indexing.geoms):
    parent = geom.parent.name.split("/")[-1] if geom.parent is not None else ""
    if "stilt" not in parent:
      continue
    ids.append(int(robot.indexing.geom_ids[i]))
    visual += int(not (geom.name or "").endswith("_collision"))

  assert visual == 10, f"expected 10 stilt visual meshes, found {visual}"
  z = env.sim.model.geom_pos[:, ids, 2]
  fitted = env.stilt_fitted.bool()
  assert float(z[fitted].max()) < 1.0, "fitted stilt geoms are not on the robot"
  assert float(z[~fitted].min()) > 1.0, (
    "removed stilt geoms are still on the robot — the bare robot will render "
    "with stilts attached"
  )


def test_the_tip_site_follows_whichever_sole_is_live(env):
  """foot_height, foot_clearance and foot_slip all read these sites."""
  env.reset()
  robot = env.scene["robot"]
  names = [s.name.split("/")[-1] for s in robot.indexing.sites]
  ids = [
    int(robot.indexing.site_ids[i]) for i, n in enumerate(names) if "stilt_tip" in n
  ]
  assert len(ids) == 2

  z = env.sim.model.site_pos[:, ids, 2]
  fitted = env.stilt_fitted.bool()
  # In the ankle frame: the stilt plate sole at -0.4425, the foot sole at -0.035.
  assert float(z[fitted].max()) == pytest.approx(-0.4425, abs=1e-3)
  assert float(z[~fitted].min()) == pytest.approx(-0.035, abs=1e-3)
