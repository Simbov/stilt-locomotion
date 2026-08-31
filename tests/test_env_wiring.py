def test_base_lin_vel_is_critic_only():
  """The actor must not see base_lin_vel — there is no sensor for it.

  The unitree_hg LowState carries only IMU and motor states, so the deploy
  runtime can only zero-fill this term. Run 8 was trained with it as ground
  truth and used it to correct its own drift; stubbed to zeros on hardware
  (2026-08-31) it was told it was stationary while it walked and behaved
  erratically. Reproduce with scripts/check_base_lin_vel_stub.py.

  It stays in the critic, which is training-only and may use privileged state.
  """
  from mjlab.tasks.registry import load_env_cfg

  import envs.stilt_g1  # noqa: F401

  for play in (False, True):
    cfg = load_env_cfg("Mjlab-Velocity-Flat-Stilt-G1", play=play)
    actor = cfg.observations["actor"].terms
    critic = cfg.observations["critic"].terms
    assert "base_lin_vel" not in actor, (
      f"base_lin_vel is back in the actor obs (play={play}); it is not "
      "measurable on the G1 and the deploy runtime can only zero it"
    )
    assert "base_lin_vel" in critic, (
      "base_lin_vel should stay in the critic — it is free privileged state"
    )


def test_actor_observation_width_matches_the_deployed_runtime():
  """6 terms x 5 frames = 480. Pinned because the C++ runtime is hand-configured.

  deploy.yaml is generated from ONNX metadata, but the buffer the runtime
  assembles has to agree with it exactly, and a silent width change produces a
  policy that loads and merely walks badly.
  """
  from mjlab.envs import ManagerBasedRlEnv

  from envs.stilt_g1.env_cfgs import stilt_g1_flat_env_cfg

  cfg = stilt_g1_flat_env_cfg(play=True)
  cfg.scene.num_envs = 1
  env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  try:
    obs, _ = env.reset()
    assert obs["actor"].shape[-1] == 480, (
      f"actor obs is {obs['actor'].shape[-1]} wide, expected 480 "
      "(6 terms x 5 frames). Update deploy/README.md's layout table too."
    )
  finally:
    env.close()
