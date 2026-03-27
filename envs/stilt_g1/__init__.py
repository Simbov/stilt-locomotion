"""Stilt G1 velocity task — registers with mjlab task registry."""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import stilt_g1_flat_env_cfg
from .rl_cfg import stilt_g1_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Stilt-G1",
  env_cfg=stilt_g1_flat_env_cfg(),
  play_env_cfg=stilt_g1_flat_env_cfg(play=True),
  rl_cfg=stilt_g1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
