"""Does this policy survive the hardware `base_lin_vel` stub?

On hardware there is no body-velocity sensor: the `unitree_hg` LowState carries
only IMU and motor states, so obs[0:15] is zero-filled (see deploy/README.md).
In sim that term is ground truth, and the policy uses it to notice and correct
its own drift. Zeroing it can turn a policy that tracks correctly into one that
accelerates without limit.

That is exactly what happened to Run 8 on the robot, 2026-08-31:

    uv run python scripts/check_base_lin_vel_stub.py 0.0          ->  x +0.04 m in 15 s
    uv run python scripts/check_base_lin_vel_stub.py 0.0 --stub   ->  x +1.05 m, monotonic
    uv run python scripts/check_base_lin_vel_stub.py 0.4          ->  vx_true +0.38  (correct)
    uv run python scripts/check_base_lin_vel_stub.py 0.4 --stub   ->  vx_true +2.16, still rising

RUN THIS AGAINST ANY NEW CHECKPOINT BEFORE TAKING IT TO THE ROBOT. If `--stub`
does not closely match the un-stubbed run, the policy is not deployable on this
hardware, however good its training curves look.

`vx_true` is the number to trust; world-frame x is not monotonic when the robot
curves as it runs away.
"""

import os
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

_orig = torch.load
torch.load = lambda *a, **k: _orig(*a, **{**k, "map_location": "cpu"})

from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.rl import RslRlVecEnvWrapper  # noqa: E402
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner  # noqa: E402

from envs.stilt_g1.env_cfgs import stilt_g1_flat_env_cfg  # noqa: E402
from envs.stilt_g1.rl_cfg import stilt_g1_ppo_runner_cfg  # noqa: E402

CMD_VX = float(sys.argv[1]) if len(sys.argv) > 1 else 0.4
STUB = "--stub" in sys.argv
CKPT = os.environ.get(
  "STILT_CKPT",
  "logs/rsl_rl/stilt_g1_velocity/2026-08-13_20-35-42_run8-stilts-on-off/model_5999.pt",
)

cfg = stilt_g1_flat_env_cfg(play=True)
cfg.scene.num_envs = 1
cfg.events["stilts_fitted"].params["fitted_probability"] = 0.0  # bare, like the robot

# Where does base_lin_vel live in the flattened actor vector, if at all?
# History is term-major (each term's frames contiguous, oldest first), so the
# term occupies the FIRST width*history entries when it is first in the group.
_actor_terms = list(cfg.observations["actor"].terms.keys())
if "base_lin_vel" not in _actor_terms:
  print(
    "base_lin_vel is not in the actor observation for this config, so there is\n"
    "nothing for the hardware stub to zero and this check does not apply.\n"
    "That is the intended state from Run 9 onward — the fix for the 2026-08-31\n"
    "hardware fault was to remove the term, not to compensate for it.\n"
    "Invariant is pinned by tests/test_env_wiring.py instead.\n"
    "To reproduce the original Run 8 fault, check out a commit from before the\n"
    "term was removed."
  )
  raise SystemExit(0)
if _actor_terms[0] != "base_lin_vel":
  raise SystemExit(
    f"base_lin_vel is not the first actor term (order: {_actor_terms}); the "
    "0:15 slice assumption no longer holds. Fix the slice before trusting this."
  )
STUB_SLICE = slice(0, 3 * cfg.observations["actor"].history_length)

raw = ManagerBasedRlEnv(cfg=cfg, device="cpu")
env = RslRlVecEnvWrapper(raw, clip_actions=None)
with tempfile.TemporaryDirectory() as tmp:
  runner = VelocityOnPolicyRunner(env, asdict(stilt_g1_ppo_runner_cfg()), tmp, "cpu")
  runner.load(CKPT)
  policy = runner.get_inference_policy(device="cpu")

cm = raw.command_manager
term = [n for n in cm.active_terms][0]
print(f"command term: {term}  CMD_VX={CMD_VX}  stub={STUB}")

robot = raw.scene["robot"]
start = None
print(f"{'step':>5} {'t(s)':>6} {'grav_x':>8} {'x(m)':>8} {'z(m)':>7} {'vx_true':>8}")
with torch.inference_mode():
  obs, _ = env.reset()
  if STUB:
    obs["actor"][:, STUB_SLICE] = 0.0
  for step in range(750):  # 15 s at 50 Hz
    cm.get_command(term)[:, 0] = CMD_VX
    cm.get_command(term)[:, 1:] = 0.0
    if STUB:
      obs["actor"][:, STUB_SLICE] = 0.0  # <-- the hardware base_lin_vel stub
    action = policy(obs)
    obs, _, _, _ = env.step(action)
    cm.get_command(term)[:, 0] = CMD_VX
    cm.get_command(term)[:, 1:] = 0.0
    pos = robot.data.root_link_pos_w[0]
    if start is None:
      start = pos.clone()
    if step % 50 == 0 or step == 749:
      g = float(obs["actor"][0, 42])
      vx = float(robot.data.root_link_lin_vel_b[0, 0])
      print(
        f"{step:5d} {step / 50:6.1f} {g:+8.3f} "
        f"{float(pos[0] - start[0]):+8.3f} {float(pos[2]):7.3f} {vx:+8.3f}"
      )
