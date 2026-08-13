"""Drive a checkpoint with the stilts ON and then OFF, and report both.

Run 8 trains ONE policy for two morphologies. The training curves cannot settle
whether it learned both or averaged them, because every aggregate is a mixture.
This pins the morphology, sweeps fixed commands through it, and reports what the
robot actually achieved in each.

    uv run python scripts/eval_both_modes.py \
        --checkpoint logs/rsl_rl/stilt_g1_velocity/<run>/model_5999.pt

Achieved velocity is measured in the body frame, the same frame the command is
expressed in, after discarding a settling window.
"""

from __future__ import annotations

import argparse
import math
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np

if str(Path(__file__).parent.parent) not in sys.path:
  sys.path.insert(0, str(Path(__file__).parent.parent))

# (vx, vy, yaw) commands to hold, in m/s and rad/s.
COMMANDS: list[tuple[float, float, float]] = [
  (0.0, 0.0, 0.0),
  (0.2, 0.0, 0.0),
  (0.4, 0.0, 0.0),
  (0.6, 0.0, 0.0),
  (0.8, 0.0, 0.0),
  (-0.4, 0.0, 0.0),
  (0.0, 0.4, 0.0),
  (0.0, 0.0, 0.6),
]
SETTLE_STEPS = 100
SAMPLE_STEPS = 250


def evaluate(
  checkpoint: Path,
  fitted: bool,
  stilt_mass_kg: float,
  device: str,
  commands: list[tuple[float, float, float]] | None = None,
  episodes: int = 1,
):
  import torch

  original_load = torch.load
  torch.load = lambda *a, **k: original_load(*a, **{**k, "map_location": device})

  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

  from envs.stilt_g1.env_cfgs import stilt_g1_flat_env_cfg
  from envs.stilt_g1.rl_cfg import stilt_g1_ppo_runner_cfg

  cfg = stilt_g1_flat_env_cfg(play=True)
  cfg.scene.num_envs = 1
  cfg.events["stilts_fitted"].params["fitted_probability"] = 1.0 if fitted else 0.0
  alpha = 0.5 * math.log(stilt_mass_kg / 2.8)
  cfg.events["stilt_mass"].params["alpha_range"] = (alpha, alpha)

  raw = ManagerBasedRlEnv(cfg=cfg, device=device)
  env = RslRlVecEnvWrapper(raw, clip_actions=None)
  with tempfile.TemporaryDirectory() as tmp:
    runner = VelocityOnPolicyRunner(env, asdict(stilt_g1_ppo_runner_cfg()), tmp, device)
    runner.load(str(checkpoint))
    policy = runner.get_inference_policy(device=device)

  robot = raw.scene["robot"]
  command = raw.command_manager.get_term("twist")

  rows = []
  with torch.inference_mode():
    for cmd in commands or COMMANDS:
      target = torch.tensor(cmd, device=device)
      # Repeat the command over several fresh episodes. One episode is not
      # enough: reset randomises the initial pose, the terrain and (when fitted)
      # the brace stiffness, and yaw tracking in particular varies a lot between
      # episodes — a single sample can read as "turns the wrong way".
      per_episode = []
      falls = 0
      for _ in range(episodes):
        obs, _ = env.reset()
        assert bool(raw.stilt_fitted[0] > 0.5) == fitted, (
          "morphology draw was not pinned"
        )
        terminated = False
        for _ in range(SETTLE_STEPS):
          command.command[:] = target
          obs, _, dones, _ = env.step(policy(obs))
          terminated |= bool(dones[0])

        vx, vy, yaw, height = [], [], [], []
        for _ in range(SAMPLE_STEPS):
          command.command[:] = target
          obs, _, dones, _ = env.step(policy(obs))
          terminated |= bool(dones[0])
          lin = robot.data.root_link_lin_vel_b[0]
          ang = robot.data.root_link_ang_vel_b[0]
          vx.append(float(lin[0]))
          vy.append(float(lin[1]))
          yaw.append(float(ang[2]))
          height.append(
            float(robot.data.root_link_pos_w[0, 2] - raw.scene.env_origins[0, 2])
          )
        falls += int(terminated)
        per_episode.append((np.mean(vx), np.mean(vy), np.mean(yaw), np.mean(height)))

      arr = np.array(per_episode)
      rows.append(
        {
          "cmd": cmd,
          "vx": float(arr[:, 0].mean()),
          "vy": float(arr[:, 1].mean()),
          "yaw": float(arr[:, 2].mean()),
          "yaw_sd": float(arr[:, 2].std()),
          "height": float(arr[:, 3].mean()),
          "falls": falls,
          "episodes": episodes,
        }
      )
  return rows


def report(label: str, rows: list[dict]) -> None:
  print(f"\n=== {label} ===")
  print(
    f"  {'command (vx,vy,yaw)':>22}  {'vx':>7} {'vy':>7} {'yaw':>7} {'yaw sd':>7}"
    f"  {'pelvis':>7}  {'falls':>6}"
  )
  for r in rows:
    c = f"({r['cmd'][0]:+.1f},{r['cmd'][1]:+.1f},{r['cmd'][2]:+.1f})"
    print(
      f"  {c:>22}  {r['vx']:+7.3f} {r['vy']:+7.3f} {r['yaw']:+7.3f} {r['yaw_sd']:7.3f}"
      f"  {r['height']:7.3f}  {str(r['falls']) + '/' + str(r['episodes']):>6}"
    )

  # Tracking error against the commanded component that is non-zero.
  errs = []
  for r in rows:
    cx, cy, cyaw = r["cmd"]
    errs.append(math.hypot(cx - r["vx"], cy - r["vy"]))
  print(f"  mean |planar velocity error| = {np.mean(errs):.3f} m/s")
  print(
    f"  falls = {sum(r['falls'] for r in rows)}"
    f" / {sum(r['episodes'] for r in rows)} episodes"
  )


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--stilt-mass", type=float, default=2.8)
  parser.add_argument("--device", default="cpu")
  parser.add_argument(
    "--episodes",
    type=int,
    default=5,
    help="Fresh episodes averaged per command. Yaw needs several.",
  )
  parser.add_argument(
    "--commands",
    help="Semicolon-separated vx,vy,yaw triples to sweep instead of the default set.",
  )
  args = parser.parse_args()

  commands = None
  if args.commands:
    commands = [
      tuple(float(v) for v in triple.split(",")) for triple in args.commands.split(";")
    ]

  on = evaluate(
    args.checkpoint, True, args.stilt_mass, args.device, commands, args.episodes
  )
  off = evaluate(
    args.checkpoint, False, args.stilt_mass, args.device, commands, args.episodes
  )
  report(f"STILTS ON ({args.stilt_mass:.2f} kg per side)", on)
  report("STILTS OFF (bare robot)", off)


if __name__ == "__main__":
  main()
