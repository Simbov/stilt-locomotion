#!/usr/bin/env python3
"""Write simulation telemetry in the robot's CSV format, with ground truth.

    uv run python scripts/sim_telemetry.py                 # bare feet, Run 9
    uv run python scripts/sim_telemetry.py --stilts
    uv run python scripts/analyze_hardware_log.py logs/hardware/sim/<file>.csv

Two jobs:

1. Validate the leg odometry in analyze_hardware_log.py. The file carries
   `true_x`, `true_y`, `true_z` from the simulator, and the analyser prints them
   next to its estimate.
2. A sim baseline in exactly the format the robot writes, so a hardware session
   can be read against the same command schedule.

Rows follow the robot's semantics: the state the policy observed, the action it
produced from it, and the command it saw. There is no accelerometer or motor
temperature in sim; those columns are zero.
"""

from __future__ import annotations

import argparse
import csv
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import yaml

_orig_load = torch.load
torch.load = lambda *a, **k: _orig_load(*a, **{**k, "map_location": "cpu"})

from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.rl import RslRlVecEnvWrapper  # noqa: E402
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner  # noqa: E402

from envs.stilt_g1.env_cfgs import stilt_g1_flat_env_cfg  # noqa: E402
from envs.stilt_g1.rl_cfg import stilt_g1_ppo_runner_cfg  # noqa: E402
from scripts.analyze_hardware_log import (  # noqa: E402
  DEPLOY_YAML,
  N_JOINTS,
  joint_names,
  telemetry_header,
)

ROOT = Path(__file__).resolve().parent.parent
RUN = "2026-08-31_13-35-11_run9-no-baselinvel"
STEP_S = 0.02

# (seconds, vx, vy, wz): the gates in the order they are run on the robot.
SCHEDULE = [
  (6, 0.0, 0.0, 0.0),
  (5, 0.4, 0.0, 0.0),
  (5, 0.0, 0.0, 0.0),
  (5, 0.0, 0.3, 0.0),
  (5, 0.0, 0.0, 0.5),
  (5, -0.3, 0.0, 0.0),
  (8, 0.0, 0.0, 0.0),
]


def main() -> int:
  ap = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
  )
  ap.add_argument("--stilts", action="store_true", help="stilts fitted (default: bare)")
  ap.add_argument(
    "--ckpt",
    type=Path,
    default=ROOT / "logs" / "rsl_rl" / "stilt_g1_velocity" / RUN / "model_5999.pt",
  )
  ap.add_argument("--out", type=Path)
  args = ap.parse_args()
  mode = "stilts" if args.stilts else "bare"
  out = (
    args.out
    or ROOT / "logs" / "hardware" / "sim" / f"{args.ckpt.parent.name}_{mode}.csv"
  )

  cfg = stilt_g1_flat_env_cfg(play=True)
  cfg.scene.num_envs = 1
  cfg.events["stilts_fitted"].params["fitted_probability"] = 1.0 if args.stilts else 0.0

  raw = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  env = RslRlVecEnvWrapper(raw, clip_actions=None)
  with tempfile.TemporaryDirectory() as tmp:
    runner = VelocityOnPolicyRunner(env, asdict(stilt_g1_ppo_runner_cfg()), tmp, "cpu")
    runner.load(str(args.ckpt))
    policy = runner.get_inference_policy(device="cpu")

  robot = raw.scene["robot"]
  command = raw.command_manager.get_term("twist")
  names, _ = joint_names()
  if list(robot.joint_names) != names:
    raise SystemExit(
      "sim joint order differs from the policy's; columns would be mislabelled"
    )

  deploy = yaml.safe_load(DEPLOY_YAML.read_text())
  action_cfg = deploy["actions"]["JointPositionAction"]
  scale = np.array(action_cfg["scale"])
  offset = np.array(action_cfg["offset"])
  origin = raw.scene.env_origins[0].numpy()

  out.parent.mkdir(parents=True, exist_ok=True)
  wall0 = time.time()
  segment, step, total = 1, 0, 0
  with out.open("w", newline="") as f, torch.inference_mode():
    writer = csv.writer(f)
    writer.writerow(telemetry_header() + ["true_x", "true_y", "true_z"])
    obs, _ = env.reset()
    for seconds, vx, vy, wz in SCHEDULE:
      target = torch.tensor([vx, vy, wz])
      for _ in range(round(seconds / STEP_S)):
        command.command[:] = target
        d = robot.data
        state = [
          d.root_link_quat_w[0].numpy(),
          d.root_link_ang_vel_b[0].numpy(),
          np.zeros(3),
          np.array([vx, vy, wz]),
          d.joint_pos[0].numpy(),
          d.joint_vel[0].numpy(),
          d.qfrc_actuator[0].numpy(),
        ]
        true_pos = d.root_link_pos_w[0].numpy() - origin

        action = policy(obs)
        a = action[0].numpy()
        step += 1
        t = total * STEP_S
        head = [f"{wall0 + t:.6f}", f"{t:.6f}", segment, step, step, "20.000", "0.000"]
        values = np.concatenate(state + [offset + scale * a, a, np.zeros(N_JOINTS)])
        writer.writerow(
          head + [f"{v:.6g}" for v in values] + [f"{v:.6f}" for v in true_pos]
        )

        obs, _, dones, _ = env.step(action)
        total += 1
        if bool(dones[0]):
          print(f"episode ended at t={t:.1f} s; starting segment {segment + 1}")
          segment, step = segment + 1, 0

  print(f"wrote {out} ({total} steps, {mode})")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
