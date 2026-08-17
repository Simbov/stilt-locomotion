"""Record the trained policy walking, once with the stilts on and once without.

Two clips from the same checkpoint and the same config — the policy is never
told which morphology it is in, so the only difference between the runs is the
`stilts_fitted` draw being pinned.

    uv run python scripts/record_videos.py \
        --checkpoint logs/rsl_rl/stilt_g1_velocity/<run>/model_5999.pt

Writes stilts_on.mp4 and stilts_off.mp4 next to each other. The command is held
fixed rather than resampled so the clip shows a steady gait instead of the
policy chasing a new random target every few seconds.
"""

from __future__ import annotations

import argparse
import math
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

if str(Path(__file__).parent.parent) not in sys.path:
  sys.path.insert(0, str(Path(__file__).parent.parent))

# Held command (vx, vy, yaw). 0.5 m/s forward sits in the band the policy tracks
# well in both morphologies, so the two clips are directly comparable.
COMMAND = (0.5, 0.0, 0.0)
SETTLE_STEPS = 50
# Camera distance as a multiple of standing height. 2.4 keeps the whole robot
# in frame with room around it; lower crops the feet, higher makes it a speck.
FRAMING = 2.4
FPS = 50  # the control rate: step_dt is 0.02 s, so this plays at real time.


def record(
  checkpoint: Path,
  fitted: bool,
  out_path: Path,
  seconds: float,
  width: int,
  height: int,
  device: str,
) -> None:
  import imageio.v2 as imageio
  import torch

  original_load = torch.load
  torch.load = lambda *a, **k: original_load(*a, **{**k, "map_location": device})

  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner
  from mjlab.utils.spec_config import MaterialCfg

  from envs.stilt_g1.env_cfgs import stilt_g1_flat_env_cfg
  from envs.stilt_g1.rl_cfg import stilt_g1_ppo_runner_cfg
  from envs.stilt_g1.stilt_robot import (
    STILT_FITTED_SPAWN_HEIGHT,
    STILT_SPAWN_HEIGHT,
  )

  cfg = stilt_g1_flat_env_cfg(play=True)
  cfg.scene.num_envs = 1
  cfg.events["stilts_fitted"].params["fitted_probability"] = 1.0 if fitted else 0.0
  cfg.viewer.width = width
  cfg.viewer.height = height
  # Camera distance scales with how tall the robot is standing, so the two
  # clips frame it at the same apparent size. The camera tracks torso_link and
  # the frame reaches roughly distance*tan(fovy/2) below it, so a fixed distance
  # either crops the stilt plates 1.3 m down or leaves the bare robot tiny.
  standing_height = (STILT_FITTED_SPAWN_HEIGHT if fitted else STILT_SPAWN_HEIGHT) + 0.55
  cfg.viewer.distance = FRAMING * standing_height
  cfg.viewer.elevation = -6.0

  # Plain grey ground instead of mjlab's blue checkerboard. Dropping the texture
  # entirely and setting reflectance to 0 also kills the mirror sheen, which
  # otherwise reads as a second robot below the floor.
  cfg.scene.terrain.textures = ()
  cfg.scene.terrain.materials = (
    MaterialCfg(
      name="groundplane",
      rgba=(0.55, 0.55, 0.57, 1.0),
      reflectance=0.0,
      geom_names_expr=("terrain$",),
    ),
  )

  raw = ManagerBasedRlEnv(cfg=cfg, device=device, render_mode="rgb_array")
  env = RslRlVecEnvWrapper(raw, clip_actions=None)
  with tempfile.TemporaryDirectory() as tmp:
    runner = VelocityOnPolicyRunner(env, asdict(stilt_g1_ppo_runner_cfg()), tmp, device)
    runner.load(str(checkpoint))
    policy = runner.get_inference_policy(device=device)

  command = raw.command_manager.get_term("twist")
  robot = raw.scene["robot"]
  target = torch.tensor(COMMAND, device=device)
  frames = []
  fell = False

  # The offscreen renderer keeps one MjvCamera across frames, so the azimuth can
  # be steered as it goes. That is what makes a true side-on shot possible:
  # `reset_base` randomises the spawn heading and the twist command is in the
  # BODY frame, so the robot walks off in whatever direction it happens to face.
  # A fixed world azimuth would give a different aspect every take, and the yaw
  # drifts several tenths of a radian over ten seconds anyway. Locking the
  # camera 90 degrees off the live heading holds the profile no matter what.
  camera = raw._offline_renderer._cam  # noqa: SLF001 - no public accessor

  def side_on_azimuth() -> float:
    w, x, y, z = (float(v) for v in robot.data.root_link_quat_w[0])
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return math.degrees(yaw) + 90.0

  with torch.inference_mode():
    obs, _ = env.reset()
    assert bool(raw.stilt_fitted[0] > 0.5) == fitted, "morphology draw was not pinned"
    for _ in range(SETTLE_STEPS):
      command.command[:] = target
      obs, _, _, _ = env.step(policy(obs))

    for _ in range(int(seconds * FPS)):
      command.command[:] = target
      obs, _, dones, _ = env.step(policy(obs))
      fell |= bool(dones[0])
      camera.azimuth = side_on_azimuth()
      frame = raw.render()
      if frame is not None:
        frames.append(frame)

  out_path.parent.mkdir(parents=True, exist_ok=True)
  imageio.mimsave(str(out_path), frames, fps=FPS, macro_block_size=1)
  state = "stilts ON " if fitted else "stilts OFF"
  note = "  (EPISODE ENDED — the robot fell)" if fell else ""
  print(f"{state}  {len(frames)} frames -> {out_path}{note}")
  raw.close()


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--out-dir", type=Path, default=Path("logs/videos"))
  parser.add_argument("--seconds", type=float, default=10.0)
  parser.add_argument("--width", type=int, default=960)
  parser.add_argument("--height", type=int, default=720)
  parser.add_argument("--device", default="cpu")
  args = parser.parse_args()

  for fitted, name in ((True, "stilts_on.mp4"), (False, "stilts_off.mp4")):
    record(
      args.checkpoint,
      fitted,
      args.out_dir / name,
      args.seconds,
      args.width,
      args.height,
      args.device,
    )


if __name__ == "__main__":
  main()
