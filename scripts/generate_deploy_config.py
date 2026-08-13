"""Regenerate deploy.yaml (and a golden I/O vector) from a trained ONNX policy.

Every per-joint array in the deploy config — PD gains, action scales, the
standing pose that doubles as the action offset — is read out of the trained
policy. None of it can be hand-derived, and a plausible-but-wrong deployment
config is worse than an obviously stale one, so this is the only supported way
to produce the file.

    uv run python scripts/generate_deploy_config.py \
        --run logs/rsl_rl/stilt_g1_velocity/<run>

Writes:
  deploy/config/g1_stilt/deploy.yaml       the runtime config
  deploy/config/g1_stilt/reference_io.json golden (observation, action) pairs

The golden vectors exist because the observation layout is the easiest thing to
get wrong on the runtime side, and getting it wrong yields a policy that loads
and runs and merely walks badly. Feed a recorded observation to the deployed
ONNX and check the action matches to ~1e-4 BEFORE putting the robot on the
ground. See deploy/README.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if str(Path(__file__).parent.parent) not in sys.path:
  sys.path.insert(0, str(Path(__file__).parent.parent))

# mjlab's observation term names -> the names the unitree_rl_mjlab runtime uses.
RUNTIME_TERM_NAME = {
  "base_lin_vel": "base_lin_vel",
  "base_ang_vel": "base_ang_vel",
  "projected_gravity": "projected_gravity",
  "joint_pos": "joint_pos_rel",
  "joint_vel": "joint_vel_rel",
  "actions": "last_action",
  "command": "velocity_commands",
}

TERM_NOTE = {
  "base_lin_vel": (
    "linear velocity in body frame. Requires the onboard estimator; if that\n"
    "  # term is unavailable, register a zero-fill stub (see README)."
  ),
  "base_ang_vel": "angular velocity from the IMU gyroscope.",
  "projected_gravity": "gravity projected into the body frame, from the IMU quaternion.",
  "joint_pos": "encoder positions MINUS default_joint_pos.",
  "joint_vel": "encoder velocities.",
  "actions": "previous raw policy output, before scale/offset.",
  "command": "joystick twist [vx, vy, yaw].",
}


def read_metadata(onnx_path: Path) -> dict[str, str]:
  import onnx

  model = onnx.load(str(onnx_path))
  meta = {p.key: p.value for p in model.metadata_props}
  shape = [d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim]
  meta["_input_dim"] = str(shape[-1])
  out = [d.dim_value for d in model.graph.output[0].type.tensor_type.shape.dim]
  meta["_output_dim"] = str(out[-1])
  return meta


def floats(meta: dict[str, str], key: str) -> list[float]:
  return [float(v) for v in meta[key].split(",")]


def fmt_rows(values: list[float], names: list[str], indent: int) -> str:
  """Lay a 29-vector out in leg / waist / arm groups, so it is readable.

  Every row is indented, including the first: these blocks open with a bare
  ``[`` on the preceding line, so the first row is not a continuation.
  """
  groups = [(0, 6), (6, 12), (12, 15), (15, 22), (22, 29)]
  pad = " " * indent
  lines = []
  for a, b in groups:
    body = ", ".join(f"{v:7.3f}" for v in values[a:b])
    tail = "," if b < len(values) else ""
    label = names[a].replace("_joint", "").rsplit("_", 1)[0]
    lines.append(f"{pad}{body}{tail}  # {label}")
  return "\n".join(lines)


def golden_vectors(run_dir: Path, checkpoint: Path, count: int) -> list[dict]:
  """Capture (observation, action) pairs straight out of the sim.

  These are the ground truth for validating the runtime's observation
  assembly. The morphology is pinned OFF, because the bare robot is the first
  hardware test — but the vectors validate wiring, not behaviour, so either
  morphology would do.
  """
  import tempfile
  from dataclasses import asdict

  import torch

  original_load = torch.load
  torch.load = lambda *a, **k: original_load(*a, **{**k, "map_location": "cpu"})

  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

  from envs.stilt_g1.env_cfgs import stilt_g1_flat_env_cfg
  from envs.stilt_g1.rl_cfg import stilt_g1_ppo_runner_cfg

  cfg = stilt_g1_flat_env_cfg(play=True)
  cfg.scene.num_envs = 1
  cfg.events["stilts_fitted"].params["fitted_probability"] = 0.0

  raw = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  env = RslRlVecEnvWrapper(raw, clip_actions=None)
  with tempfile.TemporaryDirectory() as tmp:
    runner = VelocityOnPolicyRunner(env, asdict(stilt_g1_ppo_runner_cfg()), tmp, "cpu")
    runner.load(str(checkpoint))
    policy = runner.get_inference_policy(device="cpu")

  # obs is a TensorDict of observation GROUPS. Only "actor" goes to the ONNX;
  # "critic" is training-only privileged state and is not available on hardware.
  pairs = []
  with torch.inference_mode():
    obs, _ = env.reset()
    for step in range(200):
      action = policy(obs)
      if step >= 200 - count:
        flat = obs["actor"].detach().cpu().numpy().reshape(-1)
        act = action.detach().cpu().numpy().reshape(-1)
        pairs.append(
          {
            "step": step,
            "observation": [round(float(v), 6) for v in flat],
            "action": [round(float(v), 6) for v in act],
          }
        )
      obs, _, _, _ = env.step(action)
  return pairs


def render_yaml(meta: dict[str, str], run_dir: Path, obs_offsets: list[tuple]) -> str:
  names = meta["joint_names"].split(",")
  n = len(names)
  pose = floats(meta, "default_joint_pos")
  stiff = floats(meta, "joint_stiffness")
  damp = floats(meta, "joint_damping")
  scale = floats(meta, "action_scale")
  history = int(float(meta["observation_terms_history_length"].split(",")[0]))

  layout = "\n".join(
    f"#   [{a}:{b}]".ljust(16) + f"{RUNTIME_TERM_NAME[t]:<20} {history} frames x {w}"
    for t, a, b, w in obs_offsets
  )

  obs_block = []
  for term, a, b, width in obs_offsets:
    runtime = RUNTIME_TERM_NAME[term]
    ones = ", ".join(["1.0"] * width)
    params = "\n    params: {command_name: twist}" if term == "command" else ""
    obs_block.append(
      f"  # [{a}:{b}] — {TERM_NOTE[term]}\n"
      f"  {runtime}:{params}\n"
      f"    scale: [{ones}]\n"
      f"    history_length: {history}"
    )

  return f"""\
# ============================================================================
# Deployment config for the Stilt G1 velocity policy.
#
# GENERATED — do not hand-edit. Regenerate with:
#   uv run python scripts/generate_deploy_config.py --run {run_dir}
#
# Source: {run_dir.name}
# Policy: {meta["_input_dim"]} inputs -> {meta["_output_dim"]} actions
#
# ONE POLICY, TWO MORPHOLOGIES. This config is correct with the stilts bolted
# on and with them removed. The policy is not told which — it infers it from
# {history} frames of observation history. There is no mode switch to set, and
# nothing in this file changes between the two.
#
# The four ankle motors stay in NORMAL PD POSITION MODE in both cases. Do not
# put them in damping mode; the policy is trained to drive them into the
# brace's stiffness and is counting on that authority.
# ============================================================================
#
# OBSERVATION LAYOUT — {meta["_input_dim"]} inputs. Read this carefully.
#
# The vector is NOT {history} stacked frames of the full observation. Each term
# contributes ALL {history} of its frames contiguously, oldest first, in this
# order:
#
{layout}
#
# Within each block index 0 is the OLDEST frame. Getting this wrong produces a
# policy that loads, runs, and walks badly. Validate against reference_io.json
# before putting the robot on the ground.

step_dt: 0.02  # 50 Hz control loop

# G1 29-DOF joint ordering, from the ONNX joint_names metadata.
joint_ids_map: [{",".join(str(i) for i in range(n))}]

# Standing pose. Shank vertical with the ankle at the stilt brace's neutral
# angle — the only pose the assembled stilt stands upright in, and shared with
# the bare robot so the neutral action means the same thing in both.
default_joint_pos: [
{fmt_rows(pose, names, 2)}
]

# PD gains — must match training exactly.
stiffness: [
{fmt_rows(stiff, names, 2)}
]

damping: [
{fmt_rows(damp, names, 2)}
]

# Capped at what the policy was actually trained on (the final command
# curriculum stage). Commanding beyond this is extrapolation: the policy
# saturates rather than tracking, and it was never rewarded for trying.
commands:
  twist:
    ranges:
      lin_vel_x:  [-0.6, 0.8]
      lin_vel_y:  [-0.5, 0.5]
      ang_vel_z:  [-0.6, 0.6]

actions:
  JointPositionAction:
    joint_names: [".*"]
    # Final target = (raw policy output * scale) + offset.
    scale: [
{fmt_rows(scale, names, 6)}
    ]
    offset: [
{fmt_rows(pose, names, 6)}
    ]
    clip: null

# Observation normalisation is baked into the ONNX, so every scale is 1.0.
# Do NOT add extra scaling here.
observations:
{chr(10).join(obs_block)}
"""


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--run", type=Path, required=True, help="Training run directory")
  parser.add_argument(
    "--out", type=Path, default=Path("deploy/config/g1_stilt/deploy.yaml")
  )
  parser.add_argument("--golden", type=int, default=3, help="Golden I/O pairs to write")
  args = parser.parse_args()

  onnx_files = sorted(args.run.glob("*.onnx"))
  if not onnx_files:
    raise SystemExit(f"no .onnx in {args.run}")
  meta = read_metadata(onnx_files[0])

  terms = meta["observation_names"].split(",")
  history = int(float(meta["observation_terms_history_length"].split(",")[0]))
  n_joints = len(meta["joint_names"].split(","))
  width = {"joint_pos": n_joints, "joint_vel": n_joints, "actions": n_joints}

  offsets, cursor = [], 0
  for term in terms:
    w = width.get(term, 3)
    offsets.append((term, cursor, cursor + w * history, w))
    cursor += w * history
  assert cursor == int(meta["_input_dim"]), (
    f"layout came to {cursor} but the model takes {meta['_input_dim']}"
  )

  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(render_yaml(meta, args.run, offsets))
  print(f"wrote {args.out}")

  checkpoints = sorted(args.run.glob("model_*.pt"), key=lambda p: p.stat().st_mtime)
  pairs = golden_vectors(args.run, checkpoints[-1], args.golden)
  golden_path = args.out.parent / "reference_io.json"
  golden_path.write_text(
    json.dumps(
      {
        "run": args.run.name,
        "checkpoint": checkpoints[-1].name,
        "note": (
          "Feed 'observation' to the deployed ONNX; 'action' is what it must "
          "return, to ~1e-4. Captured from the sim with the stilts OFF."
        ),
        "input_dim": int(meta["_input_dim"]),
        "output_dim": int(meta["_output_dim"]),
        "pairs": pairs,
      },
      indent=2,
    )
  )
  print(f"wrote {golden_path}  ({len(pairs)} pairs)")


if __name__ == "__main__":
  main()
