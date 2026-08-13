"""Structural load and stress analysis for the stilt, from a trained policy.

Regenerates every number in docs/stilt-structural-report. Run:

    uv run python scripts/analyse_stilt_loads.py \
        --checkpoint logs/rsl_rl/stilt_g1_velocity/<run>/model_5999.pt

Three stages, each printed with its provenance so the report can cite it:

  1. MEASURED (CAD)  — section geometry read off the source STL
  2. MEASURED (SIM)  — load distributions sampled from the policy
  3. COMPUTED        — stresses from 1 and 2 by closed-form beam theory

The two-post frame is the part worth understanding: the posts sit 135 mm apart
fore-aft, so a fore-aft moment is reacted as a push-pull couple between them
(axial), while only the lateral moment actually bends the tubes.
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

# Post centres measured from the CAD at mesh x = 86.6 and 221.6 mm.
POST_SPACING_M = 0.135
# Rear post centre and heel capsule, ankle frame — the ground-plate cantilever.
HEEL_CANTILEVER_M = (60 - 28) / 1000
# Plate thickness from voxelising the CAD (see measure_geometry).
PLATE_THICKNESS_MM = 16.0
PLATE_WIDTH_MM = 80.0
PLATE_EFFECTIVE_FRACTION = 0.70  # cutouts; see report caveats

COMMANDS = [
  (0.4, 0, 0),
  (0.6, 0, 0),
  (0.8, 0, 0),
  (0, 0.4, 0),
  (0, 0, 0.6),
  (-0.4, 0, 0),
]
SETTLE_STEPS = 40
SAMPLE_STEPS = 250

MATERIALS = {
  # name: (yield or UTS MPa, design allowable MPa, source note)
  "6061-T6 aluminium": (240, 80, "min yield; allowable = yield/3"),
  "6063-T5 extrusion": (145, 48, "min yield; allowable = yield/3"),
  "PLA (FDM, XY)": (40, 12, "typical printed UTS; allowable derated for creep+fatigue"),
  "PLA (FDM, Z)": (22, 7, "layer adhesion governs"),
  "PA6-CF (FDM)": (90, 30, "typical printed UTS"),
}


def tube_section(a_mm: float, t_mm: float) -> tuple[float, float, float]:
  """Area, second moment and section modulus of a square tube."""
  b = a_mm - 2 * t_mm
  area = a_mm**2 - b**2
  inertia = (a_mm**4 - b**4) / 12
  return area, inertia, inertia / (a_mm / 2)


def measure_geometry(source_stl: Path) -> dict:
  """Read section geometry straight off the CAD rather than assuming it."""
  import trimesh

  mesh = trimesh.load(str(source_stl), force="mesh")
  parts = mesh.split(only_watertight=False)

  wanted = {
    (40.0, 40.0, 250.0): "post_outer",
    (35.0, 35.0, 250.0): "post_inner",
    (220.0, 80.0, 35.0): "ground_plate",
    (220.0, 110.0, 35.0): "mount_plate",
  }
  out: dict[str, dict] = {}
  for part in parts:
    key = tuple(np.round(part.extents, 1))
    name = wanted.get(key)  # type: ignore[arg-type]
    if name is None or name in out:
      continue
    volume_cm3 = abs(part.volume) / 1000
    bbox_cm3 = float(np.prod(part.extents)) / 1000
    fill = volume_cm3 / bbox_cm3
    entry = {"extents_mm": key, "volume_cm3": volume_cm3, "fill_frac": fill}
    if name.startswith("post"):
      # Solve wall thickness from the fill fraction of a square tube.
      a = min(key[0], key[1])
      entry["wall_mm"] = a / 2 * (1 - math.sqrt(max(0.0, 1 - fill)))
    out[name] = entry
  return out


def sample_loads(checkpoint: Path, stilt_mass_kg: float) -> np.ndarray:
  """Sample the ground-reaction wrench about the post-pair centroid.

  Returns |Fz|, |Fxy|, |Mx| (lateral), |My| (fore-aft) per frame per side.
  """
  import torch

  original_load = torch.load
  torch.load = lambda *a, **k: original_load(*a, **{**k, "map_location": "cpu"})

  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

  from envs.stilt_g1.env_cfgs import STILT_CONTACT_SENSOR, stilt_g1_flat_env_cfg
  from envs.stilt_g1.rl_cfg import stilt_g1_ppo_runner_cfg

  alpha = 0.5 * math.log(stilt_mass_kg / 2.8)
  cfg = stilt_g1_flat_env_cfg(play=True)
  cfg.scene.num_envs = 1
  cfg.events["stilt_mass"].params["alpha_range"] = (alpha, alpha)
  # The policy is trained on a 50/50 stilts-on/off draw; pin it on, or half the
  # sampled episodes would be the bare robot and report zero stilt load.
  cfg.events["stilts_fitted"].params["fitted_probability"] = 1.0

  raw = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  env = RslRlVecEnvWrapper(raw, clip_actions=None)
  with tempfile.TemporaryDirectory() as tmp:
    runner = VelocityOnPolicyRunner(env, asdict(stilt_g1_ppo_runner_cfg()), tmp, "cpu")
    runner.load(str(checkpoint))
    policy = runner.get_inference_policy(device="cpu")

  robot = raw.scene["robot"]
  sensor = raw.scene.sensors[STILT_CONTACT_SENSOR]
  names = list(sensor.primary_names)
  command = raw.command_manager.get_term("twist")
  body_names = [b.name.split("/")[-1] for b in robot.indexing.bodies]
  capsules = [f"{b}{i}" for b in "lr" for i in range(1, 5)]

  rows: list[tuple[float, float, float, float]] = []
  heel: list[
    float
  ] = []  # capsule l1, the heel-most contact - drives the plate cantilever
  with torch.inference_mode():
    for cmd in COMMANDS:
      obs, _ = env.reset()
      for _ in range(SETTLE_STEPS):
        command.command[:] = torch.tensor(cmd)
        obs, _, _, _ = env.step(policy(obs))
      for _ in range(SAMPLE_STEPS):
        command.command[:] = torch.tensor(cmd)
        obs, _, _, _ = env.step(policy(obs))
        for side in ("left", "right"):
          centre = robot.data.body_link_pos_w[
            0, body_names.index(f"{side}_stilt_post_outer")
          ].numpy()
          force = np.zeros(3)
          moment = np.zeros(3)
          for cap in capsules:
            col = names.index(f"{side}_stilt_{cap}_collision")
            f = sensor.data.force[0, col].numpy().astype(float)
            p = sensor.data.pos[0, col].numpy().astype(float)
            force += f
            moment += np.cross(p - centre, f)
          rows.append((force[2], math.hypot(force[0], force[1]), moment[0], moment[1]))
          heel.append(
            abs(
              float(sensor.data.force[0, names.index(f"{side}_stilt_l1_collision"), 2])
            )
          )
  return np.abs(np.array(rows)), np.array(heel)


def report(
  geometry: dict, loads: np.ndarray, heel: np.ndarray, stilt_mass_kg: float
) -> None:
  labels = [
    "axial Fz (N)",
    "shear Fxy (N)",
    "lateral moment Mx (Nm)",
    "fore-aft moment My (Nm)",
  ]

  print("\n[MEASURED - CAD] section geometry, from the source STL")
  for name, g in sorted(geometry.items()):
    wall = f"  wall {g['wall_mm']:.1f} mm" if "wall_mm" in g else ""
    print(
      f"  {name:14s} {str(g['extents_mm']):>26}  vol {g['volume_cm3']:6.1f} cm3"
      f"  fill {100 * g['fill_frac']:4.1f}%{wall}"
    )

  print(
    f"\n[MEASURED - SIM] ground reaction, {stilt_mass_kg:.2f} kg/stilt, n={len(loads)}"
  )
  print(f"  {'quantity':26s}{'median':>10}{'p95':>10}{'p99':>10}{'max':>10}")
  for i, lab in enumerate(labels):
    col = loads[:, i]
    print(
      f"  {lab:26s}{np.median(col):10.1f}{np.percentile(col, 95):10.1f}"
      f"{np.percentile(col, 99):10.1f}{col.max():10.1f}"
    )

  fz, mx, my = loads[:, 0], loads[:, 2], loads[:, 3]
  print(
    "\n[COMPUTED] per-post stress (two-post frame: My -> axial couple, Mx -> bending)"
  )
  for name, a, t in (("post_outer", 40, 2.5), ("post_inner", 35, 2.4)):
    area, inertia, modulus = tube_section(a, t)
    axial = fz / 2 + my / POST_SPACING_M
    sigma = axial / area + (mx / 2) * 1000 / modulus
    print(
      f"  {name:12s} A={area:5.0f} mm2 I={inertia:7.0f} mm4 Z={modulus:6.0f} mm3"
      f" | sigma median {np.median(sigma):5.1f}  p99 {np.percentile(sigma, 99):5.1f}"
      f"  max {sigma.max():5.1f} MPa"
    )

  z_plate = PLATE_EFFECTIVE_FRACTION * PLATE_WIDTH_MM * PLATE_THICKNESS_MM**2 / 6
  print(
    f"\n[COMPUTED] ground-plate cantilever, arm {HEEL_CANTILEVER_M * 1000:.0f} mm,"
    f" Z={z_plate:.0f} mm3 ({PLATE_EFFECTIVE_FRACTION:.0%} of {PLATE_WIDTH_MM:.0f} mm width)"
  )
  for lab, f in (("p99 ", float(np.percentile(heel, 99))), ("max ", float(heel.max()))):
    moment = f * HEEL_CANTILEVER_M
    print(
      f"  heel capsule {lab} {f:7.0f} N  ->  M {moment:5.1f} Nm"
      f"  ->  sigma {moment * 1000 / z_plate:5.1f} MPa"
    )

  print("\n[STANDARD] material allowables")
  for name, (ultimate, allow, note) in MATERIALS.items():
    print(
      f"  {name:22s} {ultimate:4.0f} MPa yield/UTS   {allow:3.0f} MPa design   ({note})"
    )


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument(
    "--source-stl",
    type=Path,
    default=Path.home() / "Downloads" / "Assembled 40.7cm.STL",
  )
  parser.add_argument("--stilt-mass", type=float, default=2.8)
  args = parser.parse_args()

  geometry = measure_geometry(args.source_stl)
  loads, heel = sample_loads(args.checkpoint, args.stilt_mass)
  report(geometry, loads, heel, args.stilt_mass)


if __name__ == "__main__":
  main()
