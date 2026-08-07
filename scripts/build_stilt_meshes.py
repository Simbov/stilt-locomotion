"""Partition the assembled stilt STL into per-segment meshes and inertials.

Single source of truth for the stilt segment geometry in assets/mjcf/g1/g1.xml.
Rerun whenever the CAD changes:

    uv run python scripts/build_stilt_meshes.py

It writes one decimated STL per segment and prints ready-to-paste MJCF
<inertial> blocks.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh

REPO_ROOT = Path(__file__).parent.parent
SOURCE_STL = Path.home() / "Downloads" / "Assembled 40.7cm.STL"
DEFAULT_OUT = REPO_ROOT / "assets" / "mjcf" / "g1" / "assets"

SEGMENTS = (
  "stilt_mount",
  "stilt_brace",
  "stilt_post_outer",
  "stilt_post_inner",
  "stilt_plate",
)

# Mesh point (mm) that maps onto the ankle_roll_link origin.
MESH_ORIGIN_MM = (114.1, 66.2, 667.3)

# Triangle budget per segment, ~15k total per stilt (spec §4.1).
FACE_BUDGET = {
  "stilt_mount": 3000,
  "stilt_brace": 3000,
  "stilt_post_outer": 2000,
  "stilt_post_inner": 2000,
  "stilt_plate": 5000,
}

# Anchor parts, identified by bounding-box extents in mm (rounded).
_ANCHOR_EXTENTS: dict[tuple[float, float, float], str] = {
  (220.0, 110.0, 35.0): "stilt_mount",
  (30.0, 60.0, 200.0): "stilt_brace",
  (30.0, 60.0, 150.0): "stilt_brace",
  (45.0, 120.0, 30.0): "stilt_brace",
  (40.0, 40.0, 250.0): "stilt_post_outer",
  (35.0, 35.0, 250.0): "stilt_post_inner",
  (220.0, 80.0, 35.0): "stilt_plate",
}

# 1 mount plate + 1 ground plate + 2 outer posts + 2 inner posts + 3 brace parts.
_EXPECTED_ANCHORS = 9


@dataclass(frozen=True)
class SegmentProps:
  """Rigid-body properties of one stilt segment, in ankle-frame metres."""

  mass: float
  com: tuple[float, float, float]
  diaginertia: tuple[float, float, float]
  quat: tuple[float, float, float, float]
  z_min: float
  z_max: float


def uniform_density(mesh: trimesh.Trimesh, total_mass_kg: float) -> float:
  """Density (kg/m^3) that makes the summed solid volume weigh total_mass_kg."""
  volume_mm3 = sum(abs(c.volume) for c in mesh.split(only_watertight=False))
  return total_mass_kg / (volume_mm3 * 1e-9)


def partition(mesh: trimesh.Trimesh) -> dict[str, list[trimesh.Trimesh]]:
  """Assign every connected solid to a segment by nearest anchor centroid."""
  components = mesh.split(only_watertight=False)

  anchors: list[tuple[np.ndarray, str]] = []
  for comp in components:
    key = tuple(np.round(comp.extents, 0))
    name = _ANCHOR_EXTENTS.get(key)  # type: ignore[arg-type]
    if name is not None:
      anchors.append((comp.center_mass, name))
  if len(anchors) != _EXPECTED_ANCHORS:
    raise RuntimeError(
      f"expected {_EXPECTED_ANCHORS} anchor solids, found {len(anchors)} — "
      "CAD changed, update _ANCHOR_EXTENTS"
    )

  groups: dict[str, list[trimesh.Trimesh]] = {name: [] for name in SEGMENTS}
  for comp in components:
    centroid = comp.center_mass
    nearest = min(anchors, key=lambda a: float(np.linalg.norm(centroid - a[0])))
    groups[nearest[1]].append(comp)
  return groups


def segment_properties(
  parts: list[trimesh.Trimesh],
  density: float,
  origin_mm: tuple[float, float, float],
) -> SegmentProps:
  """Mass, COM and principal inertia of a segment, expressed in ankle frame."""
  origin_m = np.asarray(origin_mm) * 1e-3

  mass = 0.0
  weighted_com = np.zeros(3)
  for part in parts:
    part_mass = abs(part.volume) * 1e-9 * density
    mass += part_mass
    weighted_com += part_mass * part.center_mass * 1e-3
  com = weighted_com / mass

  inertia = np.zeros((3, 3))
  for part in parts:
    part_mass = abs(part.volume) * 1e-9 * density
    # trimesh reports inertia for unit density in mm^5; rescale to this part.
    part_inertia = part.moment_inertia / abs(part.volume) * 1e-6 * part_mass
    offset = part.center_mass * 1e-3 - com
    inertia += part_inertia + part_mass * (
      np.dot(offset, offset) * np.eye(3) - np.outer(offset, offset)
    )

  eigenvalues, eigenvectors = np.linalg.eigh(inertia)
  # eigh may return a reflection; MuJoCo needs a proper rotation.
  if np.linalg.det(eigenvectors) < 0:
    eigenvectors[:, 0] *= -1.0
  transform = np.eye(4)
  transform[:3, :3] = eigenvectors
  quat = trimesh.transformations.quaternion_from_matrix(transform)

  bounds = np.array([p.bounds for p in parts])
  return SegmentProps(
    mass=float(mass),
    com=tuple((com - origin_m).tolist()),
    diaginertia=tuple(eigenvalues.tolist()),
    quat=tuple(quat.tolist()),
    z_min=float(bounds[:, 0, 2].min() * 1e-3 - origin_m[2]),
    z_max=float(bounds[:, 1, 2].max() * 1e-3 - origin_m[2]),
  )


def export_segment(
  parts: list[trimesh.Trimesh],
  name: str,
  origin_mm: tuple[float, float, float],
  out_dir: Path,
) -> tuple[Path, int]:
  """Merge, re-origin and decimate one segment, then write it as STL."""
  merged = trimesh.util.concatenate(parts)
  merged.apply_translation(-np.asarray(origin_mm))

  budget = FACE_BUDGET[name]
  if len(merged.faces) > budget:
    merged = merged.simplify_quadric_decimation(face_count=budget)

  path = out_dir / f"{name}.STL"
  merged.export(path)
  return path, len(merged.faces)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--source", type=Path, default=SOURCE_STL)
  parser.add_argument("--total-mass", type=float, default=2.8)
  parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
  args = parser.parse_args()

  mesh = trimesh.load(str(args.source), force="mesh")
  density = uniform_density(mesh, args.total_mass)
  groups = partition(mesh)

  args.out.mkdir(parents=True, exist_ok=True)
  print(f"effective uniform density: {density:.0f} kg/m^3\n")

  total = 0.0
  faces = 0
  for name in SEGMENTS:
    props = segment_properties(groups[name], density, MESH_ORIGIN_MM)
    path, face_count = export_segment(groups[name], name, MESH_ORIGIN_MM, args.out)
    total += props.mass
    faces += face_count
    print(
      f"<!-- {name}: {len(groups[name])} solids, "
      f"z {props.z_min:+.4f}..{props.z_max:+.4f} m, "
      f"{path.name} ({face_count} faces) -->"
    )
    print(
      f'<inertial pos="{props.com[0]:.5f} {props.com[1]:.5f} {props.com[2]:.5f}"\n'
      f'  quat="{props.quat[0]:.6f} {props.quat[1]:.6f} '
      f'{props.quat[2]:.6f} {props.quat[3]:.6f}"\n'
      f'  mass="{props.mass:.4f}"\n'
      f'  diaginertia="{props.diaginertia[0]:.6f} '
      f'{props.diaginertia[1]:.6f} {props.diaginertia[2]:.6f}"/>\n'
    )
  print(f"total mass: {total:.3f} kg over {faces} faces per stilt")


if __name__ == "__main__":
  main()
