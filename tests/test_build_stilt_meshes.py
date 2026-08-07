"""The generator must reproduce the segment masses recorded in the spec."""

import pytest
import trimesh

from scripts.build_stilt_meshes import (
  MESH_ORIGIN_MM,
  SEGMENTS,
  SOURCE_STL,
  partition,
  segment_properties,
  uniform_density,
)

# From docs/superpowers/specs/2026-08-07-new-stilt-design.md §4.
EXPECTED_MASS_KG = {
  "stilt_mount": 0.636,
  "stilt_brace": 0.999,
  "stilt_post_outer": 0.313,
  "stilt_post_inner": 0.288,
  "stilt_plate": 0.565,
}
TOTAL_MASS_KG = 2.8


@pytest.fixture(scope="module")
def parts():
  mesh = trimesh.load(str(SOURCE_STL), force="mesh")
  return partition(mesh), uniform_density(mesh, TOTAL_MASS_KG)


def test_every_segment_is_populated(parts):
  groups, _ = parts
  assert set(groups) == set(SEGMENTS)
  for name in SEGMENTS:
    assert groups[name], f"segment {name} got no solids"


def test_partition_is_a_complete_disjoint_cover(parts):
  groups, _ = parts
  assigned = sum(len(v) for v in groups.values())
  assert assigned == 66, f"expected all 66 solids assigned, got {assigned}"


def test_segment_masses_match_spec(parts):
  groups, density = parts
  for name, expected in EXPECTED_MASS_KG.items():
    props = segment_properties(groups[name], density, MESH_ORIGIN_MM)
    assert props.mass == pytest.approx(expected, abs=0.002), name


def test_total_mass_is_conserved(parts):
  groups, density = parts
  total = sum(
    segment_properties(groups[n], density, MESH_ORIGIN_MM).mass for n in SEGMENTS
  )
  assert total == pytest.approx(TOTAL_MASS_KG, abs=0.005)


def test_ground_contact_is_at_minus_442_5_mm(parts):
  groups, density = parts
  plate = segment_properties(groups["stilt_plate"], density, MESH_ORIGIN_MM)
  assert plate.z_min == pytest.approx(-0.4425, abs=1e-4)
