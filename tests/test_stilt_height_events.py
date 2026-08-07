"""Spawn-height correction must exactly cancel the sampled telescope offset."""

import pytest

from envs.stilt_g1.events import spawn_height_correction


def test_taller_stilt_raises_the_robot():
  """A more negative body_pos z means a longer stilt, so the root must rise."""
  assert spawn_height_correction(sampled_z=-0.05, nominal_z=0.0) == pytest.approx(+0.05)


def test_shorter_stilt_lowers_the_robot():
  assert spawn_height_correction(sampled_z=+0.03, nominal_z=0.0) == pytest.approx(-0.03)


def test_nominal_height_needs_no_correction():
  assert spawn_height_correction(sampled_z=0.0, nominal_z=0.0) == pytest.approx(0.0)
