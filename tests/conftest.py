"""Shared fixtures for stilt-locomotion tests."""

from pathlib import Path

import mujoco
import pytest

REPO_ROOT = Path(__file__).parent.parent
G1_XML = REPO_ROOT / "assets" / "mjcf" / "g1" / "g1.xml"

SEGMENTS = (
  "stilt_mount",
  "stilt_brace",
  "stilt_post_outer",
  "stilt_post_inner",
  "stilt_plate",
)


@pytest.fixture(scope="session")
def stilt_model():
  """Compile the stilt G1 MJCF and run one forward pass."""
  spec = mujoco.MjSpec.from_file(str(G1_XML))
  model = spec.compile()
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  return model, data
