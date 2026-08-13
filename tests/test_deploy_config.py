"""The deploy config must agree with the ONNX it was generated from.

A deployment config that looks plausible but disagrees with the trained policy
is worse than an obviously stale one — it produces a robot that stands up and
then behaves wrongly. These checks are cheap and catch the whole class.

They skip cleanly when the run directory is absent, so the suite still passes on
a fresh clone with no logs.
"""

from pathlib import Path

import numpy as np
import pytest

yaml = pytest.importorskip("yaml")

ROOT = Path(__file__).parent.parent
DEPLOY_YAML = ROOT / "deploy" / "config" / "g1_stilt" / "deploy.yaml"
REFERENCE_IO = ROOT / "deploy" / "config" / "g1_stilt" / "reference_io.json"


@pytest.fixture(scope="module")
def config():
  if not DEPLOY_YAML.exists():
    pytest.skip("no deploy.yaml")
  return yaml.safe_load(DEPLOY_YAML.read_text())


@pytest.fixture(scope="module")
def metadata(config):
  """ONNX metadata for the run the config names in its header."""
  onnx = pytest.importorskip("onnx")
  source = None
  for line in DEPLOY_YAML.read_text().splitlines():
    if line.startswith("# Source:"):
      source = line.split(":", 1)[1].strip()
      break
  if source is None:
    pytest.skip("deploy.yaml has no Source header — hand-written?")
  run_dir = ROOT / "logs" / "rsl_rl" / "stilt_g1_velocity" / source
  files = sorted(run_dir.glob("*.onnx"))
  if not files:
    pytest.skip(f"no ONNX for run {source}")
  model = onnx.load(str(files[0]))
  meta = {p.key: p.value for p in model.metadata_props}
  meta["_input_dim"] = model.graph.input[0].type.tensor_type.shape.dim[-1].dim_value
  meta["_output_dim"] = model.graph.output[0].type.tensor_type.shape.dim[-1].dim_value
  return meta


def _floats(meta, key):
  return np.array([float(v) for v in meta[key].split(",")])


@pytest.mark.parametrize(
  "yaml_path,meta_key",
  [
    ("default_joint_pos", "default_joint_pos"),
    ("stiffness", "joint_stiffness"),
    ("damping", "joint_damping"),
  ],
)
def test_per_joint_arrays_match_the_policy(config, metadata, yaml_path, meta_key):
  got = np.array(config[yaml_path], dtype=float)
  expected = _floats(metadata, meta_key)
  assert got.shape == expected.shape
  np.testing.assert_allclose(got, expected, atol=1e-6)


def test_action_scale_and_offset_match_the_policy(config, metadata):
  action = config["actions"]["JointPositionAction"]
  np.testing.assert_allclose(
    np.array(action["scale"], dtype=float), _floats(metadata, "action_scale"), atol=1e-6
  )
  # The offset IS the standing pose: a zero action must command it.
  np.testing.assert_allclose(
    np.array(action["offset"], dtype=float),
    _floats(metadata, "default_joint_pos"),
    atol=1e-6,
  )


def test_observation_widths_sum_to_the_policy_input(config, metadata):
  """The single easiest thing to get wrong, and it fails silently at runtime."""
  total = sum(
    len(term["scale"]) * term["history_length"]
    for term in config["observations"].values()
  )
  assert total == metadata["_input_dim"]


def test_every_observation_term_carries_the_history(config, metadata):
  expected = int(float(metadata["observation_terms_history_length"].split(",")[0]))
  assert expected > 1, "this policy has no history; the rest of this test is moot"
  for name, term in config["observations"].items():
    assert term["history_length"] == expected, (
      f"{name} has history_length {term['history_length']}, policy needs {expected}"
    )


def test_commands_do_not_exceed_what_was_trained(config):
  """Beyond the trained range the policy saturates rather than tracking."""
  ranges = config["commands"]["twist"]["ranges"]
  trained = {
    "lin_vel_x": (-0.6, 0.8),
    "lin_vel_y": (-0.5, 0.5),
    "ang_vel_z": (-0.6, 0.6),
  }
  for axis, (low, high) in trained.items():
    got_low, got_high = ranges[axis]
    assert got_low >= low and got_high <= high, (
      f"{axis} deploys {ranges[axis]} but was only trained on ({low}, {high})"
    )


def test_the_golden_vectors_reproduce_through_the_onnx():
  """Pins the observation layout end to end. See scripts/verify_deploy_io.py."""
  import json

  ort = pytest.importorskip("onnxruntime")
  onnx = pytest.importorskip("onnx")
  del onnx
  if not REFERENCE_IO.exists():
    pytest.skip("no reference_io.json")

  ref = json.loads(REFERENCE_IO.read_text())
  run_dir = ROOT / "logs" / "rsl_rl" / "stilt_g1_velocity" / ref["run"]
  files = sorted(run_dir.glob("*.onnx"))
  if not files:
    pytest.skip(f"no ONNX for run {ref['run']}")

  session = ort.InferenceSession(str(files[0]))
  name = session.get_inputs()[0].name
  for pair in ref["pairs"]:
    obs = np.asarray(pair["observation"], dtype=np.float32).reshape(1, -1)
    actual = session.run(None, {name: obs})[0].reshape(-1)
    np.testing.assert_allclose(
      actual, np.asarray(pair["action"], dtype=np.float32), atol=1e-4
    )
