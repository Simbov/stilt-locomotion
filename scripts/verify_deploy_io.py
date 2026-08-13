"""Check a deployed ONNX reproduces the sim's actions on recorded observations.

Run this wherever the policy is about to be deployed — including on the robot's
own compute, against the exact .onnx file that shipped there. It catches a
corrupted or wrong-version model file, an ONNX Runtime that disagrees, and the
single most likely deployment mistake: assembling the 495-dim observation in the
wrong order.

    uv run python scripts/verify_deploy_io.py \
        --onnx <policy>.onnx \
        --reference deploy/config/g1_stilt/reference_io.json

This validates the MODEL and the observation LAYOUT. It cannot tell you whether
the runtime is filling those observations with the right sensor values — for
that, log a real observation off the robot while it is held still in the
standing pose and compare it term by term against the reference.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

TOLERANCE = 1e-4


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--onnx", type=Path, required=True)
  parser.add_argument(
    "--reference",
    type=Path,
    default=Path("deploy/config/g1_stilt/reference_io.json"),
  )
  parser.add_argument("--tolerance", type=float, default=TOLERANCE)
  args = parser.parse_args()

  import onnxruntime as ort

  ref = json.loads(args.reference.read_text())
  session = ort.InferenceSession(str(args.onnx))
  spec_in = session.get_inputs()[0]
  spec_out = session.get_outputs()[0]

  print(f"model      {args.onnx}")
  print(f"reference  {args.reference}  (run {ref['run']}, {ref['checkpoint']})")
  print(f"signature  {spec_in.name} {spec_in.shape} -> {spec_out.shape}")

  if spec_in.shape[-1] != ref["input_dim"]:
    print(
      f"FAIL: model takes {spec_in.shape[-1]} inputs, reference has "
      f"{ref['input_dim']}. Wrong policy version."
    )
    return 1

  worst = 0.0
  for pair in ref["pairs"]:
    obs = np.asarray(pair["observation"], dtype=np.float32).reshape(1, -1)
    expected = np.asarray(pair["action"], dtype=np.float32)
    actual = session.run(None, {spec_in.name: obs})[0].reshape(-1)
    error = float(np.abs(actual - expected).max())
    worst = max(worst, error)
    print(f"  step {pair['step']:>4}  max|onnx - sim| = {error:.3e}")

  if worst < args.tolerance:
    print(f"PASS  worst {worst:.3e} < {args.tolerance:g}")
    return 0
  print(f"FAIL  worst {worst:.3e} >= {args.tolerance:g}")
  print(
    "If the model itself is right, suspect the observation layout: the vector\n"
    "is per-term history, NOT stacked frames. See deploy/README.md."
  )
  return 1


if __name__ == "__main__":
  sys.exit(main())
