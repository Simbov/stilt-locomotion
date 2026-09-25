"""D-pad Down drops the G1 to Passive from every state that has the L2+B kill."""

import shutil
from pathlib import Path

import pytest
import yaml

from scripts import prepare_runtime as pr

CFG = (
  Path(__file__).parent.parent / "unitree_rl_mjlab/deploy/robots/g1/config/config.yaml"
)


def test_quick_damp_patch_lands_on_the_stock_config(tmp_path):
  if not CFG.is_file():
    pytest.skip("unitree_rl_mjlab submodule not checked out")
  p = tmp_path / "config.yaml"
  shutil.copy(CFG, p)
  pr.patch_quick_damp(p)

  fsm = yaml.safe_load(p.read_text())["FSM"]
  stock = yaml.safe_load(CFG.read_text())["FSM"]
  for state, body in stock.items():
    if state == "_":
      continue
    kill = body.get("transitions", {}).get("Passive")
    if kill is None:
      continue
    assert (
      fsm[state]["transitions"]["Passive"] == "(LT + B.on_pressed) | down.on_pressed"
    )
  # Nothing else is bound to D-pad Down, so it can't fire a different transition.
  others = [
    expr
    for state, body in fsm.items()
    if state != "_"
    for target, expr in body.get("transitions", {}).items()
    if target != "Passive"
  ]
  assert not any("down" in e for e in others)
