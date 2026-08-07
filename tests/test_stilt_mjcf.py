"""The stilt MJCF must match the geometry and topology in the spec."""

import mujoco
import pytest

SEGMENT_BODIES = (
  "stilt_mount",
  "stilt_brace",
  "stilt_post_outer",
  "stilt_post_inner",
  "stilt_plate",
)
SIDES = ("left", "right")


def _body_id(model, name):
  return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)


def test_ankle_joints_are_gone(stilt_model):
  model, _ = stilt_model
  names = {
    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)
  }
  offenders = {n for n in names if n and "ankle" in n}
  assert not offenders, f"ankle joints still present: {offenders}"


def test_ankle_links_survive(stilt_model):
  """The bodies must stay — the foot_swing_height contact subtree depends on them."""
  model, _ = stilt_model
  for side in SIDES:
    assert _body_id(model, f"{side}_ankle_roll_link") >= 0


def test_all_segment_bodies_exist(stilt_model):
  model, _ = stilt_model
  for side in SIDES:
    for segment in SEGMENT_BODIES:
      assert _body_id(model, f"{side}_{segment}") >= 0, f"{side}_{segment}"


def test_segments_add_no_degrees_of_freedom(stilt_model):
  model, _ = stilt_model
  for side in SIDES:
    for segment in SEGMENT_BODIES:
      bid = _body_id(model, f"{side}_{segment}")
      assert model.body_dofnum[bid] == 0, f"{side}_{segment} has DoFs"


def test_segment_chain_is_nested_correctly(stilt_model):
  model, _ = stilt_model
  expected_parent = {
    "stilt_mount": "ankle_roll_link",
    "stilt_brace": "stilt_mount",
    "stilt_post_outer": "stilt_mount",
    "stilt_post_inner": "stilt_post_outer",
    "stilt_plate": "stilt_post_inner",
  }
  for side in SIDES:
    for child, parent in expected_parent.items():
      bid = _body_id(model, f"{side}_{child}")
      pid = model.body_parentid[bid]
      assert mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, pid) == (
        f"{side}_{parent}"
      ), f"{side}_{child}"


def test_stilt_mass_per_side(stilt_model):
  model, _ = stilt_model
  for side in SIDES:
    total = sum(model.body_mass[_body_id(model, f"{side}_{s}")] for s in SEGMENT_BODIES)
    assert total == pytest.approx(2.8, abs=0.01), side


def test_tip_sites_at_ground_contact(stilt_model):
  model, data = stilt_model
  for side in SIDES:
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_stilt_tip")
    assert sid >= 0
    ankle = _body_id(model, f"{side}_ankle_roll_link")
    relative_z = data.site_xpos[sid][2] - data.xpos[ankle][2]
    assert relative_z == pytest.approx(-0.4425, abs=1e-3), side


def test_contact_capsule_names_are_preserved(stilt_model):
  """_STILT_GEOM_NAMES, foot_friction DR and STILT_G1_COLLISION match on these."""
  model, _ = stilt_model
  names = {
    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) for i in range(model.ngeom)
  }
  for side in SIDES:
    for block in ("l", "r"):
      for i in range(1, 5):
        assert f"{side}_stilt_{block}{i}_collision" in names


def test_contact_capsules_reach_the_ground(stilt_model):
  model, data = stilt_model
  for side in SIDES:
    ankle = _body_id(model, f"{side}_ankle_roll_link")
    for block in ("l", "r"):
      for i in range(1, 5):
        gid = mujoco.mj_name2id(
          model, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_stilt_{block}{i}_collision"
        )
        bottom = data.geom_xpos[gid][2] - model.geom_size[gid][0]
        assert bottom - data.xpos[ankle][2] == pytest.approx(-0.4425, abs=1e-3)
