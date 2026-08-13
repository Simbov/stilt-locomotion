"""Stilt G1 velocity task — registers with mjlab task registry."""

from __future__ import annotations

import math
import threading
import time

import torch
import viser
from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import RecomputeLevel
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner
from mjlab.viewer.base import EnvProtocol

from .env_cfgs import stilt_g1_flat_env_cfg
from .loads import SECTIONS, section_loads_from_sensor, sensor_capsule_forces
from .rl_cfg import stilt_g1_ppo_runner_cfg

# Per-segment baseline mass from the MJCF, summing to 2.8 kg per stilt.
_SEGMENT_BASELINE_KG: dict[str, float] = {
  "stilt_mount": 0.6358,
  "stilt_brace": 0.9992,
  "stilt_post_outer": 0.3125,
  "stilt_post_inner": 0.2881,
  "stilt_plate": 0.5645,
}
_NOMINAL_MASS_KG = sum(_SEGMENT_BASELINE_KG.values())

# Full curriculum alpha range: 0.9 kg (α=-0.55) to 7.6 kg (α=0.5) per stilt.
_ALPHA_MIN, _ALPHA_MAX = -0.55, 0.5
# Upper bound of each per-segment mass slider — set well above the training
# range so out-of-distribution robustness can be probed interactively.
MASS_PLAY_MAX_KG: float = 8.0

# Denominators for the load bars (N for forces, Nm for moments). These are
# display scales, not engineering limits — adjust to taste.
SECTION_LIMIT = {
  "axial": 1500.0,
  "shear": 500.0,
  "bending": 150.0,
  "torsion": 50.0,
}
# Bar scale for per-capsule ground pressure (N).
CAPSULE_LIMIT_N: float = 400.0

# Actuator effort limits (Nm).
_LIMIT = {
  "hip": 88.0,  # 7520-14
  "knee": 139.0,  # 7520-22
  "ankle": 50.0,  # 5020, both pitch and roll
}

_MONITOR_JOINTS: list[tuple[str, str, float]] = [
  # (joint_name, display_name, limit_Nm)
  ("left_hip_pitch_joint", "L hip pitch ", _LIMIT["hip"]),
  ("right_hip_pitch_joint", "R hip pitch ", _LIMIT["hip"]),
  ("left_knee_joint", "L knee      ", _LIMIT["knee"]),
  ("right_knee_joint", "R knee      ", _LIMIT["knee"]),
  # The ankles are back and actuated. Worth watching whenever the stilts are
  # fitted: the brace spring is what they are working against, and the clamp
  # stiffness is a guess, so this is where an over-stiff brace shows up first.
  ("left_ankle_pitch_joint", "L ankle pit ", _LIMIT["ankle"]),
  ("right_ankle_pitch_joint", "R ankle pit ", _LIMIT["ankle"]),
  ("left_ankle_roll_joint", "L ankle roll", _LIMIT["ankle"]),
  ("right_ankle_roll_joint", "R ankle roll", _LIMIT["ankle"]),
]

_SEGMENT_LABEL = {
  "stilt_plate": "plate     ",
  "stilt_post_inner": "post inner",
  "stilt_post_outer": "post outer",
  "stilt_mount": "mount     ",
  "stilt_brace": "brace     ",
}


def _stilt_mass_play_gui(server: viser.ViserServer, env: EnvProtocol) -> None:
  """Add Stilt Mass and joint torque monitor folders to the Controls tab."""
  raw_env = env.unwrapped
  try:
    term_cfg = raw_env.event_manager.get_term_cfg("stilt_mass")
  except (AttributeError, ValueError):
    return

  robot = raw_env.scene["robot"]
  joint_names = list(robot.joint_names)

  # Resolve joint indices; skip any that don't exist.
  monitor_entries: list[tuple[int, str, float]] = []
  for jname, display, limit in _MONITOR_JOINTS:
    try:
      monitor_entries.append((joint_names.index(jname), display, limit))
    except ValueError:
      pass

  body_names = [b.name.split("/")[-1] for b in robot.indexing.bodies]

  def _global_body_id(name: str) -> int | None:
    try:
      return int(robot.indexing.body_ids[body_names.index(name)].item())
    except (ValueError, IndexError):
      return None

  # --- Stilts fitted or not ---
  # The policy is trained on a 50/50 draw, so by default a single-env viewer
  # would flip morphology at random on every reset, which makes it impossible to
  # watch either one. This pins the draw. It takes effect at the NEXT reset
  # rather than immediately: mass, contact geometry, tip sites, ankle stiffness,
  # standing pose and root height all have to change together, and the reset
  # event is the one place that does all six consistently.
  try:
    fitted_cfg = raw_env.event_manager.get_term_cfg("stilts_fitted")
  except (AttributeError, ValueError):
    fitted_cfg = None

  fitted_readback = None
  if fitted_cfg is not None:
    with server.gui.add_folder("Stilts"):
      fitted_mode = server.gui.add_dropdown(
        "fitted",
        ("always on", "always off", "randomised 50/50"),
        initial_value="always on",
      )
      fitted_readback = server.gui.add_markdown("*—*")
      fitted_cfg.params["fitted_probability"] = 1.0

      @fitted_mode.on_update
      def _(_) -> None:
        fitted_cfg.params["fitted_probability"] = {
          "always on": 1.0,
          "always off": 0.0,
          "randomised 50/50": 0.5,
        }[fitted_mode.value]

  # --- Per-segment stilt mass, editable live ---
  with server.gui.add_folder("Stilt Mass"):
    randomize_cb = server.gui.add_checkbox("Randomize on reset", initial_value=False)
    master_slider = server.gui.add_slider(
      "master ×", min=0.1, max=3.0, step=0.05, initial_value=1.0
    )
    segment_sliders = {
      segment: server.gui.add_slider(
        _SEGMENT_LABEL[segment].strip() + " (kg)",
        min=0.0,
        max=MASS_PLAY_MAX_KG,
        step=0.01,
        initial_value=baseline,
      )
      for segment, baseline in _SEGMENT_BASELINE_KG.items()
    }
    mass_readback = server.gui.add_markdown("*sim mass: —*")

    def _apply_segment(segment: str, target_kg: float) -> None:
      """Set one segment's mass on both stilts, live.

      pseudo_inertia scales mass and inertia together, so alpha is the
      log-scale multiplier that takes the baseline to the requested mass.
      """
      import contextlib

      alpha = 0.5 * math.log(max(target_kg, 1e-6) / _SEGMENT_BASELINE_KG[segment])
      asset_cfg = SceneEntityCfg(
        "robot", body_names=[f"left_{segment}", f"right_{segment}"]
      )
      asset_cfg.resolve(raw_env.scene)

      sim_lock = getattr(raw_env, "sim_lock", contextlib.nullcontext())
      with sim_lock:
        all_ids = torch.arange(
          raw_env.num_envs, dtype=torch.int64, device=raw_env.device
        )
        dr.pseudo_inertia(
          raw_env, all_ids, alpha_range=(alpha, alpha), asset_cfg=asset_cfg
        )
        # requires_model_fields only annotates; recomputation is manual here.
        raw_env.sim.recompute_constants(RecomputeLevel.set_const)
        # cinert -> qfrc_bias, so the torque monitor updates while paused.
        raw_env.sim.forward()
        if hasattr(raw_env, "sim_scene"):
          raw_env.sim_scene.request_update()

    def _refresh_readback() -> None:
      """Show body_mass and cinert[9] per segment; they must agree."""
      try:
        lines = []
        total = 0.0
        for segment in _SEGMENT_BASELINE_KG:
          gid = _global_body_id(f"left_{segment}")
          if gid is None:
            continue
          sim_mass = float(raw_env.sim.model.body_mass[0, gid].item())
          cinert_mass = float(raw_env.sim.data.cinert[0, gid, 9].item())
          total += sim_mass
          lines.append(
            f"`{_SEGMENT_LABEL[segment]}` **{sim_mass:.3f}** / "
            f"cinert **{cinert_mass:.3f}** kg"
          )
        lines.append(f"`total     ` **{total:.3f} kg** per stilt")
        mass_readback.content = "\n\n".join(lines)
      except Exception as e:
        mass_readback.content = f"*readback failed: {e}*"

    def _apply_all() -> None:
      if randomize_cb.value:
        # Hand control back to the reset event over the full trained range.
        term_cfg.params["alpha_range"] = (_ALPHA_MIN, _ALPHA_MAX)
        return
      term_cfg.params["alpha_range"] = (0.0, 0.0)
      for segment, slider in segment_sliders.items():
        _apply_segment(segment, slider.value * master_slider.value)
      _refresh_readback()

    for _slider in (*segment_sliders.values(), master_slider):

      @_slider.on_update
      def _(_) -> None:
        _apply_all()

    @randomize_cb.on_update
    def _(_) -> None:
      _apply_all()

    # Populate the readback immediately; otherwise it reads "sim mass: —" until
    # the first slider move, which looks like the panel is broken.
    _refresh_readback()

  # --- Stilt section loads ---
  with server.gui.add_folder("Stilt Loads"):
    server.gui.add_markdown(
      "*brace row is inertial load only — **not** the clamp reaction. "
      "The split between the sole bolts and the shank clamp is statically "
      "indeterminate; the sim gives the total wrench only.*"
    )
    loads_md = server.gui.add_markdown("*—*")

  # --- Ground pressure distribution ---
  with server.gui.add_folder("Ground Pressure"):
    pressure_md = server.gui.add_markdown("*—*")

  # --- Joint torque monitor: hip, knee ---
  with server.gui.add_folder("Joint Torques"):
    torque_md = server.gui.add_markdown(
      _torque_text(monitor_entries, [0.0] * len(monitor_entries))
    )
    # qfrc_bias = gravity + Coriolis forces, computed by RNE from cinert.
    # Unlike qfrc_actuator (PD output), this is DIRECTLY proportional to
    # stilt mass and changes immediately when the slider moves.
    server.gui.add_markdown("**Gravity load (qfrc\\_bias):**")
    bias_md = server.gui.add_markdown(
      _bias_text(monitor_entries, [0.0] * len(monitor_entries))
    )

  def _poll() -> None:
    import contextlib

    sim_lock = getattr(raw_env, "sim_lock", contextlib.nullcontext())
    v_adr = robot.indexing.joint_v_adr
    while True:
      try:
        with sim_lock:
          qfrc = robot.data.qfrc_actuator  # (num_envs, num_joints)
          torques = [float(qfrc[0, idx].item()) for idx, _, _ in monitor_entries]

          # qfrc_bias: mass-dependent gravity/Coriolis forces in joint space.
          # Changes visibly with stilt mass even if policy adapts its gait.
          qfrc_b = raw_env.sim.data.qfrc_bias[:, v_adr]
          biases = [float(qfrc_b[0, idx].item()) for idx, _, _ in monitor_entries]

        torque_md.content = _torque_text(monitor_entries, torques)
        bias_md.content = _bias_text(monitor_entries, biases)
      except Exception:
        pass

      # Load panels are read outside the sim lock: the sensor tensors are
      # already materialised, and a stale frame is harmless at 10 Hz.
      try:
        loads_md.content = _loads_text(raw_env)
        pressure_md.content = _pressure_text(raw_env)
      except Exception as e:
        loads_md.content = f"*loads unavailable: {e}*"

      # What the sim is ACTUALLY running, not what the dropdown asks for — the
      # two differ until the next reset.
      if fitted_readback is not None:
        flag = getattr(raw_env, "stilt_fitted", None)
        state = "—" if flag is None else ("**ON**" if flag[0] > 0.5 else "**OFF**")
        fitted_readback.content = f"stilts currently {state} in the sim"

      time.sleep(0.1)

  threading.Thread(target=_poll, daemon=True).start()


def _bar(value: float, limit: float, width: int = 14) -> str:
  filled = min(width, int(abs(value) / limit * width))
  return "█" * filled + "░" * (width - filled)


def _loads_text(raw_env) -> str:
  """Section loads for both stilts, ground-up."""
  lines: list[str] = []
  for side in ("left", "right"):
    loads = section_loads_from_sensor(raw_env, side)
    lines.append(f"**{side}**")
    for section in SECTIONS:
      load = loads[section]
      lines.append(
        f"`{_SEGMENT_LABEL[section]}` "
        f"`ax {load.axial:6.0f}N` `sh {load.shear:5.0f}N` "
        f"`bd {load.bending:6.1f}Nm` {_bar(load.bending, SECTION_LIMIT['bending'])}"
      )
  return "\n\n".join(lines)


def _pressure_text(raw_env) -> str:
  """Per-capsule ground reaction, heel (l1) to toe (r4)."""
  lines: list[str] = []
  for side in ("left", "right"):
    forces = sensor_capsule_forces(raw_env, side)
    total = sum(forces.values())
    lines.append(f"**{side}**  total `{total:6.0f} N`")
    for name, value in forces.items():
      lines.append(f"`{name:>2}` `{value:6.0f}N` {_bar(value, CAPSULE_LIMIT_N)}")
  return "\n\n".join(lines)


def _bias_text(entries: list[tuple[int, str, float]], biases: list[float]) -> str:
  lines: list[str] = []
  for (_, name, limit), b in zip(entries, biases, strict=True):
    pct = abs(b) / limit * 100
    filled = int(pct / 5)
    bar = "█" * filled + "░" * (20 - filled)
    lines.append(f"`{name}` `{b:+7.1f}Nm` {pct:4.0f}%  {bar}")
  return "\n\n".join(lines)


def _torque_text(entries: list[tuple[int, str, float]], torques: list[float]) -> str:
  lines: list[str] = []
  for (_, name, limit), t in zip(entries, torques, strict=True):
    pct = abs(t) / limit * 100
    filled = int(pct / 5)
    bar = "█" * filled + "░" * (20 - filled)
    lines.append(f"`{name}` `{t:+7.1f}Nm` {pct:4.0f}%  {bar}")
  return "\n\n".join(lines)


register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Stilt-G1",
  env_cfg=stilt_g1_flat_env_cfg(),
  play_env_cfg=stilt_g1_flat_env_cfg(play=True),
  rl_cfg=stilt_g1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
