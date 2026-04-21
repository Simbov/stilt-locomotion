"""Stilt G1 velocity task — registers with mjlab task registry."""

from __future__ import annotations

import math
import threading
import time

import torch
import viser

from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import RecomputeLevel
from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner
from mjlab.viewer.base import EnvProtocol

from .env_cfgs import stilt_g1_flat_env_cfg
from .rl_cfg import stilt_g1_ppo_runner_cfg

# Default stilt mass from the MJCF (body_mass = 1.5 kg per stilt).
_NOMINAL_MASS_KG = 1.5
# Full curriculum alpha range: 0.5 kg (α=-0.55) to 6.0 kg (α=0.69).
_ALPHA_MIN, _ALPHA_MAX = -0.55, 0.69
# Upper bound of the viewer mass slider — set above the training range so
# out-of-distribution robustness can be probed interactively.
MASS_PLAY_MAX_KG: float = 20.0

# Actuator effort limits (Nm).
_LIMIT = {
    "ankle": 50.0,  # 2× 5020 motors
    "hip": 88.0,  # 7520-14
    "knee": 139.0,  # 7520-22
}

_MONITOR_JOINTS: list[tuple[str, str, float]] = [
    # (joint_name, display_name, limit_Nm)
    ("left_hip_pitch_joint", "L hip pitch ", _LIMIT["hip"]),
    ("right_hip_pitch_joint", "R hip pitch ", _LIMIT["hip"]),
    ("left_knee_joint", "L knee      ", _LIMIT["knee"]),
    ("right_knee_joint", "R knee      ", _LIMIT["knee"]),
    ("left_ankle_pitch_joint", "L ank pitch ", _LIMIT["ankle"]),
    ("right_ankle_pitch_joint", "R ank pitch ", _LIMIT["ankle"]),
]


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

    # Find the stilt body index for the mass readback sanity check.
    body_names = list(robot.body_names)
    try:
        left_stilt_local_id = body_names.index("left_stilt")
    except ValueError:
        left_stilt_local_id = None

    with server.gui.add_folder("Stilt Mass"):
        mass_slider = server.gui.add_slider(
            "Mass (kg)",
            min=0.0,
            max=MASS_PLAY_MAX_KG,
            step=0.05,
            initial_value=_NOMINAL_MASS_KG,
        )
        randomize_cb = server.gui.add_checkbox(
            "Randomize on reset",
            initial_value=False,
        )
        mass_readback = server.gui.add_markdown("*sim mass: —*")

        def _apply(alpha_range: tuple[float, float]) -> None:
            # Retrieve sim lock from env if available (set by ViserPlayViewer).
            import contextlib
            sim_lock = getattr(raw_env, "sim_lock", contextlib.nullcontext())

            with sim_lock:
                term_cfg.params["alpha_range"] = alpha_range
                all_ids = torch.arange(
                    raw_env.num_envs, dtype=torch.int64, device=raw_env.device
                )
                dr.pseudo_inertia(
                    raw_env,
                    all_ids,
                    alpha_range=alpha_range,
                    asset_cfg=term_cfg.params["asset_cfg"],
                )
                # Recompute model constants (body_mass -> cinert).
                raw_env.sim.recompute_constants(RecomputeLevel.set_const)
                # Recompute dynamics (cinert -> qfrc_bias). Essential for the
                # Joint Torque monitor to update while the viewer is paused.
                raw_env.sim.forward()
                
                # Force a viewer refresh (if running in a viewer).
                if hasattr(raw_env, "sim_scene"):
                    raw_env.sim_scene.request_update()

            # Readback: confirm write landed in Warp model AND propagated to cinert.
            # cinert[body, 9] is the mass component of composite inertia — recomputed
            # by set_const_0 → smooth.com_pos → _cinert kernel. If it matches
            # body_mass, the mass change has reached the dynamics path.
            if left_stilt_local_id is not None:
                try:
                    global_body_id = int(
                        robot.indexing.body_ids[left_stilt_local_id].item()
                    )
                    sim_mass = float(
                        raw_env.sim.model.body_mass[0, global_body_id].item()
                    )
                    cinert_mass = float(
                        raw_env.sim.data.cinert[0, global_body_id, 9].item()
                    )
                    mass_readback.content = (
                        f"*sim mass: **{sim_mass:.3f} kg** "
                        f"| cinert\\[9\\]: **{cinert_mass:.3f} kg***"
                    )
                except Exception as e:
                    mass_readback.content = f"*readback failed: {e}*"

        def _current_alpha_range() -> tuple[float, float]:
            if randomize_cb.value:
                return (_ALPHA_MIN, _ALPHA_MAX)
            mass = max(mass_slider.value, 1e-6)
            alpha = 0.5 * math.log(mass / _NOMINAL_MASS_KG)
            return (alpha, alpha)

        @mass_slider.on_update
        def _(_) -> None:
            if not randomize_cb.value:
                _apply(_current_alpha_range())

        @randomize_cb.on_update
        def _(_) -> None:
            _apply(_current_alpha_range())

    # --- Joint torque monitor: hip, knee, ankle ---
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
            time.sleep(0.1)

    threading.Thread(target=_poll, daemon=True).start()


def _bias_text(entries: list[tuple[int, str, float]], biases: list[float]) -> str:
    lines: list[str] = []
    for (_, name, limit), b in zip(entries, biases):
        pct = abs(b) / limit * 100
        filled = int(pct / 5)
        bar = "█" * filled + "░" * (20 - filled)
        lines.append(f"`{name}` `{b:+7.1f}Nm` {pct:4.0f}%  {bar}")
    return "\n\n".join(lines)


def _torque_text(entries: list[tuple[int, str, float]], torques: list[float]) -> str:
    lines: list[str] = []
    for (_, name, limit), t in zip(entries, torques):
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
