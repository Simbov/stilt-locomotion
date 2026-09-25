#!/usr/bin/env python3
"""Turn a stock unitree_rl_mjlab checkout into the tree we ship to the G1.

    uv run python scripts/prepare_runtime.py            # -> build/runtime/
    uv run python scripts/prepare_runtime.py --probe    # + observation probe
    uv run python scripts/prepare_runtime.py --check    # report, change nothing

Patch it HERE, not on the robot: the robot is air-gapped, has no editor
(`nano` is not installed), and its clock is wrong, so every fix applied over
SSH is slow and hard to verify. This produces a ship-ready `deploy/` tree that
`scripts/ship_to_robot.sh` sends as one tarball.

The patches, all of which upstream needs and none of which touch the robot's
system directories:

1. KeyBase shim — the G1's SDK has `Button<T>` and `Axis` with no common base;
   unitree_rl_mjlab was written against a newer SDK that has `KeyBase`.
2. Drop `fmt` from link_libraries — listed but unused, and not installed.
3. Point the build at ~/unitree_sdk2 rather than the /usr/local install.
   THIS IS THE ONE THAT MATTERS. See deploy/README.md: the installed SDK
   predates the G1 `unitree_hg` DDS types, and `sudo make install`-ing a newer
   one over it (as the May instructions said) would break the lab's own
   software. The home-directory SDK already has them.
4. `--probe` only: print the newest frame of three observation terms at 2 Hz
   for the first 60 s of Velocity, then go silent. Use it on a robot you have
   not deployed to before, to prove the IMU and encoders land in the right
   slots. Indices are derived from the config, not hardcoded.
5. Telemetry, on unless `--no-telemetry`: one CSV row per policy step with
   timing, IMU, command, and per joint q, dq, torque estimate, temperature, PD
   target and raw action, written off the control thread to ~/telemetry/.
   Source is deploy/patches/telemetry.h; read it back with
   scripts/analyze_hardware_log.py. The stock runtime logs only FSM
   transitions, which could not quantify the 2026-09-14 on-the-spot creep.
6. Quick damp: D-pad Down alone drops to Passive from any state, alongside
   the stock L2+B chord, which proved too slow to hit mid-fall.

`base_lin_vel` deliberately has NO patch here. Run 8 needed a zero-fill stub
because its actor observation included a term the G1 cannot measure; that is
exactly what broke it on hardware. From Run 9 the term is critic-only and does
not appear in the deployed observation at all.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "unitree_rl_mjlab" / "deploy"
OUT_DEFAULT = ROOT / "build" / "runtime"

# Excluded from the shipped tree: the robot is aarch64, so the x64 runtime is
# 20 MB of dead weight.
EXCLUDE = {"onnxruntime-linux-x64-1.22.0"}

SDK_BLOCK = """
# --- Local SDK wiring (added by scripts/prepare_runtime.py) ---------------
# The system-installed SDK in /usr/local predates the G1 `unitree_hg` DDS
# types: /usr/local/include/unitree/idl holds only go2 and ros2, and
# /usr/local/lib/libunitree_sdk2.a has ZERO unitree_hg symbols. That is what
# produced the `get_type_props` link errors on 2026-05-09.
#
# ~/unitree_sdk2 on the robot is the newer SDK and already carries them (95
# unitree_hg symbols in its prebuilt lib), with a matching bundled cyclonedds
# in thirdparty/. Point at it with BEFORE so it wins over /usr/local.
#
# Do NOT `sudo make install` a newer SDK over /usr/local instead. It is a
# shared lab machine and other software is built against what is there.
set(UNITREE_SDK_ROOT "$ENV{HOME}/unitree_sdk2")
if(NOT EXISTS "${UNITREE_SDK_ROOT}/include/unitree/idl/hg")
  message(FATAL_ERROR
    "No unitree_hg IDL under ${UNITREE_SDK_ROOT}. This robot's home-directory "
    "SDK is too old; see deploy/README.md before installing anything.")
endif()
include_directories(BEFORE
  ${UNITREE_SDK_ROOT}/include
  ${UNITREE_SDK_ROOT}/thirdparty/include
)
link_directories(BEFORE
  ${UNITREE_SDK_ROOT}/lib/aarch64
  ${UNITREE_SDK_ROOT}/thirdparty/lib/aarch64
)
# Load the SDK's own cyclonedds at runtime, not /usr/local's different build.
set(CMAKE_BUILD_RPATH "${UNITREE_SDK_ROOT}/thirdparty/lib/aarch64")
set(CMAKE_INSTALL_RPATH "${UNITREE_SDK_ROOT}/thirdparty/lib/aarch64")
# --------------------------------------------------------------------------
"""

KEYBASE_SHIM = """// KeyBase: compat shim — Button<T> and Axis have no common base in this SDK
struct KeyBase {
    bool pressed = false;
    bool on_pressed = false;
    bool on_released = false;
    float pressed_time = 0.0f;  // stub; hold-time transitions unused here
};
template<typename T>
inline KeyBase make_key_base(const ::unitree::common::Button<T>& b) {
    return {b.pressed, b.on_pressed, b.on_released, 0.0f};
}
inline KeyBase make_key_base(const ::unitree::common::Axis& a) {
    return {a.pressed, a.on_pressed, a.on_released, 0.0f};
}

"""

PROBE = """
        // --- Bring-up probe (scripts/prepare_runtime.py --probe) ----------
        // Newest frame of three terms at 2 Hz for the first 60 s of Velocity,
        // then silent. Offsets are computed from the observation widths in
        // deploy.yaml, so they follow the layout rather than assuming it.
        if (episode_length <= 3000 && episode_length % 25 == 0) {
            const auto it = obs.find("obs");
            if (it != obs.end()) {
                const auto& o = it->second;
                const size_t H = @H@, NJ = @NJ@;
                const size_t ang = 0, grav = 3 * H, jp = 6 * H;
                const size_t cmd = o.size() - 3;   // newest command frame
                if (o.size() >= 6 * H + NJ * H) {
                    std::printf(
                        "[probe %4ld] dim=%zu grav=[%+.2f %+.2f %+.2f] "
                        "ang=[%+.2f %+.2f %+.2f] Ljp=[%+.2f %+.2f %+.2f "
                        "%+.2f %+.2f] CMD=[%+.4f %+.4f %+.4f]\\n",
                        episode_length, o.size(),
                        o[grav + 3*(H-1)], o[grav + 3*(H-1) + 1], o[grav + 3*(H-1) + 2],
                        o[ang  + 3*(H-1)], o[ang  + 3*(H-1) + 1], o[ang  + 3*(H-1) + 2],
                        o[jp + NJ*(H-1)], o[jp + NJ*(H-1) + 1], o[jp + NJ*(H-1) + 2],
                        o[jp + NJ*(H-1) + 3], o[jp + NJ*(H-1) + 4],
                        o[cmd], o[cmd + 1], o[cmd + 2]);
                    std::fflush(stdout);
                }
            }
        }
        // ------------------------------------------------------------------
"""


def patch_keybase(p: Path) -> str:
  c = before = p.read_text()
  c = c.replace(
    "// Retrieve KeyBase from UnitreeJoystick (case-insensitive)\n",
    KEYBASE_SHIM + "// Retrieve KeyBase from UnitreeJoystick (case-insensitive)\n",
  )
  c = c.replace("inline const KeyBase& GetKey(", "inline KeyBase GetKey(")
  c = c.replace(
    "const KeyBase* (*)(const UnitreeJoystick&)", "KeyBase (*)(const UnitreeJoystick&)"
  )
  c = re.sub(
    r"->const KeyBase\*\{ return &static_cast<const KeyBase&>\(j\.(\w+)\); \}",
    r"->KeyBase{ return make_key_base(j.\1); }",
    c,
  )
  c = c.replace("return *it->second(joy);", "return it->second(joy);")
  c = c.replace(
    "const KeyBase& kb = GetKey(joy, a.name);",
    "const KeyBase kb = GetKey(joy, a.name);",
  )
  if c == before:
    raise SystemExit("KeyBase shim: no anchors matched — upstream changed?")
  p.write_text(c)
  return f"KeyBase shim ({c.count('make_key_base(j.')} call sites rewritten)"


def patch_cmake(p: Path) -> list[str]:
  s = p.read_text()
  notes = []
  if "\n  fmt\n" in s:
    s = s.replace("\n  fmt\n", "\n")
    notes.append("dropped unused fmt from link_libraries")
  anchor = "set(CMAKE_CXX_STANDARD 17)\n"
  if anchor not in s:
    raise SystemExit("CMakeLists: CMAKE_CXX_STANDARD anchor missing")
  s = s.replace(anchor, anchor + SDK_BLOCK, 1)
  notes.append("build points at ~/unitree_sdk2 (no sudo, no /usr/local write)")
  p.write_text(s)
  return notes


def patch_probe(p: Path, history: int, n_joints: int) -> str:
  s = p.read_text()
  if "#include <cstdio>" not in s:
    s = s.replace("#include <iostream>", "#include <iostream>\n#include <cstdio>", 1)
  anchor = "        auto obs = observation_manager->compute();\n"
  if anchor not in s:
    raise SystemExit("probe: observation_manager->compute() anchor missing")
  s = s.replace(
    anchor,
    anchor + PROBE.replace("@H@", str(history)).replace("@NJ@", str(n_joints)),
    1,
  )
  p.write_text(s)
  return f"observation probe (history={history}, {n_joints} joints)"


TELEMETRY_H = ROOT / "deploy" / "patches" / "telemetry.h"

TELEMETRY_DATA = """    std::vector<float> joint_ids_map;

    // --- Telemetry (scripts/prepare_runtime.py) --------------------------
    // Policy joint order. Filled by BaseArticulation::update, read by
    // telemetry::record.
    std::vector<float> tlm_tau_est;
    std::vector<int16_t> tlm_motor_temp;
    Eigen::Vector3f tlm_imu_acc = Eigen::Vector3f::Zero();
    uint32_t tlm_tick = 0;
"""

TELEMETRY_UPDATE = """            data.joint_vel[i] = lowstate->msg_.motor_state()[data.joint_ids_map[i]].dq();
        }
        // --- Telemetry (scripts/prepare_runtime.py) ----------------------
        // Still under the lowstate lock taken above.
        {
            const size_t n = data.joint_ids_map.size();
            data.tlm_tau_est.resize(n);
            data.tlm_motor_temp.resize(n);
            for (size_t i = 0; i < n; ++i) {
                const auto& m = lowstate->msg_.motor_state()[static_cast<size_t>(data.joint_ids_map[i])];
                data.tlm_tau_est[i] = m.tau_est();
                data.tlm_motor_temp[i] = m.temperature()[0];
            }
            for (int i = 0; i < 3; ++i) {
                data.tlm_imu_acc[i] = lowstate->msg_.imu_state().accelerometer()[i];
            }
            data.tlm_tick = lowstate->msg_.tick();
        }
"""

TELEMETRY_STEP = """        const auto tlm_t0 = std::chrono::steady_clock::now();
        auto action = alg->act(obs);
        const auto tlm_t1 = std::chrono::steady_clock::now();
        action_manager->process_action(action);
        telemetry::record(this, obs, action,
            std::chrono::duration<double, std::milli>(tlm_t1 - tlm_t0).count());
"""


def _replace_once(p: Path, anchor: str, new: str, what: str) -> None:
  s = p.read_text()
  n = s.count(anchor)
  if n != 1:
    raise SystemExit(f"telemetry: {what} anchor matched {n} times in {p.name}")
  p.write_text(s.replace(anchor, new, 1))


def patch_telemetry(out: Path) -> str:
  """Per-step CSV recorder. `out` is the deploy/ tree; only include/ is touched.

  G1 only: the articulation patch reads the unitree_hg MotorState layout, so a
  go2 build from the same tree would not compile.
  """
  shutil.copy(TELEMETRY_H, out / "include" / "telemetry.h")
  _replace_once(
    out / "include/isaaclab/assets/articulation/articulation.h",
    "    std::vector<float> joint_ids_map;\n",
    TELEMETRY_DATA,
    "ArticulationData",
  )
  _replace_once(
    out / "include/unitree_articulation.h",
    "            data.joint_vel[i] = lowstate->msg_.motor_state()"
    "[data.joint_ids_map[i]].dq();\n        }\n",
    TELEMETRY_UPDATE,
    "BaseArticulation::update",
  )
  env_h = out / "include/isaaclab/envs/manager_based_rl_env.h"
  _replace_once(
    env_h,
    '#include "isaaclab/utils/utils.h"\n',
    '#include "isaaclab/utils/utils.h"\n#include <chrono>\n#include "telemetry.h"\n',
    "env include",
  )
  _replace_once(
    env_h,
    "        episode_length = 0;\n",
    "        episode_length = 0;\n        telemetry::Recorder::get().new_segment();\n",
    "env reset",
  )
  _replace_once(
    env_h,
    "        auto action = alg->act(obs);\n        action_manager->process_action(action);\n",
    TELEMETRY_STEP,
    "env step",
  )
  return "telemetry recorder -> ~/telemetry/g1_<time>.csv"


STOCK_KILL = "Passive: LT + B.on_pressed\n"
QUICK_KILL = "Passive: (LT + B.on_pressed) | down.on_pressed\n"


def patch_quick_damp(p: Path) -> str:
  """D-pad Down alone -> Passive, from every state that has the L2+B kill.

  L2+B is a two-hand chord and was too slow to hit when a stilted run started
  going wrong. L2+B still works. D-pad Down is otherwise unbound in g1_ctrl.
  """
  s = p.read_text()
  n = s.count(STOCK_KILL)
  if n == 0:
    raise SystemExit("quick damp: no 'Passive: LT + B.on_pressed' transitions found")
  p.write_text(s.replace(STOCK_KILL, QUICK_KILL))
  return f"D-pad Down -> Passive (damping) in {n} states, alongside L2+B"


def deployed_layout() -> tuple[int, int]:
  """History length and joint count, read from the generated deploy.yaml."""
  import yaml

  cfg = yaml.safe_load((ROOT / "deploy/config/g1_stilt/deploy.yaml").read_text())
  obs = cfg["observations"]
  h = obs["base_ang_vel"]["history_length"]
  nj = len(obs["joint_pos_rel"]["scale"])
  return h, nj


def main() -> int:
  ap = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
  )
  ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
  ap.add_argument(
    "--probe",
    action="store_true",
    help="include the bring-up observation probe (see docstring)",
  )
  ap.add_argument(
    "--no-telemetry",
    action="store_true",
    help="leave out the per-step CSV recorder (see docstring)",
  )
  ap.add_argument(
    "--check", action="store_true", help="report what would happen; change nothing"
  )
  args = ap.parse_args()

  if not SRC.is_dir():
    print(
      f"No runtime source at {SRC}.\n"
      "The unitree_rl_mjlab submodule is not checked out. Run:\n"
      "    git submodule update --init unitree_rl_mjlab",
      file=sys.stderr,
    )
    return 1

  history, n_joints = deployed_layout()
  print(f"source   {SRC}")
  print(f"layout   {history} frames x ({n_joints} joints) from deploy.yaml")
  if args.check:
    print("--check: no files written.")
    return 0

  out = args.out / "deploy"
  if out.exists():
    shutil.rmtree(out)
  out.parent.mkdir(parents=True, exist_ok=True)
  shutil.copytree(SRC, out, ignore=shutil.ignore_patterns(*EXCLUDE))

  notes = [patch_keybase(out / "include/unitree_joystick_dsl.hpp")]
  notes += patch_cmake(out / "robots/g1/CMakeLists.txt")
  notes.append(patch_quick_damp(out / "robots/g1/config/config.yaml"))
  if args.probe:
    notes.append(
      patch_probe(
        out / "include/isaaclab/envs/manager_based_rl_env.h", history, n_joints
      )
    )
  if not args.no_telemetry:
    notes.append(patch_telemetry(out))

  for n in notes:
    print(f"  + {n}")
  size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
  print(f"\nready    {out}  ({size / 1e6:.0f} MB)")
  print("ship it  ./scripts/ship_to_robot.sh")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
