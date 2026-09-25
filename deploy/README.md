# G1 hardware deployment — reference

How the deployed policy actually runs, and why the pieces are arranged the way
they are. For the step-by-step, use
[`BRINGUP_CHECKLIST.md`](BRINGUP_CHECKLIST.md).

The runtime is [unitree_rl_mjlab](https://github.com/unitreerobotics/unitree_rl_mjlab)'s
C++ binary `g1_ctrl`, driving ONNX Runtime 1.22.0 at 50 Hz. mjlab exports the
`.onnx` at the end of every training run.

**Field history:** first successful walk 2026-05-09 (stock G1 policy). First
stilt-policy session 2026-08-31 — the deployment path worked, the policy did
not; see [`docs/RUN_RESULTS.md`](../docs/RUN_RESULTS.md).

---

## The robot

| | |
|---|---|
| OS / kernel | Ubuntu 20.04.6, Linux 5.10.104-tegra (Jetson Orin), `aarch64` |
| SSH | `unitree@192.168.123.164`, password `123` (Unitree factory default) |
| Interface | `eth0`; **no DHCP** — give your laptop a static `192.168.123.222/24` |
| Internet | none — everything is transferred from the laptop |
| Prereqs present | `libyaml-cpp-dev`, `libeigen3-dev`, `libboost-program-options-dev`, cmake 4.2.1, gcc 9.4 |
| Body velocity sensor | **none.** See below — this shaped the policy design |

---

## Build: never install anything system-wide

`scripts/prepare_runtime.py` patches a clean copy of the runtime **on the
laptop** and `scripts/ship_to_robot.sh --runtime` sends it. Three patches, all
required, none touching the robot's system directories:

1. **KeyBase shim** — the robot's SDK has `Button<T>` and `Axis` with no common
   base; unitree_rl_mjlab expects a newer SDK that has `KeyBase`.
2. **Drop `fmt`** from `link_libraries` — listed, unused, not installed.
3. **Point the build at `~/unitree_sdk2`** rather than `/usr/local`.

### Why #3, and why the old instructions were wrong

The 2026-05 procedure said to `sudo make install` a newer `unitree_sdk2` into
`/usr/local`. That fixes the symptom and creates a much worse problem: these are
shared lab machines whose other software is built against what is already
there.

The symptom was dozens of `get_type_props` undefined references — the
`/usr/local` SDK predates the G1's `unitree_hg` DDS types. It still does:

| | `unitree_hg` symbols | `include/unitree/idl` |
|---|---|---|
| `/usr/local/lib/libunitree_sdk2.a` | **0** | `go2`, `ros2` |
| `~/unitree_sdk2/lib/aarch64/libunitree_sdk2.a` | **95** | `go2`, `hg`, `hg_doubleimu`, `ros2` |

The home-directory SDK already has everything, including a matching bundled
cyclonedds under `thirdparty/`. So the patch adds `include_directories(BEFORE)`,
`link_directories(BEFORE)` and an RPATH pointing there. No sudo, nothing
installed, no lab software touched. Verify after building:

```sh
readelf -d g1_ctrl | grep RUNPATH     # -> ~/unitree_sdk2/thirdparty/lib/aarch64
ldd g1_ctrl | grep -E "ddsc|onnx"     # -> must NOT be /usr/local
```

CMake 4.2.1 configures this fine. The `cmake_minimum_required(VERSION 3.0)` in
`thirdparty/cnpy` is inert — cnpy is globbed as sources, never
`add_subdirectory`'d.

---

## The lowcmd channel

The robot's built-in `master_service` publishes motor commands continuously.
`g1_ctrl` detects that at startup:

```
[critical] The other process is using the lowcmd channel, please close it first.
```

…and then **continues anyway** — upstream commented out the `exit(0)`. Two
controllers then write to the motors at once. Put the robot into damping/debug
mode from the remote first; the absence of that log line is the go/no-go test.
Don't kill `master_service` (needs sudo, not a systemd unit, may need a reboot
to restore).

---

## `deploy.yaml`

**Generated, never hand-edited.** The gains, action scales and standing pose all
come out of the trained policy's ONNX metadata:

```sh
uv run python scripts/generate_deploy_config.py --run logs/rsl_rl/stilt_g1_velocity/<run>
```

That also writes `reference_io.json` — recorded (observation, action) pairs from
the sim, which `scripts/verify_deploy_io.py` replays through the shipped `.onnx`.

### Four rules the C++ parser enforces, none of them obvious

Each of these was a startup-fatal bug at some point. All are handled by the
generator and pinned by `tests/test_deploy_config.py`.

1. **`commands:` must be keyed `base_velocity`.** The `velocity_commands` term
   hardcodes `cfg["commands"]["base_velocity"]["ranges"]` — mjlab's own command
   name (`twist`) leaves the lookup undefined and it throws at the first step.
2. **Every observation term needs a `params:` key, even `{}`.**
   `ObservationManager::_prapare_terms` decides single-group vs multi-group by
   probing `cfg.begin()->second["params"].IsDefined()`. Without it the whole
   block parses as a map of *groups*, each term name becomes a group name, and
   startup throws `Observation term 'scale' is not registered.`
3. **The ONNX input tensor must be named `obs`.** `OrtRunner::act` looks the
   group up by the graph's input name. The single-group path names it `obs`;
   mjlab exports `obs`. If a future export renames it, rename the group.
4. **Leave `use_gym_history` unset.** See the layout section below.

### Observation term names

The YAML keys must match the runtime's `REGISTER_OBSERVATION` macros, which are
*not* the same as mjlab's names. The generator maps them:

| mjlab name | runtime key | computed from |
|---|---|---|
| `base_ang_vel` | `base_ang_vel` | IMU gyroscope |
| `projected_gravity` | `projected_gravity` | IMU quaternion |
| `joint_pos` | `joint_pos_rel` | encoder position − `default_joint_pos` |
| `joint_vel` | `joint_vel_rel` | encoder velocity |
| `actions` | `last_action` | previous raw policy output |
| `command` | `velocity_commands` | joystick, clamped to `commands.base_velocity.ranges` |

All scales are 1.0 — normalisation is baked into the ONNX.

### History layout — term-major, oldest first

The 480-input vector is **not** five stacked frames. Each term contributes all
five of its frames contiguously:

| offset | term | layout |
|---|---|---|
| 0:15 | `base_ang_vel` | 5 × 3 |
| 15:30 | `projected_gravity` | 5 × 3 |
| 30:175 | `joint_pos_rel` | 5 × 29 |
| 175:320 | `joint_vel_rel` | 5 × 29 |
| 320:465 | `last_action` | 5 × 29 |
| 465:480 | `velocity_commands` | 5 × 3 |

Index 0 within each block is the **oldest** frame.

**The stock runtime already produces exactly this** — no C++ change is needed
for history. `ObservationTermCfg::get()` concatenates the term's whole deque
(oldest at the front) and `compute_group` walks terms in YAML order. Just set
`history_length: 5` per term and leave `use_gym_history` unset; setting it true
switches to a frame-major layout, and the result loads, runs, and walks badly
with no error at all.

---

## `base_lin_vel`: why it is gone

**The G1 has no body-velocity sensor.** `unitree_hg` `LowState_` carries
`imu_state_` (quaternion, gyro, accelerometer) and motor states. Nothing else.

Runs up to and including Run 8 had `base_lin_vel` in the actor observation, and
the deploy runtime zero-filled it with a patch to `State_RLBase.cpp`. That
looked harmless and was not: the policy had learned to use that term to notice
and correct its own drift, so on hardware — permanently told it was stationary —
it walked forward at a zero command and would not track. Reproduced in sim by
zeroing the same slice.

**From Run 9 the term is critic-only.** It does not appear in the deployed
observation, there is nothing to zero-fill, and the `State_RLBase.cpp` patch is
no longer needed. The policy infers its motion from the 5 frames of history it
already gets.

The general rule, now pinned by `tests/test_env_wiring.py`:

> **Anything in the actor observation must be measurable on the real robot.**
> The critic may use whatever privileged state it likes — it is training-only.

---

## Ankles, and the two morphologies

**The robot is always the stock 29-DoF G1, and all four ankle motors stay in
normal PD position mode**, stilts on or off. The action vector is 29. The brace
stiffens the ankle mechanically, and the policy is trained against exactly that
(ankle stiffness randomised 10–500 Nm/rad in the fitted half of training from Run 10; 150–2000 up to Run 9).
Putting the motors in damping would remove authority the policy is counting on.

One policy handles both morphologies and **is not told which** — it infers from
observation history. **No configuration changes between them.** Fit the stilts
or don't; expect the first few steps after a change to be the shakiest.

Watch ankle motor temperature on early stilted runs. The clamp stiffness has
never been measured and the training range is an engineering guess; if the
motors run hot, measure it and retrain with the range corrected rather than
changing the control mode.

---

## Joystick

| Combo | Transition |
|---|---|
| `L2 + D-pad Up` | Passive → FixStand |
| `R2 + A` | FixStand → Velocity (policy runs) |
| `L2 + B` | any → Passive (**damping — the robot drops if unsupported**) |
| `D-pad Down` | any → Passive (same as `L2 + B`, one thumb; added by `prepare_runtime.py`) |

Left stick forward = `+vx`, left stick left = `+vy`, right stick left = `+yaw`.
Commands are clamped to the trained range, so full deflection is not an
overspeed risk.

`g1_ctrl -n eth0`. There is no `--config` flag; `--help` prints and then aborts.

---

## Files

| Path | Purpose |
|---|---|
| `scripts/prepare_runtime.py` | patch a clean runtime checkout, on the laptop |
| `scripts/ship_to_robot.sh` | checksum + transfer, one command |
| `scripts/generate_deploy_config.py` | `deploy.yaml` + `reference_io.json` from a run |
| `scripts/verify_deploy_io.py` | replay golden vectors through the shipped ONNX |
| `deploy/outbox/*.sh` | robot-side identify / install / pose-match |
| `deploy/config/g1_stilt/` | the generated config and golden vectors |
