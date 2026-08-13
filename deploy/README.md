# G1 Hardware Deployment

Deploys a trained ONNX policy to the Unitree G1 (29-DOF) using
[unitree_rl_mjlab](https://github.com/unitreerobotics/unitree_rl_mjlab)'s C++ runtime.
The ONNX file is produced automatically by mjlab at the end of every training run.

This README documents what was actually done to get the g1_velocity policy running
on the physical robot (first successful test: 2026-05-09).

---

## Robot hardware facts (confirmed in field)

| Item | Value |
|---|---|
| OS | Ubuntu 20.04.6 LTS |
| Kernel | Linux 5.10.104-tegra (Jetson Orin) |
| Architecture | `aarch64` |
| SSH address | `192.168.123.164` (Ethernet) |
| SSH user | `unitree` / password `123` |
| Ethernet interface | `eth0` |
| Internet access | None (air-gapped) — all files must be transferred from laptop |
| Pre-installed SDK | unitree_sdk2 v2.0.0 (too old — see patch notes below) |
| Pre-installed deps | `libyaml-cpp-dev`, `libeigen3-dev`, `libboost-program-options-dev` all present |

---

## Prerequisites

On your **laptop** (macOS or Linux, with internet):
- `git`, `scp`
- `uv run python` with `onnx` package (for verifying ONNX metadata)

On the **robot's onboard computer** (already present on this G1):
- All system deps already installed — no `apt install` needed
- ONNX Runtime 1.22.0 — transfer from laptop (see Step 3)
- unitree_sdk2 — needs updating from v2.0.0 (see Step 5)

---

## One-time setup (first deployment only)

### Step 1 — SSH into the robot

```bash
ssh unitree@192.168.123.164
# password: 123
```

### Step 2 — Clone unitree_rl_mjlab onto the robot

The robot has no internet. Clone on your laptop and transfer:

```bash
# On laptop:
git clone https://github.com/unitreerobotics/unitree_rl_mjlab.git
cd ..
tar czf unitree_rl_mjlab.tar.gz unitree_rl_mjlab/
scp unitree_rl_mjlab.tar.gz unitree@192.168.123.164:~/

# On robot:
tar xzf unitree_rl_mjlab.tar.gz
```

### Step 3 — Transfer ONNX Runtime 1.22.0

```bash
# On laptop:
curl -LO https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/onnxruntime-linux-aarch64-1.22.0.tgz
scp onnxruntime-linux-aarch64-1.22.0.tgz unitree@192.168.123.164:~/

# On robot — the repo already has thirdparty/ set up, just unpack there:
# (onnxruntime was already present in unitree_rl_mjlab/deploy/thirdparty/ on this robot)
```

### Step 4 — Update unitree_sdk2 (SDK v2.0.0 → latest)

The pre-installed SDK v2.0.0 is missing `libunitree_hg_idl_cpp.a` — the compiled
DDS type registrations for G1 (`unitree_hg` message types). Without it, linking fails
with dozens of `get_type_props` undefined reference errors.

```bash
# On laptop:
git clone https://github.com/unitreerobotics/unitree_sdk2.git
cd ..
tar czf unitree_sdk2.tar.gz unitree_sdk2/
scp unitree_sdk2.tar.gz unitree@192.168.123.164:~/

# On robot:
mkdir unitree_sdk2_new
tar xzf unitree_sdk2.tar.gz -C unitree_sdk2_new
cd unitree_sdk2_new/unitree_sdk2
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local
sudo make install -j$(nproc)
```

### Step 5 — Apply three source patches to unitree_rl_mjlab

All three patches are needed. Apply them once; they persist across rebuilds.

#### 5a. Add `base_lin_vel` zero-fill (`State_RLBase.cpp`)

The policy's obs[0:3] is body-frame linear velocity — ground-truth in sim, no
direct sensor on hardware. Add a zero-fill registration in the `isaaclab` namespace
block in `~/unitree_rl_mjlab/deploy/robots/g1/src/State_RLBase.cpp`:

```cpp
REGISTER_OBSERVATION(base_lin_vel)
{
    // No direct body-velocity sensor on hardware — zero-fill.
    // Policy uses this as measured-speed feedback, not commanded speed.
    return std::vector<float>{0.f, 0.f, 0.f};
}
```

Use `cat >` to write since `nano` is not installed:
```bash
cat > ~/unitree_rl_mjlab/deploy/robots/g1/src/State_RLBase.cpp << 'EOF'
#include "FSM/State_RLBase.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include <unordered_map>

namespace isaaclab
{

REGISTER_OBSERVATION(keyboard_velocity_commands)
{
    std::string key = FSMState::keyboard->key();
    static auto cfg = env->cfg["commands"]["base_velocity"]["ranges"];

    static std::unordered_map<std::string, std::vector<float>> key_commands = {
        {"w", {1.0f, 0.0f, 0.0f}},
        {"s", {-1.0f, 0.0f, 0.0f}},
        {"a", {0.0f, 1.0f, 0.0f}},
        {"d", {0.0f, -1.0f, 0.0f}},
        {"q", {0.0f, 0.0f, 1.0f}},
        {"e", {0.0f, 0.0f, -1.0f}}
    };
    std::vector<float> cmd = {0.0f, 0.0f, 0.0f};
    if (key_commands.find(key) != key_commands.end())
    {
        cmd = key_commands[key];
    }
    return cmd;
}

REGISTER_OBSERVATION(base_lin_vel)
{
    // No direct body-velocity sensor on hardware — zero-fill.
    // Policy uses this as measured-speed feedback, not commanded speed.
    return std::vector<float>{0.f, 0.f, 0.f};
}

}

State_RLBase::State_RLBase(int state_mode, std::string state_string)
: FSMState(state_mode, state_string) 
{
    auto cfg = param::config["FSM"][state_string];
    auto policy_dir = param::parser_policy_dir(cfg["policy_dir"].as<std::string>());

    env = std::make_unique<isaaclab::ManagerBasedRLEnv>(
        YAML::LoadFile(policy_dir / "params" / "deploy.yaml"),
        std::make_shared<unitree::BaseArticulation<LowState_t::SharedPtr>>(FSMState::lowstate)
    );
    env->alg = std::make_unique<isaaclab::OrtRunner>(policy_dir / "exported" / "policy.onnx");

    this->registered_checks.emplace_back(
        std::make_pair(
            [&]()->bool{ return isaaclab::mdp::bad_orientation(env.get(), 1.0); },
            FSMStringMap.right.at("Passive")
        )
    );
}

void State_RLBase::run()
{
    auto action = env->action_manager->processed_actions();
    for(int i(0); i < env->robot->data.joint_ids_map.size(); i++) {
        lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].q() = action[i];
    }
}
EOF
```

#### 5b. Add `KeyBase` shim (`unitree_joystick_dsl.hpp`)

SDK v2.0.0's `UnitreeJoystick` has `Button<T>` and `Axis` as separate classes with
no common base. `unitree_rl_mjlab` was written against a newer SDK that has a `KeyBase`
base class. Apply the shim with Python:

```bash
python3 << 'EOF'
import re

path = '/home/unitree/unitree_rl_mjlab/deploy/include/unitree_joystick_dsl.hpp'
with open(path, 'r') as f:
    content = f.read()

shim = """// KeyBase: compat shim — Button<T> and Axis have no common base in this SDK version
struct KeyBase {
    bool pressed = false;
    bool on_pressed = false;
    bool on_released = false;
    float pressed_time = 0.0f;  // stub; hold-time transitions not used in this config
};
template<typename T>
inline KeyBase make_key_base(const ::unitree::common::Button<T>& b) {
    return {b.pressed, b.on_pressed, b.on_released, 0.0f};
}
inline KeyBase make_key_base(const ::unitree::common::Axis& a) {
    return {a.pressed, a.on_pressed, a.on_released, 0.0f};
}

"""

content = content.replace(
    '// Retrieve KeyBase from UnitreeJoystick (case-insensitive)\n',
    shim + '// Retrieve KeyBase from UnitreeJoystick (case-insensitive)\n')
content = content.replace('inline const KeyBase& GetKey(', 'inline KeyBase GetKey(')
content = content.replace(
    'const KeyBase* (*)(const UnitreeJoystick&)',
    'KeyBase (*)(const UnitreeJoystick&)')
content = re.sub(
    r'->const KeyBase\*\{ return &static_cast<const KeyBase&>\(j\.(\w+)\); \}',
    r'->KeyBase{ return make_key_base(j.\1); }',
    content)
content = content.replace('return *it->second(joy);', 'return it->second(joy);')
content = content.replace(
    'const KeyBase& kb = GetKey(joy, a.name);',
    'const KeyBase kb = GetKey(joy, a.name);')

with open(path, 'w') as f:
    f.write(content)

print("Patched successfully")
EOF
```

#### 5c. Remove unused `fmt` dependency (`CMakeLists.txt`)

`libfmt` is listed in `link_libraries` but not used anywhere in the source.
It's not installed on the robot. Remove it:

```bash
python3 -c "
path = '/home/unitree/unitree_rl_mjlab/deploy/robots/g1/CMakeLists.txt'
with open(path) as f: content = f.read()
content = content.replace('  fmt\n', '')
with open(path, 'w') as f: f.write(content)
print('done')
"
```

---

## Per-policy setup

For each new policy you want to deploy, do the following.

### Step 6 — Verify ONNX metadata (on laptop)

```bash
uv run python -c "
import onnx
m = onnx.load('logs/rsl_rl/g1_velocity/<run>/<run>.onnx')
for p in m.metadata_props: print(p.key, ':', p.value)
"
```

Check `observation_names` — the order defines the obs vector. Must match `deploy.yaml`.

### Step 7 — Create policy directory on robot

```bash
# On robot:
mkdir -p ~/unitree_rl_mjlab/deploy/robots/g1/config/policy/velocity/<name>/params
mkdir -p ~/unitree_rl_mjlab/deploy/robots/g1/config/policy/velocity/<name>/exported
```

### Step 8 — Transfer ONNX and deploy.yaml

```bash
# On laptop — transfer ONNX:
RUN=<run_timestamp>
scp logs/rsl_rl/g1_velocity/$RUN/$RUN.onnx \
    unitree@192.168.123.164:~/unitree_rl_mjlab/deploy/robots/g1/config/policy/velocity/<name>/exported/policy.onnx

# Transfer deploy.yaml:
scp deploy/config/g1_velocity/deploy.yaml \
    unitree@192.168.123.164:~/unitree_rl_mjlab/deploy/robots/g1/config/policy/velocity/<name>/params/deploy.yaml
```

### Step 9 — Update config.yaml to point at your policy

```bash
# On robot — edit the Velocity policy_dir:
python3 -c "
path = '/home/unitree/unitree_rl_mjlab/deploy/robots/g1/config/config.yaml'
with open(path) as f: content = f.read()
content = content.replace(
    'policy_dir: config/policy/velocity/simon',
    'policy_dir: config/policy/velocity/<name>')
with open(path, 'w') as f: f.write(content)
print('done')
"
```

### Step 10 — Build

```bash
cd ~/unitree_rl_mjlab/deploy/robots/g1/build
cmake ..
make -j$(nproc)
```

Clock skew warnings (`modification time ... in the future`) are harmless — the robot's
clock is set to epoch (1970). The binary is still valid.

---

## Running on the robot

### Safety checklist (every time)
- [ ] Robot suspended on gantry / harness
- [ ] Correct feet fitted for policy (bare feet for g1_velocity, stilts for stilt policy)
- [ ] Flat, clear floor beneath robot
- [ ] Joystick charged and paired
- [ ] E-stop within reach — zero all sticks to stop motion
- [ ] Second person dedicated to E-stop at all times

### Start sequence

```bash
cd ~/unitree_rl_mjlab/deploy/robots/g1/build
./g1_ctrl -n eth0
```

The terminal will print the policy directory it loaded — verify it says your policy name.

**Joystick transitions** (confirmed on physical robot):

| Button combo | Transition |
|---|---|
| **L2 + D-pad Up** | Passive → FixStand (stands up over ~2 sec) |
| **R2 + A** | FixStand → Velocity (your policy activates) |
| **L2 + B** | Any → Passive (joints go to damping — robot drops if not on gantry) |

After activating Velocity, keep sticks centred for a few seconds and observe
standing stability before commanding motion. Left stick = forward/back/lateral,
right stick = yaw.

---

## deploy.yaml format notes

The runtime registers obs terms by name. Names in the YAML must match what the
C++ `REGISTER_OBSERVATION` macros in `observations.h` define. The registered
names (confirmed from source inspection) are:

| YAML key | What it computes |
|---|---|
| `base_lin_vel` | zero-fill (added via patch) |
| `base_ang_vel` | IMU gyroscope |
| `projected_gravity` | gravity vector from IMU quaternion |
| `joint_pos_rel` | encoder position − `default_joint_pos` |
| `joint_vel_rel` | encoder velocity |
| `last_action` | previous policy output (raw, before scale/offset) |
| `velocity_commands` | joystick vx/vy/yaw, reads from `commands.base_velocity` |

**Important:** the obs order in `deploy.yaml` defines the obs vector order. It must
match training exactly. The policy was trained with:
`base_lin_vel, base_ang_vel, projected_gravity, joint_pos, joint_vel, actions, command`
which maps to the runtime names above.

The command section must use `base_velocity` as the key (not `twist`) because
`velocity_commands` hardcodes `commands.base_velocity.ranges` in the C++ source.

All scales are 1.0 — obs normalisation is baked into the ONNX.

---

## Config values reference

All numerical values in `deploy.yaml` come from ONNX metadata embedded at training.
To regenerate from any checkpoint:

```python
import onnx
m = onnx.load("path/to/policy.onnx")
for p in m.metadata_props:
    print(p.key, ":", p.value)
```

| YAML field | ONNX metadata key | Notes |
|---|---|---|
| `stiffness` | `joint_stiffness` | PD kp per joint |
| `damping` | `joint_damping` | PD kd per joint |
| `default_joint_pos` | `default_joint_pos` | Standing pose + action offset |
| `actions.scale` | `action_scale` | Per-joint action multiplier |
| `actions.offset` | `default_joint_pos` | Same as standing pose |

---

## Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `get_type_props` undefined reference | SDK v2.0.0 missing `libunitree_hg_idl_cpp.a` | Update unitree_sdk2 (Step 4) |
| `KeyBase` does not name a type | SDK `Button`/`Axis` lack common base | Apply `KeyBase` shim (Step 5b) |
| `cannot find -lfmt` | `libfmt` not installed, not needed | Remove from CMakeLists.txt (Step 5c) |
| `unrecognised option '--config'` | Wrong flag — there is no `--config` flag | Use `./g1_ctrl -n eth0` |
| `--help` causes abort | Known quirk — the help flag works but then aborts | Ignore the abort, read the output |
| Clock skew warnings in make | Robot clock set to 1970 | Harmless — binary is valid |
| Policy not loading | Wrong `policy_dir` in `config.yaml` | Check with `grep policy_dir config.yaml` |

---

## Ankle handling (stilt hardware)

**Rewritten 2026-08-13. The previous version of this section said the four ankle
motors must be put in damping mode, and that the action vector is 25. Both are
wrong — do not follow any copy of that instruction.**

The robot is the stock 29-DOF G1 in every configuration. **All four ankle motors
stay in normal PD position mode and are driven by the policy, stilts on or off.**
The action vector is **29**.

The brace does stiffen the ankle when the stilts are bolted on, but it does so
mechanically, and the policy is trained against exactly that: ankle joint
stiffness randomised 150–2000 Nm/rad in the fitted half of the training envs, and
zero in the other half. It has learned to command the ankle into a clamp that may
or may not be there. Putting the motors in damping mode would take away authority
the policy is counting on.

Watch ankle motor temperature on the first stilted runs anyway. The clamp
stiffness is unmeasured — the 150–2000 Nm/rad range is an engineering guess — so
if the real brace is stiffer than the top of that range the motors will do more
static work than training predicted. If they run hot, measure the actual clamp
stiffness and retrain with the range corrected; do not paper over it by changing
the control mode.

### One policy, two morphologies

Run 8 onwards trains a single policy that walks with the stilts fitted and with
them removed. It is **not** told which — it infers the morphology from 5 frames of
observation history. Two things follow for deployment:

- The runtime must buffer 5 frames of observation and feed the policy a 495-dim
  vector. A single-frame runtime will not work with this policy.

  **The layout is per-term, not per-frame.** This is the easy thing to get
  wrong: the vector is *not* five 99-dim frames concatenated. Each observation
  term contributes all five of its frames contiguously, oldest first, in term
  order:

  | offset | term | layout |
  |---|---|---|
  | 0:15 | `base_lin_vel` | 5 frames × 3 |
  | 15:30 | `base_ang_vel` | 5 × 3 |
  | 30:45 | `projected_gravity` | 5 × 3 |
  | 45:190 | `joint_pos` | 5 × 29 |
  | 190:335 | `joint_vel` | 5 × 29 |
  | 335:480 | `actions` | 5 × 29 |
  | 480:495 | `command` | 5 × 3 |

  Within each block, index 0 is the OLDEST frame and index 4 the newest. Getting
  this wrong produces a policy that runs without error and walks badly, so
  verify it against a recorded sim rollout before putting weight on it.
- No configuration switch is needed when the stilts come on or off, and there is
  no "stilt mode" flag to set. Fit them or don't; the policy adapts within a few
  control steps. Expect the first few steps after a change to be the shakiest.

> `deploy/config/g1_stilt/deploy.yaml` is currently **superseded** — it still
> holds Run 5's arrays and is single-frame. Regenerate it from the Run 8 ONNX
> metadata after training; do not hand-edit it.
