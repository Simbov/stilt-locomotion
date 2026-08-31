# Bare-robot bring-up — Run 8 policy

**Goal for this session: get the Run 8 policy standing and walking on the G1
with NO stilts fitted.** Stilts stay in the box. The same policy handles both
morphologies, so nothing in the config changes when they eventually go on — but
the bare robot is the easier half and it is the whole of today.

Policy: `2026-08-13_20-35-42_run8-stilts-on-off`, 495 inputs → 29 actions.
Everything before Run 8 is void — do not deploy an older checkpoint.

Reference for the *why* behind any step: [`deploy/README.md`](README.md).

---

## The fast path (do this; the detail below is the fallback)

Everything is pre-staged in [`deploy/outbox/`](outbox/). Three commands:

```sh
./scripts/ship_to_robot.sh                 # laptop: verify + scp the bundle to ~/run8/
ssh unitree@192.168.123.164                # password: 123
bash ~/run8/00_identify_robot.sh           # same robot? setup done? -> verdict
bash ~/run8/01_install_policy.sh           # install, checksum, sanity-check, point the FSM
cd ~/unitree_rl_mjlab/deploy/robots/g1/build && cmake .. && make -j$(nproc)
```

`00_identify_robot.sh` is read-only and prints a verdict telling you whether to
skip Part 2. `01_install_policy.sh` is idempotent, refuses to run on a bad
checksum, and re-checks the config against the runtime's parsing rules before
touching anything. `02_match_fixstand_pose.sh` is optional (see 3.5) and has a
`--revert`.

Then go to **Part 4** (safety) and **Part 5** (the gates). Parts 1–3 below are
what those scripts do, written out, for when something disagrees.

---

## Part 0 — On the laptop, before you leave

**Done 2026-08-30. Everything below except 0.4 is already green — the results
are recorded here so you do not have to re-run them tomorrow.** Re-run them only
if you touch the config, the policy, or the repo between now and then.

- [x] **0.1 — Tests green.**
  ```sh
  uv run pytest tests/ -q
  ```
  → **65 passed.**

- [x] **0.2 — The ONNX reproduces the sim.** This is the single check that pins
  the 495-dim observation layout end to end.
  ```sh
  uv run python scripts/verify_deploy_io.py \
    --onnx logs/rsl_rl/stilt_g1_velocity/2026-08-13_20-35-42_run8-stilts-on-off/2026-08-13_20-35-42_run8-stilts-on-off.onnx
  ```
  → **PASS, worst 4.768e-06.** Signature `obs [1, 495] -> [1, 29]`. If this ever
  fails, stop — nothing downstream is worth doing.

- [x] **0.3 — Checksums of the two files you are shipping.** Compare these on
  the robot after `scp` (step 3.3):

  | file | md5 | size |
  |---|---|---|
  | `…run8-stilts-on-off.onnx` | `2f881e1bfe107223cdc7a57ed2a60573` | 1 695 005 B |
  | `deploy/config/g1_stilt/deploy.yaml` | `03f07877d380ae969d8ec44b972593eb` | 6 242 B |

  Regenerating `deploy.yaml` changes its md5 — if you do, take a fresh one with
  `md5 -q deploy/config/g1_stilt/deploy.yaml`.

- [ ] **0.4 — Physical, still to do tonight.** Charge the joystick. Charge the
  laptop. Pack the Ethernet cable. Pack the hoist/harness fittings.

- [x] **0.5 — Both files present and current.**
  - `deploy/config/g1_stilt/deploy.yaml` — regenerated 2026-08-30, `commands:`
    keyed `base_velocity`, `params:` on all seven obs terms, 495 obs dims,
    `use_gym_history` absent
  - the `.onnx` above, 495 → 29

**What changed since the last field trip.** Two bugs in `deploy.yaml` were found
by reading the runtime source and fixed today. Both were startup-fatal, so the
symptom would have been an exception at `R2 + A`, not a bad gait:

| was | now | why it mattered |
|---|---|---|
| `commands: twist:` | `commands: base_velocity:` | `velocity_commands` hardcodes `cfg["commands"]["base_velocity"]["ranges"]` |
| obs terms had no `params:` key | `params: {}` on every term | the runtime probes the first term's `params` to tell one group from many; without it each term name is read as a *group* name |

The good news from the same source read: **the stock runtime already supports
the 5-frame history in exactly the layout the policy wants** (term-major, oldest
frame first). No C++ change is needed for it. Just do not set
`use_gym_history: true` — that flips it to frame-major and the robot walks
badly with no error.

---

## Part 1 — Is this the same robot as last time?

**We never wrote down a model name, and the robot does not carry one.** Its
hostname is just `ubuntu`, and nothing in the May session log mentions a
G1-E1/E2/E3 style label — that naming would be the lab's, not the robot's, so if
the machines are labelled on the outside you will have to read the sticker.

What we *do* have is a hardware fingerprint from the May session, which
identifies the exact machine beyond doubt:

| | |
|---|---|
| eth0 MAC | `3c:6d:66:2b:d1:f0` |
| wlan0 MAC | `fc:23:cd:8f:70:79` |
| IP / user | `192.168.123.164`, `unitree` / `123`, iface `eth0` |
| hostname | `ubuntu` (generic — not an identifier) |
| home dirs | `FALCON`, `QCR_G1`, `g1plus_pc4_unitree_install`, `inspire_hand_ws`, `ws_livox`, `xr_teleoperate`, `walking_deployment` |

`QCR_G1` and `g1plus_pc4_unitree_install` are the most distinctive of those —
they suggest the lab's own build, and no stock robot would have them.

Ethernet in, then:

```sh
ssh unitree@192.168.123.164   # password: 123
bash ~/run8/00_identify_robot.sh
```

The script compares the MAC and the home-directory fingerprint, then checks all
six setup markers and prints a verdict. If you would rather do it by hand, this
is the same thing:

```sh
echo "--- binary:"      ; ls -l ~/unitree_rl_mjlab/deploy/robots/g1/build/g1_ctrl 2>&1 | tail -1
echo "--- base_lin_vel:"; grep -c "REGISTER_OBSERVATION(base_lin_vel)" ~/unitree_rl_mjlab/deploy/robots/g1/src/State_RLBase.cpp 2>&1
echo "--- KeyBase shim:"; grep -c "compat shim" ~/unitree_rl_mjlab/deploy/include/unitree_joystick_dsl.hpp 2>&1
echo "--- fmt removed:" ; grep -c "^  fmt$" ~/unitree_rl_mjlab/deploy/robots/g1/CMakeLists.txt 2>&1
echo "--- hg idl lib:"  ; ls -l /usr/local/lib/libunitree_hg_idl_cpp.a 2>&1 | tail -1
echo "--- policies:"    ; ls ~/unitree_rl_mjlab/deploy/robots/g1/config/policy/velocity/ 2>&1
echo "--- policy_dir:"  ; grep -A3 "^  Velocity:" ~/unitree_rl_mjlab/deploy/robots/g1/config/config.yaml | grep policy_dir
```

Read it like this:

| Line | Same robot, ready | Fresh robot |
|---|---|---|
| binary | a `g1_ctrl` file exists | `No such file` |
| base_lin_vel | `1` | `0` or no such file |
| KeyBase shim | `1` | `0` |
| fmt removed | `0` | `1` |
| hg idl lib | the `.a` exists | `No such file` |
| policies | lists `simon` | lists `v0` only |

- **All six rows in the left column → skip Part 2 entirely.** Go to Part 3.
- **Any row in the right column → do Part 2**, but only the steps that row
  covers. They are independent and each is safe to re-run.
- **Mixed / confusing → do all of Part 2.** Every step is idempotent.

> If it is a *different* physical robot, also re-confirm the IP. `192.168.123.164`
> is the address that worked last time; if SSH times out, check the robot's own
> network config before assuming anything is broken.

---

## Part 2 — One-time setup (skip if Part 1 says the robot is ready)

Full detail and rationale in [`deploy/README.md`](README.md) Steps 2–5. Short
form, in order:

- [ ] **2.1** Ship `unitree_rl_mjlab` (the robot is air-gapped — tar it on the
      laptop and `scp`).
- [ ] **2.2** ONNX Runtime 1.22.0 aarch64 into `deploy/thirdparty/`. The tarball
      is already in this repo root: `onnxruntime-linux-aarch64-1.22.0.tgz`.
- [ ] **2.3** Build and install current `unitree_sdk2` — the pre-installed v2.0.0
      lacks `libunitree_hg_idl_cpp.a` and linking fails with dozens of
      `get_type_props` undefined references.
- [ ] **2.4** Patch `State_RLBase.cpp` for the `base_lin_vel` zero-fill:
      ```sh
      cd ~/unitree_rl_mjlab && bash ~/apply_base_lin_vel_stub.sh
      ```
      (`scp deploy/patches/apply_base_lin_vel_stub.sh` across first. It is
      safe to run twice — it exits without change if the stub is there.)
- [ ] **2.5** `KeyBase` shim in `unitree_joystick_dsl.hpp`.
- [ ] **2.6** Drop `fmt` from `deploy/robots/g1/CMakeLists.txt`.

Steps 2.4–2.6 are the three source patches; all three are required and all three
persist across rebuilds.

---

## Part 3 — Install the Run 8 policy

**`bash ~/run8/01_install_policy.sh` does 3.1–3.4 in one go, with the checksum
check and a config sanity-check built in.** The manual steps follow for
reference.


- [ ] **3.1 — Make the policy directory (on the robot):**
  ```sh
  mkdir -p ~/unitree_rl_mjlab/deploy/robots/g1/config/policy/velocity/stilt_run8/{params,exported}
  ```

- [ ] **3.2 — Ship the two files (from the laptop, in the repo root):**
  ```sh
  RUN=2026-08-13_20-35-42_run8-stilts-on-off
  DEST=unitree@192.168.123.164:~/unitree_rl_mjlab/deploy/robots/g1/config/policy/velocity/stilt_run8
  scp logs/rsl_rl/stilt_g1_velocity/$RUN/$RUN.onnx $DEST/exported/policy.onnx
  scp deploy/config/g1_stilt/deploy.yaml           $DEST/params/deploy.yaml
  ```

- [ ] **3.3 — Confirm the transfer (on the robot):**
  ```sh
  md5sum ~/unitree_rl_mjlab/deploy/robots/g1/config/policy/velocity/stilt_run8/exported/policy.onnx
  ```
  Must equal what you wrote down in 0.3.

- [ ] **3.4 — Point the FSM at it (on the robot):**
  ```sh
  python3 -c "import re; p='/home/unitree/unitree_rl_mjlab/deploy/robots/g1/config/config.yaml'; s=open(p).read(); s=re.sub(r'policy_dir: config/policy/velocity(/\S+)?', 'policy_dir: config/policy/velocity/stilt_run8', s, count=1); open(p,'w').write(s); print([l for l in s.splitlines() if 'policy_dir' in l])"
  ```
  Check the printed list: the **Velocity** entry must read
  `config/policy/velocity/stilt_run8`. The Mimic entry is separate and must not
  change.

- [ ] **3.5 — (Optional but recommended) Match the FixStand pose to the policy.**
  `R2 + A` hands over in one control step, and the two poses differ: FixStand
  targets the stock crouch (knee 0.3, ankle_pitch −0.2, elbow 0.87), the policy
  wants the shared stilt pose (knee 0.1, ankle 0, elbow 0.6). The gains step at
  the same instant too. Expect a visible settle. If you want it smooth, edit the
  FixStand `qs` block in `config.yaml` to the `default_joint_pos` line from
  `deploy.yaml`. Robot-side config only — it does not touch the policy.
  ```sh
  bash ~/run8/02_match_fixstand_pose.sh          # --revert undoes it
  ```
  It backs `config.yaml` up first and aborts if the block is not in the shape it
  expects. No rebuild needed — `config.yaml` is read at startup.

- [ ] **3.6 — Build:**
  ```sh
  cd ~/unitree_rl_mjlab/deploy/robots/g1/build && cmake .. && make -j$(nproc)
  ```
  Clock-skew warnings ("modification time in the future") are harmless — the
  robot's clock sits at 1970. If nothing in Part 2 was needed, a rebuild is not
  strictly required for a new policy either, but it costs a minute and removes
  a variable.

---

## Part 4 — Safety, before power

- [ ] Robot on the gantry / hoist, harness taking weight, feet clear of the floor
- [ ] **Bare feet fitted. No stilts anywhere near the robot today.**
- [ ] Floor flat and clear, 3 m in every direction
- [ ] Joystick charged and paired; **`L2 + B` = Passive** is the kill — know it
      by feel, and remember it drops the robot if the hoist is slack
- [ ] Second person on the hoist and on `L2 + B`, doing nothing else
- [ ] Phone recording — the failures are fast and the video is the only record

---

## Part 5 — Bring-up, in order, with gates

Run it under `tee` so the terminal output survives:

```sh
cd ~/unitree_rl_mjlab/deploy/robots/g1/build && ./g1_ctrl -n eth0 2>&1 | tee ~/run8_$(date +%s).log
```

The first lines print the policy directory it loaded. **Verify it says
`stilt_run8` before touching the joystick.** There is no `--config` flag;
`--help` prints and then aborts, which is a known quirk.

| Gate | Action | Pass looks like | Stop if |
|---|---|---|---|
| **5.1** | Binary starts | Loads `stilt_run8`, no exception, no `not registered` error | Any throw at startup — see troubleshooting |
| **5.2** | `L2 + D-pad Up` → FixStand | Stands to the crouch over ~2 s, quiet, no buzzing | Any joint oscillating or hot |
| **5.3** | `R2 + A` → Velocity, **feet still off the floor**, sticks centred | Legs straighten slightly to the policy pose and hold. Small settle at the handover is expected (3.5) | Sustained oscillation, chattering, or a joint driving hard against a limit |
| **5.4** | Lower onto the floor, sticks centred, hoist still attached but slack | Stands. Pelvis ≈ **0.79 m**. Small postural corrections are normal | It sags, creeps down, or steps continuously with zero command |
| **5.5** | Forward, small: left stick to ~¼ | Walks. Roughly 0.2 m/s | Any fall |
| **5.6** | Work up: 0.4, then 0.6, then full | See the table below | |
| **5.7** | Backward, then lateral | Both work, both undershoot | |
| **5.8** | Yaw, several attempts | Turns, weakly and inconsistently — see below | |

**What "correct" looks like.** These are the measured sim numbers for the bare
morphology; hardware will be worse, but the *shape* should match:

| commanded vx | expect achieved |
|---|---|
| 0.2 | ~0.21 |
| 0.4 | ~0.39 |
| 0.6 | ~0.47 (a dip here is expected, not a fault) |
| 0.8 | ~0.56 — it saturates, it does not track |
| −0.4 | ~−0.27 (undershoots ~35%) |
| vy 0.4 | ~0.25 |
| yaw 0.6 | ~0.33, **sd 0.37** |

**Do not read a single yaw episode as a sign error or a broken policy.** The
episode-to-episode spread on yaw is wider than its own mean. Try it five times
before concluding anything.

Stick mapping, from the runtime source: left stick forward = `+vx`, left stick
left = `+vy`, right stick left = `+yaw`. Commands are clamped to the trained
range — `vx [-0.6, 0.8]`, `vy [-0.5, 0.5]`, `yaw [-0.6, 0.6]` — so full
deflection is not an overspeed risk.

**Abort criteria — go to Passive and stop the session:**
- Any motor audibly hot or a temperature warning
- Two falls at the same command
- Any behaviour that changes between identical commands in a way you cannot
  explain

---

## Part 6 — Optional: prove the observations are being filled correctly

`verify_deploy_io.py` proves the model and the layout. It cannot prove the
runtime is putting the right *sensor values* into those slots. If you want that
certainty before the robot takes weight, add a temporary print to
`~/unitree_rl_mjlab/deploy/include/isaaclab/envs/manager_based_rl_env.h`, in
`step()`, right after `auto obs = observation_manager->compute();`:

```cpp
if (episode_length % 50 == 0) {
    const auto& o = obs.at("obs");
    printf("dim=%zu grav=[%.2f %.2f %.2f] ang=[%.2f %.2f %.2f] jp0..3=[%.2f %.2f %.2f %.2f]\n",
           o.size(), o[42], o[43], o[44], o[27], o[28], o[29], o[45+4*29], o[46+4*29], o[47+4*29], o[48+4*29]);
}
```

Add `#include <cstdio>` at the top of that header if it does not compile.

Those indices read the **newest** frame of each block (index 4 of 5). Rebuild,
then hold the robot still in the FixStand pose and read one line:

- `dim` = **495**
- `grav` ≈ `(0, 0, −1)` hanging level; tilt the robot and the vector should tilt
  with it
- `ang` ≈ `(0, 0, 0)` while still
- `jp0..3` is `joint_pos_rel` for left hip pitch/roll/yaw and left knee. In the
  FixStand pose that is **`0, 0, 0, +0.20`** — the knee offset between the two
  poses. Seeing exactly that is a positive test that the joint ordering and the
  `default_joint_pos` subtraction are both right.

**Revert this print and rebuild before the walking gates.** A `printf` at 50 Hz
in the control loop is not something to leave in.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Observation term 'scale' is not registered.` | An obs term is missing its `params:` key, so the block parsed as groups | Re-ship `deploy.yaml` from this repo — the current one is correct |
| `Observation term 'base_lin_vel' is not registered.` | The `State_RLBase.cpp` patch is missing on this robot | Part 2.4, then rebuild |
| `Input name obs not found in observations` | The obs group name does not match the ONNX input | The group is named `obs` on the single-group path; check every term has `params:` |
| Exception at the first control step, after `R2 + A` | `commands:` keyed something other than `base_velocity` | Re-ship `deploy.yaml` |
| `get_type_props` undefined reference | SDK v2.0.0 | Part 2.3 |
| `KeyBase does not name a type` | SDK `Button`/`Axis` have no common base | Part 2.5 |
| `cannot find -lfmt` | `libfmt` listed but unused and not installed | Part 2.6 |
| `ModuleNotFoundError: yaml` during install | robot python has no pyyaml | Harmless — the script skips that check and says so; it was verified on the laptop |
| `scp`/`ssh` asks for the password three times | no key installed | Optional: `ssh-copy-id unitree@192.168.123.164` once, then it stops asking |
| `unrecognised option '--config'` | No such flag exists | `./g1_ctrl -n eth0` |
| Clock skew warnings during `make` | Robot clock is at 1970 | Harmless |
| Loads the wrong policy | `policy_dir` | `grep policy_dir config.yaml` |
| Walks, but badly and vaguely | The observation layout — the one failure mode with no error message | Part 6, and re-read the layout table in `deploy.yaml` |

---

## After the session

- [ ] Pull the `~/run8_*.log` file off the robot
- [ ] Note achieved-vs-commanded for each gate against the table in Part 5
- [ ] Note anything that differed from sim — especially yaw, and anything about
      the FixStand → Velocity handover
- [ ] Ankle motor temperatures, even bare. They matter far more once the stilts
      go on, so a bare-robot baseline is worth having.
