# G1 bring-up runbook

Taking a trained policy to the physical robot. Written after the 2026-08-31
session, which took about an hour of setup to reach the first joystick press;
following this should take **ten minutes on a robot we know, twenty on a new
one.**

Current policy: **Run 9**, `2026-08-31_13-35-11_run9-no-baselinvel`,
480 inputs → 29 actions. Results: [`docs/RUN_RESULTS.md`](../docs/RUN_RESULTS.md).
Why things are the way they are: [`deploy/README.md`](README.md).

> **Do not deploy Run 8 or earlier.** Run 8's observation includes
> `base_lin_vel`, which the G1 cannot measure; on hardware it drifts at a zero
> command and will not track. Run 9 removed the term.

---

## Known robots

Identity comes from the eth0 MAC — the robots carry no name and every hostname
is `ubuntu`. `00_identify_robot.sh` checks this for you.

| eth0 MAC | wlan0 MAC | First seen | Notes |
|---|---|---|---|
| `3c:6d:66:2b:d1:f0` | `fc:23:cd:8f:70:79` | 2026-05-09 | home dirs incl. `FALCON`, `inspire_hand_ws`, `ws_livox`, `xr_teleoperate` |
| `3c:6d:66:a3:e5:73` | `fc:23:cd:91:36:15` | 2026-08-31 | QCR lab build; `QCR_G1`, `g1plus_pc4_unitree_install`. Runtime installed under `~/unitree_rl_mjlab` |
| `3c:6d:66:a3:e0:76` | — | 2026-09-17 | shipped without a G1 SDK (only go2/ros2 in `~/unitree_sdk2-main` and `/usr/local`); `~/unitree_sdk2` copied from the laptop, no sudo. cmake 3.16.3. `ROS2`, `controller_ws`, `Bag_HumanoidGuide` |

**Add a row whenever you meet a new one.** Both are 29-DoF G1s; `g1_ctrl`
checks that itself at startup and exits if not.

---

## Part 0 — Laptop, before you leave

```sh
uv run pytest tests/ -q                                    # expect 70 passed
uv run python scripts/verify_deploy_io.py --onnx logs/rsl_rl/stilt_g1_velocity/<run>/<run>.onnx
```

`verify_deploy_io.py` must say **PASS**. It pins the observation layout end to
end, and it is the only check that catches a `deploy.yaml` that does not
describe the policy beside it.

If you retrained, regenerate the config first — never hand-edit it:

```sh
uv run python scripts/generate_deploy_config.py --run logs/rsl_rl/stilt_g1_velocity/<run>
```

Physical: charge the joystick and the laptop, pack the Ethernet cable and the
USB adapter, pack the harness fittings.

---

## Part 1 — Connect

Two things that are not obvious and cost an hour the first time.

**1. There is no DHCP on the robot's network.** Your adapter will sit on a
self-assigned `169.254.x.x` and nothing will ping. Give it a static address on
the robot's subnet (`en13` is the AX88179A USB adapter — check yours with
`networksetup -listnetworkserviceorder`):

```sh
sudo ifconfig en13 192.168.123.222 netmask 255.255.255.0 up
```

That form is temporary and disappears when you unplug, which is what you want.
Verify: `ping -c3 192.168.123.164`.

**2. Install your SSH key**, or every command needs the password typed and
nothing can be scripted:

```sh
ssh-copy-id unitree@192.168.123.164        # password: 123
```

> If the link drops mid-session the interface loses its address silently — you
> get `Operation timed out` and `status: inactive` on the interface. Re-run the
> `ifconfig` line. Nothing on the robot is lost.

---

## Part 2 — Which robot is this, and what is on it?

```sh
./scripts/ship_to_robot.sh                 # policy only, ~2 s
ssh unitree@192.168.123.164
bash ~/run8/00_identify_robot.sh
```

The script prints the MAC, compares it to the known-robot list, checks every
setup marker, and gives a verdict. Then take one of two paths.

### Path A — known robot, verdict says setup complete

Skip to **Part 3**. Nothing else is needed; the runtime and the built binary
persist across reboots.

**Exception: a runtime from before 2026-09-14 has no telemetry recorder.** Ship
`./scripts/ship_to_robot.sh --runtime` once. Afterwards the `g1_ctrl` output
includes a `[telemetry] writing ...` line as soon as Velocity starts.

### Path B — new robot, or anything missing

The whole setup is now **one command from the laptop** (`Ctrl-D` out of the
robot first):

```sh
./scripts/ship_to_robot.sh --runtime
```

That runs `scripts/prepare_runtime.py` to patch a clean copy of the runtime on
the laptop, then ships and unpacks it. ~31 MB, a few seconds. Patch on the
laptop, never on the robot: the robot is air-gapped, has no editor (`nano` is
not installed) and its clock is wrong.

Add `--probe` on a robot you have never deployed to — it compiles in an
observation probe that proves the IMU and encoders land in the right slots.
See Part 6.

**Prerequisites on the robot, all true on both machines so far:**
`libyaml-cpp-dev`, `libeigen3-dev`, `libboost-program-options-dev`, cmake
(4.2.1 works fine), gcc 9.4, and **`~/unitree_sdk2` containing
`include/unitree/idl/hg`**. `00_identify_robot.sh` checks the last one.

> ### Do NOT install a newer unitree_sdk2 over /usr/local
>
> The 2026-05 instructions said to `sudo make install` one. **Don't.** These are
> shared lab machines and other software is built against what is in
> `/usr/local`. It is also unnecessary: `~/unitree_sdk2` already carries the
> `unitree_hg` DDS types (95 symbols in its prebuilt lib, against zero in the
> `/usr/local` one), and `prepare_runtime.py` points the build at it with
> include/link paths and an RPATH. Nothing is installed system-wide and no sudo
> is needed anywhere in this runbook.

---

## Part 3 — Install the policy and build

```sh
ssh unitree@192.168.123.164
bash ~/run8/01_install_policy.sh
cd ~/unitree_rl_mjlab/deploy/robots/g1/build && cmake .. && make -j$(nproc)
```

`01_install_policy.sh` verifies the checksums it was shipped with, installs,
re-checks `deploy.yaml` against the runtime's parsing rules, and points the
Velocity FSM at the policy. It is idempotent and refuses to proceed on a bad
checksum.

Build takes ~40 s. Clock-skew warnings are harmless — the robot's clock is
wrong. Confirm the binary resolves the right libraries:

```sh
ldd ~/unitree_rl_mjlab/deploy/robots/g1/build/g1_ctrl | grep -E "ddsc|onnx"
```

Both should point into `~/unitree_sdk2/...` and `~/unitree_rl_mjlab/...`, not
`/usr/local`.

---

## Part 4 — Release the lowcmd channel ⚠️

**This is the step that stops you, and it is not a software fix.** The robot's
built-in `master_service` publishes motor commands continuously. `g1_ctrl`
detects this and prints

```
[critical] The other process is using the lowcmd channel, please close it first.
```

— and then **carries on anyway**, because upstream commented out the `exit(0)`.
Two controllers then fight over the motors.

Put the robot into damping / debug mode **from the remote** so the built-in
controller hands over. Exact combo depends on firmware; check the manual if the
usual `L2 + B` then `L2 + R2` does not do it.

**The test is exact and needs no guessing:** launch `g1_ctrl` and look for that
line. Gone = channel released. Present = do not touch the joystick.

Do not kill `master_service` to get around this. It needs sudo, it is not a
systemd unit, and restoring it may need a reboot.

---

## Part 5 — Safety, before power

- [ ] Robot on the gantry, harness attached
- [ ] **Correct feet for the policy** — bare feet, or stilts fitted. Run 9
      handles both with no config change
- [ ] Joystick on; **`L2 + B` = Passive** is the kill (D-pad Down is added but unproven on hardware — see 2026-09-25), known by feel
- [ ] Second person on the hoist, doing nothing else
- [ ] Phone recording — failures are fast

**Launching drops every joint into damping.** An unsupported robot collapses at
that instant.

---

## Part 6 — Bring-up gates

```sh
cd ~/unitree_rl_mjlab/deploy/robots/g1/build
./g1_ctrl -n eth0 2>&1 | tee ~/run_$(date +%s).log
```

Verify the banner says your policy directory **before touching the joystick**.
A wrong `deploy.yaml` throws here, not later.

**Testing without the laptop cable.** The robot has no tmux, and `g1_ctrl`
dies with the SSH session. Check the banner and the lowcmd line in the
foreground first, stop it, then relaunch detached:

```sh
setsid -f bash -c 'sleep infinity | ./g1_ctrl -n eth0 > ~/run_$(date +%s).log 2>&1' < /dev/null > /dev/null 2>&1
```

`sleep infinity` is load-bearing: the runtime's keyboard thread reads stdin,
and on a closed stdin it spins a core next to the policy thread.

**Every policy step is recorded** to `~/telemetry/g1_<time>.csv` (IMU, command,
per-joint q/dq/torque/temperature/target, loop timing), about 5 MB per minute.
Note the gate times as you go; they match the `FSM:` lines in the run log.

| Gate | Action | Pass | Stop if |
|---|---|---|---|
| 6.1 | Launch | loads your policy; **no lowcmd critical** | any throw, or the critical line |
| 6.2 | `L2 + D-pad Up` → FixStand | stands to the crouch over ~2 s, quiet | buzzing, or a joint fighting |
| 6.3 | `R2 + A` → Velocity, **weight on the harness**, sticks centred | holds the policy's pose | sustained oscillation |
| 6.4 | Lower to full weight, zero command | **stands still.** Pelvis ~0.79 m bare, ~1.20 m on stilts | it creeps, sags, or marches |
| 6.5 | Work up: 0.2, 0.4, 0.6, then full | see the table in `docs/RUN_RESULTS.md` | any fall |
| 6.6 | Backward, lateral, yaw (several attempts) | all track, undershooting | |

**Gate 6.4 is the one that matters now.** It is exactly what Run 8 failed. Run 9
holds position to within 4 cm over 15 s in sim with no velocity feedback at all.

**FixStand hands over into a different pose.** `R2 + A` switches in one control
step and the poses differ (knee 0.3 → 0.1, ankle −0.2 → 0), with the gains
stepping too. Expect a visible settle; it is not a fault.
`bash ~/run8/02_match_fixstand_pose.sh` removes it if you would rather.

**If you shipped `--probe`**, the first 60 s of Velocity print at 2 Hz:

```
[probe 25] dim=480 grav=[-0.03 -0.00 -1.00] ang=[...] Ljp=[...] CMD=[...]
```

Held still in the FixStand pose, expect `dim` = **480**, `grav ≈ (0,0,−1)`,
`ang ≈ 0`, `CMD` exactly zero with your hands off, and the left leg reading
about `(0, 0, 0, +0.20, −0.20)` — the knee and ankle difference between the two
poses. Those two non-zero numbers landing in the right slots is a positive test
of joint ordering *and* the `default_joint_pos` subtraction.

**Abort and go to Passive** on: any motor hot, two falls at the same command,
or behaviour that changes between identical commands.

---

## Part 7 — Finishing

1. `L2 + B` → Passive, then stop the controller: `kill $(pgrep -x g1_ctrl)`
   — **use the PID, not `pkill -f`**, which matches your own SSH command line
   and kills the session instead.
2. Pull the logs, one remote pattern per `scp` (two paths inside one quoted
   argument fail with `No such file or directory`), then analyse:
   ```sh
   scp unitree@192.168.123.164:'~/run_*.log' logs/hardware/<date>/
   scp unitree@192.168.123.164:'~/telemetry/g1_*.csv' logs/hardware/<date>/
   uv run python scripts/analyze_hardware_log.py logs/hardware/<date>/g1_<time>.csv
   ```
   Add `--stilts` if they were fitted. For the matching sim baseline, run
   `uv run python scripts/sim_telemetry.py` and analyse its output the same way.
3. **Restore the built-in controller** — reverse the Part 4 remote sequence, or
   power-cycle. Confirm with
   `~/unitree_sdk2/build/bin/g1_loco_client --network_interface=eth0 --get_fsm_id`;
   a reply (`current fsm_id: 65535`) means the RPC path is back. Silence means
   it is still in debug mode.
4. Take the robot off the gantry, unplug the Ethernet. The laptop's temporary
   IP disappears on its own.

**Leave `~/unitree_rl_mjlab` and `~/run8` on the robot.** They are in the user's
home, nothing system-wide, and next visit becomes Path A.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ping` times out, interface `inactive` | link dropped; address lost | re-run the Part 1 `ifconfig` |
| `Observation term 'scale' is not registered.` | an obs term lacks `params:` | re-ship `deploy.yaml`; the generator emits it |
| `Observation term 'X' is not registered.` | `deploy.yaml` names a term the runtime lacks | check the table in `deploy/README.md` |
| Throw at the first control step | `commands:` not keyed `base_velocity` | re-ship `deploy.yaml` |
| `Input name obs not found in observations` | group name ≠ ONNX input name | every term needs `params:` (see above) |
| `get_type_props` undefined reference | build found the `/usr/local` SDK | `prepare_runtime.py` fixes this; **do not** install an SDK |
| `KeyBase does not name a type` | old SDK joystick classes | `prepare_runtime.py` |
| `cannot find -lfmt` | unused dep | `prepare_runtime.py` |
| lowcmd critical at launch | built-in controller active | Part 4; redo `L2 + B`, `L2 + R2`, or `g1_loco_client --network_interface=eth0 --damp` |
| No `[telemetry] writing` line | runtime predates the recorder | `ship_to_robot.sh --runtime`, rebuild |
| Clock skew during `make` | robot clock is wrong | harmless |
| Walks forward at zero command | **you are running Run 8 or earlier** | deploy Run 9+ |
