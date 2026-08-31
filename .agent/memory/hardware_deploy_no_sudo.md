---
name: hardware-deploy-no-sudo
description: How unitree_rl_mjlab was built on the 2026-08 G1 without sudo or touching /usr/local — the May SDK-install step is unnecessary
metadata: 
  node_type: memory
  type: project
  originSessionId: e47aed7e-46a3-4be7-b3b4-d8ee09eda3d2
  modified: 2026-08-31T01:46:18.315Z
---

The `deploy/README.md` "one-time setup" tells you to `sudo make install` a newer
`unitree_sdk2` into `/usr/local`. **Don't.** On the QCR lab G1 used on
2026-08-31 (eth0 MAC `3c:6d:66:a3:e5:73` — *not* the May robot) that would
overwrite the headers and `libunitree_sdk2.a` that the lab's own software is
built against.

It is also unnecessary. `~/unitree_sdk2` on that robot is already the newer SDK:
its prebuilt `lib/aarch64/libunitree_sdk2.a` carries 95 `unitree_hg` symbols
(the `get_type_props<unitree_hg::msg::dds_::...>` ones whose absence caused the
May link errors), and `thirdparty/` bundles a matching cyclonedds. The
`/usr/local` install has **zero** `unitree_hg` symbols and only `go2`/`ros2` IDL.

The fix is include/link paths in **our own copy** of
`deploy/robots/g1/CMakeLists.txt` — `include_directories(BEFORE ...)`,
`link_directories(BEFORE ...)` at `$ENV{HOME}/unitree_sdk2`, plus
`CMAKE_BUILD_RPATH` so the binary loads the SDK's cyclonedds rather than
`/usr/local`'s different build. Verify with `readelf -d g1_ctrl | grep RUNPATH`
and `ldd`. No sudo, nothing installed system-wide.

Also true of that robot:

- **CMake is 4.2.1.** It configures fine — the g1 CMakeLists declares 3.12, and
  the `cmake_minimum_required(VERSION 3.0)` in `thirdparty/cnpy` is inert
  because cnpy is globbed as sources, never `add_subdirectory`'d.
- **Only `deploy/` needs shipping** (~31 MB after dropping the x64 onnxruntime).
  The 224 MB `src/` training half is irrelevant on the robot, and the aarch64
  onnxruntime is already unpacked in `deploy/thirdparty/`.
- **`master_service` holds the lowcmd channel** until the robot is put into
  damping/debug mode from the remote. `g1_ctrl` warns
  (`The other process is using the lowcmd channel`) but upstream commented out
  the `exit(0)`, so it carries on with two controllers fighting. The absence of
  that log line is the clean go/no-go test.
- Laptop needs a static `192.168.123.222/24` on the USB-Ethernet adapter; there
  is no DHCP on the robot's network.

Findings from the session itself: [[hardware-forward-drift]].
