#!/usr/bin/env python3
"""Turn a G1 telemetry CSV into the numbers a hardware report needs.

    uv run python scripts/analyze_hardware_log.py logs/hardware/<date>/g1_*.csv
    uv run python scripts/analyze_hardware_log.py <csv> --stilts     # stilts fitted
    uv run python scripts/analyze_hardware_log.py <csv> --md out.md  # also save

The CSV is written on the robot by deploy/patches/telemetry.h (compiled in by
scripts/prepare_runtime.py), one row per 50 Hz policy step, and lands in
~/telemetry/ on the robot. `scripts/sim_telemetry.py` writes the same format from
simulation, plus ground-truth base position, which is how the odometry below
is validated.

What it reports, per segment (one segment per entry into a policy state):

- Loop timing: step period percentiles and overruns, ONNX inference time.
- Posture: tilt and the projected-gravity components the policy sees.
- Command windows: every command held for 2 s or more, with the velocity the
  robot actually achieved, estimated by leg odometry.
- Standing still: drift at zero command. This is the number the 2026-09-14
  session could only describe as "a slight creep".
- Joints: peak torque estimate, PD tracking error, motor temperature rise.

LEG ODOMETRY. The G1 has no body-velocity sensor, so velocity is estimated from
kinematics: pose the MJCF with the IMU orientation and the encoder angles, take
the lower foot as the stance foot, and assume it does not move in the world. The
pelvis moves by minus the stance foot's displacement relative to it. It is blind
to foot slip and accumulates IMU yaw drift, so trust window means, not samples.
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MJCF = ROOT / "assets" / "mjcf" / "g1" / "g1.xml"
DEPLOY_YAML = ROOT / "deploy" / "config" / "g1_stilt" / "deploy.yaml"

N_JOINTS = 29
STEP_MS = 20.0
OVERRUN_MS = 25.0
FALL_TILT_DEG = 45.0
CMD_BIN = 0.1  # commands within one bin count as held
ZERO_CMD = 0.05  # every axis below this counts as a zero command
MIN_WINDOW_S = 2.0
SETTLE_S = 0.5  # dropped from the start of each window
STANCE_HYSTERESIS_M = 0.005


def telemetry_header() -> list[str]:
  """Column order written by telemetry::header() in deploy/patches/telemetry.h."""
  cols = [
    "wall_s", "mono_s", "segment", "step", "tick", "dt_ms", "infer_ms",
    "quat_w", "quat_x", "quat_y", "quat_z", "gyro_x", "gyro_y", "gyro_z",
    "acc_x", "acc_y", "acc_z", "cmd_vx", "cmd_vy", "cmd_wz",
  ]  # fmt: skip
  for field in ("q", "dq", "tau_est", "q_des", "action", "temp"):
    cols += [f"{field}_{j:02d}" for j in range(N_JOINTS)]
  return cols


def load(path: Path) -> dict[str, np.ndarray]:
  header = path.open().readline().strip().split(",")
  data = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
  return {name: data[:, i] for i, name in enumerate(header)}


def per_joint(log: dict[str, np.ndarray], field: str) -> np.ndarray:
  return np.stack([log[f"{field}_{j:02d}"] for j in range(N_JOINTS)], axis=1)


def joint_names() -> tuple[list[str], str]:
  """Policy-order joint names from the ONNX that deploy.yaml was generated from."""
  try:
    import onnx

    source = next(
      line.split(":", 1)[1].strip()
      for line in DEPLOY_YAML.read_text().splitlines()
      if line.startswith("# Source:")
    )
    (onnx_path,) = (ROOT / "logs" / "rsl_rl" / "stilt_g1_velocity" / source).glob(
      "*.onnx"
    )
    meta = {p.key: p.value for p in onnx.load(str(onnx_path)).metadata_props}
    names = meta["joint_names"].split(",")
    if len(names) == N_JOINTS:
      return names, source
  except (ImportError, StopIteration, ValueError, KeyError, FileNotFoundError):
    pass
  return [f"j{j:02d}" for j in range(N_JOINTS)], "unknown (index names)"


def rotation(quat: np.ndarray) -> np.ndarray:
  """(N, 4) wxyz body-to-world quaternions -> (N, 3, 3) rotation matrices."""
  q = quat / np.linalg.norm(quat, axis=1, keepdims=True)
  w, x, y, z = q.T
  return np.stack(
    [
      np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
      np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
      np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ],
    axis=1,
  )


def to_heading(vec_w: np.ndarray, yaw: np.ndarray) -> np.ndarray:
  """Rotate world-frame planar vectors into the robot's heading frame."""
  c, s = np.cos(yaw), np.sin(yaw)
  return np.stack(
    [c * vec_w[:, 0] + s * vec_w[:, 1], -s * vec_w[:, 0] + c * vec_w[:, 1]], -1
  )


def leg_odometry(
  log: dict[str, np.ndarray], names: list[str], stilts: bool
) -> np.ndarray:
  """(N, 2) planar pelvis position in the IMU's world frame, starting at zero."""
  import mujoco

  model = mujoco.MjModel.from_xml_path(str(MJCF))
  data = mujoco.MjData(model)
  adr = np.array([model.joint(n).qposadr[0] for n in names])
  point = "stilt_tip" if stilts else "foot"
  sites = [model.site(f"{side}_{point}").id for side in ("left", "right")]

  quat = np.stack([log["quat_w"], log["quat_x"], log["quat_y"], log["quat_z"]], axis=1)
  quat /= np.linalg.norm(quat, axis=1, keepdims=True)
  q = per_joint(log, "q")
  n = len(q)
  feet = np.empty((n, 2, 3))
  for t in range(n):
    data.qpos[:] = model.qpos0
    data.qpos[0:3] = 0.0
    data.qpos[3:7] = quat[t]
    data.qpos[adr] = q[t]
    mujoco.mj_kinematics(model, data)
    feet[t, 0] = data.site_xpos[sites[0]]
    feet[t, 1] = data.site_xpos[sites[1]]

  stance = np.empty(n, dtype=int)
  s = int(np.argmin(feet[0, :, 2]))
  for t in range(n):
    if feet[t, 1 - s, 2] < feet[t, s, 2] - STANCE_HYSTERESIS_M:
      s = 1 - s
    stance[t] = s

  idx = np.arange(1, n)
  step = feet[idx, stance[1:], :2] - feet[idx - 1, stance[1:], :2]
  return np.vstack([np.zeros((1, 2)), -np.cumsum(step, axis=0)])


def command_windows(t: np.ndarray, cmd: np.ndarray) -> list[tuple[int, int]]:
  """Index ranges where the command stayed in one bin for MIN_WINDOW_S."""
  key = np.round(cmd / CMD_BIN).astype(int)
  change = np.flatnonzero(np.any(np.diff(key, axis=0) != 0, axis=1)) + 1
  starts = np.r_[0, change]
  ends = np.r_[change, len(t)]
  windows = []
  for a, b in zip(starts, ends, strict=True):
    a = int(np.searchsorted(t, t[a] + SETTLE_S))
    if a < b - 1 and t[b - 1] - t[a] >= MIN_WINDOW_S:
      windows.append((a, int(b)))
  return windows


def table(headers: list[str], rows: list[list[str]]) -> str:
  lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
  lines += ["| " + " | ".join(r) + " |" for r in rows]
  return "\n".join(lines)


def clock(wall_s: float) -> str:
  return dt.datetime.fromtimestamp(wall_s).strftime("%H:%M:%S")


def analyse(path: Path, stilts: bool) -> str:
  log = load(path)
  names, source = joint_names()
  has_truth = "true_x" in log
  out = [
    f"# G1 telemetry: `{path.name}`",
    "",
    f"Joint names from `{source}`. Odometry contact point: "
    f"{'stilt tips' if stilts else 'foot soles'}. "
    "Times are the robot's clock, which is not set correctly; match them to the "
    "`FSM:` lines of the run log.",
  ]

  segments = np.unique(log["segment"]).astype(int)
  session, timing, posture, windows_rows, still_rows = [], [], [], [], []
  joint_seg = {}

  for seg in segments:
    m = log["segment"] == seg
    L = {k: v[m] for k, v in log.items()}
    t = L["mono_s"] - L["mono_s"][0]
    n = len(t)
    duration = t[-1] if n > 1 else 0.0
    session.append([str(seg), clock(L["wall_s"][0]), f"{duration:.1f} s", str(n)])

    dts = L["dt_ms"][1:]
    inf = L["infer_ms"]
    if len(dts):
      timing.append(
        [str(seg)]
        + [f"{np.percentile(dts, p):.1f}" for p in (50, 99)]
        + [f"{dts.max():.1f}", str(int((dts > OVERRUN_MS).sum()))]
        + [f"{np.percentile(inf, p):.2f}" for p in (50, 99)]
        + [f"{inf.max():.2f}"]
      )

    quat = np.stack([L["quat_w"], L["quat_x"], L["quat_y"], L["quat_z"]], axis=1)
    R = rotation(quat)
    tilt = np.degrees(np.arccos(np.clip(R[:, 2, 2], -1.0, 1.0)))
    grav_b = -R[:, 2, :]  # world -z expressed in the body frame
    yaw = np.arctan2(R[:, 1, 0], R[:, 0, 0])
    posture.append(
      [
        str(seg),
        f"{tilt.mean():.1f}°",
        f"{tilt.max():.1f}°",
        f"{grav_b[:, 0].mean():+.3f}",
        f"{grav_b[:, 1].mean():+.3f}",
        "**yes**" if tilt.max() > FALL_TILT_DEG else "no",
      ]
    )

    cmd = np.stack([L["cmd_vx"], L["cmd_vy"], L["cmd_wz"]], axis=1)
    if n > 2:
      pos = leg_odometry(L, names, stilts)
      v_h = to_heading(np.gradient(pos, t, axis=0), yaw)
      if has_truth:
        true_pos = np.stack([L["true_x"], L["true_y"]], axis=1)
        true_v_h = to_heading(np.gradient(true_pos, t, axis=0), yaw)
      for a, b in command_windows(t, cmd):
        span = t[b - 1] - t[a]
        c = cmd[a:b].mean(axis=0)
        row = [
          str(seg),
          f"{t[a]:.0f} s",
          f"{span:.1f} s",
          f"{c[0]:+.2f} {c[1]:+.2f} {c[2]:+.2f}",
          f"{v_h[a:b, 0].mean():+.2f} {v_h[a:b, 1].mean():+.2f}",
          f"{L['gyro_z'][a:b].mean():+.2f}",
        ]
        if has_truth:
          row.append(f"{true_v_h[a:b, 0].mean():+.2f} {true_v_h[a:b, 1].mean():+.2f}")
        windows_rows.append(row)

        if np.all(np.abs(c) < ZERO_CMD):
          disp_w = pos[b - 1] - pos[a]
          disp_h = to_heading(disp_w[None, :], np.array([yaw[a:b].mean()]))[0]
          dist = float(np.linalg.norm(disp_w))
          row = [
            str(seg),
            f"{t[a]:.0f} s",
            f"{span:.1f} s",
            f"{dist * 100:.1f} cm",
            f"{dist / span * 100:.1f} cm/s",
            f"{disp_h[0] * 100:+.1f} / {disp_h[1] * 100:+.1f} cm",
            f"{np.degrees(yaw[b - 1] - yaw[a]):+.1f}°",
          ]
          if has_truth:
            true_disp = true_pos[b - 1] - true_pos[a]
            row.append(f"{np.linalg.norm(true_disp) * 100:.1f} cm")
          still_rows.append(row)

    q, q_des = per_joint(L, "q"), per_joint(L, "q_des")
    tau, temp = per_joint(L, "tau_est"), per_joint(L, "temp")
    joint_seg[seg] = (
      np.abs(tau).max(axis=0),
      np.degrees(np.sqrt(((q_des - q) ** 2).mean(axis=0))),
      temp[0],
      temp.max(axis=0),
    )

  out += ["", "## Segments", "", table(["seg", "start", "duration", "steps"], session)]
  out += [
    "",
    f"## Loop timing (target {STEP_MS:.0f} ms)",
    "",
    table(
      [
        "seg",
        "dt p50 ms",
        "dt p99",
        "dt max",
        f"> {OVERRUN_MS:.0f} ms",
        "infer p50",
        "p99",
        "max",
      ],
      timing,
    ),
  ]
  out += [
    "",
    "## Posture",
    "",
    "`grav x/y` are the projected-gravity components the policy observes.",
    "",
    table(
      [
        "seg",
        "tilt mean",
        "tilt max",
        "grav x",
        "grav y",
        f"fell (> {FALL_TILT_DEG:.0f}°)",
      ],
      posture,
    ),
  ]

  win_headers = ["seg", "at", "held", "cmd vx vy wz", "odom vx vy", "gyro wz"]
  if has_truth:
    win_headers.append("true vx vy")
  out += [
    "",
    "## Command tracking",
    "",
    f"Every command held for {MIN_WINDOW_S:.0f} s or more, first {SETTLE_S} s dropped. "
    "Velocities in m/s in the heading frame (x forward, y left), yaw rate in rad/s.",
    "",
    table(win_headers, windows_rows) if windows_rows else "_No held commands._",
  ]

  still_headers = ["seg", "at", "held", "drift", "speed", "fwd / left", "yaw change"]
  if has_truth:
    still_headers.append("true drift")
  out += [
    "",
    "## Standing still (zero command)",
    "",
    table(still_headers, still_rows) if still_rows else "_No zero-command windows._",
  ]

  out += ["", "## Joints", ""]
  for seg, (peak, track, t0, tmax) in joint_seg.items():
    order = np.argsort(-peak)[:10]
    rows = [
      [
        names[j],
        f"{peak[j]:.1f}",
        f"{track[j]:.2f}",
        f"{t0[j]:.0f} → {tmax[j]:.0f} °C",
      ]
      for j in order
    ]
    hottest = int(np.argmax(tmax))
    heat = (
      f" Hottest motor: `{names[hottest]}` at {tmax[hottest]:.0f} °C."
      if tmax.max() > 0
      else " No temperature data (simulation)."
    )
    out += [
      f"Segment {seg}, ten highest peak torques.{heat}",
      "",
      table(["joint", "peak |tau_est| Nm", "RMS track err °", "temperature"], rows),
      "",
    ]
  return "\n".join(out)


def main() -> int:
  ap = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
  )
  ap.add_argument("csv", type=Path)
  ap.add_argument("--stilts", action="store_true", help="odometry on the stilt tips")
  ap.add_argument("--md", type=Path, help="also write the report to this file")
  args = ap.parse_args()

  report = analyse(args.csv, args.stilts)
  print(report)
  if args.md:
    args.md.write_text(report + "\n")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
