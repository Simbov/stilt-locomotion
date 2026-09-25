"""The hardware telemetry recorder patches cleanly and writes the CSV we read back.

The robot is air-gapped and every rebuild there costs a trip, so both halves are
checked on the laptop: the text patches must land on the stock unitree_rl_mjlab
source (including alongside the probe), and deploy/patches/telemetry.h is
compiled against a stand-in env and run, so the column layout the analyser
depends on is pinned by an actual file rather than by reading the C++.
"""

import csv
import shutil
import subprocess
from pathlib import Path

import pytest

from scripts import prepare_runtime as pr
from scripts.analyze_hardware_log import telemetry_header

ROOT = Path(__file__).parent.parent
SRC = ROOT / "unitree_rl_mjlab" / "deploy"


@pytest.fixture
def stock_include(tmp_path):
  if not (SRC / "include").is_dir():
    pytest.skip("unitree_rl_mjlab submodule not checked out")
  out = tmp_path / "deploy"
  shutil.copytree(SRC / "include", out / "include")
  return out


def test_telemetry_patch_lands_on_the_stock_runtime(stock_include):
  pr.patch_telemetry(stock_include)
  inc = stock_include / "include"
  env_h = (inc / "isaaclab/envs/manager_based_rl_env.h").read_text()
  assert '#include "telemetry.h"' in env_h
  assert "telemetry::Recorder::get().new_segment();" in env_h
  assert "telemetry::record(this, obs, action," in env_h
  assert (
    "tlm_tau_est" in (inc / "isaaclab/assets/articulation/articulation.h").read_text()
  )
  assert "m.tau_est()" in (inc / "unitree_articulation.h").read_text()
  assert (inc / "telemetry.h").read_bytes() == pr.TELEMETRY_H.read_bytes()


def test_telemetry_and_probe_patches_compose(stock_include):
  env_h = stock_include / "include/isaaclab/envs/manager_based_rl_env.h"
  pr.patch_probe(env_h, history=5, n_joints=29)
  pr.patch_telemetry(stock_include)
  s = env_h.read_text()
  assert "Bring-up probe" in s
  assert "telemetry::record(this" in s


HARNESS = r"""
#include "telemetry.h"

struct Quat {
  float w() const { return 1.0f; }
  float x() const { return 0.0f; }
  float y() const { return 0.0f; }
  float z() const { return 0.0f; }
};
struct Data {
  Quat root_quat_w;
  std::vector<float> root_ang_vel_b{0.1f, 0.2f, 0.3f};
  std::vector<float> joint_pos, joint_vel, tlm_tau_est;
  std::vector<int16_t> tlm_motor_temp;
  std::vector<float> tlm_imu_acc{0.0f, 0.0f, -9.81f};
  uint32_t tlm_tick = 0;
};
struct Robot { Data data; };
struct Actions {
  std::vector<float> q;
  std::vector<float> processed_actions() const { return q; }
};
struct Env { long episode_length; Robot* robot; Actions* action_manager; };

int main() {
  Robot robot;
  Actions actions;
  Env env{0, &robot, &actions};
  auto& d = robot.data;
  d.joint_pos.assign(29, 0.0f);
  d.joint_vel.assign(29, 0.0f);
  d.tlm_tau_est.assign(29, 0.0f);
  d.tlm_motor_temp.assign(29, 0);
  actions.q.assign(29, 0.5f);
  std::vector<float> action(29, 0.0f);
  for (int seg = 0; seg < 2; ++seg) {
    telemetry::Recorder::get().new_segment();
    for (int s = 1; s <= 60; ++s) {
      env.episode_length = s;
      d.tlm_tick = 1000 * seg + s;
      for (int j = 0; j < 29; ++j) {
        d.joint_pos[j] = 0.01f * j;
        d.joint_vel[j] = -1.0f * j;
        d.tlm_tau_est[j] = float(s + j);
        d.tlm_motor_temp[j] = int16_t(30 + j);
        action[j] = float(j - 14);
      }
      std::unordered_map<std::string, std::vector<float>> obs{
        {"obs", {9.0f, 9.0f, 0.4f, -0.1f, 0.25f}}};
      telemetry::record(&env, obs, action, 1.5);
    }
  }
  return 0;  // Recorder destructor flushes
}
"""


def _compilers():
  names = ["c++", "g++", "clang++"] + [f"g++-{v}" for v in range(20, 8, -1)]
  return [p for p in map(shutil.which, names) if p]


def test_recorder_writes_the_documented_csv(tmp_path):
  compilers = _compilers()
  if not compilers:
    pytest.skip("no C++ compiler")
  (tmp_path / "h.cpp").write_text(HARNESS)
  flags = ["-std=c++17", "-Wall", "-Werror", "-pthread", f"-I{pr.TELEMETRY_H.parent}"]

  # A broken local linker (seen with the macOS 27 Command Line Tools, which
  # cannot link even an empty main) must not hide a header that does not
  # compile, so the syntax check runs regardless.
  syntax = subprocess.run(
    [compilers[0], *flags, "-fsyntax-only", "h.cpp"],
    cwd=tmp_path,
    capture_output=True,
    text=True,
  )
  assert syntax.returncode == 0, syntax.stderr

  exe = tmp_path / "h"
  for cxx in compilers:
    build = subprocess.run(
      [cxx, *flags, "h.cpp", "-o", str(exe)], cwd=tmp_path, capture_output=True
    )
    if build.returncode == 0:
      break
  else:
    pytest.skip("no C++ toolchain here can link; header syntax-checked only")

  out_dir = tmp_path / "telemetry"
  run = subprocess.run(
    [str(exe)],
    env={"G1_TELEMETRY_DIR": str(out_dir)},
    check=True,
    capture_output=True,
    text=True,
  )
  assert "[telemetry] writing" in run.stdout

  (csv_path,) = out_dir.glob("g1_*.csv")
  with csv_path.open() as f:
    rows = list(csv.DictReader(f))
  header = csv_path.read_text().splitlines()[0].split(",")

  assert header == telemetry_header()
  assert len(rows) == 120
  assert all(None not in r and len(r) == len(header) for r in rows)
  assert {r["segment"] for r in rows} == {"1", "2"}

  first, last = rows[0], rows[119]
  assert first["step"] == "1" and float(first["dt_ms"]) == 0.0
  assert last["step"] == "60" and last["tick"] == "1060"
  assert float(last["q_05"]) == pytest.approx(0.05)
  assert float(last["dq_03"]) == pytest.approx(-3.0)
  assert float(last["tau_est_28"]) == pytest.approx(88.0)
  assert float(last["q_des_10"]) == pytest.approx(0.5)
  assert float(last["action_00"]) == pytest.approx(-14.0)
  assert last["temp_28"] == "58"
  assert float(last["infer_ms"]) == pytest.approx(1.5)
  assert [float(last[k]) for k in ("cmd_vx", "cmd_vy", "cmd_wz")] == pytest.approx(
    [0.4, -0.1, 0.25]
  )
  assert float(last["acc_z"]) == pytest.approx(-9.81)
