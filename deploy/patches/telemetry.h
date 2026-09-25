// Per-step telemetry for the G1 policy runtime.
//
// Injected into unitree_rl_mjlab by scripts/prepare_runtime.py and read back by
// scripts/analyze_hardware_log.py. One CSV row per policy step (50 Hz): loop
// timing, the IMU, the command the policy saw, and per joint the encoder state,
// the motor's own torque estimate and temperature, the PD target and the raw
// policy output.
//
// The stock runtime logs only FSM transitions, which is why the 2026-09-14
// session could not put a number on "walks on the spot with a slight creep".
//
// The policy thread only copies a fixed-size row into a pre-reserved buffer
// under a mutex. Formatting and disk writes happen on a background thread every
// 250 ms, so a slow disk cannot stall the control loop. Killing the process
// loses at most the last 250 ms.
//
// Output: $G1_TELEMETRY_DIR if set, else ~/telemetry/, one file per process.
// About 5 MB per minute of policy control.
#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace telemetry
{

constexpr int kJoints = 29;

struct Row
{
    double wall_s = 0;     // system clock, matches the spdlog timestamps
    double mono_s = 0;     // steady clock, use this for timing
    uint32_t segment = 0;  // increments on every policy reset (FSM entry)
    int64_t step = 0;
    uint32_t tick = 0;     // LowState tick
    float dt_ms = 0;       // since the previous row of this segment
    float infer_ms = 0;
    float quat[4] = {};    // w x y z, straight from the IMU
    float gyro[3] = {};
    float acc[3] = {};
    float cmd[3] = {};     // newest command frame the policy saw: vx vy wz
    float q[kJoints] = {};
    float dq[kJoints] = {};
    float tau_est[kJoints] = {};
    float q_des[kJoints] = {};   // PD target after scale and offset
    float action[kJoints] = {};  // raw policy output
    int16_t temp[kJoints] = {};
};

// Joint columns are indexed in policy order; the analyser maps them to names
// from the ONNX metadata.
inline std::string header()
{
    std::string h = "wall_s,mono_s,segment,step,tick,dt_ms,infer_ms,"
                    "quat_w,quat_x,quat_y,quat_z,gyro_x,gyro_y,gyro_z,"
                    "acc_x,acc_y,acc_z,cmd_vx,cmd_vy,cmd_wz";
    for (const char* field : {"q", "dq", "tau_est", "q_des", "action", "temp"}) {
        for (int j = 0; j < kJoints; ++j) {
            char col[32];
            std::snprintf(col, sizeof col, ",%s_%02d", field, j);
            h += col;
        }
    }
    return h;
}

class Recorder
{
public:
    static Recorder& get()
    {
        static Recorder instance;
        return instance;
    }

    void new_segment()
    {
        std::lock_guard<std::mutex> lock(mu_);
        ++segment_;
        have_prev_ = false;
    }

    void push(Row& r)
    {
        std::lock_guard<std::mutex> lock(mu_);
        r.segment = segment_;
        r.dt_ms = have_prev_ ? static_cast<float>((r.mono_s - prev_mono_s_) * 1e3) : 0.0f;
        prev_mono_s_ = r.mono_s;
        have_prev_ = true;
        if (pending_.size() < kMaxPending) {
            pending_.push_back(r);  // capacity is reserved: never allocates
        } else {
            ++dropped_;
        }
    }

    ~Recorder()
    {
        stop_ = true;
        if (writer_.joinable()) writer_.join();
        if (file_) std::fclose(file_);
    }

private:
    static constexpr size_t kMaxPending = 1500;  // 30 s at 50 Hz

    Recorder()
    {
        pending_.reserve(kMaxPending);
        writing_.reserve(kMaxPending);
        writer_ = std::thread([this] {
            while (!stop_) {
                std::this_thread::sleep_for(std::chrono::milliseconds(250));
                flush();
            }
            flush();
        });
    }

    void flush()
    {
        uint64_t dropped = 0;
        {
            std::lock_guard<std::mutex> lock(mu_);
            pending_.swap(writing_);
            std::swap(dropped, dropped_);
        }
        if (dropped) {
            std::fprintf(stderr, "[telemetry] dropped %llu rows, writer fell behind\n",
                         static_cast<unsigned long long>(dropped));
        }
        if (writing_.empty()) return;
        if (file_ || open()) {
            for (const Row& r : writing_) write(r);
            std::fflush(file_);
        }
        writing_.clear();  // keeps its capacity for the next swap
    }

    bool open()
    {
        if (open_failed_) return false;
        namespace fs = std::filesystem;
        const char* dir_env = std::getenv("G1_TELEMETRY_DIR");
        const char* home = std::getenv("HOME");
        const fs::path dir = dir_env ? fs::path(dir_env) : fs::path(home ? home : ".") / "telemetry";
        std::error_code ec;
        fs::create_directories(dir, ec);

        char name[64];
        const std::time_t now = std::time(nullptr);
        std::strftime(name, sizeof name, "g1_%Y%m%d-%H%M%S.csv", std::localtime(&now));
        const std::string path = (dir / name).string();

        file_ = std::fopen(path.c_str(), "w");
        if (!file_) {
            std::fprintf(stderr, "[telemetry] cannot open %s, telemetry disabled\n", path.c_str());
            open_failed_ = true;
            return false;
        }
        std::fprintf(file_, "%s\n", header().c_str());
        std::printf("[telemetry] writing %s\n", path.c_str());
        std::fflush(stdout);
        return true;
    }

    void write(const Row& r)
    {
        std::fprintf(file_, "%.6f,%.6f,%u,%lld,%u,%.3f,%.3f", r.wall_s, r.mono_s, r.segment,
                     static_cast<long long>(r.step), r.tick, r.dt_ms, r.infer_ms);
        const auto put = [this](const float* v, int n) {
            for (int i = 0; i < n; ++i) std::fprintf(file_, ",%.6g", v[i]);
        };
        put(r.quat, 4);
        put(r.gyro, 3);
        put(r.acc, 3);
        put(r.cmd, 3);
        put(r.q, kJoints);
        put(r.dq, kJoints);
        put(r.tau_est, kJoints);
        put(r.q_des, kJoints);
        put(r.action, kJoints);
        for (int j = 0; j < kJoints; ++j) std::fprintf(file_, ",%d", r.temp[j]);
        std::fputc('\n', file_);
    }

    std::mutex mu_;
    std::vector<Row> pending_;
    std::vector<Row> writing_;  // touched only by the writer thread
    uint32_t segment_ = 0;
    uint64_t dropped_ = 0;
    double prev_mono_s_ = 0;
    bool have_prev_ = false;

    std::atomic<bool> stop_{false};
    std::thread writer_;
    std::FILE* file_ = nullptr;
    bool open_failed_ = false;
};

// Called from ManagerBasedRLEnv::step(). A template so this header does not
// need the env's definition, which includes it.
template <class Env>
inline void record(Env* env, const std::unordered_map<std::string, std::vector<float>>& obs,
                   const std::vector<float>& action, double infer_ms)
{
    using namespace std::chrono;
    Row r;
    r.wall_s = duration<double>(system_clock::now().time_since_epoch()).count();
    r.mono_s = duration<double>(steady_clock::now().time_since_epoch()).count();
    r.step = env->episode_length;
    r.infer_ms = static_cast<float>(infer_ms);

    const auto& d = env->robot->data;
    r.tick = d.tlm_tick;
    r.quat[0] = d.root_quat_w.w();
    r.quat[1] = d.root_quat_w.x();
    r.quat[2] = d.root_quat_w.y();
    r.quat[3] = d.root_quat_w.z();
    for (int i = 0; i < 3; ++i) {
        r.gyro[i] = d.root_ang_vel_b[i];
        r.acc[i] = d.tlm_imu_acc[i];
    }

    // velocity_commands is the last observation term and history is term-major,
    // so the newest command frame is the last three entries of the vector.
    if (!obs.empty()) {
        const auto& o = obs.begin()->second;
        if (o.size() >= 3) {
            for (int i = 0; i < 3; ++i) r.cmd[i] = o[o.size() - 3 + i];
        }
    }

    const auto q_des = env->action_manager->processed_actions();
    const int n = std::min<int>(kJoints, static_cast<int>(d.joint_pos.size()));
    for (int j = 0; j < n; ++j) {
        r.q[j] = d.joint_pos[j];
        r.dq[j] = d.joint_vel[j];
        if (j < static_cast<int>(d.tlm_tau_est.size())) r.tau_est[j] = d.tlm_tau_est[j];
        if (j < static_cast<int>(d.tlm_motor_temp.size())) r.temp[j] = d.tlm_motor_temp[j];
        if (j < static_cast<int>(q_des.size())) r.q_des[j] = q_des[j];
        if (j < static_cast<int>(action.size())) r.action[j] = action[j];
    }
    Recorder::get().push(r);
}

}  // namespace telemetry
