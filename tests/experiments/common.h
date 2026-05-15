#pragma once

// --- Comparative Study: Shared Scaffolding ---
//
// Header-only helpers shared by all five experiment binaries. Covers:
//   - Compile-time timing constants (matching simulation_test.cpp and hardware_test.cpp).
//   - PerfStats: per-tick controller timing (O(1) record, sorted report at end).
//   - make_anchor_pose(): canonical workspace anchor used by both existing tests.
//   - run_joint_ptp_sim(): Phase 0/1/3 sim runner (joint-PTP via JointPController).
//   - run_joint_ptp_hw(): Phase 0/1/3 hardware runner (joint-PTP via JointPController).
//   - setup_results_dir(): creates tests/results/exp_<id>/ if absent.
//   - make_logger(): DataLogger configured for per-experiment output paths.
//
// Usage pattern in each experiment binary:
//   #include "experiments/common.h"
//   // ... setup model, sim/hw ...
//   auto logger = make_logger("1a", model, trial_name);
//
// All helpers are in namespace xarm_geo::experiments.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <numbers>
#include <numeric>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <Eigen/Dense>

#include <xarm_geo/core/system.h>
#include <xarm_geo/diagnostics/logger.h>
#include <xarm_geo/examples/controllers/joint_p_controller.h>
#include <xarm_geo/examples/trajectories/joint_ptp.h>
#include <xarm_geo/interfaces/simulation.h>
#include <xarm_geo/modelling/kinematics.h>

#ifdef XARM_GEO_HAS_REAL_XARM
#include <xarm_geo/interfaces/hardware.h>
#endif

namespace xarm_geo::experiments {

    // -------------------------------------------------------------------------
    // Timing constants
    // -------------------------------------------------------------------------

    // Physics step used by MuJoCo (500 Hz). Controller is decimated relative
    // to this; the safety filter / Opt IK runs at kSimulationControlPeriodS.
    inline constexpr double kSimulationPhysicsPeriodS = 0.002;  // 500 Hz

    // Controller runs every kSafetyDecimationFactor physics steps.
    // Effective controller rate: 4 * 0.002 s = 0.008 s = 125 Hz.
    inline constexpr int kSafetyDecimationFactor = 4;

    // Controller period in simulation (= hardware period for matched rates).
    inline constexpr double kSimulationControlPeriodS =
        kSafetyDecimationFactor * kSimulationPhysicsPeriodS;  // 0.008 s

    // Hardware control period.  Must equal kSimulationControlPeriodS so that
    // the same gain set works on both sim and hardware without re-tuning.
    inline constexpr double kHardwareControlPeriodS = 0.008;  // 125 Hz

    // -------------------------------------------------------------------------
    // PerfStats — per-tick controller timing
    // -------------------------------------------------------------------------

    struct PerfStats {
        std::vector<double> samples_us;

        void reserve(std::size_t n) { samples_us.reserve(n); }

        void record(std::chrono::steady_clock::time_point t0,
                    std::chrono::steady_clock::time_point t1) {
            samples_us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
        }

        void report(std::string_view label) const {
            if (samples_us.empty()) {
                std::cout << label << ": no samples recorded.\n";
                return;
            }
            std::vector<double> sorted = samples_us;
            std::sort(sorted.begin(), sorted.end());
            const std::size_t n = sorted.size();
            const double mean =
                std::accumulate(sorted.begin(), sorted.end(), 0.0) / static_cast<double>(n);
            auto percentile = [&](double p) -> double {
                const std::size_t idx =
                    std::max(std::size_t{0},
                             static_cast<std::size_t>(std::ceil(p * static_cast<double>(n))) - 1);
                return sorted[std::min(idx, n - 1)];
            };
            std::cout << "\n"
                      << label << " (n=" << n << " samples):\n"
                      << "  p50:  " << percentile(0.50) << " us\n"
                      << "  p95:  " << percentile(0.95) << " us\n"
                      << "  p99:  " << percentile(0.99) << " us\n"
                      << "  max:  " << sorted.back() << " us\n"
                      << "  mean: " << mean << " us\n"
                      << "  [budget: " << (kSafetyDecimationFactor * 2000.0)
                      << " us per controller tick  |  2000 us per physics step]\n";
        }
    };

    // -------------------------------------------------------------------------
    // make_anchor_pose()
    // -------------------------------------------------------------------------
    // Returns the canonical workspace anchor used by simulation_test.cpp:
    //   center = (radius * cos(q0), radius * sin(q0), height)
    //   rot    = exp(UnitZ * (q0 - 3*pi/2))
    // where q0 = q_home[0] = 3*pi/2.

    [[nodiscard]] inline auto make_anchor_pose(const Eigen::VectorXd &q_home, double height = 0.35,
                                               double radius = 0.35) -> manifold::SE3 {
        const double q0 = q_home[0];
        const Eigen::Vector3d center(radius * std::cos(q0), radius * std::sin(q0), height);
        const manifold::SO3 rot =
            manifold::SO3::exp(Eigen::Vector3d::UnitZ() * (q0 - (1.5 * std::numbers::pi)));
        return manifold::SE3(rot, center);
    }

    // -------------------------------------------------------------------------
    // run_joint_ptp_sim() — Simulation Phase 0/1/3 runner
    // -------------------------------------------------------------------------
    // Drives a JointPController against a JointPTP trajectory in velocity mode.
    // Decimates controller updates at kSafetyDecimationFactor; renders at ~60 Hz.
    // Returns false on any interface error or early sim shutdown.

    inline auto run_joint_ptp_sim(Simulation &sim, const Model &model, Data &data,
                                  controllers::JointPController &joint_controller,
                                  const trajectories::JointPTP &traj, JointState &state,
                                  JointVelocity &control_target) -> bool {
        const double duration = traj.duration();
        double t = 0.0;
        const double physics_dt = kSimulationPhysicsPeriodS;
        double render_dt = 1.0 / 60.0;
        double last_render_t = 0.0;
        const auto dt_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::duration<double>(kSimulationControlPeriodS));
        JointTarget joint_target(traj.dof());
        std::int64_t tick = 0;

        while (t < duration && sim.is_running()) {
            const auto step_start = std::chrono::steady_clock::now();
            if (sim.read(state) != InterfaceStatus::OK) { return false; }

            if (tick % kSafetyDecimationFactor == 0) {
                if (traj.evaluate(t, joint_target) != TrajectoryStatus::OK) { return false; }
                const JointControllerContext ctx{state, joint_target, dt_ns};
                if (joint_controller.update(model, data, ctx, control_target) !=
                    ControllerStatus::OK) {
                    std::cerr << "JointPController update failed.\n";
                    return false;
                }
                if (sim.write(control_target) != InterfaceStatus::OK) { return false; }
            }

            sim.step();
            t += physics_dt;
            ++tick;

            if (t - last_render_t >= render_dt) {
                sim.update_scene();
                sim.render();
                last_render_t = t;
            }
            std::this_thread::sleep_until(step_start + std::chrono::duration<double>(physics_dt));
        }
        control_target.v.setZero();
        sim.write(control_target);
        sim.step();
        return true;
    }

#ifdef XARM_GEO_HAS_REAL_XARM
    // -------------------------------------------------------------------------
    // run_joint_ptp_hw() — Hardware Phase 0/1/3 runner
    // -------------------------------------------------------------------------
    // Drives a JointPController against any JointTrajectory on real hardware.
    // Rate-limited to kHardwareControlPeriodS via sleep_until.

    template <JointTrajectory T>
    auto run_joint_ptp_hw(Hardware &hw, const Model &model, Data &data,
                          controllers::JointPController &controller, const T &traj,
                          JointState &state, JointVelocity &control_target) -> bool {
        const double duration = traj.duration();
        const double dt = kHardwareControlPeriodS;
        const auto dt_ns =
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::duration<double>(dt));
        JointTarget target(traj.dof());
        auto next_tick = std::chrono::steady_clock::now();

        for (double t = 0.0; t < duration && hw.is_running(); t += dt) {
            if (hw.read(state) != InterfaceStatus::OK) { return false; }
            if (traj.evaluate(t, target) != TrajectoryStatus::OK) { return false; }
            const JointControllerContext ctx{state, target, dt_ns};
            if (controller.update(model, data, ctx, control_target) != ControllerStatus::OK) {
                std::cerr << "JointPController update failed.\n";
                return false;
            }
            if (hw.write(control_target) != InterfaceStatus::OK) { return false; }
            next_tick += std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                std::chrono::duration<double>(dt));
            std::this_thread::sleep_until(next_tick);
        }
        return true;
    }
#endif  // XARM_GEO_HAS_REAL_XARM

    // -------------------------------------------------------------------------
    // setup_results_dir()
    // -------------------------------------------------------------------------
    // Creates tests/results/exp_<id>/ (and any intermediate directories).

    inline void setup_results_dir(std::string_view exp_id) {
        const std::filesystem::path dir =
            std::filesystem::path("tests/results") / ("exp_" + std::string(exp_id));
        std::filesystem::create_directories(dir);
    }

    // -------------------------------------------------------------------------
    // make_logger()
    // -------------------------------------------------------------------------
    // Constructs a DataLogger writing to tests/results/exp_<id>/<trial_name>.csv.
    // Returns std::nullopt when log_data is false.

    [[nodiscard]] inline auto make_logger(std::string_view exp_id, const Model &model,
                                          const std::string &trial_name, bool log_data)
        -> std::optional<diagnostics::DataLogger> {
        if (!log_data) { return std::nullopt; }
        const std::string path =
            "tests/results/exp_" + std::string(exp_id) + "/" + trial_name + ".csv";
        return diagnostics::DataLogger(model, diagnostics::DataLogger::Config{
                                                  .output_path = path,
                                                  .trial_name = trial_name,
                                              });
    }

}  // namespace xarm_geo::experiments
