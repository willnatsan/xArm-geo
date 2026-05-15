// --- Experiment 1B: Smooth Complex Trajectory ---
//
// Compares the two gradient formulations of GeometricPController on a smooth,
// continuously-moving SE(3) trajectory (PipeInspection).  No disturbance; no
// safety layer.  The experiment isolates the effect of gradient choice on
// steady-state tracking accuracy and transient behaviour.
//
// Variants
// --------
//   A) GeometricPController   -- Lie-group (trace-based) gradient (default)
//   B) GeometricPController   -- Lie-algebra (log-map) gradient
//
// Both variants share identical gains (kp_pos = kp_rot = 8.0), feedforward
// enabled, constraint_aware = false.  The sim runs in VELOCITY mode.
//
// Trajectory: PipeInspection (circular arc in YZ plane, 15 s).
//
// Usage
// -----
//   ./build/exp_1b_smooth_trajectory [--log true] [--variants AB]
//
// Analysis
// --------
//   pixi run report-exp 1b
//   pixi run plot-exp   1b

#include <chrono>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <numbers>
#include <optional>
#include <string>
#include <thread>

#include <Eigen/Dense>

#include <xarm_geo/core/system.h>
#include <xarm_geo/diagnostics/logger.h>
#include <xarm_geo/examples/controllers/geometric_p_controller.h>
#include <xarm_geo/examples/controllers/joint_p_controller.h>
#include <xarm_geo/examples/trajectories/joint_ptp.h>
#include <xarm_geo/examples/trajectories/pipe_inspection.h>
#include <xarm_geo/interfaces/simulation.h>
#include <xarm_geo/modelling/collision.h>
#include <xarm_geo/modelling/kinematics.h>
#include <xarm_geo/modelling/optimal_kinematics.h>
#include <xarm_geo/utils/model_builder.h>

#include "experiments/common.h"

namespace {

    using namespace xarm_geo;
    using namespace xarm_geo::experiments;

    constexpr double kTrajDuration = 15.0;
    constexpr double kKpPos = 8.0;
    constexpr double kKpRot = 8.0;
    constexpr double kPhaseDuration = 3.0;

    // -------------------------------------------------------------------------
    // run_variant()
    // -------------------------------------------------------------------------

    auto run_variant(Simulation &sim, const Model &model, Data &data, CollisionModel &col_model,
                     CollisionData &col_data, controllers::GeometricPController &ctrl,
                     const trajectories::PipeInspection &traj, const Eigen::VectorXd &q_home,
                     bool log_data, std::string_view suffix) -> bool {

        // IK for trajectory start pose.
        TaskTarget start_target;
        if (traj.evaluate(0.0, start_target) != TrajectoryStatus::OK) { return false; }

        const bool ik_ok =
            optimal_inverse_kinematics(model, data, col_model, col_data, q_home, start_target.pose);
        if (!ik_ok) {
            std::cerr << "[1B] IK failed for trajectory start pose.\n";
            return false;
        }
        const Eigen::VectorXd q_start = data.q_out;

        controllers::JointPController joint_ctrl(model);
        joint_ctrl.kp.setConstant(5.0);
        joint_ctrl.use_feedforward = true;

        JointState state(model.dof);
        JointVelocity vel_target(model.dof);

        if (sim.read(state) != InterfaceStatus::OK) { return false; }

        std::cout << "  [Phase 1] Home -> Trajectory Start...\n";
        trajectories::JointPTP approach(q_home, q_start, kPhaseDuration);
        if (!run_joint_ptp_sim(sim, model, data, joint_ctrl, approach, state, vel_target)) {
            std::cerr << "  [Phase 1] Failed.\n";
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(200));

        // Phase 2: PipeInspection trajectory.
        std::cout << "  [Phase 2] Executing PipeInspection trajectory...\n";

        const std::string trial_name = diagnostics::make_trial_name(
            "sim", controllers::GeometricPController::kName, trajectories::PipeInspection::kName,
            /*constraint=*/ctrl.constraint_aware,
            /*feedforward=*/ctrl.use_feedforward, suffix);

        auto logger = make_logger("1b", model, trial_name, log_data);

        const double physics_dt = kSimulationPhysicsPeriodS;
        double render_dt = 1.0 / 60.0;
        double last_render_t = 0.0;
        const auto ctrl_dt_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::duration<double>(kSimulationControlPeriodS));

        PerfStats perf;
        perf.reserve(
            static_cast<std::size_t>(kTrajDuration / (kSafetyDecimationFactor * physics_dt)) + 32);

        double t = 0.0;
        std::int64_t tick = 0;
        TaskTarget task_target;

        while (t < kTrajDuration && sim.is_running()) {
            const auto step_start = std::chrono::steady_clock::now();
            if (sim.read(state) != InterfaceStatus::OK) { break; }

            if (tick % kSafetyDecimationFactor == 0) {
                if (traj.evaluate(t, task_target) != TrajectoryStatus::OK) { break; }
                const TaskControllerContext ctx{state, task_target, ctrl_dt_ns};
                const auto t0 = std::chrono::steady_clock::now();
                const ControllerStatus cs = ctrl.update(model, data, ctx, vel_target);
                const auto t1 = std::chrono::steady_clock::now();
                perf.record(t0, t1);

                if (cs != ControllerStatus::OK) {
                    std::cerr << "  [1B] Controller update failed at t=" << t << "\n";
                    break;
                }
                if (sim.write(vel_target) != InterfaceStatus::OK) { break; }

                if (logger) {
                    diagnostics::LogSample s;
                    diagnostics::fill_task_sample(s, t, tick, state, task_target, data);
                    s.controller_status = static_cast<std::uint8_t>(cs);
                    diagnostics::fill_velocity_diagnostics(s, ctrl);
                    logger->log(s);
                }
            }

            sim.step();
            t += physics_dt;
            ++tick;

            if (t - last_render_t >= render_dt) {
                sim.set_marker(task_target.pose);
                sim.update_scene();
                sim.render();
                last_render_t = t;
            }
            std::this_thread::sleep_until(step_start + std::chrono::duration<double>(physics_dt));
        }
        logger.reset();

        perf.report("[Phase 2] GeometricPController PipeInspection");

        // Phase 3: Return home.
        std::cout << "  [Phase 3] Returning home...\n";
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        if (sim.read(state) != InterfaceStatus::OK) { return false; }
        trajectories::JointPTP ret(state.q, q_home, kPhaseDuration);
        if (!run_joint_ptp_sim(sim, model, data, joint_ctrl, ret, state, vel_target)) {
            std::cerr << "  [Phase 3] Failed.\n";
            return false;
        }

        if (log_data) {
            std::cout << "  -> Logged: tests/results/exp_1b/" << trial_name << ".csv\n";
        }
        return true;
    }

}  // namespace

// =============================================================================
// main
// =============================================================================

auto main(int argc, char *argv[]) -> int {
    using namespace xarm_geo;
    using namespace xarm_geo::experiments;

    bool log_data = false;
    std::string variants_str = "AB";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "--log" || arg == "-l") && i + 1 < argc) {
            const std::string v = argv[++i];
            log_data = (v == "1" || v == "true");
        } else if (arg == "--variants" && i + 1 < argc) {
            variants_str = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [options]\n"
                      << "Options:\n"
                      << "  --log <false|true>    Log data to CSV (default: false)\n"
                      << "  --variants <AB>       Variants to run (default: AB)\n"
                      << "                          A = LieGroup gradient\n"
                      << "                          B = LieAlgebra gradient\n";
            return 0;
        }
    }

    setup_results_dir("1b");

    Model model = build_model(6, "XI130412C23L45");
    Data data(model);
    Simulation sim(model);

    CollisionModel col_model = build_collision_model(model, true);
    CollisionData col_data(col_model);

    Eigen::VectorXd q_home = Eigen::VectorXd::Zero(model.dof);
    q_home[0] = 1.5 * std::numbers::pi;
    sim.set_joint_positions(q_home);

    JointState state(model.dof);
    if (sim.read(state) != InterfaceStatus::OK) { return 1; }
    data.q = state.q;
    compute_jacobians(model, data);

    const manifold::SE3 anchor = make_anchor_pose(q_home);
    const trajectories::PipeInspection traj(anchor, kTrajDuration);

    std::cout << "=== Experiment 1B: Smooth Complex Trajectory (PipeInspection) ===\n"
              << "Variants: " << variants_str << "\n"
              << "Log data: " << (log_data ? "yes" : "no") << "\n\n";

    // -------------------------------------------------------------------------
    // Variant A — LieGroup gradient
    // -------------------------------------------------------------------------
    if (variants_str.find('A') != std::string::npos) {
        std::cout << "--- Variant A: GeometricPController (LieGroup) ---\n";
        controllers::GeometricPController ctrl_a(model);
        ctrl_a.gains.kp_pos.setConstant(kKpPos);
        ctrl_a.gains.kp_rot.setConstant(kKpRot);
        ctrl_a.use_feedforward = true;
        ctrl_a.gradient = GradientType::LieGroup;
        ctrl_a.constraint_aware = false;

        if (!run_variant(sim, model, data, col_model, col_data, ctrl_a, traj, q_home, log_data,
                         /*suffix=*/"")) {
            std::cerr << "[1B] Variant A failed.\n";
            return 1;
        }
        std::cout << "--- Variant A complete. ---\n\n";
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    // -------------------------------------------------------------------------
    // Variant B — LieAlgebra (log-map) gradient
    // -------------------------------------------------------------------------
    if (variants_str.find('B') != std::string::npos) {
        std::cout << "--- Variant B: GeometricPController (LieAlgebra) ---\n";
        controllers::GeometricPController ctrl_b(model);
        ctrl_b.gains.kp_pos.setConstant(kKpPos);
        ctrl_b.gains.kp_rot.setConstant(kKpRot);
        ctrl_b.use_feedforward = true;
        ctrl_b.gradient = GradientType::LieAlgebra;
        ctrl_b.constraint_aware = false;

        if (!run_variant(sim, model, data, col_model, col_data, ctrl_b, traj, q_home, log_data,
                         /*suffix=*/"lie")) {
            std::cerr << "[1B] Variant B failed.\n";
            return 1;
        }
        std::cout << "--- Variant B complete. ---\n\n";
    }

    sim.shutdown();
    std::cout << "=== Experiment 1B complete. ===\n";
    if (log_data) {
        std::cout << "Results in: tests/results/exp_1b/\n"
                  << "  pixi run report-exp 1b\n"
                  << "  pixi run plot-exp   1b\n";
    }
    return 0;
}
