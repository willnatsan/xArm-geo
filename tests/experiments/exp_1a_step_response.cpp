// --- Experiment 1A: Large Angle Step Response ---
//
// Comparative study of three kinematic (velocity-mode) P controllers tracking
// a large-angle orientation step (178 deg about (1,1,1)/sqrt(3) relative to
// the anchor).
//
// Variants
// --------
//   A) EuclideanPController       -- naive, non-geometric baseline
//   B) GeometricPController       -- SE(3) Lie-group gradient (almost global)
//   C) GeometricPController       -- SE(3) Lie-algebra (log-map) gradient (global)
//
// All variants share identical gains (kp_pos = kp_rot = 8.0) and flags
// (feedforward = true, constraint_aware = false) so execution differences
// attribute purely to the control law.  The sim runs in VELOCITY mode
// throughout; torque is not used.
//
// The LargeOrientationStep trajectory holds the anchor for step_time = 0 s,
// then immediately jumps to the target.  Phase 1 drives the robot to the
// anchor (not the target) so Phase 2 begins with the full 178 deg error at
// t = 0.  Total duration = LargeOrientationStep::hold_duration (8 s), which
// is long enough to capture the full settling transient.
//
// Usage
// -----
//   ./build/exp_1a_step_response [--log true] [--variants ABC]
//
// Analysis
// --------
// After running, from the repo root:
//   pixi run report-exp 1a
//   pixi run plot-exp 1a
//
// Or for individual 4-panel diagnostics:
//   pixi run analyse plot trial tests/results/exp_1a/<trial>.csv

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
#include <xarm_geo/examples/controllers/euclidean_p_controller.h>
#include <xarm_geo/examples/controllers/geometric_p_controller.h>
#include <xarm_geo/examples/controllers/joint_p_controller.h>
#include <xarm_geo/examples/trajectories/joint_ptp.h>
#include <xarm_geo/examples/trajectories/large_orientation_step.h>
#include <xarm_geo/interfaces/simulation.h>
#include <xarm_geo/modelling/collision.h>
#include <xarm_geo/modelling/kinematics.h>
#include <xarm_geo/modelling/optimal_kinematics.h>
#include <xarm_geo/utils/model_builder.h>

#include "experiments/common.h"

namespace {

    using namespace xarm_geo;
    using namespace xarm_geo::experiments;

    // Shared proportional gains (identical across all three variants).
    constexpr double kKpPos = 8.0;
    constexpr double kKpRot = 8.0;

    // Joint-PTP approach / return duration.
    constexpr double kPhaseDuration = 3.0;

    // Step magnitude.  At 178 deg we are 2 deg from the SO(3) antipodal set
    // theta = pi, exercising the trace-gradient saddle (Variants A/B) and
    // pushing Variant C's log-map gradient close to its dr_exp singularity.
    constexpr double kStepAngleDeg = 178.0;

    // --- Variant descriptor ---
    struct Variant {
        char id;           // 'A', 'B', 'C'
        std::string name;  // human-readable for console output
    };

    // -------------------------------------------------------------------------
    // run_variant()
    // -------------------------------------------------------------------------
    // Executes one step-response variant.
    //   controller_name : kName of the controller being run (for trial naming).
    //   gradient        : LieGroup or LieAlgebra (only used for Geometric variants).
    //   use_euclidean   : true -> use EuclideanPController instead of GeometricP.
    //   suffix          : appended to trial name for disambiguation (e.g. "lie").

    template <typename TaskCtrl>
    auto run_variant(Simulation &sim, const Model &model, Data &data, CollisionModel &col_model,
                     CollisionData &col_data, TaskCtrl &ctrl, std::string_view controller_name,
                     const Eigen::VectorXd &q_home, const manifold::SE3 &anchor, bool log_data,
                     std::string_view suffix) -> bool {

        // Build the step trajectory: step_time = 0 so the jump is immediate at
        // t = 0.  hold_duration provides the settling window.
        trajectories::LargeOrientationStep step_traj(anchor,
                                                     Eigen::Vector3d::Ones() / std::sqrt(3.0),
                                                     kStepAngleDeg * std::numbers::pi / 180.0);

        // --- Phase 1: Home -> Anchor (pre-step initial condition) ---
        //
        // Drive to the anchor pose, NOT the target.  This ensures Phase 2 begins
        // with the full orientation error (178 deg) on the very first tick.
        const bool ik_ok = optimal_inverse_kinematics(model, data, col_model, col_data, q_home,
                                                      step_traj.anchor());
        if (!ik_ok) {
            std::cerr << "[1A] IK failed for anchor pose.\n";
            return false;
        }
        const Eigen::VectorXd q_start = data.q_out;

        controllers::JointPController joint_ctrl(model);
        joint_ctrl.kp.setConstant(5.0);
        joint_ctrl.use_feedforward = true;

        JointState state(model.dof);
        JointVelocity vel_target(model.dof);

        if (sim.read(state) != InterfaceStatus::OK) { return false; }

        std::cout << "  [Phase 1] Home -> Anchor...\n";
        trajectories::JointPTP approach(q_home, q_start, kPhaseDuration);
        if (!run_joint_ptp_sim(sim, model, data, joint_ctrl, approach, state, vel_target)) {
            std::cerr << "  [Phase 1] Failed.\n";
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(200));

        // --- Phase 2: Step Response ---
        std::cout << "  [Phase 2] Running step response...\n";

        // Sanity: log initial error so regressions are immediately visible.
        if (sim.read(state) != InterfaceStatus::OK) { return false; }
        data.q = state.q;
        compute_jacobians(model, data);
        const manifold::SE3 g_e = data.ee_pose.inverse() * step_traj.target();
        const manifold::SE3::Twist xi_e0 = g_e.log();
        std::cout << "  [Phase 2] Init ||p_err|| = " << xi_e0.head<3>().norm()
                  << " m,  ||log(R_e)|| = " << xi_e0.tail<3>().norm() << " rad\n";

        const std::string trial_name = diagnostics::make_trial_name(
            "sim", controller_name, trajectories::LargeOrientationStep::kName,
            /*constraint=*/ctrl.constraint_aware,
            /*feedforward=*/ctrl.use_feedforward, suffix);

        auto logger = make_logger("1a", model, trial_name, log_data);

        const double physics_dt = kSimulationPhysicsPeriodS;
        double render_dt = 1.0 / 60.0;
        double last_render_t = 0.0;
        const auto ctrl_dt_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::duration<double>(kSimulationControlPeriodS));

        PerfStats perf;
        {
            const std::size_t expected =
                static_cast<std::size_t>(step_traj.duration() /
                                         (kSafetyDecimationFactor * physics_dt)) +
                32;
            perf.reserve(expected);
        }

        double t = 0.0;
        std::int64_t tick = 0;
        TaskTarget task_target;

        while (t < step_traj.duration() && sim.is_running()) {
            const auto step_start = std::chrono::steady_clock::now();
            if (sim.read(state) != InterfaceStatus::OK) { break; }

            if (tick % kSafetyDecimationFactor == 0) {
                if (step_traj.evaluate(t, task_target) != TrajectoryStatus::OK) { break; }

                const TaskControllerContext ctx{state, task_target, ctrl_dt_ns};
                const auto t0 = std::chrono::steady_clock::now();
                const ControllerStatus cs = ctrl.update(model, data, ctx, vel_target);
                const auto t1 = std::chrono::steady_clock::now();
                perf.record(t0, t1);

                if (cs != ControllerStatus::OK) {
                    std::cerr << "  [1A] Controller update failed at t=" << t << "\n";
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
        logger.reset();  // flush CSV to disk before Phase 3

        const std::string mode_tag =
            "[Phase 2] Kinematic step response (" + std::string(controller_name) + ")";
        perf.report(mode_tag);

        // --- Phase 3: Return Home ---
        std::cout << "  [Phase 3] Returning home...\n";
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        if (sim.read(state) != InterfaceStatus::OK) { return false; }
        trajectories::JointPTP ret(state.q, q_home, kPhaseDuration);
        if (!run_joint_ptp_sim(sim, model, data, joint_ctrl, ret, state, vel_target)) {
            std::cerr << "  [Phase 3] Failed.\n";
            return false;
        }

        if (log_data) {
            std::cout << "  -> Logged: tests/results/exp_1a/" << trial_name << ".csv\n";
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

    // --- CLI parsing ---
    bool log_data = false;
    std::string variants_str = "ABC";  // run all three by default

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
                      << "  --log <false|true>        Log data to CSV (default: false)\n"
                      << "  --variants <ABC>          Variants to run (default: ABC)\n"
                      << "                              A = EuclideanPController\n"
                      << "                              B = GeometricPController (LieGroup)\n"
                      << "                              C = GeometricPController (LieAlgebra)\n";
            return 0;
        }
    }

    // --- Setup ---
    setup_results_dir("1a");

    Model model = build_model(6, "XI130412C23L45");
    // Velocity mode: gravity compensation is handled by MuJoCo actuators.
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

    std::cout << "=== Experiment 1A: Large Angle Step Response ===\n"
              << "Variants: " << variants_str << "\n"
              << "Log data: " << (log_data ? "yes" : "no") << "\n\n";

    // -------------------------------------------------------------------------
    // Variant A — EuclideanPController
    // -------------------------------------------------------------------------
    if (variants_str.find('A') != std::string::npos) {
        std::cout << "--- Variant A: EuclideanPController ---\n";
        controllers::EuclideanPController ctrl_a(model);
        ctrl_a.gains.kp_pos.setConstant(kKpPos);
        ctrl_a.gains.kp_rot.setConstant(kKpRot);
        ctrl_a.use_feedforward = true;
        ctrl_a.constraint_aware = false;

        if (!run_variant(sim, model, data, col_model, col_data, ctrl_a,
                         controllers::EuclideanPController::kName, q_home, anchor, log_data,
                         /*suffix=*/"")) {
            std::cerr << "[1A] Variant A failed.\n";
            return 1;
        }
        std::cout << "--- Variant A complete. ---\n\n";
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    // -------------------------------------------------------------------------
    // Variant B — GeometricPController (Lie-group gradient, default)
    // -------------------------------------------------------------------------
    if (variants_str.find('B') != std::string::npos) {
        std::cout << "--- Variant B: GeometricPController (LieGroup gradient) ---\n";
        controllers::GeometricPController ctrl_b(model);
        ctrl_b.gains.kp_pos.setConstant(kKpPos);
        ctrl_b.gains.kp_rot.setConstant(kKpRot);
        ctrl_b.use_feedforward = true;
        ctrl_b.gradient = GradientType::LieGroup;
        ctrl_b.constraint_aware = false;

        if (!run_variant(sim, model, data, col_model, col_data, ctrl_b,
                         controllers::GeometricPController::kName, q_home, anchor, log_data,
                         /*suffix=*/"")) {
            std::cerr << "[1A] Variant B failed.\n";
            return 1;
        }
        std::cout << "--- Variant B complete. ---\n\n";
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    // -------------------------------------------------------------------------
    // Variant C — GeometricPController (Lie-algebra / log-map gradient)
    // -------------------------------------------------------------------------
    if (variants_str.find('C') != std::string::npos) {
        std::cout << "--- Variant C: GeometricPController (LieAlgebra gradient) ---\n";
        controllers::GeometricPController ctrl_c(model);
        ctrl_c.gains.kp_pos.setConstant(kKpPos);
        ctrl_c.gains.kp_rot.setConstant(kKpRot);
        ctrl_c.use_feedforward = true;
        ctrl_c.gradient = GradientType::LieAlgebra;
        ctrl_c.constraint_aware = false;

        // Suffix "lie" distinguishes this CSV from Variant B (same controller kName).
        if (!run_variant(sim, model, data, col_model, col_data, ctrl_c,
                         controllers::GeometricPController::kName, q_home, anchor, log_data,
                         /*suffix=*/"lie")) {
            std::cerr << "[1A] Variant C failed.\n";
            return 1;
        }
        std::cout << "--- Variant C complete. ---\n\n";
    }

    sim.shutdown();
    std::cout << "=== Experiment 1A complete. ===\n";
    if (log_data) {
        std::cout << "Results in: tests/results/exp_1a/\n"
                  << "  pixi run report-exp 1a\n"
                  << "  pixi run plot-exp   1a\n";
    }
    return 0;
}
