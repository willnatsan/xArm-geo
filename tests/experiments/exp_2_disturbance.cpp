// --- Experiment 2: Smooth Complex Trajectory with Unmodelled Disturbance ---
//
// Three controllers track a WingInspection trajectory while an unmodelled
// constant wrench is injected at the EE link from t = 5 s to t = 10 s.
// The experiment compares how well each controller rejects the disturbance and
// whether it recovers zero steady-state error once the disturbance is removed.
//
// Variants
// --------
//   A) GeometricPController  (velocity / kinematic mode)
//        -- Sees disturbance only through changes in measured joint velocities.
//           Steady-state error expected; no integral term.
//   B) GeometricPDController (torque / dynamic mode)
//        -- Damping-dominated rejection; persistent steady-state offset expected.
//   C) GeometricPIDController (torque / dynamic mode)
//        -- Bhat-intrinsic integral; drives steady-state error to zero after
//           the disturbance is removed (and partially during it).
//
// All variants use the same trajectory (WingInspection, 15 s) and receive
// identical disturbance schedules:
//   ConstantWrench([10, 0, 0, 0, 0, 0] N, body-frame, t_start=5 s, t_end=10 s)
//   applied via Simulation::apply_external_wrench("link_eef", ...).
//
// Controller gains
// ----------------
//   Variant A (kinematic):  kp_pos = kp_rot = 8.0
//   Variants B/C (torque):  kp_pos = 2000, kp_rot = 100,
//                           kd_lin = 280,  kd_ang = 5.0
//   Variant C extra:        ki_lin = ki_ang = 2.0 [1/s]  (Bhat formulation)
//
// The simulation is reset between variants (Phase 0: home) so all three start
// from identical conditions.
//
// Usage
// -----
//   ./build/exp_2_disturbance [--log true] [--variants ABC]
//
// Analysis
// --------
//   pixi run report-exp 2
//   pixi run plot-exp   2

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
#include <xarm_geo/examples/controllers/geometric_pd_controller.h>
#include <xarm_geo/examples/controllers/geometric_pid_controller.h>
#include <xarm_geo/examples/controllers/joint_p_controller.h>
#include <xarm_geo/examples/trajectories/joint_ptp.h>
#include <xarm_geo/examples/trajectories/wing_inspection.h>
#include <xarm_geo/interfaces/simulation.h>
#include <xarm_geo/modelling/collision.h>
#include <xarm_geo/modelling/kinematics.h>
#include <xarm_geo/modelling/optimal_kinematics.h>
#include <xarm_geo/utils/disturbance.h>
#include <xarm_geo/utils/model_builder.h>

#include "experiments/common.h"

namespace {

    using namespace xarm_geo;
    using namespace xarm_geo::experiments;

    // -------------------------------------------------------------------------
    // make_wrench()
    // -------------------------------------------------------------------------
    // Constructs a manifold::SE3::Wrench (CoTangent<SE3d>) from six scalars.
    // CoTangent only accepts a single VectorType argument; this helper builds
    // the underlying Eigen vector and wraps it.
    [[nodiscard]] inline auto make_wrench(double fx, double fy, double fz, double tx, double ty,
                                          double tz) -> manifold::SE3::Wrench {
        manifold::SE3::Wrench w;
        w.coeffs << fx, fy, fz, tx, ty, tz;
        return w;
    }

    constexpr double kTrajDuration = 15.0;
    constexpr double kPhaseDuration = 3.0;

    // Disturbance: 10 N force in EE body-frame +X, active from t=5 s to t=10 s.
    constexpr double kDisturbanceStartS = 5.0;
    constexpr double kDisturbanceEndS = 10.0;

    // Kinematic-mode gains.
    constexpr double kKpPosKin = 8.0;
    constexpr double kKpRotKin = 8.0;

    // Dynamic-mode gains (matches exp_3a and exp_3b for consistency).
    constexpr double kKpPos = 2000.0;
    constexpr double kKpRot = 100.0;
    constexpr double kKdLin = 280.0;
    constexpr double kKdAng = 5.0;

    // Integral gains for PID variant (Bhat formulation, units [1/s]).
    constexpr double kKiLin = 20.0;
    constexpr double kKiAng = 20.0;

    // EE body name for wrench injection (matches the xArm6 MJCF body tree).
    constexpr std::string_view kEEBodyName = "link_eef";

    // -------------------------------------------------------------------------
    // run_approach()
    // -------------------------------------------------------------------------
    // Shared Phase 1: moves from q_home to q_start via joint-PTP.

    auto run_approach(Simulation &sim, const Model &model, Data &data, CollisionModel &col_model,
                      CollisionData &col_data, const manifold::SE3 &traj_start_pose,
                      const Eigen::VectorXd &q_home, Eigen::VectorXd &q_start_out) -> bool {

        const bool ik_ok =
            optimal_inverse_kinematics(model, data, col_model, col_data, q_home, traj_start_pose);
        if (!ik_ok) {
            std::cerr << "[2] IK failed for trajectory start pose.\n";
            return false;
        }
        q_start_out = data.q_out;

        controllers::JointPController joint_ctrl(model);
        joint_ctrl.kp.setConstant(5.0);
        joint_ctrl.use_feedforward = true;

        JointState state(model.dof);
        JointVelocity vel(model.dof);
        if (sim.read(state) != InterfaceStatus::OK) { return false; }

        trajectories::JointPTP approach(q_home, q_start_out, kPhaseDuration);
        if (!run_joint_ptp_sim(sim, model, data, joint_ctrl, approach, state, vel)) {
            std::cerr << "[2] Phase 1 approach failed.\n";
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        return true;
    }

    // -------------------------------------------------------------------------
    // run_return()
    // -------------------------------------------------------------------------
    // Shared Phase 3: returns to q_home.

    auto run_return(Simulation &sim, const Model &model, Data &data, const Eigen::VectorXd &q_home)
        -> bool {
        controllers::JointPController joint_ctrl(model);
        joint_ctrl.kp.setConstant(5.0);
        joint_ctrl.use_feedforward = true;
        JointState state(model.dof);
        JointVelocity vel(model.dof);
        if (sim.read(state) != InterfaceStatus::OK) { return false; }
        trajectories::JointPTP ret(state.q, q_home, kPhaseDuration);
        return run_joint_ptp_sim(sim, model, data, joint_ctrl, ret, state, vel);
    }

    // -------------------------------------------------------------------------
    // Phase 2 loop: kinematic (velocity) mode
    // -------------------------------------------------------------------------

    auto run_phase2_kinematic(Simulation &sim, const Model &model, Data &data,
                              controllers::GeometricPController &ctrl,
                              const trajectories::WingInspection &traj, bool log_data) -> bool {

        const std::string trial_name = diagnostics::make_trial_name(
            "sim", controllers::GeometricPController::kName, trajectories::WingInspection::kName,
            /*constraint=*/ctrl.constraint_aware,
            /*feedforward=*/ctrl.use_feedforward);

        auto logger = make_logger("2", model, trial_name, log_data,
                                  {{"disturbance_start_s", kDisturbanceStartS},
                                   {"disturbance_end_s", kDisturbanceEndS},
                                   {"disturbance_force_x_N", 10.0}});

        const ConstantWrench disturbance(make_wrench(10.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                                         kDisturbanceStartS, kDisturbanceEndS);

        const double physics_dt = kSimulationPhysicsPeriodS;
        double render_dt = 1.0 / 60.0;
        double last_render_t = 0.0;
        const auto ctrl_dt_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::duration<double>(kSimulationControlPeriodS));

        PerfStats perf;
        perf.reserve(
            static_cast<std::size_t>(kTrajDuration / (kSafetyDecimationFactor * physics_dt)) + 32);

        JointState state(model.dof);
        JointVelocity vel(model.dof);
        double t = 0.0;
        std::int64_t tick = 0;
        TaskTarget task_target;

        while (t < kTrajDuration && sim.is_running()) {
            const auto step_start = std::chrono::steady_clock::now();
            if (sim.read(state) != InterfaceStatus::OK) { break; }

            // Apply or clear the external wrench every physics step.
            const manifold::SE3::Wrench w = disturbance.evaluate(t);
            if (w.coeffs.norm() > 0.0) {
                sim.apply_external_wrench(std::string(kEEBodyName), w);
            } else {
                sim.clear_external_wrenches();
            }

            if (tick % kSafetyDecimationFactor == 0) {
                if (traj.evaluate(t, task_target) != TrajectoryStatus::OK) { break; }
                const TaskControllerContext ctx{state, task_target, ctrl_dt_ns};
                const auto t0 = std::chrono::steady_clock::now();
                const ControllerStatus cs = ctrl.update(model, data, ctx, vel);
                const auto t1 = std::chrono::steady_clock::now();
                perf.record(t0, t1);

                if (cs != ControllerStatus::OK) {
                    std::cerr << "[2A] Controller failed at t=" << t << "\n";
                    break;
                }
                if (sim.write(vel) != InterfaceStatus::OK) { break; }

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
        sim.clear_external_wrenches();
        logger.reset();
        perf.report("[Phase 2] GeometricPController (kinematic) + disturbance");
        if (log_data) {
            std::cout << "  -> Logged: tests/results/exp_2/" << trial_name << ".csv\n";
        }
        return true;
    }

    // -------------------------------------------------------------------------
    // Phase 2 loop: dynamic (torque) mode
    // -------------------------------------------------------------------------
    // Templated on controller type; handles both GeometricPDController and
    // GeometricPIDController (both derive from DynamicTaskControllerBase).

    template <typename DynCtrl>
    auto run_phase2_torque(Simulation &sim, const Model &model, Data &data, DynCtrl &ctrl,
                           std::string_view controller_name,
                           const trajectories::WingInspection &traj, bool log_data) -> bool {

        const std::string trial_name = diagnostics::make_trial_name(
            "sim", controller_name, trajectories::WingInspection::kName,
            /*constraint=*/ctrl.constraint_aware,
            /*feedforward=*/ctrl.use_feedforward);

        auto logger = make_logger("2", model, trial_name, log_data,
                                  {{"disturbance_start_s", kDisturbanceStartS},
                                   {"disturbance_end_s", kDisturbanceEndS},
                                   {"disturbance_force_x_N", 10.0}});

        const ConstantWrench disturbance(make_wrench(10.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                                         kDisturbanceStartS, kDisturbanceEndS);

        const double physics_dt = kSimulationPhysicsPeriodS;
        double render_dt = 1.0 / 60.0;
        double last_render_t = 0.0;
        const auto ctrl_dt_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::duration<double>(kSimulationControlPeriodS));

        PerfStats perf;
        perf.reserve(
            static_cast<std::size_t>(kTrajDuration / (kSafetyDecimationFactor * physics_dt)) + 32);

        JointState state(model.dof);
        JointTorque tau(model.dof);
        double t = 0.0;
        std::int64_t tick = 0;
        TaskTarget task_target;

        while (t < kTrajDuration && sim.is_running()) {
            const auto step_start = std::chrono::steady_clock::now();
            if (sim.read(state) != InterfaceStatus::OK) { break; }

            const manifold::SE3::Wrench w = disturbance.evaluate(t);
            if (w.coeffs.norm() > 0.0) {
                sim.apply_external_wrench(std::string(kEEBodyName), w);
            } else {
                sim.clear_external_wrenches();
            }

            if (tick % kSafetyDecimationFactor == 0) {
                if (traj.evaluate(t, task_target) != TrajectoryStatus::OK) { break; }
                const TaskControllerContext ctx{state, task_target, ctrl_dt_ns};
                const auto t0 = std::chrono::steady_clock::now();
                const ControllerStatus cs = ctrl.update(model, data, ctx, tau);
                const auto t1 = std::chrono::steady_clock::now();
                perf.record(t0, t1);

                if (cs != ControllerStatus::OK) {
                    std::cerr << "[2] Torque controller failed at t=" << t << "\n";
                    break;
                }
                if (sim.write(tau) != InterfaceStatus::OK) { break; }

                if (logger) {
                    diagnostics::LogSample s;
                    diagnostics::fill_task_sample(s, t, tick, state, task_target, data);
                    s.controller_status = static_cast<std::uint8_t>(cs);
                    diagnostics::fill_torque_diagnostics(s, ctrl);
                    // Log the Bhat integrator state when the controller exposes it
                    // (GeometricPIDController only; GeometricPDController does not).
                    if constexpr (requires { ctrl.integrator_state(); }) {
                        diagnostics::fill_integrator_diagnostics(s, ctrl.integrator_state());
                    }
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
        sim.clear_external_wrenches();
        logger.reset();
        perf.report(std::string("[Phase 2] ") + std::string(controller_name) +
                    " (torque) + disturbance");
        if (log_data) {
            std::cout << "  -> Logged: tests/results/exp_2/" << trial_name << ".csv\n";
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
    std::string variants_str = "ABC";

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
                      << "  --variants <ABC>      Variants to run (default: ABC)\n"
                      << "                          A = GeometricPController  (velocity)\n"
                      << "                          B = GeometricPDController (torque)\n"
                      << "                          C = GeometricPIDController (torque)\n";
            return 0;
        }
    }

    setup_results_dir("2");

    // Gravity is needed for torque mode (RNEA bias compensation).
    // We set it on the model up-front; variant A (kinematic) ignores it.
    Model model = build_model(6, "XI130412C23L45");
    model.gravity = Eigen::Vector3d{0.0, 0.0, -9.81};

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
    const trajectories::WingInspection traj(anchor, kTrajDuration);

    // Pre-compute the trajectory start pose once for IK.
    TaskTarget traj_start;
    if (traj.evaluate(0.0, traj_start) != TrajectoryStatus::OK) { return 1; }

    std::cout << "=== Experiment 2: WingInspection + ConstantWrench Disturbance ===\n"
              << "Disturbance: 10 N body-frame +X from t=" << kDisturbanceStartS
              << " s to t=" << kDisturbanceEndS << " s\n"
              << "Variants: " << variants_str << "\n"
              << "Log data: " << (log_data ? "yes" : "no") << "\n\n";

    // -------------------------------------------------------------------------
    // Variant A — GeometricPController (kinematic / velocity mode)
    // -------------------------------------------------------------------------
    if (variants_str.find('A') != std::string::npos) {
        std::cout << "--- Variant A: GeometricPController (velocity mode) ---\n";

        // Kinematic mode: gravity compensation handled by MuJoCo actuators.
        // No mode switch needed; sim starts in VELOCITY mode by default.

        Eigen::VectorXd q_start;
        if (!run_approach(sim, model, data, col_model, col_data, traj_start.pose, q_home,
                          q_start)) {
            return 1;
        }

        controllers::GeometricPController ctrl_a(model);
        ctrl_a.gains.kp_pos.setConstant(kKpPosKin);
        ctrl_a.gains.kp_rot.setConstant(kKpRotKin);
        ctrl_a.use_feedforward = true;
        ctrl_a.constraint_aware = false;

        std::cout << "  [Phase 2] Executing with disturbance...\n";
        if (!run_phase2_kinematic(sim, model, data, ctrl_a, traj, log_data)) {
            std::cerr << "[2] Variant A failed.\n";
            return 1;
        }

        std::cout << "  [Phase 3] Returning home...\n";
        if (!run_return(sim, model, data, q_home)) { return 1; }
        std::cout << "--- Variant A complete. ---\n\n";
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    // -------------------------------------------------------------------------
    // Variant B — GeometricPDController (torque mode)
    // -------------------------------------------------------------------------
    if (variants_str.find('B') != std::string::npos) {
        std::cout << "--- Variant B: GeometricPDController (torque mode) ---\n";

        // Switch to TORQUE mode; the Model already has gravity set.
        sim.set_control_mode(ControlMode::TORQUE);

        Eigen::VectorXd q_start;
        // Phase 1 uses joint-PTP in velocity mode; switch before approach.
        sim.set_control_mode(ControlMode::VELOCITY);
        if (!run_approach(sim, model, data, col_model, col_data, traj_start.pose, q_home,
                          q_start)) {
            return 1;
        }
        sim.set_control_mode(ControlMode::TORQUE);

        controllers::GeometricPDController ctrl_b(model);
        ctrl_b.gains.kp_pos.setConstant(kKpPos);
        ctrl_b.gains.kp_rot.setConstant(kKpRot);
        ctrl_b.gains.kd_lin.setConstant(kKdLin);
        ctrl_b.gains.kd_ang.setConstant(kKdAng);
        ctrl_b.use_feedforward = true;
        ctrl_b.constraint_aware = false;

        std::cout << "  [Phase 2] Executing with disturbance...\n";
        if (!run_phase2_torque(sim, model, data, ctrl_b, controllers::GeometricPDController::kName,
                               traj, log_data)) {
            std::cerr << "[2] Variant B failed.\n";
            return 1;
        }

        std::cout << "  [Phase 3] Returning home...\n";
        sim.set_control_mode(ControlMode::VELOCITY);
        if (!run_return(sim, model, data, q_home)) { return 1; }
        std::cout << "--- Variant B complete. ---\n\n";
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    // -------------------------------------------------------------------------
    // Variant C — GeometricPIDController (torque mode)
    // -------------------------------------------------------------------------
    if (variants_str.find('C') != std::string::npos) {
        std::cout << "--- Variant C: GeometricPIDController (torque mode) ---\n";

        sim.set_control_mode(ControlMode::VELOCITY);
        Eigen::VectorXd q_start;
        if (!run_approach(sim, model, data, col_model, col_data, traj_start.pose, q_home,
                          q_start)) {
            return 1;
        }
        sim.set_control_mode(ControlMode::TORQUE);

        controllers::GeometricPIDController ctrl_c(model);
        ctrl_c.gains.kp_pos.setConstant(kKpPos);
        ctrl_c.gains.kp_rot.setConstant(kKpRot);
        ctrl_c.gains.kd_lin.setConstant(kKdLin);
        ctrl_c.gains.kd_ang.setConstant(kKdAng);
        // Bhat-intrinsic integrator gains [1/s]; tau_i ~ 0.5 s.
        ctrl_c.gains.ki_lin.setConstant(kKiLin);
        ctrl_c.gains.ki_ang.setConstant(kKiAng);
        ctrl_c.use_feedforward = true;
        ctrl_c.constraint_aware = false;
        ctrl_c.reset();  // ensure integrator state is zero at trajectory start

        std::cout << "  [Phase 2] Executing with disturbance...\n";
        if (!run_phase2_torque(sim, model, data, ctrl_c, controllers::GeometricPIDController::kName,
                               traj, log_data)) {
            std::cerr << "[2] Variant C failed.\n";
            return 1;
        }

        std::cout << "  [Phase 3] Returning home...\n";
        sim.set_control_mode(ControlMode::VELOCITY);
        if (!run_return(sim, model, data, q_home)) { return 1; }
        std::cout << "--- Variant C complete. ---\n\n";
    }

    sim.shutdown();
    std::cout << "=== Experiment 2 complete. ===\n";
    if (log_data) {
        std::cout << "Results in: tests/results/exp_2/\n"
                  << "  pixi run report-exp 2\n"
                  << "  pixi run plot-exp   2\n";
    }
    return 0;
}
