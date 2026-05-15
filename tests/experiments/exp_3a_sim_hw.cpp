// --- Experiment 3A: Smooth Complex Trajectory (Simulation + Hardware) ---
//
// Four variants of TiltingCircle tracking spanning two interfaces (sim / hw)
// and two control modalities (velocity / torque).  The goal is to compare
// simulation and hardware behaviour for the same controller, and to show the
// effect of the admittance layer that bridges a torque-mode controller to the
// xArm SDK's velocity-only hardware interface.
//
// Variants
// --------
//   A) Simulation,  velocity mode:  GeometricPController  (OptIK off)
//   B) Simulation,  torque mode:    GeometricPDController (ASIF off)
//   C) Hardware,    velocity mode:  GeometricPController  (OptIK on)
//   D) Hardware,    velocity mode + admittance:
//                   GeometricPDController (torque) -> AdmittanceLayer -> write(v)
//                   (ASIF on the PD controller; admittance translates tau to v)
//
// Simulation variants (A, B) run by default.  Hardware variants (C, D) require
// the --hw <ip> flag and BUILD_WITH_REAL_XARM.
//
// Admittance (Variant D)
// ----------------------
// The xArm SDK only accepts JointVelocity commands.  GeometricPDController
// outputs JointTorque.  AdmittanceLayer maps:
//   v = D_v^{-1} * tau
// followed by a direction-preserving rescale to satisfy |v_i| <= v_max_i.
// Per-joint damping D_v defaults to 5 N.m.s/rad; tunable via --damping.
//
// Usage
// -----
//   ./build/exp_3a_sim_hw [--log true] [--variants AB]
//   ./build/exp_3a_sim_hw --hw <ip> [--log true] [--variants CD] [--damping <d>]
//
// Analysis
// --------
//   pixi run report-exp 3a
//   pixi run plot-exp   3a

#include <chrono>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <numbers>
#include <optional>
#include <string>
#include <thread>

#include <Eigen/Dense>

#include <xarm_geo/control/admittance.h>
#include <xarm_geo/core/system.h>
#include <xarm_geo/diagnostics/logger.h>
#include <xarm_geo/examples/controllers/geometric_p_controller.h>
#include <xarm_geo/examples/controllers/geometric_pd_controller.h>
#include <xarm_geo/examples/controllers/joint_p_controller.h>
#include <xarm_geo/examples/trajectories/joint_ptp.h>
#include <xarm_geo/examples/trajectories/tilting_circle.h>
#include <xarm_geo/interfaces/simulation.h>
#include <xarm_geo/modelling/collision.h>
#include <xarm_geo/modelling/kinematics.h>
#include <xarm_geo/modelling/optimal_kinematics.h>
#include <xarm_geo/trajectory/adapters.h>
#include <xarm_geo/trajectory/validate.h>
#include <xarm_geo/utils/model_builder.h>

#ifdef XARM_GEO_HAS_REAL_XARM
#include <xarm_geo/interfaces/hardware.h>
#endif

#include "experiments/common.h"

namespace {

    using namespace xarm_geo;
    using namespace xarm_geo::experiments;

    // TiltingCircle run at half speed (TimeScaledTask wrapping 15 s curve).
    constexpr double kCircleBaseDuration = 15.0;
    constexpr double kHalfSpeed = 2.0;

    // Torque-mode gains (critically damped; from validate_torque.cpp).
    constexpr double kKpPos = 4000.0;
    constexpr double kKpRot = 60.0;
    constexpr double kKdLin = 280.0;
    constexpr double kKdAng = 2.4;

    // Kinematic-mode gains.
    constexpr double kKpPosKin = 8.0;
    constexpr double kKpRotKin = 8.0;

    // Approach / return duration for joint-PTP phases.
    constexpr double kPhaseDuration = 4.0;  // matches hardware_test.cpp

    // -------------------------------------------------------------------------
    // Variant A: sim, velocity mode, GeometricPController
    // -------------------------------------------------------------------------

    auto run_variant_A(Simulation &sim, const Model &model, Data &data, CollisionModel &col_model,
                       CollisionData &col_data, const Eigen::VectorXd &q_home, bool log_data)
        -> bool {

        trajectories::TiltingCircle circle_inner(make_anchor_pose(q_home), kCircleBaseDuration);
        TimeScaledTask traj{std::move(circle_inner), kHalfSpeed};

        TaskTarget start_target;
        if (traj.evaluate(0.0, start_target) != TrajectoryStatus::OK) { return false; }
        if (!optimal_inverse_kinematics(model, data, col_model, col_data, q_home,
                                        start_target.pose)) {
            std::cerr << "[3A-A] IK failed.\n";
            return false;
        }
        const Eigen::VectorXd q_start = data.q_out;

        controllers::JointPController joint_ctrl(model);
        joint_ctrl.kp.setConstant(5.0);
        joint_ctrl.use_feedforward = true;

        JointState state(model.dof);
        JointVelocity vel(model.dof);
        if (sim.read(state) != InterfaceStatus::OK) { return false; }

        std::cout << "  [Phase 1] Approaching start...\n";
        trajectories::JointPTP approach(q_home, q_start, kPhaseDuration);
        if (!run_joint_ptp_sim(sim, model, data, joint_ctrl, approach, state, vel)) {
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(200));

        controllers::GeometricPController ctrl(model);
        ctrl.gains.kp_pos.setConstant(kKpPosKin);
        ctrl.gains.kp_rot.setConstant(kKpRotKin);
        ctrl.use_feedforward = true;
        ctrl.constraint_aware = false;

        const std::string trial_name = diagnostics::make_trial_name(
            "sim", controllers::GeometricPController::kName, trajectories::TiltingCircle::kName,
            ctrl.constraint_aware, ctrl.use_feedforward);
        auto logger = make_logger("3a", model, trial_name, log_data);

        const double traj_dur = traj.duration();
        const double physics_dt = kSimulationPhysicsPeriodS;
        double render_dt = 1.0 / 60.0;
        double last_render_t = 0.0;
        const auto ctrl_dt_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::duration<double>(kSimulationControlPeriodS));

        PerfStats perf;
        perf.reserve(static_cast<std::size_t>(traj_dur / (kSafetyDecimationFactor * physics_dt)) +
                     32);

        double t = 0.0;
        std::int64_t tick = 0;
        TaskTarget task_target;

        std::cout << "  [Phase 2] GeometricPController (velocity)...\n";
        while (t < traj_dur && sim.is_running()) {
            const auto step_start = std::chrono::steady_clock::now();
            if (sim.read(state) != InterfaceStatus::OK) { break; }

            if (tick % kSafetyDecimationFactor == 0) {
                if (traj.evaluate(t, task_target) != TrajectoryStatus::OK) { break; }
                const TaskControllerContext ctx{state, task_target, ctrl_dt_ns};
                const auto t0 = std::chrono::steady_clock::now();
                const ControllerStatus cs = ctrl.update(model, data, ctx, vel);
                perf.record(t0, std::chrono::steady_clock::now());

                if (cs != ControllerStatus::OK) { break; }
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
        logger.reset();
        perf.report("[3A-A] GeometricPController (sim, velocity)");

        std::cout << "  [Phase 3] Returning home...\n";
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        if (sim.read(state) != InterfaceStatus::OK) { return false; }
        trajectories::JointPTP ret(state.q, q_home, kPhaseDuration);
        if (!run_joint_ptp_sim(sim, model, data, joint_ctrl, ret, state, vel)) { return false; }
        if (log_data) {
            std::cout << "  -> Logged: tests/results/exp_3a/" << trial_name << ".csv\n";
        }
        return true;
    }

    // -------------------------------------------------------------------------
    // Variant B: sim, torque mode, GeometricPDController
    // -------------------------------------------------------------------------

    auto run_variant_B(Simulation &sim, const Model &model, Data &data, CollisionModel &col_model,
                       CollisionData &col_data, const Eigen::VectorXd &q_home, bool log_data)
        -> bool {

        trajectories::TiltingCircle circle_inner(make_anchor_pose(q_home), kCircleBaseDuration);
        TimeScaledTask traj{std::move(circle_inner), kHalfSpeed};

        TaskTarget start_target;
        if (traj.evaluate(0.0, start_target) != TrajectoryStatus::OK) { return false; }
        if (!optimal_inverse_kinematics(model, data, col_model, col_data, q_home,
                                        start_target.pose)) {
            std::cerr << "[3A-B] IK failed.\n";
            return false;
        }
        const Eigen::VectorXd q_start = data.q_out;

        controllers::JointPController joint_ctrl(model);
        joint_ctrl.kp.setConstant(5.0);
        joint_ctrl.use_feedforward = true;

        JointState state(model.dof);
        JointVelocity vel(model.dof);
        JointTorque tau(model.dof);

        if (sim.read(state) != InterfaceStatus::OK) { return false; }

        std::cout << "  [Phase 1] Approaching start (velocity mode)...\n";
        trajectories::JointPTP approach(q_home, q_start, kPhaseDuration);
        if (!run_joint_ptp_sim(sim, model, data, joint_ctrl, approach, state, vel)) {
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(200));

        // Switch to torque mode for Phase 2.
        sim.set_control_mode(ControlMode::TORQUE);

        controllers::GeometricPDController ctrl(model);
        ctrl.gains.kp_pos.setConstant(kKpPos);
        ctrl.gains.kp_rot.setConstant(kKpRot);
        ctrl.gains.kd_lin.setConstant(kKdLin);
        ctrl.gains.kd_ang.setConstant(kKdAng);
        ctrl.use_feedforward = true;
        ctrl.constraint_aware = false;

        const std::string trial_name = diagnostics::make_trial_name(
            "sim", controllers::GeometricPDController::kName, trajectories::TiltingCircle::kName,
            ctrl.constraint_aware, ctrl.use_feedforward);
        auto logger = make_logger("3a", model, trial_name, log_data);

        const double traj_dur = traj.duration();
        const double physics_dt = kSimulationPhysicsPeriodS;
        double render_dt = 1.0 / 60.0;
        double last_render_t = 0.0;
        const auto ctrl_dt_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::duration<double>(kSimulationControlPeriodS));

        PerfStats perf;
        perf.reserve(static_cast<std::size_t>(traj_dur / (kSafetyDecimationFactor * physics_dt)) +
                     32);

        double t = 0.0;
        std::int64_t tick = 0;
        TaskTarget task_target;

        std::cout << "  [Phase 2] GeometricPDController (torque)...\n";
        while (t < traj_dur && sim.is_running()) {
            const auto step_start = std::chrono::steady_clock::now();
            if (sim.read(state) != InterfaceStatus::OK) { break; }

            if (tick % kSafetyDecimationFactor == 0) {
                if (traj.evaluate(t, task_target) != TrajectoryStatus::OK) { break; }
                const TaskControllerContext ctx{state, task_target, ctrl_dt_ns};
                const auto t0 = std::chrono::steady_clock::now();
                const ControllerStatus cs = ctrl.update(model, data, ctx, tau);
                perf.record(t0, std::chrono::steady_clock::now());

                if (cs != ControllerStatus::OK) { break; }
                if (sim.write(tau) != InterfaceStatus::OK) { break; }

                if (logger) {
                    diagnostics::LogSample s;
                    diagnostics::fill_task_sample(s, t, tick, state, task_target, data);
                    s.controller_status = static_cast<std::uint8_t>(cs);
                    diagnostics::fill_torque_diagnostics(s, ctrl);
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
        perf.report("[3A-B] GeometricPDController (sim, torque)");

        // Phase 3: switch back to velocity for the return move.
        std::cout << "  [Phase 3] Returning home (velocity mode)...\n";
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        sim.set_control_mode(ControlMode::VELOCITY);
        if (sim.read(state) != InterfaceStatus::OK) { return false; }
        trajectories::JointPTP ret(state.q, q_home, kPhaseDuration);
        if (!run_joint_ptp_sim(sim, model, data, joint_ctrl, ret, state, vel)) { return false; }
        if (log_data) {
            std::cout << "  -> Logged: tests/results/exp_3a/" << trial_name << ".csv\n";
        }
        return true;
    }

#ifdef XARM_GEO_HAS_REAL_XARM

    // -------------------------------------------------------------------------
    // Variant C: hardware, velocity mode, GeometricPController + OptIK
    // -------------------------------------------------------------------------

    auto run_variant_C(Hardware &hw, const Model &model, Data &data, CollisionModel &col_model,
                       CollisionData &col_data, const Eigen::VectorXd &q_home, bool log_data)
        -> bool {

        trajectories::TiltingCircle circle_inner(make_anchor_pose(q_home), kCircleBaseDuration);
        TimeScaledTask traj{std::move(circle_inner), kHalfSpeed};

        TaskTarget start_target;
        if (traj.evaluate(0.0, start_target) != TrajectoryStatus::OK) { return false; }
        if (!optimal_inverse_kinematics(model, data, col_model, col_data, q_home,
                                        start_target.pose)) {
            std::cerr << "[3A-C] IK failed.\n";
            return false;
        }
        const Eigen::VectorXd q_start = data.q_out;

        controllers::JointPController joint_ctrl(model);
        joint_ctrl.kp.setConstant(5.0);
        joint_ctrl.use_feedforward = true;
        joint_ctrl.constraint_aware = true;

        JointState state(model.dof);
        JointVelocity vel(model.dof);
        hw.read(state);

        std::cout << "  [Phase 0] Home...\n";
        trajectories::JointPTP h_traj(state.q, q_home, kPhaseDuration);
        if (!run_joint_ptp_hw(hw, model, data, joint_ctrl, h_traj, state, vel)) { return false; }

        std::cout << "  [Phase 1] Approaching start...\n";
        trajectories::JointPTP a_traj(q_home, q_start, kPhaseDuration);
        if (!run_joint_ptp_hw(hw, model, data, joint_ctrl, a_traj, state, vel)) { return false; }

        controllers::GeometricPController ctrl(model);
        ctrl.gains.kp_pos.setConstant(kKpPosKin);
        ctrl.gains.kp_rot.setConstant(kKpRotKin);
        ctrl.use_feedforward = true;
        ctrl.constraint_aware = true;
        ctrl.attach_collision(col_model, col_data);
        ctrl.optimal_ik_options.dt = kHardwareControlPeriodS;

        // Pre-flight validation.
        const auto val = validate_trajectory(model, data, col_model, col_data, traj, q_start, ctrl);
        if (val.status != ValidationStatus::OK) {
            std::cerr << "[3A-C] Safety validation failed: " << val.reason << "\n";
            return false;
        }

        const std::string trial_name = diagnostics::make_trial_name(
            "hardware", controllers::GeometricPController::kName,
            trajectories::TiltingCircle::kName, ctrl.constraint_aware, ctrl.use_feedforward);
        auto logger = make_logger("3a", model, trial_name, log_data);

        const double dt = kHardwareControlPeriodS;
        const auto dt_ns =
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::duration<double>(dt));
        auto next_tick = std::chrono::steady_clock::now();
        std::int64_t tick = 0;
        TaskTarget task_target;

        std::cout << "  [Phase 2] GeometricPController (hw, velocity, OptIK)...\n";
        for (double t = 0.0; t < traj.duration() && hw.is_running(); t += dt, ++tick) {
            if (hw.read(state) != InterfaceStatus::OK) { return false; }
            if (traj.evaluate(t, task_target) != TrajectoryStatus::OK) { break; }

            const TaskControllerContext ctx{state, task_target, dt_ns};
            const ControllerStatus cs = ctrl.update(model, data, ctx, vel);
            if (cs != ControllerStatus::OK) {
                std::cerr << "[3A-C] Controller failed.\n";
                return false;
            }
            if (hw.write(vel) != InterfaceStatus::OK) { return false; }

            if (logger) {
                diagnostics::LogSample s;
                diagnostics::fill_task_sample(s, t, tick, state, task_target, data);
                s.controller_status = static_cast<std::uint8_t>(cs);
                diagnostics::fill_velocity_diagnostics(s, ctrl);
                logger->log(s);
            }

            next_tick += std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                std::chrono::duration<double>(dt));
            std::this_thread::sleep_until(next_tick);
        }
        logger.reset();

        std::cout << "  [Phase 3] Returning home...\n";
        hw.read(state);
        trajectories::JointPTP r_traj(state.q, q_home, kPhaseDuration);
        if (!run_joint_ptp_hw(hw, model, data, joint_ctrl, r_traj, state, vel)) { return false; }
        if (log_data) {
            std::cout << "  -> Logged: tests/results/exp_3a/" << trial_name << ".csv\n";
        }
        return true;
    }

    // -------------------------------------------------------------------------
    // Variant D: hardware, velocity mode, GeometricPDController + AdmittanceLayer
    // -------------------------------------------------------------------------
    // GeometricPDController outputs JointTorque; AdmittanceLayer translates it
    // to JointVelocity via  v = D_v^{-1} * tau  followed by a direction-
    // preserving velocity-limit rescale.  The PD controller also has
    // constraint_aware = true so ASIF certifies the torque command *before* the
    // admittance translation.

    auto run_variant_D(Hardware &hw, const Model &model, Data &data, CollisionModel &col_model,
                       CollisionData &col_data, const Eigen::VectorXd &q_home, bool log_data,
                       double admittance_damping) -> bool {

        trajectories::TiltingCircle circle_inner(make_anchor_pose(q_home), kCircleBaseDuration);
        TimeScaledTask traj{std::move(circle_inner), kHalfSpeed};

        TaskTarget start_target;
        if (traj.evaluate(0.0, start_target) != TrajectoryStatus::OK) { return false; }
        if (!optimal_inverse_kinematics(model, data, col_model, col_data, q_home,
                                        start_target.pose)) {
            std::cerr << "[3A-D] IK failed.\n";
            return false;
        }
        const Eigen::VectorXd q_start = data.q_out;

        controllers::JointPController joint_ctrl(model);
        joint_ctrl.kp.setConstant(5.0);
        joint_ctrl.use_feedforward = true;
        joint_ctrl.constraint_aware = true;

        JointState state(model.dof);
        JointVelocity vel(model.dof);
        JointTorque tau(model.dof);
        hw.read(state);

        std::cout << "  [Phase 0] Home...\n";
        trajectories::JointPTP h_traj(state.q, q_home, kPhaseDuration);
        if (!run_joint_ptp_hw(hw, model, data, joint_ctrl, h_traj, state, vel)) { return false; }

        std::cout << "  [Phase 1] Approaching start...\n";
        trajectories::JointPTP a_traj(q_home, q_start, kPhaseDuration);
        if (!run_joint_ptp_hw(hw, model, data, joint_ctrl, a_traj, state, vel)) { return false; }

        controllers::GeometricPDController ctrl(model);
        ctrl.gains.kp_pos.setConstant(kKpPos);
        ctrl.gains.kp_rot.setConstant(kKpRot);
        ctrl.gains.kd_lin.setConstant(kKdLin);
        ctrl.gains.kd_ang.setConstant(kKdAng);
        ctrl.use_feedforward = true;
        ctrl.constraint_aware = true;
        ctrl.attach_collision(col_model, col_data);

        AdmittanceLayer admittance(model.dof, admittance_damping);

        const std::string trial_name = diagnostics::make_trial_name(
            "hardware", controllers::GeometricPDController::kName,
            trajectories::TiltingCircle::kName, ctrl.constraint_aware, ctrl.use_feedforward,
            /*suffix=*/"admittance");
        auto logger = make_logger("3a", model, trial_name, log_data);

        const double dt = kHardwareControlPeriodS;
        const auto dt_ns =
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::duration<double>(dt));
        auto next_tick = std::chrono::steady_clock::now();
        std::int64_t tick = 0;
        TaskTarget task_target;

        std::cout << "  [Phase 2] GeometricPDController + AdmittanceLayer (hw)...\n";
        for (double t = 0.0; t < traj.duration() && hw.is_running(); t += dt, ++tick) {
            if (hw.read(state) != InterfaceStatus::OK) { return false; }
            if (traj.evaluate(t, task_target) != TrajectoryStatus::OK) { break; }

            const TaskControllerContext ctx{state, task_target, dt_ns};
            const ControllerStatus cs = ctrl.update(model, data, ctx, tau);
            if (cs != ControllerStatus::OK) {
                std::cerr << "[3A-D] PD controller failed.\n";
                return false;
            }

            // Translate ASIF-certified tau -> velocity via admittance map.
            // tau (JointTorque) already holds the ASIF-certified command because
            // DynamicTaskControllerBase writes tau_safe into out.tau when
            // constraint_aware = true.  Use the model-aware overload to also
            // enforce |v_i| <= v_max_i via direction-preserving rescale.
            admittance.apply(model, tau, vel);

            if (hw.write(vel) != InterfaceStatus::OK) { return false; }

            if (logger) {
                diagnostics::LogSample s;
                diagnostics::fill_task_sample(s, t, tick, state, task_target, data);
                s.controller_status = static_cast<std::uint8_t>(cs);
                // Torque triplet from the PD controller (with ASIF diagnostics).
                diagnostics::fill_torque_diagnostics(s, ctrl);
                // Also log the admittance-derived velocity that was actually sent to
                // the hardware.  All three triplet entries are the same value because
                // AdmittanceLayer::apply(model, ...) combines the raw map and the
                // direction-preserving velocity-limit rescale in one step with no
                // separately observable intermediate.  The Python analysis suite will
                // read these under the v_ctrl / v_des / v_safe column group.
                s.v_ctrl = vel.v;
                s.v_des = vel.v;
                s.v_safe = vel.v;
                logger->log(s);
            }

            next_tick += std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                std::chrono::duration<double>(dt));
            std::this_thread::sleep_until(next_tick);
        }
        logger.reset();

        std::cout << "  [Phase 3] Returning home...\n";
        hw.read(state);
        trajectories::JointPTP r_traj(state.q, q_home, kPhaseDuration);
        if (!run_joint_ptp_hw(hw, model, data, joint_ctrl, r_traj, state, vel)) { return false; }
        if (log_data) {
            std::cout << "  -> Logged: tests/results/exp_3a/" << trial_name << ".csv\n";
        }
        return true;
    }

#endif  // XARM_GEO_HAS_REAL_XARM

}  // namespace

// =============================================================================
// main
// =============================================================================

auto main(int argc, char *argv[]) -> int {
    using namespace xarm_geo;
    using namespace xarm_geo::experiments;

    bool log_data = false;
    std::string variants_str;  // auto-set below based on --hw presence
    std::string robot_ip;
    double admittance_damping = 5.0;  // N.m.s/rad, conservative default

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "--log" || arg == "-l") && i + 1 < argc) {
            const std::string v = argv[++i];
            log_data = (v == "1" || v == "true");
        } else if (arg == "--variants" && i + 1 < argc) {
            variants_str = argv[++i];
        } else if (arg == "--hw" && i + 1 < argc) {
            robot_ip = argv[++i];
        } else if (arg == "--damping" && i + 1 < argc) {
            admittance_damping = std::stod(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            std::cout
                << "Usage: " << argv[0] << " [options]\n"
                << "Options:\n"
                << "  --log <false|true>     Log data to CSV (default: false)\n"
                << "  --variants <ABCD>      Variants to run\n"
                << "                           A = Sim,  velocity, GeometricP\n"
                << "                           B = Sim,  torque,   GeometricPD\n"
                << "                           C = Hw,   velocity, GeometricP + OptIK\n"
                << "                           D = Hw,   velocity, GeometricPD + Admittance\n"
                << "  --hw <ip>              Robot IP for hardware variants\n"
                << "  --damping <d>          Admittance damping D_v (N.m.s/rad, default: 5.0)\n";
            return 0;
        }
    }

    const bool use_hardware = !robot_ip.empty();

    // Default variant selection: sim variants if no --hw given, hw variants if given.
    if (variants_str.empty()) { variants_str = use_hardware ? "CD" : "AB"; }

    // Guard: hardware variants require the SDK and --hw.
    const bool want_hw =
        variants_str.find('C') != std::string::npos || variants_str.find('D') != std::string::npos;
    if (want_hw && !use_hardware) {
        std::cerr << "[3A] Hardware variants (C, D) require --hw <ip>.\n";
        return 1;
    }

#ifndef XARM_GEO_HAS_REAL_XARM
    if (want_hw) {
        std::cerr << "[3A] Binary was compiled without real xArm SDK support "
                     "(BUILD_WITH_REAL_XARM=OFF). "
                     "Hardware variants are unavailable.\n";
        return 1;
    }
#endif

    setup_results_dir("3a");

    // Gravity always set; velocity-mode variants ignore it in the controller
    // (MuJoCo/SDK handles compensation), but it's needed for torque mode.
    Model model = build_model(6, "XI130412C23L45");
    model.gravity = Eigen::Vector3d{0.0, 0.0, -9.81};

    Data data(model);
    CollisionModel col_model = build_collision_model(model, true);
    CollisionData col_data(col_model);

    Eigen::VectorXd q_home = Eigen::VectorXd::Zero(model.dof);
    q_home[0] = 1.5 * std::numbers::pi;

    std::cout << "=== Experiment 3A: TiltingCircle (Sim + Hardware) ===\n"
              << "Variants: " << variants_str << "\n"
              << "Log data: " << (log_data ? "yes" : "no") << "\n\n";

    // ---- Simulation variants (A, B) ----------------------------------------
    const bool want_sim =
        variants_str.find('A') != std::string::npos || variants_str.find('B') != std::string::npos;

    if (want_sim) {
        Simulation sim(model);
        sim.set_joint_positions(q_home);

        JointState state(model.dof);
        if (sim.read(state) != InterfaceStatus::OK) { return 1; }
        data.q = state.q;
        compute_jacobians(model, data);

        if (variants_str.find('A') != std::string::npos) {
            std::cout << "--- Variant A: GeometricPController (sim, velocity) ---\n";
            if (!run_variant_A(sim, model, data, col_model, col_data, q_home, log_data)) {
                std::cerr << "[3A] Variant A failed.\n";
                return 1;
            }
            std::cout << "--- Variant A complete. ---\n\n";
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }

        if (variants_str.find('B') != std::string::npos) {
            std::cout << "--- Variant B: GeometricPDController (sim, torque) ---\n";
            if (!run_variant_B(sim, model, data, col_model, col_data, q_home, log_data)) {
                std::cerr << "[3A] Variant B failed.\n";
                return 1;
            }
            std::cout << "--- Variant B complete. ---\n\n";
        }

        sim.shutdown();
    }

    // ---- Hardware variants (C, D) -------------------------------------------
#ifdef XARM_GEO_HAS_REAL_XARM
    if (want_hw) {
        Hardware hw(model.dof, robot_ip);
        if (!hw.is_running()) {
            std::cerr << "[3A] Failed to connect to robot at " << robot_ip << "\n";
            return 1;
        }

        JointState state(model.dof);
        hw.read(state);
        data.q = state.q;
        compute_jacobians(model, data);

        if (variants_str.find('C') != std::string::npos) {
            std::cout << "--- Variant C: GeometricPController (hw, velocity + OptIK) ---\n";
            if (!run_variant_C(hw, model, data, col_model, col_data, q_home, log_data)) {
                std::cerr << "[3A] Variant C failed.\n";
                hw.shutdown();
                return 1;
            }
            std::cout << "--- Variant C complete. ---\n\n";
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        }

        if (variants_str.find('D') != std::string::npos) {
            std::cout << "--- Variant D: GeometricPDController + Admittance (hw) ---\n"
                      << "  Admittance damping D_v = " << admittance_damping << " N.m.s/rad\n";
            if (!run_variant_D(hw, model, data, col_model, col_data, q_home, log_data,
                               admittance_damping)) {
                std::cerr << "[3A] Variant D failed.\n";
                hw.shutdown();
                return 1;
            }
            std::cout << "--- Variant D complete. ---\n\n";
        }

        hw.shutdown();
    }
#endif  // XARM_GEO_HAS_REAL_XARM

    std::cout << "=== Experiment 3A complete. ===\n";
    if (log_data) {
        std::cout << "Results in: tests/results/exp_3a/\n"
                  << "  pixi run report-exp 3a\n"
                  << "  pixi run plot-exp   3a\n";
    }
    return 0;
}
