// --- Experiment 3A: Smooth Complex Trajectory (Simulation + Hardware) ---
//
// Five variants of TiltingCircle tracking spanning two interfaces (sim / hw)
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
//                   GeometricPDController (torque) -> AdmittanceLayer -> safe_velocity_projection
//   E) Simulation,  velocity mode + admittance:
//                   GeometricPDController (torque) -> AdmittanceLayer -> safe_velocity_projection
//                   Sim-domain analogue of Variant D; useful for offline tuning.
//
// Simulation variants (A, B) run by default.  E is opt-in (--variants E).
// Hardware variants (C, D) require --hw <ip> and BUILD_WITH_REAL_XARM.
//
// Admittance (Variants D, E)
// --------------------------
// AdmittanceLayer implements a 1st-order ODE  M_v v_dot + D_v v = tau  with a
// velocity feedforward port that bypasses the low-pass for trajectory tracking.
// M_v = diag(M(q_start)) * mass_scale, D_v = cutoff * M_v.
// Tunable via --cutoff (rad/s, default 30) and --mass-scale (default 1.0).
//
// Usage
// -----
//   ./build/exp_3a_sim_hw [--log true] [--variants AB]
//   ./build/exp_3a_sim_hw [--log true] [--variants E] [--cutoff <w>] [--mass-scale <s>]
//   ./build/exp_3a_sim_hw --hw <ip> [--log true] [--variants CD] [--cutoff <w>] [--mass-scale <s>]
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
#include <xarm_geo/utils/debug.h>
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
    // Used for simulation Variant B where torque is applied directly.
    constexpr double kKpPos = 4000.0;
    constexpr double kKpRot = 60.0;
    constexpr double kKdLin = 280.0;
    constexpr double kKdAng = 2.4;

    // Hardware gains for Variant D (admittance layer).
    // The stateful admittance provides its own low-pass filtering, so these can
    // be set at the same scale as the simulation torque gains (kKpPos / kKpRot).
    // The feedforward velocity bypass means trajectory tracking does not rely
    // solely on the P term, so kd can be tuned independently for damping.
    // Starting point: same as simulation. Raise if steady-state error is large;
    // lower if oscillation persists after tuning the admittance cutoff.
    constexpr double kKpPosHw = 4000.0;
    constexpr double kKpRotHw = 60.0;
    constexpr double kKdLinHw = 280.0;
    constexpr double kKdAngHw = 2.4;

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

    // -------------------------------------------------------------------------
    // Variant E: sim, velocity mode, GeometricPDController + AdmittanceLayer
    // -------------------------------------------------------------------------
    // Sim-domain analogue of Variant D.  The simulation runs in velocity mode
    // so the inner actuator servo dynamics are present, mirroring the hardware
    // cascade.  Useful for tuning --cutoff and --mass-scale offline.
    //
    // Key difference from Variant D: bias_compensation = Full because the sim
    // does not apply internal gravity compensation (unlike the xArm SDK).

    auto run_variant_E(Simulation &sim, const Model &model, Data &data, CollisionModel &col_model,
                       CollisionData &col_data, const Eigen::VectorXd &q_home, bool log_data,
                       double admittance_cutoff, double admittance_mass_scale) -> bool {

        trajectories::TiltingCircle circle_inner(make_anchor_pose(q_home), kCircleBaseDuration);
        TimeScaledTask traj{std::move(circle_inner), kHalfSpeed};

        TaskTarget start_target;
        if (traj.evaluate(0.0, start_target) != TrajectoryStatus::OK) { return false; }
        if (!optimal_inverse_kinematics(model, data, col_model, col_data, q_home,
                                        start_target.pose)) {
            std::cerr << "[3A-E] IK failed.\n";
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

        std::cout << "  [Phase 1] Approaching start...\n";
        trajectories::JointPTP approach(q_home, q_start, kPhaseDuration);
        if (!run_joint_ptp_sim(sim, model, data, joint_ctrl, approach, state, vel)) {
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(200));

        // Sim stays in VELOCITY mode throughout (mirrors hardware mode 4).

        // --- Torque controller (no ASIF; bias_compensation = Full for sim). ---
        controllers::GeometricPDController ctrl(model);
        ctrl.gains.kp_pos.setConstant(kKpPosHw);
        ctrl.gains.kp_rot.setConstant(kKpRotHw);
        ctrl.gains.kd_lin.setConstant(kKdLinHw);
        ctrl.gains.kd_ang.setConstant(kKdAngHw);
        ctrl.use_feedforward = true;
        ctrl.constraint_aware = false;
        // Sim has no internal gravity compensation; inject h(q,v) here.
        ctrl.bias_compensation = BiasCompensation::Full;

        // --- Admittance layer: same sizing as Variant D. ---
        AdmittanceOptions adm_opts;
        adm_opts.mass_diag = make_inertia_diag(model, data, q_start) * admittance_mass_scale;
        adm_opts.damping_diag = adm_opts.mass_diag * admittance_cutoff;
        AdmittanceLayer admittance(model.dof, adm_opts);

        if (sim.read(state) != InterfaceStatus::OK) { return false; }
        admittance.seed_from(state.v);

        OptimalIKOptions proj_opts;
        proj_opts.dt = kSimulationControlPeriodS;

        Eigen::VectorXd v_ff(model.dof);
        Eigen::VectorXd v_projected(model.dof);

        const std::string trial_name = diagnostics::make_trial_name(
            "sim", controllers::GeometricPDController::kName, trajectories::TiltingCircle::kName,
            /*constraint=*/true, ctrl.use_feedforward,
            /*suffix=*/"admittance");
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

        std::cout << "  [Phase 2] GeometricPDController + AdmittanceLayer (sim, velocity)...\n";
        while (t < traj_dur && sim.is_running()) {
            const auto step_start = std::chrono::steady_clock::now();
            if (sim.read(state) != InterfaceStatus::OK) { break; }

            if (tick % kSafetyDecimationFactor == 0) {
                if (traj.evaluate(t, task_target) != TrajectoryStatus::OK) { break; }
                const TaskControllerContext ctx{state, task_target, ctrl_dt_ns};
                const auto t0 = std::chrono::steady_clock::now();

                // 1. Torque PD. compute_jacobians runs inside ctrl.update.
                const ControllerStatus cs = ctrl.update(model, data, ctx, tau);
                if (cs != ControllerStatus::OK) { break; }

                // 2. Velocity feedforward: DLS-IDK of Ad_{g_e} * xi_d.
                {
                    const manifold::SE3 g_e = data.ee_pose.inverse() * task_target.pose;
                    inverse_diff_kinematics(model, data, g_e.Ad() * task_target.twist);
                    v_ff = data.v_out;
                }

                // 3. Admittance step.
                admittance.apply(model, state.q, tau, v_ff, ctrl_dt_ns, vel);

                // 4. Kinematic safety projection.
                const OptimalIKStatus proj_status = safe_velocity_projection(
                    model, data, col_model, col_data, vel.v, v_projected, proj_opts);
                if (proj_status == OptimalIKStatus::OK || proj_status == OptimalIKStatus::RELAXED) {
                    vel.v = v_projected;
                } else {
                    debug::log("safe_velocity_projection failed; using admittance output directly");
                }

                perf.record(t0, std::chrono::steady_clock::now());
                if (sim.write(vel) != InterfaceStatus::OK) { break; }

                if (logger) {
                    diagnostics::LogSample s;
                    diagnostics::fill_task_sample(s, t, tick, state, task_target, data);
                    s.controller_status = static_cast<std::uint8_t>(cs);
                    diagnostics::fill_torque_diagnostics(s, ctrl);
                    diagnostics::fill_admittance_diagnostics(s, admittance);
                    if (proj_status == OptimalIKStatus::OK ||
                        proj_status == OptimalIKStatus::RELAXED) {
                        s.v_safe = v_projected;
                    }
                    s.optik_invoked = true;
                    s.optik_modified =
                        (proj_status == OptimalIKStatus::OK ||
                         proj_status == OptimalIKStatus::RELAXED) &&
                        (v_projected - admittance.last_tick_diagnostics().v_safe).norm() >
                            optik_defaults::kModifiedTol;
                    s.optik_status = static_cast<std::uint8_t>(proj_status);
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
        perf.report("[3A-E] GeometricPDController + AdmittanceLayer (sim, velocity)");

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
    // Cascade:
    //   1. GeometricPDController (torque, constraint_aware=false; see
    //      docs/admittance_and_safety.md for why ASIF does not transfer through
    //      the admittance onto a velocity-mode interface).
    //   2. AdmittanceLayer (1st-order ODE: M_v v_dot + D_v v = tau) with a
    //      velocity feedforward v_ff = DLS-IDK(Ad_{g_e} * xi_d) that bypasses
    //      the low-pass so moving targets are tracked without phase lag.
    //   3. safe_velocity_projection: projects the admittance output to the
    //      nearest joint velocity satisfying hard position/velocity limits and
    //      soft self-collision avoidance (kinematic CBFs via OptIK QP).
    //
    // Admittance sizing: M_v = diag(M(q_start)) * mass_scale, D_v = cutoff * M_v.
    // Default cutoff 30 rad/s (~5 Hz) is well below the inner-servo bandwidth.
    // Tunable via --cutoff <rad/s> (break frequency) and --mass-scale <s> (M_v multiplier).

    auto run_variant_D(Hardware &hw, const Model &model, Data &data, CollisionModel &col_model,
                       CollisionData &col_data, const Eigen::VectorXd &q_home, bool log_data,
                       double admittance_cutoff, double admittance_mass_scale) -> bool {

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

        // --- Torque controller (no ASIF: its torque constraints don't transfer
        //     through the admittance onto the velocity-mode SDK servo). ---
        controllers::GeometricPDController ctrl(model);
        ctrl.gains.kp_pos.setConstant(kKpPosHw);
        ctrl.gains.kp_rot.setConstant(kKpRotHw);
        ctrl.gains.kd_lin.setConstant(kKdLinHw);
        ctrl.gains.kd_ang.setConstant(kKdAngHw);
        ctrl.use_feedforward = true;
        ctrl.constraint_aware = false;
        // xArm SDK applies gravity compensation internally; do not double-count.
        ctrl.bias_compensation = BiasCompensation::None;

        // --- Admittance layer: M_v from robot inertia, D_v = cutoff * M_v. ---
        AdmittanceOptions adm_opts;
        adm_opts.mass_diag = make_inertia_diag(model, data, q_start) * admittance_mass_scale;
        adm_opts.damping_diag = adm_opts.mass_diag * admittance_cutoff;
        AdmittanceLayer admittance(model.dof, adm_opts);

        // Seed from the arm's current velocity so the first tick has no step.
        hw.read(state);
        admittance.seed_from(state.v);

        // --- Kinematic safety filter options (dt must match control period). ---
        OptimalIKOptions proj_opts;
        proj_opts.dt = kHardwareControlPeriodS;

        // Pre-allocated scratch.
        Eigen::VectorXd v_ff(model.dof);
        Eigen::VectorXd v_projected(model.dof);

        const std::string trial_name = diagnostics::make_trial_name(
            "hardware", controllers::GeometricPDController::kName,
            trajectories::TiltingCircle::kName, /*constraint=*/true, ctrl.use_feedforward,
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

            // 1. Torque PD. compute_jacobians runs inside ctrl.update.
            const ControllerStatus cs = ctrl.update(model, data, ctx, tau);
            if (cs != ControllerStatus::OK) {
                std::cerr << "[3A-D] PD controller failed.\n";
                return false;
            }

            // 2. Velocity feedforward: DLS-IDK of Ad_{g_e} * xi_d.
            //    Bypasses the admittance low-pass so trajectory tracking does
            //    not incur phase lag through the filter.
            {
                const manifold::SE3 g_e = data.ee_pose.inverse() * task_target.pose;
                inverse_diff_kinematics(model, data, g_e.Ad() * task_target.twist);
                v_ff = data.v_out;
            }

            // 3. Admittance step (stateful 1st-order ODE + FF bypass + rescale).
            admittance.apply(model, state.q, tau, v_ff, dt_ns, vel);

            // 4. Kinematic safety projection: hard position/velocity limits +
            //    soft self-collision avoidance via kinematic CBF QP.
            //    compute_jacobians was already called inside ctrl.update above.
            const OptimalIKStatus proj_status = safe_velocity_projection(
                model, data, col_model, col_data, vel.v, v_projected, proj_opts);
            if (proj_status == OptimalIKStatus::INFEASIBLE ||
                proj_status == OptimalIKStatus::ERROR) {
                debug::log("safe_velocity_projection failed; using admittance output directly");
                // Admittance output already velocity-limit rescaled; use as fallback.
            } else {
                vel.v = v_projected;
            }

            if (hw.write(vel) != InterfaceStatus::OK) { return false; }

            if (logger) {
                diagnostics::LogSample s;
                diagnostics::fill_task_sample(s, t, tick, state, task_target, data);
                s.controller_status = static_cast<std::uint8_t>(cs);
                // Torque triplet (tau_ctrl, tau_des, tau_safe) from the PD controller.
                diagnostics::fill_torque_diagnostics(s, ctrl);
                // Velocity triplet from admittance: v_ctrl=v_state, v_des=v_state+v_ff,
                // v_safe=post-rescale. v_safe is overwritten below if projection ran.
                diagnostics::fill_admittance_diagnostics(s, admittance);
                // If projection succeeded, update v_safe to reflect the projected value.
                if (proj_status == OptimalIKStatus::OK || proj_status == OptimalIKStatus::RELAXED) {
                    s.v_safe = v_projected;
                }
                s.optik_invoked = true;
                s.optik_modified =
                    (proj_status == OptimalIKStatus::OK ||
                     proj_status == OptimalIKStatus::RELAXED) &&
                    (v_projected - admittance.last_tick_diagnostics().v_safe).norm() >
                        optik_defaults::kModifiedTol;
                s.optik_status = static_cast<std::uint8_t>(proj_status);
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
    // Admittance cutoff frequency (rad/s).  D_v = cutoff * M_v(q_start).
    double admittance_cutoff = 30.0;
    // Uniform scale on M_v; 1.0 = robot inertia.  Raise to slow the filter.
    double admittance_mass_scale = 1.0;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "--log" || arg == "-l") && i + 1 < argc) {
            const std::string v = argv[++i];
            log_data = (v == "1" || v == "true");
        } else if (arg == "--variants" && i + 1 < argc) {
            variants_str = argv[++i];
        } else if (arg == "--hw" && i + 1 < argc) {
            robot_ip = argv[++i];
        } else if (arg == "--cutoff" && i + 1 < argc) {
            admittance_cutoff = std::stod(argv[++i]);
        } else if (arg == "--mass-scale" && i + 1 < argc) {
            admittance_mass_scale = std::stod(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            std::cout
                << "Usage: " << argv[0] << " [options]\n"
                << "Options:\n"
                << "  --log <false|true>     Log data to CSV (default: false)\n"
                << "  --variants <ABCDE>     Variants to run\n"
                << "                           A = Sim,  velocity, GeometricP\n"
                << "                           B = Sim,  torque,   GeometricPD\n"
                << "                           C = Hw,   velocity, GeometricP + OptIK\n"
                << "                           D = Hw,   velocity, GeometricPD + Admittance\n"
                << "                           E = Sim,  velocity, GeometricPD + Admittance\n"
                << "  --hw <ip>              Robot IP for hardware variants\n"
                << "  --cutoff <w>           Admittance break frequency (rad/s, default: 30.0)\n"
                << "                           D_v = cutoff * M_v(q_start)  [variants D, E]\n"
                << "  --mass-scale <s>       Uniform scale on M_v (default: 1.0)  [variants D, "
                   "E]\n";
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

    // ---- Simulation variants (A, B, E) -------------------------------------
    const bool want_sim = variants_str.find('A') != std::string::npos ||
                          variants_str.find('B') != std::string::npos ||
                          variants_str.find('E') != std::string::npos;

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
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }

        if (variants_str.find('E') != std::string::npos) {
            std::cout << "--- Variant E: GeometricPDController + Admittance (sim, velocity) ---\n"
                      << "  cutoff = " << admittance_cutoff << " rad/s"
                      << "  mass_scale = " << admittance_mass_scale << "\n";
            if (!run_variant_E(sim, model, data, col_model, col_data, q_home, log_data,
                               admittance_cutoff, admittance_mass_scale)) {
                std::cerr << "[3A] Variant E failed.\n";
                return 1;
            }
            std::cout << "--- Variant E complete. ---\n\n";
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
                      << "  cutoff = " << admittance_cutoff << " rad/s"
                      << "  mass_scale = " << admittance_mass_scale << "\n";
            if (!run_variant_D(hw, model, data, col_model, col_data, q_home, log_data,
                               admittance_cutoff, admittance_mass_scale)) {
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
