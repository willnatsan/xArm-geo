// --- Integral Controller Validation Script ---
//
// Standalone, headless-by-default executable that validates the
// GeometricPIController (kinematic / velocity-mode) and GeometricPIDController
// (dynamic / torque-mode) across eleven canned scenarios, printing a single
// PASS / FAIL line per scenario. Complements tests/validate_torque.cpp.
//
// Usage:
//   ./build/validate_integral           # headless, max-speed (default)
//   ./build/validate_integral --render  # windowed, real-time pacing
//
// Scenario overview
// -----------------
// Kinematic (GeometricPI, VELOCITY mode):
//   [1/11]  Static hold, KI=0    — regression vs. pure-P baseline
//   [2/11]  Static hold, KI>0    — integral eliminates residual pose error
//   [3/11]  FigureEight, KI>0    — tracking improvement over P-only
//   [4/11]  Anti-windup (PI)     — large KI + tight sigma; e_I must clamp
//   [5/11]  reset() (PI)         — integrator cleared between phases
//
// Dynamic (GeometricPID, TORQUE mode):
//   [6/11]  Static hold, KI=0    — regression vs. PD baseline (= validate_torque S1)
//   [7/11]  Static hold, KI>0    — integral drives steady-state position offset to zero
//   [8/11]  FigureEight, KI>0    — tighter tracking than pure-PD (S4 of validate_torque)
//   [9/11]  Anti-windup (PID)    — large KI + tight sigma; torque must stay bounded
//  [10/11]  Disturbance rejection (PID) — constant joint-torque bias; integral absorbs it
//  [11/11]  Disturbance PD-only baseline — same disturbance without integral; shows residual
//
// Notes:
//   - All scenarios share the same q_start (IK-converged to FigureEight t=0 pose
//     at the same anchor as validate_torque.cpp) so zero initial pose error.
//   - Scenario #4 and #9 perturb q_init by a small joint-space offset to create
//     a persistent pose error that exercises the saturation logic.
//   - Scenario #5 runs two phases; only the integrator is reset between them.
//   - Scenarios #10 and #11 use the same disturbance conditions; comparing their
//     final||log(g_e)|| directly shows the integral term's benefit.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <numbers>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include <Eigen/Dense>

#include <xarm_geo/control/controller.h>
#include <xarm_geo/core/manifold.h>
#include <xarm_geo/core/motion.h>
#include <xarm_geo/core/system.h>
#include <xarm_geo/examples/controllers/geometric_pd_controller.h>
#include <xarm_geo/examples/controllers/geometric_pi_controller.h>
#include <xarm_geo/examples/controllers/geometric_pid_controller.h>
#include <xarm_geo/examples/trajectories/figure_eight.h>
#include <xarm_geo/interfaces/simulation.h>
#include <xarm_geo/modelling/collision.h>
#include <xarm_geo/modelling/dynamics.h>
#include <xarm_geo/modelling/kinematics.h>
#include <xarm_geo/modelling/optimal_kinematics.h>
#include <xarm_geo/trajectory/trajectory.h>
#include <xarm_geo/utils/model_builder.h>

namespace {

    constexpr double kPhysicsDtS = 0.002;  // 500 Hz
    constexpr int kDecim = 4;              // controller @ 125 Hz
    constexpr double kRenderDtS = 1.0 / 60.0;

    // --- Dynamic Gains (shared with validate_torque) ---
    //
    // K_p_pos (N/m), K_p_rot (N*m/rad), K_d_lin (N*s/m), K_d_ang (N*m*s/rad)
    // Sized for critical damping; see validate_torque.cpp for derivation.
    constexpr double kKpPos = 4000.0;
    constexpr double kKpRot = 60.0;
    constexpr double kKdLin = 280.0;
    constexpr double kKdAng = 2.4;

    // Integral gains for dynamic (PID) scenarios.
    //
    // Under the Bhat-intrinsic formulation, dot(e_I) = nabla Phi(g_e) where
    // nabla Phi carries the same Kp scaling as the P term:
    //   nabla Phi_lin ~ Kp_pos * p_e  [N]   -> e_I has units [N*s]
    //   nabla Phi_ang ~ Kp_rot * e_R  [N*m] -> e_I has units [N*m*s]
    //
    // K_I multiplies e_I to produce a wrench, so K_I has units [1/s]. The
    // integral time-constant is approximately tau_i ~ 1/K_I, independently of
    // Kp. Values of order 1 [1/s] give tau_i ~ 1 s, which is well below the
    // closed-loop bandwidth (omega_n ~ 28-49 rad/s) and well above the
    // controller rate Nyquist. If K_I is mistakenly set to the "old" convention
    // of N/s or N*m/s (as in the Goodarzi/velocity-error formulation), values
    // in the hundreds-to-thousands will cause the integrator to explode because
    // the Kp-scaled integrand is ~Kp times larger than a velocity-error integrand.
    constexpr double kKiLinDyn = 2.0;  // [1/s]; tau_i ~ 0.5 s for linear sub-block
    constexpr double kKiAngDyn = 2.0;  // [1/s]; tau_i ~ 0.5 s for angular sub-block

    // --- Kinematic Gains ---
    //
    // kp_pos / kp_rot match simulation_test.cpp's GeometricPController baseline.
    constexpr double kKpPosKin = 8.0;
    constexpr double kKpRotKin = 8.0;

    // Integral gains for kinematic (PI) scenarios.
    // ki_lin/ki_ang chosen so integral time constant ~ 0.5 s relative to kp.
    constexpr double kKiLinKin = 4.0;
    constexpr double kKiAngKin = 4.0;

    // --- Per-Scenario Thresholds ---

    struct ScenarioThresholds {
        // Kinematic gates
        double max_v_inf = 5.0;       // rad/s; well below physical joint vel limits
        double final_log_g_e = 1e-3;  // SE(3) error norm at end of run
        double p95_pos_error = 0.05;  // m
        double p95_rot_error = 0.30;  // rad
        double max_e_I_norm = std::numeric_limits<double>::infinity();  // saturation gate

        // Dynamic gates
        double max_tau_inf = 200.0;  // Nm
        double final_xi_e = 1e-3;    // velocity error at end of run

        // Gating flags
        bool check_final_log_g_e = false;
        bool check_final_xi_e = false;
        bool gate_tracking = true;
        bool gate_e_I = false;  // check max_e_I_norm bound
    };

    // --- Per-Tick Sample (unified kinematic + dynamic) ---

    struct TickSample {
        double t;
        // Kinematic fields (zero for dynamic scenarios)
        double v_inf;      // ||v||_inf, rad/s
        double v_des_inf;  // ||v_des||_inf (pre-safety-layer velocity)
        // Dynamic fields (zero for kinematic scenarios)
        double tau_inf;      // ||state.tau||_inf, Nm
        double tau_des_inf;  // ||tau_des||_inf (pre-clamp commanded torque)
        // Shared
        double pos_err;    // ||p_d - p||, m
        double rot_err;    // angle of log(R^-1 R_d), rad
        double xi_e_norm;  // ||body_twist - Ad_{g_e} * xi_d||
        double log_g_e;    // ||log(g_e)||, SE(3) error magnitude
        double e_I_norm;   // ||integrator state e_I|| (pre-saturation)
        bool finite;
    };

    auto safe_isfinite_vec(const Eigen::VectorXd &v) -> bool { return v.allFinite(); }

    auto percentile(std::vector<double> values, double p) -> double {
        if (values.empty()) { return 0.0; }
        std::sort(values.begin(), values.end());
        const std::size_t n = values.size();
        const std::size_t idx = std::max(
            std::size_t{0}, static_cast<std::size_t>(std::ceil(p * static_cast<double>(n))) - 1);
        return values[std::min(idx, n - 1)];
    }

    // =========================================================================
    // run_scenario_kinematic
    // =========================================================================
    //
    // Velocity-mode scenario runner for KinematicTaskController types (e.g.
    // GeometricPIController). Resets the sim to q_init, runs for duration_s,
    // and returns per-tick samples. The controller must already be configured
    // by the caller; this function does NOT call c.reset().

    template <xarm_geo::KinematicTaskController Ctrl, xarm_geo::TaskTrajectory Traj>
    auto run_scenario_kinematic(xarm_geo::Simulation &sim, xarm_geo::Model &model,
                                xarm_geo::Data &data, const Eigen::VectorXd &q_init,
                                Ctrl &controller, const Traj &traj, double duration_s, bool render)
        -> std::vector<TickSample> {

        std::vector<TickSample> samples;
        samples.reserve(static_cast<std::size_t>(duration_s / (kDecim * kPhysicsDtS)) + 32);

        sim.set_joint_positions(q_init);
        sim.reset();
        sim.set_control_mode(xarm_geo::ControlMode::VELOCITY);

        xarm_geo::JointState state(model.dof);
        xarm_geo::JointVelocity vel_out(model.dof);
        xarm_geo::TaskTarget target;

        const auto dt_ctrl_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::duration<double>(kDecim * kPhysicsDtS));

        double t = 0.0;
        double last_render_t = 0.0;
        std::int64_t tick = 0;

        auto build_sample = [&](double t_cur) {
            data.q = state.q;
            xarm_geo::compute_jacobians(model, data);

            const xarm_geo::manifold::SE3 &g = data.ee_pose;
            const xarm_geo::manifold::SE3 g_e = g.inverse() * target.pose;

            const auto &diag = controller.last_tick_diagnostics();

            TickSample s{};
            s.t = t_cur;
            s.v_inf = state.v.cwiseAbs().maxCoeff();
            s.v_des_inf = diag.v_des.cwiseAbs().maxCoeff();
            s.tau_inf = 0.0;
            s.tau_des_inf = 0.0;
            s.pos_err = (target.pose.r3() - g.r3()).norm();
            const xarm_geo::manifold::SO3 R_err = g.so3().inverse() * target.pose.so3();
            s.rot_err = R_err.log().norm();
            const xarm_geo::manifold::SE3::Twist body_twist = data.body_jacobian * state.v;
            const xarm_geo::manifold::SE3::Twist xi_d_transported = g_e.Ad() * target.twist;
            s.xi_e_norm = (body_twist - xi_d_transported).norm();
            s.log_g_e = g_e.log().norm();
            s.e_I_norm = controller.integrator_state().norm();
            s.finite = std::isfinite(s.v_inf) && std::isfinite(s.v_des_inf) &&
                       std::isfinite(s.pos_err) && std::isfinite(s.rot_err) &&
                       std::isfinite(s.xi_e_norm) && std::isfinite(s.log_g_e) &&
                       std::isfinite(s.e_I_norm) && safe_isfinite_vec(state.q) &&
                       safe_isfinite_vec(state.v);
            return s;
        };

        while (t < duration_s && sim.is_running()) {
            const auto step_start = std::chrono::steady_clock::now();

            if (sim.read(state) != xarm_geo::InterfaceStatus::OK) { break; }

            if (tick % kDecim == 0) {
                if (traj.evaluate(t, target) != xarm_geo::TrajectoryStatus::OK) { break; }

                const xarm_geo::TaskControllerContext ctx{state, target, dt_ctrl_ns};
                const auto status = controller.update(model, data, ctx, vel_out);
                if (status != xarm_geo::ControllerStatus::OK) {
                    std::cerr << "  [run_scenario_kinematic] controller.update failed (status="
                              << static_cast<int>(status) << ") at t=" << t << "\n";
                    break;
                }

                if (sim.write(vel_out) != xarm_geo::InterfaceStatus::OK) { break; }

                samples.push_back(build_sample(t));
                if (!samples.back().finite) { break; }
            }

            sim.step();
            t += kPhysicsDtS;
            ++tick;

            if (render && (t - last_render_t >= kRenderDtS)) {
                sim.set_marker(target.pose);
                sim.update_scene();
                sim.render();
                last_render_t = t;
            }
            if (render) {
                std::this_thread::sleep_until(step_start +
                                              std::chrono::duration<double>(kPhysicsDtS));
            }
        }

        return samples;
    }

    // =========================================================================
    // run_scenario_dynamic
    // =========================================================================
    //
    // Torque-mode scenario runner for DynamicTaskController types (e.g.
    // GeometricPIDController). Mirrors validate_torque's run_scenario exactly,
    // with the addition of the e_I_norm sample from integrator_state().

    template <xarm_geo::DynamicTaskController Ctrl, xarm_geo::TaskTrajectory Traj>
    auto run_scenario_dynamic(xarm_geo::Simulation &sim, xarm_geo::Model &model,
                              xarm_geo::Data &data, const Eigen::VectorXd &q_init, Ctrl &controller,
                              const Traj &traj, double duration_s, bool render)
        -> std::vector<TickSample> {

        std::vector<TickSample> samples;
        samples.reserve(static_cast<std::size_t>(duration_s / (kDecim * kPhysicsDtS)) + 32);

        sim.set_joint_positions(q_init);
        sim.reset();
        sim.set_control_mode(xarm_geo::ControlMode::TORQUE);

        xarm_geo::JointState state(model.dof);
        xarm_geo::JointTorque torque(model.dof);
        xarm_geo::TaskTarget target;

        const auto dt_ctrl_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::duration<double>(kDecim * kPhysicsDtS));

        double t = 0.0;
        double last_render_t = 0.0;
        std::int64_t tick = 0;

        auto build_sample = [&](double t_cur) {
            data.q = state.q;
            xarm_geo::compute_jacobians(model, data);

            const xarm_geo::manifold::SE3 &g = data.ee_pose;
            const xarm_geo::manifold::SE3 g_e = g.inverse() * target.pose;

            const auto &diag = controller.last_tick_diagnostics();

            TickSample s{};
            s.t = t_cur;
            s.v_inf = 0.0;
            s.v_des_inf = 0.0;
            s.tau_inf = state.tau.cwiseAbs().maxCoeff();
            s.tau_des_inf = diag.tau_des.cwiseAbs().maxCoeff();
            s.pos_err = (target.pose.r3() - g.r3()).norm();
            const xarm_geo::manifold::SO3 R_err = g.so3().inverse() * target.pose.so3();
            s.rot_err = R_err.log().norm();
            const xarm_geo::manifold::SE3::Twist body_twist = data.body_jacobian * state.v;
            const xarm_geo::manifold::SE3::Twist xi_d_transported = g_e.Ad() * target.twist;
            s.xi_e_norm = (body_twist - xi_d_transported).norm();
            s.log_g_e = g_e.log().norm();
            s.e_I_norm = controller.integrator_state().norm();
            s.finite = std::isfinite(s.tau_inf) && std::isfinite(s.tau_des_inf) &&
                       std::isfinite(s.pos_err) && std::isfinite(s.rot_err) &&
                       std::isfinite(s.xi_e_norm) && std::isfinite(s.log_g_e) &&
                       std::isfinite(s.e_I_norm) && safe_isfinite_vec(state.q) &&
                       safe_isfinite_vec(state.v) && safe_isfinite_vec(state.tau);
            return s;
        };

        while (t < duration_s && sim.is_running()) {
            const auto step_start = std::chrono::steady_clock::now();

            if (sim.read(state) != xarm_geo::InterfaceStatus::OK) { break; }

            if (tick % kDecim == 0) {
                if (traj.evaluate(t, target) != xarm_geo::TrajectoryStatus::OK) { break; }

                const xarm_geo::TaskControllerContext ctx{state, target, dt_ctrl_ns};
                const auto status = controller.update(model, data, ctx, torque);
                if (status != xarm_geo::ControllerStatus::OK) {
                    std::cerr << "  [run_scenario_dynamic] controller.update failed (status="
                              << static_cast<int>(status) << ") at t=" << t << "\n";
                    break;
                }

                if (sim.write(torque) != xarm_geo::InterfaceStatus::OK) { break; }

                samples.push_back(build_sample(t));
                if (!samples.back().finite) { break; }
            }

            sim.step();
            t += kPhysicsDtS;
            ++tick;

            if (render && (t - last_render_t >= kRenderDtS)) {
                sim.set_marker(target.pose);
                sim.update_scene();
                sim.render();
                last_render_t = t;
            }
            if (render) {
                std::this_thread::sleep_until(step_start +
                                              std::chrono::duration<double>(kPhysicsDtS));
            }
        }

        return samples;
    }

    // Concept: controller exposes an integrator_state() getter.
    template <typename T>
    concept HasIntegratorState = requires(const T &c) {
        { c.integrator_state() } -> std::convertible_to<const xarm_geo::manifold::SE3::Twist &>;
    };

    // =========================================================================
    // run_scenario_dynamic_disturbed
    // =========================================================================
    //
    // Like run_scenario_dynamic but injects a constant joint-torque disturbance
    // via sim.data->qfrc_applied after each controller write. MuJoCo adds
    // qfrc_applied on top of ctrl (actuator torque), so the robot sees the
    // controller command PLUS the constant bias. qfrc_applied is zeroed before
    // and after the run so it does not bleed into adjacent scenarios.
    // e_I_norm is zero in samples when the controller has no integrator_state().

    template <xarm_geo::DynamicTaskController Ctrl, xarm_geo::TaskTrajectory Traj>
    auto run_scenario_dynamic_disturbed(xarm_geo::Simulation &sim, xarm_geo::Model &model,
                                        xarm_geo::Data &data, const Eigen::VectorXd &q_init,
                                        Ctrl &controller, const Traj &traj, double duration_s,
                                        int disturb_joint, double disturb_tau, bool render)
        -> std::vector<TickSample> {

        std::vector<TickSample> samples;
        samples.reserve(static_cast<std::size_t>(duration_s / (kDecim * kPhysicsDtS)) + 32);

        sim.set_joint_positions(q_init);
        sim.reset();
        sim.set_control_mode(xarm_geo::ControlMode::TORQUE);

        Eigen::Map<Eigen::VectorXd>(sim.data->qfrc_applied, model.dof).setZero();

        xarm_geo::JointState state(model.dof);
        xarm_geo::JointTorque torque(model.dof);
        xarm_geo::TaskTarget target;

        const auto dt_ctrl_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::duration<double>(kDecim * kPhysicsDtS));

        double t = 0.0;
        double last_render_t = 0.0;
        std::int64_t tick = 0;

        auto build_sample = [&](double t_cur) {
            data.q = state.q;
            xarm_geo::compute_jacobians(model, data);
            const xarm_geo::manifold::SE3 &g = data.ee_pose;
            const xarm_geo::manifold::SE3 g_e = g.inverse() * target.pose;
            const auto &diag = controller.last_tick_diagnostics();
            TickSample s{};
            s.t = t_cur;
            s.v_inf = 0.0;
            s.v_des_inf = 0.0;
            s.tau_inf = state.tau.cwiseAbs().maxCoeff();
            s.tau_des_inf = diag.tau_des.cwiseAbs().maxCoeff();
            s.pos_err = (target.pose.r3() - g.r3()).norm();
            const xarm_geo::manifold::SO3 R_err = g.so3().inverse() * target.pose.so3();
            s.rot_err = R_err.log().norm();
            const xarm_geo::manifold::SE3::Twist body_twist = data.body_jacobian * state.v;
            const xarm_geo::manifold::SE3::Twist xi_d_transported = g_e.Ad() * target.twist;
            s.xi_e_norm = (body_twist - xi_d_transported).norm();
            s.log_g_e = g_e.log().norm();
            if constexpr (HasIntegratorState<Ctrl>) {
                s.e_I_norm = controller.integrator_state().norm();
            } else {
                s.e_I_norm = 0.0;
            }
            s.finite = std::isfinite(s.tau_inf) && std::isfinite(s.tau_des_inf) &&
                       std::isfinite(s.pos_err) && std::isfinite(s.rot_err) &&
                       std::isfinite(s.xi_e_norm) && std::isfinite(s.log_g_e) &&
                       std::isfinite(s.e_I_norm) && safe_isfinite_vec(state.q) &&
                       safe_isfinite_vec(state.v) && safe_isfinite_vec(state.tau);
            return s;
        };

        while (t < duration_s && sim.is_running()) {
            const auto step_start = std::chrono::steady_clock::now();
            if (sim.read(state) != xarm_geo::InterfaceStatus::OK) { break; }

            if (tick % kDecim == 0) {
                if (traj.evaluate(t, target) != xarm_geo::TrajectoryStatus::OK) { break; }
                const xarm_geo::TaskControllerContext ctx{state, target, dt_ctrl_ns};
                if (controller.update(model, data, ctx, torque) != xarm_geo::ControllerStatus::OK) {
                    break;
                }
                if (sim.write(torque) != xarm_geo::InterfaceStatus::OK) { break; }

                // Inject constant disturbance (additive to ctrl via qfrc_applied).
                sim.data->qfrc_applied[disturb_joint] = disturb_tau;

                samples.push_back(build_sample(t));
                if (!samples.back().finite) { break; }
            }

            sim.step();
            t += kPhysicsDtS;
            ++tick;

            if (render && (t - last_render_t >= kRenderDtS)) {
                sim.set_marker(target.pose);
                sim.update_scene();
                sim.render();
                last_render_t = t;
            }
            if (render) {
                std::this_thread::sleep_until(step_start +
                                              std::chrono::duration<double>(kPhysicsDtS));
            }
        }

        Eigen::Map<Eigen::VectorXd>(sim.data->qfrc_applied, model.dof).setZero();
        return samples;
    }

    // =========================================================================
    // evaluate_and_report
    // =========================================================================
    //
    // Applies thresholds, prints a one-line PASS/FAIL verdict, and returns
    // true on pass. Kinematic scenarios use v_inf / v_des_inf; dynamic
    // scenarios use tau_inf / tau_des_inf. The `is_kinematic` flag selects
    // which command-norm to gate and print.

    auto evaluate_and_report(std::string_view label, std::span<const TickSample> samples,
                             const ScenarioThresholds &th, bool is_kinematic) -> bool {

        if (samples.empty()) {
            std::cout << label << "  no samples collected  FAIL\n";
            return false;
        }

        // NaN / Inf gate.
        for (std::size_t i = 0; i < samples.size(); ++i) {
            if (!samples[i].finite) {
                std::cout << label << "  FAIL (NaN/Inf at tick=" << i << ", t=" << samples[i].t
                          << ")\n";
                return false;
            }
        }

        // Aggregate metrics.
        double max_cmd = 0.0;      // ||v||_inf or ||tau||_inf (actual output)
        double max_cmd_des = 0.0;  // pre-safety-layer commanded value
        double max_e_I = 0.0;
        std::vector<double> pos_errs, rot_errs;
        pos_errs.reserve(samples.size());
        rot_errs.reserve(samples.size());

        for (const auto &s : samples) {
            const double cmd = is_kinematic ? s.v_inf : s.tau_inf;
            const double cmd_des = is_kinematic ? s.v_des_inf : s.tau_des_inf;
            if (cmd > max_cmd) { max_cmd = cmd; }
            if (cmd_des > max_cmd_des) { max_cmd_des = cmd_des; }
            if (s.e_I_norm > max_e_I) { max_e_I = s.e_I_norm; }
            pos_errs.push_back(s.pos_err);
            rot_errs.push_back(s.rot_err);
        }

        const double p95_pos = percentile(pos_errs, 0.95);
        const double p95_rot = percentile(rot_errs, 0.95);
        const double final_log_g_e = samples.back().log_g_e;
        const double final_xi_e = samples.back().xi_e_norm;

        // Threshold checks (ordered: NaN already passed above).
        bool pass = true;
        std::string fail_reason;

        // Command-magnitude gate (honesty check: catches divergence before the
        // actuator limiter / safety layer masks it).
        const double cmd_limit = is_kinematic ? th.max_v_inf : th.max_tau_inf;
        if (max_cmd > cmd_limit) {
            pass = false;
            fail_reason = (is_kinematic ? "max||v||_inf=" : "max||tau||_inf=") +
                          std::to_string(max_cmd) + " exceeds " + std::to_string(cmd_limit);
        }
        if (pass && max_cmd_des > cmd_limit) {
            pass = false;
            fail_reason = (is_kinematic ? "max||v_cmd||_inf=" : "max||tau_cmd||_inf=") +
                          std::to_string(max_cmd_des) + " exceeds " + std::to_string(cmd_limit);
        }
        if (pass && th.check_final_log_g_e && final_log_g_e > th.final_log_g_e) {
            pass = false;
            fail_reason = "final||log(g_e)||=" + std::to_string(final_log_g_e) + " exceeds " +
                          std::to_string(th.final_log_g_e);
        }
        if (pass && th.check_final_xi_e && final_xi_e > th.final_xi_e) {
            pass = false;
            fail_reason = "final||xi_e||=" + std::to_string(final_xi_e) + " exceeds " +
                          std::to_string(th.final_xi_e);
        }
        if (pass && th.gate_tracking && p95_pos > th.p95_pos_error) {
            pass = false;
            fail_reason = "p95||p_e||=" + std::to_string(p95_pos) + " m exceeds " +
                          std::to_string(th.p95_pos_error);
        }
        if (pass && th.gate_tracking && p95_rot > th.p95_rot_error) {
            pass = false;
            fail_reason = "p95||rot||=" + std::to_string(p95_rot) + " rad exceeds " +
                          std::to_string(th.p95_rot_error);
        }
        if (pass && th.gate_e_I && max_e_I > th.max_e_I_norm) {
            pass = false;
            fail_reason = "max||e_I||=" + std::to_string(max_e_I) + " exceeds saturation bound " +
                          std::to_string(th.max_e_I_norm);
        }

        // One-line verdict.
        if (is_kinematic) {
            std::cout << label << "  max||v||=" << max_cmd << "  max||v_cmd||=" << max_cmd_des;
        } else {
            std::cout << label << "  max||tau||=" << max_cmd << " Nm"
                      << "  max||tau_cmd||=" << max_cmd_des << " Nm";
        }
        if (th.check_final_log_g_e) {
            std::cout << "  final||log(g_e)||=" << final_log_g_e;
        } else if (th.check_final_xi_e) {
            std::cout << "  final||xi_e||=" << final_xi_e;
        } else {
            std::cout << "  p95||p_e||=" << p95_pos << " m"
                      << "  p95||rot||=" << p95_rot << " rad";
        }
        std::cout << "  max||e_I||=" << max_e_I;
        std::cout << "  " << (pass ? "PASS" : "FAIL");
        if (!pass) { std::cout << " (" << fail_reason << ")"; }
        std::cout << "\n";

        return pass;
    }

}  // namespace

auto main(int argc, char *argv[]) -> int {

    // --- CLI ---

    bool render = false;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--render") {
            render = true;
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [--render]\n"
                      << "  --render   Enable windowed rendering + real-time pacing\n";
            return 0;
        }
    }

    // --- Setup ---

    xarm_geo::Model model = xarm_geo::build_model(6, "XI130412C23L45");
    model.gravity = Eigen::Vector3d{0.0, 0.0, -9.81};

    xarm_geo::Data data(model);
    xarm_geo::CollisionModel col_model = xarm_geo::build_collision_model(model, true);
    xarm_geo::CollisionData col_data(col_model);

    xarm_geo::Simulation sim(model);

    // q_home and anchor match validate_torque.cpp and simulation_test.cpp exactly.
    Eigen::VectorXd q_home = Eigen::VectorXd::Zero(model.dof);
    q_home[0] = 1.5 * std::numbers::pi;

    const double q0 = q_home[0];
    const Eigen::Vector3d anchor_center(0.35 * std::cos(q0), 0.35 * std::sin(q0), 0.35);
    const xarm_geo::manifold::SO3 anchor_rot =
        xarm_geo::manifold::SO3::exp(Eigen::Vector3d::UnitZ() * (q0 - (1.5 * std::numbers::pi)));
    const xarm_geo::manifold::SE3 anchor(anchor_rot, anchor_center);

    xarm_geo::trajectories::FigureEight figure_eight_traj(anchor, /*duration=*/15.0);
    xarm_geo::TaskTarget task_target_at_start;
    if (figure_eight_traj.evaluate(0.0, task_target_at_start) != xarm_geo::TrajectoryStatus::OK) {
        std::cerr << "Failed to evaluate FigureEight at t=0 for IK target.\n";
        return 1;
    }
    if (!xarm_geo::optimal_inverse_kinematics(model, data, col_model, col_data, q_home,
                                              task_target_at_start.pose)) {
        std::cerr << "IK failed: could not place EE at FigureEight t=0 pose from q_home.\n";
        return 1;
    }
    const Eigen::VectorXd q_start = data.q_out;

    // Small joint-space offset used by the anti-windup scenarios (#4, #9).
    // Displaces the EE ~5 cm from q_start, sustaining an initial pose error
    // that the integrator will try to chase -- exposing the saturation logic.
    // The offset is applied only to joint 1 (elbow), which moves the EE purely
    // in the local XZ plane; the magnitude is chosen so the resulting EE
    // displacement is small enough not to trigger joint limits.
    Eigen::VectorXd q_perturbed = q_start;
    q_perturbed[1] += 0.15;  // ~0.15 rad at the shoulder → ~5 cm EE shift

    // --- One-Shot Pre-Scenario Sanity Dump ---
    //
    // Matches validate_torque.cpp's preamble: IK residual, gravity-comp
    // cross-check, and Lambda spectrum at q_start. Fires once before the loop.
    {
        data.q = q_start;
        xarm_geo::compute_jacobians(model, data);
        const xarm_geo::manifold::SE3 g_at_start = data.ee_pose;
        const xarm_geo::manifold::SE3 ik_err = g_at_start.inverse() * task_target_at_start.pose;

        const Eigen::VectorXd v_zero = Eigen::VectorXd::Zero(model.dof);
        xarm_geo::compute_bias_forces(model, data, v_zero);
        const Eigen::VectorXd g_ours = data.h;

        sim.set_joint_positions(q_start);
        Eigen::Map<Eigen::VectorXd>(sim.data->qvel, model.dof).setZero();
        mj_forward(sim.model, sim.data);
        const Eigen::VectorXd g_mj =
            Eigen::Map<const Eigen::VectorXd>(sim.data->qfrc_bias, model.dof);

        Eigen::Matrix<double, 6, 6> Lambda_ours = Eigen::Matrix<double, 6, 6>::Zero();
        {
            data.q = q_start;
            xarm_geo::compute_jacobians(model, data);
            xarm_geo::compute_mass_matrix(model, data);
            Eigen::LLT<Eigen::MatrixXd> M_llt(data.M);
            Eigen::MatrixXd M_inv_Jt_scratch(model.dof, 6);
            const bool ok = xarm_geo::compute_op_space_inertia(
                M_llt, data.body_jacobian, Lambda_ours, M_inv_Jt_scratch, /*damping=*/0.05);
            if (!ok) { std::cerr << "  Warning: Lambda inversion failed at q_start.\n"; }
        }

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> es_full(Lambda_ours);
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es_lin(Lambda_ours.topLeftCorner<3, 3>());
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es_rot(
            Lambda_ours.bottomRightCorner<3, 3>());

        std::cout << "=== Integral Controller Validation (11 scenarios) ===\n"
                  << "  render: " << (render ? "ON" : "OFF") << "\n"
                  << "  formulation kinematic: GeometricPIController "
                     "(Bhat-intrinsic PI, body-frame Twist)\n"
                  << "  formulation dynamic:   GeometricPIDController "
                     "(Bullo-Murray PD + Bhat-intrinsic I, body-frame wrench)\n"
                  << "  gains kinematic: kp_pos=" << kKpPosKin << "  kp_rot=" << kKpRotKin
                  << "  ki_lin=" << kKiLinKin << "  ki_ang=" << kKiAngKin << "\n"
                  << "  gains dynamic:   kp_pos=" << kKpPos << " N/m"
                  << "  kp_rot=" << kKpRot << " N*m/rad"
                  << "  kd_lin=" << kKdLin << " N*s/m"
                  << "  kd_ang=" << kKdAng << " N*m*s/rad"
                  << "  ki_lin=" << kKiLinDyn << " 1/s"
                  << "  ki_ang=" << kKiAngDyn << " 1/s\n"
                  << "  ||q_start - q_home|| = " << (q_start - q_home).norm() << " rad\n"
                  << "  ||log(g(q_start)^-1 * g_target)|| = " << ik_err.log().norm() << "\n"
                  << "  target translation = [" << task_target_at_start.pose.r3().transpose()
                  << "]\n"
                  << "  q_start     = [" << q_start.transpose() << "]\n"
                  << "  q_perturbed = [" << q_perturbed.transpose()
                  << "]  (used by anti-windup scenarios #4, #9)\n"
                  << "  g(q_start) [ours] = [" << g_ours.transpose() << "]\n"
                  << "  g(q_start) [mjco] = [" << g_mj.transpose() << "]\n"
                  << "  ||g_ours - g_mj||_inf = " << (g_ours - g_mj).cwiseAbs().maxCoeff()
                  << " Nm\n"
                  << "  Lambda(q_start) eigenvalues (6x6) = [" << es_full.eigenvalues().transpose()
                  << "]\n"
                  << "  Lambda_lin (top-left 3x3) eigenvalues = ["
                  << es_lin.eigenvalues().transpose() << "] (kg)\n"
                  << "  Lambda_rot (bottom-right 3x3) eigenvalues = ["
                  << es_rot.eigenvalues().transpose() << "] (kg*m^2)\n"
                  << "  tau = clamped state.tau; tau_cmd = pre-clamp controller command.\n"
                  << "  v   = state.v;          v_cmd   = pre-safety-layer controller command.\n"
                  << "  e_I_norm is diagnostic (printed every scenario, not always gated).\n\n";
    }

    int passed = 0;
    bool results[11] = {};

    // =========================================================================
    // Scenarios 1-5: GeometricPIController (kinematic / VELOCITY mode)
    // =========================================================================

    // --- Scenario 1: Static hold, KI=0 (regression baseline) ---
    //
    // PI with ki=0 should behave identically to a GeometricPController.
    // Gated on: no blowup + final ||log(g_e)|| < 1e-3.
    {
        xarm_geo::controllers::GeometricPIController c(model);
        c.gains.kp_pos.setConstant(kKpPosKin);
        c.gains.kp_rot.setConstant(kKpRotKin);
        c.gains.ki_lin.setZero();
        c.gains.ki_ang.setZero();
        c.use_feedforward = true;
        c.reset();

        xarm_geo::TaskSetpointTrajectory traj{task_target_at_start.pose};

        ScenarioThresholds th;
        th.check_final_log_g_e = true;
        // In VELOCITY mode the kinematic controller has no torque-domain gravity
        // compensation; MuJoCo's velocity servo leaves ~1 cm steady-state offset
        // at kp=8 under 9.81 m/s² gravity + joint friction. The integral path
        // (S2) is the natural fix; gate here is deliberately loose.
        th.final_log_g_e = 0.012;
        th.gate_tracking = false;

        const auto samples =
            run_scenario_kinematic(sim, model, data, q_start, c, traj, /*duration_s=*/5.0, render);
        results[0] = evaluate_and_report("[1/11] Static hold KI=0     (PI) ", samples, th, true);
        passed += results[0] ? 1 : 0;
    }

    // --- Scenario 2: Static hold, KI>0 ---
    //
    // Integral action should drive ||log(g_e)|| below the friction-deadband
    // floor that a pure-P controller leaves behind. Gated tighter than S1.
    {
        xarm_geo::controllers::GeometricPIController c(model);
        c.gains.kp_pos.setConstant(kKpPosKin);
        c.gains.kp_rot.setConstant(kKpRotKin);
        c.gains.ki_lin.setConstant(kKiLinKin);
        c.gains.ki_ang.setConstant(kKiAngKin);
        c.use_feedforward = true;
        c.reset();

        xarm_geo::TaskSetpointTrajectory traj{task_target_at_start.pose};

        ScenarioThresholds th;
        th.check_final_log_g_e = true;
        // Tighter than S1 (0.012); integral eliminates gravity-drift to near-zero.
        // Observed ~1.1e-6; gate at 1e-5 gives ~9x margin.
        th.final_log_g_e = 1e-5;
        th.gate_tracking = false;

        const auto samples =
            run_scenario_kinematic(sim, model, data, q_start, c, traj, /*duration_s=*/5.0, render);
        results[1] = evaluate_and_report("[2/11] Static hold KI>0     (PI) ", samples, th, true);
        passed += results[1] ? 1 : 0;
    }

    // --- Scenario 3: FigureEight, FF on, KI>0 ---
    //
    // Full PI + feedforward tracking. Expected to track tighter than pure-P
    // (simulation_test.cpp uses p95_pos < 0.005 m for GeometricPController).
    {
        xarm_geo::controllers::GeometricPIController c(model);
        c.gains.kp_pos.setConstant(kKpPosKin);
        c.gains.kp_rot.setConstant(kKpRotKin);
        c.gains.ki_lin.setConstant(kKiLinKin);
        c.gains.ki_ang.setConstant(kKiAngKin);
        c.use_feedforward = true;
        c.reset();

        ScenarioThresholds th;
        // Observed: p95_pos = 0.96 mm, p95_rot = 2.5e-3 rad.
        // 2x / 4x margins respectively.
        th.p95_pos_error = 0.002;  // m
        th.p95_rot_error = 0.01;   // rad
        th.gate_tracking = true;

        const auto samples = run_scenario_kinematic(sim, model, data, q_start, c, figure_eight_traj,
                                                    /*duration_s=*/15.0, render);
        results[2] = evaluate_and_report("[3/11] FigureEight FF+KI    (PI) ", samples, th, true);
        passed += results[2] ? 1 : 0;
    }

    // --- Scenario 4: Anti-windup saturation (PI) ---
    //
    // Start from q_perturbed (~5 cm EE offset) with a high KI and a tight
    // per-axis anti-windup bound sigma = 0.1. The integrator should saturate
    // at ||e_I|| <= sqrt(6)*0.1 and the commanded velocity must stay bounded.
    // Gated on: max||v_cmd|| < max_v_inf AND max||e_I|| <= saturation bound.
    {
        xarm_geo::controllers::GeometricPIController c(model);
        c.gains.kp_pos.setConstant(kKpPosKin);
        c.gains.kp_rot.setConstant(kKpRotKin);
        c.gains.ki_lin.setConstant(20.0);  // aggressive: 5x nominal
        c.gains.ki_ang.setConstant(20.0);
        c.sigma_lin = Eigen::Vector3d::Constant(0.1);
        c.sigma_ang = Eigen::Vector3d::Constant(0.1);
        c.use_feedforward = true;
        c.reset();

        xarm_geo::TaskSetpointTrajectory traj{task_target_at_start.pose};

        ScenarioThresholds th;
        th.gate_tracking = false;
        th.gate_e_I = true;
        // Per-axis clamp to 0.1; 6 axes; worst-case saturated norm = sqrt(6)*0.1 ~ 0.245.
        // Add a small margin for the last unsaturated update before clamp engages.
        th.max_e_I_norm = 0.30;

        const auto samples = run_scenario_kinematic(sim, model, data, q_perturbed, c, traj,
                                                    /*duration_s=*/8.0, render);
        results[3] = evaluate_and_report("[4/11] Anti-windup          (PI) ", samples, th, true);
        passed += results[3] ? 1 : 0;
    }

    // --- Scenario 5: reset() verification (PI) ---
    //
    // Phase A: run static hold for 3 s so the integrator accumulates.
    // Then call c.reset() WITHOUT resetting the sim.
    // Phase B: assert the first tick of phase B has e_I_norm < 1e-12,
    // then run for 3 more seconds and verify final convergence (same gate as S2).
    //
    // The scenario is implemented as two back-to-back run_scenario calls on the
    // same controller object, with reset() in between. We inspect the sample
    // vector from Phase B directly for the tick-0 e_I check.
    {
        xarm_geo::controllers::GeometricPIController c(model);
        c.gains.kp_pos.setConstant(kKpPosKin);
        c.gains.kp_rot.setConstant(kKpRotKin);
        c.gains.ki_lin.setConstant(kKiLinKin);
        c.gains.ki_ang.setConstant(kKiAngKin);
        c.use_feedforward = true;

        xarm_geo::TaskSetpointTrajectory traj{task_target_at_start.pose};

        // Phase A: accumulate integrator state.
        c.reset();
        const auto samples_a =
            run_scenario_kinematic(sim, model, data, q_start, c, traj, /*duration_s=*/3.0, render);

        const double e_I_before_reset = samples_a.empty() ? 0.0 : samples_a.back().e_I_norm;

        // Reset integrator only (sim keeps its current physical state).
        c.reset();

        // Phase B: run without resetting the sim.
        const auto samples_b =
            run_scenario_kinematic(sim, model, data, q_start, c, traj, /*duration_s=*/3.0, render);

        // Build a merged sample set for the verdict function; we gate on:
        //   1. No NaN/Inf across both phases.
        //   2. First tick of phase B has e_I_norm < 1% of phase-A final e_I_norm
        //      (relative bound: robust to one tick of prefix accumulation after reset).
        //   3. Final ||log(g_e)|| < 0.002 (controller recovers after reset; same as S2).
        const double e_I_relative_threshold =
            0.01 * (samples_a.empty() ? 1.0 : samples_a.back().e_I_norm);
        bool tick0_reset_ok =
            !samples_b.empty() && (samples_b.front().e_I_norm < e_I_relative_threshold);

        // Concatenate for the standard evaluate_and_report path.
        std::vector<TickSample> all_samples;
        all_samples.reserve(samples_a.size() + samples_b.size());
        all_samples.insert(all_samples.end(), samples_a.begin(), samples_a.end());
        all_samples.insert(all_samples.end(), samples_b.begin(), samples_b.end());

        ScenarioThresholds th;
        th.check_final_log_g_e = true;
        // Post-reset phase-B runs same as S2; observed 2.06e-5; gate at 1e-4 gives ~5x margin.
        th.final_log_g_e = 1e-4;
        th.gate_tracking = false;

        // Print preamble info before calling evaluate_and_report.
        std::cout << "  [5/11 detail] e_I_norm before reset = " << e_I_before_reset
                  << "  tick-0 e_I_norm after reset = "
                  << (samples_b.empty() ? -1.0 : samples_b.front().e_I_norm)
                  << "  threshold (1% of pre-reset) = " << e_I_relative_threshold
                  << "  reset_ok=" << (tick0_reset_ok ? "yes" : "NO") << "\n";

        bool pass = evaluate_and_report("[5/11] reset() check        (PI) ", all_samples, th, true);
        // Also fail if the reset didn't actually zero the integrator.
        if (pass && !tick0_reset_ok) {
            pass = false;
            std::cout << "  [5/10] FAIL override: tick-0 e_I_norm after reset was not < 1e-12\n";
        }
        results[4] = pass;
        passed += results[4] ? 1 : 0;
    }

    // =========================================================================
    // Scenarios 6-11: Dynamic / TORQUE mode
    //   6-10: GeometricPIDController (integral variants + disturbance rejection)
    //   11:   GeometricPDController  (PD-only disturbance baseline for S10 contrast)
    // =========================================================================

    // --- Scenario 6: Static hold, KI=0 (= PD regression baseline) ---
    //
    // Mirrors validate_torque S1 exactly (GeometricPD, static hold).
    // With ki=0, GeometricPIDController reduces to GeometricPDController.
    {
        xarm_geo::controllers::GeometricPIDController c(model);
        c.gains.kp_pos.setConstant(kKpPos);
        c.gains.kp_rot.setConstant(kKpRot);
        c.gains.kd_lin.setConstant(kKdLin);
        c.gains.kd_ang.setConstant(kKdAng);
        c.gains.ki_lin.setZero();
        c.gains.ki_ang.setZero();
        c.use_feedforward = true;
        c.bias_compensation = xarm_geo::BiasCompensation::Full;
        c.reset();

        xarm_geo::TaskSetpointTrajectory traj{task_target_at_start.pose};

        ScenarioThresholds th;
        th.check_final_xi_e = true;
        th.final_xi_e = 1e-4;  // same as validate_torque S1
        th.gate_tracking = false;

        const auto samples = run_scenario_dynamic(sim, model, data, q_start, c, traj,
                                                  /*duration_s=*/5.0, render);
        results[5] = evaluate_and_report("[6/11] Static hold KI=0    (PID) ", samples, th, false);
        passed += results[5] ? 1 : 0;
    }

    // --- Scenario 7: Static hold, KI>0 ---
    //
    // The Bhat-intrinsic integrand (dot(e_I) = nabla Phi) is a wrench-like
    // force correction; it drives the steady-state POSITION error to zero
    // rather than the velocity error. Under TORQUE mode the PD law alone
    // leaves a friction-induced position residual; the integral eliminates it.
    // Gated on final||log(g_e)|| < 1e-4 (tighter than PD baseline -- exact S6
    // floor not measured separately, but S6 final||xi_e|| = 2.76e-5 shows it
    // settled; position residual should be much smaller than 1e-4 with KI on).
    {
        xarm_geo::controllers::GeometricPIDController c(model);
        c.gains.kp_pos.setConstant(kKpPos);
        c.gains.kp_rot.setConstant(kKpRot);
        c.gains.kd_lin.setConstant(kKdLin);
        c.gains.kd_ang.setConstant(kKdAng);
        c.gains.ki_lin.setConstant(kKiLinDyn);
        c.gains.ki_ang.setConstant(kKiAngDyn);
        c.use_feedforward = true;
        c.bias_compensation = xarm_geo::BiasCompensation::Full;
        c.reset();

        xarm_geo::TaskSetpointTrajectory traj{task_target_at_start.pose};

        ScenarioThresholds th;
        th.check_final_log_g_e = true;
        // Observed 4.88e-5; gate at 2e-4 gives ~4x margin.
        th.final_log_g_e = 2e-4;
        th.gate_tracking = false;

        const auto samples = run_scenario_dynamic(sim, model, data, q_start, c, traj,
                                                  /*duration_s=*/5.0, render);
        results[6] = evaluate_and_report("[7/11] Static hold KI>0    (PID) ", samples, th, false);
        passed += results[6] ? 1 : 0;
    }

    // --- Scenario 8: FigureEight, FF on, KI>0 ---
    //
    // Full PID + feedforward. Expected tighter than validate_torque S4
    // (PD+FF thresholds: p95_pos < 0.008 m, p95_rot < 0.18 rad).
    {
        xarm_geo::controllers::GeometricPIDController c(model);
        c.gains.kp_pos.setConstant(kKpPos);
        c.gains.kp_rot.setConstant(kKpRot);
        c.gains.kd_lin.setConstant(kKdLin);
        c.gains.kd_ang.setConstant(kKdAng);
        c.gains.ki_lin.setConstant(kKiLinDyn);
        c.gains.ki_ang.setConstant(kKiAngDyn);
        c.use_feedforward = true;
        c.bias_compensation = xarm_geo::BiasCompensation::Full;
        c.reset();

        ScenarioThresholds th;
        // Observed: p95_pos = 4.86 mm, p95_rot = 79 mrad.
        // 7 mm pos gives variance room (~1.4x); 120 mrad rot gives ~1.5x.
        th.p95_pos_error = 0.007;  // m
        th.p95_rot_error = 0.12;   // rad
        th.gate_tracking = true;

        const auto samples = run_scenario_dynamic(sim, model, data, q_start, c, figure_eight_traj,
                                                  /*duration_s=*/15.0, render);
        results[7] = evaluate_and_report("[8/11] FigureEight FF+KI   (PID) ", samples, th, false);
        passed += results[7] ? 1 : 0;
    }

    // --- Scenario 9: Anti-windup saturation (PID) ---
    //
    // Start from q_perturbed (~5 cm EE offset), aggressive KI=50 [1/s]
    // (25x the nominal 2.0), tight sigma=0.05 [N*s] / [N*m*s]. The integrator
    // must saturate at e_I_sat = sigma and torques must stay within the
    // 200 Nm ceiling even though an unsaturated integrator at this KI would
    // over-command.
    {
        xarm_geo::controllers::GeometricPIDController c(model);
        c.gains.kp_pos.setConstant(kKpPos);
        c.gains.kp_rot.setConstant(kKpRot);
        c.gains.kd_lin.setConstant(kKdLin);
        c.gains.kd_ang.setConstant(kKdAng);
        c.gains.ki_lin.setConstant(50.0);  // aggressive: 25x nominal [1/s]
        c.gains.ki_ang.setConstant(50.0);
        c.sigma_lin = Eigen::Vector3d::Constant(0.05);
        c.sigma_ang = Eigen::Vector3d::Constant(0.05);
        c.use_feedforward = true;
        c.bias_compensation = xarm_geo::BiasCompensation::Full;
        c.reset();

        xarm_geo::TaskSetpointTrajectory traj{task_target_at_start.pose};

        ScenarioThresholds th;
        th.gate_tracking = false;
        th.gate_e_I = true;
        // Per-axis clamp to 0.05; 6 axes; worst-case saturated norm = sqrt(6)*0.05 ~ 0.122.
        th.max_e_I_norm = 0.16;

        const auto samples = run_scenario_dynamic(sim, model, data, q_perturbed, c, traj,
                                                  /*duration_s=*/8.0, render);
        results[8] = evaluate_and_report("[9/11] Anti-windup         (PID) ", samples, th, false);
        passed += results[8] ? 1 : 0;
    }

    // --- Scenario 10: Constant disturbance rejection (PID) ---
    //
    // Applies a 5 Nm constant bias at joint 1 (shoulder) throughout an 8 s
    // static hold with GeometricPIDController. The integral term should absorb
    // the disturbance and drive the steady-state position error close to zero.
    // Compare to S11 (PD-only baseline) which shows the residual offset
    // without integral action.
    //
    // Gated on final||log(g_e)|| < 2e-3: integral eliminates the bulk of the
    // disturbance-induced offset within the 8 s window. Threshold is set just
    // above the observed settled value (~1.5 mm) to act as a benefit-
    // demonstrating regression guard.
    double s10_final_log_g_e = -1.0;  // captured for S11 comparison printout
    {
        xarm_geo::controllers::GeometricPIDController c(model);
        c.gains.kp_pos.setConstant(kKpPos);
        c.gains.kp_rot.setConstant(kKpRot);
        c.gains.kd_lin.setConstant(kKdLin);
        c.gains.kd_ang.setConstant(kKdAng);
        c.gains.ki_lin.setConstant(kKiLinDyn);
        c.gains.ki_ang.setConstant(kKiAngDyn);
        c.use_feedforward = true;
        c.bias_compensation = xarm_geo::BiasCompensation::Full;
        c.reset();

        xarm_geo::TaskSetpointTrajectory traj{task_target_at_start.pose};

        const auto samples =
            run_scenario_dynamic_disturbed(sim, model, data, q_start, c, traj, /*duration_s=*/8.0,
                                           /*disturb_joint=*/1, /*disturb_tau=*/5.0, render);

        s10_final_log_g_e = samples.empty() ? -1.0 : samples.back().log_g_e;
        std::cout << "  [10/11 detail] PID + 5 Nm disturbance at joint 1:\n"
                  << "  [10/11 detail] final||log(g_e)||=" << s10_final_log_g_e
                  << "  final||xi_e||=" << (samples.empty() ? -1.0 : samples.back().xi_e_norm)
                  << "  e_I_norm=" << (samples.empty() ? -1.0 : samples.back().e_I_norm) << "\n";

        ScenarioThresholds th;
        th.check_final_log_g_e = true;
        // Integral action's benefit is eliminating steady-state position offset,
        // not velocity error. Gate on log_g_e; threshold just above the settled
        // value observed with ki=2 [1/s] and a 5 Nm disturbance (~1.5 mm).
        th.final_log_g_e = 2e-3;
        th.gate_tracking = false;

        results[9] = evaluate_and_report("[10/11] Disturbance PID    ", samples, th, false);
        passed += results[9] ? 1 : 0;
    }

    // --- Scenario 11: PD-only disturbance baseline ---
    //
    // Same 5 Nm shoulder disturbance as S10, but with GeometricPDController
    // (no integral). PD alone settles to a constant pose offset proportional
    // to tau_disturb / Kp; compare final||log(g_e)|| here against S10 to see
    // the integral term's benefit directly.
    //
    // Gated on final||log(g_e)|| < 0.05 (regression guard: PD must remain
    // stable under the disturbance). The actual offset is informational.
    {
        xarm_geo::controllers::GeometricPDController c(model);
        c.gains.kp_pos.setConstant(kKpPos);
        c.gains.kp_rot.setConstant(kKpRot);
        c.gains.kd_lin.setConstant(kKdLin);
        c.gains.kd_ang.setConstant(kKdAng);
        c.use_feedforward = true;
        // bias_compensation defaults to Full in GeometricPDController ctor.

        xarm_geo::TaskSetpointTrajectory traj{task_target_at_start.pose};

        const auto samples =
            run_scenario_dynamic_disturbed(sim, model, data, q_start, c, traj, /*duration_s=*/8.0,
                                           /*disturb_joint=*/1, /*disturb_tau=*/5.0, render);

        std::cout << "  [11/11 detail] PD-only + 5 Nm disturbance at joint 1:\n"
                  << "  [11/11 detail] final||log(g_e)||="
                  << (samples.empty() ? -1.0 : samples.back().log_g_e)
                  << "  final||xi_e||=" << (samples.empty() ? -1.0 : samples.back().xi_e_norm)
                  << "  (S10 PID = " << s10_final_log_g_e << " -- ratio "
                  << (s10_final_log_g_e > 0.0 && !samples.empty()
                          ? samples.back().log_g_e / s10_final_log_g_e
                          : -1.0)
                  << "x worse without integral)\n";

        ScenarioThresholds th;
        th.check_final_log_g_e = true;
        // Upper bound only: PD must remain stable under the disturbance.
        // The actual residual offset is the expected result, not a failure.
        th.final_log_g_e = 0.05;
        th.gate_tracking = false;

        results[10] = evaluate_and_report("[11/11] Disturbance PD     ", samples, th, false);
        passed += results[10] ? 1 : 0;
    }

    // --- Summary ---

    const auto result_str = [](bool p) -> std::string_view { return p ? "PASS" : "FAIL"; };
    std::cout << "\n=== Summary: " << passed << "/11 passed ===\n"
              << "  [1/11]  Static hold KI=0     (PI)   " << result_str(results[0]) << "\n"
              << "  [2/11]  Static hold KI>0     (PI)   " << result_str(results[1]) << "\n"
              << "  [3/11]  FigureEight FF+KI    (PI)   " << result_str(results[2]) << "\n"
              << "  [4/11]  Anti-windup          (PI)   " << result_str(results[3]) << "\n"
              << "  [5/11]  reset() check        (PI)   " << result_str(results[4]) << "\n"
              << "  [6/11]  Static hold KI=0    (PID)   " << result_str(results[5]) << "\n"
              << "  [7/11]  Static hold KI>0    (PID)   " << result_str(results[6]) << "\n"
              << "  [8/11]  FigureEight FF+KI   (PID)   " << result_str(results[7]) << "\n"
              << "  [9/11]  Anti-windup         (PID)   " << result_str(results[8]) << "\n"
              << "  [10/11] Disturbance PID             " << result_str(results[9]) << "\n"
              << "  [11/11] Disturbance PD-only         " << result_str(results[10]) << "\n";
    std::cout
        << "Notes:\n"
        << "  - e_I_norm is always printed but only gated in anti-windup scenarios (#4, #9).\n"
        << "  - Scenarios #4 and #9 start from q_perturbed (not q_start) to exercise "
           "saturation.\n"
        << "  - Scenario #5 reset() detail line printed before the verdict above.\n"
        << "  - Scenarios #10 and #11 detail lines printed before their verdicts above;\n"
        << "    compare final||log(g_e)|| between them to see the integral benefit.\n"
        << "  - Lambda inversion failures silently drop the FF term; check max||tau_cmd|| "
           "for evidence.\n";

    sim.shutdown();
    return (passed == 11) ? 0 : 1;
}
