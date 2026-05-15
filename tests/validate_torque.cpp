// --- Torque Controller Validation Script ---
//
// Standalone, headless-by-default executable that exercises the dynamic task
// controllers across five canned scenarios and prints a single PASS / FAIL
// line per scenario. Complements (does not replace) tests/simulation_test.cpp.
//
// Usage:
//   ./build/validate_torque           # headless, max-speed (default)
//   ./build/validate_torque --render  # windowed, real-time pacing for
//                                     # visual inspection of any FAIL case
//
// Notes:
//   - All scenarios start from the same q_start (matches simulation_test.cpp)
//     and use the EE pose at q_start as the trajectory anchor; the first tick
//     therefore has zero pose error and Phase-1 approach is sidestepped.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iostream>
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
#include <xarm_geo/examples/controllers/euclidean_pd_controller.h>
#include <xarm_geo/examples/controllers/geometric_pd_controller.h>
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

    // --- Canonical-Form Gain Set (Bullo-Murray / Maithripala / Seo) ---
    //
    // Wrench-direct PD with Lambda-scaled FF. Gains have physical units:
    //   K_p_pos  (N/m)        : task-space linear stiffness
    //   K_p_rot  (N*m/rad)    : task-space rotational stiffness
    //   K_d_lin  (N*s/m)      : linear damping  (zeta = 1)
    //   K_d_ang  (N*m*s/rad)  : rotational damping  (zeta = 1)
    //
    // Sized for critical damping against the operational-space inertia
    // eigenvalues at q_start (xArm 6, with motor armature):
    //   sigma_min(Lambda_lin) ~ 5     kg     -> omega_n_lin = sqrt(4000/5)    ~ 28 rad/s
    //   sigma_min(Lambda_rot) ~ 0.025 kg*m^2 -> omega_n_rot = sqrt(60/0.025)  ~ 49 rad/s
    //   K_d_lin = 2*sqrt(K_p_pos*sigma_min(Lambda_lin)) ~ 283  -> 280
    //   K_d_ang = 2*sqrt(K_p_rot*sigma_min(Lambda_rot)) ~ 2.45 -> 2.4
    //
    // K_p_rot is raised above the sigma_min-aligned critical-damping target
    // to overcome the per-joint friction floor (MJCF frictionloss="1").
    // Empirically this halves the friction-deadband-driven p95_rot offset.
    // omega_n_rot ~ 49 rad/s stays below the 62.5 rad/s controller-rate
    // Nyquist at 125 Hz; further raises risk discrete-time instability.
    //
    // Both subspaces share the same convention; numerical asymmetry reflects
    // the physical anisotropy of Lambda(q), not a structural difference.
    constexpr double kKpPos = 4000.0;  // N/m
    constexpr double kKpRot = 60.0;    // N*m/rad
    constexpr double kKdLin = 280.0;   // N*s/m
    constexpr double kKdAng = 2.4;     // N*m*s/rad

    // --- Per-Scenario Thresholds ---
    //
    // A scenario PASS requires: no NaNs/Infs at any tick; max||tau||_inf below
    // the joint-torque ceiling; and (when gated) p95 position / rotation /
    // velocity-error metrics below the configured bounds.

    struct ScenarioThresholds {
        double max_tau_inf = 200.0;     // Nm; well below xArm 6 joint limits
        double final_xi_e = 1e-3;       // only used when check_final_xi_e
        double p95_pos_error = 0.05;    // m
        double p95_rot_error = 0.3;     // rad
        bool check_final_xi_e = false;  // static-hold gating only
        bool gate_tracking = true;      // false -> pass purely on no-blowup
    };

    // --- Per-Tick Sample ---

    struct TickSample {
        double t;
        double tau_inf;      // ||state.tau||_inf, Nm  (clamped by MuJoCo actuator_ctrlrange)
        double tau_des_inf;  // ||tau_des||_inf, Nm  (controller's pre-clamp commanded torque)
        double pos_err;      // ||p_d - p||, m
        double rot_err;      // angle of log(R^-1 R_d), rad
        double xi_e_norm;    // ||body_twist - Ad_{g_e} * xi_d||, 1/s
        double log_g_e;      // ||log(g_e)||, SE(3) error magnitude
        double q_drift;      // ||q - q_start||, joint-space drift from scenario start
        bool finite;         // all of the above are finite
    };

    auto safe_isfinite_vec(const Eigen::VectorXd &v) -> bool { return v.allFinite(); }

    // Nearest-rank p95 over an unsorted buffer. Sorts a copy; no allocation
    // inside the hot loop because samples are collected once and the sort
    // happens out-of-loop.
    auto percentile(std::vector<double> values, double p) -> double {
        if (values.empty()) { return 0.0; }
        std::sort(values.begin(), values.end());
        const std::size_t n = values.size();
        const std::size_t idx = std::max(
            std::size_t{0}, static_cast<std::size_t>(std::ceil(p * static_cast<double>(n))) - 1);
        return values[std::min(idx, n - 1)];
    }

    // --- Scenario Runner ---
    //
    // Reset the sim, run the trajectory for `duration_s` seconds, return the
    // collected per-controller-tick samples. Caller evaluates thresholds.
    //
    // Note: the controller hook reads ctx.dt (used by integrator-bearing
    // controllers); we set it to the controller period, not the physics
    // period, mirroring simulation_test.cpp.

    template <xarm_geo::DynamicTaskController Ctrl, xarm_geo::TaskTrajectory Traj>
    auto run_scenario(xarm_geo::Simulation &sim, xarm_geo::Model &model, xarm_geo::Data &data,
                      const Eigen::VectorXd &q_start, Ctrl &controller, const Traj &traj,
                      double duration_s, bool render) -> std::vector<TickSample> {

        std::vector<TickSample> samples;
        samples.reserve(static_cast<std::size_t>(duration_s / (kDecim * kPhysicsDtS)) + 32);

        // Reset state and switch to torque mode.
        sim.set_joint_positions(q_start);
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

        // Helper: refresh data.q, recompute kinematics, build sample.
        auto build_sample = [&](double t_cur) {
            data.q = state.q;
            xarm_geo::compute_jacobians(model, data);

            const xarm_geo::manifold::SE3 &g = data.ee_pose;
            const xarm_geo::manifold::SE3 g_e = g.inverse() * target.pose;

            TickSample s{};
            s.t = t_cur;
            s.tau_inf = state.tau.cwiseAbs().maxCoeff();
            s.tau_des_inf = controller.last_tick_diagnostics().tau_des.cwiseAbs().maxCoeff();
            s.pos_err = (target.pose.r3() - g.r3()).norm();
            // Rotation error angle via SO(3) log of R^-1 R_d.
            const xarm_geo::manifold::SO3 R_err = g.so3().inverse() * target.pose.so3();
            s.rot_err = R_err.log().norm();
            const xarm_geo::manifold::SE3::Twist body_twist = data.body_jacobian * state.v;
            const xarm_geo::manifold::SE3::Twist xi_e = body_twist - (g_e.Ad() * target.twist);
            s.xi_e_norm = xi_e.norm();
            s.log_g_e = g_e.log().norm();
            s.q_drift = (state.q - q_start).norm();
            s.finite = std::isfinite(s.tau_inf) && std::isfinite(s.tau_des_inf) &&
                       std::isfinite(s.pos_err) && std::isfinite(s.rot_err) &&
                       std::isfinite(s.xi_e_norm) && std::isfinite(s.log_g_e) &&
                       std::isfinite(s.q_drift) && safe_isfinite_vec(state.q) &&
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
                    std::cerr << "  [run_scenario] controller.update failed (status="
                              << static_cast<int>(status) << ") at t=" << t << "\n";
                    break;
                }

                if (sim.write(torque) != xarm_geo::InterfaceStatus::OK) { break; }

                samples.push_back(build_sample(t));

                // Early-out on NaN/Inf: bail before the simulator amplifies it.
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

    // --- Verdict & Reporting ---
    //
    // Apply thresholds to the collected samples, print the one-line verdict,
    // and return a pass/fail bool for the summary tally.

    auto evaluate_and_report(std::string_view label, std::span<const TickSample> samples,
                             const ScenarioThresholds &th) -> bool {
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
        double max_tau = 0.0;
        double max_tau_t = 0.0;
        double max_tau_des = 0.0;
        std::vector<double> pos_errs;
        std::vector<double> rot_errs;
        pos_errs.reserve(samples.size());
        rot_errs.reserve(samples.size());
        for (const auto &s : samples) {
            if (s.tau_inf > max_tau) {
                max_tau = s.tau_inf;
                max_tau_t = s.t;
            }
            if (s.tau_des_inf > max_tau_des) { max_tau_des = s.tau_des_inf; }
            pos_errs.push_back(s.pos_err);
            rot_errs.push_back(s.rot_err);
        }

        const double p95_pos = percentile(pos_errs, 0.95);
        const double p95_rot = percentile(rot_errs, 0.95);
        const double final_xi_e = samples.back().xi_e_norm;

        // Threshold checks.
        bool pass = true;
        std::string fail_reason;

        if (max_tau > th.max_tau_inf) {
            pass = false;
            fail_reason = "max||tau||_inf=" + std::to_string(max_tau) + " Nm exceeds " +
                          std::to_string(th.max_tau_inf) + " @ t=" + std::to_string(max_tau_t);
        }
        // Pre-clamp controller honesty check: max||tau_cmd||_inf is the
        // unbounded commanded torque before MuJoCo's actuator_ctrlrange clamp.
        // Always gated regardless of gate_tracking, so a scenario cannot
        // "pass" by virtue of clamping hiding a wildly over-commanded wrench.
        if (pass && max_tau_des > th.max_tau_inf) {
            pass = false;
            fail_reason = "max||tau_cmd||_inf=" + std::to_string(max_tau_des) + " Nm exceeds " +
                          std::to_string(th.max_tau_inf);
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

        // One-line verdict. tau = clamped (state.tau); tau_cmd = pre-clamp (controller).
        std::cout << label << "  max||tau||=" << max_tau << " Nm"
                  << "  max||tau_cmd||=" << max_tau_des << " Nm";
        if (th.check_final_xi_e) {
            std::cout << "  final||xi_e||=" << final_xi_e;
        } else {
            std::cout << "  p95||p_e||=" << p95_pos << " m"
                      << "  p95||rot||=" << p95_rot << " rad";
        }
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
    model.gravity = Eigen::Vector3d{0.0, 0.0, -9.81};  // torque mode bakes gravity into RNEA

    xarm_geo::Data data(model);
    xarm_geo::CollisionModel col_model = xarm_geo::build_collision_model(model, true);
    xarm_geo::CollisionData col_data(col_model);

    xarm_geo::Simulation sim(model);

    // q_home matches simulation_test.cpp's q_home.
    Eigen::VectorXd q_home = Eigen::VectorXd::Zero(model.dof);
    q_home[0] = 1.5 * std::numbers::pi;

    // Trajectory anchor: matches simulation_test.cpp's construction so the
    // validation runs use the same workspace region as the original failure
    // case (offset 0.35 m from the base, 0.35 m elevation, rotated about Z
    // by the base-joint angle).
    const double q0 = q_home[0];
    const Eigen::Vector3d anchor_center(0.35 * std::cos(q0), 0.35 * std::sin(q0), 0.35);
    const xarm_geo::manifold::SO3 anchor_rot =
        xarm_geo::manifold::SO3::exp(Eigen::Vector3d::UnitZ() * (q0 - (1.5 * std::numbers::pi)));
    const xarm_geo::manifold::SE3 anchor(anchor_rot, anchor_center);

    // Solve collision-aware IK to the FigureEight t=0 pose. Mirrors
    // simulation_test.cpp's Phase-1 pattern: the trajectory bakes its own
    // tool-down construction on top of the anchor; the raw anchor with
    // identity orientation is tool-up and unreachable from q_home without
    // a wrist flip the collision-aware IK refuses.
    //
    // All five scenarios share this start pose (zero initial pose error).
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

    // --- One-Shot Pre-Scenario Sanity Dump ---
    //
    // IK convergence quality, gravity-compensation cross-check, and Lambda
    // spectrum at q_start. Fires once before the scenario loop.
    //
    // If ||log(g(q_start) * g_target^-1)|| is non-trivial the controllers see
    // a step-input pose error on tick 0 of every scenario.
    //
    // If ||g_ours - g_mj||_inf exceeds ~1 Nm the gravity compensation is
    // structurally wrong -- the arm will fall faster than the PD can recover.
    {
        data.q = q_start;
        xarm_geo::compute_jacobians(model, data);
        const xarm_geo::manifold::SE3 g_at_start = data.ee_pose;
        const xarm_geo::manifold::SE3 ik_err = g_at_start.inverse() * task_target_at_start.pose;

        // Our RNEA gravity-comp at q_start (v = 0 => h(q,0) = g(q)).
        const Eigen::VectorXd v_zero = Eigen::VectorXd::Zero(model.dof);
        xarm_geo::compute_bias_forces(model, data, v_zero);
        const Eigen::VectorXd g_ours = data.h;

        // MuJoCo's internal qfrc_bias at q_start with qvel = 0.
        sim.set_joint_positions(q_start);
        Eigen::Map<Eigen::VectorXd>(sim.data->qvel, model.dof).setZero();
        mj_forward(sim.model, sim.data);
        const Eigen::VectorXd g_mj =
            Eigen::Map<const Eigen::VectorXd>(sim.data->qfrc_bias, model.dof);

        // Lambda(q_start): compare our M(q) path against MuJoCo's mj_fullM.
        // ||Lambda_ours - Lambda_mj||_F should be small; large values indicate
        // M(q) disagreement. Lambda enters only the FF term; the PD term is
        // wrench-direct and unaffected by Lambda anisotropy.
        Eigen::Matrix<double, 6, 6> Lambda_ours = Eigen::Matrix<double, 6, 6>::Zero();
        Eigen::Matrix<double, 6, 6> Lambda_mj = Eigen::Matrix<double, 6, 6>::Zero();
        {
            // Our Lambda via compute_op_space_inertia.
            data.q = q_start;
            xarm_geo::compute_jacobians(model, data);
            xarm_geo::compute_mass_matrix(model, data);
            Eigen::LLT<Eigen::MatrixXd> M_llt(data.M);
            Eigen::MatrixXd M_inv_Jt_scratch(model.dof, 6);
            const bool ok_ours = xarm_geo::compute_op_space_inertia(
                M_llt, data.body_jacobian, Lambda_ours, M_inv_Jt_scratch, /*damping=*/0.05);
            if (!ok_ours) { std::cerr << "  Warning: Lambda_ours inversion failed at q_start.\n"; }

            // MuJoCo's M via mj_fullM, then Lambda_mj using our J_b.
            Eigen::MatrixXd M_mj_dense(model.dof, model.dof);
            std::vector<double> M_mj_vec(model.dof * model.dof);
            mj_fullM(sim.model, M_mj_vec.data(), sim.data->qM);
            for (int i = 0; i < model.dof; ++i) {
                for (int j = 0; j < model.dof; ++j) {
                    M_mj_dense(i, j) = M_mj_vec[(i * model.dof) + j];
                }
            }
            Eigen::LLT<Eigen::MatrixXd> M_mj_llt(M_mj_dense);
            if (M_mj_llt.info() == Eigen::Success) {
                Eigen::MatrixXd M_mj_inv_Jt = M_mj_llt.solve(data.body_jacobian.transpose());
                Eigen::Matrix<double, 6, 6> M_op_mj = data.body_jacobian * M_mj_inv_Jt;
                Eigen::LLT<Eigen::Matrix<double, 6, 6>> Mop_mj_llt(M_op_mj);
                if (Mop_mj_llt.info() == Eigen::Success) {
                    Lambda_mj = Mop_mj_llt.solve(Eigen::Matrix<double, 6, 6>::Identity());
                }
            }
        }

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> es_full(Lambda_ours);
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es_lin(Lambda_ours.topLeftCorner<3, 3>());
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es_rot(
            Lambda_ours.bottomRightCorner<3, 3>());

        std::cout << "=== Torque Controller Validation (5 scenarios) ===\n"
                  << "  render: " << (render ? "ON" : "OFF") << "\n"
                  << "  formulation S1/S3/S4: Bullo-Murray / Maithripala / Seo "
                     "(wrench-direct PD + Lambda-scaled FF)\n"
                  << "  formulation S2/S5: Euclidean world-frame PD + Lambda-scaled FF "
                     "(no ad*-coupling; ZYX-Euler orientation error)\n"
                  << "  gains: K_p_pos=" << kKpPos << " N/m"
                  << "  K_p_rot=" << kKpRot << " N*m/rad"
                  << "  K_d_lin=" << kKdLin << " N*s/m"
                  << "  K_d_ang=" << kKdAng << " N*m*s/rad  (zeta=1)\n"
                  << "  ||q_start - q_home|| = " << (q_start - q_home).norm() << " rad\n"
                  << "  ||log(g(q_start)^-1 * g_target)|| = " << ik_err.log().norm() << "\n"
                  << "  target translation = [" << task_target_at_start.pose.r3().transpose()
                  << "]\n"
                  << "  q_start = [" << q_start.transpose() << "]\n"
                  << "  g(q_start) [ours] = [" << g_ours.transpose() << "]\n"
                  << "  g(q_start) [mjco] = [" << g_mj.transpose() << "]\n"
                  << "  ||g_ours - g_mj||_inf = " << (g_ours - g_mj).cwiseAbs().maxCoeff()
                  << " Nm\n"
                  << "  ||g_ours||_inf = " << g_ours.cwiseAbs().maxCoeff() << " Nm  |  "
                  << "||g_mj||_inf = " << g_mj.cwiseAbs().maxCoeff() << " Nm\n"
                  << "  Lambda(q_start) [ours] eigenvalues (6x6) = ["
                  << es_full.eigenvalues().transpose() << "]\n"
                  << "  Lambda_lin (top-left 3x3) eigenvalues = ["
                  << es_lin.eigenvalues().transpose() << "] (kg)\n"
                  << "  Lambda_rot (bottom-right 3x3) eigenvalues = ["
                  << es_rot.eigenvalues().transpose() << "] (kg*m^2)\n"
                  << "  sigma_min(Lambda_lin)/sigma_min(Lambda_rot) = "
                  << (es_rot.eigenvalues().minCoeff() > 0.0
                          ? es_lin.eigenvalues().minCoeff() / es_rot.eigenvalues().minCoeff()
                          : std::numeric_limits<double>::infinity())
                  << "  (anisotropy ratio)\n"
                  << "  ||Lambda_ours - Lambda_mj||_F = " << (Lambda_ours - Lambda_mj).norm()
                  << "  (Frobenius)\n"
                  << "  tau = clamped state.tau; tau_cmd = controller's pre-clamp command.\n\n";
    }

    int passed = 0;
    bool s1_pass = false;
    bool s2_pass = false;
    bool s3_pass = false;
    bool s4_pass = false;
    bool s5_pass = false;

    // --- Scenario 1: Static hold (GeometricPD) ---
    //
    // Sanity check: setpoint regulation at the IK-converged start pose with
    // zero reference twist / acceleration. Under wrench-direct PD this is
    // pure spring-damper recovery from any tick-0 residual error; FF is
    // structurally zero (xi_d = 0, a_d = 0). Gating: final ||xi_e|| < 1e-4.
    {
        xarm_geo::controllers::GeometricPDController c(model);
        c.gains.kp_pos.setConstant(kKpPos);
        c.gains.kp_rot.setConstant(kKpRot);
        c.gains.kd_lin.setConstant(kKdLin);
        c.gains.kd_ang.setConstant(kKdAng);
        c.use_feedforward = true;
        c.constraint_aware = false;

        xarm_geo::TaskSetpointTrajectory traj{task_target_at_start.pose};

        ScenarioThresholds th;
        th.check_final_xi_e = true;
        th.final_xi_e = 1e-4;  // 3.6x margin over Stage 2 measured 2.8e-5
        th.gate_tracking = false;

        const auto samples =
            run_scenario(sim, model, data, q_start, c, traj, /*duration_s=*/5.0, render);
        s1_pass = evaluate_and_report("[1/5] Static hold (GeometricPD)            ", samples, th);
        passed += s1_pass ? 1 : 0;
    }

    // --- Scenario 2: Static hold (EuclideanPD) ---
    //
    // Baseline contrast for S1. EuclideanPDController uses ZYX-Euler
    // orientation error + Lambda-scaled FF without ad*-coupling (see its
    // class doc). At zero pose error this is essentially the same as S1
    // since the gradient differences vanish at the identity rotation.
    // Gating: final ||xi_e|| < 5e-4.
    {
        xarm_geo::controllers::EuclideanPDController c(model);
        c.gains.kp_pos.setConstant(kKpPos);
        c.gains.kp_rot.setConstant(kKpRot);
        c.gains.kd_lin.setConstant(kKdLin);
        c.gains.kd_ang.setConstant(kKdAng);
        c.use_feedforward = true;
        c.constraint_aware = false;

        xarm_geo::TaskSetpointTrajectory traj{task_target_at_start.pose};

        ScenarioThresholds th;
        th.check_final_xi_e = true;
        th.final_xi_e = 5e-4;  // 3.6x margin over Stage 2 measured 1.4e-4
        th.gate_tracking = false;

        const auto samples =
            run_scenario(sim, model, data, q_start, c, traj, /*duration_s=*/5.0, render);
        s2_pass = evaluate_and_report("[2/5] Static hold (EuclideanPD)            ", samples, th);
        passed += s2_pass ? 1 : 0;
    }

    // --- Scenario 3: FigureEight, FF off (GeometricPD) ---
    //
    // Pure-PD tracking; isolates the wrench-direct spring-damper from the
    // FF contribution. Tracking error reflects the asymptotic offset of a
    // stiff PD against a non-zero reference acceleration (no anticipation).
    //
    // figure_eight_traj was built earlier for the IK target sampling; reuse
    // it directly to avoid re-fitting the spline.
    {
        xarm_geo::controllers::GeometricPDController c(model);
        c.gains.kp_pos.setConstant(kKpPos);
        c.gains.kp_rot.setConstant(kKpRot);
        c.gains.kd_lin.setConstant(kKdLin);
        c.gains.kd_ang.setConstant(kKdAng);
        c.use_feedforward = false;
        c.constraint_aware = false;

        ScenarioThresholds th;
        // Thresholds set at 1.7-1.9x above Stage 2 measured performance
        // (p95_pos 0.0053 m, p95_rot 0.107 rad). Looser than S4/S5 to
        // preserve the FF=off vs FF=on tracking contrast.
        th.p95_pos_error = 0.010;
        th.p95_rot_error = 0.18;
        th.gate_tracking = true;

        const auto samples = run_scenario(sim, model, data, q_start, c, figure_eight_traj,
                                          /*duration_s=*/15.0, render);
        s3_pass = evaluate_and_report("[3/5] FigureEight FF=off (GeometricPD)     ", samples, th);
        passed += s3_pass ? 1 : 0;
    }

    // --- Scenario 4: FigureEight, FF on (GeometricPD) ---
    //
    // Full canonical law: PD + Lambda * d/dt(Ad xi_d). Expected to track
    // tightly because the FF cancels reference-acceleration-induced offset
    // and the ad-coupling term inside d/dt(Ad xi_d) compensates for body-
    // frame transport effects (Bullo-Murray Theorem 6, Maithripala eq. 17).
    {
        xarm_geo::controllers::GeometricPDController c(model);
        c.gains.kp_pos.setConstant(kKpPos);
        c.gains.kp_rot.setConstant(kKpRot);
        c.gains.kd_lin.setConstant(kKdLin);
        c.gains.kd_ang.setConstant(kKdAng);
        c.use_feedforward = true;
        c.constraint_aware = false;

        ScenarioThresholds th;
        // Thresholds set at 2.0-1.7x above Stage 2 measured performance
        // (p95_pos 0.0039 m, p95_rot 0.107 rad). Tighter than S3 to capture
        // the FF benefit on the linear subspace.
        th.p95_pos_error = 0.008;
        th.p95_rot_error = 0.18;
        th.gate_tracking = true;

        const auto samples = run_scenario(sim, model, data, q_start, c, figure_eight_traj,
                                          /*duration_s=*/15.0, render);
        s4_pass = evaluate_and_report("[4/5] FigureEight FF=on  (GeometricPD)     ", samples, th);
        passed += s4_pass ? 1 : 0;
    }

    // --- Scenario 5: FigureEight, FF on (EuclideanPD) ---
    //
    // Baseline contrast for S4: same trajectory, same gains, same FF flag, but
    // the non-geometric Euclidean PD law (world-frame, ZYX-Euler orientation
    // error, naive FF without ad*-coupling -- see euclidean_pd_controller.h
    // for the two retained naivetes). Tracking quality is expected to be
    // meaningfully worse than S4 because the FF residual does not vanish at
    // perfect tracking and the Euler-error mapping degrades away from identity
    // rotation. Thresholds are deliberately loose (regression-guard rather than
    // performance qualification) -- the primary gate is max||tau||_inf which
    // catches any re-introduction of the lever-arm divergence bug.
    //
    // Regression guard: exercises the lever-arm correction
    // F_w.tail<3>() += p x F_w.head<3>() in EuclideanPDController. Without
    // it, Ad_g^T introduces a p x K_d * xi_w cross-coupling that destabilises
    // dynamic tracking (visible as unbounded divergence) even though static-
    // hold (S2) remains marginally stable.
    {
        xarm_geo::controllers::EuclideanPDController c(model);
        c.gains.kp_pos.setConstant(kKpPos);
        c.gains.kp_rot.setConstant(kKpRot);
        c.gains.kd_lin.setConstant(kKdLin);
        c.gains.kd_ang.setConstant(kKdAng);
        c.use_feedforward = true;
        c.constraint_aware = false;

        ScenarioThresholds th;
        // Thresholds are lax: Euclidean baseline is expected to exhibit
        // jitter and oscillation from the retained non-geometric naivetes.
        // Primary regression guard is max||tau||_inf (divergence detection).
        th.p95_pos_error = 0.05;
        th.p95_rot_error = 0.50;
        th.gate_tracking = true;

        const auto samples = run_scenario(sim, model, data, q_start, c, figure_eight_traj,
                                          /*duration_s=*/15.0, render);
        s5_pass = evaluate_and_report("[5/5] FigureEight FF=on  (EuclideanPD)     ", samples, th);
        passed += s5_pass ? 1 : 0;
    }

    // --- Summary ---

    const auto result_str = [](bool p) -> std::string_view { return p ? "PASS" : "FAIL"; };
    std::cout << "\n=== Summary: " << passed << "/5 passed ===\n"
              << "  [1/5] Static hold (GeometricPD)            " << result_str(s1_pass) << "\n"
              << "  [2/5] Static hold (EuclideanPD)            " << result_str(s2_pass) << "\n"
              << "  [3/5] FigureEight FF=off (GeometricPD)     " << result_str(s3_pass) << "\n"
              << "  [4/5] FigureEight FF=on  (GeometricPD)     " << result_str(s4_pass) << "\n"
              << "  [5/5] FigureEight FF=on  (EuclideanPD)     " << result_str(s5_pass) << "\n";
    std::cout << "Note: Lambda inversion failures silently drop the FF term for that tick;\n"
              << "      check max||tau_cmd|| for evidence of repeated failures.\n";

    sim.shutdown();
    return (passed == 5) ? 0 : 1;
}
