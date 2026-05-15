// --- Experiment 3B: Obstacle-Passing Line Trajectory (Simulation + Hardware) ---
//
// Three variants execute an ObstacleLine trajectory that, without safety
// filtering, would pass directly through a sphere obstacle registered in the
// CollisionModel.  Each variant uses a safety-aware controller to deflect the
// EE path online.
//
// Variants
// --------
//   A) Simulation:  GeometricPController  + Optimal IK (constraint_aware = true)
//                   -- velocity-level QP deflects the joint-velocity command.
//   B) Simulation:  GeometricPDController + ASIF       (constraint_aware = true)
//                   -- torque-level ASIF deflects the joint-torque command.
//   C) Hardware:    GeometricPController  + Optimal IK (constraint_aware = true)
//                   -- same as A but on real hardware.
//                   Requires --hw <ip>; BUILD_WITH_REAL_XARM must be enabled.
//
// Obstacle geometry
// -----------------
// A coal::Sphere of radius 0.05 m is registered in the CollisionModel at the
// Euclidean midpoint of the line (stationary in the world frame, parent joint 0).
// The line endpoints are separated by 0.4 m along the world-frame Y-axis from
// the anchor, both at the anchor height.  A straight path between them passes
// through the sphere centre.
//
// On hardware (Variant C), place a physical obstacle of equivalent size at the
// matching world-frame position before running.  The code registers the sphere
// at a fixed pose; you are responsible for physical placement.
//
// Pre-flight safety check
// -----------------------
// Before each variant, the experiment validates the UNCONSTRAINED trajectory
// (a scratch GeometricPController with constraint_aware = false) to confirm
// the straight path is indeed in collision.  The check is expected to fail;
// the experiment logs the result and continues with the safety-aware variant.
//
// Usage
// -----
//   ./build/exp_3b_obstacle [--log true] [--variants AB]
//   ./build/exp_3b_obstacle --hw <ip> [--log true] [--variants C]
//
// Analysis
// --------
//   pixi run report-exp 3b
//   pixi run plot-exp   3b
//   # 3-D path plots (plot_3d_path) will show the deflection around the obstacle.

#include <chrono>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <memory>
#include <numbers>
#include <optional>
#include <string>
#include <thread>

#include <Eigen/Dense>
#include <coal/shape/geometric_shapes.h>

#include <xarm_geo/core/system.h>
#include <xarm_geo/diagnostics/logger.h>
#include <xarm_geo/examples/controllers/geometric_p_controller.h>
#include <xarm_geo/examples/controllers/geometric_pd_controller.h>
#include <xarm_geo/examples/controllers/joint_p_controller.h>
#include <xarm_geo/examples/trajectories/joint_ptp.h>
#include <xarm_geo/examples/trajectories/obstacle_line.h>
#include <xarm_geo/interfaces/simulation.h>
#include <xarm_geo/modelling/collision.h>
#include <xarm_geo/modelling/dynamics.h>
#include <xarm_geo/modelling/kinematics.h>
#include <xarm_geo/modelling/optimal_kinematics.h>
#include <xarm_geo/trajectory/validate.h>
#include <xarm_geo/utils/model_builder.h>

#ifdef XARM_GEO_HAS_REAL_XARM
#include <xarm_geo/interfaces/hardware.h>
#endif

#include "experiments/common.h"

namespace {

    using namespace xarm_geo;
    using namespace xarm_geo::experiments;

    // Line half-extent along the world-frame X-axis (left → right).
    // Total line length = 2 * kLineHalfExtent = 0.4 m.
    constexpr double kLineHalfExtent = 0.20;  // metres

    // Obstacle sphere radius.
    constexpr double kObstacleRadius = 0.05;  // metres

    // Trajectory duration.
    constexpr double kTrajDuration = 8.0;  // seconds

    // Obstacle visualisation colour: semi-transparent red-orange.
    // Applied each render frame via Simulation::draw_sphere(); purely visual.
    constexpr float kObstacleRGBA[4] = {1.0f, 0.3f, 0.1f, 0.6f};

    // Phase approach / return duration.
    constexpr double kPhaseDuration = 3.0;

    // Kinematic gains.
    constexpr double kKpPosKin = 8.0;
    constexpr double kKpRotKin = 8.0;

    // Torque-mode translational gains (same as validate_torque.cpp / exp_2).
    constexpr double kKpPos = 4000.0;
    constexpr double kKdLin = 280.0;

    // Rotational gains are raised 3× (kp_rot) and ~1.7× (kd_ang) relative to
    // the baseline values from validate_torque.cpp.  The higher kp_rot makes
    // orientation tracking 3× more expensive for the ASIF QP to trade away
    // against the obstacle CBF, reducing the "wrist-twist" artefact observed
    // when the safety filter deflects around the obstacle.  kd_ang is scaled by
    // sqrt(3) to maintain approximate critical damping.
    //
    //   Baseline: kp_rot = 60,  kd_ang = 2.4   (omega_n_rot ≈ 49 rad/s @ kp=60)
    //   3B trial: kp_rot = 180, kd_ang = 4.2   (omega_n_rot ≈ 85 rad/s; still
    //             below 125 Hz Nyquist = 392 rad/s; may see slight damped ringing
    //             but remains well within the stable envelope in simulation.)
    constexpr double kKpRot = 180.0;
    constexpr double kKdAng = 4.2;

    // -------------------------------------------------------------------------
    // build_obstacle_collision_model()
    // -------------------------------------------------------------------------
    // Returns a CollisionModel with the standard robot self-collision geometry
    // PLUS a static sphere obstacle at the line midpoint.
    // `midpoint` is the Euclidean midpoint in world frame; we express it as a
    // world-fixed pose (parent joint = 0).

    [[nodiscard]] auto build_obstacle_collision_model(const Model &model,
                                                      const Eigen::Vector3d &midpoint)
        -> CollisionModel {

        CollisionModel col = build_collision_model(model, /*self_collision=*/true);

        // Register the obstacle sphere at the midpoint of the line.
        const manifold::SE3 obstacle_pose(manifold::SO3::Identity(), midpoint);
        auto sphere = std::make_shared<coal::Sphere>(kObstacleRadius);
        col.add_geometry("obstacle_3b", /*parent_joint=*/0, obstacle_pose, sphere);

        // Add all collision pairs; adjacent-link self-collision is already
        // handled by build_collision_model (allowed pairs are set internally).
        col.add_all_collision_pairs();

        return col;
    }

    // -------------------------------------------------------------------------
    // make_obstacle_activation_distances()
    // -------------------------------------------------------------------------
    // Returns a per-pair activation-distance vector for the supplied collision
    // model.  Pairs that involve the "obstacle_3b" geometry get a 0.10 m
    // activation zone (enough lead time for the safety filter to deflect around
    // the sphere); all other (self-collision) pairs keep the library default
    // of 0.05 m so they are not over-activated.

    [[nodiscard]] auto make_obstacle_activation_distances(const CollisionModel &col)
        -> Eigen::VectorXd {

        const int n = static_cast<int>(col.collision_pairs.size());
        Eigen::VectorXd act(n);
        act.setConstant(0.05);  // library default for self-collision pairs

        for (int k = 0; k < n; ++k) {
            const auto &pair = col.collision_pairs[k];
            const std::string &n1 = col.geometries[pair.obj1_idx].name;
            const std::string &n2 = col.geometries[pair.obj2_idx].name;
            if (n1 == "obstacle_3b" || n2 == "obstacle_3b") {
                act[k] = 0.10;  // 10 cm activation for obstacle pairs
            }
        }
        return act;
    }

    // -------------------------------------------------------------------------
    // make_line_trajectory()
    // -------------------------------------------------------------------------
    // Returns an ObstacleLine trajectory whose midpoint coincides with the
    // obstacle sphere centre.  The line is horizontal, oriented along the
    // world-frame X axis (left −X to right +X), at constant height z = anchor.z
    // and constant Y = anchor.y (in front of the arm).
    //
    // Geometry (q0 = 3π/2, anchor at (0, −0.35, 0.35)):
    //   start = (−0.20,  −0.35, 0.35)   [left, world −X]
    //   end   = (+0.20,  −0.35, 0.35)   [right, world +X]
    //   obstacle midpoint = (0, −0.35, 0.35)
    //
    // The EE sweeps laterally in front of the arm at a comfortable 0.35 m
    // reach distance.  All points satisfy the conservative workspace bounds
    // (|x| < 0.6, |y| < 0.6, 0 < z < 2.0).
    //
    // Orientation: tool-down, matching the convention of all other example
    // trajectories (TiltingCircle, PipeInspection, etc.).  The anchor rotation
    // is Identity at q0 = 3π/2, so "no rotation" would be tool-up.  We compose
    // a π-flip about the local X axis (same as TiltingCircle line 61) to point
    // the EE downward.  ObstacleLine takes explicit start/end SE(3) poses, so
    // we build tool-down endpoints here rather than baking the flip into the
    // trajectory class (keeping ObstacleLine generic for other use cases).

    [[nodiscard]] auto make_line_trajectory(const manifold::SE3 &anchor)
        -> trajectories::ObstacleLine {

        // π-flip about local X → tool-down (EE z-axis points toward −world Z).
        const manifold::SO3 tool_down =
            manifold::SO3::exp(std::numbers::pi * Eigen::Vector3d::UnitX());
        const manifold::SO3 endpoint_rot = anchor.so3() * tool_down;

        // World-frame X axis: the arm sweeps left → right when viewed from above.
        const Eigen::Vector3d start_pos = anchor.r3() - kLineHalfExtent * Eigen::Vector3d::UnitX();
        const Eigen::Vector3d end_pos = anchor.r3() + kLineHalfExtent * Eigen::Vector3d::UnitX();

        const manifold::SE3 start(endpoint_rot, start_pos);
        const manifold::SE3 end(endpoint_rot, end_pos);

        return trajectories::ObstacleLine(start, end, kTrajDuration);
    }

    // -------------------------------------------------------------------------
    // validate_unconstrained()
    // -------------------------------------------------------------------------
    // Verifies that the raw (unconstrained) path actually collides; logs the
    // result but does NOT abort — a collision here is EXPECTED.

    auto validate_unconstrained(const Model &model, Data &data, CollisionModel &col_model,
                                CollisionData &col_data, const trajectories::ObstacleLine &traj,
                                const Eigen::VectorXd &q_start, std::string_view variant_tag)
        -> void {

        controllers::GeometricPController scratch(model);
        scratch.gains.kp_pos.setConstant(kKpPosKin);
        scratch.gains.kp_rot.setConstant(kKpRotKin);
        scratch.use_feedforward = true;
        scratch.constraint_aware = false;

        const auto val =
            validate_trajectory(model, data, col_model, col_data, traj, q_start, scratch);
        if (val.status == ValidationStatus::OK) {
            std::cout << "  [" << variant_tag
                      << "] Pre-flight (unconstrained): trajectory is "
                         "COLLISION-FREE (unexpected -- obstacle may not be in path).\n";
        } else {
            std::cout << "  [" << variant_tag
                      << "] Pre-flight (unconstrained): COLLISION DETECTED at t="
                      << val.failure_time << " s  (" << val.reason << ") -- as expected.\n";
        }
    }

    // -------------------------------------------------------------------------
    // Variant A: sim, velocity mode, GeometricPController + OptIK
    // -------------------------------------------------------------------------

    auto run_variant_A(Simulation &sim, const Model &model, Data &data, CollisionModel &col_model,
                       CollisionData &col_data, const trajectories::ObstacleLine &traj,
                       const Eigen::VectorXd &q_home, bool log_data) -> bool {

        // Per-pair activation: obstacle pairs at 10 cm, self-collision at 5 cm.
        const Eigen::VectorXd pair_activation = make_obstacle_activation_distances(col_model);

        // IK for trajectory start: use the same per-pair activation so the start
        // pose IK sees the obstacle with the same 10 cm safety zone.
        OptimalIKOptions ik_opts;
        ik_opts.per_pair_activation_distance = pair_activation;
        TaskTarget start_target;
        if (traj.evaluate(0.0, start_target) != TrajectoryStatus::OK) { return false; }
        if (!optimal_inverse_kinematics(model, data, col_model, col_data, q_home, start_target.pose,
                                        ik_opts)) {
            std::cerr << "[3B-A] IK failed for start pose.\n";
            return false;
        }
        const Eigen::VectorXd q_start = data.q_out;

        // Pre-flight: confirm the unconstrained path collides.
        validate_unconstrained(model, data, col_model, col_data, traj, q_start, "3B-A");

        controllers::JointPController joint_ctrl(model);
        joint_ctrl.kp.setConstant(5.0);
        joint_ctrl.use_feedforward = true;

        JointState state(model.dof);
        JointVelocity vel(model.dof);
        if (sim.read(state) != InterfaceStatus::OK) { return false; }

        std::cout << "  [Phase 1] Home -> Line Start...\n";
        trajectories::JointPTP approach(q_home, q_start, kPhaseDuration);
        if (!run_joint_ptp_sim(sim, model, data, joint_ctrl, approach, state, vel)) {
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(200));

        controllers::GeometricPController ctrl(model);
        ctrl.gains.kp_pos.setConstant(kKpPosKin);
        ctrl.gains.kp_rot.setConstant(kKpRotKin);
        ctrl.use_feedforward = true;
        ctrl.constraint_aware = true;
        ctrl.attach_collision(col_model, col_data);
        // Forward per-pair activation so runtime OptIK uses 10 cm for obstacle pairs.
        ctrl.optimal_ik_options.per_pair_activation_distance = pair_activation;

        const std::string trial_name = diagnostics::make_trial_name(
            "sim", controllers::GeometricPController::kName, trajectories::ObstacleLine::kName,
            ctrl.constraint_aware, ctrl.use_feedforward);
        auto logger = make_logger("3b", model, trial_name, log_data);

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

        std::cout << "  [Phase 2] GeometricPController + OptIK (obstacle avoidance)...\n";
        // On QP failure the controller does not write vel; reuse the last
        // successfully-issued command instead of zeroing, which avoids the
        // discontinuous deceleration that causes catch-up oscillation after
        // the obstacle is passed.
        JointVelocity vel_last_good(model.dof);
        vel_last_good.v.setZero();
        bool have_last_good_a = false;
        bool first_failure_a = true;
        while (t < kTrajDuration && sim.is_running()) {
            const auto step_start = std::chrono::steady_clock::now();
            if (sim.read(state) != InterfaceStatus::OK) { break; }

            if (tick % kSafetyDecimationFactor == 0) {
                if (traj.evaluate(t, task_target) != TrajectoryStatus::OK) { break; }
                const TaskControllerContext ctx{state, task_target, ctrl_dt_ns};
                const auto t0 = std::chrono::steady_clock::now();
                const ControllerStatus cs = ctrl.update(model, data, ctx, vel);
                perf.record(t0, std::chrono::steady_clock::now());

                if (cs == ControllerStatus::INFEASIBLE || cs == ControllerStatus::MAX_ITERS) {
                    if (first_failure_a) {
                        std::cerr << "  [3B-A] Safety filter "
                                  << (cs == ControllerStatus::INFEASIBLE ? "INFEASIBLE"
                                                                         : "MAX_ITERS")
                                  << " at t=" << t << " s  (holding last good command)\n";
                        first_failure_a = false;
                    }
                    // Reuse the last good command to avoid discontinuous deceleration.
                    if (have_last_good_a) {
                        vel = vel_last_good;
                    } else {
                        vel.v.setZero();  // fallback if no good command yet
                    }
                } else if (cs != ControllerStatus::OK) {
                    std::cerr << "[3B-A] Controller error at t=" << t << "\n";
                    break;
                } else {
                    vel_last_good = vel;
                    have_last_good_a = true;
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
                // Draw the obstacle sphere so it is visible in the MuJoCo viewer.
                sim.draw_sphere(traj.midpoint(), kObstacleRadius, kObstacleRGBA);
                sim.render();
                last_render_t = t;
            }
            std::this_thread::sleep_until(step_start + std::chrono::duration<double>(physics_dt));
        }
        logger.reset();
        perf.report("[3B-A] GeometricPController + OptIK");

        std::cout << "  [Phase 3] Returning home...\n";
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        if (sim.read(state) != InterfaceStatus::OK) { return false; }
        trajectories::JointPTP ret(state.q, q_home, kPhaseDuration);
        if (!run_joint_ptp_sim(sim, model, data, joint_ctrl, ret, state, vel)) { return false; }
        if (log_data) {
            std::cout << "  -> Logged: tests/results/exp_3b/" << trial_name << ".csv\n";
        }
        return true;
    }

    // -------------------------------------------------------------------------
    // Variant B: sim, torque mode, GeometricPDController + ASIF
    // -------------------------------------------------------------------------

    auto run_variant_B(Simulation &sim, const Model &model, Data &data, CollisionModel &col_model,
                       CollisionData &col_data, const trajectories::ObstacleLine &traj,
                       const Eigen::VectorXd &q_home, bool log_data) -> bool {

        const Eigen::VectorXd pair_activation = make_obstacle_activation_distances(col_model);

        OptimalIKOptions ik_opts;
        ik_opts.per_pair_activation_distance = pair_activation;
        TaskTarget start_target;
        if (traj.evaluate(0.0, start_target) != TrajectoryStatus::OK) { return false; }
        if (!optimal_inverse_kinematics(model, data, col_model, col_data, q_home, start_target.pose,
                                        ik_opts)) {
            std::cerr << "[3B-B] IK failed for start pose.\n";
            return false;
        }
        const Eigen::VectorXd q_start = data.q_out;

        validate_unconstrained(model, data, col_model, col_data, traj, q_start, "3B-B");

        controllers::JointPController joint_ctrl(model);
        joint_ctrl.kp.setConstant(5.0);
        joint_ctrl.use_feedforward = true;

        JointState state(model.dof);
        JointVelocity vel(model.dof);
        JointTorque tau(model.dof);

        // Phase 1: velocity mode approach.
        sim.set_control_mode(ControlMode::VELOCITY);
        if (sim.read(state) != InterfaceStatus::OK) { return false; }

        std::cout << "  [Phase 1] Home -> Line Start (velocity)...\n";
        trajectories::JointPTP approach(q_home, q_start, kPhaseDuration);
        if (!run_joint_ptp_sim(sim, model, data, joint_ctrl, approach, state, vel)) {
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(200));

        sim.set_control_mode(ControlMode::TORQUE);

        controllers::GeometricPDController ctrl(model);
        ctrl.gains.kp_pos.setConstant(kKpPos);
        ctrl.gains.kp_rot.setConstant(kKpRot);
        ctrl.gains.kd_lin.setConstant(kKdLin);
        ctrl.gains.kd_ang.setConstant(kKdAng);
        ctrl.use_feedforward = true;
        ctrl.constraint_aware = true;
        ctrl.attach_collision(col_model, col_data);
        // Forward per-pair activation so ASIF uses 10 cm for obstacle pairs.
        ctrl.asif_options.per_pair_activation_distance = pair_activation;

        const std::string trial_name = diagnostics::make_trial_name(
            "sim", controllers::GeometricPDController::kName, trajectories::ObstacleLine::kName,
            ctrl.constraint_aware, ctrl.use_feedforward);
        auto logger = make_logger("3b", model, trial_name, log_data);

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

        std::cout << "  [Phase 2] GeometricPDController + ASIF (obstacle avoidance)...\n";
        // On ASIF failure the controller does not write tau; reuse the last
        // successfully-certified torque command.  This avoids the gravity-comp
        // hold, which was causing the arm to freeze and accumulate tracking
        // error, leading to oscillatory INFEASIBLE / MAX_ITERS sequences.
        JointTorque tau_last_good(model.dof);
        tau_last_good.tau.setZero();
        bool have_last_good_b = false;
        bool first_failure_b = true;
        while (t < kTrajDuration && sim.is_running()) {
            const auto step_start = std::chrono::steady_clock::now();
            if (sim.read(state) != InterfaceStatus::OK) { break; }

            if (tick % kSafetyDecimationFactor == 0) {
                if (traj.evaluate(t, task_target) != TrajectoryStatus::OK) { break; }
                const TaskControllerContext ctx{state, task_target, ctrl_dt_ns};
                const auto t0 = std::chrono::steady_clock::now();
                const ControllerStatus cs = ctrl.update(model, data, ctx, tau);
                perf.record(t0, std::chrono::steady_clock::now());

                if (cs == ControllerStatus::INFEASIBLE || cs == ControllerStatus::MAX_ITERS) {
                    if (first_failure_b) {
                        std::cerr << "  [3B-B] Safety filter "
                                  << (cs == ControllerStatus::INFEASIBLE ? "INFEASIBLE"
                                                                         : "MAX_ITERS")
                                  << " at t=" << t << " s  (holding last good torque command)\n";
                        first_failure_b = false;
                    }
                    // Reuse the last ASIF-certified torque to avoid discontinuous motion.
                    if (have_last_good_b) {
                        tau = tau_last_good;
                    } else {
                        // No good command yet: gravity compensation as absolute fallback.
                        compute_gravity_forces(model, data);
                        tau.tau = data.g;
                    }
                } else if (cs != ControllerStatus::OK) {
                    std::cerr << "[3B-B] Controller error at t=" << t << "\n";
                    break;
                } else {
                    tau_last_good = tau;
                    have_last_good_b = true;
                }
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
                // Draw the obstacle sphere so it is visible in the MuJoCo viewer.
                sim.draw_sphere(traj.midpoint(), kObstacleRadius, kObstacleRGBA);
                sim.render();
                last_render_t = t;
            }
            std::this_thread::sleep_until(step_start + std::chrono::duration<double>(physics_dt));
        }
        logger.reset();
        perf.report("[3B-B] GeometricPDController + ASIF");

        std::cout << "  [Phase 3] Returning home (velocity)...\n";
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        sim.set_control_mode(ControlMode::VELOCITY);
        if (sim.read(state) != InterfaceStatus::OK) { return false; }
        trajectories::JointPTP ret(state.q, q_home, kPhaseDuration);
        if (!run_joint_ptp_sim(sim, model, data, joint_ctrl, ret, state, vel)) { return false; }
        if (log_data) {
            std::cout << "  -> Logged: tests/results/exp_3b/" << trial_name << ".csv\n";
        }
        return true;
    }

#ifdef XARM_GEO_HAS_REAL_XARM

    // -------------------------------------------------------------------------
    // Variant C: hardware, velocity mode, GeometricPController + OptIK
    // -------------------------------------------------------------------------

    auto run_variant_C(Hardware &hw, const Model &model, Data &data, CollisionModel &col_model,
                       CollisionData &col_data, const trajectories::ObstacleLine &traj,
                       const Eigen::VectorXd &q_home, bool log_data) -> bool {

        const Eigen::VectorXd pair_activation = make_obstacle_activation_distances(col_model);

        OptimalIKOptions ik_opts;
        ik_opts.per_pair_activation_distance = pair_activation;
        TaskTarget start_target;
        if (traj.evaluate(0.0, start_target) != TrajectoryStatus::OK) { return false; }
        if (!optimal_inverse_kinematics(model, data, col_model, col_data, q_home, start_target.pose,
                                        ik_opts)) {
            std::cerr << "[3B-C] IK failed for start pose.\n";
            return false;
        }
        const Eigen::VectorXd q_start = data.q_out;

        validate_unconstrained(model, data, col_model, col_data, traj, q_start, "3B-C");

        controllers::JointPController joint_ctrl(model);
        joint_ctrl.kp.setConstant(5.0);
        joint_ctrl.use_feedforward = true;
        joint_ctrl.constraint_aware = true;

        JointState state(model.dof);
        JointVelocity vel(model.dof);
        hw.read(state);

        std::cout << "  [Phase 0] Moving to home...\n";
        trajectories::JointPTP h_traj(state.q, q_home, kPhaseDuration);
        if (!run_joint_ptp_hw(hw, model, data, joint_ctrl, h_traj, state, vel)) { return false; }

        std::cout << "  [Phase 1] Approaching line start...\n";
        trajectories::JointPTP a_traj(q_home, q_start, kPhaseDuration);
        if (!run_joint_ptp_hw(hw, model, data, joint_ctrl, a_traj, state, vel)) { return false; }

        controllers::GeometricPController ctrl(model);
        ctrl.gains.kp_pos.setConstant(kKpPosKin);
        ctrl.gains.kp_rot.setConstant(kKpRotKin);
        ctrl.use_feedforward = true;
        ctrl.constraint_aware = true;
        ctrl.attach_collision(col_model, col_data);
        ctrl.optimal_ik_options.dt = kHardwareControlPeriodS;
        // Forward per-pair activation so runtime OptIK uses 10 cm for obstacle pairs.
        ctrl.optimal_ik_options.per_pair_activation_distance = pair_activation;

        // Pre-flight validation with the safety-aware controller.
        const auto val = validate_trajectory(model, data, col_model, col_data, traj, q_start, ctrl);
        if (val.status != ValidationStatus::OK) {
            std::cerr << "[3B-C] Safety validation failed: " << val.reason
                      << " at t=" << val.failure_time << "\n"
                      << "  Aborting hardware run.\n";
            return false;
        }
        std::cout << "  [3B-C] Pre-flight (constrained): PASSED.\n";

        const std::string trial_name = diagnostics::make_trial_name(
            "hardware", controllers::GeometricPController::kName, trajectories::ObstacleLine::kName,
            ctrl.constraint_aware, ctrl.use_feedforward);
        auto logger = make_logger("3b", model, trial_name, log_data);

        const double dt = kHardwareControlPeriodS;
        const auto dt_ns =
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::duration<double>(dt));
        auto next_tick = std::chrono::steady_clock::now();
        std::int64_t tick = 0;
        TaskTarget task_target;

        std::cout << "  [Phase 2] GeometricPController + OptIK (hw, obstacle avoidance)...\n";
        JointVelocity vel_last_good_c(model.dof);
        vel_last_good_c.v.setZero();
        bool have_last_good_c = false;
        bool first_failure_c = true;
        for (double t = 0.0; t < kTrajDuration && hw.is_running(); t += dt, ++tick) {
            if (hw.read(state) != InterfaceStatus::OK) { return false; }
            if (traj.evaluate(t, task_target) != TrajectoryStatus::OK) { break; }

            const TaskControllerContext ctx{state, task_target, dt_ns};
            const ControllerStatus cs = ctrl.update(model, data, ctx, vel);

            if (cs == ControllerStatus::INFEASIBLE || cs == ControllerStatus::MAX_ITERS) {
                if (first_failure_c) {
                    std::cerr << "  [3B-C] Safety filter "
                              << (cs == ControllerStatus::INFEASIBLE ? "INFEASIBLE" : "MAX_ITERS")
                              << " at t=" << t << " s  (holding last good command)\n";
                    first_failure_c = false;
                }
                // Reuse last good command to avoid discontinuous deceleration.
                if (have_last_good_c) {
                    vel = vel_last_good_c;
                } else {
                    vel.v.setZero();
                }
            } else if (cs != ControllerStatus::OK) {
                std::cerr << "[3B-C] Controller error.\n";
                return false;
            } else {
                vel_last_good_c = vel;
                have_last_good_c = true;
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
            std::cout << "  -> Logged: tests/results/exp_3b/" << trial_name << ".csv\n";
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
    std::string variants_str;
    std::string robot_ip;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "--log" || arg == "-l") && i + 1 < argc) {
            const std::string v = argv[++i];
            log_data = (v == "1" || v == "true");
        } else if (arg == "--variants" && i + 1 < argc) {
            variants_str = argv[++i];
        } else if (arg == "--hw" && i + 1 < argc) {
            robot_ip = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [options]\n"
                      << "Options:\n"
                      << "  --log <false|true>    Log data to CSV (default: false)\n"
                      << "  --variants <ABC>      Variants to run\n"
                      << "                          A = Sim,  velocity, GeometricP + OptIK\n"
                      << "                          B = Sim,  torque,   GeometricPD + ASIF\n"
                      << "                          C = Hw,   velocity, GeometricP + OptIK\n"
                      << "  --hw <ip>             Robot IP for hardware variant\n";
            return 0;
        }
    }

    const bool use_hardware = !robot_ip.empty();
    if (variants_str.empty()) { variants_str = use_hardware ? "C" : "AB"; }

    const bool want_hw = variants_str.find('C') != std::string::npos;
    if (want_hw && !use_hardware) {
        std::cerr << "[3B] Variant C requires --hw <ip>.\n";
        return 1;
    }

#ifndef XARM_GEO_HAS_REAL_XARM
    if (want_hw) {
        std::cerr << "[3B] Binary was compiled without real xArm SDK support "
                     "(BUILD_WITH_REAL_XARM=OFF). "
                     "Hardware variant is unavailable.\n";
        return 1;
    }
#endif

    setup_results_dir("3b");

    // Gravity set for torque-mode variant B.
    Model model = build_model(6, "XI130412C23L45");
    model.gravity = Eigen::Vector3d{0.0, 0.0, -9.81};

    Data data(model);

    Eigen::VectorXd q_home = Eigen::VectorXd::Zero(model.dof);
    q_home[0] = 1.5 * std::numbers::pi;

    const manifold::SE3 anchor = make_anchor_pose(q_home);
    const trajectories::ObstacleLine line_traj = make_line_trajectory(anchor);
    const Eigen::Vector3d obstacle_pos = line_traj.midpoint();

    // Build collision model with the static sphere obstacle.
    CollisionModel col_model = build_obstacle_collision_model(model, obstacle_pos);
    CollisionData col_data(col_model);

    std::cout << "=== Experiment 3B: Obstacle-Passing Line Trajectory ===\n"
              << "Obstacle: sphere r=" << kObstacleRadius << " m at [" << obstacle_pos.transpose()
              << "]\n"
              << "Line: start=[" << line_traj.start_pose().r3().transpose() << "] -> end=["
              << line_traj.end_pose().r3().transpose() << "]\n"
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
            std::cout << "--- Variant A: GeometricPController + OptIK (sim) ---\n";
            if (!run_variant_A(sim, model, data, col_model, col_data, line_traj, q_home,
                               log_data)) {
                std::cerr << "[3B] Variant A failed.\n";
                return 1;
            }
            std::cout << "--- Variant A complete. ---\n\n";
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }

        if (variants_str.find('B') != std::string::npos) {
            std::cout << "--- Variant B: GeometricPDController + ASIF (sim) ---\n";
            if (!run_variant_B(sim, model, data, col_model, col_data, line_traj, q_home,
                               log_data)) {
                std::cerr << "[3B] Variant B failed.\n";
                return 1;
            }
            std::cout << "--- Variant B complete. ---\n\n";
        }

        sim.shutdown();
    }

    // ---- Hardware variant (C) -----------------------------------------------
#ifdef XARM_GEO_HAS_REAL_XARM
    if (want_hw) {
        std::cout << "--- Variant C: GeometricPController + OptIK (hardware) ---\n"
                  << "  NOTE: Ensure a physical sphere obstacle of radius " << kObstacleRadius
                  << " m is placed at [" << obstacle_pos.transpose()
                  << "] in the robot workspace.\n";

        Hardware hw(model.dof, robot_ip);
        if (!hw.is_running()) {
            std::cerr << "[3B] Failed to connect to robot at " << robot_ip << "\n";
            return 1;
        }

        JointState state(model.dof);
        hw.read(state);
        data.q = state.q;
        compute_jacobians(model, data);

        if (!run_variant_C(hw, model, data, col_model, col_data, line_traj, q_home, log_data)) {
            std::cerr << "[3B] Variant C failed.\n";
            hw.shutdown();
            return 1;
        }
        std::cout << "--- Variant C complete. ---\n\n";
        hw.shutdown();
    }
#endif  // XARM_GEO_HAS_REAL_XARM

    std::cout << "=== Experiment 3B complete. ===\n";
    if (log_data) {
        std::cout << "Results in: tests/results/exp_3b/\n"
                  << "  pixi run report-exp 3b\n"
                  << "  pixi run plot-exp   3b\n"
                  << "  # 3-D path plots show EE deflection around obstacle:\n"
                  << "  pixi run analyse plot trial tests/results/exp_3b/<trial>.csv\n";
    }
    return 0;
}
