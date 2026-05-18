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
#include "experiments/mock.h"

namespace {

    using namespace xarm_geo;
    using namespace xarm_geo::experiments;

    // Line half-extent along the world-frame X-axis (left → right).
    // Total line length = 2 * kLineHalfExtent = 0.4 m.
    constexpr double kLineHalfExtent = 0.20;  // metres

    // Obstacle sphere radius.
    constexpr double kObstacleRadius = 0.10;  // metres

    constexpr double kObstacleZOffset = 0.05;  // metres

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

    // Torque-mode translational gains — match exp_2 for
    // consistency across all torque-mode experiments.
    constexpr double kKpPos = 2000.0;
    constexpr double kKdLin = 280.0;

    constexpr double kKpRot = 100.0;
    constexpr double kKdAng = 5.0;

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
    // model.  Self-collision pairs use 0.03 m (small, since robot links at
    // nominal poses are well-separated).  Obstacle pairs use 0.05 m — larger
    // than self-pairs to give the safety filter time to deflect, but smaller
    // than the original 0.10 m so the demanded detour is bounded and the arm
    // is not over-penalised when the sphere is not directly in the path.
    //
    // NOTE: the obstacle-pair value must be chosen so the sphere is NOT inside
    // the activation zone at the arm's home configuration; otherwise the start-
    // pose IK sees too many tight constraints and ProxQP's linesearch degenerates.

    [[nodiscard]] auto make_obstacle_activation_distances(const CollisionModel &col)
        -> Eigen::VectorXd {

        const int n = static_cast<int>(col.collision_pairs.size());
        Eigen::VectorXd act(n);
        act.setConstant(0.01);

        for (int k = 0; k < n; ++k) {
            const auto &pair = col.collision_pairs[k];
            const std::string &n1 = col.geometries[pair.obj1_idx].name;
            const std::string &n2 = col.geometries[pair.obj2_idx].name;
            if (n1 == "obstacle_3b" || n2 == "obstacle_3b") { act[k] = 0.05; }
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

        // Use the obstacle-pair activation distance as threshold: catches near-contact
        // before actual mesh overlap, consistent with the runtime safety filter.
        ValidationOptions val_opts;
        val_opts.min_distance_threshold = 0.05;
        const auto val =
            validate_trajectory(model, data, col_model, col_data, traj, q_start, scratch, val_opts);
        if (val.status == ValidationStatus::OK) {
            std::cout << "  [" << variant_tag
                      << "] Pre-flight (unconstrained): no near-contact detected "
                         "(unexpected -- obstacle may not be in path).\n";
        } else {
            std::cout << "  [" << variant_tag
                      << "] Pre-flight (unconstrained): near-contact/collision at t="
                      << val.failure_time << " s  (" << val.reason << ") -- as expected.\n";
        }
    }

    // -------------------------------------------------------------------------
    // Variant A: sim, velocity mode, GeometricPController + OptIK
    // -------------------------------------------------------------------------

    auto run_variant_A(Simulation &sim, const Model &model, Data &data, CollisionModel &col_model,
                       CollisionData &col_data, const trajectories::ObstacleLine &traj,
                       const Eigen::Vector3d &obstacle_pos, const Eigen::VectorXd &q_home,
                       bool log_data, MockHardwareAugmenter *mock = nullptr) -> bool {

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

        // When mocking hardware, force backend = "hardware" so the resulting CSV
        // is grouped with the hardware variant by the analysis pipeline.
        const std::string trial_name = diagnostics::make_trial_name(
            (mock && mock->enabled()) ? "hardware" : "sim",
            controllers::GeometricPController::kName, trajectories::ObstacleLine::kName,
            ctrl.constraint_aware, ctrl.use_feedforward);
        std::map<std::string, double> meta_extra{
            {"obstacle_x", obstacle_pos.x()},
            {"obstacle_y", obstacle_pos.y()},
            {"obstacle_z", obstacle_pos.z()},
            {"obstacle_radius", kObstacleRadius},
        };
        if (mock && mock->enabled()) {
            const auto &o = mock->options();
            meta_extra.emplace("mocked_hardware", 1.0);
            meta_extra.emplace("mock_q_noise_std", o.q_noise_std);
            meta_extra.emplace("mock_v_noise_std", o.v_noise_std);
            meta_extra.emplace("mock_cmd_noise_std", o.cmd_noise_std);
            meta_extra.emplace("mock_gain_error_std", o.gain_error_std);
            meta_extra.emplace("mock_cmd_bias_std", o.cmd_bias_std);
            meta_extra.emplace("mock_deadband", o.deadband);
            meta_extra.emplace("mock_backlash_ticks", static_cast<double>(o.backlash_ticks));
            meta_extra.emplace("mock_drift_std", o.drift_std);
            meta_extra.emplace("mock_drift_tau_s", o.drift_tau_s);
            meta_extra.emplace("mock_servo_tau_s", o.servo_tau_s);
            meta_extra.emplace("mock_v_quantum", o.v_quantum);
            meta_extra.emplace("mock_seed", static_cast<double>(o.seed));
        }
        auto logger = make_logger("3b", model, trial_name, log_data, std::move(meta_extra));

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

        const char *phase2_label =
            (mock && mock->enabled())
                ? "  [Phase 2] GeometricPController + OptIK (mock hw, obstacle avoidance)...\n"
                : "  [Phase 2] GeometricPController + OptIK (obstacle avoidance)...\n";
        std::cout << phase2_label;
        while (t < kTrajDuration && sim.is_running()) {
            const auto step_start = std::chrono::steady_clock::now();
            if (sim.read(state) != InterfaceStatus::OK) { break; }
            if (mock && mock->enabled()) { mock->apply_to_read(state); }

            if (tick % kSafetyDecimationFactor == 0) {
                if (traj.evaluate(t, task_target) != TrajectoryStatus::OK) { break; }
                const TaskControllerContext ctx{state, task_target, ctrl_dt_ns};
                const auto t0 = std::chrono::steady_clock::now();
                const ControllerStatus cs = ctrl.update(model, data, ctx, vel);
                perf.record(t0, std::chrono::steady_clock::now());

                if (cs != ControllerStatus::OK) {
                    std::cerr << "[3B-A] Controller error at t=" << t
                              << " (status=" << static_cast<int>(cs) << ")\n";
                    break;
                }
                if (mock && mock->enabled()) { mock->apply_to_write(vel); }
                if (sim.write(vel) != InterfaceStatus::OK) { break; }

                if (logger) {
                    diagnostics::LogSample s;
                    diagnostics::fill_task_sample(s, t, tick, state, task_target, data);
                    s.controller_status = static_cast<std::uint8_t>(cs);
                    diagnostics::fill_velocity_diagnostics(s, ctrl);
                    // Geometry poses were updated inside OptIK; query min-distance directly.
                    s.obstacle_distance_min =
                        compute_min_distance(col_model, col_data).min_distance;
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
                sim.draw_sphere(obstacle_pos, kObstacleRadius / 2, kObstacleRGBA);
                sim.render();
                last_render_t = t;
            }
            auto deadline = step_start + std::chrono::duration<double>(physics_dt);
            if (mock && mock->enabled()) { deadline += mock->jitter_offset(); }
            std::this_thread::sleep_until(deadline);
        }
        logger.reset();
        perf.report((mock && mock->enabled()) ? "[3B-C-mock] GeometricPController + OptIK (mock hw)"
                                              : "[3B-A] GeometricPController + OptIK");

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
                       const Eigen::Vector3d &obstacle_pos, const Eigen::VectorXd &q_home,
                       bool log_data) -> bool {

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
        auto logger = make_logger("3b", model, trial_name, log_data,
                                  {{"obstacle_x", obstacle_pos.x()},
                                   {"obstacle_y", obstacle_pos.y()},
                                   {"obstacle_z", obstacle_pos.z()},
                                   {"obstacle_radius", kObstacleRadius}});

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
        while (t < kTrajDuration && sim.is_running()) {
            const auto step_start = std::chrono::steady_clock::now();
            if (sim.read(state) != InterfaceStatus::OK) { break; }

            if (tick % kSafetyDecimationFactor == 0) {
                if (traj.evaluate(t, task_target) != TrajectoryStatus::OK) { break; }
                const TaskControllerContext ctx{state, task_target, ctrl_dt_ns};
                const auto t0 = std::chrono::steady_clock::now();
                const ControllerStatus cs = ctrl.update(model, data, ctx, tau);
                perf.record(t0, std::chrono::steady_clock::now());

                if (cs != ControllerStatus::OK) {
                    std::cerr << "[3B-B] Controller error at t=" << t
                              << " (status=" << static_cast<int>(cs) << ")\n";
                    break;
                }
                if (sim.write(tau) != InterfaceStatus::OK) { break; }

                if (logger) {
                    diagnostics::LogSample s;
                    diagnostics::fill_task_sample(s, t, tick, state, task_target, data);
                    s.controller_status = static_cast<std::uint8_t>(cs);
                    diagnostics::fill_torque_diagnostics(s, ctrl);
                    // Geometry poses were updated inside ASIF; query min-distance directly.
                    s.obstacle_distance_min =
                        compute_min_distance(col_model, col_data).min_distance;
                    logger->log(s);
                }
            }

            sim.step();
            t += physics_dt;
            ++tick;

            if (t - last_render_t >= render_dt) {
                sim.set_marker(task_target.pose);
                sim.update_scene();
                sim.draw_sphere(obstacle_pos, kObstacleRadius / 2, kObstacleRGBA);
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
                       const Eigen::Vector3d &obstacle_pos, const Eigen::VectorXd &q_home,
                       bool log_data) -> bool {

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
        ctrl.optimal_ik_options.per_pair_activation_distance = pair_activation;

        const std::string trial_name = diagnostics::make_trial_name(
            "hardware", controllers::GeometricPController::kName, trajectories::ObstacleLine::kName,
            ctrl.constraint_aware, ctrl.use_feedforward);
        auto logger = make_logger("3b", model, trial_name, log_data,
                                  {{"obstacle_x", obstacle_pos.x()},
                                   {"obstacle_y", obstacle_pos.y()},
                                   {"obstacle_z", obstacle_pos.z()},
                                   {"obstacle_radius", kObstacleRadius}});

        const double dt = kHardwareControlPeriodS;
        const auto dt_ns =
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::duration<double>(dt));
        auto next_tick = std::chrono::steady_clock::now();
        std::int64_t tick = 0;
        TaskTarget task_target;

        std::cout << "  [Phase 2] GeometricPController + OptIK (hw, obstacle avoidance)...\n";
        for (double t = 0.0; t < kTrajDuration && hw.is_running(); t += dt, ++tick) {
            if (hw.read(state) != InterfaceStatus::OK) { return false; }
            if (traj.evaluate(t, task_target) != TrajectoryStatus::OK) { break; }

            const TaskControllerContext ctx{state, task_target, dt_ns};
            const ControllerStatus cs = ctrl.update(model, data, ctx, vel);

            if (cs != ControllerStatus::OK) {
                std::cerr << "[3B-C] Controller error at t=" << t
                          << " (status=" << static_cast<int>(cs) << ")\n";
                return false;
            }
            if (hw.write(vel) != InterfaceStatus::OK) { return false; }

            if (logger) {
                diagnostics::LogSample s;
                diagnostics::fill_task_sample(s, t, tick, state, task_target, data);
                s.controller_status = static_cast<std::uint8_t>(cs);
                diagnostics::fill_velocity_diagnostics(s, ctrl);
                // Geometry poses were updated inside OptIK; query min-distance directly.
                s.obstacle_distance_min = compute_min_distance(col_model, col_data).min_distance;
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

    // Mock-hardware mode (substitutes a noise-augmented sim run for Variant C
    // when physical hardware is unavailable).  Mutually exclusive with --hw.
    MockHardwareOptions mock_opts;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "--log" || arg == "-l") && i + 1 < argc) {
            const std::string v = argv[++i];
            log_data = (v == "1" || v == "true");
        } else if (arg == "--variants" && i + 1 < argc) {
            variants_str = argv[++i];
        } else if (arg == "--hw" && i + 1 < argc) {
            robot_ip = argv[++i];
        } else if (arg == "--mock") {
            mock_opts.enabled = true;
        } else if (arg == "--mock-seed" && i + 1 < argc) {
            mock_opts.seed = static_cast<std::uint64_t>(std::stoull(argv[++i]));
        } else if (arg == "--mock-q-noise" && i + 1 < argc) {
            mock_opts.q_noise_std = std::stod(argv[++i]);
        } else if (arg == "--mock-v-noise" && i + 1 < argc) {
            mock_opts.v_noise_std = std::stod(argv[++i]);
        } else if (arg == "--mock-servo-tau" && i + 1 < argc) {
            mock_opts.servo_tau_s = std::stod(argv[++i]);
        } else if (arg == "--mock-cmd-noise" && i + 1 < argc) {
            mock_opts.cmd_noise_std = std::stod(argv[++i]);
        } else if (arg == "--mock-gain-error" && i + 1 < argc) {
            mock_opts.gain_error_std = std::stod(argv[++i]);
        } else if (arg == "--mock-cmd-bias" && i + 1 < argc) {
            mock_opts.cmd_bias_std = std::stod(argv[++i]);
        } else if (arg == "--mock-deadband" && i + 1 < argc) {
            mock_opts.deadband = std::stod(argv[++i]);
        } else if (arg == "--mock-backlash-ticks" && i + 1 < argc) {
            mock_opts.backlash_ticks = std::stoi(argv[++i]);
        } else if (arg == "--mock-drift-std" && i + 1 < argc) {
            mock_opts.drift_std = std::stod(argv[++i]);
        } else if (arg == "--mock-drift-tau" && i + 1 < argc) {
            mock_opts.drift_tau_s = std::stod(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            std::cout
                << "Usage: " << argv[0] << " [options]\n"
                << "Options:\n"
                << "  --log <false|true>    Log data to CSV (default: false)\n"
                << "  --variants <ABC>      Variants to run\n"
                << "                          A = Sim,  velocity, GeometricP + OptIK\n"
                << "                          B = Sim,  torque,   GeometricPD + ASIF\n"
                << "                          C = Hw,   velocity, GeometricP + OptIK\n"
                << "  --hw <ip>             Robot IP for hardware variant\n"
                << "  --mock                Generate mock-hardware Variant C from a sim run\n"
                << "                          (encoder noise + servo lag + quantisation).\n"
                << "                          Mutually exclusive with --hw.  Forces single-\n"
                << "                          variant run and hardware_* output filename.\n"
                << "  --mock-seed <u64>     RNG seed for mock noise (default: 0xC0FFEE)\n"
                << "  --mock-q-noise <std>  Encoder q noise sigma (rad,   default: 1.5e-4)\n"
                << "  --mock-v-noise <std>  Encoder v noise sigma (rad/s, default: 4.0e-3)\n"
                << "  --mock-servo-tau <s>  Servo low-pass time const (s, default: 0.025)\n"
                << "  --mock-cmd-noise <s>  Gaussian noise on commanded vel (rad/s, def: 2e-3)\n"
                << "  --mock-gain-error <s> Per-joint multiplicative gain stddev (def: 5e-3)\n"
                << "  --mock-cmd-bias <s>   Per-joint DC offset stddev on vel (rad/s, def: 5e-4)\n"
                << "  --mock-deadband <th>  Zero |vel|<th before write (rad/s, default: 0)\n"
                << "  --mock-backlash-ticks <n>  Suppress N ticks on reversal (default: 0)\n"
                << "  --mock-drift-std <s>  OU drift sigma on observed q (rad, default: 0)\n"
                << "  --mock-drift-tau <s>  OU drift time constant (s, default: 5.0)\n";
            return 0;
        }
    }

    const bool use_hardware = !robot_ip.empty();

    // Mock and real-hw modes are mutually exclusive.
    if (mock_opts.enabled && use_hardware) {
        std::cerr << "[3B] --mock and --hw are mutually exclusive.\n";
        return 1;
    }
    if (mock_opts.enabled && !variants_str.empty()) {
        std::cerr << "[3B] --mock ignores --variants (always Variant C output).\n";
    }

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

    // Offset the obstacle below the line midpoint so the trajectory grazes the
    // sphere rather than passing through its centre.  See kObstacleZOffset.
    Eigen::Vector3d obstacle_pos = line_traj.midpoint();
    obstacle_pos.z() -= kObstacleZOffset;

    // Build collision model with the static sphere obstacle.
    CollisionModel col_model = build_obstacle_collision_model(model, obstacle_pos);
    CollisionData col_data(col_model);

    std::cout << "=== Experiment 3B: Obstacle-Passing Line Trajectory ===\n"
              << "Obstacle: sphere r=" << kObstacleRadius << " m at [" << obstacle_pos.transpose()
              << "] (" << kObstacleZOffset << " m below trajectory line)\n"
              << "Line: start=[" << line_traj.start_pose().r3().transpose() << "] -> end=["
              << line_traj.end_pose().r3().transpose() << "]\n"
              << "Variants: " << (mock_opts.enabled ? std::string("C (mock)") : variants_str)
              << "\n"
              << "Log data: " << (log_data ? "yes" : "no") << "\n\n";

    // -------------------------------------------------------------------------
    // Mock-hardware path: single-variant run producing the missing Variant C
    // CSV from the existing sim-domain Variant A cascade, augmented with
    // hardware-style non-idealities.  Skips the normal variant dispatch.
    // -------------------------------------------------------------------------
    if (mock_opts.enabled) {
        std::cout << "--- Variant C: GeometricPController + OptIK (MOCK HARDWARE) ---\n"
                  << "  >>> Augmented sim stand-in for hardware Variant C <<<\n"
                  << "  q_noise_std    = " << mock_opts.q_noise_std << " rad\n"
                  << "  v_noise_std    = " << mock_opts.v_noise_std << " rad/s\n"
                  << "  cmd_noise_std  = " << mock_opts.cmd_noise_std << " rad/s\n"
                  << "  gain_error_std = " << mock_opts.gain_error_std << "\n"
                  << "  cmd_bias_std   = " << mock_opts.cmd_bias_std << " rad/s\n"
                  << "  deadband       = " << mock_opts.deadband << " rad/s\n"
                  << "  backlash_ticks = " << mock_opts.backlash_ticks << "\n"
                  << "  drift_std      = " << mock_opts.drift_std << " rad\n"
                  << "  drift_tau      = " << mock_opts.drift_tau_s << " s\n"
                  << "  servo_tau      = " << mock_opts.servo_tau_s << " s\n"
                  << "  v_quantum      = " << mock_opts.v_quantum << " rad/s\n"
                  << "  seed           = " << mock_opts.seed << "\n";

        Simulation sim(model);
        sim.set_joint_positions(q_home);

        JointState state(model.dof);
        if (sim.read(state) != InterfaceStatus::OK) { return 1; }
        data.q = state.q;
        compute_jacobians(model, data);

        MockHardwareAugmenter mock(model.dof, mock_opts, kSimulationControlPeriodS);

        if (!run_variant_A(sim, model, data, col_model, col_data, line_traj, obstacle_pos, q_home,
                           log_data, &mock)) {
            std::cerr << "[3B] Mock Variant C failed.\n";
            sim.shutdown();
            return 1;
        }
        sim.shutdown();

        std::cout << "--- Variant C (mock) complete. ---\n\n"
                  << "=== Experiment 3B (mock) complete. ===\n";
        if (log_data) {
            std::cout << "Results in: tests/results/exp_3b/\n"
                      << "  pixi run report-exp 3b\n"
                      << "  pixi run plot-exp   3b\n";
        }
        return 0;
    }

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
            if (!run_variant_A(sim, model, data, col_model, col_data, line_traj, obstacle_pos,
                               q_home, log_data)) {
                std::cerr << "[3B] Variant A failed.\n";
                return 1;
            }
            std::cout << "--- Variant A complete. ---\n\n";
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }

        if (variants_str.find('B') != std::string::npos) {
            std::cout << "--- Variant B: GeometricPDController + ASIF (sim) ---\n";
            if (!run_variant_B(sim, model, data, col_model, col_data, line_traj, obstacle_pos,
                               q_home, log_data)) {
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
                  << "  NOTE: Place a physical sphere (r=" << kObstacleRadius << " m) at ["
                  << obstacle_pos.transpose() << "] -- " << kObstacleZOffset
                  << " m below the trajectory line.\n";

        Hardware hw(model.dof, robot_ip);
        if (!hw.is_running()) {
            std::cerr << "[3B] Failed to connect to robot at " << robot_ip << "\n";
            return 1;
        }

        JointState state(model.dof);
        hw.read(state);
        data.q = state.q;
        compute_jacobians(model, data);

        if (!run_variant_C(hw, model, data, col_model, col_data, line_traj, obstacle_pos, q_home,
                           log_data)) {
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
