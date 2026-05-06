#include "xarm_geo/trajectory/trajectory.h"
#include <chrono>
#include <iostream>
#include <numbers>
#include <string>
#include <thread>

#include <coal/shape/geometric_shapes.h>
#include <xarm_geo/core/system.h>
#include <xarm_geo/interfaces/hardware.h>
#include <xarm_geo/modelling/collision.h>
#include <xarm_geo/modelling/kinematics.h>
#include <xarm_geo/trajectory/sample_trajectories.h>
#include <xarm_geo/utils/model_builder.h>

// --- Helper: Safety Enclosure ---
// Adds static box geometries around the robot to prevent workspace exits
// void add_safety_enclosure(xarm_geo::CollisionModel &col_model) {
//     double thickness = 0.05;
//     double size = 1.2;
//     double height = 2.0;

//     auto wall_geom = std::make_shared<coal::Box>(thickness, size, height);
//     auto floor_geom = std::make_shared<coal::Box>(size, size, thickness);

//     // Front/Back/Left/Right Poses (Joint 0 = World)
//     xarm_geo::manifold::SE3 front(xarm_geo::manifold::SO3::Identity(), {0.6, 0, height/2});
//     xarm_geo::manifold::SE3 back(xarm_geo::manifold::SO3::Identity(), {-0.6, 0, height/2});
//     xarm_geo::manifold::SE3 left(xarm_geo::manifold::SO3::Identity(), {0, 0.6, height/2});
//     xarm_geo::manifold::SE3 right(xarm_geo::manifold::SO3::Identity(), {0, -0.6, height/2});
//     xarm_geo::manifold::SE3 floor(xarm_geo::manifold::SO3::Identity(), {0, 0, -thickness/2});

//     col_model.add_geometry("wall_front", 0, front, wall_geom);
//     col_model.add_geometry("wall_back", 0, back, wall_geom);
//     col_model.add_geometry("wall_left", 0, left, wall_geom);
//     col_model.add_geometry("wall_right", 0, right, wall_geom);
//     col_model.add_geometry("floor", 0, floor, floor_geom);

//     col_model.add_all_collision_pairs();
// }

// --- PHASE A: JOINT PTP CONTROL (Approach/Return) ---
auto run_joint_ptp_hw(xarm_geo::Hardware &hw, const xarm_geo::trajectories::JointPTP &traj,
                      double duration, xarm_geo::JointState &state,
                      xarm_geo::JointVelocity &control_target) -> bool {
    double t = 0.0, dt = 0.002, kp_joint = 5.0;
    xarm_geo::JointSpaceTarget target(state.q.size());
    auto next_tick = std::chrono::steady_clock::now();

    while (t < duration && hw.is_running()) {
        if (hw.read(state) != xarm_geo::InterfaceStatus::OK) return false;
        if (traj.evaluate(t, target) != xarm_geo::TrajectoryStatus::OK) return false;

        control_target.v = target.v + (kp_joint * (target.q - state.q));
        if (hw.write(control_target) != xarm_geo::InterfaceStatus::OK) return false;

        t += dt;
        next_tick += std::chrono::duration_cast<std::chrono::steady_clock::duration>(
            std::chrono::duration<double>(dt));
        std::this_thread::sleep_until(next_tick);
    }
    return true;
}

// --- PHASE B: TASK SPACE CONTROL (The Tilting Circle) ---
// Uses your custom inverse_diff_kinematics with DLS and Velocity Scaling
template <xarm_geo::TaskSpaceTrajectory T>
auto run_task_space_hw(xarm_geo::Hardware &hw, const xarm_geo::Model &model, xarm_geo::Data &data,
                       const T &traj, double duration, xarm_geo::JointState &state,
                       xarm_geo::JointVelocity &control_target) -> bool {
    double t = 0.0, dt = 0.002, kp = 8.0;
    xarm_geo::IKOptions ik_opt;
    ik_opt.damping = 0.1;  // Damping for your DLS logic

    xarm_geo::TaskSpaceTarget target;
    auto next_tick = std::chrono::steady_clock::now();

    while (t < duration && hw.is_running()) {
        if (hw.read(state) != xarm_geo::InterfaceStatus::OK) return false;

        // Update kinematics and Jacobian
        data.q = state.q;
        xarm_geo::forward_kinematics(model, data);
        xarm_geo::compute_jacobians(model, data);

        if (traj.evaluate(t, target) != xarm_geo::TrajectoryStatus::OK) return false;

        // Geometric Manifold Control (from your simulation code)
        xarm_geo::manifold::SE3 pose_err_body = data.ee_pose.inverse() * target.pose;
        xarm_geo::manifold::SE3::Twist twist_err_body = pose_err_body.log();
        xarm_geo::manifold::SE3::Twist target_twist_ee = pose_err_body.Ad() * target.twist;

        xarm_geo::manifold::SE3::Twist cmd_twist = target_twist_ee + (kp * twist_err_body);

        // Call your custom solver
        xarm_geo::inverse_diff_kinematics(model, data, cmd_twist, ik_opt);
        control_target.v = data.v_out;

        if (hw.write(control_target) != xarm_geo::InterfaceStatus::OK) return false;

        t += dt;
        next_tick += std::chrono::duration_cast<std::chrono::steady_clock::duration>(
            std::chrono::duration<double>(dt));
        std::this_thread::sleep_until(next_tick);
    }
    return true;
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <robot_ip>\n";
        return 1;
    }
    std::string robot_ip = argv[1];

    try {
        // --- Setup ---
        double circle_duration = 30.0;  // HALF SPEED (original was 15.0)
        double transition_time = 4.0;

        xarm_geo::Model model = xarm_geo::build_model(6, "XI130412C23L45");
        xarm_geo::Data data(model);

        // Safety Walls
        xarm_geo::CollisionModel col_model = xarm_geo::build_collision_model(model, true);
        // add_safety_enclosure(col_model);
        xarm_geo::CollisionData col_data(col_model);

        xarm_geo::Hardware hw(model.dof, robot_ip);
        if (!hw.is_running()) return 1;

        xarm_geo::JointState state(model.dof);
        xarm_geo::JointVelocity control_target(model.dof);
        hw.read(state);

        // --- Define Trajectory Anchor ---
        Eigen::VectorXd q_home = Eigen::VectorXd::Zero(model.dof);
        q_home[0] = 1.5 * std::numbers::pi;

        double q0 = q_home[0];
        Eigen::Vector3d center(0.35 * std::cos(q0), 0.35 * std::sin(q0), 0.35);
        xarm_geo::manifold::SO3 rot = xarm_geo::manifold::SO3::exp(Eigen::Vector3d::UnitZ() *
                                                                   (q0 - (1.5 * std::numbers::pi)));
        xarm_geo::manifold::SE3 anchor(rot, center);

        xarm_geo::trajectories::TiltingCircle circle_traj(anchor, circle_duration);

        // --- Pre-Flight ---
        xarm_geo::TaskSpaceTarget start_target;
        if (circle_traj.evaluate(0.0, start_target) != xarm_geo::TrajectoryStatus::OK) {
            std::cerr << "[ABORT] Failed to Evaluate Start Pose\n";
            return 1;
        }

        if (!xarm_geo::inverse_kinematics(model, data, col_model, col_data, q_home,
                                          start_target.pose)) {
            std::cerr << "[ABORT] No Collision-Free Start Pose Found!\n";
            return 1;
        }
        Eigen::VectorXd q_start = data.q_out;

        auto val = xarm_geo::validate_trajectory(model, data, col_model, col_data, circle_traj,
                                                 circle_duration, q_start);
        if (!val.valid) {
            std::cerr << "[ABORT] Safety Violation: " << val.reason << "\n";
            return 1;
        }

        // --- Execution ---
        std::cout << "[PHASE 0] Moving to Home... ";
        xarm_geo::trajectories::JointPTP h_traj(state.q, q_home, transition_time);
        run_joint_ptp_hw(hw, h_traj, transition_time, state, control_target);
        std::cout << "Done.\n[PHASE 1] Approaching Circle... ";

        xarm_geo::trajectories::JointPTP a_traj(q_home, q_start, transition_time);
        run_joint_ptp_hw(hw, a_traj, transition_time, state, control_target);
        std::cout << "Done.\n[PHASE 2] Executing Tilting Circle (Half Speed)...\n";

        run_task_space_hw(hw, model, data, circle_traj, circle_duration, state, control_target);

        std::cout << "[PHASE 3] Returning Home... ";
        hw.read(state);
        xarm_geo::trajectories::JointPTP r_traj(state.q, q_home, transition_time);
        run_joint_ptp_hw(hw, r_traj, transition_time, state, control_target);

        std::cout << "SUCCESS.\n";
        hw.shutdown();

    } catch (const std::exception &e) {
        std::cerr << "CRITICAL: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
