#pragma once

#include <algorithm>
#include <concepts>
#include <string>

#include <Eigen/Dense>

#include <xarm_geo/core/manifold.h>
#include <xarm_geo/modelling/collision.h>
#include <xarm_geo/modelling/kinematics.h>

namespace xarm_geo {

    enum class TrajectoryStatus : std::uint8_t { OK, ERROR };

    // --- Task Space Trajectory ---

    // Note: All spatial quantities below are expressed in the end-effector BODY frame.
    struct TaskSpaceTarget {
        manifold::SE3 pose;
        manifold::SE3::Twist twist;
        manifold::SE3::SpatialAcceleration spatial_acc;
    };

    template <typename T>
    concept TaskSpaceTrajectory = requires(const T &traj, double t, TaskSpaceTarget &target) {
        { traj.evaluate(t, target) } -> std::same_as<TrajectoryStatus>;
    };

    // --- Joint Space Trajectory ---

    struct JointSpaceTarget {
        Eigen::VectorXd q;
        Eigen::VectorXd v;
        Eigen::VectorXd a;
        explicit JointSpaceTarget(int dof)
            : q(Eigen::VectorXd::Zero(dof)), v(Eigen::VectorXd::Zero(dof)),
              a(Eigen::VectorXd::Zero(dof)) {}
    };

    template <typename T>
    concept JointSpaceTrajectory = requires(const T &traj, double t, JointSpaceTarget &target) {
        { traj.evaluate(t, target) } -> std::same_as<TrajectoryStatus>;
    };

    // --- Trajectory Validation (Collision Detection) ---

    struct ValidationOptions {
        double integration_dt = 0.002;
        double collision_check_dt = 0.05;
    };

    struct ValidationResult {
        bool valid = true;
        double failure_time = -1.0;
        std::string reason;  // Human-readable failure reason
    };

    // Note: Trajectory Validation Functions need to be Implemented in Header due to Templating

    template <TaskSpaceTrajectory T>
    auto validate_trajectory(const Model &model, Data &data, const CollisionModel &col_model,
                             CollisionData &col_data, const T &trajectory, double duration,
                             const Eigen::Ref<const Eigen::VectorXd> &q_start,
                             const ValidationOptions &options = {}) -> ValidationResult {

        ValidationResult result;
        TaskSpaceTarget target;

        Eigen::VectorXd q_curr = q_start;

        // Set Up Controller Gains
        // TODO: Remove when Controller Function Parameter Integrated
        double kp = 8;

        // Set Up Integration & Collision Checking Loop
        double dt = options.integration_dt;
        int num_steps = static_cast<int>(duration / dt);
        double last_collision_t = -options.collision_check_dt;

        for (int s = 0; s <= num_steps; ++s) {
            // Compute Current State
            data.q = q_curr;
            compute_jacobians(model, data);

            // Evaluate Desired Pose @ Current Timestep
            double t = std::min(s * dt, duration);
            if (trajectory.evaluate(t, target) != TrajectoryStatus::OK) {
                result.valid = false;
                result.failure_time = t;
                result.reason = "trajectory_evaluate_failed";
                return result;
            }

            // Collision Checking
            if ((t - last_collision_t) >= options.collision_check_dt || s == num_steps) {
                update_geometry_poses(model, data, col_model, col_data);
                if (compute_collisions(col_model, col_data)) {
                    result.valid = false;
                    result.failure_time = t;
                    result.reason = "collision_detected";
                    return result;
                }
                last_collision_t = t;  // Reset the Timer
            }

            // Forward Simulate Controller Execution (If Not at Final Step)
            // TODO: Replace w/ Controller Function Parameter (Not Hardcoded to Geometric Diff. IK)
            if (s < num_steps) {
                manifold::SE3 pose_err_body = data.ee_pose.inverse() * target.pose;
                manifold::SE3::Twist twist_err_body = pose_err_body.log();

                manifold::SE3::Twist target_twist_ee = pose_err_body.Ad() * target.twist;
                manifold::SE3::Twist cmd_twist = target_twist_ee + (kp * twist_err_body);

                inverse_diff_kinematics(model, data, cmd_twist);
                q_curr += data.v_out * dt;
            }
        }

        return result;
    }

    template <JointSpaceTrajectory T>
    auto validate_trajectory(const Model &model, Data &data, const CollisionModel &col_model,
                             CollisionData &col_data, const T &trajectory, double duration,
                             const ValidationOptions &options = {}) -> ValidationResult {

        ValidationResult result;
        JointSpaceTarget target(model.dof);

        double dt = options.integration_dt;
        int num_steps = static_cast<int>(duration / dt);

        for (int s = 0; s <= num_steps; ++s) {
            double t = std::min(s * dt, duration);

            if (trajectory.evaluate(t, target) != TrajectoryStatus::OK) {
                result.valid = false;
                result.failure_time = t;
                result.reason = "trajectory_evaluate_failed";
                return result;
            }

            data.q = target.q;
            forward_kinematics(model, data);
            update_geometry_poses(model, data, col_model, col_data);

            if (compute_collisions(col_model, col_data)) {
                result.valid = false;
                result.failure_time = t;
                result.reason = "collision_detected";
                return result;
            }
        }

        return result;
    }

}  // namespace xarm_geo
