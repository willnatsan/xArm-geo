#pragma once

#include <concepts>
#include <string>

#include <Eigen/Dense>

#include <xarm_geo/core/manifold.h>
#include <xarm_geo/modelling/collision.h>
#include <xarm_geo/modelling/kinematics.h>

namespace xarm_geo {

    enum class TrajectoryStatus { OK, ERROR };

    // --- Task Space Trajectory ---

    struct TaskSpaceTarget {
        manifold::SE3 pose;
        manifold::SE3::Twist twist;
    };

    template <typename T>
    concept TaskSpaceTrajectory = requires(const T &traj, double t, TaskSpaceTarget &target) {
        { traj.evaluate(t, target) } -> std::same_as<TrajectoryStatus>;
    };

    // --- Joint Space Trajectory ---

    struct JointSpaceTarget {
        Eigen::VectorXd q;
        Eigen::VectorXd v;
        explicit JointSpaceTarget(int dof)
            : q(Eigen::VectorXd::Zero(dof)), v(Eigen::VectorXd::Zero(dof)) {}
    };

    template <typename T>
    concept JointSpaceTrajectory = requires(const T &traj, double t, JointSpaceTarget &target) {
        { traj.evaluate(t, target) } -> std::same_as<TrajectoryStatus>;
    };

    // --- Trajectory Validation (Collision Detection) ---

    struct ValidationOptions {
        int num_samples = 100;
        int interpolation_steps = 5;
        IKOptions ik_options = {};
    };

    struct ValidationResult {
        bool valid = true;
        double failure_time = -1.0;
        int failure_sample = -1;
        std::string reason;  // Human-readable failure reason
    };

    // Note: Trajectory Validation Functions need to be Implemented in Header due to Templating

    template <TaskSpaceTrajectory T>
    auto validate_trajectory(const Model &model, Data &data, const CollisionModel &col_model,
                             CollisionData &col_data, const T &trajectory, double duration,
                             const Eigen::Ref<const Eigen::VectorXd> &q_seed,
                             const ValidationOptions &options = {}) -> ValidationResult {

        ValidationResult result;
        TaskSpaceTarget target;
        Eigen::VectorXd q_prev = q_seed;

        for (int s = 0; s <= options.num_samples; ++s) {
            double t = (static_cast<double>(s) / options.num_samples) * duration;

            if (trajectory.evaluate(t, target) != TrajectoryStatus::OK) {
                result.valid = false;
                result.failure_time = t;
                result.failure_sample = s;
                result.reason = "trajectory_evaluate_failed";
                return result;
            }

            // Use Previous Solution as Seed for Collision-Aware IK
            bool ik_ok = inverse_kinematics(model, data, col_model, col_data, q_prev, target.pose,
                                            options.ik_options);
            if (!ik_ok) {
                result.valid = false;
                result.failure_time = t;
                result.failure_sample = s;
                result.reason = "ik_failed_or_collision";
                return result;
            }

            Eigen::VectorXd q_current = data.q_out;

            // Check Interpolated Configurations between q_prev and q_current
            if (s > 0 && options.interpolation_steps > 0) {
                for (int k = 1; k < options.interpolation_steps; ++k) {
                    double alpha = static_cast<double>(k) / options.interpolation_steps;
                    Eigen::VectorXd q_interp = (1.0 - alpha) * q_prev + alpha * q_current;

                    forward_kinematics(model, data, q_interp);
                    update_geometry_poses(model, data, col_model, col_data);

                    if (compute_collisions(col_model, col_data)) {
                        result.valid = false;
                        result.failure_time = t;
                        result.failure_sample = s;
                        result.reason = "interpolation_collision";
                        return result;
                    }
                }
            }

            q_prev = q_current;
        }

        return result;
    }

    template <JointSpaceTrajectory T>
    auto validate_trajectory(const Model &model, Data &data, const CollisionModel &col_model,
                             CollisionData &col_data, const T &trajectory, double duration,
                             const ValidationOptions &options = {}) -> ValidationResult {

        ValidationResult result;
        JointSpaceTarget target(model.dof);
        Eigen::VectorXd q_prev;

        for (int s = 0; s <= options.num_samples; ++s) {
            double t = (static_cast<double>(s) / options.num_samples) * duration;

            if (trajectory.evaluate(t, target) != TrajectoryStatus::OK) {
                result.valid = false;
                result.failure_time = t;
                result.failure_sample = s;
                result.reason = "trajectory_evaluate_failed";
                return result;
            }

            Eigen::VectorXd q_current = target.q;

            // Check Current Configuration
            forward_kinematics(model, data, q_current);
            update_geometry_poses(model, data, col_model, col_data);

            if (compute_collisions(col_model, col_data)) {
                result.valid = false;
                result.failure_time = t;
                result.failure_sample = s;
                result.reason = "collision";
                return result;
            }

            // Check Interpolated Configurations between q_prev and q_current
            if (s > 0 && options.interpolation_steps > 0) {
                for (int k = 1; k < options.interpolation_steps; ++k) {
                    double alpha = static_cast<double>(k) / options.interpolation_steps;
                    Eigen::VectorXd q_interp = (1.0 - alpha) * q_prev + alpha * q_current;

                    forward_kinematics(model, data, q_interp);
                    update_geometry_poses(model, data, col_model, col_data);

                    if (compute_collisions(col_model, col_data)) {
                        result.valid = false;
                        result.failure_time = t;
                        result.failure_sample = s;
                        result.reason = "interpolation_collision";
                        return result;
                    }
                }
            }

            q_prev = q_current;
        }

        return result;
    }

}  // namespace xarm_geo
