#pragma once

#include <concepts>

#include <Eigen/Dense>

#include <xarm_geo/core/manifold.h>

namespace xarm_geo {

    enum class TrajectoryStatus { OK, ERROR };

    // --- Task Space ---

    struct TaskSpaceTarget {
        manifold::SE3 pose;
        manifold::SE3::Twist twist;
    };

    template <typename T>
    concept TaskSpaceTrajectory = requires(const T &traj, double t, TaskSpaceTarget &target) {
        { traj.evaluate(t, target) } -> std::same_as<TrajectoryStatus>;
    };

    // --- Joint Space ---

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

}  // namespace xarm_geo
