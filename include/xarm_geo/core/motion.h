#pragma once

#include <concepts>
#include <type_traits>

#include <Eigen/Dense>

namespace xarm_geo {

    // Empty Base Struct & Concept Definition (For Compile-Time Checking)

    struct JointMotionBase {};

    template <typename T>
    concept JointMotion = std::derived_from<std::remove_cvref_t<T>, JointMotionBase>;

    // --- Joint Motion Types (For Interfaces & Controllers) ---

    struct JointPosition : JointMotionBase {
        Eigen::VectorXd q;
        explicit JointPosition(int dof) : q(Eigen::VectorXd::Zero(dof)) {}
    };

    struct JointVelocity : JointMotionBase {
        Eigen::VectorXd v;
        explicit JointVelocity(int dof) : v(Eigen::VectorXd::Zero(dof)) {}
    };

    struct JointTorque : JointMotionBase {
        Eigen::VectorXd tau;
        explicit JointTorque(int dof) : tau(Eigen::VectorXd::Zero(dof)) {}
    };

    struct JointState : JointMotionBase {
        Eigen::VectorXd q;
        Eigen::VectorXd v;
        Eigen::VectorXd tau;
        explicit JointState(int dof)
            : q(Eigen::VectorXd::Zero(dof)), v(Eigen::VectorXd::Zero(dof)),
              tau(Eigen::VectorXd::Zero(dof)) {}
    };
}  // namespace xarm_geo
