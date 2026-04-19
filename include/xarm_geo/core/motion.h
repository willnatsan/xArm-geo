#pragma once

#include <concepts>
#include <type_traits>

#include <Eigen/Dense>

namespace xarm_geo {

    // --- Observation Type (Read from Interface | Controller Feedback) ---

    struct JointState {
        Eigen::VectorXd q;
        Eigen::VectorXd v;
        Eigen::VectorXd tau;
        explicit JointState(int dof)
            : q(Eigen::VectorXd::Zero(dof)), v(Eigen::VectorXd::Zero(dof)),
              tau(Eigen::VectorXd::Zero(dof)) {}
    };

    // --- Command Types (Write to Interface | Controller Output) ---

    struct JointCommandBase {};

    template <typename T>
    concept JointCommand = std::derived_from<std::remove_cvref_t<T>, JointCommandBase>;

    struct JointPosition : JointCommandBase {
        Eigen::VectorXd q;
        explicit JointPosition(int dof) : q(Eigen::VectorXd::Zero(dof)) {}
    };

    struct JointVelocity : JointCommandBase {
        Eigen::VectorXd v;
        explicit JointVelocity(int dof) : v(Eigen::VectorXd::Zero(dof)) {}
    };

    struct JointTorque : JointCommandBase {
        Eigen::VectorXd tau;
        explicit JointTorque(int dof) : tau(Eigen::VectorXd::Zero(dof)) {}
    };

}  // namespace xarm_geo
