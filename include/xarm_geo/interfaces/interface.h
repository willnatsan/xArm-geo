#pragma once

#include <concepts>

#include <Eigen/Dense>

namespace xarm_geo {

    // --- Joint Motion Types (For Reading / Writing) ---

    struct JointPosition {
        Eigen::VectorXd q;
        explicit JointPosition(int dof) : q(Eigen::VectorXd::Zero(dof)) {}
    };

    struct JointVelocity {
        Eigen::VectorXd v;
        explicit JointVelocity(int dof) : v(Eigen::VectorXd::Zero(dof)) {}
    };

    struct JointTorque {
        Eigen::VectorXd tau;
        explicit JointTorque(int dof) : tau(Eigen::VectorXd::Zero(dof)) {}
    };

    // --- Hardware / Simulation Concepts ---

    // Base Lifecycle Requirements
    template <typename T>
    concept Interface = requires(T interface) {
        // Must have an initialised method to check for successful initialisation
        { interface.initialised() } -> std::same_as<bool>;

        // Must have a shutdown method to safely disengage the system
        { interface.shutdown() };
    };

    // Defines the Joint Motion Type that can be READ from the Interface
    template <typename T, typename JointMotion>
    concept Observable = Interface<T> && requires(T interface, JointMotion &data) {
        { interface.read(data) };
    };

    // Defines the Joint Motion Type that can be WRITTEN to the Interface
    template <typename T, typename JointMotion>
    concept Controllable = Interface<T> && requires(T interface, const JointMotion &cmd) {
        { interface.write(cmd) };
    };

}  // namespace xarm_geo
