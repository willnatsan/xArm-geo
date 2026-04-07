#pragma once

#include <chrono>
#include <concepts>

#include <Eigen/Dense>

namespace xarm_geo {
    enum class InterfaceStatus { OK, ERROR };

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
    // Note: No Dedicated `init()` Method -> Assuming RAII (Initialisation in Constructor)
    template <typename T>
    concept Interface = requires(T interface) {
        { interface.is_running() } -> std::same_as<bool>;
        { interface.shutdown() } -> std::same_as<void>;
    };

    // Defines the Joint Motion Type that can be READ from the Interface
    template <typename T, typename JointMotion>
    concept Observable = Interface<T> && requires(T interface, JointMotion &data) {
        { interface.read(data) } -> std::same_as<InterfaceStatus>;
        { interface.read_time() } -> std::same_as<std::chrono::nanoseconds>;
    };

    // Defines the Joint Motion Type that can be WRITTEN to the Interface
    template <typename T, typename JointMotion>
    concept Controllable = Interface<T> && requires(T interface, const JointMotion &cmd) {
        { interface.write(cmd) } -> std::same_as<InterfaceStatus>;
    };

}  // namespace xarm_geo
