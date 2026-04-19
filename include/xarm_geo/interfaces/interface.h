#pragma once

#include <chrono>
#include <concepts>
#include <cstdint>

#include <xarm_geo/core/motion.h>

namespace xarm_geo {

    enum class InterfaceStatus : std::uint8_t { OK, ERROR };

    // --- Interface Concepts ---

    // Base Interface: Lifecycle Management + State Reading (JointState)
    // Note: No Dedicated `init()` Method -> Assuming RAII (Initialisation in Constructor)
    template <typename T>
    concept Interface = requires(T interface, JointState &state) {
        { interface.is_running() } -> std::same_as<bool>;
        { interface.shutdown() } -> std::same_as<void>;
        { interface.read(state) } noexcept -> std::same_as<InterfaceStatus>;
        { interface.read_time() } noexcept -> std::same_as<std::chrono::nanoseconds>;
    };

    // Extends Interface w/ Command Writing (JointPosition | JointVelocity | JointTorque) Capability
    template <typename T, typename J>
    concept Controllable = Interface<T> && JointCommand<J> && requires(T interface, const J &cmd) {
        { interface.write(cmd) } noexcept -> std::same_as<InterfaceStatus>;
    };

}  // namespace xarm_geo
