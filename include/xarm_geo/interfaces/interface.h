#pragma once

#include <chrono>
#include <concepts>
#include <cstdint>

#include <Eigen/Dense>

#include <xarm_geo/core/motion.h>

namespace xarm_geo {

    enum class InterfaceStatus : std::uint8_t { OK, ERROR };

    // --- Hardware / Simulation Concepts ---

    // Base Interface Lifecycle Requirements
    // Note: No Dedicated `init()` Method -> Assuming RAII (Initialisation in Constructor)
    template <typename T>
    concept Interface = requires(T interface) {
        { interface.is_running() } -> std::same_as<bool>;
        { interface.shutdown() } -> std::same_as<void>;
    };

    // Defines the Joint Motion Type that can be READ from the Interface
    template <typename T, typename J>
    concept Observable = Interface<T> && JointMotion<J> && requires(T interface, J &data) {
        { interface.read(data) } noexcept -> std::same_as<InterfaceStatus>;
        { interface.read_time() } noexcept -> std::same_as<std::chrono::nanoseconds>;
    };

    // Defines the Joint Motion Type that can be WRITTEN to the Interface
    template <typename T, typename J>
    concept Controllable = Interface<T> && JointMotion<J> && requires(T interface, const J &cmd) {
        { interface.write(cmd) } noexcept -> std::same_as<InterfaceStatus>;
    };

}  // namespace xarm_geo
