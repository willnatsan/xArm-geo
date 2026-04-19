#pragma once

#include <chrono>
#include <concepts>
#include <cstdint>

#include <xarm_geo/core/motion.h>
#include <xarm_geo/core/system.h>
#include <xarm_geo/trajectory/trajectory.h>

namespace xarm_geo {

    // --- Controller Metadata ---

    enum class ControllerStatus : std::uint8_t { OK, ERROR };

    struct ControllerContext {
        std::chrono::nanoseconds dt;
    };

    // Union Concept for Supported Reference Target Types.
    template <typename R>
    concept ControllerReference = std::same_as<std::remove_cvref_t<R>, JointSpaceTarget> ||
                                  std::same_as<std::remove_cvref_t<R>, TaskSpaceTarget>;

    // --- Controller Concepts / Interfaces ---
    // Reference: TaskSpaceTarget || JointSpaceTarget
    // Command: JointPosition || JointVelocity || JointTorque

    template <typename T, typename Reference, typename Command>
    concept ModelFreeController =
        ControllerReference<Reference> && JointCommand<Command> &&
        requires(T &controller, const JointState &fb, const Reference &ref,
                 const ControllerContext &ctx, Command &out) {
            { controller.update(fb, ref, ctx, out) } noexcept -> std::same_as<ControllerStatus>;
            { controller.reset() } noexcept -> std::same_as<void>;
        };

    template <typename T, typename Reference, typename Command>
    concept ModelBasedController =
        ControllerReference<Reference> && JointCommand<Command> &&
        requires(T &controller, const Model &model, Data &data, const JointState &fb,
                 const Reference &ref, const ControllerContext &ctx, Command &out) {
            {
                controller.update(model, data, fb, ref, ctx, out)
            } noexcept -> std::same_as<ControllerStatus>;
            { controller.reset() } noexcept -> std::same_as<void>;
        };

    // Union Concept for Generic Controller
    template <typename T, typename Reference, typename Command>
    concept Controller =
        ModelFreeController<T, Reference, Command> || ModelBasedController<T, Reference, Command>;

}  // namespace xarm_geo
