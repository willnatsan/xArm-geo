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

    // --- Controller Concepts / Interfaces ---
    // Note: `Feedback` is the measured state; `Command` is the actuation output.

    template <typename T, typename Feedback, typename Command>
    concept ModelFreeController =
        JointMotion<Feedback> && JointMotion<Command> &&
        requires(T &controller, const Feedback &fb, const JointSpaceTarget &ref,
                 const ControllerContext &ctx, Command &out) {
            { controller.update(fb, ref, ctx, out) } noexcept -> std::same_as<ControllerStatus>;
            { controller.reset() } noexcept -> std::same_as<void>;
        };

    template <typename T, typename Feedback, typename Command>
    concept ModelBasedController =
        JointMotion<Feedback> && JointMotion<Command> &&
        requires(T &controller, const Model &model, Data &data, const Feedback &fb,
                 const JointSpaceTarget &ref, const ControllerContext &ctx, Command &out) {
            {
                controller.update(model, data, fb, ref, ctx, out)
            } noexcept -> std::same_as<ControllerStatus>;
            { controller.reset() } noexcept -> std::same_as<void>;
        };

    // Union Concept for Generic Controller
    template <typename T, typename Feedback, typename Command>
    concept Controller =
        ModelFreeController<T, Feedback, Command> || ModelBasedController<T, Feedback, Command>;

}  // namespace xarm_geo
