#pragma once

#include <algorithm>
#include <concepts>
#include <limits>
#include <tuple>
#include <utility>

#include <xarm_geo/core/manifold.h>

namespace xarm_geo {

    // --- Disturbance Profile Concept ---
    //
    // Time-parameterised external wrench schedule for closed-loop robustness
    // tests. By convention, the returned wrench is in the body frame at the
    // body's frame origin, consistent with `Simulation::apply_external_wrench`.
    // Profiles must be deterministic and have no mutable state.

    template <typename T>
    concept DisturbanceProfile = requires(const T &profile, double t) {
        { profile.evaluate(t) } -> std::same_as<manifold::SE3::Wrench>;
    };

    // --- Constant Wrench ---
    //
    // Step-on / step-off disturbance: returns `w` for t in [t_start, t_end],
    // zero otherwise. Defaults: t_start = 0, t_end = +infinity.

    class ConstantWrench {
    public:
        explicit ConstantWrench(manifold::SE3::Wrench w, double t_start = 0.0,
                                double t_end = std::numeric_limits<double>::infinity())
            : w_(w), t_start_(t_start), t_end_(t_end) {}

        [[nodiscard]] auto evaluate(double t) const noexcept -> manifold::SE3::Wrench {
            if (t >= t_start_ && t <= t_end_) { return w_; }
            return manifold::SE3::Wrench();
        }

    private:
        manifold::SE3::Wrench w_;
        double t_start_, t_end_;
    };

    // --- Ramp Wrench ---
    //
    // Zero before t_start, linearly increasing from zero to `w_peak` over
    // `t_ramp` seconds, then held at `w_peak` until t_end (default +infinity).
    // Zero after t_end.

    class RampWrench {
    public:
        RampWrench(manifold::SE3::Wrench w_peak, double t_start, double t_ramp,
                   double t_end = std::numeric_limits<double>::infinity())
            : w_peak_(w_peak), t_start_(t_start), t_ramp_(t_ramp), t_end_(t_end) {}

        [[nodiscard]] auto evaluate(double t) const noexcept -> manifold::SE3::Wrench {
            if (t < t_start_ || t > t_end_) { return manifold::SE3::Wrench(); }

            // Linear Ramp from Zero to w_peak Over t_ramp Seconds.
            const double alpha =
                (t_ramp_ > 0.0) ? std::clamp((t - t_start_) / t_ramp_, 0.0, 1.0) : 1.0;
            return alpha * w_peak_;
        }

    private:
        manifold::SE3::Wrench w_peak_;
        double t_start_, t_ramp_, t_end_;
    };

    // --- Sum Disturbance ---
    //
    // Evaluates N profiles at the same t and returns their sum. Profiles may
    // overlap in time; their wrenches are added component-wise.

    template <DisturbanceProfile... Profiles> class SumDisturbance {
    public:
        explicit SumDisturbance(Profiles... profiles) : profiles_(std::move(profiles)...) {}

        [[nodiscard]] auto evaluate(double t) const noexcept -> manifold::SE3::Wrench {
            manifold::SE3::Wrench sum;
            std::apply([&](const auto &...p) { ((sum += p.evaluate(t)), ...); }, profiles_);
            return sum;
        }

    private:
        std::tuple<Profiles...> profiles_;
    };

    // --- Compile-Time Concept Verifications ---
    static_assert(DisturbanceProfile<ConstantWrench>);
    static_assert(DisturbanceProfile<RampWrench>);
    static_assert(DisturbanceProfile<SumDisturbance<ConstantWrench, RampWrench>>);

}  // namespace xarm_geo
