#pragma once

#include <cmath>
#include <numbers>
#include <utility>

#include <xarm_geo/core/manifold.h>
#include <xarm_geo/trajectory/trajectory.h>

namespace xarm_geo::trajectories {

    // --- Figure-Eight Trajectory ---
    //
    // Lissajous figure-eight in the XY plane with sinusoidal Z-variation,
    // anchored at a user-supplied SE(3) pose. Tool-down orientation with
    // oscillating yaw and pitch.

    class FigureEight final : public AnalyticTaskTrajectory {
    public:
        explicit FigureEight(const manifold::SE3 &anchor, double duration = 15.0,
                             double omega = 1.0, double size_x = 0.15, double size_y = 0.10,
                             double size_z = 0.03)
            : AnalyticTaskTrajectory(anchor, duration), omega_(omega), sx_(size_x), sy_(size_y),
              sz_(size_z) {
            build_spline();
        }

    protected:
        [[nodiscard]] auto sample(double t) const
            -> std::pair<manifold::SO3, Eigen::Vector3d> override {
            const Eigen::Vector3d local_pos(sx_ * std::sin(omega_ * t),
                                            sy_ * std::sin(2.0 * omega_ * t),
                                            sz_ * std::sin(omega_ * t));

            // Tool-down roll offset; oscillating yaw and pitch.
            const double roll = std::numbers::pi + (0.3 * std::cos(omega_ * t));
            const double pitch = 0.5 * std::sin(2.0 * omega_ * t);
            const double yaw = 0.8 * std::sin(omega_ * t);

            return {manifold::rpy_to_SO3(roll, pitch, yaw), local_pos};
        }

    private:
        double omega_, sx_, sy_, sz_;
    };

    // --- Compile-Time Concept Verification ---
    static_assert(TaskTrajectory<FigureEight>);

}  // namespace xarm_geo::trajectories
