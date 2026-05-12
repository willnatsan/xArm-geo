#pragma once

#include <algorithm>
#include <cmath>
#include <numbers>
#include <utility>

#include <xarm_geo/core/manifold.h>
#include <xarm_geo/trajectory/trajectory.h>

namespace xarm_geo::trajectories {

    // --- Tilting Circle Trajectory ---
    //
    // Circular orbit in the XY plane that smoothly transitions from
    // horizontal to vertical at the trajectory midpoint via a min-jerk tilt
    // ramp (C^2 continuous polynomial). The end-effector points radially
    // inward throughout. Sampled at 100 points and fitted to a degree-5
    // B-spline for C^4 continuity.

    class TiltingCircle final : public AnalyticTaskTrajectory {
    public:
        explicit TiltingCircle(const manifold::SE3 &anchor, double duration = 15.0,
                               double omega = 2.0, double R = 0.12)
            : AnalyticTaskTrajectory(anchor, duration), omega_(omega), R_(R) {
            build_spline();
        }

    protected:
        [[nodiscard]] auto sample(double t) const
            -> std::pair<manifold::SO3, Eigen::Vector3d> override {
            const double transition_start = duration() / 2.0;
            const double transition_duration = 1.0;

            // Min-Jerk Tilt Ramp (C^2 Continuous) to Avoid Acceleration Spikes at
            // Boundaries. Polynomial: s(p) = 10p^3 - 15p^4 + 6p^5 with s(0) = 0,
            // s(1) = 1, s'(0) = s'(1) = 0.
            const double progress =
                std::clamp((t - transition_start) / transition_duration, 0.0, 1.0);
            const double p2 = progress * progress;
            const double p3 = p2 * progress;
            const double p4 = p3 * progress;
            const double p5 = p4 * progress;
            const double s = (10.0 * p3) - (15.0 * p4) + (6.0 * p5);
            const double tilt = s * (std::numbers::pi / 2.0);

            // Circular Path in XY Plane (Horizontal).
            const Eigen::Vector3d p_base(R_ * std::cos(omega_ * t), R_ * std::sin(omega_ * t), 0.0);

            // Tilt the Circular Orbit Plane About X-Axis Using SO3 Action on Vector
            // (Smoothly Transitions from Horizontal to Vertical Circle at Midpoint).
            const manifold::SO3 tilt_rot = manifold::SO3::exp(tilt * Eigen::Vector3d::UnitX());
            const Eigen::Vector3d local_pos = tilt_rot * p_base;

            // End-Effector Points Radially Inward: Pi Flips Tool Down, Tilt Compensates
            // for the Orbit Plane Rotation to Maintain Normal-to-Circle Orientation.
            const manifold::SO3 local_rot =
                manifold::SO3::exp((std::numbers::pi - tilt) * Eigen::Vector3d::UnitX());

            return {local_rot, local_pos};
        }

    private:
        double omega_, R_;
    };

    // --- Compile-Time Concept Verification ---
    static_assert(TaskTrajectory<TiltingCircle>);

}  // namespace xarm_geo::trajectories
