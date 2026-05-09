#pragma once

#include <algorithm>
#include <cmath>
#include <numbers>
#include <vector>

#include <smooth/spline/bspline.hpp>

#include <xarm_geo/core/manifold.h>
#include <xarm_geo/trajectory/trajectory.h>

namespace xarm_geo::trajectories {

    // --- Example: Tilting Circle Trajectory (Task-Space) ---
    //
    // Circular orbit in the XY plane that smoothly transitions from
    // horizontal to vertical at the trajectory midpoint via a min-jerk tilt
    // ramp (C^2 continuous polynomial). The end-effector points radially
    // inward throughout. Sampled at 100 points and fitted to a degree-5
    // B-spline.

    class TiltingCircle {
    public:
        explicit TiltingCircle(const manifold::SE3 &anchor, double duration = 15.0,
                               double omega = 2.0, double R = 0.12) {

            constexpr int num_samples = 100;
            std::vector<manifold::SE3> waypoints;
            waypoints.reserve(num_samples);

            const double transition_start = duration / 2.0;
            const double transition_duration = 1.0;

            for (int i = 0; i < num_samples; ++i) {
                const double t = (static_cast<double>(i) / (num_samples - 1)) * duration;

                // Min-Jerk Tilt Ramp (C^2 Continuous) to Avoid Acceleration Spikes at
                // Boundaries. Polynomial: s(p) = 10p^3 - 15p^4 + 6p^5 with s(0) = 0,
                // s(1) = 1, s'(0) = s'(1) = 0.
                const double progress =
                    std::clamp((t - transition_start) / transition_duration, 0.0, 1.0);
                const double progress2 = progress * progress;
                const double progress3 = progress2 * progress;
                const double progress4 = progress3 * progress;
                const double progress5 = progress4 * progress;
                const double s = (10.0 * progress3) - (15.0 * progress4) + (6.0 * progress5);
                const double tilt = s * (std::numbers::pi / 2.0);

                // Circular Path in XY Plane (Horizontal).
                const Eigen::Vector3d p_base(R * std::cos(omega * t), R * std::sin(omega * t), 0.0);

                // Tilt the Circular Orbit Plane About X-Axis Using SO3 Action on Vector
                // (Smoothly Transitions from Horizontal to Vertical Circle at Midpoint).
                const manifold::SO3 tilt_rot = manifold::SO3::exp(tilt * Eigen::Vector3d::UnitX());
                const Eigen::Vector3d local_pos = tilt_rot * p_base;

                // End-Effector Points Radially Inward: Pi Flips Tool Down, Tilt Compensates
                // for the Orbit Plane Rotation to Maintain Normal-to-Circle Orientation.
                const manifold::SO3 local_rot =
                    manifold::SO3::exp((std::numbers::pi - tilt) * Eigen::Vector3d::UnitX());

                waypoints.emplace_back(anchor * manifold::SE3(local_rot, local_pos));
            }

            spline_ = build_se3_spline<5>(waypoints, duration);
        }

        [[nodiscard]] auto evaluate(double t, TaskTarget &target) const -> TrajectoryStatus {
            manifold::SE3::Tangent vel;
            manifold::SE3::Tangent acc;

            target.pose = spline_(t, smooth::OptTangent<manifold::SE3>{vel},
                                  smooth::OptTangent<manifold::SE3>{acc});
            target.twist = vel;
            target.spatial_acc = acc;

            return TrajectoryStatus::OK;
        }

    private:
        manifold::SE3::Spline<5> spline_;
    };

    // --- Compile-Time Concept Verification ---
    static_assert(TaskTrajectory<TiltingCircle>);

}  // namespace xarm_geo::trajectories
