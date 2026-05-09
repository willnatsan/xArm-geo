#pragma once

#include <cmath>
#include <numbers>
#include <vector>

#include <smooth/spline/bspline.hpp>

#include <xarm_geo/core/manifold.h>
#include <xarm_geo/trajectory/trajectory.h>

namespace xarm_geo::trajectories {

    // --- Example: Figure-Eight Trajectory (Task-Space) ---
    //
    // Lissajous figure-eight pattern in the XY plane with sinusoidal
    // Z-variation, anchored at a user-supplied SE(3) pose. Time-varying ZYX
    // Euler orientation (oscillating yaw / pitch with pi-offset roll to
    // point the tool downward). Sampled at 100 points and fitted to a
    // degree-5 B-spline for C^4 continuity.

    class FigureEight {
    public:
        explicit FigureEight(const manifold::SE3 &anchor, double duration = 15.0,
                             double omega = 1.0, double size_x = 0.15, double size_y = 0.10,
                             double size_z = 0.03) {

            constexpr int num_samples = 100;
            std::vector<manifold::SE3> waypoints;
            waypoints.reserve(num_samples);

            for (int i = 0; i < num_samples; ++i) {
                const double t = (static_cast<double>(i) / (num_samples - 1)) * duration;

                // Lissajous Figure-Eight Pattern in XY Plane with Sinusoidal Z-Variation.
                const Eigen::Vector3d local_pos(size_x * std::sin(omega * t),
                                                size_y * std::sin(2.0 * omega * t),
                                                size_z * std::sin(omega * t));

                // Time-Varying ZYX Euler Orientation: Oscillating Yaw/Pitch with Pi-Offset
                // Roll to Point Tool Downward.
                const double roll = std::numbers::pi + (0.3 * std::cos(omega * t));
                const double pitch = 0.5 * std::sin(2.0 * omega * t);
                const double yaw = 0.8 * std::sin(omega * t);
                const manifold::SO3 local_rot = manifold::rpy_to_SO3(roll, pitch, yaw);

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
    static_assert(TaskTrajectory<FigureEight>);

}  // namespace xarm_geo::trajectories
