#pragma once

#include <cmath>
#include <numbers>
#include <vector>

#include <smooth/spline/bspline.hpp>

#include <xarm_geo/core/manifold.h>
#include <xarm_geo/trajectory/trajectory.h>

namespace xarm_geo::trajectories {

    // --- Example: Wing Inspection Trajectory (Task-Space) ---
    //
    // Parabolic wing-surface scan: X sweeps laterally, Y scans along span,
    // Z follows the curvature (z = -curvature * x^2). Pitch aligns the
    // tool normal to the surface gradient; roll of pi points the tool
    // downward into the surface. Sampled at 100 points and fitted to a
    // degree-5 B-spline.

    class WingInspection {
    public:
        explicit WingInspection(const manifold::SE3 &anchor, double duration = 15.0,
                                double omega_sweep = 0.5, double omega_scan = 2.0,
                                double sweep_amp = 0.20, double scan_amp = 0.05,
                                double curvature = 2.5) {

            constexpr int num_samples = 100;
            std::vector<manifold::SE3> waypoints;
            waypoints.reserve(num_samples);

            for (int i = 0; i < num_samples; ++i) {
                const double t = (static_cast<double>(i) / (num_samples - 1)) * duration;

                // Parabolic Wing Surface: X Sweeps Laterally, Y Scans Along Span, Z Follows
                // Curvature.
                const double x_wing = sweep_amp * std::sin(omega_sweep * t);
                const double y_wing = scan_amp * std::cos(omega_scan * t);
                const double z_wing = -curvature * (x_wing * x_wing);

                const Eigen::Vector3d local_pos(x_wing, y_wing, z_wing);

                // Pitch Aligns Tool Normal to Wing Surface Gradient (atan of dz/dx),
                // Roll of Pi Points Tool Downward into Surface.
                const manifold::SO3 local_rot =
                    manifold::SO3::exp(std::atan(2.0 * curvature * x_wing) *
                                       Eigen::Vector3d::UnitY()) *
                    manifold::SO3::exp(std::numbers::pi * Eigen::Vector3d::UnitX());

                waypoints.emplace_back(anchor * manifold::SE3(local_rot, local_pos));
            }

            spline_ = build_se3_spline<5>(waypoints, duration);
        }

        [[nodiscard]] auto evaluate(double t, TaskSpaceTarget &target) const -> TrajectoryStatus {
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
    static_assert(TaskSpaceTrajectory<WingInspection>);

}  // namespace xarm_geo::trajectories
