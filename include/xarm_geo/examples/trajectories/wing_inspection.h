#pragma once

#include <cmath>
#include <numbers>
#include <utility>

#include <xarm_geo/core/manifold.h>
#include <xarm_geo/trajectory/trajectory.h>

namespace xarm_geo::trajectories {

    // --- Wing Inspection Trajectory ---
    //
    // Parabolic wing-surface scan: X sweeps laterally, Y scans along span,
    // Z follows the curvature (z = -curvature * x^2). Pitch aligns the
    // tool normal to the surface gradient; roll of pi points the tool
    // downward into the surface. Sampled at 100 points and fitted to a
    // degree-5 B-spline.

    class WingInspection final : public AnalyticTaskTrajectory {
    public:
        explicit WingInspection(const manifold::SE3 &anchor, double duration = 15.0,
                                double omega_sweep = 0.5, double omega_scan = 2.0,
                                double sweep_amp = 0.20, double scan_amp = 0.05,
                                double curvature = 2.5)
            : AnalyticTaskTrajectory(anchor, duration), omega_sweep_(omega_sweep),
              omega_scan_(omega_scan), sweep_amp_(sweep_amp), scan_amp_(scan_amp),
              curvature_(curvature) {
            build_spline();
        }

    protected:
        [[nodiscard]] auto sample(double t) const
            -> std::pair<manifold::SO3, Eigen::Vector3d> override {
            // Parabolic Wing Surface: X Sweeps Laterally, Y Scans Along Span, Z Follows
            // Curvature.
            const double x_wing = sweep_amp_ * std::sin(omega_sweep_ * t);
            const double y_wing = scan_amp_ * std::cos(omega_scan_ * t);
            const double z_wing = -curvature_ * (x_wing * x_wing);
            const Eigen::Vector3d local_pos(x_wing, y_wing, z_wing);

            // Pitch Aligns Tool Normal to Wing Surface Gradient (atan of dz/dx),
            // Roll of Pi Points Tool Downward into Surface.
            const manifold::SO3 local_rot =
                manifold::SO3::exp(std::atan(2.0 * curvature_ * x_wing) *
                                   Eigen::Vector3d::UnitY()) *
                manifold::SO3::exp(std::numbers::pi * Eigen::Vector3d::UnitX());

            return {local_rot, local_pos};
        }

    private:
        double omega_sweep_, omega_scan_, sweep_amp_, scan_amp_, curvature_;
    };

    // --- Compile-Time Concept Verification ---
    static_assert(TaskTrajectory<WingInspection>);

}  // namespace xarm_geo::trajectories
