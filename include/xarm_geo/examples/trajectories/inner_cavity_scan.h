#pragma once

#include <cmath>
#include <numbers>
#include <utility>

#include <xarm_geo/core/manifold.h>
#include <xarm_geo/trajectory/trajectory.h>

namespace xarm_geo::trajectories {

    // --- Example: Inner Cavity Scan Trajectory (Task-Space) ---
    //
    // Small linear oscillation along the X-axis to probe cavity depth,
    // combined with an aggressive multi-axis orientation sweep at
    // incommensurate frequencies to maximise viewing-angle coverage inside
    // the cavity. Sampled at 100 points and fitted to a degree-5 B-spline.

    class InnerCavityScan final : public AnalyticTaskTrajectory {
    public:
        explicit InnerCavityScan(const manifold::SE3 &anchor, double duration = 15.0,
                                 double omega = 0.5, double pos_amp = 0.05)
            : AnalyticTaskTrajectory(anchor, duration), omega_(omega), pos_amp_(pos_amp) {
            build_spline();
        }

    protected:
        [[nodiscard]] auto sample(double t) const
            -> std::pair<manifold::SO3, Eigen::Vector3d> override {
            // Small Linear Oscillation Along X-Axis to Probe Cavity Depth.
            const Eigen::Vector3d local_pos(pos_amp_ * std::sin(omega_ * t), 0.0, 0.0);

            // Aggressive Multi-Axis Orientation Sweep with Incommensurate Frequencies
            // to Maximise Viewing Angle Coverage Inside the Cavity.
            const double roll_target = 1.5 * std::sin(1.1 * omega_ * t);
            const double pitch_target = std::numbers::pi - (1.2 * std::sin(0.9 * omega_ * t));
            const double yaw_target = 2.0 * std::cos(1.3 * omega_ * t);

            return {manifold::rpy_to_SO3(roll_target, pitch_target, yaw_target), local_pos};
        }

    private:
        double omega_, pos_amp_;
    };

    // --- Compile-Time Concept Verification ---
    static_assert(TaskTrajectory<InnerCavityScan>);

}  // namespace xarm_geo::trajectories
