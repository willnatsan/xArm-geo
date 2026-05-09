#pragma once

#include <cmath>
#include <numbers>
#include <vector>

#include <smooth/spline/bspline.hpp>

#include <xarm_geo/core/manifold.h>
#include <xarm_geo/trajectory/trajectory.h>

namespace xarm_geo::trajectories {

    // --- Example: Inner Cavity Scan Trajectory (Task-Space) ---
    //
    // Small linear oscillation along the X-axis to probe cavity depth,
    // combined with an aggressive multi-axis orientation sweep at
    // incommensurate frequencies to maximise viewing-angle coverage inside
    // the cavity. Sampled at 100 points and fitted to a degree-5 B-spline.

    class InnerCavityScan {
    public:
        explicit InnerCavityScan(const manifold::SE3 &anchor, double duration = 15.0,
                                 double omega = 0.5, double pos_amp = 0.05) {

            constexpr int num_samples = 100;
            std::vector<manifold::SE3> waypoints;
            waypoints.reserve(num_samples);

            for (int i = 0; i < num_samples; ++i) {
                const double t = (static_cast<double>(i) / (num_samples - 1)) * duration;

                // Small Linear Oscillation Along X-Axis to Probe Cavity Depth.
                const Eigen::Vector3d local_pos(pos_amp * std::sin(omega * t), 0.0, 0.0);

                // Aggressive Multi-Axis Orientation Sweep with Incommensurate Frequencies
                // to Maximise Viewing Angle Coverage Inside the Cavity.
                const double roll_target = 1.5 * std::sin(1.1 * omega * t);
                const double pitch_target = std::numbers::pi - (1.2 * std::sin(0.9 * omega * t));
                const double yaw_target = 2.0 * std::cos(1.3 * omega * t);

                const manifold::SO3 local_rot =
                    manifold::rpy_to_SO3(roll_target, pitch_target, yaw_target);

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
    static_assert(TaskTrajectory<InnerCavityScan>);

}  // namespace xarm_geo::trajectories
