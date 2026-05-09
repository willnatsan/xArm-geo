#pragma once

#include <cmath>
#include <numbers>
#include <vector>

#include <smooth/spline/bspline.hpp>

#include <xarm_geo/core/manifold.h>
#include <xarm_geo/trajectory/trajectory.h>

namespace xarm_geo::trajectories {

    // --- Example: Pipe Inspection Trajectory (Task-Space) ---
    //
    // Circular arc in the YZ plane simulating travel around a pipe interior.
    // Pitch tracks the arc angle to keep the tool normal to the pipe wall.
    // Sampled at 100 points and fitted to a degree-5 B-spline.

    class PipeInspection {
    public:
        explicit PipeInspection(const manifold::SE3 &anchor, double duration = 15.0,
                                double omega = 0.4, double radius = 0.06) {

            constexpr int num_samples = 100;
            std::vector<manifold::SE3> waypoints;
            waypoints.reserve(num_samples);

            for (int i = 0; i < num_samples; ++i) {
                const double t = (static_cast<double>(i) / (num_samples - 1)) * duration;

                // Circular Arc in YZ Plane Simulating Travel Around Pipe Interior.
                const double y_pipe = radius * std::sin(omega * t);
                const double z_pipe = radius * (1.0 - std::cos(omega * t));

                const Eigen::Vector3d local_pos(0.0, y_pipe, -z_pipe);

                // Pitch Tracks Arc Angle to Keep Tool Normal to Pipe Wall.
                const double pitch_target = std::numbers::pi - (omega * t);

                const manifold::SO3 local_rot =
                    manifold::SO3::exp(pitch_target * Eigen::Vector3d::UnitY());

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
    static_assert(TaskTrajectory<PipeInspection>);

}  // namespace xarm_geo::trajectories
