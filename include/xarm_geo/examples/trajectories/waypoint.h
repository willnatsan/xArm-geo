#pragma once

#include <string_view>
#include <vector>

#include <smooth/spline/bspline.hpp>

#include <xarm_geo/core/manifold.h>
#include <xarm_geo/trajectory/trajectory.h>

namespace xarm_geo::trajectories {

    // --- Waypoint Trajectory ---
    //
    // Task-space trajectory interpolating a list of SE(3) waypoints via a
    // degree-5 B-spline.

    class Waypoint {
    public:
        static constexpr std::string_view kName = "Waypoint";

        Waypoint(const std::vector<manifold::SE3> &waypoints, double duration)
            : spline_(build_se3_spline<5>(waypoints, duration)), duration_(duration) {}

        [[nodiscard]] auto evaluate(double t, TaskTarget &target) const -> TrajectoryStatus {
            if (t < 0.0 || t > duration_) { return TrajectoryStatus::OUT_OF_DOMAIN; }

            manifold::SE3::Tangent vel;
            manifold::SE3::Tangent acc;

            // smooth's analytical spline derivatives give body twist and spatial acceleration.
            target.pose = spline_(t, smooth::OptTangent<manifold::SE3>{vel},
                                  smooth::OptTangent<manifold::SE3>{acc});
            target.twist = vel;
            target.spatial_acc = acc;

            return TrajectoryStatus::OK;
        }

        [[nodiscard]] auto duration() const noexcept -> double { return duration_; }

    private:
        manifold::SE3::Spline<5> spline_;
        double duration_;
    };

    // --- Compile-Time Concept Verification ---
    static_assert(TaskTrajectory<Waypoint>);

}  // namespace xarm_geo::trajectories
