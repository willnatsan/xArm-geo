#pragma once

#include <vector>

#include <smooth/spline/bspline.hpp>

#include <xarm_geo/core/manifold.h>
#include <xarm_geo/trajectory/trajectory.h>

namespace xarm_geo::trajectories {

    // --- Example: Waypoint Trajectory (Task-Space) ---
    //
    // Reference implementation of a task-space trajectory that interpolates
    // through a user-supplied list of SE(3) waypoints using a degree-5
    // B-spline.

    class Waypoint {
    public:
        Waypoint(const std::vector<manifold::SE3> &waypoints, double duration)
            : spline_(build_se3_spline<5>(waypoints, duration)) {}

        [[nodiscard]] auto evaluate(double t, TaskSpaceTarget &target) const -> TrajectoryStatus {
            manifold::SE3::Tangent vel;
            manifold::SE3::Tangent acc;

            // Using Analytical Derivatives from `smooth` to get Body Twist & Body Spatial
            // Acceleration.
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
    static_assert(TaskSpaceTrajectory<Waypoint>);

}  // namespace xarm_geo::trajectories
