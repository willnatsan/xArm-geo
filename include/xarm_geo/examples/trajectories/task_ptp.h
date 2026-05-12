#pragma once

#include <algorithm>

#include <Eigen/Dense>

#include <xarm_geo/core/manifold.h>
#include <xarm_geo/trajectory/trajectory.h>

namespace xarm_geo::trajectories {

    // --- Task Point-to-Point Trajectory ---
    //
    // Minimum-jerk SE(3) geodesic interpolation between two poses:
    //   g(tau)       = g_start * exp(s(tau) * delta),   delta = log(g_start^{-1} * g_end)
    //   s(tau)       = 10*tau^3 - 15*tau^4 + 6*tau^5,   tau = t / duration
    //   xi_b(t)      = s_dot(t)  * delta
    //   d/dt xi_b(t) = s_ddot(t) * delta
    //
    // Endpoints are exact rest-to-rest: body twist and spatial acceleration vanish
    // at t = 0 and t = duration. The path is a single SE(3) screw motion.

    class TaskPTP final {
    public:
        TaskPTP(const manifold::SE3 &start, const manifold::SE3 &end, double duration)
            : start_(start), delta_((start_.inverse() * end).log()), duration_(duration) {}

        [[nodiscard]] auto evaluate(double t, TaskTarget &target) const -> TrajectoryStatus {
            if (duration_ <= 0.0) {
                target.pose = start_ * manifold::SE3::exp(delta_);
                target.twist = manifold::SE3::Twist::Zero();
                target.spatial_acc = manifold::SE3::SpatialAcceleration::Zero();
                return TrajectoryStatus::OK;
            }

            t = std::clamp(t, 0.0, duration_);

            const double tau = t / duration_;
            const double tau2 = tau * tau;
            const double tau3 = tau2 * tau;
            const double tau4 = tau3 * tau;
            const double tau5 = tau4 * tau;

            const double s = (10.0 * tau3) - (15.0 * tau4) + (6.0 * tau5);
            const double s_dot = ((30.0 * tau2) - (60.0 * tau3) + (30.0 * tau4)) / duration_;
            const double s_ddot =
                ((60.0 * tau) - (180.0 * tau2) + (120.0 * tau3)) / (duration_ * duration_);

            target.pose = start_ * manifold::SE3::exp(s * delta_);
            target.twist = s_dot * delta_;
            target.spatial_acc = s_ddot * delta_;

            return TrajectoryStatus::OK;
        }

        [[nodiscard]] auto duration() const noexcept -> double { return duration_; }

    private:
        manifold::SE3 start_;
        manifold::SE3::Twist delta_;
        double duration_;
    };

    // --- Compile-Time Concept Verification ---
    static_assert(TaskTrajectory<TaskPTP>);

}  // namespace xarm_geo::trajectories
