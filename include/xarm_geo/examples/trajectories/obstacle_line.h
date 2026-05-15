#pragma once

#include <cmath>
#include <string_view>
#include <utility>

#include <Eigen/Dense>

#include <xarm_geo/core/manifold.h>
#include <xarm_geo/trajectory/trajectory.h>

namespace xarm_geo::trajectories {

    // --- Obstacle Line Trajectory ---
    //
    // Straight-line SE(3) trajectory from `start` to `end` with min-jerk
    // time-scaling (10t^3 - 15t^4 + 6t^5) so the EE accelerates from rest,
    // reaches peak speed at the midpoint, and decelerates to rest at the end.
    // Both translation and orientation follow the same scalar profile so they
    // complete simultaneously with zero boundary velocities and accelerations.
    //
    // Orientation is interpolated via right-trivialized SLERP on SO(3):
    //   R(s) = R_start * exp(s * log(R_start^{-1} * R_end))
    //
    // The trajectory is designed to pass through any obstacle placed at the
    // geometric midpoint of the line; Experiment 3B registers such an obstacle
    // in the CollisionModel and relies on the safety layer (OptIK / ASIF) to
    // deflect the EE path at runtime. The trajectory itself is intentionally
    // naive -- it does not avoid the obstacle -- so the comparison between a
    // constraint-unaware controller and a safety-aware controller is clean.
    //
    // The anchor convention differs from other AnalyticTaskTrajectory subclasses:
    // `start_pose` is passed as the anchor. sample(t) returns a LOCAL pose
    // (displacement from start), which the base composes with the anchor to give
    // the world-frame SE(3) target. `end_pose` is stored as the total
    // start-relative displacement SE3(R_rel, p_rel).
    //
    // Public accessors:
    //   start_pose()  -- start (= anchor; world-frame)
    //   end_pose()    -- end (world-frame, reconstructed from anchor * rel_)
    //   midpoint()    -- Euclidean midpoint of the translational segment (world-frame)

    class ObstacleLine final : public AnalyticTaskTrajectory {
    public:
        static constexpr std::string_view kName = "ObstacleLine";

        // `start` and `end` are world-frame SE(3) poses.
        // `duration` is the total trajectory time in seconds.
        ObstacleLine(const manifold::SE3 &start, const manifold::SE3 &end, double duration = 8.0)
            : AnalyticTaskTrajectory(start, duration), start_(start), end_(end),
              // Start-relative displacement: R_rel = R_start^{-1} * R_end,
              // p_rel = R_start^{-1} * (p_end - p_start).
              rel_(start.inverse() * end) {
            build_spline();
        }

        [[nodiscard]] auto start_pose() const noexcept -> const manifold::SE3 & { return start_; }
        [[nodiscard]] auto end_pose() const noexcept -> const manifold::SE3 & { return end_; }
        [[nodiscard]] auto midpoint() const noexcept -> Eigen::Vector3d {
            return 0.5 * (start_.r3() + end_.r3());
        }

    protected:
        // Returns (local_rot, local_pos) relative to `start_` (the anchor).
        // The base class composes: world_pose = anchor * SE3(local_rot, local_pos).
        [[nodiscard]] auto sample(double t) const
            -> std::pair<manifold::SO3, Eigen::Vector3d> override {

            const double T = duration();
            const double tau = (T > 0.0) ? (t / T) : 1.0;

            // Min-jerk profile: s(tau) = 10t^3 - 15t^4 + 6t^5.
            // s(0) = 0, s(1) = 1, s'(0) = s'(1) = s''(0) = s''(1) = 0.
            const double tau2 = tau * tau;
            const double tau3 = tau2 * tau;
            const double tau4 = tau3 * tau;
            const double tau5 = tau4 * tau;
            const double s = (10.0 * tau3) - (15.0 * tau4) + (6.0 * tau5);

            // Local (start-relative) position: linear interpolation scaled by s.
            const Eigen::Vector3d p_rel = rel_.r3();
            const Eigen::Vector3d local_pos = s * p_rel;

            // Local (start-relative) orientation: SLERP via right-trivialized log.
            // log(R_rel) is the axis-angle vector; scaling by s gives SLERP on SO(3).
            const Eigen::Vector3d log_R_rel = rel_.so3().log();
            const manifold::SO3 local_rot = manifold::SO3::exp(s * log_R_rel);

            return {local_rot, local_pos};
        }

    private:
        manifold::SE3 start_;
        manifold::SE3 end_;
        manifold::SE3 rel_;  // start^{-1} * end (start-relative displacement)
    };

    // --- Compile-Time Concept Verification ---
    static_assert(TaskTrajectory<ObstacleLine>);

}  // namespace xarm_geo::trajectories
