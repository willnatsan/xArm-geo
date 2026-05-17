#pragma once

#include <cassert>
#include <cmath>
#include <numbers>
#include <string_view>

#include <Eigen/Dense>

#include <xarm_geo/core/manifold.h>
#include <xarm_geo/trajectory/trajectory.h>

namespace xarm_geo::trajectories {

    // --- Large Orientation Step Trajectory ---
    //
    // Timed SE(3) step input: holds `anchor` for `step_time` seconds, then
    // jumps to a pose rotated `angle` rad about `axis` from the anchor (zero
    // translational offset).  Default: 175° about (1,1,1)/sqrt(3), placing the
    // commanded error near the SO(3) trace-gradient saddle set at theta = pi.
    //
    // `target.twist` and `target.spatial_acc` are zero throughout -- this is a
    // pure setpoint change, not a smooth motion.  Feedforward in geometric
    // controllers therefore contributes nothing across the discontinuity.
    //
    // `duration() = step_time + hold_duration` (finite; eligible for adapters
    // and the trajectory validator).
    //
    // For step-response experiments: drive the plant to `traj.anchor()` before
    // entering the control loop so that the full error is present from the
    // first controller tick.

    class LargeOrientationStep final {
    public:
        static constexpr std::string_view kName = "LargeOrientationStep";

        explicit LargeOrientationStep(const manifold::SE3 &anchor,
                                      const Eigen::Vector3d &axis = Eigen::Vector3d::Ones() /
                                                                    std::sqrt(3.0),
                                      double angle = 175.0 * std::numbers::pi / 180.0,
                                      double step_time = 0.0, double hold_duration = 8.0)
            : anchor_(anchor), axis_unit_(axis.normalized()), angle_(angle), step_time_(step_time),
              hold_duration_(hold_duration), total_duration_(step_time + hold_duration),
              target_(anchor_ * manifold::SE3(manifold::SO3::exp(angle_ * axis_unit_),
                                              Eigen::Vector3d::Zero())) {
            assert(axis.norm() > 0.0 && "LargeOrientationStep: axis must be non-zero");
            assert(step_time >= 0.0 && "LargeOrientationStep: step_time must be >= 0");
            assert(hold_duration > 0.0 && "LargeOrientationStep: hold_duration must be > 0");
        }

        [[nodiscard]] auto evaluate(double t, TaskTarget &out) const noexcept -> TrajectoryStatus {
            if (t < 0.0 || t > total_duration_) { return TrajectoryStatus::OUT_OF_DOMAIN; }

            out.pose = (t < step_time_) ? anchor_ : target_;
            out.twist = manifold::SE3::Twist::Zero();
            out.spatial_acc = manifold::SE3::SpatialAcceleration::Zero();

            return TrajectoryStatus::OK;
        }

        [[nodiscard]] auto duration() const noexcept -> double { return total_duration_; }

        [[nodiscard]] auto anchor() const noexcept -> const manifold::SE3 & { return anchor_; }
        [[nodiscard]] auto target() const noexcept -> const manifold::SE3 & { return target_; }
        [[nodiscard]] auto axis() const noexcept -> const Eigen::Vector3d & { return axis_unit_; }
        [[nodiscard]] auto angle() const noexcept -> double { return angle_; }
        [[nodiscard]] auto step_time() const noexcept -> double { return step_time_; }
        [[nodiscard]] auto hold_duration() const noexcept -> double { return hold_duration_; }

    private:
        manifold::SE3 anchor_;
        Eigen::Vector3d axis_unit_;
        double angle_;
        double step_time_;
        double hold_duration_;
        double total_duration_;
        manifold::SE3 target_;
    };

    // --- Compile-Time Concept Verification ---
    static_assert(TaskTrajectory<LargeOrientationStep>);

}  // namespace xarm_geo::trajectories
