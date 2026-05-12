#pragma once

#include <cassert>
#include <cmath>
#include <limits>
#include <numbers>

#include <Eigen/Dense>

#include <xarm_geo/core/manifold.h>
#include <xarm_geo/trajectory/trajectory.h>

namespace xarm_geo::trajectories {

    // --- Large Orientation Step Trajectory ---
    //
    // Discrete SE(3) setpoint step: zero translational error, pure rotational
    // error of `angle` rad about `axis` relative to `anchor`. The default
    // (175 deg about (1, 1, 1) / sqrt(3)) places the commanded error near the
    // SO(3) trace-gradient saddle set at theta = pi, exercising almost-global
    // stability under large initial orientation error.

    class LargeOrientationStep final {
    public:
        explicit LargeOrientationStep(const manifold::SE3 &anchor,
                                      const Eigen::Vector3d &axis = Eigen::Vector3d::Ones() /
                                                                    std::sqrt(3.0),
                                      double angle = 175.0 * std::numbers::pi / 180.0)
            : anchor_(anchor), axis_unit_(axis.normalized()), angle_(angle),
              target_(anchor_ * manifold::SE3(manifold::SO3::exp(angle_ * axis_unit_),
                                              Eigen::Vector3d::Zero())) {
            assert(axis.norm() > 0.0 && "LargeOrientationStep: axis must be non-zero");
        }

        [[nodiscard]] auto evaluate(double t, TaskTarget &out) const noexcept -> TrajectoryStatus {
            if (t < 0.0) { return TrajectoryStatus::OUT_OF_DOMAIN; }

            out.pose = target_;
            out.twist = manifold::SE3::Twist::Zero();
            out.spatial_acc = manifold::SE3::SpatialAcceleration::Zero();

            return TrajectoryStatus::OK;
        }

        [[nodiscard]] static auto duration() noexcept -> double {
            return std::numeric_limits<double>::infinity();
        }

        [[nodiscard]] auto anchor() const noexcept -> const manifold::SE3 & { return anchor_; }
        [[nodiscard]] auto target() const noexcept -> const manifold::SE3 & { return target_; }
        [[nodiscard]] auto axis() const noexcept -> const Eigen::Vector3d & { return axis_unit_; }
        [[nodiscard]] auto angle() const noexcept -> double { return angle_; }

    private:
        manifold::SE3 anchor_;
        Eigen::Vector3d axis_unit_;
        double angle_;
        manifold::SE3 target_;
    };

    // --- Compile-Time Concept Verification ---
    static_assert(TaskTrajectory<LargeOrientationStep>);

}  // namespace xarm_geo::trajectories
