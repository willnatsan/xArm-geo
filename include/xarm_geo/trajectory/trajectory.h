#pragma once

#include <concepts>

#include <Eigen/Dense>

#include <xarm_geo/core/manifold.h>

namespace xarm_geo {
    template <typename T>
    concept Trajectory = requires(const T &traj, double t) {
        { traj.evaluate(t) } -> std::convertible_to<manifold::SE3>;
    };
}  // namespace xarm_geo
