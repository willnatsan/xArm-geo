#pragma once

#include <xarm_geo/core/data.h>

namespace xarm_geo {
    void forward_dynamics(const Model &model, State &state,
                          const Eigen::Ref<const Eigen::VectorXd> &q,
                          const Eigen::Ref<const Eigen::VectorXd> &v,
                          const Eigen::Ref<const Eigen::VectorXd> &a);

    // Note; `ee_wrench` must be expressed in End-Effector Frame (!)
    void inverse_dynamics(const Model &model, State &state,
                          const Eigen::Ref<const Eigen::VectorXd> &q,
                          const Eigen::Ref<const Eigen::VectorXd> &v,
                          const Eigen::Ref<const Eigen::VectorXd> &a,
                          const manifold::SE3::Wrench &ee_wrench = manifold::SE3::Wrench());
}  // namespace xarm_geo
