#pragma once

#include <xarm_geo/core/data.h>

namespace xarm_geo {
    void forward_kinematics(const Model &model, State &state,
                            const Eigen::Ref<const Eigen::VectorXd> &q);

    void forward_kinematics(const Model &model, State &state,
                            const Eigen::Ref<const Eigen::VectorXd> &q,
                            const Eigen::Ref<const Eigen::VectorXd> &v);

    void compute_jacobians(const Model &model, State &state,
                           const Eigen::Ref<const Eigen::VectorXd> &q);
}  // namespace xarm_geo
