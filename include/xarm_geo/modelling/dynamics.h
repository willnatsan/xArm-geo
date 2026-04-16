#pragma once

#include <xarm_geo/core/system.h>

namespace xarm_geo {

    void forward_dynamics(const Model &model, Data &data,
                          const Eigen::Ref<const Eigen::VectorXd> &q,
                          const Eigen::Ref<const Eigen::VectorXd> &v,
                          const Eigen::Ref<const Eigen::VectorXd> &tau,
                          const manifold::SE3::Wrench &ee_wrench = manifold::SE3::Wrench());

    // Note: `forward_kinematics()` must be called beforehand to populate `pose_tree_local` (!)
    // Note: `ee_wrench` must be expressed in the End-Effector Frame (!)
    void inverse_dynamics(const Model &model, Data &data,
                          const Eigen::Ref<const Eigen::VectorXd> &q,
                          const Eigen::Ref<const Eigen::VectorXd> &v,
                          const Eigen::Ref<const Eigen::VectorXd> &a,
                          const manifold::SE3::Wrench &ee_wrench = manifold::SE3::Wrench());

    void compute_mass_matrix(const Model &model, Data &data,
                             const Eigen::Ref<const Eigen::VectorXd> &q);

    void compute_bias_forces(const Model &model, Data &data,
                             const Eigen::Ref<const Eigen::VectorXd> &q,
                             const Eigen::Ref<const Eigen::VectorXd> &v,
                             const manifold::SE3::Wrench &ee_wrench = manifold::SE3::Wrench());

}  // namespace xarm_geo
