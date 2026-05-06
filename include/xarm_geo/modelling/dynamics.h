#pragma once

#include <xarm_geo/core/system.h>

namespace xarm_geo {

    // --- Dynamics ---
    //
    // All dynamics functions read the joint configuration `q` from `data.q`.
    // Joint velocity `v`, joint acceleration `a`, and joint torque `tau` are
    // signals (inputs/outputs of the algorithm being parameterised) and remain
    // explicit parameters.
    //
    // Note: `forward_kinematics()` must be called beforehand to populate
    //       `pose_tree_local` (!)
    // Note: `ee_wrench` must be expressed in the End-Effector Frame (!)
    // Note: gravity defaults to zero (assumes external compensation, e.g.
    //       xArm SDK). Set `model.gravity = (0, 0, -9.81)` for torque-mode use.

    void forward_dynamics(const Model &model, Data &data,
                          const Eigen::Ref<const Eigen::VectorXd> &v,
                          const Eigen::Ref<const Eigen::VectorXd> &tau,
                          const manifold::SE3::Wrench &ee_wrench = manifold::SE3::Wrench());

    void inverse_dynamics(const Model &model, Data &data,
                          const Eigen::Ref<const Eigen::VectorXd> &v,
                          const Eigen::Ref<const Eigen::VectorXd> &a,
                          const manifold::SE3::Wrench &ee_wrench = manifold::SE3::Wrench());

    void compute_mass_matrix(const Model &model, Data &data);

    void compute_bias_forces(const Model &model, Data &data,
                             const Eigen::Ref<const Eigen::VectorXd> &v,
                             const manifold::SE3::Wrench &ee_wrench = manifold::SE3::Wrench());

}  // namespace xarm_geo
