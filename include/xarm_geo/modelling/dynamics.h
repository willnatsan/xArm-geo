#pragma once

#include <xarm_geo/core/system.h>

namespace xarm_geo {

    // --- Dynamics ---
    //
    // All functions read q from data.q; v, a, tau are explicit signal parameters.
    // `forward_kinematics()` must be run beforehand to populate pose_tree_local.
    // `ee_wrench` is expressed in the end-effector frame.
    // `model.gravity` defaults to zero (assumes external compensation).

    void forward_dynamics(const Model &model, Data &data,
                          const Eigen::Ref<const Eigen::VectorXd> &v,
                          const Eigen::Ref<const Eigen::VectorXd> &tau,
                          const manifold::SE3::Wrench &ee_wrench = manifold::SE3::Wrench());

    void inverse_dynamics(const Model &model, Data &data,
                          const Eigen::Ref<const Eigen::VectorXd> &v,
                          const Eigen::Ref<const Eigen::VectorXd> &a,
                          const manifold::SE3::Wrench &ee_wrench = manifold::SE3::Wrench());

    void compute_mass_matrix(const Model &model, Data &data);

    // Computes Full Bias Forces (Gravity + Coriolis) -> data.h
    void compute_bias_forces(const Model &model, Data &data,
                             const Eigen::Ref<const Eigen::VectorXd> &v,
                             const manifold::SE3::Wrench &ee_wrench = manifold::SE3::Wrench());

    // Computes Gravity Bias Forces -> data.g
    void compute_gravity_forces(const Model &model, Data &data);

    // Coriolis bilinear product: out = C(q, v_a) * v_b.
    //
    // Uses the symmetric Levi-Civita Christoffel form, so C(q, v_a) * v_b ==
    // C(q, v_b) * v_a. Useful for passivity-based controllers that evaluate
    // the Coriolis feedforward at a velocity other than the actual joint
    // velocity (e.g. Bullo & Murray geometric PD, Slotine-Li regressor).
    //
    // Calls compute_gravity_forces internally (unless g_fresh = true) and
    // overwrites data.g, data.tau_out, and data.rnea workspaces.
    void compute_coriolis_times(const Model &model, Data &data,
                                const Eigen::Ref<const Eigen::VectorXd> &v_a,
                                const Eigen::Ref<const Eigen::VectorXd> &v_b,
                                Eigen::Ref<Eigen::VectorXd> out, bool g_fresh = false);

}  // namespace xarm_geo
