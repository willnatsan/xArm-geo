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
    // Note: gravity defaults to zero (assumes external compensation, e.g. xArm SDK)

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

    // Computes the Coriolis Bilinear Product: out = C(q, v_a) * v_b.
    //
    // C is the joint-space Coriolis matrix derived from the symmetric
    // Levi-Civita Christoffel symbols of the kinetic-energy metric M(q).
    // In this form the Coriolis bilinear is symmetric in its two velocity
    // arguments:
    //                     C(q, v_a) * v_b == C(q, v_b) * v_a.
    //
    // Useful for geometric / passivity-based controllers that need a
    // Coriolis feedforward evaluated at a velocity OTHER than the actual
    // joint velocity -- e.g. Bullo & Murray's geometric PD ( C(q, q_dot) *
    // r_dot ) or Slotine--Li's regressor ( C(q, q_dot) * r_dot_filtered ).
    //
    // Note: this function calls compute_gravity_forces internally and so
    // overwrites data.g, data.tau_out, and data.rnea workspaces.
    void compute_coriolis_times(const Model &model, Data &data,
                                const Eigen::Ref<const Eigen::VectorXd> &v_a,
                                const Eigen::Ref<const Eigen::VectorXd> &v_b,
                                Eigen::Ref<Eigen::VectorXd> out);

}  // namespace xarm_geo
