#pragma once

#include <xarm_geo/core/system.h>

namespace xarm_geo {

    struct IKOptions {
        double damping = 1e-2;     // Levenberg-Marquardt damping factor
        double tolerance = 1e-4;   // convergence threshold
        double max_iters = 50;     // max iterations per restart
        double max_restarts = 10;  // max restart attempts
        double dt = 1.0;           // step size: q_out += v_out * dt (rad/s -> rad)
    };

    // --- Kinematics ---
    //
    // All functions read q from data.q (the canonical robot state). Joint
    // velocity `v` is treated as an explicit signal where needed.
    // inverse_kinematics takes an explicit q_init seed (distinct from data.q).
    //
    // These are pure primitives -- joint/velocity limits and collision
    // avoidance are NOT enforced. For constraint-aware variants, see
    // optimal_kinematics.h.

    void forward_kinematics(const Model &model, Data &data);

    void forward_kinematics(const Model &model, Data &data,
                            const Eigen::Ref<const Eigen::VectorXd> &v);

    void compute_jacobians(const Model &model, Data &data);

    // Point-velocity Jacobian (3 x dof) of a world-space point p_world rigidly
    // attached to link `parent_joint`.
    void compute_point_jacobian(const Model &model, const Data &data, std::size_t parent_joint,
                                const Eigen::Ref<const Eigen::Vector3d> &p_world,
                                Eigen::Ref<Eigen::MatrixXd> J_out);

    // Damped least-squares inverse differential kinematics (Levenberg-Marquardt).
    // Solves (J^T J + lambda^2 I) v = J^T xi_target.
    void inverse_diff_kinematics(const Model &model, Data &data,
                                 const manifold::SE3::Twist &target_twist,
                                 const IKOptions &options = IKOptions());

    auto inverse_kinematics(const Model &model, Data &data,
                            const Eigen::Ref<const Eigen::VectorXd> &q_init,
                            const manifold::SE3 &target_pose,
                            const IKOptions &options = IKOptions()) -> bool;

}  // namespace xarm_geo
