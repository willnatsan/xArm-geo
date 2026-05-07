#pragma once

#include <xarm_geo/core/system.h>

namespace xarm_geo {

    struct IKOptions {
        double damping = 1e-2;     // Levenberg-Marquadt Damping Factor
        double tolerance = 1e-4;   // Convergence Threshold for IK
        double max_iters = 50;     // Maximum Iterations for IK
        double max_restarts = 10;  // Maximum Restart Attempts for IK
        double dt = 1.0;           // Step Size: q_out += v_out * dt (rad/s -> rad)
    };

    // --- Kinematics ---
    //
    // All kinematics functions read the joint configuration from `data.q`
    // (the canonical robot state). The user must set `data.q` to the desired
    // configuration before calling these.
    //
    // Joint velocity `v` is treated as a signal and remains an explicit
    // parameter where needed (the velocity-flavoured forward_kinematics).
    //
    // The position-level inverse_kinematics takes an explicit `q_init` seed,
    // since the IK initial guess is conceptually distinct from the current state.
    //
    // Constraints (joint velocity / position limits, collision avoidance) are
    // NOT enforced here. These are pure kinematics primitives. For
    // constraint-aware variants, see optimal_kinematics.h.

    void forward_kinematics(const Model &model, Data &data);

    void forward_kinematics(const Model &model, Data &data,
                            const Eigen::Ref<const Eigen::VectorXd> &v);

    void compute_jacobians(const Model &model, Data &data);

    // Compute the point-velocity Jacobian (3 x dof) of a world-space point
    // `p_world` rigidly attached to link `parent_joint`, using
    void compute_point_jacobian(const Model &model, const Data &data, std::size_t parent_joint,
                                const Eigen::Ref<const Eigen::Vector3d> &p_world,
                                Eigen::Ref<Eigen::MatrixXd> J_out);

    // Damped least-squares inverse differential kinematics (Levenberg–Marquardt Damping).
    // Solves the Tikhonov-regularised normal equations:
    //     (J^T J + lambda^2 I) v = J^T xi_target
    void inverse_diff_kinematics(const Model &model, Data &data,
                                 const manifold::SE3::Twist &target_twist,
                                 const IKOptions &options = IKOptions());

    auto inverse_kinematics(const Model &model, Data &data,
                            const Eigen::Ref<const Eigen::VectorXd> &q_init,
                            const manifold::SE3 &target_pose,
                            const IKOptions &options = IKOptions()) -> bool;

}  // namespace xarm_geo
