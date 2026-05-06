#pragma once

#include <xarm_geo/core/system.h>
#include <xarm_geo/modelling/collision.h>

namespace xarm_geo {

    struct IKOptions {
        double damping = 1e-2;       // Levenberg-Marquadt Damping Factor
        double tolerance = 1e-4;     // Convergence Threshold for IK
        double max_iters = 50;       // Maximum Iterations for IK
        double max_restarts = 10;    // Maximum Restart Attempts for IK
        bool apply_scaling = false;  // Apply Joint velocity Scaling
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
    // since the IK initial guess is conceptually distinct from the current
    // robot state -- a user often wants to seed IK from a "home" configuration
    // rather than from wherever the robot currently is. The result is written
    // to `data.q_out`; `data.q` is not modified.

    void forward_kinematics(const Model &model, Data &data);

    void forward_kinematics(const Model &model, Data &data,
                            const Eigen::Ref<const Eigen::VectorXd> &v);

    void compute_jacobians(const Model &model, Data &data);

    // Compute the point-velocity Jacobian (3 x dof) of a world-space point
    // `p_world` rigidly attached to link `parent_joint`, using
    // `data.space_jacobian` (which must be up-to-date for `data.q` -- caller
    // must run `compute_jacobians(model, data)` first).
    //
    // Convention: parent_joint == 0 corresponds to the static environment
    // (returns an all-zero Jacobian). For parent_joint k >= 1, only the first
    // k columns of J_out are non-zero (the joints downstream of the geometry
    // do not move it).
    void compute_point_jacobian(const Model &model, const Data &data,
                                std::size_t parent_joint,
                                const Eigen::Ref<const Eigen::Vector3d> &p_world,
                                Eigen::Ref<Eigen::MatrixXd> J_out);

    void inverse_diff_kinematics(const Model &model, Data &data,
                                 const manifold::SE3::Twist &target_twist,
                                 const IKOptions &options = IKOptions());

    auto inverse_kinematics(const Model &model, Data &data,
                            const Eigen::Ref<const Eigen::VectorXd> &q_init,
                            const manifold::SE3 &target_pose,
                            const IKOptions &options = IKOptions()) -> bool;

    // Note: Overloaded IK w/ Collision Detection
    auto inverse_kinematics(const Model &model, Data &data, const CollisionModel &col_model,
                            CollisionData &col_data,
                            const Eigen::Ref<const Eigen::VectorXd> &q_init,
                            const manifold::SE3 &target_pose,
                            const IKOptions &options = IKOptions()) -> bool;

}  // namespace xarm_geo
