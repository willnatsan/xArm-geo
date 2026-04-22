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

    void forward_kinematics(const Model &model, Data &data,
                            const Eigen::Ref<const Eigen::VectorXd> &q);

    void forward_kinematics(const Model &model, Data &data,
                            const Eigen::Ref<const Eigen::VectorXd> &q,
                            const Eigen::Ref<const Eigen::VectorXd> &v);

    void compute_jacobians(const Model &model, Data &data,
                           const Eigen::Ref<const Eigen::VectorXd> &q);

    void inverse_diff_kinematics(const Model &model, Data &data,
                                 const manifold::SE3::Twist &target_twist,
                                 const IKOptions &options = IKOptions());

    auto inverse_kinematics(const Model &model, Data &data,
                            const Eigen::Ref<const Eigen::VectorXd> &q,
                            const manifold::SE3 &target_pose,
                            const IKOptions &options = IKOptions()) -> bool;

    // Note: Overloaded IK w/ Collision Detection
    auto inverse_kinematics(const Model &model, Data &data, const CollisionModel &col_model,
                            CollisionData &col_data, const Eigen::Ref<const Eigen::VectorXd> &q,
                            const manifold::SE3 &target_pose,
                            const IKOptions &options = IKOptions()) -> bool;

}  // namespace xarm_geo
