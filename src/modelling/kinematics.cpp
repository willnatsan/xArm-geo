#include <algorithm>
#include <numbers>
#include <random>

#include <xarm_geo/modelling/kinematics.h>

namespace xarm_geo {

    void forward_kinematics(const Model &model, Data &data,
                            const Eigen::Ref<const Eigen::VectorXd> &q) {

        manifold::SE3 T = manifold::SE3::Identity();
        data.pose_tree[0] = T;
        data.pose_tree_local[0] = T;

        for (int i = 0; i < model.dof; ++i) {
            const manifold::SE3::Tangent twist_i = model.screw_axes_space[i] * q[i];
            T *= manifold::SE3::exp(twist_i);
            const manifold::SE3 T_global = T * model.home_pose_tree[i + 1];

            data.pose_tree[i + 1] = T_global;
            data.pose_tree_local[i + 1] = data.pose_tree[i].inverse() * T_global;
        }

        data.ee_pose = T * model.home_pose;

        data.pose_tree[model.dof + 1] = data.ee_pose;
        data.pose_tree_local[model.dof + 1] = data.pose_tree[model.dof].inverse() * data.ee_pose;
    };

    void forward_kinematics(const Model &model, Data &data,
                            const Eigen::Ref<const Eigen::VectorXd> &q,
                            const Eigen::Ref<const Eigen::VectorXd> &v) {

        compute_jacobians(model, data, q);

        data.space_twist = data.space_jacobian * v;
        data.body_twist = data.body_jacobian * v;
        data.frame_twist = data.frame_jacobian * v;
    };

    void compute_jacobians(const Model &model, Data &data,
                           const Eigen::Ref<const Eigen::VectorXd> &q) {

        manifold::SE3 T = manifold::SE3::Identity();
        data.pose_tree[0] = T;
        data.pose_tree_local[0] = T;

        for (int i = 0; i < model.dof; ++i) {
            data.space_jacobian.col(i) = T.Ad() * model.screw_axes_space[i];

            const manifold::SE3::Tangent twist_i = model.screw_axes_space[i] * q[i];
            T *= manifold::SE3::exp(twist_i);
            const manifold::SE3 T_global = T * model.home_pose_tree[i + 1];

            data.pose_tree[i + 1] = T_global;
            data.pose_tree_local[i + 1] = data.pose_tree[i].inverse() * T_global;
        }

        data.ee_pose = T * model.home_pose;

        data.pose_tree[model.dof + 1] = data.ee_pose;
        data.pose_tree_local[model.dof + 1] = data.pose_tree[model.dof].inverse() * data.ee_pose;

        data.body_jacobian = data.ee_pose.inverse().Ad() * data.space_jacobian;

        // Frame Jacobian uses Rotation-Only Adjoint (Hybrid Frame: World-Aligned, EE-Origin)
        manifold::SE3 T_rot_only(data.ee_pose.so3(), Eigen::Vector3d::Zero());
        data.frame_jacobian = T_rot_only.Ad() * data.body_jacobian;
    };

    void inverse_diff_kinematics(const Model &model, Data &data,
                                 const manifold::SE3::Twist &target_twist,
                                 const IKOptions &options) {

        const auto &J = data.body_jacobian;  // Alias for brevity

        // Compute DLS Matrix: A = J^T * J
        // .noalias() prevents hidden Heap Allocation
        data.ik.A.noalias() = J.transpose() * J;

        // Apply Damping Factor: A = A + lambda^2 * I (lambda = Damping Factor)
        data.ik.A.diagonal().array() += (options.damping * options.damping);

        // Compute the Target Vector: b = J^T * V
        data.ik.b.noalias() = J.transpose() * target_twist;

        // Solve for Joint Velocities w/ Cholesky Decomposition (LDLT)
        data.v_out = data.ik.A.ldlt().solve(data.ik.b);

        // Joint Velocity Scaling (Ensure Limits are not Breached)
        if (options.apply_scaling) {
            double max_scale_factor = 1.0;
            for (int i = 0; i < model.dof; ++i) {
                double abs_vel = std::abs(data.v_out(i));
                double limit = model.limits[i].q_vel_max;

                if (limit > 0.0 && abs_vel > limit) {
                    double scale = abs_vel / limit;
                    max_scale_factor = std::max(scale, max_scale_factor);
                }
            }
            if (max_scale_factor > 1.0) { data.v_out /= max_scale_factor; }
        }
    };

    auto inverse_kinematics(const Model &model, Data &data,
                            const Eigen::Ref<const Eigen::VectorXd> &q,
                            const manifold::SE3 &target_pose, const IKOptions &options) -> bool {

        // Initialise the Current Guess from the Initial Starting Configuration
        data.q_guess = q;

        // Setup Random Number Generator for Subsequent Attempts
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(-std::numbers::pi, std::numbers::pi);

        for (int attempt = 0; attempt < options.max_restarts; ++attempt) {
            data.q_out = data.q_guess;

            for (int iter = 0; iter < options.max_iters; ++iter) {
                // Update Kinematic Tree (As `data.q_out` changes every iteration)
                compute_jacobians(model, data, data.q_out);

                // Compute Body-Frame Error Using Right-Minus: log(ee^{-1} * target)
                manifold::SE3::Twist V_err = target_pose - data.ee_pose;

                // If Converged -> Return
                if (V_err.norm() < options.tolerance) { return true; }

                // If Not Converged -> Compute Joint Step & Apply
                inverse_diff_kinematics(model, data, V_err, options);
                data.q_out += data.v_out;

                for (int i = 0; i < model.dof; ++i) {
                    // Wrap Joint Angle to [midpoint - pi, midpoint + pi]
                    double midpoint = 0.5 * (model.limits[i].q_min + model.limits[i].q_max);
                    data.q_out[i] = manifold::wrap_to_range(data.q_out[i], midpoint);

                    // Clamp Joint to Specified Joint Limits
                    data.q_out[i] =
                        std::clamp(data.q_out[i], model.limits[i].q_min, model.limits[i].q_max);
                }
            }

            // IK Loop Finished without returning `true` -> Go to Next Attempt w/ Random Seed
            for (int i = 0; i < model.dof; ++i) { data.q_guess[i] = dis(gen); }
        }

        // If Loop Terminates without returning `true`, IK Failed to Converge
        return false;
    };

    auto inverse_kinematics(const Model &model, Data &data, const CollisionModel &col_model,
                            CollisionData &col_data, const Eigen::Ref<const Eigen::VectorXd> &q,
                            const manifold::SE3 &target_pose, const IKOptions &options) -> bool {

        // Initialise the Current Guess from the Initial Starting Configuration
        data.q_guess = q;

        // Setup Random Number Generator for Subsequent Attempts
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(-std::numbers::pi, std::numbers::pi);

        for (int attempt = 0; attempt < options.max_restarts; ++attempt) {
            data.q_out = data.q_guess;

            for (int iter = 0; iter < options.max_iters; ++iter) {
                // Update Kinematic Tree (As `data.q_out` changes every iteration)
                compute_jacobians(model, data, data.q_out);

                // Compute Body-Frame Error Using Right-Minus: log(ee^{-1} * target)
                manifold::SE3::Twist V_err = target_pose - data.ee_pose;

                // If Converged -> Check for Collisions & Return if None
                if (V_err.norm() < options.tolerance) {
                    update_geometry_poses(model, data, col_model, col_data);
                    if (!compute_collisions(col_model, col_data)) { return true; }
                }

                // If Not Converged -> Compute Joint Step & Apply
                inverse_diff_kinematics(model, data, V_err, options);
                data.q_out += data.v_out;

                for (int i = 0; i < model.dof; ++i) {
                    // Wrap Joint Angle to [midpoint - pi, midpoint + pi]
                    double midpoint = 0.5 * (model.limits[i].q_min + model.limits[i].q_max);
                    data.q_out[i] = manifold::wrap_to_range(data.q_out[i], midpoint);

                    // Clamp Joint to Specified Joint Limits
                    data.q_out[i] =
                        std::clamp(data.q_out[i], model.limits[i].q_min, model.limits[i].q_max);
                }
            }

            // IK Loop Finished without returning `true` -> Go to Next Attempt w/ Random Seed
            for (int i = 0; i < model.dof; ++i) { data.q_guess[i] = dis(gen); }
        }

        // If Loop Terminates without returning `true`, IK Failed to Converge
        return false;
    }

}  // namespace xarm_geo
