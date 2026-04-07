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

        Eigen::Matrix<double, 6, 6> rot = Eigen::Matrix<double, 6, 6>::Zero();
        rot.topLeftCorner<3, 3>() = data.ee_pose.so3().matrix();
        rot.bottomRightCorner<3, 3>() = data.ee_pose.so3().matrix();
        data.frame_jacobian = rot * data.body_jacobian;
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

                // Compute Error in se(3)
                manifold::SE3 T_err = data.ee_pose.inverse() * target_pose;
                manifold::SE3::Twist V_err = T_err.log();

                // Check for Convergence
                if (V_err.norm() < options.tolerance) { return true; }

                // If not Converged -> Compute Joint Step & Apply
                inverse_diff_kinematics(model, data, V_err, options);
                data.q_out += data.v_out;

                // Wrap & Clamp Joints
                for (int i = 0; i < model.dof; ++i) {
                    // 1. Wrap joint angles to [-pi, pi]
                    while (data.q_out[i] > std::numbers::pi)
                        data.q_out[i] -= 2.0 * std::numbers::pi;
                    while (data.q_out[i] < -std::numbers::pi)
                        data.q_out[i] += 2.0 * std::numbers::pi;

                    // 2. Clamp joint to specified joint limits
                    data.q_out[i] = std::max(model.limits[i].q_min,
                                             std::min(data.q_out[i], model.limits[i].q_max));
                }
            }

            // IK Loop Finished without returning `true` -> Go to Next Attempt w/ Random Seed
            for (int i = 0; i < model.dof; ++i) { data.q_guess[i] = dis(gen); }
        }

        // If Loop Terminates without returning `true`, IK Failed to Converge
        return false;
    };
}  // namespace xarm_geo
