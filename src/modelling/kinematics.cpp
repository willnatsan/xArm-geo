#include <xarm_geo/modelling/kinematics.h>

namespace xarm_geo {
    void forward_kinematics(const Model &model, State &state,
                            const Eigen::Ref<const Eigen::VectorXd> &q) {
        manifold::SE3 T = manifold::SE3::Identity();
        state.pose_tree[0] = T;
        state.pose_tree_local[0] = T;

        for (int i = 0; i < model.dof; ++i) {
            const manifold::SE3::Tangent twist_i = model.screw_axes_space[i] * q[i];
            T *= manifold::SE3::exp(twist_i);
            const manifold::SE3 T_global = T * model.home_pose_tree[i + 1];

            state.pose_tree[i + 1] = T_global;
            state.pose_tree_local[i + 1] = state.pose_tree[i].inverse() * T_global;
        }

        state.ee_pose = T * model.home_pose;

        state.pose_tree[model.dof + 1] = state.ee_pose;
        state.pose_tree_local[model.dof + 1] = state.pose_tree[model.dof].inverse() * state.ee_pose;
    };

    void forward_kinematics(const Model &model, State &state,
                            const Eigen::Ref<const Eigen::VectorXd> &q,
                            const Eigen::Ref<const Eigen::VectorXd> &v) {
        compute_jacobians(model, state, q);

        state.space_twist = state.space_jacobian * v;
        state.body_twist = state.body_jacobian * v;
        state.frame_twist = state.frame_jacobian * v;
    };

    void compute_jacobians(const Model &model, State &state,
                           const Eigen::Ref<const Eigen::VectorXd> &q) {
        manifold::SE3 T = manifold::SE3::Identity();
        state.pose_tree[0] = T;
        state.pose_tree_local[0] = T;

        for (int i = 0; i < model.dof; ++i) {
            state.space_jacobian.col(i) = T.Ad() * model.screw_axes_space[i];

            const manifold::SE3::Tangent twist_i = model.screw_axes_space[i] * q[i];
            T *= manifold::SE3::exp(twist_i);
            const manifold::SE3 T_global = T * model.home_pose_tree[i + 1];

            state.pose_tree[i + 1] = T_global;
            state.pose_tree_local[i + 1] = state.pose_tree[i].inverse() * T_global;
        }

        state.ee_pose = T * model.home_pose;

        state.pose_tree[model.dof + 1] = state.ee_pose;
        state.pose_tree_local[model.dof + 1] = state.pose_tree[model.dof].inverse() * state.ee_pose;

        state.body_jacobian = state.ee_pose.inverse().Ad() * state.space_jacobian;

        Eigen::Matrix<double, 6, 6> rot = Eigen::Matrix<double, 6, 6>::Zero();
        rot.topLeftCorner<3, 3>() = state.ee_pose.so3().matrix();
        rot.bottomRightCorner<3, 3>() = state.ee_pose.so3().matrix();
        state.frame_jacobian = rot * state.body_jacobian;
    };

    void inverse_diff_kinematics(const Model &model, State &state,
                                 const manifold::SE3::Twist &target_twist,
                                 const IKOptions &options) {

        const auto &J = state.body_jacobian;  // Alias for brevity

        // Compute DLS Matrix: A = J^T * J
        // .noalias() prevents hidden Heap Allocation
        state.A.noalias() = J.transpose() * J;

        // Apply Damping Factor: A = A + lambda^2 * I (lambda = Damping Factor)
        state.A.diagonal().array() += (options.damping * options.damping);

        // Compute the Target Vector: b = J^T * V
        state.b.noalias() = J.transpose() * target_twist;

        // Solve for Joint Velocities w/ Cholesky Decomposition (LDLT)
        state.v = state.A.ldlt().solve(state.b);
    };

    auto inverse_kinematics(const Model &model, State &state,
                            const Eigen::Ref<const Eigen::VectorXd> &q,
                            const manifold::SE3 &target_pose, const IKOptions &options) -> bool {

        // Initialise the Output `state.q` from the Initial Starting Configuration `q`
        state.q = q;

        for (int iter = 0; iter < options.max_iters; ++iter) {
            // Update Kinematic Tree (As `state.q` changes every iteration)
            compute_jacobians(model, state, state.q);

            // Compute Error in SE(3)
            manifold::SE3 T_err = state.ee_pose.inverse() * target_pose;

            // Map Error to Lie Algebra se(3)
            manifold::SE3::Twist V_err = T_err.log();

            // Check for Convergence
            if (V_err.norm() < options.tolerance) { return true; }

            // If not Converged -> Compute Joint Step & Apply
            inverse_diff_kinematics(model, state, V_err, options);
            state.q += state.v;
        }

        // If Loop Terminates without returning `true`, IK FAILED
        return false;
    };
}  // namespace xarm_geo
