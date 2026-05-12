#include <algorithm>
#include <cassert>
#include <random>

#include <xarm_geo/modelling/kinematics.h>

namespace xarm_geo {

    void forward_kinematics(const Model &model, Data &data) {

        assert(data.q.size() == model.dof && "data.q size must equal model.dof");

        const auto &q = data.q;

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
                            const Eigen::Ref<const Eigen::VectorXd> &v) {

        assert(v.size() == model.dof && "v size must equal model.dof");

        compute_jacobians(model, data);

        data.space_twist = data.space_jacobian * v;
        data.body_twist = data.body_jacobian * v;
        data.frame_twist = data.frame_jacobian * v;
    };

    void compute_jacobians(const Model &model, Data &data) {

        assert(data.q.size() == model.dof && "data.q size must equal model.dof");

        const auto &q = data.q;

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

        // Frame jacobian: hybrid frame (world-aligned, EE-origin) via rotation-only adjoint.
        manifold::SE3 T_rot_only(data.ee_pose.so3(), Eigen::Vector3d::Zero());
        data.frame_jacobian = T_rot_only.Ad() * data.body_jacobian;
    };

    void compute_point_jacobian(const Model &model, const Data &data, std::size_t parent_joint,
                                const Eigen::Ref<const Eigen::Vector3d> &p_world,
                                Eigen::Ref<Eigen::MatrixXd> J_out) {

        assert(parent_joint <= static_cast<std::size_t>(model.dof) &&
               "parent_joint must be in [0, model.dof]");
        assert(J_out.rows() == 3 && J_out.cols() == model.dof &&
               "J_out must be sized (3 x model.dof)");

        J_out.setZero();

        // parent_joint == 0 -> environment, no joint contributes.
        if (parent_joint == 0) { return; }

        const int n_active = static_cast<int>(parent_joint);
        const auto &J_space = data.space_jacobian;

        // `smooth` SE(3) tangent layout is [v_lin; omega]. Point velocity for a
        // world point p attached to link L is v_origin + omega x p.
        for (int i = 0; i < n_active; ++i) {
            const Eigen::Vector3d v_lin = J_space.col(i).head<3>();
            const Eigen::Vector3d omega = J_space.col(i).tail<3>();
            J_out.col(i) = v_lin + omega.cross(p_world);
        }
    };

    void inverse_diff_kinematics(const Model &model, Data &data,
                                 const manifold::SE3::Twist &target_twist,
                                 const IKOptions &options) {

        assert(data.body_jacobian.cols() == model.dof &&
               "body_jacobian not sized for model.dof; run compute_jacobians first");

        const auto &J = data.body_jacobian;

        // DLS: (J^T J + lambda^2 I) v = J^T V_target. A is strictly PD -> LLT.
        data.ik.A.noalias() = J.transpose() * J;
        data.ik.A.diagonal().array() += (options.damping * options.damping);
        data.ik.b.noalias() = J.transpose() * target_twist;
        data.v_out = data.ik.A.llt().solve(data.ik.b);
    };

    auto inverse_kinematics(const Model &model, Data &data,
                            const Eigen::Ref<const Eigen::VectorXd> &q_init,
                            const manifold::SE3 &target_pose, const IKOptions &options) -> bool {

        assert(q_init.size() == model.dof && "q_init size must equal model.dof");
        assert(options.dt > 0.0 && "IKOptions::dt must be positive");

        data.q_guess = q_init;

        for (int attempt = 0; attempt < options.max_restarts; ++attempt) {
            data.q_out = data.q_guess;

            for (int iter = 0; iter < options.max_iters; ++iter) {
                data.q = data.q_out;
                compute_jacobians(model, data);

                // Right-minus body-frame error: log(ee^{-1} * target).
                manifold::SE3::Twist V_err = target_pose - data.ee_pose;

                if (V_err.norm() < options.tolerance) { return true; }

                // data.v_out is rad/s; integrate over options.dt for the delta-q step.
                inverse_diff_kinematics(model, data, V_err, options);
                data.q_out += data.v_out * options.dt;

                // Wrap to [midpoint - pi, midpoint + pi], then clamp to joint limits.
                for (int i = 0; i < model.dof; ++i) {
                    double midpoint = 0.5 * (model.limits[i].q_min + model.limits[i].q_max);
                    data.q_out[i] = manifold::wrap_to_range(data.q_out[i], midpoint);
                    data.q_out[i] =
                        std::clamp(data.q_out[i], model.limits[i].q_min, model.limits[i].q_max);
                }
            }

            // Restart from a random per-joint seed.
            for (int i = 0; i < model.dof; ++i) {
                std::uniform_real_distribution<> dis(model.limits[i].q_min, model.limits[i].q_max);
                data.q_guess[i] = dis(data.rng);
            }
        }

        return false;
    };

}  // namespace xarm_geo
