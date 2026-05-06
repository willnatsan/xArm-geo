#include <cassert>

#include <xarm_geo/control/sample_controllers.h>
#include <xarm_geo/modelling/dynamics.h>
#include <xarm_geo/utils/debug.h>

namespace xarm_geo {

    // --- GeometricPController ---

    GeometricPController::GeometricPController(const Model &model) : dof_(model.dof) {
        assert(dof_ > 0 && "GeometricPController: model.dof must be > 0");
    }

    auto GeometricPController::compute_command_twist(const Model & /*model*/, Data &data,
                                                     const JointState & /*fb*/,
                                                     const TaskSpaceTarget &ref,
                                                     const ControllerContext & /*ctx*/,
                                                     manifold::SE3::Twist &cmd_twist) noexcept
        -> ControllerStatus {

        // Body-frame configuration error: g_e = g^{-1} * g_d.
        const manifold::SE3 g_e = data.ee_pose.inverse() * ref.pose;

        // Body-frame gradient (NF or log-map per `gradient`).
        const manifold::SE3::Twist grad =
            (gradient == GradientType::LieAlgebra)
                ? se3_lie_algebra_gradient(g_e, gains.kp_pos, gains.kp_rot)
                : se3_lie_group_gradient(g_e, gains.kp_pos, gains.kp_rot);

        // Transport the reference twist into the current body frame.
        const manifold::SE3::Twist ad_xi_d = g_e.Ad() * ref.twist;

        // Command twist:
        //   use_feedforward = true  -> xi_c = Ad * xi_d - grad
        //   use_feedforward = false -> xi_c =          - grad
        if (use_feedforward) {
            cmd_twist = ad_xi_d - grad;
        } else {
            cmd_twist = -grad;
        }

        return ControllerStatus::OK;
    }

    // --- GeometricPDController ---

    GeometricPDController::GeometricPDController(const Model &model)
        : dof_(model.dof), M_inv_Jt_(Eigen::MatrixXd::Zero(model.dof, 6)) {

        assert(dof_ > 0 && "GeometricPDController: model.dof must be > 0");
        lambda_.setZero();
    }

    auto GeometricPDController::compute_task_wrench(const Model &model, Data &data,
                                                    const JointState &fb,
                                                    const TaskSpaceTarget &ref,
                                                    const ControllerContext & /*ctx*/,
                                                    manifold::SE3::Twist &task_wrench) noexcept
        -> ControllerStatus {

        // Current body-frame end-effector twist.
        const manifold::SE3::Twist body_twist = data.body_jacobian * fb.v;

        // Body-frame configuration error and transported reference twist.
        const manifold::SE3 g_e = data.ee_pose.inverse() * ref.pose;
        const manifold::SE3::Twist grad =
            (gradient == GradientType::LieAlgebra)
                ? se3_lie_algebra_gradient(g_e, gains.kp_pos, gains.kp_rot)
                : se3_lie_group_gradient(g_e, gains.kp_pos, gains.kp_rot);
        ad_xi_d_ = g_e.Ad() * ref.twist;

        // Body-frame velocity error.
        xi_e_ = body_twist - ad_xi_d_;

        // P + D wrench (body frame). K_D applied per-axis on linear/angular.
        task_wrench.head<3>().noalias() =
            -grad.head<3>() - gains.kd_lin.cwiseProduct(xi_e_.head<3>());
        task_wrench.tail<3>().noalias() =
            -grad.tail<3>() - gains.kd_ang.cwiseProduct(xi_e_.tail<3>());

        // Optional inertial feedforward via operational-space inertia.
        if (use_feedforward) {
            compute_mass_matrix(model, data);
            M_llt_.compute(data.M);

            if (M_llt_.info() == Eigen::Success) {
                // M_inv_Jt = M^{-1} * J_b^T   ((dof x dof) x (dof x 6) -> (dof x 6))
                M_inv_Jt_.noalias() = M_llt_.solve(data.body_jacobian.transpose());

                // Lambda = (J_b * M_inv_Jt)^{-1}  (6 x 6)
                lambda_.noalias() = data.body_jacobian * M_inv_Jt_;
                lambda_ = lambda_.inverse().eval();

                // Closed-form d/dt(Ad * xi_d).
                d_ad_xi_d_ = se3_transported_acc(g_e, xi_e_, ad_xi_d_, ref.spatial_acc);

                // F_FF = Lambda * d/dt(Ad * xi_d) - ad_{xi_e}^* * Lambda * (Ad * xi_d)
                const manifold::SE3::Twist Lambda_ad_xi_d = lambda_ * ad_xi_d_;
                task_wrench.noalias() += lambda_ * d_ad_xi_d_;
                task_wrench.noalias() -= manifold::SE3::ad(xi_e_).transpose() * Lambda_ad_xi_d;
            } else {
                debug::log("Cholesky failed on M(q); FF dropped this tick");
            }
        }

        return ControllerStatus::OK;
    }

}  // namespace xarm_geo
