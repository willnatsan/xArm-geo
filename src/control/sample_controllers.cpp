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

    // --- GeometricPIController ---

    GeometricPIController::GeometricPIController(const Model &model) : dof_(model.dof) {
        assert(dof_ > 0 && "GeometricPIController: model.dof must be > 0");
    }

    void GeometricPIController::reset() noexcept { e_I_.setZero(); }

    auto GeometricPIController::compute_command_twist(const Model & /*model*/, Data &data,
                                                      const JointState & /*fb*/,
                                                      const TaskSpaceTarget &ref,
                                                      const ControllerContext &ctx,
                                                      manifold::SE3::Twist &cmd_twist) noexcept
        -> ControllerStatus {

        // Body-frame configuration error: g_e = g^{-1} * g_d.
        const manifold::SE3 g_e = data.ee_pose.inverse() * ref.pose;

        // Hardcoded to Lie-group gradient (integrating log-map gradient is
        // unsafe near theta=pi due to branch-cut accumulation).
        const manifold::SE3::Twist grad = se3_lie_group_gradient(g_e, gains.kp_pos, gains.kp_rot);

        // Integrator update: dot(e_I) = nabla Phi(g_e) (kinematic mixed-state form).
        const double dt = std::chrono::duration<double>(ctx.dt).count();
        e_I_.noalias() += grad * dt;

        // Per-axis saturation for anti-windup (defaults +inf -> no clamping).
        manifold::SE3::Twist e_I_sat;
        e_I_sat.head<3>() = e_I_.head<3>().cwiseMax(-sigma_lin).cwiseMin(sigma_lin);
        e_I_sat.tail<3>() = e_I_.tail<3>().cwiseMax(-sigma_ang).cwiseMin(sigma_ang);

        // Integral wrench contribution (per-axis K_I).
        manifold::SE3::Twist integral_term;
        integral_term.head<3>() = gains.ki_lin.cwiseProduct(e_I_sat.head<3>());
        integral_term.tail<3>() = gains.ki_ang.cwiseProduct(e_I_sat.tail<3>());

        // Transport the reference twist into the current body frame.
        const manifold::SE3::Twist ad_xi_d = g_e.Ad() * ref.twist;

        // Command twist:
        //   xi_c = (use_ff ? Ad * xi_d : 0) - grad - K_I * sat(e_I)
        if (use_feedforward) {
            cmd_twist = ad_xi_d - grad - integral_term;
        } else {
            cmd_twist = -grad - integral_term;
        }

        return ControllerStatus::OK;
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

    // --- GeometricPIDController ---

    GeometricPIDController::GeometricPIDController(const Model &model)
        : dof_(model.dof), M_inv_Jt_(Eigen::MatrixXd::Zero(model.dof, 6)) {

        assert(dof_ > 0 && "GeometricPIDController: model.dof must be > 0");
        lambda_.setZero();
    }

    void GeometricPIDController::reset() noexcept { e_I_.setZero(); }

    auto GeometricPIDController::compute_task_wrench(const Model &model, Data &data,
                                                     const JointState &fb,
                                                     const TaskSpaceTarget &ref,
                                                     const ControllerContext &ctx,
                                                     manifold::SE3::Twist &task_wrench) noexcept
        -> ControllerStatus {

        // Current body-frame end-effector twist.
        const manifold::SE3::Twist body_twist = data.body_jacobian * fb.v;

        // Body-frame configuration error and transported reference twist.
        // Hardcoded to Lie-group gradient (integrating log-map gradient is
        // unsafe near theta=pi due to branch-cut accumulation).
        const manifold::SE3 g_e = data.ee_pose.inverse() * ref.pose;
        const manifold::SE3::Twist grad = se3_lie_group_gradient(g_e, gains.kp_pos, gains.kp_rot);
        ad_xi_d_ = g_e.Ad() * ref.twist;

        // Body-frame velocity error.
        xi_e_ = body_twist - ad_xi_d_;

        // Integrator update: dot(e_I) = xi_e + c2 * nabla Phi(g_e) (Goodarzi mixed-state).
        const double dt = std::chrono::duration<double>(ctx.dt).count();
        e_I_.noalias() += (xi_e_ + c2 * grad) * dt;

        // Per-axis saturation for anti-windup (defaults +inf -> no clamping).
        manifold::SE3::Twist e_I_sat;
        e_I_sat.head<3>() = e_I_.head<3>().cwiseMax(-sigma_lin).cwiseMin(sigma_lin);
        e_I_sat.tail<3>() = e_I_.tail<3>().cwiseMax(-sigma_ang).cwiseMin(sigma_ang);

        // P + D + I wrench (body frame). K_D and K_I applied per-axis on linear/angular.
        task_wrench.head<3>().noalias() = -grad.head<3>() -
                                          gains.kd_lin.cwiseProduct(xi_e_.head<3>()) -
                                          gains.ki_lin.cwiseProduct(e_I_sat.head<3>());
        task_wrench.tail<3>().noalias() = -grad.tail<3>() -
                                          gains.kd_ang.cwiseProduct(xi_e_.tail<3>()) -
                                          gains.ki_ang.cwiseProduct(e_I_sat.tail<3>());

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
