#pragma once

#include <cassert>
#include <chrono>
#include <limits>

#include <Eigen/Dense>

#include <xarm_geo/control/controller.h>
#include <xarm_geo/control/tracking.h>
#include <xarm_geo/core/manifold.h>
#include <xarm_geo/core/system.h>
#include <xarm_geo/modelling/dynamics.h>
#include <xarm_geo/utils/debug.h>

namespace xarm_geo::controllers {

    // --- Example: Geometric PID Controller (Dynamic, Task-Space) ---
    //
    // Reference implementation of an SE(3)-tracking dynamic PID controller
    // built on `DynamicTaskControllerBase`. Mixed-state integral (Goodarzi
    // et al. 2013): integrator accumulates xi_e + c2 * nabla Phi(g_e),
    // passed through per-axis saturation for anti-windup.
    //
    //     F_task   = - nabla Phi(g_e) - K_D * xi_e - K_I * sat(e_I) + F_FF
    //     dot(e_I) = xi_e + c2 * nabla Phi(g_e)
    //
    // F_FF is the operational-space feedforward block (see
    // GeometricPDController). Always uses the Lie-group gradient;
    // integrating the log-map gradient is unsafe near theta = pi due to
    // branch-cut accumulation. Caller must invoke reset() to zero the
    // integrator state between distinct trajectories.

    class GeometricPIDController final : public DynamicTaskControllerBase {
    public:
        explicit GeometricPIDController(const Model &model)
            : DynamicTaskControllerBase(model), M_inv_Jt_(Eigen::MatrixXd::Zero(model.dof, 6)) {

            assert(model.dof > 0 && "GeometricPIDController: model.dof must be > 0");
            lambda_.setZero();
        }

        void reset() noexcept { e_I_.setZero(); }

        // --- Public Configuration ---
        SE3TrackingGains gains;
        bool use_feedforward = true;
        double c2 = 0.0;  // mixed-state coupling (Goodarzi); set > 0 to enable mixing
        Eigen::Vector3d sigma_lin =
            Eigen::Vector3d::Constant(std::numeric_limits<double>::infinity());
        Eigen::Vector3d sigma_ang =
            Eigen::Vector3d::Constant(std::numeric_limits<double>::infinity());

    protected:
        auto compute_command_wrench(const Model &model, Data &data,
                                    const TaskControllerContext &ctx,
                                    manifold::SE3::Wrench &cmd_wrench) noexcept -> bool override {

            // Current body-frame end-effector twist.
            const manifold::SE3::Twist body_twist = data.body_jacobian * ctx.fb.v;

            // Body-frame configuration error and transported reference twist.
            // Hardcoded to Lie-group gradient (integrating log-map gradient is
            // unsafe near theta = pi due to branch-cut accumulation).
            const manifold::SE3 g_e = data.ee_pose.inverse() * ctx.ref.pose;
            const manifold::SE3::Twist grad =
                se3_lie_group_gradient(g_e, gains.kp_pos, gains.kp_rot);
            ad_xi_d_ = g_e.Ad() * ctx.ref.twist;

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
            cmd_wrench.head<3>().noalias() = -grad.head<3>() -
                                             gains.kd_lin.cwiseProduct(xi_e_.head<3>()) -
                                             gains.ki_lin.cwiseProduct(e_I_sat.head<3>());
            cmd_wrench.tail<3>().noalias() = -grad.tail<3>() -
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
                    d_ad_xi_d_ = se3_transported_acc(g_e, xi_e_, ad_xi_d_, ctx.ref.spatial_acc);

                    // F_FF = Lambda * d/dt(Ad * xi_d) - ad_{xi_e}^* * Lambda * (Ad * xi_d)
                    const manifold::SE3::Twist Lambda_ad_xi_d = lambda_ * ad_xi_d_;
                    cmd_wrench.noalias() += lambda_ * d_ad_xi_d_;
                    cmd_wrench.noalias() -= manifold::SE3::ad(xi_e_).transpose() * Lambda_ad_xi_d;
                } else {
                    debug::log("Cholesky failed on M(q); FF dropped this tick");
                }
            }

            return true;
        }

    private:
        manifold::SE3::Twist e_I_ = manifold::SE3::Twist::Zero();  // integrator state

        // --- Per-Tick Scratch (Pre-Allocated at Construction) ---
        // Sized from model.dof; zero allocation in compute_command_wrench().
        Eigen::LLT<Eigen::MatrixXd> M_llt_;
        Eigen::MatrixXd M_inv_Jt_;            // (dof x 6)
        Eigen::Matrix<double, 6, 6> lambda_;  // Operational-space inertia
        manifold::SE3::Twist ad_xi_d_;        // Ad_{g_e} * xi_d
        manifold::SE3::Twist d_ad_xi_d_;      // d/dt(Ad_{g_e} * xi_d)
        manifold::SE3::Twist xi_e_;           // velocity error
    };

    // --- Compile-Time Concept Verification ---
    static_assert(xarm_geo::DynamicTaskController<GeometricPIDController>);

}  // namespace xarm_geo::controllers
