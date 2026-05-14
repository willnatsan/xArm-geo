#pragma once

#include <cassert>
#include <string_view>

#include <Eigen/Dense>

#include <xarm_geo/control/controller.h>
#include <xarm_geo/control/feedback.h>
#include <xarm_geo/core/manifold.h>
#include <xarm_geo/core/system.h>

namespace xarm_geo::controllers {

    // --- Geometric PID Controller (Bullo-Murray / Maithripala / Seo + Goodarzi integrator) ---
    //
    // SE(3)-tracking dynamic PID, mirroring GeometricPDController's wrench-
    // direct PD + Lambda-scaled FF structure (Bullo-Murray 1999 / Maithripala
    // 2006 / Seo 2023) and adding a Goodarzi-style mixed-state integrator with
    // per-axis anti-windup saturation. Body-frame law:
    //
    //     cmd_wrench = nabla Phi(g_e) - K_d * xi_e - K_I * sat(e_I)         [F_PID]
    //                + Lambda(q) * d/dt(Ad_{g_e} xi_d)                       [F_FF, if enabled]
    //
    //     dot(e_I) = xi_e + c2 * nabla Phi(g_e)             (Goodarzi mixed state; c2=0 default)
    //
    // se3_lie_group_gradient_wrench returns +nabla Phi as a body-frame Wrench;
    // see feedback.h for the sign convention and chain-rule derivation through g_e.
    //
    // The integrator term is wrench-direct (same dimensional convention as
    // the P and D terms), so K_I carries physical units of N/s for the linear
    // sub-block and N*m/s for the angular sub-block.
    //
    // Hardcoded to the Lie-group gradient: the log-map branch cut at theta=pi
    // makes integration unsafe at antipodes. Call reset() to zero the
    // integrator between trajectories. See GeometricPDController for the
    // detailed rationale on the wrench-direct PD + Lambda-scaled FF split.

    class GeometricPIDController final : public DynamicTaskControllerBase {
    public:
        static constexpr std::string_view kName = "GeometricPIDController";

        // Recommended default; users may override after construction.
        static constexpr BiasCompensation kRecommendedBiasCompensation = BiasCompensation::Full;

        explicit GeometricPIDController(const Model &model)
            : DynamicTaskControllerBase(model), M_inv_Jt_(Eigen::MatrixXd::Zero(model.dof, 6)) {

            assert(model.dof > 0 && "GeometricPIDController: model.dof must be > 0");
            lambda_.setZero();
            bias_compensation = kRecommendedBiasCompensation;
        }

        void reset() noexcept { e_I_.setZero(); }

        // --- Public Configuration ---
        SE3FeedbackGains gains;
        bool use_feedforward = true;
        double c2 = 0.0;  // mixed-state coupling (Goodarzi); 0 disables mixing
        Eigen::Vector3d sigma_lin =
            Eigen::Vector3d::Constant(std::numeric_limits<double>::infinity());
        Eigen::Vector3d sigma_ang =
            Eigen::Vector3d::Constant(std::numeric_limits<double>::infinity());

        double lambda_damping = 0.05;  // Damped least-squares regularisation for Lambda(q)

    protected:
        auto compute_command_wrench(const Model & /*model*/, Data & /*data*/, KinematicsCache &kin,
                                    DynamicsCache &dyn, const TaskControllerContext &ctx,
                                    manifold::SE3::Wrench &cmd_wrench) noexcept -> bool override {

            // Body-frame configuration error, transported reference twist, velocity error.
            const manifold::SE3::Twist body_twist = kin.body_jacobian() * ctx.fb.v;
            const manifold::SE3 g_e = kin.ee_pose().inverse() * ctx.ref.pose;
            const manifold::SE3::Wrench grad =
                se3_lie_group_gradient_wrench(g_e, gains.kp_pos, gains.kp_rot);
            ad_xi_d_ = g_e.Ad() * ctx.ref.twist;
            xi_e_ = body_twist - ad_xi_d_;

            // Mixed-state integrator (Goodarzi):
            //     dot(e_I) = xi_e + c2 * nabla Phi(g_e)
            //              = xi_e + c2 * grad
            // c2 = 0 disables mixing; default.
            const double dt = std::chrono::duration<double>(ctx.dt).count();
            e_I_.noalias() += (xi_e_ + c2 * grad.coeffs) * dt;

            // Per-axis anti-windup saturation (defaults +inf -> no clamping).
            manifold::SE3::Twist e_I_sat;
            e_I_sat.head<3>() = e_I_.head<3>().cwiseMax(-sigma_lin).cwiseMin(sigma_lin);
            e_I_sat.tail<3>() = e_I_.tail<3>().cwiseMax(-sigma_ang).cwiseMin(sigma_ang);

            // F_PID = nabla Phi - K_d * xi_e - K_I * sat(e_I)
            cmd_wrench.head<3>().noalias() = grad.head<3>() -
                                             gains.kd_lin.cwiseProduct(xi_e_.head<3>()) -
                                             gains.ki_lin.cwiseProduct(e_I_sat.head<3>());
            cmd_wrench.tail<3>().noalias() = grad.tail<3>() -
                                             gains.kd_ang.cwiseProduct(xi_e_.tail<3>()) -
                                             gains.ki_ang.cwiseProduct(e_I_sat.tail<3>());

            // F_FF: Lambda(q) * d/dt(Ad_{g_e} xi_d)
            if (use_feedforward) {
                M_llt_.compute(dyn.M());
                if (compute_op_space_inertia(M_llt_, kin.body_jacobian(), lambda_, M_inv_Jt_,
                                             lambda_damping)) {
                    d_ad_xi_d_ = se3_transported_acc(g_e, xi_e_, ad_xi_d_, ctx.ref.spatial_acc);
                    cmd_wrench.noalias() += lambda_ * d_ad_xi_d_;
                }
            }

            return true;
        }

    private:
        manifold::SE3::Twist e_I_ = manifold::SE3::Twist::Zero();

        // --- Pre-Allocated Per-Tick Scratch ---
        Eigen::LLT<Eigen::MatrixXd> M_llt_;
        Eigen::MatrixXd M_inv_Jt_;            // (dof x 6)
        Eigen::Matrix<double, 6, 6> lambda_;  // operational-space inertia
        manifold::SE3::Twist ad_xi_d_;        // Ad_{g_e} * xi_d
        manifold::SE3::Twist d_ad_xi_d_;      // d/dt(Ad_{g_e} * xi_d)
        manifold::SE3::Twist xi_e_;           // velocity error
    };

    // --- Compile-Time Concept Verification ---
    static_assert(xarm_geo::DynamicTaskController<GeometricPIDController>);
    static_assert(xarm_geo::ResettableController<GeometricPIDController>);

}  // namespace xarm_geo::controllers
