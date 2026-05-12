#pragma once

#include <cassert>

#include <Eigen/Dense>

#include <xarm_geo/control/controller.h>
#include <xarm_geo/control/feedback.h>
#include <xarm_geo/core/manifold.h>
#include <xarm_geo/core/system.h>
#include <xarm_geo/utils/debug.h>

namespace xarm_geo::controllers {

    // --- Geometric PD Controller (Maithripala) ---
    //
    // SE(3)-tracking dynamic PD. Body-frame command wrench:
    //   F_task = -nabla Phi(g_e) - K_D * xi_e
    //          + ( Lambda(q) * d/dt(Ad * xi_d)
    //              - ad_{xi_e}^* * Lambda(q) * Ad * xi_d )         [if use_feedforward]
    //
    // The dual-adjoint coupling is evaluated at the velocity error xi_e, so
    // the FF residual vanishes at perfect tracking and the closed loop is
    // passive. See Maithripala et al. (2006).

    class GeometricPDController final : public DynamicTaskControllerBase {
    public:
        // Recommended default; users may override after construction (e.g. None
        // when running against xArm SDK gravity compensation).
        static constexpr BiasCompensation kRecommendedBiasCompensation = BiasCompensation::Full;

        explicit GeometricPDController(const Model &model)
            : DynamicTaskControllerBase(model), M_inv_Jt_(Eigen::MatrixXd::Zero(model.dof, 6)) {

            assert(model.dof > 0 && "GeometricPDController: model.dof must be > 0");
            lambda_.setZero();
            bias_compensation = kRecommendedBiasCompensation;
        }

        // --- Public Configuration ---
        SE3FeedbackGains gains;
        bool use_feedforward = true;
        GradientType gradient = GradientType::LieGroup;

    protected:
        auto compute_command_wrench(const Model & /*model*/, Data & /*data*/, KinematicsCache &kin,
                                    DynamicsCache &dyn, const TaskControllerContext &ctx,
                                    manifold::SE3::Wrench &cmd_wrench) noexcept -> bool override {

            // Body-frame configuration error, transported reference twist, velocity error.
            const manifold::SE3::Twist body_twist = kin.body_jacobian() * ctx.fb.v;
            const manifold::SE3 g_e = kin.ee_pose().inverse() * ctx.ref.pose;
            const manifold::SE3::Twist grad =
                (gradient == GradientType::LieAlgebra)
                    ? se3_lie_algebra_gradient(g_e, gains.kp_pos, gains.kp_rot)
                    : se3_lie_group_gradient(g_e, gains.kp_pos, gains.kp_rot);
            ad_xi_d_ = g_e.Ad() * ctx.ref.twist;
            xi_e_ = body_twist - ad_xi_d_;

            // P + D wrench, per-axis K_D on linear/angular.
            cmd_wrench.head<3>().noalias() =
                -grad.head<3>() - gains.kd_lin.cwiseProduct(xi_e_.head<3>());
            cmd_wrench.tail<3>().noalias() =
                -grad.tail<3>() - gains.kd_ang.cwiseProduct(xi_e_.tail<3>());

            // F_FF = Lambda * d/dt(Ad * xi_d) - ad_{xi_e}^* * Lambda * (Ad * xi_d).
            if (use_feedforward) {
                M_llt_.compute(dyn.M());

                if (M_llt_.info() == Eigen::Success) {
                    M_inv_Jt_.noalias() = M_llt_.solve(kin.body_jacobian().transpose());
                    lambda_.noalias() = kin.body_jacobian() * M_inv_Jt_;
                    lambda_ = lambda_.inverse().eval();

                    d_ad_xi_d_ = se3_transported_acc(g_e, xi_e_, ad_xi_d_, ctx.ref.spatial_acc);

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
        // --- Pre-Allocated Per-Tick Scratch ---
        Eigen::LLT<Eigen::MatrixXd> M_llt_;
        Eigen::MatrixXd M_inv_Jt_;            // (dof x 6)
        Eigen::Matrix<double, 6, 6> lambda_;  // operational-space inertia
        manifold::SE3::Twist ad_xi_d_;        // Ad_{g_e} * xi_d
        manifold::SE3::Twist d_ad_xi_d_;      // d/dt(Ad_{g_e} * xi_d)
        manifold::SE3::Twist xi_e_;           // velocity error
    };

    // --- Compile-Time Concept Verification ---
    static_assert(xarm_geo::DynamicTaskController<GeometricPDController>);

}  // namespace xarm_geo::controllers
