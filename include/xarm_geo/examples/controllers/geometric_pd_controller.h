#pragma once

#include <cassert>
#include <string_view>

#include <Eigen/Dense>

#include <xarm_geo/control/controller.h>
#include <xarm_geo/control/feedback.h>
#include <xarm_geo/core/manifold.h>
#include <xarm_geo/core/system.h>

namespace xarm_geo::controllers {

    // --- Geometric PD Controller (Bullo-Murray / Maithripala / Seo) ---
    //
    // SE(3)-tracking dynamic PD on the Lie group, in the body frame of g.
    // Combines:
    //   - Maithripala (TAC 2006) body-frame configuration error g_e = g^{-1} g_d
    //     and velocity error xi_e = xi - Ad_{g_e} xi_d.
    //   - Bullo & Murray (Automatica 1999) PD-plus-feedforward structure:
    //         F_PD = nabla Phi(g_e) - K_d * xi_e           (wrench-direct)
    //         F_FF = Lambda(q) * d/dt(Ad_{g_e} xi_d)       (Lambda-scaled FF)
    //     The PD term is NOT premultiplied by Lambda; only the feedforward
    //     reference-acceleration term carries the operational-space inertia.
    //   - Seo et al. (IFAC 2023) left-invariant SE(3) gradient with matrix
    //     gains and proof of asymptotic stability for the manipulator case.
    //
    // Body-frame command wrench:
    //
    //     cmd_wrench = nabla Phi(g_e) - K_d * xi_e                    [F_PD]
    //                + Lambda(q) * d/dt(Ad_{g_e} xi_d)                [F_FF, if enabled]
    //
    // se3_*_gradient_wrench returns +nabla Phi as a body-frame Wrench; see
    // feedback.h for the sign convention and chain-rule derivation through g_e.

    class GeometricPDController final : public DynamicTaskControllerBase {
    public:
        static constexpr std::string_view kName = "GeometricPDController";

        // Recommended default; users may override after construction
        // (e.g. None when running against xArm SDK gravity compensation).
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
        double lambda_damping = 0.05;  // Damped least-squares regularisation for Lambda(q).

    protected:
        auto compute_command_wrench(const Model & /*model*/, Data & /*data*/, KinematicsCache &kin,
                                    DynamicsCache &dyn, const TaskControllerContext &ctx,
                                    manifold::SE3::Wrench &cmd_wrench) noexcept -> bool override {

            // Body-frame configuration error, transported reference twist, velocity error.
            const manifold::SE3::Twist body_twist = kin.body_jacobian() * ctx.fb.v;
            const manifold::SE3 g_e = kin.ee_pose().inverse() * ctx.ref.pose;
            const manifold::SE3::Wrench grad =
                (gradient == GradientType::LieAlgebra)
                    ? se3_lie_algebra_gradient_wrench(g_e, gains.kp_pos, gains.kp_rot)
                    : se3_lie_group_gradient_wrench(g_e, gains.kp_pos, gains.kp_rot);
            ad_xi_d_ = g_e.Ad() * ctx.ref.twist;
            xi_e_ = body_twist - ad_xi_d_;

            // F_PD = nabla Phi(g_e) - K_d * xi_e
            cmd_wrench.head<3>().noalias() =
                grad.head<3>() - gains.kd_lin.cwiseProduct(xi_e_.head<3>());
            cmd_wrench.tail<3>().noalias() =
                grad.tail<3>() - gains.kd_ang.cwiseProduct(xi_e_.tail<3>());

            // F_FF: Lambda(q) * d/dt(Ad_{g_e} xi_d)
            if (use_feedforward) {
                M_llt_.compute(dyn.M());
                if (compute_op_space_inertia(M_llt_, kin.body_jacobian(), lambda_, M_inv_Jt_,
                                             lambda_damping)) {
                    d_ad_xi_d_ = se3_transported_acc(g_e, xi_e_, ad_xi_d_, ctx.ref.spatial_acc);
                    cmd_wrench += manifold::SE3::Wrench(lambda_ * d_ad_xi_d_);
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
