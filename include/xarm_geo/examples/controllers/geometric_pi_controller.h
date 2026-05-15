#pragma once

#include <cassert>
#include <chrono>
#include <limits>
#include <string_view>

#include <Eigen/Dense>

#include <xarm_geo/control/controller.h>
#include <xarm_geo/control/feedback.h>
#include <xarm_geo/core/manifold.h>
#include <xarm_geo/core/system.h>

namespace xarm_geo::controllers {

    // --- Geometric PI Controller ---
    //
    // SE(3)-tracking kinematic PI. Adds an intrinsic integral term (Bhat &
    // Bernstein 2015) to GeometricPController's P + feedforward law, with
    // per-axis back-calculation anti-windup. Body-frame command twist:
    //
    //   xi_c     = Ad_{g_e} * xi_d + nabla Phi(g_e) + K_I * sat(e_I)
    //   dot(e_I) = nabla Phi(g_e)        (Bhat 2015 eq. 5; intrinsic-gradient integrand)
    //   e_I      <- sat(e_I)             (back-calculation anti-windup)
    //
    // nabla Phi (se3_lie_group_gradient) is computed once and reused for both
    // the P term and the integrand. The integral is ADDED (same sign as the P
    // term): e_I accumulates +nabla Phi so K_I * sat(e_I) reinforces the
    // proportional correction (see feedback.h).
    //
    // Why Bhat (not Goodarzi 2013): Goodarzi's dot(e_I) = xi_e + c2*e_R couples
    // a velocity error to an attitude-error vector and carries a stability bound
    // that depends on K_p. Bhat 2015 proves that the intrinsic-gradient integrand
    // is covariant-derivative-correct on Lie groups and reduces to textbook PI in
    // the Euclidean limit.
    //
    // Back-calculation anti-windup: e_I_ is replaced by sat(e_I_) each tick.
    // sigma_lin / sigma_ang default to +inf (no clamping).
    //
    // Hardcoded to the Lie-group gradient (branch-cut-free). Call reset() to
    // zero the integrator between trajectories.

    class GeometricPIController final : public KinematicTaskControllerBase {
    public:
        static constexpr std::string_view kName = "GeometricPIController";

        explicit GeometricPIController(const Model &model) : KinematicTaskControllerBase(model) {
            assert(model.dof > 0 && "GeometricPIController: model.dof must be > 0");
        }

        void reset() noexcept { e_I_.setZero(); }
        [[nodiscard]] auto integrator_state() const noexcept -> const manifold::SE3::Twist & {
            return e_I_;
        }

        // --- Public Configuration ---
        SE3FeedbackGains gains;
        bool use_feedforward = true;
        Eigen::Vector3d sigma_lin =
            Eigen::Vector3d::Constant(std::numeric_limits<double>::infinity());
        Eigen::Vector3d sigma_ang =
            Eigen::Vector3d::Constant(std::numeric_limits<double>::infinity());

    protected:
        auto compute_command_twist(const Model & /*model*/, Data & /*data*/, KinematicsCache &kin,
                                   const TaskControllerContext &ctx,
                                   manifold::SE3::Twist &cmd_twist) noexcept -> bool override {

            const manifold::SE3 g_e = kin.ee_pose().inverse() * ctx.ref.pose;
            const manifold::SE3::Twist grad =
                se3_lie_group_gradient(g_e, gains.kp_pos, gains.kp_rot);

            // Bhat-intrinsic integrand: dot(e_I) = nabla Phi(g_e).
            // Reuses the same gradient as the P term; no velocity-error coupling.
            const double dt = std::chrono::duration<double>(ctx.dt).count();
            e_I_.noalias() += grad * dt;

            // Per-axis anti-windup saturation (defaults +inf -> no clamping).
            manifold::SE3::Twist e_I_sat;
            e_I_sat.head<3>() = e_I_.head<3>().cwiseMax(-sigma_lin).cwiseMin(sigma_lin);
            e_I_sat.tail<3>() = e_I_.tail<3>().cwiseMax(-sigma_ang).cwiseMin(sigma_ang);

            manifold::SE3::Twist integral_term;
            integral_term.head<3>() = gains.ki_lin.cwiseProduct(e_I_sat.head<3>());
            integral_term.tail<3>() = gains.ki_ang.cwiseProduct(e_I_sat.tail<3>());

            // Back-calculation anti-windup: clamp the raw integrator state to the
            // saturated value so e_I_ cannot grow unboundedly during saturation.
            e_I_ = e_I_sat;

            const manifold::SE3::Twist ad_xi_d = g_e.Ad() * ctx.ref.twist;
            if (use_feedforward) {
                cmd_twist = ad_xi_d + grad + integral_term;
            } else {
                cmd_twist = grad + integral_term;
            }

            return true;
        }

    private:
        manifold::SE3::Twist e_I_ = manifold::SE3::Twist::Zero();  // integrator state
    };

    // --- Compile-Time Concept Verification ---
    static_assert(xarm_geo::KinematicTaskController<GeometricPIController>);
    static_assert(xarm_geo::ResettableController<GeometricPIController>);

}  // namespace xarm_geo::controllers
