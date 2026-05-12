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
    // SE(3)-tracking kinematic PI. Mixed-state integral (Goodarzi et al. 2013);
    // the integrator accumulates nabla Phi(g_e) through per-axis saturation
    // for anti-windup:
    //   xi_c     = Ad_{g_e} * xi_d - nabla Phi(g_e) - K_I * sat(e_I)
    //   dot(e_I) = nabla Phi(g_e)
    //
    // Hardcoded to the Lie-group gradient -- the log-map gradient's branch
    // cut near theta = pi makes it unsafe to integrate. Call reset() to zero
    // the integrator between distinct trajectories.

    class GeometricPIController final : public KinematicTaskControllerBase {
    public:
        static constexpr std::string_view kName = "GeometricPIController";

        explicit GeometricPIController(const Model &model) : KinematicTaskControllerBase(model) {
            assert(model.dof > 0 && "GeometricPIController: model.dof must be > 0");
        }

        void reset() noexcept { e_I_.setZero(); }

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

            // Mixed-state integrator: dot(e_I) = nabla Phi(g_e).
            const double dt = std::chrono::duration<double>(ctx.dt).count();
            e_I_.noalias() += grad * dt;

            // Per-axis anti-windup saturation (defaults +inf -> no clamping).
            manifold::SE3::Twist e_I_sat;
            e_I_sat.head<3>() = e_I_.head<3>().cwiseMax(-sigma_lin).cwiseMin(sigma_lin);
            e_I_sat.tail<3>() = e_I_.tail<3>().cwiseMax(-sigma_ang).cwiseMin(sigma_ang);

            manifold::SE3::Twist integral_term;
            integral_term.head<3>() = gains.ki_lin.cwiseProduct(e_I_sat.head<3>());
            integral_term.tail<3>() = gains.ki_ang.cwiseProduct(e_I_sat.tail<3>());

            const manifold::SE3::Twist ad_xi_d = g_e.Ad() * ctx.ref.twist;
            if (use_feedforward) {
                cmd_twist = ad_xi_d - grad - integral_term;
            } else {
                cmd_twist = -grad - integral_term;
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
