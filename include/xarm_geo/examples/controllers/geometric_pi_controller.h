#pragma once

#include <cassert>
#include <chrono>
#include <limits>

#include <Eigen/Dense>

#include <xarm_geo/control/controller.h>
#include <xarm_geo/control/feedback.h>
#include <xarm_geo/core/manifold.h>
#include <xarm_geo/core/system.h>

namespace xarm_geo::controllers {

    // --- Example: Geometric PI Controller (Kinematic, Task-Space) ---
    //
    // Reference implementation of an SE(3)-tracking kinematic PI controller
    // built on `KinematicTaskControllerBase`. Mixed-state integral
    // (Goodarzi et al. 2013): integrator accumulates nabla Phi(g_e), passed
    // through per-axis saturation for anti-windup.
    //
    //     xi_c     = Ad_{g_e} * xi_d - nabla Phi(g_e) - K_I * sat(e_I)
    //     dot(e_I) = nabla Phi(g_e)
    //
    // Always uses the Lie-group gradient; integrating the log-map gradient is
    // unsafe near theta = pi due to branch-cut accumulation. Caller must
    // invoke reset() to zero the integrator state between distinct
    // trajectories.

    class GeometricPIController final : public KinematicTaskControllerBase {
    public:
        explicit GeometricPIController(const Model &model) : KinematicTaskControllerBase(model) {
            assert(model.dof > 0 && "GeometricPIController: model.dof must be > 0");
        }

        void reset() noexcept { e_I_.setZero(); }

        // --- Public Configuration ---
        SE3TrackingGains gains;
        bool use_feedforward = true;  // Kinematic feedforward (Transported reference twist)
        Eigen::Vector3d sigma_lin =
            Eigen::Vector3d::Constant(std::numeric_limits<double>::infinity());
        Eigen::Vector3d sigma_ang =
            Eigen::Vector3d::Constant(std::numeric_limits<double>::infinity());

    protected:
        auto compute_command_twist(const Model & /*model*/, Data & /*data*/, KinematicsCache &kin,
                                   const TaskControllerContext &ctx,
                                   manifold::SE3::Twist &cmd_twist) noexcept -> bool override {

            // Body-frame configuration error: g_e = g^{-1} * g_d.
            const manifold::SE3 g_e = kin.ee_pose().inverse() * ctx.ref.pose;

            // Hardcoded to Lie-group gradient (integrating log-map gradient is
            // unsafe near theta = pi due to branch-cut accumulation).
            const manifold::SE3::Twist grad =
                se3_lie_group_gradient(g_e, gains.kp_pos, gains.kp_rot);

            // Integrator update: dot(e_I) = nabla Phi(g_e) (kinematic mixed-state form).
            const double dt = std::chrono::duration<double>(ctx.dt).count();
            e_I_.noalias() += grad * dt;

            // Per-axis saturation for anti-windup (defaults +inf -> no clamping).
            manifold::SE3::Twist e_I_sat;
            e_I_sat.head<3>() = e_I_.head<3>().cwiseMax(-sigma_lin).cwiseMin(sigma_lin);
            e_I_sat.tail<3>() = e_I_.tail<3>().cwiseMax(-sigma_ang).cwiseMin(sigma_ang);

            // Integral twist contribution (per-axis K_I).
            manifold::SE3::Twist integral_term;
            integral_term.head<3>() = gains.ki_lin.cwiseProduct(e_I_sat.head<3>());
            integral_term.tail<3>() = gains.ki_ang.cwiseProduct(e_I_sat.tail<3>());

            // Transport the reference twist into the current body frame.
            const manifold::SE3::Twist ad_xi_d = g_e.Ad() * ctx.ref.twist;

            // Command twist:
            //   xi_c = (use_ff ? Ad * xi_d : 0) - grad - K_I * sat(e_I)
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
