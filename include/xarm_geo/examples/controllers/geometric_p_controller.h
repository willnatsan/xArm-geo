#pragma once

#include <cassert>
#include <string_view>

#include <Eigen/Dense>

#include <xarm_geo/control/controller.h>
#include <xarm_geo/control/feedback.h>
#include <xarm_geo/core/manifold.h>
#include <xarm_geo/core/system.h>

namespace xarm_geo::controllers {

    // --- Geometric P Controller ---
    //
    // SE(3)-tracking kinematic P controller. Body-frame command twist:
    //   xi_c = Ad_{g_e} * xi_d - nabla Phi(g_e)    (use_feedforward = true)
    //   xi_c =                 - nabla Phi(g_e)    (use_feedforward = false)

    class GeometricPController final : public KinematicTaskControllerBase {
    public:
        static constexpr std::string_view kName = "GeometricPController";

        explicit GeometricPController(const Model &model) : KinematicTaskControllerBase(model) {
            assert(model.dof > 0 && "GeometricPController: model.dof must be > 0");
        }

        // --- Public Configuration ---
        SE3FeedbackGains gains;
        bool use_feedforward = true;
        GradientType gradient = GradientType::LieGroup;

    protected:
        auto compute_command_twist(const Model & /*model*/, Data & /*data*/, KinematicsCache &kin,
                                   const TaskControllerContext &ctx,
                                   manifold::SE3::Twist &cmd_twist) noexcept -> bool override {

            // Body-frame configuration error and gradient.
            const manifold::SE3 g_e = kin.ee_pose().inverse() * ctx.ref.pose;
            const manifold::SE3::Twist grad =
                (gradient == GradientType::LieAlgebra)
                    ? se3_lie_algebra_gradient(g_e, gains.kp_pos, gains.kp_rot)
                    : se3_lie_group_gradient(g_e, gains.kp_pos, gains.kp_rot);

            // Reference twist transported into the current body frame.
            const manifold::SE3::Twist ad_xi_d = g_e.Ad() * ctx.ref.twist;

            if (use_feedforward) {
                cmd_twist = ad_xi_d - grad;
            } else {
                cmd_twist = -grad;
            }
            return true;
        }
    };

    // --- Compile-Time Concept Verification ---
    static_assert(xarm_geo::KinematicTaskController<GeometricPController>);

}  // namespace xarm_geo::controllers
