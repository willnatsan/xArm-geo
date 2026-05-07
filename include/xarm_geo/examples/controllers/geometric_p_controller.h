#pragma once

#include <cassert>

#include <Eigen/Dense>

#include <xarm_geo/control/controller.h>
#include <xarm_geo/control/tracking.h>
#include <xarm_geo/core/manifold.h>
#include <xarm_geo/core/system.h>

namespace xarm_geo {

    // --- Example: Geometric P Controller (Kinematic, Task-Space) ---
    //
    // Reference implementation of an SE(3)-tracking kinematic P controller
    // built on `KinematicTaskControllerBase`. Overrides compute_command_twist
    // and lets the base handle size checks, kinematics refresh, and
    // IDK / Optimal IDK routing.
    //
    // Body-frame command twist:
    //
    //     xi_c = Ad_{g_e} * xi_d - nabla Phi(g_e)         (use_feedforward = true)
    //     xi_c =                 - nabla Phi(g_e)         (use_feedforward = false)
    //
    // Users may copy this file and modify the control law to taste; the rest
    // of the pipeline is provided by the base.

    class GeometricPController final : public KinematicTaskControllerBase {
    public:
        explicit GeometricPController(const Model &model) : KinematicTaskControllerBase(model) {
            assert(model.dof > 0 && "GeometricPController: model.dof must be > 0");
        }

        // --- Public Configuration ---
        SE3TrackingGains gains;
        bool use_feedforward = true;
        GradientType gradient = GradientType::LieGroup;

    protected:
        auto compute_command_twist(const Model & /*model*/, Data &data,
                                   const TaskControllerContext &ctx,
                                   manifold::SE3::Twist &cmd_twist) noexcept -> bool override {

            // Body-frame configuration error: g_e = g^{-1} * g_d.
            const manifold::SE3 g_e = data.ee_pose.inverse() * ctx.ref.pose;

            // Body-frame gradient (NF or log-map per `gradient`).
            const manifold::SE3::Twist grad =
                (gradient == GradientType::LieAlgebra)
                    ? se3_lie_algebra_gradient(g_e, gains.kp_pos, gains.kp_rot)
                    : se3_lie_group_gradient(g_e, gains.kp_pos, gains.kp_rot);

            // Transport the reference twist into the current body frame.
            const manifold::SE3::Twist ad_xi_d = g_e.Ad() * ctx.ref.twist;

            // Command twist:
            //   use_feedforward = true  -> xi_c = Ad * xi_d - grad
            //   use_feedforward = false -> xi_c =          - grad
            if (use_feedforward) {
                cmd_twist = ad_xi_d - grad;
            } else {
                cmd_twist = -grad;
            }

            return true;
        }
    };

    // --- Compile-Time Concept Verification ---
    static_assert(KinematicTaskController<GeometricPController>);

}  // namespace xarm_geo
