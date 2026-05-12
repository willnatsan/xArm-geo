#pragma once

#include <cassert>

#include <Eigen/Dense>

#include <xarm_geo/control/controller.h>
#include <xarm_geo/control/feedback.h>
#include <xarm_geo/core/manifold.h>
#include <xarm_geo/core/system.h>
#include <xarm_geo/modelling/collision.h>
#include <xarm_geo/modelling/kinematics.h>
#include <xarm_geo/modelling/optimal_kinematics.h>
#include <xarm_geo/safety/barriers.h>
#include <xarm_geo/safety/constraints.h>
#include <xarm_geo/safety/tasks.h>
#include <xarm_geo/utils/debug.h>

namespace xarm_geo::controllers {

    // --- Posture-Biased Geometric P Controller (Concept-Only) ---
    //
    // Demonstrates the concept-based escape valve from the controller
    // architecture: this class deliberately does NOT inherit from
    // KinematicTaskControllerBase, and instead satisfies the
    // KinematicTaskController concept by implementing update() directly.
    // Use this pattern when you need to customise something the bases do
    // not expose -- here, the Optimal IDK task set is augmented with a
    // PostureTask biasing the null space toward a preferred configuration
    // (joint midpoints by default).
    //
    // Control law mirrors GeometricPController:
    //   xi_c = Ad_{g_e} * xi_d - nabla Phi(g_e)    (use_feedforward = true)
    //   xi_c =                 - nabla Phi(g_e)    (use_feedforward = false)

    class PostureBiasedPController {
    public:
        PostureBiasedPController(const Model &model, const CollisionModel &col_model,
                                 CollisionData &col_data)
            : col_model_(&col_model), col_data_(&col_data) {

            assert(model.dof > 0 && "PostureBiasedPController: model.dof must be > 0");

            // Default posture reference: joint midpoints.
            posture_q_ref.resize(model.dof);
            for (int i = 0; i < model.dof; ++i) {
                const double mid = 0.5 * (model.limits[i].q_min + model.limits[i].q_max);
                assert(std::isfinite(mid) &&
                       "PostureBiasedPController: joint midpoint is non-finite; override "
                       "posture_q_ref after construction");
                posture_q_ref[i] = mid;
            }

            // Small uniform weight so the posture task never dominates EE tracking.
            posture_weight = Eigen::VectorXd::Constant(model.dof, 0.01);
        }

        // --- Public Configuration ---
        SE3FeedbackGains gains;
        bool use_feedforward = true;
        GradientType gradient = GradientType::LieGroup;
        Eigen::VectorXd posture_q_ref;
        Eigen::VectorXd posture_weight;
        double collision_activation_distance = 0.05;  // d_safe (m)
        double collision_barrier_alpha = 5.0;
        OptimalIKOptions optimal_ik_options;

        // --- Update Hook ---
        //
        // Satisfies the KinematicTaskController concept directly (no base class).
        auto update(const Model &model, Data &data, const TaskControllerContext &ctx,
                    JointVelocity &out) noexcept -> ControllerStatus {

            if (ctx.fb.q.size() != model.dof || out.v.size() != model.dof) {
                return ControllerStatus::SIZE_MISMATCH;
            }
            if (posture_q_ref.size() != model.dof || posture_weight.size() != model.dof) {
                debug::log("PostureBiasedPController: posture_q_ref / posture_weight "
                           "size mismatch with model.dof");
                return ControllerStatus::SIZE_MISMATCH;
            }

            data.q = ctx.fb.q;
            compute_jacobians(model, data);

            // Geometric P control law (body frame).
            const manifold::SE3 g_e = data.ee_pose.inverse() * ctx.ref.pose;
            const manifold::SE3::Twist grad =
                (gradient == GradientType::LieAlgebra)
                    ? se3_lie_algebra_gradient(g_e, gains.kp_pos, gains.kp_rot)
                    : se3_lie_group_gradient(g_e, gains.kp_pos, gains.kp_rot);
            const manifold::SE3::Twist ad_xi_d = g_e.Ad() * ctx.ref.twist;
            manifold::SE3::Twist cmd_twist;
            if (use_feedforward) {
                cmd_twist = ad_xi_d - grad;
            } else {
                cmd_twist = -grad;
            }

            // --- Augmented Optimal IDK Task Set ---
            const double step_dt = (optimal_ik_options.dt > 0.0) ? optimal_ik_options.dt : 1.0;

            TwistTask twist_task;
            twist_task.target_twist = cmd_twist;
            twist_task.dt = step_dt;

            PostureTask posture_task;
            posture_task.q_ref = posture_q_ref;
            posture_task.q_curr = ctx.fb.q;
            posture_task.weight = posture_weight;

            VelocityLimit vlim(model, step_dt);
            PositionLimit plim(model);

            CollisionBarrier cbar(model, *col_model_, collision_activation_distance);
            cbar.alpha = collision_barrier_alpha;
            cbar.dt = step_dt;

            const Task *task_ptrs[2] = {&twist_task, &posture_task};
            const Constraint *constraint_ptrs[2] = {&vlim, &plim};
            const KinematicBarrier *barrier_ptrs[1] = {&cbar};

            std::span<const Task *const> tasks(task_ptrs);
            std::span<const Constraint *const> constraints(constraint_ptrs);
            std::span<const KinematicBarrier *const> barriers(barrier_ptrs);

            const OptimalIKStatus status =
                optimal_inverse_diff_kinematics(model, data, col_model_, col_data_, tasks,
                                                constraints, barriers, optimal_ik_options);

            if (status == OptimalIKStatus::OK) { out.v = data.v_out; }
            return to_controller_status(status);
        }

    private:
        const CollisionModel *col_model_;
        CollisionData *col_data_;
    };

    // --- Compile-Time Concept Verification ---
    static_assert(xarm_geo::KinematicTaskController<PostureBiasedPController>);

}  // namespace xarm_geo::controllers
