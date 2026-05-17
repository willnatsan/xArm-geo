#pragma once

#include <span>

#include <xarm_geo/core/system.h>
#include <xarm_geo/modelling/collision.h>
#include <xarm_geo/modelling/kinematics.h>
#include <xarm_geo/safety/barriers.h>
#include <xarm_geo/safety/constraints.h>
#include <xarm_geo/safety/tasks.h>

namespace xarm_geo {

    // --- Optimal IK Defaults ---

    namespace optik_defaults {
        // Threshold on ||v_safe - v_des||_2 above which optik_modified is set.
        // Sized just above the QP solver's per-element numerical noise floor.
        inline constexpr double kModifiedTol = 1e-4;
    }  // namespace optik_defaults

    // --- Optimal IK Status & Options ---

    // OK       : strict solution, slack delta ≈ 0
    // RELAXED  : QP solved but slack delta > 1e-6; barrier(s) softened slightly
    // INFEASIBLE / MAX_ITERS / ERROR : true solver failure (rare once relaxation is on)
    enum class OptimalIKStatus : std::uint8_t { OK, RELAXED, INFEASIBLE, MAX_ITERS, ERROR };

    struct OptimalIKOptions {
        double regularisation = 1e-12;  // Tikhonov Regularisation on H
        double dt = 0.002;              // QP / runtime control timestep (s)
        double ik_step_dt = 0.1;        // Position-level IK iteration step (s)
        int max_iters_qp = 20;          // ProxQP Inner Iterations
        bool warmstart = true;          // Reuse Previous QP Solution
        double tolerance = 1e-4;        // Convergence Threshold (Position-Level IK)
        int max_iters = 50;             // Maximum Iterations (Position-Level IK)
        int max_restarts = 10;          // Maximum Restart Attempts (Position-Level IK)

        // The QP minimises relax_cost * delta^2 subject to delta >= 0, making all
        // barrier constraints softly relaxable.
        double relax_cost = 1e6;

        // Per-pair activation distance override for the collision barrier.
        Eigen::VectorXd per_pair_activation_distance;
    };

    // --- Composable API: One Velocity-Level QP Step ---
    //
    // Solves a single QP step:
    //
    //     min_{dq, delta}  sum_t  0.5 * ||J_t dq + alpha_t e_t||^2_W_t
    //                           + reg * ||dq||^2
    //                           + relax_cost * delta^2
    //     s.t.     l_c <= G_c dq <= u_c           (hard constraints)
    //              G_b dq - delta <= b_b           (CBF inequalities, softened)
    //              delta >= 0
    //
    // The slack variable delta makes the problem always feasible when
    // opts.relax_cost < inf.  Status is RELAXED when delta > 1e-6.
    //
    // Writes the resulting joint velocity into data.v_out (units: rad/s, the
    // QP's decision variable dq is divided by opts.dt before being written).
    //
    // The collision pointers may be nullptr if no KinematicBarrier needs them.
    //
    // Note: Torque/effort limits are NOT enforced here. The OptIK decision
    // variable is dq; torques are a dynamic quantity. Use the joint-torque
    // ASIF filter (safety/asif.h) for torque-bound enforcement.

    auto optimal_inverse_diff_kinematics(const Model &model, Data &data,
                                         const CollisionModel *col_model, CollisionData *col_data,
                                         std::span<const Task *const> tasks,
                                         std::span<const Constraint *const> constraints,
                                         std::span<const KinematicBarrier *const> barriers,
                                         const OptimalIKOptions &opts = OptimalIKOptions())
        -> OptimalIKStatus;

    // --- Convenience Overloads: Default Safety Set ---
    //
    // The "common case": automatically enforces joint position limits, joint
    // velocity limits, and self/environment collision avoidance derived from
    // the supplied CollisionModel.

    auto optimal_inverse_diff_kinematics(const Model &model, Data &data,
                                         const CollisionModel &col_model, CollisionData &col_data,
                                         const manifold::SE3::Twist &target_twist,
                                         const OptimalIKOptions &opts = OptimalIKOptions())
        -> OptimalIKStatus;

    auto optimal_inverse_kinematics(const Model &model, Data &data, const CollisionModel &col_model,
                                    CollisionData &col_data,
                                    const Eigen::Ref<const Eigen::VectorXd> &q_init,
                                    const manifold::SE3 &target_pose,
                                    const OptimalIKOptions &opts = OptimalIKOptions()) -> bool;

}  // namespace xarm_geo
