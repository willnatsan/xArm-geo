#pragma once

#include <span>

#include <xarm_geo/core/system.h>
#include <xarm_geo/modelling/collision.h>
#include <xarm_geo/modelling/kinematics.h>
#include <xarm_geo/safety/barriers.h>
#include <xarm_geo/safety/constraints.h>
#include <xarm_geo/safety/tasks.h>

namespace xarm_geo {

    // --- Optimal IK Status & Options ---

    enum class OptimalIKStatus : std::uint8_t { OK, INFEASIBLE, MAX_ITERS, ERROR };

    struct OptimalIKOptions {
        double regularisation = 1e-12;  // Tikhonov Regularisation on H
        double dt = 0.002;              // QP / runtime control timestep (s). Sets the physical
                                        // meaning of v_out (rad/s) and the velocity-limit window
                                        // for the differential-IK overloads. Should equal the
                                        // controller step period.
        double ik_step_dt = 0.1;        // Position-level IK iteration step (s). Used only by
                                        // optimal_inverse_kinematics to size the per-iteration
                                        // Newton step. Decoupled from `dt` so the offline IK can
                                        // take sensible large steps (≈ 18 deg/joint/iter at the
                                        // default) without being capped by the real-time velocity
                                        // limit that applies during closed-loop control.
        int max_iters_qp = 50;          // ProxQP Inner Iterations
        bool warmstart = true;          // Reuse Previous QP Solution
        double tolerance = 1e-4;        // Convergence Threshold (Position-Level IK)
        int max_iters = 50;             // Maximum Iterations (Position-Level IK)
        int max_restarts = 10;          // Maximum Restart Attempts (Position-Level IK)
    };

    // --- Composable API: One Velocity-Level QP Step ---
    //
    // Solves a single QP step:
    //
    //     min_dq   sum_t  0.5 * ||J_t dq + alpha_t e_t||^2_W_t  +  reg * ||dq||^2
    //     s.t.     l_c <= G_c dq <= u_c           (hard constraints)
    //              G_b dq <= b_b                  (CBF inequalities)
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
