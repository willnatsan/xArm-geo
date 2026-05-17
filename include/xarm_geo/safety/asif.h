#pragma once

#include <span>

#include <Eigen/Dense>

#include <xarm_geo/core/system.h>
#include <xarm_geo/modelling/collision.h>
#include <xarm_geo/safety/barriers.h>

namespace xarm_geo {

    // --- Active-Set Invariance Filtering (ASIF) for Joint Torques ---
    //
    // Projects a nominal torque tau_des to the closest tau_safe that, applied
    // through M(q) v_dot + h(q, v) = tau, keeps the closed-loop trajectory in
    // the safe set defined by the supplied DynamicBarriers:
    //   min_tau   0.5 (tau - tau_des)^T diag(W) (tau - tau_des) + reg ||tau||^2
    //   s.t.      A_cbf tau <= b_cbf      (one block per DynamicBarrier)
    //             tau_min  <= tau <= tau_max
    //
    // Solved with ProxQP; M^-1 is factorised once and reused across barriers.
    // asif_validate() forward-simulates one step to catch QP linearisation errors.

    // --- ASIF Defaults ---
    //
    // Shared by asif_filter (convenience overload in asif.cpp) and the dynamic
    // controller bases (controller.cpp) so the defaults stay in one place.

    namespace asif_defaults {
        inline constexpr double kBarrierAlpha0 = 25.0;
        inline constexpr double kBarrierAlpha1 = 10.0;
        inline constexpr double kCollisionActivationDistance = 0.05;

        // Threshold on ||tau_safe - tau_des||_2 above which asif_modified is set.
        // Sized just above the QP solver's per-element numerical noise floor.
        inline constexpr double kModifiedTol = 1e-3;
    }  // namespace asif_defaults

    // --- ASIF Status & Options ---

    // OK       : strict solution, slack delta ≈ 0
    // RELAXED  : QP solved but slack delta > 1e-6; barrier(s) softened slightly
    // INFEASIBLE / MAX_ITERS / ERROR : true solver failure (rare once relaxation is on)
    enum class ASIFStatus : std::uint8_t { OK, RELAXED, INFEASIBLE, MAX_ITERS, ERROR };

    struct ASIFOptions {
        double regularisation = 1e-12;  // Tikhonov diagonal regulariser
        int max_iters_qp = 20;          // ProxQP inner iterations
        bool warmstart = true;          // reuse previous QP solution

        // Slack-variable relaxation cost for CBF barrier rows.
        double relax_cost = 1e6;

        // Task-consistent QP cost weights (6-vector: [w_pos x3, w_rot x3]).
        // When size() == 6, replaces the joint-space cost with an operational-space
        // form that penalises task-space acceleration deviation:
        //
        //     H = (J * M^-1)^T * diag(W_task) * (J * M^-1) + reg * I
        //     g = -H * tau_des
        //
        // DynamicTaskControllerBase sets this automatically (w_pos=1, w_rot=0.1).
        // DynamicJointControllerBase leaves it empty (joint-space cost preserved).
        Eigen::VectorXd W_task;

        // Hard torque box. Empty -> no torque bounds. The convenience overload
        // auto-populates these from model.limits[i].tau_max (symmetric).
        Eigen::VectorXd tau_min;
        Eigen::VectorXd tau_max;

        // Per-joint cost weights (diagonal). Empty -> unit weights.
        // Ignored when W_task is populated (task-consistent cost takes precedence).
        Eigen::VectorXd weight;

        // Per-pair activation distance override for the DynCollisionBarrier.
        // Used by the convenience overload of asif_filter that builds the
        // default DynCollisionBarrier internally.  When size() ==
        // col_model.collision_pairs.size(), entry k overrides the library
        // default for pair k.  The AABB-cull threshold passed to
        // compute_min_distance() is automatically set to the maximum value.
        // Empty by default (no override; all pairs use the 5 cm default).
        Eigen::VectorXd per_pair_activation_distance;
    };

    // --- Composable ASIF Filter ---
    //
    // Projects a nominal torque tau_des to the closest tau_safe that keeps the
    // closed-loop trajectory in the safe set defined by the supplied barriers:
    //
    //   min_{tau, delta}  0.5 (tau - tau_des)^T H (tau - tau_des) + relax_cost * delta^2
    //   s.t.              A_cbf tau - delta <= b_cbf   (one block per DynamicBarrier)
    //                     tau_min <= tau <= tau_max
    //                     delta >= 0
    //
    // H is either diag(W) + reg*I (joint-space, default) or the task-consistent
    // form (J*M^-1)^T diag(W_task) (J*M^-1) + reg*I when opts.W_task.size()==6.
    //
    // The slack variable delta makes the problem always feasible; status is
    // RELAXED when delta > 1e-6.  Solved with ProxQP.
    //
    // Pre-conditions: compute_jacobians, compute_mass_matrix, and
    // compute_bias_forces have already been called for the current data.q;
    // the filter does NOT recompute them. If any barrier needs collision
    // data, the caller must also have run update_geometry_poses and
    // compute_min_distance.

    auto asif_filter(const Model &model, Data &data, const CollisionModel *col_model,
                     CollisionData *col_data, const Eigen::Ref<const Eigen::VectorXd> &v,
                     const Eigen::Ref<const Eigen::VectorXd> &tau_des,
                     std::span<const DynamicBarrier *const> barriers,
                     Eigen::Ref<Eigen::VectorXd> tau_safe, const ASIFOptions &opts = ASIFOptions())
        -> ASIFStatus;

    // --- Convenience Overload (Default Safety Set) ---
    //
    // Installs DynPositionBarrier + DynVelocityBarrier + DynCollisionBarrier
    // (alpha_0 = 25, alpha_1 = 10, activation_distance = 5 cm). Internally
    // runs compute_mass_matrix, compute_bias_forces, update_geometry_poses,
    // and compute_min_distance; the caller must still have run
    // compute_jacobians beforehand.
    //
    // Passes opts.W_task and opts.relax_cost through to the composable filter.

    auto asif_filter(const Model &model, Data &data, const CollisionModel &col_model,
                     CollisionData &col_data, const Eigen::Ref<const Eigen::VectorXd> &v,
                     const Eigen::Ref<const Eigen::VectorXd> &tau_des,
                     Eigen::Ref<Eigen::VectorXd> tau_safe, const ASIFOptions &opts = ASIFOptions())
        -> ASIFStatus;

    // --- Post-Solve Validator ---
    //
    // Forward-simulates one step with the full nonlinear dynamics:
    //   a      = M^-1 (tau_safe - h_bias)
    //   v_next = v + a * dt
    //   q_next = q + v * dt + 0.5 * a * dt^2
    //
    // and evaluates each barrier's evaluate_at(q_next, v_next). Returns false
    // if any h_min < -tolerance (caller falls back to a known-safe action).
    //
    // `live_data` is read-only; predicted state is written into the
    // user-supplied scratch instances (left in an undefined state on exit).
    // Reuses live_data.asif.M_llt from the most recent asif_filter call.
    // `scratch_col_data` may be nullptr if no collision barrier is present.

    auto asif_validate(const Model &model, const Data &live_data, Data &scratch_data,
                       const CollisionModel *col_model, CollisionData *scratch_col_data,
                       const Eigen::Ref<const Eigen::VectorXd> &v,
                       const Eigen::Ref<const Eigen::VectorXd> &tau_safe, double dt,
                       std::span<const DynamicBarrier *const> barriers, double tolerance = 0.0)
        -> bool;

}  // namespace xarm_geo
