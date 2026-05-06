#pragma once

#include <span>

#include <Eigen/Dense>

#include <xarm_geo/core/system.h>
#include <xarm_geo/modelling/collision.h>
#include <xarm_geo/safety/barriers.h>

namespace xarm_geo {

    // --- Active-Set Invariance Filtering (ASIF) for Joint Torques ---
    //
    // Takes a nominal torque tau_des and returns the closest tau_safe that,
    // when applied through the dynamics  M(q) v_dot + h(q, v) = tau,  keeps
    // the closed-loop trajectory inside the safe set defined by the supplied
    // DynamicBarriers (see safety/barriers.h):
    //
    //     min_tau     0.5 (tau - tau_des)^T diag(W) (tau - tau_des) + reg ||tau||^2
    //     s.t.        A_cbf tau <= b_cbf       (one block per DynamicBarrier)
    //                 tau_min <= tau <= tau_max
    //
    // Solved via ProxQP. M^-1 is factorised once per call and reused by all
    // barriers. The accompanying asif_validate() forward-simulates one step
    // and checks the actual barrier values at the predicted next-state to
    // catch QP linearisation errors.

    // --- ASIF Status & Options ---

    enum class ASIFStatus : std::uint8_t { OK, INFEASIBLE, MAX_ITERS, ERROR };

    struct ASIFOptions {
        double regularisation = 1e-12;  // Tikhonov regulariser added to H diagonal
        int max_iters_qp = 50;          // ProxQP inner iterations
        bool warmstart = true;          // Reuse previous QP solution

        // Hard torque box bounds. Empty -> no torque bounds enforced.
        // The convenience overload auto-populates these from
        // model.limits[i].tau_max (symmetric: tau_min = -tau_max) when left
        // empty by the caller.
        Eigen::VectorXd tau_min;
        Eigen::VectorXd tau_max;

        // Per-joint cost weights (diagonal). Empty -> unit weights.
        Eigen::VectorXd weight;
    };

    // --- Composable ASIF Filter ---
    //
    // Pre-conditions:
    //   - data.q is the current joint configuration.
    //   - compute_jacobians(model, data) has been called.
    //   - data.M and data.h are populated (compute_mass_matrix +
    //     compute_bias_forces); the filter does NOT recompute these.
    //
    // Writes the filtered torque to `tau_safe`. If any barrier requires
    // collision data, the caller must pass valid col_model/col_data AND
    // have called update_geometry_poses + compute_min_distance beforehand.

    auto asif_filter(const Model &model, Data &data, const CollisionModel *col_model,
                     CollisionData *col_data, const Eigen::Ref<const Eigen::VectorXd> &v,
                     const Eigen::Ref<const Eigen::VectorXd> &tau_des,
                     std::span<const DynamicBarrier *const> barriers,
                     Eigen::Ref<Eigen::VectorXd> tau_safe, const ASIFOptions &opts = ASIFOptions())
        -> ASIFStatus;

    // --- Convenience Overload (Default Safety Set) ---
    //
    // Default barriers: DynPositionBarrier + DynVelocityBarrier +
    // DynCollisionBarrier (alpha_0 = 25, alpha_1 = 10, activation_distance =
    // 5 cm). Internally runs compute_mass_matrix, compute_bias_forces,
    // update_geometry_poses, and compute_min_distance (the user is still
    // expected to have called compute_jacobians beforehand).

    auto asif_filter(const Model &model, Data &data, const CollisionModel &col_model,
                     CollisionData &col_data, const Eigen::Ref<const Eigen::VectorXd> &v,
                     const Eigen::Ref<const Eigen::VectorXd> &tau_des,
                     Eigen::Ref<Eigen::VectorXd> tau_safe, const ASIFOptions &opts = ASIFOptions())
        -> ASIFStatus;

    // --- Post-Solve Validator ---
    //
    // Forward-simulates one step using tau_safe and the full nonlinear
    // dynamics:
    //     a      = M^-1 (tau_safe - h_bias)
    //     v_next = v + a * dt
    //     q_next = q + v * dt + 0.5 * a * dt^2
    //
    // and evaluates each DynamicBarrier's `evaluate_at(q_next, v_next)`.
    // Returns false if any h_min < -tolerance (caller falls back to a
    // known-safe action, e.g. tau_safe = h_bias).
    //
    // `live_data` is read-only; predicted kinematic / collision state is
    // written into the user-supplied `scratch_data` and (optionally)
    // `scratch_col_data`, which must be pre-allocated alongside the live
    // instances. Scratch is left in an undefined state on exit. Reuses
    // live_data.asif.M_llt (factorised by the most recent asif_filter call).
    // `scratch_col_data` may be nullptr if no collision-using barrier is
    // present.

    auto asif_validate(const Model &model, const Data &live_data, Data &scratch_data,
                       const CollisionModel *col_model, CollisionData *scratch_col_data,
                       const Eigen::Ref<const Eigen::VectorXd> &v,
                       const Eigen::Ref<const Eigen::VectorXd> &tau_safe, double dt,
                       std::span<const DynamicBarrier *const> barriers, double tolerance = 0.0)
        -> bool;

}  // namespace xarm_geo
