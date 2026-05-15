#include <algorithm>
#include <cassert>
#include <random>

#include <proxsuite/proxqp/dense/dense.hpp>

#include <xarm_geo/modelling/optimal_kinematics.h>
#include <xarm_geo/safety/barriers.h>
#include <xarm_geo/safety/constraints.h>
#include <xarm_geo/safety/tasks.h>

namespace {
    constexpr double CollisionActivationDistance = 0.05;  // metres
    constexpr double CollisionBarrierAlpha = 5.0;
    constexpr double CollisionBarrierMargin = 0.0;

    // Threshold above which the slack variable is considered active.
    constexpr double kSlackActiveThreshold = 1e-6;
}  // namespace

namespace xarm_geo {

    namespace {

        // --- Helpers ---

        // Ensure the ProxQP solver is sized for (n, m_eq, m_in). On dimension
        // change, reconstruct and reset init; otherwise reuse to keep the
        // warm-start path active.
        void ensure_qp(Data::OptIKWorkspace &ws, int n, int m_eq, int m_in) {
            if (ws.qp == nullptr || ws.current_n != n || ws.current_m_eq != m_eq ||
                ws.current_m_in != m_in) {
                ws.qp = std::make_unique<proxsuite::proxqp::dense::QP<double>>(n, m_eq, m_in);
                ws.current_n = n;
                ws.current_m_eq = m_eq;
                ws.current_m_in = m_in;
                ws.initialised = false;
                ws.solved_once = false;
            }
        }

        // Accumulate one task into (H, g):
        //   H += J^T * W^T W * J + lm_damping * I
        //   g -= J^T * W^T W * (alpha * e)
        void accumulate_task(const Task &task, const Model &model, Data &data,
                             Eigen::Ref<Eigen::MatrixXd> H, Eigen::Ref<Eigen::VectorXd> g,
                             Eigen::MatrixXd &J_scratch, Eigen::VectorXd &e_scratch) {

            const int rows = task.rows();
            const int dof = model.dof;

            if (J_scratch.rows() < rows || J_scratch.cols() != dof) { J_scratch.resize(rows, dof); }
            if (e_scratch.size() < rows) { e_scratch.resize(rows); }

            auto J = J_scratch.topRows(rows);
            auto e = e_scratch.head(rows);

            task.compute(model, data, J, e);

            if (task.weight.size() == rows) {
                const auto W = task.weight.asDiagonal();
                H.noalias() += J.transpose() * W * W * J;
                g.noalias() -= J.transpose() * W * W * (task.gain * e);
            } else {
                H.noalias() += J.transpose() * J;
                g.noalias() -= J.transpose() * (task.gain * e);
            }

            if (task.lm_damping > 0.0) { H.diagonal().array() += task.lm_damping; }
        }

    }  // namespace

    // --- Composable Velocity-Level Solver ---

    auto optimal_inverse_diff_kinematics(const Model &model, Data &data,
                                         const CollisionModel *col_model, CollisionData *col_data,
                                         std::span<const Task *const> tasks,
                                         std::span<const Constraint *const> constraints,
                                         std::span<const KinematicBarrier *const> barriers,
                                         const OptimalIKOptions &opts) -> OptimalIKStatus {

        assert(opts.dt > 0.0 && "OptimalIKOptions::dt must be positive");
        assert(data.q.size() == model.dof && "data.q size must equal model.dof");

        const int dof = model.dof;
        auto &ws = data.optik;

        // When relax_cost is finite we augment the decision variable with a
        // non-negative scalar slack delta that softens all barrier rows.
        // n_dec = dof + 1 (with slack) or dof (strict / no barriers).
        int n_barrier = 0;
        for (const KinematicBarrier *b : barriers) {
            if (b != nullptr) { n_barrier += b->rows(); }
        }
        const bool use_slack = std::isfinite(opts.relax_cost) && (n_barrier > 0);
        const int n_dec = dof + (use_slack ? 1 : 0);

        // --- Cost matrices ---
        ws.H.setZero(n_dec, n_dec);
        ws.g.setZero(n_dec);

        for (const Task *task : tasks) {
            if (task == nullptr) { continue; }
            accumulate_task(*task, model, data, ws.H.topLeftCorner(dof, dof), ws.g.head(dof),
                            ws.J_task, ws.e_task);
        }

        // Tikhonov regularisation on the joint-velocity block only.
        ws.H.topLeftCorner(dof, dof).diagonal().array() += opts.regularisation;

        // Slack cost: relax_cost * delta^2, occupying the bottom-right corner.
        if (use_slack) { ws.H(dof, dof) = opts.relax_cost; }

        // Refresh collision geometry for any barrier that needs it.
        if (col_model != nullptr && col_data != nullptr && !barriers.empty()) {
            update_geometry_poses(model, data, *col_model, *col_data);
            double cull_threshold = CollisionActivationDistance;
            for (const KinematicBarrier *b : barriers) {
                if (b != nullptr) {
                    cull_threshold = std::max(cull_threshold, b->max_activation_distance());
                }
            }
            (void)compute_min_distance(*col_model, *col_data, cull_threshold);
        }

        // --- Inequality rows ---
        // Hard constraints (two-sided):   l_c <= G_c * dq <= u_c
        // Barrier rows (one-sided):       G_b * dq - delta <= b_b  (slack column = -1)
        // Slack bound (box):              0 <= delta < inf
        int n_constraint = 0;
        for (const Constraint *c : constraints) {
            if (c != nullptr) { n_constraint += c->rows(); }
        }
        const int n_slack_box = use_slack ? 1 : 0;
        const int n_in = n_constraint + n_barrier + n_slack_box;

        if (n_in > 0) {
            if (ws.A.rows() != n_in || ws.A.cols() != n_dec) {
                ws.A.resize(n_in, n_dec);
                ws.l.resize(n_in);
                ws.u.resize(n_in);
            }
            ws.A.setZero();

            int row = 0;

            // Hard constraints — joint block only; slack column stays zero.
            for (const Constraint *c : constraints) {
                if (c == nullptr) { continue; }
                const int r = c->rows();
                c->compute(model, data, ws.A.block(row, 0, r, dof), ws.l.segment(row, r),
                           ws.u.segment(row, r));
                row += r;
            }

            // Barrier rows — G_b in joint block, -1 in slack column.
            for (const KinematicBarrier *b : barriers) {
                if (b == nullptr) { continue; }
                const int r = b->rows();
                b->compute(model, data, col_model, col_data, ws.A.block(row, 0, r, dof),
                           ws.u.segment(row, r));
                ws.l.segment(row, r).setConstant(-std::numeric_limits<double>::infinity());
                if (use_slack) { ws.A.block(row, dof, r, 1).setConstant(-1.0); }
                row += r;
            }

            // delta >= 0  expressed as  0 <= delta <= +inf.
            if (use_slack) {
                ws.A(row, dof) = 1.0;
                ws.l(row) = 0.0;
                ws.u(row) = std::numeric_limits<double>::infinity();
            }
        } else {
            ws.A.resize(0, n_dec);
            ws.l.resize(0);
            ws.u.resize(0);
        }

        ensure_qp(ws, n_dec, /*m_eq=*/0, /*m_in=*/n_in);

        Eigen::MatrixXd A_eq(0, n_dec);
        Eigen::VectorXd b_eq(0);

        if (n_in > 0) {
            if (!ws.initialised) {
                ws.qp->init(ws.H, ws.g, A_eq, b_eq, ws.A, ws.l, ws.u);
                ws.initialised = true;
            } else {
                ws.qp->update(ws.H, ws.g, A_eq, b_eq, ws.A, ws.l, ws.u);
            }
        } else {
            Eigen::MatrixXd A_in(0, n_dec);
            Eigen::VectorXd l_in(0), u_in(0);
            if (!ws.initialised) {
                ws.qp->init(ws.H, ws.g, A_eq, b_eq, A_in, l_in, u_in);
                ws.initialised = true;
            } else {
                ws.qp->update(ws.H, ws.g, A_eq, b_eq, A_in, l_in, u_in);
            }
        }

        ws.qp->settings.max_iter = opts.max_iters_qp;
        // Guard: only warm-start after at least one successful solve so the
        // LDLT factorisation is populated (proxsuite 0.7.2 assertion otherwise).
        ws.qp->settings.initial_guess =
            (opts.warmstart && ws.solved_once)
                ? proxsuite::proxqp::InitialGuessStatus::WARM_START_WITH_PREVIOUS_RESULT
                : proxsuite::proxqp::InitialGuessStatus::NO_INITIAL_GUESS;

        ws.qp->solve();

        const auto qp_status = ws.qp->results.info.status;
        if (qp_status == proxsuite::proxqp::QPSolverOutput::PROXQP_SOLVED) {
            ws.solved_once = true;
            // Extract joint-velocity block; convert dq -> rad/s.
            const Eigen::VectorXd &x = ws.qp->results.x;
            const double inv_dt = (opts.dt > 0.0) ? (1.0 / opts.dt) : 1.0;
            data.v_out = x.head(dof) * inv_dt;
            const double delta = use_slack ? x[dof] : 0.0;
            return (delta > kSlackActiveThreshold) ? OptimalIKStatus::RELAXED : OptimalIKStatus::OK;
        }
        if (qp_status == proxsuite::proxqp::QPSolverOutput::PROXQP_PRIMAL_INFEASIBLE ||
            qp_status == proxsuite::proxqp::QPSolverOutput::PROXQP_DUAL_INFEASIBLE) {
            return OptimalIKStatus::INFEASIBLE;
        }
        if (qp_status == proxsuite::proxqp::QPSolverOutput::PROXQP_MAX_ITER_REACHED) {
            return OptimalIKStatus::MAX_ITERS;
        }
        return OptimalIKStatus::ERROR;
    }

    // --- Convenience Overloads (Default Safety Set) ---
    //
    // Builds a single TwistTask or FrameTask plus VelocityLimit, PositionLimit
    // and CollisionBarrier, then forwards to the composable solver. The soft
    // PositionBarrier is intentionally omitted (redundant with PositionLimit);
    // users wanting graceful slow-down should call the composable API.

    auto optimal_inverse_diff_kinematics(const Model &model, Data &data,
                                         const CollisionModel &col_model, CollisionData &col_data,
                                         const manifold::SE3::Twist &target_twist,
                                         const OptimalIKOptions &opts) -> OptimalIKStatus {

        const double step_dt = (opts.dt > 0.0) ? opts.dt : 1.0;

        TwistTask task;
        task.target_twist = target_twist;
        task.dt = step_dt;

        VelocityLimit vlim(model, step_dt);
        PositionLimit plim(model);

        CollisionBarrier cbar(model, col_model, CollisionActivationDistance);
        cbar.alpha = CollisionBarrierAlpha;
        cbar.margin = CollisionBarrierMargin;
        cbar.dt = step_dt;
        // Forward per-pair override; ignored when empty (fallback to scalar default).
        cbar.per_pair_activation_distance = opts.per_pair_activation_distance;

        const Task *task_ptrs[1] = {&task};
        const Constraint *constraint_ptrs[2] = {&vlim, &plim};
        const KinematicBarrier *barrier_ptrs[1] = {&cbar};

        std::span<const Task *const> tasks(task_ptrs);
        std::span<const Constraint *const> constraints(constraint_ptrs);
        std::span<const KinematicBarrier *const> barriers(barrier_ptrs);

        return optimal_inverse_diff_kinematics(model, data, &col_model, &col_data, tasks,
                                               constraints, barriers, opts);
    }

    auto optimal_inverse_kinematics(const Model &model, Data &data, const CollisionModel &col_model,
                                    CollisionData &col_data,
                                    const Eigen::Ref<const Eigen::VectorXd> &q_init,
                                    const manifold::SE3 &target_pose, const OptimalIKOptions &opts)
        -> bool {

        assert(q_init.size() == model.dof && "q_init size must equal model.dof");
        assert(opts.dt > 0.0 && "OptimalIKOptions::dt must be positive");
        assert(opts.ik_step_dt > 0.0 && "OptimalIKOptions::ik_step_dt must be positive");

        // Use ik_step_dt for the position-level IK loop, not the runtime `dt`.
        // This decouples the offline IK's Newton-step size from the physical
        // controller period; each iteration can now cover a sensible joint-space
        // displacement (≈ ik_step_dt * q_vel_max per joint) rather than a single
        // real-time control tick (0.002 s × π ≈ 0.36 deg).
        const double step_dt = opts.ik_step_dt;

        // Build a local options copy with dt = step_dt so that the composable
        // solver emits v_out = x_qp / step_dt, and the outer integration
        //   q_out += v_out * step_dt
        // collapses back to exactly x_qp (the QP dq), preserving the
        // velocity-limit bound while using the larger IK step window.
        OptimalIKOptions ik_opts = opts;
        ik_opts.dt = step_dt;

        FrameTask task;
        task.target = target_pose;
        task.gain = 1.0;

        VelocityLimit vlim(model, step_dt);
        PositionLimit plim(model);

        CollisionBarrier cbar(model, col_model, CollisionActivationDistance);
        cbar.alpha = CollisionBarrierAlpha;
        cbar.margin = CollisionBarrierMargin;
        cbar.dt = step_dt;
        // Forward per-pair override; ignored when empty (fallback to scalar default).
        cbar.per_pair_activation_distance = opts.per_pair_activation_distance;

        const Task *task_ptrs[1] = {&task};
        const Constraint *constraint_ptrs[2] = {&vlim, &plim};
        const KinematicBarrier *barrier_ptrs[1] = {&cbar};

        std::span<const Task *const> tasks(task_ptrs);
        std::span<const Constraint *const> constraints(constraint_ptrs);
        std::span<const KinematicBarrier *const> barriers(barrier_ptrs);

        data.q_guess = q_init;

        for (int attempt = 0; attempt < opts.max_restarts; ++attempt) {
            data.q_out = data.q_guess;

            for (int iter = 0; iter < opts.max_iters; ++iter) {
                data.q = data.q_out;
                compute_jacobians(model, data);

                // Right-minus body-frame pose error.
                const manifold::SE3::Twist V_err = target_pose - data.ee_pose;
                if (V_err.norm() < opts.tolerance) {
                    // Sanity check: CollisionBarrier already enforced avoidance inside the QP.
                    update_geometry_poses(model, data, col_model, col_data);
                    if (!compute_collisions(col_model, col_data)) { return true; }
                    break;  // Collision at the converged pose -> next restart.
                }

                const auto status = optimal_inverse_diff_kinematics(
                    model, data, &col_model, &col_data, tasks, constraints, barriers, ik_opts);
                if (status != OptimalIKStatus::OK) { break; }

                // Composable solver returns rad/s; integrate over step_dt.
                data.q_out += data.v_out * step_dt;

                for (int i = 0; i < model.dof; ++i) {
                    const double midpoint = 0.5 * (model.limits[i].q_min + model.limits[i].q_max);
                    data.q_out[i] = manifold::wrap_to_range(data.q_out[i], midpoint);
                    data.q_out[i] =
                        std::clamp(data.q_out[i], model.limits[i].q_min, model.limits[i].q_max);
                }
            }

            // Random per-joint restart.
            for (int i = 0; i < model.dof; ++i) {
                std::uniform_real_distribution<> dis(model.limits[i].q_min, model.limits[i].q_max);
                data.q_guess[i] = dis(data.rng);
            }
        }

        return false;
    }

}  // namespace xarm_geo
