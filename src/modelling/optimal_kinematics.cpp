#include <algorithm>
#include <cassert>
#include <random>

#include <proxsuite/proxqp/dense/dense.hpp>

#include <xarm_geo/modelling/optimal_kinematics.h>
#include <xarm_geo/safety/barriers.h>
#include <xarm_geo/safety/constraints.h>
#include <xarm_geo/safety/tasks.h>

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
        void accumulate_task(const Task &task, const Model &model, Data &data, Eigen::MatrixXd &H,
                             Eigen::VectorXd &g, Eigen::MatrixXd &J_scratch,
                             Eigen::VectorXd &e_scratch) {

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

        // Cost matrices.
        ws.H.setZero(dof, dof);
        ws.g.setZero(dof);

        for (const Task *task : tasks) {
            if (task == nullptr) { continue; }
            accumulate_task(*task, model, data, ws.H, ws.g, ws.J_task, ws.e_task);
        }

        // Tikhonov regularisation for strict PD.
        ws.H.diagonal().array() += opts.regularisation;

        // Refresh collision pre-requisites; barriers without collision needs ignore col_data.
        if (col_model != nullptr && col_data != nullptr && !barriers.empty()) {
            update_geometry_poses(model, data, *col_model, *col_data);
            (void)compute_min_distance(*col_model, *col_data);
        }

        // Inequality rows.
        int n_in = 0;
        for (const Constraint *c : constraints) {
            if (c != nullptr) { n_in += c->rows(); }
        }
        for (const KinematicBarrier *b : barriers) {
            if (b != nullptr) { n_in += b->rows(); }
        }

        // Assemble (A, l, u).
        if (n_in > 0) {
            if (ws.A.rows() != n_in || ws.A.cols() != dof) {
                ws.A.resize(n_in, dof);
                ws.l.resize(n_in);
                ws.u.resize(n_in);
            }
            ws.A.setZero();

            // Constraints (hard, two-sided): l_c <= G_c * dq <= u_c.
            int row = 0;
            for (const Constraint *c : constraints) {
                if (c == nullptr) { continue; }
                const int r = c->rows();
                c->compute(model, data, ws.A.middleRows(row, r), ws.l.segment(row, r),
                           ws.u.segment(row, r));
                row += r;
            }

            // Barriers (one-sided): G_b * dq <= b_b  (l = -inf, u = b_b).
            for (const KinematicBarrier *b : barriers) {
                if (b == nullptr) { continue; }
                const int r = b->rows();
                b->compute(model, data, col_model, col_data, ws.A.middleRows(row, r),
                           ws.u.segment(row, r));
                ws.l.segment(row, r).setConstant(-std::numeric_limits<double>::infinity());
                row += r;
            }
        } else {
            ws.A.resize(0, dof);
            ws.l.resize(0);
            ws.u.resize(0);
        }

        // First call -> init(); subsequent calls -> update() (reuses factorisation).
        ensure_qp(ws, dof, /*m_eq=*/0, /*m_in=*/n_in);

        // Empty equality-constraint placeholders.
        Eigen::MatrixXd A_eq(0, dof);
        Eigen::VectorXd b_eq(0);

        if (n_in > 0) {
            if (!ws.initialised) {
                ws.qp->init(ws.H, ws.g, A_eq, b_eq, ws.A, ws.l, ws.u);
                ws.initialised = true;
            } else {
                ws.qp->update(ws.H, ws.g, A_eq, b_eq, ws.A, ws.l, ws.u);
            }
        } else {
            Eigen::MatrixXd A_in(0, dof);
            Eigen::VectorXd l_in(0), u_in(0);
            if (!ws.initialised) {
                ws.qp->init(ws.H, ws.g, A_eq, b_eq, A_in, l_in, u_in);
                ws.initialised = true;
            } else {
                ws.qp->update(ws.H, ws.g, A_eq, b_eq, A_in, l_in, u_in);
            }
        }

        ws.qp->settings.max_iter = opts.max_iters_qp;
        // Use WARM_START_WITH_PREVIOUS_RESULT only after at least one successful
        // solve, so the LDLT factorisation is already populated. On the very
        // first solve of a newly-constructed QP the proxsuite 0.7.2 PrimalLDLT
        // backend skips setup_factorization (leaving ldl.dim()==0), which then
        // triggers an assertion inside active_set_change -> rank_r_update.
        ws.qp->settings.initial_guess =
            (opts.warmstart && ws.solved_once)
                ? proxsuite::proxqp::InitialGuessStatus::WARM_START_WITH_PREVIOUS_RESULT
                : proxsuite::proxqp::InitialGuessStatus::NO_INITIAL_GUESS;

        ws.qp->solve();

        const auto status = ws.qp->results.info.status;
        if (status == proxsuite::proxqp::QPSolverOutput::PROXQP_SOLVED) {
            ws.solved_once = true;
            // QP solution is dq over a step of opts.dt; convert back to rad/s.
            data.v_out = (opts.dt > 0.0) ? (ws.qp->results.x / opts.dt) : ws.qp->results.x;
            return OptimalIKStatus::OK;
        }
        if (status == proxsuite::proxqp::QPSolverOutput::PROXQP_PRIMAL_INFEASIBLE ||
            status == proxsuite::proxqp::QPSolverOutput::PROXQP_DUAL_INFEASIBLE) {
            return OptimalIKStatus::INFEASIBLE;
        }
        if (status == proxsuite::proxqp::QPSolverOutput::PROXQP_MAX_ITER_REACHED) {
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

    namespace {
        constexpr double CollisionActivationDistance = 0.05;  // metres
        constexpr double CollisionBarrierAlpha = 5.0;
        constexpr double CollisionBarrierMargin = 0.0;
    }  // namespace

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
