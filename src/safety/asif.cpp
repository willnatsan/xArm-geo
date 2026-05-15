#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>

#include <proxsuite/proxqp/dense/dense.hpp>

#include <xarm_geo/modelling/dynamics.h>
#include <xarm_geo/modelling/kinematics.h>
#include <xarm_geo/safety/asif.h>
#include <xarm_geo/utils/debug.h>

namespace {
    // Threshold above which the slack variable is considered active.
    constexpr double kSlackActiveThreshold = 1e-6;
}  // namespace

namespace xarm_geo {

    namespace {

        // Ensure the ProxQP solver is sized for (n, m_eq, m_in). On dimension
        // change, reconstruct and reset init; otherwise reuse to keep the
        // warm-start path active.
        void ensure_qp(Data::ASIFWorkspace &ws, int n, int m_eq, int m_in) {
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

    }  // namespace

    // --- Composable ASIF Filter ---

    auto asif_filter(const Model &model, Data &data, const CollisionModel *col_model,
                     CollisionData *col_data, const Eigen::Ref<const Eigen::VectorXd> &v,
                     const Eigen::Ref<const Eigen::VectorXd> &tau_des,
                     std::span<const DynamicBarrier *const> barriers,
                     Eigen::Ref<Eigen::VectorXd> tau_safe, const ASIFOptions &opts) -> ASIFStatus {

        assert(v.size() == model.dof && "v size must equal model.dof");
        assert(tau_des.size() == model.dof && "tau_des size must equal model.dof");
        assert(tau_safe.size() == model.dof && "tau_safe size must equal model.dof");
        assert(data.body_jacobian.cols() == model.dof &&
               "body_jacobian not sized for model.dof; run compute_jacobians first");
        assert(data.M.rows() == model.dof && data.M.cols() == model.dof &&
               "data.M not populated; run compute_mass_matrix first");
        assert(data.h.size() == model.dof && "data.h not populated; run compute_bias_forces first");

        const int dof = model.dof;
        auto &ws = data.asif;

        // M^-1 and M^-1 * h cached once; reused by all barriers and the task-cost branch.
        ws.M_llt.compute(data.M);
        if (ws.M_llt.info() != Eigen::Success) {
            debug::log("Cholesky failed on data.M (not PD)");
            return ASIFStatus::ERROR;
        }
        ws.M_inv = ws.M_llt.solve(Eigen::MatrixXd::Identity(dof, dof));
        ws.M_inv_h = ws.M_llt.solve(data.h);

        // Count CBF rows; decide whether to use the slack augmentation.
        int n_cbf = 0;
        for (const DynamicBarrier *bar : barriers) {
            if (bar != nullptr) { n_cbf += bar->rows(); }
        }
        const bool use_slack = std::isfinite(opts.relax_cost) && (n_cbf > 0);
        const int n_dec = dof + (use_slack ? 1 : 0);

        // --- Cost matrices ---
        // Task-consistent form when W_task is supplied (size 6):
        //     H = (J * M^-1)^T diag(W_task) (J * M^-1) + reg * I
        //     g = -H * tau_des
        // Otherwise joint-space form:
        //     H = diag(w_i) + reg * I,   g = -diag(w_i) * tau_des
        ws.H.setZero(n_dec, n_dec);
        ws.g.setZero(n_dec);

        const bool use_task_cost = (opts.W_task.size() == 6);
        if (use_task_cost) {
            // JM_inv = J * M^-1  (6 x dof)
            const Eigen::MatrixXd JM_inv = data.body_jacobian * ws.M_inv;
            // H_task = JM_inv^T * diag(W_task) * JM_inv  (dof x dof)
            ws.H.topLeftCorner(dof, dof).noalias() =
                JM_inv.transpose() * opts.W_task.asDiagonal() * JM_inv;
            ws.H.topLeftCorner(dof, dof).diagonal().array() += opts.regularisation;
            ws.g.head(dof).noalias() = -ws.H.topLeftCorner(dof, dof) * tau_des;
        } else {
            const bool use_weight = (opts.weight.size() == dof);
            for (int i = 0; i < dof; ++i) {
                const double w_i = use_weight ? opts.weight[i] : 1.0;
                ws.H(i, i) = w_i + opts.regularisation;
                ws.g[i] = -w_i * tau_des[i];
            }
        }

        // Slack cost: relax_cost * delta^2.
        if (use_slack) { ws.H(dof, dof) = opts.relax_cost; }

        // --- Inequality rows ---
        // CBF rows (one-sided):   A * tau - delta <= b   (slack col = -1)
        // Torque box:             tau_min <= tau <= tau_max  (two-sided, no slack)
        // Slack bound:            0 <= delta < inf
        const bool has_tau_max = (opts.tau_max.size() == dof);
        const bool has_tau_min = (opts.tau_min.size() == dof);
        const int n_box = (has_tau_max || has_tau_min) ? dof : 0;
        const int n_slack_box = use_slack ? 1 : 0;
        const int n_in = n_cbf + n_box + n_slack_box;

        if (n_in > 0) {
            if (ws.A.rows() != n_in || ws.A.cols() != n_dec) {
                ws.A.resize(n_in, n_dec);
                ws.l.resize(n_in);
                ws.u.resize(n_in);
            }
            ws.A.setZero();

            int row = 0;

            // CBF rows — torque block + slack column.
            for (const DynamicBarrier *bar : barriers) {
                if (bar == nullptr) { continue; }
                const int r = bar->rows();
                bar->compute_torque_constraint(model, data, col_model, col_data, v,
                                               ws.A.block(row, 0, r, dof), ws.u.segment(row, r));
                ws.l.segment(row, r).setConstant(-std::numeric_limits<double>::infinity());
                if (use_slack) { ws.A.block(row, dof, r, 1).setConstant(-1.0); }
                row += r;
            }

            // Torque box — torque block only; slack column stays zero.
            if (n_box > 0) {
                ws.A.block(row, 0, dof, dof).setIdentity();
                ws.l.segment(row, dof) =
                    has_tau_min
                        ? opts.tau_min
                        : Eigen::VectorXd::Constant(dof, -std::numeric_limits<double>::infinity());
                ws.u.segment(row, dof) =
                    has_tau_max
                        ? opts.tau_max
                        : Eigen::VectorXd::Constant(dof, std::numeric_limits<double>::infinity());
                row += dof;
            }

            // delta >= 0.
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
        // Guard: only warm-start after at least one successful solve (proxsuite 0.7.2).
        ws.qp->settings.initial_guess =
            (opts.warmstart && ws.solved_once)
                ? proxsuite::proxqp::InitialGuessStatus::WARM_START_WITH_PREVIOUS_RESULT
                : proxsuite::proxqp::InitialGuessStatus::NO_INITIAL_GUESS;

        ws.qp->solve();

        const auto qp_status = ws.qp->results.info.status;
        if (qp_status == proxsuite::proxqp::QPSolverOutput::PROXQP_SOLVED) {
            ws.solved_once = true;
            const Eigen::VectorXd &x = ws.qp->results.x;
            tau_safe = x.head(dof);
            const double delta = use_slack ? x[dof] : 0.0;
            return (delta > kSlackActiveThreshold) ? ASIFStatus::RELAXED : ASIFStatus::OK;
        }
        if (qp_status == proxsuite::proxqp::QPSolverOutput::PROXQP_PRIMAL_INFEASIBLE ||
            qp_status == proxsuite::proxqp::QPSolverOutput::PROXQP_DUAL_INFEASIBLE) {
            return ASIFStatus::INFEASIBLE;
        }
        if (qp_status == proxsuite::proxqp::QPSolverOutput::PROXQP_MAX_ITER_REACHED) {
            return ASIFStatus::MAX_ITERS;
        }
        return ASIFStatus::ERROR;
    }

    // --- Convenience Overload (Default Safety Set) ---

    auto asif_filter(const Model &model, Data &data, const CollisionModel &col_model,
                     CollisionData &col_data, const Eigen::Ref<const Eigen::VectorXd> &v,
                     const Eigen::Ref<const Eigen::VectorXd> &tau_des,
                     Eigen::Ref<Eigen::VectorXd> tau_safe, const ASIFOptions &opts) -> ASIFStatus {

        compute_mass_matrix(model, data);
        compute_bias_forces(model, data, v);

        DynPositionBarrier pbar(model);
        pbar.alpha_0 = asif_defaults::kBarrierAlpha0;
        pbar.alpha_1 = asif_defaults::kBarrierAlpha1;

        DynVelocityBarrier vbar(model);
        vbar.alpha_0 = asif_defaults::kBarrierAlpha0;

        DynCollisionBarrier cbar(model, col_model, asif_defaults::kCollisionActivationDistance);
        cbar.alpha_0 = asif_defaults::kBarrierAlpha0;
        cbar.alpha_1 = asif_defaults::kBarrierAlpha1;
        cbar.per_pair_activation_distance = opts.per_pair_activation_distance;

        // Collision pre-requisites for DynCollisionBarrier.  Use cbar's reported
        // max_activation_distance() as the AABB-cull threshold so pairs whose
        // per-pair activation exceeds the scalar default are not prematurely culled.
        update_geometry_poses(model, data, col_model, col_data);
        (void)compute_min_distance(col_model, col_data, cbar.max_activation_distance());

        const DynamicBarrier *barrier_ptrs[3] = {&pbar, &vbar, &cbar};
        std::span<const DynamicBarrier *const> barriers(barrier_ptrs);

        // Auto-populate symmetric torque box from model.limits[i].tau_max if
        // the caller left it empty. +inf entries are treated as unbounded.
        ASIFOptions opts_eff = opts;
        if (opts_eff.tau_max.size() != model.dof) {
            opts_eff.tau_max.resize(model.dof);
            for (int i = 0; i < model.dof; ++i) { opts_eff.tau_max[i] = model.limits[i].tau_max; }
        }
        if (opts_eff.tau_min.size() != model.dof) { opts_eff.tau_min = -opts_eff.tau_max; }

        return asif_filter(model, data, &col_model, &col_data, v, tau_des, barriers, tau_safe,
                           opts_eff);
    }

    // --- Post-Solve Validator ---

    auto asif_validate(const Model &model, const Data &live_data, Data &scratch_data,
                       const CollisionModel *col_model, CollisionData *scratch_col_data,
                       const Eigen::Ref<const Eigen::VectorXd> &v,
                       const Eigen::Ref<const Eigen::VectorXd> &tau_safe, double dt,
                       std::span<const DynamicBarrier *const> barriers, double tolerance) -> bool {

        assert(v.size() == model.dof && "v size must equal model.dof");
        assert(tau_safe.size() == model.dof && "tau_safe size must equal model.dof");
        assert(dt > 0.0 && "asif_validate::dt must be positive");

        // Single-step Euler integrator with tau held constant over dt; reuses
        // the LLT factorisation from the most recent asif_filter call.
        const Eigen::VectorXd a = live_data.asif.M_llt.solve(tau_safe - live_data.h);
        const Eigen::VectorXd v_next = v + a * dt;
        const Eigen::VectorXd q_next = live_data.q + v * dt + 0.5 * a * (dt * dt);

        scratch_data.q = q_next;
        forward_kinematics(model, scratch_data);

        if (col_model != nullptr && scratch_col_data != nullptr) {
            update_geometry_poses(model, scratch_data, *col_model, *scratch_col_data);
            (void)compute_min_distance(*col_model, *scratch_col_data,
                                       asif_defaults::kCollisionActivationDistance);
        }

        // Each barrier reads from scratch_data; live_data is never touched.
        for (const DynamicBarrier *bar : barriers) {
            if (bar == nullptr) { continue; }
            const double h_min =
                bar->evaluate_at(model, scratch_data, col_model, scratch_col_data, q_next, v_next);
            if (h_min < -tolerance) { return false; }
        }

        return true;
    }

}  // namespace xarm_geo
