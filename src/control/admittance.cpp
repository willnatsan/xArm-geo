#include <algorithm>
#include <cassert>
#include <cmath>
#include <span>

#include <xarm_geo/control/admittance.h>
#include <xarm_geo/modelling/dynamics.h>
#include <xarm_geo/safety/asif.h>
#include <xarm_geo/safety/barriers.h>
#include <xarm_geo/safety/constraints.h>
#include <xarm_geo/safety/tasks.h>
#include <xarm_geo/utils/debug.h>

namespace xarm_geo {

    // --- AdmittanceLayer ---

    AdmittanceLayer::AdmittanceLayer(int dof, const AdmittanceOptions &opts) : dof_(dof) {

        assert(dof_ > 0 && "AdmittanceLayer: dof must be > 0");
        assert(opts.mass_diag.size() == dof_ && "mass_diag size must equal dof");
        assert(opts.damping_diag.size() == dof_ && "damping_diag size must equal dof");
        assert((opts.mass_diag.array() > 0.0).all() && "mass_diag must be strictly positive");
        assert((opts.damping_diag.array() > 0.0).all() && "damping_diag must be strictly positive");

        m_inv_diag_ = opts.mass_diag.cwiseInverse();
        d_diag_ = opts.damping_diag;
        v_state_ = Eigen::VectorXd::Zero(dof_);

        if (opts.stiffness_diag.size() > 0) {
            assert(opts.stiffness_diag.size() == dof_ && "stiffness_diag size must equal dof");
            assert(opts.q_anchor.size() == dof_ &&
                   "q_anchor size must equal dof when stiffness set");
            assert((opts.stiffness_diag.array() >= 0.0).all() &&
                   "stiffness_diag must be non-negative");
            k_diag_ = opts.stiffness_diag;
            q_anchor_ = opts.q_anchor;
        }

        diag_.v_state = Eigen::VectorXd::Zero(dof_);
        diag_.v_ff = Eigen::VectorXd::Zero(dof_);
        diag_.v_des = Eigen::VectorXd::Zero(dof_);
        diag_.v_safe = Eigen::VectorXd::Zero(dof_);
    }

    void AdmittanceLayer::reset() noexcept { v_state_.setZero(); }

    void AdmittanceLayer::seed_from(Eigen::Ref<const Eigen::VectorXd> v) noexcept {
        assert(v.size() == dof_ && "seed v size must equal dof");
        v_state_ = v;
    }

    void AdmittanceLayer::apply(const Model &model, Eigen::Ref<const Eigen::VectorXd> q,
                                const JointTorque &tau_in, Eigen::Ref<const Eigen::VectorXd> v_ff,
                                std::chrono::nanoseconds dt, JointVelocity &v_out) noexcept {

        assert(q.size() == dof_ && "q size must equal dof");
        assert(tau_in.tau.size() == dof_ && "tau size must equal dof");
        assert(v_ff.size() == dof_ && "v_ff size must equal dof");
        assert(v_out.v.size() == dof_ && "v_out size must equal dof");
        assert(dt.count() > 0 && "dt must be positive");

        const double dt_s = std::chrono::duration<double>(dt).count();

        if (!tau_in.tau.allFinite()) {
            debug::log("non-finite tau; holding v_state, emitting v_ff");
        } else {
            // M_v v_dot + D_v v + K_v (q - q_anchor) = tau
            // Forward Euler: v_state += dt * M_v^{-1} * (tau - D_v*v - K_v*(q-q_anchor))
            Eigen::VectorXd rhs = tau_in.tau - d_diag_.cwiseProduct(v_state_);
            if (k_diag_.size() == dof_) { rhs -= k_diag_.cwiseProduct(q - q_anchor_); }
            v_state_ += dt_s * m_inv_diag_.cwiseProduct(rhs);
        }

        // v_des = admittance state + feedforward bypass.
        const Eigen::VectorXd v_des = v_state_ + v_ff;

        // Direction-preserving rescale to |v_i| <= q_vel_max_i.
        double max_ratio = 0.0;
        for (int i = 0; i < dof_; ++i) {
            const double v_max = model.limits[i].q_vel_max;
            if (v_max > 0.0) { max_ratio = std::max(max_ratio, std::abs(v_des[i]) / v_max); }
        }
        const bool rescaled = max_ratio > 1.0;
        v_out.v = rescaled ? v_des * (1.0 / max_ratio) : v_des;

        diag_.v_state = v_state_;
        diag_.v_ff = v_ff;
        diag_.v_des = v_des;
        diag_.v_safe = v_out.v;
        diag_.max_ratio = max_ratio;
        diag_.rescaled = rescaled;
    }

    void AdmittanceLayer::apply(const Model &model, Eigen::Ref<const Eigen::VectorXd> q,
                                const JointTorque &tau_in, std::chrono::nanoseconds dt,
                                JointVelocity &v_out) noexcept {
        const Eigen::VectorXd v_ff = Eigen::VectorXd::Zero(dof_);
        apply(model, q, tau_in, v_ff, dt, v_out);
    }

    // --- Inertia Helpers ---

    auto make_inertia_diag(const Model &model, Data &data,
                           Eigen::Ref<const Eigen::VectorXd> q_anchor) -> Eigen::VectorXd {

        assert(q_anchor.size() == model.dof && "q_anchor size must equal model.dof");

        const Eigen::VectorXd saved_q = data.q;
        data.q = q_anchor;
        compute_mass_matrix(model, data);
        data.q = saved_q;

        Eigen::VectorXd diag = data.M.diagonal();
        if (model.joint_armature.size() == model.dof) { diag += model.joint_armature; }
        return diag;
    }

    auto make_inertia_weighted_damping(const Model &model, Data &data,
                                       Eigen::Ref<const Eigen::VectorXd> q_anchor,
                                       double cutoff_rad_s) -> Eigen::VectorXd {

        assert(cutoff_rad_s > 0.0 && "cutoff_rad_s must be positive");
        return make_inertia_diag(model, data, q_anchor) * cutoff_rad_s;
    }

    // --- Safe Velocity Projection ---

    auto safe_velocity_projection(const Model &model, Data &data, const CollisionModel &col_model,
                                  CollisionData &col_data, Eigen::Ref<const Eigen::VectorXd> v_in,
                                  Eigen::Ref<Eigen::VectorXd> v_safe, const OptimalIKOptions &opts)
        -> OptimalIKStatus {

        assert(v_in.size() == model.dof && "v_in size must equal model.dof");
        assert(v_safe.size() == model.dof && "v_safe size must equal model.dof");
        assert(opts.dt > 0.0 && "OptimalIKOptions::dt must be positive");
        assert(data.body_jacobian.cols() == model.dof &&
               "body_jacobian not sized; run compute_jacobians first");

        const double dt = opts.dt;

        // Posture task: minimise ||dq - v_in * dt||^2.
        // J = I,  e = q_ref - q_curr = v_in * dt.  With gain = 1.0:
        //   optimal dq = v_in * dt   ->   v_out = dq / dt = v_in.
        PostureTask task;
        task.q_ref = data.q + v_in * dt;
        task.q_curr = data.q;
        task.gain = 1.0;

        VelocityLimit vlim(model, dt);
        PositionLimit plim(model);

        // Soft position CBF: graceful slow-down near joint limits.
        PositionBarrier pbar(model);
        pbar.alpha = asif_defaults::kBarrierAlpha0;
        pbar.dt = dt;

        // Soft collision CBF: self / environment avoidance.
        CollisionBarrier cbar(model, col_model, asif_defaults::kCollisionActivationDistance);
        cbar.alpha = asif_defaults::kBarrierAlpha0;
        cbar.dt = dt;
        cbar.per_pair_activation_distance = opts.per_pair_activation_distance;

        const Task *task_ptrs[1] = {&task};
        const Constraint *constraint_ptrs[2] = {&vlim, &plim};
        const KinematicBarrier *barrier_ptrs[2] = {&pbar, &cbar};

        const OptimalIKStatus status = optimal_inverse_diff_kinematics(
            model, data, &col_model, &col_data, std::span<const Task *const>(task_ptrs),
            std::span<const Constraint *const>(constraint_ptrs),
            std::span<const KinematicBarrier *const>(barrier_ptrs), opts);

        if (status == OptimalIKStatus::OK || status == OptimalIKStatus::RELAXED) {
            v_safe = data.v_out;
        } else {
            // Fallback: pass through the admittance output unmodified so the
            // caller can decide whether to write it or drop the tick.
            debug::log("safe_velocity_projection failed; passing v_in through");
            v_safe = v_in;
        }

        return status;
    }

}  // namespace xarm_geo
