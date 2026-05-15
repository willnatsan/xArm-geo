#include <cassert>

#include <xarm_geo/control/controller.h>
#include <xarm_geo/utils/debug.h>

namespace xarm_geo {

    // --- Kinematic Task Controller Base Class ---

    KinematicTaskControllerBase::KinematicTaskControllerBase(const Model & /*model*/) noexcept {
        // No joint-sized scratch; ctor kept for symmetry with the other bases.
    }

    void KinematicTaskControllerBase::attach_collision(const CollisionModel &col_model,
                                                       CollisionData &col_data) noexcept {
        col_model_ = &col_model;
        col_data_ = &col_data;
    }

    void KinematicTaskControllerBase::detach_collision() noexcept {
        col_model_ = nullptr;
        col_data_ = nullptr;
    }

    auto KinematicTaskControllerBase::update(const Model &model, Data &data,
                                             const TaskControllerContext &ctx,
                                             JointVelocity &out) noexcept -> ControllerStatus {

        if (ctx.fb.q.size() != model.dof || out.v.size() != model.dof) {
            return ControllerStatus::SIZE_MISMATCH;
        }

        data.q = ctx.fb.q;

        // Eager kinematics refresh: task-space -> joint-space routing always needs them.
        compute_jacobians(model, data);

        KinematicsCache kin(model, data, /*kin_fresh=*/true);

        manifold::SE3::Twist cmd_twist;
        if (!compute_command_twist(model, data, kin, ctx, cmd_twist)) {
            return ControllerStatus::HOOK_FAILED;
        }

        // DLS-IDK always runs first; result is the unconstrained (pre-QP) command.
        // When constraint_aware, the DLS output is saved as v_des before the OptIK
        // QP may reshape it, giving a meaningful v_des vs v_safe comparison.
        inverse_diff_kinematics(model, data, cmd_twist, ik_options);
        const Eigen::VectorXd v_des_scratch = data.v_out;  // pre-QP DLS solution

        if (constraint_aware) {
            if (col_model_ == nullptr || col_data_ == nullptr) {
                debug::log("constraint_aware=true but no collision attached; NOT_CONFIGURED");
                return ControllerStatus::NOT_CONFIGURED;
            }

            const OptimalIKStatus status = optimal_inverse_diff_kinematics(
                model, data, *col_model_, *col_data_, cmd_twist, optimal_ik_options);

            if (status != OptimalIKStatus::OK && status != OptimalIKStatus::RELAXED) {
                debug::log("optimal_inverse_diff_kinematics failed");
                diag_.v_ctrl = v_des_scratch;
                diag_.v_des = v_des_scratch;
                diag_.v_safe = Eigen::VectorXd::Zero(model.dof);
                diag_.optik_invoked = true;
                diag_.optik_status = status;
                diag_.optik_modified = false;
                diag_.optik_delta = 0.0;
                return to_controller_status(status);
            }
        }

        out.v = data.v_out;

        diag_.v_ctrl = v_des_scratch;
        diag_.v_des = v_des_scratch;
        diag_.v_safe = data.v_out;
        diag_.optik_invoked = constraint_aware;
        diag_.optik_status = OptimalIKStatus::OK;
        diag_.optik_modified = constraint_aware && ((diag_.v_safe - diag_.v_des).norm() > 1e-9);
        diag_.optik_delta = 0.0;  // not exposed from the convenience overload path

        return ControllerStatus::OK;
    }

    // --- Dynamic Task Controller Base Class ---

    DynamicTaskControllerBase::DynamicTaskControllerBase(const Model &model)
        : tau_ctrl_(Eigen::VectorXd::Zero(model.dof)), tau_des_(Eigen::VectorXd::Zero(model.dof)),
          tau_safe_(Eigen::VectorXd::Zero(model.dof)) {}

    void DynamicTaskControllerBase::attach_collision(const CollisionModel &col_model,
                                                     CollisionData &col_data) noexcept {
        col_model_ = &col_model;
        col_data_ = &col_data;
    }

    void DynamicTaskControllerBase::detach_collision() noexcept {
        col_model_ = nullptr;
        col_data_ = nullptr;
    }

    auto DynamicTaskControllerBase::update(const Model &model, Data &data,
                                           const TaskControllerContext &ctx,
                                           JointTorque &out) noexcept -> ControllerStatus {

        if (ctx.fb.q.size() != model.dof || ctx.fb.v.size() != model.dof ||
            out.tau.size() != model.dof) {
            return ControllerStatus::SIZE_MISMATCH;
        }

        data.q = ctx.fb.q;

        // Eager kinematics refresh: task-space -> joint-space routing always needs them.
        compute_jacobians(model, data);

        KinematicsCache kin(model, data, /*kin_fresh=*/true);
        DynamicsCache dyn(model, data, ctx.fb.v, /*m_fresh=*/false, /*h_fresh=*/false,
                          /*g_fresh=*/false);

        manifold::SE3::Wrench cmd_wrench;
        if (!compute_command_wrench(model, data, kin, dyn, ctx, cmd_wrench)) {
            return ControllerStatus::HOOK_FAILED;
        }

        // Project body-frame wrench into joint torques.
        tau_ctrl_.noalias() = data.body_jacobian.transpose() * cmd_wrench.coeffs;

        // Bias-force compensation.
        switch (bias_compensation) {
        case BiasCompensation::None:
            tau_des_ = tau_ctrl_;
            break;
        case BiasCompensation::GravityOnly:
            tau_des_ = tau_ctrl_ + dyn.g();
            break;
        case BiasCompensation::Full:
            tau_des_ = tau_ctrl_ + dyn.h();
            break;
        }

        // Optional ASIF certification.
        if (constraint_aware) {
            if (col_model_ == nullptr || col_data_ == nullptr) {
                debug::log("constraint_aware=true but no collision attached; NOT_CONFIGURED");
                return ControllerStatus::NOT_CONFIGURED;
            }

            (void)dyn.M();
            (void)dyn.h();

            DynPositionBarrier pbar(model);
            pbar.alpha_0 = asif_defaults::kBarrierAlpha0;
            pbar.alpha_1 = asif_defaults::kBarrierAlpha1;

            DynVelocityBarrier vbar(model);
            vbar.alpha_0 = asif_defaults::kBarrierAlpha0;

            DynCollisionBarrier cbar(model, *col_model_,
                                     asif_defaults::kCollisionActivationDistance);
            cbar.alpha_0 = asif_defaults::kBarrierAlpha0;
            cbar.alpha_1 = asif_defaults::kBarrierAlpha1;
            cbar.per_pair_activation_distance = asif_options.per_pair_activation_distance;

            update_geometry_poses(model, data, *col_model_, *col_data_);
            (void)compute_min_distance(*col_model_, *col_data_, cbar.max_activation_distance());

            const DynamicBarrier *bar_ptrs[3] = {&pbar, &vbar, &cbar};

            ASIFOptions opts_eff = asif_options;
            if (opts_eff.tau_max.size() != model.dof) {
                opts_eff.tau_max.resize(model.dof);
                for (int i = 0; i < model.dof; ++i) {
                    opts_eff.tau_max[i] = model.limits[i].tau_max;
                }
            }
            if (opts_eff.tau_min.size() != model.dof) { opts_eff.tau_min = -opts_eff.tau_max; }

            // Task-consistent cost: always on for task-mode dynamic controllers.
            // Prevents the QP from exploiting cheap wrist torques to satisfy the CBF
            // at the expense of orientation tracking (wrist-twist artefact).
            // w_pos = 1.0, w_rot = 0.1 (≈ L_char^2 for L_char ≈ 0.32 m).
            if (opts_eff.W_task.size() != 6) {
                opts_eff.W_task.resize(6);
                opts_eff.W_task << 1.0, 1.0, 1.0, 0.1, 0.1, 0.1;
            }

            const ASIFStatus asif_status =
                asif_filter(model, data, col_model_, col_data_, ctx.fb.v, tau_des_,
                            std::span<const DynamicBarrier *const>(bar_ptrs), tau_safe_, opts_eff);

            if (asif_status != ASIFStatus::OK && asif_status != ASIFStatus::RELAXED) {
                debug::log("asif_filter failed");
                diag_.tau_ctrl = tau_ctrl_;
                diag_.tau_des = tau_des_;
                diag_.tau_safe = Eigen::VectorXd::Zero(model.dof);
                diag_.asif_invoked = true;
                diag_.asif_status = asif_status;
                diag_.asif_modified = false;
                diag_.asif_delta = 0.0;
                return to_controller_status(asif_status);
            }

            out.tau = tau_safe_;

            diag_.tau_ctrl = tau_ctrl_;
            diag_.tau_des = tau_des_;
            diag_.tau_safe = tau_safe_;
            diag_.asif_invoked = true;
            diag_.asif_status = asif_status;
            diag_.asif_modified = (tau_safe_ - tau_des_).norm() > 1e-9;
            diag_.asif_delta = 0.0;  // not exposed from the convenience overload path
        } else {
            out.tau = tau_des_;
            tau_safe_ = tau_des_;

            diag_.tau_ctrl = tau_ctrl_;
            diag_.tau_des = tau_des_;
            diag_.tau_safe = tau_des_;
            diag_.asif_invoked = false;
            diag_.asif_status = ASIFStatus::OK;
            diag_.asif_modified = false;
            diag_.asif_delta = 0.0;
        }

        return ControllerStatus::OK;
    }

    // --- Kinematic Joint Controller Base Class ---

    KinematicJointControllerBase::KinematicJointControllerBase(const Model &model)
        : v_ctrl_(model.dof) {}

    auto KinematicJointControllerBase::update(const Model &model, Data &data,
                                              const JointControllerContext &ctx,
                                              JointVelocity &out) noexcept -> ControllerStatus {

        if (ctx.fb.q.size() != model.dof || out.v.size() != model.dof) {
            return ControllerStatus::SIZE_MISMATCH;
        }

        data.q = ctx.fb.q;

        KinematicsCache kin(model, data, /*kin_fresh=*/false);

        if (!compute_command_velocity(model, data, kin, ctx, v_ctrl_)) {
            return ControllerStatus::HOOK_FAILED;
        }

        // Direction-preserving velocity-limit rescale: divide the whole vector by the
        // largest abs_vel / limit factor so the worst joint hits its limit exactly and
        // the others scale down proportionally.
        bool rescaled = false;
        if (constraint_aware) {
            double max_excess_factor = 1.0;
            for (int i = 0; i < model.dof; ++i) {
                const double abs_vel = std::abs(v_ctrl_.v(i));
                const double limit = model.limits[i].q_vel_max;
                if (limit > 0.0 && abs_vel > limit) {
                    const double excess = abs_vel / limit;
                    max_excess_factor = std::max(excess, max_excess_factor);
                }
            }
            if (max_excess_factor > 1.0) {
                v_ctrl_.v /= max_excess_factor;
                rescaled = true;
            }
        }

        out.v = v_ctrl_.v;

        diag_.v_ctrl = v_ctrl_.v;
        diag_.v_des = v_ctrl_.v;
        diag_.v_safe = v_ctrl_.v;
        diag_.optik_invoked = false;
        diag_.optik_status = OptimalIKStatus::OK;
        diag_.optik_modified = rescaled;
        diag_.optik_delta = 0.0;

        return ControllerStatus::OK;
    }

    // --- Dynamic Joint Controller Base Class ---

    DynamicJointControllerBase::DynamicJointControllerBase(const Model &model)
        : tau_ctrl_(model.dof), tau_des_(Eigen::VectorXd::Zero(model.dof)),
          tau_safe_(Eigen::VectorXd::Zero(model.dof)) {}

    void DynamicJointControllerBase::attach_collision(const CollisionModel &col_model,
                                                      CollisionData &col_data) noexcept {
        col_model_ = &col_model;
        col_data_ = &col_data;
    }

    void DynamicJointControllerBase::detach_collision() noexcept {
        col_model_ = nullptr;
        col_data_ = nullptr;
    }

    auto DynamicJointControllerBase::update(const Model &model, Data &data,
                                            const JointControllerContext &ctx,
                                            JointTorque &out) noexcept -> ControllerStatus {

        if (ctx.fb.q.size() != model.dof || ctx.fb.v.size() != model.dof ||
            out.tau.size() != model.dof) {
            return ControllerStatus::SIZE_MISMATCH;
        }

        data.q = ctx.fb.q;

        KinematicsCache kin(model, data, /*kin_fresh=*/false);
        DynamicsCache dyn(model, data, ctx.fb.v, /*m_fresh=*/false, /*h_fresh=*/false,
                          /*g_fresh=*/false);

        if (!compute_command_torque(model, data, kin, dyn, ctx, tau_ctrl_)) {
            return ControllerStatus::HOOK_FAILED;
        }

        // Bias-force compensation.
        switch (bias_compensation) {
        case BiasCompensation::None:
            tau_des_ = tau_ctrl_.tau;
            break;
        case BiasCompensation::GravityOnly:
            tau_des_ = tau_ctrl_.tau + dyn.g();
            break;
        case BiasCompensation::Full:
            tau_des_ = tau_ctrl_.tau + dyn.h();
            break;
        }

        // Optional ASIF certification.
        if (constraint_aware) {
            if (col_model_ == nullptr || col_data_ == nullptr) {
                debug::log("constraint_aware=true but no collision attached; NOT_CONFIGURED");
                return ControllerStatus::NOT_CONFIGURED;
            }

            // This base uses lazy kinematics; the composable asif_filter asserts
            // body_jacobian must be sized for the composable asif_filter assert.
            kin.refresh();

            (void)dyn.M();
            (void)dyn.h();

            DynPositionBarrier pbar(model);
            pbar.alpha_0 = asif_defaults::kBarrierAlpha0;
            pbar.alpha_1 = asif_defaults::kBarrierAlpha1;

            DynVelocityBarrier vbar(model);
            vbar.alpha_0 = asif_defaults::kBarrierAlpha0;

            DynCollisionBarrier cbar(model, *col_model_,
                                     asif_defaults::kCollisionActivationDistance);
            cbar.alpha_0 = asif_defaults::kBarrierAlpha0;
            cbar.alpha_1 = asif_defaults::kBarrierAlpha1;
            cbar.per_pair_activation_distance = asif_options.per_pair_activation_distance;

            update_geometry_poses(model, data, *col_model_, *col_data_);
            (void)compute_min_distance(*col_model_, *col_data_, cbar.max_activation_distance());

            const DynamicBarrier *bar_ptrs[3] = {&pbar, &vbar, &cbar};

            ASIFOptions opts_eff = asif_options;
            if (opts_eff.tau_max.size() != model.dof) {
                opts_eff.tau_max.resize(model.dof);
                for (int i = 0; i < model.dof; ++i) {
                    opts_eff.tau_max[i] = model.limits[i].tau_max;
                }
            }
            if (opts_eff.tau_min.size() != model.dof) { opts_eff.tau_min = -opts_eff.tau_max; }
            // Joint-mode base: leave W_task empty (joint-space cost preserved).

            const ASIFStatus asif_status =
                asif_filter(model, data, col_model_, col_data_, ctx.fb.v, tau_des_,
                            std::span<const DynamicBarrier *const>(bar_ptrs), tau_safe_, opts_eff);

            if (asif_status != ASIFStatus::OK && asif_status != ASIFStatus::RELAXED) {
                debug::log("asif_filter failed");
                diag_.tau_ctrl = tau_ctrl_.tau;
                diag_.tau_des = tau_des_;
                diag_.tau_safe = Eigen::VectorXd::Zero(model.dof);
                diag_.asif_invoked = true;
                diag_.asif_status = asif_status;
                diag_.asif_modified = false;
                diag_.asif_delta = 0.0;
                return to_controller_status(asif_status);
            }

            out.tau = tau_safe_;

            diag_.tau_ctrl = tau_ctrl_.tau;
            diag_.tau_des = tau_des_;
            diag_.tau_safe = tau_safe_;
            diag_.asif_invoked = true;
            diag_.asif_status = asif_status;
            diag_.asif_modified = (tau_safe_ - tau_des_).norm() > 1e-9;
            diag_.asif_delta = 0.0;
        } else {
            out.tau = tau_des_;
            tau_safe_ = tau_des_;

            diag_.tau_ctrl = tau_ctrl_.tau;
            diag_.tau_des = tau_des_;
            diag_.tau_safe = tau_des_;
            diag_.asif_invoked = false;
            diag_.asif_status = ASIFStatus::OK;
            diag_.asif_modified = false;
            diag_.asif_delta = 0.0;
        }

        return ControllerStatus::OK;
    }

}  // namespace xarm_geo
