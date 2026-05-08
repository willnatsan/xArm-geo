#include <cassert>

#include <xarm_geo/control/controller.h>
#include <xarm_geo/modelling/dynamics.h>
#include <xarm_geo/utils/debug.h>

namespace xarm_geo {

    // --- KinematicTaskControllerBase ---

    KinematicTaskControllerBase::KinematicTaskControllerBase(const Model & /*model*/) noexcept {
        // No joint-sized scratch on this base; ctor exists for symmetry and
        // to keep the construction-style uniform across all four bases.
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

        // Live-phase size checks: status return (honours noexcept; no UB in release).
        if (ctx.fb.q.size() != model.dof || out.v.size() != model.dof) {
            return ControllerStatus::SIZE_MISMATCH;
        }

        // Sync canonical state and refresh kinematic tree + Jacobians.
        data.q = ctx.fb.q;
        compute_jacobians(model, data);

        // Derived class produces the body-frame command twist.
        manifold::SE3::Twist cmd_twist;
        if (!compute_command_twist(model, data, ctx, cmd_twist)) {
            return ControllerStatus::HOOK_FAILED;
        }

        // Solver path: standard DLS-IDK or safety-aware optimal IDK.
        if (constraint_aware) {
            if (col_model_ == nullptr || col_data_ == nullptr) {
                debug::log("constraint_aware=true but no collision attached; NOT_CONFIGURED");
                return ControllerStatus::NOT_CONFIGURED;
            }

            const OptimalIKStatus status = optimal_inverse_diff_kinematics(
                model, data, *col_model_, *col_data_, cmd_twist, optimal_ik_options);

            if (status != OptimalIKStatus::OK) {
                debug::log("optimal_inverse_diff_kinematics failed");
                return to_controller_status(status);
            }
        } else {
            inverse_diff_kinematics(model, data, cmd_twist, ik_options);
        }

        out.v = data.v_out;
        return ControllerStatus::OK;
    }

    // --- DynamicTaskControllerBase ---

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

        // Sync canonical state and refresh kinematic tree + Jacobians.
        data.q = ctx.fb.q;
        compute_jacobians(model, data);

        // Optional dynamics refresh: populate data.M, data.h, and data.g
        // before the hook so the user's control law can read them directly.
        //
        // data.g is only refreshed when the policy is GravityOnly to avoid
        // a redundant RNEA pass for Full / None.
        bool h_fresh = false;
        bool g_fresh = false;
        if (refresh_dynamics) {
            compute_mass_matrix(model, data);
            compute_bias_forces(model, data, ctx.fb.v);
            h_fresh = true;
            if (bias_compensation == BiasCompensation::GravityOnly) {
                compute_gravity_forces(model, data);
                g_fresh = true;
            }
        }

        // Derived class produces the body-frame end-effector wrench.
        manifold::SE3::Wrench cmd_wrench;
        if (!compute_command_wrench(model, data, ctx, cmd_wrench)) {
            return ControllerStatus::HOOK_FAILED;
        }

        // Project task-space wrench to joint torques.
        tau_ctrl_.noalias() = data.body_jacobian.transpose() * cmd_wrench.coeffs;

        // Bias-force compensation: append the term selected by `bias_compensation`.
        switch (bias_compensation) {
        case BiasCompensation::None:
            tau_des_ = tau_ctrl_;
            break;
        case BiasCompensation::GravityOnly:
            if (!g_fresh) { compute_gravity_forces(model, data); }
            tau_des_ = tau_ctrl_ + data.g;
            break;
        case BiasCompensation::Full:
            if (!h_fresh) { compute_bias_forces(model, data, ctx.fb.v); }
            tau_des_ = tau_ctrl_ + data.h;
            break;
        }

        // Optional ASIF certification stage.
        if (constraint_aware) {
            if (col_model_ == nullptr || col_data_ == nullptr) {
                debug::log("constraint_aware=true but no collision attached; NOT_CONFIGURED");
                return ControllerStatus::NOT_CONFIGURED;
            }

            // NOTE: this convenience overload of asif_filter recomputes
            // data.M and data.h internally. If the user has already
            // populated them (refresh_dynamics or bias_compensation == Full).
            //
            // TODO: switch to the composable overload of
            // asif_filter once the default-barrier list is plumbed through.
            const ASIFStatus status = asif_filter(model, data, *col_model_, *col_data_, ctx.fb.v,
                                                  tau_des_, tau_safe_, asif_options);

            if (status != ASIFStatus::OK) {
                debug::log("asif_filter failed");
                return to_controller_status(status);
            }

            out.tau = tau_safe_;
        } else {
            out.tau = tau_des_;
        }

        return ControllerStatus::OK;
    }

    // --- KinematicJointControllerBase ---

    KinematicJointControllerBase::KinematicJointControllerBase(const Model &model)
        : v_ctrl_(model.dof) {}

    auto KinematicJointControllerBase::update(const Model &model, Data &data,
                                              const JointControllerContext &ctx,
                                              JointVelocity &out) noexcept -> ControllerStatus {

        if (ctx.fb.q.size() != model.dof || out.v.size() != model.dof) {
            return ControllerStatus::SIZE_MISMATCH;
        }

        // Sync canonical state.
        data.q = ctx.fb.q;
        if (refresh_kinematics) { compute_jacobians(model, data); }

        if (!compute_command_velocity(model, data, ctx, v_ctrl_)) {
            return ControllerStatus::HOOK_FAILED;
        }

        // Direction-preserving velocity-limit rescale.
        if (constraint_aware) {
            double max_scale_factor = 1.0;
            for (int i = 0; i < model.dof; ++i) {
                const double abs_vel = std::abs(v_ctrl_.v(i));
                const double limit = model.limits[i].q_vel_max;
                if (limit > 0.0 && abs_vel > limit) {
                    const double scale = abs_vel / limit;
                    max_scale_factor = std::max(scale, max_scale_factor);
                }
            }
            if (max_scale_factor > 1.0) { v_ctrl_.v /= max_scale_factor; }
        }

        out.v = v_ctrl_.v;
        return ControllerStatus::OK;
    }

    // --- DynamicJointControllerBase ---

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

        // Sync canonical state.
        data.q = ctx.fb.q;
        if (refresh_kinematics) { compute_jacobians(model, data); }

        // Optional dynamics refresh: populate data.M, data.h, and data.g
        // before the hook so the user's control law can read them directly.
        //
        // data.g is only refreshed when the policy is GravityOnly to avoid
        // a redundant RNEA pass for Full / None.
        bool h_fresh = false;
        bool g_fresh = false;
        if (refresh_dynamics) {
            compute_mass_matrix(model, data);
            compute_bias_forces(model, data, ctx.fb.v);
            h_fresh = true;
            if (bias_compensation == BiasCompensation::GravityOnly) {
                compute_gravity_forces(model, data);
                g_fresh = true;
            }
        }

        if (!compute_command_torque(model, data, ctx, tau_ctrl_)) {
            return ControllerStatus::HOOK_FAILED;
        }

        // Bias-force compensation: append the term selected by `bias_compensation`.
        switch (bias_compensation) {
        case BiasCompensation::None:
            tau_des_ = tau_ctrl_.tau;
            break;
        case BiasCompensation::GravityOnly:
            if (!g_fresh) { compute_gravity_forces(model, data); }
            tau_des_ = tau_ctrl_.tau + data.g;
            break;
        case BiasCompensation::Full:
            if (!h_fresh) { compute_bias_forces(model, data, ctx.fb.v); }
            tau_des_ = tau_ctrl_.tau + data.h;
            break;
        }

        // Optional ASIF certification stage.
        if (constraint_aware) {
            if (col_model_ == nullptr || col_data_ == nullptr) {
                debug::log("constraint_aware=true but no collision attached; NOT_CONFIGURED");
                return ControllerStatus::NOT_CONFIGURED;
            }

            // NOTE: see DynamicTaskControllerBase for the same TODO regarding
            // redundant data.M / data.h recomputation inside asif_filter.
            const ASIFStatus status = asif_filter(model, data, *col_model_, *col_data_, ctx.fb.v,
                                                  tau_des_, tau_safe_, asif_options);

            if (status != ASIFStatus::OK) {
                debug::log("asif_filter failed");
                return to_controller_status(status);
            }

            out.tau = tau_safe_;
        } else {
            out.tau = tau_des_;
        }

        return ControllerStatus::OK;
    }

}  // namespace xarm_geo
