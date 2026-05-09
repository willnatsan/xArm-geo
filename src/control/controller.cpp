#include <cassert>

#include <xarm_geo/control/controller.h>
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

        // Sync canonical state
        data.q = ctx.fb.q;

        // Mandatory kinematics refresh: populate data.body_jacobian, data.ee_pose, etc.
        // Note: Mandatory, as updated kinematics are always required (task-space -> joint-space)
        compute_jacobians(model, data);

        // Construct hook-side cache (kinematics already fresh; reads through `kin` are free).
        KinematicsCache kin(model, data, /*kin_fresh=*/true);

        // Derived class produces the body-frame command twist.
        manifold::SE3::Twist cmd_twist;
        if (!compute_command_twist(model, data, kin, ctx, cmd_twist)) {
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

        // Sync canonical state
        data.q = ctx.fb.q;

        // Mandatory kinematics refresh: always required for task-space -> joint-space routing.
        compute_jacobians(model, data);

        // Construct hook-side caches; kinematics already fresh; dynamics is always lazy
        // (first cache access trigger the relevant RNEA / CRBA pass if not already
        // populated by a prior access).
        KinematicsCache kin(model, data, /*kin_fresh=*/true);
        DynamicsCache dyn(model, data, ctx.fb.v, /*m_fresh=*/false, /*h_fresh=*/false,
                          /*g_fresh=*/false);

        // Derived class produces the body-frame end-effector wrench.
        manifold::SE3::Wrench cmd_wrench;
        if (!compute_command_wrench(model, data, kin, dyn, ctx, cmd_wrench)) {
            return ControllerStatus::HOOK_FAILED;
        }

        // Project task-space wrench to joint torques.
        tau_ctrl_.noalias() = data.body_jacobian.transpose() * cmd_wrench.coeffs;

        // Bias-force compensation: append the term selected by `bias_compensation`.
        // Reads route through `dyn` so the cache memoises across hook + base usage.
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

        // Optional ASIF certification stage.
        if (constraint_aware) {
            if (col_model_ == nullptr || col_data_ == nullptr) {
                debug::log("constraint_aware=true but no collision attached; NOT_CONFIGURED");
                return ControllerStatus::NOT_CONFIGURED;
            }

            // TODO: switch to the composable asif_filter overload to avoid recomputing
            // data.M / data.h when the base has already populated them.
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

        // Construct hook-side cache; kinematics is lazy
        // (compute_jacobians runs on first access if the hook needs kinematic state).
        KinematicsCache kin(model, data, /*kin_fresh=*/false);

        if (!compute_command_velocity(model, data, kin, ctx, v_ctrl_)) {
            return ControllerStatus::HOOK_FAILED;
        }

        // Direction-preserving velocity-limit rescale.
        //
        // For each joint, compute the over-limit factor abs_vel / limit; track the largest
        // over all joints. If any joint exceeds its limit, divide the whole velocity vector
        // by that factor -- this clamps the worst-offending joint exactly to its limit and
        // reduces the rest proportionally, preserving the direction of v_ctrl in joint space.
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
            if (max_excess_factor > 1.0) { v_ctrl_.v /= max_excess_factor; }
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

        // Construct hook-side caches; both kinematics and dynamics are lazy
        // (first cache access triggers the relevant computation if needed).
        KinematicsCache kin(model, data, /*kin_fresh=*/false);
        DynamicsCache dyn(model, data, ctx.fb.v, /*m_fresh=*/false, /*h_fresh=*/false,
                          /*g_fresh=*/false);

        if (!compute_command_torque(model, data, kin, dyn, ctx, tau_ctrl_)) {
            return ControllerStatus::HOOK_FAILED;
        }

        // Bias-force compensation: append the term selected by `bias_compensation`.
        // Reads route through `dyn` so the cache memoises across hook + base usage.
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

        // Optional ASIF certification stage.
        if (constraint_aware) {
            if (col_model_ == nullptr || col_data_ == nullptr) {
                debug::log("constraint_aware=true but no collision attached; NOT_CONFIGURED");
                return ControllerStatus::NOT_CONFIGURED;
            }

            // TODO: see DynamicTaskControllerBase for the same asif_filter redundancy.
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
