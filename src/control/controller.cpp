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

        // DLS-IDK by default; optimal (safety-aware) IDK when constraint_aware.
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

            // TODO: see DynamicTaskControllerBase::update for the same asif_filter redundancy.
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
