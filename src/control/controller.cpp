#include <cassert>

#include <xarm_geo/control/controller.h>
#include <xarm_geo/modelling/dynamics.h>
#include <xarm_geo/utils/debug.h>

namespace xarm_geo {

    // --- KinematicTaskController ---

    void KinematicTaskController::attach_collision(const CollisionModel &col_model,
                                                   CollisionData &col_data) noexcept {
        col_model_ = &col_model;
        col_data_ = &col_data;
    }

    void KinematicTaskController::detach_collision() noexcept {
        col_model_ = nullptr;
        col_data_ = nullptr;
    }

    auto KinematicTaskController::update(const Model &model, Data &data, const JointState &fb,
                                         const TaskSpaceTarget &ref, const ControllerContext &ctx,
                                         JointVelocity &out) noexcept -> ControllerStatus {

        assert(fb.q.size() == model.dof && "fb.q size must equal model.dof");
        assert(out.v.size() == model.dof && "out.v size must equal model.dof");

        // Sync canonical state and refresh kinematic tree + Jacobians.
        data.q = fb.q;
        compute_jacobians(model, data);

        // Derived class produces the body-frame command twist.
        manifold::SE3::Twist cmd_twist;
        const ControllerStatus hook_status =
            compute_command_twist(model, data, fb, ref, ctx, cmd_twist);
        if (hook_status != ControllerStatus::OK) { return hook_status; }

        // Solver path: standard DLS-IDK or safety-aware optimal IDK.
        if (constraint_aware) {
            if (col_model_ == nullptr || col_data_ == nullptr) {
                debug::log("constraint_aware=true but no collision attached; returning ERROR");
                return ControllerStatus::ERROR;
            }

            const OptimalIKStatus status = optimal_inverse_diff_kinematics(
                model, data, *col_model_, *col_data_, cmd_twist, optimal_ik_options);

            if (status != OptimalIKStatus::OK) {
                debug::log("optimal_inverse_diff_kinematics failed");
                return ControllerStatus::ERROR;
            }
        } else {
            inverse_diff_kinematics(model, data, cmd_twist, ik_options);
        }

        out.v = data.v_out;
        return ControllerStatus::OK;
    }

    // --- DynamicTaskController ---

    void DynamicTaskController::attach_collision(const CollisionModel &col_model,
                                                 CollisionData &col_data) noexcept {
        col_model_ = &col_model;
        col_data_ = &col_data;
    }

    void DynamicTaskController::detach_collision() noexcept {
        col_model_ = nullptr;
        col_data_ = nullptr;
    }

    auto DynamicTaskController::update(const Model &model, Data &data, const JointState &fb,
                                       const TaskSpaceTarget &ref, const ControllerContext &ctx,
                                       JointTorque &out) noexcept -> ControllerStatus {

        assert(fb.q.size() == model.dof && "fb.q size must equal model.dof");
        assert(fb.v.size() == model.dof && "fb.v size must equal model.dof");
        assert(out.tau.size() == model.dof && "out.tau size must equal model.dof");

        // Lazy-resize the joint-torque assembly scratch on first use.
        if (tau_task_joint_.size() != model.dof) {
            tau_task_joint_.setZero(model.dof);
            tau_des_.setZero(model.dof);
            tau_safe_.setZero(model.dof);
        }

        // Sync canonical state and refresh kinematic tree + Jacobians.
        data.q = fb.q;
        compute_jacobians(model, data);

        // Derived class produces the body-frame end-effector wrench.
        manifold::SE3::Twist task_wrench;
        const ControllerStatus hook_status =
            compute_task_wrench(model, data, fb, ref, ctx, task_wrench);
        if (hook_status != ControllerStatus::OK) { return hook_status; }

        // Project task-space wrench to joint torques.
        tau_task_joint_.noalias() = data.body_jacobian.transpose() * task_wrench;

        // Bias-force compensation (gravity + Coriolis if model.gravity != 0).
        compute_bias_forces(model, data, fb.v);
        tau_des_ = tau_task_joint_ + data.h;

        // Optional ASIF certification stage.
        if (constraint_aware) {
            if (col_model_ == nullptr || col_data_ == nullptr) {
                debug::log("constraint_aware=true but no collision attached; returning ERROR");
                return ControllerStatus::ERROR;
            }

            const ASIFStatus status = asif_filter(model, data, *col_model_, *col_data_, fb.v,
                                                  tau_des_, tau_safe_, asif_options);

            if (status != ASIFStatus::OK) {
                debug::log("asif_filter failed");
                return ControllerStatus::ERROR;
            }

            out.tau = tau_safe_;
        } else {
            out.tau = tau_des_;
        }

        return ControllerStatus::OK;
    }

}  // namespace xarm_geo
