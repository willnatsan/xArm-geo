#pragma once

#include <chrono>
#include <cstdint>

#include <Eigen/Dense>

#include <xarm_geo/core/manifold.h>
#include <xarm_geo/core/motion.h>
#include <xarm_geo/core/system.h>
#include <xarm_geo/modelling/collision.h>
#include <xarm_geo/modelling/kinematics.h>
#include <xarm_geo/modelling/optimal_kinematics.h>
#include <xarm_geo/safety/asif.h>
#include <xarm_geo/trajectory/trajectory.h>

namespace xarm_geo {

    // --- Controller Metadata ---

    enum class ControllerStatus : std::uint8_t { OK, ERROR };

    struct ControllerContext {
        std::chrono::nanoseconds dt;
    };

    // --- Abstract Base: KinematicTaskController ---
    //
    // Common pipeline for SE(3)-tracking velocity-mode controllers:
    //   1. Sync data.q from feedback and refresh kinematic tree + Jacobians
    //      (compute_jacobians).
    //   2. Derived class produces a body-frame command twist via the
    //      compute_command_twist() hook.
    //   3. Base routes the twist through:
    //        - inverse_diff_kinematics             (constraint_aware = false)
    //        - optimal_inverse_diff_kinematics     (constraint_aware = true,
    //                                               requires attach_collision)
    //   4. Result is written to JointVelocity out.
    //
    // Note: derived classes should NOT redefine update(); override only the
    // protected hook compute_command_twist().

    class KinematicTaskController {
    public:
        virtual ~KinematicTaskController() = default;

        // Final entry point. Calls compute_command_twist() then dispatches.
        auto update(const Model &model, Data &data, const JointState &fb,
                    const TaskSpaceTarget &ref, const ControllerContext &ctx,
                    JointVelocity &out) noexcept -> ControllerStatus;

        // --- Optional Collision Attachment ---
        //
        // Required when constraint_aware = true. The pointers are non-owning;
        // the caller must ensure col_model / col_data outlive the controller.
        void attach_collision(const CollisionModel &col_model, CollisionData &col_data) noexcept;
        void detach_collision() noexcept;

        // --- Public Configuration ---
        bool constraint_aware = false;
        IKOptions ik_options;
        OptimalIKOptions optimal_ik_options;

    protected:
        virtual auto compute_command_twist(const Model &model, Data &data, const JointState &fb,
                                           const TaskSpaceTarget &ref, const ControllerContext &ctx,
                                           manifold::SE3::Twist &cmd_twist) noexcept
            -> ControllerStatus = 0;

        const CollisionModel *col_model_ = nullptr;
        CollisionData *col_data_ = nullptr;
    };

    // --- Abstract Base: DynamicTaskController ---
    //
    // Common pipeline for SE(3)-tracking torque-mode controllers:
    //   1. Sync data.q from feedback and refresh kinematic tree + Jacobians
    //      (compute_jacobians).
    //   2. Derived class produces a body-frame end-effector wrench via the
    //      compute_task_wrench() hook. Derived classes may compute additional
    //      dynamics quantities (e.g. operational-space inertia) themselves;
    //      the base does NOT call compute_mass_matrix() before the hook.
    //   3. Base projects the wrench to joint torques: tau = J_b^T * F + h(q,v)
    //      via compute_bias_forces.
    //   4. Optional ASIF filter (constraint_aware = true; requires
    //      attach_collision) certifies the resulting torque.
    //   5. Result is written to JointTorque out.
    //
    // Note: derived classes should NOT redefine update(); override only the
    // protected hook compute_task_wrench().

    class DynamicTaskController {
    public:
        virtual ~DynamicTaskController() = default;

        auto update(const Model &model, Data &data, const JointState &fb,
                    const TaskSpaceTarget &ref, const ControllerContext &ctx,
                    JointTorque &out) noexcept -> ControllerStatus;

        // --- Optional Collision Attachment ---
        //
        // Required when constraint_aware = true. The pointers are non-owning;
        // the caller must ensure col_model / col_data outlive the controller.
        void attach_collision(const CollisionModel &col_model, CollisionData &col_data) noexcept;
        void detach_collision() noexcept;

        // --- Public Configuration ---
        bool constraint_aware = false;
        ASIFOptions asif_options;

    protected:
        virtual auto compute_task_wrench(const Model &model, Data &data, const JointState &fb,
                                         const TaskSpaceTarget &ref, const ControllerContext &ctx,
                                         manifold::SE3::Twist &task_wrench) noexcept
            -> ControllerStatus = 0;

        const CollisionModel *col_model_ = nullptr;
        CollisionData *col_data_ = nullptr;

        // Pre-allocated scratch for the base's joint-torque assembly. Sized
        // lazily on first update() call (when size() != model.dof); zero
        // allocation on subsequent calls.
        Eigen::VectorXd tau_task_joint_;
        Eigen::VectorXd tau_des_;
        Eigen::VectorXd tau_safe_;
    };

}  // namespace xarm_geo
