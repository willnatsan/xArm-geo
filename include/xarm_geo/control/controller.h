#pragma once

#include <chrono>
#include <concepts>
#include <cstdint>

#include <Eigen/Dense>

#include <vector>

#include <xarm_geo/core/manifold.h>
#include <xarm_geo/core/motion.h>
#include <xarm_geo/core/system.h>
#include <xarm_geo/modelling/collision.h>
#include <xarm_geo/modelling/dynamics.h>
#include <xarm_geo/modelling/kinematics.h>
#include <xarm_geo/modelling/optimal_kinematics.h>
#include <xarm_geo/safety/asif.h>
#include <xarm_geo/trajectory/trajectory.h>

// --- Controller Architecture Notes ---
//
// All four base classes follow the same template-method shape:
//
//   update(model, data, ctx, out)
//     1. Size checks.
//     2. data.q <- ctx.fb.q.
//     3. Refresh kinematics (always for task-space bases; gated for
//        joint-space bases via refresh_kinematics).
//     4. Refresh dynamics (gated for the dynamic bases via refresh_dynamics).
//     5. Construct KinematicsCache / DynamicsCache reflecting freshness (if needed).
//     6. Hook: compute_command_<X>(...).  <-- this is what the user writes.
//     7. Post-processing:
//          - kinematic-task : route through (optimal_)inverse_diff_kinematics
//          - dynamic-task   : project J_b^T F, then optional bias add,
//                             then optional ASIF
//          - kinematic-joint: optional direction-preserving rescale
//          - dynamic-joint  : optional bias add, then optional ASIF
//     8. Write to `out` exactly once at the end.
//
// Hook-side access pattern:
//   - data.q is canonical and always fresh.
//   - Kinematic state (Jacobians, EE pose, ...) is read through the
//     `KinematicsCache` argument: kin.body_jacobian(), kin.ee_pose(), ...
//   - Dynamic state (M, h, g) is read through the `DynamicsCache` argument:
//     dyn.M(), dyn.h(), dyn.g().
//   - The hook should NOT read data.ee_pose / data.body_jacobian / data.M /
//     data.h / data.g directly: the cache accessors are the canonical path
//     and guarantee freshness without redundant work.

// --- Customising the constraint / safety set ---
//
// The base classes wrap the *convenience* overloads of:
//   - optimal_inverse_diff_kinematics  (kinematic-task base)
//   - asif_filter                      (dynamic-task & dynamic-joint bases)
//
// Those overloads install an opinionated default safety set:
//
//   Optimal IDK : TwistTask + VelocityLimit + PositionLimit +
//                 CollisionBarrier (activation 0.05 m, alpha 5.0).
//   ASIF        : DynPositionBarrier + DynVelocityBarrier +
//                 DynCollisionBarrier (alpha_0 25, alpha_1 10,
//                 activation 0.05 m). Torque box auto-filled from
//                 model.limits[i].tau_max.
//
// To install custom tasks / constraints / kinematic barriers / dynamic
// barriers (or to use a different filtering strategy entirely), DO NOT
// subclass the base. Instead, write your own class that satisfies the
// relevant *Controller concept (defined at the bottom of this file).
//
// Worked Example:
//   - xarm_geo/examples/controllers/custom_controller.h   (PostureBiasedPController:
//                                                          kinematic-task with an
//                                                          augmented Optimal IDK
//                                                          task set)

// TODO: the convenience overload of `asif_filter` recomputes `data.M` and
// `data.h` even when the base has already populated them. Switch to the
// composable overload to share the work.

namespace xarm_geo {

    // --- Bias-Force Cancellation Policy (Dynamic Controllers Only) ---
    //
    //   None         : out = tau_ctrl
    //                  (e.g. explicit feedback linearisation, IDA-PBC).
    //
    //   GravityOnly  : out = tau_ctrl + g(q)
    //                  Base injects gravity only; Coriolis is left to the natural dynamics.
    //                  (Bullo & Murray geometric PD, Slotine--Li adaptive, natural-PD, etc.).
    //
    //   Full         : out = tau_ctrl + h(q, q_dot)  with  h = C(q,q_dot)*q_dot + g(q)
    //                  Base injects the full bias forces.
    //                  ()Computed Torque Control / inverse-dynamics linearisation)
    //
    // Cost: GravityOnly and Full each incur exactly one RNEA pass before the
    // hook (or share one with refresh_dynamics if enabled). None is free.

    enum class BiasCompensation : std::uint8_t { None, GravityOnly, Full };

    // --- Controller Metadata ---

    // Origin column legend:
    //   - base   : produced by the *ControllerBase update() pipeline itself.
    //   - hook   : produced by the user-supplied compute_command_* hook.
    //   - OptIK  : forwarded from optimal_inverse_diff_kinematics
    //              (only reachable from the kinematic-task base).
    //   - ASIF   : forwarded from asif_filter
    //              (only reachable from the dynamic-task and dynamic-joint bases).

    enum class ControllerStatus : std::uint8_t {
        OK,              // origin: base / hook
        SIZE_MISMATCH,   // origin: base   -- fb / out vector sizes != model.dof
        NOT_CONFIGURED,  // origin: base   -- constraint_aware = true but no collision attached
        HOOK_FAILED,     // origin: hook   -- user-supplied hook returned non-OK
        INFEASIBLE,      // origin: OptIK or ASIF (mutually exclusive per base; see legend)
        MAX_ITERS,       // origin: OptIK or ASIF (mutually exclusive per base; see legend)
        SOLVER_ERROR,    // origin: OptIK or ASIF (mutually exclusive per base; see legend)
    };

    // --- Solver Status Mapping (Helper Functions)
    //
    // Translate an upstream solver status (OptimalIKStatus, ASIFStatus) into
    // the corresponding ControllerStatus.

    [[nodiscard]] constexpr auto to_controller_status(OptimalIKStatus s) noexcept
        -> ControllerStatus {
        switch (s) {
        case OptimalIKStatus::OK:
            return ControllerStatus::OK;
        case OptimalIKStatus::INFEASIBLE:
            return ControllerStatus::INFEASIBLE;
        case OptimalIKStatus::MAX_ITERS:
            return ControllerStatus::MAX_ITERS;
        case OptimalIKStatus::ERROR:
            return ControllerStatus::SOLVER_ERROR;
        }
        return ControllerStatus::SOLVER_ERROR;
    }

    [[nodiscard]] constexpr auto to_controller_status(ASIFStatus s) noexcept -> ControllerStatus {
        switch (s) {
        case ASIFStatus::OK:
            return ControllerStatus::OK;
        case ASIFStatus::INFEASIBLE:
            return ControllerStatus::INFEASIBLE;
        case ASIFStatus::MAX_ITERS:
            return ControllerStatus::MAX_ITERS;
        case ASIFStatus::ERROR:
            return ControllerStatus::SOLVER_ERROR;
        }
        return ControllerStatus::SOLVER_ERROR;
    }

    struct TaskControllerContext {
        const JointState &fb;
        const TaskSpaceTarget &ref;
        std::chrono::nanoseconds dt;
    };

    struct JointControllerContext {
        const JointState &fb;
        const JointSpaceTarget &ref;
        std::chrono::nanoseconds dt;
    };

    // --- KinematicsCache (Hook-Side Lazy Accessor for Kinematics State) ---
    //
    // Passed by reference into kinematic-controller hooks. Wraps `data` and
    // exposes lazy accessors for the kinematic fields (Jacobians, EE pose,
    // pose tree). On first access, computes and caches; subsequent calls in
    // the same tick are free.
    //
    // Inside a hook, read kinematic state through this cache rather than
    // touching `data` directly: this guarantees freshness without redundant
    // work and is robust to future changes in the base's refresh policy.

    class KinematicsCache {
    public:
        KinematicsCache(const Model &model, Data &data, bool kin_fresh) noexcept
            : model_(model), data_(data), kin_fresh_(kin_fresh) {}

        // Refresh the full kinematic state (pose tree, EE pose, Jacobians).
        auto refresh() -> void {
            if (!kin_fresh_) {
                compute_jacobians(model_, data_);
                kin_fresh_ = true;
            }
        }

        [[nodiscard]] auto ee_pose() -> const manifold::SE3 & {
            refresh();
            return data_.ee_pose;
        }
        [[nodiscard]] auto body_jacobian() -> const manifold::SE3::Jacobian & {
            refresh();
            return data_.body_jacobian;
        }
        [[nodiscard]] auto space_jacobian() -> const manifold::SE3::Jacobian & {
            refresh();
            return data_.space_jacobian;
        }
        [[nodiscard]] auto frame_jacobian() -> const manifold::SE3::Jacobian & {
            refresh();
            return data_.frame_jacobian;
        }
        [[nodiscard]] auto pose_tree() -> const std::vector<manifold::SE3> & {
            refresh();
            return data_.pose_tree;
        }
        [[nodiscard]] auto pose_tree_local() -> const std::vector<manifold::SE3> & {
            refresh();
            return data_.pose_tree_local;
        }

    private:
        const Model &model_;
        Data &data_;
        bool kin_fresh_;
    };

    // --- DynamicsCache (Hook-Side Lazy Accessor for Dynamics State) ---
    //
    // Passed by reference into dynamic-controller hooks. Wraps `data` and
    // exposes lazy accessors for the joint-space mass matrix M, the full
    // bias forces h = C(q,v)v + g(q), and the gravity-only bias g. On first
    // access, computes and caches; subsequent calls in the same tick are free.
    //
    // Note: g(q) is also computed implicitly when h is recomputed via RNEA;
    // however this cache treats them independently because the eager-refresh
    // path differs (refresh_dynamics populates `data.h` but only populates
    // `data.g` when the bias_compensation policy explicitly asks for it).

    class DynamicsCache {
    public:
        DynamicsCache(const Model &model, Data &data, const Eigen::VectorXd &v, bool m_fresh,
                      bool h_fresh, bool g_fresh) noexcept
            : model_(model), data_(data), v_(v), m_fresh_(m_fresh), h_fresh_(h_fresh),
              g_fresh_(g_fresh) {}

        [[nodiscard]] auto M() -> const Eigen::MatrixXd & {
            if (!m_fresh_) {
                compute_mass_matrix(model_, data_);
                m_fresh_ = true;
            }
            return data_.M;
        }

        [[nodiscard]] auto h() -> const Eigen::VectorXd & {
            if (!h_fresh_) {
                compute_bias_forces(model_, data_, v_);
                h_fresh_ = true;
            }
            return data_.h;
        }

        [[nodiscard]] auto g() -> const Eigen::VectorXd & {
            if (!g_fresh_) {
                compute_gravity_forces(model_, data_);
                g_fresh_ = true;
            }
            return data_.g;
        }

    private:
        const Model &model_;
        Data &data_;
        const Eigen::VectorXd &v_;
        bool m_fresh_;
        bool h_fresh_;
        bool g_fresh_;
    };

    // --- Abstract Base: KinematicTaskControllerBase ---
    //
    // SE(3)-tracking velocity-mode controllers.
    //
    // Refresh policy:
    //   - kinematics : always (compute_jacobians is unconditional).
    //   - dynamics   : never  (kinematic controller; M, h not needed).
    //
    // Public config:
    //   - constraint_aware (default false): route through
    //     optimal_inverse_diff_kinematics; requires attach_collision.
    //   - ik_options, optimal_ik_options.
    //
    // Note: derived classes should NOT redefine update(); override only the
    // protected hook compute_command_twist().

    class KinematicTaskControllerBase {
    public:
        explicit KinematicTaskControllerBase(const Model &model) noexcept;
        virtual ~KinematicTaskControllerBase() = default;

        auto update(const Model &model, Data &data, const TaskControllerContext &ctx,
                    JointVelocity &out) noexcept -> ControllerStatus;

        void attach_collision(const CollisionModel &col_model, CollisionData &col_data) noexcept;
        void detach_collision() noexcept;

        // --- Public Configuration ---

        bool constraint_aware = false;
        IKOptions ik_options;
        OptimalIKOptions optimal_ik_options;

    protected:
        virtual auto compute_command_twist(const Model &model, Data &data, KinematicsCache &kin,
                                           const TaskControllerContext &ctx,
                                           manifold::SE3::Twist &cmd_twist) noexcept -> bool = 0;

        const CollisionModel *col_model_ = nullptr;
        CollisionData *col_data_ = nullptr;
    };

    // --- Abstract Base: DynamicTaskControllerBase ---
    //
    // SE(3)-tracking torque-mode controllers.
    //
    // Refresh policy:
    //   - kinematics : always (compute_jacobians is unconditional).
    //   - dynamics   : gated by refresh_dynamics (default false). When
    //                  enabled, the base populates data.M and data.h up
    //                  front; data.g is populated up front iff the user's
    //                  bias_compensation policy is GravityOnly.
    //
    // Public config:
    //   - refresh_dynamics   (default false): see above.
    //   - bias_compensation  (default None): see `BiasCompensation` enum for semantics.
    //   - constraint_aware   (default false): apply ASIF to the resulting torque;
    //                                         requires attach_collision.
    //   - asif_options.
    //
    // Pipeline (after hook):
    //   tau_ctrl = J_b^T * cmd_wrench
    //   switch (bias_compensation):
    //     None        -> tau_des = tau_ctrl
    //     GravityOnly -> tau_des = tau_ctrl + data.g
    //     Full        -> tau_des = tau_ctrl + data.h
    //   out.tau  = constraint_aware ? asif_filter(tau_des) : tau_des
    //
    // Note: derived classes should NOT redefine update(); override only the
    // protected hook compute_command_wrench().

    class DynamicTaskControllerBase {
    public:
        explicit DynamicTaskControllerBase(const Model &model);
        virtual ~DynamicTaskControllerBase() = default;

        auto update(const Model &model, Data &data, const TaskControllerContext &ctx,
                    JointTorque &out) noexcept -> ControllerStatus;

        void attach_collision(const CollisionModel &col_model, CollisionData &col_data) noexcept;
        void detach_collision() noexcept;

        // --- Public Configuration ---

        BiasCompensation bias_compensation = BiasCompensation::None;
        bool refresh_dynamics = false;
        bool constraint_aware = false;
        ASIFOptions asif_options;

    protected:
        virtual auto compute_command_wrench(const Model &model, Data &data, KinematicsCache &kin,
                                            DynamicsCache &dyn, const TaskControllerContext &ctx,
                                            manifold::SE3::Wrench &cmd_wrench) noexcept -> bool = 0;

        const CollisionModel *col_model_ = nullptr;
        CollisionData *col_data_ = nullptr;

        // Pre-allocated joint-sized scratch (sized in the constructor).
        // tau_ctrl_ : J_b^T * cmd_wrench (control torque from the hook, projected to joint space).
        // tau_des_  : tau_ctrl_ plus the bias-compensation term selected by `bias_compensation`.
        // tau_safe_ : ASIF-certified torque (only used when constraint_aware).
        Eigen::VectorXd tau_ctrl_;
        Eigen::VectorXd tau_des_;
        Eigen::VectorXd tau_safe_;
    };

    // --- Abstract Base: KinematicJointControllerBase ---
    //
    // Joint-space velocity-mode controllers.
    //
    // Refresh policy:
    //   - kinematics : gated by refresh_kinematics (default false).
    //   - dynamics   : never.
    //
    // Public config:
    //   - constraint_aware    (default false): direction-preserving rescale
    //                         so |v_i| <= model.limits[i].q_vel_max.
    //   - refresh_kinematics  (default false): call compute_jacobians before
    //                         the hook (some hooks read ee-side info).
    //
    // Pipeline (after hook):
    //   out.v = constraint_aware ? rescale(v_ctrl) : v_ctrl

    class KinematicJointControllerBase {
    public:
        explicit KinematicJointControllerBase(const Model &model);
        virtual ~KinematicJointControllerBase() = default;

        auto update(const Model &model, Data &data, const JointControllerContext &ctx,
                    JointVelocity &out) noexcept -> ControllerStatus;

        // --- Public Configuration ---
        bool constraint_aware = false;
        bool refresh_kinematics = false;  // call compute_jacobians before the hook

    protected:
        virtual auto compute_command_velocity(const Model &model, Data &data, KinematicsCache &kin,
                                              const JointControllerContext &ctx,
                                              JointVelocity &v_ctrl) noexcept -> bool = 0;

        // Pre-allocated joint-sized scratch (sized in the constructor).
        // v_ctrl_ : the joint velocity produced by the user's control law.
        JointVelocity v_ctrl_;
    };

    // --- Abstract Base: DynamicJointControllerBase ---
    //
    // Joint-space torque-mode controllers.
    //
    // Refresh policy:
    //   - kinematics : gated by refresh_kinematics (default false).
    //   - dynamics   : gated by refresh_dynamics  (default false). When
    //                  enabled, the base populates data.M and data.h up
    //                  front; data.g is populated up front iff the user's
    //                  bias_compensation policy is GravityOnly.
    //
    // Public config:
    //   - refresh_kinematics (default false).
    //   - refresh_dynamics   (default false).
    //   - bias_compensation  (default None): see `BiasCompensation` enum for semantics.
    //   - constraint_aware   (default false): apply ASIF; requires attach_collision.
    //   - asif_options.
    //
    // Pipeline (after hook):
    //   switch (bias_compensation):
    //     None        -> tau_des = tau_ctrl
    //     GravityOnly -> tau_des = tau_ctrl + data.g
    //     Full        -> tau_des = tau_ctrl + data.h
    //   out.tau = constraint_aware ? asif_filter(tau_des) : tau_des

    class DynamicJointControllerBase {
    public:
        explicit DynamicJointControllerBase(const Model &model);
        virtual ~DynamicJointControllerBase() = default;

        auto update(const Model &model, Data &data, const JointControllerContext &ctx,
                    JointTorque &out) noexcept -> ControllerStatus;

        void attach_collision(const CollisionModel &col_model, CollisionData &col_data) noexcept;
        void detach_collision() noexcept;

        // --- Public Configuration ---
        BiasCompensation bias_compensation = BiasCompensation::None;
        bool refresh_kinematics = false;
        bool refresh_dynamics = false;
        bool constraint_aware = false;
        ASIFOptions asif_options;

    protected:
        virtual auto compute_command_torque(const Model &model, Data &data, KinematicsCache &kin,
                                            DynamicsCache &dyn, const JointControllerContext &ctx,
                                            JointTorque &tau_ctrl) noexcept -> bool = 0;

        const CollisionModel *col_model_ = nullptr;
        CollisionData *col_data_ = nullptr;

        // Pre-allocated joint-sized scratch (sized in the constructor).
        // tau_ctrl_ : the joint torque produced by the user's control law.
        // tau_des_  : tau_ctrl_ plus the bias-compensation term selected by `bias_compensation`.
        // tau_safe_ : ASIF-certified torque (only used when constraint_aware).
        JointTorque tau_ctrl_;
        Eigen::VectorXd tau_des_;
        Eigen::VectorXd tau_safe_;
    };

    // --- Published Concepts ---
    //
    // Recommended usage: write controllers as subclasses; use these concepts
    // when authoring generic library code that takes "any controller of category X".
    //
    // Scope of the check: these concepts verify the *signature* of update()
    // (and the inheritance shortcut), not behavioural correctness. Satisfying the
    // concept is necessary but not sufficient; the doc-blocks are the source of truth.

    template <typename T>
    concept KinematicTaskController =
        std::derived_from<T, KinematicTaskControllerBase> ||
        requires(T &c, const Model &m, Data &d, const TaskControllerContext &ctx,
                 JointVelocity &out) {
            { c.update(m, d, ctx, out) } noexcept -> std::same_as<ControllerStatus>;
        };

    template <typename T>
    concept DynamicTaskController = std::derived_from<T, DynamicTaskControllerBase> ||
                                    requires(T &c, const Model &m, Data &d,
                                             const TaskControllerContext &ctx, JointTorque &out) {
                                        {
                                            c.update(m, d, ctx, out)
                                        } noexcept -> std::same_as<ControllerStatus>;
                                    };

    template <typename T>
    concept KinematicJointController =
        std::derived_from<T, KinematicJointControllerBase> ||
        requires(T &c, const Model &m, Data &d, const JointControllerContext &ctx,
                 JointVelocity &out) {
            { c.update(m, d, ctx, out) } noexcept -> std::same_as<ControllerStatus>;
        };

    template <typename T>
    concept DynamicJointController = std::derived_from<T, DynamicJointControllerBase> ||
                                     requires(T &c, const Model &m, Data &d,
                                              const JointControllerContext &ctx, JointTorque &out) {
                                         {
                                             c.update(m, d, ctx, out)
                                         } noexcept -> std::same_as<ControllerStatus>;
                                     };

    // --- Optional Capability Concepts ---
    //
    // Opt-in concepts that controller authors can satisfy to advertise
    // additional capabilities to harness / runtime code. None of these are
    // required by the four base classes; they are pure capability tags.

    // ResettableController : the controller carries internal state (e.g. an integrator)
    //                        that must be zeroed between distinct trajectories.
    template <typename T>
    concept ResettableController = requires(T &c) {
        { c.reset() } noexcept -> std::same_as<void>;
    };

}  // namespace xarm_geo
