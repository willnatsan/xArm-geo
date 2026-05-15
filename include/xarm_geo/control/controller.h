#pragma once

#include <chrono>
#include <concepts>
#include <cstdint>
#include <vector>

#include <Eigen/Dense>

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
// All four base classes follow the same template-method shape: size checks,
// sync data.q from ctx.fb.q, refresh kinematics (eager on task-space bases,
// lazy on joint-space bases), construct KinematicsCache/DynamicsCache,
// invoke the user-supplied hook, run base-specific post-processing, then
// write `out` exactly once.
//
// Inside the hook, read kinematic state through the KinematicsCache (kin.X())
// and dynamic state through the DynamicsCache (dyn.M/h/g()) -- never touch
// the underlying data.* fields directly. The caches guarantee freshness and
// memoise across hook + base usage.
//
// Constraint-aware bases install an opinionated default safety set:
//   - kinematic task base  : optimal_inverse_diff_kinematics convenience overload
//                            (TwistTask + joint/velocity limits + collision barriers).
//   - dynamic bases        : composable asif_filter overload with DynPositionBarrier +
//                            DynVelocityBarrier + DynCollisionBarrier (gains from
//                            asif_defaults); M and h are taken from the DynamicsCache
//                            so they are not recomputed when the hook already used them.
// To use a custom safety set, write a class satisfying the relevant *Controller
// concept directly rather than subclassing -- see
// xarm_geo/examples/controllers/custom_controller.h.

namespace xarm_geo {

    // --- Bias-Force Cancellation Policy ---
    //
    // Applies only to dynamic controllers. Selects the bias term added to the
    // hook's tau_ctrl before optional ASIF certification:
    //   None         : out = tau_ctrl                       (free).
    //   GravityOnly  : out = tau_ctrl + g(q)                (1 RNEA pass).
    //   Full         : out = tau_ctrl + h(q, v),
    //                  where h = C(q,v)*v + g(q)            (1 RNEA pass).

    enum class BiasCompensation : std::uint8_t { None, GravityOnly, Full };

    // --- Controller Status ---
    //
    // Origins:  base  = update() pipeline;  hook  = user-supplied hook;
    //           OptIK = optimal_inverse_diff_kinematics (kinematic-task only);
    //           ASIF  = asif_filter (dynamic bases only).

    enum class ControllerStatus : std::uint8_t {
        OK,              // base / hook
        SIZE_MISMATCH,   // base  -- fb / out vector sizes != model.dof
        NOT_CONFIGURED,  // base  -- constraint_aware set but no collision attached
        HOOK_FAILED,     // hook  -- user-supplied hook returned non-OK
        INFEASIBLE,      // OptIK or ASIF
        MAX_ITERS,       // OptIK or ASIF
        SOLVER_ERROR,    // OptIK or ASIF
    };

    // --- Solver Status Mapping ---
    //
    // Translate an upstream solver status into the corresponding ControllerStatus.

    [[nodiscard]] constexpr auto to_controller_status(OptimalIKStatus s) noexcept
        -> ControllerStatus {
        switch (s) {
        case OptimalIKStatus::OK:
        case OptimalIKStatus::RELAXED:  // slack was non-zero but QP solved; treat as success
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
        case ASIFStatus::RELAXED:  // slack was non-zero but QP solved; treat as success
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
        const TaskTarget &ref;
        std::chrono::nanoseconds dt;
    };

    struct JointControllerContext {
        const JointState &fb;
        const JointTarget &ref;
        std::chrono::nanoseconds dt;
    };

    // --- Hook-Side Kinematics Cache ---
    //
    // Lazy accessor passed into controller hooks. Wraps `data` and computes
    // each kinematic field (Jacobians, EE pose, pose tree) at most once per
    // tick on first access.

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

    // --- Hook-Side Dynamics Cache ---
    //
    // Lazy accessor for the joint-space mass matrix M, full bias forces
    // h = C(q,v)v + g(q), and gravity-only bias g. Each is computed at most
    // once per tick on first access.

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

    // --- Per-Tick Diagnostic Snapshots ---
    //
    // Populated at the end of every successful update() call and retrievable
    // via last_tick_diagnostics(). Callers (e.g. DataLogger fill helpers) read
    // these immediately after update() to assemble a LogSample without any
    // extra allocations.

    // Kinematic bases (velocity triplet).
    struct KinematicTickDiagnostics {
        Eigen::VectorXd v_ctrl;  // raw hook output, before any safety-layer shaping
        Eigen::VectorXd v_des;   // == v_ctrl (no bias-compensation for kinematic bases)
        Eigen::VectorXd v_safe;  // post OptIK / direction-preserving rescale

        // NOTE: KinematicJointControllerBase has no IK layer; `optik_invoked` is
        // always false and `optik_modified` is repurposed to flag direction-preserving
        // velocity-limit rescaling. Conflating these under one schema keeps the Python
        // side uniform -- revisit if a third kinematic safety layer is ever added.
        OptimalIKStatus optik_status = OptimalIKStatus::OK;
        bool optik_invoked =
            false;  // true only for KinematicTaskControllerBase with constraint_aware
        bool optik_modified =
            false;                 // ||v_safe - v_des|| > eps (or rescale clipped for joint base)
        double optik_delta = 0.0;  // slack magnitude from last QP solve (0 when strict)
    };

    // Dynamic bases (torque triplet).
    struct DynamicTickDiagnostics {
        Eigen::VectorXd tau_ctrl;  // raw hook output, before bias compensation
        Eigen::VectorXd tau_des;   // post bias-compensation; intended command without ASIF
        Eigen::VectorXd tau_safe;  // post ASIF certification; == tau_des when ASIF is off

        ASIFStatus asif_status = ASIFStatus::OK;
        bool asif_invoked = false;   // true only when constraint_aware
        bool asif_modified = false;  // ||tau_safe - tau_des|| > eps
        double asif_delta = 0.0;     // slack magnitude from last QP solve (0 when strict)
    };

    // --- Kinematic Task Controller Base Class ---
    //
    // SE(3)-tracking velocity-mode controllers. Eager kinematics refresh; no
    // dynamics. When constraint_aware is set the base routes through
    // optimal_inverse_diff_kinematics and a collision model must be attached.
    //
    // Derived classes override only compute_command_twist().

    class KinematicTaskControllerBase {
    public:
        explicit KinematicTaskControllerBase(const Model &model) noexcept;
        virtual ~KinematicTaskControllerBase() = default;

        auto update(const Model &model, Data &data, const TaskControllerContext &ctx,
                    JointVelocity &out) noexcept -> ControllerStatus;

        void attach_collision(const CollisionModel &col_model, CollisionData &col_data) noexcept;
        void detach_collision() noexcept;

        [[nodiscard]] auto last_tick_diagnostics() const noexcept
            -> const KinematicTickDiagnostics & {
            return diag_;
        }

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

    private:
        KinematicTickDiagnostics diag_;
    };

    // --- Dynamic Task Controller Base Class ---
    //
    // SE(3)-tracking torque-mode controllers. Eager kinematics, lazy dynamics.
    // Post-hook pipeline:
    //   tau = J_b^T * cmd_wrench  +  bias term (per `bias_compensation`)
    //   out.tau = constraint_aware ? asif_filter(tau) : tau
    //
    // Derived classes override only compute_command_wrench().

    class DynamicTaskControllerBase {
    public:
        explicit DynamicTaskControllerBase(const Model &model);
        virtual ~DynamicTaskControllerBase() = default;

        auto update(const Model &model, Data &data, const TaskControllerContext &ctx,
                    JointTorque &out) noexcept -> ControllerStatus;

        void attach_collision(const CollisionModel &col_model, CollisionData &col_data) noexcept;
        void detach_collision() noexcept;

        [[nodiscard]] auto last_tick_diagnostics() const noexcept
            -> const DynamicTickDiagnostics & {
            return diag_;
        }

        // --- Public Configuration ---

        BiasCompensation bias_compensation = BiasCompensation::None;
        bool constraint_aware = false;
        ASIFOptions asif_options;

    protected:
        virtual auto compute_command_wrench(const Model &model, Data &data, KinematicsCache &kin,
                                            DynamicsCache &dyn, const TaskControllerContext &ctx,
                                            manifold::SE3::Wrench &cmd_wrench) noexcept -> bool = 0;

        const CollisionModel *col_model_ = nullptr;
        CollisionData *col_data_ = nullptr;

        // Pre-allocated joint-sized scratch: control torque, bias-compensated
        // torque, ASIF-certified torque.
        Eigen::VectorXd tau_ctrl_;
        Eigen::VectorXd tau_des_;
        Eigen::VectorXd tau_safe_;

    private:
        DynamicTickDiagnostics diag_;
    };

    // --- Kinematic Joint Controller Base Class ---
    //
    // Joint-space velocity-mode controllers. Lazy kinematics; no dynamics.
    // When constraint_aware is set, applies a direction-preserving rescale
    // so |v_i| <= model.limits[i].q_vel_max.
    //
    // Derived classes override only compute_command_velocity().

    class KinematicJointControllerBase {
    public:
        explicit KinematicJointControllerBase(const Model &model);
        virtual ~KinematicJointControllerBase() = default;

        auto update(const Model &model, Data &data, const JointControllerContext &ctx,
                    JointVelocity &out) noexcept -> ControllerStatus;

        [[nodiscard]] auto last_tick_diagnostics() const noexcept
            -> const KinematicTickDiagnostics & {
            return diag_;
        }

        // --- Public Configuration ---
        bool constraint_aware = false;

    protected:
        virtual auto compute_command_velocity(const Model &model, Data &data, KinematicsCache &kin,
                                              const JointControllerContext &ctx,
                                              JointVelocity &v_ctrl) noexcept -> bool = 0;

        // Joint-sized scratch for the hook's output velocity.
        JointVelocity v_ctrl_;

    private:
        KinematicTickDiagnostics diag_;
    };

    // --- Dynamic Joint Controller Base Class ---
    //
    // Joint-space torque-mode controllers. Lazy kinematics and dynamics.
    // Post-hook pipeline:
    //   tau = tau_ctrl + bias term (per `bias_compensation`)
    //   out.tau = constraint_aware ? asif_filter(tau) : tau
    //
    // Derived classes override only compute_command_torque().

    class DynamicJointControllerBase {
    public:
        explicit DynamicJointControllerBase(const Model &model);
        virtual ~DynamicJointControllerBase() = default;

        auto update(const Model &model, Data &data, const JointControllerContext &ctx,
                    JointTorque &out) noexcept -> ControllerStatus;

        void attach_collision(const CollisionModel &col_model, CollisionData &col_data) noexcept;
        void detach_collision() noexcept;

        [[nodiscard]] auto last_tick_diagnostics() const noexcept
            -> const DynamicTickDiagnostics & {
            return diag_;
        }

        // --- Public Configuration ---
        BiasCompensation bias_compensation = BiasCompensation::None;
        bool constraint_aware = false;
        ASIFOptions asif_options;

    protected:
        virtual auto compute_command_torque(const Model &model, Data &data, KinematicsCache &kin,
                                            DynamicsCache &dyn, const JointControllerContext &ctx,
                                            JointTorque &tau_ctrl) noexcept -> bool = 0;

        const CollisionModel *col_model_ = nullptr;
        CollisionData *col_data_ = nullptr;

        // Pre-allocated joint-sized scratch: control torque, bias-compensated
        // torque, ASIF-certified torque.
        JointTorque tau_ctrl_;
        Eigen::VectorXd tau_des_;
        Eigen::VectorXd tau_safe_;

    private:
        DynamicTickDiagnostics diag_;
    };

    // --- Published Controller Concepts ---
    //
    // Concepts for generic code that takes "any controller of category X".
    // They check the signature of update() (and the inheritance shortcut),
    // not behavioural correctness.

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
    // Pure capability tags. Controllers opt in to advertise extra features:
    //   ResettableController  : exposes reset() to zero internal state
    //                           (e.g. integrators) between trajectories.
    //   ConvergenceObservable : exposes converged() reporting whether the
    //                           error has been below threshold for the last
    //                           min_consecutive ticks.

    template <typename T>
    concept ResettableController = requires(T &c) {
        { c.reset() } noexcept -> std::same_as<void>;
    };

    template <typename T>
    concept ConvergenceObservable = requires(const T &c) {
        { c.converged() } noexcept -> std::same_as<bool>;
    };

}  // namespace xarm_geo
