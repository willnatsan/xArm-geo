#pragma once

#include <concepts>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include <Eigen/Dense>
#include <unsupported/Eigen/Splines>

#include <xarm_geo/core/manifold.h>
#include <xarm_geo/modelling/collision.h>
#include <xarm_geo/modelling/kinematics.h>

namespace xarm_geo {

    // --- Trajectory Status ---
    //
    // Returned by every evaluate() call.
    //   OK              : evaluation succeeded; target has been filled.
    //   OUT_OF_DOMAIN   : t < 0 or t > duration(); target is unchanged.
    //   NOT_INITIALISED : trajectory object is not ready (e.g. build_spline()
    //                     was never called); target is unchanged.
    //   SOLVER_ERROR    : an internal solver (IK, spline, ...) failed.
    //   ERROR           : catch-all for errors not covered above.

    enum class TrajectoryStatus : std::uint8_t {
        OK,
        OUT_OF_DOMAIN,
        NOT_INITIALISED,
        SOLVER_ERROR,
        ERROR,
    };

    [[nodiscard]] constexpr auto to_string(TrajectoryStatus s) noexcept -> const char * {
        switch (s) {
        case TrajectoryStatus::OK:
            return "OK";
        case TrajectoryStatus::OUT_OF_DOMAIN:
            return "OUT_OF_DOMAIN";
        case TrajectoryStatus::NOT_INITIALISED:
            return "NOT_INITIALISED";
        case TrajectoryStatus::SOLVER_ERROR:
            return "SOLVER_ERROR";
        case TrajectoryStatus::ERROR:
            return "ERROR";
        }
        return "UNKNOWN";
    }

    // --- Task-Space Trajectory ---
    //
    // All spatial quantities (twist, spatial_acc) are expressed in the
    // end-effector body frame.

    struct TaskTarget {
        manifold::SE3 pose;
        manifold::SE3::Twist twist;
        manifold::SE3::SpatialAcceleration spatial_acc;
    };

    // A type T satisfies TaskTrajectory if it provides:
    //
    //   evaluate(double t, TaskTarget &) const -> TrajectoryStatus
    //       Fill `target` with the reference pose, body twist, and body
    //       spatial acceleration at time t.
    //
    //   duration() const -> (convertible to double)
    //       Return the total duration of the trajectory in seconds.
    //       Infinite-duration / setpoint trajectories should return
    //       std::numeric_limits<double>::infinity().

    template <typename T>
    concept TaskTrajectory = requires(const T &traj, double t, TaskTarget &target) {
        { traj.evaluate(t, target) } -> std::same_as<TrajectoryStatus>;
        { traj.duration() } -> std::convertible_to<double>;
    };

    // --- Joint-Space Trajectory ---

    struct JointTarget {
        Eigen::VectorXd q;
        Eigen::VectorXd v;
        Eigen::VectorXd a;
        explicit JointTarget(int dof)
            : q(Eigen::VectorXd::Zero(dof)), v(Eigen::VectorXd::Zero(dof)),
              a(Eigen::VectorXd::Zero(dof)) {}
    };

    // A type T satisfies JointTrajectory if it provides:
    //
    //   evaluate(double t, JointTarget &) const -> TrajectoryStatus
    //       Fill `target` with q, v, a at time t.
    //
    //   duration() const -> (convertible to double)
    //       Total duration in seconds (infinity for setpoints).
    //
    //   dof() const -> (convertible to int)
    //       Degrees of freedom; callers use this to size JointTarget before
    //       the first evaluate() call.

    template <typename T>
    concept JointTrajectory = requires(const T &traj, double t, JointTarget &target) {
        { traj.evaluate(t, target) } -> std::same_as<TrajectoryStatus>;
        { traj.duration() } -> std::convertible_to<double>;
        { traj.dof() } -> std::convertible_to<int>;
    };

    // --- Setpoint Trajectory Adapters ---
    //
    // Time-invariant constant-reference trajectories that satisfy the
    // TaskTrajectory / JointTrajectory concepts. duration() returns +infinity
    // because a setpoint has no natural time horizon.

    struct TaskSetpointTrajectory {
        manifold::SE3 pose;

        [[nodiscard]] auto evaluate(double /*t*/, TaskTarget &out) const noexcept
            -> TrajectoryStatus {
            out.pose = pose;
            out.twist = manifold::SE3::Twist::Zero();
            out.spatial_acc = manifold::SE3::SpatialAcceleration::Zero();
            return TrajectoryStatus::OK;
        }

        [[nodiscard]] static auto duration() noexcept -> double {
            return std::numeric_limits<double>::infinity();
        }
    };

    struct JointSetpointTrajectory {
        Eigen::VectorXd q;

        explicit JointSetpointTrajectory(int dof) : q(Eigen::VectorXd::Zero(dof)) {}

        [[nodiscard]] auto evaluate(double /*t*/, JointTarget &out) const noexcept
            -> TrajectoryStatus {
            out.q = q;
            out.v.setZero();
            out.a.setZero();
            return TrajectoryStatus::OK;
        }

        [[nodiscard]] static auto duration() noexcept -> double {
            return std::numeric_limits<double>::infinity();
        }

        [[nodiscard]] auto dof() const noexcept -> int { return static_cast<int>(q.size()); }
    };

    static_assert(TaskTrajectory<TaskSetpointTrajectory>);
    static_assert(JointTrajectory<JointSetpointTrajectory>);

    // --- AnalyticTaskTrajectory ---
    //
    // Abstract base class for task-space trajectories defined by a closed-form
    // analytic curve. Derived classes supply the curve geometry via the pure
    // virtual sample() method; this base handles spline fitting, storage, and
    // evaluation.
    //
    // Usage pattern (mirrors example controllers):
    //
    //   class MyTrajectory final : public AnalyticTaskTrajectory {
    //   public:
    //       MyTrajectory(const manifold::SE3 &anchor, double duration, double my_param)
    //           : AnalyticTaskTrajectory(anchor, duration), my_param_(my_param) {
    //           build_spline();   // must be called after derived members are initialised
    //       }
    //
    //   protected:
    //       auto sample(double t) const
    //           -> std::pair<manifold::SO3, Eigen::Vector3d> override {
    //           // Return (local_rot, local_pos) relative to anchor at time t.
    //       }
    //
    //   private:
    //       double my_param_;
    //   };
    //
    // evaluate() returns:
    //   NOT_INITIALISED  if build_spline() was never called.
    //   OUT_OF_DOMAIN    if t < 0 or t > duration().
    //   OK               on success (target.pose, .twist, .spatial_acc filled).
    //
    // Note: virtual dispatch in sample() occurs only during build_spline()
    // (once at construction, over num_samples points). The hot-path evaluate()
    // reads from the pre-built spline directly — no virtual call per tick.

    class AnalyticTaskTrajectory {
    public:
        AnalyticTaskTrajectory(manifold::SE3 anchor, double duration, int num_samples = 100);
        virtual ~AnalyticTaskTrajectory() = default;

        // Non-copyable due to the spline (large internal state); movable.
        AnalyticTaskTrajectory(const AnalyticTaskTrajectory &) = delete;
        AnalyticTaskTrajectory &operator=(const AnalyticTaskTrajectory &) = delete;
        AnalyticTaskTrajectory(AnalyticTaskTrajectory &&) = default;
        AnalyticTaskTrajectory &operator=(AnalyticTaskTrajectory &&) = default;

        [[nodiscard]] auto evaluate(double t, TaskTarget &target) const -> TrajectoryStatus;
        [[nodiscard]] auto duration() const noexcept -> double;

    protected:
        // Override to define the trajectory geometry.
        // Returns (local_rot, local_pos) expressed relative to the anchor.
        // The base composes: anchor * SE3(local_rot, local_pos).
        // Called num_samples times during build_spline(); not called at runtime.
        [[nodiscard]] virtual auto sample(double t) const
            -> std::pair<manifold::SO3, Eigen::Vector3d> = 0;

        // Sample the curve and fit the degree-5 B-spline.
        // Must be called exactly once from the derived constructor, after all
        // derived members that sample() depends on have been initialised.
        void build_spline();

    private:
        manifold::SE3 anchor_;
        double duration_;
        int num_samples_;
        manifold::SE3::Spline<5> spline_;
        bool initialised_ = false;
    };

    // --- AnalyticJointTrajectory ---
    //
    // Abstract base class for joint-space trajectories defined by a closed-form
    // analytic curve. Mirror of AnalyticTaskTrajectory for joint space.
    //
    // Derived classes supply joint positions via the pure virtual sample()
    // method. The base fits a degree-5 Eigen spline through the sampled
    // configurations and provides analytic velocity (v = dq/dt) and
    // acceleration (a = d²q/dt²) via chain-rule scaling of the spline
    // derivatives.
    //
    // Usage pattern:
    //
    //   class MyJointTrajectory final : public AnalyticJointTrajectory {
    //   public:
    //       MyJointTrajectory(int dof, double duration, double my_param)
    //           : AnalyticJointTrajectory(dof, duration), my_param_(my_param) {
    //           build_spline();   // must be called after derived members are initialised
    //       }
    //
    //   protected:
    //       auto sample(double t) const -> Eigen::VectorXd override {
    //           // Return q (size == dof()) at time t.
    //       }
    //
    //   private:
    //       double my_param_;
    //   };
    //
    // evaluate() returns:
    //   NOT_INITIALISED  if build_spline() was never called.
    //   OUT_OF_DOMAIN    if t < 0 or t > duration().
    //   OK               on success (target.q, .v, .a filled).

    class AnalyticJointTrajectory {
    public:
        AnalyticJointTrajectory(int dof, double duration, int num_samples = 100);
        virtual ~AnalyticJointTrajectory() = default;

        // Non-copyable; movable.
        AnalyticJointTrajectory(const AnalyticJointTrajectory &) = delete;
        AnalyticJointTrajectory &operator=(const AnalyticJointTrajectory &) = delete;
        AnalyticJointTrajectory(AnalyticJointTrajectory &&) = default;
        AnalyticJointTrajectory &operator=(AnalyticJointTrajectory &&) = default;

        [[nodiscard]] auto evaluate(double t, JointTarget &target) const -> TrajectoryStatus;
        [[nodiscard]] auto duration() const noexcept -> double;
        [[nodiscard]] auto dof() const noexcept -> int;

    protected:
        // Override to define the joint configuration at time t.
        // Returns a VectorXd of size dof(). Called num_samples times during
        // build_spline(); not called at runtime.
        [[nodiscard]] virtual auto sample(double t) const -> Eigen::VectorXd = 0;

        // Sample the curve and fit the degree-5 Eigen spline.
        // Must be called once from the derived constructor, after all derived
        // members that sample() depends on have been initialised.
        void build_spline();

    private:
        int dof_;
        double duration_;
        int num_samples_;
        Eigen::Spline<double, Eigen::Dynamic> spline_;
        bool initialised_ = false;
    };

    // --- B-Spline Construction Helper ---
    //
    // Build a degree-D SE(3) B-spline from a sequence of waypoints, padding
    // the boundaries with D repeated copies of the first / last waypoint to
    // satisfy the knot multiplicity required for degree-D continuity at the
    // endpoints.

    template <std::size_t D = 5>
    [[nodiscard]] inline auto build_se3_spline(const std::vector<manifold::SE3> &waypoints,
                                               double duration) -> manifold::SE3::Spline<D> {
        if (waypoints.size() < 2) {
            throw std::invalid_argument("build_se3_spline: at least 2 waypoints required");
        }

        std::vector<manifold::SE3> padded;
        padded.reserve(waypoints.size() + 2 * D);
        for (std::size_t i = 0; i < D; ++i) { padded.push_back(waypoints.front()); }
        for (const auto &wp : waypoints) { padded.push_back(wp); }
        for (std::size_t i = 0; i < D; ++i) { padded.push_back(waypoints.back()); }

        const double dt = duration / static_cast<double>(waypoints.size() - 1);
        return manifold::SE3::Spline<D>(0.0, dt, padded);
    }

}  // namespace xarm_geo
