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
    // Returned by every evaluate() call. Only OK guarantees that `target`
    // has been written; all other statuses leave it unchanged.

    enum class TrajectoryStatus : std::uint8_t {
        OK,
        OUT_OF_DOMAIN,    // t < 0 or t > duration()
        NOT_INITIALISED,  // e.g. build_spline() was never called
        SOLVER_ERROR,     // internal solver (IK, spline, ...) failed
        ERROR,            // catch-all
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
    // Spatial quantities (twist, spatial_acc) are in the desired end-effector body frame.
    // Setpoint / infinite-duration trajectories should return +infinity from duration().

    struct TaskTarget {
        manifold::SE3 pose;
        manifold::SE3::Twist twist;
        manifold::SE3::SpatialAcceleration spatial_acc;
    };

    template <typename T>
    concept TaskTrajectory = requires(const T &traj, double t, TaskTarget &target) {
        { traj.evaluate(t, target) } -> std::same_as<TrajectoryStatus>;
        { traj.duration() } -> std::convertible_to<double>;
    };

    // --- Joint-Space Trajectory ---
    //
    // dof() is used by callers to size JointTarget before the first evaluate().

    struct JointTarget {
        Eigen::VectorXd q;
        Eigen::VectorXd v;
        Eigen::VectorXd a;
        explicit JointTarget(int dof)
            : q(Eigen::VectorXd::Zero(dof)), v(Eigen::VectorXd::Zero(dof)),
              a(Eigen::VectorXd::Zero(dof)) {}
    };

    template <typename T>
    concept JointTrajectory = requires(const T &traj, double t, JointTarget &target) {
        { traj.evaluate(t, target) } -> std::same_as<TrajectoryStatus>;
        { traj.duration() } -> std::convertible_to<double>;
        { traj.dof() } -> std::convertible_to<int>;
    };

    // --- Setpoint Trajectory Adapters ---
    //
    // Time-invariant constant-reference trajectories satisfying the concepts.
    // duration() is +infinity (no natural time horizon).

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

    // --- Analytic Task Trajectory Base Class ---
    //
    // Abstract base for task-space trajectories defined by a closed-form
    // analytic curve. Derived classes supply the curve geometry via the
    // virtual sample() method; the base fits a degree-5 SE(3) B-spline once
    // at construction and serves evaluate() from it -- no virtual dispatch
    // per tick.
    //
    // See docs/authoring_trajectories.md for the authoring pattern.

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
        // Override to define the trajectory geometry. Returns (local_rot, local_pos)
        // relative to the anchor; the base composes anchor * SE3(local_rot, local_pos).
        // Called only during build_spline(), never at runtime.
        [[nodiscard]] virtual auto sample(double t) const
            -> std::pair<manifold::SO3, Eigen::Vector3d> = 0;

        // Sample the curve and fit the spline. Must be called once from the
        // derived constructor after all members read by sample() are initialised.
        void build_spline();

    private:
        manifold::SE3 anchor_;
        double duration_;
        int num_samples_;
        manifold::SE3::Spline<5> spline_;
        bool initialised_ = false;
    };

    // --- Analytic Joint Trajectory Base Class ---
    //
    // Joint-space mirror of AnalyticTaskTrajectory. Derived classes supply
    // joint positions via sample(); the base fits a degree-5 Eigen spline
    // and provides v = dq/dt and a = d²q/dt² via chain-rule scaling.
    //
    // See docs/authoring_trajectories.md for the authoring pattern.

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
        // Override to return q at time t (size must equal dof()). Called only
        // during build_spline(), never at runtime.
        [[nodiscard]] virtual auto sample(double t) const -> Eigen::VectorXd = 0;

        // Sample the curve and fit the spline. Must be called once from the
        // derived constructor after all members read by sample() are initialised.
        void build_spline();

    private:
        int dof_;
        double duration_;
        int num_samples_;
        Eigen::Spline<double, Eigen::Dynamic> spline_;
        bool initialised_ = false;
    };

    // --- SE(3) B-Spline Construction Helper ---
    //
    // Build a degree-D SE(3) B-spline from waypoints, padding the boundaries
    // with D repeated copies of the first/last waypoint to satisfy endpoint
    // knot multiplicity for degree-D continuity.

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
