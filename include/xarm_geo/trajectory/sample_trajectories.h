#pragma once

#include <vector>

#include <smooth/spline/bspline.hpp>

#include <xarm_geo/trajectory/trajectory.h>

namespace xarm_geo::trajectories {

    // --- Task Space Trajectories ---

    struct FigureEight {
        manifold::SE3 anchor;
        double omega = 1.0;
        double size_x = 0.15;
        double size_y = 0.10;
        double size_z = 0.03;

        explicit FigureEight(const manifold::SE3 &anchor) : anchor(anchor) {}
        [[nodiscard]] auto evaluate(double t, TaskSpaceTarget &target) const -> TrajectoryStatus;
    };

    struct WingInspection {
        manifold::SE3 anchor;
        double omega_sweep = 0.5;
        double omega_scan = 2.0;
        double sweep_amp = 0.20;
        double scan_amp = 0.05;
        double curvature = 2.5;

        explicit WingInspection(const manifold::SE3 &anchor) : anchor(anchor) {}
        [[nodiscard]] auto evaluate(double t, TaskSpaceTarget &target) const -> TrajectoryStatus;
    };

    struct TiltingCircle {
        manifold::SE3 anchor;
        double omega = 2.0;
        double R = 0.12;
        double transition_start = 7.5;
        double transition_duration = 1.0;

        explicit TiltingCircle(const manifold::SE3 &anchor, double duration)
            : anchor(anchor), transition_start(duration / 2) {}
        [[nodiscard]] auto evaluate(double t, TaskSpaceTarget &target) const -> TrajectoryStatus;
    };

    class Waypoint {
    public:
        Waypoint(const std::vector<manifold::SE3> &waypoints, double duration);
        [[nodiscard]] auto evaluate(double t, TaskSpaceTarget &target) const -> TrajectoryStatus;

    private:
        manifold::SE3::Spline<5> spline_;  // Internal B-Spline State
    };

    // --- Joint Space Trajectories ---

    class JointPTP {
    public:
        JointPTP(const Eigen::Ref<const Eigen::VectorXd> &q_start,
                 const Eigen::Ref<const Eigen::VectorXd> &q_end, double duration);
        [[nodiscard]] auto evaluate(double t, JointSpaceTarget &target) const -> TrajectoryStatus;

    private:
        Eigen::VectorXd q_start_;
        Eigen::VectorXd delta_q_;
        double duration_;
    };

    // --- Compile-Time Concept Verifications ---

    static_assert(TaskSpaceTrajectory<FigureEight>);
    static_assert(TaskSpaceTrajectory<WingInspection>);
    static_assert(TaskSpaceTrajectory<TiltingCircle>);
    static_assert(TaskSpaceTrajectory<Waypoint>);
    static_assert(JointSpaceTrajectory<JointPTP>);

}  // namespace xarm_geo::trajectories
