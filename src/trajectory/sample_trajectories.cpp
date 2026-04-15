#include <xarm_geo/trajectory/sample_trajectories.h>

namespace xarm_geo {
    auto FigureEightTrajectory::evaluate(double t) const -> manifold::SE3 {
        Eigen::Vector3d pos(center_x + (size_x * std::sin(omega * t)),
                            center_y + (size_y * std::sin(2.0 * omega * t)),
                            center_z + (size_z * std::sin(omega * t)));

        Eigen::Quaterniond rot =
            Eigen::AngleAxisd(0.8 * std::sin(omega * t), Eigen::Vector3d::UnitZ()) *
            Eigen::AngleAxisd(std::numbers::pi + (0.3 * std::cos(omega * t)),
                              Eigen::Vector3d::UnitY()) *
            Eigen::AngleAxisd(0.5 * std::sin(2.0 * omega * t), Eigen::Vector3d::UnitX());

        return {manifold::SO3(rot), pos};
    }

    auto WingInspectionTrajectory::evaluate(double t) const -> manifold::SE3 {
        double y_wing = sweep_amp * std::sin(omega_sweep * t);
        double x_wing = scan_amp * std::cos(omega_scan * t);
        double z_wing = -curvature * (y_wing * y_wing);

        Eigen::Vector3d pos(center_x + x_wing, center_y + y_wing, center_z + z_wing);

        Eigen::Quaterniond rot =
            Eigen::AngleAxisd(0.0, Eigen::Vector3d::UnitZ()) *
            Eigen::AngleAxisd(std::numbers::pi, Eigen::Vector3d::UnitY()) *
            Eigen::AngleAxisd(std::atan(2.0 * curvature * y_wing), Eigen::Vector3d::UnitX());

        return {manifold::SO3(rot), pos};
    }

    auto TiltingCircleTrajectory::evaluate(double t) const -> manifold::SE3 {
        double tilt = 0.0;
        if (t > transition_start) {
            double progress = (t - transition_start) / transition_duration;
            tilt = std::min(progress, 1.0) * (std::numbers::pi / 2.0);
        }

        Eigen::Vector3d p_base(R * std::cos(omega * t), R * std::sin(omega * t), 0.0);
        Eigen::Matrix3d R_tilt =
            Eigen::AngleAxisd(tilt, Eigen::Vector3d::UnitY()).toRotationMatrix();

        Eigen::Vector3d pos = Eigen::Vector3d(center_x, center_y, center_z) + R_tilt * p_base;

        Eigen::Quaterniond rot =
            Eigen::AngleAxisd(0.0, Eigen::Vector3d::UnitZ()) *
            Eigen::AngleAxisd(std::numbers::pi - tilt, Eigen::Vector3d::UnitY()) *
            Eigen::AngleAxisd(0.0, Eigen::Vector3d::UnitX());

        return {manifold::SO3(rot), pos};
    }

    WaypointTrajectory::WaypointTrajectory(const std::vector<manifold::SE3> &waypoints,
                                           double duration) {
        if (waypoints.size() < 2) {
            throw std::invalid_argument("WaypointTrajectory requires at least 2 waypoints.");
        }

        // Build the Waypoints
        std::vector<manifold::SE3> base_waypoints;
        for (int i = 0; i < 5; ++i) { base_waypoints.push_back(waypoints.front()); }
        for (const auto &waypoint : waypoints) { base_waypoints.push_back(waypoint); }
        for (int i = 0; i < 5; ++i) { base_waypoints.push_back(waypoints.back()); }

        // Calculate Timestep
        double dt = duration / static_cast<double>(waypoints.size() - 1);

        // Build the Spline
        this->spline_ = manifold::SE3::Spline<5>(0.0, dt, base_waypoints);
    }

    auto WaypointTrajectory::evaluate(double t) const -> manifold::SE3 {
        // Query the Continuous B-Spline at time `t`
        return this->spline_(t);
    }
}  // namespace xarm_geo
