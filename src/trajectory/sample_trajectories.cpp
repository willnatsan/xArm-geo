#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>

#include <xarm_geo/trajectory/sample_trajectories.h>

namespace xarm_geo::trajectories {

    // --- Task Space Trajectories ---

    auto FigureEight::evaluate(double t, TaskSpaceTarget &target) const -> TrajectoryStatus {
        // Local Lambda to Compute Pose at any given Time
        auto compute = [this](double time) -> manifold::SE3 {
            Eigen::Vector3d pos(center_x + (size_x * std::sin(omega * time)),
                                center_y + (size_y * std::sin(2.0 * omega * time)),
                                center_z + (size_z * std::sin(omega * time)));

            Eigen::Quaterniond rot =
                Eigen::AngleAxisd(0.8 * std::sin(omega * time), Eigen::Vector3d::UnitZ()) *
                Eigen::AngleAxisd(std::numbers::pi + (0.3 * std::cos(omega * time)),
                                  Eigen::Vector3d::UnitY()) *
                Eigen::AngleAxisd(0.5 * std::sin(2.0 * omega * time), Eigen::Vector3d::UnitX());

            return {manifold::SO3(rot), pos};
        };

        target.pose = compute(t);

        // Numerical Differentiation for Feedforward Twist
        double dt = 0.001;
        manifold::SE3 pose_next = compute(t + dt);
        target.twist = (target.pose.inverse() * pose_next).log() / dt;

        return TrajectoryStatus::OK;
    }

    auto WingInspection::evaluate(double t, TaskSpaceTarget &target) const -> TrajectoryStatus {
        // Local Lambda to Compute Pose at any given Time
        auto compute = [this](double time) -> manifold::SE3 {
            double y_wing = sweep_amp * std::sin(omega_sweep * time);
            double x_wing = scan_amp * std::cos(omega_scan * time);
            double z_wing = -curvature * (y_wing * y_wing);

            Eigen::Vector3d pos(center_x + x_wing, center_y + y_wing, center_z + z_wing);

            Eigen::Quaterniond rot =
                Eigen::AngleAxisd(0.0, Eigen::Vector3d::UnitZ()) *
                Eigen::AngleAxisd(std::numbers::pi, Eigen::Vector3d::UnitY()) *
                Eigen::AngleAxisd(std::atan(2.0 * curvature * y_wing), Eigen::Vector3d::UnitX());

            return {manifold::SO3(rot), pos};
        };

        target.pose = compute(t);

        // Numerical Differentiation for Feedforward Twist
        double dt = 0.001;
        manifold::SE3 pose_next = compute(t + dt);
        target.twist = (target.pose.inverse() * pose_next).log() / dt;

        return TrajectoryStatus::OK;
    }

    auto TiltingCircle::evaluate(double t, TaskSpaceTarget &target) const -> TrajectoryStatus {
        // Local Lambda to Compute Pose at any given Time
        auto compute = [this](double time) -> manifold::SE3 {
            double tilt = 0.0;
            if (time > transition_start) {
                double progress = (time - transition_start) / transition_duration;
                tilt = std::min(progress, 1.0) * (std::numbers::pi / 2.0);
            }

            Eigen::Vector3d p_base(R * std::cos(omega * time), R * std::sin(omega * time), 0.0);
            Eigen::Matrix3d R_tilt =
                Eigen::AngleAxisd(tilt, Eigen::Vector3d::UnitY()).toRotationMatrix();

            Eigen::Vector3d pos = Eigen::Vector3d(center_x, center_y, center_z) + R_tilt * p_base;

            Eigen::Quaterniond rot =
                Eigen::AngleAxisd(0.0, Eigen::Vector3d::UnitZ()) *
                Eigen::AngleAxisd(std::numbers::pi - tilt, Eigen::Vector3d::UnitY()) *
                Eigen::AngleAxisd(0.0, Eigen::Vector3d::UnitX());

            return {manifold::SO3(rot), pos};
        };

        target.pose = compute(t);

        // Numerical Differentiation for Feedforward Twist
        double dt = 0.001;
        manifold::SE3 pose_next = compute(t + dt);
        target.twist = (target.pose.inverse() * pose_next).log() / dt;

        return TrajectoryStatus::OK;
    }

    Waypoint::Waypoint(const std::vector<manifold::SE3> &waypoints, double duration) {
        if (waypoints.size() < 2) {
            throw std::invalid_argument("Waypoint Trajectory Requires at least 2 Waypoints.");
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

    auto Waypoint::evaluate(double t, TaskSpaceTarget &target) const -> TrajectoryStatus {
        // Query the Continuous B-Spline
        target.pose = this->spline_(t);

        // Numerical Differentiation for Feedforward Twist
        double dt = 0.001;
        manifold::SE3 pose_next = this->spline_(t + dt);
        target.twist = (target.pose.inverse() * pose_next).log() / dt;

        return TrajectoryStatus::OK;
    }

    // --- Joint Space Trajectories ---

    JointPTP::JointPTP(const Eigen::Ref<const Eigen::VectorXd> &q_start,
                       const Eigen::Ref<const Eigen::VectorXd> &q_end, double duration)
        : q_start_(q_start), duration_(duration) {

        if (q_start_.size() != q_end.size()) {
            throw std::invalid_argument("Start & End Joint Configurations Size Mismatch!");
        }

        this->delta_q_ = q_end - this->q_start_;
    }

    auto JointPTP::evaluate(double t, JointSpaceTarget &target) const -> TrajectoryStatus {
        if (target.q.size() != this->q_start_.size() || target.v.size() != this->q_start_.size()) {
            return TrajectoryStatus::ERROR;
        }

        if (duration_ <= 0.0) {
            target.q = q_start_ + delta_q_;
            target.v.setZero();
            return TrajectoryStatus::OK;
        }

        t = std::clamp(t, 0.0, duration_);

        // Minimum Jerk Polynomial Formulation
        double tau = t / duration_;
        double tau2 = tau * tau;
        double tau3 = tau2 * tau;
        double tau4 = tau3 * tau;
        double tau5 = tau4 * tau;

        double s = (10.0 * tau3) - (15.0 * tau4) + (6.0 * tau5);
        double ds_dt = (30.0 * tau2 - 60.0 * tau3 + 30.0 * tau4) / duration_;

        target.q = q_start_ + (s * delta_q_);
        target.v = ds_dt * delta_q_;

        return TrajectoryStatus::OK;
    }

}  // namespace xarm_geo::trajectories
