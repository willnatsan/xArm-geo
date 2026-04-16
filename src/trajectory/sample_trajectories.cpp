#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>

#include <xarm_geo/trajectory/sample_trajectories.h>

namespace xarm_geo::trajectories {

    // --- Task Space Trajectories ---

    auto FigureEight::evaluate(double t, TaskSpaceTarget &target) const -> TrajectoryStatus {
        auto compute = [this](double time) -> manifold::SE3 {
            Eigen::Vector3d local_pos(size_x * std::sin(omega * time),
                                      size_y * std::sin(2.0 * omega * time),
                                      size_z * std::sin(omega * time));

            Eigen::Quaterniond local_rot =
                Eigen::AngleAxisd(0.8 * std::sin(omega * time), Eigen::Vector3d::UnitZ()) *
                Eigen::AngleAxisd(std::numbers::pi + (0.3 * std::cos(omega * time)),
                                  Eigen::Vector3d::UnitX()) *
                Eigen::AngleAxisd(0.5 * std::sin(2.0 * omega * time), Eigen::Vector3d::UnitY());

            return anchor * manifold::SE3(manifold::SO3(local_rot), local_pos);
        };

        target.pose = compute(t);

        double dt = 0.001;
        manifold::SE3 pose_next = compute(t + dt);
        target.twist = (target.pose.inverse() * pose_next).log() / dt;

        return TrajectoryStatus::OK;
    }

    auto WingInspection::evaluate(double t, TaskSpaceTarget &target) const -> TrajectoryStatus {
        auto compute = [this](double time) -> manifold::SE3 {
            double x_wing = sweep_amp * std::sin(omega_sweep * time);
            double y_wing = scan_amp * std::cos(omega_scan * time);
            double z_wing = -curvature * (x_wing * x_wing);

            Eigen::Vector3d local_pos(x_wing, y_wing, z_wing);

            Eigen::Quaterniond local_rot =
                Eigen::AngleAxisd(0.0, Eigen::Vector3d::UnitZ()) *
                Eigen::AngleAxisd(std::atan(2.0 * curvature * x_wing), Eigen::Vector3d::UnitY()) *
                Eigen::AngleAxisd(std::numbers::pi, Eigen::Vector3d::UnitX());

            return anchor * manifold::SE3(manifold::SO3(local_rot), local_pos);
        };

        target.pose = compute(t);

        double dt = 0.001;
        manifold::SE3 pose_next = compute(t + dt);
        target.twist = (target.pose.inverse() * pose_next).log() / dt;

        return TrajectoryStatus::OK;
    }

    auto InnerCavityScan::evaluate(double t, TaskSpaceTarget &target) const -> TrajectoryStatus {
        auto compute = [this](double time) -> manifold::SE3 {
            Eigen::Vector3d local_pos(pos_amp * std::sin(omega * time), 0.0, 0.0);

            double roll_target = 1.5 * std::sin(1.1 * omega * time);
            double pitch_target = std::numbers::pi - (1.2 * std::sin(0.9 * omega * time));
            double yaw_target = 2.0 * std::cos(1.3 * omega * time);

            Eigen::Quaterniond local_rot =
                Eigen::AngleAxisd(yaw_target, Eigen::Vector3d::UnitZ()) *
                Eigen::AngleAxisd(pitch_target, Eigen::Vector3d::UnitY()) *
                Eigen::AngleAxisd(roll_target, Eigen::Vector3d::UnitX());

            return anchor * manifold::SE3(manifold::SO3(local_rot), local_pos);
        };

        target.pose = compute(t);

        // Numerical Differentiation for Feedforward Twist
        double dt = 0.001;
        manifold::SE3 pose_next = compute(t + dt);
        target.twist = (target.pose.inverse() * pose_next).log() / dt;

        return TrajectoryStatus::OK;
    }

    auto PipeInspection::evaluate(double t, TaskSpaceTarget &target) const -> TrajectoryStatus {
        auto compute = [this](double time) -> manifold::SE3 {
            double y_pipe = radius * std::sin(omega * time);
            double z_pipe = radius * (1.0 - std::cos(omega * time));

            Eigen::Vector3d local_pos(0.0, y_pipe, -z_pipe);

            double pitch_target = std::numbers::pi - (omega * time);

            Eigen::Quaterniond local_rot =
                Eigen::AngleAxisd(0.0, Eigen::Vector3d::UnitZ()) *
                Eigen::AngleAxisd(pitch_target, Eigen::Vector3d::UnitY()) *
                Eigen::AngleAxisd(0.0, Eigen::Vector3d::UnitX());

            return anchor * manifold::SE3(manifold::SO3(local_rot), local_pos);
        };

        target.pose = compute(t);

        // Numerical Differentiation for Feedforward Twist
        double dt = 0.001;
        manifold::SE3 pose_next = compute(t + dt);
        target.twist = (target.pose.inverse() * pose_next).log() / dt;

        return TrajectoryStatus::OK;
    }

    auto TiltingCircle::evaluate(double t, TaskSpaceTarget &target) const -> TrajectoryStatus {
        auto compute = [this](double time) -> manifold::SE3 {
            double tilt = 0.0;
            if (time > transition_start) {
                double progress = (time - transition_start) / transition_duration;
                tilt = std::min(progress, 1.0) * (std::numbers::pi / 2.0);
            }

            Eigen::Vector3d p_base(R * std::cos(omega * time), R * std::sin(omega * time), 0.0);
            Eigen::Matrix3d R_tilt =
                Eigen::AngleAxisd(tilt, Eigen::Vector3d::UnitX()).toRotationMatrix();
            Eigen::Vector3d local_pos = R_tilt * p_base;

            Eigen::Quaterniond local_rot =
                Eigen::AngleAxisd(0.0, Eigen::Vector3d::UnitZ()) *
                Eigen::AngleAxisd(0.0, Eigen::Vector3d::UnitY()) *
                Eigen::AngleAxisd(std::numbers::pi - tilt, Eigen::Vector3d::UnitX());

            return anchor * manifold::SE3(manifold::SO3(local_rot), local_pos);
        };

        target.pose = compute(t);

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

        // Normalise for Shortest Path [-pi, pi]
        for (auto &val : this->delta_q_) { val = std::remainder(val, 2.0 * std::numbers::pi); }
    }

    auto JointPTP::evaluate(double t, JointSpaceTarget &target) const -> TrajectoryStatus {
        if (target.q.size() != this->q_start_.size() || target.v.size() != this->q_start_.size()) {
            return TrajectoryStatus::ERROR;
        }

        if (this->duration_ <= 0.0) {
            target.q = this->q_start_ + this->delta_q_;
            target.v.setZero();
            return TrajectoryStatus::OK;
        }

        t = std::clamp(t, 0.0, this->duration_);

        // Minimum Jerk Polynomial Formulation
        double tau = t / this->duration_;
        double tau2 = tau * tau;
        double tau3 = tau2 * tau;
        double tau4 = tau3 * tau;
        double tau5 = tau4 * tau;

        double s = (10.0 * tau3) - (15.0 * tau4) + (6.0 * tau5);
        double ds_dt = (30.0 * tau2 - 60.0 * tau3 + 30.0 * tau4) / this->duration_;

        target.q = this->q_start_ + (s * this->delta_q_);
        target.v = ds_dt * this->delta_q_;

        return TrajectoryStatus::OK;
    }

}  // namespace xarm_geo::trajectories
