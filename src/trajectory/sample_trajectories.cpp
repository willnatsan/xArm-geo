#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>

#include <xarm_geo/trajectory/sample_trajectories.h>

namespace xarm_geo::trajectories {

    // --- Task Space Trajectories ---

    Waypoint::Waypoint(const std::vector<manifold::SE3> &waypoints, double duration) {
        if (waypoints.size() < 2) {
            throw std::invalid_argument("Waypoint Trajectory Requires at least 2 Waypoints.");
        }

        // Build the Waypoints
        std::vector<manifold::SE3> base_waypoints;
        base_waypoints.reserve(5);
        for (int i = 0; i < 5; ++i) { base_waypoints.push_back(waypoints.front()); }
        for (const auto &waypoint : waypoints) { base_waypoints.push_back(waypoint); }
        for (int i = 0; i < 5; ++i) { base_waypoints.push_back(waypoints.back()); }

        // Calculate Timestep
        double dt = duration / static_cast<double>(waypoints.size() - 1);

        // Build the Spline
        this->spline_ = manifold::SE3::Spline<5>(0.0, dt, base_waypoints);
    }

    auto Waypoint::evaluate(double t, TaskSpaceTarget &target) const -> TrajectoryStatus {
        manifold::SE3::Tangent vel;
        manifold::SE3::Tangent acc;

        // Using Analytical Derivatives from `smooth` to get Body Twist & Body Spatial Acceleration
        target.pose = this->spline_(t, smooth::OptTangent<manifold::SE3>{vel},
                                    smooth::OptTangent<manifold::SE3>{acc});
        target.twist = vel;
        target.spatial_acc = acc;

        return TrajectoryStatus::OK;
    }

    FigureEight::FigureEight(const manifold::SE3 &anchor, double duration, double omega,
                             double size_x, double size_y, double size_z) {

        // Sample the Analytic Trajectory at Regular Intervals to Fit a B-Spline
        constexpr int num_samples = 100;
        std::vector<manifold::SE3> waypoints;
        waypoints.reserve(num_samples);

        for (int i = 0; i < num_samples; ++i) {
            double t = (static_cast<double>(i) / (num_samples - 1)) * duration;

            // Lissajous Figure-Eight Pattern in XY Plane with Sinusoidal Z-Variation
            Eigen::Vector3d local_pos(size_x * std::sin(omega * t),
                                      size_y * std::sin(2.0 * omega * t),
                                      size_z * std::sin(omega * t));

            // Time-Varying ZYX Euler Orientation: Oscillating Yaw/Pitch with
            // Pi-Offset Roll to Point Tool Downward
            double roll = std::numbers::pi + (0.3 * std::cos(omega * t));
            double pitch = 0.5 * std::sin(2.0 * omega * t);
            double yaw = 0.8 * std::sin(omega * t);
            manifold::SO3 local_rot = manifold::rpy_to_SO3(roll, pitch, yaw);

            waypoints.emplace_back(anchor * manifold::SE3(local_rot, local_pos));
        }

        // Pad Boundary with Repeated Endpoints to Satisfy Degree-5 B-Spline Knot Requirements
        std::vector<manifold::SE3> padded_waypoints;
        padded_waypoints.reserve(num_samples + 10);
        for (int i = 0; i < 5; ++i) { padded_waypoints.push_back(waypoints.front()); }
        for (const auto &waypoint : waypoints) { padded_waypoints.push_back(waypoint); }
        for (int i = 0; i < 5; ++i) { padded_waypoints.push_back(waypoints.back()); }

        double dt = duration / static_cast<double>(num_samples - 1);
        this->spline_ = manifold::SE3::Spline<5>(0.0, dt, padded_waypoints);
    }

    auto FigureEight::evaluate(double t, TaskSpaceTarget &target) const -> TrajectoryStatus {
        manifold::SE3::Tangent vel;
        manifold::SE3::Tangent acc;

        // Evaluate B-Spline with Analytical First & Second Derivatives via smooth::OptTangent
        target.pose = this->spline_(t, smooth::OptTangent<manifold::SE3>{vel},
                                    smooth::OptTangent<manifold::SE3>{acc});
        target.twist = vel;
        target.spatial_acc = acc;

        return TrajectoryStatus::OK;
    }

    WingInspection::WingInspection(const manifold::SE3 &anchor, double duration, double omega_sweep,
                                   double omega_scan, double sweep_amp, double scan_amp,
                                   double curvature) {

        constexpr int num_samples = 100;
        std::vector<manifold::SE3> waypoints;
        waypoints.reserve(num_samples);

        for (int i = 0; i < num_samples; ++i) {
            double t = (static_cast<double>(i) / (num_samples - 1)) * duration;

            // Parabolic Wing Surface: X Sweeps Laterally, Y Scans Along Span, Z Follows Curvature
            double x_wing = sweep_amp * std::sin(omega_sweep * t);
            double y_wing = scan_amp * std::cos(omega_scan * t);
            double z_wing = -curvature * (x_wing * x_wing);

            Eigen::Vector3d local_pos(x_wing, y_wing, z_wing);

            // Pitch Aligns Tool Normal to Wing Surface Gradient (atan of dz/dx),
            // Roll of Pi Points Tool Downward into Surface
            manifold::SO3 local_rot =
                manifold::SO3::exp(std::atan(2.0 * curvature * x_wing) * Eigen::Vector3d::UnitY()) *
                manifold::SO3::exp(std::numbers::pi * Eigen::Vector3d::UnitX());

            waypoints.emplace_back(anchor * manifold::SE3(local_rot, local_pos));
        }

        std::vector<manifold::SE3> padded_waypoints;
        padded_waypoints.reserve(num_samples + 10);
        for (int i = 0; i < 5; ++i) { padded_waypoints.push_back(waypoints.front()); }
        for (const auto &waypoint : waypoints) { padded_waypoints.push_back(waypoint); }
        for (int i = 0; i < 5; ++i) { padded_waypoints.push_back(waypoints.back()); }

        double dt = duration / static_cast<double>(num_samples - 1);
        this->spline_ = manifold::SE3::Spline<5>(0.0, dt, padded_waypoints);
    }

    auto WingInspection::evaluate(double t, TaskSpaceTarget &target) const -> TrajectoryStatus {
        manifold::SE3::Tangent vel;
        manifold::SE3::Tangent acc;

        target.pose = this->spline_(t, smooth::OptTangent<manifold::SE3>{vel},
                                    smooth::OptTangent<manifold::SE3>{acc});
        target.twist = vel;
        target.spatial_acc = acc;

        return TrajectoryStatus::OK;
    }

    InnerCavityScan::InnerCavityScan(const manifold::SE3 &anchor, double duration, double omega,
                                     double pos_amp) {

        constexpr int num_samples = 100;
        std::vector<manifold::SE3> waypoints;
        waypoints.reserve(num_samples);

        for (int i = 0; i < num_samples; ++i) {
            double t = (static_cast<double>(i) / (num_samples - 1)) * duration;

            // Small Linear Oscillation Along X-Axis to Probe Cavity Depth
            Eigen::Vector3d local_pos(pos_amp * std::sin(omega * t), 0.0, 0.0);

            // Aggressive Multi-Axis Orientation Sweep with Incommensurate Frequencies
            // to Maximise Viewing Angle Coverage Inside the Cavity
            double roll_target = 1.5 * std::sin(1.1 * omega * t);
            double pitch_target = std::numbers::pi - (1.2 * std::sin(0.9 * omega * t));
            double yaw_target = 2.0 * std::cos(1.3 * omega * t);

            manifold::SO3 local_rot = manifold::rpy_to_SO3(roll_target, pitch_target, yaw_target);

            waypoints.emplace_back(anchor * manifold::SE3(local_rot, local_pos));
        }

        std::vector<manifold::SE3> padded_waypoints;
        padded_waypoints.reserve(num_samples + 10);
        for (int i = 0; i < 5; ++i) { padded_waypoints.push_back(waypoints.front()); }
        for (const auto &waypoint : waypoints) { padded_waypoints.push_back(waypoint); }
        for (int i = 0; i < 5; ++i) { padded_waypoints.push_back(waypoints.back()); }

        double dt = duration / static_cast<double>(num_samples - 1);
        this->spline_ = manifold::SE3::Spline<5>(0.0, dt, padded_waypoints);
    }

    auto InnerCavityScan::evaluate(double t, TaskSpaceTarget &target) const -> TrajectoryStatus {
        manifold::SE3::Tangent vel;
        manifold::SE3::Tangent acc;

        target.pose = this->spline_(t, smooth::OptTangent<manifold::SE3>{vel},
                                    smooth::OptTangent<manifold::SE3>{acc});
        target.twist = vel;
        target.spatial_acc = acc;

        return TrajectoryStatus::OK;
    }

    PipeInspection::PipeInspection(const manifold::SE3 &anchor, double duration, double omega,
                                   double radius) {

        constexpr int num_samples = 100;
        std::vector<manifold::SE3> waypoints;
        waypoints.reserve(num_samples);

        for (int i = 0; i < num_samples; ++i) {
            double t = (static_cast<double>(i) / (num_samples - 1)) * duration;

            // Circular Arc in YZ Plane Simulating Travel Around Pipe Interior
            double y_pipe = radius * std::sin(omega * t);
            double z_pipe = radius * (1.0 - std::cos(omega * t));

            Eigen::Vector3d local_pos(0.0, y_pipe, -z_pipe);

            // Pitch Tracks Arc Angle to Keep Tool Normal to Pipe Wall
            double pitch_target = std::numbers::pi - (omega * t);

            manifold::SO3 local_rot = manifold::SO3::exp(pitch_target * Eigen::Vector3d::UnitY());

            waypoints.emplace_back(anchor * manifold::SE3(local_rot, local_pos));
        }

        std::vector<manifold::SE3> padded_waypoints;
        padded_waypoints.reserve(num_samples + 10);
        for (int i = 0; i < 5; ++i) { padded_waypoints.push_back(waypoints.front()); }
        for (const auto &waypoint : waypoints) { padded_waypoints.push_back(waypoint); }
        for (int i = 0; i < 5; ++i) { padded_waypoints.push_back(waypoints.back()); }

        double dt = duration / static_cast<double>(num_samples - 1);
        this->spline_ = manifold::SE3::Spline<5>(0.0, dt, padded_waypoints);
    }

    auto PipeInspection::evaluate(double t, TaskSpaceTarget &target) const -> TrajectoryStatus {
        manifold::SE3::Tangent vel;
        manifold::SE3::Tangent acc;

        target.pose = this->spline_(t, smooth::OptTangent<manifold::SE3>{vel},
                                    smooth::OptTangent<manifold::SE3>{acc});
        target.twist = vel;
        target.spatial_acc = acc;

        return TrajectoryStatus::OK;
    }

    TiltingCircle::TiltingCircle(const manifold::SE3 &anchor, double duration, double omega,
                                 double R) {

        constexpr int num_samples = 100;
        std::vector<manifold::SE3> waypoints;
        waypoints.reserve(num_samples);

        double transition_start = duration / 2.0;
        double transition_duration = 1.0;

        for (int i = 0; i < num_samples; ++i) {
            double t = (static_cast<double>(i) / (num_samples - 1)) * duration;

            // Min-Jerk Tilt Ramp (C^2 Continuous) to Avoid Acceleration Spikes at Boundaries
            // Polynomial: s(p) = 10p^3 - 15p^4 + 6p^5 with s(0)=0, s(1)=1, s'(0)=s'(1)=0
            double progress = std::clamp((t - transition_start) / transition_duration, 0.0, 1.0);
            double progress2 = progress * progress;
            double progress3 = progress2 * progress;
            double progress4 = progress3 * progress;
            double progress5 = progress4 * progress;
            double s = (10.0 * progress3) - (15.0 * progress4) + (6.0 * progress5);
            double tilt = s * (std::numbers::pi / 2.0);

            // Circular Path in XY Plane (Horizontal)
            Eigen::Vector3d p_base(R * std::cos(omega * t), R * std::sin(omega * t), 0.0);

            // Tilt the Circular Orbit Plane About X-Axis Using SO3 Action on Vector
            // (Smoothly Transitions from Horizontal to Vertical Circle at Midpoint)
            manifold::SO3 tilt_rot = manifold::SO3::exp(tilt * Eigen::Vector3d::UnitX());
            Eigen::Vector3d local_pos = tilt_rot * p_base;

            // End-Effector Points Radially Inward: Pi Flips Tool Down, Tilt Compensates for
            // the Orbit Plane Rotation to Maintain Normal-to-Circle Orientation
            manifold::SO3 local_rot =
                manifold::SO3::exp((std::numbers::pi - tilt) * Eigen::Vector3d::UnitX());

            waypoints.emplace_back(anchor * manifold::SE3(local_rot, local_pos));
        }

        std::vector<manifold::SE3> padded_waypoints;
        padded_waypoints.reserve(num_samples + 10);
        for (int i = 0; i < 5; ++i) { padded_waypoints.push_back(waypoints.front()); }
        for (const auto &waypoint : waypoints) { padded_waypoints.push_back(waypoint); }
        for (int i = 0; i < 5; ++i) { padded_waypoints.push_back(waypoints.back()); }

        double dt = duration / static_cast<double>(num_samples - 1);
        this->spline_ = manifold::SE3::Spline<5>(0.0, dt, padded_waypoints);
    }

    auto TiltingCircle::evaluate(double t, TaskSpaceTarget &target) const -> TrajectoryStatus {
        manifold::SE3::Tangent vel;
        manifold::SE3::Tangent acc;

        target.pose = this->spline_(t, smooth::OptTangent<manifold::SE3>{vel},
                                    smooth::OptTangent<manifold::SE3>{acc});
        target.twist = vel;
        target.spatial_acc = acc;

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
        for (auto &val : this->delta_q_) { val = manifold::wrap_to_pi(val); }
    }

    auto JointPTP::evaluate(double t, JointSpaceTarget &target) const -> TrajectoryStatus {
        if (target.q.size() != this->q_start_.size() || target.v.size() != this->q_start_.size() ||
            target.a.size() != this->q_start_.size()) {
            return TrajectoryStatus::ERROR;
        }

        if (this->duration_ <= 0.0) {
            target.q = this->q_start_ + this->delta_q_;
            target.v.setZero();
            target.a.setZero();
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
        double s_dot = ((30.0 * tau2) - (60.0 * tau3) + (30.0 * tau4)) / this->duration_;
        double s_ddot =
            ((60.0 * tau) - (180.0 * tau2) + (120.0 * tau3)) / (this->duration_ * this->duration_);

        target.q = this->q_start_ + (s * this->delta_q_);
        target.v = s_dot * this->delta_q_;
        target.a = s_ddot * this->delta_q_;

        return TrajectoryStatus::OK;
    }

}  // namespace xarm_geo::trajectories
