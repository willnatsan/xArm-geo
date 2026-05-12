#include <stdexcept>

#include <smooth/spline/bspline.hpp>
#include <unsupported/Eigen/Splines>

#include <xarm_geo/trajectory/trajectory.h>

namespace xarm_geo {

    // --- Analytic Task Trajectory Base Class ---

    AnalyticTaskTrajectory::AnalyticTaskTrajectory(manifold::SE3 anchor, double duration,
                                                   int num_samples)
        : anchor_(std::move(anchor)), duration_(duration), num_samples_(num_samples) {
        if (num_samples_ < 2) {
            throw std::invalid_argument("AnalyticTaskTrajectory: num_samples must be >= 2");
        }
        if (duration_ <= 0.0) {
            throw std::invalid_argument("AnalyticTaskTrajectory: duration must be > 0");
        }
    }

    void AnalyticTaskTrajectory::build_spline() {
        std::vector<manifold::SE3> waypoints;
        waypoints.reserve(static_cast<std::size_t>(num_samples_));

        for (int i = 0; i < num_samples_; ++i) {
            const double t =
                (static_cast<double>(i) / static_cast<double>(num_samples_ - 1)) * duration_;
            auto [local_rot, local_pos] = sample(t);
            waypoints.emplace_back(anchor_ * manifold::SE3(local_rot, local_pos));
        }

        spline_ = build_se3_spline<5>(waypoints, duration_);
        initialised_ = true;
    }

    auto AnalyticTaskTrajectory::evaluate(double t, TaskTarget &target) const -> TrajectoryStatus {
        if (!initialised_) { return TrajectoryStatus::NOT_INITIALISED; }
        if (t < 0.0 || t > duration_) { return TrajectoryStatus::OUT_OF_DOMAIN; }

        manifold::SE3::Tangent vel;
        manifold::SE3::Tangent acc;

        target.pose = spline_(t, smooth::OptTangent<manifold::SE3>{vel},
                              smooth::OptTangent<manifold::SE3>{acc});
        target.twist = vel;
        target.spatial_acc = acc;

        return TrajectoryStatus::OK;
    }

    auto AnalyticTaskTrajectory::duration() const noexcept -> double { return duration_; }

    // --- Analytic Joint Trajectory Base Class ---

    AnalyticJointTrajectory::AnalyticJointTrajectory(int dof, double duration, int num_samples)
        : dof_(dof), duration_(duration), num_samples_(num_samples) {
        if (dof_ < 1) { throw std::invalid_argument("AnalyticJointTrajectory: dof must be >= 1"); }
        if (duration_ <= 0.0) {
            throw std::invalid_argument("AnalyticJointTrajectory: duration must be > 0");
        }
        if (num_samples_ < 2) {
            throw std::invalid_argument("AnalyticJointTrajectory: num_samples must be >= 2");
        }
    }

    void AnalyticJointTrajectory::build_spline() {
        // Eigen::SplineFitting expects points as columns of a (dof x num_samples) matrix.
        Eigen::MatrixXd pts(dof_, num_samples_);

        for (int i = 0; i < num_samples_; ++i) {
            const double t =
                (static_cast<double>(i) / static_cast<double>(num_samples_ - 1)) * duration_;
            const Eigen::VectorXd q = sample(t);

            if (q.size() != dof_) {
                throw std::invalid_argument(
                    "AnalyticJointTrajectory: sample() returned wrong size; "
                    "expected dof=" +
                    std::to_string(dof_) + ", got " + std::to_string(static_cast<int>(q.size())));
            }
            pts.col(i) = q;
        }

        // Uniform parameterisation in [0, 1]; evaluate() scales t -> u = t / duration_.
        Eigen::RowVectorXd params(num_samples_);
        for (int i = 0; i < num_samples_; ++i) {
            params(i) = static_cast<double>(i) / static_cast<double>(num_samples_ - 1);
        }

        constexpr int kDegree = 5;
        spline_ = Eigen::SplineFitting<Eigen::Spline<double, Eigen::Dynamic>>::Interpolate(
            pts, kDegree, params);

        initialised_ = true;
    }

    auto AnalyticJointTrajectory::evaluate(double t, JointTarget &target) const
        -> TrajectoryStatus {
        if (!initialised_) { return TrajectoryStatus::NOT_INITIALISED; }
        if (t < 0.0 || t > duration_) { return TrajectoryStatus::OUT_OF_DOMAIN; }

        const double u = t / duration_;

        // derivatives(u, 2) columns: q, dq/du, d²q/du². Chain-rule to dt.
        const auto derivs = spline_.derivatives(u, 2);
        target.q = derivs.col(0);
        target.v = derivs.col(1) / duration_;
        target.a = derivs.col(2) / (duration_ * duration_);

        return TrajectoryStatus::OK;
    }

    auto AnalyticJointTrajectory::duration() const noexcept -> double { return duration_; }
    auto AnalyticJointTrajectory::dof() const noexcept -> int { return dof_; }

}  // namespace xarm_geo
