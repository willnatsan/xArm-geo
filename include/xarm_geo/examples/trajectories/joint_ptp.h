#pragma once

#include <algorithm>
#include <stdexcept>

#include <Eigen/Dense>

#include <xarm_geo/core/manifold.h>
#include <xarm_geo/trajectory/trajectory.h>

namespace xarm_geo::trajectories {

    // --- Example: Joint Point-to-Point Trajectory (Joint-Space) ---
    //
    // Minimum-jerk polynomial point-to-point motion in joint space:
    //
    //     q(tau)     = q_start + s(tau) * delta_q
    //     s(tau)     = 10*tau^3 - 15*tau^4 + 6*tau^5,    tau = t / duration.
    //
    // delta_q is wrapped to [-pi, pi] per joint to take the shortest path.

    class JointPTP {
    public:
        JointPTP(const Eigen::Ref<const Eigen::VectorXd> &q_start,
                 const Eigen::Ref<const Eigen::VectorXd> &q_end, double duration)
            : q_start_(q_start), duration_(duration) {

            if (q_start_.size() != q_end.size()) {
                throw std::invalid_argument("Start & End Joint Configurations Size Mismatch!");
            }

            delta_q_ = q_end - q_start_;

            // Normalise for Shortest Path [-pi, pi].
            for (auto &val : delta_q_) { val = manifold::wrap_to_pi(val); }
        }

        [[nodiscard]] auto evaluate(double t, JointTarget &target) const -> TrajectoryStatus {
            if (target.q.size() != q_start_.size() || target.v.size() != q_start_.size() ||
                target.a.size() != q_start_.size()) {
                return TrajectoryStatus::ERROR;
            }

            if (duration_ <= 0.0) {
                target.q = q_start_ + delta_q_;
                target.v.setZero();
                target.a.setZero();
                return TrajectoryStatus::OK;
            }

            t = std::clamp(t, 0.0, duration_);

            // Minimum Jerk Polynomial Formulation.
            const double tau = t / duration_;
            const double tau2 = tau * tau;
            const double tau3 = tau2 * tau;
            const double tau4 = tau3 * tau;
            const double tau5 = tau4 * tau;

            const double s = (10.0 * tau3) - (15.0 * tau4) + (6.0 * tau5);
            const double s_dot = ((30.0 * tau2) - (60.0 * tau3) + (30.0 * tau4)) / duration_;
            const double s_ddot =
                ((60.0 * tau) - (180.0 * tau2) + (120.0 * tau3)) / (duration_ * duration_);

            target.q = q_start_ + (s * delta_q_);
            target.v = s_dot * delta_q_;
            target.a = s_ddot * delta_q_;

            return TrajectoryStatus::OK;
        }

    private:
        Eigen::VectorXd q_start_;
        Eigen::VectorXd delta_q_;
        double duration_;
    };

    // --- Compile-Time Concept Verification ---
    static_assert(JointTrajectory<JointPTP>);

}  // namespace xarm_geo::trajectories
