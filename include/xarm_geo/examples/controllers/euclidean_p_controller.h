#pragma once

#include <cassert>
#include <cmath>
#include <numbers>

#include <Eigen/Dense>

#include <xarm_geo/control/controller.h>
#include <xarm_geo/control/feedback.h>
#include <xarm_geo/core/manifold.h>
#include <xarm_geo/core/system.h>

namespace xarm_geo::controllers {

    // --- Euclidean P Controller (Non-Geometric Baseline) ---
    //
    // Kinematic, task-space. Textbook naive baseline provided as a contrast
    // point against GeometricPController -- not recommended for production.
    //
    // The control law is intentionally non-geometric:
    //   - World-frame position error e_p = p_d - p.
    //   - ZYX-Euler orientation error (singular at pitch = +-pi/2; gimbal
    //     lock is NOT handled).
    //   - Naive feedforward: reference twist added in space frame with no
    //     Ad-transport into the current EE frame.
    //
    // The hook returns a body-frame twist (the base routes through J_b);
    // we compute xi_w first and apply xi_b = Ad_{g^{-1}} * xi_w at the end
    // (a frame change, not control logic).

    class EuclideanPController final : public KinematicTaskControllerBase {
    public:
        explicit EuclideanPController(const Model &model) : KinematicTaskControllerBase(model) {
            assert(model.dof > 0 && "EuclideanPController: model.dof must be > 0");
        }

        // --- Public Configuration ---
        SE3FeedbackGains gains;
        bool use_feedforward = true;

    protected:
        auto compute_command_twist(const Model & /*model*/, Data & /*data*/, KinematicsCache &kin,
                                   const TaskControllerContext &ctx,
                                   manifold::SE3::Twist &cmd_twist) noexcept -> bool override {

            const manifold::SE3 &g = kin.ee_pose();
            const Eigen::Matrix3d R = g.so3().matrix();
            const Eigen::Matrix3d R_d = ctx.ref.pose.so3().matrix();

            const Eigen::Vector3d e_pos = ctx.ref.pose.r3() - g.r3();

            // ZYX-Euler orientation error, wrapped to (-pi, pi].
            const Eigen::Vector3d rpy = so3_to_rpy_zyx(R);
            const Eigen::Vector3d rpy_d = so3_to_rpy_zyx(R_d);
            Eigen::Vector3d e_rpy = rpy_d - rpy;
            for (int i = 0; i < 3; ++i) { e_rpy[i] = wrap_to_pi(e_rpy[i]); }

            // P law: world-frame linear velocity; angular velocity as RPY-rate
            // mapped through E(rpy) to spatial omega.
            const Eigen::Vector3d v_lin_w = gains.kp_pos.cwiseProduct(e_pos);
            const Eigen::Vector3d rpy_rate_cmd = gains.kp_rot.cwiseProduct(e_rpy);
            const Eigen::Vector3d omega_w = rpy_rate_to_spatial_omega(rpy, rpy_rate_cmd);

            manifold::SE3::Twist xi_w;
            xi_w.head<3>() = v_lin_w;
            xi_w.tail<3>() = omega_w;

            // Naive feedforward: add reference twist in space frame (no Ad-transport).
            if (use_feedforward) { xi_w += ctx.ref.twist; }

            // Frame change for the base pipeline; not part of the law.
            cmd_twist = g.inverse().Ad() * xi_w;
            return true;
        }

    private:
        // --- ZYX-Euler Helpers (File-Local) ---

        static auto so3_to_rpy_zyx(const Eigen::Matrix3d &R) noexcept -> Eigen::Vector3d {
            const double pitch = std::asin(std::clamp(-R(2, 0), -1.0, 1.0));
            double yaw = 0.0;
            double roll = 0.0;
            if (std::abs(std::cos(pitch)) > 1e-6) {
                yaw = std::atan2(R(1, 0), R(0, 0));
                roll = std::atan2(R(2, 1), R(2, 2));
            } else {
                // Gimbal lock: yaw underdetermined; convention is yaw = 0.
                yaw = 0.0;
                roll = std::atan2(-R(0, 1), R(1, 1));
            }
            return Eigen::Vector3d(roll, pitch, yaw);
        }

        // Standard E(rpy) for ZYX; singular at pitch = +-pi/2 (the baseline pitfall).
        static auto rpy_rate_to_spatial_omega(const Eigen::Vector3d &rpy,
                                              const Eigen::Vector3d &rpy_rate) noexcept
            -> Eigen::Vector3d {
            const double roll = rpy[0];
            const double pitch = rpy[1];
            const double yaw = rpy[2];
            const double d_roll = rpy_rate[0];
            const double d_pitch = rpy_rate[1];
            const double d_yaw = rpy_rate[2];

            const Eigen::AngleAxisd Rz(yaw, Eigen::Vector3d::UnitZ());
            const Eigen::AngleAxisd Ry(pitch, Eigen::Vector3d::UnitY());

            return Eigen::Vector3d::UnitZ() * d_yaw + Rz * (Eigen::Vector3d::UnitY() * d_pitch) +
                   Rz * Ry * (Eigen::Vector3d::UnitX() * d_roll);
        }

        // Wrap an angle to (-pi, pi].
        static auto wrap_to_pi(double a) noexcept -> double {
            constexpr double pi = std::numbers::pi;
            while (a > pi) { a -= 2.0 * pi; }
            while (a < -pi) { a += 2.0 * pi; }
            return a;
        }
    };

    // --- Compile-Time Concept Verification ---
    static_assert(xarm_geo::KinematicTaskController<EuclideanPController>);

}  // namespace xarm_geo::controllers
