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

    // --- Example: Euclidean P Controller (Kinematic, Task-Space, NON-GEOMETRIC BASELINE) ---
    //
    // Textbook "naive" baseline provided as a contrast point against
    // `GeometricPController`. NOT recommended for production use.
    //
    // The control law is intentionally non-geometric:
    //
    //   - Position error is expressed in the WORLD frame: e_p = p_d - p.
    //   - Orientation error is expressed as a difference of ZYX Euler
    //     angles (RPY): e_rpy = wrap_to_pi(rpy_d - rpy). This singular
    //     parameterisation exhibits gimbal lock near pitch = +- pi/2
    //     and discontinuities at the chart boundary; both are precisely
    //     the failure modes the geometric SE(3) formulation avoids.
    //   - The angular component is mapped back to a spatial angular
    //     velocity via the standard RPY-rate matrix E(rpy), which is
    //     singular at gimbal lock. NO special handling is provided --
    //     this is the textbook formulation.
    //   - Feedforward is "naive": the reference twist is added directly
    //     in space frame, with NO Ad-transport into the current EE frame
    //     (contrast: GeometricPController uses Ad_{g_e} * xi_d).
    //
    // Frame interop note:
    //
    //   The control law produces a space-frame twist xi_w. The
    //   KinematicTaskControllerBase pipeline expects a BODY-frame command
    //   twist (it uses J_b for the downstream DLS / Optimal-IDK call).
    //   We therefore apply a single Ad transport at the end:
    //
    //       xi_b = Ad_{g^{-1}} * xi_w
    //
    //   This is a pure frame change. It is mathematically equivalent to
    //   projecting xi_w through the space Jacobian J_s (since J_s = Ad_g
    //   * J_b), so the JOINT-VELOCITY OUTPUT is identical to the
    //   "world-frame DLS" formulation common in textbooks. The control
    //   law itself remains coordinate-based.

    class EuclideanPController final : public KinematicTaskControllerBase {
    public:
        explicit EuclideanPController(const Model &model) : KinematicTaskControllerBase(model) {
            assert(model.dof > 0 && "EuclideanPController: model.dof must be > 0");
        }

        // --- Public Configuration ---
        // Matches GeometricPController's `gains` surface; only kp_pos and kp_rot are used.
        SE3FeedbackGains gains;
        bool use_feedforward = true;  // Naive feedforward (target twist added in space frame)

    protected:
        auto compute_command_twist(const Model & /*model*/, Data & /*data*/, KinematicsCache &kin,
                                   const TaskControllerContext &ctx,
                                   manifold::SE3::Twist &cmd_twist) noexcept -> bool override {

            const manifold::SE3 &g = kin.ee_pose();
            const Eigen::Matrix3d R = g.so3().matrix();
            const Eigen::Matrix3d R_d = ctx.ref.pose.so3().matrix();

            // World-frame position error.
            const Eigen::Vector3d e_pos = ctx.ref.pose.r3() - g.r3();

            // ZYX-Euler orientation error. wrap_to_pi to keep the
            // componentwise difference in (-pi, pi].
            const Eigen::Vector3d rpy = so3_to_rpy_zyx(R);
            const Eigen::Vector3d rpy_d = so3_to_rpy_zyx(R_d);
            Eigen::Vector3d e_rpy = rpy_d - rpy;
            for (int i = 0; i < 3; ++i) { e_rpy[i] = wrap_to_pi(e_rpy[i]); }

            // P law: linear velocity directly in world frame; angular velocity
            // is RPY-rate scaled by Kp_rot, then mapped to spatial omega via E(rpy).
            const Eigen::Vector3d v_lin_w = gains.kp_pos.cwiseProduct(e_pos);
            const Eigen::Vector3d rpy_rate_cmd = gains.kp_rot.cwiseProduct(e_rpy);
            const Eigen::Vector3d omega_w = rpy_rate_to_spatial_omega(rpy, rpy_rate_cmd);

            // Assemble space-frame twist [linear; angular] (matches the smooth Twist layout).
            manifold::SE3::Twist xi_w;
            xi_w.head<3>() = v_lin_w;
            xi_w.tail<3>() = omega_w;

            // Naive feedforward: add the reference twist in space frame.
            // No Ad-transport into the current EE body frame.
            if (use_feedforward) { xi_w += ctx.ref.twist; }

            // Frame change to body frame so the base can route through the
            // body-Jacobian DLS / Optimal-IDK solver. This is a pure transport,
            // not part of the control law.
            cmd_twist = g.inverse().Ad() * xi_w;
            return true;
        }

    private:
        // --- ZYX-Euler Helpers (File-Local, Match the Original Inline Baseline) ---

        // Extract ZYX (yaw-pitch-roll) Euler angles from a rotation matrix.
        // Returns (roll, pitch, yaw) in the [0]=roll, [1]=pitch, [2]=yaw layout
        // that pairs with `rpy_rate_to_spatial_omega` below.
        static auto so3_to_rpy_zyx(const Eigen::Matrix3d &R) noexcept -> Eigen::Vector3d {
            const double pitch = std::asin(std::clamp(-R(2, 0), -1.0, 1.0));
            double yaw = 0.0;
            double roll = 0.0;
            if (std::abs(std::cos(pitch)) > 1e-6) {
                yaw = std::atan2(R(1, 0), R(0, 0));
                roll = std::atan2(R(2, 1), R(2, 2));
            } else {
                // Gimbal lock: yaw underdetermined; pick yaw = 0 by convention.
                yaw = 0.0;
                roll = std::atan2(-R(0, 1), R(1, 1));
            }
            return Eigen::Vector3d(roll, pitch, yaw);
        }

        // Map ZYX RPY-rate to spatial angular velocity:
        //   omega = Rz(yaw) * Ry(pitch) * (e_x * d_roll)
        //         + Rz(yaw)             * (e_y * d_pitch)
        //         +                       (e_z * d_yaw)
        // i.e., the standard E(rpy) matrix for ZYX convention. Singular at
        // pitch = +- pi/2; NOT handled specially -- this is the baseline pitfall.
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

        // Wrap an angle to (-pi, pi]. File-local to avoid pulling in
        // any "geometric" helper from the public API.
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
