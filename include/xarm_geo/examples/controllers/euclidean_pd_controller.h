#pragma once

#include <cassert>
#include <cmath>
#include <numbers>

#include <Eigen/Dense>

#include <xarm_geo/control/controller.h>
#include <xarm_geo/control/feedback.h>
#include <xarm_geo/core/manifold.h>
#include <xarm_geo/core/system.h>
#include <xarm_geo/utils/debug.h>

namespace xarm_geo::controllers {

    // --- Euclidean PD Controller (Non-Geometric Baseline) ---
    //
    // Dynamic, task-space. Textbook naive baseline provided as a contrast
    // point against GeometricPDController -- not recommended for production.
    //
    // The control law is intentionally non-geometric:
    //   - World-frame position error e_p = p_d - p.
    //   - Orientation error as a ZYX-Euler difference (singular at
    //     pitch = +-pi/2; gimbal lock is NOT handled).
    //   - Velocity error as a raw 6-vector difference (no Ad-transport of
    //     the reference into the current EE frame).
    //   - Optional feedforward F_FF = Lambda_w * a_d with no ad^* coupling
    //     term, so the FF residual does NOT vanish at perfect tracking.
    //
    // The hook returns a body-frame wrench (the base projects via J_b^T);
    // we compute F_w first and apply a single co-tangent transport
    // F_b = Ad_g^T * F_w at the end (a frame change, not control logic).

    class EuclideanPDController final : public DynamicTaskControllerBase {
    public:
        // Mirrors GeometricPDController so the A/B comparison varies only the control law.
        static constexpr BiasCompensation kRecommendedBiasCompensation = BiasCompensation::Full;

        explicit EuclideanPDController(const Model &model)
            : DynamicTaskControllerBase(model), M_inv_Jst_(Eigen::MatrixXd::Zero(model.dof, 6)) {

            assert(model.dof > 0 && "EuclideanPDController: model.dof must be > 0");
            lambda_w_.setZero();
            bias_compensation = kRecommendedBiasCompensation;
        }

        // --- Public Configuration ---
        SE3FeedbackGains gains;
        bool use_feedforward = true;

    protected:
        auto compute_command_wrench(const Model & /*model*/, Data & /*data*/, KinematicsCache &kin,
                                    DynamicsCache &dyn, const TaskControllerContext &ctx,
                                    manifold::SE3::Wrench &cmd_wrench) noexcept -> bool override {

            const manifold::SE3 &g = kin.ee_pose();
            const Eigen::Matrix3d R = g.so3().matrix();
            const Eigen::Matrix3d R_d = ctx.ref.pose.so3().matrix();

            // --- World-Frame Configuration Error ---

            const Eigen::Vector3d e_pos = ctx.ref.pose.r3() - g.r3();

            const Eigen::Vector3d rpy = so3_to_rpy_zyx(R);
            const Eigen::Vector3d rpy_d = so3_to_rpy_zyx(R_d);
            Eigen::Vector3d e_rpy = rpy_d - rpy;
            for (int i = 0; i < 3; ++i) { e_rpy[i] = wrap_to_pi(e_rpy[i]); }

            // --- World-Frame Velocity Error ---
            //
            // xi_w = Ad_g * (J_b * v); avoids reading data.space_jacobian inside the hook.
            // The reference twist is taken as a raw space-frame 6-vector (no Ad-transport).
            const manifold::SE3::Twist body_twist = kin.body_jacobian() * ctx.fb.v;
            const manifold::SE3::Twist xi_w = g.Ad() * body_twist;
            const manifold::SE3::Twist &xi_d_w = ctx.ref.twist;

            const Eigen::Vector3d e_vel_lin = xi_d_w.head<3>() - xi_w.head<3>();
            const Eigen::Vector3d e_vel_ang = xi_d_w.tail<3>() - xi_w.tail<3>();

            // --- Space-Frame PD Wrench ---
            //
            // Kp_rot acts on the singular Euler-angle error e_rpy (gimbal-lock pitfall);
            // map to a space-frame torque via E(rpy) for unit-consistent wrenches.
            const Eigen::Vector3d kp_rpy = gains.kp_rot.cwiseProduct(e_rpy);
            const Eigen::Vector3d kp_rpy_w = rpy_rate_to_spatial_omega(rpy, kp_rpy);

            manifold::SE3::Wrench F_w;
            F_w.head<3>() = gains.kp_pos.cwiseProduct(e_pos) + gains.kd_lin.cwiseProduct(e_vel_lin);
            F_w.tail<3>() = kp_rpy_w + gains.kd_ang.cwiseProduct(e_vel_ang);

            // --- Optional Naive Inertial Feedforward ---
            //
            // F_FF = Lambda_w(q) * a_d_w  with  Lambda_w = (J_s M^-1 J_s^T)^{-1}.
            // Compute J_s = Ad_g * J_b to avoid touching data.space_jacobian.
            // No ad_{xi_e}^* * Lambda * xi_d coupling term -- that's what makes
            // this baseline "naive"; the geometric variant adds it back.
            if (use_feedforward) {
                M_llt_.compute(dyn.M());
                if (M_llt_.info() == Eigen::Success) {
                    const manifold::SE3::Jacobian J_s = g.Ad() * kin.body_jacobian();

                    M_inv_Jst_.noalias() = M_llt_.solve(J_s.transpose());
                    lambda_w_.noalias() = J_s * M_inv_Jst_;
                    lambda_w_ = lambda_w_.inverse().eval();

                    // Raw 6-vector reference spatial acceleration; no se3_transported_acc.
                    F_w.noalias() += lambda_w_ * ctx.ref.spatial_acc;
                } else {
                    debug::log("EuclideanPDController: Cholesky failed on M(q); FF dropped");
                }
            }

            // Frame change for the base pipeline: F_b = Ad_g^T * F_w (not part of the law).
            cmd_wrench = g.Ad().transpose() * F_w;
            return true;
        }

    private:
        // --- Pre-Allocated Per-Tick Scratch ---
        Eigen::LLT<Eigen::MatrixXd> M_llt_;
        Eigen::MatrixXd M_inv_Jst_;             // (dof x 6)
        Eigen::Matrix<double, 6, 6> lambda_w_;  // operational-space inertia (world frame)

        // --- ZYX-Euler Helpers (File-Local) ---
        //
        // Duplicated across example headers so each one stays self-contained.

        static auto so3_to_rpy_zyx(const Eigen::Matrix3d &R) noexcept -> Eigen::Vector3d {
            const double pitch = std::asin(std::clamp(-R(2, 0), -1.0, 1.0));
            double yaw = 0.0;
            double roll = 0.0;
            if (std::abs(std::cos(pitch)) > 1e-6) {
                yaw = std::atan2(R(1, 0), R(0, 0));
                roll = std::atan2(R(2, 1), R(2, 2));
            } else {
                yaw = 0.0;
                roll = std::atan2(-R(0, 1), R(1, 1));
            }
            return Eigen::Vector3d(roll, pitch, yaw);
        }

        static auto rpy_rate_to_spatial_omega(const Eigen::Vector3d &rpy,
                                              const Eigen::Vector3d &rpy_rate) noexcept
            -> Eigen::Vector3d {
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

        static auto wrap_to_pi(double a) noexcept -> double {
            constexpr double pi = std::numbers::pi;
            while (a > pi) { a -= 2.0 * pi; }
            while (a < -pi) { a += 2.0 * pi; }
            return a;
        }
    };

    // --- Compile-Time Concept Verification ---
    static_assert(xarm_geo::DynamicTaskController<EuclideanPDController>);

}  // namespace xarm_geo::controllers
