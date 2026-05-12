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

    // --- Example: Euclidean PD Controller (Dynamic, Task-Space, NON-GEOMETRIC BASELINE) ---
    //
    // Textbook "naive" baseline provided as a contrast point against
    // `GeometricPDController`. NOT recommended for production use.
    //
    // The control law is intentionally non-geometric:
    //
    //   - Position error in WORLD frame: e_p = p_d - p.
    //   - Orientation error as a difference of ZYX Euler angles (RPY):
    //     e_rpy = wrap_to_pi(rpy_d - rpy). Singular parameterisation;
    //     gimbal lock at pitch = +- pi/2 is NOT handled.
    //   - Velocity error: e_v = xi_d - xi, with BOTH treated as ordinary
    //     6-vectors in world frame (linear + spatial-angular). No
    //     Ad-transport of the reference into the current EE frame.
    //   - Feedforward (when enabled): naive operational-space inertia
    //     coupling F_FF = Lambda_w * a_d, with NO ad^* correction term.
    //     The Maithripala FF residual therefore does NOT vanish at perfect
    //     tracking; passivity is NOT preserved.
    //
    // Frame interop note:
    //
    //   The control law produces a space-frame wrench F_w. The
    //   DynamicTaskControllerBase pipeline expects a BODY-frame wrench
    //   (it projects via J_b^T to obtain joint torques). We therefore
    //   apply a single co-tangent transport at the end:
    //
    //       F_b = Ad_g^T * F_w
    //
    //   This is a pure frame change (virtual work is frame-invariant:
    //   J_b^T * F_b = J_s^T * F_w). It is NOT part of the control law.

    class EuclideanPDController final : public DynamicTaskControllerBase {
    public:
        // Default bias-compensation policy chosen to mirror GeometricPDController, so the
        // A/B comparison varies only the control law.
        static constexpr BiasCompensation kRecommendedBiasCompensation = BiasCompensation::Full;

        explicit EuclideanPDController(const Model &model)
            : DynamicTaskControllerBase(model), M_inv_Jst_(Eigen::MatrixXd::Zero(model.dof, 6)) {

            assert(model.dof > 0 && "EuclideanPDController: model.dof must be > 0");
            lambda_w_.setZero();
            bias_compensation = kRecommendedBiasCompensation;
        }

        // --- Public Configuration ---
        // Matches GeometricPDController's `gains` surface; kp_pos, kp_rot, kd_lin, kd_ang used.
        SE3FeedbackGains gains;
        bool use_feedforward = true;  // Naive op-space inertial FF (no ad^* coupling)

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
            // Current spatial twist: xi_w = J_s * v. We work via Ad on the body
            // twist (avoids reading data.space_jacobian inside a hook):
            //     xi_w = Ad_g * (J_b * v)
            const manifold::SE3::Twist body_twist = kin.body_jacobian() * ctx.fb.v;
            const manifold::SE3::Twist xi_w = g.Ad() * body_twist;

            // The reference twist is treated naively as a space-frame twist:
            // raw 6-vector (v_lin_ref ; omega_ref) with no Ad-transport.
            const manifold::SE3::Twist &xi_d_w = ctx.ref.twist;

            // Velocity error: linear component in world frame, angular
            // component as a spatial omega error (mapped from the RPY-rate
            // command path, *not* via the orientation error gradient).
            const Eigen::Vector3d e_vel_lin = xi_d_w.head<3>() - xi_w.head<3>();
            const Eigen::Vector3d e_vel_ang = xi_d_w.tail<3>() - xi_w.tail<3>();

            // --- P + D Wrench (Space Frame) ---
            //
            // F_w_lin = Kp_pos * e_pos   + Kd_lin * e_vel_lin
            // F_w_ang = Kp_rot * e_rpy_w + Kd_ang * e_vel_ang
            //
            // Crucially, the angular Kp acts on the (singular) Euler-angle
            // error e_rpy directly, not on a Lie-algebra element. This is the
            // textbook formulation and the source of the gimbal-lock pitfall.
            //
            // The angular error is mapped to a space-frame torque axis by
            // E(rpy) so the units of the wrench are consistent: torque about
            // world axes.
            const Eigen::Vector3d kp_rpy = gains.kp_rot.cwiseProduct(e_rpy);
            const Eigen::Vector3d kp_rpy_w = rpy_rate_to_spatial_omega(rpy, kp_rpy);

            manifold::SE3::Wrench F_w;
            F_w.head<3>() = gains.kp_pos.cwiseProduct(e_pos) + gains.kd_lin.cwiseProduct(e_vel_lin);
            F_w.tail<3>() = kp_rpy_w + gains.kd_ang.cwiseProduct(e_vel_ang);

            // --- Optional Naive Inertial Feedforward ---
            //
            // F_FF = Lambda_w(q) * a_d_w
            //
            // where Lambda_w = ( J_s * M^-1 * J_s^T )^{-1} is the
            // operational-space inertia in world frame. We compute J_s via
            // J_s = Ad_g * J_b to avoid touching data.space_jacobian inside
            // the hook.
            //
            // NOTE: NO ad_{xi_e}^* * Lambda * xi_d term. The geometric variant
            // includes this so the FF residual vanishes at perfect tracking;
            // omitting it is exactly what makes this baseline "naive".
            if (use_feedforward) {
                M_llt_.compute(dyn.M());
                if (M_llt_.info() == Eigen::Success) {
                    const manifold::SE3::Jacobian J_s = g.Ad() * kin.body_jacobian();

                    M_inv_Jst_.noalias() = M_llt_.solve(J_s.transpose());
                    lambda_w_.noalias() = J_s * M_inv_Jst_;
                    lambda_w_ = lambda_w_.inverse().eval();

                    // a_d_w: raw 6-vector reference spatial acceleration. No
                    // se3_transported_acc -- by design.
                    F_w.noalias() += lambda_w_ * ctx.ref.spatial_acc;
                } else {
                    debug::log("EuclideanPDController: Cholesky failed on M(q); FF dropped");
                }
            }

            // --- Frame Change to Body Frame for the Base Pipeline ---
            //
            // F_b = Ad_g^T * F_w. Co-tangent transport; not part of the control law.
            cmd_wrench = g.Ad().transpose() * F_w;
            return true;
        }

    private:
        // --- Per-Tick Scratch (Pre-Allocated at Construction) ---
        Eigen::LLT<Eigen::MatrixXd> M_llt_;
        Eigen::MatrixXd M_inv_Jst_;             // (dof x 6)
        Eigen::Matrix<double, 6, 6> lambda_w_;  // Operational-space inertia (world frame)

        // --- ZYX-Euler Helpers (File-Local, Match `EuclideanPController`) ---
        //
        // Duplicated rather than shared, to keep each example header
        // self-contained (matches how the geometric example controllers are
        // also self-contained).

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
