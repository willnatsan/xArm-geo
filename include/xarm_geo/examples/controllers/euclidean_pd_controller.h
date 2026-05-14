#pragma once

#include <cassert>
#include <cmath>
#include <numbers>
#include <string_view>

#include <Eigen/Dense>

#include <xarm_geo/control/controller.h>
#include <xarm_geo/control/feedback.h>
#include <xarm_geo/core/manifold.h>
#include <xarm_geo/core/system.h>

namespace xarm_geo::controllers {

    // --- Euclidean PD Controller (Non-Geometric Baseline) ---
    //
    // Dynamic, task-space. Provided as a contrast point against
    // GeometricPDController (Bullo-Murray / Maithripala / Seo form);
    // intentionally retains two non-geometric defects so the A/B comparison
    // isolates the contribution of the geometric error generator and the
    // geometric FF coupling.
    //
    // World-frame law (mirrors GeometricPDController's wrench-direct PD +
    // Lambda-scaled FF structure):
    //   F_w_pd  = K_p * e_pos       + K_d * e_vel_lin             [PD linear, wrench-direct]
    //           + E(rpy) * K_p * e_rpy + K_d * e_vel_ang          [PD angular, see below]
    //
    // F_w_pd is built as "force-and-torque applied at the EE" in world frame.
    // The smooth SE(3) Ad convention treats the wrench covector as
    // (force, moment-about-world-origin), so a lever-arm correction is applied
    // before the spatial-to-body transform:
    //
    //   F_w.tail<3>() += p x F_w.head<3>()                        [tau_origin = tau_ee + p x f]
    //
    // The FF term Lambda_w(q) * a_d_w is derived from J_s = Ad_g * J_b and is
    // already a spatial wrench in (force, moment-about-origin) form; it does not
    // require the lever-arm correction and is added after it.
    //
    //   F_w     = F_w_pd (lever-arm corrected) + Lambda_w * a_d_w [if use_feedforward]
    //   F_b     = Ad_g^T * F_w                                    [base pipeline]
    //
    // Retained naivetes (intentional, distinguish this baseline from the
    // geometric variant):
    //   1. Orientation error as a ZYX-Euler difference (singular at pitch =
    //      +-pi/2; gimbal lock is NOT handled). The geometric variant uses
    //      Maithripala's smooth navigation-function gradient on SO(3) / SE(3).
    //   2. Feedforward F_FF_w = Lambda_w * a_d_w omits the ad-coupling term
    //      -ad_{xi_e}(Ad_{g_d} xi_d) that appears inside d/dt(Ad_{g_d} xi_d),
    //      so the FF residual does not vanish at perfect tracking. The
    //      geometric variant uses the full Bullo-Murray transported
    //      acceleration via se3_transported_acc.

    class EuclideanPDController final : public DynamicTaskControllerBase {
    public:
        static constexpr std::string_view kName = "EuclideanPDController";

        // Recommended default; users may override after construction.
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
        double lambda_damping = 0.05;  // Damped least-squares regularisation for Lambda(q).

    protected:
        auto compute_command_wrench(const Model & /*model*/, Data & /*data*/, KinematicsCache &kin,
                                    DynamicsCache &dyn, const TaskControllerContext &ctx,
                                    manifold::SE3::Wrench &cmd_wrench) noexcept -> bool override {

            const manifold::SE3 &g = kin.ee_pose();
            const Eigen::Matrix3d R = g.so3().matrix();
            const Eigen::Matrix3d R_d = ctx.ref.pose.so3().matrix();

            // World-Frame Configuration Error
            const Eigen::Vector3d e_pos = ctx.ref.pose.r3() - g.r3();

            const Eigen::Vector3d rpy = so3_to_rpy_zyx(R);
            const Eigen::Vector3d rpy_d = so3_to_rpy_zyx(R_d);
            Eigen::Vector3d e_rpy = rpy_d - rpy;
            for (int i = 0; i < 3; ++i) { e_rpy[i] = wrap_to_pi(e_rpy[i]); }

            // World-Frame Velocity Error.
            // ctx.ref.twist is a body twist in the g_d frame (per the TaskTarget contract);
            // transport via Ad_{g_d} so it lives in the same world frame as xi_w.
            const manifold::SE3::Twist body_twist = kin.body_jacobian() * ctx.fb.v;
            const manifold::SE3::Twist xi_w = g.Ad() * body_twist;
            const manifold::SE3::Twist xi_d_w = ctx.ref.pose.Ad() * ctx.ref.twist;

            const Eigen::Vector3d e_vel_lin = xi_d_w.head<3>() - xi_w.head<3>();
            const Eigen::Vector3d e_vel_ang = xi_d_w.tail<3>() - xi_w.tail<3>();

            // World-frame PD wrench: produced directly in wrench units
            // (no Lambda-scaling on the PD term -- see GeometricPDController doc).
            const Eigen::Vector3d kp_rpy = gains.kp_rot.cwiseProduct(e_rpy);
            const Eigen::Vector3d kp_rpy_w = rpy_rate_to_spatial_omega(rpy, kp_rpy);

            manifold::SE3::Wrench F_w;
            F_w.head<3>() = gains.kp_pos.cwiseProduct(e_pos) + gains.kd_lin.cwiseProduct(e_vel_lin);
            F_w.tail<3>() = kp_rpy_w + gains.kd_ang.cwiseProduct(e_vel_ang);

            // Lever-arm correction. Without this, Ad_g^T introduces a velocity-dependent
            // torque coupling (p x K_d * xi_w) that destabilises tracking trajectories.
            F_w.tail<3>() += g.r3().cross(F_w.head<3>());

            // Feedforward: Lambda_w(q) * a_d_w with the naive (no ad^* coupling) form.
            // Lambda_w is derived from J_s = Ad_g * J_b, so Lambda_w * a_d_w is already
            // a spatial wrench in (force, moment-about-origin) form; no lever-arm needed.
            if (use_feedforward) {
                const manifold::SE3::Jacobian J_s = g.Ad() * kin.body_jacobian();
                M_llt_.compute(dyn.M());
                if (compute_op_space_inertia(M_llt_, J_s, lambda_w_, M_inv_Jst_, lambda_damping)) {
                    // ctx.ref.spatial_acc is body-frame (d/dt of ctx.ref.twist in the g_d frame);
                    // transport via Ad_{g_d} into world frame before scaling.
                    const manifold::SE3::SpatialAcceleration a_d_w =
                        ctx.ref.pose.Ad() * ctx.ref.spatial_acc;
                    F_w.coeffs.noalias() += lambda_w_ * a_d_w;
                }
            }

            // Spatial-to-body wrench: F_b = Ad_g^T * F_w (not part of the law).
            // F_w is now a fully corrected spatial wrench (force, moment-about-origin).
            cmd_wrench = g.Ad().transpose() * F_w;
            return true;
        }

    private:
        // --- Pre-Allocated Per-Tick Scratch ---
        Eigen::LLT<Eigen::MatrixXd> M_llt_;
        Eigen::MatrixXd M_inv_Jst_;             // (dof x 6)
        Eigen::Matrix<double, 6, 6> lambda_w_;  // operational-space inertia (world frame)

        // --- ZYX-Euler Helpers (File-Local) ---
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
