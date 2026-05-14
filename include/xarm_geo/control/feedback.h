#pragma once

#include <cstdint>

#include <Eigen/Dense>

#include <xarm_geo/core/manifold.h>

namespace xarm_geo {

    // --- SE(3) Feedback Gains ---
    //
    // Defaults are zero to surface mistuning loudly.
    struct SE3FeedbackGains {
        Eigen::Vector3d kp_pos = Eigen::Vector3d::Zero();  // K_p linear (body frame)
        Eigen::Vector3d kp_rot = Eigen::Vector3d::Zero();  // K_R angular (body frame)
        Eigen::Vector3d kd_lin = Eigen::Vector3d::Zero();  // K_D linear (PD / PID)
        Eigen::Vector3d kd_ang = Eigen::Vector3d::Zero();  // K_D angular (PD / PID)
        Eigen::Vector3d ki_lin = Eigen::Vector3d::Zero();  // K_I linear (PI / PID)
        Eigen::Vector3d ki_ang = Eigen::Vector3d::Zero();  // K_I angular (PI / PID)
    };

    // --- Error Gradient Selector ---
    //
    // Tag for choosing between the two gradient implementations below:
    //   LieGroup   : smooth (trace-based) gradient -> almost global stability.
    //   LieAlgebra : discontinuous (log-map) gradient -> global stability.

    enum class GradientType : std::uint8_t { LieGroup, LieAlgebra };

    // --- SE(3) Feedback Building Blocks ---
    //
    // Body-frame primitives for SE(3) tracking and setpoint regulation.
    // Stateless, allocation-free, independent of Model / Data.
    //
    // Body-frame gradient of the SE(3) navigation function (Maithripala 2006):
    //
    //     Phi(g_e) = 0.5 * p_e^T K_p p_e + 0.5 * tr(K_R (I - R_e))
    //     nabla Phi = ( R_e^T K_p p_e ;  0.5 * (K_R R_e - R_e^T K_R)^vee )
    //
    // Returns +nabla Phi in the body frame. K_p, K_R are diagonal (per-axis) gains.
    //
    // Note: Under the left-error convention g_e = g^{-1} g_d, descent is achieved by
    // ADDING grad (not subtracting); the chain rule through g_e flips the sign
    // relative to the naive "subtract gradient" rule:
    //   cmd_twist  = ad_xi_d + grad       (kinematic P law)
    //   cmd_wrench = grad - K_d * xi_e    (dynamic PD law)
    [[nodiscard]] auto se3_lie_group_gradient(const manifold::SE3 &g_e,
                                              const Eigen::Vector3d &kp_pos,
                                              const Eigen::Vector3d &kp_rot)
        -> manifold::SE3::Twist;

    // SE(3) log-map gradient with right-Jacobian correction.
    //
    //     nabla Phi_log(g_e) = Ad_{g_e} * dr_exp(log(g_e)) * K * log(g_e)^vee
    //
    // Returns +nabla Phi_log in the body frame; same sign convention as
    // se3_lie_group_gradient. Reduces to k * log(g_e) under isotropic scalar
    // gain (Prabhu 2020); the Ad and dr_exp factors restore exponential convergence
    // for per-axis K. Discontinuous on the antipodal set theta = pi.
    [[nodiscard]] auto se3_lie_algebra_gradient(const manifold::SE3 &g_e,
                                                const Eigen::Vector3d &kp_pos,
                                                const Eigen::Vector3d &kp_rot)
        -> manifold::SE3::Twist;

    // Wrench-typed variants of the two gradient helpers above.
    //
    // Use these in dynamic controllers, where the gradient is composed with
    // damping into a body-frame wrench (cmd_wrench = grad - K_d * xi_e).
    //
    // Same numerical value and sign convention as the Twist variants; the only
    // difference is the static return type (For pedagogical consistency). See the
    // Twist variants above for the full sign-convention and chain-rule derivation
    [[nodiscard]] auto se3_lie_group_gradient_wrench(const manifold::SE3 &g_e,
                                                     const Eigen::Vector3d &kp_pos,
                                                     const Eigen::Vector3d &kp_rot)
        -> manifold::SE3::Wrench;

    [[nodiscard]] auto se3_lie_algebra_gradient_wrench(const manifold::SE3 &g_e,
                                                       const Eigen::Vector3d &kp_pos,
                                                       const Eigen::Vector3d &kp_rot)
        -> manifold::SE3::Wrench;

    // Body-frame velocity error: xi_e = xi - Ad_{g_e} * xi_d.
    // `body_twist` is the current EE body twist (J_b * v); `target_twist_body`
    // is the reference body twist in g_d's frame (Ad_{g_e} transports it).
    [[nodiscard]] auto se3_velocity_error(const manifold::SE3::Twist &body_twist,
                                          const manifold::SE3 &g_e,
                                          const manifold::SE3::Twist &target_twist_body)
        -> manifold::SE3::Twist;

    // Closed-form time derivative of the transported reference twist.
    //
    //     d/dt(Ad_{g_e} * xi_d) = -ad_{xi_e}(Ad_{g_e} * xi_d) + Ad_{g_e} * a_d
    //
    // (Equivalent to the xi-form via ad_v(v) = 0 and xi = xi_e + Ad_{g_e} * xi_d.)
    // Takes xi_e to avoid re-passing xi (already consumed by the K_D term).
    // `ad_xi_d` is the cached Ad_{g_e} * xi_d; `spatial_acc_body` is d/dt(xi_d).
    [[nodiscard]] auto
    se3_transported_acc(const manifold::SE3 &g_e, const manifold::SE3::Twist &xi_e,
                        const manifold::SE3::Twist &ad_xi_d,
                        const manifold::SE3::SpatialAcceleration &spatial_acc_body)
        -> manifold::SE3::Twist;

    // --- Operational-Space Inertia with DLS Regularisation ---
    //
    // Lambda(q) = (J * M(q)^-1 * J^T)^-1 is the manipulator's effective
    // inertia at the task frame in which J is expressed (body frame when J =
    // J_b, world frame when J = J_s = Ad_g * J_b). It is configuration-
    // dependent and becomes ill-conditioned near task singularities -- a
    // direct inverse blows up exactly where the operational-space wrench
    // most needs to stay bounded.
    //
    // This helper computes Lambda via a Cholesky-based inversion of M and
    // a damped-least-squares fallback: when |det(M_op)| < min_det the inner
    // matrix M_op = J * M^-1 * J^T is replaced by (M_op + damping^2 * I)
    // before inversion, capping sigma_max(Lambda) at ~1/damping^2.
    //
    // Returns false iff M_llt.info() != Eigen::Success, in which case both
    // outputs are left untouched. Callers should drop Lambda-scaled terms on
    // failure and either fall back to a raw PD or skip the tick.
    [[nodiscard]] auto compute_op_space_inertia(const Eigen::LLT<Eigen::MatrixXd> &M_llt,
                                                const manifold::SE3::Jacobian &J,
                                                Eigen::Matrix<double, 6, 6> &Lambda_out,
                                                Eigen::MatrixXd &M_inv_Jt_scratch,
                                                double damping = 0.05,
                                                double min_det = 1e-6) noexcept -> bool;

}  // namespace xarm_geo
