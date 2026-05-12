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
    // Conventions:
    //   - g_e = g^{-1} * g_d (body-frame configuration error).
    //   - Twists / wrenches in the body frame unless stated otherwise.
    //   - Tangent layout follows `smooth`: [linear; angular].

    // SE(3) navigation-function gradient in the body frame of g.
    //
    //     Phi(g_e) = 0.5 * tr(K_R (I - R_e)) + 0.5 * p_e^T K_p p_e
    //
    //     nabla Phi = ( R_e^T * K_p * p_e ;  0.5 * (K_R R_e - R_e^T K_R)^vee )
    //
    // K_p, K_R are diagonal (per-axis) gain vectors. Returns a 6-vector in
    // body-frame [linear; angular] layout.
    [[nodiscard]] auto se3_lie_group_gradient(const manifold::SE3 &g_e,
                                              const Eigen::Vector3d &kp_pos,
                                              const Eigen::Vector3d &kp_rot)
        -> manifold::SE3::Twist;

    // SE(3) log-map gradient with Jacobian correction.
    //
    //     nabla Phi_log(g_e) = - Ad_{g_e} * dr_exp(log(g_e)) * K * log(g_e)^vee
    //
    // Reduces to -k * log(g_e) under isotropic scalar gain (Prabhu-Saxena-Sastry
    // 2020); the correction restores exponential convergence with per-axis K.
    // Discontinuous on the antipodal set theta = pi.
    [[nodiscard]] auto se3_lie_algebra_gradient(const manifold::SE3 &g_e,
                                                const Eigen::Vector3d &kp_pos,
                                                const Eigen::Vector3d &kp_rot)
        -> manifold::SE3::Twist;

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

}  // namespace xarm_geo
