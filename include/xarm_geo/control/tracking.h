#pragma once

#include <Eigen/Dense>

#include <xarm_geo/core/manifold.h>

namespace xarm_geo {

    // --- SE(3) Tracking Gains ---
    //
    // Per-axis diagonal gains (body-frame). Defaults are zero -- a
    // default-constructed SE3TrackingGains yields a no-op controller; users
    // must set gains explicitly. Empty defaults surface mistuning loudly.

    struct SE3TrackingGains {
        Eigen::Vector3d kp_pos = Eigen::Vector3d::Zero();  // K_p (linear, body frame)
        Eigen::Vector3d kp_rot = Eigen::Vector3d::Zero();  // K_R (angular, body frame)
        Eigen::Vector3d kd_lin = Eigen::Vector3d::Zero();  // K_D linear (PD only)
        Eigen::Vector3d kd_ang = Eigen::Vector3d::Zero();  // K_D angular (PD only)
    };

    // --- SE(3) Tracking: Free-Function Building Blocks ---
    //
    // Body-frame primitives for SE(3) tracking controllers. These helpers are
    // pure functions: stateless, allocation-free, and independent of `Model`
    // / `Data`. Reusable in any controller variant (P, PD, PID, MPC,
    // impedance, ...).
    //
    // Frame conventions (consistent with the rest of the library):
    //   - g_e = g^{-1} * g_d         (body-frame configuration error;
    //                                 lives in the tangent space at the
    //                                 current end-effector pose g)
    //   - All twists / wrenches are expressed in the body frame of the
    //     end-effector unless stated otherwise.
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

    // SE(3) log-map gradient in the body frame of g, with Jacobian correction.
    //
    //     nabla Phi_log(g_e) = - Ad_{g_e} * dr_exp(log(g_e)) * K * log(g_e)^vee
    //
    // Reduces to -k * log(g_e) under scalar isotropic gain (Prabhu-Saxena-Sastry
    // 2020); the correction restores exponential closed-loop convergence with
    // per-axis K. Discontinuous on the antipodal set theta = pi (smooth's
    // atan2-based log returns +/-pi*n_hat there).
    [[nodiscard]] auto se3_lie_algebra_gradient(const manifold::SE3 &g_e,
                                                const Eigen::Vector3d &kp_pos,
                                                const Eigen::Vector3d &kp_rot)
        -> manifold::SE3::Twist;

    // Body-frame velocity error.
    //
    //     xi_e = xi - Ad_{g_e} * xi_d
    //
    // `body_twist` is the current end-effector body-frame twist (J_b * v).
    // `target_twist_body` is the reference body-frame twist (lives in the
    // body frame of g_d; the Ad_{g_e} transport pulls it into the body frame
    // of g for comparison).
    [[nodiscard]] auto se3_velocity_error(const manifold::SE3::Twist &body_twist,
                                          const manifold::SE3 &g_e,
                                          const manifold::SE3::Twist &target_twist_body)
        -> manifold::SE3::Twist;

    // Closed-form time derivative of the transported reference twist.
    //
    //     d/dt(Ad_{g_e} * xi_d) = -ad_{xi}(Ad_{g_e} * xi_d) + Ad_{g_e} * a_d
    //
    // Using ad_v(v) = 0 together with the velocity-error decomposition
    // xi = xi_e + Ad_{g_e} * xi_d, this is equivalent to
    //
    //     d/dt(Ad_{g_e} * xi_d) = -ad_{xi_e}(Ad_{g_e} * xi_d) + Ad_{g_e} * a_d
    //
    // We take xi_e to avoid re-passing xi (already consumed by the K_D term
    // in the dynamic PD law). `ad_xi_d` is the cached value of Ad_{g_e} *
    // xi_d (avoids recomputing). `spatial_acc_body` is the reference
    // body-frame spatial acceleration (= d/dt(xi_d), as written by every
    // TaskSpaceTrajectory in src/trajectory/sample_trajectories.cpp via
    // smooth's BSpline analytical derivative).
    [[nodiscard]] auto
    se3_transported_acc(const manifold::SE3 &g_e, const manifold::SE3::Twist &xi_e,
                        const manifold::SE3::Twist &ad_xi_d,
                        const manifold::SE3::SpatialAcceleration &spatial_acc_body)
        -> manifold::SE3::Twist;

}  // namespace xarm_geo
