#include <xarm_geo/control/tracking.h>

namespace xarm_geo {

    auto se3_lie_group_gradient(const manifold::SE3 &g_e, const Eigen::Vector3d &kp_pos,
                                const Eigen::Vector3d &kp_rot) -> manifold::SE3::Twist {

        // Decoupled SE(3) navigation function gradient in body frame of g
        // (where g_e = g^{-1} * g_d):
        //   grad_lin = R_e^T * K_p * p_e
        //   grad_ang = 0.5 * (K_R * R_e - R_e^T * K_R)^vee
        // Layout: [linear; angular] (matches `smooth`'s tangent layout; see
        // kinematics.cpp:97).

        const Eigen::Matrix3d R_e = g_e.so3().matrix();
        const Eigen::Vector3d p_e = g_e.r3();

        manifold::SE3::Twist grad;
        grad.head<3>().noalias() = R_e.transpose() * kp_pos.cwiseProduct(p_e);

        const Eigen::Matrix3d K_R = kp_rot.asDiagonal();
        const Eigen::Matrix3d S = 0.5 * (K_R * R_e - R_e.transpose() * K_R);
        grad[3] = S(2, 1);
        grad[4] = S(0, 2);
        grad[5] = S(1, 0);

        return grad;
    }

    auto se3_lie_algebra_gradient(const manifold::SE3 &g_e, const Eigen::Vector3d &kp_pos,
                                  const Eigen::Vector3d &kp_rot) -> manifold::SE3::Twist {

        // Log-map gradient with Jacobian correction:
        //   nabla Phi_log = - Ad_{g_e} * dr_exp(xi_e) * K * xi_e   where xi_e = log(g_e).
        //
        // The Ad and dr_exp factors collapse to identity when K = k*I (BCH commutation).

        const manifold::SE3::Twist xi_e = g_e.log();

        manifold::SE3::Twist K_xi_e;
        K_xi_e.head<3>() = kp_pos.cwiseProduct(xi_e.head<3>());
        K_xi_e.tail<3>() = kp_rot.cwiseProduct(xi_e.tail<3>());

        return -(g_e.Ad() * manifold::SE3::dr_exp(xi_e) * K_xi_e);
    }

    auto se3_velocity_error(const manifold::SE3::Twist &body_twist, const manifold::SE3 &g_e,
                            const manifold::SE3::Twist &target_twist_body) -> manifold::SE3::Twist {

        // Body-frame velocity error: xi_e = xi - Ad_{g_e} * xi_d.
        return body_twist - (g_e.Ad() * target_twist_body);
    }

    auto se3_transported_acc(const manifold::SE3 &g_e, const manifold::SE3::Twist &xi_e,
                             const manifold::SE3::Twist &ad_xi_d,
                             const manifold::SE3::SpatialAcceleration &spatial_acc_body)
        -> manifold::SE3::Twist {

        // Closed-form time derivative of the transported reference twist:
        //   d/dt(Ad_{g_e} * xi_d) = -ad_{xi_e}(Ad_{g_e} * xi_d) + Ad_{g_e} * a_d
        // (See header for the equivalence with -ad_{xi}; we take xi_e here.)
        return (-manifold::SE3::ad(xi_e) * ad_xi_d) + (g_e.Ad() * spatial_acc_body);
    }

}  // namespace xarm_geo
