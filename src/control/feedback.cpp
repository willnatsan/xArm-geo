#include <xarm_geo/control/feedback.h>

namespace xarm_geo {

    auto se3_lie_group_gradient(const manifold::SE3 &g_e, const Eigen::Vector3d &kp_pos,
                                const Eigen::Vector3d &kp_rot) -> manifold::SE3::Twist {

        // Body-frame gradient of the decoupled SE(3) navigation function
        // (Maithripala 2006 / Bullo-Murray 1999):
        //   Phi(g_e) = 0.5 * p_e^T K_p p_e + 0.5 * tr(K_R (I - R_e))
        //   nabla Phi = ( R_e^T K_p p_e ;  0.5 * (K_R R_e - R_e^T K_R)^vee )

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

        // Log-map gradient with right-Jacobian correction:
        //   nabla Phi_log = Ad_{g_e} * dr_exp(xi_e) * K * xi_e,   xi_e = log(g_e).

        const manifold::SE3::Twist xi_e = g_e.log();

        manifold::SE3::Twist K_xi_e;
        K_xi_e.head<3>() = kp_pos.cwiseProduct(xi_e.head<3>());
        K_xi_e.tail<3>() = kp_rot.cwiseProduct(xi_e.tail<3>());

        return g_e.Ad() * manifold::SE3::dr_exp(xi_e) * K_xi_e;
    }

    auto se3_lie_group_gradient_wrench(const manifold::SE3 &g_e, const Eigen::Vector3d &kp_pos,
                                       const Eigen::Vector3d &kp_rot) -> manifold::SE3::Wrench {
        return {se3_lie_group_gradient(g_e, kp_pos, kp_rot)};
    }

    auto se3_lie_algebra_gradient_wrench(const manifold::SE3 &g_e, const Eigen::Vector3d &kp_pos,
                                         const Eigen::Vector3d &kp_rot) -> manifold::SE3::Wrench {
        return {se3_lie_algebra_gradient(g_e, kp_pos, kp_rot)};
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

    auto compute_op_space_inertia(const Eigen::LLT<Eigen::MatrixXd> &M_llt,
                                  const manifold::SE3::Jacobian &J,
                                  Eigen::Matrix<double, 6, 6> &Lambda_out,
                                  Eigen::MatrixXd &M_inv_Jt_scratch, double damping,
                                  double min_det) noexcept -> bool {

        // Bail if M is not PD; caller drops Lambda-scaled terms this tick.
        if (M_llt.info() != Eigen::Success) { return false; }

        // M_op = J * M^-1 * J^T  via one triangular solve (no explicit M^-1).
        M_inv_Jt_scratch.noalias() = M_llt.solve(J.transpose());
        Eigen::Matrix<double, 6, 6> M_op;
        M_op.noalias() = J * M_inv_Jt_scratch;

        // DLS guard: |det(M_op)| collapses toward zero at task singularities;
        // adding damping^2 * I preserves the SPD structure and caps sigma_max
        // of the resulting Lambda at ~1/damping^2.
        const double det_M_op = M_op.determinant();
        if (std::abs(det_M_op) < min_det) { M_op.diagonal().array() += damping * damping; }

        // M_op is SPD by construction (J*M^-1*J^T with M PD); LLT-invert.
        Eigen::LLT<Eigen::Matrix<double, 6, 6>> M_op_llt(M_op);
        if (M_op_llt.info() != Eigen::Success) { return false; }
        Lambda_out = M_op_llt.solve(Eigen::Matrix<double, 6, 6>::Identity());

        return true;
    }

}  // namespace xarm_geo
