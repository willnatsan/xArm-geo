#include <random>

#include <xarm_geo/core/system.h>

namespace xarm_geo {
    Data::Data(const Model &model) : rng(std::random_device{}()) {
        int dof = model.dof;

        // --- Public Outputs ---
        q.setZero(dof);

        pose_tree.resize(dof + 2, manifold::SE3::Identity());
        pose_tree_local.resize(dof + 2, manifold::SE3::Identity());

        space_jacobian.setZero(6, dof);
        body_jacobian.setZero(6, dof);
        frame_jacobian.setZero(6, dof);

        q_guess.setZero(dof);
        q_out.setZero(dof);
        v_out.setZero(dof);

        M.setZero(dof, dof);
        h.setZero(dof);
        g.setZero(dof);
        tau_out.setZero(dof);
        a_out.setZero(dof);

        // --- Internal Workspaces ---
        ik.A.setZero(dof, dof);
        ik.b.setZero(dof);

        rnea.v_links.resize(dof + 2, manifold::SE3::Tangent::Zero());
        rnea.a_links.resize(dof + 2, manifold::SE3::Tangent::Zero());
        rnea.f_links.resize(dof + 2, manifold::SE3::Wrench());
        rnea.T_i_parent_cache.resize(dof + 1, manifold::SE3::Identity());

        bias.a_zero.setZero(dof);

        coriolis.v_sum.setZero(dof);
        coriolis.b_sum.setZero(dof);
        coriolis.b_a.setZero(dof);
        coriolis.b_b.setZero(dof);

        crba.I_C.assign(dof, manifold::SE3::SpatialInertia::Zero());
        crba.F_scratch = manifold::SE3::Wrench();

        collision.point_jacobian_1.setZero(3, dof);
        collision.point_jacobian_2.setZero(3, dof);

        // OptIK: H/g sized to dof; A/l/u grow on demand as constraints register.
        optik.H.setZero(dof, dof);
        optik.g.setZero(dof);
        optik.A.resize(0, dof);
        optik.l.resize(0);
        optik.u.resize(0);

        // Per-task scratch: 6 rows for the default FrameTask on SE(3).
        optik.J_task.setZero(6, dof);
        optik.e_task.setZero(6);

        optik.qp.reset();
        optik.current_n = 0;
        optik.current_m_eq = 0;
        optik.current_m_in = 0;
        optik.initialised = false;

        asif.H.setZero(dof, dof);
        asif.g.setZero(dof);
        asif.A.resize(0, dof);
        asif.l.resize(0);
        asif.u.resize(0);

        asif.M_llt = Eigen::LLT<Eigen::MatrixXd>(dof);
        asif.M_inv.setZero(dof, dof);
        asif.M_inv_h.setZero(dof);

        asif.qp.reset();
        asif.current_n = 0;
        asif.current_m_eq = 0;
        asif.current_m_in = 0;
        asif.initialised = false;
    }
}  // namespace xarm_geo
