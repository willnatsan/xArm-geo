#include <xarm_geo/core/system.h>

namespace xarm_geo {
    Data::Data(const Model &model) {
        int dof = model.dof;

        // --- Kinematics Outputs ---
        pose_tree.resize(dof + 2, manifold::SE3::Identity());
        pose_tree_local.resize(dof + 2, manifold::SE3::Identity());

        space_jacobian.setZero(6, dof);
        body_jacobian.setZero(6, dof);
        frame_jacobian.setZero(6, dof);

        q_guess.setZero(dof);
        q_out.setZero(dof);
        v_out.setZero(dof);

        // --- Dynamics Outputs ---
        M.setZero(dof, dof);
        h.setZero(dof);
        tau_out.setZero(dof);
        a_out.setZero(dof);

        // --- Internal IK Workspace ---
        ik.A.setZero(dof, dof);
        ik.b.setZero(dof);

        // --- Internal RNEA Workspace ---
        rnea.v_links.resize(dof + 2, manifold::SE3::Tangent::Zero());
        rnea.a_links.resize(dof + 2, manifold::SE3::Tangent::Zero());
        rnea.f_links.resize(dof + 2, manifold::SE3::Wrench());
        rnea.T_i_parent_cache.resize(dof + 1, manifold::SE3::Identity());

        // --- Internal CRBA Workspace ---
        crba.v_zero.setZero(dof);
        crba.a_zero.setZero(dof);
        crba.a_ei.setZero(dof);
    }
}  // namespace xarm_geo
