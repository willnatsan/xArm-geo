#include <xarm_geo/core/data.h>

namespace xarm_geo {
    State::State(const Model &model) {
        pose_tree.resize(model.dof + 2);
        pose_tree_local.resize(model.dof + 2);

        space_jacobian.setZero(6, model.dof);
        body_jacobian.setZero(6, model.dof);
        frame_jacobian.setZero(6, model.dof);

        v_links.resize(model.dof + 2, manifold::SE3::Tangent::Zero());
        a_links.resize(model.dof + 2, manifold::SE3::Tangent::Zero());
        f_links.resize(model.dof + 2, manifold::SE3::Wrench());

        T_i_parent_cache.resize(model.dof + 1, manifold::SE3::Identity());

        M.setZero(model.dof, model.dof);
        h.setZero(model.dof);
        tau.setZero(model.dof);
        a.setZero(model.dof);

        v_zero.setZero(model.dof);
        a_zero.setZero(model.dof);
        a_ei.setZero(model.dof);
    }

}  // namespace xarm_geo
