#include <xarm_geo/core/data.h>

namespace xarm_geo {
    State::State(const Model &model) {
        pose_tree.resize(model.dof + 1);
        pose_tree_local.resize(model.dof + 1);

        space_jacobian.setZero(6, model.dof);
        body_jacobian.setZero(6, model.dof);
        frame_jacobian.setZero(6, model.dof);

        v_links.resize(model.dof + 1, manifold::SE3::Tangent::Zero());
        a_links.resize(model.dof + 1, manifold::SE3::Tangent::Zero());
        f_links.resize(model.dof + 1, manifold::SE3::Wrench());

        M.setZero(model.dof, model.dof);
        h.setZero(model.dof);
        tau.setZero(model.dof);
        q_accel.setZero(model.dof);
    }

}  // namespace xarm_geo
