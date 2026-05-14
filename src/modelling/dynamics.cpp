#include <cassert>

#include <xarm_geo/modelling/dynamics.h>
#include <xarm_geo/utils/debug.h>

namespace xarm_geo {
    void forward_dynamics(const Model &model, Data &data,
                          const Eigen::Ref<const Eigen::VectorXd> &v,
                          const Eigen::Ref<const Eigen::VectorXd> &tau,
                          const manifold::SE3::Wrench &ee_wrench) {

        assert(v.size() == model.dof && "v size must equal model.dof");
        assert(tau.size() == model.dof && "tau size must equal model.dof");

        compute_bias_forces(model, data, v, ee_wrench);
        compute_mass_matrix(model, data);

        // Solve M(q) * a = tau - h for the joint accelerations.
        Eigen::VectorXd tau_diff = tau - data.h;
        Eigen::LLT<Eigen::MatrixXd> M_llt(data.M);

        if (M_llt.info() != Eigen::Success) { debug::log("Cholesky failed on data.M (not PD)"); }

        data.a_out = M_llt.solve(tau_diff);
    };

    void inverse_dynamics(const Model &model, Data &data,
                          const Eigen::Ref<const Eigen::VectorXd> &v,
                          const Eigen::Ref<const Eigen::VectorXd> &a,
                          const manifold::SE3::Wrench &ee_wrench) {

        assert(v.size() == model.dof && "v size must equal model.dof");
        assert(a.size() == model.dof && "a size must equal model.dof");
        assert(static_cast<int>(data.pose_tree_local.size()) == model.dof + 2 &&
               "pose_tree_local not populated; run forward_kinematics first");

        // --- Initial Conditions ---
        //
        // a_links[0] = (-g, 0) folds gravity into RNEA. With model.gravity == 0
        // (default) this matches the externally-compensated convention.
        data.rnea.v_links[0] = manifold::SE3::Twist::Zero();
        manifold::SE3::SpatialAcceleration a0 = manifold::SE3::SpatialAcceleration::Zero();
        a0.head<3>() = -model.gravity;
        data.rnea.a_links[0] = a0;

        // --- Forward Pass (Links 1 .. model.dof) ---
        //
        // Joint i drives link i+1; cache child-to-parent transforms in T_i_parent_cache.
        for (int i = 0; i < model.dof; ++i) {
            int link_idx = i + 1;
            int parent_idx = i;

            data.rnea.T_i_parent_cache[i] = data.pose_tree_local[link_idx].inverse();
            auto Ad_T = data.rnea.T_i_parent_cache[i].Ad();

            manifold::SE3::Twist joint_v = model.screw_axes_local[i] * v[i];
            data.rnea.v_links[link_idx] = (Ad_T * data.rnea.v_links[parent_idx]) + joint_v;

            manifold::SE3::SpatialAcceleration joint_a = model.screw_axes_local[i] * a[i];
            auto ad_V = manifold::SE3::ad(data.rnea.v_links[link_idx]);

            data.rnea.a_links[link_idx] =
                (Ad_T * data.rnea.a_links[parent_idx]) + (ad_V * joint_v) + joint_a;
        }

        // End-effector to last-link transform.
        data.rnea.T_i_parent_cache[model.dof] = data.pose_tree_local[model.dof + 1].inverse();

        // --- Backward Pass (model.dof .. 1) ---
        //
        // Apply external wrench at the end-effector (index model.dof + 1), then
        // walk back assembling Coriolis + inertial + child wrenches per link
        // and project onto the screw axis to recover joint torques.
        data.rnea.f_links[model.dof + 1] = ee_wrench;

        for (int i = model.dof - 1; i >= 0; --i) {
            int link_idx = i + 1;
            int child_idx = i + 2;

            auto ad_V_T = manifold::SE3::ad(data.rnea.v_links[link_idx]).transpose();
            manifold::SE3::Wrench F_coriolis(ad_V_T * model.spatial_inertias_link[i] *
                                             data.rnea.v_links[link_idx]);

            manifold::SE3::Wrench F_inertial(model.spatial_inertias_link[i] *
                                             data.rnea.a_links[link_idx]);

            auto Ad_T_child = data.rnea.T_i_parent_cache[i + 1].Ad();
            manifold::SE3::Wrench F_child_pulled_back =
                Ad_T_child.transpose() * data.rnea.f_links[child_idx];

            data.rnea.f_links[link_idx] = F_child_pulled_back + F_inertial - F_coriolis;
            data.tau_out[i] = data.rnea.f_links[link_idx].dot(model.screw_axes_local[i]);
        }
    }

    void compute_mass_matrix(const Model &model, Data &data) {

        assert(static_cast<int>(data.pose_tree_local.size()) == model.dof + 2 &&
               "pose_tree_local not populated; run forward_kinematics first");

        // Composite Rigid Body Algorithm (Featherstone), body-frame variant.
        // Joint i drives link i+1; link k's parent is link k-1. Requires
        // pose_tree_local to be fresh (forward_kinematics or compute_jacobians).
        // M is symmetric by construction.

        const int dof = model.dof;

        for (int k = 1; k <= dof; ++k) {
            data.rnea.T_i_parent_cache[k - 1] = data.pose_tree_local[k].inverse();
        }

        for (int i = 0; i < dof; ++i) { data.crba.I_C[i] = model.spatial_inertias_link[i]; }

        // Leaves-to-base composite-inertia accumulation: I_C[parent] += Ad^T * I_C[child] * Ad.
        for (int k = dof; k >= 2; --k) {
            const auto Ad = data.rnea.T_i_parent_cache[k - 1].Ad();
            data.crba.I_C[k - 2].noalias() += Ad.transpose() * data.crba.I_C[k - 1] * Ad;
        }

        // Columns of M: F = I_C[i] * S_i transported up the chain and dotted with each ancestor.
        data.M.setZero();
        for (int i = 0; i < dof; ++i) {
            data.crba.F_scratch.noalias() = data.crba.I_C[i] * model.screw_axes_local[i];

            data.M(i, i) = data.crba.F_scratch.dot(model.screw_axes_local[i]);

            for (int j = i; j >= 1; --j) {
                const manifold::SE3::TangentMap Ad = data.rnea.T_i_parent_cache[j].Ad();
                data.crba.F_scratch = Ad.transpose() * data.crba.F_scratch;

                const double m_ij = data.crba.F_scratch.dot(model.screw_axes_local[j - 1]);
                data.M(i, j - 1) = m_ij;
                data.M(j - 1, i) = m_ij;
            }
        }

        // Add motor-armature reflected inertia diagonally (Featherstone §6.1).
        // CRBA's link-only pass omits rotor inertia. Guard allows callers that
        // omit joint_armature to see no change.
        if (model.joint_armature.size() == model.dof) {
            for (int i = 0; i < model.dof; ++i) { data.M(i, i) += model.joint_armature[i]; }
        }
    };

    void compute_bias_forces(const Model &model, Data &data,
                             const Eigen::Ref<const Eigen::VectorXd> &v,
                             const manifold::SE3::Wrench &ee_wrench) {

        assert(v.size() == model.dof && "v size must equal model.dof");

        // Inverse dynamics with a = 0 gives tau = h(q, v) (includes gravity when nonzero).
        inverse_dynamics(model, data, v, data.bias.a_zero, ee_wrench);
        data.h = data.tau_out;
    };

    void compute_gravity_forces(const Model &model, Data &data) {

        // Inverse dynamics with v = 0 and a = 0 gives tau = g(q).
        // Reuses bias.a_zero as both arguments to avoid an extra zero vector.
        inverse_dynamics(model, data, data.bias.a_zero, data.bias.a_zero);
        data.g = data.tau_out;
    };

    void compute_coriolis_times(const Model &model, Data &data,
                                const Eigen::Ref<const Eigen::VectorXd> &v_a,
                                const Eigen::Ref<const Eigen::VectorXd> &v_b,
                                Eigen::Ref<Eigen::VectorXd> out, bool g_fresh) {

        assert(v_a.size() == model.dof && "v_a size must equal model.dof");
        assert(v_b.size() == model.dof && "v_b size must equal model.dof");
        assert(out.size() == model.dof && "out size must equal model.dof");

        // Polarisation identity for the symmetric Levi-Civita form:
        //   2 * C(q, v_a) * v_b = RNEA(q, v_a+v_b, 0) - RNEA(q, v_a, 0) - RNEA(q, v_b, 0) + g(q).
        // Each RNEA(q, v, 0) returns C(q, v) * v + g(q); the combination
        // cancels the three g(q) terms and isolates 2 * C(q, v_a) * v_b.

        inverse_dynamics(model, data, v_a, data.bias.a_zero);
        data.coriolis.b_a = data.tau_out;

        inverse_dynamics(model, data, v_b, data.bias.a_zero);
        data.coriolis.b_b = data.tau_out;

        data.coriolis.v_sum = v_a + v_b;
        inverse_dynamics(model, data, data.coriolis.v_sum, data.bias.a_zero);
        data.coriolis.b_sum = data.tau_out;

        // Skip the gravity recompute when the caller has already populated data.g (4 -> 3 RNEA).
        if (!g_fresh) { compute_gravity_forces(model, data); }

        out.noalias() =
            0.5 * (data.coriolis.b_sum - data.coriolis.b_a - data.coriolis.b_b + data.g);
    };
}  // namespace xarm_geo
