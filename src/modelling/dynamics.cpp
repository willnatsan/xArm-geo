#include <cassert>
#include <iostream>

#include <xarm_geo/modelling/dynamics.h>

namespace xarm_geo {
    void forward_dynamics(const Model &model, Data &data,
                          const Eigen::Ref<const Eigen::VectorXd> &v,
                          const Eigen::Ref<const Eigen::VectorXd> &tau,
                          const manifold::SE3::Wrench &ee_wrench) {

        assert(v.size() == model.dof && "v size must equal model.dof");
        assert(tau.size() == model.dof && "tau size must equal model.dof");

        // Compute Bias Forces & Mass Matrix
        compute_bias_forces(model, data, v, ee_wrench);
        compute_mass_matrix(model, data);

        // Rearrange Dynamics Equation to solve for Joint Accelerations
        // tau = M(q) * a + h
        // M(q) * a = tau - h
        Eigen::VectorXd tau_diff = tau - data.h;

        // Solve for Joint Accelerations w/ Cholesky Decomposition
        Eigen::LLT<Eigen::MatrixXd> M_llt(data.M);
#ifndef NDEBUG
        if (M_llt.info() != Eigen::Success) {
            std::cerr << "[xarm_geo::forward_dynamics] Cholesky failed on data.M (not PD).\n";
        }
#endif
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
        // Setting a_links[0] = (-g, 0) folds gravity into RNEA. With
        // model.gravity == 0 (default) this matches the externally-compensated
        // convention.
        data.rnea.v_links[0] = manifold::SE3::Twist::Zero();
        manifold::SE3::SpatialAcceleration a0 = manifold::SE3::SpatialAcceleration::Zero();
        a0.head<3>() = -model.gravity;
        data.rnea.a_links[0] = a0;

        // --- Forward Pass (Iterate through moving links 1 to model.dof) ---
        for (int i = 0; i < model.dof; ++i) {
            int link_idx = i + 1;  // Current link (1 to dof)
            int parent_idx = i;    // Parent link (0 to dof-1)

            // Cache Transform from Current Frame to Parent Frame
            data.rnea.T_i_parent_cache[i] = data.pose_tree_local[link_idx].inverse();
            auto Ad_T = data.rnea.T_i_parent_cache[i].Ad();

            // Compute Spatial Twist for Current Frame
            manifold::SE3::Twist joint_v = model.screw_axes_local[i] * v[i];
            data.rnea.v_links[link_idx] = (Ad_T * data.rnea.v_links[parent_idx]) + joint_v;

            // Compute Spatial Acceleration for Current Frame
            manifold::SE3::SpatialAcceleration joint_a = model.screw_axes_local[i] * a[i];
            auto ad_V = manifold::SE3::ad(data.rnea.v_links[link_idx]);

            data.rnea.a_links[link_idx] =
                (Ad_T * data.rnea.a_links[parent_idx]) + (ad_V * joint_v) + joint_a;
        }

        // Cache the End-Effector to Last-Link Transform
        data.rnea.T_i_parent_cache[model.dof] = data.pose_tree_local[model.dof + 1].inverse();

        // --- Backward Pass (Iterate from model.dof down to 1) ---

        // Apply External Wrench at the End-Effector (Index model.dof + 1)
        data.rnea.f_links[model.dof + 1] = ee_wrench;

        for (int i = model.dof - 1; i >= 0; --i) {
            int link_idx = i + 1;
            int child_idx = i + 2;

            // Compute Coriolis Wrench
            auto ad_V_T = manifold::SE3::ad(data.rnea.v_links[link_idx]).transpose();
            manifold::SE3::Wrench F_coriolis(ad_V_T * model.spatial_inertias_link[i] *
                                             data.rnea.v_links[link_idx]);

            // Compute Inertial Wrench
            manifold::SE3::Wrench F_inertial(model.spatial_inertias_link[i] *
                                             data.rnea.a_links[link_idx]);

            // Project Child Wrench to Current Link Frame
            auto Ad_T_child = data.rnea.T_i_parent_cache[i + 1].Ad();
            manifold::SE3::Wrench F_child_pulled_back(Ad_T_child.transpose() *
                                                      data.rnea.f_links[child_idx].coeffs);

            // Compute Total Wrench on Link i
            data.rnea.f_links[link_idx] = F_child_pulled_back + F_inertial - F_coriolis;

            // Extract Joint Torques
            data.tau_out[i] = data.rnea.f_links[link_idx].dot(model.screw_axes_local[i]);
        }
    }

    void compute_mass_matrix(const Model &model, Data &data) {

        assert(static_cast<int>(data.pose_tree_local.size()) == model.dof + 2 &&
               "pose_tree_local not populated; run forward_kinematics first");

        // Composite Rigid Body Algorithm (Featherstone), body-frame variant.
        // Convention: joint i drives link i+1; link k's parent is link k-1.
        // Requires pose_tree_local to be up-to-date (call forward_kinematics
        // or compute_jacobians first). M is symmetric by construction.

        const int dof = model.dof;

        // Cache Child-to-Parent Transforms (Index k-1 -> link k frame to link k-1 frame)
        for (int k = 1; k <= dof; ++k) {
            data.rnea.T_i_parent_cache[k - 1] = data.pose_tree_local[k].inverse();
        }

        // Seed Composite Inertias w/ Each Link's Spatial Inertia
        for (int i = 0; i < dof; ++i) { data.crba.I_C[i] = model.spatial_inertias_link[i]; }

        // Leaves-to-Base Composite-Inertia Accumulation
        // I_C[parent] += Ad^T * I_C[child] * Ad
        for (int k = dof; k >= 2; --k) {
            const auto Ad = data.rnea.T_i_parent_cache[k - 1].Ad();
            data.crba.I_C[k - 2].noalias() += Ad.transpose() * data.crba.I_C[k - 1] * Ad;
        }

        // Write Columns of M (Wrench F = I_C[i+1] * S_i, transported up the chain)
        data.M.setZero();
        for (int i = 0; i < dof; ++i) {
            data.crba.F_scratch.coeffs.noalias() = data.crba.I_C[i] * model.screw_axes_local[i];

            // Diagonal Entry
            data.M(i, i) = data.crba.F_scratch.coeffs.dot(model.screw_axes_local[i]);

            // Off-Diagonals: Transport F up the chain, dot w/ each ancestor's screw
            for (int j = i; j >= 1; --j) {
                const auto Ad_T = data.rnea.T_i_parent_cache[j].Ad().transpose();
                data.crba.F_scratch.coeffs = Ad_T * data.crba.F_scratch.coeffs;

                const double m_ij = data.crba.F_scratch.coeffs.dot(model.screw_axes_local[j - 1]);
                data.M(i, j - 1) = m_ij;
                data.M(j - 1, i) = m_ij;
            }
        }
    };

    void compute_bias_forces(const Model &model, Data &data,
                             const Eigen::Ref<const Eigen::VectorXd> &v,
                             const manifold::SE3::Wrench &ee_wrench) {

        assert(v.size() == model.dof && "v size must equal model.dof");

        // Evaluating the Inverse Dynamics w/ Zero Joint Accelerations
        // The equation simplifies to: tau = h(q, v)
        // (h includes gravity when `model.gravity` is non-zero)
        inverse_dynamics(model, data, v, data.bias.a_zero, ee_wrench);

        // Store resulting Joint Torques as the Bias Forces `h`
        data.h = data.tau_out;
    };
}  // namespace xarm_geo
