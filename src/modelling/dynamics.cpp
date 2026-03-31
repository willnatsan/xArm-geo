#include <xarm_geo/modelling/dynamics.h>

namespace xarm_geo {
    void forward_dynamics(const Model &model, Data &data,
                          const Eigen::Ref<const Eigen::VectorXd> &q,
                          const Eigen::Ref<const Eigen::VectorXd> &v,
                          const Eigen::Ref<const Eigen::VectorXd> &tau,
                          const manifold::SE3::Wrench &ee_wrench) {

        // Compute Bias Forces & Mass Matrix
        compute_bias_forces(model, data, q, v, ee_wrench);
        compute_mass_matrix(model, data, q);

        // Rearrange Dynamics Equation to solve for Joint Accelerations
        // tau = M(q) * a + h
        // M(q) * a = tau - h
        Eigen::VectorXd tau_diff = tau - data.h;

        // Solve for Joint Accelerations w/ Cholesky Decomposition
        data.a_out = data.M.llt().solve(tau_diff);
    };

    void inverse_dynamics(const Model &model, Data &data,
                          const Eigen::Ref<const Eigen::VectorXd> &q,
                          const Eigen::Ref<const Eigen::VectorXd> &v,
                          const Eigen::Ref<const Eigen::VectorXd> &a,
                          const manifold::SE3::Wrench &ee_wrench) {

        // --- Initial Conditions (Assuming Gravity Term is Compensated Externally) ---
        data.rnea.v_links[0] = manifold::SE3::Twist::Zero();
        data.rnea.a_links[0] = manifold::SE3::SpatialAcceleration::Zero();

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

    void compute_mass_matrix(const Model &model, Data &data,
                             const Eigen::Ref<const Eigen::VectorXd> &q) {

        // Evaluating the Inverse Dynamics w/ Zero Joint Velocities + Unit Joint Accelerations
        // The equation simplifies to: tau = M(q) * a
        for (int i = 0; i < model.dof; ++i) {
            data.crba.a_ei[i] = 1.0;

            // Resulting Joint Torque equals i-th column of Mass Matrix
            inverse_dynamics(model, data, q, data.crba.v_zero, data.crba.a_ei);
            data.M.col(i) = data.tau_out;

            data.crba.a_ei[i] = 0.0;  // Reset for next iteration
        }

        // Enforcing Symmetry for Numerical Stability
        data.M = 0.5 * (data.M + data.M.transpose());
    };

    void compute_bias_forces(const Model &model, Data &data,
                             const Eigen::Ref<const Eigen::VectorXd> &q,
                             const Eigen::Ref<const Eigen::VectorXd> &v,
                             const manifold::SE3::Wrench &ee_wrench) {

        // Evaluating the Inverse Dynamics w/ Zero Joint Accelerations
        // The equation simplifies to: tau = h(q, v)
        inverse_dynamics(model, data, q, v, data.crba.a_zero, ee_wrench);

        // Store resulting Joint Torques as the Bias Forces `h`
        data.h = data.tau_out;
    };
}  // namespace xarm_geo
