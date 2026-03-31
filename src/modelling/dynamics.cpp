#include "xarm_geo/core/manifold.h"
#include <xarm_geo/modelling/dynamics.h>
#include <xarm_geo/modelling/kinematics.h>

namespace xarm_geo {
    void forward_dynamics(const Model &model, State &state,
                          const Eigen::Ref<const Eigen::VectorXd> &q,
                          const Eigen::Ref<const Eigen::VectorXd> &v,
                          const Eigen::Ref<const Eigen::VectorXd> &a) {};

    void inverse_dynamics(const Model &model, State &state,
                          const Eigen::Ref<const Eigen::VectorXd> &q,
                          const Eigen::Ref<const Eigen::VectorXd> &v,
                          const Eigen::Ref<const Eigen::VectorXd> &a,
                          const manifold::SE3::Wrench &ee_wrench) {

        // --- Initial Conditions (Assuming Gravity Term is Compensated Externally) ---
        state.v_links[0] = manifold::SE3::Twist::Zero();
        state.a_links[0] = manifold::SE3::SpatialAcceleration::Zero();

        // Computing Forward Kinematics to get Link Frames for this Configuration
        forward_kinematics(model, state, q);

        // --- Forward Pass (Iterate through moving links 1 to model.dof) ---
        for (int i = 0; i < model.dof; ++i) {
            int link_idx = i + 1;  // Current link (1 to dof)
            int parent_idx = i;    // Parent link (0 to dof-1)

            // Cache Transform from Current Frame to Parent Frame
            state.T_i_parent_cache[i] = state.pose_tree_local[link_idx].inverse();
            auto Ad_T = state.T_i_parent_cache[i].Ad();

            // Compute Spatial Twist for Current Frame
            manifold::SE3::Twist joint_v = model.screw_axes_local[i] * v[i];
            state.v_links[link_idx] = (Ad_T * state.v_links[parent_idx]) + joint_v;

            // Compute Spatial Acceleration for Current Frame
            manifold::SE3::SpatialAcceleration joint_a = model.screw_axes_local[i] * a[i];
            auto ad_V = manifold::SE3::ad(state.v_links[link_idx]);

            state.a_links[link_idx] =
                (Ad_T * state.a_links[parent_idx]) + (ad_V * joint_v) + joint_a;
        }

        // Cache the End-Effector to Last-Link Transform
        state.T_i_parent_cache[model.dof] = state.pose_tree_local[model.dof + 1].inverse();

        // --- Backward Pass (Iterate from model.dof down to 1) ---

        // Apply External Wrench at the End-Effector (Index model.dof + 1)
        state.f_links[model.dof + 1] = ee_wrench;

        for (int i = model.dof - 1; i >= 0; --i) {
            int link_idx = i + 1;
            int child_idx = i + 2;

            // Compute Coriolis Wrench
            auto ad_V_T = manifold::SE3::ad(state.v_links[link_idx]).transpose();
            manifold::SE3::Wrench F_coriolis(ad_V_T * model.spatial_inertias_link[i] *
                                             state.v_links[link_idx]);

            // Compute Inertial Wrench
            manifold::SE3::Wrench F_inertial(model.spatial_inertias_link[i] *
                                             state.a_links[link_idx]);

            // Project Child Wrench to Current Link Frame
            auto Ad_T_child = state.T_i_parent_cache[i + 1].Ad();
            manifold::SE3::Wrench F_child_pulled_back(Ad_T_child.transpose() *
                                                      state.f_links[child_idx].coeffs);

            // Compute Total Wrench on Link i
            state.f_links[link_idx] = F_child_pulled_back + F_inertial - F_coriolis;

            // Extract Joint Torques
            state.tau[i] = state.f_links[link_idx].dot(model.screw_axes_local[i]);
        }
    }
}  // namespace xarm_geo
