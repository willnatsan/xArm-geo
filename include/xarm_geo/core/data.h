#pragma once

#include <string>
#include <vector>

#include <xarm_geo/core/manifold.h>

namespace xarm_geo {

    // --- Joint Position & Velocity Limits ---

    struct JointLimits {
        double q_min = -2 * M_PI;
        double q_max = 2 * M_PI;
        double q_vel_max = M_PI;
    };

    // --- Immutable Model Data ---
    // Represents the physical properties and static configuration

    struct Model {
        // Metadata
        int dof;
        std::string urdf_file;
        std::string mjcf_file;

        // Kinematic Parameters
        manifold::SE3 home_pose;
        std::vector<manifold::SE3> home_pose_tree;
        std::vector<manifold::SE3::Twist> screw_axes_space;
        std::vector<manifold::SE3::Twist> screw_axes_local;

        // Dynamic Parameters
        std::vector<manifold::SE3::SpatialInertia> spatial_inertias_link;

        // Constraints
        std::vector<JointLimits> limits;
    };

    // --- Mutable Intermediate Data ---
    // Represents the state at a specific instance in time

    struct State {
        // --- Kinematic Cache ---
        manifold::SE3 ee_pose;
        std::vector<manifold::SE3> pose_tree;
        std::vector<manifold::SE3> pose_tree_local;

        // 6xN Jacobians mapping Joint Velocity -> SE(3) Tangent Space
        manifold::SE3::Jacobian space_jacobian;
        manifold::SE3::Jacobian body_jacobian;
        manifold::SE3::Jacobian frame_jacobian;

        // End-Effector twists
        manifold::SE3::Twist space_twist;
        manifold::SE3::Twist body_twist;
        manifold::SE3::Twist frame_twist;

        // Internal Link Twists and Accelerations (Needed for RNEA Forward Pass)
        std::vector<manifold::SE3::Twist> v_links;
        std::vector<manifold::SE3::SpatialAcceleration> a_links;

        // Local Transforms between Links (Needed for RNEA Backward Pass)
        std::vector<manifold::SE3> T_i_parent_cache;

        // --- Dynamic Cache ---
        Eigen::MatrixXd M;    // Joint-space Mass Matrix
        Eigen::VectorXd h;    // Bias Forces (Coriolis + Gravity)
        Eigen::VectorXd tau;  // Joint Torques (Results of Inverse Dynamics)
        Eigen::VectorXd a;    // Joint Accelerations (Results of Forward Dynamics)

        // Internal Spatial Wrenches (Needed for RNEA Backward Pass)
        std::vector<manifold::SE3::Wrench> f_links;

        // --- Constants & Simple Values ---
        Eigen::VectorXd v_zero;  // Zero Joint Velocity Vector
        Eigen::VectorXd a_zero;  // Zero Joint Acceleration Vector
        Eigen::VectorXd a_ei;    // Unit Joint Acceleration Vector

        // --- Pre-Allocation ---
        explicit State(const Model &model);
    };
}  // namespace xarm_geo
