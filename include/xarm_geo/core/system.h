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
        // --- Metadata ---
        int dof;
        std::string urdf_file;
        std::string mjcf_file;

        // --- Kinematic Parameters ---
        manifold::SE3 home_pose;
        std::vector<manifold::SE3> home_pose_tree;
        std::vector<manifold::SE3::Twist> screw_axes_space;
        std::vector<manifold::SE3::Twist> screw_axes_local;

        // --- Dynamic Parameters ---
        std::vector<manifold::SE3::SpatialInertia> spatial_inertias_link;

        // --- Constraints ---
        std::vector<JointLimits> limits;
    };

    // --- Mutable Intermediate Data ---
    // Pre-Allocated Memory Cache & Output Buffer for ALL Algorithms

    struct Data {
        // --- Kinematics Outputs ---
        manifold::SE3 ee_pose;
        std::vector<manifold::SE3> pose_tree;
        std::vector<manifold::SE3> pose_tree_local;

        manifold::SE3::Jacobian space_jacobian;
        manifold::SE3::Jacobian body_jacobian;
        manifold::SE3::Jacobian frame_jacobian;

        manifold::SE3::Twist space_twist;
        manifold::SE3::Twist body_twist;
        manifold::SE3::Twist frame_twist;

        Eigen::VectorXd q_out;  // Results of Inverse Kinematics
        Eigen::VectorXd v_out;  // Results of Inverse Differential Kinematics

        // --- Dynamics Outputs ---
        Eigen::MatrixXd M;        // Joint-space Mass Matrix
        Eigen::VectorXd h;        // Bias Forces (Coriolis + Gravity)
        Eigen::VectorXd tau_out;  // Results of Inverse Dynamics
        Eigen::VectorXd a_out;    // Results of Forward Dynamics

        // --- Internal Algorithm Workspaces (Zero-Allocation Scratchpads) ---

        // Workspace for Inverse Kinematics (Damped Least Squares)
        struct IKWorkspace {
            Eigen::MatrixXd A;  // DLS matrix: J^T * J + lambda^2 * I
            Eigen::VectorXd b;  // Target vector: J^T * V_err
        } ik;

        // Workspace for Inverse Dynamics (RNEA)
        struct RNEAWorkspace {
            std::vector<manifold::SE3::Twist> v_links;
            std::vector<manifold::SE3::SpatialAcceleration> a_links;
            std::vector<manifold::SE3::Wrench> f_links;
            std::vector<manifold::SE3> T_i_parent_cache;
        } rnea;

        // Workspace for Forward Dynamcis (CRBA)
        struct CRBAWorkspace {
            Eigen::VectorXd v_zero;  // Cached zero velocity vector
            Eigen::VectorXd a_zero;  // Cached zero acceleration vector
            Eigen::VectorXd a_ei;    // Moving unit acceleration vector
        } crba;

        // --- Pre-Allocation Constructor ---
        explicit Data(const Model &model);
    };
}  // namespace xarm_geo
