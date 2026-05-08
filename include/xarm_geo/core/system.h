#pragma once

#include <limits>
#include <memory>
#include <numbers>
#include <random>
#include <string>
#include <vector>

#include <proxsuite/proxqp/dense/dense.hpp>

#include <xarm_geo/core/manifold.h>

namespace xarm_geo {

    // --- Joint Position, Velocity & Effort Limits ---

    struct JointLimits {
        double q_min = -2 * std::numbers::pi;
        double q_max = 2 * std::numbers::pi;
        double q_vel_max = std::numbers::pi;

        // Symmetric Joint Torque Bound: |tau_i| <= tau_max (N*m).
        // Sourced from URDF <limit effort="..."> attribute; +inf if absent.
        double tau_max = std::numeric_limits<double>::infinity();
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

        // World-Frame Gravitational Acceleration (m/s^2).
        // Default zero -> assumes external gravity compensation (e.g. xArm SDK).
        // Set to (0, 0, -9.81) for torque-mode use w/ RNEA / ASIF.
        Eigen::Vector3d gravity = Eigen::Vector3d::Zero();

        // --- Constraints ---
        std::vector<JointLimits> limits;
    };

    // --- Mutable Intermediate Data ---
    // Pre-Allocated Memory Cache & Output Buffer for ALL Algorithms

    struct Data {
        // --- Canonical Robot State (Source of Truth) ---
        //
        // `data.q` is the joint configuration that all kinematics-derived
        // fields below (pose_tree, *_jacobian, ee_pose, ...) are computed
        // from. The user is responsible for setting `data.q` to the desired
        // configuration BEFORE calling kinematics/dynamics/IK algorithms.
        //
        // After a kinematics call, `data.q` and the derived fields are
        // synchronised. After IK, the result is written to `data.q_out` (a
        // separate field) so that `data.q` remains under user control.
        //
        // Note: joint velocity `v` is NOT stored in Data -- it is treated as
        // a signal (input to dynamics, output of IDK) and passed through APIs
        // explicitly.
        Eigen::VectorXd q;

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

        Eigen::VectorXd q_guess;  // Intermediate Guess for Inverse Kinematics
        Eigen::VectorXd q_out;    // Results of Inverse Kinematics
        Eigen::VectorXd v_out;    // Results of Inverse Differential Kinematics

        // --- Dynamics Outputs ---
        Eigen::MatrixXd M;        // Joint-space Mass Matrix (CRBA)
        Eigen::VectorXd h;        // Full Bias Forces: C(q,v)v + g(q)
        Eigen::VectorXd g;        // Gravity Forces: g(q)  (zero when model.gravity == 0)
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

        // Workspace for Bias-Force Computation (RNEA w/ Zero Acceleration)
        struct BiasWorkspace {
            Eigen::VectorXd a_zero;
        } bias;

        // Workspace for Coriolis-Times Computation (Symmetric Levi-Civita Form)
        struct CoriolisWorkspace {
            Eigen::VectorXd v_sum;  // v_a + v_b
            Eigen::VectorXd b_sum;  // RNEA(q, v_a + v_b, 0)
            Eigen::VectorXd b_a;    // RNEA(q, v_a, 0)
            Eigen::VectorXd b_b;    // RNEA(q, v_b, 0)
        } coriolis;

        // Workspace for Composite Rigid Body Algorithm
        struct CRBAWorkspace {
            std::vector<manifold::SE3::SpatialInertia> I_C;  // Composite Inertias (size: dof)
            manifold::SE3::Wrench F_scratch;                 // Wrench Scratchpad
        } crba;

        // Workspace for Collision-Aware Algorithms (Pre-Allocated Scratchpads
        // shared between CollisionBarrier and the future DynCollisionBarrier).
        struct CollisionWorkspace {
            Eigen::MatrixXd point_jacobian_1;  // (3 x dof) scratch for witness 1
            Eigen::MatrixXd point_jacobian_2;  // (3 x dof) scratch for witness 2
        } collision;

        // Workspace for Optimal Inverse (Diff.) Kinematics (QP via ProxQP)
        struct OptIKWorkspace {
            // QP Cost Matrices (Hessian + Gradient)
            Eigen::MatrixXd H;
            Eigen::VectorXd g;

            // QP Constraint Matrices (Hard Constraints + Barrier Inequalities)
            Eigen::MatrixXd A;
            Eigen::VectorXd l;
            Eigen::VectorXd u;

            // Per-Task Scratch Buffers (Sized to the Largest Task seen so far)
            Eigen::MatrixXd J_task;
            Eigen::VectorXd e_task;

            // ProxQP Solver Instance (Lazy-Init, Kept Warm Across Calls)
            std::unique_ptr<proxsuite::proxqp::dense::QP<double>> qp;
            int current_n = 0;         // Current Number of Decision Variables
            int current_m_eq = 0;      // Current Number of Equality Constraints
            int current_m_in = 0;      // Current Number of Inequality Constraints
            bool initialised = false;  // First call -> qp->init(); subsequent -> qp->update()
        } optik;

        // Workspace for Joint-Torque ASIF Filter (QP via ProxQP)
        // QP form: min 0.5 * (tau - tau_des)^T (I + eps*I) (tau - tau_des)  s.t.   A * tau <= u
        struct ASIFWorkspace {
            Eigen::MatrixXd H;
            Eigen::VectorXd g;

            Eigen::MatrixXd A;
            Eigen::VectorXd l;
            Eigen::VectorXd u;

            // Cached Mass Matrix Factorisation (LLT) - reused for all M^-1 * x
            Eigen::LLT<Eigen::MatrixXd> M_llt;

            // Cached dense inverse: M_inv = M^-1 * I, computed once per
            // asif_filter() call. Dynamic barriers read this to avoid
            // refactoring or repeated back-substitution.
            Eigen::MatrixXd M_inv;

            // Cached vector M^-1 * h_bias, computed once per filter call.
            Eigen::VectorXd M_inv_h;

            // ProxQP Solver Instance (Lazy-Init, Kept Warm Across Calls)
            std::unique_ptr<proxsuite::proxqp::dense::QP<double>> qp;
            int current_n = 0;
            int current_m_eq = 0;
            int current_m_in = 0;
            bool initialised = false;  // First call -> qp->init(); subsequent -> qp->update()
        } asif;

        // --- PRNG (Seeded Once; Reused by IK Random Restarts) ---
        std::mt19937 rng;

        // --- Pre-Allocation Constructor ---
        explicit Data(const Model &model);
    };
}  // namespace xarm_geo
