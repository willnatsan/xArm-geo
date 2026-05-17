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
        double tau_max = std::numeric_limits<double>::infinity();
    };

    // --- Immutable Model ---
    //
    // Physical properties and static configuration.

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

        // Per-joint reflected motor + harmonic-drive inertia (kg*m^2).
        // Added diagonally to M(q) by compute_mass_matrix; CRBA's link-only
        // pass omits this rotor contribution. Default empty -> no-op.
        Eigen::VectorXd joint_armature;

        // World-frame gravitational acceleration (m/s^2). Default zero
        // assumes external compensation; set (0, 0, -9.81) for RNEA / ASIF.
        Eigen::Vector3d gravity = Eigen::Vector3d::Zero();

        // --- Constraints ---
        std::vector<JointLimits> limits;
    };

    // --- Mutable Data ---
    //
    // Pre-allocated scratch and output buffers for every algorithm.

    struct Data {
        // --- Canonical Robot State ---
        //
        // data.q is the joint configuration that all kinematics-derived fields
        // are computed from. Set data.q before calling any kinematics /
        // dynamics / IK routine. IK results go to data.q_out so data.q stays
        // under user control. Joint velocity v is treated as a signal and
        // passed explicitly through APIs rather than stored here.
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

        Eigen::VectorXd q_guess;  // IK intermediate guess
        Eigen::VectorXd q_out;    // IK result
        Eigen::VectorXd v_out;    // IDK result

        // --- Dynamics Outputs ---
        Eigen::MatrixXd M;        // joint-space mass matrix (CRBA)
        Eigen::VectorXd h;        // full bias forces: C(q,v)v + g(q)
        Eigen::VectorXd g;        // gravity forces g(q) (zero when model.gravity == 0)
        Eigen::VectorXd tau_out;  // inverse-dynamics result
        Eigen::VectorXd a_out;    // forward-dynamics result

        // --- Internal Workspaces (Zero-Allocation Scratchpads) ---

        // Damped Least Squares IK.
        struct IKWorkspace {
            Eigen::MatrixXd A;  // J^T * J + lambda^2 * I
            Eigen::VectorXd b;  // J^T * V_err
        } ik;

        // Recursive Newton-Euler Algorithm.
        struct RNEAWorkspace {
            std::vector<manifold::SE3::Twist> v_links;
            std::vector<manifold::SE3::SpatialAcceleration> a_links;
            std::vector<manifold::SE3::Wrench> f_links;
            std::vector<manifold::SE3> T_i_parent_cache;
        } rnea;

        // RNEA with zero acceleration; used for bias-force computation.
        struct BiasWorkspace {
            Eigen::VectorXd a_zero;
        } bias;

        // Symmetric Levi-Civita Coriolis-times product.
        struct CoriolisWorkspace {
            Eigen::VectorXd v_sum;  // v_a + v_b
            Eigen::VectorXd b_sum;  // RNEA(q, v_a + v_b, 0)
            Eigen::VectorXd b_a;    // RNEA(q, v_a, 0)
            Eigen::VectorXd b_b;    // RNEA(q, v_b, 0)
        } coriolis;

        // Composite Rigid Body Algorithm.
        struct CRBAWorkspace {
            std::vector<manifold::SE3::SpatialInertia> I_C;
            manifold::SE3::Wrench F_scratch;
        } crba;

        // Shared by CollisionBarrier and DynCollisionBarrier.
        struct CollisionWorkspace {
            Eigen::MatrixXd point_jacobian_1;  // (3 x dof) witness 1
            Eigen::MatrixXd point_jacobian_2;  // (3 x dof) witness 2
        } collision;

        // Optimal inverse (diff.) kinematics (ProxQP).
        struct OptIKWorkspace {
            // Cost matrices.
            Eigen::MatrixXd H;
            Eigen::VectorXd g;

            // Constraint matrices (hard constraints + barrier inequalities).
            Eigen::MatrixXd A;
            Eigen::VectorXd l;
            Eigen::VectorXd u;

            // Per-task scratch, sized to the largest task seen so far.
            Eigen::MatrixXd J_task;
            Eigen::VectorXd e_task;

            // Lazy-init ProxQP instance, kept warm across calls.
            std::unique_ptr<proxsuite::proxqp::dense::QP<double>> qp;
            int current_n = 0;
            int current_m_eq = 0;
            int current_m_in = 0;
            bool initialised = false;
            // True once qp->solve() has completed at least once. Two purposes:
            //   1. Guards against passing WARM_START_WITH_PREVIOUS_RESULT to a
            //      freshly-constructed QP whose LDLT has never been factorised
            //      (proxsuite 0.7.2 bug: the first solve skips
            //      setup_factorization when refactorize==false, leaving
            //      ldl.dim()==0 and causing a rank_r_update assertion).
            //   2. Callers may explicitly set this to false at the top of an
            //      entry point (e.g. optimal_inverse_kinematics) to force the
            //      next solve onto NO_INITIAL_GUESS, isolating it from stale
            //      primal/dual values left by a prior caller.
            bool solved_once = false;
        } optik;

        // Joint-torque ASIF filter (ProxQP).
        // QP form: min 0.5 * (tau - tau_des)^T (I + eps*I) (tau - tau_des)  s.t.  A * tau <= u.
        struct ASIFWorkspace {
            Eigen::MatrixXd H;
            Eigen::VectorXd g;

            Eigen::MatrixXd A;
            Eigen::VectorXd l;
            Eigen::VectorXd u;

            // Mass matrix factorisation reused for every M^-1 * x access.
            Eigen::LLT<Eigen::MatrixXd> M_llt;

            // Dense M^-1 and M^-1 * h, computed once per asif_filter() call so
            // dynamic barriers avoid repeated back-substitution.
            Eigen::MatrixXd M_inv;
            Eigen::VectorXd M_inv_h;

            std::unique_ptr<proxsuite::proxqp::dense::QP<double>> qp;
            int current_n = 0;
            int current_m_eq = 0;
            int current_m_in = 0;
            bool initialised = false;
            // See OptIKWorkspace::solved_once for rationale.
            bool solved_once = false;
        } asif;

        // Seeded once; reused by IK random restarts.
        std::mt19937 rng;

        // Pre-allocation constructor.
        explicit Data(const Model &model);
    };
}  // namespace xarm_geo
