#pragma once

#include <cstdint>

#include <Eigen/Dense>

#include <xarm_geo/control/controller.h>
#include <xarm_geo/control/tracking.h>
#include <xarm_geo/core/manifold.h>
#include <xarm_geo/core/system.h>

namespace xarm_geo {

    // Selects the SE(3) tracking gradient used by the geometric controllers.
    //   LieGroup :   Smooth Function (via Trace Function) -> Almost Global Stability
    //   LieAlgebra : Discontinuous Function (via Log-Map) -> Global Stability
    enum class GradientType : std::uint8_t { LieGroup, LieAlgebra };

    // --- Geometric P Controller (Kinematic) ---
    //
    // Reference: TaskSpaceTarget; Command: JointVelocity.
    //
    // Body-frame command twist:
    //
    //     xi_c = Ad_{g_e} * xi_d - nabla Phi(g_e)         (use_feedforward = true)
    //     xi_c =                 - nabla Phi(g_e)         (use_feedforward = false)

    class GeometricPController final : public KinematicTaskController {
    public:
        explicit GeometricPController(const Model &model);

        // --- Public Configuration ---
        SE3TrackingGains gains;
        bool use_feedforward = true;
        GradientType gradient = GradientType::LieGroup;

    protected:
        auto compute_command_twist(const Model &model, Data &data, const JointState &fb,
                                   const TaskSpaceTarget &ref, const ControllerContext &ctx,
                                   manifold::SE3::Twist &cmd_twist) noexcept
            -> ControllerStatus override;

    private:
        int dof_;
    };

    // --- Geometric PD Controller (Dynamic) ---
    //
    // Reference: TaskSpaceTarget; Command: JointTorque.
    //
    // Body-frame end-effector wrench:
    //
    //     F_task = - nabla Phi(g_e)
    //              - K_D * xi_e
    //              + ( Lambda(q) * d/dt(Ad * xi_d)
    //                  - ad_{xi_e}^* * Lambda(q) * Ad * xi_d )    [if use_feedforward]
    //
    // Operational-space inertia: Lambda(q) = (J_b * M^{-1} * J_b^T)^{-1}.
    // M(q) is factorised internally each tick when use_feedforward = true.
    // If the factorisation fails (M not PD), the FF term is dropped for
    // that tick (debug-logged) and the P+D path still runs.

    class GeometricPDController final : public DynamicTaskController {
    public:
        explicit GeometricPDController(const Model &model);

        // --- Public Configuration ---
        SE3TrackingGains gains;
        bool use_feedforward = true;
        GradientType gradient = GradientType::LieGroup;

    protected:
        auto compute_task_wrench(const Model &model, Data &data, const JointState &fb,
                                 const TaskSpaceTarget &ref, const ControllerContext &ctx,
                                 manifold::SE3::Twist &task_wrench) noexcept
            -> ControllerStatus override;

    private:
        int dof_;

        // --- Per-Tick Scratch (Pre-Allocated at Construction) ---
        // Sized from model.dof; zero allocation in compute_task_wrench().
        Eigen::LLT<Eigen::MatrixXd> M_llt_;
        Eigen::MatrixXd M_inv_Jt_;            // (dof x 6)
        Eigen::Matrix<double, 6, 6> lambda_;  // Operational-space inertia
        manifold::SE3::Twist ad_xi_d_;        // Ad_{g_e} * xi_d
        manifold::SE3::Twist d_ad_xi_d_;      // d/dt(Ad_{g_e} * xi_d)
        manifold::SE3::Twist xi_e_;           // velocity error
    };

}  // namespace xarm_geo
