#pragma once

#include <cassert>

#include <Eigen/Dense>

#include <xarm_geo/control/controller.h>
#include <xarm_geo/core/manifold.h>
#include <xarm_geo/core/motion.h>
#include <xarm_geo/core/system.h>
#include <xarm_geo/modelling/dynamics.h>
#include <xarm_geo/utils/debug.h>

namespace xarm_geo::controllers {

    // --- Example: Joint-Space PD Controller (Dynamic, Bullo & Murray) ---
    //
    // Reference implementation of a joint-space tracking dynamic PD
    // controller built on `DynamicJointControllerBase`. Faithfully recreates
    // the Bullo & Murray geometric PD law specialised to R^n.
    //
    // Geometric primitives (Bullo & Murray on R^n):
    //   - Configuration error  : e   = q - r
    //   - Velocity error       : e_d = q_dot - r_dot       (transport = I_n)
    //   - Error gradient       : grad Phi(q, r) = K_p * (q - r)
    //
    // Joint torque command (the hook returns this):
    //
    //     tau_PD = - K_p * (q - r) - K_d * (q_dot - r_dot)
    //     tau_FF = M(q) * r_ddot + C(q, q_dot) * r_dot         [both FF flags = true]
    //     tau_ctrl = tau_PD + tau_FF
    //
    // The Coriolis bilinear C(q, q_dot) * r_dot is computed via the
    // symmetric polarisation identity in `compute_coriolis_times`.

    class JointPDController final : public DynamicJointControllerBase {
    public:
        // The bias_compensation policy that pairs correctly with this controller's geometric
        // structure. Set on bias_compensation in the constructor; users may override after
        // construction to deviate (e.g. set None for real-hardware xArm SDK with gravity comp).
        static constexpr BiasCompensation kRecommendedBiasCompensation =
            BiasCompensation::GravityOnly;

        explicit JointPDController(const Model &model)
            : DynamicJointControllerBase(model), kp(Eigen::VectorXd::Zero(model.dof)),
              kd(Eigen::VectorXd::Zero(model.dof)), Mqdd_r_(Eigen::VectorXd::Zero(model.dof)),
              C_qdot_rdot_(Eigen::VectorXd::Zero(model.dof)) {

            assert(model.dof > 0 && "JointPDController: model.dof must be > 0");
            bias_compensation = kRecommendedBiasCompensation;
        }

        // --- Public Configuration ---
        Eigen::VectorXd kp;           // Per-joint diagonal proportional gain.
        Eigen::VectorXd kd;           // Per-joint diagonal derivative gain.
        bool use_inertial_ff = true;  // Include M(q) * r_ddot in the FF.
        bool use_coriolis_ff = true;  // Include C(q, q_dot) * r_dot in the FF.

    protected:
        auto compute_command_torque(const Model &model, Data &data, KinematicsCache & /*kin*/,
                                    DynamicsCache &dyn, const JointControllerContext &ctx,
                                    JointTorque &tau_ctrl) noexcept -> bool override {

            // Hook-side size checks (gain vectors + joint-space target).
            if (kp.size() != model.dof || kd.size() != model.dof || ctx.ref.q.size() != model.dof ||
                ctx.ref.v.size() != model.dof || ctx.ref.a.size() != model.dof) {
                debug::log("JointPDController: kp / kd / ref.{q,v,a} size mismatch with model.dof");
                return false;
            }

            // PD term: tau_PD = - K_p * (q - r) - K_d * (q_dot - r_dot)
            tau_ctrl.tau.noalias() =
                -kp.cwiseProduct(ctx.fb.q - ctx.ref.q) - kd.cwiseProduct(ctx.fb.v - ctx.ref.v);

            // Inertial feedforward: tau_FF += M(q) * r_ddot.
            if (use_inertial_ff) {
                Mqdd_r_.noalias() = dyn.M() * ctx.ref.a;
                tau_ctrl.tau.noalias() += Mqdd_r_;
            }

            // Geometric Coriolis feedforward: tau_FF += C(q, q_dot) * r_dot,
            // via the symmetric Levi-Civita Christoffel form. Pull g(q) through
            // the cache first, so compute_coriolis_times can skip its internal
            // gravity recomputation regardless of bias_compensation.
            if (use_coriolis_ff) {
                (void)dyn.g();  // ensure data.g is fresh and cached
                compute_coriolis_times(model, data, ctx.fb.v, ctx.ref.v, C_qdot_rdot_,
                                       /*g_fresh=*/true);
                tau_ctrl.tau.noalias() += C_qdot_rdot_;
            }

            return true;
        }

    private:
        // --- Per-Tick Scratch (Pre-Allocated at Construction) ---
        // Sized from model.dof; zero allocation in compute_command_torque().
        Eigen::VectorXd Mqdd_r_;       // M(q) * r_ddot
        Eigen::VectorXd C_qdot_rdot_;  // C(q, q_dot) * r_dot
    };

    // --- Compile-Time Concept Verification ---
    static_assert(xarm_geo::DynamicJointController<JointPDController>);

}  // namespace xarm_geo::controllers
