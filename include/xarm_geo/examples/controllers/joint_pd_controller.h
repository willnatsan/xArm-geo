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

    // --- Joint-Space PD Controller (Bullo & Murray) ---
    //
    // Joint-space dynamic PD specialised from the Bullo & Murray law on R^n:
    //   tau_PD = -K_p * (q - r) - K_d * (q_dot - r_dot)
    //   tau_FF = M(q) * r_ddot + C(q, q_dot) * r_dot         [both FF flags = true]
    //   tau_ctrl = tau_PD + tau_FF
    //
    // C(q, q_dot) * r_dot uses the symmetric Christoffel form via
    // compute_coriolis_times.

    class JointPDController final : public DynamicJointControllerBase {
    public:
        // Recommended default; users may override after construction.
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
        Eigen::VectorXd kp;           // per-joint proportional gain
        Eigen::VectorXd kd;           // per-joint derivative gain
        bool use_inertial_ff = true;  // include M(q) * r_ddot
        bool use_coriolis_ff = true;  // include C(q, q_dot) * r_dot

    protected:
        auto compute_command_torque(const Model &model, Data &data, KinematicsCache & /*kin*/,
                                    DynamicsCache &dyn, const JointControllerContext &ctx,
                                    JointTorque &tau_ctrl) noexcept -> bool override {

            if (kp.size() != model.dof || kd.size() != model.dof || ctx.ref.q.size() != model.dof ||
                ctx.ref.v.size() != model.dof || ctx.ref.a.size() != model.dof) {
                debug::log("JointPDController: kp / kd / ref.{q,v,a} size mismatch with model.dof");
                return false;
            }

            // PD term.
            tau_ctrl.tau.noalias() =
                -kp.cwiseProduct(ctx.fb.q - ctx.ref.q) - kd.cwiseProduct(ctx.fb.v - ctx.ref.v);

            if (use_inertial_ff) {
                Mqdd_r_.noalias() = dyn.M() * ctx.ref.a;
                tau_ctrl.tau.noalias() += Mqdd_r_;
            }

            // Pull g(q) through the cache first so compute_coriolis_times can
            // skip its internal gravity recomputation regardless of bias_compensation.
            if (use_coriolis_ff) {
                (void)dyn.g();
                compute_coriolis_times(model, data, ctx.fb.v, ctx.ref.v, C_qdot_rdot_,
                                       /*g_fresh=*/true);
                tau_ctrl.tau.noalias() += C_qdot_rdot_;
            }

            return true;
        }

    private:
        // --- Pre-Allocated Per-Tick Scratch ---
        Eigen::VectorXd Mqdd_r_;       // M(q) * r_ddot
        Eigen::VectorXd C_qdot_rdot_;  // C(q, q_dot) * r_dot
    };

    // --- Compile-Time Concept Verification ---
    static_assert(xarm_geo::DynamicJointController<JointPDController>);

}  // namespace xarm_geo::controllers
