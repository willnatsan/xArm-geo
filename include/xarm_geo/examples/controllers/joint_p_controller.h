#pragma once

#include <cassert>

#include <Eigen/Dense>

#include <xarm_geo/control/controller.h>
#include <xarm_geo/core/manifold.h>
#include <xarm_geo/core/motion.h>
#include <xarm_geo/core/system.h>
#include <xarm_geo/utils/debug.h>

namespace xarm_geo::controllers {

    // --- Example: Joint-Space P Controller (Kinematic) ---
    //
    // Reference implementation of a joint-space tracking kinematic P
    // controller built on `KinematicJointControllerBase`.
    //
    // Geometric primitives (Bullo & Murray on R^n):
    //   - Configuration error  : e   = q - r
    //   - Velocity error       : e_d = q_dot - r_dot      (transport = I_n)
    //   - Error gradient       : grad Phi(q, r) = K_p * (q - r)
    //
    // Joint-velocity command:
    //
    //     q_dot_c = r_dot - K_p * (q - r)        [use_feedforward = true]
    //     q_dot_c =       - K_p * (q - r)        [use_feedforward = false]
    //
    // Here K_p is a per-joint diagonal proportional gain vector. r_dot acts
    // as a velocity feedforward to track moving references; the proportional
    // term corrects positional drift.

    class JointPController final : public KinematicJointControllerBase {
    public:
        explicit JointPController(const Model &model)
            : KinematicJointControllerBase(model), kp(Eigen::VectorXd::Zero(model.dof)) {

            assert(model.dof > 0 && "JointPController: model.dof must be > 0");
        }

        // --- Public Configuration ---
        Eigen::VectorXd kp;           // Per-joint diagonal proportional gain.
        bool use_feedforward = true;  // Include r_dot in the command.

    protected:
        auto compute_command_velocity(const Model &model, Data & /*data*/,
                                      const JointControllerContext &ctx,
                                      JointVelocity &v_ctrl) noexcept -> bool override {

            // Hook-side size checks (gain vector + joint-space target).
            if (kp.size() != model.dof || ctx.ref.q.size() != model.dof ||
                ctx.ref.v.size() != model.dof) {
                debug::log("JointPController: kp / ref.q / ref.v size mismatch with model.dof");
                return false;
            }

            // Joint-velocity command:
            //   use_feedforward = true  -> q_dot_c = r_dot - K_p * (q - r)
            //   use_feedforward = false -> q_dot_c =       - K_p * (q - r)
            if (use_feedforward) {
                v_ctrl.v.noalias() = ctx.ref.v - kp.cwiseProduct(ctx.fb.q - ctx.ref.q);
            } else {
                v_ctrl.v.noalias() = -kp.cwiseProduct(ctx.fb.q - ctx.ref.q);
            }

            return true;
        }
    };

    // --- Compile-Time Concept Verification ---
    static_assert(xarm_geo::KinematicJointController<JointPController>);

}  // namespace xarm_geo::controllers
