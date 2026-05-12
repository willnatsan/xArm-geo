#pragma once

#include <cassert>

#include <Eigen/Dense>

#include <xarm_geo/control/controller.h>
#include <xarm_geo/core/manifold.h>
#include <xarm_geo/core/motion.h>
#include <xarm_geo/core/system.h>
#include <xarm_geo/utils/debug.h>

namespace xarm_geo::controllers {

    // --- Joint-Space P Controller ---
    //
    // Joint-space kinematic P (Bullo & Murray on R^n). Joint-velocity command:
    //   q_dot_c = r_dot - K_p * (q - r)        [use_feedforward = true]
    //   q_dot_c =       - K_p * (q - r)        [use_feedforward = false]

    class JointPController final : public KinematicJointControllerBase {
    public:
        explicit JointPController(const Model &model)
            : KinematicJointControllerBase(model), kp(Eigen::VectorXd::Zero(model.dof)) {

            assert(model.dof > 0 && "JointPController: model.dof must be > 0");
        }

        // --- Public Configuration ---
        Eigen::VectorXd kp;           // per-joint diagonal proportional gain
        bool use_feedforward = true;  // include r_dot in the command

    protected:
        auto compute_command_velocity(const Model &model, Data & /*data*/,
                                      KinematicsCache & /*kin*/, const JointControllerContext &ctx,
                                      JointVelocity &v_ctrl) noexcept -> bool override {

            if (kp.size() != model.dof || ctx.ref.q.size() != model.dof ||
                ctx.ref.v.size() != model.dof) {
                debug::log("JointPController: kp / ref.q / ref.v size mismatch with model.dof");
                return false;
            }

            const Eigen::VectorXd p_term = kp.cwiseProduct(ctx.fb.q - ctx.ref.q);
            if (use_feedforward) {
                v_ctrl.v.noalias() = ctx.ref.v - p_term;
            } else {
                v_ctrl.v.noalias() = -p_term;
            }
            return true;
        }
    };

    // --- Compile-Time Concept Verification ---
    static_assert(xarm_geo::KinematicJointController<JointPController>);

}  // namespace xarm_geo::controllers
