#pragma once

#include <Eigen/Dense>

#include <xarm_geo/core/motion.h>
#include <xarm_geo/core/system.h>

namespace xarm_geo {

    // --- Admittance Layer ---
    //
    // Stateless joint-space torque-to-velocity map: v = D_v^{-1} * tau.
    // No reference, no state, no safety -- compose with asif_filter or an
    // equivalent upstream layer if those guarantees are required.
    //
    // The model-aware apply() overload additionally enforces
    // |v_i| <= model.limits[i].q_vel_max via direction-preserving rescale.

    class AdmittanceLayer {
    public:
        // Diagonal damping; size must equal dof, all entries strictly positive.
        AdmittanceLayer(int dof, Eigen::Ref<const Eigen::VectorXd> damping_diag);

        // Convenience overload: scalar damping on every joint.
        AdmittanceLayer(int dof, double damping);

        // Pure map: v = D_v^{-1} * tau. No bounds applied.
        void apply(const JointTorque &tau_in, JointVelocity &v_out) const noexcept;

        // Pure map followed by direction-preserving rescale to satisfy
        // |v_i| <= model.limits[i].q_vel_max for all i.
        void apply(const Model &model, const JointTorque &tau_in,
                   JointVelocity &v_out) const noexcept;

    private:
        int dof_;
        Eigen::VectorXd d_inv_diag_;  // Precomputed 1 / D_v diagonal.
    };

}  // namespace xarm_geo
