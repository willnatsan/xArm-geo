#include <algorithm>
#include <cassert>
#include <cmath>

#include <xarm_geo/control/admittance.h>
#include <xarm_geo/utils/debug.h>

namespace xarm_geo {

    AdmittanceLayer::AdmittanceLayer(int dof, Eigen::Ref<const Eigen::VectorXd> damping_diag)
        : dof_(dof), d_inv_diag_(Eigen::VectorXd::Zero(dof)) {

        assert(dof_ > 0 && "AdmittanceLayer: dof must be > 0");
        assert(damping_diag.size() == dof_ && "damping_diag size must equal dof");
        assert((damping_diag.array() > 0.0).all() &&
               "damping_diag must be strictly positive on every joint");

        d_inv_diag_ = damping_diag.cwiseInverse();
    }

    AdmittanceLayer::AdmittanceLayer(int dof, double damping)
        : AdmittanceLayer(dof, Eigen::VectorXd::Constant(dof, damping)) {}

    void AdmittanceLayer::apply(const JointTorque &tau_in, JointVelocity &v_out) const noexcept {

        assert(tau_in.tau.size() == dof_ && "tau size must equal dof");
        assert(v_out.v.size() == dof_ && "v_out size must equal dof");

        if (!tau_in.tau.allFinite()) {
            debug::log("non-finite tau; emitting zero velocity");
            v_out.v.setZero();
            return;
        }

        v_out.v.noalias() = d_inv_diag_.cwiseProduct(tau_in.tau);
    }

    void AdmittanceLayer::apply(const Model &model, const JointTorque &tau_in,
                                JointVelocity &v_out) const noexcept {

        assert(model.dof == dof_ && "model.dof must equal layer dof");

        apply(tau_in, v_out);

        // Direction-preserving rescale against q_vel_max.
        // Style mirrors src/modelling/kinematics.cpp:131-144.
        double max_ratio = 0.0;
        for (int i = 0; i < dof_; ++i) {
            const double v_max = model.limits[i].q_vel_max;
            if (v_max > 0.0) { max_ratio = std::max(max_ratio, std::abs(v_out.v[i]) / v_max); }
        }
        if (max_ratio > 1.0) { v_out.v *= (1.0 / max_ratio); }
    }

}  // namespace xarm_geo
