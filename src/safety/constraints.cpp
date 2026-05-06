#include <cassert>
#include <limits>

#include <xarm_geo/safety/constraints.h>

namespace xarm_geo {

    // --- VelocityLimit ---
    // -dt * v_max <= dq <= dt * v_max,    G = I.

    void VelocityLimit::compute(const Model &model, Data & /*data*/, Eigen::Ref<Eigen::MatrixXd> G,
                                Eigen::Ref<Eigen::VectorXd> l,
                                Eigen::Ref<Eigen::VectorXd> u) const {

        assert(dt > 0.0 && "VelocityLimit::dt must be positive");

        G.setIdentity();

        const bool use_override = (v_max.size() == model.dof);

        for (int i = 0; i < model.dof; ++i) {
            const double v_max_i = use_override ? v_max[i] : model.limits[i].q_vel_max;
            const double bound = dt * v_max_i;
            l[i] = -bound;
            u[i] = bound;
        }
    }

    // --- PositionLimit ---
    // q_min - data.q <= dq <= q_max - data.q,    G = I.

    void PositionLimit::compute(const Model &model, Data &data, Eigen::Ref<Eigen::MatrixXd> G,
                                Eigen::Ref<Eigen::VectorXd> l,
                                Eigen::Ref<Eigen::VectorXd> u) const {

        G.setIdentity();

        if (data.q.size() != model.dof) {
            // Defensive: data.q not populated -> emit infinite (effectively absent) bounds.
            l.setConstant(-std::numeric_limits<double>::infinity());
            u.setConstant(std::numeric_limits<double>::infinity());
            return;
        }

        for (int i = 0; i < model.dof; ++i) {
            l[i] = model.limits[i].q_min - data.q[i];
            u[i] = model.limits[i].q_max - data.q[i];
        }
    }

}  // namespace xarm_geo
