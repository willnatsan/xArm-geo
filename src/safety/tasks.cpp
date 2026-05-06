#include <cassert>

#include <xarm_geo/safety/tasks.h>

namespace xarm_geo {

    // --- FrameTask ---
    // Body-Frame Pose Tracking
    //   J = body_jacobian (6 x dof)
    //   e = log(ee.inverse() * target)   [Right-Minus on SE(3)]
    //
    // Note: Caller must have run `compute_jacobians(model, data)` so that
    //       data.body_jacobian and data.ee_pose are up-to-date for data.q.

    void FrameTask::compute(const Model & /*model*/, Data &data, Eigen::Ref<Eigen::MatrixXd> J,
                            Eigen::Ref<Eigen::VectorXd> e) const {

        J = data.body_jacobian;
        e = (target - data.ee_pose);
    }

    // --- PostureTask ---
    // Joint-Space Tracking: J = I, e = q_ref - q_curr.
    // If q_curr has not been refreshed (size mismatch), fall back to e = q_ref.
    // The fallback is deliberate (lets the user defer setting q_curr) -- not an
    // assert -- but callers should refresh q_curr each tick for correct behaviour.

    void PostureTask::compute(const Model & /*model*/, Data & /*data*/,
                              Eigen::Ref<Eigen::MatrixXd> J, Eigen::Ref<Eigen::VectorXd> e) const {

        J.setIdentity();
        if (q_curr.size() == q_ref.size()) {
            e = q_ref - q_curr;
        } else {
            e = q_ref;
        }
    }

    // --- TwistTask ---
    // Body-Frame Twist Tracking
    //   J = body_jacobian (6 x dof)
    //   e = target_twist * dt
    //
    // Note: Caller must have run `compute_jacobians(model, data)` so that
    //       data.body_jacobian is up-to-date for data.q. `dt` must be > 0.

    void TwistTask::compute(const Model & /*model*/, Data &data, Eigen::Ref<Eigen::MatrixXd> J,
                            Eigen::Ref<Eigen::VectorXd> e) const {

        assert(dt > 0.0 && "TwistTask::dt must be set to a positive value before solving");

        J = data.body_jacobian;
        e = target_twist * dt;
    }

}  // namespace xarm_geo
