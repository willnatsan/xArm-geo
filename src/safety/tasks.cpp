#include <cassert>

#include <xarm_geo/safety/tasks.h>

namespace xarm_geo {

    // --- Frame Task (Body-Frame Pose Tracking) ---
    //
    // J = body_jacobian (6 x dof);  e = log(ee.inverse() * target)  [right-minus on SE(3)].
    // Caller must have run compute_jacobians(model, data) for the current data.q.

    void FrameTask::compute(const Model & /*model*/, Data &data, Eigen::Ref<Eigen::MatrixXd> J,
                            Eigen::Ref<Eigen::VectorXd> e) const {

        J = data.body_jacobian;
        e = (target - data.ee_pose);
    }

    // --- Posture Task (Joint-Space Tracking) ---
    //
    // J = I,  e = q_ref - q_curr. If q_curr has not been refreshed (size mismatch),
    // falls back to e = q_ref to let callers defer setting q_curr.

    void PostureTask::compute(const Model & /*model*/, Data & /*data*/,
                              Eigen::Ref<Eigen::MatrixXd> J, Eigen::Ref<Eigen::VectorXd> e) const {

        J.setIdentity();
        if (q_curr.size() == q_ref.size()) {
            e = q_ref - q_curr;
        } else {
            e = q_ref;
        }
    }

    // --- Twist Task (Body-Frame Twist Tracking) ---
    //
    // J = body_jacobian (6 x dof);  e = target_twist * dt.
    // Caller must have run compute_jacobians(model, data) for the current data.q.

    void TwistTask::compute(const Model & /*model*/, Data &data, Eigen::Ref<Eigen::MatrixXd> J,
                            Eigen::Ref<Eigen::VectorXd> e) const {

        assert(dt > 0.0 && "TwistTask::dt must be set to a positive value before solving");

        J = data.body_jacobian;
        e = target_twist * dt;
    }

}  // namespace xarm_geo
