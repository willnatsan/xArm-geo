#pragma once

#include <Eigen/Dense>

#include <xarm_geo/core/manifold.h>
#include <xarm_geo/core/system.h>

namespace xarm_geo {

    // --- Task Base Class ---
    //
    // A Task contributes a quadratic cost to the Optimal IK QP:
    //
    //     0.5 * (J * dq + alpha * e)^T W^T W (J * dq + alpha * e)
    //
    // which expands to:
    //
    //     0.5 * dq^T H_task dq + g_task^T dq    (+ const)
    //
    // with H_task = J^T W^T W J + lm_damping * I  and  g_task = J^T W^T W (alpha * e).
    //
    // The Task is responsible for filling J (rows() x dof) and e (rows()) when
    // `compute()` is called.

    struct Task {
        virtual ~Task() = default;

        // Number of rows the Task contributes to the residual e (== rows of J).
        [[nodiscard]] virtual auto rows() const noexcept -> int = 0;

        // Fill J (rows x dof) and e (rows) using current `data` (assumes
        // `compute_jacobians(model, data)` has been called for the current
        // value of `data.q`).
        virtual void compute(const Model &model, Data &data, Eigen::Ref<Eigen::MatrixXd> J,
                             Eigen::Ref<Eigen::VectorXd> e) const = 0;

        double gain = 1.0;        // Task-Space "P-Gain"
        double lm_damping = 0.0;  // Per-Task Levenberg-Marquardt Damping

        // Diagonal Weight Vector (Size: rows()). Empty -> Unit Weights.
        Eigen::VectorXd weight;
    };

    // --- Frame (Task-Space Pose) Task ---
    //
    // Tracks an SE(3) target pose using the body Jacobian and a right-minus
    // (log) error: e = log(ee.inverse() * target).

    struct FrameTask final : public Task {
        FrameTask() : target(manifold::SE3::Identity()) { weight.setOnes(6); }

        manifold::SE3 target;

        [[nodiscard]] auto rows() const noexcept -> int override { return 6; }

        void compute(const Model &model, Data &data, Eigen::Ref<Eigen::MatrixXd> J,
                     Eigen::Ref<Eigen::VectorXd> e) const override;
    };

    // --- Posture (Joint-Space) Task ---
    //
    // Tracks a joint-space reference: J = I, e = q_ref - q_curr.
    // Useful as a low-priority secondary task for null-space behaviour
    // (e.g., bias toward a "natural" posture).
    //
    // The user is responsible for refreshing `q_curr` (typically from
    // `data.q`) before each solve.

    struct PostureTask final : public Task {
        PostureTask() = default;

        Eigen::VectorXd q_ref;   // Reference Joint Configuration
        Eigen::VectorXd q_curr;  // Current Joint Configuration (refreshed by user)

        [[nodiscard]] auto rows() const noexcept -> int override {
            return static_cast<int>(q_ref.size());
        }

        void compute(const Model &model, Data &data, Eigen::Ref<Eigen::MatrixXd> J,
                     Eigen::Ref<Eigen::VectorXd> e) const override;
    };

    // --- Twist (Task-Space Velocity) Task ---
    //
    // Tracks a commanded body-frame spatial twist (the velocity-level analogue
    // of FrameTask).
    //
    // IMPORTANT: `dt` MUST be set to a positive value before solving.
    // The QP variable is dq, so the residual is `J dq - target_twist * dt`.
    // Leaving dt at 0 silently produces a zero residual (asserted in compute).

    struct TwistTask final : public Task {
        TwistTask() : target_twist(manifold::SE3::Twist::Zero()) { weight.setOnes(6); }

        manifold::SE3::Twist target_twist;  // Commanded Body-Frame Twist (rad/s, m/s)
        double dt = 0.0;                    // Step length used to convert v -> dq (must be > 0)

        [[nodiscard]] auto rows() const noexcept -> int override { return 6; }

        void compute(const Model &model, Data &data, Eigen::Ref<Eigen::MatrixXd> J,
                     Eigen::Ref<Eigen::VectorXd> e) const override;
    };

}  // namespace xarm_geo
