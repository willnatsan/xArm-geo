#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <string>

#include <Eigen/Dense>

// validate.h depends on controller.h (for TaskControllerContext, ControllerStatus,
// JointState, JointVelocity) and trajectory.h (for concepts, targets, status).
// controller.h itself includes trajectory.h, so trajectory.h is guaranteed to be
// processed first — this is what breaks the cycle:
//
//   trajectory.h  (leaf — no controller dep)
//   controller.h  → trajectory.h
//   validate.h    → controller.h + trajectory.h  (both fully resolved)
//
// Users who only need to author trajectories include trajectory.h.
// Users who need to validate them include validate.h (or the umbrella).

#include <xarm_geo/control/controller.h>
#include <xarm_geo/modelling/collision.h>
#include <xarm_geo/modelling/kinematics.h>
#include <xarm_geo/trajectory/trajectory.h>

namespace xarm_geo {

    // --- Validation Status ---
    //
    // Structured outcome of validate_trajectory(). Callers can switch on this
    // for differentiated error handling, or compare directly to OK for the
    // common case.

    enum class ValidationStatus : std::uint8_t {
        OK,
        COLLISION,
        TRAJECTORY_OUT_OF_DOMAIN,
        TRAJECTORY_NOT_INITIALISED,
        TRAJECTORY_SOLVER_ERROR,
        TRAJECTORY_ERROR,
        CONTROLLER_FAILED,
    };

    [[nodiscard]] constexpr auto to_string(ValidationStatus s) noexcept -> const char * {
        switch (s) {
        case ValidationStatus::OK:
            return "OK";
        case ValidationStatus::COLLISION:
            return "COLLISION";
        case ValidationStatus::TRAJECTORY_OUT_OF_DOMAIN:
            return "TRAJECTORY_OUT_OF_DOMAIN";
        case ValidationStatus::TRAJECTORY_NOT_INITIALISED:
            return "TRAJECTORY_NOT_INITIALISED";
        case ValidationStatus::TRAJECTORY_SOLVER_ERROR:
            return "TRAJECTORY_SOLVER_ERROR";
        case ValidationStatus::TRAJECTORY_ERROR:
            return "TRAJECTORY_ERROR";
        case ValidationStatus::CONTROLLER_FAILED:
            return "CONTROLLER_FAILED";
        }
        return "UNKNOWN";
    }

    struct ValidationOptions {
        double integration_dt = 0.002;
        double collision_check_dt = 0.05;
    };

    struct ValidationResult {
        ValidationStatus status = ValidationStatus::OK;
        double failure_time = -1.0;
        std::string reason;
    };

    // Internal helper: map TrajectoryStatus -> ValidationStatus.
    [[nodiscard]] inline auto trajectory_to_validation_status(TrajectoryStatus ts) noexcept
        -> ValidationStatus {
        switch (ts) {
        case TrajectoryStatus::OK:
            return ValidationStatus::OK;
        case TrajectoryStatus::OUT_OF_DOMAIN:
            return ValidationStatus::TRAJECTORY_OUT_OF_DOMAIN;
        case TrajectoryStatus::NOT_INITIALISED:
            return ValidationStatus::TRAJECTORY_NOT_INITIALISED;
        case TrajectoryStatus::SOLVER_ERROR:
            return ValidationStatus::TRAJECTORY_SOLVER_ERROR;
        case TrajectoryStatus::ERROR:
            return ValidationStatus::TRAJECTORY_ERROR;
        }
        return ValidationStatus::TRAJECTORY_ERROR;
    }

    // --- Trajectory Validation ---
    //
    // Forward-simulate a trajectory and check for collisions at each step.
    //
    // Task-space overloads: simulate closed-loop execution with a user-supplied
    // KinematicTaskController. Pass the same controller instance (with the same
    // configuration) you intend to use at runtime. Validation accuracy depends
    // on the controller and gains matching runtime.
    //
    // Joint-space overloads: sample the trajectory directly (open-loop).
    //
    // Two overload families for each domain:
    //   (1) Default — reads duration from trajectory.duration(). Rejects
    //       infinite-duration trajectories with TRAJECTORY_ERROR; use the
    //       explicit-duration overload for setpoints.
    //   (2) Explicit-duration — the caller supplies a finite duration (e.g.
    //       for setpoint trajectories). No infinity check.

    // --- Task-space validator: explicit-duration overload ---

    template <TaskTrajectory T, typename Controller>
    auto validate_trajectory(const Model &model, Data &data, const CollisionModel &col_model,
                             CollisionData &col_data, const T &trajectory,
                             const Eigen::Ref<const Eigen::VectorXd> &q_start,
                             Controller &controller, double explicit_duration,
                             const ValidationOptions &options = {}) -> ValidationResult {

        ValidationResult result;
        TaskTarget target;
        Eigen::VectorXd q_curr = q_start;

        const double dt = options.integration_dt;
        const int num_steps = static_cast<int>(explicit_duration / dt);
        double last_collision_t = -options.collision_check_dt;

        const auto dt_ns =
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::duration<double>(dt));

        JointState fb_state(model.dof);
        JointVelocity cmd_vel(model.dof);

        for (int s = 0; s <= num_steps; ++s) {
            const double t = std::min(s * dt, explicit_duration);

            const auto eval_status = trajectory.evaluate(t, target);
            if (eval_status != TrajectoryStatus::OK) {
                result.status = trajectory_to_validation_status(eval_status);
                result.failure_time = t;
                result.reason =
                    std::string{"trajectory_evaluate_returned_"} + to_string(eval_status);
                return result;
            }

            data.q = q_curr;
            compute_jacobians(model, data);

            if ((t - last_collision_t) >= options.collision_check_dt || s == num_steps) {
                update_geometry_poses(model, data, col_model, col_data);
                if (compute_collisions(col_model, col_data)) {
                    result.status = ValidationStatus::COLLISION;
                    result.failure_time = t;
                    result.reason = "collision_detected";
                    return result;
                }
                last_collision_t = t;
            }

            if (s < num_steps) {
                fb_state.q = q_curr;
                fb_state.v.setZero();
                fb_state.tau.setZero();

                const TaskControllerContext ctx{fb_state, target, dt_ns};

                if (controller.update(model, data, ctx, cmd_vel) != ControllerStatus::OK) {
                    result.status = ValidationStatus::CONTROLLER_FAILED;
                    result.failure_time = t;
                    result.reason = "controller_update_failed";
                    return result;
                }

                q_curr += cmd_vel.v * dt;
            }
        }

        return result;
    }

    // --- Task-space validator: default overload (reads duration from trajectory) ---

    template <TaskTrajectory T, typename Controller>
    auto validate_trajectory(const Model &model, Data &data, const CollisionModel &col_model,
                             CollisionData &col_data, const T &trajectory,
                             const Eigen::Ref<const Eigen::VectorXd> &q_start,
                             Controller &controller, const ValidationOptions &options = {})
        -> ValidationResult {

        const double duration = trajectory.duration();
        if (!std::isfinite(duration)) {
            return {ValidationStatus::TRAJECTORY_ERROR, -1.0,
                    "infinite_duration_use_explicit_duration_overload"};
        }

        return validate_trajectory(model, data, col_model, col_data, trajectory, q_start,
                                   controller, duration, options);
    }

    // --- Joint-space validator: explicit-duration overload ---

    template <JointTrajectory T>
    auto validate_trajectory(const Model &model, Data &data, const CollisionModel &col_model,
                             CollisionData &col_data, const T &trajectory, double explicit_duration,
                             const ValidationOptions &options = {}) -> ValidationResult {

        ValidationResult result;
        JointTarget target(model.dof);

        const double dt = options.integration_dt;
        const int num_steps = static_cast<int>(explicit_duration / dt);

        for (int s = 0; s <= num_steps; ++s) {
            const double t = std::min(s * dt, explicit_duration);

            const auto eval_status = trajectory.evaluate(t, target);
            if (eval_status != TrajectoryStatus::OK) {
                result.status = trajectory_to_validation_status(eval_status);
                result.failure_time = t;
                result.reason =
                    std::string{"trajectory_evaluate_returned_"} + to_string(eval_status);
                return result;
            }

            data.q = target.q;
            forward_kinematics(model, data);
            update_geometry_poses(model, data, col_model, col_data);

            if (compute_collisions(col_model, col_data)) {
                result.status = ValidationStatus::COLLISION;
                result.failure_time = t;
                result.reason = "collision_detected";
                return result;
            }
        }

        return result;
    }

    // --- Joint-space validator: default overload (reads duration from trajectory) ---

    template <JointTrajectory T>
    auto validate_trajectory(const Model &model, Data &data, const CollisionModel &col_model,
                             CollisionData &col_data, const T &trajectory,
                             const ValidationOptions &options = {}) -> ValidationResult {

        const double duration = trajectory.duration();
        if (!std::isfinite(duration)) {
            return {ValidationStatus::TRAJECTORY_ERROR, -1.0,
                    "infinite_duration_use_explicit_duration_overload"};
        }

        return validate_trajectory(model, data, col_model, col_data, trajectory, duration, options);
    }

}  // namespace xarm_geo
