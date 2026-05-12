#include <chrono>
#include <iostream>
#include <numbers>
#include <optional>
#include <string>
#include <thread>

#include <Eigen/Dense>
#include <coal/shape/geometric_shapes.h>

#include <xarm_geo/control/controller.h>
#include <xarm_geo/core/system.h>
#include <xarm_geo/diagnostics/logger.h>
#include <xarm_geo/examples/controllers/geometric_p_controller.h>
#include <xarm_geo/examples/controllers/joint_p_controller.h>
#include <xarm_geo/examples/trajectories/joint_ptp.h>
#include <xarm_geo/examples/trajectories/tilting_circle.h>
#include <xarm_geo/interfaces/hardware.h>
#include <xarm_geo/modelling/collision.h>
#include <xarm_geo/modelling/optimal_kinematics.h>
#include <xarm_geo/trajectory/adapters.h>
#include <xarm_geo/trajectory/trajectory.h>
#include <xarm_geo/trajectory/validate.h>
#include <xarm_geo/utils/model_builder.h>

// --- Optional Safety Enclosure ---
// Static box geometries around the robot to prevent workspace exits.
// Uncomment the call site in main() to enable.
// void add_safety_enclosure(xarm_geo::CollisionModel &col_model) {
//     double thickness = 0.05;
//     double size = 1.2;
//     double height = 2.0;
//
//     auto wall_geom = std::make_shared<coal::Box>(thickness, size, height);
//     auto floor_geom = std::make_shared<coal::Box>(size, size, thickness);
//
//     xarm_geo::manifold::SE3 front(xarm_geo::manifold::SO3::Identity(), {0.6, 0, height/2});
//     xarm_geo::manifold::SE3 back (xarm_geo::manifold::SO3::Identity(), {-0.6, 0, height/2});
//     xarm_geo::manifold::SE3 left (xarm_geo::manifold::SO3::Identity(), {0, 0.6, height/2});
//     xarm_geo::manifold::SE3 right(xarm_geo::manifold::SO3::Identity(), {0, -0.6, height/2});
//     xarm_geo::manifold::SE3 floor(xarm_geo::manifold::SO3::Identity(), {0, 0, -thickness/2});
//
//     col_model.add_geometry("wall_front", 0, front, wall_geom);
//     col_model.add_geometry("wall_back",  0, back,  wall_geom);
//     col_model.add_geometry("wall_left",  0, left,  wall_geom);
//     col_model.add_geometry("wall_right", 0, right, wall_geom);
//     col_model.add_geometry("floor",      0, floor, floor_geom);
//     col_model.add_all_collision_pairs();
// }

namespace {

    // --- Phase Runners ---

    // Joint-PTP execution loop: drives a JointPController against a JointTrajectory.
    template <xarm_geo::JointTrajectory T>
    auto run_joint_ptp_hw(xarm_geo::Hardware &hw, const xarm_geo::Model &model,
                          xarm_geo::Data &data, xarm_geo::controllers::JointPController &controller,
                          const T &traj, xarm_geo::JointState &state,
                          xarm_geo::JointVelocity &control_target) -> bool {

        const double duration = traj.duration();
        const double dt = 0.002;
        const auto dt_ns =
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::duration<double>(dt));

        xarm_geo::JointTarget target(traj.dof());
        auto next_tick = std::chrono::steady_clock::now();

        for (double t = 0.0; t < duration && hw.is_running(); t += dt) {
            if (hw.read(state) != xarm_geo::InterfaceStatus::OK) { return false; }
            if (traj.evaluate(t, target) != xarm_geo::TrajectoryStatus::OK) { return false; }

            const xarm_geo::JointControllerContext ctx{state, target, dt_ns};
            if (controller.update(model, data, ctx, control_target) !=
                xarm_geo::ControllerStatus::OK) {
                std::cerr << "JointPController update failed.\n";
                return false;
            }
            if (hw.write(control_target) != xarm_geo::InterfaceStatus::OK) { return false; }

            next_tick += std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                std::chrono::duration<double>(dt));
            std::this_thread::sleep_until(next_tick);
        }
        return true;
    }

    // Task-space execution loop: drives a GeometricPController against a TaskTrajectory.
    // When `logger` is non-null, one LogSample is recorded per tick.
    template <xarm_geo::TaskTrajectory T>
    auto run_task_space_hw(xarm_geo::Hardware &hw, const xarm_geo::Model &model,
                           xarm_geo::Data &data,
                           xarm_geo::controllers::GeometricPController &controller, const T &traj,
                           xarm_geo::JointState &state, xarm_geo::JointVelocity &control_target,
                           xarm_geo::diagnostics::DataLogger *logger) -> bool {

        const double duration = traj.duration();
        const double dt = 0.002;
        const auto dt_ns =
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::duration<double>(dt));

        xarm_geo::TaskTarget target;
        auto next_tick = std::chrono::steady_clock::now();
        std::int64_t tick = 0;

        for (double t = 0.0; t < duration && hw.is_running(); t += dt, ++tick) {
            if (hw.read(state) != xarm_geo::InterfaceStatus::OK) { return false; }
            if (traj.evaluate(t, target) != xarm_geo::TrajectoryStatus::OK) { return false; }

            const xarm_geo::TaskControllerContext ctx{state, target, dt_ns};
            const xarm_geo::ControllerStatus ctrl_status =
                controller.update(model, data, ctx, control_target);

            if (ctrl_status != xarm_geo::ControllerStatus::OK) {
                std::cerr << "GeometricPController update failed.\n";
                return false;
            }
            if (hw.write(control_target) != xarm_geo::InterfaceStatus::OK) { return false; }

            if (logger) {
                xarm_geo::diagnostics::LogSample s;
                xarm_geo::diagnostics::fill_task_sample(s, t, tick, state, target, data);
                s.controller_status = static_cast<std::uint8_t>(ctrl_status);
                xarm_geo::diagnostics::fill_velocity_diagnostics(s, controller);
                logger->log(s);
            }

            next_tick += std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                std::chrono::duration<double>(dt));
            std::this_thread::sleep_until(next_tick);
        }
        return true;
    }

}  // namespace

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <robot_ip> [--log <false|true>]\n";
        return 1;
    }
    const std::string robot_ip = argv[1];

    bool log_data = false;
    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--log" && i + 1 < argc) {
            const std::string val = argv[++i];
            log_data = (val == "1" || val == "true");
        }
    }

    try {
        // --- Setup ---
        const double circle_base_duration = 15.0;
        constexpr double half_speed = 2.0;
        const double transition_time = 4.0;

        xarm_geo::Model model = xarm_geo::build_model(6, "XI130412C23L45");
        xarm_geo::Data data(model);

        xarm_geo::CollisionModel col_model = xarm_geo::build_collision_model(model, true);
        // add_safety_enclosure(col_model);
        xarm_geo::CollisionData col_data(col_model);

        xarm_geo::Hardware hw(model.dof, robot_ip);
        if (!hw.is_running()) { return 1; }

        xarm_geo::JointState state(model.dof);
        xarm_geo::JointVelocity control_target(model.dof);
        hw.read(state);

        // --- Define Trajectory Anchor ---
        Eigen::VectorXd q_home = Eigen::VectorXd::Zero(model.dof);
        q_home[0] = 1.5 * std::numbers::pi;

        const double q0 = q_home[0];
        const Eigen::Vector3d center(0.35 * std::cos(q0), 0.35 * std::sin(q0), 0.35);
        const xarm_geo::manifold::SO3 rot = xarm_geo::manifold::SO3::exp(
            Eigen::Vector3d::UnitZ() * (q0 - (1.5 * std::numbers::pi)));
        const xarm_geo::manifold::SE3 anchor(rot, center);

        // Half-speed tilting circle (TimeScaledTask wrapping the canonical 15s curve).
        xarm_geo::trajectories::TiltingCircle circle_inner(anchor, circle_base_duration);
        xarm_geo::TimeScaledTask circle_traj{std::move(circle_inner), half_speed};

        // --- Pre-Flight: Find A Collision-Free Start Pose ---
        xarm_geo::TaskTarget start_target;
        if (circle_traj.evaluate(0.0, start_target) != xarm_geo::TrajectoryStatus::OK) {
            std::cerr << "[ABORT] Failed to Evaluate Start Pose\n";
            return 1;
        }

        if (!xarm_geo::optimal_inverse_kinematics(model, data, col_model, col_data, q_home,
                                                  start_target.pose)) {
            std::cerr << "[ABORT] No Collision-Free Start Pose Found!\n";
            return 1;
        }
        const Eigen::VectorXd q_start = data.q_out;

        // --- Controller Setup ---
        xarm_geo::controllers::JointPController joint_controller(model);
        joint_controller.kp.setConstant(5.0);
        joint_controller.use_feedforward = true;
        joint_controller.constraint_aware = true;

        xarm_geo::controllers::GeometricPController p_controller(model);
        p_controller.gains.kp_pos.setConstant(8.0);
        p_controller.gains.kp_rot.setConstant(8.0);
        p_controller.use_feedforward = true;
        p_controller.constraint_aware = true;
        p_controller.attach_collision(col_model, col_data);

        const auto val = xarm_geo::validate_trajectory(model, data, col_model, col_data,
                                                       circle_traj, q_start, p_controller);
        if (val.status != xarm_geo::ValidationStatus::OK) {
            std::cerr << "[ABORT] Safety Violation: " << val.reason << "\n";
            return 1;
        }

        // --- Logger Setup ---
        //
        // Constructed before Phase 2 and destroyed (flushed) immediately after,
        // so the CSV lands on disk before the return-to-home phase begins.
        std::optional<xarm_geo::diagnostics::DataLogger> logger;
        if (log_data) {
            const std::string trial_name = xarm_geo::diagnostics::make_trial_name(
                "hardware", xarm_geo::controllers::GeometricPController::kName,
                xarm_geo::trajectories::TiltingCircle::kName,
                /*constraint=*/p_controller.constraint_aware,
                /*feedforward=*/p_controller.use_feedforward);

            logger.emplace(model, xarm_geo::diagnostics::DataLogger::Config{
                                      .output_path = "tests/results/" + trial_name + ".csv",
                                      .trial_name = trial_name,
                                  });
        }

        // --- Execution ---
        std::cout << "[PHASE 0] Moving to Home... ";
        const xarm_geo::trajectories::JointPTP h_traj(state.q, q_home, transition_time);
        run_joint_ptp_hw(hw, model, data, joint_controller, h_traj, state, control_target);
        std::cout << "Done.\n[PHASE 1] Approaching Circle... ";

        const xarm_geo::trajectories::JointPTP a_traj(q_home, q_start, transition_time);
        run_joint_ptp_hw(hw, model, data, joint_controller, a_traj, state, control_target);
        std::cout
            << "Done.\n[PHASE 2] Executing Tilting Circle (Half Speed via TimeScaledTask)...\n";

        run_task_space_hw(hw, model, data, p_controller, circle_traj, state, control_target,
                          logger ? &(*logger) : nullptr);

        // logger goes out of scope here and flushes to disk before Phase 3 begins.
        logger.reset();

        std::cout << "[PHASE 3] Returning Home... ";
        hw.read(state);
        const xarm_geo::trajectories::JointPTP r_traj(state.q, q_home, transition_time);
        run_joint_ptp_hw(hw, model, data, joint_controller, r_traj, state, control_target);

        std::cout << "SUCCESS.\n";
        hw.shutdown();

    } catch (const std::exception &e) {
        std::cerr << "CRITICAL: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
