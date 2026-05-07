#include <chrono>
#include <iostream>
#include <numbers>
#include <string>
#include <thread>

#include <Eigen/Dense>
#include <coal/shape/geometric_shapes.h>

#include <xarm_geo/control/controller.h>
#include <xarm_geo/core/system.h>
#include <xarm_geo/examples/controllers/geometric_p_controller.h>
#include <xarm_geo/interfaces/hardware.h>
#include <xarm_geo/modelling/collision.h>
#include <xarm_geo/modelling/optimal_kinematics.h>
#include <xarm_geo/trajectory/sample_trajectories.h>
#include <xarm_geo/trajectory/trajectory.h>
#include <xarm_geo/utils/model_builder.h>

// --- Helper: Safety Enclosure ---
// Adds static box geometries around the robot to prevent workspace exits.
// Disabled by default; uncomment the call site in main() to enable.
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

    // --- Simple Joint-Space PD Controller ---
    //
    // File-local concrete subclass of KinematicJointControllerBase. Computes
    //   v = v_ref + kp * (q_ref - q)
    class JointPDController final : public xarm_geo::KinematicJointControllerBase {
    public:
        explicit JointPDController(const xarm_geo::Model &model)
            : xarm_geo::KinematicJointControllerBase(model) {}

        // --- Public Configuration ---
        double kp = 5.0;

    protected:
        auto compute_command_velocity(const xarm_geo::Model & /*model*/, xarm_geo::Data & /*data*/,
                                      const xarm_geo::JointControllerContext &ctx,
                                      xarm_geo::JointVelocity &v_ctrl) noexcept -> bool override {
            v_ctrl.v = ctx.ref.v + kp * (ctx.ref.q - ctx.fb.q);
            return true;
        }
    };

    // --- Phase Runners ---

    // Joint-PTP execution loop. Drives a JointPDController against the JointSpaceTrajectory.
    template <xarm_geo::JointSpaceTrajectory T>
    auto run_joint_ptp_hw(xarm_geo::Hardware &hw, const xarm_geo::Model &model,
                          xarm_geo::Data &data, JointPDController &controller, const T &traj,
                          double duration, xarm_geo::JointState &state,
                          xarm_geo::JointVelocity &control_target) -> bool {

        const double dt = 0.002;
        const auto dt_ns =
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::duration<double>(dt));

        xarm_geo::JointSpaceTarget target(model.dof);
        auto next_tick = std::chrono::steady_clock::now();

        for (double t = 0.0; t < duration && hw.is_running(); t += dt) {
            if (hw.read(state) != xarm_geo::InterfaceStatus::OK) { return false; }
            if (traj.evaluate(t, target) != xarm_geo::TrajectoryStatus::OK) { return false; }

            const xarm_geo::JointControllerContext ctx{state, target, dt_ns};
            if (controller.update(model, data, ctx, control_target) !=
                xarm_geo::ControllerStatus::OK) {
                std::cerr << "JointPDController update failed.\n";
                return false;
            }
            if (hw.write(control_target) != xarm_geo::InterfaceStatus::OK) { return false; }

            next_tick += std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                std::chrono::duration<double>(dt));
            std::this_thread::sleep_until(next_tick);
        }
        return true;
    }

    // Task-space execution loop. Drives a GeometricPController against the TaskSpaceTrajectory.
    template <xarm_geo::TaskSpaceTrajectory T>
    auto run_task_space_hw(xarm_geo::Hardware &hw, const xarm_geo::Model &model,
                           xarm_geo::Data &data, xarm_geo::GeometricPController &controller,
                           const T &traj, double duration, xarm_geo::JointState &state,
                           xarm_geo::JointVelocity &control_target) -> bool {

        const double dt = 0.002;
        const auto dt_ns =
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::duration<double>(dt));

        xarm_geo::TaskSpaceTarget target;
        auto next_tick = std::chrono::steady_clock::now();

        for (double t = 0.0; t < duration && hw.is_running(); t += dt) {
            if (hw.read(state) != xarm_geo::InterfaceStatus::OK) { return false; }
            if (traj.evaluate(t, target) != xarm_geo::TrajectoryStatus::OK) { return false; }

            const xarm_geo::TaskControllerContext ctx{state, target, dt_ns};
            if (controller.update(model, data, ctx, control_target) !=
                xarm_geo::ControllerStatus::OK) {
                std::cerr << "GeometricPController update failed.\n";
                return false;
            }
            if (hw.write(control_target) != xarm_geo::InterfaceStatus::OK) { return false; }

            next_tick += std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                std::chrono::duration<double>(dt));
            std::this_thread::sleep_until(next_tick);
        }
        return true;
    }

}  // namespace

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <robot_ip>\n";
        return 1;
    }
    const std::string robot_ip = argv[1];

    try {
        // --- Setup ---
        const double circle_duration = 30.0;  // HALF SPEED (original was 15.0)
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

        xarm_geo::trajectories::TiltingCircle circle_traj(anchor, circle_duration);

        // --- Pre-Flight: Find a Collision-Free Start Pose ---
        xarm_geo::TaskSpaceTarget start_target;
        if (circle_traj.evaluate(0.0, start_target) != xarm_geo::TrajectoryStatus::OK) {
            std::cerr << "[ABORT] Failed to Evaluate Start Pose\n";
            return 1;
        }

        // optimal_inverse_kinematics enforces collision avoidance and joint
        // limits as hard constraints inside the QP.
        if (!xarm_geo::optimal_inverse_kinematics(model, data, col_model, col_data, q_home,
                                                  start_target.pose)) {
            std::cerr << "[ABORT] No Collision-Free Start Pose Found!\n";
            return 1;
        }
        const Eigen::VectorXd q_start = data.q_out;

        const auto val = xarm_geo::validate_trajectory(model, data, col_model, col_data,
                                                       circle_traj, circle_duration, q_start);
        if (!val.valid) {
            std::cerr << "[ABORT] Safety Violation: " << val.reason << "\n";
            return 1;
        }

        // --- Controller Setup ---
        // Joint-space PD controller for the PTP phases.
        JointPDController joint_controller(model);
        joint_controller.constraint_aware = true;
        joint_controller.kp = 5.0;

        // Task-space geometric P controller
        xarm_geo::GeometricPController p_controller(model);
        p_controller.gains.kp_pos.setConstant(8.0);
        p_controller.gains.kp_rot.setConstant(8.0);
        p_controller.use_feedforward = true;
        p_controller.constraint_aware = true;
        p_controller.attach_collision(col_model, col_data);

        // --- Execution ---
        std::cout << "[PHASE 0] Moving to Home... ";
        const xarm_geo::trajectories::JointPTP h_traj(state.q, q_home, transition_time);
        run_joint_ptp_hw(hw, model, data, joint_controller, h_traj, transition_time, state,
                         control_target);
        std::cout << "Done.\n[PHASE 1] Approaching Circle... ";

        const xarm_geo::trajectories::JointPTP a_traj(q_home, q_start, transition_time);
        run_joint_ptp_hw(hw, model, data, joint_controller, a_traj, transition_time, state,
                         control_target);
        std::cout << "Done.\n[PHASE 2] Executing Tilting Circle (Half Speed)...\n";

        run_task_space_hw(hw, model, data, p_controller, circle_traj, circle_duration, state,
                          control_target);

        std::cout << "[PHASE 3] Returning Home... ";
        hw.read(state);
        const xarm_geo::trajectories::JointPTP r_traj(state.q, q_home, transition_time);
        run_joint_ptp_hw(hw, model, data, joint_controller, r_traj, transition_time, state,
                         control_target);

        std::cout << "SUCCESS.\n";
        hw.shutdown();

    } catch (const std::exception &e) {
        std::cerr << "CRITICAL: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
