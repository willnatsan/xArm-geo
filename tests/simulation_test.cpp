#include <format>
#include <fstream>
#include <iostream>
#include <numbers>
#include <stdexcept>
#include <thread>

#include <xarm_geo/control/sample_controllers.h>
#include <xarm_geo/core/system.h>
#include <xarm_geo/interfaces/simulation.h>
#include <xarm_geo/modelling/collision.h>
#include <xarm_geo/modelling/kinematics.h>
#include <xarm_geo/trajectory/sample_trajectories.h>
#include <xarm_geo/utils/model_builder.h>

struct TestParams {
    bool geometric = true;
    int trajectory_mode = 2;
    bool show_marker = true;
    bool log_data = false;
    bool check_trajectory = false;
    bool torque_mode = false;  // false: kinematic P + JointVelocity; true: dynamic PD + JointTorque
    bool feedforward = true;   // controller FF toggle
    bool constraint_aware = false;  // route through optimal_inverse_diff_kinematics (kinematic) /
                                    // asif_filter (dynamic)
};

auto run_joint_ptp(xarm_geo::Simulation &sim, const xarm_geo::trajectories::JointPTP &traj,
                   double duration, xarm_geo::JointState &state,
                   xarm_geo::JointVelocity &control_target) -> bool {

    double t = 0.0;
    double physics_dt = 0.002;
    double render_dt = 1.0 / 60.0;
    double last_render_t = 0.0;
    double kp_joint = 5.0;

    xarm_geo::JointSpaceTarget joint_target(state.q.size());

    while (t < duration && sim.is_running()) {
        if (sim.read(state) != xarm_geo::InterfaceStatus::OK) { return false; }

        if (traj.evaluate(t, joint_target) != xarm_geo::TrajectoryStatus::OK) { return false; }

        for (size_t i = 0; i < control_target.v.size(); ++i) {
            control_target.v[i] = joint_target.v[i] + (kp_joint * (joint_target.q[i] - state.q[i]));
        }

        if (sim.write(control_target) != xarm_geo::InterfaceStatus::OK) { return false; }

        sim.step();
        t += physics_dt;

        if (t - last_render_t >= render_dt) {
            auto start_render = std::chrono::steady_clock::now();

            sim.update_scene();
            sim.render();
            last_render_t = t;

            std::this_thread::sleep_until(start_render + std::chrono::duration<double>(render_dt));
        }
    }

    control_target.v.setZero();
    sim.write(control_target);
    sim.step();

    return true;
}

template <xarm_geo::TaskSpaceTrajectory T>
auto run_simulation(xarm_geo::Model &model, xarm_geo::Data &data,
                    xarm_geo::CollisionModel &col_model, xarm_geo::CollisionData &col_data,
                    xarm_geo::Simulation &sim, const T &trajectory, double duration,
                    const Eigen::VectorXd &q_home, const TestParams &params) -> int {

    xarm_geo::JointState state(model.dof);
    xarm_geo::JointVelocity control_target(model.dof);

    if (sim.read(state) != xarm_geo::InterfaceStatus::OK) { return 1; }

    xarm_geo::TaskSpaceTarget task_target;
    if (trajectory.evaluate(0.0, task_target) != xarm_geo::TrajectoryStatus::OK) { return 1; }

    // --- Layer 1: Collision-Aware IK for Start Pose ---

    bool ik_success =
        xarm_geo::inverse_kinematics(model, data, col_model, col_data, q_home, task_target.pose);
    if (!ik_success) {
        std::cerr << "IK FAILED for Trajectory Start Pose (No Collision-Free Solution)\n";
        return 1;
    }
    Eigen::VectorXd q_start = data.q_out;

    // --- Layer 2: Pre-Flight Trajectory Validation ---

    if (params.check_trajectory) {
        std::cout << "\n[PRE-FLIGHT] Validating Trajectory for Collision-Free Feasibility...\n";

        auto validation = xarm_geo::validate_trajectory(model, data, col_model, col_data,
                                                        trajectory, duration, q_start);
        if (!validation.valid) {
            std::cerr << "Trajectory Validation FAILED at t=" << validation.failure_time << ": "
                      << validation.reason << "\n";
            return 1;
        }

        std::cout << "[PRE-FLIGHT] Trajectory Validated!\n";
    }

    // --- PHASE 1: HOME TO START ---

    double start_duration = 3.0;

    std::cout << "\n[PHASE 1] Moving from Home to Trajectory Start...\n";
    xarm_geo::trajectories::JointPTP approach_traj(q_home, q_start, start_duration);

    if (!run_joint_ptp(sim, approach_traj, start_duration, state, control_target)) {
        std::cerr << "Failed during Phase 1 Approach.\n";
        return 1;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    // --- PHASE 2: MAIN TRAJECTORY EXECUTION ---

    std::cout << "\n[PHASE 2] Executing Task Space Trajectory...\n";

    std::ofstream log_file;
    if (params.log_data) {
        std::string filename =
            std::format("tests/results/trajectory_log_{}.csv", params.trajectory_mode);
        log_file.open(filename);
        log_file << "time,target_x,target_y,target_z,actual_x,actual_y,actual_z,"
                 << "target_roll,target_pitch,target_yaw,actual_roll,actual_pitch,actual_yaw\n";
    }

    std::cout << "Starting Simulation.\n"
              << "  Torque Mode: " << (params.torque_mode ? "ON" : "OFF") << "\n"
              << "  Geometric Mode: " << (params.geometric ? "ON" : "OFF") << "\n"
              << "  Feedforward: " << (params.feedforward ? "ON" : "OFF") << "\n"
              << "  Constraint Aware: " << (params.constraint_aware ? "ON" : "OFF") << "\n"
              << "  Trajectory: " << params.trajectory_mode << "\n"
              << "  Marker: " << (params.show_marker ? "ON" : "OFF") << "\n"
              << "  Data Logging: " << (params.log_data ? "ON" : "OFF") << "\n";

    double t = 0.0;
    double physics_dt = 0.002;
    double render_dt = 1.0 / 60.0;
    double last_render_t = 0.0;

    // --- Controller Setup ---
    //
    // Kinematic mode: GeometricPController emits JointVelocity. Defaults
    // (kp_pos = kp_rot = 8) preserve the previously hard-coded behaviour;

    xarm_geo::GeometricPController p_controller(model);

    p_controller.gains.kp_pos.setConstant(8.0);
    p_controller.gains.kp_rot.setConstant(8.0);

    p_controller.use_feedforward = params.feedforward;
    p_controller.constraint_aware = params.constraint_aware;
    p_controller.ik_options.apply_scaling = true;
    if (params.constraint_aware) { p_controller.attach_collision(col_model, col_data); }

    // Dynamic mode: GeometricPDController emits JointTorque. K_D is set to
    // a critically-damped baseline relative to K_P.

    xarm_geo::GeometricPDController pd_controller(model);

    pd_controller.gains.kp_pos.setConstant(100.0);
    pd_controller.gains.kp_rot.setConstant(50.0);
    pd_controller.gains.kd_lin.setConstant(20.0);
    pd_controller.gains.kd_ang.setConstant(10.0);

    pd_controller.use_feedforward = params.feedforward;
    pd_controller.compensate_bias = true;  // model.gravity is non-zero in torque mode
    pd_controller.constraint_aware = params.constraint_aware;
    if (params.constraint_aware) { pd_controller.attach_collision(col_model, col_data); }

    xarm_geo::JointTorque torque_target(model.dof);
    const auto dt_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::duration<double>(physics_dt));

    while (t < duration && sim.is_running()) {
        if (sim.read(state) != xarm_geo::InterfaceStatus::OK) { break; }

        if (trajectory.evaluate(t, task_target) != xarm_geo::TrajectoryStatus::OK) { break; }

        if (params.torque_mode) {
            // --- Dynamic Mode: GeometricPDController -> JointTorque ---
            const xarm_geo::TaskControllerContext ctx{state, task_target, dt_ns};
            if (pd_controller.update(model, data, ctx, torque_target) !=
                xarm_geo::ControllerStatus::OK) {
                std::cerr << "GeometricPDController update failed.\n";
                break;
            }
            if (sim.write(torque_target) != xarm_geo::InterfaceStatus::OK) { break; }
        } else if (params.geometric) {
            // --- Kinematic Mode: GeometricPController -> JointVelocity ---
            const xarm_geo::TaskControllerContext ctx{state, task_target, dt_ns};
            if (p_controller.update(model, data, ctx, control_target) !=
                xarm_geo::ControllerStatus::OK) {
                std::cerr << "GeometricPController update failed.\n";
                break;
            }
            if (sim.write(control_target) != xarm_geo::InterfaceStatus::OK) { break; }
        } else {
            // --- Non-Geometric Baseline (Euler-RPY) ---
            // Hand-rolled feedforward + P, kept verbatim as a comparison
            // baseline against the geometric controllers.
            data.q = state.q;
            xarm_geo::compute_jacobians(model, data);

            xarm_geo::manifold::SE3::Twist cmd_twist;
            double kp = 8;

            Eigen::Vector3d pos_err_space = task_target.pose.r3() - data.ee_pose.r3();

            Eigen::Matrix3d R_target = task_target.pose.so3().matrix();
            double target_pitch = std::asin(std::clamp(-R_target(2, 0), -1.0, 1.0));
            double target_yaw;
            double target_roll;

            if (std::abs(std::cos(target_pitch)) > 1e-6) {
                target_yaw = std::atan2(R_target(1, 0), R_target(0, 0));
                target_roll = std::atan2(R_target(2, 1), R_target(2, 2));
            } else {
                target_yaw = 0.0;
                target_roll = std::atan2(-R_target(0, 1), R_target(1, 1));
            }
            Eigen::Vector3d target_euler(target_yaw, target_pitch, target_roll);

            Eigen::Matrix3d R_curr = data.ee_pose.so3().matrix();
            double curr_pitch = std::asin(std::clamp(-R_curr(2, 0), -1.0, 1.0));
            double curr_yaw;
            double curr_roll;

            if (std::abs(std::cos(curr_pitch)) > 1e-6) {
                curr_yaw = std::atan2(R_curr(1, 0), R_curr(0, 0));
                curr_roll = std::atan2(R_curr(2, 1), R_curr(2, 2));
            } else {
                curr_yaw = 0.0;
                curr_roll = std::atan2(-R_curr(0, 1), R_curr(1, 1));
            }
            Eigen::Vector3d curr_euler(curr_yaw, curr_pitch, curr_roll);

            Eigen::Vector3d rot_err_space = target_euler - curr_euler;
            for (int i = 0; i < 3; ++i) {
                while (rot_err_space[i] > std::numbers::pi) {
                    rot_err_space[i] -= 2.0 * std::numbers::pi;
                }
                while (rot_err_space[i] < -std::numbers::pi) {
                    rot_err_space[i] += 2.0 * std::numbers::pi;
                }
            }

            double d_yaw = kp * rot_err_space[0];
            double d_pitch = kp * rot_err_space[1];
            double d_roll = kp * rot_err_space[2];

            Eigen::Vector3d omega_space =
                Eigen::Vector3d::UnitZ() * d_yaw +
                Eigen::AngleAxisd(curr_yaw, Eigen::Vector3d::UnitZ()) *
                    (Eigen::Vector3d::UnitY() * d_pitch) +
                Eigen::AngleAxisd(curr_yaw, Eigen::Vector3d::UnitZ()) *
                    Eigen::AngleAxisd(curr_pitch, Eigen::Vector3d::UnitY()) *
                    (Eigen::Vector3d::UnitX() * d_roll);

            Eigen::Matrix3d Rt = R_curr.transpose();
            cmd_twist.head<3>() = Rt * (kp * pos_err_space);  // Body linear velocity
            cmd_twist.tail<3>() = Rt * omega_space;           // Body angular velocity

            xarm_geo::manifold::SE3 pose_err = data.ee_pose.inverse() * task_target.pose;
            cmd_twist += pose_err.Ad() * task_target.twist;

            xarm_geo::IKOptions options({.apply_scaling = true});
            xarm_geo::inverse_diff_kinematics(model, data, cmd_twist, options);

            control_target.v = data.v_out;
            if (sim.write(control_target) != xarm_geo::InterfaceStatus::OK) { break; }
        }

        if (params.log_data) {
            Eigen::Vector3d actual_euler =
                data.ee_pose.so3().quat().toRotationMatrix().eulerAngles(2, 1, 0);
            Eigen::Vector3d target_euler =
                task_target.pose.so3().quat().toRotationMatrix().eulerAngles(2, 1, 0);

            log_file << t << "," << task_target.pose.r3().x() << "," << task_target.pose.r3().y()
                     << "," << task_target.pose.r3().z() << "," << data.ee_pose.r3().x() << ","
                     << data.ee_pose.r3().y() << "," << data.ee_pose.r3().z() << ","
                     << target_euler[2] << "," << target_euler[1] << "," << target_euler[0] << ","
                     << actual_euler[2] << "," << actual_euler[1] << "," << actual_euler[0] << "\n";
        }

        sim.step();
        t += physics_dt;

        if (t - last_render_t >= render_dt) {
            if (params.show_marker) { sim.set_marker(task_target.pose); }

            auto start_render = std::chrono::steady_clock::now();

            sim.update_scene();
            sim.render();
            last_render_t = t;

            std::this_thread::sleep_until(start_render + std::chrono::duration<double>(render_dt));
        }
    }

    // --- PHASE 3: END TO HOME ---

    std::cout << "\n[PHASE 3] Task Complete. Returning to Home...\n";
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    // Phase 3 uses velocity-mode joint-PTP regardless of Phase 2's actuator
    // mode. Switch back if we were in torque mode.
    if (params.torque_mode) { sim.set_control_mode(xarm_geo::ControlMode::VELOCITY); }

    double end_duration = 3.0;

    if (sim.read(state) != xarm_geo::InterfaceStatus::OK) {
        std::cerr << "Failed to Read Simulation State for Phase 3 Return.\n";
        return 1;
    }

    xarm_geo::trajectories::JointPTP return_traj(state.q, q_home, end_duration);
    if (!run_joint_ptp(sim, return_traj, end_duration, state, control_target)) {
        std::cerr << "Failed during Phase 3 Return.\n";
        return 1;
    }

    sim.shutdown();
    std::cout << "Simulation Sequence Completed Safely.\n";

    if (params.log_data) { log_file.close(); }

    return 0;
}

auto main(int argc, char *argv[]) -> int {

    TestParams params;

    // --- COMMAND LINE ARGUMENT PARSING ---

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            std::cout
                << "Usage: " << argv[0] << " [options]\n"
                << "Options:\n"
                << "  --geometric <false|true> -> Toggle Geometric Controller (default: true)\n"
                << "  --trajectory <0|1|2|3|4> -> Select Trajectory Mode (default: 2)\n"
                << "  --marker <false|true> -> Show Target Marker in simulation (default: true)\n"
                << "  --log <false|true> -> Log Data to CSV File (default: false)\n"
                << "  --validate <false|true> -> Validate Trajectory (default: false)\n"
                << "  --torque <false|true> -> Use Dynamic PD Controller in Torque Mode (default: "
                   "false)\n"
                << "  --feedforward <false|true> -> Toggle Controller Feedforward Term (default: "
                   "true)\n"
                << "  --constraint_aware <false|true> -> Route through Optimal IDK (kinematic) / "
                   "ASIF (dynamic) (default: false)\n";
            return 0;
        }

        if (arg == "--geometric" && i + 1 < argc) {
            params.geometric = (std::string(argv[++i]) == "1" || std::string(argv[i]) == "true");
        } else if (arg == "--trajectory" && i + 1 < argc) {
            try {
                params.trajectory_mode = std::stoi(argv[++i]);
                if (params.trajectory_mode < 0 || params.trajectory_mode > 4) {
                    throw std::out_of_range("Mode must be in range [0, 4]");
                }
            } catch (const std::exception &e) {
                std::cerr << "Warning: Invalid Trajectory Mode. Defaulting to Mode 2.\n";
            }
        } else if (arg == "--marker" && i + 1 < argc) {
            params.show_marker = (std::string(argv[++i]) == "1" || std::string(argv[i]) == "true");
        } else if (arg == "--log" && i + 1 < argc) {
            params.log_data = (std::string(argv[++i]) == "1" || std::string(argv[i]) == "true");
        } else if (arg == "--validate" && i + 1 < argc) {
            params.check_trajectory =
                (std::string(argv[++i]) == "1" || std::string(argv[i]) == "true");
        } else if (arg == "--torque" && i + 1 < argc) {
            params.torque_mode = (std::string(argv[++i]) == "1" || std::string(argv[i]) == "true");
        } else if (arg == "--feedforward" && i + 1 < argc) {
            params.feedforward = (std::string(argv[++i]) == "1" || std::string(argv[i]) == "true");
        } else if (arg == "--constraint_aware" && i + 1 < argc) {
            params.constraint_aware =
                (std::string(argv[++i]) == "1" || std::string(argv[i]) == "true");
        }
    }

    // --- SETUP ---

    xarm_geo::Model model = xarm_geo::build_model(6, "XI130412C23L45");

    // Torque-mode control needs gravity baked into RNEA so compute_bias_forces
    // (called by the PD controller) returns the correct h(q, v) for gravity
    // compensation. Velocity-mode leaves gravity at zero (xArm SDK / MuJoCo
    // velocity actuators handle compensation externally).
    if (params.torque_mode) { model.gravity = Eigen::Vector3d{0.0, 0.0, -9.81}; }

    xarm_geo::Data data(model);
    xarm_geo::Simulation sim(model.mjcf_file);

    if (params.torque_mode) { sim.set_control_mode(xarm_geo::ControlMode::TORQUE); }

    // Build Collision Model & Data
    xarm_geo::CollisionModel col_model = xarm_geo::build_collision_model(model, true);
    xarm_geo::CollisionData col_data(col_model);

    Eigen::VectorXd q_home = Eigen::VectorXd::Zero(model.dof);
    q_home[0] = 1.5 * std::numbers::pi;

    sim.set_joint_positions(q_home);

    xarm_geo::JointState state(model.dof);
    if (sim.read(state) != xarm_geo::InterfaceStatus::OK) { return 1; }

    data.q = state.q;
    xarm_geo::compute_jacobians(model, data);

    // Creating Anchor Pose (Centre of Trajectory)
    double q0 = q_home[0];
    Eigen::Vector3d center(0.35 * std::cos(q0), 0.35 * std::sin(q0), 0.35);

    // Rotation about Z-axis
    xarm_geo::manifold::SO3 rot =
        xarm_geo::manifold::SO3::exp(Eigen::Vector3d::UnitZ() * (q0 - (1.5 * std::numbers::pi)));

    xarm_geo::manifold::SE3 anchor_pose(rot, center);

    double traj_duration = 15.0;

    if (params.trajectory_mode == 0) {
        xarm_geo::trajectories::FigureEight traj(anchor_pose, traj_duration);
        return run_simulation(model, data, col_model, col_data, sim, traj, traj_duration, q_home,
                              params);
    }
    if (params.trajectory_mode == 1) {
        xarm_geo::trajectories::WingInspection traj(anchor_pose, traj_duration);
        return run_simulation(model, data, col_model, col_data, sim, traj, traj_duration, q_home,
                              params);
    }
    if (params.trajectory_mode == 2) {
        xarm_geo::trajectories::PipeInspection traj(anchor_pose, traj_duration);
        return run_simulation(model, data, col_model, col_data, sim, traj, traj_duration, q_home,
                              params);
    }
    if (params.trajectory_mode == 3) {
        xarm_geo::trajectories::InnerCavityScan traj(anchor_pose, traj_duration);
        return run_simulation(model, data, col_model, col_data, sim, traj, traj_duration, q_home,
                              params);
    }
    if (params.trajectory_mode == 4) {
        xarm_geo::trajectories::TiltingCircle traj(anchor_pose, traj_duration);
        return run_simulation(model, data, col_model, col_data, sim, traj, traj_duration, q_home,
                              params);
    }
}
