#include <format>
#include <fstream>
#include <iostream>
#include <numbers>
#include <thread>

#include <xarm_geo/core/system.h>
#include <xarm_geo/interfaces/simulation.h>
#include <xarm_geo/modelling/kinematics.h>
#include <xarm_geo/trajectory/sample_trajectories.h>
#include <xarm_geo/utils/model_builder.h>

template <xarm_geo::Trajectory T>
auto run_simulation(const T &trajectory, double duration, bool use_geometric_controller,
                    int trajectory_mode, bool show_marker, bool log_data) -> int {

    xarm_geo::Model model = xarm_geo::build_model(6, "XI130412C23L45");
    xarm_geo::Data data(model);
    xarm_geo::Simulation sim(model.mjcf_file);

    xarm_geo::JointPosition pos_curr(model.dof);
    xarm_geo::JointVelocity vel_target(model.dof);

    double t = 0.0;
    double physics_dt = 0.002;
    double render_dt = 1.0 / 60.0;
    double last_render_t = 0.0;

    // --- SET INITIAL CONFIGURATION (IK SNAP TO TRAJECTORY START) ---

    xarm_geo::manifold::SE3 initial_pose = trajectory.evaluate(0.0);

    if (sim.read(pos_curr) != xarm_geo::InterfaceStatus::OK) {
        std::cerr << "Error: Failed to read Initial Joint Positions.\n";
        return 1;
    }

    bool ik_success = xarm_geo::inverse_kinematics(model, data, pos_curr.q, initial_pose);
    if (!ik_success) {
        std::cerr << "IK FAILED\n";
        return 1;
    }
    sim.set_joint_positions(data.q_out);

    // --- CSV LOGGING SETUP ---

    std::string filename = std::format("tests/results/trajectory_log_{}.csv", trajectory_mode);
    std::ofstream log_file(filename);
    log_file << "time,target_x,target_y,target_z,actual_x,actual_y,actual_z,"
             << "target_roll,target_pitch,target_yaw,actual_roll,actual_pitch,actual_yaw\n";

    // --- STARTING SIMULATION ---

    std::cout << "Starting Simulation.\n"
              << "  Geometric Mode: " << (use_geometric_controller ? "ON" : "OFF") << "\n"
              << "  Trajectory: " << trajectory_mode << "\n"
              << "  Marker: " << (show_marker ? "ON" : "OFF") << "\n"
              << "  Data Logging: " << (log_data ? "ON" : "OFF") << "\n";

    while (t < duration && sim.is_running()) {
        if (sim.read(pos_curr) != xarm_geo::InterfaceStatus::OK) {
            std::cerr << "Error: Failed to read Current Joint Positions. Halting Loop.\n";
            break;
        }
        xarm_geo::compute_jacobians(model, data, pos_curr.q);

        // Generate target pose for current time t
        xarm_geo::manifold::SE3 target_pose = trajectory.evaluate(t);
        Eigen::Vector3d target_pos = target_pose.r3();

        xarm_geo::manifold::SE3::Twist cmd_twist;
        double kp = 10;  // Tracking gain

        if (use_geometric_controller) {
            xarm_geo::manifold::SE3 pose_err_body = data.ee_pose.inverse() * target_pose;
            xarm_geo::manifold::SE3::Twist twist_err_body = pose_err_body.log();

            cmd_twist = kp * twist_err_body;

        } else {
            Eigen::Vector3d pos_err_space = target_pos - data.ee_pose.r3();
            Eigen::Vector3d target_euler = target_pose.so3().matrix().eulerAngles(2, 1, 0);

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
                while (rot_err_space[i] > std::numbers::pi)
                    rot_err_space[i] -= 2.0 * std::numbers::pi;
                while (rot_err_space[i] < -std::numbers::pi)
                    rot_err_space[i] += 2.0 * std::numbers::pi;
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
            cmd_twist.head<3>() = Rt * (kp * pos_err_space);
            cmd_twist.tail<3>() = Rt * omega_space;
        }

        xarm_geo::inverse_diff_kinematics(model, data, cmd_twist);
        vel_target.v = data.v_out;
        if (sim.write(vel_target) != xarm_geo::InterfaceStatus::OK) break;

        if (log_data) {
            Eigen::Vector3d actual_euler =
                data.ee_pose.so3().quat().toRotationMatrix().eulerAngles(2, 1, 0);
            Eigen::Vector3d target_euler =
                target_pose.so3().quat().toRotationMatrix().eulerAngles(2, 1, 0);

            log_file << t << "," << target_pos.x() << "," << target_pos.y() << "," << target_pos.z()
                     << "," << data.ee_pose.r3().x() << "," << data.ee_pose.r3().y() << ","
                     << data.ee_pose.r3().z() << "," << target_euler[2] << "," << target_euler[1]
                     << "," << target_euler[0] << "," << actual_euler[2] << "," << actual_euler[1]
                     << "," << actual_euler[0] << "\n";
        }

        sim.step();
        t += physics_dt;

        if (t - last_render_t >= render_dt) {
            if (show_marker) { sim.set_marker(target_pose); }
            sim.update_scene();
            sim.render();
            last_render_t = t;
            std::this_thread::sleep_for(std::chrono::duration<double>(render_dt));
        }
    }

    sim.shutdown();
    log_file.close();
    std::cout << "Simulation Completed.\n";
    return 0;
}

auto main(int argc, char *argv[]) -> int {
    bool geometric = true;
    int trajectory_mode = 2;
    bool show_marker = true;
    bool log_data = false;

    // --- COMMAND LINE ARGUMENT PARSING ---

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            std::cout
                << "Usage: " << argv[0] << " [options]\n"
                << "Options:\n"
                << "  --geometric <false|true> -> Toggle Geometric Controller (default: true)\n"
                << "  --trajectory <0|1|2> -> Select trajectory mode (default: 2)\n"
                << "  --marker <false|true> -> Show target marker in simulation (default: true)\n"
                << "  --log <false|true> -> Log data to CSV file (default: false)\n";
            return 0;
        }

        if (arg == "--geometric" && i + 1 < argc) {
            geometric = (std::string(argv[++i]) == "1" || std::string(argv[i]) == "true");
        } else if (arg == "--trajectory" && i + 1 < argc) {
            try {
                trajectory_mode = std::stoi(argv[++i]);
            } catch (...) {}
        } else if (arg == "--marker" && i + 1 < argc) {
            show_marker = (std::string(argv[++i]) == "1" || std::string(argv[i]) == "true");
        } else if (arg == "--log" && i + 1 < argc) {
            log_data = (std::string(argv[++i]) == "1" || std::string(argv[i]) == "true");
        }
    }

    // --- RUNTIME TO COMPILE-TIME DISPATCH ---

    double duration = 15.0;

    if (trajectory_mode == 0) {
        xarm_geo::FigureEightTrajectory traj;
        return run_simulation(traj, duration, geometric, trajectory_mode, show_marker, log_data);
    } else if (trajectory_mode == 1) {
        xarm_geo::WingInspectionTrajectory traj;
        return run_simulation(traj, duration, geometric, trajectory_mode, show_marker, log_data);
    } else {
        xarm_geo::TiltingCircleTrajectory traj;
        return run_simulation(traj, duration, geometric, trajectory_mode, show_marker, log_data);
    }
}
