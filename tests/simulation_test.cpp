#include <chrono>
#include <format>
#include <fstream>
#include <iostream>
#include <numbers>
#include <thread>

#include <xarm_geo/core/system.h>
#include <xarm_geo/interfaces/simulation.h>
#include <xarm_geo/modelling/kinematics.h>
#include <xarm_geo/utils/model_builder.h>

struct TrajectoryState {
    xarm_geo::manifold::SE3 pose;
    Eigen::Vector3d pos;
    double roll;
    double pitch;
    double yaw;
};

auto generate_trajectory(int trajectory_mode, double t) -> TrajectoryState {
    double x_c = 0.35;
    double y_c = 0.0;
    double z_c = 0.30;

    double roll_target = 0.0;
    double pitch_target = std::numbers::pi;  // BASELINE: Pointing straight down
    double yaw_target = 0.0;
    Eigen::Vector3d target_pos;

    if (trajectory_mode == 0) {
        double omega = 1.0;
        double size_x = 0.15;
        double size_y = 0.10;

        double x_fig8 = size_x * std::sin(omega * t);
        double y_fig8 = size_y * std::sin(2.0 * omega * t);
        double z_bob = 0.03 * std::sin(omega * t);

        target_pos << x_c + x_fig8, y_c + y_fig8, z_c + z_bob;
        roll_target = 0.5 * std::sin(2.0 * omega * t);
        pitch_target = std::numbers::pi + 0.3 * std::cos(omega * t);
        yaw_target = 0.8 * std::sin(omega * t);

    } else if (trajectory_mode == 1) {
        double omega_sweep = 0.5;
        double omega_scan = 2.0;
        double sweep_amp = 0.20;
        double scan_amp = 0.05;
        double curvature = 2.5;

        double y_wing = sweep_amp * std::sin(omega_sweep * t);
        double x_wing = scan_amp * std::cos(omega_scan * t);
        double z_wing = -curvature * (y_wing * y_wing);

        target_pos << x_c + x_wing, y_c + y_wing, z_c + z_wing;
        roll_target = std::atan(2.0 * curvature * y_wing);
        pitch_target = std::numbers::pi;
        yaw_target = 0.0;

    } else if (trajectory_mode == 2) {
        double omega = 2.0;
        double R = 0.12;
        double transition_start = 8.0;
        double transition_duration = 1;

        double tilt = 0.0;
        if (t > transition_start) {
            double progress = (t - transition_start) / transition_duration;
            tilt = std::min(progress, 1.0) * (std::numbers::pi / 2.0);
        }

        Eigen::Vector3d p_base(R * std::cos(omega * t), R * std::sin(omega * t), 0.0);
        Eigen::Matrix3d R_tilt =
            Eigen::AngleAxisd(tilt, Eigen::Vector3d::UnitY()).toRotationMatrix();
        target_pos = Eigen::Vector3d(x_c, y_c, z_c) + R_tilt * p_base;

        roll_target = 0.0;
        pitch_target = std::numbers::pi - tilt;
        yaw_target = 0.0;
    }

    // Construct target pose
    Eigen::AngleAxisd rollAngle(roll_target, Eigen::Vector3d::UnitX());
    Eigen::AngleAxisd pitchAngle(pitch_target, Eigen::Vector3d::UnitY());
    Eigen::AngleAxisd yawAngle(yaw_target, Eigen::Vector3d::UnitZ());

    Eigen::Quaterniond target_quat = yawAngle * pitchAngle * rollAngle;
    xarm_geo::manifold::SE3 target_pose(xarm_geo::manifold::SO3(target_quat), target_pos);

    // Extract Target Euler Angles to Prevent Alias Fighting
    Eigen::Matrix3d R_target = target_quat.toRotationMatrix();
    double target_pitch_ext = std::asin(std::clamp(-R_target(2, 0), -1.0, 1.0));
    double target_yaw_ext;
    double target_roll_ext;

    if (std::abs(std::cos(target_pitch_ext)) > 1e-6) {
        target_yaw_ext = std::atan2(R_target(1, 0), R_target(0, 0));
        target_roll_ext = std::atan2(R_target(2, 1), R_target(2, 2));
    } else {
        target_yaw_ext = 0.0;
        target_roll_ext = std::atan2(-R_target(0, 1), R_target(1, 1));
    }

    return {target_pose, target_pos, target_roll_ext, target_pitch_ext, target_yaw_ext};
}

auto main(int argc, char *argv[]) -> int {
    // --- DEFAULT PARAMETERS ---
    bool use_geometric_controller = true;
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
                << "                       0 = Flat 2D Circle\n"
                << "                       1 = Curved Wing Inspection\n"
                << "                       2 = Horizontal to Vertical Circle\n"
                << "  --marker <false|true> -> Show target marker in simulation (default: true)\n"
                << "  --log <false|true> -> Log data to CSV file (default: false)\n";
            return 0;
        }

        if (arg == "--geometric") {
            if (i + 1 < argc) {
                std::string val = argv[++i];
                use_geometric_controller = (val == "1" || val == "true");
            }
        } else if (arg == "--trajectory") {
            if (i + 1 < argc) {
                try {
                    trajectory_mode = std::stoi(argv[++i]);
                } catch (...) {
                    std::cerr << "Invalid trajectory mode argument. Using default: "
                              << trajectory_mode << "\n";
                }
            }
        } else if (arg == "--marker") {
            if (i + 1 < argc) {
                std::string val = argv[++i];
                show_marker = (val == "1" || val == "true");
            }
        } else if (arg == "--log") {
            if (i + 1 < argc) {
                std::string val = argv[++i];
                log_data = (val == "1" || val == "true");
            }
        }
    }

    xarm_geo::Model model = xarm_geo::build_model(6, "XI130412C23L45");
    xarm_geo::Data data(model);
    xarm_geo::Simulation sim(model.mjcf_file);

    xarm_geo::JointPosition pos_curr(model.dof);
    xarm_geo::JointVelocity vel_target(model.dof);

    double t = 0.0;
    double physics_dt = 0.002;
    double render_dt = 1.0 / 60.0;
    double last_render_t = 0.0;

    // --- SET INITIAL CONFIGURATION (TRAJECTORY START) ---
    TrajectoryState initial = generate_trajectory(trajectory_mode, t);

    if (sim.read(pos_curr) != xarm_geo::InterfaceStatus::OK) {
        std::cerr << "Error: Failed to read Initial Joint Positions.\n";
        return 1;
    }
    bool ik_success = xarm_geo::inverse_kinematics(model, data, pos_curr.q, initial.pose);
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

    while (t < 20.0 && sim.is_running()) {
        if (sim.read(pos_curr) != xarm_geo::InterfaceStatus::OK) {
            std::cerr << "Error: Failed to read Current Joint Positions. Halting Loop.\n";
            break;
        }
        xarm_geo::compute_jacobians(model, data, pos_curr.q);

        // Generate trajectory for current time t
        TrajectoryState target = generate_trajectory(trajectory_mode, t);

        xarm_geo::manifold::SE3::Twist cmd_twist;
        double kp = 10;  // Tracking gain

        if (use_geometric_controller) {
            // GEOMETRIC APPROACH [SE(3)]:
            xarm_geo::manifold::SE3 pose_err_body = data.ee_pose.inverse() * target.pose;
            xarm_geo::manifold::SE3::Twist twist_err_body = pose_err_body.log();

            cmd_twist.head<3>() = kp * twist_err_body.head<3>();
            cmd_twist.tail<3>() = kp * twist_err_body.tail<3>();

        } else {
            // NON-GEOMETRIC APPROACH:
            Eigen::Vector3d pos_err_space = target.pos - data.ee_pose.r3();

            // ROBUST EULER EXTRACTION (ZYX / Yaw-Pitch-Roll)
            Eigen::Matrix3d R_curr = data.ee_pose.so3().matrix();

            double curr_pitch = std::asin(std::clamp(-R_curr(2, 0), -1.0, 1.0));
            double curr_yaw;
            double curr_roll;

            // Check for gimbal lock
            if (std::abs(std::cos(curr_pitch)) > 1e-6) {
                curr_yaw = std::atan2(R_curr(1, 0), R_curr(0, 0));
                curr_roll = std::atan2(R_curr(2, 1), R_curr(2, 2));
            } else {
                curr_yaw = 0.0;
                curr_roll = std::atan2(-R_curr(0, 1), R_curr(1, 1));
            }

            Eigen::Vector3d curr_euler(curr_yaw, curr_pitch, curr_roll);
            Eigen::Vector3d target_euler(target.yaw, target.pitch, target.roll);

            // ERROR AND CONTINUOUS WRAPPING
            Eigen::Vector3d rot_err_space = target_euler - curr_euler;
            for (int i = 0; i < 3; ++i) {
                while (rot_err_space[i] > std::numbers::pi)
                    rot_err_space[i] -= 2.0 * std::numbers::pi;
                while (rot_err_space[i] < -std::numbers::pi)
                    rot_err_space[i] += 2.0 * std::numbers::pi;
            }

            // MAP EULER RATES TO SPACE ANGULAR VELOCITY
            double d_yaw = kp * rot_err_space[0];
            double d_pitch = kp * rot_err_space[1];
            double d_roll = kp * rot_err_space[2];

            // Map the rates through the ZYX rotation axes to get omega in the space frame
            Eigen::Vector3d omega_space =
                Eigen::Vector3d::UnitZ() * d_yaw +
                Eigen::AngleAxisd(curr_yaw, Eigen::Vector3d::UnitZ()) *
                    (Eigen::Vector3d::UnitY() * d_pitch) +
                Eigen::AngleAxisd(curr_yaw, Eigen::Vector3d::UnitZ()) *
                    Eigen::AngleAxisd(curr_pitch, Eigen::Vector3d::UnitY()) *
                    (Eigen::Vector3d::UnitX() * d_roll);

            // TRANSFORM TO BODY FRAME TWIST
            // The IK solver expects a body twist, so we rotate space vectors back to the body
            Eigen::Matrix3d Rt = R_curr.transpose();

            cmd_twist.head<3>() = Rt * (kp * pos_err_space);  // Body linear velocity
            cmd_twist.tail<3>() = Rt * omega_space;           // Body angular velocity
        }

        // --- WRITE ---
        xarm_geo::inverse_diff_kinematics(model, data, cmd_twist);
        vel_target.v = data.v_out;
        if (sim.write(vel_target) != xarm_geo::InterfaceStatus::OK) {
            std::cerr << "Error: Failed to send Joint Velocity Commands. Halting Loop.\n";
            break;
        }

        // --- LOG DATA TO CSV ---
        if (log_data) {
            Eigen::Vector3d actual_euler =
                data.ee_pose.so3().quat().toRotationMatrix().eulerAngles(2, 1, 0);

            log_file << t << "," << target.pos.x() << "," << target.pos.y() << "," << target.pos.z()
                     << "," << data.ee_pose.r3().x() << "," << data.ee_pose.r3().y() << ","
                     << data.ee_pose.r3().z() << "," << target.roll << "," << target.pitch << ","
                     << target.yaw << "," << actual_euler[2] << "," << actual_euler[1] << ","
                     << actual_euler[0] << "\n";
        }

        // --- STEP PHYSICS & RENDER ---
        sim.step();

        t += physics_dt;

        if (t - last_render_t >= render_dt) {
            if (show_marker) { sim.set_marker(target.pose); }
            sim.update_scene();
            sim.render();
            last_render_t = t;
            std::this_thread::sleep_for(std::chrono::duration<double>(render_dt));
        }
    }

    sim.shutdown();
    log_file.close();  // Close the CSV
    std::cout << "Simulation Completed.\n";
    return 0;
}
