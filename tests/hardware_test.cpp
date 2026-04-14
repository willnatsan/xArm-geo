#include <chrono>
#include <cmath>
#include <iostream>
#include <numbers>
#include <thread>

#include <xarm_geo/core/system.h>
#include <xarm_geo/interfaces/hardware.h>
#include <xarm_geo/utils/model_builder.h>

auto main(int argc, char *argv[]) -> int {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <robot_ip>\n";
        return 1;
    }

    std::string robot_ip = argv[1];
    xarm_geo::Model model = xarm_geo::build_model(6, "XI130412C23L45");
    xarm_geo::Data data(model);

    std::cout << "--- Starting Safe Hardware Evaluation ---\n";

    try {
        // 1. Initialize Hardware
        std::cout << "Connecting to xArm at " << robot_ip << "...\n";
        xarm_geo::Hardware hw(model.dof, robot_ip);

        if (!hw.is_running()) {
            std::cerr << "[ERROR] Hardware Connected BUT Reported an Error State.\n";
            return 1;
        }
        std::cout << "Connection Successful!\n";

        xarm_geo::JointState state(model.dof);
        xarm_geo::JointVelocity cmd(model.dof);

        // Passive Read Test (2 Seconds)
        std::cout << "\nPassive Read Test (2 seconds)...\n";
        for (int i = 0; i < 20; ++i) {  // 10 Hz loop
            if (hw.read(state) == xarm_geo::InterfaceStatus::OK) {
                std::cout << "Read OK | Joint 0 Pos: " << state.q[0]
                          << " rad | Joint 0 Vel: " << state.v[0] << " rad/s\n";
            } else {
                std::cerr << "[ERROR] Failed to read Joint States!\n";
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        // Zero-Velocity Hold (2 Seconds)
        // Tests the write functionality without actually moving the robot.
        std::cout << "\nZero-Velocity Hold Test (2 seconds)...\n";
        for (int i = 0; i < 20; ++i) {
            for (int j = 0; j < model.dof; ++j) { cmd.v[j] = 0.0; }  // Ensure strict zeroes

            if (hw.write(cmd) != xarm_geo::InterfaceStatus::OK) {
                std::cerr << "[ERROR] Failed to write Zero Velocity Command!\n";
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        // Gentle Actuation Test (4 Seconds)
        // Applies a very slow, tiny sine wave to the base joint (Joint 0) only.
        std::cout << "\nGentle Actuation Test...\n";
        std::cout << "[WARNING] Robot will now move: KEEP HAND ON E-STOP !!!\n";
        std::this_thread::sleep_for(std::chrono::seconds(2));  // Buffer time for the operator

        auto start_time = std::chrono::steady_clock::now();
        double loop_dt = 0.01;  // 100 Hz control loop

        while (true) {
            auto now = std::chrono::steady_clock::now();
            double t = std::chrono::duration<double>(now - start_time).count();

            if (t > 4.0) break;  // Run for 4 seconds

            // Generate a gentle sine wave on Joint 0.
            // Amplitude: 0.05 rad/s. Period: 4 seconds.
            for (int j = 0; j < model.dof; ++j) { cmd.v[j] = 0.0; }
            cmd.v[0] = 0.05 * std::sin(2.0 * std::numbers::pi * t / 4.0);

            if (hw.write(cmd) != xarm_geo::InterfaceStatus::OK) {
                std::cerr << "[ERROR] Failed to write during Actuation!\n";
                break;
            }

            std::this_thread::sleep_for(std::chrono::duration<double>(loop_dt));
        }

        // Final Safety Shutdown
        std::cout << "\nStopping Robot...\n";
        for (int j = 0; j < model.dof; ++j) { cmd.v[j] = 0.0; }
        hw.write(cmd);

        hw.shutdown();
        std::cout << "Hardware Evaluation Complete. Shutting Down...\n";

    } catch (const std::exception &e) {
        std::cerr << "\n[CRITICAL ERROR] Exception Caught: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
