#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <numbers>
#include <numeric>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <vector>

// Note: Safety filter and controller updates are rate-limited relative to the
// physics step to reduce per-tick QP solve cost. The nominal physics rate
// is 500 Hz (physics_dt = 0.002 s); with a decimation factor of 4 the
// safety-aware controller runs at 125 Hz while MuJoCo continues at 500 Hz,
// retaining the last written command between updates.
//
// Note: Hardware Control Rate
// The xArm controller silently drops commands sent faster than 250 Hz on both
// vc_set_joint_velocity (mode 4) and set_servo_angle_j (mode 1). The UFactory
// User Manual (v1.6.1) recommends a command rate of 30-250 Hz, preferably
// 100-200 Hz, for stable motion. The effective controller rate at
// kSafetyDecimationFactor = 4 is 125 Hz, which sits cleanly inside the
// preferred 100-200 Hz band.
static constexpr int kSafetyDecimationFactor = 4;

// Physics step size used by MuJoCo and the outer simulation loop.  The
// physics rate is intrinsic to simulator fidelity (constraint-solver
// stability, contact dynamics) and is independent of the controller rate.
static constexpr double kSimulationPhysicsPeriodS = 0.002;  // 500 Hz

// This is the simulation-side analogue of kHardwareControlPeriodS in
// hardware_test.cpp.
//
// Rate reference table:
//   decim 1 -> 500 Hz  [EXCEEDS SDK ceiling; half commands dropped on hardware]
//   decim 2 -> 250 Hz  [At SDK ceiling; no headroom]
//   decim 4 -> 125 Hz  [Preferred range; recommended for hardware]
//   decim 5 -> 100 Hz  [Lower preferred bound; still fine]
//   decim 8 ->  62 Hz  [Above 30 Hz floor; motion may begin to look choppy]
//   decim 16->  31 Hz  [At discontinuity threshold; not recommended]
static constexpr double kSimulationControlPeriodS =
    kSafetyDecimationFactor * kSimulationPhysicsPeriodS;  // 0.008 s = 125 Hz

// --- Per-Tick Controller Timing ---
//
// Lightweight, non-intrusive timing helper for the Phase 2 controller loop.
// record() is O(1) with pre-allocated storage; report() sorts a copy at
// end-of-trial, so no allocations occur inside the hot loop.
//
// All durations are in microseconds (µs).

namespace {

    struct PerfStats {
        std::vector<double> samples_us;

        // Pre-allocate storage to avoid heap allocation inside the control loop.
        void reserve(std::size_t n) { samples_us.reserve(n); }

        void record(std::chrono::steady_clock::time_point t0,
                    std::chrono::steady_clock::time_point t1) {
            samples_us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
        }

        void report(std::string_view label) const {
            if (samples_us.empty()) {
                std::cout << label << ": no samples recorded.\n";
                return;
            }

            std::vector<double> sorted = samples_us;
            std::sort(sorted.begin(), sorted.end());

            const std::size_t n = sorted.size();
            const double mean =
                std::accumulate(sorted.begin(), sorted.end(), 0.0) / static_cast<double>(n);

            auto percentile = [&](double p) -> double {
                // Nearest-rank method.
                const std::size_t idx =
                    std::max(std::size_t{0},
                             static_cast<std::size_t>(std::ceil(p * static_cast<double>(n))) - 1);
                return sorted[std::min(idx, n - 1)];
            };

            std::cout << "\n"
                      << label << " (n=" << n << " samples):\n"
                      << "  p50:  " << percentile(0.50) << " us\n"
                      << "  p95:  " << percentile(0.95) << " us\n"
                      << "  p99:  " << percentile(0.99) << " us\n"
                      << "  max:  " << sorted.back() << " us\n"
                      << "  mean: " << mean << " us\n"
                      << "  [budget: " << (kSafetyDecimationFactor * 2000.0)
                      << " us per controller tick  |  2000 us per physics step]\n";
        }
    };

}  // namespace

#include <xarm_geo/core/system.h>
#include <xarm_geo/diagnostics/logger.h>
#include <xarm_geo/examples/controllers/euclidean_p_controller.h>
#include <xarm_geo/examples/controllers/euclidean_pd_controller.h>
#include <xarm_geo/examples/controllers/geometric_p_controller.h>
#include <xarm_geo/examples/controllers/geometric_pd_controller.h>
#include <xarm_geo/examples/controllers/joint_p_controller.h>
#include <xarm_geo/examples/trajectories/figure_eight.h>
#include <xarm_geo/examples/trajectories/inner_cavity_scan.h>
#include <xarm_geo/examples/trajectories/joint_ptp.h>
#include <xarm_geo/examples/trajectories/pipe_inspection.h>
#include <xarm_geo/examples/trajectories/tilting_circle.h>
#include <xarm_geo/examples/trajectories/wing_inspection.h>
#include <xarm_geo/interfaces/simulation.h>
#include <xarm_geo/modelling/collision.h>
#include <xarm_geo/modelling/kinematics.h>
#include <xarm_geo/trajectory/validate.h>
#include <xarm_geo/utils/model_builder.h>

struct TestParams {
    bool geometric = true;
    int trajectory_mode = 2;
    bool show_marker = true;
    bool log_data = false;
    bool check_trajectory = false;
    bool torque_mode = false;
    bool feedforward = true;
    bool constraint_aware = true;
};

auto run_joint_ptp(xarm_geo::Simulation &sim, const xarm_geo::Model &model, xarm_geo::Data &data,
                   xarm_geo::controllers::JointPController &joint_controller,
                   const xarm_geo::trajectories::JointPTP &traj, xarm_geo::JointState &state,
                   xarm_geo::JointVelocity &control_target) -> bool {

    const double duration = traj.duration();
    double t = 0.0;
    const double physics_dt = kSimulationPhysicsPeriodS;
    double render_dt = 1.0 / 60.0;
    double last_render_t = 0.0;

    // ctx.dt must reflect the actual period between controller invocations
    // (kSimulationControlPeriodS), not the physics step (kSimulationPhysicsPeriodS).
    const auto dt_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::duration<double>(kSimulationControlPeriodS));

    xarm_geo::JointTarget joint_target(traj.dof());

    std::int64_t tick = 0;
    while (t < duration && sim.is_running()) {
        // Capture the step deadline at the top of each iteration so the
        // sleep_until at the bottom paces the simulation to real-time.
        // This mirrors a hardware control loop where the robot (physics) always
        // runs at wall-clock rate; render is a best-effort overlay inside the
        // same budget and does not gate the physics rate.
        const auto step_start = std::chrono::steady_clock::now();

        if (sim.read(state) != xarm_geo::InterfaceStatus::OK) { return false; }

        // Controller and write are decimated; MuJoCo retains the last ctrl
        // command between updates so the actuator output is held constant.
        if (tick % kSafetyDecimationFactor == 0) {
            if (traj.evaluate(t, joint_target) != xarm_geo::TrajectoryStatus::OK) { return false; }

            const xarm_geo::JointControllerContext ctx{state, joint_target, dt_ns};
            if (joint_controller.update(model, data, ctx, control_target) !=
                xarm_geo::ControllerStatus::OK) {
                std::cerr << "JointPController update failed.\n";
                return false;
            }

            if (sim.write(control_target) != xarm_geo::InterfaceStatus::OK) { return false; }
        }

        sim.step();
        t += physics_dt;
        ++tick;

        if (t - last_render_t >= render_dt) {
            sim.update_scene();
            sim.render();
            last_render_t = t;
            // No sleep here: render is a best-effort overlay; the per-step
            // sleep_until below provides the real-time pacing.
        }

        // Pace the loop to real-time: sleep until one physics_dt has elapsed
        // since the start of this iteration. If the work took longer than
        // physics_dt (controller overrun), sleep_until returns immediately and
        // the simulation falls behind real-time -- the same honest behaviour
        // as a hardware control loop missing a deadline.
        std::this_thread::sleep_until(step_start + std::chrono::duration<double>(physics_dt));
    }

    control_target.v.setZero();
    sim.write(control_target);
    sim.step();

    return true;
}

template <xarm_geo::TaskTrajectory T>
auto run_simulation(xarm_geo::Model &model, xarm_geo::Data &data,
                    xarm_geo::CollisionModel &col_model, xarm_geo::CollisionData &col_data,
                    xarm_geo::Simulation &sim, const T &trajectory, const Eigen::VectorXd &q_home,
                    const TestParams &params, std::string_view traj_name) -> int {

    const double duration = trajectory.duration();

    xarm_geo::JointState state(model.dof);
    xarm_geo::JointVelocity control_target(model.dof);

    if (sim.read(state) != xarm_geo::InterfaceStatus::OK) { return 1; }

    xarm_geo::TaskTarget task_target;
    if (trajectory.evaluate(0.0, task_target) != xarm_geo::TrajectoryStatus::OK) { return 1; }

    // --- Layer 1: Collision-Aware IK for Start Pose ---

    bool ik_success = xarm_geo::optimal_inverse_kinematics(model, data, col_model, col_data, q_home,
                                                           task_target.pose);
    if (!ik_success) {
        std::cerr << "IK FAILED for Trajectory Start Pose (No Collision-Free Solution)\n";
        return 1;
    }
    Eigen::VectorXd q_start = data.q_out;

    // --- Controller Setup ---
    //
    // Geometric and Euclidean variants share identical gains and flags so
    // execution differences attribute to the control law, not tuning.

    // Kinematic pair (JointVelocity output).
    xarm_geo::controllers::GeometricPController p_controller(model);
    p_controller.gains.kp_pos.setConstant(8.0);
    p_controller.gains.kp_rot.setConstant(8.0);
    p_controller.use_feedforward = params.feedforward;
    p_controller.constraint_aware = params.constraint_aware;
    if (params.constraint_aware) { p_controller.attach_collision(col_model, col_data); }

    // --- Layer 2: Pre-Flight Trajectory Validation ---

    if (params.check_trajectory) {
        std::cout << "\n[PRE-FLIGHT] Validating Trajectory for Collision-Free Feasibility...\n";

        auto validation = xarm_geo::validate_trajectory(model, data, col_model, col_data,
                                                        trajectory, q_start, p_controller);
        if (validation.status != xarm_geo::ValidationStatus::OK) {
            std::cerr << "Trajectory Validation FAILED at t=" << validation.failure_time << ": "
                      << validation.reason << "\n";
            return 1;
        }

        std::cout << "[PRE-FLIGHT] Trajectory Validated!\n";
    }

    // --- Joint-Space PTP Controller (Phases 1 and 3) ---

    xarm_geo::controllers::JointPController joint_controller(model);
    joint_controller.kp.setConstant(5.0);
    joint_controller.use_feedforward = true;

    // --- Phase 1: Home To Start ---

    double start_duration = 3.0;

    std::cout << "\n[PHASE 1] Moving from Home to Trajectory Start...\n";
    xarm_geo::trajectories::JointPTP approach_traj(q_home, q_start, start_duration);

    if (!run_joint_ptp(sim, model, data, joint_controller, approach_traj, state, control_target)) {
        std::cerr << "Failed during Phase 1 Approach.\n";
        return 1;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    // Switch to torque mode only after Phase 1 has completed. Phase 1 uses
    // JointPController (JointVelocity output), which requires VELOCITY mode.
    // Phase 3 restores VELOCITY before its joint-PTP return move.
    if (params.torque_mode) { sim.set_control_mode(xarm_geo::ControlMode::TORQUE); }

    // --- Phase 2: Main Trajectory Execution ---

    std::cout << "\n[PHASE 2] Executing Task Space Trajectory...\n";

    std::cout << "Starting Simulation.\n"
              << "  Torque Mode: " << (params.torque_mode ? "ON" : "OFF") << "\n"
              << "  Geometric Mode: " << (params.geometric ? "ON" : "OFF") << "\n"
              << "  Feedforward: " << (params.feedforward ? "ON" : "OFF") << "\n"
              << "  Constraint Aware: " << (params.constraint_aware ? "ON" : "OFF") << "\n"
              << "  Trajectory: " << params.trajectory_mode << "\n"
              << "  Marker: " << (params.show_marker ? "ON" : "OFF") << "\n"
              << "  Data Logging: " << (params.log_data ? "ON" : "OFF") << "\n";

    double t = 0.0;
    const double physics_dt = kSimulationPhysicsPeriodS;
    double render_dt = 1.0 / 60.0;
    double last_render_t = 0.0;

    xarm_geo::controllers::EuclideanPController p_baseline(model);
    p_baseline.gains.kp_pos.setConstant(8.0);
    p_baseline.gains.kp_rot.setConstant(8.0);
    p_baseline.use_feedforward = params.feedforward;
    p_baseline.constraint_aware = params.constraint_aware;
    if (params.constraint_aware) { p_baseline.attach_collision(col_model, col_data); }

    // Dynamic pair (JointTorque output). K_D is critically-damped relative to K_P.
    xarm_geo::controllers::GeometricPDController pd_controller(model);
    pd_controller.gains.kp_pos.setConstant(100.0);
    pd_controller.gains.kp_rot.setConstant(50.0);
    pd_controller.gains.kd_lin.setConstant(20.0);
    pd_controller.gains.kd_ang.setConstant(10.0);
    pd_controller.use_feedforward = params.feedforward;
    pd_controller.constraint_aware = params.constraint_aware;
    if (params.constraint_aware) { pd_controller.attach_collision(col_model, col_data); }

    xarm_geo::controllers::EuclideanPDController pd_baseline(model);
    pd_baseline.gains.kp_pos.setConstant(100.0);
    pd_baseline.gains.kp_rot.setConstant(50.0);
    pd_baseline.gains.kd_lin.setConstant(20.0);
    pd_baseline.gains.kd_ang.setConstant(10.0);
    pd_baseline.use_feedforward = params.feedforward;
    pd_baseline.constraint_aware = params.constraint_aware;
    if (params.constraint_aware) { pd_baseline.attach_collision(col_model, col_data); }

    xarm_geo::JointTorque torque_target(model.dof);
    // ctx.dt must reflect the actual controller invocation period
    // (kSimulationControlPeriodS = 8 ms at decim 4), not the physics step
    // (2 ms).  GeometricPIController and GeometricPIDController consume
    // ctx.dt for integrator accumulation; using physics_dt here would
    // integrate at 1/kSafetyDecimationFactor of the intended rate.
    const auto dt_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::duration<double>(kSimulationControlPeriodS));

    // Determine the active controller name for the trial filename.
    const std::string_view controller_name =
        params.torque_mode
            ? (params.geometric ? xarm_geo::controllers::GeometricPDController::kName
                                : xarm_geo::controllers::EuclideanPDController::kName)
            : (params.geometric ? xarm_geo::controllers::GeometricPController::kName
                                : xarm_geo::controllers::EuclideanPController::kName);

    // DataLogger is constructed (and pre-allocated) before the loop;
    // the destructor flushes to disk automatically after Phase 2 ends.
    std::optional<xarm_geo::diagnostics::DataLogger> logger;
    if (params.log_data) {
        const std::string trial_name = xarm_geo::diagnostics::make_trial_name(
            "sim", controller_name, traj_name, params.constraint_aware, params.feedforward);

        logger.emplace(model, xarm_geo::diagnostics::DataLogger::Config{
                                  .output_path = "tests/results/" + trial_name + ".csv",
                                  .trial_name = trial_name,
                              });
    }

    // Pre-allocate the timing buffer for the expected number of controller
    // ticks (duration / (kSafetyDecimationFactor * physics_dt)) plus headroom,
    // so push_back inside the hot loop never allocates.
    PerfStats perf_phase2;
    {
        const std::size_t expected_ctrl_ticks =
            static_cast<std::size_t>(duration / (kSafetyDecimationFactor * physics_dt)) + 32;
        perf_phase2.reserve(expected_ctrl_ticks);
    }

    std::int64_t tick = 0;
    while (t < duration && sim.is_running()) {
        // Capture the step deadline at the top of each iteration so the
        // sleep_until at the bottom paces the simulation to real-time.
        const auto step_start = std::chrono::steady_clock::now();

        if (sim.read(state) != xarm_geo::InterfaceStatus::OK) { break; }

        // Controller, write, and logging are decimated to reduce per-tick QP
        // cost. MuJoCo retains the last ctrl command between updates.
        if (tick % kSafetyDecimationFactor == 0) {
            if (trajectory.evaluate(t, task_target) != xarm_geo::TrajectoryStatus::OK) { break; }

            const xarm_geo::TaskControllerContext ctx{state, task_target, dt_ns};

            xarm_geo::ControllerStatus ctrl_status = xarm_geo::ControllerStatus::OK;

            const auto ctrl_t0 = std::chrono::steady_clock::now();

            if (params.torque_mode) {
                // Dynamic Mode: (Geometric|Euclidean)PDController -> JointTorque.
                ctrl_status = params.geometric
                                  ? pd_controller.update(model, data, ctx, torque_target)
                                  : pd_baseline.update(model, data, ctx, torque_target);

                if (ctrl_status != xarm_geo::ControllerStatus::OK) {
                    std::cerr << (params.geometric ? "GeometricPDController"
                                                   : "EuclideanPDController")
                              << " update failed.\n";
                    break;
                }
                if (sim.write(torque_target) != xarm_geo::InterfaceStatus::OK) { break; }
            } else {
                // Kinematic Mode: (Geometric|Euclidean)PController -> JointVelocity.
                ctrl_status = params.geometric
                                  ? p_controller.update(model, data, ctx, control_target)
                                  : p_baseline.update(model, data, ctx, control_target);

                if (ctrl_status != xarm_geo::ControllerStatus::OK) {
                    std::cerr << (params.geometric ? "GeometricPController"
                                                   : "EuclideanPController")
                              << " update failed.\n";
                    break;
                }
                if (sim.write(control_target) != xarm_geo::InterfaceStatus::OK) { break; }
            }

            const auto ctrl_t1 = std::chrono::steady_clock::now();
            perf_phase2.record(ctrl_t0, ctrl_t1);

            if (logger) {
                xarm_geo::diagnostics::LogSample s;
                xarm_geo::diagnostics::fill_task_sample(s, t, tick, state, task_target, data);
                s.controller_status = static_cast<std::uint8_t>(ctrl_status);

                if (params.torque_mode) {
                    if (params.geometric) {
                        xarm_geo::diagnostics::fill_torque_diagnostics(s, pd_controller);
                    } else {
                        xarm_geo::diagnostics::fill_torque_diagnostics(s, pd_baseline);
                    }
                } else {
                    if (params.geometric) {
                        xarm_geo::diagnostics::fill_velocity_diagnostics(s, p_controller);
                    } else {
                        xarm_geo::diagnostics::fill_velocity_diagnostics(s, p_baseline);
                    }
                }

                logger->log(s);
            }
        }

        sim.step();
        t += physics_dt;
        ++tick;

        if (t - last_render_t >= render_dt) {
            if (params.show_marker) { sim.set_marker(task_target.pose); }
            sim.update_scene();
            sim.render();
            last_render_t = t;
            // No sleep here: render is best-effort; the per-step sleep_until
            // below provides real-time pacing.
        }

        // Pace to real-time: sleep until one physics_dt has elapsed since the
        // start of this iteration. Overruns return immediately, faithfully
        // signalling missed deadlines as a hardware loop would.
        std::this_thread::sleep_until(step_start + std::chrono::duration<double>(physics_dt));
    }

    // logger goes out of scope here and flushes to disk before Phase 3 begins.

    // Print per-tick controller update timing for the dissertation / hardware-
    // readiness assessment. Sorting and printing happen out of the hot loop.
    const std::string_view mode_tag =
        params.torque_mode ? (params.constraint_aware ? "[PHASE 2] Torque, constraint_aware"
                                                      : "[PHASE 2] Torque, unconstrained")
                           : (params.constraint_aware ? "[PHASE 2] Kinematic, constraint_aware"
                                                      : "[PHASE 2] Kinematic, unconstrained");
    perf_phase2.report(mode_tag);

    // --- Phase 3: End To Home ---

    std::cout << "\n[PHASE 3] Task Complete. Returning to Home...\n";
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    // Phase 3 always uses velocity-mode joint-PTP; switch back if Phase 2 was torque.
    if (params.torque_mode) { sim.set_control_mode(xarm_geo::ControlMode::VELOCITY); }

    double end_duration = 3.0;

    if (sim.read(state) != xarm_geo::InterfaceStatus::OK) {
        std::cerr << "Failed to Read Simulation State for Phase 3 Return.\n";
        return 1;
    }

    xarm_geo::trajectories::JointPTP return_traj(state.q, q_home, end_duration);
    if (!run_joint_ptp(sim, model, data, joint_controller, return_traj, state, control_target)) {
        std::cerr << "Failed during Phase 3 Return.\n";
        return 1;
    }

    sim.shutdown();
    std::cout << "Simulation Sequence Completed Safely.\n";

    return 0;
}

auto main(int argc, char *argv[]) -> int {

    TestParams params;

    // --- Command Line Argument Parsing ---

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

    // --- Setup ---

    xarm_geo::Model model = xarm_geo::build_model(6, "XI130412C23L45");

    // Torque mode bakes gravity into RNEA so h(q, v) compensates for it.
    // Velocity mode delegates compensation to xArm SDK / MuJoCo actuators.
    if (params.torque_mode) { model.gravity = Eigen::Vector3d{0.0, 0.0, -9.81}; }

    xarm_geo::Data data(model);
    xarm_geo::Simulation sim(model.mjcf_file);

    xarm_geo::CollisionModel col_model = xarm_geo::build_collision_model(model, true);
    xarm_geo::CollisionData col_data(col_model);

    Eigen::VectorXd q_home = Eigen::VectorXd::Zero(model.dof);
    q_home[0] = 1.5 * std::numbers::pi;

    sim.set_joint_positions(q_home);

    xarm_geo::JointState state(model.dof);
    if (sim.read(state) != xarm_geo::InterfaceStatus::OK) { return 1; }

    data.q = state.q;
    xarm_geo::compute_jacobians(model, data);

    // Trajectory anchor pose: centre offset from base, rotated about Z.
    double q0 = q_home[0];
    Eigen::Vector3d center(0.35 * std::cos(q0), 0.35 * std::sin(q0), 0.35);
    xarm_geo::manifold::SO3 rot =
        xarm_geo::manifold::SO3::exp(Eigen::Vector3d::UnitZ() * (q0 - (1.5 * std::numbers::pi)));

    xarm_geo::manifold::SE3 anchor_pose(rot, center);

    constexpr double traj_duration = 15.0;

    if (params.trajectory_mode == 0) {
        xarm_geo::trajectories::FigureEight traj(anchor_pose, traj_duration);
        return run_simulation(model, data, col_model, col_data, sim, traj, q_home, params,
                              xarm_geo::trajectories::FigureEight::kName);
    }
    if (params.trajectory_mode == 1) {
        xarm_geo::trajectories::WingInspection traj(anchor_pose, traj_duration);
        return run_simulation(model, data, col_model, col_data, sim, traj, q_home, params,
                              xarm_geo::trajectories::WingInspection::kName);
    }
    if (params.trajectory_mode == 2) {
        xarm_geo::trajectories::PipeInspection traj(anchor_pose, traj_duration);
        return run_simulation(model, data, col_model, col_data, sim, traj, q_home, params,
                              xarm_geo::trajectories::PipeInspection::kName);
    }
    if (params.trajectory_mode == 3) {
        xarm_geo::trajectories::InnerCavityScan traj(anchor_pose, traj_duration);
        return run_simulation(model, data, col_model, col_data, sim, traj, q_home, params,
                              xarm_geo::trajectories::InnerCavityScan::kName);
    }
    if (params.trajectory_mode == 4) {
        xarm_geo::trajectories::TiltingCircle traj(anchor_pose, traj_duration);
        return run_simulation(model, data, col_model, col_data, sim, traj, q_home, params,
                              xarm_geo::trajectories::TiltingCircle::kName);
    }
}
