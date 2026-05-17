#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include <Eigen/Dense>

#include <xarm_geo/control/admittance.h>
#include <xarm_geo/control/controller.h>
#include <xarm_geo/core/manifold.h>
#include <xarm_geo/core/motion.h>
#include <xarm_geo/core/system.h>
#include <xarm_geo/trajectory/trajectory.h>

namespace xarm_geo::diagnostics {

    // --- Log Sample ---
    //
    // One row of the wide-CSV schema. Written once per logged tick.
    //
    // Column groups:
    //   time          : t (s), tick (monotonic counter)
    //   joint state   : q, v, tau_measured (size = dof)
    //   joint refs    : q_ref, v_ref, a_ref (joint-space phases only; empty otherwise)
    //   task state    : ee_pose_actual, ee_pose_target (quaternion xyzw), ee_twist_actual/target
    //   torque triplet: tau_ctrl, tau_des, tau_safe (dynamic controllers; empty in velocity mode)
    //   velocity trip.: v_ctrl, v_des, v_safe (kinematic controllers; empty in torque mode)
    //   diagnostics   : controller_status, asif_status, asif_invoked, asif_modified,
    //                   optik_status, optik_invoked, optik_modified
    //
    // Empty VectorXd fields (size == 0) are written as blank CSV cells.
    // Rotations are stored as quaternion (qx, qy, qz, qw); no Euler anywhere.

    struct LogSample {
        // --- Time ---
        double t = 0.0;
        std::int64_t tick = 0;

        // --- Joint State ---
        Eigen::VectorXd q;
        Eigen::VectorXd v;
        Eigen::VectorXd tau_measured;

        // --- Joint References (joint-space phases only) ---
        Eigen::VectorXd q_ref;
        Eigen::VectorXd v_ref;
        Eigen::VectorXd a_ref;

        // --- Task-Space Actual & Target ---
        manifold::SE3 ee_pose_actual;
        manifold::SE3 ee_pose_target;
        manifold::SE3::Twist ee_twist_actual = manifold::SE3::Twist::Zero();
        manifold::SE3::Twist ee_twist_target = manifold::SE3::Twist::Zero();

        // --- Torque Triplet (dynamic controllers only) ---
        //
        // tau_ctrl : raw hook output, before bias compensation.
        // tau_des  : post bias-compensation; what would be sent without ASIF.
        // tau_safe : post ASIF certification; == tau_des when ASIF is off.
        Eigen::VectorXd tau_ctrl;
        Eigen::VectorXd tau_des;
        Eigen::VectorXd tau_safe;

        // --- Velocity Triplet (kinematic controllers only) ---
        //
        // v_ctrl : raw hook output, before any safety-layer shaping.
        // v_des  : == v_ctrl (no bias-compensation equivalent for kinematic bases).
        // v_safe : post OptIK / direction-preserving velocity-limit rescale.
        Eigen::VectorXd v_ctrl;
        Eigen::VectorXd v_des;
        Eigen::VectorXd v_safe;

        // --- Controller / Solver Diagnostics ---
        //
        // 0xFF signals "not invoked this tick" for status bytes.
        std::uint8_t controller_status = 0;
        std::uint8_t asif_status = 0xFF;
        bool asif_invoked = false;
        bool asif_modified = false;  // ||tau_safe - tau_des|| > eps

        std::uint8_t optik_status = 0xFF;
        bool optik_invoked = false;
        bool optik_modified = false;  // ||v_safe - v_des|| > eps
    };

    // --- DataLogger ---
    //
    // RAM-buffered, flush-on-close logger. Call log() once per tick inside the
    // control loop (O(1), no I/O); flush() or the destructor writes the CSV and
    // a JSON sidecar to disk after the trial ends.
    //
    // Pre-allocate with Config::reserve_samples to avoid any heap allocation
    // during the trial. For a 30 s trial at 500 Hz that is ~15 000 rows.
    //
    // TODO: add flush_strategy = Sync | Async to Config if trials grow past
    // ~minutes and RAM budget becomes a concern.

    class DataLogger {
    public:
        struct Config {
            std::string output_path;              // full path including filename, e.g.
                                                  // "tests/results/trial.csv"
            std::string trial_name;               // written into the JSON sidecar
            std::size_t reserve_samples = 16000;  // pre-allocated rows
            int decimation = 1;                   // log every Nth call; 1 = log all
            std::int64_t skip_first = 0;          // drop the first N calls (warm-up)
        };

        // Captures tau_max, q_max, v_max from the model for the sidecar.
        DataLogger(const Model &model, Config cfg);

        // Calls flush() if the buffer has not been written yet.
        ~DataLogger();

        // Non-copyable; movable.
        DataLogger(const DataLogger &) = delete;
        DataLogger &operator=(const DataLogger &) = delete;
        DataLogger(DataLogger &&) = default;
        DataLogger &operator=(DataLogger &&) = default;

        // Record one sample. O(1) amortised; never touches the filesystem.
        void log(const LogSample &sample) noexcept;

        // Write the buffered samples to CSV and emit a JSON sidecar
        // (<output_path>.meta.json). Idempotent: second call is a no-op.
        void flush();

        [[nodiscard]] auto size() const noexcept -> std::size_t { return buf_.size(); }

    private:
        Config cfg_;
        int dof_;
        Eigen::VectorXd tau_max_;
        Eigen::VectorXd q_min_;
        Eigen::VectorXd q_max_;
        Eigen::VectorXd v_max_;
        std::vector<LogSample> buf_;
        std::int64_t call_count_ = 0;
        bool flushed_ = false;

        void write_csv() const;
        void write_sidecar() const;
        void write_csv_header(std::ostream &os) const;
        void write_csv_row(std::ostream &os, const LogSample &s) const;
    };

    // --- Trial Name Builder ---
    //
    // Composes a self-describing trial filename stem from the controller class
    // name, trajectory class name, and active flags.  Example output:
    //   "sim_GeometricPDController_PipeInspection_safe_ff"
    //
    // `backend`      : "sim" or "hardware".
    // `controller`   : controller::kName (or any string_view).
    // `trajectory`   : trajectory::kName (or any string_view).
    // `constraint`   : true if any safety routing layer was active (ASIF for
    //                  dynamic controllers, OptIK or velocity rescale for
    //                  kinematic controllers).
    // `feedforward`  : true if the controller's use_feedforward flag was on.
    // `suffix`       : optional extra tag appended after "_ff" / "_noff".

    [[nodiscard]] auto make_trial_name(std::string_view backend, std::string_view controller,
                                       std::string_view trajectory, bool constraint,
                                       bool feedforward, std::string_view suffix = "")
        -> std::string;

    // --- Sample Assembly Helpers ---
    //
    // Convenience free functions that fill the common groups of a LogSample
    // from library types. Call them in sequence; each touches only its own
    // group of fields and leaves the rest unchanged.

    // Fills: t, tick, q, v, tau_measured, ee_pose_actual, ee_twist_actual,
    //        ee_pose_target, ee_twist_target.
    void fill_task_sample(LogSample &s, double t, std::int64_t tick, const JointState &fb,
                          const TaskTarget &ref, const Data &data) noexcept;

    // Fills: t, tick, q, v, tau_measured, q_ref, v_ref, a_ref.
    void fill_joint_sample(LogSample &s, double t, std::int64_t tick, const JointState &fb,
                           const JointTarget &ref) noexcept;

    // Fills the torque triplet + ASIF diagnostic fields from a dynamic controller's
    // last-tick snapshot. Call immediately after update().
    void fill_torque_diagnostics(LogSample &s, const DynamicTaskControllerBase &c) noexcept;
    void fill_torque_diagnostics(LogSample &s, const DynamicJointControllerBase &c) noexcept;

    // Fills the velocity triplet + OptIK/rescale diagnostic fields from a kinematic
    // controller's last-tick snapshot. Call immediately after update().
    void fill_velocity_diagnostics(LogSample &s, const KinematicTaskControllerBase &c) noexcept;
    void fill_velocity_diagnostics(LogSample &s, const KinematicJointControllerBase &c) noexcept;

    // Fills the velocity triplet from an AdmittanceLayer's last-tick snapshot:
    //   v_ctrl <- v_state  (admittance ODE state, pre-feedforward)
    //   v_des  <- v_state + v_ff  (pre-rescale)
    //   v_safe <- post velocity-limit rescale
    // Does not touch the optik_* fields; fill those separately if safe_velocity_projection
    // was also called (see exp_3a_sim_hw.cpp for the combined pattern).
    void fill_admittance_diagnostics(LogSample &s, const AdmittanceLayer &a) noexcept;

}  // namespace xarm_geo::diagnostics
