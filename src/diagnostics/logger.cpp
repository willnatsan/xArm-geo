#include <xarm_geo/diagnostics/logger.h>

#include <format>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace xarm_geo::diagnostics {

    // --- DataLogger ---

    DataLogger::DataLogger(const Model &model, Config cfg) : cfg_(std::move(cfg)), dof_(model.dof) {

        tau_max_.resize(dof_);
        q_min_.resize(dof_);
        q_max_.resize(dof_);
        v_max_.resize(dof_);

        for (int i = 0; i < dof_; ++i) {
            tau_max_[i] = model.limits[i].tau_max;
            q_min_[i] = model.limits[i].q_min;
            q_max_[i] = model.limits[i].q_max;
            v_max_[i] = model.limits[i].q_vel_max;
        }

        buf_.reserve(cfg_.reserve_samples);
    }

    DataLogger::~DataLogger() {
        if (!flushed_) {
            try {
                flush();
            } catch (const std::exception &e) {
                std::cerr << "[DataLogger] flush() in destructor threw: " << e.what() << '\n';
            }
        }
    }

    void DataLogger::log(const LogSample &sample) noexcept {
        ++call_count_;

        if (call_count_ <= cfg_.skip_first) { return; }

        const std::int64_t effective = call_count_ - cfg_.skip_first - 1;
        if ((effective % cfg_.decimation) != 0) { return; }

        buf_.push_back(sample);
    }

    void DataLogger::flush() {
        if (flushed_) { return; }
        flushed_ = true;

        if (buf_.empty()) { return; }

        write_csv();
        write_sidecar();
    }

    // --- CSV Writer ---

    namespace {

        // Write one double, or an empty cell if the value is not finite.
        void write_double(std::ostream &os, double v) {
            if (std::isfinite(v)) { os << v; }
            // else: leave blank (e.g. tau_max = +inf from default JointLimits)
        }

        // Write every element of a VectorXd as comma-separated cells.
        // If the vector is empty, write `n` blank cells.
        void write_vector(std::ostream &os, const Eigen::VectorXd &vec, int n) {
            if (vec.size() == n) {
                for (int i = 0; i < n; ++i) {
                    if (i > 0) { os << ','; }
                    write_double(os, vec[i]);
                }
            } else {
                // Vector absent for this row (e.g. torques in a velocity-mode trial).
                for (int i = 0; i < n; ++i) {
                    if (i > 0) { os << ','; }
                }
            }
        }

        // Write bool as 0 / 1.
        void write_bool(std::ostream &os, bool b) { os << (b ? '1' : '0'); }

        // Write a uint8 status byte; 255 (0xFF) is written as blank.
        void write_status(std::ostream &os, std::uint8_t s) {
            if (s != 0xFF) { os << static_cast<int>(s); }
        }

        // Column header block for a per-joint vector: prefix.0, prefix.1, ...
        void write_vector_header(std::ostream &os, std::string_view prefix, int n,
                                 bool leading_comma = true) {
            for (int i = 0; i < n; ++i) {
                if (leading_comma || i > 0) { os << ','; }
                os << prefix << '.' << i;
            }
        }

    }  // namespace

    void DataLogger::write_csv_header(std::ostream &os) const {
        // Time.
        os << "t,tick";

        // Joint state.
        write_vector_header(os, "q", dof_);
        write_vector_header(os, "v", dof_);
        write_vector_header(os, "tau_measured", dof_);

        // Joint references (present only in joint-space phases; blank otherwise).
        write_vector_header(os, "q_ref", dof_);
        write_vector_header(os, "v_ref", dof_);
        write_vector_header(os, "a_ref", dof_);

        // Task-space actual.
        os << ",ee_actual.x,ee_actual.y,ee_actual.z"
           << ",ee_actual.qx,ee_actual.qy,ee_actual.qz,ee_actual.qw";

        // Task-space target.
        os << ",ee_target.x,ee_target.y,ee_target.z"
           << ",ee_target.qx,ee_target.qy,ee_target.qz,ee_target.qw";

        // EE twist actual and target (body frame, [v; omega]).
        os << ",ee_twist_actual.vx,ee_twist_actual.vy,ee_twist_actual.vz"
           << ",ee_twist_actual.wx,ee_twist_actual.wy,ee_twist_actual.wz";
        os << ",ee_twist_target.vx,ee_twist_target.vy,ee_twist_target.vz"
           << ",ee_twist_target.wx,ee_twist_target.wy,ee_twist_target.wz";

        // Torque triplet.
        write_vector_header(os, "tau_ctrl", dof_);
        write_vector_header(os, "tau_des", dof_);
        write_vector_header(os, "tau_safe", dof_);

        // Velocity triplet.
        write_vector_header(os, "v_ctrl", dof_);
        write_vector_header(os, "v_des", dof_);
        write_vector_header(os, "v_safe", dof_);

        // Diagnostics.
        os << ",controller_status"
           << ",asif_status,asif_invoked,asif_modified"
           << ",optik_status,optik_invoked,optik_modified";

        os << '\n';
    }

    void DataLogger::write_csv_row(std::ostream &os, const LogSample &s) const {
        // Time.
        os << s.t << ',' << s.tick;

        // Joint state.
        os << ',';
        write_vector(os, s.q, dof_);
        os << ',';
        write_vector(os, s.v, dof_);
        os << ',';
        write_vector(os, s.tau_measured, dof_);

        // Joint references.
        os << ',';
        write_vector(os, s.q_ref, dof_);
        os << ',';
        write_vector(os, s.v_ref, dof_);
        os << ',';
        write_vector(os, s.a_ref, dof_);

        // Task-space actual (position then quaternion xyzw).
        const Eigen::Vector3d p_a = s.ee_pose_actual.r3();
        const Eigen::Quaterniond q_a = s.ee_pose_actual.so3().quat();
        os << ',' << p_a.x() << ',' << p_a.y() << ',' << p_a.z() << ',' << q_a.x() << ',' << q_a.y()
           << ',' << q_a.z() << ',' << q_a.w();

        // Task-space target.
        const Eigen::Vector3d p_t = s.ee_pose_target.r3();
        const Eigen::Quaterniond q_t = s.ee_pose_target.so3().quat();
        os << ',' << p_t.x() << ',' << p_t.y() << ',' << p_t.z() << ',' << q_t.x() << ',' << q_t.y()
           << ',' << q_t.z() << ',' << q_t.w();

        // EE twists (body frame, linear then angular).
        for (int i = 0; i < 6; ++i) { os << ',' << s.ee_twist_actual[i]; }
        for (int i = 0; i < 6; ++i) { os << ',' << s.ee_twist_target[i]; }

        // Torque triplet.
        os << ',';
        write_vector(os, s.tau_ctrl, dof_);
        os << ',';
        write_vector(os, s.tau_des, dof_);
        os << ',';
        write_vector(os, s.tau_safe, dof_);

        // Velocity triplet.
        os << ',';
        write_vector(os, s.v_ctrl, dof_);
        os << ',';
        write_vector(os, s.v_des, dof_);
        os << ',';
        write_vector(os, s.v_safe, dof_);

        // Diagnostics.
        os << ',';
        write_status(os, s.controller_status);
        os << ',';
        write_status(os, s.asif_status);
        os << ',';
        write_bool(os, s.asif_invoked);
        os << ',';
        write_bool(os, s.asif_modified);
        os << ',';
        write_status(os, s.optik_status);
        os << ',';
        write_bool(os, s.optik_invoked);
        os << ',';
        write_bool(os, s.optik_modified);

        os << '\n';
    }

    void DataLogger::write_csv() const {
        std::ofstream os(cfg_.output_path);
        if (!os) {
            throw std::runtime_error(
                std::format("[DataLogger] could not open '{}' for writing", cfg_.output_path));
        }

        write_csv_header(os);
        for (const auto &s : buf_) { write_csv_row(os, s); }
    }

    // --- JSON Sidecar Writer ---
    //
    // Hand-rolled minimal JSON; no external dep. Format:
    //   { "trial_name": "...", "dof": N, "dt_nominal_s": 0.002,
    //     "tau_max": [...], "q_min": [...], "q_max": [...], "v_max": [...] }

    void DataLogger::write_sidecar() const {
        const std::string path = cfg_.output_path + ".meta.json";
        std::ofstream os(path);
        if (!os) {
            std::cerr << "[DataLogger] warning: could not open sidecar '" << path << "'\n";
            return;
        }

        const double dt_nominal = (buf_.size() >= 2) ? (buf_[1].t - buf_[0].t) : 0.0;

        os << "{\n";
        os << "  \"trial_name\": \"" << cfg_.trial_name << "\",\n";
        os << "  \"dof\": " << dof_ << ",\n";
        os << "  \"dt_nominal_s\": " << dt_nominal << ",\n";
        os << "  \"decimation\": " << cfg_.decimation << ",\n";

        auto write_json_vec = [&](std::string_view key, const Eigen::VectorXd &vec) {
            os << "  \"" << key << "\": [";
            for (int i = 0; i < dof_; ++i) {
                if (i > 0) { os << ", "; }
                write_double(os, vec[i]);
            }
            os << "]";
        };

        write_json_vec("tau_max", tau_max_);
        os << ",\n";
        write_json_vec("q_min", q_min_);
        os << ",\n";
        write_json_vec("q_max", q_max_);
        os << ",\n";
        write_json_vec("v_max", v_max_);
        os << "\n}\n";
    }

    // --- Trial Name Builder ---

    auto make_trial_name(std::string_view backend, std::string_view controller,
                         std::string_view trajectory, bool constraint, bool feedforward,
                         std::string_view suffix) -> std::string {
        std::string name;
        name.reserve(64);

        name += backend;
        name += '_';
        name += controller;
        name += '_';
        name += trajectory;

        if (constraint) { name += "_asif"; }
        name += feedforward ? "_ff" : "_noff";

        if (!suffix.empty()) {
            name += '_';
            name += suffix;
        }

        return name;
    }

    // --- Sample Assembly Helpers ---

    void fill_task_sample(LogSample &s, double t, std::int64_t tick, const JointState &fb,
                          const TaskTarget &ref, const Data &data) noexcept {
        s.t = t;
        s.tick = tick;
        s.q = fb.q;
        s.v = fb.v;
        s.tau_measured = fb.tau;
        s.ee_pose_actual = data.ee_pose;
        s.ee_pose_target = ref.pose;
        // Body-frame actual twist: J_b * v.  The caller must have already called
        // compute_jacobians() (standard in every task-controller update path).
        s.ee_twist_actual = data.body_jacobian * fb.v;
        s.ee_twist_target = ref.twist;
    }

    void fill_joint_sample(LogSample &s, double t, std::int64_t tick, const JointState &fb,
                           const JointTarget &ref) noexcept {
        s.t = t;
        s.tick = tick;
        s.q = fb.q;
        s.v = fb.v;
        s.tau_measured = fb.tau;
        s.q_ref = ref.q;
        s.v_ref = ref.v;
        s.a_ref = ref.a;
    }

    // --- Diagnostic Fill Helpers ---

    void fill_torque_diagnostics(LogSample &s, const DynamicTaskControllerBase &c) noexcept {
        const auto &d = c.last_tick_diagnostics();
        s.tau_ctrl = d.tau_ctrl;
        s.tau_des = d.tau_des;
        s.tau_safe = d.tau_safe;
        s.asif_invoked = d.asif_invoked;
        s.asif_modified = d.asif_modified;
        s.asif_status = static_cast<std::uint8_t>(d.asif_status);
    }

    void fill_torque_diagnostics(LogSample &s, const DynamicJointControllerBase &c) noexcept {
        const auto &d = c.last_tick_diagnostics();
        s.tau_ctrl = d.tau_ctrl;
        s.tau_des = d.tau_des;
        s.tau_safe = d.tau_safe;
        s.asif_invoked = d.asif_invoked;
        s.asif_modified = d.asif_modified;
        s.asif_status = static_cast<std::uint8_t>(d.asif_status);
    }

    void fill_velocity_diagnostics(LogSample &s, const KinematicTaskControllerBase &c) noexcept {
        const auto &d = c.last_tick_diagnostics();
        s.v_ctrl = d.v_ctrl;
        s.v_des = d.v_des;
        s.v_safe = d.v_safe;
        s.optik_invoked = d.optik_invoked;
        s.optik_modified = d.optik_modified;
        s.optik_status = static_cast<std::uint8_t>(d.optik_status);
    }

    void fill_velocity_diagnostics(LogSample &s, const KinematicJointControllerBase &c) noexcept {
        const auto &d = c.last_tick_diagnostics();
        s.v_ctrl = d.v_ctrl;
        s.v_des = d.v_des;
        s.v_safe = d.v_safe;
        s.optik_invoked = d.optik_invoked;
        s.optik_modified = d.optik_modified;
        s.optik_status = static_cast<std::uint8_t>(d.optik_status);
    }

}  // namespace xarm_geo::diagnostics
