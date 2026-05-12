#include <stdexcept>

#include <xarm_geo/interfaces/hardware.h>

namespace xarm_geo {

    // --- Constructors & Destructors ---

    Hardware::Hardware(int dof, const std::string &robot_ip) : dof_(dof) {
        // Default angles in radians (second arg = true).
        this->arm = std::make_unique<XArmAPI>(robot_ip, true);

        if (!this->arm->is_connected()) {
            throw std::runtime_error("Failed to connect to xArm @ IP: " + robot_ip);
        }

        this->arm->clean_error();
        this->arm->clean_warn();
        this->arm->motion_enable(true);

        // Mode 4 = joint velocity control; state 0 = ready.
        this->arm->set_mode(4);
        this->arm->set_state(0);

        // Flush the velocity command buffer.
        this->stop();
    }

    Hardware::~Hardware() { this->shutdown(); }

    // --- Interface Concept: Lifecycle & State Reading ---

    auto Hardware::is_running() const -> bool {
        return this->arm->is_connected() && !this->arm->has_err_warn();
    }

    void Hardware::shutdown() {
        this->stop();

        if (this->arm && this->arm->is_connected()) { this->arm->disconnect(); }
    }

    auto Hardware::read(JointState &state) noexcept -> InterfaceStatus {
        // xArm SDK requires fixed-size buffers of 7.
        float pos[7] = {0.0F};
        float vel[7] = {0.0F};
        float torque[7] = {0.0F};

        int res = this->arm->get_joint_states(pos, vel, torque);
        if (res != 0) { return InterfaceStatus::ERROR; }

        for (int i = 0; i < this->dof_; ++i) {
            state.q[i] = static_cast<double>(pos[i]);
            state.v[i] = static_cast<double>(vel[i]);
            state.tau[i] = static_cast<double>(torque[i]);
        }

        this->last_read_time_ = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch());

        return InterfaceStatus::OK;
    }

    auto Hardware::read_time() const noexcept -> std::chrono::nanoseconds {
        return this->last_read_time_;
    }

    // --- Controllable Concept: Command Writing ---

    auto Hardware::write(const JointVelocity &cmd) noexcept -> InterfaceStatus {
        float vel[7] = {0.0F};

        for (int i = 0; i < this->dof_; ++i) { vel[i] = static_cast<float>(cmd.v[i]); }
        int res = this->arm->vc_set_joint_velocity(vel);

        return (res == 0) ? InterfaceStatus::OK : InterfaceStatus::ERROR;
    }

    // --- Additional Utility Methods ---

    void Hardware::stop() {
        if (this->arm && this->arm->is_connected()) {
            float zero_vel[7] = {0.0F};
            this->arm->vc_set_joint_velocity(zero_vel);
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
    }

    void Hardware::disable_motors() {
        this->stop();

        if (this->arm && this->arm->is_connected()) { this->arm->motion_enable(false); }
    }

}  // namespace xarm_geo
