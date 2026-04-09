#include <stdexcept>

#include <xarm_geo/interfaces/hardware.h>

namespace xarm_geo {

    // --- Constructors & Destructors ---

    Hardware::Hardware(int dof, const std::string &robot_ip) : dof_(dof) {
        // Initialise xArm SDK
        // Note: Setting Default Angles to Radians
        this->arm = std::make_unique<XArmAPI>(robot_ip, true);

        if (!this->arm->is_connected()) {
            throw std::runtime_error("Failed to connect to xArm @ IP: " + robot_ip);
        }

        // xArm Setup
        this->arm->clean_error();
        this->arm->clean_warn();
        this->arm->motion_enable(true);

        // Set xArm State & Control Mode to Joint Velocity Control
        this->arm->set_mode(4);
        this->arm->set_state(0);
    }

    Hardware::~Hardware() { this->shutdown(); }

    // --- Concept: Lifecycle Management ---

    auto Hardware::is_running() const -> bool {
        return this->arm->is_connected() && !this->arm->has_err_warn();
    }

    void Hardware::shutdown() {
        if (this->arm && this->arm->is_connected()) {
            this->arm->motion_enable(false);
            this->arm->disconnect();
        }
    }

    // --- Concept: Observable ---

    auto Hardware::read(JointState &data) -> InterfaceStatus {
        // Setting Up Data Arrays
        // Note: xArm SDK expects Fixed Size of 7
        float pos[7] = {0.0F};
        float vel[7] = {0.0F};
        float torque[7] = {0.0F};

        // Reading Joint State Data
        int res = this->arm->get_joint_states(pos, vel, torque);
        if (res != 0) { return InterfaceStatus::ERROR; }

        for (int i = 0; i < this->dof_; ++i) {
            data.q[i] = static_cast<double>(pos[i]);
            data.v[i] = static_cast<double>(vel[i]);
            data.tau[i] = static_cast<double>(torque[i]);
        }

        this->last_read_time_ = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch());

        return InterfaceStatus::OK;
    }

    auto Hardware::read_time() const -> std::chrono::nanoseconds { return this->last_read_time_; }

    // --- Concept: Controllable ---

    auto Hardware::write(const JointVelocity &cmd) -> InterfaceStatus {
        // Setting up Command Array
        // Note: xArm SDK expects Fixed Size of 7
        float vel[7] = {0.0F};

        // Sending Velocity Command
        for (int i = 0; i < this->dof_; ++i) { vel[i] = static_cast<float>(cmd.v[i]); }
        int res = this->arm->vc_set_joint_velocity(vel);

        return (res == 0) ? InterfaceStatus::OK : InterfaceStatus::ERROR;
    }
}  // namespace xarm_geo
