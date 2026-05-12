#pragma once

#include <memory>

#include <xarm/wrapper/xarm_api.h>

#include <xarm_geo/interfaces/interface.h>

namespace xarm_geo {

    class Hardware {

    public:
        // --- Direct SDK Access ---

        std::unique_ptr<XArmAPI> arm;

        // --- Constructors & Destructor ---

        explicit Hardware(int dof, const std::string &robot_ip);
        ~Hardware();

        // --- Interface Concept: Lifecycle & State Reading ---

        [[nodiscard]] auto is_running() const -> bool;
        void shutdown();
        auto read(JointState &state) noexcept -> InterfaceStatus;
        [[nodiscard]] auto read_time() const noexcept -> std::chrono::nanoseconds;

        // --- Controllable Concept: Command Writing ---

        auto write(const JointVelocity &cmd) noexcept -> InterfaceStatus;

        // --- Utility ---

        void stop();
        void disable_motors();

    private:
        int dof_;
        std::chrono::nanoseconds last_read_time_{0};
    };

    // --- Compile-Time Concept Verifications ---

    static_assert(Interface<Hardware>);
    static_assert(Controllable<Hardware, JointVelocity>);

}  // namespace xarm_geo
