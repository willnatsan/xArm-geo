#pragma once

#include <memory>

#include <xarm/wrapper/xarm_api.h>

#include <xarm_geo/interfaces/interface.h>

namespace xarm_geo {
    class Hardware {
    public:
        // --- Direct Access (If Needed) ---
        std::unique_ptr<XArmAPI> arm;

        // --- Constructors & Destructors ---
        explicit Hardware(int dof, const std::string &robot_ip);
        ~Hardware();

        // --- Concept: Interface ---
        [[nodiscard]] auto is_running() const -> bool;
        void shutdown();

        // --- Concept: Observable (READ) ---
        auto read(JointState &data) -> InterfaceStatus;
        [[nodiscard]] auto read_time() const -> std::chrono::nanoseconds;

        // --- Concept: Controllable (WRITE) ---
        auto write(const JointVelocity &cmd) -> InterfaceStatus;

    private:
        int dof_;
        std::chrono::nanoseconds last_read_time_{0};
    };

    // --- Compile-Time Concept Verifications ---
    static_assert(Interface<Hardware>);
    static_assert(Observable<Hardware, JointState>);
    static_assert(Controllable<Hardware, JointVelocity>);
}  // namespace xarm_geo
