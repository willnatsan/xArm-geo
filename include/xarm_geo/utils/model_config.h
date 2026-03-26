#pragma once

#include <string>

namespace xarm_geo::internal {
    struct ParsedSN {
        bool is_valid = false;
        int model_type = -1;
        int mass_type = 1;  // Default
    };

    [[nodiscard]] auto parse_serial_number(const std::string &robot_sn, bool model1300) -> ParsedSN;
    [[nodiscard]] auto get_kinematic_file(int num_dof, const std::string &robot_type)
        -> std::string;
    [[nodiscard]] auto get_inertial_file(int num_dof, const std::string &robot_type,
                                         const ParsedSN &sn) -> std::string;
    [[nodiscard]] auto get_description_files(int dof, const std::string &robot_type)
        -> std::pair<std::string, std::string>;
}  // namespace xarm_geo::internal
