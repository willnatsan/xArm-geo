#include <iostream>

#include <xarm_geo/utils/data_config.h>

namespace xarm_geo::internal {
    [[nodiscard]] auto parse_serial_number(const std::string &robot_sn, bool model1300)
        -> ParsedSN {
        ParsedSN parsed;

        if (robot_sn.length() < 14 ||
            (robot_sn[0] != 'X' && robot_sn[0] != 'L' && robot_sn[0] != 'F')) {
            std::cerr << "Invalid xArm Serial Number; Sticking with defaults...\n";
            return parsed;  // is_valid remains false
        }

        parsed.is_valid = true;
        int model_num = model1300 ? 1300 : -1;

        // --- Parse model_type ---
        if (robot_sn[0] == 'F' && robot_sn[1] == 'X') parsed.model_type = 12;
        else if (robot_sn[0] == 'L') parsed.model_type = 9;
        else if (robot_sn[0] == 'X') {
            if (robot_sn[1] == 'F') parsed.model_type = 5;
            else if (robot_sn[1] == 'I') parsed.model_type = 6;
            else if (robot_sn[1] == 'S') parsed.model_type = 7;
        }

        // --- Parse model_num ---
        try {
            model_num = std::stoi(robot_sn.substr(2, 4));
        } catch (const std::exception &e) {
            std::cerr << "Warning: Could not parse model_num from SN. Using default.\n";
        }

        if (model_num == 1250) parsed.model_type = 8;
        if (model_num == 1380) parsed.model_type = 11;

        // --- Parse date_int ---
        int date_int = 0;
        try {
            std::string date_str = isdigit(robot_sn[8])
                                       ? (robot_sn.substr(8, 2) + robot_sn.substr(6, 2))
                                       : ("2" + robot_sn.substr(9, 1) + robot_sn.substr(6, 2));
            date_int = std::stoi(date_str);
        } catch (const std::exception &e) {
            std::cerr << "Warning: Could not parse date_int from SN.\n";
        }

        // --- Parse hd_type & mass_type ---
        char hd_type = '\0';
        if (robot_sn[11] == 'B' || robot_sn[11] == 'L' || robot_sn[11] == 'X' ||
            robot_sn[11] == 'D' || robot_sn[11] == 'A') {
            hd_type = robot_sn[11];
        } else if (robot_sn.length() >= 16 && robot_sn[14] == '_') {
            hd_type = robot_sn[15];
        }

        if (hd_type == 'B') {
            parsed.mass_type = (date_int >= 2004) ? 11 : 1;
        } else {
            if (hd_type == 'L') parsed.mass_type = 2;
            else if (hd_type == 'A') parsed.mass_type = 3;
            else if (hd_type == 'D') parsed.mass_type = 4;
        }

        return parsed;
    }

    [[nodiscard]] auto get_kinematic_file(int num_dof, const std::string &robot_type)
        -> std::string {
        if (num_dof == 6 && robot_type == "lite") return "lite6_default_kinematics.yaml";
        if (num_dof == 6 && robot_type == "uf850") return "uf850_default_kinematics.yaml";
        if (num_dof == 6) return "xarm6_default_kinematics.yaml";
        if (num_dof == 7 && robot_type == "xarm7_mirror")
            return "xarm7_mirror_default_kinematics.yaml";
        if (num_dof == 7) return "xarm7_default_kinematics.yaml";
        if (num_dof == 5) return "xarm5_default_kinematics.yaml";

        std::cerr << "Invalid xArm DOF / Type [Kinematics].\n";
        return "";
    }

    [[nodiscard]] auto get_inertial_file(int num_dof, const std::string &robot_type,
                                         const ParsedSN &sn) -> std::string {
        // Fallback guess
        std::string res =
            "xarm" + std::to_string(num_dof) + "_type" + std::to_string(num_dof) + "_HT_BR2.yaml";
        if (num_dof == 6 && robot_type == "lite") res = "xarm6_type9_HT_BR2.yaml";
        if (num_dof == 6 && robot_type == "uf850") res = "xarm6_type12_HT_LDBR2.yaml";
        if (num_dof == 7 && robot_type == "xarm7_mirror") res = "xarm7_type13_HT_BR2.yaml";

        if (!sn.is_valid) return res;

        // Valid SN logic
        if (num_dof == 5 && sn.model_type == 5) {
            if (sn.mass_type == 1) return "xarm5_type5_HT_BR.yaml";
            if (sn.mass_type == 2) return "xarm5_type5_HT_LD.yaml";
            if (sn.mass_type == 3) return "xarm5_type5_UJ_BR_2403.yaml";
            if (sn.mass_type == 11) return "xarm5_type5_HT_BR2.yaml";
        } else if (num_dof == 6) {
            if (robot_type == "uf850" && sn.model_type == 12) return "xarm6_type12_HT_LDBR2.yaml";
            if (robot_type == "lite" && sn.model_type == 9) return "xarm6_type9_HT_BR2.yaml";

            if (sn.model_type == 6) {
                if (sn.mass_type == 1) return "xarm6_type6_HT_BR.yaml";
                if (sn.mass_type == 2) return "xarm6_type6_HT_LD.yaml";
                if (sn.mass_type == 3) return "xarm6_type6_UJ_BR_2403.yaml";
                if (sn.mass_type == 11) return "xarm6_type6_HT_BR2.yaml";
            }
            if (sn.model_type == 8) return "xarm6_type8_HT2_BR2.yaml";
            if (sn.model_type == 11) return "xarm6_type11_HT_LD.yaml";
        } else if (num_dof == 7) {
            if (sn.model_type == 3) return "xarm7_type3_YT_SP.yaml";
            if (sn.model_type == 7) {
                if (sn.mass_type == 1) return "xarm7_type7_HT_BR.yaml";
                if (sn.mass_type == 2) return "xarm7_type7_HT_LD.yaml";
                if (sn.mass_type == 3) return "xarm7_type7_UJ_BR_2403.yaml";
                if (sn.mass_type == 11) return "xarm7_type7_HT_BR2.yaml";
            }
        }

        return res;  // Safety net
    }

    [[nodiscard]] auto get_description_files(int num_dof, const std::string &robot_type)
        -> std::pair<std::string, std::string> {
        if (num_dof == 6 && robot_type == "lite") return {"lite6.urdf", "lite6_scene.xml"};
        if (num_dof == 6 && robot_type == "uf850") return {"uf850.urdf", "uf850_scene.xml"};
        if (num_dof == 6) return {"xarm6.urdf", "xarm6_scene.xml"};
        if (num_dof == 7 && robot_type == "xarm7_mirror")
            return {"xarm7_mirror.urdf", "xarm7_mirror_scene.xml"};
        if (num_dof == 7) return {"xarm7.urdf", "xarm7_scene.xml"};
        if (num_dof == 5) return {"xarm5.urdf", "xarm5_scene.xml"};

        std::cerr << "Invalid xArm DOF / Type [Descriptions]\n";
        return {"", ""};
    }
}  // namespace xarm_geo::internal
