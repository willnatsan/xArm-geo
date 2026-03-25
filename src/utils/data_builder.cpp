#include <iostream>

#include <tinyxml2.h>
#include <yaml-cpp/yaml.h>

#include <xarm_geo/core/manifold.h>
#include <xarm_geo/utils/data_builder.h>
#include <xarm_geo/utils/data_config.h>
#include <xarm_geo_config.h>

namespace xarm_geo::internal {
    void load_kinematic_params(xarm_geo::Model &model, const std::string &kinematic_file) {
        const YAML::Node config = YAML::LoadFile(KINEMATIC_PARAMS_PATH + kinematic_file);
        const YAML::Node kinematics = config["kinematics"];
        auto pose_curr = xarm_geo::manifold::SE3::Identity();

        for (auto const &joint : kinematics) {
            const YAML::Node joint_data = joint.second;

            const auto x = joint_data["x"].as<double>();
            const auto y = joint_data["y"].as<double>();
            const auto z = joint_data["z"].as<double>();
            const auto roll = joint_data["roll"].as<double>();
            const auto pitch = joint_data["pitch"].as<double>();
            const auto yaw = joint_data["yaw"].as<double>();

            Eigen::Vector3d translation(x, y, z);
            Eigen::Quaterniond quat = Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()) *
                                      Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) *
                                      Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX());
            xarm_geo::manifold::SO3 rotation(quat);
            xarm_geo::manifold::SE3 transform(rotation, translation);

            pose_curr *= transform;
            model.home_pose_tree.emplace_back(pose_curr);

            const auto q = pose_curr.r3();
            const auto w = pose_curr.so3() * Eigen::Vector3d::UnitZ();
            const auto v = -w.cross(q);

            xarm_geo::manifold::SE3::Twist screw_axis_space;
            screw_axis_space.head<3>() = v;
            screw_axis_space.tail<3>() = w;
            model.screw_axes_space.push_back(screw_axis_space);
        }

        model.home_pose = pose_curr;

        const xarm_geo::manifold::SE3 M_inv = model.home_pose.inverse();
        for (auto const &screw_axis_space : model.screw_axes_space) {
            model.screw_axes_body.emplace_back(M_inv.Ad() * screw_axis_space);
        }
    }

    void load_inertial_params(xarm_geo::Model &model, const std::string &inertial_file) {
        const YAML::Node config = YAML::LoadFile(INERTIAL_PARAMS_PATH + inertial_file);
        for (int i = 0; i < model.dof; i++) {
            const YAML::Node link = config["link" + std::to_string(i + 1)];

            auto mass = link["mass"].as<double>();

            Eigen::Vector3d com_pos;
            com_pos << link["origin"]["x"].as<double>(), link["origin"]["y"].as<double>(),
                link["origin"]["z"].as<double>();

            Eigen::Matrix3d com_inertia;
            com_inertia << link["inertia"]["ixx"].as<double>(), link["inertia"]["ixy"].as<double>(),
                link["inertia"]["ixz"].as<double>(), link["inertia"]["ixy"].as<double>(),
                link["inertia"]["iyy"].as<double>(), link["inertia"]["iyz"].as<double>(),
                link["inertia"]["ixz"].as<double>(), link["inertia"]["iyz"].as<double>(),
                link["inertia"]["izz"].as<double>();

            xarm_geo::manifold::SE3::SpatialInertia spatial_inertia_com =
                xarm_geo::manifold::SE3::SpatialInertia::Zero();
            spatial_inertia_com.topLeftCorner(3, 3) = com_inertia;
            spatial_inertia_com.bottomRightCorner(3, 3) = mass * Eigen::Matrix3d::Identity();
            model.spatial_inertias_com.emplace_back(spatial_inertia_com);

            // Reference Frame Change - CoM -> Link Origin
            // Assuming `origin` defines Transform from Link Origin -> CoM
            xarm_geo::manifold::SE3 T_origin_com(xarm_geo::manifold::SO3::Identity(), com_pos);
            Eigen::Matrix<double, 6, 6> Ad_T_com_origin = T_origin_com.inverse().Ad();

            xarm_geo::manifold::SE3::SpatialInertia spatial_inertia_link =
                Ad_T_com_origin.transpose() * model.spatial_inertias_com[i] * Ad_T_com_origin;
            model.spatial_inertias_link.push_back(spatial_inertia_link);
        }
    }

    void load_constraint_params(xarm_geo::Model &model, const std::string &urdf_file) {
        tinyxml2::XMLDocument urdf;
        std::string urdf_path = URDF_PATH + urdf_file;

        if (const tinyxml2::XMLError err = urdf.LoadFile(urdf_path.c_str());
            err != tinyxml2::XML_SUCCESS) {
            std::cerr << "Error loading URDF file: " << urdf_file << "\n";
            return;
        }

        const tinyxml2::XMLElement *robot = urdf.FirstChildElement("robot");
        if (!robot) {
            std::cerr << "Error parsing <robot> from URDF file: " << urdf_file << "\n";
            return;
        }

        model.limits.clear();
        model.limits.reserve(model.dof);

        std::map<std::string, int> link_to_joint_map;

        // Parsing Joints
        int joint_idx = 0;
        for (const tinyxml2::XMLElement *child = robot->FirstChildElement(); child != nullptr;
             child = child->NextSiblingElement()) {
            if (std::string(child->Name()) != "joint") { continue; }

            const char *type_attr = child->Attribute("type");
            const char *name_attr = child->Attribute("name");
            std::string joint_name = name_attr ? name_attr : "unknown";
            if (const std::string type(type_attr ? type_attr : ""); type != "revolute") {
                continue;
            }

            const tinyxml2::XMLElement *limit = child->FirstChildElement("limit");
            if (!limit) {
                std::cerr << "Error parsing <limit> of " << joint_name
                          << " from URDF file: " << urdf_file << "\n";
                return;
            }

            xarm_geo::JointLimits limits_curr;
            limits_curr.q_min = limit->DoubleAttribute("lower", -2 * M_PI);
            limits_curr.q_max = limit->DoubleAttribute("upper", 2 * M_PI);
            limits_curr.q_vel_max = limit->DoubleAttribute("velocity", M_PI);
            model.limits.push_back(limits_curr);

            const tinyxml2::XMLElement *child_link = child->FirstChildElement("child");
            if (child_link) {
                const char *link_name = child_link->Attribute("link");
                if (link_name) { link_to_joint_map[std::string(link_name)] = joint_idx; }
            }

            joint_idx++;
            if (joint_idx >= model.dof) { break; }
        }

        if (joint_idx < model.dof) {
            std::cerr << "Warning: URDF contained fewer revolute joints (" << joint_idx
                      << ") than DOF (" << model.dof << ")." << "\n";
        }
    }
}  // namespace xarm_geo::internal

namespace xarm_geo {
    [[nodiscard]] auto build_model(int dof, const std::string &robot_sn,
                                   const std::string &robot_type, bool modell1300) -> Model {
        Model model;
        model.dof = dof;

        internal::ParsedSN sn_parsed = internal::parse_serial_number(robot_sn, modell1300);

        std::string kinematic_file = internal::get_kinematic_file(dof, robot_type);
        std::string inertial_file = internal::get_inertial_file(dof, robot_type, sn_parsed);

        std::pair<std::string, std::string> desc_files =
            internal::get_description_files(dof, robot_type);
        model.urdf_file = desc_files.first;
        model.mjcf_file = desc_files.second;

        internal::load_kinematic_params(model, kinematic_file);
        internal::load_inertial_params(model, inertial_file);
        internal::load_constraint_params(model, model.urdf_file);

        return model;
    }
}  // namespace xarm_geo
