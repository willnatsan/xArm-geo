#include <iostream>
#include <limits>
#include <numbers>

#include <coal/mesh_loader/loader.h>
#include <coal/serialization/BVH_model.h>
#include <coal/shape/geometric_shapes.h>
#include <tinyxml2.h>
#include <yaml-cpp/yaml.h>

#include <xarm_geo/core/manifold.h>
#include <xarm_geo/utils/model_builder.h>
#include <xarm_geo/utils/model_config.h>
#include <xarm_geo/utils/parsing_utils.h>
#include <xarm_geo_config.h>

namespace xarm_geo::internal {
    void load_kinematic_params(xarm_geo::Model &model, const std::string &kinematic_file) {
        const YAML::Node config = YAML::LoadFile(KINEMATIC_PARAMS_PATH + kinematic_file);
        const YAML::Node kinematics = config["kinematics"];

        auto pose_curr = manifold::SE3::Identity();
        model.home_pose_tree.emplace_back(pose_curr);

        for (auto const &joint : kinematics) {
            const YAML::Node joint_data = joint.second;

            const auto x = joint_data["x"].as<double>();
            const auto y = joint_data["y"].as<double>();
            const auto z = joint_data["z"].as<double>();
            const auto roll = joint_data["roll"].as<double>();
            const auto pitch = joint_data["pitch"].as<double>();
            const auto yaw = joint_data["yaw"].as<double>();

            Eigen::Vector3d translation(x, y, z);
            manifold::SO3 rotation = manifold::rpy_to_SO3(roll, pitch, yaw);
            manifold::SE3 transform(rotation, translation);

            pose_curr *= transform;
            model.home_pose_tree.emplace_back(pose_curr);

            // Screw Axis for Revolute Z-axis Joint: Transform Local Z-axis to Spatial Frame
            manifold::SE3::Twist S_local_z = manifold::SE3::Twist::Zero();
            S_local_z.tail<3>() = Eigen::Vector3d::UnitZ();
            model.screw_axes_space.emplace_back(pose_curr.Ad() * S_local_z);
        }

        manifold::SE3 T_flange_to_ee = manifold::SE3::Identity();
        if (config["end_effector"]) {
            const YAML::Node ee_data = config["end_effector"];
            const auto x = ee_data["x"].as<double>(0.0);
            const auto y = ee_data["y"].as<double>(0.0);
            const auto z = ee_data["z"].as<double>(0.0);
            const auto roll = ee_data["roll"].as<double>(0.0);
            const auto pitch = ee_data["pitch"].as<double>(0.0);
            const auto yaw = ee_data["yaw"].as<double>(0.0);

            Eigen::Vector3d ee_trans(x, y, z);
            manifold::SO3 ee_rot = manifold::rpy_to_SO3(roll, pitch, yaw);

            T_flange_to_ee = manifold::SE3(ee_rot, ee_trans);
        }

        pose_curr *= T_flange_to_ee;

        model.home_pose_tree.emplace_back(pose_curr);
        model.home_pose = pose_curr;

        for (size_t i = 0; i < model.screw_axes_space.size(); ++i) {
            // home_pose_tree[0] is the base frame.
            // home_pose_tree[i + 1] is the home pose of link i relative to the base frame.
            const manifold::SE3 link_i_home_inv = model.home_pose_tree[i + 1].inverse();

            // Map the spatial screw axis into the local frame of the corresponding link
            model.screw_axes_local.emplace_back(link_i_home_inv.Ad() * model.screw_axes_space[i]);
        }
    }

    void load_inertial_params(xarm_geo::Model &model, const std::string &inertial_file) {
        const YAML::Node config = YAML::LoadFile(INERTIAL_PARAMS_PATH + inertial_file);
        std::vector<manifold::SE3::SpatialInertia> spatial_inertias_com;

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

            manifold::SE3::SpatialInertia spatial_inertia_com =
                manifold::SE3::SpatialInertia::Zero();
            spatial_inertia_com.topLeftCorner(3, 3) = com_inertia;
            spatial_inertia_com.bottomRightCorner(3, 3) = mass * Eigen::Matrix3d::Identity();
            spatial_inertias_com.emplace_back(spatial_inertia_com);

            // Reference Frame Change - CoM -> Link Origin (Joint Frame)
            // Assuming `origin` defines Transform from Link Origin -> CoM
            manifold::SE3 T_origin_com(manifold::SO3::Identity(), com_pos);
            Eigen::Matrix<double, 6, 6> Ad_T_com_origin = T_origin_com.inverse().Ad();

            manifold::SE3::SpatialInertia spatial_inertia_link =
                Ad_T_com_origin.transpose() * spatial_inertias_com[i] * Ad_T_com_origin;

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

        std::unordered_map<std::string, int> link_to_joint_map;

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
            limits_curr.q_min = limit->DoubleAttribute("lower", -2 * std::numbers::pi);
            limits_curr.q_max = limit->DoubleAttribute("upper", 2 * std::numbers::pi);
            limits_curr.q_vel_max = limit->DoubleAttribute("velocity", std::numbers::pi);
            limits_curr.tau_max =
                limit->DoubleAttribute("effort", std::numeric_limits<double>::infinity());
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

    void load_geometry_params(CollisionModel &col_model, const Model &kin_model) {
        tinyxml2::XMLDocument urdf;
        std::string urdf_path = URDF_PATH + kin_model.urdf_file;

        if (urdf.LoadFile(urdf_path.c_str()) != tinyxml2::XML_SUCCESS) {
            std::cerr << "Error loading URDF file: " << urdf_path << "\n";
            return;
        }

        const tinyxml2::XMLElement *robot = urdf.FirstChildElement("robot");
        if (!robot) {
            std::cerr << "Error parsing <robot> from URDF file: " << urdf_path << "\n";
            return;
        }

        // 1. Build Link Name -> Joint ID Map
        std::unordered_map<std::string, size_t> link_to_joint_map;
        size_t joint_idx = 0;

        for (const tinyxml2::XMLElement *child = robot->FirstChildElement("joint");
             child != nullptr; child = child->NextSiblingElement("joint")) {

            const tinyxml2::XMLElement *child_link = child->FirstChildElement("child");
            if (!child_link) continue;

            const char *link_name = child_link->Attribute("link");
            if (!link_name) continue;

            const char *type_attr = child->Attribute("type");
            std::string type = type_attr ? type_attr : "";

            if (type == "revolute" || type == "continuous") {
                // It's an active joint. Map it and increment the index.
                link_to_joint_map[std::string(link_name)] = joint_idx;
                joint_idx++;
                if (joint_idx >= kin_model.dof) break;
            } else if (type == "fixed") {
                // It's a bolted part. Map it to the CURRENT joint index so it moves
                // with the parent kinematic frame, rather than being glued to the World
                // (Note: joint_idx is already pointing to the next available index,
                // so we use joint_idx - 1, safely bounded at 0).
                size_t parent_idx = (joint_idx > 0) ? (joint_idx - 1) : 0;
                link_to_joint_map[std::string(link_name)] = parent_idx;
            }
        }

        static coal::CachedMeshLoader mesh_loader;

        // 2. Parse Links and Extract Collision Geometry
        for (const tinyxml2::XMLElement *link = robot->FirstChildElement("link"); link;
             link = link->NextSiblingElement("link")) {

            const char *name_attr = link->Attribute("name");
            if (!name_attr) continue;
            std::string link_name = name_attr;

            size_t current_joint_idx = 0;
            // .contains() is C++20. If compiling older, use .count() or .find()
            if (link_to_joint_map.contains(link_name)) {
                current_joint_idx = link_to_joint_map[link_name];
            }

            for (const tinyxml2::XMLElement *col = link->FirstChildElement("collision"); col;
                 col = col->NextSiblingElement("collision")) {

                manifold::SE3 offset = parse_origin(col->FirstChildElement("origin"));
                const tinyxml2::XMLElement *geom = col->FirstChildElement("geometry");
                if (!geom) continue;

                // Handle Meshes
                const tinyxml2::XMLElement *mesh_xml = geom->FirstChildElement("mesh");
                if (mesh_xml) {
                    const char *file = mesh_xml->Attribute("filename");
                    if (!file) continue;

                    std::string file_path(file);
                    const std::string file_prefix = "file://";

                    // If the path starts with "file://", strip it out
                    if (file_path.starts_with(file_prefix)) {
                        file_path.erase(0, file_prefix.length());
                    }

                    Eigen::Vector3d scale =
                        parse_vec3(mesh_xml->Attribute("scale"), Eigen::Vector3d::Ones());
                    coal::Vec3s coal_scale(scale.x(), scale.y(), scale.z());

                    try {
                        auto mesh_geom = mesh_loader.load(file_path, coal_scale);
                        col_model.add_geometry(link_name + "_col", current_joint_idx, offset,
                                               mesh_geom);
                    } catch (const std::exception &e) {
                        std::cerr << "Failed to load mesh: " << file_path << " | " << e.what()
                                  << "\n";
                    }
                }
            }
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

    [[nodiscard]] auto build_collision_model(const Model &kin_model, bool add_ground_plane)
        -> CollisionModel {
        CollisionModel col_model;

        // Load the Geometry from the URDF
        internal::load_geometry_params(col_model, kin_model);

        // Inject Robot-Specific Allowed Collision Matrix (Based on Official SRDF)
        // TODO: Add for Remaining Manipulators (xArm5, xArm7, etc.) OR Integrate SRDF Reading
        if (kin_model.dof == 6) {
            col_model.disable_collision_pair("link1_col", "link3_col");
            col_model.disable_collision_pair("link2_col", "link4_col");
            col_model.disable_collision_pair("link_base_col", "link2_col");
            col_model.disable_collision_pair("link_base_col", "link3_col");
            col_model.disable_collision_pair("link3_col", "link5_col");
            col_model.disable_collision_pair("link3_col", "link6_col");
            col_model.disable_collision_pair("link4_col", "link6_col");
        }

        // Add the Infinite Ground Plane (If Requested)
        if (add_ground_plane) {
            auto ground_geom = std::make_shared<coal::Halfspace>(coal::Vec3s(0, 0, 1), 0.0);

            manifold::SE3 ground_pose = manifold::SE3::Identity();
            ground_pose.r3() << 0.0, 0.0, 0.0;

            col_model.add_geometry("ground_plane", 0, ground_pose, ground_geom);
        }

        // Compile the Collision Pairs
        col_model.add_all_collision_pairs();

        return col_model;
    }
}  // namespace xarm_geo
