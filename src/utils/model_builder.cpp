#include <cmath>
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

            // Spatial screw axis for the revolute joint. Assumes axis = local Z
            // (true for all xArm variants; URDFs with non-Z <axis xyz=...>
            // attributes will be silently incorrect).
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

        // Map spatial screw axes into each link's local frame.
        // home_pose_tree[i+1] is the home pose of link i relative to the base.
        for (size_t i = 0; i < model.screw_axes_space.size(); ++i) {
            const manifold::SE3 link_i_home_inv = model.home_pose_tree[i + 1].inverse();
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

            // Spatial inertia block layout for smooth's [v_lin; omega] tangent
            // convention: mass*I_3 in the top-left (linear) block, rotational
            // inertia tensor in the bottom-right (angular) block. Swapping the
            // two blocks produces a ~400x error in g(q) and M(q).
            manifold::SE3::SpatialInertia spatial_inertia_com =
                manifold::SE3::SpatialInertia::Zero();
            spatial_inertia_com.topLeftCorner(3, 3) = mass * Eigen::Matrix3d::Identity();
            spatial_inertia_com.bottomRightCorner(3, 3) = com_inertia;
            spatial_inertias_com.emplace_back(spatial_inertia_com);

            // CoM -> link origin frame change (origin defines link-origin -> CoM).
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

        // Warn once if any joint lacked <effort>; ASIF / torque controllers
        // cannot bound those joints.
        bool any_inf_tau = false;
        for (const auto &lim : model.limits) {
            if (!std::isfinite(lim.tau_max)) {
                any_inf_tau = true;
                break;
            }
        }
        if (any_inf_tau) {
            std::cerr << "Warning: URDF " << urdf_file
                      << " has joints without <limit effort=...>; tau_max defaulted to +inf. "
                         "ASIF / torque controllers will not enforce torque bounds on these joints."
                      << "\n";
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
                // Active joint -> map and advance the index.
                link_to_joint_map[std::string(link_name)] = joint_idx;
                joint_idx++;
                if (joint_idx >= kin_model.dof) break;
            } else if (type == "fixed") {
                // Fixed link -> bind it to the parent's joint frame (so it
                // moves with the parent, not the world). joint_idx points at
                // the next free slot; use the previous index, floored at 0.
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
            if (link_to_joint_map.contains(link_name)) {
                current_joint_idx = link_to_joint_map[link_name];
            }

            for (const tinyxml2::XMLElement *col = link->FirstChildElement("collision"); col;
                 col = col->NextSiblingElement("collision")) {

                manifold::SE3 offset = parse_origin(col->FirstChildElement("origin"));
                const tinyxml2::XMLElement *geom = col->FirstChildElement("geometry");
                if (!geom) continue;

                // TODO: Collision-geometry performance.
                // Triangle meshes are loaded directly from the URDF here. coal::distance
                // on mesh-mesh pairs uses BVH traversal and costs ~200-800 us per pair.
                // Replacing each link with a bounding primitive (e.g. a single capsule per
                // link) would reduce this to ~5 us per pair via a closed-form solve.
                const tinyxml2::XMLElement *mesh_xml = geom->FirstChildElement("mesh");
                if (mesh_xml) {
                    const char *file = mesh_xml->Attribute("filename");
                    if (!file) continue;

                    std::string file_path(file);
                    const std::string file_prefix = "file://";

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

        // Motor-armature reflected inertia for the xArm 6. The link inertial
        // YAMLs describe rigid-body inertias only; harmonic-drive rotor inertia
        // is omitted, making Lambda_rot orders of magnitude too small without
        // this correction. Values are currently only approximate estimates.
        //
        // TODO: populate joint_armature (and matching MJCF <joint armature=...>)
        // for xarm5, xarm7, lite6, uf850 to enable proper torque-mode FF tracking.
        if (dof == 6 && robot_type != "lite" && robot_type != "uf850") {
            model.joint_armature.resize(6);
            model.joint_armature << 1.0, 0.5, 0.2, 0.07, 0.03, 0.02;
        }

        return model;
    }

    [[nodiscard]] auto build_collision_model(const Model &kin_model, bool add_ground_plane)
        -> CollisionModel {
        CollisionModel col_model;

        internal::load_geometry_params(col_model, kin_model);

        // Robot-specific allowed-collision matrix (from official SRDF).
        // xArm6 only; other variants will see spurious self-collision pairs.
        // TODO: replace with SRDF parsing for general support.
        if (kin_model.dof == 6) {
            col_model.disable_collision_pair("link1_col", "link3_col");
            col_model.disable_collision_pair("link2_col", "link4_col");
            col_model.disable_collision_pair("link_base_col", "link2_col");
            col_model.disable_collision_pair("link_base_col", "link3_col");
            col_model.disable_collision_pair("link3_col", "link5_col");
            col_model.disable_collision_pair("link3_col", "link6_col");
            col_model.disable_collision_pair("link4_col", "link6_col");
        }

        if (add_ground_plane) {
            auto ground_geom = std::make_shared<coal::Halfspace>(coal::Vec3s(0, 0, 1), 0.0);

            manifold::SE3 ground_pose = manifold::SE3::Identity();
            ground_pose.r3() << 0.0, 0.0, 0.0;

            col_model.add_geometry("ground_plane", 0, ground_pose, ground_geom);
        }

        col_model.add_all_collision_pairs();

        return col_model;
    }
}  // namespace xarm_geo
