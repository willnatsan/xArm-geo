#include <sstream>

#include <coal/mesh_loader/loader.h>
#include <coal/serialization/BVH_model.h>
#include <coal/shape/convex.h>
#include <tinyxml2.h>

#include <xarm_geo/utils/parsing_utils.h>

namespace xarm_geo::internal {

    auto make_convex(const std::shared_ptr<coal::CollisionGeometry> &geom)
        -> std::shared_ptr<coal::Convex<coal::Triangle>> {
        const auto mesh = std::dynamic_pointer_cast<coal::BVHModel<coal::OBBRSS>>(geom);
        if (!mesh) { return nullptr; }

        auto convex = std::make_shared<coal::Convex<coal::Triangle>>(
            mesh->vertices,      // std::shared_ptr<std::vector<coal::Vec3s>>
            mesh->num_vertices,  // unsigned int
            mesh->tri_indices,   // std::shared_ptr<std::vector<coal::Triangle>>
            mesh->num_tris       // unsigned int
        );

        return convex;
    }

    auto parse_vec3(const char *str, const Eigen::Vector3d &default_val) -> Eigen::Vector3d {
        if (!str) { return default_val; }
        std::stringstream ss(str);

        double x;
        double y;
        double z;

        ss >> x >> y >> z;
        return {x, y, z};
    }

    auto parse_origin(const tinyxml2::XMLElement *xml) -> manifold::SE3 {
        if (!xml) { return manifold::SE3::Identity(); }

        const Eigen::Vector3d xyz = parse_vec3(xml->Attribute("xyz"));
        const Eigen::Vector3d rpy = parse_vec3(xml->Attribute("rpy"));

        // Build rotation from RPY (ZYX Euler angles)
        manifold::SO3 rotation = manifold::rpy_to_SO3(rpy.x(), rpy.y(), rpy.z());
        return {rotation, xyz};
    }

}  // namespace xarm_geo::internal
