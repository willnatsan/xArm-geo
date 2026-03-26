#pragma once

#include <Eigen/Dense>
#include <memory>
#include <xarm_geo/core/manifold.h>

// --- Forward Declarations ---
namespace tinyxml2 {
    class XMLElement;
}
namespace coal {
    class CollisionGeometry;
    template <typename PolygonT> class Convex;
    struct Triangle;
}  // namespace coal

namespace xarm_geo::internal {

    // Convert BVH mesh to convex hull for collision detection
    auto make_convex(const std::shared_ptr<coal::CollisionGeometry> &geom)
        -> std::shared_ptr<coal::Convex<coal::Triangle>>;

    // Parse 3D vector from space-separated string
    auto parse_vec3(const char *str, const Eigen::Vector3d &default_val = Eigen::Vector3d::Zero())
        -> Eigen::Vector3d;

    // Parse SE(3) pose from URDF <origin> element
    auto parse_origin(const tinyxml2::XMLElement *xml) -> manifold::SE3;

}  // namespace xarm_geo::internal
