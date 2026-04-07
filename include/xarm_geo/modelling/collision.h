#pragma once

#include <memory>
#include <string>
#include <vector>

#include <coal/collision_data.h>
#include <coal/collision_object.h>
#include <coal/shape/geometric_shapes.h>

#include <xarm_geo/core/system.h>
#include <xarm_geo_config.h>

namespace xarm_geo {

    // --- Collision Structs ---

    struct CollisionModel {

        // Description of Geometry Object (Robot / Environment)
        struct GeomObject {
            std::string name;
            size_t parent_joint;  // 0: Environment, 1...N: Robot Links
            manifold::SE3 pose;   // Relative to  `parent_joint`
            std::shared_ptr<coal::CollisionGeometry> geom;
        };

        // Indices of 2 Objects that can Collide
        struct CollisionPair {
            size_t obj1_idx;
            size_t obj2_idx;
        };

        std::vector<GeomObject> geometries;
        std::vector<CollisionPair> collision_pairs;

        // --- Setup Helpers ---

        // Returns the index of the newly added geometry
        auto add_geometry(const std::string &name, size_t parent_joint,
                          const manifold::SE3 &placement,
                          std::shared_ptr<coal::CollisionGeometry> geom) -> size_t;

        // Populates collision_pairs while ignoring adjacent links or world-vs-world checks.
        void add_all_collision_pairs();
    };

    struct CollisionData {
        std::vector<manifold::SE3> geom_poses;  // In Space Frame

        std::vector<coal::CollisionObject> collision_objects;
        std::vector<coal::CollisionRequest> collision_requests;
        std::vector<coal::CollisionResult> collision_results;

        // --- Pre-Allocation ---
        explicit CollisionData(const CollisionModel &col_model);
    };

    // --- Collision Algorithms ---

    void update_geometry_poses(const Model &kin_model, const Data &kin_data,
                               const CollisionModel &col_model, CollisionData &col_data);

    auto compute_collisions(const CollisionModel &col_model, CollisionData &col_data) -> bool;
}  // namespace xarm_geo
