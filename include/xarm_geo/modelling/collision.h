#pragma once

#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Dense>
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
        std::set<std::pair<std::string, std::string>> allowed_collision_pairs;

        // --- Setup Helpers ---

        // Returns the index of the newly added geometry
        auto add_geometry(const std::string &name, size_t parent_joint,
                          const manifold::SE3 &placement,
                          std::shared_ptr<coal::CollisionGeometry> geom) -> size_t;
        void add_all_collision_pairs();
        void disable_collision_pair(const std::string &link1, const std::string &link2);
    };

    struct CollisionData {
        std::vector<manifold::SE3> geom_poses;  // In Space Frame

        std::vector<coal::CollisionObject> collision_objects;
        std::vector<coal::CollisionRequest> collision_requests;
        std::vector<coal::CollisionResult> collision_results;

        std::vector<coal::DistanceRequest> distance_requests;
        std::vector<coal::DistanceResult> distance_results;

        // --- Pre-Allocation ---
        explicit CollisionData(const CollisionModel &col_model);
    };

    // --- Collision Algorithms ---

    void update_geometry_poses(const Model &kin_model, const Data &kin_data,
                               const CollisionModel &col_model, CollisionData &col_data);

    auto compute_collisions(const CollisionModel &col_model, CollisionData &col_data) -> bool;

    // --- Distance Algorithms ---

    struct DistanceResult {
        double min_distance = std::numeric_limits<double>::max();
        size_t closest_pair_idx = 0;                               // Index into collision_pairs
        Eigen::Vector3d nearest_point1 = Eigen::Vector3d::Zero();  // Witness point on obj1
        Eigen::Vector3d nearest_point2 = Eigen::Vector3d::Zero();  // Witness point on obj2
    };

    // Note: Requires update_geometry_poses() to have been called first.
    //
    // If activation_distance > 0, pairs whose world-AABB separation already
    // exceeds activation_distance are skipped: coal::distance is not called and
    // the per-pair DistanceResult is written with min_distance = +inf and zeroed
    // nearest_points. This is a conservative early-out (AABB distance is a
    // lower bound on true geometry distance), so no valid close pairs are missed.
    // Barriers treat inf distance as trivially safe and skip their Jacobian row.
    //
    // Pass activation_distance = 0.0 (the default) to run all pairs unconditionally.
    auto compute_min_distance(const CollisionModel &col_model, CollisionData &col_data,
                              double activation_distance = 0.0) -> DistanceResult;

}  // namespace xarm_geo
