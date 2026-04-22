#include <iostream>

#include <coal/collision.h>
#include <coal/distance.h>

#include <xarm_geo/modelling/collision.h>

namespace xarm_geo {

    // --- Collision Helpers & Constructor ---

    auto CollisionModel::add_geometry(const std::string &name, size_t parent_joint,
                                      const manifold::SE3 &placement,
                                      std::shared_ptr<coal::CollisionGeometry> geom) -> size_t {

        geometries.push_back({.name = name,
                              .parent_joint = parent_joint,
                              .pose = placement,
                              .geom = std::move(geom)});

        return geometries.size() - 1;
    }

    void CollisionModel::add_all_collision_pairs() {
        collision_pairs.clear();

        for (size_t i = 0; i < geometries.size(); ++i) {
            for (size_t j = i + 1; j < geometries.size(); ++j) {
                const auto &g1 = geometries[i];
                const auto &g2 = geometries[j];

                // Check the Allowed Collision Matrix
                if (allowed_collision_pairs.contains({g1.name, g2.name}) ||
                    allowed_collision_pairs.contains({g2.name, g1.name})) {
                    continue;
                }

                int joint_i = static_cast<int>(g1.parent_joint);
                int joint_j = static_cast<int>(g2.parent_joint);

                // Ignore Static Environment
                if (joint_i == 0 && joint_j == 0) { continue; }

                // Ignore Adjacent Links
                if (std::abs(joint_i - joint_j) <= 1) { continue; }

                collision_pairs.push_back({.obj1_idx = i, .obj2_idx = j});
            }
        }
    }

    void CollisionModel::disable_collision_pair(const std::string &link1,
                                                const std::string &link2) {
        allowed_collision_pairs.insert({link1, link2});
    }

    CollisionData::CollisionData(const CollisionModel &col_model) {
        size_t num_geoms = col_model.geometries.size();
        size_t num_pairs = col_model.collision_pairs.size();

        geom_poses.resize(num_geoms, manifold::SE3::Identity());

        // Initialise Collision Objects
        for (const auto &obj : col_model.geometries) { collision_objects.emplace_back(obj.geom); }

        // Pre-Allocate Collision Requests & Results for Every Collision Pair
        collision_requests.resize(num_pairs);
        collision_results.resize(num_pairs);

        // Pre-Allocate Distance Requests & Results for Every Collision Pair
        distance_requests.resize(num_pairs);
        distance_results.resize(num_pairs);
    }

    // --- Collision Algorithms ---

    void update_geometry_poses(const Model &kin_model, const Data &kin_data,
                               const CollisionModel &col_model, CollisionData &col_data) {
        for (size_t i = 0; i < col_model.geometries.size(); ++i) {
            const auto &obj = col_model.geometries[i];

            // If parent is 0 (World), kin_data.pose_tree[0] should be Identity.
            col_data.geom_poses[i] = kin_data.pose_tree[obj.parent_joint] * obj.pose;

            // Update the Collision Objects
            coal::Transform3s transform(col_data.geom_poses[i].so3().matrix(),
                                        col_data.geom_poses[i].r3());
            col_data.collision_objects[i].setTransform(transform);
        }
    }

    auto compute_collisions(const CollisionModel &col_model, CollisionData &col_data) -> bool {
        bool is_colliding = false;

        for (size_t i = 0; i < col_model.collision_pairs.size(); ++i) {
            const auto &pair = col_model.collision_pairs[i];

            // Clear previous Collision Result for this Collision Pair
            col_data.collision_results[i].clear();

            // Collision Detection
            coal::collide(&col_data.collision_objects[pair.obj1_idx],
                          &col_data.collision_objects[pair.obj2_idx],
                          col_data.collision_requests[i], col_data.collision_results[i]);
            if (col_data.collision_results[i].isCollision()) {
                std::cout << "[COLLISION HIT] " << col_model.geometries[pair.obj1_idx].name
                          << " and " << col_model.geometries[pair.obj2_idx].name << "\n";
                is_colliding = true;
            }
        }

        return is_colliding;
    }

    // --- Distance Algorithms ---

    auto compute_min_distance(const CollisionModel &col_model, CollisionData &col_data)
        -> DistanceResult {

        DistanceResult result;

        for (size_t i = 0; i < col_model.collision_pairs.size(); ++i) {
            const auto &pair = col_model.collision_pairs[i];

            // Clear previous Distance Result for this Collision Pair
            col_data.distance_results[i].clear();

            // Distance Computation
            coal::distance(&col_data.collision_objects[pair.obj1_idx],
                           &col_data.collision_objects[pair.obj2_idx],
                           col_data.distance_requests[i], col_data.distance_results[i]);

            double dist = col_data.distance_results[i].min_distance;

            if (dist < result.min_distance) {
                result.min_distance = dist;
                result.closest_pair_idx = i;
                result.nearest_point1 = col_data.distance_results[i].nearest_points[0];
                result.nearest_point2 = col_data.distance_results[i].nearest_points[1];
            }
        }

        return result;
    }

}  // namespace xarm_geo
