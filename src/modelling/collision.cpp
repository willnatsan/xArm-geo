#include <coal/collision.h>

#include <xarm_geo/modelling/collision.h>

namespace xarm_geo {

    // --- Collision Helpers & Constructor ---

    auto CollisionModel::add_geometry(const std::string &name, size_t parent_joint,
                                      const manifold::SE3 &placement,
                                      std::shared_ptr<coal::CollisionGeometry> geom) -> size_t {
        geometries.push_back({name, parent_joint, placement, std::move(geom)});
        return geometries.size() - 1;
    }

    void CollisionModel::add_all_collision_pairs() {
        collision_pairs.clear();
        for (size_t i = 0; i < geometries.size(); ++i) {
            for (size_t j = i + 1; j < geometries.size(); ++j) {
                int joint_i = static_cast<int>(geometries[i].parent_joint);
                int joint_j = static_cast<int>(geometries[j].parent_joint);

                // Ignore if both are static environment objects
                if (joint_i == 0 && joint_j == 0) continue;

                // Ignore adjacent robot links
                if (std::abs(joint_i - joint_j) <= 1) continue;

                collision_pairs.push_back({i, j});
            }
        }
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
    }

    // --- Collision Algorithms ---

    void update_geometry_poses(const Model &kin_model, const Data &kin_data,
                               const CollisionModel &col_model, CollisionData &col_data) {
        for (size_t i = 0; i < col_model.geometries.size(); ++i) {
            const auto &obj = col_model.geometries[i];

            // If parent is 0 (World), kin_data.pose_tree[0] should be Identity.
            col_data.geom_poses[i] = kin_data.pose_tree[obj.parent_joint] * obj.pose;

            // Update the dataful Collision Objects
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
            if (col_data.collision_results[i].isCollision()) { is_colliding = true; }
        }

        return is_colliding;
    }

}  // namespace xarm_geo
