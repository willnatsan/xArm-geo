#pragma once

#include <string>

#include <xarm_geo/core/data.h>
#include <xarm_geo/modelling/collision.h>

namespace xarm_geo {
    [[nodiscard]] auto build_model(int dof, const std::string &robot_sn,
                                   const std::string &robot_type = "", bool modell1300 = false)
        -> Model;
    [[nodiscard]] auto build_collision_model(const Model &kin_model, bool add_ground_plane = false)
        -> CollisionModel;
}  // namespace xarm_geo
