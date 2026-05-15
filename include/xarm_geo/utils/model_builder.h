#pragma once

#include <string>

#include <mujoco/mujoco.h>

#include <xarm_geo/modelling/collision.h>

namespace xarm_geo {
    [[nodiscard]] auto build_model(int dof, const std::string &robot_sn,
                                   const std::string &robot_type = "", bool modell1300 = false)
        -> Model;
    [[nodiscard]] auto build_collision_model(const Model &kin_model, bool add_ground_plane = false)
        -> CollisionModel;

    // Mirror physical parameters from xarm_geo::Model into MuJoCo's mjModel for
    // fields that the MJCF cannot express (e.g. per-joint motor armature). Calls
    // mj_setConst() after writing raw fields so that all derived MuJoCo quantities
    // are recomputed and remain consistent with the updated parameters.
    void sync_model_to_mujoco(const Model &model, mjModel *mj_model, mjData *mj_data);
}  // namespace xarm_geo
