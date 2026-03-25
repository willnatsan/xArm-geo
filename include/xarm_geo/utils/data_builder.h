#pragma once

#include <string>

#include <xarm_geo/core/data.h>

namespace xarm_geo {
    [[nodiscard]] auto build_model(int dof, const std::string &robot_sn,
                                   const std::string &robot_type = "", bool modell1300 = false)
        -> Model;
}  // namespace xarm_geo
