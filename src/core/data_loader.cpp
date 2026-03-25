#include <tinyxml2.h>
#include <yaml-cpp/yaml.h>

#include <xarm_geo/core/data_loader.h>
#include <xarm_geo/utils/data_config.h>

namespace xarm_geo {
    [[nodiscard]] auto build_model(int dof, const std::string &robot_sn,
                                   const std::string &robot_type, bool modell1300) -> Model {
        Model model;
        model.dof
    }
}  // namespace xarm_geo
