#pragma once

#include <xarm_geo/control/controller.h>
#include <xarm_geo/core/manifold.h>
#include <xarm_geo/core/system.h>
#include <xarm_geo/diagnostics/logger.h>
#include <xarm_geo/interfaces/hardware.h>
#include <xarm_geo/interfaces/simulation.h>
#include <xarm_geo/modelling/dynamics.h>
#include <xarm_geo/modelling/kinematics.h>
#include <xarm_geo/trajectory/adapters.h>
#include <xarm_geo/trajectory/trajectory.h>
#include <xarm_geo/trajectory/validate.h>
#include <xarm_geo/utils/model_builder.h>

// TODO: Stateful XArm class that uses lazy evaluation to manage stale-data
// risk without redundant re-computation in tight control loops.
