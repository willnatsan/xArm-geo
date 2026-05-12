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

// NOTE: Revisit a full stateful XArm facade (owning Model + Data + collision +
// interface, with high-level verbs like move_home() / execute_trajectory())
// only if the library grows a non-research audience -- e.g. teaching,
// scripted demos, or Python bindings where method-chaining ergonomics
// matter. At that point, design it as an additive layer over the existing
// free-function API, not a replacement.
