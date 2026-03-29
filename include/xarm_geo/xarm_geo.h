#pragma once

#include <xarm_geo/core/data.h>
#include <xarm_geo/core/manifold.h>
#include <xarm_geo/modelling/dynamics.h>
#include <xarm_geo/modelling/kinematics.h>
#include <xarm_geo/utils/model_builder.h>

// Lazy-Evaluation for Stateful XArm Class -> Managing Stale Data Risk w/o Redundant Re-Computation
// [Repetitive Control Loop Mitigates Issues w/ Branch Prediction Failures!]
