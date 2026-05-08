// Compile-only static-assert test for example controllers and trajectories.
// Add new example headers here when you create them.

// --- Controllers ---
#include <xarm_geo/examples/controllers/custom_controller.h>
#include <xarm_geo/examples/controllers/geometric_p_controller.h>
#include <xarm_geo/examples/controllers/geometric_pd_controller.h>
#include <xarm_geo/examples/controllers/geometric_pi_controller.h>
#include <xarm_geo/examples/controllers/geometric_pid_controller.h>
#include <xarm_geo/examples/controllers/joint_p_controller.h>
#include <xarm_geo/examples/controllers/joint_pd_controller.h>

// --- Trajectories ---
#include <xarm_geo/examples/trajectories/figure_eight.h>
#include <xarm_geo/examples/trajectories/inner_cavity_scan.h>
#include <xarm_geo/examples/trajectories/joint_ptp.h>
#include <xarm_geo/examples/trajectories/pipe_inspection.h>
#include <xarm_geo/examples/trajectories/tilting_circle.h>
#include <xarm_geo/examples/trajectories/waypoint.h>
#include <xarm_geo/examples/trajectories/wing_inspection.h>

auto main() -> int { return 0; }
