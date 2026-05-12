// Compile-only static-assert test for example controllers and trajectories.
// Add new example headers here when you create them.

// --- Controllers ---
#include <xarm_geo/examples/controllers/custom_controller.h>
#include <xarm_geo/examples/controllers/euclidean_p_controller.h>
#include <xarm_geo/examples/controllers/euclidean_pd_controller.h>
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
#include <xarm_geo/examples/trajectories/large_orientation_step.h>
#include <xarm_geo/examples/trajectories/pipe_inspection.h>
#include <xarm_geo/examples/trajectories/tilting_circle.h>
#include <xarm_geo/examples/trajectories/waypoint.h>
#include <xarm_geo/examples/trajectories/wing_inspection.h>

// --- Trajectory Adapters ---
#include <xarm_geo/trajectory/adapters.h>

// --- Adapter concept verifications ---
//
// Each static_assert confirms that a concrete instantiation of an adapter
// satisfies the appropriate concept. One representative example trajectory is
// used per adapter type; the adapter template mechanics are the same for all.

// ConcatenatedTask: two and three segment variants.
using ConcatTask2 = xarm_geo::ConcatenatedTask<xarm_geo::trajectories::PipeInspection,
                                               xarm_geo::trajectories::WingInspection>;
static_assert(xarm_geo::TaskTrajectory<ConcatTask2>);

using ConcatTask3 = xarm_geo::ConcatenatedTask<xarm_geo::trajectories::FigureEight,
                                               xarm_geo::trajectories::TiltingCircle,
                                               xarm_geo::trajectories::InnerCavityScan>;
static_assert(xarm_geo::TaskTrajectory<ConcatTask3>);

// ConcatenatedJoint: two segment variant.
using ConcatJoint2 =
    xarm_geo::ConcatenatedJoint<xarm_geo::trajectories::JointPTP, xarm_geo::trajectories::JointPTP>;
static_assert(xarm_geo::JointTrajectory<ConcatJoint2>);

// TimeScaledTask.
using TimeScaledT = xarm_geo::TimeScaledTask<xarm_geo::trajectories::FigureEight>;
static_assert(xarm_geo::TaskTrajectory<TimeScaledT>);

// TimeScaledJoint.
using TimeScaledJ = xarm_geo::TimeScaledJoint<xarm_geo::trajectories::JointPTP>;
static_assert(xarm_geo::JointTrajectory<TimeScaledJ>);

// OffsetTask (left and right sides produce the same type).
using OffsetT = xarm_geo::OffsetTask<xarm_geo::trajectories::TiltingCircle>;
static_assert(xarm_geo::TaskTrajectory<OffsetT>);

// ReversedTask.
using ReversedT = xarm_geo::ReversedTask<xarm_geo::trajectories::PipeInspection>;
static_assert(xarm_geo::TaskTrajectory<ReversedT>);

// ReversedJoint.
using ReversedJ = xarm_geo::ReversedJoint<xarm_geo::trajectories::JointPTP>;
static_assert(xarm_geo::JointTrajectory<ReversedJ>);

// Nested composition: TimeScaled wrapping a Concatenated.
using NestedT =
    xarm_geo::TimeScaledTask<xarm_geo::ConcatenatedTask<xarm_geo::trajectories::PipeInspection,
                                                        xarm_geo::trajectories::WingInspection>>;
static_assert(xarm_geo::TaskTrajectory<NestedT>);

// Reversed wrapping an Offset.
using NestedT2 = xarm_geo::ReversedTask<xarm_geo::OffsetTask<xarm_geo::trajectories::FigureEight>>;
static_assert(xarm_geo::TaskTrajectory<NestedT2>);

auto main() -> int { return 0; }
