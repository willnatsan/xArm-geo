# Authoring Trajectories

## Overview

Trajectories in `xarm_geo` provide time-parameterised reference signals to
controllers. A trajectory is any C++ type that satisfies one of two concepts:

```
TaskTrajectory   -- SE(3) tracking reference: (pose, twist, spatial_acc)
JointTrajectory  -- Joint-space reference: (q, v, a)
```

Both concepts are checked at compile time via `static_assert`. The control
loop, validator, and simulation/hardware harnesses are all generic over these
concepts -- any conforming type is accepted without registration or inheritance.

---

## Concepts

### TaskTrajectory

```cpp
template <typename T>
concept TaskTrajectory =
    requires(const T &traj, double t, TaskTarget &target) {
        { traj.evaluate(t, target) } -> std::same_as<TrajectoryStatus>;
        { traj.duration() } -> std::convertible_to<double>;
    };
```

`evaluate(t, target)` must fill:
- `target.pose`         -- SE(3) reference pose at time t.
- `target.twist`        -- body-frame twist (expressed in the end-effector
                           frame); xi_d = J_b * q_dot at the reference.
- `target.spatial_acc`  -- body-frame spatial acceleration; d/dt(xi_d).

All spatial quantities are in the **end-effector body frame**. This is
consistent with the rest of the library (J_b, g_e = g^{-1} * g_d, etc.).

`duration()` returns the total time in seconds.
For infinite-duration trajectories (setpoints), return
`std::numeric_limits<double>::infinity()`.

Return values of `evaluate()`:

| Status            | Meaning                                             |
|-------------------|-----------------------------------------------------|
| `OK`              | target filled successfully.                         |
| `OUT_OF_DOMAIN`   | t < 0 or t > duration(). target unchanged.          |
| `NOT_INITIALISED` | Object not ready (build step not called). Unchanged. |
| `SOLVER_ERROR`    | Internal solver failed.                             |
| `ERROR`           | Catch-all.                                          |

### JointTrajectory

```cpp
template <typename T>
concept JointTrajectory =
    requires(const T &traj, double t, JointTarget &target) {
        { traj.evaluate(t, target) } -> std::same_as<TrajectoryStatus>;
        { traj.duration() } -> std::convertible_to<double>;
        { traj.dof() } -> std::convertible_to<int>;
    };
```

`dof()` returns the number of joints. Callers construct `JointTarget(traj.dof())`
before the first `evaluate()` call.

---

## Pattern 1: Inherit from AnalyticTaskTrajectory / AnalyticJointTrajectory

**The recommended pattern** for any trajectory defined by a closed-form
analytic curve.

### AnalyticTaskTrajectory (task space)

`AnalyticTaskTrajectory` (in `xarm_geo/trajectory/trajectory.h`) is an
abstract base that handles:
- Anchor composition (`anchor * SE3(local_rot, local_pos)`).
- Uniform sampling over `[0, duration]`.
- Degree-5 SE(3) B-spline fitting (`C^4` continuity).
- `evaluate()` with domain guarding (`OUT_OF_DOMAIN`, `NOT_INITIALISED`).
- `duration()` accessor.
- Spline evaluation with analytical body-twist and spatial-acceleration
  derivatives (via `smooth`).

You supply only the **geometry** via the pure virtual `sample()` method.
The virtual calls happen only during `build_spline()` at construction
(N = `num_samples`, default 100 calls). The hot-path `evaluate()` reads
from the pre-built spline with zero virtual dispatch.

#### Minimal example

```cpp
#include <cmath>
#include <utility>
#include <xarm_geo/trajectory/trajectory.h>

namespace xarm_geo::trajectories {

class CircleXY final : public AnalyticTaskTrajectory {
public:
    explicit CircleXY(const manifold::SE3 &anchor, double duration = 10.0,
                      double omega = 1.0, double radius = 0.10)
        : AnalyticTaskTrajectory(anchor, duration),
          omega_(omega), radius_(radius) {
        build_spline();   // must be last line of constructor body
    }

protected:
    [[nodiscard]] auto sample(double t) const
        -> std::pair<manifold::SO3, Eigen::Vector3d> override {
        const Eigen::Vector3d pos(radius_ * std::cos(omega_ * t),
                                  radius_ * std::sin(omega_ * t),
                                  0.0);
        return {manifold::SO3::Identity(), pos};
    }

private:
    double omega_, radius_;
};

static_assert(TaskTrajectory<CircleXY>);

}  // namespace xarm_geo::trajectories
```

#### Rules

1. **Call `build_spline()` as the last statement in your constructor body**,
   after all member variables that `sample()` reads have been initialised.
   Calling it from the base constructor would invoke `sample()` before
   derived members exist (undefined behaviour).

2. `sample()` is called at `num_samples` uniformly-spaced times in
   `[0, duration]`. It must be deterministic and not depend on control state.

3. If `sample()` needs the total duration (e.g. for a midpoint transition),
   call the inherited public `duration()` accessor inside `sample()`.

4. `num_samples` defaults to 100 and can be overridden via the third base
   constructor argument:
   ```cpp
   : AnalyticTaskTrajectory(anchor, duration, /*num_samples=*/200)
   ```

#### Orientation conventions

Return `local_rot` such that:
- `(SO3::Identity(), local_pos)` gives a pure translation relative to anchor.
- `anchor * SE3(local_rot, local_pos)` is the world-frame target pose.

For tools pointing downward (e.g. inspection probes), a roll of `pi` about
the X-axis is the canonical "tool-down" baseline:
```cpp
manifold::SO3::exp(std::numbers::pi * Eigen::Vector3d::UnitX())
```
See `FigureEight`, `WingInspection`, `PipeInspection` for worked examples.

### AnalyticJointTrajectory (joint space)

`AnalyticJointTrajectory` is the joint-space counterpart. It fits a degree-5
`Eigen::Spline` through `num_samples` configurations returned by `sample()`,
and provides analytic velocity and acceleration via chain-rule scaling of the
spline derivatives.

The interface is symmetric with `AnalyticTaskTrajectory`:
- Constructor: `(int dof, double duration, int num_samples = 100)`.
- Override `sample(double t) -> Eigen::VectorXd` returning a `VectorXd` of
  size `dof()`.
- Call `build_spline()` at the end of the derived constructor.
- `evaluate()` fills `target.q`, `target.v`, `target.a`.
- `dof()` is inherited.

#### Minimal example

```cpp
#include <cmath>
#include <xarm_geo/trajectory/trajectory.h>

// Simple two-joint sinusoidal "wave" trajectory.
class WaveJoint final : public xarm_geo::AnalyticJointTrajectory {
public:
    WaveJoint(double duration, double omega)
        : AnalyticJointTrajectory(/*dof=*/2, duration), omega_(omega) {
        build_spline();
    }

protected:
    [[nodiscard]] auto sample(double t) const -> Eigen::VectorXd override {
        Eigen::VectorXd q(2);
        q(0) = 0.5 * std::sin(omega_ * t);
        q(1) = 0.3 * std::cos(2.0 * omega_ * t);
        return q;
    }

private:
    double omega_;
};

static_assert(xarm_geo::JointTrajectory<WaveJoint>);
```

---

## Pattern 2: Freestanding class (concept-only)

Use this when the trajectory is not naturally an analytic sampled curve:
waypoint lists, splines with non-uniform knots, loaded from file, etc.

Satisfy `TaskTrajectory` or `JointTrajectory` directly. No base class
required. The concept checks the signature, not the implementation.

```cpp
class WaypointTrajectory {
public:
    WaypointTrajectory(const std::vector<manifold::SE3> &waypoints, double duration)
        : spline_(build_se3_spline<5>(waypoints, duration)), duration_(duration) {}

    [[nodiscard]] auto evaluate(double t, TaskTarget &target) const -> TrajectoryStatus {
        if (t < 0.0 || t > duration_) { return TrajectoryStatus::OUT_OF_DOMAIN; }
        manifold::SE3::Tangent vel, acc;
        target.pose = spline_(t, smooth::OptTangent<manifold::SE3>{vel},
                              smooth::OptTangent<manifold::SE3>{acc});
        target.twist = vel;
        target.spatial_acc = acc;
        return TrajectoryStatus::OK;
    }

    [[nodiscard]] auto duration() const noexcept -> double { return duration_; }

private:
    manifold::SE3::Spline<5> spline_;
    double duration_;
};

static_assert(TaskTrajectory<WaypointTrajectory>);
```

`build_se3_spline<D>(waypoints, duration)` is a free function in
`trajectory.h` that pads boundary knots for `C^{D-1}` continuity and
returns a `smooth` B-spline.

For joint-space, also expose `dof()`:
```cpp
[[nodiscard]] auto dof() const noexcept -> int { return n_joints_; }
```

---

## Pattern 3: Constant setpoint

For setpoint regulation (no time-varying reference), use the bundled adapters:

```cpp
// Task-space constant setpoint.
xarm_geo::TaskSetpointTrajectory setpoint;
setpoint.pose = desired_pose;
// duration() returns +infinity; twist and spatial_acc are zero.

// Joint-space constant setpoint.
xarm_geo::JointSetpointTrajectory joint_setpoint(model.dof);
joint_setpoint.q = desired_q;
// duration() returns +infinity; v and a are zero.
```

Both satisfy `TaskTrajectory` / `JointTrajectory` respectively. The
controller treats the zero twist / acceleration as "pure regulation" --
the feedforward term contributes nothing and the controller drives the
error to zero via the proportional term alone.

---

## Composing Trajectories

`#include <xarm_geo/trajectory/adapters.h>` (also pulled in by `xarm_geo.h`).

All adapters are header-only class templates that satisfy the same concept as
their inner trajectory. They can be nested arbitrarily.

### ConcatenatedTask / ConcatenatedJoint

Chains two or more trajectories end-to-end. Duration is the sum of all
segment durations. All segment durations must be finite and positive.

```cpp
xarm_geo::trajectories::PipeInspection phase1(anchor, 10.0);
xarm_geo::trajectories::WingInspection  phase2(anchor, 15.0);
xarm_geo::trajectories::PipeInspection phase3(anchor, 10.0);

// CTAD: type deduced automatically.
xarm_geo::ConcatenatedTask sequence{std::move(phase1),
                                    std::move(phase2),
                                    std::move(phase3)};
// sequence.duration() == 35.0

xarm_geo::TaskTarget target;
sequence.evaluate(12.5, target);  // falls in phase2 at local t=2.5
```

Joint-space:
```cpp
xarm_geo::trajectories::JointPTP leg1(q_start, q_mid, 2.0);
xarm_geo::trajectories::JointPTP leg2(q_mid,   q_end, 2.0);
xarm_geo::ConcatenatedJoint approach{std::move(leg1), std::move(leg2)};
// approach.duration() == 4.0,  approach.dof() == model.dof
```

**Seam continuity**: derivatives are **not** smoothed at segment boundaries.
If segment i ends with non-zero velocity and segment i+1 starts at rest, the
controller will see a step change in the reference twist at the junction.
Design trajectories so that seam velocities are compatible, or accept the
discontinuity in the feed-forward term.

### TimeScaledTask / TimeScaledJoint

Plays a trajectory at a different speed. Scale > 1 slows down; scale < 1
speeds up. Both duration and derivatives are corrected together.

```cpp
xarm_geo::trajectories::TiltingCircle inner(anchor, 15.0);

// Half speed: duration becomes 30 s; twist halved; spatial_acc quartered.
xarm_geo::TimeScaledTask half_speed{std::move(inner), 2.0};
// half_speed.duration() == 30.0
```

Chain rule applied by the adapter:
```
twist_out       = twist_inner       / scale
spatial_acc_out = spatial_acc_inner / scale²
```

Note: if you previously hard-coded a doubled duration to achieve half speed
(without fixing the feedforward twist), `TimeScaledTask` is the correct
replacement — it adjusts both simultaneously.

### OffsetTask

Applies a fixed SE(3) transform to every pose. Two conventions:

```cpp
// Left action: move trajectory to a different world-frame location.
// Body twist and spatial_acc are unchanged.
xarm_geo::OffsetTask shifted{std::move(inner), new_anchor};   // OffsetSide::Left (default)

// Right action: apply a tool-frame offset (e.g. extended reach, sensor offset).
// Twist and spatial_acc are pulled back via Ad_{transform^{-1}}.
xarm_geo::OffsetTask with_sensor{std::move(inner),
                                  sensor_transform,
                                  xarm_geo::OffsetSide::Right};
```

Left-action math:
```
pose_out = transform * pose_inner
twist_out, spatial_acc_out unchanged
```

Right-action math:
```
pose_out        = pose_inner * transform
twist_out       = Ad_{T^{-1}} * twist_inner
spatial_acc_out = Ad_{T^{-1}} * spatial_acc_inner
```

### ReversedTask / ReversedJoint

Plays a trajectory backwards in time:
```
t_inner = duration - t_out
```

Derivative correction (body-frame):
```
twist_out       = -twist_inner       (sign reversal under time reversal)
spatial_acc_out =  spatial_acc_inner (second derivative; sign unchanged)
```

```cpp
xarm_geo::trajectories::PipeInspection fwd(anchor, 15.0);
xarm_geo::ReversedTask rev{std::move(fwd)};
// rev.duration() == 15.0; traverses the path in reverse
```

Joint-space:
```cpp
xarm_geo::trajectories::JointPTP ptp(q_start, q_end, 3.0);
xarm_geo::ReversedJoint back{std::move(ptp)};
// back plays q_end -> q_start over 3 s
```

### Nesting adapters

Adapters compose: any adapter output is itself a valid trajectory and can be
wrapped further.

```cpp
// Slow-motion reversed inspection:
xarm_geo::TimeScaledTask slow_rev{
    xarm_geo::ReversedTask{xarm_geo::trajectories::WingInspection{anchor, 15.0}},
    2.0  // half speed
};
// slow_rev.duration() == 30.0
```

---

## Trajectory Validation

`validate_trajectory` is defined in `<xarm_geo/trajectory/validate.h>` (also
pulled in by the `xarm_geo/xarm_geo.h` umbrella). Include it directly if you
are not using the umbrella:

```cpp
#include <xarm_geo/trajectory/validate.h>
```

It accepts any `TaskTrajectory` or `JointTrajectory`.

### ValidationResult

```cpp
struct ValidationResult {
    ValidationStatus status = ValidationStatus::OK;
    double failure_time = -1.0;
    std::string reason;
};
```

`ValidationStatus` is an enum with 7 values:

| Status                        | Meaning                                        |
|-------------------------------|------------------------------------------------|
| `OK`                          | No collision or error found.                   |
| `COLLISION`                   | Collision detected at `failure_time`.          |
| `TRAJECTORY_OUT_OF_DOMAIN`    | evaluate() returned OUT_OF_DOMAIN.             |
| `TRAJECTORY_NOT_INITIALISED`  | evaluate() returned NOT_INITIALISED.           |
| `TRAJECTORY_SOLVER_ERROR`     | evaluate() returned SOLVER_ERROR.              |
| `TRAJECTORY_ERROR`            | evaluate() returned ERROR, or infinite duration.|
| `CONTROLLER_FAILED`           | controller.update() returned non-OK.           |

Check the outcome:
```cpp
if (result.status != xarm_geo::ValidationStatus::OK) {
    std::cerr << xarm_geo::to_string(result.status)
              << " at t=" << result.failure_time
              << ": " << result.reason << "\n";
}
```

### Default overload (reads duration from trajectory)

```cpp
xarm_geo::controllers::GeometricPController controller(model);
controller.gains.kp_pos.setConstant(8.0);
controller.gains.kp_rot.setConstant(8.0);
// configure identically to runtime

auto result = xarm_geo::validate_trajectory(
    model, data, col_model, col_data,
    my_traj,      // TaskTrajectory; duration read from my_traj.duration()
    q_start,
    controller);
```

The task-space validator forward-simulates closed-loop execution via
`controller.update()` at each integration step (dt = 2 ms default).
**Validation accuracy depends on the controller and gains matching runtime.**
Pass the same controller instance you will use for execution.

For joint-space (open-loop sampling):
```cpp
auto result = xarm_geo::validate_trajectory(
    model, data, col_model, col_data, joint_traj);
```

### Explicit-duration overload (for setpoints and composed sequences)

The default overload rejects infinite-duration trajectories (setpoints) with
`TRAJECTORY_ERROR`. To validate a finite window of a setpoint, or to override
the duration of any trajectory, pass an explicit duration as the last parameter
before `ValidationOptions`:

```cpp
// Validate the first 10 s of a setpoint.
auto result = xarm_geo::validate_trajectory(
    model, data, col_model, col_data,
    setpoint, q_start, controller,
    /*explicit_duration=*/10.0);
```

`ValidationOptions` lets you tune step sizes:
```cpp
xarm_geo::ValidationOptions opts;
opts.integration_dt      = 0.002;   // controller / integration step (s)
opts.collision_check_dt  = 0.05;    // collision check interval (s)
auto result = xarm_geo::validate_trajectory(..., controller, opts);
```

---

## Provided Examples

All examples are in `include/xarm_geo/examples/trajectories/`.

| File                 | Class            | Type              | Description                                            |
|----------------------|------------------|-------------------|--------------------------------------------------------|
| `figure_eight.h`     | `FigureEight`    | Task, analytic    | Lissajous figure-eight with oscillating orientation.   |
| `tilting_circle.h`   | `TiltingCircle`  | Task, analytic    | Circular orbit with min-jerk tilt transition.          |
| `pipe_inspection.h`  | `PipeInspection` | Task, analytic    | Arc in YZ plane; pitch tracks pipe wall normal.        |
| `wing_inspection.h`  | `WingInspection` | Task, analytic    | Parabolic surface scan; pitch tracks curvature.        |
| `inner_cavity_scan.h`| `InnerCavityScan`| Task, analytic    | Linear probe + multi-axis orientation sweep.           |
| `waypoint.h`         | `Waypoint`       | Task, freestanding| B-spline through explicit SE(3) waypoints.             |
| `task_ptp.h`         | `TaskPTP`        | Task, freestanding| Minimum-jerk SE(3) geodesic between two poses; exact rest-to-rest endpoints; analytic body twist and spatial acceleration. |
| `joint_ptp.h`        | `JointPTP`       | Joint, freestanding| Minimum-jerk point-to-point joint motion.             |
| `large_orientation_step.h` | `LargeOrientationStep` | Task, freestanding | Discrete SE(3) setpoint step: zero translational, large pure-rotation error (default 175° about (1,1,1)/√3) near the SO(3) trace-gradient saddle set. |

---

## Common Pitfalls

### Calling build_spline() before derived members are initialised

`sample()` is called from `build_spline()` via virtual dispatch. Calling
`build_spline()` from the **base constructor** invokes `sample()` before the
derived object exists. Always call it from the **derived constructor body**,
as the last statement, after all members `sample()` depends on are set.

### Forgetting build_spline() entirely

`evaluate()` returns `TrajectoryStatus::NOT_INITIALISED` if `build_spline()`
was never called. This surfaces as `ValidationStatus::TRAJECTORY_NOT_INITIALISED`
in the validator. The `static_assert(TaskTrajectory<YourClass>)` does not catch
this -- it only checks method signatures.

### Mixing body-frame and world-frame quantities

`target.twist` and `target.spatial_acc` must be expressed in the
**end-effector body frame**. Using space-frame (world-frame) quantities
compiles but produces wrong feedforward commands. The `AnalyticTaskTrajectory`
base correctly provides body-frame derivatives from the B-spline via `smooth`.
If you compute derivatives manually in Pattern 2, verify the frame.

### Passing a setpoint to the default validator

`validate_trajectory` reads `trajectory.duration()`. If the trajectory has
`duration() == +infinity` (e.g. a setpoint), the default overload returns
`ValidationStatus::TRAJECTORY_ERROR` immediately with reason
`"infinite_duration_use_explicit_duration_overload"`. Use the explicit-duration
overload to validate a finite window:
```cpp
validate_trajectory(..., setpoint, q_start, controller, /*explicit_duration=*/10.0);
```

### Discontinuous derivatives at concatenation seams

`ConcatenatedTask` and `ConcatenatedJoint` do not smooth derivatives at segment
boundaries. A step change in `target.twist` at a seam is indistinguishable from
a sudden reference change and will produce a large (but bounded) control effort
spike. Ensure that the ending velocity of segment i matches the starting
velocity of segment i+1 if smooth behaviour at seams is required.

### TimeScaled derivatives vs. just changing the duration

Doubling a trajectory's constructor `duration` argument is **not** the same as
`TimeScaledTask{inner, 2.0}`. The former stretches the B-spline domain but
leaves `target.twist` unchanged (incorrect for half-speed playback). The latter
correctly halves the twist and quarters the spatial acceleration.
