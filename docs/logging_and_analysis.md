# Data Logging & Analysis

## Overview

`xarm_geo` ships a first-class data pipeline for offline evaluation of
controller trials. The split is clean:

- **C++ side (`xarm_geo::diagnostics`)**: `DataLogger` records a wide,
  schema-defined CSV plus a JSON sidecar to disk. It is RAM-buffered and
  zero-I/O inside the control loop; the file is written only after the
  trial ends.

- **Python side (`xarm_geo_analysis`)**: a standalone package that loads
  the CSV/sidecar pair, exposes typed array views, computes the full set
  of evaluation metrics, generates publication-ready plots, and prints or
  writes comparison tables.

The contract between the two sides is simple: one trial = one
`<name>.csv` + one `<name>.csv.meta.json` written by `DataLogger`.
`xarm_geo_analysis` reads those two files and needs nothing else.

This module is for **offline post-hoc analysis** of logged trials. For
online settling detection within a running control loop, see
`xarm_geo::ConvergenceMonitor` (`include/xarm_geo/control/monitor.h`).

---

## Workflow at a Glance

```
┌─────────────────────────────────┐
│  C++ test binary (sim/hardware) │
│                                 │
│  DataLogger logger(model, cfg); │
│  // ... control loop ...        │
│  logger.log(sample);            │
│  // destructor flushes to disk  │
└───────────────┬─────────────────┘
                │
                ▼
        tests/results/
          trial_name.csv
          trial_name.csv.meta.json
                │
                ▼
┌─────────────────────────────────┐
│  xarm-geo-analyse / Python API  │
│                                 │
│  Trial.load("trial_name.csv")   │
│  translational_rmse(trial)      │
│  plot_tracking_errors(trial)    │
│  summarise_experiment(exp)      │
└─────────────────────────────────┘
```

Typical sequence:

1. Run a logged C++ trial: `pixi run sim --log true [options]`.
2. Inspect it: `pixi run analyse plot trial tests/results/<name>.csv`.
3. Run a sweep; compare: `pixi run report tests/results/`.
4. Save figures for a paper: add `--save-dir paper/figures/`.

---

## C++ Side: Emitting Trials

### First-Time Setup

After cloning or pulling the new module, run `pixi install` once to
register the Python package into the pixi-managed environment:

```bash
pixi install
```

Verify the Python side is importable:

```bash
pixi run python -c "import xarm_geo_analysis; print('OK')"
pixi run analyse --help
pixi run test-analysis
```

All three should succeed before you attempt to load any trial CSV.

### Quick Start

The five-line pattern used in `simulation_test.cpp` and
`hardware_test.cpp`:

```cpp
#include <optional>
#include <xarm_geo/diagnostics/logger.h>
#include <xarm_geo/examples/controllers/geometric_p_controller.h>
#include <xarm_geo/examples/trajectories/pipe_inspection.h>

// --- Before the control loop ---

const std::string trial_name = xarm_geo::diagnostics::make_trial_name(
    "sim",
    xarm_geo::controllers::GeometricPController::kName,
    xarm_geo::trajectories::PipeInspection::kName,
    /*constraint=*/controller.constraint_aware,
    /*feedforward=*/controller.use_feedforward);

std::optional<xarm_geo::diagnostics::DataLogger> logger;
if (log_data) {
    logger.emplace(model, xarm_geo::diagnostics::DataLogger::Config{
        .output_path = "tests/results/" + trial_name + ".csv",
        .trial_name  = trial_name,
    });
}

// --- Inside the control loop ---

const xarm_geo::ControllerStatus ctrl_status =
    controller.update(model, data, ctx, control_target);

if (logger) {
    xarm_geo::diagnostics::LogSample s;
    xarm_geo::diagnostics::fill_task_sample(s, t, tick, state, task_target, data);
    s.controller_status = static_cast<std::uint8_t>(ctrl_status);
    xarm_geo::diagnostics::fill_velocity_diagnostics(s, controller);
    logger->log(s);
}

// logger goes out of scope here; destructor flushes CSV + sidecar to disk.
// (Or call logger.reset() explicitly to flush before a subsequent phase.)
```

For torque-mode trials, replace `fill_velocity_diagnostics` with
`fill_torque_diagnostics`.

### LogSample Schema

Each call to `log()` records one row with the following column groups.
Quaternion conventions: `qx, qy, qz, qw` (scalar last). No Euler angles
are used anywhere.

| Group | Columns | Present when |
|---|---|---|
| Time | `t`, `tick` | Always |
| Joint state | `q.0`..`q.{N-1}`, `v.0`..`v.{N-1}`, `tau_measured.0`..`tau_measured.{N-1}` | Always |
| Joint references | `q_ref.0`..`a_ref.{N-1}` | Joint-space phases only; blank in task-space phases |
| Task actual | `ee_actual.x/y/z`, `ee_actual.qx/qy/qz/qw`, `ee_twist_actual.vx/vy/vz/wx/wy/wz` | Always (identity pose if not filled) |
| Task target | `ee_target.x/y/z`, `ee_target.qx/qy/qz/qw`, `ee_twist_target.vx/vy/vz/wx/wy/wz` | Always |
| Torque triplet | `tau_ctrl.i`, `tau_des.i`, `tau_safe.i` | Dynamic (torque-mode) controllers only; blank in kinematic trials |
| Velocity triplet | `v_ctrl.i`, `v_des.i`, `v_safe.i` | Kinematic (velocity-mode) controllers only; blank in torque-mode trials |
| Controller status | `controller_status`, `asif_status`, `asif_invoked`, `asif_modified`, `optik_status`, `optik_invoked`, `optik_modified` | Always; 255 (0xFF) means "not invoked" for status bytes |

The torque triplet semantics:

```
tau_ctrl  -- raw hook output, before bias-force compensation.
tau_des   -- post bias-compensation; the command without ASIF.
tau_safe  -- post ASIF certification; == tau_des when ASIF is off.
```

The velocity triplet semantics:

```
v_ctrl  -- raw hook output, before any safety-layer shaping.
v_des   -- == v_ctrl (no bias-compensation equivalent for kinematic bases).
v_safe  -- post OptIK / direction-preserving velocity-limit rescale.
```

### Sample Assembly Helpers

Four free functions in `xarm_geo/diagnostics/logger.h` fill the standard
column groups from library types. Call them in sequence immediately after
`update()`; each touches only its own group and leaves the rest unchanged.

```cpp
// State side: fills t, tick, q, v, tau_measured, ee_actual/target, ee_twist.
// Precondition: compute_jacobians() has been called (guaranteed by every
// task-controller update path).
void fill_task_sample(LogSample &s, double t, std::int64_t tick,
                      const JointState &fb, const TaskTarget &ref,
                      const Data &data) noexcept;

// State side: fills t, tick, q, v, tau_measured, q_ref, v_ref, a_ref.
void fill_joint_sample(LogSample &s, double t, std::int64_t tick,
                       const JointState &fb, const JointTarget &ref) noexcept;

// Controller side (dynamic bases): fills tau_ctrl/des/safe + ASIF diagnostics.
void fill_torque_diagnostics(LogSample &s,
                              const DynamicTaskControllerBase &c) noexcept;
void fill_torque_diagnostics(LogSample &s,
                              const DynamicJointControllerBase &c) noexcept;

// Controller side (kinematic bases): fills v_ctrl/des/safe + OptIK diagnostics.
void fill_velocity_diagnostics(LogSample &s,
                                const KinematicTaskControllerBase &c) noexcept;
void fill_velocity_diagnostics(LogSample &s,
                                const KinematicJointControllerBase &c) noexcept;
```

Typical combination for a kinematic task-space trial:

```cpp
xarm_geo::diagnostics::LogSample s;
xarm_geo::diagnostics::fill_task_sample(s, t, tick, state, task_target, data);
s.controller_status = static_cast<std::uint8_t>(ctrl_status);
xarm_geo::diagnostics::fill_velocity_diagnostics(s, p_controller);
logger->log(s);
```

Typical combination for a dynamic (torque-mode) task-space trial:

```cpp
xarm_geo::diagnostics::LogSample s;
xarm_geo::diagnostics::fill_task_sample(s, t, tick, state, task_target, data);
s.controller_status = static_cast<std::uint8_t>(ctrl_status);
xarm_geo::diagnostics::fill_torque_diagnostics(s, pd_controller);
logger->log(s);
```

### Trial Naming

`make_trial_name` composes a self-describing filename stem:

```
{sim|hardware}_{ControllerClass}_{TrajectoryClass}[_safe][_ff|_noff]
```

Examples:

```
sim_GeometricPDController_PipeInspection_safe_ff
sim_EuclideanPController_FigureEight_ff
hardware_GeometricPController_TiltingCircle_safe_ff
```

The `_safe` tag appears when `constraint_aware` is true, covering ASIF in
torque-mode trials, OptIK in kinematic task-space trials, and the
direction-preserving velocity rescale in kinematic joint-space trials. The
trailing `_ff` / `_noff` reflects the controller's `use_feedforward` flag.

Controller and trajectory names come from `kName`:

```cpp
// Reads the kName of the active controller and trajectory.
const std::string trial_name = xarm_geo::diagnostics::make_trial_name(
    "sim",
    xarm_geo::controllers::GeometricPDController::kName,
    xarm_geo::trajectories::TiltingCircle::kName,
    /*constraint=*/pd_controller.constraint_aware,
    /*feedforward=*/pd_controller.use_feedforward);
```

All example controllers and trajectories already declare `kName`. For a
user-defined class, add one line to the class body:

```cpp
class MyController final : public DynamicTaskControllerBase {
public:
    static constexpr std::string_view kName = "MyController";
    // ...
};
```

An optional `suffix` argument appends a custom tag after `_ff`:

```cpp
xarm_geo::diagnostics::make_trial_name("sim", ctrl, traj, true, true, "run2");
// -> "sim_GeometricPController_PipeInspection_safe_ff_run2"
```

### Controller Diagnostics Surface

All four controller base classes expose the last tick's triplet via
`last_tick_diagnostics()`. The `fill_*` helpers call this internally, but
you can also read it directly:

```cpp
// Dynamic bases return DynamicTickDiagnostics.
const auto &d = pd_controller.last_tick_diagnostics();
// d.tau_ctrl, d.tau_des, d.tau_safe
// d.asif_invoked, d.asif_modified, d.asif_status

// Kinematic bases return KinematicTickDiagnostics.
const auto &k = p_controller.last_tick_diagnostics();
// k.v_ctrl, k.v_des, k.v_safe
// k.optik_invoked, k.optik_modified, k.optik_status
```

Note on `KinematicJointControllerBase`: `optik_invoked` is always
`false` (there is no IK layer in the joint-space kinematic base).
`optik_modified` is repurposed to flag whether the direction-preserving
velocity-limit rescale clipped the command this tick. Conflating these
two mechanisms under one field keeps the Python schema uniform; if a
further kinematic safety layer is ever added, a dedicated field would be
warranted.

### Decimation and Warm-Up

`DataLogger::Config` exposes two optional knobs:

```cpp
xarm_geo::diagnostics::DataLogger::Config cfg{
    .output_path    = "tests/results/trial.csv",
    .trial_name     = trial_name,
    .reserve_samples = 16000,  // pre-allocate rows (default 16000)
    .decimation     = 5,       // log every 5th call; 1 = log all (default)
    .skip_first     = 250,     // drop the first 250 calls as warm-up (default 0)
};
```

`decimation` is useful for long hardware sessions where every-tick
logging at 500 Hz over several minutes would produce impractically large
files. The Python loader reads the actual `t` column rather than assuming
a fixed sample interval, so all integration-based metrics (effort
integrals, intervention integrals, settling time) remain correct for
decimated logs without any extra configuration.

`skip_first` discards an initial transient period (e.g. the first few
hundred ticks during approach) so the metric window starts at steady-state
execution.

---

## Python Side: Analysing Trials

### CLI Sub-Commands

The `xarm-geo-analyse` entry point (also reachable as
`pixi run analyse`) exposes four sub-commands:

**`plot trial <csv> [--save-dir DIR] [--format pdf|png] [--dpi N]`**

Opens (or saves) four diagnostic figures for one trial:

1. Three-panel tracking-error time series: translational (mm),
   rotational geodesic (deg), and weighted Riemannian SE(3) error, each
   with its settling band marked.
2. Two-panel settling visualisation: shaded ±band, vertical settling-time
   annotation per axis.
3. 3-D EE path with quaternion-driven orientation triads (actual path
   + target path overlaid).
4. Torque triplet (`tau_ctrl` / `tau_des` / `tau_safe` per joint, with
   ±`tau_max` band and ASIF-active shading) for torque-mode trials, or
   velocity triplet for kinematic trials.

**`plot compare <csv1> <csv2> ... [--save-dir DIR] [--format pdf|png] [--dpi N]`**

Three overlay figures, one line per trial:

- Translational error vs time.
- Rotational geodesic error vs time.
- Riemannian SE(3) error vs time.

**`report <csv-or-dir> [-o output]`**

Loads every CSV in a directory (or a single CSV) and prints a
rectangular metric table. Pass `-o summary.md` for a markdown file or
`-o summary.csv` for a CSV file suitable for further analysis in
pandas/Excel.

**`metric <name> <csv>`**

Prints a single scalar metric to stdout. Intended for CI scripts or
quick inspection:

```bash
pixi run analyse metric trans_rmse_m tests/results/trial.csv
# 0.003421
pixi run analyse metric safety_intervention_integral tests/results/trial.csv
# 0.3621
```

### Saving for LaTeX

Both plot sub-commands accept output flags. PDF is the default because it
is a vector format and can be included directly in LaTeX with no
conversion:

```bash
pixi run analyse plot trial \
    tests/results/sim_GeometricPDController_PipeInspection_safe_ff.csv \
    --save-dir paper/figures/phase2/

```

Produces:

```
paper/figures/phase2/
├── sim_GeometricPDController_PipeInspection_safe_ff_tracking_errors.pdf
├── sim_GeometricPDController_PipeInspection_safe_ff_settling.pdf
├── sim_GeometricPDController_PipeInspection_safe_ff_3d_path.pdf
└── sim_GeometricPDController_PipeInspection_safe_ff_torque_triplet.pdf
```

Include in LaTeX:

```latex
\includegraphics[width=\linewidth]{figures/phase2/sim_GeometricPDController_PipeInspection_safe_ff_tracking_errors.pdf}
```

All figures are saved with `bbox_inches="tight"` (no surrounding
whitespace). For slides or quick previews, use `--format png --dpi 200`:

```bash
pixi run analyse plot compare trial_a.csv trial_b.csv \
    --save-dir slides/ --format png --dpi 200
```

### Programmatic API

The CLI is a thin wrapper over a normal importable Python API. Use it
directly in notebooks or analysis scripts.

**Loading trials:**

```python
from xarm_geo_analysis import Trial, Experiment

# Single trial.
trial = Trial.load("tests/results/sim_GeometricPDController_PipeInspection_safe_ff.csv")

print(trial.name)   # "sim_GeometricPDController_PipeInspection_safe_ff"
print(trial.dof)    # 6
print(trial.dt)     # 0.002

# Typed array views (all return numpy arrays).
p = trial.p_actual()    # (N, 3)  EE position
R = trial.R_actual()    # scipy Rotation (N,)
tau = trial.tau_safe()  # (N, 6)  post-ASIF torques; NaN rows in velocity mode

# Whole directory as an Experiment.
exp = Experiment.load_dir("tests/results/")
```

**Computing metrics:**

```python
from xarm_geo_analysis.metrics import (
    translational_rmse, rotational_rmse, riemannian_rmse,
    steady_state_rmse, settling_time,
    normalized_control_effort, safety_intervention_integral,
    asif_activity,
)

print(f"Trans RMSE : {translational_rmse(trial)*1e3:.2f} mm")
print(f"Rot RMSE   : {rotational_rmse(trial):.4f} rad")
print(f"Riem RMSE  : {riemannian_rmse(trial):.4f}")
print(f"Settling   : {settling_time(trial)}")
print(f"Effort     : {normalized_control_effort(trial):.2f}")
print(f"ASIF integ.: {safety_intervention_integral(trial):.4f}")
print(asif_activity(trial))
```

**Generating plots:**

```python
from xarm_geo_analysis.plotting import (
    plot_tracking_errors, plot_settling, plot_3d_path,
    plot_torque_triplet, overlay,
)

fig = plot_tracking_errors(trial)
fig.savefig("tracking.pdf", bbox_inches="tight")

# Multi-trial overlay.
fig2 = overlay(exp, translational_error, ylabel="Trans. error (m)")
fig2.savefig("compare.pdf", bbox_inches="tight")
```

**Generating the summary table:**

```python
from xarm_geo_analysis.report import summarise_experiment, to_markdown, to_csv

df = summarise_experiment(exp)
print(to_markdown(df))      # prints markdown table
to_csv(df, "summary.csv")   # writes CSV
```

---

## Metric Reference

Full enumeration of every scalar produced by `summarise()`. Vector
sub-metrics (per-joint) are expanded to `<key>.{joint_idx}` columns in
the experiment DataFrame.

### Tracking

| Name | Units | Applies to |
|---|---|---|
| `trans_rmse_m` | m | Both |
| `rot_rmse_rad` | rad | Both |
| `riem_rmse` | — | Both |
| `ss_trans_rmse_m` | m | Both |
| `ss_rot_rmse_rad` | rad | Both |
| `ss_riem_rmse` | — | Both |

Steady-state metrics (`ss_*`) are computed over the last 20 % of the
trial by default. The Riemannian error uses weights
`w_trans = w_rot = 1.0`; the combined band radius is
`sqrt((5e-3)² + (π/180)²) ≈ 0.0181`.

### Transient

| Name | Units | Applies to |
|---|---|---|
| `max_overshoot_trans_m` | m | Both |
| `max_overshoot_rot_rad` | rad | Both |
| `max_overshoot_riem` | — | Both |
| `post_overshoot_trans_m` | m | Both |
| `post_overshoot_rot_rad` | rad | Both |
| `post_overshoot_riem` | — | Both |
| `settling_trans_s` | s | Both |
| `settling_rot_s` | s | Both |
| `settling_riem_s` | s | Both |

`max_overshoot_*` is the peak error over the full trial. `post_overshoot_*`
is the peak error after the minimum-error point in the last 50 % of the
trial (meaningful for step-like phases). Settling time uses bands of
5 mm (translational) and 1° (rotational). `NaN` means the trial never
settled within the recorded window.

### Effort

| Name | Units | Applies to | NaN when |
|---|---|---|---|
| `norm_control_effort` | — | Torque mode | Velocity-mode trial, or `tau_max` missing from sidecar |
| `norm_velocity_effort` | — | Kinematic mode | Torque-mode trial, or `v_max` missing from sidecar |
| `tau_sat_fraction` | — | Torque mode | As above |
| `v_sat_fraction` | — | Kinematic mode | As above |

`norm_control_effort` = `∫ Σ (τᵢ / τ_max,ᵢ)² dt`.
`norm_velocity_effort` = `∫ Σ (vᵢ / v_max,ᵢ)² dt`.
Saturation fractions are the fraction of ticks where any joint exceeded
95 % of its limit.

### Safety — ASIF (dynamic / torque-mode)

| Name | Units | NaN when |
|---|---|---|
| `safety_intervention_integral` | N·m·s | Velocity-mode trial |
| `asif_invocation_rate` | — | Velocity-mode trial |
| `asif_modification_rate` | — | Velocity-mode trial |
| `asif_mean_delta_tau` | N·m | Velocity-mode trial |
| `asif_max_delta_tau` | N·m | Velocity-mode trial |
| `asif_infeasible_count` | count | Velocity-mode trial |
| `asif_max_iters_count` | count | Velocity-mode trial |
| `asif_per_joint_mean_delta.0` .. `.{dof-1}` | N·m | Velocity-mode trial |

`safety_intervention_integral` = `∫ ‖τ_des − τ_safe‖₂ dt`. This is the
defining Phase 3 metric: it quantifies how much the ASIF layer had to
filter the geometric controller's desired torque. Returns 0.0 when ASIF
was off (τ_safe == τ_des by construction).

### Safety — OptIK / Velocity Rescale (kinematic mode)

| Name | Units | NaN when |
|---|---|---|
| `kinematic_intervention_integral` | rad/s·s | Torque-mode trial |
| `optik_invocation_rate` | — | Torque-mode trial |
| `optik_modification_rate` | — | Torque-mode trial |
| `optik_mean_delta_v` | rad/s | Torque-mode trial |
| `optik_max_delta_v` | rad/s | Torque-mode trial |
| `optik_infeasible_count` | count | Torque-mode trial |
| `optik_max_iters_count` | count | Torque-mode trial |
| `optik_per_joint_mean_delta.0` .. `.{dof-1}` | rad/s | Torque-mode trial |

Note: for `KinematicJointControllerBase` trials, `optik_invocation_rate`
is always 0 (there is no QP in that base), and `optik_modification_rate`
reflects direction-preserving velocity-limit rescaling rather than QP
activity.

---

## Adding a New Metric

Four steps, illustrated with a hypothetical `peak_velocity_norm` metric
(the maximum ‖v‖₂ observed over the trial).

**Step 1 — Add the function to the appropriate metrics module.**

The function accepts a `Trial` and returns a scalar float (or a
`np.ndarray` for time-series metrics). Place it in
`python/xarm_geo_analysis/src/xarm_geo_analysis/metrics/effort.py` (or
whichever module best fits the category):

```python
# In metrics/effort.py

def peak_velocity_norm(trial: Trial) -> float:
    """Maximum ||v||_2 over the trial (rad/s)."""
    v = trial.v()   # (N, dof)
    norms = np.linalg.norm(v, axis=1)   # (N,)
    return float(np.max(norms))
```

**Step 2 — Re-export from `metrics/__init__.py`.**

```python
# In metrics/__init__.py
from xarm_geo_analysis.metrics.effort import (
    ...
    peak_velocity_norm,
)

__all__ = [
    ...
    "peak_velocity_norm",
]
```

**Step 3 — Include it in `summarise()` in `report.py`.**

```python
# In report.py
from xarm_geo_analysis.metrics.effort import (
    ...
    peak_velocity_norm,
)

def summarise(trial: Trial) -> dict[str, float]:
    row: dict[str, float] = {}
    ...
    row["peak_velocity_norm"] = peak_velocity_norm(trial)
    ...
    return row
```

**Step 4 — Add a synthetic unit test in `tests/test_metrics.py`.**

```python
class TestPeakVelocityNorm:
    def test_constant_velocity(self):
        """Constant v = [1, 0, ..., 0] => peak norm == 1.0."""
        from xarm_geo_analysis.metrics.effort import peak_velocity_norm

        p = np.zeros((N, 3))
        q = _identity_rotation_quat(N)
        trial = _build_trial(p, p, q, q)

        # Override the v column in the dataframe directly.
        trial.df["v.0"] = 1.0
        for i in range(1, DOF):
            trial.df[f"v.{i}"] = 0.0

        assert peak_velocity_norm(trial) == pytest.approx(1.0, rel=1e-10)
```

Run `pixi run test-analysis` to verify.

---

## Troubleshooting

**All ASIF metrics are NaN.**
The trial was logged in kinematic (velocity) mode. The torque triplet
columns are blank, so `safety_intervention_integral` and related metrics
return NaN by design. Use the OptIK equivalents
(`kinematic_intervention_integral`, `optik_activity`) instead.

**All effort metrics are NaN.**
The `.meta.json` sidecar is missing or was deleted, so `tau_max` and
`v_max` are unavailable. Re-run the C++ trial with logging enabled to
regenerate the sidecar alongside the CSV. All tracking and transient
metrics remain unaffected (they do not need the sidecar).

**Plot windows do not appear.**
You are in a headless terminal or SSH session without X forwarding.
Use `--save-dir` to write files to disk instead:

```bash
pixi run analyse plot trial trial.csv --save-dir /tmp/figures/
```

**`ValueError: requires a torque-mode trial` from `plot_torque_triplet`.**
The trial was logged in kinematic mode. Use `plot_velocity_triplet`
instead. `trial.is_torque_mode()` and `trial.is_kinematic_mode()` can be
used to branch programmatically.

**`Empty experiment` from `Experiment.load_dir`.**
The directory contains no `.csv` files, or the default glob pattern
`*.csv` did not match. Check that the C++ trial ran with `--log true`
and that the output path points to the right directory. The default
output path in `simulation_test.cpp` is `tests/results/`; run the binary
from the repository root or adjust `DataLogger::Config::output_path`.

**`optik_invocation_rate` is 0 for a constraint-aware kinematic trial.**
This is expected for `KinematicJointControllerBase` trials (e.g.
`JointPController` / `JointPDController`). The joint-space kinematic base
applies a direction-preserving velocity rescale rather than an OptIK QP.
Use `optik_modification_rate` to see how often the rescale actually
clipped, and `kinematic_intervention_integral` to quantify the aggregate
effect.
