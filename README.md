# xArm Geometric Modelling & Control

A C++20 research library for geometric modelling and control of the UFACTORY xArm line of manipulators, with a Python analysis suite for evaluating controller performance through structured experiments.

## Overview

**xArm-geo** is a C++20 library that implements geometric (SE(3) / Lie-group) control for the UFACTORY xArm manipulator family — xArm5, xArm6, xArm7, Lite6, and UF850. It supplies the modelling primitives needed to do this end-to-end: forward and inverse kinematics, rigid-body dynamics, and collision detection.

Controllers are organised along two axes: task-space versus joint-space, and kinematic (velocity-mode) versus dynamic (torque-mode). On top of these, the library provides two composable safety filters — an Active Set Invariance Filter (ASIF) for torque mode and an Optimal Inverse Differential Kinematics (OptIK) filter for velocity mode — both formulated as QPs with Control Barrier Function (CBF) constraints. A first-order admittance layer bridges torque controllers onto velocity-mode hardware.

The library ships with five comparative study experiments (Exp 1A – 3B) covering step response, smooth trajectory tracking, disturbance rejection, sim-vs-hardware transfer, and obstacle avoidance. Each experiment runs against a MuJoCo physics simulation or, where applicable, a physical robot. A companion Python package (`xarm_geo_analysis`) ingests the logged CSV data and produces scalar metrics tables and publication-ready PDF figures.

## Repository Layout

```
include/xarm_geo/        Public C++ API
├── core/                Lie-Group Types and Robot Model Containers
├── modelling/           Kinematics, Dynamics, and Collision
├── control/             Controller Bases, Feedback Primitives, and Admittance Layer
├── safety/              ASIF and OptIK Filters with CBF Barriers
├── trajectory/          Task and Joint Trajectory Concepts and Adapters
├── interfaces/          MuJoCo Simulation and xArm Hardware Backends
├── diagnostics/         CSV+JSON Data Logger
├── utils/               URDF/YAML Model Builder
└── examples/            Reference Controllers and Trajectories
src/                     Library Implementation
tests/experiments/       Comparative Study Experiment Binaries (Exp 1A – 3B)
python/                  Python Analysis Package (xarm_geo_analysis)
assets/                  Robot Description Files (URDF/MJCF) and Per-Robot Config
docs/                    Usage Guides
paper/figures/           Generated Publication Plots (PDF Output)
```

## Prerequisites

- **[Pixi](https://pixi.sh)** — manages all C++ and Python dependencies (Eigen, MuJoCo, ProxSuite, coal, matplotlib, pandas, …) via conda-forge. No other package manager is needed.
- **Git** — required for dependencies fetched at build time (`smooth` Lie-group library, xArm C++ SDK).
- **Linux x86-64** — the only supported platform (`linux-64`).

## Installation & Build

```bash
# 1. Clone the repository
git clone <url> && cd xArm-geo

# 2. Install all dependencies (first run downloads the conda environment — allow a few minutes)
pixi install

# 3. Build in release mode
#    This also auto-generates URDF/MJCF robot description files from xacro sources.
pixi run build_release
```

> **Debug build:** `pixi run build` builds in Debug mode with additional checks. Use `build_release` when running experiments, as the control loops are timing-sensitive.

## Running the Experiments

Each experiment compiles to a standalone binary and is launched via a `pixi run` task. All simulation variants require no additional arguments; results are written to `tests/results/exp_<id>/`.

| Experiment | Command | Description |
|---|---|---|
| **1A** — Step Response | `pixi run exp-1a` | Large-Angle (178°) Orientation Step; Compares EuclideanP, GeometricP (Lie-Group Gradient), and GeometricP (Lie-Algebra Gradient) in Velocity Mode |
| **1B** — Smooth Tracking | `pixi run exp-1b` | Continuous Trajectory Tracking with Geodesic Error and Control Effort Comparison |
| **2** — Disturbance Rejection | `pixi run exp-2` | External Wrench Disturbance with PID Integrator Rejection |
| **3A** — Sim vs Hardware | `pixi run exp-3a-sim` | Five Variants Across Velocity, Torque, and Admittance-Cascaded Torque Modes |
| **3B** — Obstacle Avoidance | `pixi run exp-3b-sim` | Line Trajectory Through Obstacle; CBF/HOCBF Safety Filters Deflect Collision |

### Hardware variants

Experiments 3A and 3B can run against a physical xArm robot. Pass the robot's IP address as an argument:

```bash
pixi run exp-3a-hw 192.168.1.221
pixi run exp-3b-hw 192.168.1.221
```

The robot must be reachable on the network and in a state that accepts velocity commands before launching.

## Analysing Results

After running an experiment, generate a scalar metrics table and publication figures with:

```bash
pixi run report-exp <id>   # prints a summary table of tracking, effort, and safety metrics
pixi run plot-exp <id>     # generates PDF figures → paper/figures/exp_<id>/
```

Replace `<id>` with the experiment identifier: `1a`, `1b`, `2`, `3a`, or `3b`. For example:

```bash
pixi run report-exp 1a
pixi run plot-exp 1a
```

Individual trial diagnostics can also be plotted directly:

```bash
pixi run analyse plot trial tests/results/exp_1a/<trial>.csv
```

## Development Setup

This repository uses [pre-commit](https://pre-commit.com) to auto-format C/C++ (via `clang-format`) and Python (via `ruff`) on every commit.

After cloning, install the lint environment and the git hook once:

```bash
pixi install -e lint
pixi run -e lint install-hooks
```

From then on, `git commit` will automatically format staged files. If a hook modifies a file, the commit is aborted; re-stage the changes and commit again.

To run all hooks against the entire repository on demand:

```bash
pixi run -e lint lint
```

## Documentation

Guides for using and extending the library:

- [`docs/authoring_controllers.md`](docs/authoring_controllers.md) — writing custom controllers
- [`docs/authoring_trajectories.md`](docs/authoring_trajectories.md) — writing custom trajectories
- [`docs/logging_and_analysis.md`](docs/logging_and_analysis.md) — logging trials and computing metrics
- [`docs/admittance_and_safety.md`](docs/admittance_and_safety.md) — admittance layer and safety filter composition
