"""
Safety-layer diagnostic plots: obstacle distance and intervention magnitude.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from xarm_geo_analysis.metrics.safety import (
    min_distance_series,
)
from xarm_geo_analysis.plotting.style import (
    ALPHA_BAND_FILL,
    ALPHA_PRIMARY,
    ALPHA_SECONDARY,
    COLOR_DES,
    COLOR_LIMIT,
    COLOR_SAFE,
    LW_PRIMARY,
    LW_REFERENCE,
    LW_SECONDARY,
    TITLE_SEP,
)
from xarm_geo_analysis.trial import Experiment, Trial


def plot_obstacle_distance(
    experiment: Experiment,
    figsize: tuple[float, float] = (10, 4),
) -> Figure:
    """Minimum distance to the nearest collision pair vs. time.

    Draws a horizontal red line at d = 0 (contact / penetration boundary).
    One line per trial.

    Returns
    -------
    matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    prop_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, trial in enumerate(experiment):
        d = min_distance_series(trial)
        if not np.any(np.isfinite(d)):
            continue
        color = prop_cycle[i % len(prop_cycle)]
        ax.plot(
            trial.t(),
            d,
            color=color,
            linewidth=LW_PRIMARY,
            alpha=ALPHA_PRIMARY,
            label=trial.name,
        )

    ax.axhline(
        0.0,
        color=COLOR_LIMIT,
        linewidth=LW_REFERENCE,
        linestyle="--",
        label="d = 0 (contact)",
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Min. distance to obstacle (m)")
    ax.set_title("Obstacle Distance vs. Time")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_intervention_norm(
    trial: Trial,
    figsize: tuple[float, float] = (10, 4),
) -> Figure:
    """Commanded vs. safe command norm over time with shaded intervention area.

    For torque-mode trials: plots ‖τ_des‖ vs. ‖τ_safe‖ and shades the delta.
    For kinematic trials: plots ‖v_des‖ vs. ‖v_safe‖.

    Returns
    -------
    matplotlib Figure.
    """
    t = trial.t()

    if trial.is_torque_mode():
        des_mat = trial.tau_des()
        safe_mat = trial.tau_safe()
        ylabel = r"$\|\tau\|_2$  (Nm)"
        label_des = r"$\|\tau_\mathrm{des}\|$"
        label_safe = r"$\|\tau_\mathrm{safe}\|$"
    else:
        des_mat = trial.v_des()
        safe_mat = trial.v_safe()
        ylabel = r"$\|v\|_2$  (rad/s)"
        label_des = r"$\|v_\mathrm{des}\|$"
        label_safe = r"$\|v_\mathrm{safe}\|$"

    if not np.any(np.isfinite(des_mat)):
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title(f"{trial.name}{TITLE_SEP}No command data available")
        return fig

    des_norm = np.linalg.norm(des_mat, axis=1)
    safe_norm = np.linalg.norm(safe_mat, axis=1)

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(
        t,
        des_norm,
        color=COLOR_DES,
        linewidth=LW_SECONDARY,
        alpha=ALPHA_SECONDARY,
        linestyle="--",
        label=label_des,
    )
    ax.plot(
        t,
        safe_norm,
        color=COLOR_SAFE,
        linewidth=LW_PRIMARY,
        alpha=ALPHA_PRIMARY,
        label=label_safe,
    )

    # Shade the area between des and safe: this is the safety layer's footprint.
    ax.fill_between(
        t,
        des_norm,
        safe_norm,
        where=np.isfinite(des_norm) & np.isfinite(safe_norm),
        alpha=ALPHA_BAND_FILL,
        color=COLOR_LIMIT,
        label="Safety intervention",
    )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{trial.name}{TITLE_SEP}Command vs. Safe Command")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_command_norm_overlay(
    experiment: Experiment,
    kind: str = "auto",
    figsize: tuple[float, float] = (10, 5),
) -> Figure:
    """Overlay ‖cmd_safe‖₂(t) for all trials in an experiment.

    Useful for Exp 1B: confirms control effort is identical across
    Lie-Group and Lie-Algebra gradient formulations on smooth trajectories.

    Parameters
    ----------
    kind : "torque", "velocity", or "auto".

    Returns
    -------
    matplotlib Figure.
    """
    from xarm_geo_analysis.metrics.effort import command_norm_series

    fig, ax = plt.subplots(figsize=figsize)
    prop_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, trial in enumerate(experiment):
        norms = command_norm_series(trial, kind=kind)
        if not np.any(np.isfinite(norms)):
            continue
        label_suffix = "τ" if trial.is_torque_mode() else "v"
        ax.plot(
            trial.t(),
            norms,
            color=prop_cycle[i % len(prop_cycle)],
            linewidth=LW_PRIMARY,
            alpha=ALPHA_PRIMARY,
            label=f"{trial.name}  ‖{label_suffix}_safe‖",
        )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"$\|\mathrm{cmd\_safe}\|_2$")
    ax.set_title("Control Effort Comparison")
    ax.legend()
    fig.tight_layout()
    return fig
