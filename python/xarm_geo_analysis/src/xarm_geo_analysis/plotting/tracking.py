"""
Tracking-error and 3-D path plots.

All rotation data is read from quaternion columns; no Euler angles are used.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from xarm_geo_analysis.metrics.tracking import (
    error_twist_norm,
    riemannian_se3_error,
    rotational_geodesic_error,
    translational_error,
)
from xarm_geo_analysis.metrics.transient import _BAND_RIEM, _BAND_ROT_RAD, _BAND_TRANS_M
from xarm_geo_analysis.plotting.style import (
    ALPHA_BAND_FILL,
    ALPHA_PATH_MUTED,
    ALPHA_PRIMARY,
    COLOR_ACTUAL,
    COLOR_BAND_OK,
    COLOR_LIMIT,
    COLOR_RIEM,
    COLOR_ROT,
    COLOR_TARGET,
    COLOR_TRANS,
    LW_PATH,
    LW_PRIMARY,
    LW_REFERENCE,
    TITLE_SEP,
    style_axes_3d,
)
from xarm_geo_analysis.trial import Experiment, Trial

# Axis-length for EE frame triads in the 3-D path plot (metres).
_TRIAD_LENGTH: float = 0.02
# Downsample factor for triads (every Nth sample).
_TRIAD_STEP: int = 100


# ---------------------------------------------------------------------------
# Tracking-error time series (3-panel)
# ---------------------------------------------------------------------------


def plot_tracking_errors(
    trial: Trial,
    w_trans: float = 1.0,
    w_rot: float = 1.0,
    figsize: tuple[float, float] = (10, 8),
) -> Figure:
    """Three-panel figure: translational, rotational, and Riemannian error vs time.

    Parameters
    ----------
    trial         : the trial to plot.
    w_trans, w_rot: weights forwarded to riemannian_se3_error.
    figsize       : matplotlib figure size.

    Returns
    -------
    matplotlib Figure (caller is responsible for plt.show() / savefig()).
    """
    t = trial.t()
    e_trans = translational_error(trial) * 1e3  # convert to mm
    e_rot = np.degrees(rotational_geodesic_error(trial))
    e_riem = riemannian_se3_error(trial, w_trans, w_rot)

    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)

    axes[0].plot(
        t, e_trans, color=COLOR_TRANS, linewidth=LW_PRIMARY, alpha=ALPHA_PRIMARY
    )
    axes[0].axhline(
        _BAND_TRANS_M * 1e3,
        color=COLOR_LIMIT,
        linestyle="--",
        linewidth=LW_REFERENCE,
        label="5 mm Band",
    )
    axes[0].set_ylabel("Translational Error (mm)")
    axes[0].legend()
    axes[0].set_title(f"{trial.name}{TITLE_SEP}Tracking Errors")

    axes[1].plot(t, e_rot, color=COLOR_ROT, linewidth=LW_PRIMARY, alpha=ALPHA_PRIMARY)
    axes[1].axhline(
        math.degrees(_BAND_ROT_RAD),
        color=COLOR_LIMIT,
        linestyle="--",
        linewidth=LW_REFERENCE,
        label="1° Band",
    )
    axes[1].set_ylabel("Rotational Error (deg)")
    axes[1].legend()

    axes[2].plot(t, e_riem, color=COLOR_RIEM, linewidth=LW_PRIMARY, alpha=ALPHA_PRIMARY)
    axes[2].axhline(
        _BAND_RIEM,
        color=COLOR_LIMIT,
        linestyle="--",
        linewidth=LW_REFERENCE,
        label=f"Riemannian Band ({_BAND_RIEM:.4f})",
    )
    axes[2].set_ylabel("Riemannian Error")
    axes[2].set_xlabel("Time (s)")
    axes[2].legend()

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Settling visualisation
# ---------------------------------------------------------------------------


def plot_settling(
    trial: Trial,
    trans_band_m: float = _BAND_TRANS_M,
    rot_band_rad: float = _BAND_ROT_RAD,
    figsize: tuple[float, float] = (10, 5),
) -> Figure:
    """Two-panel figure with band shading and settling-time annotation.

    Parameters
    ----------
    trial         : the trial to plot.
    trans_band_m  : translational band radius (metres).
    rot_band_rad  : rotational band radius (radians).
    figsize       : matplotlib figure size.

    Returns
    -------
    matplotlib Figure.
    """
    from xarm_geo_analysis.metrics.transient import settling_time

    t = trial.t()
    e_trans = translational_error(trial) * 1e3
    e_rot = np.degrees(rotational_geodesic_error(trial))
    st = settling_time(trial, trans_band_m, rot_band_rad)

    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

    # Translational panel.
    band_mm = trans_band_m * 1e3
    axes[0].plot(
        t,
        e_trans,
        color=COLOR_TRANS,
        linewidth=LW_PRIMARY,
        alpha=ALPHA_PRIMARY,
        label="Translational Error",
    )
    axes[0].axhspan(
        0,
        band_mm,
        alpha=ALPHA_BAND_FILL,
        color=COLOR_BAND_OK,
        label=f"±{band_mm:.1f} mm Band",
    )
    if math.isfinite(st["trans_s"]):
        axes[0].axvline(
            st["trans_s"],
            color=COLOR_LIMIT,
            linestyle="--",
            linewidth=LW_REFERENCE,
            label=f"Settled at {st['trans_s']:.2f} s",
        )
    axes[0].set_ylabel("Translational Error (mm)")
    axes[0].legend()
    axes[0].set_title(f"{trial.name}{TITLE_SEP}Settling Response")

    # Rotational panel.
    band_deg = math.degrees(rot_band_rad)
    axes[1].plot(
        t,
        e_rot,
        color=COLOR_ROT,
        linewidth=LW_PRIMARY,
        alpha=ALPHA_PRIMARY,
        label="Rotational Error",
    )
    axes[1].axhspan(
        0,
        band_deg,
        alpha=ALPHA_BAND_FILL,
        color=COLOR_BAND_OK,
        label=f"±{band_deg:.1f}° Band",
    )
    if math.isfinite(st["rot_s"]):
        axes[1].axvline(
            st["rot_s"],
            color=COLOR_LIMIT,
            linestyle="--",
            linewidth=LW_REFERENCE,
            label=f"Settled at {st['rot_s']:.2f} s",
        )
    axes[1].set_ylabel("Rotational Error (deg)")
    axes[1].set_xlabel("Time (s)")
    axes[1].legend()

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 3-D path with EE frame triads
# ---------------------------------------------------------------------------


def plot_3d_path(
    trial: Trial,
    triad_step: int = _TRIAD_STEP,
    axis_length: float = _TRIAD_LENGTH,
    figsize: tuple[float, float] = (10, 8),
) -> Figure:
    """3-D line plot of actual vs target EE path with orientation triads.

    Rotations are read from quaternion columns — no Euler-angle conversion.

    Parameters
    ----------
    trial       : the trial to plot.
    triad_step  : draw a frame triad every this many samples.
    axis_length : arrow length in metres.
    figsize     : matplotlib figure size.

    Returns
    -------
    matplotlib Figure.
    """
    p_actual = trial.p_actual()
    p_target = trial.p_target()
    R_actual = trial.R_actual()

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(
        p_target[:, 0],
        p_target[:, 1],
        p_target[:, 2],
        color=COLOR_TARGET,
        alpha=ALPHA_PATH_MUTED,
        linewidth=LW_PATH,
        label="Target Path",
    )
    ax.plot(
        p_actual[:, 0],
        p_actual[:, 1],
        p_actual[:, 2],
        color=COLOR_ACTUAL,
        alpha=ALPHA_PATH_MUTED,
        linewidth=LW_PATH,
        label="Actual Path",
    )

    # Draw orientation triads at every triad_step sample.
    rot_matrices = R_actual.as_matrix()  # (N, 3, 3)
    for i in range(0, len(p_actual), triad_step):
        pos = p_actual[i]
        R = rot_matrices[i]
        for col, color in zip(range(3), ("r", "g", "b")):
            ax.quiver(
                *pos,
                *R[:, col],
                color=color,
                length=axis_length,
                normalize=True,
                arrow_length_ratio=0.3,
            )

    # Dummy legend entries for triad axis colours.
    ax.plot([], [], color="r", label="Local X")
    ax.plot([], [], color="g", label="Local Y")
    ax.plot([], [], color="b", label="Local Z")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(f"{trial.name}{TITLE_SEP}End-Effector Path")
    ax.legend()

    style_axes_3d(ax)

    # Equal aspect ratio.
    all_pts = np.vstack([p_actual, p_target])
    mins = all_pts.min(axis=0)
    maxs = all_pts.max(axis=0)
    mid = (mins + maxs) / 2
    half = np.max(maxs - mins) / 2
    ax.set_xlim3d(mid[0] - half, mid[0] + half)
    ax.set_ylim3d(mid[1] - half, mid[1] + half)
    ax.set_zlim3d(mid[2] - half, mid[2] + half)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Multi-trial 3-D path overlay
# ---------------------------------------------------------------------------


def plot_3d_paths_compare(
    experiment: Experiment,
    obstacle: Optional[Tuple[np.ndarray, float]] = None,
    show_reference: bool = True,
    figsize: tuple[float, float] = (10, 8),
) -> Figure:
    """3-D overlay of actual EE paths across all trials in an experiment.

    Parameters
    ----------
    experiment    : Experiment containing the trials to compare.
    obstacle      : (center, radius) tuple to draw a sphere.  If None the
                    function auto-reads obstacle_{x,y,z,radius} from the first
                    trial's sidecar meta; omits the sphere when both sources
                    are absent.
    show_reference: if True, plot the target path of the first trial as a
                    dashed reference (all trials share the same target for
                    comparative experiments).
    figsize       : matplotlib figure size.

    Returns
    -------
    matplotlib Figure.
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    all_pts: list[np.ndarray] = []

    # Dashed reference from the first trial.
    trials = list(experiment)
    if show_reference and trials:
        p_ref = trials[0].p_target()
        ax.plot(
            p_ref[:, 0],
            p_ref[:, 1],
            p_ref[:, 2],
            color=COLOR_TARGET,
            linewidth=LW_REFERENCE,
            linestyle="--",
            alpha=ALPHA_PATH_MUTED,
            label="Reference (unconstrained)",
        )
        all_pts.append(p_ref)

    # Actual paths, one colour per trial.
    prop_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, trial in enumerate(trials):
        p = trial.p_actual()
        color = prop_cycle[i % len(prop_cycle)]
        ax.plot(
            p[:, 0],
            p[:, 1],
            p[:, 2],
            color=color,
            linewidth=LW_PATH,
            alpha=ALPHA_PRIMARY,
            label=trial.name,
        )
        all_pts.append(p)

    # Obstacle sphere.
    if obstacle is None and trials:
        m = trials[0].meta
        if "obstacle_x" in m and "obstacle_radius" in m:
            center = np.array([m["obstacle_x"], m["obstacle_y"], m["obstacle_z"]])
            obstacle = (center, float(m["obstacle_radius"]))

    if obstacle is not None:
        center, radius = obstacle
        u, v = np.mgrid[0 : 2 * np.pi : 24j, 0 : np.pi : 13j]
        xs = center[0] + radius * np.cos(u) * np.sin(v)
        ys = center[1] + radius * np.sin(u) * np.sin(v)
        zs = center[2] + radius * np.cos(v)
        ax.plot_wireframe(
            xs,
            ys,
            zs,
            color="red",
            alpha=0.25,
            linewidth=0.5,
            label=f"Obstacle (r={radius:.2f} m)",
        )
        all_pts.append(
            np.array(
                [
                    [center[0] - radius, center[1] - radius, center[2] - radius],
                    [center[0] + radius, center[1] + radius, center[2] + radius],
                ]
            )
        )

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("End-Effector Path Comparison")
    ax.legend()
    style_axes_3d(ax)

    if all_pts:
        pts = np.vstack(all_pts)
        mins = pts.min(axis=0)
        maxs = pts.max(axis=0)
        mid = (mins + maxs) / 2
        half = max(np.max(maxs - mins) / 2, 0.05)
        ax.set_xlim3d(mid[0] - half, mid[0] + half)
        ax.set_ylim3d(mid[1] - half, mid[1] + half)
        ax.set_zlim3d(mid[2] - half, mid[2] + half)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Error-twist overlay
# ---------------------------------------------------------------------------


def plot_error_twist_overlay(
    experiment: Experiment,
    figsize: tuple[float, float] = (10, 5),
) -> Figure:
    """Overlay ‖ξ_e‖₂(t) for all trials in an experiment.

    Useful for Exp 1A: highlights the difference between LieGroup (error
    twist starts near zero, ramps up) and LieAlgebra (peaks immediately).

    Returns
    -------
    matplotlib Figure.
    """
    from xarm_geo_analysis.plotting.style import LW_PRIMARY, ALPHA_PRIMARY

    fig, ax = plt.subplots(figsize=figsize)
    for trial in experiment:
        ax.plot(
            trial.t(),
            error_twist_norm(trial),
            linewidth=LW_PRIMARY,
            alpha=ALPHA_PRIMARY,
            label=trial.name,
        )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"$\|\xi_e\|_2$  (m/s + rad/s)")
    ax.set_title("Error Twist Norm Comparison")
    ax.legend()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Per-axis 1-D position zoom
# ---------------------------------------------------------------------------


def plot_axis_zoom(
    trial: Trial,
    axis: str = "x",
    t_window: Optional[Tuple[float, float]] = None,
    figsize: tuple[float, float] = (10, 4),
) -> Figure:
    """Actual vs. target position along a single Cartesian axis.

    Parameters
    ----------
    axis     : "x", "y", or "z".
    t_window : (t_start, t_end) to zoom; None = full trial.
    figsize  : matplotlib figure size.

    Returns
    -------
    matplotlib Figure.
    """
    ax_idx = {"x": 0, "y": 1, "z": 2}
    if axis not in ax_idx:
        raise ValueError(f"Unknown axis '{axis}'; expected x|y|z")
    idx = ax_idx[axis]

    t = trial.t()
    actual = trial.p_actual()[:, idx]
    target = trial.p_target()[:, idx]

    if t_window is not None:
        mask = (t >= t_window[0]) & (t <= t_window[1])
        t, actual, target = t[mask], actual[mask], target[mask]

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(
        t,
        target * 1e3,
        color=COLOR_TARGET,
        linewidth=LW_REFERENCE,
        linestyle="--",
        alpha=ALPHA_PATH_MUTED,
        label="Target",
    )
    ax.plot(
        t,
        actual * 1e3,
        color=COLOR_ACTUAL,
        linewidth=LW_PRIMARY,
        alpha=ALPHA_PRIMARY,
        label="Actual",
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(f"{axis.upper()} position (mm)")
    ax.set_title(f"{trial.name}{TITLE_SEP}{axis.upper()}-Axis Position")
    ax.legend()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Error distribution box plot
# ---------------------------------------------------------------------------


def plot_error_boxplot(
    experiment: Experiment,
    kind: str = "translational",
    figsize: tuple[float, float] = (8, 5),
) -> Figure:
    """Box plot of per-trial tracking error distribution.

    One box per trial; useful for Exp 3A four-way comparison
    (Sim Kin / Sim Dyn / HW Kin / HW Dyn+Admittance).

    Parameters
    ----------
    kind    : "translational" (mm), "rotational" (deg), or "riemannian".
    figsize : matplotlib figure size.

    Returns
    -------
    matplotlib Figure.
    """
    series: list[np.ndarray] = []
    labels: list[str] = []
    for trial in experiment:
        if kind == "translational":
            err = translational_error(trial) * 1e3
            unit = "mm"
        elif kind == "rotational":
            err = np.degrees(rotational_geodesic_error(trial))
            unit = "deg"
        elif kind == "riemannian":
            err = riemannian_se3_error(trial)
            unit = ""
        else:
            raise ValueError(f"Unknown kind '{kind}'")
        series.append(err[np.isfinite(err)])
        # Shorten label to avoid overlong x-tick text.
        labels.append(trial.name.replace("sim_", "S:").replace("hardware_", "HW:"))

    fig, ax = plt.subplots(figsize=figsize)
    ax.boxplot(series, labels=labels, notch=False, patch_artist=True)
    ax.set_ylabel(
        f"{kind.capitalize()} Error ({unit})" if unit else f"{kind.capitalize()} Error"
    )
    ax.set_title("Tracking Error Distribution")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right", fontsize=8)
    fig.tight_layout()
    return fig
