"""
Tracking-error and 3-D path plots.

All rotation data is read from quaternion columns; no Euler angles are used.
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from xarm_geo_analysis.metrics.tracking import (
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
from xarm_geo_analysis.trial import Trial

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
