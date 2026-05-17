"""
Disturbance-response and PID integrator plots.
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from xarm_geo_analysis.metrics.tracking import (
    rotational_geodesic_error,
    translational_error,
)
from xarm_geo_analysis.metrics.transient import (
    _BAND_ROT_RAD,
    _BAND_TRANS_M,
    integrator_state,
)
from xarm_geo_analysis.plotting.style import (
    ALPHA_BAND_FILL,
    ALPHA_PRIMARY,
    COLOR_LIMIT,
    COLOR_ROT,
    COLOR_TRANS,
    LW_PRIMARY,
    LW_REFERENCE,
    TITLE_SEP,
)
from xarm_geo_analysis.trial import Experiment, Trial


def plot_tracking_with_disturbance(
    experiment: Experiment,
    kind: str = "translational",
    figsize: tuple[float, float] = (10, 5),
) -> Figure:
    """Overlay tracking errors across trials with a disturbance window marker.

    The disturbance window is read from the sidecar meta fields
    ``disturbance_start_s`` and ``disturbance_end_s`` of each trial.  When
    present, a vertical dashed line is drawn at ``disturbance_start_s`` and a
    shaded band covers the active interval.  The window is omitted when the
    fields are absent.

    Parameters
    ----------
    experiment : Experiment containing the trials to overlay.
    kind       : "translational" (mm) or "rotational" (deg).
    figsize    : matplotlib figure size.

    Returns
    -------
    matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=figsize)

    prop_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    disturbance_drawn = False

    for i, trial in enumerate(experiment):
        t = trial.t()
        if kind == "translational":
            err = translational_error(trial) * 1e3
            ylabel = "Translational Error (mm)"
            band = _BAND_TRANS_M * 1e3
        elif kind == "rotational":
            err = np.degrees(rotational_geodesic_error(trial))
            ylabel = "Rotational Error (deg)"
            band = math.degrees(_BAND_ROT_RAD)
        else:
            raise ValueError(
                f"Unknown kind '{kind}'; expected translational|rotational"
            )

        color = prop_cycle[i % len(prop_cycle)]
        ax.plot(
            t,
            err,
            color=color,
            linewidth=LW_PRIMARY,
            alpha=ALPHA_PRIMARY,
            label=trial.name,
        )

        # Draw disturbance markers once (all trials share the same schedule).
        if not disturbance_drawn:
            t_start = trial.meta.get("disturbance_start_s")
            t_end = trial.meta.get("disturbance_end_s")
            if t_start is not None:
                ax.axvline(
                    t_start,
                    color=COLOR_LIMIT,
                    linestyle="--",
                    linewidth=LW_REFERENCE,
                    label="Disturbance on",
                )
            if t_start is not None and t_end is not None:
                ax.axvspan(
                    t_start,
                    t_end,
                    alpha=ALPHA_BAND_FILL,
                    color=COLOR_LIMIT,
                    label="Disturbance active",
                )
            if t_start is not None:
                disturbance_drawn = True

    ax.axhline(
        band,
        color=COLOR_LIMIT,
        linestyle=":",
        linewidth=LW_REFERENCE,
        alpha=0.5,
        label=f"±{band:.1f} band",
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.set_title("Tracking Error under Disturbance")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_integrator_state(
    trial: Trial,
    figsize: tuple[float, float] = (10, 5),
) -> Figure:
    """Two-panel plot of the PID integrator state norm vs. time.

    Upper panel: ‖e_I_lin‖₂ (translational part of e_I).
    Lower panel: ‖e_I_ang‖₂ (rotational part of e_I).

    All-NaN series (non-PID trial) produces a labelled empty figure.

    Returns
    -------
    matplotlib Figure.
    """
    t = trial.t()
    e = integrator_state(trial)  # (N, 6)

    lin_norm = np.linalg.norm(e[:, :3], axis=1)
    ang_norm = np.linalg.norm(e[:, 3:], axis=1)

    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

    axes[0].plot(
        t, lin_norm, color=COLOR_TRANS, linewidth=LW_PRIMARY, alpha=ALPHA_PRIMARY
    )
    axes[0].set_ylabel(r"$\|e_{I,\mathrm{lin}}\|_2$")
    axes[0].set_title(f"{trial.name}{TITLE_SEP}PID Integrator State")

    axes[1].plot(
        t, ang_norm, color=COLOR_ROT, linewidth=LW_PRIMARY, alpha=ALPHA_PRIMARY
    )
    axes[1].set_ylabel(r"$\|e_{I,\mathrm{ang}}\|_2$")
    axes[1].set_xlabel("Time (s)")

    fig.tight_layout()
    return fig
