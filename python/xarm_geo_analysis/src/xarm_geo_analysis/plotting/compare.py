"""
Multi-trial overlay plots for experiment-level comparison.
"""

from __future__ import annotations

from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from xarm_geo_analysis.plotting.style import ALPHA_PRIMARY, LW_PRIMARY
from xarm_geo_analysis.trial import Experiment, Trial


def overlay(
    experiment: Experiment,
    metric_fn: Callable[[Trial], np.ndarray],
    ylabel: str = "",
    title: str = "",
    figsize: tuple[float, float] = (10, 5),
) -> Figure:
    """Overlay a time-series metric across all trials in an experiment.

    Parameters
    ----------
    experiment : the Experiment to plot.
    metric_fn  : a callable ``f(trial) -> np.ndarray`` returning a 1-D array
                 (e.g. ``translational_error``).
    ylabel     : y-axis label.
    title      : figure title (defaults to a Title Case rendering of the metric
                 function name).
    figsize    : matplotlib figure size.

    Returns
    -------
    matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=figsize)

    for trial in experiment:
        t = trial.t()
        y = metric_fn(trial)
        ax.plot(t, y, linewidth=LW_PRIMARY, label=trial.name, alpha=ALPHA_PRIMARY)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel or "")

    if title:
        derived_title = title
    else:
        raw = getattr(metric_fn, "__name__", "")
        derived_title = raw.replace("_", " ").title()
    ax.set_title(derived_title)

    ax.legend()
    fig.tight_layout()
    return fig
