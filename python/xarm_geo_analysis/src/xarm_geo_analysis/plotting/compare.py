"""
Multi-trial overlay plots for experiment-level comparison.
"""

from __future__ import annotations

from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

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
    title      : figure title (defaults to the metric function name).
    figsize    : matplotlib figure size.

    Returns
    -------
    matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=figsize)

    for trial in experiment:
        t = trial.t()
        y = metric_fn(trial)
        ax.plot(t, y, linewidth=1.0, label=trial.name, alpha=0.85)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel or "")
    ax.set_title(title or getattr(metric_fn, "__name__", ""))
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig
