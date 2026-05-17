"""
Central styling configuration for all xarm-geo-analysis plots.

Call ``apply_style()`` once before creating any figures (done automatically
by ``plotting/__init__.py``).  All semantic constants defined here should be
imported directly by the individual plot modules rather than hard-coding
values inline.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Semantic line-width constants
# ---------------------------------------------------------------------------

LW_PRIMARY = 2.0  # bold main data trace
LW_SECONDARY = 1.4  # supporting trace (e.g. ctrl / des alongside safe)
LW_REFERENCE = 1.2  # reference lines, dashed bands, settling markers
LW_PATH = 2.0  # 3-D EE path lines

# ---------------------------------------------------------------------------
# Semantic alpha constants
# ---------------------------------------------------------------------------

ALPHA_PRIMARY = 0.95
ALPHA_SECONDARY = 0.75
ALPHA_BAND_FILL = 0.15  # axhspan / axvspan background shading
ALPHA_OVERLAY = 0.18  # ASIF / OptIK activation overlay
ALPHA_PATH_MUTED = 0.55  # muted path in 3-D plot

# ---------------------------------------------------------------------------
# Semantic colour constants
# ---------------------------------------------------------------------------

COLOR_TRANS = "steelblue"
COLOR_ROT = "darkorange"
COLOR_RIEM = "purple"
COLOR_LIMIT = "red"  # limit bands, settling markers, safety overlays
COLOR_SAFE = "green"  # safe / within-band signal
COLOR_CTRL = "steelblue"  # controlled command
COLOR_DES = "darkorange"  # desired command
COLOR_BAND_OK = "green"  # settling / tolerance band fill
COLOR_TARGET = "gray"  # target / reference path
COLOR_ACTUAL = "black"  # actual / measured path

# ---------------------------------------------------------------------------
# Title separator (plain ASCII, renders in every PDF/font backend)
# ---------------------------------------------------------------------------

TITLE_SEP = " - "

# ---------------------------------------------------------------------------
# rcParams defaults
# ---------------------------------------------------------------------------

_STYLE: dict = {
    # Font
    "font.family": "serif",
    "mathtext.fontset": "cm",
    # Sizes
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    # Weight
    "axes.titleweight": "bold",
    # Lines
    "lines.linewidth": LW_PRIMARY,
    # Grid
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
    # Legend
    "legend.frameon": True,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "0.8",
    "legend.loc": "best",
    # Save
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
}


def apply_style() -> None:
    """Apply publication-ready rcParams.  Idempotent — safe to call multiple times."""
    plt.rcParams.update(_STYLE)


def style_axes_3d(ax) -> None:
    """Apply consistent styling to a 3-D axes object."""
    pane_color = (0.97, 0.97, 0.97, 1.0)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = True
        pane.set_facecolor(pane_color)
        pane.set_edgecolor("0.8")
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
