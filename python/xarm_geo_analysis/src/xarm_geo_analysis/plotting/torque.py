"""
Torque-triplet and velocity-triplet overlay plots.

Each plot shows ctrl / des / safe overlaid for one or all joints, with
joint limits as shaded bands and ASIF / OptIK activation as shaded intervals.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from xarm_geo_analysis.plotting.style import (
    ALPHA_BAND_FILL,
    ALPHA_OVERLAY,
    ALPHA_PRIMARY,
    ALPHA_SECONDARY,
    COLOR_CTRL,
    COLOR_DES,
    COLOR_LIMIT,
    COLOR_SAFE,
    LW_PRIMARY,
    LW_SECONDARY,
    TITLE_SEP,
)
from xarm_geo_analysis.trial import Trial


def plot_torque_triplet(
    trial: Trial,
    joint_idx: int | None = None,
    figsize: tuple[float, float] = (10, 4),
) -> Figure:
    """Overlay tau_ctrl / tau_des / tau_safe for a single joint (or all joints).

    Parameters
    ----------
    trial     : the trial to plot.
    joint_idx : joint index to plot (0-based).  If None, one subplot per joint.
    figsize   : matplotlib figure size (per-joint mode uses height * dof).

    Returns
    -------
    matplotlib Figure.
    """
    if not trial.is_torque_mode():
        raise ValueError("plot_torque_triplet requires a torque-mode trial")

    t = trial.t()
    tau_ctrl = trial.tau_ctrl()
    tau_des = trial.tau_des()
    tau_safe = trial.tau_safe()
    tau_max = trial.tau_max
    asif_mod = trial.asif_modified_mask()

    if joint_idx is not None:
        joints = [joint_idx]
        fig, axes_list = plt.subplots(1, 1, figsize=figsize)
        axes_list = [axes_list]
    else:
        joints = list(range(trial.dof))
        fig, axes_list = plt.subplots(
            trial.dof, 1, figsize=(figsize[0], figsize[1] * trial.dof), sharex=True
        )

    for ax, j in zip(axes_list, joints):
        ax.plot(
            t,
            tau_ctrl[:, j],
            color=COLOR_CTRL,
            linewidth=LW_SECONDARY,
            alpha=ALPHA_SECONDARY,
            label="Controlled Torque",
        )
        ax.plot(
            t,
            tau_des[:, j],
            color=COLOR_DES,
            linewidth=LW_SECONDARY,
            alpha=ALPHA_SECONDARY,
            linestyle="--",
            label="Desired Torque",
        )
        ax.plot(
            t,
            tau_safe[:, j],
            color=COLOR_SAFE,
            linewidth=LW_PRIMARY,
            alpha=ALPHA_PRIMARY,
            label="Safe Torque",
        )

        # Joint torque limit band.
        if tau_max is not None and np.isfinite(tau_max[j]):
            ax.axhspan(
                -tau_max[j],
                tau_max[j],
                alpha=ALPHA_BAND_FILL,
                color=COLOR_LIMIT,
                label=f"±Torque Limit ({tau_max[j]:.1f} Nm)",
            )

        # Shade intervals where ASIF modified the command.
        ax.fill_between(
            t,
            ax.get_ylim()[0],
            ax.get_ylim()[1],
            where=asif_mod,
            color=COLOR_LIMIT,
            alpha=ALPHA_OVERLAY,
            label="ASIF Active",
        )

        ax.set_ylabel(f"Joint {j} Torque (Nm)")
        ax.legend()

    axes_list[-1].set_xlabel("Time (s)")
    axes_list[0].set_title(f"{trial.name}{TITLE_SEP}Torque Triplet")
    fig.tight_layout()
    return fig


def plot_velocity_triplet(
    trial: Trial,
    joint_idx: int | None = None,
    figsize: tuple[float, float] = (10, 4),
) -> Figure:
    """Overlay v_ctrl / v_des / v_safe for a single joint (or all joints).

    Parameters
    ----------
    trial     : the trial to plot.
    joint_idx : joint index (0-based).  None = one subplot per joint.
    figsize   : matplotlib figure size.

    Returns
    -------
    matplotlib Figure.
    """
    if not trial.is_kinematic_mode():
        raise ValueError("plot_velocity_triplet requires a kinematic-mode trial")

    t = trial.t()
    v_ctrl = trial.v_ctrl()
    v_des = trial.v_des()
    v_safe = trial.v_safe()
    v_max = trial.v_max
    optik_mod = trial.optik_modified_mask()

    if joint_idx is not None:
        joints = [joint_idx]
        fig, axes_list = plt.subplots(1, 1, figsize=figsize)
        axes_list = [axes_list]
    else:
        joints = list(range(trial.dof))
        fig, axes_list = plt.subplots(
            trial.dof, 1, figsize=(figsize[0], figsize[1] * trial.dof), sharex=True
        )

    for ax, j in zip(axes_list, joints):
        ax.plot(
            t,
            v_ctrl[:, j],
            color=COLOR_CTRL,
            linewidth=LW_SECONDARY,
            alpha=ALPHA_SECONDARY,
            label="Controlled Velocity",
        )
        ax.plot(
            t,
            v_des[:, j],
            color=COLOR_DES,
            linewidth=LW_SECONDARY,
            alpha=ALPHA_SECONDARY,
            linestyle="--",
            label="Desired Velocity",
        )
        ax.plot(
            t,
            v_safe[:, j],
            color=COLOR_SAFE,
            linewidth=LW_PRIMARY,
            alpha=ALPHA_PRIMARY,
            label="Safe Velocity",
        )

        if v_max is not None and np.isfinite(v_max[j]):
            ax.axhspan(
                -v_max[j],
                v_max[j],
                alpha=ALPHA_BAND_FILL,
                color=COLOR_LIMIT,
                label=f"±Velocity Limit ({v_max[j]:.2f} rad/s)",
            )

        ax.fill_between(
            t,
            ax.get_ylim()[0],
            ax.get_ylim()[1],
            where=optik_mod,
            color=COLOR_LIMIT,
            alpha=ALPHA_OVERLAY,
            label="OptIK / Rescale Active",
        )

        ax.set_ylabel(f"Joint {j} Velocity (rad/s)")
        ax.legend()

    axes_list[-1].set_xlabel("Time (s)")
    axes_list[0].set_title(f"{trial.name}{TITLE_SEP}Velocity Triplet")
    fig.tight_layout()
    return fig
