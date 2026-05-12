"""
Torque-triplet and velocity-triplet overlay plots.

Each plot shows ctrl / des / safe overlaid for one or all joints, with
joint limits as shaded bands and ASIF / OptIK activation marked as ticks.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

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
            color="steelblue",
            linewidth=0.8,
            label="tau_ctrl",
            alpha=0.7,
        )
        ax.plot(
            t,
            tau_des[:, j],
            color="darkorange",
            linewidth=0.8,
            label="tau_des",
            alpha=0.7,
        )
        ax.plot(t, tau_safe[:, j], color="green", linewidth=1.0, label="tau_safe")

        # Joint torque limit band.
        if tau_max is not None and np.isfinite(tau_max[j]):
            ax.axhspan(
                -tau_max[j],
                tau_max[j],
                alpha=0.07,
                color="red",
                label=f"±tau_max ({tau_max[j]:.1f} Nm)",
            )

        # Vertical ticks where ASIF modified the command.
        ax.fill_between(
            t,
            ax.get_ylim()[0],
            ax.get_ylim()[1],
            where=asif_mod,
            color="red",
            alpha=0.12,
            label="ASIF active",
        )

        ax.set_ylabel(f"Joint {j} (Nm)")
        ax.legend(fontsize=7, loc="upper right")

    axes_list[-1].set_xlabel("Time (s)")
    axes_list[0].set_title(f"{trial.name} — Torque Triplet")
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
            t, v_ctrl[:, j], color="steelblue", linewidth=0.8, label="v_ctrl", alpha=0.7
        )
        ax.plot(
            t, v_des[:, j], color="darkorange", linewidth=0.8, label="v_des", alpha=0.7
        )
        ax.plot(t, v_safe[:, j], color="green", linewidth=1.0, label="v_safe")

        if v_max is not None and np.isfinite(v_max[j]):
            ax.axhspan(
                -v_max[j],
                v_max[j],
                alpha=0.07,
                color="red",
                label=f"±v_max ({v_max[j]:.2f} rad/s)",
            )

        ax.fill_between(
            t,
            ax.get_ylim()[0],
            ax.get_ylim()[1],
            where=optik_mod,
            color="red",
            alpha=0.12,
            label="OptIK/rescale active",
        )

        ax.set_ylabel(f"Joint {j} (rad/s)")
        ax.legend(fontsize=7, loc="upper right")

    axes_list[-1].set_xlabel("Time (s)")
    axes_list[0].set_title(f"{trial.name} — Velocity Triplet")
    fig.tight_layout()
    return fig
