"""
Safety-layer metrics.

ASIF metrics apply to dynamic (torque-mode) trials.
OptIK / rescale metrics apply to kinematic (velocity-mode) trials.

Both return NaN cleanly when the relevant data is absent (e.g. ASIF metrics
on a velocity-mode trial where tau_* columns are blank).
"""

from __future__ import annotations

import numpy as np

from xarm_geo_analysis.trial import Trial


# ---------------------------------------------------------------------------
# ASIF (dynamic controllers)
# ---------------------------------------------------------------------------


def safety_intervention_integral(trial: Trial) -> float:
    """Integral of ||tau_des - tau_safe||_2 dt.

    This is the defining Phase 3 metric: it quantifies exactly how much the
    ASIF layer had to filter the geometric controller's desired torque.

    Returns 0.0 when ASIF was never active (tau_safe == tau_des).
    Returns NaN when torque data is absent (velocity-mode trial).
    """
    tau_des = trial.tau_des()
    tau_safe = trial.tau_safe()

    if not np.any(np.isfinite(tau_des)):
        return float("nan")

    delta = np.linalg.norm(tau_des - tau_safe, axis=1)  # (N,)
    return float(np.trapezoid(delta, x=trial.t()))


def asif_activity(trial: Trial) -> dict[str, object]:
    """Detailed breakdown of ASIF layer activity.

    Returns
    -------
    dict with keys:
        invocation_rate     : fraction of ticks where asif_invoked == True.
        modification_rate   : fraction of ticks where asif_modified == True.
        mean_delta_tau      : mean ||tau_safe - tau_des|| over invoked ticks.
        max_delta_tau       : maximum ||tau_safe - tau_des|| over invoked ticks.
        per_joint_mean_delta: (dof,) mean |tau_safe_i - tau_des_i| per joint.
        infeasible_count    : number of ticks where asif_status == 1 (INFEASIBLE).
        max_iters_count     : number of ticks where asif_status == 2 (MAX_ITERS).

    All values are NaN / 0 when the trial has no torque data.
    """
    nan_result: dict[str, object] = {
        "invocation_rate": float("nan"),
        "modification_rate": float("nan"),
        "mean_delta_tau": float("nan"),
        "max_delta_tau": float("nan"),
        "per_joint_mean_delta": np.full(trial.dof, float("nan")),
        "infeasible_count": 0,
        "max_iters_count": 0,
    }

    tau_des = trial.tau_des()
    if not np.any(np.isfinite(tau_des)):
        return nan_result

    invoked = trial.asif_invoked_mask()
    modified = trial.asif_modified_mask()

    tau_safe = trial.tau_safe()
    delta = np.linalg.norm(tau_des - tau_safe, axis=1)  # (N,)
    per_joint = np.abs(tau_des - tau_safe)  # (N, dof)

    invoked_mask = invoked & np.isfinite(delta)
    mean_delta = float(np.mean(delta[invoked_mask])) if invoked_mask.any() else 0.0
    max_delta = float(np.max(delta[invoked_mask])) if invoked_mask.any() else 0.0
    per_joint_mean = (
        np.mean(per_joint[invoked_mask], axis=0)
        if invoked_mask.any()
        else np.zeros(trial.dof)
    )

    # asif_status byte: 1 = INFEASIBLE, 2 = MAX_ITERS (matches ASIFStatus enum order).
    status = trial.df["asif_status"].to_numpy(dtype=float)
    infeasible_count = int(np.sum(status == 1))
    max_iters_count = int(np.sum(status == 2))

    return {
        "invocation_rate": float(np.mean(invoked)),
        "modification_rate": float(np.mean(modified)),
        "mean_delta_tau": mean_delta,
        "max_delta_tau": max_delta,
        "per_joint_mean_delta": per_joint_mean,
        "infeasible_count": infeasible_count,
        "max_iters_count": max_iters_count,
    }


# ---------------------------------------------------------------------------
# OptIK / velocity-rescale (kinematic controllers)
# ---------------------------------------------------------------------------


def kinematic_intervention_integral(trial: Trial) -> float:
    """Integral of ||v_des - v_safe||_2 dt.

    The velocity-mode analogue of safety_intervention_integral.  For
    KinematicJointControllerBase, v_safe reflects direction-preserving
    velocity-limit rescaling rather than a QP solve (see codebase note on
    the optik_modified repurposing).

    Returns 0.0 when OptIK / rescaling never changed the command.
    Returns NaN when velocity data is absent (torque-mode trial).
    """
    v_des = trial.v_des()
    v_safe = trial.v_safe()

    if not np.any(np.isfinite(v_des)):
        return float("nan")

    delta = np.linalg.norm(v_des - v_safe, axis=1)  # (N,)
    return float(np.trapezoid(delta, x=trial.t()))


def optik_activity(trial: Trial) -> dict[str, object]:
    """Detailed breakdown of OptIK / velocity-rescale layer activity.

    Mirrors asif_activity() for kinematic controllers.

    Note: for KinematicJointControllerBase, optik_invoked is always False and
    optik_modified reflects velocity-limit rescaling rather than QP infeasibility
    (see the controller base comment on this repurposing).

    Returns
    -------
    dict with keys:
        invocation_rate     : fraction of ticks where optik_invoked == True.
        modification_rate   : fraction of ticks where optik_modified == True.
        mean_delta_v        : mean ||v_safe - v_des|| over modified ticks.
        max_delta_v         : maximum ||v_safe - v_des|| over modified ticks.
        per_joint_mean_delta: (dof,) mean |v_safe_i - v_des_i| per joint.
        infeasible_count    : ticks where optik_status == 1.
        max_iters_count     : ticks where optik_status == 2.
    """
    nan_result: dict[str, object] = {
        "invocation_rate": float("nan"),
        "modification_rate": float("nan"),
        "mean_delta_v": float("nan"),
        "max_delta_v": float("nan"),
        "per_joint_mean_delta": np.full(trial.dof, float("nan")),
        "infeasible_count": 0,
        "max_iters_count": 0,
    }

    v_des = trial.v_des()
    if not np.any(np.isfinite(v_des)):
        return nan_result

    invoked = trial.optik_invoked_mask()
    modified = trial.optik_modified_mask()

    v_safe = trial.v_safe()
    delta = np.linalg.norm(v_des - v_safe, axis=1)
    per_joint = np.abs(v_des - v_safe)

    modified_mask = modified & np.isfinite(delta)
    mean_delta = float(np.mean(delta[modified_mask])) if modified_mask.any() else 0.0
    max_delta = float(np.max(delta[modified_mask])) if modified_mask.any() else 0.0
    per_joint_mean = (
        np.mean(per_joint[modified_mask], axis=0)
        if modified_mask.any()
        else np.zeros(trial.dof)
    )

    status = trial.df["optik_status"].to_numpy(dtype=float)
    infeasible_count = int(np.sum(status == 1))
    max_iters_count = int(np.sum(status == 2))

    return {
        "invocation_rate": float(np.mean(invoked)),
        "modification_rate": float(np.mean(modified)),
        "mean_delta_v": mean_delta,
        "max_delta_v": max_delta,
        "per_joint_mean_delta": per_joint_mean,
        "infeasible_count": infeasible_count,
        "max_iters_count": max_iters_count,
    }
