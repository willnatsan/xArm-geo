"""
Control-effort metrics.

Normalized effort integrals quantify how hard the robot works relative to
its joint limits.  NaN is returned when the required data is absent
(e.g. torque effort for a velocity-mode trial, or missing sidecar limits).
"""

from __future__ import annotations

import numpy as np

from xarm_geo_analysis.trial import Trial


# ---------------------------------------------------------------------------
# Integrated effort
# ---------------------------------------------------------------------------


def normalized_control_effort(trial: Trial) -> float:
    """Normalized torque effort integral: integral of sum_i (tau_i / tau_max_i)^2 dt.

    Uses tau_safe (the actually-applied torque) when available, falling back
    to tau_des.  Returns NaN for velocity-mode trials or when tau_max is absent.
    """
    tau_max = trial.tau_max
    if tau_max is None:
        return float("nan")

    tau = trial.tau_safe()
    if not np.any(np.isfinite(tau)):
        # Velocity-mode trial: torque triplet is blank.
        return float("nan")

    # Replace +inf limits with NaN so those joints don't contribute.
    tau_max_safe = np.where(np.isfinite(tau_max), tau_max, np.nan)

    # Normalized per-joint squared torque, summed across joints each tick.
    normalized = tau / tau_max_safe[np.newaxis, :]  # (N, dof)
    integrand = np.nansum(normalized**2, axis=1)  # (N,)

    return float(np.trapezoid(integrand, x=trial.t()))


def normalized_velocity_effort(trial: Trial) -> float:
    """Normalized velocity effort integral: integral of sum_i (v_i / v_max_i)^2 dt.

    The velocity-mode analogue of normalized_control_effort.
    Returns NaN for torque-mode trials or when v_max is absent.
    """
    v_max = trial.v_max
    if v_max is None:
        return float("nan")

    v = trial.v_safe()
    if not np.any(np.isfinite(v)):
        return float("nan")

    v_max_safe = np.where(np.isfinite(v_max), v_max, np.nan)
    normalized = v / v_max_safe[np.newaxis, :]
    integrand = np.nansum(normalized**2, axis=1)

    return float(np.trapezoid(integrand, x=trial.t()))


# ---------------------------------------------------------------------------
# Saturation fractions
# ---------------------------------------------------------------------------


def joint_torque_saturation_fraction(
    trial: Trial,
    threshold: float = 0.95,
) -> float:
    """Fraction of ticks where any |tau_i| / tau_max_i > threshold.

    Returns NaN when tau_max or torque data is absent.
    """
    tau_max = trial.tau_max
    if tau_max is None:
        return float("nan")

    tau = trial.tau_safe()
    if not np.any(np.isfinite(tau)):
        return float("nan")

    tau_max_safe = np.where(np.isfinite(tau_max), tau_max, np.nan)
    ratio = np.abs(tau) / tau_max_safe[np.newaxis, :]
    saturated = np.any(ratio > threshold, axis=1)
    return float(np.mean(saturated))


def joint_velocity_saturation_fraction(
    trial: Trial,
    threshold: float = 0.95,
) -> float:
    """Fraction of ticks where any |v_i| / v_max_i > threshold.

    Returns NaN when v_max or velocity data is absent.
    """
    v_max = trial.v_max
    if v_max is None:
        return float("nan")

    v = trial.v()
    if not np.any(np.isfinite(v)):
        return float("nan")

    v_max_safe = np.where(np.isfinite(v_max), v_max, np.nan)
    ratio = np.abs(v) / v_max_safe[np.newaxis, :]
    saturated = np.any(ratio > threshold, axis=1)
    return float(np.mean(saturated))


# ---------------------------------------------------------------------------
# Per-joint diagnostics
# ---------------------------------------------------------------------------


def per_joint_position_rmse(trial: Trial) -> np.ndarray:
    """RMSE of joint-position tracking error per joint (dof,).

    Requires q_ref columns to be present (joint-space phases).
    Returns an all-NaN array when q_ref is absent.
    """
    q = trial.q()
    q_ref = trial._vec_cols("q_ref")

    if not np.any(np.isfinite(q_ref)):
        return np.full(trial.dof, float("nan"))

    err = q - q_ref
    return np.sqrt(np.nanmean(err**2, axis=0))
