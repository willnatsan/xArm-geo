"""
Transient-response metrics: overshoot and settling time.

Settling-time band defaults match the plan specification:
    translational : 5 mm
    rotational    : 1 degree
    riemannian    : sqrt((5e-3)^2 + (pi/180)^2)  ~ 0.0181

"Never settled" is reported as float('nan').
"""

from __future__ import annotations

import math

import numpy as np

from xarm_geo_analysis.metrics.tracking import (
    riemannian_se3_error,
    rotational_geodesic_error,
    translational_error,
)
from xarm_geo_analysis.trial import Trial

# Default band radii.
_BAND_TRANS_M: float = 5e-3
_BAND_ROT_RAD: float = math.radians(1.0)
_BAND_RIEM: float = math.sqrt(_BAND_TRANS_M**2 + _BAND_ROT_RAD**2)


# ---------------------------------------------------------------------------
# Overshoot
# ---------------------------------------------------------------------------


def max_overshoot(
    trial: Trial,
    kind: str = "translational",
    w_trans: float = 1.0,
    w_rot: float = 1.0,
) -> float:
    """Peak error magnitude over the entire trial.

    This is always well-defined for tracking trajectories: it is simply the
    maximum of the chosen error time-series.

    Parameters
    ----------
    kind          : "translational" | "rotational" | "riemannian".
    w_trans, w_rot: weights used when kind == "riemannian".

    Returns
    -------
    Scalar float in the units of the chosen error (metres / radians).
    """
    err = _error_series(trial, kind, w_trans, w_rot)
    return float(np.max(err))


def post_target_overshoot(
    trial: Trial,
    kind: str = "translational",
    w_trans: float = 1.0,
    w_rot: float = 1.0,
    window_fraction: float = 0.5,
) -> float:
    """Peak error after the minimum-error point within the last ``window_fraction``.

    Intended for step-like phases (e.g. a setpoint trajectory) where a
    meaningful "past the target" overshoot can be identified.  The algorithm:
      1. Restrict attention to the last ``window_fraction`` of the trial.
      2. Find the index of minimum error within that window (the closest the
         robot got to the target).
      3. Report the maximum error from that index to the end of the window.

    Returns NaN if fewer than 3 samples are available in the window.
    """
    err = _error_series(trial, kind, w_trans, w_rot)
    n = len(err)
    start = max(0, int(n * (1.0 - window_fraction)))
    window = err[start:]

    if len(window) < 3:
        return float("nan")

    min_idx = int(np.argmin(window))
    post = window[min_idx:]
    return float(np.max(post)) if len(post) > 0 else float("nan")


# ---------------------------------------------------------------------------
# Settling time
# ---------------------------------------------------------------------------


def settling_time(
    trial: Trial,
    trans_band_m: float = _BAND_TRANS_M,
    rot_band_rad: float = _BAND_ROT_RAD,
    w_trans: float = 1.0,
    w_rot: float = 1.0,
) -> dict[str, float]:
    """Time after which each error remains within its band for the rest of the trial.

    Algorithm: find the *last* sample where the error exceeds the band.
    Settling time is t[i + 1].  If no sample exceeds the band, settling
    occurred at t[0] (settled from the start).  If every sample exceeds the
    band, NaN is returned (never settled).

    Parameters
    ----------
    trans_band_m  : translational band radius in metres (default 5 mm).
    rot_band_rad  : rotational band radius in radians (default 1 degree).
    w_trans, w_rot: weights for the combined Riemannian band
                    (band radius = sqrt((trans_band_m)^2 + (rot_band_rad)^2)).

    Returns
    -------
    dict with keys:
        "trans_s"  : translational settling time (s) or NaN.
        "rot_s"    : rotational settling time (s) or NaN.
        "riem_s"   : combined Riemannian settling time (s) or NaN.
    """
    t = trial.t()
    riem_band = math.sqrt(trans_band_m**2 + rot_band_rad**2)

    trans_s = _settling_from_series(translational_error(trial), t, trans_band_m)
    rot_s = _settling_from_series(rotational_geodesic_error(trial), t, rot_band_rad)
    riem_s = _settling_from_series(
        riemannian_se3_error(trial, w_trans, w_rot), t, riem_band
    )

    return {"trans_s": trans_s, "rot_s": rot_s, "riem_s": riem_s}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _settling_from_series(err: np.ndarray, t: np.ndarray, band: float) -> float:
    """Return settling time for a single error series and band radius."""
    exceeded = np.where(err > band)[0]
    if len(exceeded) == 0:
        # Never exceeded the band: settled immediately.
        return float(t[0])
    last_exceeded = int(exceeded[-1])
    if last_exceeded + 1 >= len(t):
        # The last sample still exceeds the band: never settled.
        return float("nan")
    return float(t[last_exceeded + 1])


def _error_series(
    trial: Trial,
    kind: str,
    w_trans: float,
    w_rot: float,
) -> np.ndarray:
    if kind == "translational":
        return translational_error(trial)
    elif kind == "rotational":
        return rotational_geodesic_error(trial)
    elif kind == "riemannian":
        return riemannian_se3_error(trial, w_trans, w_rot)
    else:
        raise ValueError(
            f"Unknown kind '{kind}'; expected translational|rotational|riemannian"
        )
