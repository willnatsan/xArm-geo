"""
Tracking-error metrics.

All functions accept a Trial and return either:
  - a 1-D numpy array (instantaneous time-series), or
  - a scalar float (summary statistic).

Riemannian SE(3) error uses the right log-map:
    xi = Log( g_actual^{-1} * g_target )
with configurable diagonal weights on the translational and rotational parts.

No Euler angles are used anywhere in this module.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from xarm_geo_analysis.trial import Trial


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _se3_log_magnitude(
    p_err: np.ndarray,
    q_err: np.ndarray,
    w_trans: float = 1.0,
    w_rot: float = 1.0,
) -> np.ndarray:
    """Compute ||Log(g_err)||_W per row.

    Uses the closed-form SO(3) log to avoid the cost of 4x4 matrix logm:
        Log(g) = (xi_t, xi_R) where xi_R = Log_SO3(R) and xi_t is derived
        from the SE(3) left-trivialized formula.  For pure-error analysis
        (where we only need the magnitude and not the full twist direction)
        we use the simpler weighted Euclidean norm of the components:
            ||Log(g)||_W = sqrt( w_trans^2 ||t||^2 + w_rot^2 ||xi_R||^2 )

    This is not the full SE(3) screw magnitude (which couples t and xi_R
    through the T matrix), but it is the standard metric used in robotics
    tracking-error literature and matches the plan specification.

    Parameters
    ----------
    p_err : (N, 3) translational part of the relative pose.
    q_err : (N, 4) quaternion (xyzw) of the relative rotation.
    w_trans, w_rot : scalar weights.

    Returns
    -------
    (N,) array of weighted magnitudes.
    """
    # Translational magnitude (simple Euclidean).
    t_norm = np.linalg.norm(p_err, axis=1)  # (N,)

    # Rotational magnitude via SO(3) log-map: ||Log(R)|| = angle of rotation.
    rotvec = Rotation.from_quat(q_err).as_rotvec()  # (N, 3)
    r_norm = np.linalg.norm(rotvec, axis=1)  # (N,)

    return np.sqrt((w_trans * t_norm) ** 2 + (w_rot * r_norm) ** 2)


def _relative_pose(trial: Trial) -> tuple[np.ndarray, np.ndarray]:
    """Return (p_err, q_err) for g_actual^{-1} * g_target.

    g_err = g_actual^{-1} * g_target
      R_err = R_actual^T * R_target
      t_err = R_actual^T * (t_target - t_actual)

    Returns
    -------
    p_err : (N, 3)
    q_err : (N, 4) xyzw
    """
    p_a = trial.p_actual()  # (N, 3)
    p_t = trial.p_target()  # (N, 3)
    R_a = trial.R_actual()  # Rotation(N,)
    R_t = trial.R_target()  # Rotation(N,)

    R_err = R_a.inv() * R_t
    p_err = R_a.inv().apply(p_t - p_a)

    return p_err, R_err.as_quat()


# ---------------------------------------------------------------------------
# Instantaneous error time-series
# ---------------------------------------------------------------------------


def translational_error(trial: Trial) -> np.ndarray:
    """||p_actual - p_target||_2 at each tick.

    Returns
    -------
    (N,) array in metres.
    """
    return np.linalg.norm(trial.p_actual() - trial.p_target(), axis=1)


def rotational_geodesic_error(trial: Trial) -> np.ndarray:
    """Geodesic (log-map) rotation error at each tick.

    ||Log(R_actual^T * R_target)||  =  angle of the relative rotation  (radians).

    Returns
    -------
    (N,) array in radians.
    """
    R_a = trial.R_actual()
    R_t = trial.R_target()
    R_err = R_a.inv() * R_t
    rotvec = R_err.as_rotvec()  # (N, 3)
    return np.linalg.norm(rotvec, axis=1)


def riemannian_se3_error(
    trial: Trial,
    w_trans: float = 1.0,
    w_rot: float = 1.0,
) -> np.ndarray:
    """Weighted Riemannian SE(3) error at each tick.

    ||Log(g_actual^{-1} * g_target)||_W
      = sqrt( (w_trans * ||t_err||)^2 + (w_rot * ||Log(R_err)||)^2 )

    Parameters
    ----------
    w_trans : weight on the translational component (default 1.0, units m).
    w_rot   : weight on the rotational component (default 1.0, units rad).
              With both at 1.0, the combined settling band of ~0.0181 follows
              from sqrt((5e-3)^2 + (pi/180)^2).

    Returns
    -------
    (N,) array.
    """
    p_err, q_err = _relative_pose(trial)
    return _se3_log_magnitude(p_err, q_err, w_trans=w_trans, w_rot=w_rot)


# ---------------------------------------------------------------------------
# Scalar summaries
# ---------------------------------------------------------------------------


def translational_rmse(trial: Trial) -> float:
    """RMSE of translational error over the full trial (metres)."""
    return float(np.sqrt(np.mean(translational_error(trial) ** 2)))


def rotational_rmse(trial: Trial) -> float:
    """RMSE of geodesic rotational error over the full trial (radians)."""
    return float(np.sqrt(np.mean(rotational_geodesic_error(trial) ** 2)))


def riemannian_rmse(
    trial: Trial,
    w_trans: float = 1.0,
    w_rot: float = 1.0,
) -> float:
    """RMSE of the weighted Riemannian SE(3) error over the full trial."""
    return float(np.sqrt(np.mean(riemannian_se3_error(trial, w_trans, w_rot) ** 2)))


def steady_state_rmse(
    trial: Trial,
    last_fraction: float = 0.2,
    kind: str = "translational",
    w_trans: float = 1.0,
    w_rot: float = 1.0,
) -> float:
    """RMSE computed over the last ``last_fraction`` of the trial.

    Parameters
    ----------
    last_fraction : fraction of total samples to include (default 0.2 = last 20%).
    kind          : "translational" | "rotational" | "riemannian".
    w_trans, w_rot: weights used when kind == "riemannian".

    Returns
    -------
    Scalar float; NaN if the trial has fewer than 2 samples.
    """
    n = len(trial.df)
    if n < 2:
        return float("nan")

    start = max(0, int(n * (1.0 - last_fraction)))

    if kind == "translational":
        err = translational_error(trial)[start:]
    elif kind == "rotational":
        err = rotational_geodesic_error(trial)[start:]
    elif kind == "riemannian":
        err = riemannian_se3_error(trial, w_trans, w_rot)[start:]
    else:
        raise ValueError(
            f"Unknown kind '{kind}'; expected translational|rotational|riemannian"
        )

    return float(np.sqrt(np.mean(err**2)))
