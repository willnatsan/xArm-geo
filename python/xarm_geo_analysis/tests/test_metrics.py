"""
Synthetic unit tests for all metric functions.

Each test builds a minimal Trial from synthetic numpy data so that the
expected values are analytically known.  No CSV files on disk are required.
"""

from __future__ import annotations

import math
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.transform import Rotation

from xarm_geo_analysis.trial import Trial, Experiment


# ---------------------------------------------------------------------------
# Helpers: build a synthetic Trial
# ---------------------------------------------------------------------------

DOF = 6
N = 500  # number of samples
DT = 0.002  # seconds per tick
T = np.linspace(0.0, (N - 1) * DT, N)

TAU_MAX = np.array([50.0, 50.0, 30.0, 30.0, 20.0, 20.0])
V_MAX = np.array([3.14, 3.14, 3.14, 3.14, 6.28, 6.28])


def _identity_rotation_quat(n: int) -> np.ndarray:
    """(n, 4) array of identity quaternions in xyzw convention."""
    q = np.zeros((n, 4))
    q[:, 3] = 1.0  # w = 1
    return q


def _rotz_quat(angles: np.ndarray) -> np.ndarray:
    """(N,) angles -> (N, 4) xyzw quaternions for Rz(angle)."""
    # Use from_rotvec with explicit (N, 3) input; unambiguous across scipy versions.
    rotvecs = np.zeros((len(angles), 3))
    rotvecs[:, 2] = angles
    return Rotation.from_rotvec(rotvecs).as_quat()


def _build_trial(
    p_actual: np.ndarray,
    p_target: np.ndarray,
    q_actual: np.ndarray,
    q_target: np.ndarray,
    tau_des: np.ndarray | None = None,
    tau_safe: np.ndarray | None = None,
    tau_ctrl: np.ndarray | None = None,
    v_ctrl: np.ndarray | None = None,
    v_des: np.ndarray | None = None,
    v_safe: np.ndarray | None = None,
    asif_invoked: np.ndarray | None = None,
    asif_modified: np.ndarray | None = None,
    optik_invoked: np.ndarray | None = None,
    optik_modified: np.ndarray | None = None,
    dof: int = DOF,
) -> Trial:
    """Assemble a minimal Trial dataframe from arrays."""
    n = len(p_actual)
    data: dict[str, np.ndarray | list] = {}

    data["t"] = T[:n]
    data["tick"] = np.arange(n)

    # Joint state (synthetic zeros).
    for i in range(dof):
        data[f"q.{i}"] = np.zeros(n)
        data[f"v.{i}"] = np.zeros(n)
        data[f"tau_measured.{i}"] = np.zeros(n)

    # Joint refs (absent).
    for prefix in ("q_ref", "v_ref", "a_ref"):
        for i in range(dof):
            data[f"{prefix}.{i}"] = np.full(n, np.nan)

    # Task-space actual.
    data["ee_actual.x"] = p_actual[:, 0]
    data["ee_actual.y"] = p_actual[:, 1]
    data["ee_actual.z"] = p_actual[:, 2]
    data["ee_actual.qx"] = q_actual[:, 0]
    data["ee_actual.qy"] = q_actual[:, 1]
    data["ee_actual.qz"] = q_actual[:, 2]
    data["ee_actual.qw"] = q_actual[:, 3]

    # Task-space target.
    data["ee_target.x"] = p_target[:, 0]
    data["ee_target.y"] = p_target[:, 1]
    data["ee_target.z"] = p_target[:, 2]
    data["ee_target.qx"] = q_target[:, 0]
    data["ee_target.qy"] = q_target[:, 1]
    data["ee_target.qz"] = q_target[:, 2]
    data["ee_target.qw"] = q_target[:, 3]

    for c in ("vx", "vy", "vz", "wx", "wy", "wz"):
        data[f"ee_twist_actual.{c}"] = np.zeros(n)
        data[f"ee_twist_target.{c}"] = np.zeros(n)

    def _fill_triplet(prefix, arr):
        if arr is not None:
            for i in range(dof):
                data[f"{prefix}.{i}"] = arr[:, i]
        else:
            for i in range(dof):
                data[f"{prefix}.{i}"] = np.full(n, np.nan)

    _fill_triplet("tau_ctrl", tau_ctrl)
    _fill_triplet("tau_des", tau_des)
    _fill_triplet("tau_safe", tau_safe)
    _fill_triplet("v_ctrl", v_ctrl)
    _fill_triplet("v_des", v_des)
    _fill_triplet("v_safe", v_safe)

    data["controller_status"] = np.zeros(n, dtype=int)
    data["asif_status"] = np.full(n, 255)
    data["asif_invoked"] = (
        asif_invoked if asif_invoked is not None else np.zeros(n, dtype=bool)
    )
    data["asif_modified"] = (
        asif_modified if asif_modified is not None else np.zeros(n, dtype=bool)
    )
    data["optik_status"] = np.full(n, 255)
    data["optik_invoked"] = (
        optik_invoked if optik_invoked is not None else np.zeros(n, dtype=bool)
    )
    data["optik_modified"] = (
        optik_modified if optik_modified is not None else np.zeros(n, dtype=bool)
    )

    df = pd.DataFrame(data)
    meta = {
        "trial_name": "synthetic",
        "dof": dof,
        "dt_nominal_s": DT,
        "tau_max": TAU_MAX.tolist(),
        "v_max": V_MAX.tolist(),
        "q_min": (np.full(dof, -math.pi)).tolist(),
        "q_max": (np.full(dof, math.pi)).tolist(),
    }
    return Trial(name="synthetic", df=df, meta=meta)


# ---------------------------------------------------------------------------
# Tracking metric tests
# ---------------------------------------------------------------------------


class TestTranslationalError:
    def test_zero_error(self):
        """Identical actual and target => zero error."""
        from xarm_geo_analysis.metrics.tracking import (
            translational_error,
            translational_rmse,
        )

        p = np.zeros((N, 3))
        q = _identity_rotation_quat(N)
        trial = _build_trial(p, p, q, q)

        err = translational_error(trial)
        assert err.shape == (N,)
        np.testing.assert_allclose(err, 0.0, atol=1e-12)
        assert translational_rmse(trial) == pytest.approx(0.0, abs=1e-12)

    def test_constant_offset(self):
        """Constant 10 mm X-offset => every error == 0.01 m."""
        from xarm_geo_analysis.metrics.tracking import (
            translational_error,
            translational_rmse,
        )

        p_actual = np.zeros((N, 3))
        p_target = np.zeros((N, 3))
        p_target[:, 0] = 0.01  # 10 mm offset

        q = _identity_rotation_quat(N)
        trial = _build_trial(p_actual, p_target, q, q)

        err = translational_error(trial)
        np.testing.assert_allclose(err, 0.01, rtol=1e-10)
        assert translational_rmse(trial) == pytest.approx(0.01, rel=1e-10)


class TestRotationalGeodesicError:
    def test_zero_error(self):
        """Identical rotations => zero geodesic error."""
        from xarm_geo_analysis.metrics.tracking import rotational_geodesic_error

        p = np.zeros((N, 3))
        q = _identity_rotation_quat(N)
        trial = _build_trial(p, p, q, q)

        err = rotational_geodesic_error(trial)
        np.testing.assert_allclose(err, 0.0, atol=1e-12)

    def test_constant_rotation_error(self):
        """Rz(30°) actual vs identity target => error == 30° everywhere."""
        from xarm_geo_analysis.metrics.tracking import rotational_geodesic_error

        angle_rad = math.radians(30.0)
        p = np.zeros((N, 3))
        q_actual = _rotz_quat(np.full(N, angle_rad))
        q_target = _identity_rotation_quat(N)
        trial = _build_trial(p, p, q_actual, q_target)

        err = rotational_geodesic_error(trial)
        np.testing.assert_allclose(err, angle_rad, rtol=1e-8)

    def test_growing_rotation_error(self):
        """Linearly growing angle => error grows linearly."""
        from xarm_geo_analysis.metrics.tracking import rotational_geodesic_error

        angles = np.linspace(0.0, math.radians(90.0), N)
        p = np.zeros((N, 3))
        q_actual = _rotz_quat(angles)
        q_target = _identity_rotation_quat(N)
        trial = _build_trial(p, p, q_actual, q_target)

        err = rotational_geodesic_error(trial)
        np.testing.assert_allclose(err, angles, rtol=1e-8)


class TestRiemannianSE3Error:
    def test_zero_error(self):
        """Zero pose error => zero Riemannian error."""
        from xarm_geo_analysis.metrics.tracking import riemannian_se3_error

        p = np.zeros((N, 3))
        q = _identity_rotation_quat(N)
        trial = _build_trial(p, p, q, q)

        err = riemannian_se3_error(trial)
        np.testing.assert_allclose(err, 0.0, atol=1e-12)

    def test_pure_translation(self):
        """Pure translation error: Riemannian error == w_trans * ||t_err||."""
        from xarm_geo_analysis.metrics.tracking import riemannian_se3_error

        p_actual = np.zeros((N, 3))
        p_target = np.zeros((N, 3))
        p_target[:, 0] = 0.05  # 5 cm offset

        q = _identity_rotation_quat(N)
        trial = _build_trial(p_actual, p_target, q, q)

        err = riemannian_se3_error(trial, w_trans=1.0, w_rot=1.0)
        np.testing.assert_allclose(err, 0.05, rtol=1e-8)

    def test_weights_scale_components(self):
        """Doubling w_rot doubles the rotational contribution."""
        from xarm_geo_analysis.metrics.tracking import riemannian_se3_error

        p = np.zeros((N, 3))
        angle = math.radians(10.0)
        q_actual = _rotz_quat(np.full(N, angle))
        q_target = _identity_rotation_quat(N)
        trial = _build_trial(p, p, q_actual, q_target)

        e1 = riemannian_se3_error(trial, w_trans=1.0, w_rot=1.0)
        e2 = riemannian_se3_error(trial, w_trans=1.0, w_rot=2.0)
        # e1 = angle (purely rotational), e2 = 2 * angle.
        np.testing.assert_allclose(e1, angle, rtol=1e-8)
        np.testing.assert_allclose(e2, 2.0 * angle, rtol=1e-8)


class TestSteadyStateRMSE:
    def test_last_fraction_zero_error(self):
        """If the last 20% has zero error, steady-state RMSE is zero."""
        from xarm_geo_analysis.metrics.tracking import steady_state_rmse

        p_actual = np.zeros((N, 3))
        p_target = np.zeros((N, 3))
        # First 80%: 10 mm error; last 20%: zero.
        split = int(N * 0.8)
        p_target[:split, 0] = 0.01

        q = _identity_rotation_quat(N)
        trial = _build_trial(p_actual, p_target, q, q)

        ss = steady_state_rmse(trial, last_fraction=0.2, kind="translational")
        assert ss == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Transient metric tests
# ---------------------------------------------------------------------------


class TestSettlingTime:
    def test_never_exceeded_band(self):
        """Error always within band => settled at t[0]."""
        from xarm_geo_analysis.metrics.transient import settling_time

        p = np.zeros((N, 3))  # zero error
        q = _identity_rotation_quat(N)
        trial = _build_trial(p, p, q, q)

        st = settling_time(trial)
        assert st["trans_s"] == pytest.approx(T[0], abs=1e-12)
        assert st["rot_s"] == pytest.approx(T[0], abs=1e-12)

    def test_never_settled(self):
        """Error always outside band => NaN."""
        from xarm_geo_analysis.metrics.transient import settling_time

        p_actual = np.zeros((N, 3))
        p_target = np.full((N, 3), 1.0)  # 1 m offset >> 5 mm band
        q = _identity_rotation_quat(N)
        trial = _build_trial(p_actual, p_target, q, q)

        st = settling_time(trial)
        assert math.isnan(st["trans_s"])

    def test_settles_at_known_time(self):
        """Error drops below band at a known index => settling time is t[index+1]."""
        from xarm_geo_analysis.metrics.transient import settling_time, _BAND_TRANS_M

        p_actual = np.zeros((N, 3))
        p_target = np.zeros((N, 3))
        settle_idx = N // 2
        # Before settle_idx: 10 mm (above band). From settle_idx onward: 0.
        p_target[:settle_idx, 0] = 0.01
        q = _identity_rotation_quat(N)
        trial = _build_trial(p_actual, p_target, q, q)

        st = settling_time(trial, trans_band_m=_BAND_TRANS_M)
        expected_t = T[
            settle_idx
        ]  # t[last_exceeded + 1] where last_exceeded == settle_idx - 1
        assert st["trans_s"] == pytest.approx(expected_t, rel=1e-6)


class TestMaxOvershoot:
    def test_equals_peak_error(self):
        """max_overshoot == maximum of the error series."""
        from xarm_geo_analysis.metrics.transient import max_overshoot
        from xarm_geo_analysis.metrics.tracking import translational_error

        p_actual = np.zeros((N, 3))
        p_target = np.zeros((N, 3))
        p_target[N // 3, 0] = 0.08  # spike at one point

        q = _identity_rotation_quat(N)
        trial = _build_trial(p_actual, p_target, q, q)

        assert max_overshoot(trial, "translational") == pytest.approx(
            np.max(translational_error(trial)), rel=1e-10
        )


# ---------------------------------------------------------------------------
# Effort metric tests
# ---------------------------------------------------------------------------


class TestNormalizedEffort:
    def _torque_trial(self, tau_val: float) -> Trial:
        """Trial where tau_safe == tau_val for all joints and ticks."""
        p = np.zeros((N, 3))
        q = _identity_rotation_quat(N)
        tau = np.full((N, DOF), tau_val)
        return _build_trial(p, p, q, q, tau_ctrl=tau, tau_des=tau, tau_safe=tau)

    def test_zero_torque(self):
        """Zero torque => zero effort."""
        from xarm_geo_analysis.metrics.effort import normalized_control_effort

        trial = self._torque_trial(0.0)
        assert normalized_control_effort(trial) == pytest.approx(0.0, abs=1e-12)

    def test_constant_half_limit(self):
        """tau_i == 0.5 * tau_max_i for all joints and ticks.

        Integrand = sum_i (0.5)^2 = dof * 0.25.
        Integral  = integrand * total_time.
        """
        from xarm_geo_analysis.metrics.effort import normalized_control_effort

        tau = np.column_stack([0.5 * TAU_MAX[i] * np.ones(N) for i in range(DOF)])
        p = np.zeros((N, 3))
        q = _identity_rotation_quat(N)
        trial = _build_trial(p, p, q, q, tau_ctrl=tau, tau_des=tau, tau_safe=tau)

        expected_integrand = DOF * 0.25
        total_time = T[-1] - T[0]
        expected = expected_integrand * total_time

        result = normalized_control_effort(trial)
        assert result == pytest.approx(expected, rel=1e-3)

    def test_velocity_mode_returns_nan(self):
        """No torque data => NaN."""
        from xarm_geo_analysis.metrics.effort import normalized_control_effort

        p = np.zeros((N, 3))
        q = _identity_rotation_quat(N)
        trial = _build_trial(p, p, q, q)  # all tau columns are NaN
        assert math.isnan(normalized_control_effort(trial))


# ---------------------------------------------------------------------------
# Safety metric tests
# ---------------------------------------------------------------------------


class TestSafetyInterventionIntegral:
    def test_no_asif_intervention(self):
        """tau_safe == tau_des => integral is 0."""
        from xarm_geo_analysis.metrics.safety import safety_intervention_integral

        p = np.zeros((N, 3))
        q = _identity_rotation_quat(N)
        tau = np.ones((N, DOF)) * 5.0
        trial = _build_trial(p, p, q, q, tau_ctrl=tau, tau_des=tau, tau_safe=tau)

        assert safety_intervention_integral(trial) == pytest.approx(0.0, abs=1e-10)

    def test_constant_intervention(self):
        """Constant ||tau_des - tau_safe||_2 = delta => integral == delta * T."""
        from xarm_geo_analysis.metrics.safety import safety_intervention_integral

        p = np.zeros((N, 3))
        q = _identity_rotation_quat(N)
        tau_des = np.ones((N, DOF)) * 10.0
        tau_safe = np.ones((N, DOF)) * 8.0  # constant 2 Nm difference per joint
        delta = np.linalg.norm(tau_des[0] - tau_safe[0])  # sqrt(DOF) * 2
        trial = _build_trial(
            p, p, q, q, tau_ctrl=tau_des, tau_des=tau_des, tau_safe=tau_safe
        )

        total_time = T[-1] - T[0]
        expected = delta * total_time
        assert safety_intervention_integral(trial) == pytest.approx(expected, rel=1e-3)

    def test_velocity_mode_returns_nan(self):
        """No torque data => NaN."""
        from xarm_geo_analysis.metrics.safety import safety_intervention_integral

        p = np.zeros((N, 3))
        q = _identity_rotation_quat(N)
        trial = _build_trial(p, p, q, q)
        assert math.isnan(safety_intervention_integral(trial))


class TestKinematicInterventionIntegral:
    def test_no_intervention(self):
        """v_des == v_safe => integral is 0."""
        from xarm_geo_analysis.metrics.safety import kinematic_intervention_integral

        p = np.zeros((N, 3))
        q = _identity_rotation_quat(N)
        v = np.ones((N, DOF)) * 0.5
        trial = _build_trial(p, p, q, q, v_ctrl=v, v_des=v, v_safe=v)

        assert kinematic_intervention_integral(trial) == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# Trial / Experiment loader tests
# ---------------------------------------------------------------------------


class TestTrialLoad:
    def _write_trial_csv(self, tmp_path: Path) -> Path:
        """Write a minimal valid CSV + sidecar and return the CSV path."""
        p = np.zeros((N, 3))
        q = _identity_rotation_quat(N)
        trial = _build_trial(p, p, q, q)

        csv_path = tmp_path / "trial.csv"
        trial.df.to_csv(csv_path, index=False)

        sidecar = tmp_path / "trial.csv.meta.json"
        sidecar.write_text(json.dumps(trial.meta))

        return csv_path

    def test_roundtrip(self, tmp_path):
        """Saved and re-loaded trial has the same shape and metadata."""
        from xarm_geo_analysis.trial import Trial

        csv_path = self._write_trial_csv(tmp_path)
        loaded = Trial.load(csv_path)

        assert loaded.dof == DOF
        assert len(loaded.df) == N
        assert loaded.dt == pytest.approx(DT, rel=1e-6)
        np.testing.assert_allclose(loaded.tau_max, TAU_MAX)

    def test_experiment_load_dir(self, tmp_path):

        for i in range(3):
            p = np.zeros((N, 3))
            q = _identity_rotation_quat(N)
            trial = _build_trial(p, p, q, q)
            trial.df.to_csv(tmp_path / f"trial_{i}.csv", index=False)

        exp = Experiment.load_dir(tmp_path)
        assert len(exp) == 3


# ---------------------------------------------------------------------------
# New metric tests
# ---------------------------------------------------------------------------


def _build_trial_with_extras(
    e_I: np.ndarray | None = None,
    obstacle_dist: np.ndarray | None = None,
    v_ctrl: np.ndarray | None = None,
    v_des: np.ndarray | None = None,
    v_safe: np.ndarray | None = None,
    tau_ctrl: np.ndarray | None = None,
    tau_des: np.ndarray | None = None,
    tau_safe: np.ndarray | None = None,
) -> Trial:
    """Minimal trial builder that also accepts e_I and obstacle distance columns."""
    p = np.zeros((N, 3))
    q = _identity_rotation_quat(N)
    trial = _build_trial(
        p,
        p,
        q,
        q,
        v_ctrl=v_ctrl,
        v_des=v_des,
        v_safe=v_safe,
        tau_ctrl=tau_ctrl,
        tau_des=tau_des,
        tau_safe=tau_safe,
    )
    if e_I is not None:
        for j, c in enumerate(
            ["e_I.vx", "e_I.vy", "e_I.vz", "e_I.wx", "e_I.wy", "e_I.wz"]
        ):
            trial.df[c] = e_I[:, j]
    if obstacle_dist is not None:
        trial.df["obstacle_distance_min"] = obstacle_dist
    return trial


class TestErrorTwist:
    def test_zero_twist_zero_error(self):
        """Both twists zero => error twist is zero."""
        from xarm_geo_analysis.metrics.tracking import error_twist, error_twist_norm

        p = np.zeros((N, 3))
        q = _identity_rotation_quat(N)
        trial = _build_trial(p, p, q, q)
        # twist columns default to zeros in _build_trial
        xi_e = error_twist(trial)
        assert xi_e.shape == (N, 6)
        np.testing.assert_allclose(xi_e, 0.0, atol=1e-12)
        np.testing.assert_allclose(error_twist_norm(trial), 0.0, atol=1e-12)

    def test_nonzero_actual_twist(self):
        """Constant actual twist, zero target => error == actual."""
        from xarm_geo_analysis.metrics.tracking import error_twist

        p = np.zeros((N, 3))
        q = _identity_rotation_quat(N)
        trial = _build_trial(p, p, q, q)
        # Set actual twist to a constant value; target remains zero.
        for j, c in enumerate(
            [
                "ee_twist_actual.vx",
                "ee_twist_actual.vy",
                "ee_twist_actual.vz",
                "ee_twist_actual.wx",
                "ee_twist_actual.wy",
                "ee_twist_actual.wz",
            ]
        ):
            trial.df[c] = float(j + 1)

        xi_e = error_twist(trial)
        for j in range(6):
            np.testing.assert_allclose(xi_e[:, j], float(j + 1), atol=1e-10)


class TestCommandNormSeries:
    def test_velocity_mode_constant(self):
        """Constant v_safe of ones => norm == sqrt(dof) for all ticks."""
        from xarm_geo_analysis.metrics.effort import command_norm_series

        v = np.ones((N, DOF))
        trial = _build_trial_with_extras(v_ctrl=v, v_des=v, v_safe=v)
        norms = command_norm_series(trial, kind="velocity")
        assert norms.shape == (N,)
        np.testing.assert_allclose(norms, math.sqrt(DOF), rtol=1e-10)

    def test_torque_mode_zero(self):
        """Zero tau_safe => norm == 0."""
        from xarm_geo_analysis.metrics.effort import command_norm_series

        tau = np.zeros((N, DOF))
        trial = _build_trial_with_extras(tau_ctrl=tau, tau_des=tau, tau_safe=tau)
        norms = command_norm_series(trial, kind="torque")
        np.testing.assert_allclose(norms, 0.0, atol=1e-12)

    def test_missing_triplet_returns_nan(self):
        """No command data => all NaN."""
        from xarm_geo_analysis.metrics.effort import command_norm_series

        trial = _build_trial_with_extras()  # all triplets absent
        norms = command_norm_series(trial, kind="velocity")
        assert np.all(np.isnan(norms))


class TestIntegratorStateNorm:
    def test_blank_columns_nan(self):
        """No e_I columns => all NaN."""
        from xarm_geo_analysis.metrics.transient import integrator_state_norm

        trial = _build_trial_with_extras()
        norms = integrator_state_norm(trial)
        assert np.all(np.isnan(norms))

    def test_constant_e_I(self):
        """Constant e_I = [1, 0, ...] => norm == 1 everywhere."""
        from xarm_geo_analysis.metrics.transient import integrator_state_norm

        e_I = np.zeros((N, 6))
        e_I[:, 0] = 1.0
        trial = _build_trial_with_extras(e_I=e_I)
        norms = integrator_state_norm(trial)
        assert norms.shape == (N,)
        np.testing.assert_allclose(norms, 1.0, rtol=1e-10)


class TestMinDistanceSeries:
    def test_missing_column_nan(self):
        """No obstacle column => all NaN."""
        from xarm_geo_analysis.metrics.safety import min_distance_series

        trial = _build_trial_with_extras()
        d = min_distance_series(trial)
        assert np.all(np.isnan(d))

    def test_constant_distance(self):
        """Constant 0.1 m distance => series is constant 0.1."""
        from xarm_geo_analysis.metrics.safety import min_distance_series

        d_val = np.full(N, 0.1)
        trial = _build_trial_with_extras(obstacle_dist=d_val)
        d = min_distance_series(trial)
        np.testing.assert_allclose(d, 0.1, rtol=1e-10)


class TestInterventionMagnitudeSeries:
    def test_no_intervention_velocity(self):
        """v_des == v_safe => series is zero."""
        from xarm_geo_analysis.metrics.safety import intervention_magnitude_series

        v = np.ones((N, DOF))
        trial = _build_trial_with_extras(v_ctrl=v, v_des=v, v_safe=v)
        delta = intervention_magnitude_series(trial, kind="velocity")
        np.testing.assert_allclose(delta, 0.0, atol=1e-12)

    def test_constant_intervention_torque(self):
        """Constant 1 Nm difference per joint => norm == sqrt(dof)."""
        from xarm_geo_analysis.metrics.safety import intervention_magnitude_series

        tau_des = np.ones((N, DOF)) * 5.0
        tau_safe = np.ones((N, DOF)) * 4.0
        trial = _build_trial_with_extras(
            tau_ctrl=tau_des, tau_des=tau_des, tau_safe=tau_safe
        )
        delta = intervention_magnitude_series(trial, kind="torque")
        np.testing.assert_allclose(delta, math.sqrt(DOF), rtol=1e-10)

    def test_missing_returns_nan(self):
        """No torque data => all NaN."""
        from xarm_geo_analysis.metrics.safety import intervention_magnitude_series

        trial = _build_trial_with_extras()
        delta = intervention_magnitude_series(trial, kind="torque")
        assert np.all(np.isnan(delta))
