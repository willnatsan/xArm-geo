"""
Scalar summary table: one row per trial, one column per metric.

``summarise(trial)`` returns a flat dict of every scalar metric.
``summarise_experiment(exp)`` returns a pandas DataFrame with one row per trial.
``to_markdown(df)`` and ``to_csv(df, path)`` handle export.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from xarm_geo_analysis.metrics.effort import (
    joint_torque_saturation_fraction,
    joint_velocity_saturation_fraction,
    normalized_control_effort,
    normalized_velocity_effort,
)
from xarm_geo_analysis.metrics.safety import (
    asif_activity,
    kinematic_intervention_integral,
    optik_activity,
    safety_intervention_integral,
)
from xarm_geo_analysis.metrics.tracking import (
    riemannian_rmse,
    rotational_rmse,
    steady_state_rmse,
    translational_rmse,
)
from xarm_geo_analysis.metrics.transient import (
    max_overshoot,
    post_target_overshoot,
    settling_time,
)
from xarm_geo_analysis.trial import Experiment, Trial


def summarise(trial: Trial) -> dict[str, float]:
    """Compute every scalar metric for one trial and return as a flat dict.

    Vector-valued sub-metrics (e.g. per_joint_mean_delta) are expanded to
    ``<key>.<joint_idx>`` entries.  NaN is used where the metric is not
    applicable for the trial's mode.
    """
    row: dict[str, float] = {}

    # --- Tracking ---
    row["trans_rmse_m"] = translational_rmse(trial)
    row["rot_rmse_rad"] = rotational_rmse(trial)
    row["riem_rmse"] = riemannian_rmse(trial)

    row["ss_trans_rmse_m"] = steady_state_rmse(trial, kind="translational")
    row["ss_rot_rmse_rad"] = steady_state_rmse(trial, kind="rotational")
    row["ss_riem_rmse"] = steady_state_rmse(trial, kind="riemannian")

    # --- Transient ---
    row["max_overshoot_trans_m"] = max_overshoot(trial, kind="translational")
    row["max_overshoot_rot_rad"] = max_overshoot(trial, kind="rotational")
    row["max_overshoot_riem"] = max_overshoot(trial, kind="riemannian")

    row["post_overshoot_trans_m"] = post_target_overshoot(trial, kind="translational")
    row["post_overshoot_rot_rad"] = post_target_overshoot(trial, kind="rotational")
    row["post_overshoot_riem"] = post_target_overshoot(trial, kind="riemannian")

    st = settling_time(trial)
    row["settling_trans_s"] = st["trans_s"]
    row["settling_rot_s"] = st["rot_s"]
    row["settling_riem_s"] = st["riem_s"]

    # --- Effort ---
    row["norm_control_effort"] = normalized_control_effort(trial)
    row["norm_velocity_effort"] = normalized_velocity_effort(trial)
    row["tau_sat_fraction"] = joint_torque_saturation_fraction(trial)
    row["v_sat_fraction"] = joint_velocity_saturation_fraction(trial)

    # --- Safety ---
    row["safety_intervention_integral"] = safety_intervention_integral(trial)
    row["kinematic_intervention_integral"] = kinematic_intervention_integral(trial)

    asif = asif_activity(trial)
    row["asif_invocation_rate"] = float(asif["invocation_rate"])
    row["asif_modification_rate"] = float(asif["modification_rate"])
    row["asif_mean_delta_tau"] = float(asif["mean_delta_tau"])
    row["asif_max_delta_tau"] = float(asif["max_delta_tau"])
    row["asif_infeasible_count"] = float(asif["infeasible_count"])
    row["asif_max_iters_count"] = float(asif["max_iters_count"])
    for i, v in enumerate(np.atleast_1d(asif["per_joint_mean_delta"])):
        row[f"asif_per_joint_mean_delta.{i}"] = float(v)

    ok = optik_activity(trial)
    row["optik_invocation_rate"] = float(ok["invocation_rate"])
    row["optik_modification_rate"] = float(ok["modification_rate"])
    row["optik_mean_delta_v"] = float(ok["mean_delta_v"])
    row["optik_max_delta_v"] = float(ok["max_delta_v"])
    row["optik_infeasible_count"] = float(ok["infeasible_count"])
    row["optik_max_iters_count"] = float(ok["max_iters_count"])
    for i, v in enumerate(np.atleast_1d(ok["per_joint_mean_delta"])):
        row[f"optik_per_joint_mean_delta.{i}"] = float(v)

    return row


def summarise_experiment(exp: Experiment) -> pd.DataFrame:
    """One-row-per-trial DataFrame of all scalar metrics."""
    rows = []
    for trial in exp:
        row = {"trial_name": trial.name}
        row.update(summarise(trial))
        rows.append(row)
    return pd.DataFrame(rows).set_index("trial_name")


def to_markdown(df: pd.DataFrame) -> str:
    """Return a GitHub-flavoured markdown table string."""
    return df.to_markdown(floatfmt=".4f")


def to_csv(df: pd.DataFrame, path: str | Path) -> None:
    """Write the summary DataFrame to a CSV file."""
    df.to_csv(path)
