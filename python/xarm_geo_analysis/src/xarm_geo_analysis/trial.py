"""
Trial and Experiment — core data containers.

A Trial wraps one wide-CSV log emitted by DataLogger together with its
JSON sidecar.  An Experiment is a named collection of trials that can be
summarised and compared as a unit.

CSV column conventions (all written by DataLogger):
    t, tick
    q.0 .. q.{dof-1}
    v.0 .. v.{dof-1}
    tau_measured.0 .. tau_measured.{dof-1}
    q_ref.0 ..  a_ref.{dof-1}          (joint-space phases; may be blank)
    ee_actual.x/y/z, ee_actual.qx/qy/qz/qw
    ee_target.x/y/z, ee_target.qx/qy/qz/qw
    ee_twist_actual.vx/vy/vz/wx/wy/wz
    ee_twist_target.vx/vy/vz/wx/wy/wz
    tau_ctrl.0 ..  tau_safe.{dof-1}    (dynamic; may be blank)
    v_ctrl.0  ..  v_safe.{dof-1}       (kinematic; may be blank)
    controller_status, asif_status, asif_invoked, asif_modified
    optik_status, optik_invoked, optik_modified

Rotations are stored as quaternions (qx, qy, qz, qw); no Euler angles.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation


# ---------------------------------------------------------------------------
# Trial
# ---------------------------------------------------------------------------


@dataclass
class Trial:
    """One logged control trial."""

    name: str
    df: pd.DataFrame
    meta: dict  # parsed sidecar JSON

    # --- Construction ---

    @classmethod
    def load(cls, csv_path: str | Path) -> "Trial":
        """Load a trial from a DataLogger CSV and its accompanying .meta.json sidecar.

        The sidecar is expected at ``<csv_path>.meta.json``.  If it is absent,
        ``meta`` will be an empty dict and limit-dependent metrics will return NaN.
        """
        csv_path = Path(csv_path)
        df = pd.read_csv(csv_path)

        sidecar = csv_path.parent / (csv_path.name + ".meta.json")
        meta: dict = {}
        if sidecar.exists():
            with sidecar.open() as f:
                meta = json.load(f)

        name = meta.get("trial_name", csv_path.stem)
        return cls(name=name, df=df, meta=meta)

    # --- Metadata helpers ---

    @property
    def dof(self) -> int:
        return int(self.meta.get("dof", self._infer_dof()))

    @property
    def dt(self) -> float:
        """Nominal sample period in seconds (from sidecar, or inferred from data)."""
        if "dt_nominal_s" in self.meta and self.meta["dt_nominal_s"] > 0:
            return float(self.meta["dt_nominal_s"])
        t = self.t()
        return float(np.mean(np.diff(t))) if len(t) > 1 else 0.002

    @property
    def tau_max(self) -> np.ndarray | None:
        """Per-joint torque limits from sidecar (None if absent or all-inf)."""
        if "tau_max" not in self.meta:
            return None
        v = np.array(self.meta["tau_max"], dtype=float)
        return v if np.any(np.isfinite(v)) else None

    @property
    def v_max(self) -> np.ndarray | None:
        """Per-joint velocity limits from sidecar (None if absent)."""
        if "v_max" not in self.meta:
            return None
        return np.array(self.meta["v_max"], dtype=float)

    @property
    def q_min(self) -> np.ndarray | None:
        if "q_min" not in self.meta:
            return None
        return np.array(self.meta["q_min"], dtype=float)

    @property
    def q_max(self) -> np.ndarray | None:
        if "q_max" not in self.meta:
            return None
        return np.array(self.meta["q_max"], dtype=float)

    # --- Time ---

    def t(self) -> np.ndarray:
        """Timestamp vector (seconds)."""
        return self.df["t"].to_numpy(dtype=float)

    # --- Joint state ---

    def q(self) -> np.ndarray:
        """Joint positions (N, dof)."""
        return self._vec_cols("q")

    def v(self) -> np.ndarray:
        """Joint velocities (N, dof)."""
        return self._vec_cols("v")

    def tau_measured(self) -> np.ndarray:
        """Measured joint torques (N, dof)."""
        return self._vec_cols("tau_measured")

    # --- Task-space actual ---

    def p_actual(self) -> np.ndarray:
        """EE position (N, 3)."""
        return self.df[["ee_actual.x", "ee_actual.y", "ee_actual.z"]].to_numpy(
            dtype=float
        )

    def R_actual(self) -> Rotation:
        """EE orientation as a scipy Rotation object (length N)."""
        q = self.df[
            ["ee_actual.qx", "ee_actual.qy", "ee_actual.qz", "ee_actual.qw"]
        ].to_numpy(dtype=float)
        return Rotation.from_quat(q)

    def twist_actual(self) -> np.ndarray:
        """Body-frame EE twist (N, 6): [vx, vy, vz, wx, wy, wz]."""
        cols = [f"ee_twist_actual.{c}" for c in ("vx", "vy", "vz", "wx", "wy", "wz")]
        return self.df[cols].to_numpy(dtype=float)

    # --- Task-space target ---

    def p_target(self) -> np.ndarray:
        """Target EE position (N, 3)."""
        return self.df[["ee_target.x", "ee_target.y", "ee_target.z"]].to_numpy(
            dtype=float
        )

    def R_target(self) -> Rotation:
        """Target EE orientation as a scipy Rotation object (length N)."""
        q = self.df[
            ["ee_target.qx", "ee_target.qy", "ee_target.qz", "ee_target.qw"]
        ].to_numpy(dtype=float)
        return Rotation.from_quat(q)

    def twist_target(self) -> np.ndarray:
        """Target body-frame EE twist (N, 6)."""
        cols = [f"ee_twist_target.{c}" for c in ("vx", "vy", "vz", "wx", "wy", "wz")]
        return self.df[cols].to_numpy(dtype=float)

    # --- Torque triplet ---

    def tau_ctrl(self) -> np.ndarray:
        """Raw hook torque output (N, dof); NaN rows in velocity-mode trials."""
        return self._vec_cols("tau_ctrl")

    def tau_des(self) -> np.ndarray:
        """Post-bias-compensation torque (N, dof)."""
        return self._vec_cols("tau_des")

    def tau_safe(self) -> np.ndarray:
        """Post-ASIF torque (N, dof); == tau_des when ASIF is off."""
        return self._vec_cols("tau_safe")

    # --- Velocity triplet ---

    def v_ctrl(self) -> np.ndarray:
        """Raw hook velocity output (N, dof); NaN rows in torque-mode trials."""
        return self._vec_cols("v_ctrl")

    def v_des(self) -> np.ndarray:
        """Pre-safety-layer velocity (N, dof)."""
        return self._vec_cols("v_des")

    def v_safe(self) -> np.ndarray:
        """Post-OptIK / rescale velocity (N, dof)."""
        return self._vec_cols("v_safe")

    # --- Boolean / status masks ---

    def asif_invoked_mask(self) -> np.ndarray:
        """Boolean array; True on ticks where ASIF was invoked."""
        return self.df["asif_invoked"].to_numpy(dtype=bool)

    def asif_modified_mask(self) -> np.ndarray:
        """Boolean array; True on ticks where ASIF changed the command."""
        return self.df["asif_modified"].to_numpy(dtype=bool)

    def optik_invoked_mask(self) -> np.ndarray:
        return self.df["optik_invoked"].to_numpy(dtype=bool)

    def optik_modified_mask(self) -> np.ndarray:
        return self.df["optik_modified"].to_numpy(dtype=bool)

    # --- PID integrator state ---

    def e_I(self) -> np.ndarray:
        """Bhat integrator state (N, 6); NaN rows for non-PID trials."""
        cols = ["e_I.vx", "e_I.vy", "e_I.vz", "e_I.wx", "e_I.wy", "e_I.wz"]
        missing = [c for c in cols if c not in self.df.columns]
        if missing:
            return np.full((len(self.df), 6), np.nan)
        return self.df[cols].to_numpy(dtype=float)

    # --- Obstacle distance ---

    def obstacle_distance_min(self) -> np.ndarray:
        """Per-tick minimum distance to nearest collision pair (N,).

        Returns NaN for trials where distance was not recorded (non-obstacle
        experiments).
        """
        col = "obstacle_distance_min"
        if col not in self.df.columns:
            return np.full(len(self.df), np.nan)
        return self.df[col].to_numpy(dtype=float)

    def is_torque_mode(self) -> bool:
        """True if the torque triplet contains any finite values."""
        tau = self.tau_ctrl()
        return bool(np.any(np.isfinite(tau)))

    def is_kinematic_mode(self) -> bool:
        return not self.is_torque_mode()

    # --- Internal helpers ---

    def _vec_cols(self, prefix: str) -> np.ndarray:
        """Return (N, dof) array for columns prefix.0 .. prefix.{dof-1}.

        Columns that were written as blank by DataLogger (absent triplet) arrive
        as NaN from pandas; they are preserved as-is so metric functions can
        detect and handle them gracefully.
        """
        cols = [f"{prefix}.{i}" for i in range(self.dof)]
        missing = [c for c in cols if c not in self.df.columns]
        if missing:
            return np.full((len(self.df), self.dof), np.nan)
        return self.df[cols].to_numpy(dtype=float)

    def _infer_dof(self) -> int:
        """Infer dof from the number of q.i columns present."""
        i = 0
        while f"q.{i}" in self.df.columns:
            i += 1
        return i if i > 0 else 6  # sensible fallback


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------


@dataclass
class Experiment:
    """A named collection of Trial objects for multi-trial comparison.

    Trials are keyed by their name string.  Load from a directory of CSVs with
    ``Experiment.load_dir()``, or build manually with ``Experiment([t1, t2])``.
    """

    trials: dict[str, Trial] = field(default_factory=dict)

    def __init__(
        self, trials: Sequence[Trial] | dict[str, Trial] | None = None
    ) -> None:
        if trials is None:
            self.trials = {}
        elif isinstance(trials, dict):
            self.trials = trials
        else:
            self.trials = {t.name: t for t in trials}

    @classmethod
    def load_dir(cls, directory: str | Path, pattern: str = "*.csv") -> "Experiment":
        """Load all CSV files matching ``pattern`` in ``directory`` as trials."""
        directory = Path(directory)
        trials = [Trial.load(p) for p in sorted(directory.glob(pattern))]
        return cls(trials)

    def add(self, trial: Trial) -> None:
        self.trials[trial.name] = trial

    def __len__(self) -> int:
        return len(self.trials)

    def __iter__(self):
        return iter(self.trials.values())

    def __getitem__(self, name: str) -> Trial:
        return self.trials[name]
