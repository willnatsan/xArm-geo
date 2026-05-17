"""
Command-line interface: xarm-geo-analyse

Sub-commands
------------
plot trial   <csv>               4-panel diagnostic figure for one trial.
plot compare <csv1> <csv2> ...   overlay plots across multiple trials.
plot exp     <id> <dir>          Experiment-specific figure set (see below).
report       <csv-or-dir> [-o]   print / write the scalar summary table.
metric       <name> <csv>        print a single scalar metric (CI-friendly).

Experiment-specific plot sets (``plot exp <id> <dir>``)
--------------------------------------------------------
  1a   3-D path overlay + geodesic error overlay + error-twist overlay
  1b   geodesic error overlay + control-effort overlay
  2    tracking-error-with-disturbance overlay + PID integrator state
       (one integrator plot per CSV that has e_I data)
  3a   error distribution box plot + per-axis position zoom (X axis)
  3b   3-D path with obstacle + obstacle distance + intervention norms

Saving plots
------------
All plot sub-commands accept optional output flags:

    --save-dir <dir>          Write plot files into <dir> instead of displaying
                              interactively.  The directory is created if absent.
    --format   pdf|png        Output format.  Default: pdf (vector, LaTeX-ready).
                              Use png for slides / quick previews.
    --dpi      <int>          Raster DPI; only applied when --format png.
                              Default: 300.

Plots are always saved with bbox_inches="tight" (no surrounding whitespace).

PDF files can be included directly in LaTeX via:
    \\includegraphics[width=\\linewidth]{figures/trial_name_tracking_errors.pdf}

Examples
--------
    # Interactive display (default)
    xarm-geo-analyse plot trial tests/results/sim_GeometricPController_FigureEight_ff.csv

    # Experiment-specific figure set for Exp 1A, saved as PDFs
    xarm-geo-analyse plot exp 1a tests/results/exp_1a/ --save-dir paper/figures/exp_1a/

    # Generic compare overlay for ad-hoc inspection
    xarm-geo-analyse plot compare trial1.csv trial2.csv --save-dir slides/ --format png

    xarm-geo-analyse report tests/results/ -o summary.md
    xarm-geo-analyse metric trans_rmse_m tests/results/trial.csv
"""

from __future__ import annotations

import argparse
import hashlib
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def _load_trial(path: str):
    from xarm_geo_analysis.trial import Trial

    return Trial.load(path)


def _load_experiment(paths: list[str]):
    from xarm_geo_analysis.trial import Experiment, Trial

    trials = [Trial.load(p) for p in paths]
    return Experiment(trials)


# ---------------------------------------------------------------------------
# Plot output helper
# ---------------------------------------------------------------------------


def _save_or_show(
    figures: list[tuple[Figure, str]],
    args: argparse.Namespace,
) -> None:
    """Either save each figure to disk or display all interactively.

    Parameters
    ----------
    figures : list of (figure, stem) pairs, where ``stem`` is the filename
              without extension (e.g. "trial_name_tracking_errors").
    args    : parsed CLI namespace; must have attributes save_dir, format, dpi.
    """
    if args.save_dir:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        for fig, stem in figures:
            out = save_dir / f"{stem}.{args.format}"
            kwargs: dict = {"bbox_inches": "tight"}
            if args.format == "png":
                kwargs["dpi"] = args.dpi
            fig.savefig(out, format=args.format, **kwargs)
            plt.close(fig)
            print(f"Saved {out}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Sub-command handlers
# ---------------------------------------------------------------------------


def cmd_plot_trial(args: argparse.Namespace) -> None:
    from xarm_geo_analysis.plotting import (
        plot_3d_path,
        plot_settling,
        plot_torque_triplet,
        plot_tracking_errors,
        plot_velocity_triplet,
    )

    trial = _load_trial(args.csv)
    n = trial.name

    figures: list[tuple[Figure, str]] = [
        (plot_tracking_errors(trial), f"{n}_tracking_errors"),
        (plot_settling(trial), f"{n}_settling"),
        (plot_3d_path(trial), f"{n}_3d_path"),
    ]

    if trial.is_torque_mode():
        figures.append((plot_torque_triplet(trial), f"{n}_torque_triplet"))
    else:
        figures.append((plot_velocity_triplet(trial), f"{n}_velocity_triplet"))

    _save_or_show(figures, args)


def cmd_plot_compare(args: argparse.Namespace) -> None:
    import numpy as np

    from xarm_geo_analysis.metrics.tracking import (
        riemannian_se3_error,
        rotational_geodesic_error,
        translational_error,
    )
    from xarm_geo_analysis.plotting.compare import overlay

    exp = _load_experiment(args.csvs)

    # Stable short hash of the sorted trial names, for self-describing filenames.
    joined = ",".join(sorted(t.name for t in exp))
    prefix = "compare_" + hashlib.sha1(joined.encode()).hexdigest()[:8]

    # Unit-converted wrappers so compare plots match single-trial units (mm / deg).
    def trans_mm(trial):
        return translational_error(trial) * 1e3

    def rot_deg(trial):
        return np.degrees(rotational_geodesic_error(trial))

    figures: list[tuple[Figure, str]] = [
        (
            overlay(
                exp,
                trans_mm,
                ylabel="Translational Error (mm)",
                title="Translational Error Comparison",
            ),
            f"{prefix}_translational",
        ),
        (
            overlay(
                exp,
                rot_deg,
                ylabel="Rotational Error (deg)",
                title="Rotational Error Comparison",
            ),
            f"{prefix}_rotational",
        ),
        (
            overlay(
                exp,
                riemannian_se3_error,
                ylabel="Riemannian SE(3) Error",
                title="Riemannian SE(3) Error Comparison",
            ),
            f"{prefix}_riemannian",
        ),
    ]

    _save_or_show(figures, args)


def cmd_report(args: argparse.Namespace) -> None:
    from xarm_geo_analysis.report import summarise_experiment, to_csv, to_markdown
    from xarm_geo_analysis.trial import Experiment

    target = Path(args.target)
    if target.is_dir():
        exp = Experiment.load_dir(target)
    else:
        exp = _load_experiment([str(target)])

    if len(exp) == 0:
        print("No trials found.", file=sys.stderr)
        sys.exit(1)

    df = summarise_experiment(exp)

    if args.output:
        out = Path(args.output)
        if out.suffix == ".csv":
            to_csv(df, out)
            print(f"Wrote CSV report to {out}")
        else:
            out.write_text(to_markdown(df))
            print(f"Wrote markdown report to {out}")
    else:
        print(to_markdown(df))


def cmd_metric(args: argparse.Namespace) -> None:
    """Print a single named scalar metric to stdout (suitable for CI scripts)."""
    from xarm_geo_analysis.report import summarise

    trial = _load_trial(args.csv)
    row = summarise(trial)

    if args.name not in row:
        print(f"Unknown metric '{args.name}'.  Available metrics:", file=sys.stderr)
        for k in sorted(row):
            print(f"  {k}", file=sys.stderr)
        sys.exit(1)

    value = row[args.name]
    print(value if math.isfinite(value) else "nan")


def cmd_plot_exp(args: argparse.Namespace) -> None:
    """Generate the experiment-specific figure set for a results directory."""
    from xarm_geo_analysis.plotting import (
        plot_3d_paths_compare,
        plot_command_norm_overlay,
        plot_error_boxplot,
        plot_error_twist_overlay,
        plot_integrator_state,
        plot_intervention_norm,
        plot_obstacle_distance,
        plot_tracking_with_disturbance,
        plot_axis_zoom,
    )
    from xarm_geo_analysis.plotting.compare import overlay
    from xarm_geo_analysis.metrics.tracking import rotational_geodesic_error
    from xarm_geo_analysis.trial import Experiment

    exp_id = args.exp_id
    results_dir = Path(args.results_dir)

    if not results_dir.is_dir():
        print(f"Directory not found: {results_dir}", file=sys.stderr)
        sys.exit(1)

    exp = Experiment.load_dir(results_dir)
    if len(exp) == 0:
        print(f"No CSV files found in {results_dir}", file=sys.stderr)
        sys.exit(1)

    import numpy as np

    figures: list[tuple] = []

    if exp_id == "1a":
        # 3-D path overlay (all three variants)
        figures.append((plot_3d_paths_compare(exp), "exp_1a_3d_paths"))

        # Geodesic rotational error overlay
        def rot_deg(trial):
            return np.degrees(rotational_geodesic_error(trial))

        figures.append(
            (
                overlay(
                    exp,
                    rot_deg,
                    ylabel="Rotational Error (deg)",
                    title="Exp 1A: Geodesic Rotational Error",
                ),
                "exp_1a_rot_error",
            )
        )
        # Error-twist norm overlay
        figures.append((plot_error_twist_overlay(exp), "exp_1a_error_twist"))

    elif exp_id == "1b":
        # Geodesic error overlay
        def rot_deg(trial):
            return np.degrees(rotational_geodesic_error(trial))

        figures.append(
            (
                overlay(
                    exp,
                    rot_deg,
                    ylabel="Rotational Error (deg)",
                    title="Exp 1B: Geodesic Rotational Error",
                ),
                "exp_1b_rot_error",
            )
        )
        # Control-effort norm overlay
        figures.append((plot_command_norm_overlay(exp), "exp_1b_command_norm"))

    elif exp_id == "2":
        # Tracking error with disturbance markers — translational
        figures.append(
            (
                plot_tracking_with_disturbance(exp, kind="translational"),
                "exp_2_tracking_trans",
            )
        )
        # Tracking error with disturbance markers — rotational
        figures.append(
            (
                plot_tracking_with_disturbance(exp, kind="rotational"),
                "exp_2_tracking_rot",
            )
        )
        # PID integrator state (one plot per trial that has e_I data)
        for trial in exp:
            import numpy as np
            from xarm_geo_analysis.metrics.transient import integrator_state

            e = integrator_state(trial)
            if np.any(np.isfinite(e)):
                stem = trial.name.replace("/", "_")
                figures.append(
                    (plot_integrator_state(trial), f"exp_2_integrator_{stem}")
                )

    elif exp_id == "3a":
        # Error distribution box plot
        figures.append(
            (plot_error_boxplot(exp, kind="translational"), "exp_3a_boxplot_trans")
        )
        figures.append(
            (plot_error_boxplot(exp, kind="rotational"), "exp_3a_boxplot_rot")
        )
        # Per-axis zoom (X axis) for first two trials (sim kin + hw admittance)
        for trial in list(exp)[:4]:
            stem = trial.name.replace("/", "_")
            figures.append((plot_axis_zoom(trial, axis="x"), f"exp_3a_axis_x_{stem}"))

    elif exp_id == "3b":
        # 3-D path with obstacle
        figures.append((plot_3d_paths_compare(exp), "exp_3b_3d_paths"))
        # Obstacle distance over time
        figures.append((plot_obstacle_distance(exp), "exp_3b_obstacle_distance"))
        # Intervention norms for each trial
        for trial in exp:
            stem = trial.name.replace("/", "_")
            figures.append(
                (plot_intervention_norm(trial), f"exp_3b_intervention_{stem}")
            )

    else:
        print(
            f"Unknown experiment id '{exp_id}'.  Valid ids: 1a 1b 2 3a 3b",
            file=sys.stderr,
        )
        sys.exit(1)

    _save_or_show(figures, args)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _add_plot_output_args(parser: argparse.ArgumentParser) -> None:
    """Attach the shared --save-dir / --format / --dpi flags to a plot sub-parser."""
    parser.add_argument(
        "--save-dir",
        default=None,
        metavar="DIR",
        help="Write plot files into DIR instead of displaying interactively. "
        "Directory is created if it does not exist.",
    )
    parser.add_argument(
        "--format",
        default="pdf",
        choices=["pdf", "png"],
        help="Output format when --save-dir is set.  "
        "pdf (default): vector, LaTeX-ready via \\includegraphics.  "
        "png: raster, suitable for slides and quick previews.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        metavar="N",
        help="Raster DPI; only applied when --format png (default: 300).",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="xarm-geo-analyse",
        description="xArm-geo data analysis and visualisation suite",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- plot ---
    plot_p = sub.add_parser("plot", help="Plotting sub-commands")
    plot_sub = plot_p.add_subparsers(dest="plot_command", required=True)

    pt = plot_sub.add_parser("trial", help="4-panel diagnostic plot for one trial")
    pt.add_argument("csv", help="Path to trial CSV file")
    _add_plot_output_args(pt)

    pc = plot_sub.add_parser("compare", help="Multi-trial overlay plots")
    pc.add_argument("csvs", nargs="+", help="Paths to trial CSV files")
    _add_plot_output_args(pc)

    pe = plot_sub.add_parser("exp", help="Experiment-specific figure set")
    pe.add_argument(
        "exp_id",
        choices=["1a", "1b", "2", "3a", "3b"],
        help="Experiment identifier",
    )
    pe.add_argument("results_dir", help="Directory containing the experiment CSVs")
    _add_plot_output_args(pe)

    # --- report ---
    rep = sub.add_parser("report", help="Print or write scalar summary table")
    rep.add_argument("target", help="Path to a CSV file or directory of CSVs")
    rep.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output path (.md for markdown, .csv for CSV)",
    )

    # --- metric ---
    met = sub.add_parser("metric", help="Print a single metric scalar")
    met.add_argument("name", help="Metric name (e.g. trans_rmse_m)")
    met.add_argument("csv", help="Path to trial CSV file")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "plot":
        if args.plot_command == "trial":
            cmd_plot_trial(args)
        elif args.plot_command == "compare":
            cmd_plot_compare(args)
        elif args.plot_command == "exp":
            cmd_plot_exp(args)
    elif args.command == "report":
        cmd_report(args)
    elif args.command == "metric":
        cmd_metric(args)


if __name__ == "__main__":
    main()
