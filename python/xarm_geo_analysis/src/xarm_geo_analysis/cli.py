"""
Command-line interface: xarm-geo-analyse

Sub-commands
------------
plot trial   <csv>               4-panel diagnostic figure for one trial.
plot compare <csv1> <csv2> ...   overlay plots across multiple trials.
report       <csv-or-dir> [-o]   print / write the scalar summary table.
metric       <name> <csv>        print a single scalar metric (CI-friendly).

Saving plots
------------
Both plot sub-commands accept optional output flags:

    --save-dir <dir>          Write plot files into <dir> instead of displaying
                              interactively.  The directory is created if absent.
    --format   pdf|png        Output format.  Default: pdf (vector, LaTeX-ready).
                              Use png for slides / quick previews.
    --dpi      <int>          Raster DPI; only applied when --format png.
                              Default: 300.

Plots are always saved with bbox_inches="tight" (no surrounding whitespace).

PDF files can be included directly in LaTeX via:
    \\includegraphics[width=\\linewidth]{figures/trial_tracking_errors.pdf}

Examples
--------
    # Interactive display (default)
    xarm-geo-analyse plot trial tests/results/sim_GeometricPController_PipeInspection_ff.csv

    # Save LaTeX-ready PDFs
    xarm-geo-analyse plot trial tests/results/sim_GeometricPController_PipeInspection_ff.csv \\
        --save-dir paper/figures/phase2/

    # Save PNGs for slides
    xarm-geo-analyse plot compare trial1.csv trial2.csv \\
        --save-dir slides/figures/ --format png --dpi 200

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

    figures: list[tuple[Figure, str]] = [
        (
            overlay(
                exp,
                translational_error,
                ylabel="Translational error (m)",
                title="Translational error comparison",
            ),
            f"{prefix}_translational",
        ),
        (
            overlay(
                exp,
                rotational_geodesic_error,
                ylabel="Rotational error (rad)",
                title="Rotational error comparison",
            ),
            f"{prefix}_rotational",
        ),
        (
            overlay(
                exp,
                riemannian_se3_error,
                ylabel="Riemannian SE(3) error",
                title="Riemannian error comparison",
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
    elif args.command == "report":
        cmd_report(args)
    elif args.command == "metric":
        cmd_metric(args)


if __name__ == "__main__":
    main()
