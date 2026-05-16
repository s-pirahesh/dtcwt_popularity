"""
Generates the four cross-dataset comparison figures for the WSPI paper.

Each figure shows ONE metric across ALL four datasets so that metrics with
different scales and directions are never mixed in the same chart.

Output files
------------
    fig_ndcg10.png    — NDCG@10 across datasets
    fig_spearman.png  — Spearman rho across datasets
    fig_rsi10.png     — RSI@10 across datasets
    fig_deltarank.png — DeltaRank (lower is better) across datasets
    summary_table.csv — Aggregated values for the paper appendix

Usage (Windows PowerShell)
--------------------------
    python scripts/generate_paper_figures.py `
        --youtube  results/youtube_hourly/ `
        --uber-h   results/uber_hourly/ `
        --uber-30m results/uber_30min/ `
        --uber-5m  results/uber_5min/ `
        --out      paper_figures/

Each input path may be:
  - A directory produced by temporal_evaluator.py (contains a protocol/ sub-dir)
  - A single .parquet or .csv file with columns: method, ndcg@10, spearman_rho,
    rsi@10, robustness_distortion  (one row per method×window)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running directly from repo root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.cross_dataset_visualizer import CrossDatasetVisualizer


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="generate_paper_figures",
        description=(
            "Generate cross-dataset comparison figures for the WSPI paper.\n"
            "Produces four PNGs (one metric each) and a summary CSV."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--youtube",
        required=True,
        metavar="PATH",
        help="YouTube Hourly results — directory or parquet/csv file",
    )
    parser.add_argument(
        "--uber-h",
        required=True,
        metavar="PATH",
        dest="uber_h",
        help="Uber NYC Hourly results — directory or parquet/csv file",
    )
    parser.add_argument(
        "--uber-30m",
        required=True,
        metavar="PATH",
        dest="uber_30m",
        help="Uber NYC 30-min results — directory or parquet/csv file",
    )
    parser.add_argument(
        "--uber-5m",
        required=True,
        metavar="PATH",
        dest="uber_5m",
        help="Uber NYC 5-min results — directory or parquet/csv file",
    )
    parser.add_argument(
        "--out",
        required=True,
        metavar="DIR",
        help="Output directory; created if it does not exist",
    )
    return parser


def _validate_inputs(inputs: dict[str, Path]) -> bool:
    ok = True
    for label, path in inputs.items():
        if not path.exists():
            print(
                f"ERROR: {label} — path does not exist: {path}",
                file=sys.stderr,
            )
            ok = False
    return ok


def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()

    inputs: dict[str, Path] = {
        "YouTube Hourly": Path(args.youtube),
        "Uber Hourly":    Path(args.uber_h),
        "Uber 30m":       Path(args.uber_30m),
        "Uber 5m":        Path(args.uber_5m),
    }

    if not _validate_inputs(inputs):
        sys.exit(1)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading per-window results...")
    viz = CrossDatasetVisualizer(dataset_results=inputs, output_dir=out_dir)

    print("Generating figures...")
    viz.plot_all(out_dir)

    summary_path = out_dir / "summary_table.csv"
    viz.save_summary_table(summary_path)

    generated = sorted(out_dir.iterdir())
    print(f"\nDone. {len(generated)} file(s) written to: {out_dir.resolve()}")
    for f in generated:
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
