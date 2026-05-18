"""
Generate the four cross-dataset comparison figures for the WSPI paper.

This script reads the per-window 4-Layer Protocol metrics that
``temporal_evaluator.py`` writes for every method of every dataset
(layout: ``<run_dir>/protocol/<method>_protocol.parquet``) and produces
four publication-quality bar charts — one per metric (NDCG@10, Spearman ρ,
RSI@10, ΔRank) — each comparing nine methods across all four scenarios.

Grouping modes (``--group-by``)
-------------------------------
  method   (default) — X axis is method, bars within each method are the
                       four datasets (colour = dataset). Baselines and
                       proposed methods sit on grey vs pink backgrounds.

  dataset            — X axis is dataset, bars within each dataset are the
                       nine methods (colour = method). Proposed methods get
                       a hatched pattern and bold legend labels.

Usage (Windows PowerShell)
--------------------------
    python scripts/generate_paper_figures.py `
        --youtube  "results/youtube/<run_folder>" `
        --uber-h   "results/uber/<run_folder>" `
        --uber-30m "results/uber/<run_folder>" `
        --uber-5m  "results/uber/<run_folder>" `
        --out      "paper_figures/" `
        --group-by method

Output
------
    <out>/fig_ndcg10.png
    <out>/fig_spearman.png
    <out>/fig_rsi10.png
    <out>/fig_deltarank.png
    <out>/summary_table.csv     (4 datasets × 9 methods × 4 metrics)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# When --show is requested, the matplotlib backend must be set to an
# interactive one BEFORE the visualizer module imports matplotlib.  We can't
# wait for argparse: inspect sys.argv directly.
if "--show" in sys.argv:
    os.environ.setdefault("CROSS_VIZ_BACKEND", "TkAgg")

# Add the project root to sys.path so we can import the evaluation module
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.cross_dataset_visualizer import CrossDatasetVisualizer


DATASET_FLAGS = [
    ("--youtube",  "YouTube Hourly", "Path to the YouTube hourly run directory"),
    ("--uber-h",   "Uber Hourly",    "Path to the Uber NYC hourly run directory"),
    ("--uber-30m", "Uber 30m",       "Path to the Uber NYC 30-minute run directory"),
    ("--uber-5m",  "Uber 5m",        "Path to the Uber NYC 5-minute run directory"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate cross-dataset comparison figures for the WSPI paper. "
            "Each scenario flag accepts a run directory containing a protocol/ "
            "sub-folder, OR a single merged parquet/CSV file."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    for flag, _label, help_text in DATASET_FLAGS:
        parser.add_argument(flag, dest=flag.lstrip("-").replace("-", "_"),
                            required=True, help=help_text)
    parser.add_argument("--out", required=True,
                        help="Output directory for the four PNGs and summary_table.csv")
    parser.add_argument(
        "--group-by",
        dest="group_by",
        choices=["method", "dataset"],
        default="method",
        help=(
            "Grouping mode for the figures. "
            "'method' (default): X axis is method, bars are datasets. "
            "'dataset': X axis is dataset, bars are methods."
        ),
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help=(
            "After saving each figure, also open it in a viewer window. "
            "Requires an interactive matplotlib backend; on Windows the "
            "default usually works.  If not, set the environment variable "
            "CROSS_VIZ_BACKEND=TkAgg before running."
        ),
    )
    return parser.parse_args()


def _resolve_source(path_str: str, label: str) -> Path:
    """Validate the provided path and return a Path object."""
    path = Path(path_str)
    if not path.exists():
        print(f"[ERROR] {label}: path does not exist:\n        {path}",
              file=sys.stderr)
        sys.exit(1)

    if path.is_dir():
        proto_dir = path / "protocol"
        if not proto_dir.is_dir():
            print(
                f"[ERROR] {label}: run directory has no 'protocol/' sub-folder:\n"
                f"        {path}\n"
                f"        Did you point to the correct run? "
                f"Expected layout: <run_dir>/protocol/<method>_protocol.parquet",
                file=sys.stderr,
            )
            sys.exit(1)

        parquets = list(proto_dir.glob("*_protocol.parquet"))
        if not parquets:
            csvs = list(proto_dir.glob("*_protocol.csv"))
            if not csvs:
                print(
                    f"[ERROR] {label}: no *_protocol.parquet (or .csv) found under:\n"
                    f"        {proto_dir}",
                    file=sys.stderr,
                )
                sys.exit(1)

    return path


def main() -> int:
    args = parse_args()

    dataset_sources: dict[str, Path] = {}
    for flag, label, _help in DATASET_FLAGS:
        attr = flag.lstrip("-").replace("-", "_")
        dataset_sources[label] = _resolve_source(getattr(args, attr), label)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("WSPI Cross-Dataset Figure Generator")
    print("=" * 70)
    for label, src in dataset_sources.items():
        print(f"  {label:<16}  ->  {src}")
    print(f"  {'output':<16}  ->  {out_dir}")
    print(f"  {'group-by':<16}  ->  {args.group_by}")
    print("=" * 70)

    viz = CrossDatasetVisualizer(dataset_sources, output_dir=out_dir)

    print("\nProducing figures...")
    viz.plot_all(out_dir, group_by=args.group_by, show=args.show)
    print("Writing summary_table.csv...")
    viz.save_summary_table(out_dir / "summary_table.csv")

    print("\nDone. Generated files:")
    for f in sorted(out_dir.iterdir()):
        if f.is_file():
            print(f"  {f.name:<24} ({f.stat().st_size / 1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
