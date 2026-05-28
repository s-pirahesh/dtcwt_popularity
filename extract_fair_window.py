r"""
Fair-Window Results Extractor
=============================
Reads pre-computed protocol metrics from one or more completed run
directories and produces a single tidy CSV summarising every method.

It replicates the aggregation logic used by ResultsAnalyzer.get_protocol_summary:
for each method, every per-window metric column is averaged with
df[col].mean(skipna=True).

This is purely a READ + AGGREGATE step — it does NOT re-run any evaluation,
so it takes seconds, not hours.

Usage (Windows):
    python extract_fair_window.py ^
        --run results\yellow_taxi\w64_h7_nall_top_20260528_041601 ^
        --run results\youtube\w64_h7_nall_top_20260528_043027 ^
        --output results\tables\fair_window_summary.csv

You may pass --run as many times as you like.  Each run directory must
contain a 'protocol' sub-folder with <method>_protocol.parquet/csv files.
"""
import argparse
from pathlib import Path
import sys

import pandas as pd


# Metrics we want in the summary, matching the 4-layer protocol.
K_VALUES = [5, 10, 20]
SCALAR_COLS = ['spearman_rho', 'kendall_tau', 'mae', 'robustness_distortion']


def discover_methods(run_dir: Path):
    """Find all method names that have a protocol file in run_dir/protocol."""
    proto_dir = run_dir / 'protocol'
    if not proto_dir.exists():
        return []
    methods = set()
    for f in list(proto_dir.glob('*_protocol.parquet')) + \
             list(proto_dir.glob('*_protocol.csv')):
        methods.add(f.stem.replace('_protocol', ''))
    return sorted(methods)


def load_protocol(run_dir: Path, method: str):
    """Load a method's per-window protocol metrics as a DataFrame."""
    proto_dir = run_dir / 'protocol'
    for ext in ('parquet', 'csv'):
        fp = proto_dir / f'{method}_protocol.{ext}'
        if fp.exists():
            if ext == 'parquet':
                try:
                    return pd.read_parquet(fp)
                except Exception as e:
                    print(f"    ! could not read {fp.name} as parquet: {e}")
                    return None
            else:
                return pd.read_csv(fp)
    return None


def summarise_method(df: pd.DataFrame) -> dict:
    """Average every protocol metric over all windows (skipna)."""
    summary = {}

    # Layer 1 (NDCG@K) and Layer 3 (RSI@K)
    for k in K_VALUES:
        for prefix in ('ndcg', 'rsi', 'chr'):
            col = f'{prefix}@{k}'
            if col in df.columns:
                summary[col] = float(df[col].mean(skipna=True))

    # Layer 2 + Layer 4 scalar columns
    for col in SCALAR_COLS:
        if col in df.columns:
            summary[col] = float(df[col].mean(skipna=True))

    # number of windows actually analysed (non-null on a core metric)
    core = 'ndcg@10' if 'ndcg@10' in df.columns else df.columns[0]
    summary['windows'] = int(df[core].notna().sum())

    return summary


def infer_scenario_label(run_dir: Path) -> str:
    """Build a readable label like 'yellow_taxi / w64...041601'."""
    dataset = run_dir.parent.name
    run_id  = run_dir.name
    return f'{dataset}::{run_id}'


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run', action='append', required=True,
                        help='Path to a completed run directory (repeatable).')
    parser.add_argument('--output', default='results/tables/fair_window_summary.csv')
    args = parser.parse_args()

    rows = []
    for run_str in args.run:
        run_dir = Path(run_str)
        if not run_dir.exists():
            print(f"!! run directory not found: {run_dir}")
            continue

        scenario = infer_scenario_label(run_dir)
        dataset  = run_dir.parent.name
        methods  = discover_methods(run_dir)

        print(f"\n=== {scenario} ===")
        if not methods:
            print("   (no protocol files found — is there a 'protocol' sub-folder?)")
            continue

        for method in methods:
            df = load_protocol(run_dir, method)
            if df is None or df.empty:
                print(f"   {method:<15} : no data")
                continue
            summary = summarise_method(df)
            summary['method']   = method
            summary['dataset']  = dataset
            summary['run_dir']  = run_dir.name
            summary['scenario'] = scenario
            rows.append(summary)
            rsi10  = summary.get('rsi@10', float('nan'))
            ndcg10 = summary.get('ndcg@10', float('nan'))
            drank  = summary.get('robustness_distortion', float('nan'))
            print(f"   {method:<15} : RSI@10={rsi10:6.4f}  "
                  f"NDCG@10={ndcg10:6.4f}  dRank={drank:8.2f}  "
                  f"(windows={summary['windows']})")

    if not rows:
        print("\nNo results collected. Check the run paths.")
        sys.exit(1)

    # Build tidy DataFrame; order columns sensibly
    df_out = pd.DataFrame(rows)
    lead = ['scenario', 'dataset', 'method']
    metric_cols = [c for c in df_out.columns if c not in lead + ['run_dir', 'windows']]
    ordered = lead + sorted(metric_cols) + ['windows', 'run_dir']
    df_out = df_out[[c for c in ordered if c in df_out.columns]]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out, index=False, encoding='utf-8-sig')  # utf-8-sig = Excel-friendly
    print(f"\nSaved fair-window summary to: {out}")
    print(f"  ({len(df_out)} method-rows across {df_out['scenario'].nunique()} run(s))")


if __name__ == '__main__':
    main()
