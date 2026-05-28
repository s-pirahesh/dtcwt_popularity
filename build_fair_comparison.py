r"""
Fair-Window Comparison Builder
==============================
Builds the side-by-side table that answers Reviewer Comment #7:

    "Is it fair that WSPI uses a 64-slot window while baselines use 7?"

It compares each baseline's RSI@10 in two settings:
    - window = 7   (the normal run you already have)
    - window = 64  (the fair run, where all methods use 64)

If the baselines stay far below WSPI even at window=64, the fairness
concern is resolved: WSPI's advantage is structural, not a window-size
artifact.

Usage (Windows):
    python build_fair_comparison.py ^
        --normal-run results\yellow_taxi\w30_h7_nall_top_<OLD_TIMESTAMP> ^
        --fair-run   results\yellow_taxi\w64_h7_nall_top_20260528_041601 ^
        --output     results\tables\fair_comparison_yellow_taxi.csv

If you don't have the OLD normal run directory handy, you can instead pass
the RSI@10 numbers you already extracted from the *_result.txt files via
--normal-csv (a 2-column CSV: method,rsi10).  See the guide for details.
"""
import argparse
from pathlib import Path
import sys

import pandas as pd

BASELINES = ['AF', 'EWMA', 'RRD', 'VSE', 'CompoundPop', 'PFRF']
WAVELET   = ['DWT+AF', 'DTCWT+AF', 'WSPI']


def load_protocol(run_dir: Path, method: str):
    proto_dir = run_dir / 'protocol'
    for ext in ('parquet', 'csv'):
        fp = proto_dir / f'{method}_protocol.{ext}'
        if fp.exists():
            try:
                return pd.read_parquet(fp) if ext == 'parquet' else pd.read_csv(fp)
            except Exception as e:
                print(f"  ! failed to read {fp.name}: {e}")
                return None
    return None


def rsi10_of(run_dir: Path, method: str):
    df = load_protocol(run_dir, method)
    if df is None or 'rsi@10' not in df.columns:
        return None
    return float(df['rsi@10'].mean(skipna=True))


def discover_methods(run_dir: Path):
    proto = run_dir / 'protocol'
    if not proto.exists():
        return []
    out = set()
    for f in list(proto.glob('*_protocol.parquet')) + list(proto.glob('*_protocol.csv')):
        out.add(f.stem.replace('_protocol', ''))
    return sorted(out)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--normal-run', default=None,
                   help='Run dir where baselines used window=7 (optional).')
    p.add_argument('--normal-csv', default=None,
                   help='Alternative: CSV with columns method,rsi10 for window=7.')
    p.add_argument('--fair-run', required=True,
                   help='Run dir where ALL methods used window=64.')
    p.add_argument('--output', default='results/tables/fair_comparison.csv')
    args = p.parse_args()

    fair_dir = Path(args.fair_run)
    if not fair_dir.exists():
        print(f"!! fair-run not found: {fair_dir}")
        sys.exit(1)

    # --- window=7 numbers ---------------------------------------------------
    normal_rsi = {}
    if args.normal_csv:
        ncsv = pd.read_csv(args.normal_csv)
        for _, r in ncsv.iterrows():
            normal_rsi[str(r['method'])] = float(r['rsi10'])
    elif args.normal_run:
        ndir = Path(args.normal_run)
        for m in discover_methods(ndir):
            v = rsi10_of(ndir, m)
            if v is not None:
                normal_rsi[m] = v
    else:
        print("Note: no --normal-run or --normal-csv given; "
              "window=7 column will be blank.")

    # --- window=64 numbers (fair run) --------------------------------------
    fair_rsi = {}
    for m in discover_methods(fair_dir):
        v = rsi10_of(fair_dir, m)
        if v is not None:
            fair_rsi[m] = v

    wspi_fair = fair_rsi.get('WSPI')

    # --- assemble table -----------------------------------------------------
    rows = []
    all_methods = BASELINES + WAVELET
    for m in all_methods:
        w7  = normal_rsi.get(m)
        w64 = fair_rsi.get(m)
        row = {
            'method': m,
            'group': 'baseline' if m in BASELINES else 'wavelet',
            'rsi10_window7':  round(w7, 4) if w7 is not None else None,
            'rsi10_window64': round(w64, 4) if w64 is not None else None,
        }
        if w7 is not None and w64 is not None:
            row['delta_window_effect'] = round(w64 - w7, 4)
        if wspi_fair is not None and w64 is not None and m != 'WSPI':
            # how far below WSPI is this method, even at the same window=64?
            row['gap_below_WSPI_at_w64'] = round(wspi_fair - w64, 4)
        rows.append(row)

    df = pd.DataFrame(rows)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding='utf-8-sig')

    # --- console summary ----------------------------------------------------
    print("\n" + "=" * 70)
    print("FAIR-WINDOW COMPARISON (RSI@10)")
    print("=" * 70)
    print(f"{'method':<14}{'w=7':>10}{'w=64':>10}{'Δ(window)':>12}{'gap<WSPI':>12}")
    print("-" * 70)
    for _, r in df.iterrows():
        w7  = f"{r['rsi10_window7']:.4f}"  if pd.notna(r.get('rsi10_window7'))  else "  —"
        w64 = f"{r['rsi10_window64']:.4f}" if pd.notna(r.get('rsi10_window64')) else "  —"
        dlt = f"{r['delta_window_effect']:+.4f}" if pd.notna(r.get('delta_window_effect')) else "  —"
        gap = f"{r['gap_below_WSPI_at_w64']:+.4f}" if pd.notna(r.get('gap_below_WSPI_at_w64')) else "  —"
        print(f"{r['method']:<14}{w7:>10}{w64:>10}{dlt:>12}{gap:>12}")

    print("\nInterpretation:")
    print("  Δ(window)  : how much a baseline's RSI improves when given 64 slots")
    print("               instead of 7. Small Δ => window size was NOT the issue.")
    print("  gap<WSPI   : how far the baseline still sits below WSPI at window=64.")
    print("               Large positive gap => WSPI advantage is structural.")
    print(f"\nSaved table to: {out}")


if __name__ == '__main__':
    main()
