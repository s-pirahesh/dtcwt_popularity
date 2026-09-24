r"""
T1.1 side-audit — effect of POSITIONAL top-K indices on RSI
===========================================================
The V4 evaluator stores top-K as positions in the per-window score array.
When the set of eligible items changes between two windows, the same
position can point to a different item.  This script recomputes RSI@K in
compat mode twice — positional (V4) and by item id — and reports means.
Diagnostic only; nothing else changes.

    python tools/audit_rsi_indexing.py --out results/revision_v5/audit_20260924
"""
import argparse, sys, json
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from evaluation.fast_evaluator import FastEvaluator, build_default_methods

DS = {'youtube': ('data/datasets/youtube_hourly.csv', 50),
      'yellow_taxi_hourly': ('data/datasets/yellow_taxi_2025_all_hourly.csv', 24)}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', default='results/revision_v5/audit_20260924')
    ap.add_argument('--datasets', nargs='*', default=list(DS)); a = ap.parse_args()
    out = ROOT / a.out; out.mkdir(parents=True, exist_ok=True)
    rows = []
    for ds in a.datasets:
        csv, mo = DS[ds]
        for by_item in (False, True):
            fe = FastEvaluator.from_csv(ROOT / csv, mo, robustness=False, rsi_by_item=by_item, verbose=False)
            for n, fm in build_default_methods(dtcwt_level=2).items():
                df = fe.run_method(fm)
                rows.append(dict(dataset=ds, method=n, rsi_indexing='item_id' if by_item else 'positional(V4)',
                                 **{f'rsi@{k}': df[f'rsi@{k}'].mean() for k in (5, 10, 20)}))
    r = pd.DataFrame(rows)
    r.to_csv(out / 'rsi_positional_vs_item.csv', index=False)
    print(r.pivot_table(index=['dataset', 'method'], columns='rsi_indexing', values='rsi@10').round(4))

if __name__ == '__main__':
    main()
