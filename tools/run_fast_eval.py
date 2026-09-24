r"""
Command-line runner for evaluation/fast_evaluator.py (revision-srep-v5)
=======================================================================
Examples (from the project root):

  # V4-compatible run (reproduces the old protocol CSVs)
  python tools/run_fast_eval.py --data data/datasets/youtube_hourly.csv --min-obs 50 ^
         --out results/revision_v5/compat_youtube_20260924

  # corrected protocol (T1.4): exact W slots, zero-filled, RSI by item id
  python tools/run_fast_eval.py --data data/datasets/youtube_hourly.csv --min-obs 50 ^
         --mode dense --rsi-by-item --out results/revision_v5/dense_youtube_<date>

  # window sweep for the baselines (T2.2): --window AF=16 EWMA=16 ...
  python tools/run_fast_eval.py ... --window AF=16 EWMA=16 RRD=16 VSE=16 CompoundPop=16 PFRF=16

Dataset min-obs of the V4 runs: youtube 50, yellow_taxi (all granularities) 24.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.fast_evaluator import FastEvaluator, build_default_methods


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--min-obs', type=int, required=True, help='dataset min_observations')
    ap.add_argument('--out', required=True)
    ap.add_argument('--mode', choices=['compat', 'dense'], default='compat')
    ap.add_argument('--methods', nargs='*', default=None)
    ap.add_argument('--window', nargs='*', default=[], help='NAME=SLOTS overrides')
    ap.add_argument('--dtcwt-level', type=int, default=2, help='DTCWT+AF level (V4 runs: 2)')
    ap.add_argument('--wspi-level', type=int, default=3)
    ap.add_argument('--rsi-by-item', action='store_true')
    ap.add_argument('--no-robustness', action='store_true')
    ap.add_argument('--seed', type=int, default=42)
    a = ap.parse_args()

    wo = {}
    for kv in a.window:
        k, v = kv.split('=')
        wo[k] = int(v)
    fe = FastEvaluator.from_csv(ROOT / a.data if not Path(a.data).is_absolute() else a.data,
                                dataset_min_obs=a.min_obs, mode=a.mode, seed=a.seed,
                                rsi_by_item=a.rsi_by_item, robustness=not a.no_robustness)
    methods = build_default_methods(dtcwt_level=a.dtcwt_level, wspi_level=a.wspi_level,
                                    window_override=wo, names=a.methods)
    out = Path(a.out)
    if not out.is_absolute():
        out = ROOT / out
    res = fe.run(methods, out_dir=out)
    print('\nmethod        windows  ndcg@10   rho      rsi@10   robust')
    for n, df in res.items():
        print(f"{n:<12} {len(df):>8}  {df['ndcg@10'].mean():.4f}  {df['spearman_rho'].mean():.4f}  "
              f"{df['rsi@10'].mean():.4f}  {df['robustness_distortion'].mean():.2f}")
    print(f'\nSaved to {out}')


if __name__ == '__main__':
    main()
