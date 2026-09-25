r"""
Command-line runner for evaluation/protocol_v5.py (revision-srep-v5, T1.4+)
===========================================================================
Protocol V5: exact W-slot window, zero-filled slots, J=3 + reflect padding
for the three wavelet-based methods, RSI by item id, stable tie-breaking by
fixed item order, seeded robustness test.  See the module docstring.

Examples (from the project root, Windows):

  # T1.4 main run, default windows (baselines 7, wavelet-based 64)
  python tools\run_v5_eval.py --data data\datasets\youtube_hourly.csv --min-obs 50 ^
         --out results\revision_v5\T1.4_protocol_v5\youtube_hourly

  python tools\run_v5_eval.py --data data\datasets\yellow_taxi_2025_all_hourly.csv --min-obs 24 ^
         --out results\revision_v5\T1.4_protocol_v5\taxi_hourly

  # T2.1 baselines with a 16-slot window (wavelet-based stay at 64)
  # (--out results\revision_v5\T2.1_baselines_W16\<dataset>)
  python tools\run_v5_eval.py ... --window AF=16 EWMA=16 RRD=16 VSE=16 CompoundPop=16 PFRF=16

Dataset min-obs of the V4 runs: youtube 50, yellow_taxi (all granularities) 24.
Full command list: Revisions/V4/Response/Runbooks/RUN_T1.4_T2.1.md
Outputs: protocol/*.csv, comparison/summary_{all,common}_windows.csv,
metadata/protocol_v5_run.json.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.protocol_v5 import ProtocolV5Evaluator, build_v5_methods


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--min-obs', type=int, required=True, help='dataset min_observations')
    ap.add_argument('--out', required=True)
    ap.add_argument('--methods', nargs='*', default=None)
    ap.add_argument('--window', nargs='*', default=[], help='NAME=SLOTS overrides')
    ap.add_argument('--level', type=int, default=3, help='J for the three wavelet-based methods')
    ap.add_argument('--no-robustness', action='store_true')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--causal-universe', action='store_true',
                    help='item catalogue from past data only (T1.5); default: whole-file filter')
    a = ap.parse_args()

    wo = {}
    for kv in a.window:
        k, v = kv.split('=')
        wo[k] = int(v)
    data = Path(a.data)
    if not data.is_absolute():
        data = ROOT / data
    fe = ProtocolV5Evaluator.from_csv(data, dataset_min_obs=a.min_obs, seed=a.seed,
                                      robustness=not a.no_robustness,
                                      causal_universe=a.causal_universe)
    methods = build_v5_methods(level=a.level, window_override=wo, names=a.methods)
    out = Path(a.out)
    if not out.is_absolute():
        out = ROOT / out
    res = fe.run(methods, out_dir=out,
                 extra_meta=dict(data=str(a.data), dataset_min_obs=a.min_obs,
                                 window_override=wo, level=a.level,
                                 causal_universe=a.causal_universe))
    ids = set.intersection(*[set(df['window_id']) for df in res.values()])
    print(f'\ncommon windows: {len(ids)}')
    print('method        W    windows  ndcg@10  rho      rsi@10   robust  ties')
    for n, df in res.items():
        d = df[df['window_id'].isin(ids)]
        print(f"{n:<12} {methods[n].window_slots:<4} {len(d):>8}  {d['ndcg@10'].mean():.4f}  "
              f"{d['spearman_rho'].mean():.4f}  {d['rsi@10'].mean():.4f}  "
              f"{d['robustness_distortion'].mean():6.2f}  {d['ties_top21'].mean():.3f}")
    print(f'\nSaved to {out}')


if __name__ == '__main__':
    main()
