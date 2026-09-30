r"""
T1.3 — Gate G1: fast evaluator (compat mode) vs. the V4 protocol CSVs
=====================================================================
Runs evaluation/fast_evaluator.py in compat mode on the same data and
compares it, window by window, with the protocol CSVs of the V4 runs:

  YouTube      : results/youtube/main_20260612_140555/protocol
  Taxi hourly  : results/yellow_taxi/predcmp_20260830_001817_hourly/protocol

Checks per method:
  1. identical set of window_id values, identical num_items per window;
  2. max |difference| of every deterministic column
     (ndcg@K, coverage@K, kendall_tau, spearman_rho, mae, rsi@K);
     NaN must be NaN in both files.  Pass: < 1e-6 (mae: relative < 1e-9).
  3. robustness_distortion: the V4 code sampled 50 targets with the
     UNSEEDED global RNG, so only the mean over windows is compared
     (reported, not part of the pass/fail gate).
  4. runtime: fast vs. V4 (metadata/runtime_stats.json of each run folder
     when available).

Writes: <out>/g1_comparison.csv, <out>/g1_report.json and the fast
protocol CSVs under <out>/<dataset>/protocol/.

    python tools/validate_fast_evaluator.py --out results/revision_v5/g1_validation_20260924
    python tools/validate_fast_evaluator.py --datasets youtube --methods WSPI AF
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.fast_evaluator import FastEvaluator, build_default_methods

RUNS = {
    'youtube': dict(csv='data/datasets/youtube_hourly.csv', min_obs=50,
                    ref='results/youtube/main_20260612_140555'),
    'yellow_taxi_hourly': dict(csv='data/datasets/yellow_taxi_2025_all_hourly.csv', min_obs=24,
                               ref='results/yellow_taxi/predcmp_20260830_001817_hourly'),
}
DET_COLS = ['ndcg@5', 'coverage@5', 'ndcg@10', 'coverage@10', 'ndcg@20', 'coverage@20',
            'kendall_tau', 'spearman_rho', 'rsi@5', 'rsi@10', 'rsi@20']
TOL = 1e-6


def old_runtime(ref_dir, method):
    """Best-effort: V4 per-method duration in seconds."""
    for p in [ref_dir / 'metadata' / 'runtime_stats.json',
              ref_dir / 'metadata' / 'run_metadata.json']:
        try:
            d = json.loads(p.read_text(encoding='utf-8'))
            ms = d.get('methods_stats') or d.get('runtime_stats', {}).get('methods_stats', {})
            if method in ms:
                return float(ms[method].get('duration', float('nan')))
        except Exception:
            pass
    log = ref_dir / 'logs' / (method.replace('+', 'PLUS') + '.log')
    try:
        for line in log.read_text(encoding='utf-8', errors='ignore').splitlines():
            if 'Completed in' in line:
                return float(line.split('Completed in')[1].split('minutes')[0]) * 60
    except Exception:
        pass
    return float('nan')


def compare(new, old):
    r = {}
    ids_new, ids_old = set(new['window_id']), set(old['window_id'])
    r['windows_new'], r['windows_old'] = len(ids_new), len(ids_old)
    r['window_ids_equal'] = ids_new == ids_old
    m = new.merge(old, on='window_id', suffixes=('_new', '_old'))
    r['num_items_equal'] = bool((m['num_items_new'] == m['num_items_old']).all())
    r['timestamp_equal'] = bool((m['timestamp_new'] == m['timestamp_old']).all())
    worst = 0.0
    for c in DET_COLS:
        a, b = m[f'{c}_new'].values.astype(float), m[f'{c}_old'].values.astype(float)
        nan_mismatch = int((np.isnan(a) != np.isnan(b)).sum())
        both = ~np.isnan(a) & ~np.isnan(b)
        d = float(np.max(np.abs(a[both] - b[both]))) if both.any() else 0.0
        r[f'maxdiff_{c}'] = d
        r[f'nanmismatch_{c}'] = nan_mismatch
        if nan_mismatch:
            worst = max(worst, np.inf)
        worst = max(worst, d)
    a, b = m['mae_new'].values.astype(float), m['mae_old'].values.astype(float)
    both = ~np.isnan(a) & ~np.isnan(b)
    r['maxrel_mae'] = float(np.max(np.abs(a[both] - b[both]) / np.maximum(np.abs(b[both]), 1e-12))) if both.any() else 0.0
    r['max_abs_diff_all'] = worst
    # windows whose tie-sensitive columns differ, and whether ALL of them
    # have exact score ties in the top-21 (-> explained by argsort tie order)
    tie_cols = [c for c in DET_COLS if c.startswith(('ndcg', 'coverage', 'rsi'))]
    diff_w = np.zeros(len(m), dtype=bool)
    for c in tie_cols:
        a, b = m[f'{c}_new'].values.astype(float), m[f'{c}_old'].values.astype(float)
        both = ~np.isnan(a) & ~np.isnan(b)
        diff_w |= both & (np.abs(np.where(both, a - b, 0)) >= TOL)
    r['windows_differing'] = int(diff_w.sum())
    if 'ties_top21' in m.columns:
        tied = m['ties_top21'].astype(bool).values
        # RSI of window w also depends on window w-1's top-K
        prev_tied = np.r_[False, tied[:-1]]
        r['windows_with_ties_top21'] = int(tied.sum())
        r['differing_all_explained_by_ties'] = bool(np.all((tied | prev_tied)[diff_w]))
    r['robust_mean_new'] = float(np.nanmean(new['robustness_distortion']))
    r['robust_mean_old'] = float(np.nanmean(old['robustness_distortion']))
    r['PASS'] = bool(r['window_ids_equal'] and r['num_items_equal'] and r['timestamp_equal']
                     and worst < TOL and r['maxrel_mae'] < 1e-9)
    # scores identical (tau, rho, mae exact) and every differing window is a
    # tie window -> the only difference is np.argsort's tie order
    score_cols_ok = all(r[f'maxdiff_{c}'] < TOL for c in ('kendall_tau', 'spearman_rho'))
    r['PASS_up_to_ties'] = bool(r['PASS'] or (
        r['window_ids_equal'] and r['num_items_equal'] and score_cols_ok
        and r['maxrel_mae'] < 1e-9 and r.get('differing_all_explained_by_ties', False)))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='results/revision_v5/g1_validation_20260924')
    ap.add_argument('--datasets', nargs='*', default=list(RUNS))
    ap.add_argument('--methods', nargs='*', default=None)
    ap.add_argument('--no-robustness', action='store_true')
    args = ap.parse_args()
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for ds in args.datasets:
        cfg = RUNS[ds]
        ref = ROOT / cfg['ref']
        t0 = time.time()
        fe = FastEvaluator.from_csv(ROOT / cfg['csv'], dataset_min_obs=cfg['min_obs'],
                                    mode='compat', robustness=not args.no_robustness,
                                    tie_info=True)
        t_build = time.time() - t0
        methods = build_default_methods(dtcwt_level=2, names=args.methods)
        for name, fm in methods.items():
            new = fe.run_method(fm, out_dir=out / ds)
            old_p = ref / 'protocol' / f'{name}_protocol.csv'
            old = pd.read_csv(old_p, encoding='utf-8-sig')
            r = dict(dataset=ds, method=name, ref=str(old_p.relative_to(ROOT)))
            r.update(compare(new, old))
            r['fast_seconds'] = round(new.attrs['duration_s'], 2)
            r['build_seconds'] = round(t_build, 2)
            r['old_seconds'] = old_runtime(ref, name)
            rows.append(r)
            print(f"{ds:<20} {name:<12} PASS={r['PASS']!s:<5} ties_ok={r['PASS_up_to_ties']!s:<5} "
                  f"diffwin={r['windows_differing']}/{r.get('windows_with_ties_top21')} "
                  f"win {r['windows_new']}/{r['windows_old']} "
                  f"maxdiff={r['max_abs_diff_all']:.2e} mae_rel={r['maxrel_mae']:.1e} "
                  f"robust {r['robust_mean_new']:.2f}/{r['robust_mean_old']:.2f} "
                  f"time {r['fast_seconds']}s vs {r['old_seconds']:.0f}s")
    df = pd.DataFrame(rows)
    df.to_csv(out / 'g1_comparison.csv', index=False)
    rep = dict(gate='G1', tolerance=TOL, all_pass=bool(df['PASS'].all()),
               all_pass_up_to_ties=bool(df['PASS_up_to_ties'].all()),
               created=time.strftime('%Y-%m-%d %H:%M:%S'),
               n_checks=len(df), n_pass=int(df['PASS'].sum()))
    (out / 'g1_report.json').write_text(json.dumps(rep, indent=2), encoding='utf-8')
    print(json.dumps(rep, indent=2))


if __name__ == '__main__':
    main()
