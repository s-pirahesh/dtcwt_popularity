r"""
Run the prediction methods under protocol V5, for one or more horizons
=======================================================================
Same evaluator, item rule and metrics as the paper runs
(``tools/run_v5_eval.py --causal-universe``); only the method set differs and
the ground truth can look further ahead.  With horizon h, the ground truth of
window k is the sum of the h slots k..k+h-1 (``evaluation/protocol_v5.py``);
h = 1 is protocol V5.  Every method uses N = 64 slots and needs 32 observed
slots, so all methods share the windows of WSPI (WSPI is always run as the
reference).

Examples (from the project root, Windows):

  # run A on one scenario, three horizons
  python tools\run_prediction.py --data data\datasets\youtube_hourly.csv ^
         --min-obs 50 --horizons 1 8 24 --gate-profile ^
         --out results\paper2\run_A\youtube_hourly

  # run B: ARMA(1,1) in the coefficient domain against ARIMA on the raw signal
  # (statsmodels MLE, one fit per item and window; hours).  One horizon per
  # call, so that the calls can run on separate cores.
  python tools\run_prediction.py --data data\datasets\youtube_hourly.csv ^
         --min-obs 50 --horizons 8 --methods WSPI WSPI-FA ARIMA ^
         --out results\paper2\run_B\youtube_hourly

  # a long run that can be interrupted: give the same command again and it
  # continues (complete methods are skipped, the slow methods continue from
  # the last scored window); see tools/prediction_resume.py
  python tools\run_prediction.py --data data\datasets\youtube_hourly.csv ^
         --min-obs 50 --horizons 8 --methods WSPI WSPI-FA ARIMA --resume ^
         --bound-check --out results\paper2\run_B\youtube_hourly

Outputs, for every horizon h (``--out``/h<h>/):
  protocol/<method>_protocol.csv         window by window (V5 columns)
  comparison/summary_{all,common}_windows.csv
  metadata/protocol_v5_run.json          settings, versions, runtime per method
  scorer_timing.csv                      time spent inside each scorer (see below)
  gate_profile.csv, gate_hist.csv        (--gate-profile) gates of WSPI-F2 and WSPI-FG
  bound_check.csv                        (--bound-check) smallest and largest ratio
                                         P_method / P_WSPI of every clamped method
                                         against the bounds [L, U] of the method

scorer_timing.csv has one row per method: the number of scorer calls and rows,
the seconds spent inside the scorer, the milliseconds per row, the seconds of
the whole evaluation of that method (scoring, metrics and robustness test) and,
for WSPI-FA, the number of fits and of fallbacks to persistence.  The scorer is
also called by the robustness test (at most 50 rows per window); those calls
are counted.

gate_profile.csv: per window, the number of items and the statistics of the
WSPI-F2 gate (median and mean of g, share of g < 0.5, mean R and W_E, first
quartile and median of the raw product p = R(1-W_E)) and of the WSPI-FG gate
(median and mean of g, share of g = 0, which is exactly WSPI).
gate_hist.csv: counts of p, g(F2) and g(FG) over all item-windows in bins of
width 1e-4 on [0, 1].

--resume does not change any result file; the state is kept in --resume-dir
(default logs/prediction_resume/).  scorer_timing.csv and the run time in the
metadata of a resumed method are the times of an uninterrupted run (the scorer
time of the earlier sessions is added), and scorer_timing.csv gets the column
``replayed_calls`` (scorer calls answered from the journal).
"""
import argparse
import gc
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.protocol_v5 import ProtocolV5Evaluator            # noqa: E402
from methods.wspi_forecast import n_forecast_coefficients         # noqa: E402
from tools.prediction_protocol import (ALL_METHODS, MIN_OBS, RUN_A_METHODS,  # noqa: E402
                                       WINDOW, build_prediction_methods,
                                       gate_features)
from tools.prediction_resume import (CODE_FILES, ResumableEvaluator,   # noqa: E402
                                     ResumeState, add_bound_recorders,
                                     default_resume_dir, sha256_file)

HIST_BINS = 10000


def gate_profile(fe: ProtocolV5Evaluator, window: int, min_obs: int):
    """Per-window gate statistics on the item set that the methods score (same
    window and eligibility rule as ``ProtocolV5Evaluator.run_method``, with the
    horizon), and the pooled histograms."""
    rows = []
    edges = np.linspace(0.0, 1.0, HIST_BINS + 1)
    h_p = np.zeros(HIST_BINS, dtype=np.int64)
    h_g = np.zeros(HIST_BINS, dtype=np.int64)
    h_fg = np.zeros(HIST_BINS, dtype=np.int64)
    cum = np.zeros(len(fe.items)) if fe.causal_universe else None
    for k in range(fe.num_slots + 1):
        if cum is not None and k > 0:
            cum += fe.V[:, k - 1]
        if k + fe.horizon - 1 > fe.num_slots:
            break
        lo, hi = fe.window_bounds(k, window)
        if fe.any_cum[hi] - fe.any_cum[lo] == 0 or not fe.any_row[k]:
            continue
        ok = (fe.C[:, hi] - fe.C[:, lo]) >= min_obs
        if cum is not None:
            ok &= cum >= fe.universe_min_count
        idx = np.where(ok)[0]
        if len(idx) < 2:
            continue
        R, WE, g, g_fg = gate_features(fe._matrix(idx, k, window), fe.horizon, window)
        p = np.clip(R * (1.0 - WE), 0.0, 1.0)          # the product inside the gate
        h_p += np.histogram(p, bins=edges)[0]
        h_g += np.histogram(g, bins=edges)[0]
        h_fg += np.histogram(g_fg, bins=edges)[0]
        rows.append({'window_id': k, 'num_items': len(idx),
                     'g_median': float(np.median(g)), 'g_mean': float(np.mean(g)),
                     'g_below_05': float(np.mean(g < 0.5)),
                     'R_mean': float(np.mean(R)), 'WE_mean': float(np.mean(WE)),
                     'p_q25': float(np.quantile(p, 0.25)), 'p_median': float(np.median(p)),
                     'fg_g_median': float(np.median(g_fg)), 'fg_g_mean': float(np.mean(g_fg)),
                     'fg_share_zero': float(np.mean(g_fg == 0))})
    hist = pd.DataFrame({'bin_lo': edges[:-1], 'bin_hi': edges[1:],
                         'count_p': h_p, 'count_g': h_g, 'count_g_fg': h_fg})
    return pd.DataFrame(rows), hist


def timing_table(methods, res, horizon: int, stored=None) -> pd.DataFrame:
    """One row per method: time inside the scorer and the whole evaluation.
    ``stored``: scorer statistics of the methods that were not run again in
    this session (--resume)."""
    rows = []
    stored = stored or {}
    for name, fm in methods.items():
        st = stored[name] if name in stored else fm.scorer.stats()
        total = float(res[name].attrs.get('duration_s', float('nan')))
        rows.append(dict(horizon=horizon, method=name, scorer=st['scorer'],
                         windows=len(res[name]), calls=st['calls'], rows=st['rows'],
                         scorer_seconds=st['seconds'], ms_per_row=st['ms_per_row'],
                         run_seconds=total,
                         n_fits=st.get('n_fits', np.nan),
                         n_fallbacks=st.get('n_fallbacks', np.nan),
                         fallback_share=st.get('fallback_share', np.nan)))
        if 'replayed_calls' in st:
            rows[-1]['replayed_calls'] = st['replayed_calls']
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--data', required=True)
    ap.add_argument('--min-obs', type=int, required=True, help='causal item threshold')
    ap.add_argument('--out', required=True)
    ap.add_argument('--methods', nargs='*', default=None,
                    help=f'subset of {", ".join(ALL_METHODS)} (default: run A, '
                         f'{", ".join(RUN_A_METHODS)})')
    ap.add_argument('--horizons', nargs='+', type=int, default=[1],
                    help='horizons h in slots (default: 1, protocol V5)')
    ap.add_argument('--gate-profile', action='store_true')
    ap.add_argument('--no-robustness', action='store_true')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--bound-check', action='store_true',
                    help='write bound_check.csv (ratio to WSPI of the clamped methods)')
    ap.add_argument('--resume', action='store_true',
                    help='continue an interrupted run with the same command')
    ap.add_argument('--resume-dir', default=None,
                    help='folder of the resume state (default: logs/prediction_resume/<out>)')
    a = ap.parse_args(argv)

    names = a.methods or list(RUN_A_METHODS)
    if 'WSPI' not in names:
        names = ['WSPI'] + list(names)
    data = Path(a.data)
    if not data.is_absolute():
        data = ROOT / data
    out = Path(a.out)
    if not out.is_absolute():
        out = ROOT / out
    df = None
    data_sha = sha256_file(data) if a.resume else None

    for h in a.horizons:
        print(f'\n===== horizon {h} (m = {n_forecast_coefficients(h)} coefficients) =====')
        out_h = out / f'h{h}'
        methods = build_prediction_methods(horizon=h, names=names, timed=True)
        state = None
        if a.resume:
            required = ['metadata/protocol_v5_run.json', 'scorer_timing.csv',
                        'comparison/summary_common_windows.csv']
            required += [f'protocol/{n}_protocol.csv' for n in names]
            required += ['gate_profile.csv', 'gate_hist.csv'] if a.gate_profile else []
            required += ['bound_check.csv'] if a.bound_check else []
            rdir = (Path(a.resume_dir) / f'h{h}') if a.resume_dir else default_resume_dir(out_h)
            if not rdir.is_absolute():
                rdir = ROOT / rdir
            config = dict(data_sha256=data_sha, dataset_min_obs=a.min_obs, horizon=h,
                          seed=a.seed, robustness=not a.no_robustness, methods=list(names),
                          gate_profile=bool(a.gate_profile), bound_check=bool(a.bound_check),
                          python=platform.python_version(), numpy=np.__version__,
                          code={f: sha256_file(ROOT / f) for f in CODE_FILES})
            state = ResumeState(rdir, config, required)
            if state.horizon_complete(out_h):
                print(f'horizon {h} is complete in {out_h}; not run again')
                continue
            methods = state.wrap_journals(methods)
        recorders = {}
        if a.bound_check:
            methods, recorders = add_bound_recorders(methods, h)
            if state is not None:
                state.recorders = recorders
        if df is None:
            df = pd.read_csv(data)
            df['item_id'] = df['item_id'].astype(str)
        fe = ResumableEvaluator(df, dataset_min_obs=a.min_obs, seed=a.seed,
                                robustness=not a.no_robustness, causal_universe=True,
                                horizon=h, resume_state=state)
        res = fe.run(methods, out_dir=out_h,
                     extra_meta=dict(data=str(a.data), dataset_min_obs=a.min_obs,
                                     horizon=h, n_forecast_coefficients=n_forecast_coefficients(h),
                                     window=WINDOW, min_obs=MIN_OBS, causal_universe=True,
                                     runner='tools/run_prediction.py'))
        stored = None
        if state is not None:
            stored = {n: state.done[n]['stats'] for n in state.skipped}
            for n in state.journals:
                if n not in stored:
                    stored[n] = state.journals[n].stats()
        timing_table(methods, res, h, stored).to_csv(out_h / 'scorer_timing.csv',
                                                     index=False, encoding='utf-8')
        if a.bound_check:
            brows = []
            for n in methods:
                if state is not None and n in state.skipped:
                    if state.done[n].get('bound'):
                        brows.append(state.done[n]['bound'])
                elif n in recorders:
                    brows.append(recorders[n].bound_stats())
            bc = pd.DataFrame(brows)
            bc.to_csv(out_h / 'bound_check.csv', index=False, encoding='utf-8')
            for _, r in bc.iterrows():
                print(f"bound check {r['method']:<9} ratio to WSPI in [{r['min_ratio']:.6g}, "
                      f"{r['max_ratio']:.6g}]  bounds [{r['lower_bound']:.6g}, "
                      f"{r['upper_bound']:.6g}]  {r['status']}")
        if a.gate_profile:
            t0 = time.time()
            gp, gh = gate_profile(fe, WINDOW, MIN_OBS)
            gp.to_csv(out_h / 'gate_profile.csv', index=False, encoding='utf-8')
            gh.to_csv(out_h / 'gate_hist.csv', index=False, encoding='utf-8')
            print(f'gate profile: {len(gp)} windows, {time.time() - t0:.1f}s')
        ids = set.intersection(*[set(d['window_id']) for d in res.values()])
        print(f'\ncommon windows: {len(ids)}')
        print('method        windows  ndcg@10  rho      rsi@10   robust')
        for n, d_ in res.items():
            d = d_[d_['window_id'].isin(ids)]
            print(f"{n:<12} {len(d):>8}  {d['ndcg@10'].mean():.4f}  {d['spearman_rho'].mean():.4f}  "
                  f"{d['rsi@10'].mean():.4f}  {d['robustness_distortion'].mean():7.2f}")
        print(f'\nSaved to {out_h}')
        if state is not None:
            state.mark_horizon_complete(dict(out=str(out_h), methods=list(names)))
        del fe, res
        gc.collect()


if __name__ == '__main__':
    main()
