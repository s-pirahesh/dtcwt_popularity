r"""
Estimate the running time of the full prediction runs
=====================================================
Run A (ten methods, six scenarios, horizons 1, 8, 24) and run B (WSPI-FA and
ARIMA, YouTube and hourly taxi, three horizons) are estimated from timings that
were measured, never from a guess.

Run A.  For every scenario and method the time of the horizon-1 run is scaled
by the ratio  run_seconds(h, method) / run_seconds(1, method)  of a preview run
(``--preview``, normally YouTube hourly with ``--horizons 1 8 24``).  The
horizon-1 time comes from, in this order of preference,
  1. ``<base>/<scenario>/h1/scorer_timing.csv`` (``--base``: a horizon-1 run of
     that scenario with the new code), or
  2. the ``runtime_seconds`` of the stored reference run, ``thesis_v16``; WSPI-FG
     is taken as WSPI-F2 times the preview ratio FG / F2, and MA64 as
     Persistence times the preview ratio MA64 / Persistence.
Assumption: the ratio between horizons measured on the preview carries over to
the other scenarios.  (A check is the ratio itself on a second scenario.)

Run B.  The number of fits is the number of rows the scorer sees:
(sum of the items over the windows) + (at most 50 robustness rows per window),
from the reference WSPI protocol file, times (windows(h) / windows(1)).  The
seconds per fit of a scorer come from a timing sample (``--fit-timing``:
``<dir>/<scenario>/h1/scorer_timing.csv`` of ``run_prediction.py`` on the
output of ``make_timing_sample.py``: scorer_seconds / rows).  ARIMA falls back
to ``results/thesis_v16/arima_timing.csv`` if the sample has no ARIMA row.

    python tools\estimate_runtime.py --preview results\prediction\smoke\youtube_hourly ^
           [--base results\prediction\repro] [--fit-timing results\prediction\smoke\timing_sample]

Writes ``runtime_estimate.csv`` (``--out``) and prints totals in hours.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

SCENARIOS = ('youtube_hourly', 'taxi_hourly', 'taxi_30min', 'taxi_5min',
             'movielens_daily', 'movielens_weekly')
RUN_B_SCENARIOS = ('youtube_hourly', 'taxi_hourly')
HORIZONS = (1, 8, 24)
RUN_A_METHODS = ('WSPI', 'WSPI-FG', 'WSPI-F2', 'WSPI-F-YW', 'WSPI-F', 'WSPI-FT',
                 'Persistence', 'Holt', 'ARYW', 'MA64')
PROXY = {'WSPI-FG': 'WSPI-F2', 'MA64': 'Persistence'}     # methods the reference run lacks
ROBUST_ROWS = 50


def _abs(p):
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def preview_runs(preview: Path):
    """run_seconds[h][method] from <preview>/h<h>/scorer_timing.csv."""
    out = {}
    for h in HORIZONS:
        f = preview / f'h{h}' / 'scorer_timing.csv'
        if not f.exists():
            raise SystemExit(f'preview timing missing: {f}')
        t = pd.read_csv(f)
        out[h] = dict(zip(t['method'], t['run_seconds']))
    return out


def base_runs(scenario, base, reference, ratio):
    """Horizon-1 seconds per run-A method for one scenario, and where they come from."""
    if base is not None:
        f = base / scenario / 'h1' / 'scorer_timing.csv'
        if f.exists():
            t = pd.read_csv(f)
            d = dict(zip(t['method'], t['run_seconds']))
            if all(m in d for m in RUN_A_METHODS):
                return {m: d[m] for m in RUN_A_METHODS}, 'horizon-1 run (new code)'
    meta = json.loads((reference / scenario / 'metadata' / 'protocol_v5_run.json')
                      .read_text(encoding='utf-8'))['runtime_seconds']
    d = {}
    for m in RUN_A_METHODS:
        if m in meta:
            d[m] = meta[m]
        else:
            d[m] = meta[PROXY[m]] * ratio[m]           # ratio[m] = preview(h1, m) / preview(h1, proxy)
    return d, 'thesis_v16 (' + ', '.join(f'{k} as {v}' for k, v in PROXY.items()) + ')'


def rows_per_scenario(scenario, reference, base):
    """(rows at h = 1, windows at h = 1)."""
    if base is not None:
        f = base / scenario / 'h1' / 'scorer_timing.csv'
        if f.exists():
            t = pd.read_csv(f)
            r = t[t['method'] == 'WSPI'].iloc[0]
            return float(r['rows']), int(r['windows'])
    d = pd.read_csv(reference / scenario / 'protocol' / 'WSPI_protocol.csv',
                    usecols=['window_id', 'num_items'])
    return float(d['num_items'].sum() + ROBUST_ROWS * len(d)), len(d)


def seconds_per_fit(scenario, method, fit_timing, arima_timing):
    if fit_timing is not None:
        f = fit_timing / scenario / 'h1' / 'scorer_timing.csv'
        if f.exists():
            t = pd.read_csv(f)
            r = t[t['method'] == method]
            if len(r) and float(r.iloc[0]['rows']) > 0:
                return float(r.iloc[0]['scorer_seconds']) / float(r.iloc[0]['rows']), 'timing sample'
    if method == 'ARIMA' and arima_timing is not None and arima_timing.exists():
        return float(pd.read_csv(arima_timing)['seconds_per_fit'].mean()), 'arima_timing.csv'
    return float('nan'), 'not measured'


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--preview', required=True,
                    help='run_prediction.py output with --horizons 1 8 24 on one scenario')
    ap.add_argument('--reference', default='results/thesis_v16/prediction')
    ap.add_argument('--base', default=None, help='horizon-1 runs of all scenarios (optional)')
    ap.add_argument('--fit-timing', default=None, help='timing samples for ARIMA / WSPI-FA')
    ap.add_argument('--arima-timing', default='results/thesis_v16/arima_timing.csv')
    ap.add_argument('--out', default=None)
    a = ap.parse_args(argv)

    preview, reference = _abs(a.preview), _abs(a.reference)
    base = _abs(a.base) if a.base else None
    fit_timing = _abs(a.fit_timing) if a.fit_timing else None
    arima_timing = _abs(a.arima_timing) if a.arima_timing else None
    out = _abs(a.out) if a.out else preview.parent / 'runtime_estimate.csv'

    pv = preview_runs(preview)
    ratio_h = {h: {m: pv[h][m] / pv[1][m] for m in RUN_A_METHODS} for h in HORIZONS}
    ratio_proxy = {m: pv[1][m] / pv[1][PROXY[m]] for m in PROXY}
    rows = []

    for s in SCENARIOS:
        b1, basis = base_runs(s, base, reference, ratio_proxy)
        for h in HORIZONS:
            for m in RUN_A_METHODS:
                rows.append(dict(run='A', scenario=s, horizon=h, method=m,
                                 est_seconds=b1[m] * ratio_h[h][m], basis=basis))

    for s in RUN_B_SCENARIOS:
        r1, w1 = rows_per_scenario(s, reference, base)
        for h in HORIZONS:
            n_rows = r1 * (w1 - (h - 1)) / w1
            for m in ('WSPI-FA', 'ARIMA'):
                spf, src = seconds_per_fit(s, m, fit_timing, arima_timing)
                rows.append(dict(run='B', scenario=s, horizon=h, method=m,
                                 est_seconds=n_rows * spf, basis=f'{n_rows:,.0f} fits x {spf:.5f} s ({src})'))

    est = pd.DataFrame(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    est.to_csv(out, index=False, encoding='utf-8')

    hrs = lambda x: x / 3600.0                                              # noqa: E731
    A = est[est['run'] == 'A']
    print('RUN A (hours; one process per scenario runs its three horizons one after another)')
    t = A.pivot_table(index='scenario', columns='horizon', values='est_seconds', aggfunc='sum') \
         .reindex(list(SCENARIOS)).map(hrs)
    t['all horizons'] = t.sum(axis=1)
    print(t.round(2).to_string())
    print(f"  sum over scenarios (one core): {hrs(A['est_seconds'].sum()):.2f} h;  "
          f"six scenarios in parallel: {t['all horizons'].max():.2f} h (longest scenario)")
    print(f"  basis: {', '.join(sorted(set(A['basis'])))}")
    B = est[est['run'] == 'B']
    tot = lambda x: x.sum(min_count=1)                                      # noqa: E731
    print('\nRUN B (hours; one process per scenario and horizon runs WSPI-FA, then ARIMA)')
    per_method = B.pivot_table(index=['scenario', 'method'], columns='horizon',
                               values='est_seconds', aggfunc=tot, dropna=False).map(hrs)
    print(per_method.round(2).to_string())
    proc = B.groupby(['scenario', 'horizon'])['est_seconds'].apply(lambda x: x.sum(skipna=False))
    if proc.isna().any():
        print('  NaN: seconds per fit not measured (run the timing sample, then repeat)')
    else:
        print(f"  {len(proc)} processes; sum (core-hours): {hrs(proc.sum()):.1f} h;  "
              f"all in parallel: {hrs(proc.max()):.1f} h (longest process)")
        print('  the WSPI run that every call includes (minutes) is not added')
    print(f'\nwritten: {out}')


if __name__ == '__main__':
    sys.exit(main())
