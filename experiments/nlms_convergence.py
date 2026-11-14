r"""
Convergence of NLMS on the trend-band coefficients of real windows
==================================================================
WSPI-F adapts an order-2 NLMS filter (step size mu = 0.2, persistence-anchored,
on the increments) along the 16 trend-band coefficients of each window and
forecasts the next coefficient.  This script asks whether the filter has
converged by the end of the window: it records the a-priori one-step error of
NLMS at every position t of the coefficient sequence, for every item of every
window that the prediction runs evaluate (protocol V5, N = 64, 32 observed
slots, causal item rule, horizon 1), and compares it with the one-step forecast
of AR(2) Yule-Walker fitted on the same prefix a[:t] (the forecaster of
WSPI-F2).

    error(t) = |forecast of a_t - a_t| / mu_L(a)

mu_L(a) is the recency-weighted mean of the 16 coefficients of the window
(rows with mu_L <= 1e-12 are left out).  Positions t = 3, ..., 15 (0-based) are
the positions where NLMS of order 2 makes a prediction (``nlms_walk`` of
``methods/wspi_forecast.py``); AR(2) falls back to the last value for prefixes
shorter than 4 (t = 3).  If NLMS converges, its error falls with t towards the
AR(2) error.  Descriptive only (Supplementary Information).

Outputs (``--out``/h1/):
  nlms_convergence.csv        per method and t: n, mean, quantiles 0.1, 0.25,
                              0.5, 0.75, 0.9 of the error
  nlms_convergence_late.csv   per method: the same over t = 12..15 (the end of
                              the window) and the share of item-windows in which
                              NLMS has a smaller error than AR(2) there
  metadata/nlms_convergence_run.json

    python experiments\nlms_convergence.py --data data\datasets\youtube_hourly.csv ^
           --min-obs 50 --out results\paper2\run_C_nlms\youtube_hourly

``--horizons`` accepts only 1 (the forecast is one coefficient ahead);
``--resume`` skips a scenario whose output files exist.  Both are there so that
``tools/run_prediction_plan.py`` can start this script like the other runs.

Author: Sajjad Pirahesh
Date: November 2026
"""
import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import dtcwt  # noqa: E402

from evaluation.protocol_v5 import ProtocolV5Evaluator, _pad_left_reflect  # noqa: E402
from methods.wspi_forecast import NLMS_MU  # noqa: E402
from tools.prediction_protocol import (LEVEL, MIN_OBS, WINDOW, CoefFeatures,  # noqa: E402
                                       ar2_yw_rows, iter_eligible_windows,
                                       recency_mean_rows)

QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)
LATE = (12, 13, 14, 15)
ORDER = 2
EPS = 1e-8


def nlms_apriori_rows(a: np.ndarray, mu: float = NLMS_MU, eps: float = EPS):
    """A-priori one-step predictions of the order-2 NLMS walk for every row
    (row-wise replica of ``methods.wspi_forecast.nlms_walk``).  Column j is
    the prediction of a[:, ORDER + 1 + j]."""
    a = np.asarray(a, dtype=np.float64)
    nr, n = a.shape
    p = ORDER
    d = np.diff(a, axis=1)
    w = np.zeros((nr, p))
    preds = []
    for k in range(p + 1, n):
        x = d[:, k - 1 - p:k - 1][:, ::-1]
        y_hat = a[:, k - 1] + np.einsum('ij,ij->i', w, x)
        e = a[:, k] - y_hat
        w = w + mu * (e / (np.einsum('ij,ij->i', x, x) + eps))[:, None] * x
        preds.append(y_hat)
    return np.column_stack(preds)


def window_errors(X: np.ndarray, tr, target: int):
    """(positions t, NLMS errors, AR(2) errors) of one window matrix; the
    errors are (rows, positions) arrays."""
    F = CoefFeatures(_pad_left_reflect(np.asarray(X, dtype=np.float64), target), tr, LEVEL)
    a = F.a
    mu_L = recency_mean_rows(a)
    keep = np.isfinite(mu_L) & (mu_L > 1e-12)
    a, mu_L = a[keep], mu_L[keep]
    n = a.shape[1]
    ts = np.arange(ORDER + 1, n)
    if len(a) == 0:
        return ts, np.empty((0, len(ts))), np.empty((0, len(ts)))
    e_nlms = np.abs(nlms_apriori_rows(a) - a[:, ts]) / mu_L[:, None]
    e_ar = np.empty_like(e_nlms)
    for j, t in enumerate(ts):
        e_ar[:, j] = np.abs(ar2_yw_rows(a[:, :t], 1)[:, 0] - a[:, t]) / mu_L
    return ts, e_nlms, e_ar


def summarise(ts, parts_nlms, parts_ar):
    E = {'NLMS': np.concatenate(parts_nlms, axis=0), 'AR2-YW': np.concatenate(parts_ar, axis=0)}
    rows = []
    for m, e in E.items():
        for j, t in enumerate(ts):
            x = e[:, j].astype(np.float64)
            r = dict(method=m, t=int(t), n=len(x), mean=float(x.mean()) if len(x) else np.nan)
            for q in QUANTILES:
                r[f'q{q:g}'] = float(np.quantile(x, q)) if len(x) else np.nan
            rows.append(r)
    late_cols = [j for j, t in enumerate(ts) if t in LATE]
    lrows = []
    win = (E['NLMS'][:, late_cols].mean(axis=1) < E['AR2-YW'][:, late_cols].mean(axis=1))
    for m, e in E.items():
        x = e[:, late_cols].astype(np.float64).ravel()
        r = dict(method=m, t_from=min(LATE), t_to=max(LATE), n=len(x), mean=float(x.mean()))
        for q in QUANTILES:
            r[f'q{q:g}'] = float(np.quantile(x, q))
        r['share_nlms_better'] = float(win.mean())
        lrows.append(r)
    return pd.DataFrame(rows), pd.DataFrame(lrows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--data', required=True)
    ap.add_argument('--min-obs', type=int, required=True, help='causal item threshold')
    ap.add_argument('--out', required=True)
    ap.add_argument('--horizons', nargs='+', type=int, default=[1])
    ap.add_argument('--resume', action='store_true')
    a = ap.parse_args(argv)
    if list(a.horizons) != [1]:
        print('only horizon 1 (one coefficient ahead)')
        return 2
    data = Path(a.data)
    data = data if data.is_absolute() else ROOT / data
    out = Path(a.out)
    out = (out if out.is_absolute() else ROOT / out) / 'h1'
    files = [out / 'nlms_convergence.csv', out / 'nlms_convergence_late.csv',
             out / 'metadata' / 'nlms_convergence_run.json']
    if a.resume and all(f.exists() for f in files):
        print(f'complete in {out}; not run again')
        return 0
    t0 = time.time()
    df = pd.read_csv(data)
    df['item_id'] = df['item_id'].astype(str)
    fe = ProtocolV5Evaluator(df, dataset_min_obs=a.min_obs, seed=42, robustness=False,
                             causal_universe=True, horizon=1)
    tr = dtcwt.Transform1d(biort='near_sym_a', qshift='qshift_a')
    target = max(WINDOW, 2 ** (LEVEL + 1))
    pn, pa, n_win, ts = [], [], 0, None
    for k, idx in iter_eligible_windows(fe, WINDOW, MIN_OBS):
        ts, en, ea = window_errors(fe._matrix(idx, k, WINDOW), tr, target)
        pn.append(en.astype(np.float32))
        pa.append(ea.astype(np.float32))
        n_win += 1
    curve, late = summarise(ts, pn, pa)
    (out / 'metadata').mkdir(parents=True, exist_ok=True)
    curve.to_csv(files[0], index=False, encoding='utf-8')
    late.to_csv(files[1], index=False, encoding='utf-8')
    meta = dict(data=str(a.data), dataset_min_obs=a.min_obs, horizon=1, window=WINDOW,
                min_obs=MIN_OBS, causal_universe=True, level=LEVEL, nlms_mu=NLMS_MU,
                nlms_order=ORDER, error='|forecast - a_t| / mu_L(a)', positions=[int(t) for t in ts],
                late_positions=list(LATE), n_windows=n_win,
                n_item_windows=int(sum(len(x) for x in pn)),
                python=platform.python_version(), numpy=np.__version__,
                runtime_seconds=round(time.time() - t0, 1),
                created=time.strftime('%Y-%m-%d %H:%M:%S'))
    files[2].write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print(late.to_string(index=False))
    print(f'{n_win} windows, {meta["n_item_windows"]} item-windows, '
          f'{meta["runtime_seconds"]}s; saved to {out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
