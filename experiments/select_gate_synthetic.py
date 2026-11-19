r"""
Selection of the WSPI-FG backtest configuration on synthetic data
=================================================================
Chooses the number of backtest origins k in {2, 3, 4} and the backtest
source (W: coefficients of the same window; C: re-transform of the cut
window) of the forecastability gate of WSPI-FG.  Only synthetic series are
used; no evaluation dataset is read.  The specification (groups, seeds,
criterion, tolerance, cost order) was fixed before the single run and is
recorded in the pre-registration of the prediction study.

Criterion: for every synthetic window the true skill of the clamped AR(2)
forecaster against the implicit WSPI forecast is

    o = 1 - |log mu_F - log mu_true| / |log mu_L - log mu_true|

where mu_L is the recency-weighted mean of the observed trend-band
magnitudes a, mu_F = mu_L(a || f) with the clamped forecast f (lambda = 1),
and mu_true = mu_L(a || a_true) with the m true future coefficients
a_true = last m magnitudes of the window shifted forward by m * 2^(J-1)
slots.  For every configuration, group and horizon the Spearman correlation
of the gate g with o is computed; the score of a configuration is the mean
over the 7 groups x 3 horizons.  The winner has the highest score; every
configuration within TOL of the best is equivalent and the cheapest of them
is chosen (cost order W2 < W3 < W4 < C2 < C3 < C4).

Source W is the gate of the method (``backtest_gate_rows`` in
``tools/prediction_protocol.py``).  Source C exists only here: for
s = s0, ..., s0 + k_eff - 1 with s0 = ceil(m / 2) the window is cut to its
first N - 8s slots, reflect-padded on the left to N slots and transformed
again.  Cutting 8s slots and padding on the left moves the content by 2s
coefficients (an even number, so every coefficient keeps its tree).

Outputs (--out, CSV/JSON only):
  selection_cells.csv     config x group x horizon: n, Spearman(g, o)
  selection_summary.csv   config: mean Spearman, gap to best, chosen
  diagnostics.csv         group x horizon x scorer: mean |log error| of the
                          log score level, share of g = 0 (not used for the
                          selection; descriptive only)
  selection_run.json      settings, seeds, versions, chosen configuration

Example (from the project root, Windows):
  python experiments\select_gate_synthetic.py --out results\prediction\gate_selection_public

The result is identical to the single run of the pre-registration
(``tools/test_select_gate_synthetic.py`` compares them when the files exist).
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.stats import spearmanr
import dtcwt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.protocol_v5 import _pad_left_reflect                    # noqa: E402
from tools.prediction_protocol import (CLAMP, LEVEL, MIN_OBS, WINDOW,    # noqa: E402
                                       CoefFeatures, ar2_yw_rows,
                                       backtest_gate_rows, clamp_rows,
                                       gate_from_sums, recency_mean_rows,
                                       structural_gate, weighted_abs_rows)

HORIZONS = (1, 8, 24)
KS = (2, 3, 4)
SOURCES = ('W', 'C')
COST_ORDER = ['W2', 'W3', 'W4', 'C2', 'C3', 'C4']
TOL = 0.01
N_SERIES = 1000
BASE_SEED = 42
STEP = 2 ** (LEVEL - 1)                 # time step of trend-band coefficients (slots)
H_MAX = max(HORIZONS)
M_MAX = int(np.ceil(H_MAX / STEP))
LENGTH = WINDOW + M_MAX * STEP           # 64 + 24 = 88 slots
DEN_MIN = 1e-9
GROUPS = ('trend', 'lifecycle', 'random_walk', 'ar1_log', 'stationary',
          'seasonal', 'burst')


# =============================================================================
# Synthetic groups (latent rate, Poisson observations)
# =============================================================================

def _log_uniform(rng, lo, hi, n):
    return np.exp(rng.uniform(np.log(lo), np.log(hi), n))


def make_group(name: str, n: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    t = np.arange(LENGTH, dtype=np.float64)[None, :]
    if name == 'trend':
        b = _log_uniform(rng, 2, 200, n)[:, None]
        r = rng.uniform(-0.04, 0.04, n)[:, None]
        lam = b * np.exp(r * (t - (WINDOW - 1)))
    elif name == 'lifecycle':
        b = _log_uniform(rng, 5, 300, n)[:, None]
        tau = rng.uniform(10, 140, n)[:, None]
        kap = rng.uniform(1, 4, n)[:, None]
        u = (t + 1) / tau
        lam = b * u ** kap * np.exp(kap * (1 - u))
    elif name == 'random_walk':
        b = _log_uniform(rng, 2, 200, n)[:, None]
        sig = rng.uniform(0.03, 0.15, n)[:, None]
        eps = rng.standard_normal((n, LENGTH)) * sig
        lam = b * np.exp(np.cumsum(eps, axis=1))
    elif name == 'ar1_log':
        b = _log_uniform(rng, 2, 200, n)[:, None]
        phi = rng.uniform(0.8, 0.99, n)
        sig = rng.uniform(0.05, 0.3, n)
        z = np.empty((n, LENGTH))
        z[:, 0] = rng.standard_normal(n) * sig / np.sqrt(1 - phi ** 2)
        eps = rng.standard_normal((n, LENGTH)) * sig[:, None]
        for j in range(1, LENGTH):
            z[:, j] = phi * z[:, j - 1] + eps[:, j]
        lam = b * np.exp(z)
    elif name == 'stationary':
        b = _log_uniform(rng, 2, 200, n)[:, None]
        lam = np.repeat(b, LENGTH, axis=1)
    elif name == 'seasonal':
        b = _log_uniform(rng, 2, 200, n)[:, None]
        A = rng.uniform(0.2, 0.9, n)[:, None]
        ph = rng.uniform(0, 2 * np.pi, n)[:, None]
        lam = b * (1 + A * np.sin(2 * np.pi * t / 24 + ph))
    elif name == 'burst':
        b = _log_uniform(rng, 2, 200, n)[:, None]
        t0 = rng.randint(0, LENGTH, n)[:, None]
        B = b * rng.uniform(2, 20, n)[:, None]
        d = rng.uniform(2, 20, n)[:, None]
        lam = b + np.where(t >= t0, B * np.exp(-(t - t0) / d), 0.0)
    else:
        raise ValueError(name)
    return rng.poisson(lam).astype(np.float64)


# =============================================================================
# Gates
# =============================================================================

def magnitudes(tr, X: np.ndarray) -> np.ndarray:
    """|trend-band coefficients| of every row, after reflect-padding on the left to N."""
    return CoefFeatures(_pad_left_reflect(X, WINDOW), tr, LEVEL).a


def gate_W(a: np.ndarray, m: int, k: int) -> np.ndarray:
    """Source W: backtest inside the coefficients of the window."""
    if k > a.shape[1] - m - 3:
        raise ValueError('k too large for source W')
    return backtest_gate_rows(a, m, k)


def first_shift(m: int) -> int:
    """s0 = ceil(m / 2): smallest cut for which the true coefficients lie in the window (2s >= m)."""
    return int(np.ceil(m / 2))


def k_eff_C(m: int, k: int) -> int:
    """Number of origins of source C: s <= 4 keeps half of the cut window real data."""
    return min(k, 5 - first_shift(m))


def gate_C(x: np.ndarray, a: np.ndarray, m: int, k: int, tr) -> np.ndarray:
    """Source C: re-transform of the window cut by 8s slots (s = s0, ...)."""
    n = a.shape[1]
    s0 = first_shift(m)
    s_f = np.zeros(a.shape[0])
    s_mu = np.zeros(a.shape[0])
    for s in range(s0, s0 + k_eff_C(m, k)):
        b = magnitudes(tr, x[:, :WINDOW - 8 * s])      # cut 8s slots, reflect-pad to N
        f = clamp_rows(ar2_yw_rows(b, m), b[:, -1])
        r = np.repeat(recency_mean_rows(b)[:, None], m, axis=1)
        y = a[:, n - 2 * s:n - 2 * s + m]
        s_f += weighted_abs_rows(f - y)
        s_mu += weighted_abs_rows(r - y)
    return gate_from_sums(s_f, s_mu, recency_mean_rows(a) * k_eff_C(m, k))


def spearman0(x, y) -> float:
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return 0.0
    return float(spearmanr(x, y).correlation)


# =============================================================================
# Main
# =============================================================================

def run(out: Path) -> pd.DataFrame:
    """Run the selection once and write the four files into ``out``."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    tr = dtcwt.Transform1d(biort='near_sym_a', qshift='qshift_a')

    cells, diags, kept = [], [], {}
    for gi, gname in enumerate(GROUPS):
        S = make_group(gname, N_SERIES, BASE_SEED + 1000 * gi)
        x = S[:, :WINDOW]
        ok = (x > 0).sum(axis=1) >= MIN_OBS            # eligibility of protocol V5
        S, x = S[ok], x[ok]
        kept[gname] = int(ok.sum())
        F = CoefFeatures(_pad_left_reflect(x, WINDOW), tr, LEVEL)
        a = F.a
        mu_L = recency_mean_rows(a)
        g2 = structural_gate(F.R, F.WE)                 # structural gate of WSPI-F2
        for h in HORIZONS:
            m = int(np.ceil(h / STEP))
            f = clamp_rows(ar2_yw_rows(a, m), a[:, -1])
            mu_F = recency_mean_rows(np.concatenate([a, f], axis=1))
            a_true = magnitudes(tr, S[:, m * STEP:m * STEP + WINDOW])[:, -m:]
            mu_T = recency_mean_rows(np.concatenate([a, a_true], axis=1))
            with np.errstate(divide='ignore', invalid='ignore'):
                lL, lF, lT = np.log(mu_L), np.log(mu_F), np.log(mu_T)
                den = np.abs(lL - lT)
                o = 1.0 - np.abs(lF - lT) / den
            valid = np.isfinite(lL) & np.isfinite(lF) & np.isfinite(lT) & (den >= DEN_MIN)
            diag = {'WSPI': np.zeros_like(lL), 'AR2-clamp(lambda=1)': np.ones_like(lL),
                    'WSPI-F2(structural)': g2}
            for src in SOURCES:
                for k in KS:
                    cfg = f'{src}{k}'
                    g = gate_W(a, m, k) if src == 'W' else gate_C(x, a, m, k, tr)
                    cells.append({'config': cfg, 'source': src, 'k': k,
                                  'k_eff': k if src == 'W' else k_eff_C(m, k),
                                  'group': gname, 'horizon': h, 'm': m,
                                  'n_series': len(a), 'n_valid': int(valid.sum()),
                                  'spearman_g_o': spearman0(g[valid], o[valid])})
                    diag[f'WSPI-FG({cfg})'] = g
            fin = np.isfinite(lL) & np.isfinite(lF) & np.isfinite(lT)
            for name, lam in diag.items():
                lev = (1 - lam) * lL + lam * lF
                diags.append({'group': gname, 'horizon': h, 'm': m, 'scorer': name,
                              'n': int(fin.sum()),
                              'mean_abs_log_err': float(np.mean(np.abs(lev[fin] - lT[fin]))),
                              'share_lambda0': float(np.mean(lam[fin] == 0)),
                              'mean_lambda': float(np.mean(lam[fin]))})

    cdf = pd.DataFrame(cells)
    summ = cdf.groupby('config', sort=False)['spearman_g_o'].agg(['mean', 'min', 'max']).reset_index()
    summ.columns = ['config', 'mean_spearman', 'min_cell', 'max_cell']
    best = summ['mean_spearman'].max()
    summ['gap_to_best'] = best - summ['mean_spearman']
    summ['within_tol'] = summ['gap_to_best'] < TOL
    summ['cost_rank'] = summ['config'].map({c: i for i, c in enumerate(COST_ORDER)})
    chosen = summ[summ['within_tol']].sort_values('cost_rank').iloc[0]['config']
    summ['chosen'] = summ['config'] == chosen
    summ = summ.sort_values('cost_rank')

    cdf.to_csv(out / 'selection_cells.csv', index=False, encoding='utf-8')
    summ.to_csv(out / 'selection_summary.csv', index=False, encoding='utf-8')
    pd.DataFrame(diags).to_csv(out / 'diagnostics.csv', index=False, encoding='utf-8')
    meta = dict(purpose='selection of k and backtest source of the WSPI-FG gate (synthetic data only)',
                groups=list(GROUPS), n_series_per_group=N_SERIES, kept_after_min_obs=kept,
                seeds={g: BASE_SEED + 1000 * i for i, g in enumerate(GROUPS)},
                length=LENGTH, window=WINDOW, level=LEVEL, coef_step=STEP, min_obs=MIN_OBS,
                horizons=list(HORIZONS), ks=list(KS), sources=list(SOURCES), clamp=CLAMP,
                criterion='mean over groups x horizons of Spearman(g, o); o = 1 - |log mu_F - log mu_true| / |log mu_L - log mu_true|',
                den_min=DEN_MIN, zero_rel=1e-9, tolerance=TOL, cost_order=COST_ORDER,
                best_mean_spearman=float(best), chosen=str(chosen),
                versions=dict(python=platform.python_version(), numpy=np.__version__,
                              scipy=scipy.__version__, pandas=pd.__version__,
                              dtcwt=dtcwt.__version__),
                host=platform.platform(), runtime_s=round(time.time() - t_start, 1),
                created=time.strftime('%Y-%m-%d %H:%M:%S'))
    (out / 'selection_run.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    return summ


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--out', required=True)
    a_ = ap.parse_args(argv)
    out = Path(a_.out)
    if not out.is_absolute():
        out = ROOT / out
    summ = run(out)
    print(summ.to_string(index=False))
    print(f"\nchosen: {summ.loc[summ['chosen'], 'config'].iloc[0]}\nSaved to {out}")


if __name__ == '__main__':
    main()
