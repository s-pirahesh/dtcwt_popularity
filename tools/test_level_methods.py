r"""
Unit checks for the window-mean level methods WSPI-M and WSPI-H (run D)
=======================================================================
Covers the additions to ``methods/wspi_forecast.py``,
``tools/prediction_protocol.py``, ``tools/run_prediction.py`` and
``tools/run_prediction_plan.py``.

  1. Closed form: the reference class gives
        P = ((1 - s_m) mu_L + s_m f) exp(R - W_E)
     with f = a_bar (WSPI-M) and f = mu_L clip(a_bar / mu_L, 1/c, c) (WSPI-H),
     s_m = (2 - 2^(1-m)) / W_(n+m); s_m is about 0.50, 0.75, 0.98 for
     n = 16 and m = 1, 2, 6; the mean forecaster.
  2. Bound of WSPI-H: P_H / WSPI lies in [1 - s_m (1 - 1/c), 1 + s_m (c - 1)]
     and so in [1/c, c], for m = 1..6 (random search with spikes, decays and
     sparse rows); both limits are attained; WSPI-H equals WSPI-M where
     a_bar / mu_L is inside [1/c, c].  WSPI-M has no upper bound (an example
     above c).
  3. Batch scorers == reference class on full 64-slot windows, h = 1, 8, 24.
  4. The methods of runs A and C are unchanged: the default clamp anchor is a_n
     (same floats as before for every clamped method of runs A and C, checked
     against a frozen copy of the old clamped branch); the level anchor is
     refused with a gate.
  5. Names and wiring: method lists, plan D, the bound check of
     run_prediction.py uses the WSPI-H limits and passes on test rows.

    python tools/test_level_methods.py
"""
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from methods.wspi_forecast import (CLAMP, WSPIForecast, get_control_methods,  # noqa: E402
                                   get_coefficient_domain_methods, get_level_methods,
                                   level_anchor_bounds, mean_forecast, mean_share,
                                   n_forecast_coefficients, recency_mean)
from methods.wspi_assessment import WSPIAssessment                        # noqa: E402
from tools import prediction_protocol as P                                # noqa: E402
from tools import run_prediction as RP                                    # noqa: E402
from tools import run_prediction_plan as PL                               # noqa: E402
from tools.test_prediction import signals                                 # noqa: E402

HORIZONS = (1, 8, 24)
FAILED = []


def check(name, ok, detail=''):
    print(f"{'OK  ' if ok else 'FAIL'} {name} {detail}")
    if not ok:
        FAILED.append(name)
    return ok


def scorer(name, h):
    return P.build_prediction_methods(horizon=h, names=[name])[name].scorer


def random_rows(rng, n_rows=300):
    """Poisson rows with spikes in the last slots, early bursts that decay,
    sparse rows and constant rows."""
    rows = []
    for i in range(n_rows):
        x = rng.poisson(rng.gamma(1.5, 10.0), 64).astype(float)
        kind = i % 5
        if kind == 1:
            x[-rng.integers(1, 6):] *= rng.uniform(0, 30)
        elif kind == 2:
            x[:rng.integers(8, 40)] *= rng.uniform(5, 50)       # early burst, a_bar >> mu_L
        elif kind == 3:
            x[rng.random(64) < 0.8] = 0
        elif kind == 4 and i % 10 == 4:
            x[:] = 3.0
        rows.append(x)
    return np.vstack(rows)


# =============================================================================
# 1. Closed form
# =============================================================================

def test_closed_form(X):
    check('mean_share n=16: m=1, 2, 6 about 0.50, 0.75, 0.98',
          abs(mean_share(16, 1) - 0.5) < 1e-4 and abs(mean_share(16, 2) - 0.75) < 1e-4 and
          abs(mean_share(16, 6) - 0.984375) < 1e-4,
          f'({mean_share(16, 1):.6f}, {mean_share(16, 2):.6f}, {mean_share(16, 6):.6f})')
    check('mean forecast: plain mean repeated',
          np.array_equal(mean_forecast(np.array([1.0, 2.0, 6.0]), 4), np.full(4, 3.0)) and
          np.array_equal(mean_forecast(np.array([]), 2), np.zeros(2)))
    for h in HORIZONS:
        m = n_forecast_coefficients(h)
        meths = get_level_methods(horizon=h)
        worst = {'WSPI-M': 0.0, 'WSPI-H': 0.0}
        for x in X:
            obj = meths['WSPI-M']
            a, _, pyr = obj.forecast_lowpass(x)
            R, WE = obj._structure(a, pyr)
            mu_L = recency_mean(a)
            if mu_L <= 1e-12:
                continue
            s = mean_share(len(a), m)
            abar = float(np.mean(a))
            struct = np.exp(R - WE)
            want = {'WSPI-M': ((1 - s) * mu_L + s * abar) * struct,
                    'WSPI-H': ((1 - s) * mu_L + s * mu_L * np.clip(abar / mu_L, 1 / CLAMP, CLAMP))
                    * struct}
            for nm in worst:
                got = meths[nm].assess_single(x)
                worst[nm] = max(worst[nm], abs(got - want[nm]) / max(abs(want[nm]), 1e-12))
        for nm, w in worst.items():
            check(f'{nm} == closed form  h={h:<2}', w < 1e-12, f'(max rel. diff {w:.1e})')


# =============================================================================
# 2. Bound of WSPI-H
# =============================================================================

def test_bound(rng):
    worst, worst_c = 0.0, 0.0
    hit_lo = hit_hi = 0
    m_above_c = False
    same_inside = True
    for m_h in (1, 5, 9, 13, 17, 21):                      # m = ceil(h / 4) = 1..6
        m = n_forecast_coefficients(m_h)
        lo, hi = level_anchor_bounds(16, m, CLAMP)
        check(f'limits inside [1/c, c]  m={m}', 1 / CLAMP <= lo <= 1 <= hi <= CLAMP,
              f'([{lo:.4f}, {hi:.4f}])')
        meths = get_level_methods(horizon=m_h)
        base = WSPIForecast(predictor='mean', gate='none', horizon=m_h, clamp=CLAMP,
                            clamp_anchor='level', phi_d=0.0)
        wspi = WSPIAssessment()
        X = random_rows(rng)
        for x in X:
            w = base.assess_single(x)
            if w <= 0:
                continue
            ph = meths['WSPI-H'].assess_single(x)
            pm = meths['WSPI-M'].assess_single(x)
            r = ph / w
            worst = max(worst, lo / r - 1.0, r / hi - 1.0)
            worst_c = max(worst_c, (1 / CLAMP) / r - 1.0, r / CLAMP - 1.0)
            hit_lo += int(abs(r / lo - 1.0) < 1e-12)
            hit_hi += int(abs(r / hi - 1.0) < 1e-12)
            m_above_c |= pm / w > CLAMP
            a, _, _ = meths['WSPI-H'].forecast_lowpass(x)
            q = float(np.mean(a)) / recency_mean(a)
            if 1 / CLAMP < q < CLAMP:
                same_inside &= abs(ph / pm - 1.0) < 1e-12
            if m_h == 1:
                ws = wspi.assess_single(x)
                same_inside &= abs(w / ws - 1.0) < 1e-12        # phi_d = 0 is WSPI
    check('WSPI-H inside its limits, m = 1..6', worst <= 1e-9, f'(worst excess {worst:.1e})')
    check('WSPI-H inside [1/c, c], m = 1..6', worst_c <= 1e-9, f'(worst excess {worst_c:.1e})')
    check('both limits attained', hit_lo > 0 and hit_hi > 0, f'(lower {hit_lo}, upper {hit_hi})')
    check('WSPI-H == WSPI-M where a_bar / mu_L is inside [1/c, c]; phi_d = 0 is WSPI',
          bool(same_inside))
    check('WSPI-M has no upper bound (ratio above c found)', bool(m_above_c))
    obj = get_level_methods(horizon=24)['WSPI-H']
    check('ratio_bounds of the class == level_anchor_bounds',
          obj.ratio_bounds(16) == level_anchor_bounds(16, 6, CLAMP))
    try:
        get_level_methods(horizon=8)['WSPI-M'].ratio_bounds(16)
        ok = False
    except ValueError:
        ok = True
    check('WSPI-M (no clamp) has no ratio_bounds', ok)


# =============================================================================
# 3. Batch scorers against the reference class
# =============================================================================

def test_batch(X):
    for h in HORIZONS:
        refs = get_level_methods(horizon=h)
        for name, obj in refs.items():
            fast = scorer(name, h)(X)
            slow = np.array([obj.assess_single(x) for x in X])
            rel = np.abs(fast - slow) / np.maximum(np.abs(slow), 1e-9)
            check(f'scorer == class  h={h:<2} {name}', rel.max() < 1e-10,
                  f'max rel. diff {rel.max():.2e}')
        check(f'scorer names  h={h:<2}',
              scorer('WSPI-M', h).__name__ == f'wspi_m_h{h}_W64' and
              scorer('WSPI-H', h).__name__ == f'wspi_h_c4_h{h}_W64')


# =============================================================================
# 4. Runs A and C unchanged
# =============================================================================

def old_clamped_branch(X, name, h):
    """Frozen copy of the clamped branch of make_coef_scorer before run D
    (clamp anchored at the last coefficient)."""
    tr = P.dtcwt.Transform1d(biort='near_sym_a', qshift='qshift_a')
    m = n_forecast_coefficients(h)
    F = P.CoefFeatures(P._pad_left_reflect(np.asarray(X, dtype=np.float64), 64), tr, 3)
    a = F.a
    raw = P.theilsen_rows(a, m) if name == 'WSPI-FT' else P.ar2_yw_rows(a, m)
    struct = np.exp(F.R - F.WE)
    fc = P.clamp_rows(raw, a[:, -1])
    mu_L = P.recency_mean_rows(a)
    mu_hat = P.recency_mean_rows(np.concatenate([a, fc], axis=1))
    ok = np.isfinite(mu_L) & (mu_L > 1e-12)
    mu_hat = np.where(np.isfinite(mu_hat) & (mu_hat > 1e-12), mu_hat, mu_L)
    g = P.structural_gate(F.R, F.WE) if name == 'WSPI-F2' else (
        P.backtest_gate_rows(a, m) if name == 'WSPI-FG' else np.ones_like(mu_L))
    with np.errstate(divide='ignore', invalid='ignore'):
        blended = mu_L * np.power(np.where(ok, mu_hat / np.where(ok, mu_L, 1.0), 1.0), g)
    return np.where(ok, blended * struct, 0.0)


def test_unchanged(X):
    ok = True
    for h in HORIZONS:
        for name in ('WSPI-F2', 'WSPI-FT', 'WSPI-FG'):
            ok &= bool(np.array_equal(scorer(name, h)(X), old_clamped_branch(X, name, h)))
    check('run A clamped scorers == frozen old branch (same floats)', ok)
    objs = dict(get_coefficient_domain_methods(horizon=8, include_arma=False))
    objs.update(get_control_methods(horizon=8))
    check('default clamp anchor is a_n for every method of runs A and C',
          all(o.clamp_anchor == 'last' for o in objs.values()))
    for gate in ('structural', 'backtest', 'skill'):
        try:
            WSPIForecast(predictor='mean', gate=gate, clamp=CLAMP, clamp_anchor='level',
                         skill_kappa=2.0)
            ok = False
        except ValueError:
            ok = True
        check(f'level anchor refused with gate {gate}', ok)
    try:
        WSPIForecast(predictor='mean', clamp_anchor='mu')
        ok = False
    except ValueError:
        ok = True
    check('unknown clamp anchor refused', ok)


# =============================================================================
# 5. Names and wiring
# =============================================================================

def test_names(X):
    check('level method lists', P.LEVEL_METHODS == ('WSPI-M', 'WSPI-H') and
          P.RUN_D_METHODS == ('WSPI', 'WSPI-M', 'WSPI-H'))
    check('plan D methods == prediction_protocol.RUN_D_METHODS',
          tuple(PL.PLANS['D']['methods']) == P.RUN_D_METHODS)
    jobs = PL.build_jobs('D', 'results/paper2')
    check('plan D: 18 jobs with run_prediction.py, bound check, output run_D',
          len(jobs) == 18 and all(Path(j['cmd'][1]).name == 'run_prediction.py' and
                                  '--bound-check' in j['cmd'] and '--resume' in j['cmd'] and
                                  '/run_D/' in j['out'] for j in jobs))
    check('plans A, B, C unchanged',
          PL.PLANS['A']['methods'] is None and
          tuple(PL.PLANS['B']['methods']) == ('WSPI', 'WSPI-FA', 'ARIMA') and
          tuple(PL.PLANS['C']['methods']) == P.RUN_C_METHODS)
    check('bound check clamp: WSPI-H 4, WSPI-M none',
          P.control_clamp('WSPI-H') == CLAMP and P.control_clamp('WSPI-M') is None)
    check('control_bounds: WSPI-H only',
          P.control_bounds('WSPI-H', 24) == level_anchor_bounds(16, 6, CLAMP) and
          P.control_bounds('WSPI-S', 24) is None and P.control_bounds('WSPI-M', 1) is None)
    for h in HORIZONS:
        methods = P.build_prediction_methods(horizon=h, names=list(P.RUN_D_METHODS), timed=True)
        methods, rec = RP.add_control_bound_recorders(methods, {}, h)
        check(f'bound recorder only for WSPI-H  h={h:<2}', sorted(rec) == ['WSPI-H'])
        r = rec['WSPI-H']
        lo, hi = level_anchor_bounds(16, n_forecast_coefficients(h), CLAMP)
        check(f'bound recorder uses the WSPI-H limits  h={h:<2}', (r.lower, r.upper) == (lo, hi))
        methods['WSPI-H'].scorer(X)
        st = r.bound_stats()
        check(f'bound recorder OK on the test rows  h={h:<2}', st['status'] == 'OK' and
              st['n_ratios'] > 0 and lo * (1 - 1e-9) <= st['min_ratio'] and
              st['max_ratio'] <= hi * (1 + 1e-9),
              f"([{st['min_ratio']:.4f}, {st['max_ratio']:.4f}] in [{lo:.4f}, {hi:.4f}])")


def main():
    warnings.filterwarnings('ignore')
    X = signals()
    test_closed_form(X)
    test_bound(np.random.default_rng(19))
    test_batch(X)
    test_unchanged(X)
    test_names(X)
    print()
    if FAILED:
        print(f'FAILED: {len(FAILED)} check(s)')
        for f in FAILED:
            print('  -', f)
        sys.exit(1)
    print('ALL OK')


if __name__ == '__main__':
    main()
