r"""
Unit checks for the supplementary methods of run C
==================================================
Covers the additions to ``methods/wspi_forecast.py`` and
``tools/prediction_protocol.py`` (WSPI-S, WSPI-FS, SMA, EWMA and the
sensitivity variants of WSPI-F2) and the plans of ``tools/run_prediction_plan.py``.

  1. Reference class: phi_d = 0 gives exactly WSPI for WSPI-S and WSPI-FS;
     the WSPI-FS gate is never larger than the WSPI-F2 gate and equals it for a
     very large kappa; the skill factor rules (zero sums, no backtest);
     the level forecaster; the two-sided bound for WSPI-S and WSPI-FS
     (random search, m = 1..6).
  2. Batch scorers == reference class on full 64-slot windows, h = 1, 8, 24:
     WSPI-S, WSPI-FS for every kappa, the 18 sensitivity variants; batch
     backtest sums == reference sums.
  3. Identities that the run checks rely on: the sensitivity variant
     c = 4, gamma = 0.3, phi_d = 1 gives the same floats as WSPI-F2, SMA64 the
     same floats as MA64; the methods of run A are unchanged by the additions
     (scores equal a frozen copy of the run A scorer recipe).
  4. SMA<N> == MovingAverageForecast(window=N); EWMA<N> == an independent loop
     with alpha = 2 / (N + 1).
  5. Names: method lists, sensitivity names and their parsing, the clamp used by
     the bound check, the plan lists of run_prediction_plan.py and the run C
     method lists of run_prediction_stats.py; a learned method needs a trained
     scorer.

    python tools/test_supplementary_methods.py
"""
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from methods.wspi_forecast import (CLAMP, SKILL_KAPPAS, WSPIForecast,     # noqa: E402
                                   backtest_sums, get_control_methods,
                                   level_forecast, n_forecast_coefficients,
                                   aryw_forecast, ratio_bounds, recency_mean,
                                   skill_factor)
from methods.wspi_forecast import MovingAverageForecast                   # noqa: E402
from methods.wspi_assessment import WSPIAssessment                        # noqa: E402
from tools import prediction_protocol as P                                # noqa: E402
from tools import run_prediction_plan as PL                               # noqa: E402
from tools import run_prediction_stats as S                               # noqa: E402
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


# =============================================================================
# 1. Reference class
# =============================================================================

def test_reference_class(X):
    wspi = WSPIAssessment()
    ref = np.array([wspi.assess_single(x) for x in X[:60]])
    for name in ('WSPI-S', 'WSPI-FS2'):
        for h in HORIZONS:
            kw = dict(predictor='level', gate='none') if name == 'WSPI-S' else \
                dict(predictor='aryw', gate='skill', skill_kappa=2.0)
            obj = WSPIForecast(horizon=h, clamp=CLAMP, phi_d=0.0, **kw)
            got = np.array([obj.assess_single(x) for x in X[:60]])
            check(f'{name} phi_d = 0 is exactly WSPI  h={h:<2}', np.array_equal(got, ref))

    for h in HORIZONS:
        f2 = WSPIForecast(predictor='aryw', gate='structural', horizon=h)
        g2 = np.array([f2.gate_value(x) for x in X])
        worst_up, nless = 0.0, 0
        for kappa in SKILL_KAPPAS:
            fs = WSPIForecast(predictor='aryw', gate='skill', horizon=h, skill_kappa=kappa)
            gs = np.array([fs.gate_value(x) for x in X])
            worst_up = max(worst_up, float((gs - g2).max()))
            nless += int((gs < g2).sum())
        check(f'FS gate <= F2 gate (all kappa)  h={h:<2}', worst_up <= 0.0,
              f'(lower in {nless} row-kappa pairs)')
        check(f'FS gate < F2 gate somewhere     h={h:<2}', nless > 0)
        big = WSPIForecast(predictor='aryw', gate='skill', horizon=h, skill_kappa=1e300)
        gb = np.array([big.gate_value(x) for x in X])
        check(f'FS gate with kappa 1e300 == F2   h={h:<2}', np.array_equal(gb, g2))

    check('skill factor: S_F = 0 -> 1', skill_factor(0.0, 3.0, 1.0, 2.0) == 1.0)
    check('skill factor: S_F at rounding level -> 1', skill_factor(1e-12, 3.0, 1.0, 2.0) == 1.0)
    check('skill factor: S_mu zero, S_F > 0 -> 0', skill_factor(1.0, 1e-12, 1.0, 2.0) == 0.0)
    check('skill factor: kappa S_mu / S_F capped at 1',
          skill_factor(1.0, 1.0, 1.0, 2.0) == 1.0 and skill_factor(4.0, 1.0, 1.0, 2.0) == 0.5)
    check('skill factor: no backtest (scale 0, sums 0) -> 1', skill_factor(0.0, 0.0, 0.0, 2.0) == 1.0)
    s_f, s_mu, k_eff = backtest_sums(np.arange(1.0, 8.0), 6, aryw_forecast, 2, CLAMP)
    check('backtest sums: window too short -> k_eff 0', k_eff == 0 and s_f == 0 and s_mu == 0)

    check('level forecast: mean of the last two',
          np.array_equal(level_forecast(np.array([1.0, 2.0, 4.0, 8.0]), 3), np.full(3, 6.0)))
    check('level forecast: one value', np.array_equal(level_forecast(np.array([5.0]), 2),
                                                      np.full(2, 5.0)))

    # two-sided bound (Method_Design 7-2) for the clamped controls, m = 1..6
    rng = np.random.default_rng(11)
    worst = 0.0
    for m_h in (1, 3, 5, 9, 13, 21):                       # m = ceil(h / 4) = 1..6
        m = n_forecast_coefficients(m_h)
        lo, hi = ratio_bounds(16, m, CLAMP)
        for name, obj in get_control_methods(horizon=m_h).items():
            base = WSPIForecast(predictor='aryw', gate='none', horizon=m_h, clamp=CLAMP, phi_d=0.0)
            for _ in range(40):
                x = rng.poisson(rng.gamma(1.5, 10.0), 64).astype(float)
                if rng.random() < 0.3:
                    x[-rng.integers(1, 6):] *= rng.uniform(0, 20)
                w = base.assess_single(x)
                if w <= 0:
                    continue
                r = obj.assess_single(x) / w
                worst = max(worst, lo / r - 1.0, r / hi - 1.0)
    check('WSPI-S and WSPI-FS inside [L, U], m = 1..6', worst <= 1e-9, f'(worst excess {worst:.1e})')


# =============================================================================
# 2. Batch scorers against the reference class
# =============================================================================

def test_batch_against_class(X):
    for h in HORIZONS:
        refs = get_control_methods(horizon=h)
        for c, g, p in [(c, g, p) for c in P.SENS_CLAMPS for g in P.SENS_GAMMAS
                        for p in P.SENS_PHI_D]:
            refs[P.sens_name(c, g, p)] = WSPIForecast(predictor='aryw', gate='structural',
                                                      horizon=h, clamp=c, gate_gamma=g, phi_d=p)
        worst = {}
        for name, obj in refs.items():
            fast = scorer(name, h)(X)
            slow = np.array([obj.assess_single(x) for x in X])
            rel = np.abs(fast - slow) / np.maximum(np.abs(slow), 1e-9)
            worst[name] = rel.max()
        for name in ('WSPI-S',) + P.FS_METHODS:
            check(f'scorer == class  h={h:<2} {name:<10}', worst[name] < 1e-8,
                  f'max rel. diff {worst[name]:.2e}')
        sw = max(worst[n] for n in P.SENS_METHODS)
        check(f'scorer == class  h={h:<2} 18 sensitivity variants', sw < 1e-8,
              f'max rel. diff {sw:.2e}')
        # backtest sums
        tr = P.dtcwt.Transform1d(biort='near_sym_a', qshift='qshift_a')
        F = P.CoefFeatures(P._pad_left_reflect(X, 64), tr)
        m = n_forecast_coefficients(h)
        bf, bm, k_eff = P.backtest_sums_rows(F.a, m)
        ref_s = np.array([backtest_sums(a, m, aryw_forecast)[:2] for a in F.a])
        d = max(np.abs(bf - ref_s[:, 0]).max(), np.abs(bm - ref_s[:, 1]).max())
        check(f'backtest sums rows == class   h={h:<2}', d < 1e-10 and k_eff == 2,
              f'max abs. diff {d:.1e}')


# =============================================================================
# 3. Identities
# =============================================================================

def frozen_f2(X, h):
    """The run A scorer of WSPI-F2, called as run A calls it."""
    return P.make_coef_scorer('WSPI-F2', h)(X)


def test_identities(X):
    for h in HORIZONS:
        f2 = frozen_f2(X, h)
        sv = scorer(P.sens_name(4, 0.3, 1), h)(X)
        check(f'sensitivity c4 g0.3 p1 == WSPI-F2 (same floats)  h={h:<2}',
              np.array_equal(sv, f2))
        check(f'SMA64 == MA64 (same floats)                     h={h:<2}',
              np.array_equal(scorer('SMA64', h)(X), scorer('MA64', h)(X)))
        fs_inf = P.make_control_scorer('WSPI-FS', h, kappa=1e300)(X)
        check(f'WSPI-FS with kappa 1e300 == WSPI-F2             h={h:<2}',
              np.array_equal(fs_inf, f2))
        wspi = scorer('WSPI', h)(X)
        s0 = P.make_control_scorer('WSPI-S', h, phi_d=0.0)(X)
        check(f'WSPI-S with phi_d = 0 == WSPI (batch)           h={h:<2}',
              np.array_equal(s0, wspi))
    # run A recipes are unchanged: build_prediction_methods gives the same
    # scorer as the direct factory for every run A method
    ok = True
    for h in (1, 8):
        for name in P.RUN_A_METHODS:
            a = P.build_prediction_methods(horizon=h, names=[name])[name].scorer(X)
            if name == 'WSPI':
                b = P.make_v5_wspi(window=64, level=3)(X)
            elif name in P.COEF_METHODS:
                b = P.make_coef_scorer(name, h)(X)
            else:
                b = {'Persistence': P.make_persistence, 'ARYW': P.make_aryw,
                     'MA64': P.make_ma, 'Holt': lambda hh: P.make_batch_holt(horizon=hh)}[name](h)(X)
            ok &= bool(np.array_equal(a, b))
    check('run A methods: build_prediction_methods == direct factories (h = 1, 8)', ok)


# =============================================================================
# 4. Moving averages
# =============================================================================

def test_moving_averages(X):
    for h in HORIZONS:
        for n in P.MA_LENGTHS:
            ref = np.array([MovingAverageForecast(window=n, horizon=h).assess_single(x) for x in X])
            d = np.abs(scorer(f'SMA{n}', h)(X) - ref).max()
            check(f'SMA{n:<2} == MovingAverageForecast  h={h:<2}', d < 1e-9, f'({d:.1e})')
            alpha = 2.0 / (n + 1.0)
            e = X[:, 0].copy()
            for j in range(1, X.shape[1]):
                e = alpha * X[:, j] + (1.0 - alpha) * e
            d = np.abs(scorer(f'EWMA{n}', h)(X) - e * h).max()
            check(f'EWMA{n:<2} == independent loop      h={h:<2}', d < 1e-9, f'({d:.1e})')
    short = signals(n_rows=20, W=40, seed=2)
    check('SMA64 on a 40-slot window uses all slots',
          np.allclose(scorer('SMA64', 1)(short), short.mean(axis=1)))


# =============================================================================
# 5. Names
# =============================================================================

def test_names():
    check('FS names', P.FS_METHODS == ('WSPI-FS1', 'WSPI-FS1.5', 'WSPI-FS2', 'WSPI-FS3'))
    check('18 sensitivity names, one default',
          len(P.SENS_METHODS) == 18 and len(set(P.SENS_METHODS)) == 18 and
          'WSPI-F2_c4_g0.3_p1' in P.SENS_METHODS)
    check('sensitivity names parse back',
          all(P.sens_name(*P.parse_sens_name(n)) == n for n in P.SENS_METHODS) and
          P.parse_sens_name('WSPI-F2') is None and P.parse_sens_name('WSPI-F2_cx_g1_p1') is None)
    check('clamp for the bound check',
          P.control_clamp('WSPI-S') == 4 and P.control_clamp('WSPI-FS1.5') == 4 and
          P.control_clamp('WSPI-F2_c8_g0.5_p0.8') == 8 and P.control_clamp('SMA8') is None and
          P.control_clamp('GBM') is None)
    check('plan C methods == prediction_protocol.RUN_C_METHODS',
          tuple(PL.PLANS['C']['methods']) == P.RUN_C_METHODS)
    check('plan C_sens methods == prediction_protocol.RUN_C_SENS_METHODS',
          tuple(PL.PLANS['C_sens']['methods']) == P.RUN_C_SENS_METHODS)
    check('stats run C lists == prediction_protocol lists',
          tuple(S.SUPP_RUNS['C']['methods']) == P.RUN_C_METHODS and
          tuple(S.SUPP_RUNS['C_sens']['methods']) == P.RUN_C_SENS_METHODS and
          S.FS_NAMES == P.FS_METHODS)
    check('plans A and B unchanged',
          PL.PLANS['A']['methods'] is None and PL.PLANS['A']['flags'] == ('--gate-profile', '--bound-check')
          and tuple(PL.PLANS['B']['methods']) == ('WSPI', 'WSPI-FA', 'ARIMA'))
    jobs = PL.build_jobs('C_nlms', 'results/prediction')
    check('plan C_nlms: two scenarios, horizon 1, own script',
          len(jobs) == 2 and all(j['horizon'] == 1 for j in jobs) and
          all(Path(j['cmd'][1]).name == 'nlms_convergence.py' for j in jobs))
    jobs = PL.build_jobs('C', 'results/prediction')
    check('plan C: 18 jobs with run_prediction.py', len(jobs) == 18 and
          all(Path(j['cmd'][1]).name == 'run_prediction.py' for j in jobs) and
          all('--resume' in j['cmd'] and '--bound-check' in j['cmd'] for j in jobs))
    jobs_a = PL.build_jobs('A', 'results/prediction')
    check('plan A: commands as before (run_prediction.py, gate profile, no --methods)',
          len(jobs_a) == 18 and all(Path(j['cmd'][1]).name == 'run_prediction.py' and
                                    '--methods' not in j['cmd'] for j in jobs_a))
    try:
        P.build_prediction_methods(horizon=1, names=['GBM'])
        ok = False
    except ValueError:
        ok = True
    check('GBM without a trained scorer raises', ok)
    built = P.build_prediction_methods(horizon=8, names=[n for n in P.RUN_C_METHODS
                                                         if n not in P.LEARNED_METHODS]
                                       + list(P.SENS_METHODS), timed=True)
    check('every run C and C_sens method is built (timed)',
          len(built) == len(P.RUN_C_METHODS) - 2 + 18 and
          all(fm.window_slots == 64 and fm.min_obs == 32 for fm in built.values()))


def main():
    warnings.filterwarnings('ignore')
    X = signals()
    test_reference_class(X)
    test_batch_against_class(X)
    test_identities(X)
    test_moving_averages(X)
    test_names()
    print()
    if FAILED:
        print(f'FAILED: {len(FAILED)} check(s)')
        for f in FAILED:
            print('  -', f)
        sys.exit(1)
    print('ALL OK')


if __name__ == '__main__':
    main()
