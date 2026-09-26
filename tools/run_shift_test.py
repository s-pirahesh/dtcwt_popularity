r"""
Controlled shift-invariance test (revision-srep-v5, task T3.6 / E5)
===================================================================
Answers R4.7 (a controlled temporal-shift experiment comparing DTCWT with DWT and
a conventional smoothing method) and supports R3.3.  No existing module is
changed; the scorers and transforms are imported from the pipeline.

Why two parts (decided with Sajjad, 26 Sep 2026, chat 13).  When the real window
slides by one slot, two things change at once: the content (a new slot enters,
the oldest leaves) and the position of the data on the dyadic grid of the
transform.  A comparison of the score at t and t+s mixes both.  So:

  Part A - real data, circular shift (content fixed)            --data ... --out
      Every 64-slot window of every eligible item (protocol V5 eligibility:
      32 observed rows, causal catalogue with --causal-universe; windows >= 64,
      so no series is padded) is shifted circularly by s = 0..7 slots.  Both
      transforms use a circular (periodic) boundary, so only the position of the
      data on the grid changes:
        DTCWT  near_sym_a / qshift_a, J = 3, periodic extension
               (tools/run_padding_v5.dtcwt_forward_ext, mode 'periodic')
        DWT    db4, J = 3, pywt mode 'periodization'
      Per item-window, over s = 0..7:
        cv_E_L, cv_E_1, cv_E_2, cv_E_3   coefficient of variation (SD / mean,
                                         population SD) of the energy of the
                                         lowpass band and of detail levels 1..3
        sd_R, sd_WE                      SD of R (low-frequency energy ratio) and
                                         WE (normalised wavelet entropy)
      A shift by 2^J = 8 slots gives the s = 0 energies back exactly for both
      transforms (checked in every window; metadata period_max_rel_dev).  A 64-slot SMA is exactly
      invariant under a circular shift and is not listed.
      Per window: the mean over items of each metric (protocol/<T>_protocol.csv,
      T = DTCWT or DWT) and the number of items for which DTCWT varies more than DWT
      (n_dtcwt_higher_<metric>: DTCWT above DWT by more than 1e-12).

  Part B - synthetic moving event, the settings of the index    --synthetic --out
      Background: Poisson counts with a linear trend, lambda_t = 5 + 5 t / 63,
      t = 0..63 (newest = 63), one realisation per repetition (500, seed 42).
      Event: amplitude A = 10 x mean(lambda) = 75 added at start slot p0 + s,
      s = 0..8 (the event moves towards the newest slot), on the SAME background.
        shapes   spike  [1]                 (one slot)
                 burst3 [0.5, 1, 0.5]       (three slots)
        ages     middle p0 = 28  (spike age 36 -> 28 slots)
                 recent p0 = 48  (spike age 16 -> 8 slots)
      Methods (the pipeline scorers, N = 64, J = 3, library boundary handling):
        WSPI      protocol_v5.make_v5_wspi              DTCWT, symmetric
        DWT-WSPI  run_ablation_v5 DWT features, full    db4, symmetric (T3.3)
        DTCWT+AF  protocol_v5.make_v5_dtcwt_af
        DWT+AF    protocol_v5.make_v5_dwt
        SMA       sweep_methods.batch_sma               (N = 64)
        EWMA-eq   sweep_methods.make_batch_ewma_eq(64)  (alpha = 2 / 65)
      Per repetition x shape x age x method, over the 8 steps s -> s+1:
        share_wrong      share of steps in which the score goes DOWN although
                         the same event moved one slot closer to the present
                         (d log score < -1e-9)
        mean_wrong_dlog  mean size of those downward steps, mean(max(0, -d log score))
                         over the 8 steps (0 if no step goes down)
        mean_abs_dlog    mean |d log score|;  mean_dlog  mean d log score
      Per repetition x shape x age x transform (DTCWT / DWT), over s = 0..7:
        cv_E_L .. cv_E_3, sd_R, sd_WE (same definitions as part A; here the
        event also moves over a fixed background, so this is not a pure test)
      Figure example: repetition 0 (fixed rule).

Layout
  Part A  <out>/protocol/{DTCWT,DWT}_protocol.csv   one row per window >= 64
          <out>/metadata/shift_run.json
  Part B  <out>/synthetic_scores.csv     repetition x shape x age x shift x method
          <out>/synthetic_features.csv   repetition x shape x age x shift x transform
          <out>/synthetic_summary.csv    shape x age x method / transform: mean,
                                         SD and 95 % percentile bootstrap CI over
                                         repetitions (B 10000, seed 42)
          <out>/synthetic_tests.csv      WSPI against every other method, paired
                                         by repetition: Wilcoxon signed-rank,
                                         Holm per shape x age x metric
          <out>/metadata/synthetic_run.json
  --collect <root>
          <root>/shift_summary.csv   scenario x transform x metric: mean and SD
                                     over the windows, DWT / DTCWT ratio (empty when
                                     the DTCWT mean is 0, dtcwt_zero), share of
                                     item-windows in which DTCWT varies MORE than DWT
          <root>/shift_control.csv   (1) per-window item counts equal to the T1.5
                                     causal WSPI run (windows >= 64); (2) the
                                     s = 8 period check
  --stats <root> --stats-out <folder>
          tools/stats_report.analyse (imported, unchanged), reference DTCWT,
          lower is better for every metric, T1.6 blocks (YouTube 24; taxi one
          week + one day as sensitivity), one Holm family per scenario x metric.
          <folder>/<scenario>/{method_summary,paired_tests}.csv and
          <folder>/all_{method_summary,paired_tests}.csv

Examples (from the project root, Windows):

  python tools\run_shift_test.py --data data\datasets\youtube_hourly.csv --min-obs 50 ^
         --causal-universe --out results\revision_v5\T3.6_shift_invariance\youtube_hourly
  python tools\run_shift_test.py --synthetic --out results\revision_v5\T3.6_shift_invariance\synthetic
  python tools\run_shift_test.py --collect results\revision_v5\T3.6_shift_invariance
  python tools\run_shift_test.py --stats results\revision_v5\T3.6_shift_invariance ^
         --stats-out results\revision_v5\T1.6_stats\T3.6_shift_invariance

Full command list: Revisions/V4/Response/Runbooks/RUN_T3.6.md
Unit test: tools/test_shift_test.py
"""
import argparse
import json
import platform
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))

import dtcwt  # noqa: E402
import pywt  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

from evaluation.fast_evaluator import _dtcwt_forward_rows  # noqa: E402
from evaluation.protocol_v5 import (ProtocolV5Evaluator, make_v5_dtcwt_af,  # noqa: E402
                                    make_v5_dwt, make_v5_wspi)
from evaluation.sweep_methods import batch_sma, make_batch_ewma_eq  # noqa: E402
from run_ablation_v5 import DWTFeatureCache, dwt_coeffs  # noqa: E402
from run_ablation_v5 import make_scorer as ablation_scorer  # noqa: E402
from run_padding_v5 import dtcwt_forward_ext  # noqa: E402

WINDOW, LEVEL, SHIFTS = 64, 3, 8           # s = 0..7 (part A); 2^J = 8
MIN_OBS = 32                               # wavelet_min_obs(64)
FIRST_WINDOW = WINDOW                      # full 64-slot windows only
BIORT, QSHIFT = 'near_sym_a', 'qshift_a'
DWT_WAVELET = 'db4'
METRICS = ['cv_E_L', 'cv_E_1', 'cv_E_2', 'cv_E_3', 'sd_R', 'sd_WE']
TRANSFORMS = ('DTCWT', 'DWT')
REF_T15 = Path('results/revision_v5/T1.5_causal_universe/T1.4_protocol_v5')
DATASETS = {  # scenario folder -> (block, block_sens), T1.6 (tracker section E)
    'youtube_hourly': (24, None),
    'taxi_hourly': (168, 24),
    'taxi_30min': (336, 48),
    'taxi_5min': (2016, 288),
}
# part B
N_REP, SEED = 500, 42
LAM0, LAM1, AMP_X = 5.0, 10.0, 10.0
SHAPES = {'spike': [1.0], 'burst3': [0.5, 1.0, 0.5]}
AGES = {'middle': 28, 'recent': 48}
SYN_SHIFTS = 9                             # s = 0..8 -> 8 steps
WRONG_TOL = 1e-9
TIE_TOL = 1e-12                            # below this a CV / SD counts as 0
SYN_METHODS = ('WSPI', 'DWT-WSPI', 'DTCWT+AF', 'DWT+AF', 'SMA', 'EWMA-eq')


def _abs(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


# =============================================================================
# Band energies and features
# =============================================================================
def energies(low, highs) -> np.ndarray:
    """(n, J+1): energy of the lowpass band, then detail levels 1..J."""
    e_low = (np.abs(low) ** 2).sum(axis=1)
    e_h = [(np.abs(h) ** 2).sum(axis=1) for h in highs]
    return np.column_stack([e_low] + e_h)


def r_we(E: np.ndarray):
    """R and WE from band energies (last axis = bands), WSPI formulas."""
    tot = E.sum(axis=-1)
    safe = np.where(tot > 0, tot, 1.0)
    R = np.where(tot > 0, E[..., 0] / safe, 0.0)
    with np.errstate(divide='ignore', invalid='ignore'):
        P = E / safe[..., None]
        term = np.where(P > 0, P * np.log2(np.where(P > 0, P, 1.0)), 0.0)
    WE = np.where(tot > 0, -term.sum(axis=-1) / np.log2(E.shape[-1]), 0.0)
    return R, WE


def circular_energies(X: np.ndarray, transform: str, shifts=range(SHIFTS)) -> np.ndarray:
    """X (n, 64) -> band energies (S, n, J+1) of the circularly shifted rows."""
    shifts = list(shifts)
    n = X.shape[0]
    flat = np.concatenate([np.roll(X, s, axis=1) for s in shifts], axis=0)
    if transform == 'DTCWT':
        lo, hs = dtcwt_forward_ext(np.ascontiguousarray(flat.T), LEVEL, 'periodic')
        E = energies(lo.T, [h.T for h in hs])
    elif transform == 'DWT':
        c = pywt.wavedec(flat, DWT_WAVELET, level=LEVEL, mode='periodization', axis=-1)
        E = energies(c[0], list(reversed(c[1:])))
    else:
        raise ValueError(transform)
    return E.reshape(len(shifts), n, LEVEL + 1)


def shift_metrics(E: np.ndarray) -> np.ndarray:
    """E (S, n, J+1) -> (n, 6): cv of every band energy, SD of R and WE over S."""
    m = E.mean(axis=0)
    sd = E.std(axis=0)
    cv = np.where(m > 0, sd / np.where(m > 0, m, 1.0), 0.0)
    R, WE = r_we(E)
    return np.column_stack([cv, R.std(axis=0), WE.std(axis=0)])


# =============================================================================
# Part A (one scenario)
# =============================================================================
def run_real(a):
    ev = ProtocolV5Evaluator.from_csv(_abs(a.data), dataset_min_obs=a.min_obs, seed=42,
                                      robustness=False, causal_universe=a.causal_universe)
    ev.verbose = False
    out = _abs(a.out)
    recs = {t: [] for t in TRANSFORMS}
    zero_mean = {t: 0 for t in TRANSFORMS}
    period_dev = {t: 0.0 for t in TRANSFORMS}
    n_itemwin = 0
    t0 = time.time()
    cum = np.zeros(len(ev.items)) if ev.causal_universe else None
    for k in range(ev.num_slots + 1):                # same loop as run_method
        if cum is not None and k > 0:
            cum += ev.V[:, k - 1]
        lo, hi = ev.window_bounds(k, WINDOW)
        if ev.any_cum[hi] - ev.any_cum[lo] == 0 or not ev.any_row[k]:
            continue
        observed = ev.C[:, hi] - ev.C[:, lo]
        ok = observed >= MIN_OBS
        if cum is not None:
            ok &= cum >= ev.universe_min_count
        idx = np.where(ok)[0]
        if len(idx) < 2 or k < a.first_window:
            continue
        X = ev.V[idx, lo:hi]
        if X.shape[1] != WINDOW:
            raise RuntimeError(f'window {k}: {X.shape[1]} slots')
        n_itemwin += len(idx)
        vals = {}
        for t in TRANSFORMS:
            E = circular_energies(X, t, range(SHIFTS + 1))          # s = 0..8
            E0, E8 = E[0], E[SHIFTS]
            dev = np.abs(E8 - E0) / np.maximum(np.abs(E0), 1e-300)
            period_dev[t] = max(period_dev[t], float(np.max(dev[E0 > 1e-9])) if (E0 > 1e-9).any() else 0.0)
            M = shift_metrics(E[:SHIFTS])
            zero_mean[t] += int((E[:SHIFTS].mean(axis=0) <= 0).sum())
            vals[t] = M
        base = {'window_id': k,
                'timestamp': int((ev.start + k * ev.slot).timestamp() * 1000),
                'num_items': len(idx)}
        for t in TRANSFORMS:
            r = dict(base, method=t)
            for j, c in enumerate(METRICS):
                r[c] = float(vals[t][:, j].mean())
            for j, c in enumerate(METRICS):
                r[f'n_dtcwt_higher_{c}'] = int((vals['DTCWT'][:, j] > vals['DWT'][:, j] + TIE_TOL).sum())
            recs[t].append(r)
    pdir = out / 'protocol'
    pdir.mkdir(parents=True, exist_ok=True)
    cols = ['window_id', 'timestamp', 'method', 'num_items'] + METRICS + \
           [f'n_dtcwt_higher_{c}' for c in METRICS]
    for t in TRANSFORMS:
        pd.DataFrame(recs[t], columns=cols).to_csv(pdir / f'{t}_protocol.csv', index=False,
                                                   encoding='utf-8')
    meta = dict(task='T3.6 shift-invariance test (E5), part A: real data, circular shift',
                window=WINDOW, level=LEVEL, shifts=list(range(SHIFTS)),
                period_check_shift=SHIFTS, min_obs=MIN_OBS, first_window=a.first_window,
                dtcwt=dict(biort=BIORT, qshift=QSHIFT, extension='periodic',
                           code='tools/run_padding_v5.dtcwt_forward_ext'),
                dwt=dict(wavelet=DWT_WAVELET, mode='periodization'),
                metrics=METRICS, cv='population SD / mean over s = 0..7; 0 if the mean is 0',
                n_windows=len(recs['DTCWT']), n_item_windows=n_itemwin,
                zero_mean_band_cases=zero_mean,
                period_max_rel_dev=period_dev,
                causal_universe=a.causal_universe,
                universe_rule=(('total count before the test slot >= %d' if a.causal_universe
                                else 'total count over the whole file >= %d') % a.min_obs),
                slot_minutes=ev.slot.total_seconds() / 60, num_items=len(ev.items),
                data=str(a.data), dataset_min_obs=a.min_obs,
                runtime_seconds=round(time.time() - t0, 1),
                versions=_versions(), host=platform.platform(),
                created=time.strftime('%Y-%m-%d %H:%M:%S'))
    (out / 'metadata').mkdir(exist_ok=True)
    (out / 'metadata' / 'shift_run.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    d, w = pd.DataFrame(recs['DTCWT']), pd.DataFrame(recs['DWT'])
    print(f'{out.name}: {len(d)} windows, {n_itemwin} item-windows, {meta["runtime_seconds"]} s')
    for c in METRICS:
        print(f'  {c:<7} DTCWT {d[c].mean():.4f}   DWT {w[c].mean():.4f}')
    print(f'  period check (s=8 vs s=0, max rel. dev.): {period_dev}')


def _versions():
    import scipy
    return dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                dtcwt=dtcwt.__version__, pywt=pywt.__version__, scipy=scipy.__version__)


# =============================================================================
# Part B (synthetic)
# =============================================================================
def syn_scorers():
    return {'WSPI': make_v5_wspi(WINDOW, LEVEL),
            'DWT-WSPI': ablation_scorer(DWTFeatureCache(use_cache=False), 'full', 'DWT-WSPI'),
            'DTCWT+AF': make_v5_dtcwt_af(WINDOW, LEVEL),
            'DWT+AF': make_v5_dwt(WINDOW, LEVEL),
            'SMA': batch_sma,
            'EWMA-eq': make_batch_ewma_eq(WINDOW)}


def syn_background(rng) -> np.ndarray:
    lam = LAM0 + (LAM1 - LAM0) * np.arange(WINDOW) / (WINDOW - 1)
    return rng.poisson(lam).astype(np.float64)


def syn_matrix(bg: np.ndarray, shape, p0: int) -> np.ndarray:
    """(SYN_SHIFTS, 64): the background with the event at start slot p0 + s."""
    amp = AMP_X * (LAM0 + LAM1) / 2.0
    X = np.tile(bg, (SYN_SHIFTS, 1))
    sh = np.asarray(shape, dtype=np.float64) * amp
    for s in range(SYN_SHIFTS):
        X[s, p0 + s:p0 + s + len(sh)] += sh
    return X


def step_stats(scores: np.ndarray) -> dict:
    d = np.diff(np.log(np.maximum(scores, 1e-300)))
    return dict(share_wrong=float((d < -WRONG_TOL).mean()),
                mean_wrong_dlog=float(np.where(d < -WRONG_TOL, -d, 0.0).mean()),
                mean_abs_dlog=float(np.abs(d).mean()), mean_dlog=float(d.mean()))


def syn_features(X: np.ndarray, transform: str) -> pd.DataFrame:
    """Band energies, R, WE and mu_L with the index's own boundary handling."""
    if transform == 'DTCWT':
        low, highs = _dtcwt_forward_rows(dtcwt.Transform1d(biort=BIORT, qshift=QSHIFT), X, LEVEL)
    else:
        low, highs = dwt_coeffs(X, LEVEL, WINDOW)
    E = energies(low, highs)
    R, WE = r_we(E)
    lm = np.abs(low)
    w = 2.0 ** -np.arange(lm.shape[1])[::-1]
    mu = (lm * w).sum(axis=1) / w.sum()
    return pd.DataFrame({'E_L': E[:, 0], 'E_1': E[:, 1], 'E_2': E[:, 2], 'E_3': E[:, 3],
                         'R': R, 'WE': WE, 'mu_L': mu})


def boot_ci(x: np.ndarray, B: int, rng) -> tuple:
    idx = rng.randint(0, len(x), size=(B, len(x)))
    m = x[idx].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def holm(p) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    out = np.full(len(p), np.nan)
    ok = np.isfinite(p)
    q = p[ok]
    order = np.argsort(q)
    adj = np.empty(len(q))
    run = 0.0
    for i, j in enumerate(order):
        run = max(run, (len(q) - i) * q[j])
        adj[j] = min(1.0, run)
    out[ok] = adj
    return out


def run_synthetic(a):
    out = _abs(a.out)
    rng = np.random.RandomState(a.seed)
    sc = syn_scorers()
    t0 = time.time()
    srows, frows, step_rows, feat_rows = [], [], [], []
    for rep in range(a.reps):
        bg = syn_background(rng)
        for shape, sh in SHAPES.items():
            for age, p0 in AGES.items():
                X = syn_matrix(bg, sh, p0)
                key = dict(repetition=rep, shape=shape, age=age)
                for m in SYN_METHODS:
                    s = np.asarray(sc[m](X), dtype=np.float64)
                    for j in range(SYN_SHIFTS):
                        srows.append(dict(key, shift=j, method=m, score=s[j]))
                    step_rows.append(dict(key, method=m, **step_stats(s)))
                for t in TRANSFORMS:
                    F = syn_features(X, t)
                    for j in range(SYN_SHIFTS):
                        frows.append(dict(key, shift=j, transform=t, **F.iloc[j].to_dict()))
                    E = F[['E_L', 'E_1', 'E_2', 'E_3']].to_numpy()[:SHIFTS][:, None, :]
                    M = shift_metrics(E)[0]
                    feat_rows.append(dict(key, transform=t, **dict(zip(METRICS, M))))
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(srows).to_csv(out / 'synthetic_scores.csv', index=False, encoding='utf-8')
    pd.DataFrame(frows).to_csv(out / 'synthetic_features.csv', index=False, encoding='utf-8')
    steps, feats = pd.DataFrame(step_rows), pd.DataFrame(feat_rows)
    brng = np.random.RandomState(a.seed)
    rows = []
    for (shape, age, m), g in steps.groupby(['shape', 'age', 'method'], sort=False):
        for c in ('share_wrong', 'mean_wrong_dlog', 'mean_abs_dlog', 'mean_dlog'):
            x = g[c].to_numpy()
            lo, hi = boot_ci(x, a.B, brng)
            rows.append(dict(shape=shape, age=age, kind='score', name=m, metric=c,
                             n=len(x), mean=x.mean(), sd=x.std(ddof=1), ci_low=lo, ci_high=hi))
    for (shape, age, t), g in feats.groupby(['shape', 'age', 'transform'], sort=False):
        for c in METRICS:
            x = g[c].to_numpy()
            lo, hi = boot_ci(x, a.B, brng)
            rows.append(dict(shape=shape, age=age, kind='transform', name=t, metric=c,
                             n=len(x), mean=x.mean(), sd=x.std(ddof=1), ci_low=lo, ci_high=hi))
    pd.DataFrame(rows).to_csv(out / 'synthetic_summary.csv', index=False, encoding='utf-8')
    trows = []
    for (shape, age), g in steps.groupby(['shape', 'age'], sort=False):
        ref = g[g['method'] == 'WSPI'].set_index('repetition')
        for c in ('share_wrong', 'mean_wrong_dlog', 'mean_abs_dlog'):
            for m in SYN_METHODS[1:]:
                o = g[g['method'] == m].set_index('repetition').reindex(ref.index)
                d = ref[c].to_numpy() - o[c].to_numpy()
                p = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
                trows.append(dict(shape=shape, age=age, metric=c, reference='WSPI', method=m,
                                  n=len(d), mean_reference=ref[c].mean(), mean_method=o[c].mean(),
                                  diff_ref_minus_method=d.mean(), wins=int((d < 0).sum()),
                                  losses=int((d > 0).sum()), ties=int((d == 0).sum()), p=p))
        for c in METRICS:
            sel = (feats['shape'] == shape) & (feats['age'] == age)
            ref = feats[sel & (feats['transform'] == 'DTCWT')].set_index('repetition')
            o = feats[sel & (feats['transform'] == 'DWT')].set_index('repetition').reindex(ref.index)
            d = ref[c].to_numpy() - o[c].to_numpy()
            p = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
            trows.append(dict(shape=shape, age=age, metric=c, reference='DTCWT', method='DWT',
                              n=len(d), mean_reference=ref[c].mean(), mean_method=o[c].mean(),
                              diff_ref_minus_method=d.mean(), wins=int((d < 0).sum()),
                              losses=int((d > 0).sum()), ties=int((d == 0).sum()), p=p))
    tests = pd.DataFrame(trows)
    tests['p_holm'] = np.nan
    for _, g in tests.groupby(['shape', 'age', 'metric']):
        tests.loc[g.index, 'p_holm'] = holm(g['p'].to_numpy())
    tests['verdict'] = np.where(tests.p_holm >= 0.05, 'n.s.',
                                np.where(tests.diff_ref_minus_method < 0, 'ref_lower', 'ref_higher'))
    tests.to_csv(out / 'synthetic_tests.csv', index=False, encoding='utf-8')
    meta = dict(task='T3.6 shift-invariance test (E5), part B: synthetic moving event',
                window=WINDOW, level=LEVEL, repetitions=a.reps, seed=a.seed,
                background=f'Poisson, lambda_t = {LAM0} + {LAM1 - LAM0} t / 63, t = 0..63',
                amplitude=AMP_X * (LAM0 + LAM1) / 2.0, shapes=SHAPES,
                ages={k: dict(start_slot=v, spike_age_first=WINDOW - v,
                              spike_age_last=WINDOW - v - (SYN_SHIFTS - 1)) for k, v in AGES.items()},
                shifts=list(range(SYN_SHIFTS)), wrong_step=f'd log score < -{WRONG_TOL}',
                methods={m: getattr(sc[m], '__name__', m) for m in SYN_METHODS},
                transform_features=dict(DTCWT='library Transform1d, symmetric (as WSPI)',
                                        DWT='run_ablation_v5.dwt_coeffs: db4, symmetric'),
                cv_shifts=list(range(SHIFTS)), bootstrap=dict(B=a.B, seed=a.seed, kind='percentile'),
                tests='Wilcoxon signed-rank paired by repetition, Holm per shape x age x metric',
                figure_example='repetition 0',
                runtime_seconds=round(time.time() - t0, 1), versions=_versions(),
                host=platform.platform(), created=time.strftime('%Y-%m-%d %H:%M:%S'))
    (out / 'metadata').mkdir(exist_ok=True)
    (out / 'metadata' / 'synthetic_run.json').write_text(json.dumps(meta, indent=2),
                                                         encoding='utf-8')
    s = pd.DataFrame(rows)
    print(s[s['metric'].isin(['share_wrong', 'mean_wrong_dlog', 'cv_E_3', 'sd_WE'])]
          .pivot_table(index=['shape', 'age', 'name'], columns='metric', values='mean').round(4)
          .to_string())
    print(f'Saved to {out}  ({meta["runtime_seconds"]} s)')


# =============================================================================
# Collect and statistics
# =============================================================================
def scenario_dirs(root: Path):
    return [p for p in sorted(root.iterdir())
            if p.is_dir() and (p / 'protocol' / 'DTCWT_protocol.csv').exists()]


def collect(root: Path, ref_t15: Path = None):
    ref_t15 = ref_t15 or _abs(REF_T15)
    rows, ctrl = [], []
    for sd in scenario_dirs(root):
        res = {t: pd.read_csv(sd / 'protocol' / f'{t}_protocol.csv') for t in TRANSFORMS}
        n_iw = int(res['DTCWT']['num_items'].sum())
        for c in METRICS:
            m = {t: res[t][c].mean() for t in TRANSFORMS}
            for t in TRANSFORMS:
                rows.append(dict(scenario=sd.name, transform=t, metric=c,
                                 n_windows=len(res[t]), n_item_windows=n_iw,
                                 mean=m[t], sd=res[t][c].std(),
                                 ratio_dwt_over_dtcwt=(m['DWT'] / m['DTCWT'] if m['DTCWT'] > TIE_TOL
                                                       else np.nan),
                                 dtcwt_zero=bool(m['DTCWT'] <= TIE_TOL),
                                 share_item_windows_dtcwt_higher=(
                                     res['DTCWT'][f'n_dtcwt_higher_{c}'].sum() / n_iw)))
        # control 1: item counts against the T1.5 causal WSPI run
        ref = ref_t15 / sd.name / 'protocol' / 'WSPI_protocol.csv'
        c1 = dict(scenario=sd.name, check='num_items vs T1.5 WSPI (windows >= 64)',
                  reference_file=str(ref))
        if ref.exists():
            r = pd.read_csv(ref, usecols=['window_id', 'num_items'])
            r = r[r.window_id >= FIRST_WINDOW].set_index('window_id')['num_items']
            d = res['DTCWT'].set_index('window_id')['num_items']
            same = r.index.equals(d.index)
            c1.update(n_windows=len(d), n_windows_ref=len(r), same_window_ids=bool(same),
                      max_abs_diff=float((d - r).abs().max()) if same else np.nan,
                      status='equal' if same and (d == r).all() else 'DIFFERENT')
        else:
            c1['status'] = 'missing'
        ctrl.append(c1)
        mf = sd / 'metadata' / 'shift_run.json'
        pdv = json.loads(mf.read_text(encoding='utf-8'))['period_max_rel_dev'] if mf.exists() else {}
        worst = max(pdv.values()) if pdv else np.nan
        ctrl.append(dict(scenario=sd.name, check='period: s=8 equals s=0 (max rel. dev.)',
                         max_abs_diff=worst,
                         status='equal' if pdv and worst < 1e-9 else ('missing' if not pdv
                                                                      else 'DIFFERENT')))
        print(f'{sd.name}: control {[c["status"] for c in ctrl if c["scenario"] == sd.name]}')
    pd.DataFrame(rows).to_csv(root / 'shift_summary.csv', index=False, encoding='utf-8')
    pd.DataFrame(ctrl).to_csv(root / 'shift_control.csv', index=False, encoding='utf-8')
    print(f'Saved {root / "shift_summary.csv"} and {root / "shift_control.csv"}')


def stats(root: Path, out_root: Path, B: int):
    import stats_report as sr
    sr.LOWER_IS_BETTER = set(sr.LOWER_IS_BETTER) | set(METRICS)
    for sd in scenario_dirs(root):
        if sd.name not in DATASETS:
            print(f'[stats] {sd.name}: no block length, skipped')
            continue
        block, block_sens = DATASETS[sd.name]
        tmp = Path(tempfile.mkdtemp(prefix='t36_'))
        try:
            (tmp / 'protocol').mkdir()
            for t in TRANSFORMS:
                d = pd.read_csv(sd / 'protocol' / f'{t}_protocol.csv')
                d[['window_id'] + METRICS].to_csv(tmp / 'protocol' / f'{t}_protocol.csv',
                                                  index=False)
            sr.analyse(tmp, out_root / sd.name, block, block_sens, 'DTCWT', B, 42, METRICS)
            print(f'[stats] {sd.name}')
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    sr.collect(out_root)                         # all_method_summary.csv, all_paired_tests.csv


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--data')
    ap.add_argument('--min-obs', type=int, help='dataset min_observations (catalogue filter)')
    ap.add_argument('--out', help='scenario folder (part A) or synthetic folder (part B)')
    ap.add_argument('--first-window', type=int, default=FIRST_WINDOW)
    ap.add_argument('--causal-universe', action='store_true')
    ap.add_argument('--synthetic', action='store_true', help='part B')
    ap.add_argument('--reps', type=int, default=N_REP)
    ap.add_argument('--seed', type=int, default=SEED)
    ap.add_argument('--B', type=int, default=10000)
    ap.add_argument('--collect', help='T3.6 root: write shift_summary.csv and shift_control.csv')
    ap.add_argument('--stats', help='T3.6 root: block statistics DTCWT vs DWT (part A)')
    ap.add_argument('--stats-out', help='output root of --stats (always give it explicitly)')
    a = ap.parse_args()
    if a.collect:
        collect(_abs(a.collect))
    elif a.stats:
        if not a.stats_out:
            ap.error('--stats-out is required with --stats')
        stats(_abs(a.stats), _abs(a.stats_out), a.B)
    elif a.synthetic:
        if not a.out:
            ap.error('--out is required')
        run_synthetic(a)
    else:
        if not (a.data and a.min_obs is not None and a.out):
            ap.error('--data, --min-obs and --out are required')
        if a.first_window < WINDOW:
            ap.error('--first-window must be >= 64 (full windows only)')
        run_real(a)


if __name__ == '__main__':
    main()
