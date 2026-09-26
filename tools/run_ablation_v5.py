r"""
Ablation study and fusion comparison under protocol V5 (revision-srep-v5, task T3.3 / E6)
========================================================================================
Answers R4.9 (expanded ablation), R4.14 (controlled comparison of the fusion
function), R3.6 (statistics on the ablation) and part of R4.2 / R4.16.
No existing module is changed.

All variants use N = 64, J = 3, alpha = beta = 1 and the settings of the
T1.4 / T2.2 / T3.1 / T3.2 runs: exact 64-slot window, zero fill, reflect
padding (left) only while a series is shorter than 64, entry rule 32 observed
rows, horizon 1 slot, RSI by item id, stable tie-breaking, seed 42, robustness
test on.  Use --causal-universe (paper setting).  Windows >= 32 are written.

Two families (decided by Sajjad, 26 Sep 2026, chat 10).  Each family is its own
run folder, so tools/stats_report.py (unchanged) builds one Holm family per
scenario x metric x family, with WSPI as the reference.

  ablation  (8 variants; features from DTCWT or from DWT)
      WSPI           DTCWT  mu_L * exp(R - WE)        (full index)
      Trend          DTCWT  mu_L                      (trend only)
      Trend+R        DTCWT  mu_L * exp(R)
      Trend+WE       DTCWT  mu_L * exp(-WE)
      DWT-WSPI       DWT    mu_L * exp(R - WE)
      DWT-Trend      DWT    mu_L
      DWT-Trend+R    DWT    mu_L * exp(R)             (CSV only, not in the paper table)
      DWT-Trend+WE   DWT    mu_L * exp(-WE)           (CSV only, not in the paper table)

  fusion    (4 fusion functions of the same DTCWT features, unit weights, no tuning)
      WSPI           exponential   mu_L * exp(R - WE)
      Linear         linear        mu_L * (1 + R - WE)
      Product        product       mu_L * R * (1 - WE)
                     (same ranking as the geometric mean (mu_L R (1-WE))^(1/3))
      Additive       additive      mu_L + R - WE
  Only per-item fusions are compared: the score of an item depends on its own
  series only (Algorithm 1: items are independent).  Fusions that normalise
  across the items of a window are left out.

Features.  DTCWT: near_sym_a / qshift_a, the cached code of
tools/run_param_grid.py (same operations as fast_evaluator.make_batch_wspi).
DWT: db4, J = 3, pywt mode 'symmetric', reflect padding to 64 - the same
decomposition as DWT+AF in protocol V5 (protocol_v5.make_v5_dwt).  mu_L, R and
WE are computed from the DWT coefficients with exactly the WSPI formulas
(|L_J| weighted by 2^-(n-k); energy ratio; normalised entropy over J+1 bands).
Features are cached per window matrix (key = shape + BLAKE2b digest).

Layout
  <out>/ablation/protocol/<variant>_protocol.csv     windows >= 32
  <out>/ablation/comparison/summary_{all,common}_windows.csv
  <out>/fusion/protocol/<variant>_protocol.csv
  <out>/fusion/comparison/summary_{all,common}_windows.csv
  <out>/metadata/ablation_run.json
  --collect <root>:
  <root>/ablation_summary.csv  scenario x family x variant: mean and SD of every
                               metric on the windows common to the family
  <root>/ablation_control.csv  WSPI against the T1.5 causal WSPI run, and
                               Trend / Trend+R / Trend+WE against the T3.2 grid
                               points (0,0) / (1,0) / (0,1); window by window,
                               must be equal

Examples (from the project root, Windows):

  python tools\run_ablation_v5.py --data data\datasets\youtube_hourly.csv --min-obs 50 ^
         --causal-universe --out results\revision_v5\T3.3_ablation\youtube_hourly
  python tools\run_ablation_v5.py --collect results\revision_v5\T3.3_ablation

Full command list: Revisions/V4/Response/Runbooks/RUN_T3.3.md
Unit test: tools/test_ablation_v5.py
"""
import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import dtcwt  # noqa: E402
import pywt  # noqa: E402

from evaluation.fast_evaluator import FastMethod  # noqa: E402
from evaluation.protocol_v5 import (METRIC_COLUMNS, ProtocolV5Evaluator,  # noqa: E402
                                    _pad_left_reflect, _target_len, summarize)
from evaluation.sweep_methods import wavelet_min_obs  # noqa: E402
from tools.run_param_grid import WSPIFeatureCache  # noqa: E402

WINDOW, LEVEL = 64, 3
DWT_WAVELET, DWT_MODE = 'db4', 'symmetric'
REF_T15 = Path('results/revision_v5/T1.5_causal_universe/T1.4_protocol_v5')
REF_T32 = Path('results/revision_v5/T3.2_param_grid')

# name -> (transform, kind)
ABLATION = {
    'WSPI':         ('dtcwt', 'full'),
    'Trend':        ('dtcwt', 'trend'),
    'Trend+R':      ('dtcwt', 'R'),
    'Trend+WE':     ('dtcwt', 'WE'),
    'DWT-WSPI':     ('dwt', 'full'),
    'DWT-Trend':    ('dwt', 'trend'),
    'DWT-Trend+R':  ('dwt', 'R'),
    'DWT-Trend+WE': ('dwt', 'WE'),
}
FUSION = {
    'WSPI':     ('dtcwt', 'full'),
    'Linear':   ('dtcwt', 'linear'),
    'Product':  ('dtcwt', 'product'),
    'Additive': ('dtcwt', 'additive'),
}
FAMILIES = {'ablation': ABLATION, 'fusion': FUSION}
FORMULA = {'full': 'mu_L*exp(R-WE)', 'trend': 'mu_L', 'R': 'mu_L*exp(R)', 'WE': 'mu_L*exp(-WE)',
           'linear': 'mu_L*(1+R-WE)', 'product': 'mu_L*R*(1-WE)', 'additive': 'mu_L+R-WE'}
# control: variant -> T3.2 grid tag
T32_CONTROL = {'Trend': 'a0.00_b0.00', 'Trend+R': 'a1.00_b0.00', 'Trend+WE': 'a0.00_b1.00'}


def _abs(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


# =============================================================================
# Features
# =============================================================================
def wspi_features(low: np.ndarray, highs) -> tuple:
    """mu_L, R, WE from lowpass (n_items, n) and a list of highpass arrays.
    Same operations as fast_evaluator.make_batch_wspi.core."""
    lm = np.abs(low)
    n = lm.shape[1]
    w = 2.0 ** -np.arange(n)[::-1]
    mu = (lm * w).sum(axis=1) / w.sum()
    e_low = (lm ** 2).sum(axis=1)
    e_h = np.stack([(np.abs(h) ** 2).sum(axis=1) for h in highs], axis=1)
    e_tot = e_low + e_h.sum(axis=1)
    R = np.where(e_tot > 0, e_low / np.where(e_tot > 0, e_tot, 1.0), 0.0)
    E = np.concatenate([e_low[:, None], e_h], axis=1)
    tot = E.sum(axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        P = E / np.where(tot > 0, tot, 1.0)[:, None]
        term = np.where(P > 0, P * np.log2(np.where(P > 0, P, 1.0)), 0.0)
    ent = -term.sum(axis=1)
    max_ent = np.log2(E.shape[1])
    WE = np.where(tot > 0, ent / max_ent if max_ent > 0 else 0.0, 0.0)
    return mu, R, WE


def dwt_coeffs(X: np.ndarray, level: int = LEVEL, window: int = WINDOW):
    """Same decomposition as protocol_v5.make_v5_dwt (DWT+AF)."""
    Xp = _pad_left_reflect(X, _target_len(window, level))
    c = pywt.wavedec(Xp, DWT_WAVELET, level=level, mode=DWT_MODE, axis=-1)
    return c[0], list(reversed(c[1:]))          # lowpass, [level 1 .. level J]


class DWTFeatureCache:
    """mu_L, R, WE from the DWT coefficients, cached per window matrix."""

    def __init__(self, window: int = WINDOW, level: int = LEVEL, use_cache: bool = True):
        self.window, self.level = window, level
        self.use_cache = use_cache
        self.store = {}
        self.hits = 0
        self.misses = 0

    def _compute(self, X):
        low, highs = dwt_coeffs(X, self.level, self.window)
        return wspi_features(low, highs)

    def features(self, X):
        if not self.use_cache:
            self.misses += 1
            return self._compute(X)
        Xc = np.ascontiguousarray(X, dtype=np.float64)
        key = (Xc.shape, hashlib.blake2b(Xc.tobytes(), digest_size=16).digest())
        hit = self.store.get(key)
        if hit is not None:
            self.hits += 1
            return hit
        self.misses += 1
        val = self._compute(X)
        self.store[key] = val
        return val

    def clear(self):
        self.store.clear()


def fuse(kind: str, mu, R, WE, n: int):
    """Score from the features.  'full', 'R', 'WE' and 'trend' repeat the exact
    operations of make_batch_wspi / run_param_grid (expo = 0 + alpha R - beta WE)."""
    if kind == 'full':
        expo = np.zeros(n)
        expo = expo + 1.0 * R
        expo = expo - 1.0 * WE
        return mu * np.exp(expo)
    if kind == 'trend':
        return mu * np.exp(np.zeros(n))
    if kind == 'R':
        return mu * np.exp(np.zeros(n) + 1.0 * R)
    if kind == 'WE':
        return mu * np.exp(np.zeros(n) - 1.0 * WE)
    if kind == 'linear':
        return mu * (1.0 + R - WE)
    if kind == 'product':
        return mu * R * (1.0 - WE)
    if kind == 'additive':
        return mu + R - WE
    raise ValueError(kind)


def make_scorer(cache, kind: str, name: str):
    def f(X):
        mu, R, WE = cache.features(X)
        return fuse(kind, mu, R, WE, X.shape[0])
    f.__name__ = f'v5_{name}_L{LEVEL}_W{WINDOW}'
    return f


def coeff_lengths(window: int = WINDOW, level: int = LEVEL) -> dict:
    X = np.zeros((1, window))
    low, highs = dwt_coeffs(X, level, window)
    tr = dtcwt.Transform1d(biort='near_sym_a', qshift='qshift_a')
    p = tr.forward(X[0], nlevels=level)
    return dict(dwt=dict(lowpass=int(low.shape[1]), highpass=[int(h.shape[1]) for h in highs]),
                dtcwt=dict(lowpass=int(np.asarray(p.lowpass).size),
                           highpass=[int(np.asarray(h).size) for h in p.highpasses]))


# =============================================================================
# Run (one scenario)
# =============================================================================
def run(a):
    ev = ProtocolV5Evaluator.from_csv(_abs(a.data), dataset_min_obs=a.min_obs, seed=a.seed,
                                      robustness=True, causal_universe=a.causal_universe)
    ev.verbose = False
    out = _abs(a.out)
    caches = {'dtcwt': WSPIFeatureCache(window=WINDOW, level=LEVEL, use_cache=not a.no_cache),
              'dwt': DWTFeatureCache(window=WINDOW, level=LEVEL, use_cache=not a.no_cache)}
    done, rt = {}, {}
    t_all = time.time()
    for fam, variants in FAMILIES.items():
        pdir = out / fam / 'protocol'
        pdir.mkdir(parents=True, exist_ok=True)
        res = {}
        for name, (trn, kind) in variants.items():
            f = pdir / f'{name}_protocol.csv'
            key = (trn, kind)
            if key in done:                      # WSPI appears in both families
                df = done[key].copy()
                df['method'] = name
            elif a.resume and f.exists():
                df = pd.read_csv(f)
                rt[f'{fam}/{name}'] = None
                print(f'{fam:<9} {name:<13} (kept from an earlier run)')
            else:
                fm = FastMethod(name, WINDOW, wavelet_min_obs(WINDOW),
                                make_scorer(caches[trn], kind, name))
                df = ev.run_method(fm)
                rt[f'{fam}/{name}'] = round(df.attrs['duration_s'], 2)
                df = df[df['window_id'] >= a.first_window].reset_index(drop=True)
                df['method'] = name
                print(f'{fam:<9} {name:<13} {rt[f"{fam}/{name}"]:8.1f} s  '
                      f'ndcg@10 {df["ndcg@10"].mean():.4f}  rsi@10 {df["rsi@10"].mean():.4f}  '
                      f'robust {df["robustness_distortion"].mean():7.2f}', flush=True)
            if not f.exists() or key not in done:
                tmp = f.with_suffix('.tmp')
                df.to_csv(tmp, index=False, encoding='utf-8')
                tmp.replace(f)
            done[key] = df
            res[name] = df
        cdir = out / fam / 'comparison'
        cdir.mkdir(exist_ok=True)
        summarize(res, common=False).to_csv(cdir / 'summary_all_windows.csv',
                                            index=False, encoding='utf-8')
        summarize(res, common=True).to_csv(cdir / 'summary_common_windows.csv',
                                           index=False, encoding='utf-8')
    meta = dict(task='T3.3 ablation and fusion comparison (E6)', protocol='v5', mode='dense',
                window=WINDOW, level=LEVEL, alpha=1.0, beta=1.0,
                families={fam: {n: dict(transform=t, formula=FORMULA[k])
                                for n, (t, k) in v.items()} for fam, v in FAMILIES.items()},
                dtcwt=dict(biort='near_sym_a', qshift='qshift_a'),
                dwt=dict(wavelet=DWT_WAVELET, mode=DWT_MODE,
                         padding='reflect (left) to 64, as protocol_v5.make_v5_dwt'),
                coefficient_lengths=coeff_lengths(),
                min_obs=wavelet_min_obs(WINDOW), first_window_written=a.first_window,
                seed=a.seed, rsi_by_item=True, robustness=True,
                causal_universe=a.causal_universe,
                universe_rule=(('total count before the test slot >= %d' if a.causal_universe
                                else 'total count over the whole file >= %d') % a.min_obs),
                padding='reflect (left) only while the series is shorter than N',
                tie_rule='stable sort, ties by fixed item order (item ids sorted as str)',
                horizon_slots=1, slot_minutes=ev.slot.total_seconds() / 60,
                num_items=len(ev.items), num_slots=ev.num_slots + 1,
                data=str(a.data), dataset_min_obs=a.min_obs,
                feature_cache={k: dict(enabled=not a.no_cache, hits=c.hits, misses=c.misses)
                               for k, c in caches.items()},
                runtime_seconds=rt, runtime_total_seconds=round(time.time() - t_all, 1),
                versions=dict(python=platform.python_version(), numpy=np.__version__,
                              pandas=pd.__version__, dtcwt=dtcwt.__version__,
                              pywt=pywt.__version__),
                host=platform.platform(), created=time.strftime('%Y-%m-%d %H:%M:%S'))
    (out / 'metadata').mkdir(exist_ok=True)
    (out / 'metadata' / 'ablation_run.json').write_text(json.dumps(meta, indent=2),
                                                        encoding='utf-8')
    print(f'\nSaved to {out}  ({time.time() - t_all:.0f} s)')


# =============================================================================
# Collect
# =============================================================================
def compare_frames(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    """Window-by-window equality of two protocol CSVs (b cut to a's first window)."""
    a = a.set_index('window_id')
    b = b[b['window_id'] >= int(a.index.min())].set_index('window_id')
    same = a.index.equals(b.index)
    worst = 0.0
    if same:
        for m in METRIC_COLUMNS:
            if m in a and m in b and np.issubdtype(a[m].dtype, np.number):
                dd = np.nanmax(np.abs(a[m].to_numpy(float) - b[m].to_numpy(float)))
                worst = max(worst, float(dd) if np.isfinite(dd) else 0.0)
    return dict(n_windows=len(a), n_windows_ref=len(b), same_window_ids=bool(same),
                max_abs_diff=worst, status='equal' if same and worst < 1e-9 else 'DIFFERENT')


def collect(root: Path, ref_t15: Path = None, ref_t32: Path = None):
    ref_t15 = ref_t15 or _abs(REF_T15)
    ref_t32 = ref_t32 or _abs(REF_T32)
    rows, ctrl = [], []
    for scen in sorted(p for p in root.iterdir() if p.is_dir()):
        n_found = 0
        for fam, variants in FAMILIES.items():
            runs = {}
            for name in variants:
                f = scen / fam / 'protocol' / f'{name}_protocol.csv'
                if f.exists():
                    runs[name] = pd.read_csv(f)
            if not runs:
                continue
            n_found += len(runs)
            ids = sorted(set.intersection(*[set(d['window_id']) for d in runs.values()]))
            for name, d in runs.items():
                trn, kind = variants[name]
                x = d[d['window_id'].isin(ids)]
                r = {'scenario': scen.name, 'family': fam, 'variant': name, 'transform': trn,
                     'formula': FORMULA[kind], 'n_windows': len(x),
                     'first_window': int(x['window_id'].min()),
                     'last_window': int(x['window_id'].max()),
                     'same_windows_as_family': len(x) == len(d)}
                for c in METRIC_COLUMNS:
                    r[f'{c}_mean'] = x[c].mean()
                    r[f'{c}_sd'] = x[c].std()
                rows.append(r)
            if fam != 'ablation':
                continue
            checks = [('WSPI', ref_t15 / scen.name / 'protocol' / 'WSPI_protocol.csv', 'T1.5 WSPI')]
            checks += [(v, ref_t32 / scen.name / 'protocol' / f'{t}_protocol.csv', f'T3.2 {t}')
                       for v, t in T32_CONTROL.items()]
            for v, ref, label in checks:
                c = {'scenario': scen.name, 'variant': v, 'reference': label, 'reference_file': str(ref)}
                if v not in runs or not ref.exists():
                    c['status'] = 'missing'
                else:
                    c.update(compare_frames(runs[v], pd.read_csv(ref)))
                ctrl.append(c)
        if n_found:
            st = [c['status'] for c in ctrl if c['scenario'] == scen.name]
            print(f'{scen.name}: {n_found} protocol files; control {st}')
    pd.DataFrame(rows).to_csv(root / 'ablation_summary.csv', index=False, encoding='utf-8')
    pd.DataFrame(ctrl).to_csv(root / 'ablation_control.csv', index=False, encoding='utf-8')
    print(f'Saved {root / "ablation_summary.csv"} and {root / "ablation_control.csv"}')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--data')
    ap.add_argument('--min-obs', type=int, help='dataset min_observations (catalogue filter)')
    ap.add_argument('--out', help='scenario folder')
    ap.add_argument('--first-window', type=int, default=32)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--causal-universe', action='store_true')
    ap.add_argument('--resume', action='store_true', help='keep protocol CSVs already written')
    ap.add_argument('--no-cache', action='store_true', help='recompute the features for every run')
    ap.add_argument('--collect', help='T3.3 root: write ablation_summary.csv and ablation_control.csv')
    a = ap.parse_args()
    if a.collect:
        collect(_abs(a.collect))
    else:
        if not (a.data and a.min_obs is not None and a.out):
            ap.error('--data, --min-obs and --out are required')
        run(a)


if __name__ == '__main__':
    main()
