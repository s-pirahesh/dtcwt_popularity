r"""
alpha x beta grid and leakage-free 30/70 selection (revision-srep-v5, task T3.2 / E3)
====================================================================================
Two jobs, one file.  No existing module is changed.

1) Grid run (one scenario per call)
   WSPI with N = 64, J = 3 under protocol V5 for every (alpha, beta) in
   GRID x GRID, GRID = {0, 0.25, 0.5, 0.75, 1, 1.5, 2} (49 runs).  All other
   settings are those of the T1.4 / T2.2 / T3.1 runs: exact 64-slot window,
   zero fill, reflect padding only while a series is shorter than 64, entry
   rule 32 observed rows, horizon 1 slot, RSI by item id, stable tie-breaking,
   seed 42, robustness test on.  Use --causal-universe (paper setting).

   Speed: the DTCWT features (mu_L, R, WE) do not depend on alpha and beta.
   They are computed once per distinct window matrix and cached (key = shape +
   BLAKE2b digest of the bytes).  The score is then built with the exact
   operations of ``fast_evaluator.make_batch_wspi`` (expo = 0 + alpha R - beta WE;
   mu * exp(expo)), so it is bit-identical to ``protocol_v5.make_v5_wspi``
   (checked by tools/test_param_grid.py and by grid_control.csv).

2) Selection (--select), pre-registered rule (Sajjad, 26 Sep 2026, chat 9)
   Applied per scenario, separately to
     alpha_beta : the 49 grid runs (J = 3, N = 64)
     J          : the T3.1 runs of WSPI with N = 64, J in {2, 3, 4, 5}
                  (alpha = beta = 1), read from --level-root; no new run.
   a. Split: the windows common to all runs of the scenario (window_id >= 32),
      in time order.  The first floor(0.30 n) windows are the tuning part, the
      rest is the test part.  The same split point is used for alpha_beta and
      J (checked; the tool stops if the two window sets differ).
   b. Rule on the tuning part only: NDCG* = best mean NDCG@10 over the grid.
      A configuration is feasible if its mean NDCG@10 >= 0.99 NDCG*.  Among
      feasible ones, pick the highest mean RSI@10.  Ties (means rounded to 4
      decimals): higher NDCG@10, then closest to the default
      (|alpha - 1| + |beta - 1|, or |J - 3|).
   c. Report the chosen configuration and the default (alpha = beta = 1, J = 3)
      on the test part.  For tools/stats_report.py (unchanged) the test-part
      rows are written as
        <root>/selection/test_split/<scenario>/<param>/protocol/{selected,default}_protocol.csv
      (only when selected != default).

Layout
  <root>/<scenario>/protocol/a<alpha>_b<beta>_protocol.csv    windows >= 32
  <root>/<scenario>/comparison/summary_{all,common}_windows.csv
  <root>/<scenario>/metadata/param_grid_run.json
  --collect <root>:
  <root>/grid_summary.csv    scenario x alpha x beta x part (all, tune, test):
                             mean and SD of the metrics
  <root>/grid_control.csv    a1.00_b1.00 against T1.5_causal_universe WSPI
                             (window by window; must be equal)
  --select <root> --level-root <T3.1 root>:
  <root>/selection/selection.csv             one row per scenario x param
  <root>/selection/selection_candidates.csv  every configuration, tuning means,
                                             feasible / selected flags
  <root>/selection/test_split/...            (see c.)
  <root>/selection/metadata/selection_run.json

Examples (from the project root, Windows):

  python tools\run_param_grid.py --data data\datasets\youtube_hourly.csv --min-obs 50 ^
         --causal-universe --out results\revision_v5\T3.2_param_grid\youtube_hourly
  python tools\run_param_grid.py --collect results\revision_v5\T3.2_param_grid
  python tools\run_param_grid.py --select results\revision_v5\T3.2_param_grid ^
         --level-root results\revision_v5\T3.1_level_sweep

Full command list: Revisions/V4/Response/Runbooks/RUN_T3.2.md
Unit test: tools/test_param_grid.py
"""
import argparse
import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import dtcwt  # noqa: E402

from evaluation.fast_evaluator import FastMethod, _dtcwt_forward_rows  # noqa: E402
from evaluation.protocol_v5 import (METRIC_COLUMNS, ProtocolV5Evaluator,  # noqa: E402
                                    _pad_left_reflect, _target_len, summarize)
from evaluation.sweep_methods import wavelet_min_obs  # noqa: E402

GRID = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0)
DEFAULT_AB = (1.0, 1.0)
DEFAULT_J = 3
WINDOW, LEVEL = 64, 3
TUNE_SHARE = 0.30
NDCG_TOL = 0.01                       # feasible: NDCG@10 >= (1 - NDCG_TOL) * NDCG*
ACC, STAB = 'ndcg@10', 'rsi@10'
REPORT_METRICS = ('ndcg@10', 'spearman_rho', 'rsi@10', 'robustness_distortion')
REF_T15 = Path('results/revision_v5/T1.5_causal_universe/T1.4_protocol_v5')


def _abs(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def tag(alpha: float, beta: float) -> str:
    return f'a{alpha:.2f}_b{beta:.2f}'


def parse_tag(s: str):
    a, b = s.split('_')
    return float(a[1:]), float(b[1:])


# =============================================================================
# Cached WSPI scorer
# =============================================================================
class WSPIFeatureCache:
    """mu_L, R and WE of WSPI (protocol V5 padding), cached per window matrix.
    The feature code is the same as ``fast_evaluator.make_batch_wspi.core``."""

    def __init__(self, window: int = WINDOW, level: int = LEVEL,
                 biort='near_sym_a', qshift='qshift_a', use_cache: bool = True):
        self.tr = dtcwt.Transform1d(biort=biort, qshift=qshift)
        self.level = level
        self.tgt = _target_len(window, level)
        self.use_cache = use_cache
        self.store = {}
        self.hits = 0
        self.misses = 0

    def _compute(self, X: np.ndarray):
        Xp = _pad_left_reflect(X, self.tgt)
        low, highs = _dtcwt_forward_rows(self.tr, Xp, self.level)
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

    def features(self, X: np.ndarray):
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

    def scorer(self, alpha: float, beta: float):
        def f(X):
            mu, R, WE = self.features(X)
            expo = np.zeros(X.shape[0])
            expo = expo + alpha * R
            expo = expo - beta * WE
            return mu * np.exp(expo)
        f.__name__ = f'v5_wspi_L{self.level}_W{WINDOW}_{tag(alpha, beta)}'
        return f

    def clear(self):
        self.store.clear()


def grid_order(grid=GRID):
    """Default first, then row by row."""
    pairs = [(a, b) for a in grid for b in grid]
    pairs.remove(DEFAULT_AB)
    return [DEFAULT_AB] + pairs


# =============================================================================
# 1) Grid run
# =============================================================================
def run(a):
    ev = ProtocolV5Evaluator.from_csv(_abs(a.data), dataset_min_obs=a.min_obs, seed=a.seed,
                                      robustness=True, causal_universe=a.causal_universe)
    ev.verbose = False
    out = _abs(a.out)
    pdir = out / 'protocol'
    pdir.mkdir(parents=True, exist_ok=True)
    cache = WSPIFeatureCache(use_cache=not a.no_cache)
    grid = tuple(a.grid)
    res, rt = {}, {}
    t_all = time.time()
    for i, (al, be) in enumerate(grid_order(grid)):
        t = tag(al, be)
        f = pdir / f'{t}_protocol.csv'
        if a.resume and f.exists():
            res[t] = pd.read_csv(f)
            rt[t] = None
            print(f'[{i + 1:2d}/{len(grid) ** 2}] {t}  (kept from an earlier run)')
            continue
        fm = FastMethod('WSPI', WINDOW, wavelet_min_obs(WINDOW), cache.scorer(al, be))
        df = ev.run_method(fm)
        rt[t] = round(df.attrs['duration_s'], 2)
        df = df[df['window_id'] >= a.first_window].reset_index(drop=True)
        df['method'] = t
        df.to_csv(f, index=False, encoding='utf-8')
        res[t] = df
        print(f'[{i + 1:2d}/{len(grid) ** 2}] {t}  {rt[t]:7.1f} s  ndcg@10 {df[ACC].mean():.4f}  '
              f'rsi@10 {df[STAB].mean():.4f}  robust {df["robustness_distortion"].mean():6.2f}')
    (out / 'comparison').mkdir(exist_ok=True)
    summarize(res, common=False).to_csv(out / 'comparison' / 'summary_all_windows.csv',
                                        index=False, encoding='utf-8')
    summarize(res, common=True).to_csv(out / 'comparison' / 'summary_common_windows.csv',
                                       index=False, encoding='utf-8')
    meta = dict(task='T3.2 alpha x beta grid (E3)', protocol='v5', mode='dense', method='WSPI',
                window=WINDOW, level=LEVEL, grid=list(grid), default=list(DEFAULT_AB),
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
                feature_cache=dict(enabled=not a.no_cache, hits=cache.hits, misses=cache.misses),
                runtime_seconds=rt, runtime_total_seconds=round(time.time() - t_all, 1),
                versions=dict(python=platform.python_version(), numpy=np.__version__,
                              pandas=pd.__version__, dtcwt=dtcwt.__version__),
                host=platform.platform(), created=time.strftime('%Y-%m-%d %H:%M:%S'))
    (out / 'metadata').mkdir(exist_ok=True)
    (out / 'metadata' / 'param_grid_run.json').write_text(json.dumps(meta, indent=2),
                                                          encoding='utf-8')
    print(f'\nSaved to {out}  (cache hits {cache.hits}, misses {cache.misses}, '
          f'{time.time() - t_all:.0f} s)')


# =============================================================================
# Shared helpers (collect / select)
# =============================================================================
def split_ids(ids):
    """Time split of the common window ids: first floor(0.30 n) = tuning."""
    ids = sorted(ids)
    n_t = int(math.floor(TUNE_SHARE * len(ids)))
    return ids[:n_t], ids[n_t:]


def load_grid(scen_dir: Path):
    runs = {}
    for f in sorted((scen_dir / 'protocol').glob('a*_b*_protocol.csv')):
        runs[f.name[:-len('_protocol.csv')]] = pd.read_csv(f)
    return runs


def load_levels(level_root: Path, scen: str):
    runs = {}
    d = level_root / scen / f'W{WINDOW:03d}' / 'WSPI' / 'protocol'
    for f in sorted(d.glob('J*_protocol.csv')):
        runs[f.name[:-len('_protocol.csv')]] = pd.read_csv(f)
    return runs


def common_ids(runs):
    return sorted(set.intersection(*[set(d['window_id']) for d in runs.values()]))


def part_means(df: pd.DataFrame, ids, metrics=REPORT_METRICS):
    d = df[df['window_id'].isin(ids)]
    r = {'n_windows': len(d)}
    for c in metrics:
        r[f'{c}_mean'] = d[c].mean()
        r[f'{c}_sd'] = d[c].std()
    return r


# =============================================================================
# 2a) Collect
# =============================================================================
def collect(root: Path):
    rows, ctrl = [], []
    for scen in sorted(p for p in root.iterdir() if p.is_dir() and p.name != 'selection'):
        runs = load_grid(scen)
        if not runs:
            continue
        ids = common_ids(runs)
        tune, test = split_ids(ids)
        for t, d in runs.items():
            al, be = parse_tag(t)
            for part, pid in (('all', ids), ('tune', tune), ('test', test)):
                r = {'scenario': scen.name, 'alpha': al, 'beta': be, 'config': t, 'part': part,
                     'first_window': int(pid[0]), 'last_window': int(pid[-1])}
                r.update(part_means(d, pid, METRIC_COLUMNS))
                rows.append(r)
        # control: default against the T1.5 causal WSPI run
        dt = tag(*DEFAULT_AB)
        ref = _abs(REF_T15) / scen.name / 'protocol' / 'WSPI_protocol.csv'
        c = {'scenario': scen.name, 'config': dt, 'reference': str(REF_T15 / scen.name /
                                                                   'protocol' / 'WSPI_protocol.csv')}
        if dt not in runs or not ref.exists():
            c['status'] = 'missing'
        else:
            a = runs[dt].set_index('window_id')
            b = pd.read_csv(ref)
            b = b[b['window_id'] >= int(a.index.min())].set_index('window_id')
            same = a.index.equals(b.index)
            worst = 0.0
            if same:
                for m in METRIC_COLUMNS:
                    if m in a and m in b and np.issubdtype(a[m].dtype, np.number):
                        dd = np.nanmax(np.abs(a[m].to_numpy(float) - b[m].to_numpy(float)))
                        worst = max(worst, float(dd) if np.isfinite(dd) else 0.0)
            c.update(n_windows=len(a), n_windows_ref=len(b), same_window_ids=bool(same),
                     max_abs_diff=worst,
                     status='equal' if same and worst < 1e-9 else 'DIFFERENT')
        ctrl.append(c)
        print(f'{scen.name}: {len(runs)} runs, {len(ids)} common windows '
              f'(tune {len(tune)}, test {len(test)}); control {c["status"]}')
    pd.DataFrame(rows).to_csv(root / 'grid_summary.csv', index=False, encoding='utf-8')
    pd.DataFrame(ctrl).to_csv(root / 'grid_control.csv', index=False, encoding='utf-8')
    print(f'Saved {root / "grid_summary.csv"} and {root / "grid_control.csv"}')


# =============================================================================
# 2b) Select
# =============================================================================
def select_config(tab: pd.DataFrame, dist_col: str, tol: float = NDCG_TOL) -> pd.DataFrame:
    """tab: one row per configuration with tune_ndcg, tune_rsi and dist_col
    (distance to the default).  Adds 'feasible' and 'selected' columns."""
    t = tab.copy()
    best = t['tune_ndcg'].max()
    t['ndcg_star'] = best
    t['ndcg_threshold'] = (1.0 - tol) * best
    t['feasible'] = t['tune_ndcg'] >= t['ndcg_threshold']
    f = t[t['feasible']].copy()
    f['_rsi'] = f['tune_rsi'].round(4)
    f['_ndcg'] = f['tune_ndcg'].round(4)
    f = f.sort_values(['_rsi', '_ndcg', dist_col], ascending=[False, False, True], kind='mergesort')
    t['selected'] = False
    t.loc[f.index[0], 'selected'] = True
    return t


def _write_test_split(root: Path, scen: str, param: str, runs, sel: str, dflt: str, test_ids):
    d = root / 'selection' / 'test_split' / scen / param
    res = {}
    for name, key in (('selected', sel), ('default', dflt)):
        x = runs[key]
        x = x[x['window_id'].isin(test_ids)].copy()
        x['method'] = name
        res[name] = x
        (d / 'protocol').mkdir(parents=True, exist_ok=True)
        x.to_csv(d / 'protocol' / f'{name}_protocol.csv', index=False, encoding='utf-8')
    (d / 'comparison').mkdir(exist_ok=True)
    summarize(res, common=True).to_csv(d / 'comparison' / 'summary_common_windows.csv',
                                       index=False, encoding='utf-8')
    return d


def select(root: Path, level_root: Path):
    sel_rows, cand_rows = [], []
    scen_names = sorted(p.name for p in root.iterdir()
                        if p.is_dir() and p.name != 'selection' and (p / 'protocol').is_dir())
    if level_root and level_root.is_dir():
        scen_names = sorted(set(scen_names) | {p.name for p in level_root.iterdir()
                                               if (p / f'W{WINDOW:03d}' / 'WSPI').is_dir()})
    for scen in scen_names:
        sets = {}
        g = load_grid(root / scen) if (root / scen).is_dir() else {}
        if g:
            sets['alpha_beta'] = g
        if level_root:
            lv = load_levels(level_root, scen)
            if lv:
                sets['J'] = lv
        if not sets:
            continue
        idsets = {k: common_ids(v) for k, v in sets.items()}
        if len(idsets) == 2 and idsets['alpha_beta'] != idsets['J']:
            sys.exit(f'{scen}: the alpha_beta and J runs do not share the same windows '
                     f'({len(idsets["alpha_beta"])} vs {len(idsets["J"])}); stop.')
        ids = next(iter(idsets.values()))
        tune, test = split_ids(ids)
        for param, runs in sets.items():
            rows = []
            for key, d in runs.items():
                if param == 'alpha_beta':
                    al, be = parse_tag(key)
                    dist = abs(al - DEFAULT_AB[0]) + abs(be - DEFAULT_AB[1])
                    cfg = dict(alpha=al, beta=be, J=DEFAULT_J)
                else:
                    j = int(key[1:])
                    dist = abs(j - DEFAULT_J)
                    cfg = dict(alpha=DEFAULT_AB[0], beta=DEFAULT_AB[1], J=j)
                tm = part_means(d, tune)
                sm = part_means(d, test)
                r = {'scenario': scen, 'param': param, 'config': key, **cfg,
                     'dist_to_default': dist,
                     'tune_ndcg': tm[f'{ACC}_mean'], 'tune_rsi': tm[f'{STAB}_mean']}
                for c in REPORT_METRICS:
                    r[f'tune_{c}'] = tm[f'{c}_mean']
                    r[f'test_{c}'] = sm[f'{c}_mean']
                rows.append(r)
            tab = select_config(pd.DataFrame(rows), 'dist_to_default')
            dflt = tag(*DEFAULT_AB) if param == 'alpha_beta' else f'J{DEFAULT_J}'
            tab['is_default'] = tab['config'] == dflt
            cand_rows.append(tab)
            s = tab[tab['selected']].iloc[0]
            dr = tab[tab['is_default']].iloc[0]
            row = {'scenario': scen, 'param': param,
                   'n_common': len(ids), 'n_tune': len(tune), 'n_test': len(test),
                   'first_tune_window': int(tune[0]), 'last_tune_window': int(tune[-1]),
                   'first_test_window': int(test[0]), 'last_test_window': int(test[-1]),
                   'ndcg_star': s['ndcg_star'], 'ndcg_threshold': s['ndcg_threshold'],
                   'n_configs': len(tab), 'n_feasible': int(tab['feasible'].sum()),
                   'default_feasible': bool(dr['feasible']),
                   'selected': s['config'], 'selected_alpha': s['alpha'],
                   'selected_beta': s['beta'], 'selected_J': int(s['J']),
                   'default': dflt, 'same_as_default': bool(s['config'] == dflt)}
            for part in ('tune', 'test'):
                for c in REPORT_METRICS:
                    row[f'{part}_{c}_selected'] = s[f'{part}_{c}']
                    row[f'{part}_{c}_default'] = dr[f'{part}_{c}']
            if s['config'] != dflt:
                d = _write_test_split(root, scen, param, runs, s['config'], dflt, test)
                row['test_split_folder'] = str(d.relative_to(root))
            else:
                row['test_split_folder'] = ''
            sel_rows.append(row)
            print(f'{scen:<15} {param:<10} tune {len(tune):6d} / test {len(test):6d}  '
                  f'feasible {row["n_feasible"]:2d}/{len(tab):2d}  selected {s["config"]:<12} '
                  f'test ndcg@10 {s["test_ndcg@10"]:.4f} (default {dr["test_ndcg@10"]:.4f})  '
                  f'rsi@10 {s["test_rsi@10"]:.4f} (default {dr["test_rsi@10"]:.4f})')
    sd = root / 'selection'
    (sd / 'metadata').mkdir(parents=True, exist_ok=True)
    pd.DataFrame(sel_rows).to_csv(sd / 'selection.csv', index=False, encoding='utf-8')
    pd.concat(cand_rows, ignore_index=True).to_csv(sd / 'selection_candidates.csv',
                                                   index=False, encoding='utf-8')
    meta = dict(task='T3.2 leakage-free 30/70 selection (E3)',
                rule=('tuning part = first floor(0.30 n) common windows (window_id >= 32) in time '
                      'order; NDCG* = best tuning mean NDCG@10; feasible if NDCG@10 >= 0.99 NDCG*; '
                      'select max tuning mean RSI@10; ties (4 decimals) -> higher NDCG@10 -> '
                      'closest to default'),
                decided='Sajjad, 26 Sep 2026 (chat 9), before any grid result was seen',
                tune_share=TUNE_SHARE, ndcg_tolerance=NDCG_TOL, default_alpha_beta=DEFAULT_AB,
                default_J=DEFAULT_J, window=WINDOW, level_for_alpha_beta=LEVEL,
                grid_root=str(root), level_root=str(level_root) if level_root else None,
                versions=dict(python=platform.python_version(), numpy=np.__version__,
                              pandas=pd.__version__),
                created=time.strftime('%Y-%m-%d %H:%M:%S'))
    (sd / 'metadata' / 'selection_run.json').write_text(json.dumps(meta, indent=2),
                                                        encoding='utf-8')
    print(f'Saved to {sd}')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--data')
    ap.add_argument('--min-obs', type=int, help='dataset min_observations (catalogue filter)')
    ap.add_argument('--out', help='scenario folder')
    ap.add_argument('--grid', nargs='*', type=float, default=list(GRID))
    ap.add_argument('--first-window', type=int, default=32)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--causal-universe', action='store_true')
    ap.add_argument('--resume', action='store_true', help='keep protocol CSVs already written')
    ap.add_argument('--no-cache', action='store_true', help='recompute DTCWT for every run')
    ap.add_argument('--collect', help='grid root: write grid_summary.csv and grid_control.csv')
    ap.add_argument('--select', help='grid root: apply the 30/70 rule (writes <root>/selection)')
    ap.add_argument('--level-root', help='T3.1 root (results/revision_v5/T3.1_level_sweep)')
    a = ap.parse_args()
    if a.collect:
        collect(_abs(a.collect))
    elif a.select:
        select(_abs(a.select), _abs(a.level_root) if a.level_root else None)
    else:
        if not (a.data and a.min_obs is not None and a.out):
            ap.error('--data, --min-obs and --out are required')
        if 1.0 not in a.grid:
            ap.error('the grid must contain 1.0 (default)')
        run(a)


if __name__ == '__main__':
    main()
