r"""
Leakage and configuration audit for protocol V5 (revision-srep-v5, T1.5 / E12)
==============================================================================
Answers R4.13 (no temporal leakage), R3.7 (same settings for all datasets)
and the STRATA_THRESHOLDS question of the experiments plan (E12).

Five checks, one CSV each, in --out:

  window_structure.csv   For every evaluated window of every method (T1.4 and
                         T2.1 runs), the training slice rebuilt with the
                         evaluator's own ``window_bounds``: last training slot
                         < test slot, exactly W slots once k >= W, and the
                         ``padded`` flag of the protocol CSV agrees.
  truncation_test.csv    Empirical causality test.  The evaluator is rebuilt
                         on the data cut at slot T (everything after T is
                         deleted) and every method is re-run.  For windows
                         k <= T every protocol value must be identical to the
                         full run: if any score, eligibility rule, metric or
                         robustness sample used data after the test slot, the
                         values would change.  The item universe of the full
                         run is kept, so this isolates the pipeline itself.
  perturbation_test.csv  Per-window test.  For sampled windows k, the test slot
                         and the next slots (--perturb-slots) are overwritten
                         with random values and random presence; eligible
                         items, scores and the robustness value must not
                         change.  Positive control: an evaluator whose
                         training slice is shifted one slot forward (so it
                         sees the test slot) must be flagged.
  item_universe.csv      The only full-period statistic in the pipeline: the
                         catalogue filter (total count over the whole file >=
                         dataset_min_obs; replica of the V4
                         ``_select_items_from_data``).  Reports how many items
                         it removes, their largest single-slot count, and in
                         how many windows a removed item would have reached
                         the true top-10 (the 10th largest test count of the
                         eligible items).
  strata_usage.csv       Every line of project code that mentions strata /
                         STRATA_THRESHOLDS / stratification, with a flag that
                         says whether the file is on the V5 import path.
  strata_v4_runs.csv     item_selection / num_items / strata_thresholds of the
                         V4 runs the paper used.
  config_table.csv       One row per scenario x run x method from
                         metadata/protocol_v5_run.json (R3.7 table).
  metadata/leakage_audit_run.json

Examples (from the project root, Windows):

  # runs made with the causal item catalogue (the basis of the paper, T1.5)
  python tools\leakage_audit.py --causal-universe --run-root results\revision_v5\T1.5_causal_universe ^
         --out results\revision_v5\T1.5_leakage_audit\causal

  # the earlier T1.4 / T2.1 runs (whole-file catalogue)
  python tools\leakage_audit.py --out results\revision_v5\T1.5_leakage_audit\whole_file

Full command list: Revisions/V4/Response/Runbooks/RUN_T1.6_T1.5.md
"""
import argparse
import ast
import json
import platform
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.protocol_v5 import ProtocolV5Evaluator, build_v5_methods  # noqa: E402

V5 = Path('results') / 'revision_v5'     # run root; --run-root overrides
SCENARIOS = {
    # name: (data file, dataset_min_obs)
    'youtube_hourly': ('data/datasets/youtube_hourly.csv', 50),
    'taxi_hourly': ('data/datasets/yellow_taxi_2025_all_hourly.csv', 24),
    'taxi_30min': ('data/datasets/yellow_taxi_2025_all_30min.csv', 24),
    'taxi_5min': ('data/datasets/yellow_taxi_2025_all_5min.csv', 24),
}
RUN_GROUPS = ['T1.4_protocol_v5', 'T2.1_baselines_W16']
V4_RUNS = {
    'youtube_hourly': 'results/youtube/main_20260612_140555',
    'taxi_hourly': 'results/yellow_taxi/predcmp_20260830_001817_hourly',
    'taxi_30min': 'results/yellow_taxi/main_20260620_092517_30min',
    'taxi_5min': 'results/yellow_taxi/main_20260620_170044_5min',
}
COMPARE = ['num_items', 'ndcg@5', 'coverage@5', 'ndcg@10', 'coverage@10', 'ndcg@20',
           'coverage@20', 'kendall_tau', 'spearman_rho', 'mae', 'rsi@5', 'rsi@10',
           'rsi@20', 'robustness_distortion', 'ties_top21', 'padded', 'n_nonfinite']
V5_ENTRY = ['evaluation/protocol_v5.py', 'tools/run_v5_eval.py']
STRATA_RE = re.compile(r'STRATA_THRESHOLDS|strata_thresholds|StratificationSystem|stratif|strata',
                       re.IGNORECASE)
SKIP_DIRS = {'.git', '.claude', 'results', '__pycache__', 'venv', '.venv', 'env', 'data'}


def rel(p: Path) -> str:
    return p.relative_to(ROOT).as_posix()


def log(msg):
    print(msg, flush=True)


# =============================================================================
# 1. Window structure
# =============================================================================
def check_windows(ev, scen: str) -> list:
    rows = []
    for grp in RUN_GROUPS:
        run = ROOT / V5 / grp / scen
        mfile = run / 'metadata' / 'protocol_v5_run.json'
        if not mfile.exists():
            continue
        meta = json.loads(mfile.read_text(encoding='utf-8'))
        for m, info in meta['methods'].items():
            W = int(info['window_slots'])
            df = pd.read_csv(run / 'protocol' / f'{m}_protocol.csv',
                             usecols=['window_id', 'timestamp', 'padded'])
            k = df['window_id'].to_numpy()
            b = np.array([ev.window_bounds(int(x), W) for x in k])
            lo, hi = b[:, 0], b[:, 1]
            slot_ms = ev.slot.total_seconds() * 1000
            start_ms = ev.start.timestamp() * 1000
            test_ms = start_ms + k * slot_ms
            train_last_ms = start_ms + (hi - 1) * slot_ms
            full = k >= W
            rows.append({
                'scenario': scen, 'run_group': grp, 'method': m, 'W': W,
                'n_windows': len(k),
                'timestamps_match': bool(np.allclose(test_ms, df['timestamp'].to_numpy())),
                'train_end_equals_test_slot': bool((hi == k).all()),
                'last_train_before_test': bool((train_last_ms < test_ms).all()),
                'max_train_to_test_gap_slots': int((k - (hi - 1)).max()),
                'train_len_W_when_k_ge_W': bool(((hi - lo)[full] == W).all()),
                'n_short_windows': int((~full).sum()),
                # the protocol flag marks a slice shorter than W (k < W); only the
                # wavelet-based methods are then padded, the baselines use the short slice
                'padded_flag_consistent': bool(
                    (df['padded'].astype(bool).to_numpy() == ((hi - lo) < W)).all()),
            })
    return rows


# =============================================================================
# 2. Truncation test
# =============================================================================
def truncation_test(raw: pd.DataFrame, ev_full, scen: str, frac: float, seed: int,
                    min_obs: int) -> list:
    rows = []
    grp = RUN_GROUPS[0]
    run = ROOT / V5 / grp / scen
    meta = json.loads((run / 'metadata' / 'protocol_v5_run.json').read_text(encoding='utf-8'))
    T = int(frac * ev_full.num_slots)
    t_cut = ev_full.start + T * ev_full.slot
    if ev_full.causal_universe:
        # causal catalogue: the cut file is used exactly as a real run would use it
        cut = raw[raw['timestamp'] <= t_cut]
        ev = ProtocolV5Evaluator(cut, dataset_min_obs=min_obs, seed=seed, robustness=True,
                                 causal_universe=True, verbose=False)
    else:
        # whole-file catalogue: keep the full run's items to isolate the pipeline
        cut = raw[(raw['timestamp'] <= t_cut) & raw['item_id'].isin(set(ev_full.items))]
        ev = ProtocolV5Evaluator(cut, dataset_min_obs=0, seed=seed, robustness=True,
                                 verbose=False)
    if ev.start != ev_full.start:
        raise RuntimeError('truncated data starts at a different slot')
    wo = {m: int(v['window_slots']) for m, v in meta['methods'].items()}
    methods = build_v5_methods(level=int(meta.get('level', 3)), window_override=wo,
                               names=list(meta['methods']))
    for m, fm in methods.items():
        t0 = time.time()
        a = ev.run_method(fm).set_index('window_id')
        b = pd.read_csv(run / 'protocol' / f'{m}_protocol.csv').set_index('window_id')
        b = b[b.index <= T]
        common = a.index.intersection(b.index)
        r = {'scenario': scen, 'run_group': grp, 'method': m, 'W': fm.window_slots,
             'cut_slot': T, 'cut_time': str(t_cut),
             'n_windows_full_run': len(b), 'n_windows_truncated_run': len(a),
             'n_compared': len(common)}
        worst = 0.0
        nbad = 0
        for c in COMPARE:
            x = a.loc[common, c].astype(float).to_numpy()
            y = b.loc[common, c].astype(float).to_numpy()
            same = (x == y) | (np.isnan(x) & np.isnan(y))
            # relative difference (MAE is in counts, up to ~1e5)
            diff = np.where(same, 0.0, np.abs(x - y) / np.maximum(1.0, np.abs(y)))
            diff = np.nan_to_num(diff, nan=np.inf)
            worst = max(worst, float(diff.max()) if len(diff) else 0.0)
            nbad = max(nbad, int((diff > 1e-9).sum()))
        r['max_rel_diff'] = worst
        r['n_windows_differing'] = nbad
        r['identical'] = bool(len(a) == len(b) and len(common) == len(b) and worst <= 1e-9)
        r['seconds'] = round(time.time() - t0, 1)
        rows.append(r)
        log(f'   truncation {scen:<15} {m:<12} compared {len(common):>6}  '
            f'max rel diff {worst:.3g}  identical={r["identical"]}')
    return rows


# =============================================================================
# 2b. Per-window future perturbation (+ positive control)
# =============================================================================
class _LeakyV5(ProtocolV5Evaluator):
    """Positive control: the training slice is shifted one slot forward, so it
    contains the test slot.  The perturbation test must flag it."""
    def _matrix(self, idx, k, W):
        lo, hi = self.window_bounds(k, W)
        return self.V[idx, lo + 1:hi + 1]


def _window_values(ev, fm, k, seed):
    """Same steps as ProtocolV5Evaluator.run_method for one window:
    eligibility, training matrix, scores, robustness (fresh seeded RNG)."""
    lo, hi = ev.window_bounds(k, fm.window_slots)
    ok = (ev.C[:, hi] - ev.C[:, lo]) >= fm.min_obs
    if getattr(ev, 'causal_universe', False):          # running total before slot k
        ok &= ev.V[:, :k].sum(axis=1) >= ev.universe_min_count
    idx = np.where(ok)[0]
    if len(idx) < 2:
        return idx, None, None
    X = ev._matrix(idx, k, fm.window_slots)
    sc = ev._safe_score(fm.scorer, X)
    rob = ev._robustness_v5(fm, X, sc, np.random.RandomState(seed))
    return idx, sc, rob


def perturbation_test(ev, scen: str, n_windows: int, horizon: int, seed: int) -> list:
    """For sampled windows k, overwrite slots k .. k+horizon-1 (test slot and
    future) with random values and random presence, recompute the window, and
    compare eligible items, scores and robustness with the unperturbed values.
    Arrays are restored after every window."""
    run = ROOT / V5 / RUN_GROUPS[0] / scen
    meta = json.loads((run / 'metadata' / 'protocol_v5_run.json').read_text(encoding='utf-8'))
    wo = {m: int(v['window_slots']) for m, v in meta['methods'].items()}
    methods = build_v5_methods(level=int(meta.get('level', 3)), window_override=wo,
                               names=list(meta['methods']))
    leaky = _LeakyV5.__new__(_LeakyV5)
    leaky.__dict__ = ev.__dict__                      # same arrays, leaky _matrix
    cases = [(m, fm, ev, False) for m, fm in methods.items()] + \
            [('AF', methods['AF'], leaky, True)]
    rng = np.random.default_rng(seed)
    S = ev.V.shape[1]
    scale = float(ev.V[ev.M].mean()) if ev.M.any() else 1.0
    rows = []
    for m, fm, e, control in cases:
        ks = [k for k in range(fm.window_slots, ev.num_slots + 1) if ev.any_row[k]]
        ks = sorted(rng.choice(ks, size=min(n_windows, len(ks)), replace=False).tolist())
        changed = 0
        tested = 0
        for k in ks:
            idx0, sc0, rob0 = _window_values(e, fm, k, seed)
            if sc0 is None:
                continue
            hi = min(k + horizon, S)
            Vs, Ms, Cs = ev.V[:, k:hi].copy(), ev.M[:, k:hi].copy(), ev.C[:, k + 1:].copy()
            try:
                ev.M[:, k:hi] = rng.random(ev.M[:, k:hi].shape) < 0.5
                ev.V[:, k:hi] = np.where(ev.M[:, k:hi],
                                         rng.exponential(10 * scale, ev.V[:, k:hi].shape), 0.0)
                ev.C[:, k + 1:] = ev.C[:, k][:, None] + np.cumsum(ev.M[:, k:], axis=1)
                idx1, sc1, rob1 = _window_values(e, fm, k, seed)
            finally:
                ev.V[:, k:hi], ev.M[:, k:hi], ev.C[:, k + 1:] = Vs, Ms, Cs
            tested += 1
            same = (sc1 is not None and np.array_equal(idx0, idx1)
                    and np.array_equal(sc0, sc1, equal_nan=True)
                    and ((np.isnan(rob0) and np.isnan(rob1)) or rob0 == rob1))
            changed += int(not same)
        rows.append({'scenario': scen, 'run_group': RUN_GROUPS[0], 'method': m,
                     'positive_control': control, 'W': fm.window_slots,
                     'perturbed_slots': horizon, 'n_windows_tested': tested,
                     'n_windows_changed': changed,
                     'pass': (changed > 0) if control else (changed == 0)})
        log(f'   perturbation {scen:<15} {m:<12}{" (control)" if control else "          "} '
            f'tested {tested:>4}  changed {changed:>4}')
    return rows


# =============================================================================
# 3. Item universe (catalogue filter)
# =============================================================================
def item_universe(raw: pd.DataFrame, ev, scen: str, min_obs: int) -> dict:
    tot = raw.groupby('item_id')['count'].sum()
    removed = tot[tot < min_obs].index
    r = {'scenario': scen, 'dataset_min_obs': min_obs,
         'items_in_file': len(tot), 'items_kept': len(ev.items),
         'items_removed': len(removed),
         'removed_share_of_total_count': float(tot[removed].sum() / tot.sum()),
         'min_item_total_in_file': float(tot.min()),
         'removed_max_slot_count': float(raw.loc[raw['item_id'].isin(removed), 'count'].max())
         if len(removed) else 0.0}
    # 10th largest test count among items eligible for the baselines (W=7, 3 rows):
    # the broadest candidate set of any method.
    W, mo = 7, 3
    tenth = []
    rem = raw[raw['item_id'].isin(removed)]
    rem_slot = np.round((rem['timestamp'] - ev.start) / ev.slot).astype(int)
    rem_max = pd.Series(rem['count'].to_numpy(), index=rem_slot.to_numpy()).groupby(level=0).max()
    reach = 0
    nwin = 0
    for k in range(ev.num_slots + 1):
        lo, hi = ev.window_bounds(k, W)
        if ev.any_cum[hi] - ev.any_cum[lo] == 0 or not ev.any_row[k]:
            continue
        idx = np.where((ev.C[:, hi] - ev.C[:, lo]) >= mo)[0]
        if len(idx) < 2:
            continue
        act = np.sort(ev.V[idx, k])[::-1]
        t10 = act[min(9, len(act) - 1)]
        tenth.append(t10)
        nwin += 1
        if k in rem_max.index and rem_max[k] >= t10 and t10 > 0:
            reach += 1
    tenth = np.array(tenth)
    r.update({'n_windows_checked': nwin,
              'min_true_top10_threshold': float(tenth.min()) if nwin else np.nan,
              'median_true_top10_threshold': float(np.median(tenth)) if nwin else np.nan,
              'share_windows_top10_threshold_zero': float((tenth == 0).mean()) if nwin else np.nan,
              'windows_removed_item_reaches_top10': reach})
    return r


# =============================================================================
# 4. STRATA usage (static) and V4 run settings
# =============================================================================
def import_closure(entries) -> set:
    """Project files reachable by import statements from the entry files."""
    seen, todo = set(), [ROOT / e for e in entries]
    while todo:
        f = todo.pop()
        if not f.exists() or f in seen:
            continue
        seen.add(f)
        try:
            tree = ast.parse(f.read_text(encoding='utf-8'))
        except Exception:
            continue
        pkg = f.parent
        for node in ast.walk(tree):
            mods = []
            if isinstance(node, ast.ImportFrom):
                base = pkg if node.level else ROOT
                for _ in range(max(node.level - 1, 0)):
                    base = base.parent
                mod = (node.module or '').replace('.', '/')
                mods.append(base / mod if mod else base)
                for a in node.names:
                    mods.append((base / mod / a.name) if mod else (base / a.name))
            elif isinstance(node, ast.Import):
                for a in node.names:
                    mods.append(ROOT / a.name.replace('.', '/'))
            for m in mods:
                for cand in (m.with_suffix('.py'), m / '__init__.py'):
                    if cand.exists() and ROOT in cand.parents:
                        todo.append(cand)
    return seen


def strata_usage() -> tuple:
    closure = import_closure(V5_ENTRY)
    # a package __init__ runs on import but is not called by the V5 code;
    # it is reported separately
    rows = []
    for f in sorted(ROOT.rglob('*.py')):
        if any(p in SKIP_DIRS for p in f.relative_to(ROOT).parts[:-1]) or \
                f.resolve() == Path(__file__).resolve():
            continue
        try:
            lines = f.read_text(encoding='utf-8', errors='replace').splitlines()
        except Exception:
            continue
        for i, line in enumerate(lines, 1):
            if STRATA_RE.search(line):
                rows.append({'file': rel(f), 'line': i, 'text': line.strip()[:160],
                             'on_v5_import_path': f in closure and f.name != '__init__.py',
                             'package_init': f.name == '__init__.py'})
    return rows, sorted(rel(f) for f in closure)


def strata_v4() -> list:
    rows = []
    for scen, p in V4_RUNS.items():
        f = ROOT / p / 'metadata' / 'config.json'
        if not f.exists():
            rows.append({'scenario': scen, 'run': p, 'found': False})
            continue
        c = json.loads(f.read_text(encoding='utf-8'))
        rows.append({'scenario': scen, 'run': p, 'found': True,
                     'item_selection': c.get('item_selection'),
                     'num_items': c.get('num_items'),
                     'min_observations': c.get('min_observations'),
                     'strata_thresholds': json.dumps(c.get('strata_thresholds'))})
    return rows


# =============================================================================
# 5. Configuration table
# =============================================================================
def config_table() -> list:
    rows = []
    for grp in RUN_GROUPS:
        for scen in SCENARIOS:
            f = ROOT / V5 / grp / scen / 'metadata' / 'protocol_v5_run.json'
            if not f.exists():
                continue
            m = json.loads(f.read_text(encoding='utf-8'))
            for name, info in m['methods'].items():
                rows.append({'run_group': grp, 'scenario': scen, 'method': name,
                             'window_slots': info['window_slots'], 'min_obs': info['min_obs'],
                             'scorer': info['scorer'], 'level_J': m.get('level'),
                             'dataset_min_obs': m.get('dataset_min_obs'),
                             'slot_minutes': m.get('slot_minutes'),
                             'num_items': m.get('num_items'), 'num_slots': m.get('num_slots'),
                             'seed': m.get('seed'), 'tie_rule': m.get('tie_rule'),
                             'rsi_by_item': m.get('rsi_by_item'), 'mode': m.get('mode'),
                             'causal_universe': m.get('causal_universe', False),
                             'universe_rule': m.get('universe_rule'),
                             'robustness': m.get('robustness')})
    return rows


# =============================================================================
def main():
    global V5
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--out', required=True)
    ap.add_argument('--scenarios', nargs='*', default=list(SCENARIOS))
    ap.add_argument('--truncate-frac', type=float, default=0.25,
                    help='cut point of the truncation test, as a share of all slots')
    ap.add_argument('--skip-truncation', action='store_true')
    ap.add_argument('--perturb-windows', type=int, default=200,
                    help='windows per method in the perturbation test')
    ap.add_argument('--perturb-slots', type=int, default=64,
                    help='test slot + future slots overwritten in the perturbation test')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--run-root', default=str(V5),
                    help='folder that holds <run group>/<scenario> run folders')
    ap.add_argument('--causal-universe', action='store_true',
                    help='audit runs made with run_v5_eval.py --causal-universe')
    a = ap.parse_args()
    V5 = Path(a.run_root)
    out = Path(a.out)
    out = out if out.is_absolute() else ROOT / out
    (out / 'metadata').mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    su, closure = strata_usage()
    pd.DataFrame(su).to_csv(out / 'strata_usage.csv', index=False)
    pd.DataFrame(strata_v4()).to_csv(out / 'strata_v4_runs.csv', index=False)
    pd.DataFrame(config_table()).to_csv(out / 'config_table.csv', index=False)
    log(f'strata lines: {len(su)}  on V5 import path: {sum(r["on_v5_import_path"] for r in su)}')

    win, trunc, uni, pert = [], [], [], []
    for scen in a.scenarios:
        path, mo = SCENARIOS[scen]
        log(f'== {scen}')
        raw = pd.read_csv(ROOT / path)
        raw['item_id'] = raw['item_id'].astype(str)
        raw['timestamp'] = pd.to_datetime(raw['timestamp'])
        ev = ProtocolV5Evaluator(raw, dataset_min_obs=mo, seed=a.seed, verbose=False,
                                 causal_universe=a.causal_universe)
        win += check_windows(ev, scen)
        uni.append(item_universe(raw, ev, scen, mo))
        log(f'   universe: {uni[-1]["items_in_file"]} items, removed {uni[-1]["items_removed"]}, '
            f'removed item reaches top-10 in {uni[-1]["windows_removed_item_reaches_top10"]} windows')
        pert += perturbation_test(ev, scen, a.perturb_windows, a.perturb_slots, a.seed)
        pd.DataFrame(pert).to_csv(out / 'perturbation_test.csv', index=False)
        if not a.skip_truncation:
            trunc += truncation_test(raw, ev, scen, a.truncate_frac, a.seed, mo)
        # write after every scenario so a long run leaves partial results
        pd.DataFrame(win).to_csv(out / 'window_structure.csv', index=False)
        pd.DataFrame(uni).to_csv(out / 'item_universe.csv', index=False)
        if trunc:
            pd.DataFrame(trunc).to_csv(out / 'truncation_test.csv', index=False)

    w = pd.DataFrame(win)
    checks = ['timestamps_match', 'train_end_equals_test_slot', 'last_train_before_test',
              'train_len_W_when_k_ge_W', 'padded_flag_consistent']
    meta = {
        'task': 'T1.5 (E12) leakage and configuration audit',
        'scenarios': a.scenarios, 'truncate_frac': a.truncate_frac,
        'run_root': str(V5), 'causal_universe': a.causal_universe,
        'truncation_item_universe': ('causal catalogue, cut file used as is'
                                     if a.causal_universe else
                                     'items of the full run (isolates the pipeline)'),
        'seed': a.seed,
        'window_checks_all_pass': bool(w[checks].all().all()) if len(w) else None,
        'truncation_all_identical': (bool(pd.DataFrame(trunc)['identical'].all())
                                     if trunc else None),
        'perturbation_all_pass': bool(pd.DataFrame(pert)['pass'].all()) if pert else None,
        'perturb_windows': a.perturb_windows, 'perturb_slots': a.perturb_slots,
        'v5_import_closure': closure,
        'versions': {'python': platform.python_version(), 'numpy': np.__version__,
                     'pandas': pd.__version__},
        'host': platform.platform(),
        'date_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'runtime_seconds': round(time.time() - t0, 1),
    }
    (out / 'metadata' / 'leakage_audit_run.json').write_text(json.dumps(meta, indent=2),
                                                            encoding='utf-8')
    log(f"window checks all pass: {meta['window_checks_all_pass']}   "
        f"truncation identical: {meta['truncation_all_identical']}   "
        f"perturbation pass: {meta['perturbation_all_pass']}   "
        f"{meta['runtime_seconds']} s\nSaved to {out}")


if __name__ == '__main__':
    main()
