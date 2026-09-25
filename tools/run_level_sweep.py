r"""
Decomposition-level sweep under protocol V5 (revision-srep-v5, task T3.1 / E2)
=============================================================================
WSPI and DTCWT+AF are run at every level J of the grid, for two window
lengths (E2): N = 64 with J in {2, 3, 4, 5} and N = 32 with J in {2, 3, 4}.
A pair (N, J) is run only if N >= 2**(J + 1), the same rule as the window
sweep (so J = 5 is skipped at N = 32).  All other settings are those of the
window sweep T2.2 (``evaluation/sweep_methods.py``): exact N-slot window,
zero fill, reflect padding only while a series is shorter than N, entry rule
min(32, N / 2) observed rows, horizon 1 slot, RSI by item id, stable
tie-breaking, seed 42.  Use --causal-universe (paper setting).

Settings approved by Sajjad (25 Sep 2026, chat 8): methods WSPI and DTCWT+AF
(DWT+AF is left out: with db4 the largest level without boundary effects is
3 for 64 samples and 2 for 32 samples).

Layout (one folder per scenario x N x method, so that tools/stats_report.py
works unchanged with --reference J3):

  <out>/W<NNN>/<method>/protocol/J<j>_protocol.csv     windows >= --first-window
  <out>/W<NNN>/<method>/comparison/summary_{all,common}_windows.csv
  <out>/W<NNN>/<method>/metadata/level_sweep_run.json

--collect <root> writes, over every scenario folder under <root>:
  level_sweep_summary.csv  scenario x N x method x J, mean and SD of every
                           metric on the windows common to ALL (N, method, J)
                           runs of that scenario
  level_control.csv        J = 3 runs against the reference runs:
                           N = 64 -> T1.5_causal_universe/T1.4_protocol_v5/<scenario>
                           N = 32 -> T2.2_window_sweep/<scenario>/W032
                           (window by window, windows >= 32; must be equal)

Examples (from the project root, Windows):

  python tools\run_level_sweep.py --data data\datasets\youtube_hourly.csv --min-obs 50 ^
         --causal-universe --out results\revision_v5\T3.1_level_sweep\youtube_hourly
  python tools\run_level_sweep.py --collect results\revision_v5\T3.1_level_sweep

Full command list: Revisions/V4/Response/Runbooks/RUN_T3.1.md
"""
import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.protocol_v5 import METRIC_COLUMNS, ProtocolV5Evaluator, summarize  # noqa: E402
from evaluation.sweep_methods import build_sweep_methods, wavelet_min_obs  # noqa: E402

LEVEL_GRID = {64: (2, 3, 4, 5), 32: (2, 3, 4)}
LEVEL_METHODS = ('WSPI', 'DTCWT+AF')
REF_T14 = Path('results/revision_v5/T1.5_causal_universe/T1.4_protocol_v5')
REF_T22 = Path('results/revision_v5/T2.2_window_sweep')


def _abs(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def grid_pairs(windows, levels=None):
    """(N, J) pairs of the grid that satisfy N >= 2**(J+1)."""
    out = []
    for N in windows:
        for J in (levels or LEVEL_GRID.get(N, (2, 3, 4, 5))):
            if N >= 2 ** (J + 1):
                out.append((N, J))
    return out


def run(a):
    ev = ProtocolV5Evaluator.from_csv(_abs(a.data), dataset_min_obs=a.min_obs, seed=a.seed,
                                      robustness=True, causal_universe=a.causal_universe)
    out = _abs(a.out)
    for N in a.windows:
        pairs = [p for p in grid_pairs([N], a.levels) if p[0] == N]
        for m in a.methods:
            mdir = out / f'W{N:03d}' / m
            res, rt, scorers = {}, {}, {}
            for _, J in pairs:
                fm = build_sweep_methods(N, level=J, names=[m])[m]
                df = ev.run_method(fm)
                rt[f'J{J}'] = df.attrs['duration_s']
                scorers[f'J{J}'] = fm.scorer.__name__
                df = df[df['window_id'] >= a.first_window].reset_index(drop=True)
                (mdir / 'protocol').mkdir(parents=True, exist_ok=True)
                df.to_csv(mdir / 'protocol' / f'J{J}_protocol.csv', index=False, encoding='utf-8')
                res[f'J{J}'] = df
            (mdir / 'comparison').mkdir(parents=True, exist_ok=True)
            summarize(res, common=False).to_csv(mdir / 'comparison' / 'summary_all_windows.csv',
                                                index=False, encoding='utf-8')
            sc = summarize(res, common=True)
            sc.to_csv(mdir / 'comparison' / 'summary_common_windows.csv', index=False,
                      encoding='utf-8')
            meta = dict(task='T3.1 level sweep (E2)', protocol='v5', mode='dense', method=m,
                        window=N, levels=[J for _, J in pairs], min_obs=wavelet_min_obs(N),
                        first_window_written=a.first_window, seed=a.seed, rsi_by_item=True,
                        causal_universe=a.causal_universe,
                        universe_rule=(('total count before the test slot >= %d'
                                        if a.causal_universe else
                                        'total count over the whole file >= %d') % a.min_obs),
                        padding='reflect (left) only while the series is shorter than N',
                        tie_rule='stable sort, ties by fixed item order (item ids sorted as str)',
                        horizon_slots=1, slot_minutes=ev.slot.total_seconds() / 60,
                        num_items=len(ev.items), num_slots=ev.num_slots + 1,
                        data=str(a.data), dataset_min_obs=a.min_obs, scorers=scorers,
                        runtime_seconds=rt,
                        versions=dict(python=platform.python_version(), numpy=np.__version__,
                                      pandas=pd.__version__),
                        host=platform.platform(), created=time.strftime('%Y-%m-%d %H:%M:%S'))
            (mdir / 'metadata').mkdir(parents=True, exist_ok=True)
            (mdir / 'metadata' / 'level_sweep_run.json').write_text(json.dumps(meta, indent=2),
                                                                    encoding='utf-8')
            print(f'\nN={N}  {m}  common windows: {int(sc["n_windows"].iloc[0])}  '
                  f'({sum(rt.values()):.0f} s)')
            print('level  ndcg@10  rho      rsi@10   robust')
            for _, r in sc.iterrows():
                print(f"{r['method']:<5}  {r['ndcg@10_mean']:.4f}  {r['spearman_rho_mean']:.4f}  "
                      f"{r['rsi@10_mean']:.4f}  {r['robustness_distortion_mean']:6.2f}")
    print(f'\nSaved to {out}')


def _control(scen: str, N: int, m: str, df: pd.DataFrame):
    if N == 64:
        ref = _abs(REF_T14) / scen / 'protocol' / f'{m}_protocol.csv'
    elif N == 32:
        ref = _abs(REF_T22) / scen / 'W032' / 'protocol' / f'{m}_protocol.csv'
    else:
        return None
    r = {'scenario': scen, 'window': N, 'method': m, 'reference': str(ref.relative_to(ROOT))
         if ref.is_relative_to(ROOT) else str(ref)}
    if not ref.exists():
        r.update(status='reference missing')
        return r
    b = pd.read_csv(ref)
    b = b[b['window_id'] >= int(df['window_id'].min())]
    a = df.set_index('window_id')
    b = b.set_index('window_id')
    same_ids = a.index.equals(b.index)
    r['n_windows'] = len(a)
    r['n_windows_ref'] = len(b)
    r['same_window_ids'] = bool(same_ids)
    worst = 0.0
    if same_ids:
        for c in METRIC_COLUMNS:
            if c in a and c in b and np.issubdtype(a[c].dtype, np.number):
                d = np.nanmax(np.abs(a[c].to_numpy(float) - b[c].to_numpy(float)))
                worst = max(worst, float(d) if np.isfinite(d) else 0.0)
    r['max_abs_diff'] = worst
    r['status'] = 'equal' if same_ids and worst < 1e-9 else 'DIFFERENT'
    return r


def collect(root: Path):
    rows, ctrl = [], []
    for scen in sorted(p for p in root.iterdir() if p.is_dir()):
        runs = {}
        for f in sorted(scen.glob('W[0-9][0-9][0-9]/*/protocol/J*_protocol.csv')):
            N = int(f.parts[-4][1:])
            m = f.parts[-3]
            J = int(f.name[1:-len('_protocol.csv')])
            runs[(N, m, J)] = pd.read_csv(f)
        if not runs:
            continue
        ids = set.intersection(*[set(d['window_id']) for d in runs.values()])
        for (N, m, J), d in sorted(runs.items()):
            if J == 3:
                c = _control(scen.name, N, m, d)
                if c:
                    ctrl.append(c)
            d = d[d['window_id'].isin(ids)]
            r = {'scenario': scen.name, 'window': N, 'method': m, 'level': J,
                 'n_windows': len(d), 'first_window': int(d['window_id'].min()),
                 'last_window': int(d['window_id'].max())}
            for c in METRIC_COLUMNS:
                r[f'{c}_mean'] = d[c].mean()
                r[f'{c}_sd'] = d[c].std()
            r['ties_top21_share'] = d['ties_top21'].mean()
            r['padded_share'] = d['padded'].mean()
            rows.append(r)
        print(f'{scen.name}: {len(runs)} runs, {len(ids)} common windows')
    pd.DataFrame(rows).to_csv(root / 'level_sweep_summary.csv', index=False, encoding='utf-8')
    cdf = pd.DataFrame(ctrl)
    cdf.to_csv(root / 'level_control.csv', index=False, encoding='utf-8')
    print(f'Saved {root / "level_sweep_summary.csv"}  ({len(rows)} rows)')
    if len(cdf):
        print(cdf[['scenario', 'window', 'method', 'status', 'max_abs_diff']].to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data')
    ap.add_argument('--min-obs', type=int, help='dataset min_observations (catalogue filter)')
    ap.add_argument('--out', help='scenario folder')
    ap.add_argument('--windows', nargs='*', type=int, default=sorted(LEVEL_GRID, reverse=True))
    ap.add_argument('--levels', nargs='*', type=int, default=None,
                    help='override the level grid (default: 64 -> 2..5, 32 -> 2..4)')
    ap.add_argument('--methods', nargs='*', default=list(LEVEL_METHODS))
    ap.add_argument('--first-window', type=int, default=32)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--causal-universe', action='store_true')
    ap.add_argument('--collect', help='sweep root: write level_sweep_summary.csv and level_control.csv')
    a = ap.parse_args()
    if a.collect:
        collect(_abs(a.collect))
    else:
        if not (a.data and a.min_obs is not None and a.out):
            ap.error('--data, --min-obs and --out are required')
        run(a)


if __name__ == '__main__':
    main()
