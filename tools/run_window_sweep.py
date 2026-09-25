r"""
Window-length sweep under protocol V5 (revision-srep-v5, task T2.2 / E1)
=======================================================================
For every N in the grid, all methods run with the same N-slot window:
the six baselines, three simple smoothers (SMA, EWMA-eq, Holt) and, for
N >= 16, the three wavelet-based methods with J = 3.  Settings: see
``evaluation/sweep_methods.py``.

Every N gets its own run folder, laid out like the T1.4 / T2.1 runs, so that
``tools/stats_report.py`` works on it unchanged:

  <out>/W<NNN>/protocol/<method>_protocol.csv   windows >= --first-window only
  <out>/W<NNN>/comparison/summary_{all,common}_windows.csv
  <out>/W<NNN>/metadata/protocol_v5_run.json

Only windows k >= --first-window (default 32) are written, so every N and
every method is compared on the same windows (decision of 25 Sep 2026).
The RSI of window 32 still compares with the top-K of window 31.

Examples (from the project root, Windows):

  python tools\run_window_sweep.py --data data\datasets\youtube_hourly.csv --min-obs 50 ^
         --causal-universe --out results\revision_v5\T2.2_window_sweep\youtube_hourly

  # one table over all N and methods (common windows of the whole scenario)
  python tools\run_window_sweep.py --collect results\revision_v5\T2.2_window_sweep

Full command list: Revisions/V4/Response/Runbooks/RUN_T2.2.md
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
from evaluation.sweep_methods import (HOLT_ALPHA, HOLT_GAMMA, HOLT_H,  # noqa: E402
                                      SWEEP_WINDOWS, build_sweep_methods)


def run_sweep(a):
    data = Path(a.data)
    data = data if data.is_absolute() else ROOT / data
    out = Path(a.out)
    out = out if out.is_absolute() else ROOT / out
    ev = ProtocolV5Evaluator.from_csv(data, dataset_min_obs=a.min_obs, seed=a.seed,
                                      robustness=True, causal_universe=a.causal_universe)
    for N in a.windows:
        methods = build_sweep_methods(N, level=a.level, names=a.methods)
        wdir = out / f'W{N:03d}'
        res, rt = {}, {}
        for name, fm in methods.items():
            df = ev.run_method(fm)
            rt[name] = df.attrs['duration_s']
            df = df[df['window_id'] >= a.first_window].reset_index(drop=True)
            (wdir / 'protocol').mkdir(parents=True, exist_ok=True)
            df.to_csv(wdir / 'protocol' / f'{name}_protocol.csv', index=False, encoding='utf-8')
            res[name] = df
        (wdir / 'comparison').mkdir(parents=True, exist_ok=True)
        summarize(res, common=False).to_csv(wdir / 'comparison' / 'summary_all_windows.csv',
                                            index=False, encoding='utf-8')
        sc = summarize(res, common=True)
        sc.to_csv(wdir / 'comparison' / 'summary_common_windows.csv', index=False,
                  encoding='utf-8')
        meta = dict(task='T2.2 window sweep', protocol='v5', mode='dense', window=N,
                    first_window_written=a.first_window, level=a.level, seed=a.seed,
                    rsi_by_item=True, causal_universe=a.causal_universe,
                    universe_rule=(('total count before the test slot >= %d' if a.causal_universe
                                    else 'total count over the whole file >= %d') % a.min_obs),
                    tie_rule='stable sort, ties by fixed item order (item ids sorted as str)',
                    holt=dict(alpha=HOLT_ALPHA, gamma=HOLT_GAMMA, horizon=HOLT_H),
                    ewma_eq_alpha=2.0 / (N + 1.0), horizon_slots=1,
                    slot_minutes=ev.slot.total_seconds() / 60, num_items=len(ev.items),
                    num_slots=ev.num_slots + 1, data=str(a.data), dataset_min_obs=a.min_obs,
                    methods={n: dict(window_slots=m.window_slots, min_obs=m.min_obs,
                                     scorer=m.scorer.__name__) for n, m in methods.items()},
                    runtime_seconds=rt,
                    versions=dict(python=platform.python_version(), numpy=np.__version__,
                                  pandas=pd.__version__),
                    host=platform.platform(), created=time.strftime('%Y-%m-%d %H:%M:%S'))
        (wdir / 'metadata').mkdir(parents=True, exist_ok=True)
        (wdir / 'metadata' / 'protocol_v5_run.json').write_text(json.dumps(meta, indent=2),
                                                                encoding='utf-8')
        print(f'\nN={N}  common windows: {int(sc["n_windows"].iloc[0])}  '
              f'({sum(rt.values()):.0f} s)')
        print('method        ndcg@10  rho      rsi@10   robust  ties')
        for _, r in sc.iterrows():
            print(f"{r['method']:<12}  {r['ndcg@10_mean']:.4f}  {r['spearman_rho_mean']:.4f}  "
                  f"{r['rsi@10_mean']:.4f}  {r['robustness_distortion_mean']:6.2f}  "
                  f"{r['ties_top21_share']:.3f}")
    print(f'\nSaved to {out}')


def collect(root: Path):
    """sweep_summary.csv: every scenario x N x method on the windows common to
    ALL (method, N) pairs of that scenario."""
    rows = []
    for scen in sorted(p for p in root.iterdir() if p.is_dir()):
        runs = {}
        for wdir in sorted(scen.glob('W[0-9][0-9][0-9]')):
            N = int(wdir.name[1:])
            for f in sorted((wdir / 'protocol').glob('*_protocol.csv')):
                runs[(N, f.name[:-len('_protocol.csv')])] = pd.read_csv(f)
        if not runs:
            continue
        ids = set.intersection(*[set(d['window_id']) for d in runs.values()])
        for (N, m), d in sorted(runs.items()):
            d = d[d['window_id'].isin(ids)]
            r = {'scenario': scen.name, 'window': N, 'method': m, 'n_windows': len(d),
                 'first_window': int(d['window_id'].min()), 'last_window': int(d['window_id'].max())}
            for c in METRIC_COLUMNS:
                r[f'{c}_mean'] = d[c].mean()
                r[f'{c}_sd'] = d[c].std()
            r['ties_top21_share'] = d['ties_top21'].mean()
            rows.append(r)
        print(f'{scen.name}: {len(runs)} runs, {len(ids)} common windows')
    out = pd.DataFrame(rows)
    out.to_csv(root / 'sweep_summary.csv', index=False, encoding='utf-8')
    print(f'Saved {root / "sweep_summary.csv"}  ({len(out)} rows)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data')
    ap.add_argument('--min-obs', type=int, help='dataset min_observations (catalogue filter)')
    ap.add_argument('--out', help='scenario folder; one W<NNN> subfolder per window length')
    ap.add_argument('--windows', nargs='*', type=int, default=list(SWEEP_WINDOWS))
    ap.add_argument('--methods', nargs='*', default=None)
    ap.add_argument('--level', type=int, default=3)
    ap.add_argument('--first-window', type=int, default=32)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--causal-universe', action='store_true')
    ap.add_argument('--collect', help='sweep root: write sweep_summary.csv')
    a = ap.parse_args()
    if a.collect:
        p = Path(a.collect)
        collect(p if p.is_absolute() else ROOT / p)
    else:
        if not (a.data and a.min_obs is not None and a.out):
            ap.error('--data, --min-obs and --out are required')
        run_sweep(a)


if __name__ == '__main__':
    main()
