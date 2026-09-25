r"""
Unit tests for the causal item catalogue of protocol V5 (T1.5).

    python tools\test_causal_universe.py

Prints one line per test and ends with ALL OK.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.protocol_v5 import ProtocolV5Evaluator, build_v5_methods  # noqa: E402

FAILED = []


def check(name, cond):
    print(f"{'OK  ' if cond else 'FAIL'} {name}")
    if not cond:
        FAILED.append(name)


def synthetic(n_slots=200, seed=0):
    """Five steady items plus item 'late': 1 request per slot in the first half
    (total 100 over the file, so it passes a whole-file filter of 60) but only
    reaches a running total of 60 at slot 60."""
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp('2025-01-01')
    rows = []
    for s in range(n_slots):
        ts = t0 + pd.Timedelta(hours=s)
        for i in range(5):
            rows.append((ts, f'steady{i}', int(rng.integers(20, 60))))
        if s < 100:
            rows.append((ts, 'late', 1))
        if s < 3:
            rows.append((ts, 'tiny', 1))           # total 3: never passes
    return pd.DataFrame(rows, columns=['timestamp', 'item_id', 'count'])


def run(df, causal, thr=60):
    ev = ProtocolV5Evaluator(df, dataset_min_obs=thr, seed=42, robustness=True,
                             causal_universe=causal, verbose=False)
    fm = build_v5_methods(names=['AF'])['AF']
    return ev, ev.run_method(fm).set_index('window_id')


if __name__ == '__main__':
    df = synthetic()
    ev_d, d = run(df, False)
    ev_c, c = run(df, True)
    check('default mode drops the item that never passes (whole-file filter)',
          'tiny' not in set(ev_d.items) and 'late' in set(ev_d.items))
    check('causal mode keeps every item of the file', set(ev_c.items) == set(df['item_id']))
    # window 30: running total of 'late' is 30 < 60 -> excluded in causal mode only
    check('default mode lets "late" in before it has 60 requests (look-ahead)',
          d.loc[30, 'num_items'] == 6)
    check('causal mode excludes "late" until its running total reaches 60',
          c.loc[30, 'num_items'] == 5 and c.loc[61, 'num_items'] == 6)
    check('"tiny" never enters in causal mode', int(c['num_items'].max()) == 6)
    # a causal run must not depend on data after the test slot
    T = 80
    _, ct = run(df[df['timestamp'] <= df['timestamp'].min() + pd.Timedelta(hours=T)], True)
    cols = ['num_items', 'ndcg@10', 'spearman_rho', 'rsi@10', 'robustness_distortion']
    same = ct.loc[ct.index <= T, cols].equals(c.loc[c.index <= T, cols])
    check('causal mode: truncated file gives identical windows (no look-ahead)', same)
    # the whole-file filter does depend on the future: cut before 'late' reaches 60
    T2 = 40
    _, dt = run(df[df['timestamp'] <= df['timestamp'].min() + pd.Timedelta(hours=T2)], False)
    check('default mode: truncated file changes windows (look-ahead is real)',
          not dt.loc[dt.index <= T2, 'num_items'].equals(d.loc[d.index <= T2, 'num_items']))
    # threshold 0: both modes are the same
    _, d0 = run(df, False, thr=0)
    _, c0 = run(df, True, thr=0)
    check('threshold 0: causal and default modes identical', d0.equals(c0))
    print('ALL OK' if not FAILED else f'{len(FAILED)} FAILED: {FAILED}')
    sys.exit(1 if FAILED else 0)
