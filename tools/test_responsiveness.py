r"""
Unit test for evaluation/responsiveness.py and tools/run_responsiveness.py (T2.4 / E11)
======================================================================================
Synthetic data with planted entries; no real dataset is needed.

  1. event detection: planted entry found with the right t0, run length and strict flag
  2. delay of AF (W=7) = 1; RRD(64) is slower than AF(7)
  3. a method that never ranks the item: miss, delay_restricted = run length
  4. an item without history: ineligible for WSPI (min_obs 32), counted in n_ineligible_run
  5. NDCG@10 and RSI@10 of method_pass equal ProtocolV5Evaluator.run_method
     (all 9 methods, default and equal64 configurations, causal catalogue)
  6. paired test: identical delays -> difference 0, p = 1; Holm monotone
  7. example selection: two distinct events, rule applied
  8. method_ranks agrees with the stored Top-10
  9. end-to-end run of tools/run_responsiveness.py (both configs) + --collect

Run from the project root:  python tools\test_responsiveness.py
"""
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.fast_evaluator import FastMethod  # noqa: E402
from evaluation.protocol_v5 import ProtocolV5Evaluator  # noqa: E402
from evaluation.responsiveness import (ResponsivenessEvaluator, build_config, holm,  # noqa: E402
                                       paired_delays, rsi_failures, select_examples)

N_ITEMS, N_SLOTS = 40, 400
SURGE, T_SURGE, RUN = 'i30', 200, 20          # item with a planted entry
NEW, T_NEW = 'i35', 250                      # item without history
MIN_OBS = 10
FAILS = []


def check(name, cond, info=''):
    print(f"{'OK  ' if cond else 'FAIL'} {name} {info}")
    if not cond:
        FAILS.append(name)


def make_data():
    rng = np.random.default_rng(7)
    start = pd.Timestamp('2025-01-01')
    rows = []
    for j in range(N_ITEMS):
        iid = f'i{j:02d}'
        level = 60.0 - 1.2 * j                  # i00 .. i09 are the usual Top-10
        for t in range(N_SLOTS):
            if iid == NEW and t < T_NEW:
                continue
            c = rng.poisson(level)
            if iid == SURGE and T_SURGE <= t < T_SURGE + RUN:
                c = 400 + rng.poisson(20)
            if iid == NEW and t >= T_NEW:
                c = 300 + rng.poisson(10)
            if c > 0:
                rows.append((start + pd.Timedelta(hours=t), iid, int(c)))
    return pd.DataFrame(rows, columns=['timestamp', 'item_id', 'count'])


def main():
    df = make_data()
    ev = ResponsivenessEvaluator(df.copy(), MIN_OBS, causal_universe=True, verbose=False)
    events = ev.find_events(L=6, P=6, pre=10, pre_strict=20)
    s = events[events['item_id'] == SURGE]
    check('1 planted entry found', len(s) == 1 and int(s['t0'].iloc[0]) == T_SURGE,
          f"t0={s['t0'].tolist()}")
    check('1 run length', len(s) == 1 and int(s['run_len'].iloc[0]) == RUN,
          f"run={s['run_len'].tolist()}")
    check('1 strict flag', len(s) == 1 and bool(s['strict'].iloc[0]))
    check('1 no event before first window', (events['t0'] >= 32).all())

    cfgs = {c: build_config(c) for c in ('default', 'equal64')}
    dls, wins = {}, {}
    for c, methods in cfgs.items():
        parts, win = [], {}
        for n, fm in methods.items():
            top, elig, w = ev.method_pass(fm)
            win[n] = w
            parts.append(ev.delays(events, top, elig, n))
            if c == 'default' and n == 'WSPI':
                wspi_top = top
        dls[c] = pd.concat(parts, ignore_index=True)
        wins[c] = win

    def delay(c, m, item):
        eid = events.loc[events['item_id'] == item, 'event_id'].iloc[0]
        d = dls[c]
        return d[(d['event_id'] == eid) & (d['method'] == m)].iloc[0]

    af = delay('default', 'AF', SURGE)
    rrd64 = delay('equal64', 'RRD', SURGE)
    check('2 AF(7) delay = 1', af['detected'] and af['delay'] == 1, f"d={af['delay']}")
    check('2 RRD(64) slower than AF(7)', rrd64['delay_restricted'] > af['delay_restricted'],
          f"RRD64={rrd64['delay_restricted']} AF={af['delay_restricted']}")

    const = FastMethod('CONST', 7, 3, lambda X: np.zeros(X.shape[0]))
    top, elig, _ = ev.method_pass(const)
    dc = ev.delays(events, top, elig, 'CONST')
    r = dc[dc['event_id'] == s['event_id'].iloc[0]].iloc[0]
    check('3 constant scorer: miss', (not r['detected']) and r['delay_restricted'] == RUN)

    nw = events[events['item_id'] == NEW]
    check('4 new item entry found', len(nw) >= 1, f"t0={nw['t0'].tolist()}")
    if len(nw):
        w = delay('default', 'WSPI', NEW)
        a = delay('default', 'AF', NEW)
        check('4 WSPI ineligible without history', (not w['eligible_at_t0'])
              and w['n_ineligible_run'] > 0, f"n_inel={w['n_ineligible_run']}")
        check('4 AF eligible sooner', a['n_ineligible_run'] < w['n_ineligible_run'])

    # ---- 5. same NDCG@10 / RSI@10 as the protocol evaluator
    pv = ProtocolV5Evaluator(df.copy(), MIN_OBS, robustness=False, causal_universe=True,
                             verbose=False)
    worst = 0.0
    for c, methods in cfgs.items():
        for n, fm in methods.items():
            ref = pv.run_method(fm)[['window_id', 'ndcg@10', 'rsi@10']]
            j = wins[c][n].merge(ref, on='window_id', suffixes=('', '_r'))
            same_ids = len(j) == len(ref) == len(wins[c][n])
            dd = max((j['ndcg@10'] - j['ndcg@10_r']).abs().max(),
                     (j['rsi@10'] - j['rsi@10_r']).abs().max())
            worst = max(worst, dd if same_ids else np.inf)
    check('5 method_pass == run_method (18 runs)', worst < 1e-12, f'max diff {worst:g}')

    # ---- 6. paired test sanity
    d0 = dls['default'].copy()
    d0.loc[d0['method'] == 'AF', 'delay_restricted'] = \
        d0.loc[d0['method'] == 'WSPI', 'delay_restricted'].to_numpy()
    p = paired_delays(events, d0, 'main', block=24, B=500, seed=42)
    r = p[p['method'] == 'AF'].iloc[0]
    check('6 identical delays: diff 0, p 1', r['diff_ref_minus_method'] == 0 and r['p_cluster'] == 1.0)
    h = holm([0.01, 0.04, 0.03, np.nan])
    check('6 Holm', np.allclose(h[:3], [0.03, 0.06, 0.06]) and np.isnan(h[3]), str(h))

    # ---- 7. examples
    ex = select_examples(events, dls['default'])
    check('7 two distinct examples', len(ex) == 2 and ex['event_id'].nunique() == 2,
          ex[['example', 'event_id', 'diff']].to_dict('records').__str__())
    check('7 worst >= typical', ex['diff'].iloc[1] >= ex['diff'].iloc[0])

    # ---- 8. method_ranks vs stored Top-10
    fm = cfgs['default']['WSPI']
    i = int(s['item_row'].iloc[0])
    rk = ev.method_ranks(fm, i, T_SURGE - 3, T_SURGE + 5)
    ok = all(((r['rank'] <= 10) == bool((wspi_top[r['window_id']] == i).any()))
             for _, r in rk.iterrows())
    check('8 method_ranks consistent with Top-10', ok)

    tab, fail = rsi_failures(wins['default'], ev.truth_rsi(), 32)
    check('8 truth RSI in [0, 1]', tab['truth_rsi@10'].between(0, 1).all())

    # ---- 9. end to end
    import tools.run_responsiveness as rr
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        df.to_csv(tmp / 'syn.csv', index=False)
        a = types.SimpleNamespace(data=str(tmp / 'syn.csv'), min_obs=MIN_OBS, causal_universe=True,
                                  scenario='syn', out=str(tmp / 'T2.4' / 'syn'),
                                  configs=['default', 'equal64'], L=6, P=6, pre=10,
                                  pre_strict=20, first_window=32, level=3, block=24, B=300,
                                  seed=42, control_default=None, control_equal64=None)
        rr.run(a)
        rr.collect(tmp / 'T2.4')
        need = ['events.csv', 'default/delays.csv', 'equal64/paired_tests.csv',
                'default/window_rsi.csv', 'examples/example_traces.csv',
                'metadata/responsiveness_run.json']
        check('9 outputs written', all((tmp / 'T2.4' / 'syn' / f).exists() for f in need))
        check('9 collect', (tmp / 'T2.4' / 'responsiveness_summary.csv').exists())
        t = pd.read_csv(tmp / 'T2.4' / 'syn' / 'examples' / 'example_traces.csv')
        check('9 traces: 2 examples x 18 methods', t.groupby(['example', 'config', 'method'])
              .ngroups == 36)

    print(f'\n{"ALL OK" if not FAILS else "FAILED: " + ", ".join(FAILS)}')
    sys.exit(1 if FAILS else 0)


if __name__ == '__main__':
    main()
