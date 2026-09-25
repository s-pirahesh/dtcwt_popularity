r"""
Responsiveness experiment (revision-srep-v5, task T2.4 / E11)
=============================================================
Delay with which every method brings a genuine entry into its Top-10, plus
the stability-failure analysis and the graphic examples.  Design and
definitions: see ``evaluation/responsiveness.py``.

One call = one scenario, both configurations (default 7/64 and equal64):

  <out>/events.csv                      all entries (main + strict flag)
  <out>/<config>/delays.csv             event x method (delay, miss, coverage, eligibility)
  <out>/<config>/delay_summary.csv      per variant (main / strict) x method
  <out>/<config>/paired_tests.csv       WSPI vs every method, restricted delay
  <out>/<config>/window_rsi.csv         window-by-window RSI@10 of every method + truth
  <out>/<config>/rsi_failures.csv       windows where RSI@10 of WSPI is lower
  <out>/<config>/control.csv            NDCG@10 / RSI@10 vs the existing protocol run
  <out>/examples/example_events.csv     the two automatically chosen examples
  <out>/examples/example_traces.csv     count, true rank and rank of every method
  <out>/metadata/responsiveness_run.json

Examples (from the project root, Windows):

  python tools\run_responsiveness.py --data data\datasets\youtube_hourly.csv --min-obs 50 ^
         --causal-universe --scenario youtube_hourly --block 24 ^
         --control-default results\revision_v5\T1.5_causal_universe\T1.4_protocol_v5\youtube_hourly ^
         --control-equal64 results\revision_v5\T2.2_window_sweep\youtube_hourly\W064 ^
         --out results\revision_v5\T2.4_responsiveness\youtube_hourly

  # stack all scenarios into responsiveness_summary.csv, all_paired_tests.csv,
  # all_rsi_failures.csv, all_control.csv
  python tools\run_responsiveness.py --collect results\revision_v5\T2.4_responsiveness

Full command list: Revisions/V4/Response/Runbooks/RUN_T2.4.md
Unit test: tools/test_responsiveness.py
"""
import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.responsiveness import (CONFIGS, ResponsivenessEvaluator, build_config,  # noqa: E402
                                       paired_delays, rsi_failures, select_examples,
                                       summarize_delays)

TRACE_BEFORE, TRACE_AFTER = 24, 12


def absp(p):
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def control(win: dict, ref_dir, first_window: int) -> pd.DataFrame:
    rows = []
    for m, w in win.items():
        f = None if ref_dir is None else absp(ref_dir) / 'protocol' / f'{m}_protocol.csv'
        if f is None or not f.exists():
            rows.append({'method': m, 'reference_file': str(f), 'n_compared': 0,
                         'max_abs_diff_ndcg@10': np.nan, 'max_abs_diff_rsi@10': np.nan,
                         'match': False, 'note': 'reference missing'})
            continue
        r = pd.read_csv(f)[['window_id', 'ndcg@10', 'rsi@10']]
        j = w.merge(r, on='window_id', suffixes=('', '_ref'))
        j = j[j['window_id'] >= first_window]
        dn = (j['ndcg@10'] - j['ndcg@10_ref']).abs().max()
        dr = (j['rsi@10'] - j['rsi@10_ref']).abs().max()
        rows.append({'method': m, 'reference_file': str(f), 'n_compared': len(j),
                     'max_abs_diff_ndcg@10': dn, 'max_abs_diff_rsi@10': dr,
                     'match': bool(len(j) > 0 and dn < 1e-9 and dr < 1e-9), 'note': ''})
    return pd.DataFrame(rows)


def run(a):
    t_start = time.time()
    out = absp(a.out)
    ev_ = ResponsivenessEvaluator.from_csv(absp(a.data), dataset_min_obs=a.min_obs,
                                           causal_universe=a.causal_universe,
                                           first_window=a.first_window, seed=a.seed)
    events = ev_.find_events(L=a.L, P=a.P, pre=a.pre, pre_strict=a.pre_strict)
    out.mkdir(parents=True, exist_ok=True)
    events.to_csv(out / 'events.csv', index=False)
    print(f'events: {len(events)} (strict {int(events["strict"].sum()) if len(events) else 0}), '
          f'items {events["item_id"].nunique() if len(events) else 0}')
    truth = ev_.truth_rsi()
    all_dl, rt, methods_all = {}, {}, {}
    ctrl_dirs = {'default': a.control_default, 'equal64': a.control_equal64}
    for cfg in a.configs:
        methods = build_config(cfg, level=a.level)
        methods_all[cfg] = methods
        cdir = out / cfg
        cdir.mkdir(parents=True, exist_ok=True)
        win, parts = {}, []
        for name, fm in methods.items():
            t0 = time.time()
            top, elig, w = ev_.method_pass(fm)
            win[name] = w
            parts.append(ev_.delays(events, top, elig, name))
            rt[f'{cfg}/{name}'] = round(time.time() - t0, 1)
            print(f'  [{cfg}] {name:<12} W={fm.window_slots:<3} {rt[f"{cfg}/{name}"]:7.1f}s')
        dl = pd.concat(parts, ignore_index=True)
        dl.to_csv(cdir / 'delays.csv', index=False)
        all_dl[cfg] = dl
        pd.concat([summarize_delays(events, dl, v) for v in ('main', 'strict')]).to_csv(
            cdir / 'delay_summary.csv', index=False)
        pd.concat([paired_delays(events, dl, v, a.block, a.B, a.seed)
                   for v in ('main', 'strict')]).to_csv(cdir / 'paired_tests.csv', index=False)
        tab, fail = rsi_failures(win, truth, a.first_window)
        tab.to_csv(cdir / 'window_rsi.csv', index=False)
        fail.to_csv(cdir / 'rsi_failures.csv', index=False)
        c = control(win, ctrl_dirs.get(cfg), a.first_window)
        c.to_csv(cdir / 'control.csv', index=False)
        print(f'  [{cfg}] control: {int(c["match"].sum())} of {len(c)} methods match')
        s = summarize_delays(events, dl, 'main')
        print(s[['method', 'n_events', 'miss_rate', 'median_delay', 'mean_delay_restricted',
                 'share_delay_le1']].to_string(index=False, float_format='{:.3f}'.format))

    # ---- examples (rule fixed in select_examples: default config, main variant, WSPI vs AF)
    exd = out / 'examples'
    exd.mkdir(exist_ok=True)
    ex = pd.DataFrame()
    if 'default' in all_dl and len(events):
        ex = select_examples(events, all_dl['default'])
    if len(ex):
        ex.insert(0, 'scenario', a.scenario)
    ex.to_csv(exd / 'example_events.csv', index=False)
    tr = []
    for e in ex.itertuples(index=False):
        k_lo = max(0, e.t0 - TRACE_BEFORE)
        k_hi = min(ev_.nk - 1, e.t0 + e.run_len + TRACE_AFTER)
        base = pd.DataFrame({'window_id': np.arange(k_lo, k_hi + 1)})
        base['count'] = ev_.V[e.item_row, k_lo:k_hi + 1]
        rk = ev_.truth_rank[e.item_row, k_lo:k_hi + 1].astype(float)
        rk[rk >= np.iinfo(np.uint16).max] = np.nan
        base['truth_rank'] = rk
        for cfg, methods in methods_all.items():
            for name, fm in methods.items():
                r = ev_.method_ranks(fm, e.item_row, k_lo, k_hi)
                r = base.merge(r, on='window_id')
                r.insert(0, 'method', name)
                r.insert(0, 'config', cfg)
                r.insert(0, 't0', e.t0)
                r.insert(0, 'item_id', e.item_id)
                r.insert(0, 'example', e.example)
                r.insert(0, 'scenario', a.scenario)
                tr.append(r)
    (pd.concat(tr, ignore_index=True) if tr else pd.DataFrame()).to_csv(
        exd / 'example_traces.csv', index=False)

    md = out / 'metadata'
    md.mkdir(exist_ok=True)
    meta = dict(task='T2.4 (E11) responsiveness', scenario=a.scenario, data=str(a.data),
                dataset_min_obs=a.min_obs, causal_universe=a.causal_universe,
                event_rule=dict(L=a.L, P=a.P, pre=a.pre, pre_strict=a.pre_strict,
                                first_window=a.first_window, topk=10,
                                truth='real count of slot k, catalogue items, stable ties'),
                delay_rule='d = first k-t0 in [0, run_len) with the item in the method Top-10; '
                           'miss -> delay_restricted = run_len',
                configs={c: {n: dict(window_slots=m.window_slots, min_obs=m.min_obs,
                                     scorer=m.scorer.__name__) for n, m in ms.items()}
                         for c, ms in methods_all.items()},
                control_dirs=ctrl_dirs, block=a.block, B=a.B, seed=a.seed,
                stats='cluster bootstrap by t0 // block; Wilcoxon on cluster means; '
                      'Holm per scenario x config x variant',
                n_events=int(len(events)),
                n_events_strict=int(events['strict'].sum()) if len(events) else 0,
                slot_minutes=ev_.slot.total_seconds() / 60, num_items=len(ev_.items),
                num_slots=int(ev_.nk), runtime_seconds=rt,
                total_seconds=round(time.time() - t_start, 1),
                versions=dict(python=platform.python_version(), numpy=np.__version__,
                              pandas=pd.__version__),
                host=platform.platform(),
                date_utc=datetime.now(timezone.utc).isoformat(timespec='seconds'))
    (md / 'responsiveness_run.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print(f'\nSaved to {out}  ({meta["total_seconds"]} s)')


def collect(root: Path):
    names = {'delay_summary': 'responsiveness_summary', 'paired_tests': 'all_paired_tests',
             'rsi_failures': 'all_rsi_failures', 'control': 'all_control'}
    for src, dst in names.items():
        parts = []
        for scen in sorted(p for p in root.iterdir() if p.is_dir()):
            for cfg in CONFIGS:
                f = scen / cfg / f'{src}.csv'
                if f.exists():
                    d = pd.read_csv(f)
                    d.insert(0, 'config', cfg)
                    d.insert(0, 'scenario', scen.name)
                    parts.append(d)
        if parts:
            pd.concat(parts, ignore_index=True).to_csv(root / f'{dst}.csv', index=False)
            print(f'{root / (dst + ".csv")}: {len(parts)} files')
    ex = [pd.read_csv(f) for f in sorted(root.glob('*/examples/example_events.csv'))
          if f.stat().st_size > 1]
    if ex:
        pd.concat(ex, ignore_index=True).to_csv(root / 'all_example_events.csv', index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--data')
    ap.add_argument('--min-obs', type=int, help='dataset min_observations (catalogue filter)')
    ap.add_argument('--causal-universe', action='store_true')
    ap.add_argument('--scenario', help='name written in the CSVs (e.g. youtube_hourly)')
    ap.add_argument('--out')
    ap.add_argument('--configs', nargs='*', default=list(CONFIGS), choices=list(CONFIGS))
    ap.add_argument('--L', type=int, default=6, help='minimum stay in the true Top-10 (slots)')
    ap.add_argument('--P', type=int, default=6, help='slots outside the true Top-pre before t0')
    ap.add_argument('--pre', type=int, default=10)
    ap.add_argument('--pre-strict', type=int, default=20)
    ap.add_argument('--first-window', type=int, default=32)
    ap.add_argument('--level', type=int, default=3)
    ap.add_argument('--block', type=int, help='cluster length in slots (24 / 168 / 336 / 2016)')
    ap.add_argument('--B', type=int, default=10000)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--control-default', help='protocol run of the default config (T1.5 causal)')
    ap.add_argument('--control-equal64', help='protocol run of the equal-64 config (T2.2 W064)')
    ap.add_argument('--collect', help='T2.4 root: stack all scenarios')
    a = ap.parse_args()
    if a.collect:
        collect(absp(a.collect))
        return
    if not (a.data and a.min_obs is not None and a.out and a.block and a.scenario):
        ap.error('--data, --min-obs, --out, --block and --scenario are required')
    run(a)


if __name__ == '__main__':
    main()
