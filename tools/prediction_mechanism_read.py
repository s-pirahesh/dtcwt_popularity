r"""
Reading rule of the mechanism run (run E), fixed before the run
===============================================================
Reads only CSV files and writes one JSON file.  The rule is the dictionary RULE
below; it was fixed before run E was started.  The field ``rule_source`` of the
output names the design file of the study, which is not part of this repository.

Inputs (relative to results/prediction/):
  decision_D/test_gain_claims.csv            gain cells G of WSPI-H, development
  confirm/decision/test_gain_claims.csv      gain cells G of WSPI-H, confirmation
  stats_test_E/all_paired_tests.csv          reference WSPI-HL, development
  confirm/stats_test_E/all_paired_tests.csv  reference WSPI-HL, confirmation

  G = cells (scenario x horizon) where WSPI-H gains in Delta Rank against the
      best baseline of the full set (baseline_set 'A+X2', gain True).
  Verdicts are read from the reports with WSPI-HL as reference; the row of
  method WSPI-H with verdict 'ref_worse' means WSPI-H is significantly better
  than WSPI-HL.

  S+ = cells of G where WSPI-H is significantly better than WSPI-HL in Delta Rank
  S- = cells of G where WSPI-H is significantly worse
  state  1 'structural factor measured'  S+ >= 6 and S- == 0
         2 'level alone'                 S+ <= 2
         3 'partial'                     otherwise
  RSI@10 on all 30 cells: R- = cells where WSPI-H is significantly worse than
  WSPI-HL, R+ = better.  state  1 'top-ten cost measured' R- >= 15 and R+ <= 2;
  2 'no top-ten cost' R- <= 2; 3 'partial' otherwise.
  Reported only (no state): L+ = cells of G where WSPI-HL is significantly
  better than MA64 in Delta Rank, and the counts of S+ and S- on all 30 cells.

    python tools\prediction_mechanism_read.py
    python tools\prediction_mechanism_read.py --root results\prediction --out results\prediction\decision_E
"""
import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RULE = dict(G_metric='robustness_distortion', G_set='A+X2',
            S_state1=dict(S_plus_min=6, S_minus_max=0), S_state2=dict(S_plus_max=2),
            R_state1=dict(R_minus_min=15, R_plus_max=2), R_state2=dict(R_minus_max=2))
N_CELLS = 30


def gain_cells(f: Path):
    d = pd.read_csv(f)
    d = d[(d['baseline_set'] == RULE['G_set']) & (d['method'] == 'WSPI-H') &
          (d['metric'] == RULE['G_metric']) & (d['gain'].astype(str) == 'True')]
    return sorted((s, int(h)) for s, h in zip(d['scenario'], d['horizon']))


def verdicts(f: Path, method: str, metric: str):
    d = pd.read_csv(f)
    d = d[(d['reference'] == 'WSPI-HL') & (d['method'] == method) & (d['metric'] == metric)]
    return {(s, int(h)): v for s, h, v in zip(d['scenario'], d['horizon'], d['verdict'])}


def state_s(sp, sm):
    if sp >= RULE['S_state1']['S_plus_min'] and sm <= RULE['S_state1']['S_minus_max']:
        return 1
    if sp <= RULE['S_state2']['S_plus_max']:
        return 2
    return 3


def state_r(rm, rp):
    if rm >= RULE['R_state1']['R_minus_min'] and rp <= RULE['R_state1']['R_plus_max']:
        return 1
    if rm <= RULE['R_state2']['R_minus_max']:
        return 2
    return 3


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--root', default='results/prediction')
    ap.add_argument('--out', default=None, help='default: <root>/decision_E')
    a = ap.parse_args(argv)
    root = Path(a.root) if Path(a.root).is_absolute() else ROOT / a.root
    out = Path(a.out) if a.out else root / 'decision_E'
    out = out if out.is_absolute() else ROOT / out
    files = dict(g_dev=root / 'decision_D' / 'test_gain_claims.csv',
                 g_conf=root / 'confirm' / 'decision' / 'test_gain_claims.csv',
                 e_dev=root / 'stats_test_E' / 'all_paired_tests.csv',
                 e_conf=root / 'confirm' / 'stats_test_E' / 'all_paired_tests.csv')
    gone = [str(f) for f in files.values() if not f.exists()]
    if gone:
        print('missing: ' + ', '.join(gone))
        return 2
    G = {'development': gain_cells(files['g_dev']), 'confirmation': gain_cells(files['g_conf'])}
    rows, cells = [], {}
    for part, fe in (('development', files['e_dev']), ('confirmation', files['e_conf'])):
        dr_h = verdicts(fe, 'WSPI-H', 'robustness_distortion')
        rs_h = verdicts(fe, 'WSPI-H', 'rsi@10')
        dr_ma = verdicts(fe, 'MA64', 'robustness_distortion')
        cells[part] = sorted(dr_h)
        for c in sorted(dr_h):
            rows.append(dict(data=part, scenario=c[0], horizon=c[1], in_G=c in G[part],
                             drank_WSPI_H_vs_HL=dr_h[c], rsi_WSPI_H_vs_HL=rs_h.get(c),
                             drank_MA64_vs_HL=dr_ma.get(c)))
    n = sum(len(v) for v in cells.values())
    if n != N_CELLS or len(G['development']) + len(G['confirmation']) != 11:
        print(f'unexpected counts: {n} cells (want {N_CELLS}), '
              f'{len(G["development"])} + {len(G["confirmation"])} gain cells (want 8 + 3)')
        return 1
    t = pd.DataFrame(rows)
    g = t[t['in_G']]
    # verdicts are oriented with WSPI-HL as reference: 'ref_worse' = the method is better
    sp = int((g['drank_WSPI_H_vs_HL'] == 'ref_worse').sum())
    sm = int((g['drank_WSPI_H_vs_HL'] == 'ref_better').sum())
    lp = int((g['drank_MA64_vs_HL'] == 'ref_better').sum())
    rm = int((t['rsi_WSPI_H_vs_HL'] == 'ref_better').sum())
    rp = int((t['rsi_WSPI_H_vs_HL'] == 'ref_worse').sum())
    res = dict(rule=RULE, rule_source='docs/design/Mechanism_Design.md, section 4',
               G=dict((k, [f'{s}/h{h}' for s, h in v]) for k, v in G.items()),
               S_plus=sp, S_minus=sm, state_drank=state_s(sp, sm),
               S_plus_all30=int((t['drank_WSPI_H_vs_HL'] == 'ref_worse').sum()),
               S_minus_all30=int((t['drank_WSPI_H_vs_HL'] == 'ref_better').sum()),
               R_minus=rm, R_plus=rp, state_rsi=state_r(rm, rp), L_plus=lp,
               by_part={p: dict(S_plus=int((g[g.data == p]['drank_WSPI_H_vs_HL'] == 'ref_worse').sum()),
                                S_minus=int((g[g.data == p]['drank_WSPI_H_vs_HL'] == 'ref_better').sum()),
                                L_plus=int((g[g.data == p]['drank_MA64_vs_HL'] == 'ref_better').sum()),
                                R_minus=int((t[t.data == p]['rsi_WSPI_H_vs_HL'] == 'ref_better').sum()),
                                R_plus=int((t[t.data == p]['rsi_WSPI_H_vs_HL'] == 'ref_worse').sum()))
                        for p in ('development', 'confirmation')},
               inputs={k: str(v.relative_to(root).as_posix()) for k, v in files.items()},
               created=time.strftime('%Y-%m-%d %H:%M:%S'))
    out.mkdir(parents=True, exist_ok=True)
    t.to_csv(out / 'mechanism_cells.csv', index=False, encoding='utf-8')
    (out / 'mechanism.json').write_text(json.dumps(res, indent=2), encoding='utf-8')
    names = {1: 'structural factor measured', 2: 'level alone', 3: 'partial'}
    rnames = {1: 'top-ten cost measured', 2: 'no top-ten cost', 3: 'partial'}
    print(f'G: {len(G["development"])} development + {len(G["confirmation"])} confirmation cells')
    print(f'Delta Rank on G: S+ = {sp}, S- = {sm}  ->  state {res["state_drank"]} '
          f'({names[res["state_drank"]]})')
    print(f'RSI@10 on 30 cells: R- = {rm}, R+ = {rp}  ->  state {res["state_rsi"]} '
          f'({rnames[res["state_rsi"]]})')
    print(f'reported only: L+ = {lp} of 11; S+ / S- on 30 cells = '
          f'{res["S_plus_all30"]} / {res["S_minus_all30"]}')
    print(f'Saved to {out}')
    print('MECHANISM READ OK')
    return 0


if __name__ == '__main__':
    sys.exit(main())
