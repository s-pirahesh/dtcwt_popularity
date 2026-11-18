r"""
Unit checks for --block-rule of the paired tests (tools/run_prediction_stats.py)
==============================================================================
Uses the synthetic run folders of ``tools/test_prediction_stats_part.py``.

  1. Block lengths: the default rule is b_h = max(b, h) for every scenario and
     horizon (development and confirmation data); ``window`` gives
     b_h = max(b, N + h - 1) with N = 64; the window length equals the one of
     tools/prediction_protocol.py.
  2. Default unchanged: a test-part run without the option writes the same
     reports as before (block max(b, h)) and the plan names that rule.
  3. Window rule: --block-rule window --out <other folder> writes one report
     per cell; every report uses the window block; a scenario whose base block
     is already longer than N + h - 1 gives exactly the same paired tests as the
     default run; the others have fewer blocks; the plan names the window rule.
  4. The default reports are not touched by the window run (same bytes).
  5. Refusals: --block-rule window without --out, or with an --out folder that
     holds reports of the default rule.

    python tools/test_prediction_stats_block_rule.py
"""
import hashlib
import json
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools import run_prediction_stats as S                               # noqa: E402
from tools import test_prediction_stats_part as T                         # noqa: E402

FAILED = []
B_TEST = 200
LONG = 100                       # base block longer than N + 24 - 1 = 87


def check(name, ok, detail=''):
    print(f"{'OK  ' if ok else 'FAIL'} {name} {detail}")
    if not ok:
        FAILED.append(name)
    return ok


def digest(folder: Path) -> dict:
    return {f.relative_to(folder).as_posix(): hashlib.sha256(f.read_bytes()).hexdigest()
            for f in sorted(folder.rglob('*')) if f.is_file() and f.name != 'stats_plan.json'}


def refused(argv) -> bool:
    try:
        rc = S.main(argv)
    except SystemExit as e:
        return e.code == 2
    return rc == 2


def block_checks(real):
    text = (ROOT / 'tools' / 'prediction_protocol.py').read_text(encoding='utf-8')
    m = re.search(r'^WINDOW = (\d+)', text, re.M)
    check('1 window length equals tools/prediction_protocol.py', m and int(m.group(1)) == S.WINDOW,
          str(m.group(1) if m else None))
    saved = S.BLOCK_RULE
    try:
        S.BLOCK_RULE = 'max'
        ok = all(S.block_length(s, h) == max(b, h) for s, b in real.items() for h in S.HORIZONS)
        ok &= all(S.block_length(s, h, b) == max(b, h)
                  for s, b in S.CONFIRM_BLOCK.items() for h in S.HORIZONS)
        check('1 default rule max(b, h)', ok)
        S.BLOCK_RULE = 'window'
        ok = all(S.block_length(s, h) == max(b, 64 + h - 1) for s, b in real.items()
                 for h in S.HORIZONS)
        ok &= all(S.block_length(s, h, b) == max(b, 64 + h - 1)
                  for s, b in S.CONFIRM_BLOCK.items() for h in S.HORIZONS)
        check('1 window rule max(b, N + h - 1)', ok)
        S.BASE_BLOCK.update(real)
        got = {(s, h): S.block_length(s, h) for s in ('youtube_hourly', 'taxi_hourly',
                                                       'movielens_weekly') for h in S.HORIZONS}
        want = {('youtube_hourly', 1): 64, ('youtube_hourly', 8): 71, ('youtube_hourly', 24): 87,
                ('taxi_hourly', 1): 168, ('taxi_hourly', 8): 168, ('taxi_hourly', 24): 168,
                ('movielens_weekly', 1): 64, ('movielens_weekly', 8): 71,
                ('movielens_weekly', 24): 87}
        check('1 window rule, example lengths', got == want, str(got))
    finally:
        S.BLOCK_RULE = saved


def run_tests():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / 'p2'
        T.build(root)
        S.main(['--root', str(root), '--part', 'tune', '--runs', 'C', '--B', str(B_TEST),
                '--allow-missing'])
        S.main(['--root', str(root), '--part', 'test', '--runs', 'C', '--B', str(B_TEST)])
        test_dir = root / 'stats_test'
        plan = json.loads((test_dir / 'stats_plan.json').read_text())
        check('2 default plan names max(b, h)', plan.get('block_rule') == S.BLOCK_RULES['max'],
              str(plan.get('block_rule')))
        dflt = pd.read_csv(test_dir / 'all_paired_tests.csv', float_precision='round_trip')
        ok = all(int(b) == max(S.BASE_BLOCK[s], int(h))
                 for s, h, b in zip(dflt.scenario, dflt.horizon, dflt.block))
        check('2 default reports use max(b, h)', ok)
        before = digest(test_dir)

        wdir = root / 'stats_test_window'
        rc = S.main(['--root', str(root), '--part', 'test', '--runs', 'C', '--B', str(B_TEST),
                     '--block-rule', 'window', '--out', str(wdir)])
        check('3 window run exit code 1 only because B is not 10000', rc == 1, str(rc))
        check('3 module rule reset to the option', S.BLOCK_RULE == 'window')
        plan = json.loads((wdir / 'stats_plan.json').read_text())
        check('3 plan names the window rule', plan.get('block_rule') == S.BLOCK_RULES['window'],
              str(plan.get('block_rule')))
        w = pd.read_csv(wdir / 'all_paired_tests.csv', float_precision='round_trip')
        check('3 one report per cell and reference', w.groupby(['scenario', 'horizon', 'reference'])
              .ngroups == dflt.groupby(['scenario', 'horizon', 'reference']).ngroups)
        ok = all(int(b) == max(S.BASE_BLOCK[s], S.WINDOW + int(h) - 1)
                 for s, h, b in zip(w.scenario, w.horizon, w.block))
        check('3 every report uses the window block', ok)
        long_s = T.SCEN[0]
        key = ['scenario', 'horizon', 'reference', 'method', 'metric']
        a = dflt[dflt.scenario == long_s].sort_values(key).reset_index(drop=True)
        b = w[w.scenario == long_s].sort_values(key).reset_index(drop=True)
        check('3 long-block scenario: identical paired tests', a.equals(b), f'{len(a)} {len(b)}')
        short = [s for s in T.SCEN if s != long_s]
        nb_d = dflt[dflt.scenario.isin(short)].groupby(['scenario', 'horizon']).n_blocks.max()
        nb_w = w[w.scenario.isin(short)].groupby(['scenario', 'horizon']).n_blocks.max()
        check('3 short-block scenarios have fewer blocks', bool((nb_w < nb_d).all()),
              f'{nb_d.tolist()} {nb_w.tolist()}')
        check('4 default reports unchanged', digest(test_dir) == before)

        S.main(['--root', str(root), '--part', 'test', '--runs', 'C', '--B', str(B_TEST)])
        check('4 a later default call uses max again', S.BLOCK_RULE == 'max')
        check('5 refused without --out', refused(['--root', str(root), '--part', 'test', '--runs',
                                                  'C', '--B', str(B_TEST), '--block-rule',
                                                  'window']))
        check('5 refused with the default folder', refused(['--root', str(root), '--part', 'test',
                                                            '--runs', 'C', '--B', str(B_TEST),
                                                            '--block-rule', 'window', '--out',
                                                            str(test_dir)]))
        check('4 default reports unchanged after the refusals', digest(test_dir) == before)


def main():
    real = dict(S.BASE_BLOCK)
    block_checks(real)
    for s in S.BASE_BLOCK:
        S.BASE_BLOCK[s] = 4                       # short blocks for the small test series
    S.BASE_BLOCK[T.SCEN[0]] = LONG                # one scenario longer than N + h - 1
    try:
        run_tests()
    finally:
        S.BASE_BLOCK.update(real)
        S.BLOCK_RULE = 'max'
    print()
    if FAILED:
        print(f'FAILED: {len(FAILED)}')
        for f in FAILED:
            print(f'  {f}')
        sys.exit(1)
    print('ALL OK')


if __name__ == '__main__':
    main()
