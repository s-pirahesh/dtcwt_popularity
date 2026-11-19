r"""
Unit checks for the paired-test plan and the decision rules
===========================================================
Covers ``tools/run_prediction_stats.py`` and ``tools/prediction_decision.py``
on synthetic run folders with known differences between the methods.

  1. Block length max(b, h) with the registered base blocks.
  2. Window check: equal windows pass; a method with a missing window, a
     different item count or an extra method fails and gets no tests; the WSPI
     file of run B must equal the one of run A.
  3. The plan writes one report per reference (run A: WSPI-FG, WSPI-F2, WSPI;
     run B: WSPI-FA) with the block of the scenario and horizon, and the
     stacked tables; the verdicts equal a direct call of stats_report.
  4. Decision rules on constructed cases: Pareto success needs one better and
     none worse; 4 of 6 scenarios; 2 of 3 horizons -> A, otherwise B; a missing
     cell -> UNDETERMINED.  Gain claims use the best traditional baseline by
     mean (lower is better for DeltaRank).  Domain effect and bound violations
     are listed in the negative results.

    python tools/test_prediction_stats.py
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.fast_evaluator import PROTOCOL_COLUMNS                    # noqa: E402
from evaluation.protocol_v5 import V5_EXTRA_COLUMNS                       # noqa: E402
from tools import prediction_decision as D                                # noqa: E402
from tools import run_prediction_stats as S                               # noqa: E402
from tools import stats_report                                            # noqa: E402

FAILED = []
N_WIN = 960
TEST_BLOCK = 4
MAIN = D.MAIN_METRICS
COLUMNS = list(PROTOCOL_COLUMNS) + list(V5_EXTRA_COLUMNS)


def check(name, ok, detail=''):
    print(f"{'OK  ' if ok else 'FAIL'} {name} {detail}")
    if not ok:
        FAILED.append(name)
    return ok


def offsets(run, scenario, h):
    """Constructed differences (added to the shared base series)."""
    i = S.RUNS['A']['scenarios'].index(scenario)
    off = {}
    if run == 'A':
        n_good = {1: 4, 8: 4, 24: 3}[h]                # scenarios where FG beats F2 on RSI
        if i < n_good:
            off[('WSPI-FG', 'rsi@10')] = 0.05
        if i == 4 and h == 1:                           # better on RSI but worse on NDCG
            off[('WSPI-FG', 'rsi@10')] = 0.05
            off[('WSPI-FG', 'ndcg@10')] = -0.05
        off[('MA64', 'ndcg@10')] = 0.02                 # best baseline on NDCG
        off[('ARYW', 'robustness_distortion')] = -3.0   # best baseline on DeltaRank
        off[('WSPI', 'ndcg@10')] = 0.06                 # WSPI gains on NDCG
        off[('WSPI-F', 'robustness_distortion')] = 5.0  # NLMS worse than WSPI
    else:
        off[('ARIMA', 'spearman_rho')] = 0.04           # raw ARIMA better on rho
    return off


def write_run(root: Path, run: str, scenario: str, h: int, seed=0, drop=None):
    rng = np.random.default_rng([seed, h, S.RUNS['A']['scenarios'].index(scenario)])
    base = {c: rng.normal(0.8 if c != 'robustness_distortion' else 20.0, 0.02, N_WIN) for c in COLUMNS}
    off = offsets(run, scenario, h)
    d = root / f'run_{run}' / scenario / f'h{h}'
    (d / 'protocol').mkdir(parents=True, exist_ok=True)
    (d / 'metadata').mkdir(exist_ok=True)
    for m in S.RUNS[run]['methods']:
        if m == drop:
            continue
        df = pd.DataFrame({c: base[c].copy() for c in COLUMNS})
        df['window_id'] = np.arange(64, 64 + N_WIN)
        df['timestamp'] = df['window_id'] * 3600000
        df['method'] = m
        df['num_items'] = 50
        df['ties_top21'] = False
        df['padded'] = False
        df['n_nonfinite'] = 0
        for (mm, c), v in off.items():
            if mm == m:
                df[c] = df[c] + v + rng.normal(0, 0.001, N_WIN)
        df[COLUMNS].to_csv(d / 'protocol' / f'{m}_protocol.csv', index=False)
    (d / 'metadata' / 'protocol_v5_run.json').write_text(json.dumps(dict(horizon=h)))
    if run == 'B':
        pd.DataFrame([dict(method='WSPI', scorer_seconds=1.0, fallback_share=np.nan),
                      dict(method='WSPI-FA', scorer_seconds=80.0, fallback_share=0.01),
                      dict(method='ARIMA', scorer_seconds=100.0, fallback_share=np.nan)]
                     ).to_csv(d / 'scorer_timing.csv', index=False)
    pd.DataFrame([dict(horizon=h, method='WSPI-FA' if run == 'B' else 'WSPI-FG', lower_bound=0.5,
                       upper_bound=4.5, min_ratio=0.6, max_ratio=2.0, status='OK')]
                 ).to_csv(d / 'bound_check.csv', index=False)


def build(root: Path):
    for run, spec in S.RUNS.items():
        for s in spec['scenarios']:
            for h in S.HORIZONS:
                write_run(root, run, s, h)
    # run B shares the WSPI file with run A
    for s in S.RUNS['B']['scenarios']:
        for h in S.HORIZONS:
            shutil.copy(root / 'run_A' / s / f'h{h}' / 'protocol' / 'WSPI_protocol.csv',
                        root / 'run_B' / s / f'h{h}' / 'protocol' / 'WSPI_protocol.csv')


def main():
    reg = dict(S.BASE_BLOCK)
    check('registered base blocks',
          reg == {'youtube_hourly': 24, 'taxi_hourly': 168, 'taxi_30min': 336, 'taxi_5min': 2016,
                  'movielens_daily': 7, 'movielens_weekly': 4})
    check('block = max(b, h)',
          [S.block_length('movielens_daily', h) for h in (1, 8, 24)] == [7, 8, 24]
          and [S.block_length('movielens_weekly', h) for h in (1, 8, 24)] == [4, 8, 24]
          and [S.block_length('youtube_hourly', h) for h in (1, 8, 24)] == [24, 24, 24]
          and S.block_length('taxi_5min', 24) == 2016)
    check('references and bootstrap of the plan',
          S.RUNS['A']['references'] == ('WSPI-FG', 'WSPI-F2', 'WSPI')
          and S.RUNS['B']['references'] == ('WSPI-FA',) and S.N_BOOT == 10000 and S.SEED == 42)
    check('decision constants', D.MIN_SCENARIOS == 4 and D.MIN_HORIZONS == 2
          and D.BASELINES == ('Persistence', 'Holt', 'ARYW', 'MA64')
          and D.MAIN_METRICS == ('ndcg@10', 'spearman_rho', 'rsi@10', 'robustness_distortion'))

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / 'prediction'
        build(root)
        for k in S.BASE_BLOCK:                         # small blocks for the synthetic data
            S.BASE_BLOCK[k] = TEST_BLOCK
        try:
            rc = S.main(['--root', str(root), '--B', '200'])
            plan = json.loads((root / 'stats' / 'stats_plan.json').read_text())
            check('plan: 18 x 3 + 6 reports', len(plan['reports']) == 60 and not plan['missing'],
                  f"({len(plan['reports'])})")
            check('exit code 1 only because B is not the registered value', rc == 1)
            blocks = {(r['scenario'], r['horizon']): r['block'] for r in plan['reports']}
            check('block of every report is max(base, h)',
                  all(b == max(TEST_BLOCK, h) for (s, h), b in blocks.items()))
            wc = pd.read_csv(root / 'stats' / 'window_check.csv')
            check('window check clean, with the cross-run rows',
                  (wc['status'] == 'OK').all() and (wc['run'] == 'A=B').sum() == 6)
            paired = pd.read_csv(root / 'stats' / 'all_paired_tests.csv')
            direct = root / 'direct'
            _, p0, _ = stats_report.analyse(root / 'run_A' / 'taxi_30min' / 'h8', direct, 8, None,
                                            'WSPI-FG', 200, 42, None)
            p1 = paired[(paired['run'] == 'A') & (paired['scenario'] == 'taxi_30min')
                        & (paired['horizon'] == 8) & (paired['reference'] == 'WSPI-FG')]
            check('stacked table equals a direct call of stats_report',
                  list(p0['verdict']) == list(p1['verdict'])
                  and np.allclose(p0['p_block_holm'], p1['p_block_holm'], equal_nan=True))

            # ---- decision: constructed case is narrative A -------------------
            rc = D.main(['--root', str(root)])
            dec = json.loads((root / 'decision' / 'decision.json').read_text())
            cells = pd.read_csv(root / 'decision' / 'pareto_cells.csv')
            check('narrative A: horizons 1 and 8 succeed (4 of 6), horizon 24 fails (3 of 6)',
                  rc == 0 and dec['narrative'] == 'A' and dec['successful_horizons'] == [1, 8]
                  and dec['successful_cells'] == 11, f"({dec['successful_cells']} cells)")
            c = cells[(cells['scenario'] == 'movielens_daily') & (cells['horizon'] == 1)].iloc[0]
            check('better on one metric and worse on another is not a success',
                  c['n_better'] == 1 and c['n_worse'] == 1 and not c['success'])
            gc = pd.read_csv(root / 'decision' / 'gain_claims.csv')
            g = gc[(gc['scenario'] == 'taxi_hourly') & (gc['horizon'] == 8)]
            best = g.set_index(['metric', 'method'])['best_baseline']
            check('best baseline by mean: MA64 on NDCG, ARYW on DeltaRank (lower is better)',
                  best[('ndcg@10', 'WSPI')] == 'MA64'
                  and best[('robustness_distortion', 'WSPI')] == 'ARYW')
            gg = g.set_index(['metric', 'method'])
            check('gain only where the method is significantly better than that baseline',
                  bool(gg.loc[('ndcg@10', 'WSPI'), 'gain'])
                  and not bool(gg.loc[('ndcg@10', 'WSPI-F2'), 'gain'])
                  and gg.loc[('ndcg@10', 'WSPI-F2'), 'verdict'] == 'ref_worse'
                  and len(gc) == 6 * 3 * 4 * 3)
            dom = pd.read_csv(root / 'decision' / 'domain_effect.csv')
            r = dom[(dom['metric'] == 'spearman_rho')].iloc[0]
            check('domain effect: verdict, time ratio and fallback share',
                  (dom[dom['metric'] == 'spearman_rho']['verdict'] == 'ref_worse').all()
                  and abs(r['time_ratio_fa_to_arima'] - 0.8) < 1e-12 and r['fallback_share'] == 0.01
                  and len(dom) == 24)
            neg = pd.read_csv(root / 'decision' / 'negative_results.csv')
            kinds = neg['kind'].value_counts()
            check('negative results listed by kind',
                  kinds.get('gate_worse_than_structural', 0) == 1
                  and kinds.get('forecast_worse_than_wspi', 0) >= 18
                  and kinds.get('domain_worse_than_raw', 0) == 6
                  and kinds.get('worse_than_best_baseline', 0) >= 36
                  and 'bound_violation' not in kinds, str(dict(kinds)))

            # ---- narrative B: one more failing cell at horizon 8 ----------------
            f = root / 'stats' / 'all_paired_tests.csv'
            keep = f.read_bytes()
            p = pd.read_csv(f)
            sel = ((p['run'] == 'A') & (p['scenario'] == 'youtube_hourly') & (p['horizon'] == 8)
                   & (p['reference'] == 'WSPI-FG') & (p['method'] == 'WSPI-F2')
                   & (p['metric'] == 'rsi@10'))
            p.loc[sel, 'verdict'] = 'n.s.'
            p.to_csv(f, index=False)
            rc = D.main(['--root', str(root)])
            dec = json.loads((root / 'decision' / 'decision.json').read_text())
            check('narrative B when only one horizon succeeds',
                  rc == 0 and dec['narrative'] == 'B' and dec['successful_horizons'] == [1])
            p = p[~((p['scenario'] == 'taxi_5min') & (p['horizon'] == 24))]
            p.to_csv(f, index=False)
            rc = D.main(['--root', str(root)])
            dec = json.loads((root / 'decision' / 'decision.json').read_text())
            check('a missing cell gives UNDETERMINED and exit code 1',
                  rc == 1 and dec['narrative'] == 'UNDETERMINED' and dec['complete_cells'] == 17)
            f.write_bytes(keep)
            bf = root / 'run_A' / 'taxi_5min' / 'h24' / 'bound_check.csv'
            b = pd.read_csv(bf)
            b['status'] = 'VIOLATION'
            b.to_csv(bf, index=False)
            D.main(['--root', str(root)])
            neg = pd.read_csv(root / 'decision' / 'negative_results.csv')
            check('a bound violation is listed', (neg['kind'] == 'bound_violation').sum() == 1)

            # ---- window check failures ----------------------------------------
            pf = root / 'run_A' / 'taxi_hourly' / 'h1' / 'protocol'
            d = pd.read_csv(pf / 'Holt_protocol.csv')
            d.iloc[:-1].to_csv(pf / 'Holt_protocol.csv', index=False)
            d2 = pd.read_csv(root / 'run_A' / 'taxi_30min' / 'h1' / 'protocol' / 'ARYW_protocol.csv')
            d2.loc[3, 'num_items'] = 49
            d2.to_csv(root / 'run_A' / 'taxi_30min' / 'h1' / 'protocol' / 'ARYW_protocol.csv', index=False)
            shutil.copy(pf / 'WSPI_protocol.csv',
                        root / 'run_A' / 'taxi_5min' / 'h1' / 'protocol' / 'ARIMA_protocol.csv')
            w = pd.read_csv(root / 'run_B' / 'youtube_hourly' / 'h8' / 'protocol' / 'WSPI_protocol.csv')
            w.loc[0, 'ndcg@10'] += 1e-12
            w.to_csv(root / 'run_B' / 'youtube_hourly' / 'h8' / 'protocol' / 'WSPI_protocol.csv', index=False)
            shutil.rmtree(root / 'run_A' / 'movielens_weekly' / 'h24')
            rc = S.main(['--root', str(root), '--B', '200', '--out', str(root / 'stats2')])
            plan = json.loads((root / 'stats2' / 'stats_plan.json').read_text())
            check('missing window, different item count, extra method and a different WSPI file fail',
                  rc == 1 and sorted(plan['window_check_failed']) ==
                  ['run_A/taxi_30min/h1', 'run_A/taxi_5min/h1', 'run_A/taxi_hourly/h1',
                   'run_B/youtube_hourly/h8'], str(plan['window_check_failed']))
            check('a missing run folder is listed',
                  plan['missing'] == ['run_A/movielens_weekly/h24'] and len(plan['reports']) == 60 - 13,
                  f"({len(plan['reports'])} reports)")
        finally:
            S.BASE_BLOCK.update(reg)

    print()
    if FAILED:
        print(f'{len(FAILED)} FAILED: {FAILED}')
        return 1
    print('ALL OK')
    return 0


if __name__ == '__main__':
    sys.exit(main())
