r"""
Unit checks for the resume support, the bound check and the run plan
====================================================================
Covers ``tools/prediction_resume.py``, the options ``--resume`` and
``--bound-check`` of ``tools/run_prediction.py`` and ``tools/run_prediction_plan.py``.

  1. A run with --resume --bound-check writes the same result files, byte by
     byte, as a run without them (protocol and comparison files).
  2. A run that is interrupted several times (inside a fast method, inside the
     journal of WSPI-FA, inside ARIMA with a half-written last record) and
     started again gives the same files, byte by byte; the scorer statistics
     (calls, rows, fits) equal those of the uninterrupted run.
  3. A complete horizon is not run again.
  4. A resume state of a different run is refused; a changed result file is
     computed again; a journal that does not match stops the run (it is not
     turned into scores of zero by the evaluator).
  5. A scorer call that failed and was scored row by row is replayed the same way.
  6. BoundRecorder returns the scores unchanged; the ratios of the clamped
     methods are inside the bounds for h = 1, 8, 24; a scorer outside the
     bounds is reported.
  7. The run plan: 18 jobs for run A and 6 for run B, the memory limit, the
     status file and the exit code.

    python tools/test_prediction_resume.py
"""
import json
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.fast_evaluator import FastMethod                          # noqa: E402
from evaluation.protocol_v5 import ProtocolV5Evaluator, make_v5_wspi       # noqa: E402
from methods.wspi_forecast import ratio_bounds                             # noqa: E402
from tools import prediction_protocol as P                                # noqa: E402
from tools import prediction_resume as R                                  # noqa: E402
from tools import run_prediction, run_prediction_plan                     # noqa: E402

FAILED = []
METHODS = ['WSPI', 'WSPI-FG', 'WSPI-F2', 'Persistence', 'WSPI-FA', 'ARIMA']
HORIZON = 8


def check(name, ok, detail=''):
    print(f"{'OK  ' if ok else 'FAIL'} {name} {detail}")
    if not ok:
        FAILED.append(name)
    return ok


def make_data(path: Path, n_items=7, n_slots=74, seed=3):
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp('2025-01-01')
    rows = []
    for i in range(n_items):
        base = rng.gamma(2.0, 15.0) + 2
        lam = base * (1 + 0.6 * np.sin(2 * np.pi * np.arange(n_slots) / 24 + i))
        x = rng.poisson(np.clip(lam, 0.2, None))
        for k in range(n_slots):
            if x[k] > 0:
                rows.append((str(t0 + pd.Timedelta(hours=k)), f'i{i}', float(x[k])))
    pd.DataFrame(rows, columns=['timestamp', 'item_id', 'count']).to_csv(path, index=False)


def run(data, out, extra=(), methods=METHODS):
    argv = ['--data', str(data), '--min-obs', '5', '--horizons', str(HORIZON),
            '--methods', *methods, '--out', str(out), *extra]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        run_prediction.main(argv)


def result_files(out_h: Path):
    return sorted(str(f.relative_to(out_h).as_posix()) for f in out_h.rglob('*.csv')
                  if f.name not in ('scorer_timing.csv', 'bound_check.csv'))


def same_bytes(a: Path, b: Path, files):
    return [f for f in files if (a / f).read_bytes() != (b / f).read_bytes()]


class Interrupt(BaseException):
    pass


def test_resume(tmp: Path):
    data = tmp / 'data.csv'
    make_data(data)
    plain = tmp / 'plain'
    run(data, plain)
    ph = plain / f'h{HORIZON}'
    files = result_files(ph)
    check('plain run wrote protocol and comparison files',
          len(files) == len(METHODS) + 2, f'({len(files)} files)')

    # 1. uninterrupted run with --resume --bound-check
    full = tmp / 'full'
    rd = tmp / 'state_full'
    run(data, full, ['--resume', '--bound-check', '--resume-dir', str(rd)])
    diff = same_bytes(ph, full / f'h{HORIZON}', files)
    check('--resume --bound-check: result files byte-identical', not diff, str(diff))
    bc = pd.read_csv(full / f'h{HORIZON}' / 'bound_check.csv')
    check('bound_check.csv has the clamped methods of the run',
          sorted(bc['method']) == ['WSPI-F2', 'WSPI-FA', 'WSPI-FG'] and (bc['status'] == 'OK').all())
    check('journals deleted and horizon marked complete',
          not list((rd / f'h{HORIZON}').glob('*.journal'))
          and (rd / f'h{HORIZON}' / 'complete.json').exists())

    # 3. complete horizon is not run again
    before = {f: (full / f'h{HORIZON}' / f).stat().st_mtime_ns for f in files}
    run(data, full, ['--resume', '--bound-check', '--resume-dir', str(rd)])
    after = {f: (full / f'h{HORIZON}' / f).stat().st_mtime_ns for f in files}
    check('complete horizon is not run again', before == after)

    # 2. interrupted run
    inter = tmp / 'inter'
    ri = tmp / 'state_inter'
    args = ['--resume', '--bound-check', '--resume-dir', str(ri)]
    sdir = ri / f'h{HORIZON}'
    orig_run_method = ProtocolV5Evaluator.run_method
    orig_append = R.ScoreJournal.append

    def stop_in_method(name):
        def f(self, fm, out_dir=None):
            if fm.name == name:
                raise Interrupt()
            return orig_run_method(self, fm, out_dir)
        return f

    def stop_after_appends(n):
        count = [0]

        def f(self, *a):
            orig_append(self, *a)
            count[0] += 1
            if count[0] >= n:
                self.close()
                raise Interrupt()
        return f

    sessions = 0
    try:                                              # a: stop before the third method
        ProtocolV5Evaluator.run_method = stop_in_method('WSPI-F2')
        try:
            run(data, inter, args)
        except Interrupt:
            sessions += 1
        ProtocolV5Evaluator.run_method = orig_run_method
        check('session 1 stopped after two methods',
              sorted(f.name for f in sdir.glob('*.done.json')) == ['WSPI-FG.done.json', 'WSPI.done.json'])
        R.ScoreJournal.append = stop_after_appends(9)   # b: inside WSPI-FA
        try:
            run(data, inter, args)
        except Interrupt:
            sessions += 1
        check('session 2 stopped inside WSPI-FA (journal kept)',
              (sdir / 'WSPI-FA.journal').exists() and not (sdir / 'WSPI-FA.done.json').exists())
        R.ScoreJournal.append = stop_after_appends(14)  # c: again inside WSPI-FA
        try:
            run(data, inter, args)
        except Interrupt:
            sessions += 1
        R.ScoreJournal.append = stop_after_appends(60)  # d: inside ARIMA
        try:
            run(data, inter, args)
        except Interrupt:
            sessions += 1
        R.ScoreJournal.append = orig_append
        jf = sdir / 'ARIMA.journal'
        ok_j = jf.exists() and (sdir / 'WSPI-FA.done.json').exists()
        check('session 4 stopped inside ARIMA', ok_j)
        if ok_j:                                      # half-written last record
            raw = jf.read_bytes()
            jf.write_bytes(raw[:-5])
            n_before = len(R.ScoreJournal(jf).records)
            check('half-written last record is dropped',
                  jf.stat().st_size < len(raw) - 5 and n_before >= 1, f'({n_before} records kept)')
        run(data, inter, args)                         # e: runs to the end
    finally:
        ProtocolV5Evaluator.run_method = orig_run_method
        R.ScoreJournal.append = orig_append
    check('four interrupted sessions', sessions == 4, f'({sessions})')
    diff = same_bytes(ph, inter / f'h{HORIZON}', files)
    check('interrupted and resumed: result files byte-identical', not diff, str(diff))
    t0 = pd.read_csv(ph / 'scorer_timing.csv').set_index('method')
    t1 = pd.read_csv(inter / f'h{HORIZON}' / 'scorer_timing.csv').set_index('method')
    same_counts = all(t0.loc[m, c] == t1.loc[m, c] for m in METHODS for c in ('windows', 'calls', 'rows'))
    check('scorer statistics equal the uninterrupted run (windows, calls, rows)', same_counts)
    check('WSPI-FA fits equal the uninterrupted run',
          t0.loc['WSPI-FA', 'n_fits'] == t1.loc['WSPI-FA', 'n_fits'],
          f"({t1.loc['WSPI-FA', 'n_fits']:.0f} fits)")
    check('replayed calls are reported',
          t1.loc['WSPI-FA', 'replayed_calls'] > 0 and t1.loc['ARIMA', 'replayed_calls'] > 0,
          f"(WSPI-FA {t1.loc['WSPI-FA', 'replayed_calls']:.0f}, ARIMA {t1.loc['ARIMA', 'replayed_calls']:.0f})")
    b0 = pd.read_csv(full / f'h{HORIZON}' / 'bound_check.csv')
    b1 = pd.read_csv(inter / f'h{HORIZON}' / 'bound_check.csv')
    check('bound check of the resumed run equals the uninterrupted run', b0.equals(b1))

    # 4. refused / recomputed
    try:
        run_prediction.main(['--data', str(data), '--min-obs', '6', '--horizons', str(HORIZON),
                             '--methods', *METHODS, '--out', str(inter), *args])
        refused = False
    except SystemExit as e:
        refused = isinstance(e, R.ResumeError)
    check('resume state of a different run is refused', refused)
    f = inter / f'h{HORIZON}' / 'protocol' / 'WSPI-FG_protocol.csv'
    good = f.read_bytes()
    f.write_bytes(good[:len(good) // 2])
    (sdir / 'complete.json').unlink()
    run(data, inter, args)
    check('a damaged result file is computed again', f.read_bytes() == good)


def test_journal(tmp: Path):
    rng = np.random.default_rng(0)
    calls = []

    def scorer(X):
        calls.append(X.shape[0])
        if X.shape[0] == 3:
            raise ValueError('batch of three fails')
        return X.sum(axis=1) * 1.25

    Xs = [rng.random((4, 8)), rng.random((3, 8)), rng.random((5, 8))]
    path = tmp / 'j' / 'M.journal'
    ev = ProtocolV5Evaluator._safe_score
    a = R.JournalScorer(P.TimedScorer(scorer), path)
    out_a = [ev(a, X) for X in Xs]
    a.journal.close()
    n_live = len(calls)
    b = R.JournalScorer(P.TimedScorer(scorer), path)
    out_b = [ev(b, X) for X in Xs]
    check('replay gives the same scores, also for a call that failed and was scored row by row',
          all(np.array_equal(x, y) for x, y in zip(out_a, out_b)) and len(calls) == n_live,
          f'({b.replayed["calls"]} calls replayed, scorer not called again)')
    st = b.stats()
    check('statistics of a replayed scorer count the replayed calls',
          st['calls'] == a.stats()['calls'] and st['rows'] == a.stats()['rows'])
    c = R.JournalScorer(P.TimedScorer(scorer), path)
    try:
        ev(c, rng.random((4, 8)))
        stopped = False
    except SystemExit:
        stopped = True
    check('a journal that does not match stops the run (not scored as zero)', stopped)


def test_bounds():
    rng = np.random.default_rng(5)
    X = rng.poisson(rng.gamma(2.0, 20.0, size=(300, 1)) * np.ones((1, 64))).astype(float)
    X[::7, -3:] *= 40                                  # spikes at the end
    X[::11, -20:] = 0                                  # items that stop
    X[5] = 0
    for h in (1, 8, 24):
        base = P.build_prediction_methods(horizon=h, names=['WSPI', 'WSPI-FG', 'WSPI-F2',
                                                            'WSPI-FT', 'WSPI-F-YW'], timed=True)
        wrapped, rec = R.add_bound_recorders(base, h)
        same = all(np.array_equal(base[n].scorer(X), wrapped[n].scorer(X), equal_nan=True)
                   for n in base)
        lo, up = ratio_bounds(16, P.n_forecast_coefficients(h), 4.0)
        st = [rec[n].bound_stats() for n in rec]
        inside = all(s['status'] == 'OK' and s['lower_bound'] == lo and s['upper_bound'] == up
                     and s['min_ratio'] >= lo * (1 - 1e-9) and s['max_ratio'] <= up * (1 + 1e-9)
                     for s in st)
        check(f'h={h}: recorder leaves the scores unchanged; clamped methods inside the bounds',
              same and inside and sorted(rec) == ['WSPI-F2', 'WSPI-FG', 'WSPI-FT'],
              f"(min {min(s['min_ratio'] for s in st):.4g} >= {lo:.4g}, "
              f"max {max(s['max_ratio'] for s in st):.4g} <= {up:.4g})")
    wspi = make_v5_wspi(window=64, level=3)
    bad = R.BoundRecorder(lambda Z: wspi(Z) * 9.0, 'WSPI-F2', 1)
    bad(X)
    s = bad.bound_stats()
    check('a scorer outside the bound is reported', s['status'] == 'VIOLATION' and s['n_above'] > 0)


def test_plan(tmp: Path):
    ja = run_prediction_plan.build_jobs('A', 'results/prediction')
    jb = run_prediction_plan.build_jobs('B', 'results/prediction')
    check('run A: 18 jobs, run B: 6 jobs', len(ja) == 18 and len(jb) == 6)
    check('every job resumes and checks the bound; run A writes the gate profile',
          all('--resume' in j['cmd'] and '--bound-check' in j['cmd'] for j in ja + jb)
          and all('--gate-profile' in j['cmd'] for j in ja)
          and all(j['cmd'][j['cmd'].index('--methods') + 1:] == ['WSPI', 'WSPI-FA', 'ARIMA'] for j in jb))
    check('run A starts with the longest scenario', ja[0]['scenario'] == 'taxi_5min')
    marks = tmp / 'marks'
    marks.mkdir()
    code = ("import sys,time,pathlib;d=pathlib.Path(sys.argv[1]);n=sys.argv[2];"
            "(d/(n+'.run')).write_text('x');"
            "sys.stdout.write(''.join(sorted(f.stem for f in d.glob('*.run'))));"
            "time.sleep(0.6);(d/(n+'.run')).unlink();sys.exit(3 if n=='c' else 0)")
    jobs = [dict(name=n, scenario=n, horizon=1, mem_gb=m, out=n,
                 cmd=[sys.executable, '-c', code, str(marks), n])
            for n, m in (('a', 6.0), ('b', 6.0), ('c', 1.0), ('d', 1.0))]
    rc, status = run_prediction_plan.run_jobs(jobs, 4, 7.5, tmp / 'plogs', tmp / 'st' / 'plan_status.json',
                                              poll=0.1)
    seen = {n: (tmp / 'plogs' / f'{n}.log').read_text().strip().splitlines()[-1] for n in 'abcd'}
    check('memory limit: the two 6 GB jobs never run together',
          'b' not in seen['a'] and 'a' not in seen['b'], f'({seen})')
    saved = json.loads((tmp / 'st' / 'plan_status.json').read_text())['jobs']
    check('exit code 1 and the failing job in the status file',
          rc == 1 and saved['c']['exit_code'] == 3 and saved['a']['exit_code'] == 0
          and len(saved['a']['sessions']) == 1)
    order = [status[n]['sessions'][0]['ended'] for n in 'ab']
    check('status file lists every job', sorted(saved) == ['a', 'b', 'c', 'd'], str(order))


def main():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        test_journal(tmp)
        test_bounds()
        test_plan(tmp)
        test_resume(tmp)
    print()
    if FAILED:
        print(f'{len(FAILED)} FAILED: {FAILED}')
        return 1
    print('ALL OK')
    return 0


if __name__ == '__main__':
    sys.exit(main())
