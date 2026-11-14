r"""
Run the full prediction runs (run A or run B) as a set of parallel jobs
=======================================================================
One job is one call of ``tools/run_prediction.py`` for one scenario and one
horizon, in its own process, with ``--resume``.  The jobs are started as cores
and memory allow.  If the computer stops (power cut, closed window), give the
same command again: complete jobs finish in seconds, interrupted jobs continue
(see ``tools/prediction_resume.py``).

  run A   ten methods, six scenarios, horizons 1, 8, 24      -> <root>/run_A/<scenario>/h<h>/
          (WSPI, WSPI-FG, WSPI-F2, WSPI-F-YW, WSPI-F, WSPI-FT, Persistence,
          Holt, ARYW, MA64; with the gate profile and the bound check)
  run B   WSPI, WSPI-FA, ARIMA; YouTube and hourly taxi,
          horizons 1, 8, 24 (one MLE fit per item and window)  -> <root>/run_B/<scenario>/h<h>/

Supplementary runs (Experiment_Design_v2; same scenarios and horizons as run A):
  run C       WSPI, WSPI-S, WSPI-FS1, WSPI-FS1.5, WSPI-FS2, WSPI-FS3, SMA8 to SMA64,
              EWMA8 to EWMA64, GBM, GBM+W; with the bound check   -> <root>/run_C/...
  run C_sens  WSPI and the 18 configurations WSPI-F2_c<c>_g<g>_p<p>
              (c 2, 4, 8; gamma 0.1, 0.3, 0.5; phi_d 0.8, 1); bound check
                                                                 -> <root>/run_C_sens/...
  run C_nlms  convergence of NLMS on the trend-band coefficients
              (experiments/nlms_convergence.py); YouTube and hourly taxi,
              horizon 1                                          -> <root>/run_C_nlms/...

Examples (from the project root, Windows):

  python tools\run_prediction_plan.py --plan A --jobs 6 --mem-gb 12
  python tools\run_prediction_plan.py --plan B --jobs 6
  python tools\run_prediction_plan.py --plan A --dry-run        # print the commands only

Every job writes its console output to ``logs/prediction_plan/run_<plan>/
<scenario>_h<h>.log`` (appended on every start).  ``<root>/run_<plan>/
plan_status.json`` lists, for every job, the exit code, the seconds of every
session and the log file.  Exit code 0 only if every job ended with 0.

``--mem-gb`` limits the sum of the estimated peak memory of the running jobs
(``MEMORY_GB``; the dense item-by-slot matrix of MovieLens daily is the large
one).  A job that alone exceeds the limit still runs, alone.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# scenario: (data file, causal item threshold) -- the settings of the stored runs
SCENARIOS = {
    'youtube_hourly': ('data/datasets/youtube_hourly.csv', 50),
    'taxi_hourly': ('data/datasets/yellow_taxi_2025_all_hourly.csv', 24),
    'taxi_30min': ('data/datasets/yellow_taxi_2025_all_30min.csv', 24),
    'taxi_5min': ('data/datasets/yellow_taxi_2025_all_5min.csv', 24),
    'movielens_daily': ('data/datasets/movielens_v5_daily.csv', 24),
    'movielens_weekly': ('data/datasets/movielens_v5_weekly.csv', 24),
}
HORIZONS = (1, 8, 24)
RUN_C_METHODS = ('WSPI', 'WSPI-S', 'WSPI-FS1', 'WSPI-FS1.5', 'WSPI-FS2', 'WSPI-FS3',
                 'SMA8', 'SMA16', 'SMA32', 'SMA64', 'EWMA8', 'EWMA16', 'EWMA32', 'EWMA64',
                 'GBM', 'GBM+W')
RUN_C_SENS_METHODS = ('WSPI',) + tuple(f'WSPI-F2_c{c}_g{g}_p{p}' for c in (2, 4, 8)
                                       for g in (0.1, 0.3, 0.5) for p in (0.8, 1))
PLANS = {
    'A': dict(scenarios=tuple(SCENARIOS), methods=None,
              flags=('--gate-profile', '--bound-check')),
    'B': dict(scenarios=('youtube_hourly', 'taxi_hourly'),
              methods=('WSPI', 'WSPI-FA', 'ARIMA'), flags=('--bound-check',)),
    'C': dict(scenarios=tuple(SCENARIOS), methods=RUN_C_METHODS, flags=('--bound-check',)),
    'C_sens': dict(scenarios=tuple(SCENARIOS), methods=RUN_C_SENS_METHODS,
                   flags=('--bound-check',)),
    'C_nlms': dict(scenarios=('youtube_hourly', 'taxi_hourly'), methods=None, flags=(),
                   horizons=(1,), script='experiments/nlms_convergence.py'),
}
# estimated peak memory of one job (GB) and the start order (longest first)
MEMORY_GB = {'youtube_hourly': 1.0, 'taxi_hourly': 1.0, 'taxi_30min': 1.0,
             'taxi_5min': 1.5, 'movielens_daily': 6.0, 'movielens_weekly': 2.5}
ORDER = {'A': ('taxi_5min', 'movielens_daily', 'taxi_30min', 'movielens_weekly',
               'taxi_hourly', 'youtube_hourly'),
         'B': ('taxi_hourly', 'youtube_hourly')}
ORDER['C'] = ORDER['C_sens'] = ORDER['A']
ORDER['C_nlms'] = ORDER['B']


def build_jobs(plan: str, root: str, scenarios=None, horizons=None, python=None,
               data_dir=None):
    spec = PLANS[plan]
    scenarios = [s for s in ORDER[plan] if s in (scenarios or spec['scenarios'])]
    jobs = []
    for s in scenarios:
        data, min_obs = SCENARIOS[s]
        if data_dir:
            data = str(Path(data_dir) / Path(data).name)
        for h in (horizons or spec.get('horizons', HORIZONS)):
            out = f'{root}/run_{plan}/{s}'
            script = spec.get('script', 'tools/run_prediction.py')
            cmd = [python or sys.executable, str(ROOT / Path(script)),
                   '--data', data, '--min-obs', str(min_obs), '--horizons', str(h),
                   '--out', out, '--resume', *spec['flags']]
            if spec['methods']:
                cmd += ['--methods', *spec['methods']]
            jobs.append(dict(name=f'{s}_h{h}', scenario=s, horizon=h, cmd=cmd,
                             mem_gb=MEMORY_GB[s], out=f'{out}/h{h}'))
    return jobs


def _abs(p):
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def run_jobs(jobs, n_jobs: int, mem_gb: float, log_dir: Path, status_file: Path,
             poll: float = 2.0):
    log_dir.mkdir(parents=True, exist_ok=True)
    status = {}
    if status_file.exists():
        try:
            status = json.loads(status_file.read_text(encoding='utf-8')).get('jobs', {})
        except (ValueError, OSError):
            status = {}
    pending = list(jobs)
    running = []                                  # (job, process, log handle, t0)
    t_start = time.time()
    env = dict(os.environ, PYTHONUTF8='1', PYTHONIOENCODING='utf-8', PYTHONWARNINGS='ignore')

    def save():
        status_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = status_file.with_name(status_file.name + '.tmp')
        tmp.write_text(json.dumps(dict(updated=time.strftime('%Y-%m-%d %H:%M:%S'),
                                       jobs=status), indent=2), encoding='utf-8')
        os.replace(tmp, status_file)

    def stamp():
        return time.strftime('%H:%M:%S')

    try:
        while pending or running:
            # start what fits
            started = True
            while started and pending and len(running) < n_jobs:
                started = False
                used = sum(j['mem_gb'] for j, *_ in running)
                for j in pending:
                    if not running or used + j['mem_gb'] <= mem_gb:
                        log = log_dir / f"{j['name']}.log"
                        fh = open(log, 'a', encoding='utf-8')
                        fh.write(f"\n===== session started {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
                        fh.flush()
                        p = subprocess.Popen(j['cmd'], cwd=ROOT, stdout=fh,
                                             stderr=subprocess.STDOUT, env=env)
                        running.append((j, p, fh, time.time()))
                        pending.remove(j)
                        print(f"[{stamp()}] start  {j['name']}  (running {len(running)}, "
                              f"waiting {len(pending)})", flush=True)
                        started = True
                        break
            time.sleep(poll)
            for item in list(running):
                j, p, fh, t0 = item
                rc = p.poll()
                if rc is None:
                    continue
                fh.close()
                running.remove(item)
                dt = time.time() - t0
                st = status.setdefault(j['name'], dict(sessions=[]))
                st.update(scenario=j['scenario'], horizon=j['horizon'], exit_code=rc,
                          out=j['out'], log=str((log_dir / f"{j['name']}.log")))
                st['sessions'].append(dict(ended=time.strftime('%Y-%m-%d %H:%M:%S'),
                                           seconds=round(dt, 1), exit_code=rc))
                save()
                print(f"[{stamp()}] {'done ' if rc == 0 else 'ERROR'}  {j['name']}  "
                      f"exit={rc}  {dt / 60:.1f} min", flush=True)
    except KeyboardInterrupt:
        print('\ninterrupted: stopping the running jobs (give the same command to continue)')
        for j, p, fh, t0 in running:
            p.terminate()
        for j, p, fh, t0 in running:
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                p.kill()
            fh.close()
        return 130, status
    bad = [j['name'] for j in jobs if status.get(j['name'], {}).get('exit_code') != 0]
    print(f"\n{len(jobs) - len(bad)} of {len(jobs)} jobs ended with exit code 0, "
          f"{(time.time() - t_start) / 3600:.2f} h in this session")
    if bad:
        print('JOBS WITH ERRORS: ' + ', '.join(bad))
        print(f'see the log files in {log_dir}')
    else:
        print('PLAN COMPLETE')
    return (1 if bad else 0), status


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--plan', required=True, choices=sorted(PLANS))
    ap.add_argument('--jobs', type=int, default=4, help='processes at the same time')
    ap.add_argument('--mem-gb', type=float, default=12.0,
                    help='limit on the summed memory estimate of the running jobs')
    ap.add_argument('--root', default='results/paper2', help='output root')
    ap.add_argument('--scenarios', nargs='+', default=None, choices=sorted(SCENARIOS))
    ap.add_argument('--horizons', nargs='+', type=int, default=None)
    ap.add_argument('--data-dir', default=None,
                    help='folder with the data files (default: data/datasets)')
    ap.add_argument('--log-dir', default=None,
                    help='default: logs/prediction_plan/run_<plan>')
    ap.add_argument('--dry-run', action='store_true', help='print the commands and stop')
    a = ap.parse_args(argv)

    jobs = build_jobs(a.plan, a.root, a.scenarios, a.horizons, data_dir=a.data_dir)
    if a.dry_run:
        for j in jobs:
            print(f"{j['name']:<24} {j['mem_gb']:>4} GB  " + subprocess.list2cmdline(j['cmd']))
        print(f'{len(jobs)} jobs')
        return 0
    log_dir = _abs(a.log_dir) if a.log_dir else ROOT / 'logs' / 'prediction_plan' / f'run_{a.plan}'
    status_file = _abs(a.root) / f'run_{a.plan}' / 'plan_status.json'
    print(f'plan {a.plan}: {len(jobs)} jobs, {a.jobs} at a time, memory limit {a.mem_gb} GB')
    print(f'logs: {log_dir}')
    rc, _ = run_jobs(jobs, a.jobs, a.mem_gb, log_dir, status_file)
    return rc


if __name__ == '__main__':
    sys.exit(main())
