r"""
Shared parallel engine for popularity-assessment runs
=====================================================
Used by run_popularity_assessment.py (main), run_wspi_ablation.py and
run_wspi_sensitivity.py.

Model
-----
Each METHOD runs in its own short-lived process (methods are independent;
windows stay serial inside a method because RSI carries sequential state).
ALL methods write into ONE shared, TIMESTAMPED results folder, e.g.
    results/<dataset>/main_<YYYYMMDD_HHMMSS>/
Per-method files are uniquely named, so concurrent writes never collide.

Self-contained state & logs (so several runs can proceed at once with no
interference):
    <run_folder>/_status.json     -> {method: "done"/"failed"} progress
    <run_folder>/logs/<m>.log     -> per-method output

Resume
------
There are NO fixed folder names and NO external markers. To continue an
interrupted run, pass its folder explicitly:
    --resume results/<dataset>/main_<ts>
Every method NOT marked "done" in that folder's _status.json is re-run
(its partial files are cleaned first). Without --resume, a fresh timestamped
folder is created.
"""
import os
import sys
import json
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments._auto_extract import _load_protocol, _summarise_method


def resolve_cores(cores):
    cpu = os.cpu_count() or 2
    if cores is None or cores <= 0:
        return max(1, cpu - 1)
    return min(cores, cpu)


# ---------------------------------------------------------------------------
# Status file (lives INSIDE the run folder)
# ---------------------------------------------------------------------------
def _status_path(run_dir):
    return Path(run_dir) / '_status.json'


def _read_status(run_dir):
    p = _status_path(run_dir)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding='utf-8'))
        except Exception:
            return {}
    return {}


def _write_status(run_dir, status):
    # full-file rewrite (atomic-ish via temp + replace) to tolerate concurrency
    p = _status_path(run_dir)
    tmp = p.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(tmp, p)


# ---------------------------------------------------------------------------
# Worker — runs exactly ONE method/variant into the shared folder
# ---------------------------------------------------------------------------
def _harden_shared_metadata_writes():
    """Make the 4 shared metadata writes tolerant of concurrent access.

    In flat single-folder mode every method-process writes the SAME small
    metadata files (config.json, thresholds.json, runtime_stats.json,
    run_metadata.json). On Windows two simultaneous open('w') on one file
    raise PermissionError (sharing violation). The content is identical across
    methods, so we simply retry a few times and, failing that, skip silently —
    another process has already written the (identical) file.
    """
    import time as _t

    def _retry(fn):
        def wrapped(*a, **k):
            for i in range(8):
                try:
                    return fn(*a, **k)
                except (PermissionError, OSError):
                    _t.sleep(0.05 * (i + 1))
            return None   # give up quietly; a sibling wrote the same file
        return wrapped

    try:
        from evaluation.incremental_evaluator import IncrementalTemporalEvaluator as _E
        for m in ('_save_config_json', '_save_thresholds_json', '_save_runtime_stats_json'):
            if hasattr(_E, m):
                setattr(_E, m, _retry(getattr(_E, m)))
    except Exception:
        pass
    try:
        from evaluation.incremental_storage import IncrementalStorage as _S
        if hasattr(_S, 'save_metadata'):
            _S.save_metadata = _retry(_S.save_metadata)
    except Exception:
        pass


def _worker(task):
    log_path = Path(task['log_path'])
    method = task['method']
    try:
        fh = open(log_path, 'w', encoding='utf-8')
        sys.stdout = fh
        sys.stderr = fh
        print(f'>>> RUNNING METHOD: {method}', flush=True)
    except Exception:
        fh = None
    try:
        import experiments.run_popularity_assessment as runner
        import evaluation.method_configs as mc
        from evaluation.method_configs import MethodConfig

        _harden_shared_metadata_writes()   # concurrency-safe shared metadata

        spec = task.get('spec')
        if spec is not None:
            mc.METHOD_CONFIGS[method] = MethodConfig(
                name=method, window_slots=64, min_observations=32,
                description=f'WSPI variant: {method}')
            _orig = runner.create_methods_dict

            def _patched(config):
                from methods.wspi_assessment import WSPIAssessment
                d = _orig(config)
                req = set(config.methods) if config.methods else None
                if req is None or method in req:
                    d[method] = WSPIAssessment(name=method, **spec)
                return d

            runner.create_methods_dict = _patched

        kwargs = dict(
            dataset_name=task['dataset'], methods=[method],
            output_dir=Path(task['out_dir']), use_timestamp=False,
            incremental=True, num_items=task['num_items'],
            start_date=task['start_date'], end_date=task['end_date'],
            window_size=task['window_size'],
            prediction_horizon=task['horizon'],
            item_selection=task['item_selection'],
            data_path=task['data_path'], verbose=False,
        )
        if task.get('k_list'):
            kwargs['k_list'] = task['k_list']
        runner.run_temporal_evaluation(**kwargs)
        return (method, True, None)
    except Exception:
        return (method, False, traceback.format_exc())
    finally:
        if fh is not None:
            try:
                fh.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Folder helpers
# ---------------------------------------------------------------------------
def _new_run_dir(dataset, tag):
    ts = time.strftime('%Y%m%d_%H%M%S')
    p = ROOT / 'results' / dataset / f'{tag}_{ts}'
    (p / 'logs').mkdir(parents=True, exist_ok=True)
    return p


def _safe_name(method):
    return method.replace('+', 'PLUS').replace('/', '_').replace('\\', '_')


def _clean_partial(run_dir, method):
    """Remove a method's stale per-method files so a re-run starts clean."""
    for sub, pat in [('protocol', f'{method}_protocol.*'),
                     ('detailed', f'{method}_scores.*'),
                     ('summary',  f'{method}_*.*')]:
        d = run_dir / sub
        if d.exists():
            for fp in d.glob(pat):
                try:
                    fp.unlink()
                except Exception:
                    pass


def _method_complete(run_dir, method):
    return _load_protocol(run_dir, method) is not None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def run_methods_parallel(dataset_name, method_names, *, cores=-1, tag='main',
                         specs=None, data_path=None, num_items=None,
                         start_date=None, end_date=None, window_size=30,
                         prediction_horizon=7, item_selection='top',
                         k_list=None, resume=None):
    specs = specs or {}
    cores = resolve_cores(cores)

    # Decide the run folder: resume an existing one, or make a fresh timestamped one.
    if resume:
        run_dir = Path(resume)
        if not run_dir.is_absolute():
            run_dir = ROOT / run_dir
        (run_dir / 'logs').mkdir(parents=True, exist_ok=True)
    else:
        run_dir = _new_run_dir(dataset_name, tag)

    status = _read_status(run_dir)

    # 'done' if the status says so AND the protocol is actually present.
    todo, done_already = [], []
    for m in method_names:
        if status.get(m) == 'done' and _method_complete(run_dir, m):
            done_already.append(m)
        else:
            todo.append(m)

    os.environ['PYTHONUTF8'] = '1'
    os.environ['PYTHONIOENCODING'] = 'utf-8'

    print('=' * 72)
    print(f'PARALLEL ENGINE  dataset={dataset_name}  tag={tag}  cores={cores}')
    print(f'  run folder    : {run_dir}')
    print(f'  resume        : {"YES" if resume else "no (new folder)"}')
    print(f'  already done  : {len(done_already)}  {done_already}')
    print(f'  to run now    : {len(todo)}  {todo}')
    print('=' * 72)
    if not todo:
        _merge(run_dir, method_names, dataset_name, tag)
        _zip_run(run_dir)
        return run_dir

    for m in todo:
        _clean_partial(run_dir, m)
        status[m] = 'pending'
    _write_status(run_dir, status)

    tasks = [{
        'method': m, 'dataset': dataset_name, 'out_dir': str(run_dir),
        'spec': specs.get(m), 'data_path': data_path, 'num_items': num_items,
        'start_date': start_date, 'end_date': end_date,
        'window_size': window_size, 'horizon': prediction_horizon,
        'item_selection': item_selection, 'k_list': k_list,
        'log_path': str(run_dir / 'logs' / (_safe_name(m) + '.log')),
    } for m in todo]

    failed = []
    try:
        ex = ProcessPoolExecutor(max_workers=cores, max_tasks_per_child=1)
    except TypeError:
        ex = ProcessPoolExecutor(max_workers=cores)
    with ex:
        futs = {}
        for t in tasks:
            print(f'  [start] {t["method"]}', flush=True)
            futs[ex.submit(_worker, t)] = t['method']
        for fut in as_completed(futs):
            method = futs[fut]
            try:
                m, ok, err = fut.result()
            except Exception as e:
                m, ok, err = method, False, repr(e)
            if ok and _method_complete(run_dir, m):
                status[m] = 'done'
                print(f'  [done] {m}')
            else:
                status[m] = 'failed'
                failed.append(m)
                tail = (err.strip().splitlines()[-1] if err else 'no protocol written')
                print(f'  [FAIL] {m}  (logs/{m}.log): {tail}')
            _write_status(run_dir, status)   # persist after each method

    _merge(run_dir, method_names, dataset_name, tag)
    _zip_run(run_dir)
    print('=' * 72)
    print(f'FINISHED. done={sum(1 for m in method_names if status.get(m) == "done")} '
          f'failed={failed}')
    if failed:
        print('To retry only the unfinished methods, re-run with:')
        print(f'   --resume "{run_dir}"')
    return run_dir


def _merge(run_dir, method_names, dataset, tag):
    import pandas as pd
    status = _read_status(run_dir)
    rows = []
    for m in method_names:
        if status.get(m) != 'done':
            continue
        df = _load_protocol(run_dir, m)
        if df is None:
            continue
        row = {'method': m}
        row.update(_summarise_method(df))
        rows.append(row)
    if not rows:
        print('  merge: nothing completed yet.')
        return
    out = pd.DataFrame(rows)
    cols = ['method', 'rsi@10', 'rsi@5', 'ndcg@10', 'robustness_distortion',
            'spearman_rho', 'windows']
    ordered = [c for c in cols if c in out.columns] + \
              [c for c in out.columns if c not in cols]
    out = out[ordered]
    (run_dir / 'comparison').mkdir(parents=True, exist_ok=True)
    out.to_csv(run_dir / 'comparison' / f'{tag}_summary.csv',
               index=False, encoding='utf-8-sig')
    print('=' * 72)
    print(f'MERGED -> {run_dir / "comparison" / (tag + "_summary.csv")}')
    show = [c for c in ['method', 'rsi@10', 'ndcg@10', 'robustness_distortion']
            if c in out.columns]
    try:
        print(out[show].to_string(index=False))
    except Exception:
        pass


def _zip_run(run_dir):
    """Zip the whole run folder (tidy structure) next to it."""
    import shutil
    run_dir = Path(run_dir)
    archive = run_dir.parent / (run_dir.name + '.zip')
    try:
        if archive.exists():
            archive.unlink()
        shutil.make_archive(str(archive.with_suffix('')), 'zip',
                            root_dir=str(run_dir.parent), base_dir=run_dir.name)
        print(f'ZIP    -> {archive}')
    except Exception as e:
        print(f'  (zip skipped: {e})')
