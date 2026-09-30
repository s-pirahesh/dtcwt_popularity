r"""
Runtime, memory and scalability of the nine methods (revision-srep-v5, task T3.8 / E9)
=====================================================================================
Answers R1.7 (complexity and execution time against the baselines, large data),
R4.8 (implementation-level memory of the multi-scale coefficients, measured peak
memory) and R4.17 (rigorous runtime and memory; evidence for real-time claims).
No existing module is changed.  Every scorer is the exact protocol-V5 scorer of
the paper (evaluation.protocol_v5.build_v5_methods, J = 3).

Design (27 Sep 2026)
  * Threads: one thread for NumPy / BLAS (OMP, MKL, OpenBLAS, NumExpr, vecLib = 1),
    set before NumPy is imported ("--threads default" leaves the library default).
    The paper states the numbers as "on one CPU core".
  * Part A, method cost (--bench): synthetic Poisson counts (item rate
    lognormal(1, 1), seed 42).  Methods x window N in {32, 64, 128, 256} (all nine)
    plus N = 7 for the six baselines (their default).  Number of items
    M in {1, 1e3, 1e4, 1e5, 1e6}.  Items are scored in batches of CHUNK = 1e4 (one
    scorer call per batch, as a deployment would score one window); the batches
    cycle through a pool of up to 10 distinct pre-generated matrices, so data
    generation is never timed.  One warm-up call, then 10 repeats (time.perf_counter;
    for M = 1 each repeat is the mean of 200 calls).  Reported: median, quartiles
    and minimum of the time per scoring of all M items, and microseconds per
    item-window.
  * Part B, memory (--memory):
      tracemalloc_grid.csv  peak of the Python-traced allocations (NumPy reports to
                            tracemalloc) during ONE scorer call on a batch of
                            B in {1e3, 1e4, 1e5} items; the input matrix is allocated
                            before tracing starts, so it is not counted; per item.
      coefficient bytes     exact size of the stored coefficients of one item
                            (DTCWT pyramid / pywt.wavedec list), float64 / complex128.
      rss_grid.csv          resident memory of a fresh child process (N in {64, 256}):
                            M = 1e6 in batches of 1e4, and one batch of 1e5.  A thread
                            samples the current RSS every millisecond during scoring;
                            reported: sampled peak minus the RSS before scoring
                            (main value) and the OS peak minus that baseline (upper
                            bound, includes earlier peaks such as data generation).
      state_memory.csv      what an item needs between two updates (analytic, with
                            exact byte counts): the window buffer that every windowed
                            method reads (N x 8 bytes), the transient coefficients of
                            the wavelet-based methods, and the O(1) state of the
                            recursive (unwindowed) AF / EWMA.  Update cost is given
                            as an order (text), not measured.
  * Part C, real data (--real): the default causal run of the paper (protocol V5,
    baselines 7, wavelet-based 64, --causal-universe, seeded robustness test)
    is repeated with timers, one scenario per call.  Time is split into
        score_s   the scorer call that ranks the whole catalogue of a window
                  (method cost; per window -> latency),
        robust_s  the robustness test (50 targets scored again + 50 re-rankings),
        other_s   the rest of the pipeline (slicing, metrics, bookkeeping).
    Data loading and matrix building are timed once (load_s).  Control: every
    metric column and the window set must equal the T1.5 causal run
    (results/revision_v5/T1.5_causal_universe/T1.4_protocol_v5/<scenario>).
    The OS peak RSS of the whole process is recorded (full evaluation).
  * Hardware and library versions are written into every metadata JSON.

Layout (root = results/revision_v5/T3.8_runtime)
  bench/runtime_grid.csv, bench/runtime_raw.csv, bench/metadata/bench_run.json
  memory/tracemalloc_grid.csv, memory/rss_grid.csv, memory/state_memory.csv,
  memory/metadata/memory_run.json
  real/<scenario>/protocol/window_timing.csv   one row per method x window
  real/<scenario>/real_summary.csv, real/<scenario>/real_control.csv,
  real/<scenario>/metadata/runtime_real_run.json
  --collect <root>:
  <root>/runtime_paper_table.csv   one row per method: the numbers of the paper table
  <root>/real_summary_all.csv, <root>/real_control_all.csv

Examples (from the project root, Windows):

  python tools\run_runtime_v5.py --hardware
  python tools\run_runtime_v5.py --bench --out results\revision_v5\T3.8_runtime\bench
  python tools\run_runtime_v5.py --memory --out results\revision_v5\T3.8_runtime\memory
  python tools\run_runtime_v5.py --real --data data\datasets\youtube_hourly.csv --min-obs 50 ^
         --causal-universe --out results\revision_v5\T3.8_runtime\real\youtube_hourly
  python tools\run_runtime_v5.py --collect results\revision_v5\T3.8_runtime

Full command list: REPRODUCE.md
Unit test: tools/test_runtime_v5.py
"""
import os
import sys

THREAD_VARS = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
               'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')


def _thread_setting(argv):
    if '--threads' in argv:
        i = argv.index('--threads')
        if i + 1 < len(argv):
            return argv[i + 1]
    return '1'


THREADS = _thread_setting(sys.argv)
if THREADS != 'default':                     # must happen before NumPy is imported
    for _v in THREAD_VARS:
        os.environ[_v] = str(int(THREADS))

import argparse  # noqa: E402
import contextlib  # noqa: E402
import gc  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
import tracemalloc  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import dtcwt  # noqa: E402
import pywt  # noqa: E402

with contextlib.redirect_stdout(io.StringIO()):          # method_configs prints on import
    from evaluation.protocol_v5 import (ProtocolV5Evaluator, build_v5_methods,  # noqa: E402
                                        METRIC_COLUMNS)
from evaluation.fast_evaluator import FastMethod  # noqa: E402

LEVEL, SEED = 3, 42
BASELINES = ['AF', 'EWMA', 'RRD', 'VSE', 'CompoundPop', 'PFRF']
WAVELETS = ['DWT+AF', 'DTCWT+AF', 'WSPI']
METHODS = BASELINES + WAVELETS
DEFAULT_N = {**{m: 7 for m in BASELINES}, **{m: 64 for m in WAVELETS}}
BENCH_N = (32, 64, 128, 256)
BASE_N = 7
BENCH_M = (1, 1_000, 10_000, 100_000, 1_000_000)
CHUNK, POOL, REPEATS, INNER_SINGLE = 10_000, 10, 10, 200
MEM_B = (1_000, 10_000, 100_000)
RSS_N = (64, 256)
RSS_CASES = (('chunked', 1_000_000, CHUNK), ('batch', 100_000, 100_000))
RSS_SAMPLE_S = 0.001
DWT_WAVELET, DTCWT_BIORT, DTCWT_QSHIFT = 'db4', 'near_sym_a', 'qshift_a'
REF_T15 = Path('results/revision_v5/T1.5_causal_universe/T1.4_protocol_v5')
SCENARIOS = ['youtube_hourly', 'taxi_hourly', 'taxi_30min', 'taxi_5min']
CONTROL_EXTRA = ['num_items', 'ties_top21', 'padded', 'n_nonfinite']
CONTROL_REL_TOL = 1e-12       # |x - ref| / max(1, |ref|); MAE is in the thousands, so an
                              # absolute bound would fail on the last bit across platforms


def _abs(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


# =============================================================================
# Hardware, versions, memory probes
# =============================================================================
def _cpu_name():
    s = platform.system()
    try:
        if s == 'Windows':
            import winreg
            k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                               r'HARDWARE\DESCRIPTION\System\CentralProcessor\0')
            name = winreg.QueryValueEx(k, 'ProcessorNameString')[0]
            try:
                mhz = winreg.QueryValueEx(k, '~MHz')[0]
            except OSError:
                mhz = None
            return str(name).strip(), mhz
        if s == 'Linux':
            with open('/proc/cpuinfo') as f:
                for line in f:
                    if line.lower().startswith('model name'):
                        return line.split(':', 1)[1].strip(), None
        if s == 'Darwin':
            out = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'],
                                 capture_output=True, text=True).stdout.strip()
            return out, None
    except Exception:
        pass
    return platform.processor(), None


def _psutil():
    try:
        import psutil
        return psutil
    except Exception:
        return None


def total_ram_bytes():
    ps = _psutil()
    if ps is not None:
        return int(ps.virtual_memory().total)
    if platform.system() == 'Windows':
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [('dwLength', ctypes.c_ulong), ('dwMemoryLoad', ctypes.c_ulong),
                        ('ullTotalPhys', ctypes.c_ulonglong), ('ullAvailPhys', ctypes.c_ulonglong),
                        ('ullTotalPageFile', ctypes.c_ulonglong),
                        ('ullAvailPageFile', ctypes.c_ulonglong),
                        ('ullTotalVirtual', ctypes.c_ulonglong),
                        ('ullAvailVirtual', ctypes.c_ulonglong),
                        ('ullAvailExtendedVirtual', ctypes.c_ulonglong)]
        st = MEMORYSTATUSEX()
        st.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
        return int(st.ullTotalPhys)
    try:
        with open('/proc/meminfo') as f:
            for line in f:
                if line.startswith('MemTotal:'):
                    return int(line.split()[1]) * 1024
    except Exception:
        pass
    return None


def _win_mem_counters():
    import ctypes
    from ctypes import wintypes

    class PMC(ctypes.Structure):
        _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD),
                    ('PeakWorkingSetSize', ctypes.c_size_t), ('WorkingSetSize', ctypes.c_size_t),
                    ('QuotaPeakPagedPoolUsage', ctypes.c_size_t),
                    ('QuotaPagedPoolUsage', ctypes.c_size_t),
                    ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t),
                    ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
                    ('PagefileUsage', ctypes.c_size_t), ('PeakPagefileUsage', ctypes.c_size_t)]
    k32 = ctypes.WinDLL('kernel32', use_last_error=True)
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    fn = k32.K32GetProcessMemoryInfo
    fn.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
    fn.restype = wintypes.BOOL
    c = PMC()
    c.cb = ctypes.sizeof(PMC)
    if not fn(k32.GetCurrentProcess(), ctypes.byref(c), c.cb):
        raise OSError('K32GetProcessMemoryInfo failed')
    return int(c.WorkingSetSize), int(c.PeakWorkingSetSize)


def current_rss():
    """Resident set size (Windows: working set) of this process, bytes."""
    ps = _psutil()
    if ps is not None:
        return int(ps.Process().memory_info().rss)
    if platform.system() == 'Windows':
        return _win_mem_counters()[0]
    try:
        with open('/proc/self/statm') as f:
            return int(f.read().split()[1]) * os.sysconf('SC_PAGE_SIZE')
    except Exception:
        return None


def peak_rss():
    """Peak resident set size of this process since it started, bytes."""
    if platform.system() == 'Windows':
        ps = _psutil()
        if ps is not None:
            mi = ps.Process().memory_info()
            if hasattr(mi, 'peak_wset'):
                return int(mi.peak_wset)
        return _win_mem_counters()[1]
    import resource
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(r) if platform.system() == 'Darwin' else int(r) * 1024


def _blas_info():
    try:
        cfg = np.show_config(mode='dicts')
        b = cfg.get('Build Dependencies', {}).get('blas', {})
        return {k: b.get(k) for k in ('name', 'version')}
    except Exception:
        return None


def _threadpools():
    try:
        from threadpoolctl import threadpool_info
        return [{k: d.get(k) for k in ('internal_api', 'version', 'num_threads')}
                for d in threadpool_info()]
    except Exception:
        return None


def versions():
    v = dict(python=platform.python_version(), python_impl=platform.python_implementation(),
             numpy=np.__version__, pandas=pd.__version__, pywt=pywt.__version__,
             dtcwt=dtcwt.__version__)
    try:
        import scipy
        v['scipy'] = scipy.__version__
    except Exception:
        v['scipy'] = None
    ps = _psutil()
    v['psutil'] = getattr(ps, '__version__', None) if ps is not None else None
    return v


def hardware_info():
    name, mhz = _cpu_name()
    ps = _psutil()
    info = dict(system=platform.system(), release=platform.release(), version=platform.version(),
                platform=platform.platform(), machine=platform.machine(),
                node=platform.node(), cpu_name=name, cpu_base_mhz=mhz,
                cpu_identifier=platform.processor(),
                logical_cpus=os.cpu_count(),
                physical_cores=(ps.cpu_count(logical=False) if ps is not None else None),
                ram_bytes=total_ram_bytes(),
                threads_setting=THREADS,
                thread_env={v: os.environ.get(v) for v in THREAD_VARS},
                blas=_blas_info(), threadpools=_threadpools(),
                versions=versions(),
                created=time.strftime('%Y-%m-%d %H:%M:%S'))
    ram = info['ram_bytes']
    info['ram_gib'] = round(ram / 2 ** 30, 2) if ram else None
    return info


# =============================================================================
# Scorers and synthetic data
# =============================================================================
def make_scorer(method: str, N: int):
    """The protocol-V5 scorer of the paper for window N (J = 3)."""
    return build_v5_methods(level=LEVEL, window_override={method: N}, names=[method])[method].scorer


def bench_windows(method: str):
    return ((BASE_N,) if method in BASELINES else ()) + BENCH_N


def synth_chunk(rng: np.random.Generator, m: int, N: int) -> np.ndarray:
    rates = rng.lognormal(mean=1.0, sigma=1.0, size=(m, 1))
    return rng.poisson(rates, size=(m, N)).astype(np.float64)


def make_pool(N: int, M: int, chunk: int = CHUNK, pool: int = POOL, seed: int = SEED):
    """Up to `pool` distinct matrices of min(M, chunk) rows; same for every method."""
    c = min(M, chunk)
    n = min(pool, math.ceil(M / c))
    rng = np.random.default_rng([seed, N, M])
    return [synth_chunk(rng, c, N) for _ in range(n)]


def score_all(scorer, pool, M: int) -> np.ndarray:
    """Score M items in batches of len(pool[0]) rows (one scorer call per batch)."""
    c = pool[0].shape[0]
    out = np.empty(M)
    for i, start in enumerate(range(0, M, c)):
        m = min(c, M - start)
        X = pool[i % len(pool)]
        out[start:start + m] = scorer(X if m == c else X[:m])
    return out


def time_once(scorer, pool, M: int, inner: int) -> float:
    gc.collect()
    t = time.perf_counter()
    for _ in range(inner):
        score_all(scorer, pool, M)
    return (time.perf_counter() - t) / inner


# =============================================================================
# Part A: bench
# =============================================================================
RAW_COLS = ['method', 'N', 'M', 'chunk', 'n_pool', 'inner', 'repeat', 'seconds']


def run_bench(out: Path, items=BENCH_M, windows=None, repeats=REPEATS, methods=METHODS,
              resume=False, verbose=True):
    out.mkdir(parents=True, exist_ok=True)
    raw_f = out / 'runtime_raw.csv'
    done = set()
    rows = []
    if resume and raw_f.exists():
        old = pd.read_csv(raw_f)
        rows = old.to_dict('records')
        cnt = old.groupby(['method', 'N', 'M']).size()
        done = {k for k, v in cnt.items() if v >= repeats}
    t_all = time.time()
    all_N = sorted({n for m in methods for n in bench_windows(m) if windows is None or n in windows})
    for N in all_N:
        for M in items:
            todo = [m for m in methods if N in bench_windows(m) and (m, N, M) not in done]
            if not todo:
                continue
            pool = make_pool(N, M)
            inner = INNER_SINGLE if M == 1 else 1
            rows = [r for r in rows if not (r['N'] == N and r['M'] == M and r['method'] in todo)]
            for m in todo:
                sc = make_scorer(m, N)
                score_all(sc, pool[:1], min(M, pool[0].shape[0]))          # warm-up
                ts = [time_once(sc, pool, M, inner) for _ in range(repeats)]
                rows += [dict(method=m, N=N, M=M, chunk=pool[0].shape[0], n_pool=len(pool),
                              inner=inner, repeat=r, seconds=t) for r, t in enumerate(ts)]
                if verbose:
                    print(f'[bench] N={N:<4} M={M:<8} {m:<12} median {np.median(ts):10.5f} s  '
                          f'{np.median(ts) / M * 1e6:9.3f} us/item', flush=True)
            del pool
            pd.DataFrame(rows, columns=RAW_COLS).to_csv(raw_f, index=False)   # checkpoint
    raw = pd.DataFrame(rows, columns=RAW_COLS)
    grid = summarize_bench(raw)
    grid.to_csv(out / 'runtime_grid.csv', index=False)
    _write_meta(out, 'bench_run.json', dict(
        part='A bench', items=list(items), windows=all_N, repeats=repeats, chunk=CHUNK,
        pool=POOL, inner_single=INNER_SINGLE, seed=SEED, level=LEVEL,
        data='synthetic Poisson counts, item rate lognormal(1, 1)', methods=list(methods),
        wall_seconds=time.time() - t_all))
    return grid


def summarize_bench(raw: pd.DataFrame) -> pd.DataFrame:
    g = raw.groupby(['method', 'N', 'M'], sort=False)
    rows = []
    for (m, N, M), d in g:
        s = d['seconds'].to_numpy()
        q1, med, q3 = np.percentile(s, [25, 50, 75])
        rows.append(dict(method=m, N=N, M=M, chunk=int(d['chunk'].iloc[0]),
                         n_pool=int(d['n_pool'].iloc[0]), inner=int(d['inner'].iloc[0]),
                         repeats=len(s), median_s=med, q1_s=q1, q3_s=q3, min_s=s.min(),
                         us_per_item_median=med / M * 1e6, us_per_item_q1=q1 / M * 1e6,
                         us_per_item_q3=q3 / M * 1e6, items_per_s=M / med))
    return pd.DataFrame(rows)


# =============================================================================
# Part B: memory
# =============================================================================
def coef_bytes(method: str, N: int):
    """Exact bytes (and real-valued count) of the coefficients stored for one item."""
    if method in ('DTCWT+AF', 'WSPI'):
        p = dtcwt.Transform1d(biort=DTCWT_BIORT, qshift=DTCWT_QSHIFT).forward(
            np.zeros((N, 1)), nlevels=LEVEL)
        arrs = [np.asarray(p.lowpass)] + [np.asarray(h) for h in p.highpasses]
    elif method == 'DWT+AF':
        arrs = pywt.wavedec(np.zeros(N), DWT_WAVELET, level=LEVEL, mode='symmetric')
    else:
        return 0, 0
    nb = int(sum(a.nbytes for a in arrs))
    nreal = int(sum(a.size * (2 if np.iscomplexobj(a) else 1) for a in arrs))
    return nb, nreal


def traced_peak(scorer, X: np.ndarray):
    """Peak traced allocation (bytes) of one scorer call; X is not counted."""
    scorer(X[:min(10, len(X))])                    # warm-up outside tracing
    gc.collect()
    tracemalloc.start()
    base = tracemalloc.get_traced_memory()[0]
    out = scorer(X)
    peak = tracemalloc.get_traced_memory()[1] - base
    tracemalloc.stop()
    return int(peak), int(np.asarray(out).nbytes)


STATE_ORDER = {
    'AF': ('O(1) with the buffer (sliding weighted sum)', 'AF_t = x_t + AF_{t-1}/2'),
    'EWMA': ('O(1) with the buffer (sliding weighted sum)', 'e_t = a x_t + (1-a) e_{t-1}'),
    'RRD': ('O(1) amortised with the buffer (running sum, first non-zero slot)', None),
    'VSE': ('O(1) amortised with the buffer (running sum, last non-zero slot)', None),
    'CompoundPop': ('O(1) amortised with the buffer (running sums, last non-zero slot)', None),
    'PFRF': ('O(1) with the buffer (running product of a / b factors)', None),
    'DWT+AF': ('O(N): the transform of the whole window is recomputed', None),
    'DTCWT+AF': ('O(N): the transform of the whole window is recomputed', None),
    'WSPI': ('O(N): the transform of the whole window is recomputed', None),
}


def state_memory_table() -> pd.DataFrame:
    rows = []
    for m in METHODS:
        for N in bench_windows(m):
            cb, cr = coef_bytes(m, N)
            upd, rec = STATE_ORDER[m]
            rows.append(dict(method=m, N=N, window_buffer_bytes=N * 8,
                             transient_coef_bytes=cb, transient_coef_real_values=cr,
                             recursive_unwindowed_state_bytes=(8 if rec else np.nan),
                             recursive_form=rec or '', update_time_order=upd,
                             basis='analytic; byte counts exact (float64 / complex128)'))
    return pd.DataFrame(rows)


class RSSSampler:
    """Samples the current RSS in a background thread; max over the samples."""

    def __init__(self, interval=RSS_SAMPLE_S):
        self.interval, self.max, self.n = interval, 0, 0
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            r = current_rss()
            if r is not None:
                self.max = max(self.max, r)
                self.n += 1
            time.sleep(self.interval)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()
        r = current_rss()
        if r is not None:
            self.max = max(self.max, r)


def rss_child(method: str, N: int, M: int, chunk: int) -> dict:
    """Run inside a fresh process: RSS added by scoring M items in batches of chunk."""
    pool = make_pool(N, M, chunk=chunk)
    sc = make_scorer(method, N)
    sc(pool[0][:10])                               # warm-up (filters, imports)
    gc.collect()
    base = current_rss()
    peak_before = peak_rss()
    t = time.perf_counter()
    with RSSSampler() as s:
        score_all(sc, pool, M)
    dt = time.perf_counter() - t
    peak_after = peak_rss()
    return dict(method=method, N=N, M=M, chunk=chunk, n_pool=len(pool),
                input_bytes=int(sum(p.nbytes for p in pool)), rss_base_bytes=base,
                rss_sampled_peak_delta_bytes=s.max - base, n_samples=s.n,
                os_peak_before_bytes=peak_before, os_peak_after_bytes=peak_after,
                os_peak_minus_base_bytes=peak_after - base, seconds=dt)


def run_memory(out: Path, batches=MEM_B, windows=None, rss_windows=RSS_N, rss_cases=RSS_CASES,
               methods=METHODS, verbose=True):
    out.mkdir(parents=True, exist_ok=True)
    t_all = time.time()
    rows = []
    for m in methods:
        for N in bench_windows(m):
            if windows is not None and N not in windows:
                continue
            sc = make_scorer(m, N)
            cb, cr = coef_bytes(m, N)
            for B in batches:
                X = synth_chunk(np.random.default_rng([SEED, N, B]), B, N)
                pk, ob = traced_peak(sc, X)
                rows.append(dict(method=m, N=N, batch=B, traced_peak_bytes=pk,
                                 traced_peak_bytes_per_item=pk / B, output_bytes=ob,
                                 input_bytes=int(X.nbytes), coef_bytes_per_item=cb,
                                 coef_real_values_per_item=cr))
                if verbose:
                    print(f'[trace] N={N:<4} B={B:<7} {m:<12} {pk / B:10.1f} B/item', flush=True)
                del X
    pd.DataFrame(rows).to_csv(out / 'tracemalloc_grid.csv', index=False)
    state_memory_table().to_csv(out / 'state_memory.csv', index=False)
    rss = []
    for N in rss_windows:
        for case, M, ch in rss_cases:
            for m in methods:
                cmd = [sys.executable, str(Path(__file__).resolve()), '--rss-child', m, str(N),
                       str(M), str(ch), '--threads', THREADS]
                env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
                r = subprocess.run(cmd, capture_output=True, text=True, env=env)
                line = [x for x in r.stdout.splitlines() if x.startswith('RSS_JSON ')]
                if r.returncode != 0 or not line:
                    raise RuntimeError(f'rss child failed ({m}, {N}, {M}):\n{r.stderr[-2000:]}')
                d = json.loads(line[-1][len('RSS_JSON '):])
                d['case'] = case
                d['rss_sampled_peak_delta_bytes_per_item'] = d['rss_sampled_peak_delta_bytes'] / min(M, ch)
                rss.append(d)
                if verbose:
                    print(f'[rss]   N={N:<4} {case:<8} M={M:<8} {m:<12} '
                          f'{d["rss_sampled_peak_delta_bytes"] / 2 ** 20:9.1f} MiB', flush=True)
    if rss:
        pd.DataFrame(rss).to_csv(out / 'rss_grid.csv', index=False)
    _write_meta(out, 'memory_run.json', dict(
        part='B memory', batches=list(batches), rss_windows=list(rss_windows),
        rss_cases=[list(c) for c in rss_cases], rss_sample_s=RSS_SAMPLE_S, seed=SEED,
        level=LEVEL, wall_seconds=time.time() - t_all))


# =============================================================================
# Part C: real data with timers
# =============================================================================
class TimedEvaluator(ProtocolV5Evaluator):
    """ProtocolV5Evaluator with timers.  The evaluation itself is unchanged:
    the scorer is wrapped, _matrix and _robustness_v5 call the parent."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._cur_k = None
        self._in_robust = False
        self.score_rec = []                     # (window_id, n_items, seconds)
        self.robust_rec = {}                    # window_id -> seconds

    def _matrix(self, idx, k, W):
        self._cur_k = k
        return super()._matrix(idx, k, W)

    def _robustness_v5(self, fm, X, scores, rng):
        t = time.perf_counter()
        self._in_robust = True
        try:
            return super()._robustness_v5(fm, X, scores, rng)
        finally:
            self._in_robust = False
            self.robust_rec[self._cur_k] = self.robust_rec.get(self._cur_k, 0.0) + \
                time.perf_counter() - t

    def timed(self, fm: FastMethod) -> FastMethod:
        inner = fm.scorer

        def f(X):
            if self._in_robust:
                return inner(X)
            t = time.perf_counter()
            r = inner(X)
            self.score_rec.append((self._cur_k, X.shape[0], time.perf_counter() - t))
            return r
        f.__name__ = inner.__name__
        return FastMethod(fm.name, fm.window_slots, fm.min_obs, f)

    def run_timed(self, fm: FastMethod):
        self.score_rec, self.robust_rec = [], {}
        t = time.perf_counter()
        df = self.run_method(self.timed(fm), out_dir=None)
        total = time.perf_counter() - t
        sr = pd.DataFrame(self.score_rec, columns=['window_id', 'n_scored', 'score_s'])
        return df, sr, dict(self.robust_rec), total


def control_frame(df: pd.DataFrame, ref: pd.DataFrame, method: str) -> dict:
    cols = [c for c in METRIC_COLUMNS + CONTROL_EXTRA if c in df.columns and c in ref.columns]
    a = df.set_index('window_id').sort_index()
    b = ref.set_index('window_id').sort_index()
    same_windows = a.index.equals(b.index)
    maxdiff, maxrel, worst, nan_ok = 0.0, 0.0, '', True
    if same_windows:
        for c in cols:
            x = a[c].to_numpy(dtype=np.float64)
            y = b[c].to_numpy(dtype=np.float64)
            nx, ny = np.isnan(x), np.isnan(y)
            nan_ok &= bool(np.array_equal(nx, ny))
            v = ~nx & ~ny
            if v.any():
                d = np.abs(x[v] - y[v])
                rel = float(np.max(d / np.maximum(1.0, np.abs(y[v]))))
                maxdiff = max(maxdiff, float(d.max()))
                if rel > maxrel:
                    maxrel, worst = rel, c
    return dict(method=method, n_windows=len(a), n_windows_ref=len(b),
                same_windows=bool(same_windows), columns_checked=len(cols),
                nan_pattern_equal=bool(nan_ok), max_abs_diff=maxdiff,
                max_rel_diff=maxrel, worst_column=worst,
                ndcg10_mean=float(a['ndcg@10'].mean()) if len(a) else np.nan,
                ndcg10_mean_ref=float(b['ndcg@10'].mean()) if len(b) else np.nan,
                pass_=bool(same_windows and nan_ok and maxrel <= CONTROL_REL_TOL))


def run_real(a):
    out = _abs(a.out)
    (out / 'protocol').mkdir(parents=True, exist_ok=True)
    t = time.perf_counter()
    ev = TimedEvaluator.from_csv(_abs(a.data), dataset_min_obs=a.min_obs, seed=SEED,
                                 robustness=True, causal_universe=a.causal_universe,
                                 verbose=False)
    load_s = time.perf_counter() - t
    slot_s = ev.slot.total_seconds()
    methods = build_v5_methods(level=LEVEL, names=a.methods)
    ref_dir = _abs(a.ref) if a.ref else None
    timing, summ, ctrl = [], [], []
    for name, fm in methods.items():
        df, sr, rob, total = ev.run_timed(fm)
        if len(sr) != len(df) or not np.array_equal(sr['window_id'].to_numpy(),
                                                    df['window_id'].to_numpy()):
            raise RuntimeError(f'{name}: scorer calls do not match the evaluated windows')
        sr['robust_s'] = sr['window_id'].map(rob).fillna(0.0)
        sr.insert(0, 'method', name)
        sr['num_items'] = df['num_items'].to_numpy()
        timing.append(sr)
        score_s, robust_s = float(sr['score_s'].sum()), float(sr['robust_s'].sum())
        lat = sr['score_s'].to_numpy() * 1e3
        iw = int(sr['n_scored'].sum())
        summ.append(dict(method=name, window_slots=fm.window_slots, n_windows=len(df),
                         item_windows=iw, total_s=total, score_s=score_s, robust_s=robust_s,
                         other_s=total - score_s - robust_s, score_share=score_s / total,
                         us_per_item_window=score_s / iw * 1e6 if iw else np.nan,
                         latency_ms_median=float(np.median(lat)),
                         latency_ms_p95=float(np.percentile(lat, 95)),
                         latency_ms_max=float(lat.max()),
                         items_median=float(np.median(sr['n_scored'])),
                         items_max=int(sr['n_scored'].max()), slot_seconds=slot_s,
                         max_latency_over_slot=float(lat.max() / 1e3 / slot_s)))
        if ref_dir is not None:
            rf = ref_dir / 'protocol' / f'{name}_protocol.csv'
            if rf.exists():
                ctrl.append(control_frame(df, pd.read_csv(rf), name))
            else:
                ctrl.append(dict(method=name, pass_=False, note=f'missing {rf}'))
        print(f'[real] {name:<12} windows={len(df):<6} total {total:8.1f}s  score {score_s:7.2f}s  '
              f'robust {robust_s:7.1f}s  {summ[-1]["us_per_item_window"]:8.3f} us/item  '
              f'latency median {summ[-1]["latency_ms_median"]:.2f} ms max {lat.max():.2f} ms'
              + (f'  control {"OK" if ctrl[-1]["pass_"] else "FAIL"}' if ctrl else ''), flush=True)
    pd.concat(timing, ignore_index=True)[
        ['method', 'window_id', 'num_items', 'n_scored', 'score_s', 'robust_s']].to_csv(
        out / 'protocol' / 'window_timing.csv', index=False)
    sm = pd.DataFrame(summ)
    sm.insert(0, 'load_s', load_s)
    sm.to_csv(out / 'real_summary.csv', index=False)
    if ctrl:
        pd.DataFrame(ctrl).rename(columns={'pass_': 'pass'}).to_csv(out / 'real_control.csv',
                                                                     index=False)
    _write_meta(out, 'runtime_real_run.json', dict(
        part='C real data', data=str(a.data), dataset_min_obs=a.min_obs,
        causal_universe=a.causal_universe, seed=SEED, level=LEVEL, robustness=True,
        reference=str(a.ref) if a.ref else None, num_items=len(ev.items),
        num_slots=ev.num_slots + 1, slot_minutes=slot_s / 60, load_s=load_s,
        os_peak_rss_bytes=peak_rss()))
    if ctrl:
        ok = all(c['pass_'] for c in ctrl)
        print(f'control against T1.5: {"ALL PASS" if ok else "FAIL"}')


# =============================================================================
# Collect
# =============================================================================
def collect(root: Path):
    root = _abs(root)
    grid = pd.read_csv(root / 'bench' / 'runtime_grid.csv')
    tr = pd.read_csv(root / 'memory' / 'tracemalloc_grid.csv')
    rss_f = root / 'memory' / 'rss_grid.csv'
    rss = pd.read_csv(rss_f) if rss_f.exists() else None
    reals, ctrls = [], []
    for sc in SCENARIOS:
        d = root / 'real' / sc
        if (d / 'real_summary.csv').exists():
            s = pd.read_csv(d / 'real_summary.csv')
            s.insert(0, 'scenario', sc)
            reals.append(s)
        if (d / 'real_control.csv').exists():
            c = pd.read_csv(d / 'real_control.csv')
            c.insert(0, 'scenario', sc)
            ctrls.append(c)
    real = pd.concat(reals, ignore_index=True) if reals else None
    if real is not None:
        real.to_csv(root / 'real_summary_all.csv', index=False)
    if ctrls:
        cc = pd.concat(ctrls, ignore_index=True)
        cc.to_csv(root / 'real_control_all.csv', index=False)
        print(cc[['scenario', 'method', 'same_windows', 'max_abs_diff', 'max_rel_diff', 'worst_column',
                  'pass']].to_string(index=False))
        print(f'control: {int(cc["pass"].sum())} of {len(cc)} pass')

    def g(m, N, M, col='us_per_item_median'):
        r = grid[(grid['method'] == m) & (grid['N'] == N) & (grid['M'] == M)]
        return float(r[col].iloc[0]) if len(r) else np.nan

    def t(m, N, B):
        r = tr[(tr['method'] == m) & (tr['N'] == N) & (tr['batch'] == B)]
        return float(r['traced_peak_bytes_per_item'].iloc[0]) if len(r) else np.nan

    rows = []
    for m in METHODS:
        Nd = DEFAULT_N[m]
        cb, cr = coef_bytes(m, 64)
        r = dict(method=m, default_N=Nd,
                 us_per_item_default_M1e4=g(m, Nd, 10_000), us_per_item_default_M1e6=g(m, Nd, 1_000_000),
                 s_all_items_default_M1e6=g(m, Nd, 1_000_000, 'median_s'),
                 ms_single_item_call_default=g(m, Nd, 1, 'median_s') * 1e3,
                 us_per_item_N64_M1e4=g(m, 64, 10_000), us_per_item_N64_M1e6=g(m, 64, 1_000_000),
                 us_per_item_N256_M1e4=g(m, 256, 10_000),
                 traced_bytes_per_item_default_B1e4=t(m, Nd, 10_000),
                 traced_bytes_per_item_N64_B1e4=t(m, 64, 10_000),
                 coef_bytes_per_item_N64=cb, window_buffer_bytes_default=Nd * 8)
        if rss is not None:
            q = rss[(rss['method'] == m) & (rss['N'] == 64) & (rss['case'] == 'chunked')]
            r['rss_delta_MiB_N64_M1e6_chunked'] = (float(q['rss_sampled_peak_delta_bytes'].iloc[0]) / 2 ** 20
                                                   if len(q) else np.nan)
        if real is not None:
            for sc in SCENARIOS:
                q = real[(real['scenario'] == sc) & (real['method'] == m)]
                r[f'latency_ms_median_{sc}'] = float(q['latency_ms_median'].iloc[0]) if len(q) else np.nan
                r[f'latency_ms_max_{sc}'] = float(q['latency_ms_max'].iloc[0]) if len(q) else np.nan
                r[f'us_per_item_real_{sc}'] = float(q['us_per_item_window'].iloc[0]) if len(q) else np.nan
        rows.append(r)
    pt = pd.DataFrame(rows)
    pt.to_csv(root / 'runtime_paper_table.csv', index=False)
    with pd.option_context('display.width', 250, 'display.max_columns', 12):
        print(pt[['method', 'default_N', 'us_per_item_default_M1e4', 'us_per_item_N64_M1e4',
                  'ms_single_item_call_default', 'traced_bytes_per_item_default_B1e4',
                  'coef_bytes_per_item_N64']].to_string(index=False))
    print(f'written: {root / "runtime_paper_table.csv"}')


# =============================================================================
def _write_meta(out: Path, name: str, d: dict):
    md = out / 'metadata'
    md.mkdir(parents=True, exist_ok=True)
    d = dict(d, task='T3.8 (E9)', threads=THREADS, hardware=hardware_info(),
             script='tools/run_runtime_v5.py')
    (md / name).write_text(json.dumps(d, indent=2, default=str), encoding='utf-8')


def _ints(s):
    return tuple(int(x) for x in s.split(',')) if s else None


def main():
    if '--rss-child' in sys.argv:                 # internal: one measurement per process
        i = sys.argv.index('--rss-child')
        m, N, M, ch = sys.argv[i + 1], int(sys.argv[i + 2]), int(sys.argv[i + 3]), int(sys.argv[i + 4])
        print('RSS_JSON ' + json.dumps(rss_child(m, N, M, ch)))
        return
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--threads', default='1', help="'1' (paper) or 'default'")
    ap.add_argument('--hardware', action='store_true', help='print hardware / versions and exit')
    ap.add_argument('--bench', action='store_true', help='part A (synthetic method cost)')
    ap.add_argument('--memory', action='store_true', help='part B (memory)')
    ap.add_argument('--real', action='store_true', help='part C (one real scenario)')
    ap.add_argument('--collect', help='T3.8 root: paper table, real summary and control')
    ap.add_argument('--out', help='output folder')
    ap.add_argument('--data', help='dataset CSV (part C)')
    ap.add_argument('--min-obs', type=int, help='catalogue threshold (YouTube 50, taxi 24)')
    ap.add_argument('--causal-universe', action='store_true', help='paper setting (T1.5)')
    ap.add_argument('--ref', help='T1.5 causal run folder of the scenario (default: from --out name)')
    ap.add_argument('--methods', nargs='*', default=None)
    ap.add_argument('--items', help='comma list of M (default 1,1000,10000,100000,1000000)')
    ap.add_argument('--windows', help='comma list of N (default all)')
    ap.add_argument('--repeats', type=int, default=REPEATS)
    ap.add_argument('--resume', action='store_true', help='part A: keep finished rows')
    a = ap.parse_args()
    if a.hardware:
        print(json.dumps(hardware_info(), indent=2, default=str))
        return
    if a.collect:
        collect(Path(a.collect))
        return
    if not a.out:
        ap.error('--out is required')
    methods = a.methods or METHODS
    if a.bench:
        run_bench(_abs(a.out), items=_ints(a.items) or BENCH_M, windows=_ints(a.windows),
                  repeats=a.repeats, methods=methods, resume=a.resume)
    elif a.memory:
        run_memory(_abs(a.out), windows=_ints(a.windows), methods=methods)
    elif a.real:
        if not (a.data and a.min_obs):
            ap.error('--real needs --data and --min-obs')
        if a.ref is None:
            cand = REF_T15 / Path(a.out).name
            a.ref = str(cand) if _abs(cand).exists() else None
        run_real(a)
    else:
        ap.error('choose one of --hardware, --bench, --memory, --real, --collect')


if __name__ == '__main__':
    main()
