r"""
Resume support and bound check for the prediction runs
======================================================
Used by ``tools/run_prediction.py`` (``--resume``, ``--bound-check``).  Nothing
here changes a score: the evaluator, the scorers and the result files of a run
are the same with and without these options (``tools/test_prediction_resume.py``
compares the files byte by byte).

Resume.  A long run can be stopped at any time (closed window, power cut) and
started again with the same command.  The state is kept outside the result
folder, in ``--resume-dir`` (default ``logs/prediction_resume/<out>/h<h>/``):

  config.json            what the state belongs to (data file hash, settings,
                         hashes of the code files).  A different run is refused.
  <method>.done.json     written after the protocol CSV of a method is complete:
                         size and SHA-256 of the CSV, the run time and the
                         scorer statistics.  A method with a valid record is
                         not run again; its CSV is read back
                         (``float_precision='round_trip'``, exact).
  <method>.journal       only for the slow methods (WSPI-FA, ARIMA; hours per
                         method): every scorer call is appended (hash of the
                         input, the scores as float64, seconds, fits).  After
                         an interruption the evaluator starts the method again
                         from the first window, but the scorer answers from the
                         journal until it is used up, so only the windows that
                         were not scored yet cost time.  The journal is deleted
                         when the method is complete.
  complete.json          every output of the horizon is written; the horizon is
                         skipped on the next start.

Each method is evaluated independently by the evaluator (own random state for
the robustness test, own RSI memory), so skipping complete methods and
replaying exact float64 scores gives the same files as one uninterrupted run.

Bound check.  For a clamped coefficient-domain method the ratio
P_method / P_WSPI lies in [L, U] with
    L = 2^-m W_n / W_(n+m),   U = B(m, c) W_n / W_(n+m)
(``methods.wspi_forecast.ratio_bounds``; n = 16 trend coefficients, c = 4).
``BoundRecorder`` wraps the scorer of such a method, computes the WSPI score of
the same rows and keeps the smallest and largest ratio over every row the
scorer sees (all items of all windows and the series of the robustness test).

Author: Sajjad Pirahesh
Date: November 2026
"""
from __future__ import annotations

import hashlib
import json
import os
import struct
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from evaluation.fast_evaluator import FastMethod
from evaluation.protocol_v5 import ProtocolV5Evaluator, make_v5_wspi
from methods.wspi_forecast import CLAMP, n_forecast_coefficients, ratio_bounds

ROOT = Path(__file__).resolve().parents[1]

JOURNAL_METHODS = ('WSPI-FA', 'ARIMA')            # slow: one MLE fit per row
CLAMPED_METHODS = ('WSPI-F2', 'WSPI-FT', 'WSPI-FG', 'WSPI-FA')
CODE_FILES = ('evaluation/protocol_v5.py', 'evaluation/fast_evaluator.py',
              'tools/prediction_protocol.py', 'methods/wspi_forecast.py',
              'methods/forecasting_baselines.py')
MAGIC = b'PRJRNL01'
HEAD = struct.Struct('<q20sdqq')                  # rows, sha1(X), seconds, fits, fallbacks
FAILED_CALL = -1
SYNC_SECONDS = 5.0
BOUND_RTOL = 1e-9
N_TREND = 16                                      # trend coefficients of a 64-slot window, J = 3


class ResumeError(SystemExit):
    """Stops the run.  Derived from SystemExit on purpose: the evaluator
    catches ``Exception`` around a scorer call and would turn an ordinary
    error into scores of zero."""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json_atomic(path: Path, obj) -> None:
    tmp = path.with_name(path.name + '.tmp')
    with open(tmp, 'w', encoding='utf-8') as fh:
        json.dump(obj, fh, indent=2)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _digest(X: np.ndarray) -> bytes:
    X = np.ascontiguousarray(X, dtype=np.float64)
    h = hashlib.sha1(np.asarray(X.shape, dtype=np.int64).tobytes())
    h.update(X.tobytes())
    return h.digest()


# =============================================================================
# Journal of scorer calls
# =============================================================================

class ScoreJournal:
    """Append-only binary log of the calls of one scorer."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.records: List[tuple] = []            # (rows, digest, seconds, fits, fallbacks, scores)
        self._fh = None
        self._last_sync = 0.0
        self._load()

    def _load(self):
        if not self.path.exists():
            return
        data = self.path.read_bytes()
        if data[:len(MAGIC)] != MAGIC:
            raise ResumeError(f'{self.path} is not a score journal; delete it to start again')
        pos = good = len(MAGIC)
        while pos + HEAD.size <= len(data):
            rows, dg, sec, fits, fb = HEAD.unpack_from(data, pos)
            n = max(rows, 0)
            end = pos + HEAD.size + 8 * n
            if end > len(data):
                break                             # incomplete last record (interrupted write)
            scores = np.frombuffer(data, dtype='<f8', count=n, offset=pos + HEAD.size).copy()
            self.records.append((rows, dg, sec, fits, fb, scores))
            pos = good = end
        if good < len(data):
            with open(self.path, 'r+b') as fh:
                fh.truncate(good)

    def append(self, rows, dg, sec, fits, fb, scores):
        if self._fh is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            new = not self.path.exists()
            self._fh = open(self.path, 'ab')
            if new:
                self._fh.write(MAGIC)
        self._fh.write(HEAD.pack(int(rows), dg, float(sec), int(fits), int(fb)))
        if rows > 0:
            self._fh.write(np.asarray(scores, dtype='<f8').tobytes())
        now = time.time()
        if now - self._last_sync >= SYNC_SECONDS:
            self._fh.flush()
            os.fsync(self._fh.fileno())
            self._last_sync = now

    def close(self):
        if self._fh is not None:
            self._fh.flush()
            os.fsync(self._fh.fileno())
            self._fh.close()
            self._fh = None

    def delete(self):
        self.close()
        if self.path.exists():
            self.path.unlink()


class JournalScorer:
    """Scorer that answers from the journal while it lasts and appends every
    new call to it.  ``inner`` is a ``TimedScorer`` (or any scorer)."""

    def __init__(self, inner: Callable, path: Path):
        self.inner = inner
        self.__name__ = getattr(inner, '__name__', 'scorer')
        self.path = Path(path)
        self.journal: Optional[ScoreJournal] = None
        self.pos = 0
        self.replayed = dict(calls=0, rows=0, seconds=0.0, fits=0, fallbacks=0)
        self.replay_wall = 0.0

    def _counts(self):
        f = getattr(getattr(self.inner, 'scorer', self.inner), 'fallback_counts', None)
        return f() if f is not None else (0, 0)

    def __call__(self, X):
        t_in = time.perf_counter()
        if self.journal is None:
            self.journal = ScoreJournal(self.path)
        X = np.asarray(X, dtype=np.float64)
        dg = _digest(X)
        if self.pos < len(self.journal.records):
            rows, dg0, sec, fits, fb, scores = self.journal.records[self.pos]
            if dg0 != dg or (rows != FAILED_CALL and rows != X.shape[0]):
                raise ResumeError(
                    f'{self.path}: call {self.pos} does not match the stored call '
                    f'(different data, settings or code).  Delete the resume folder '
                    f'{self.path.parent} to start this horizon again.')
            self.pos += 1
            self.replayed['calls'] += 1
            self.replayed['rows'] += int(X.shape[0])
            self.replayed['seconds'] += sec
            self.replayed['fits'] += fits
            self.replayed['fallbacks'] += fb
            self.replay_wall += time.perf_counter() - t_in
            if rows == FAILED_CALL:
                raise RuntimeError('scorer call failed in the interrupted run (replayed)')
            return scores
        f0, b0 = self._counts()
        t0 = time.perf_counter()
        try:
            out = np.asarray(self.inner(X), dtype=np.float64)
        except Exception:
            f1, b1 = self._counts()
            self.journal.append(FAILED_CALL, dg, time.perf_counter() - t0, f1 - f0, b1 - b0, None)
            self.pos += 1
            raise
        f1, b1 = self._counts()
        self.journal.append(X.shape[0], dg, time.perf_counter() - t0, f1 - f0, b1 - b0, out)
        self.pos += 1
        return out

    def stats(self) -> dict:
        """Statistics of the whole method: the calls of this session plus the
        replayed calls of the earlier sessions."""
        base = self.inner.stats() if hasattr(self.inner, 'stats') else dict(
            scorer=self.__name__, calls=0, rows=0, seconds=0.0, ms_per_row=float('nan'))
        r = self.replayed
        out = dict(base)
        out['calls'] = base['calls'] + r['calls']
        out['rows'] = base['rows'] + r['rows']
        out['seconds'] = base['seconds'] + r['seconds']
        out['ms_per_row'] = (1000.0 * out['seconds'] / out['rows']) if out['rows'] else float('nan')
        if 'n_fits' in base:
            out['n_fits'] = base['n_fits'] + r['fits']
            out['n_fallbacks'] = base['n_fallbacks'] + r['fallbacks']
            out['fallback_share'] = (out['n_fallbacks'] / out['n_fits']) if out['n_fits'] else float('nan')
        out['replayed_calls'] = r['calls']
        return out


# =============================================================================
# Bound check
# =============================================================================

class BoundRecorder:
    """Keeps min and max of P_method / P_WSPI over every row the scorer sees."""

    def __init__(self, inner: Callable, method: str, horizon: int, window: int = 64,
                 level: int = 3, clamp: float = CLAMP, n_trend: int = N_TREND):
        self.inner = inner
        self.__name__ = getattr(inner, '__name__', 'scorer')
        self.method = method
        self.horizon = int(horizon)
        self.m = n_forecast_coefficients(horizon, level)
        self.n_trend = n_trend
        self.clamp = clamp
        self.lower, self.upper = ratio_bounds(n_trend, self.m, clamp)
        self._wspi = make_v5_wspi(window=window, level=level)
        self.n = 0
        self.min = float('inf')
        self.max = float('-inf')
        self.n_below = 0
        self.n_above = 0
        self.n_ref_zero = 0
        self.n_ref_zero_nonzero = 0
        self.n_nonfinite = 0

    def __call__(self, X):
        s = self.inner(X)
        sv = np.asarray(s, dtype=np.float64)
        w = np.asarray(self._wspi(np.asarray(X, dtype=np.float64)), dtype=np.float64)
        fin = np.isfinite(sv) & np.isfinite(w)
        self.n_nonfinite += int((~fin).sum())
        pos = fin & (w > 0)
        zero = fin & ~pos
        self.n_ref_zero += int(zero.sum())
        self.n_ref_zero_nonzero += int((sv[zero] != 0).sum())
        if pos.any():
            r = sv[pos] / w[pos]
            self.n += int(r.size)
            self.min = min(self.min, float(r.min()))
            self.max = max(self.max, float(r.max()))
            self.n_below += int((r < self.lower * (1.0 - BOUND_RTOL)).sum())
            self.n_above += int((r > self.upper * (1.0 + BOUND_RTOL)).sum())
        return s

    def stats(self) -> dict:
        return self.inner.stats()

    def bound_stats(self) -> dict:
        bad = self.n_below + self.n_above + self.n_ref_zero_nonzero + self.n_nonfinite
        return dict(horizon=self.horizon, method=self.method, m=self.m, n_trend=self.n_trend,
                    clamp=self.clamp, lower_bound=self.lower, upper_bound=self.upper,
                    n_ratios=self.n,
                    min_ratio=self.min if self.n else float('nan'),
                    max_ratio=self.max if self.n else float('nan'),
                    n_below=self.n_below, n_above=self.n_above,
                    n_reference_zero=self.n_ref_zero,
                    n_nonzero_at_reference_zero=self.n_ref_zero_nonzero,
                    n_nonfinite=self.n_nonfinite,
                    status='OK' if bad == 0 else 'VIOLATION')


def add_bound_recorders(methods: Dict[str, FastMethod], horizon: int):
    """Wrap the scorer of every clamped method.  Returns (methods, recorders)."""
    out, rec = {}, {}
    for name, fm in methods.items():
        if name in CLAMPED_METHODS:
            rec[name] = BoundRecorder(fm.scorer, name, horizon, window=fm.window_slots)
            out[name] = FastMethod(fm.name, fm.window_slots, fm.min_obs, rec[name])
        else:
            out[name] = fm
    return out, rec


# =============================================================================
# State of one horizon
# =============================================================================

def default_resume_dir(out_h: Path) -> Path:
    out_h = Path(out_h).resolve()
    try:
        rel = out_h.relative_to(ROOT)
        key = '__'.join(rel.parts)
    except ValueError:
        key = hashlib.sha1(str(out_h).encode('utf-8')).hexdigest()[:16] + '__' + out_h.name
    return ROOT / 'logs' / 'prediction_resume' / key


class ResumeState:
    """Resume state of one (run, horizon)."""

    def __init__(self, directory: Path, config: dict, required_files: List[str]):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.required_files = list(required_files)
        self.journals: Dict[str, JournalScorer] = {}
        self.recorders: Dict[str, BoundRecorder] = {}
        self.done: Dict[str, dict] = {}           # records read or written in this session
        self.skipped: List[str] = []
        cf = self.dir / 'config.json'
        if cf.exists():
            old = json.loads(cf.read_text(encoding='utf-8'))
            diff = sorted(k for k in set(old) | set(config) if old.get(k) != config.get(k))
            if diff:
                raise ResumeError(
                    f'The resume state in {self.dir} belongs to a different run '
                    f'(differs in: {", ".join(diff)}).  Delete that folder to start again, '
                    f'or run without --resume.')
        else:
            write_json_atomic(cf, config)

    # ---- horizon ------------------------------------------------------------
    def horizon_complete(self, out_h: Path) -> bool:
        if not (self.dir / 'complete.json').exists():
            return False
        return all((Path(out_h) / f).exists() for f in self.required_files)

    def mark_horizon_complete(self, info: Optional[dict] = None):
        write_json_atomic(self.dir / 'complete.json',
                          dict(created=time.strftime('%Y-%m-%d %H:%M:%S'), **(info or {})))

    # ---- methods ------------------------------------------------------------
    def wrap_journals(self, methods: Dict[str, FastMethod]) -> Dict[str, FastMethod]:
        out = {}
        for name, fm in methods.items():
            if name in JOURNAL_METHODS:
                js = JournalScorer(fm.scorer, self.dir / f'{name}.journal')
                self.journals[name] = js
                out[name] = FastMethod(fm.name, fm.window_slots, fm.min_obs, js)
            else:
                out[name] = fm
        return out

    def load_done(self, name: str, out_dir: Path) -> Optional[pd.DataFrame]:
        rf = self.dir / f'{name}.done.json'
        csv = Path(out_dir) / 'protocol' / f'{name}_protocol.csv'
        if not rf.exists() or not csv.exists():
            return None
        rec = json.loads(rf.read_text(encoding='utf-8'))
        if csv.stat().st_size != rec['csv_bytes'] or sha256_file(csv) != rec['csv_sha256']:
            return None                           # file changed or incomplete: run again
        df = pd.read_csv(csv, float_precision='round_trip')
        df.attrs['duration_s'] = rec['duration_s']
        self.done[name] = rec
        self.skipped.append(name)
        return df

    def mark_done(self, name: str, fm: FastMethod, df: pd.DataFrame, out_dir: Path):
        csv = Path(out_dir) / 'protocol' / f'{name}_protocol.csv'
        st = fm.scorer.stats() if hasattr(fm.scorer, 'stats') else {}
        rec = dict(method=name, windows=int(len(df)), csv_bytes=csv.stat().st_size,
                   csv_sha256=sha256_file(csv), duration_s=float(df.attrs['duration_s']),
                   stats=st, bound=(self.recorders[name].bound_stats()
                                    if name in self.recorders else None),
                   created=time.strftime('%Y-%m-%d %H:%M:%S'))
        write_json_atomic(self.dir / f'{name}.done.json', rec)
        self.done[name] = rec
        if name in self.journals and self.journals[name].journal is not None:
            self.journals[name].journal.delete()


class ResumableEvaluator(ProtocolV5Evaluator):
    """ProtocolV5Evaluator that skips the methods whose result file is complete.
    With ``resume_state=None`` it is the plain evaluator."""

    def __init__(self, *args, resume_state: Optional[ResumeState] = None, **kw):
        super().__init__(*args, **kw)
        self.resume_state = resume_state

    def run_method(self, fm: FastMethod, out_dir=None) -> pd.DataFrame:
        st = self.resume_state
        if st is None or out_dir is None:
            return super().run_method(fm, out_dir)
        df = st.load_done(fm.name, out_dir)
        if df is not None:
            if self.verbose:
                print(f'[v5] {fm.name:<12} complete ({len(df)} windows), not run again')
            return df
        df = super().run_method(fm, out_dir)
        js = st.journals.get(fm.name)
        if js is not None and js.replayed['calls']:
            # time of an uninterrupted run: this session without the replay,
            # plus the scorer time of the replayed calls
            df.attrs['duration_s'] = (df.attrs['duration_s'] - js.replay_wall
                                      + js.replayed['seconds'])
        st.mark_done(fm.name, fm, df, out_dir)
        return df
