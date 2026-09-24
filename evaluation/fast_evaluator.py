r"""
Fast vectorised evaluator (revision-srep-v5, task T1.2)
=======================================================
A drop-in, much faster re-implementation of the per-window loop of
``evaluation/incremental_evaluator.py`` (IncrementalTemporalEvaluator).

Design
------
* The data are turned ONCE into an item x slot matrix on the regular time
  grid (slot = median timestamp step, exactly as the old evaluator's
  "granularity-fix" does).
* For every window, the series of every item is cut from that matrix with
  pure index arithmetic (cumulative presence counts) — no DataFrame
  filtering inside the loop.
* Items that share the same series length are scored together with a
  batched (row-wise) version of each method: AF, EWMA, RRD, VSE,
  CompoundPop, PFRF, DWT+AF, DTCWT+AF and WSPI.  Any other method object
  with ``assess_single`` is still supported through a generic (slow)
  per-item fallback.
* All window-level metrics are computed with the SAME functions of
  ``evaluation/metrics.py`` (calculate_ndcg, calculate_coverage,
  calculate_diagnostics, calculate_rsi, calculate_rank_distortion), in the
  same order, on arrays in the same item order.

Two modes
---------
``mode='compat'`` (default) reproduces the current behaviour exactly:
  - training slice is ``[t - (W-1), t)``  -> at most W-1 slots
    (63 for the wavelet methods, 6 for the baselines);
  - the series of an item is the list of ROWS that exist in the file inside
    that slice, in time order.  Slots that are missing from the CSV are
    skipped, not zero-filled (both YouTube and Taxi files are sparse);
  - an item enters a window only if it has >= min_observations rows there;
  - a window is skipped if no selected item has a row in the train slice
    or in the test slot;
  - RSI keeps the positional top-K indices (as the old code does);
  - DTCWT+AF uses the level from EvaluationConfig (the old runs used 2),
    DWT+AF uses DWTAssessment._safe_level, WSPI uses J=3.
  - Robustness (Layer 4) follows the same algorithm, but the candidate
    sampling is seeded (``seed``) — the old code used the unseeded global
    NumPy RNG, so that column can only be matched in distribution.

``mode='dense'`` is the corrected protocol for T1.4 (NOT used for G1):
  - training slice is ``[t - W, t)`` -> exactly W slots;
  - missing slots are zero-filled;
  - eligibility: the item needs >= min_observations observed rows in the
    slice (kept identical to compat so the item sets stay comparable);
  - ``rsi_by_item=True`` is recommended (top-K compared by item id).

Output
------
One protocol CSV per method with exactly the old columns:
    window_id,timestamp,method,num_items,ndcg@5,coverage@5,...,rsi@20,
    robustness_distortion
written to ``<out_dir>/protocol/<method>_protocol.csv``; plus
``<out_dir>/metadata/fast_run.json`` (settings + runtime per method).

Usage
-----
    from evaluation.fast_evaluator import FastEvaluator, build_default_methods
    fe = FastEvaluator.from_csv('data/datasets/youtube_hourly.csv',
                                dataset_min_obs=50)
    fe.run(build_default_methods(dtcwt_level=2), out_dir='results/revision_v5/x')

or from the command line (see ``tools/run_fast_eval.py``).

Author: Sajjad (with assistance), September 2026
"""
from __future__ import annotations

import json
import math
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import pywt
import dtcwt

from .metrics import (
    calculate_ndcg,
    calculate_coverage,
    calculate_diagnostics,
    calculate_rsi,
    calculate_rank_distortion,
)

K_VALUES = [5, 10, 20]

PROTOCOL_COLUMNS = [
    'window_id', 'timestamp', 'method', 'num_items',
    'ndcg@5', 'coverage@5', 'ndcg@10', 'coverage@10', 'ndcg@20', 'coverage@20',
    'kendall_tau', 'spearman_rho', 'mae',
    'rsi@5', 'rsi@10', 'rsi@20', 'robustness_distortion',
]


# =============================================================================
# Batched scorers.  Each takes X (n_items x L, float64, oldest -> newest)
# and returns a 1-D float64 array of n_items scores.  Each one is a
# row-wise replica of the corresponding assess_single().
# =============================================================================

def _recency_weights(n: int) -> np.ndarray:
    """w[j] = 2^-(n-1-j): newest sample weight 1."""
    return 2.0 ** (-np.arange(n)[::-1].astype(np.float64))


def batch_af(X: np.ndarray) -> np.ndarray:
    # TraditionalBaselines.access_frequency: sum_i 2^-i * x[t-i]
    # summed newest -> oldest, like the original loop
    n = X.shape[1]
    out = np.zeros(X.shape[0])
    for i in range(n):
        out = out + (2.0 ** (-i)) * X[:, n - 1 - i]
    return out


def batch_ewma(X: np.ndarray, alpha: float = 0.2) -> np.ndarray:
    e = X[:, 0].astype(np.float64).copy()
    for j in range(1, X.shape[1]):
        e = alpha * X[:, j] + (1 - alpha) * e
    return e


def _first_last_nonzero(X):
    nz = X != 0
    has = nz.any(axis=1)
    first = np.argmax(nz, axis=1)
    last = X.shape[1] - 1 - np.argmax(nz[:, ::-1], axis=1)
    return has, first, last


def batch_rrd(X: np.ndarray) -> np.ndarray:
    n = X.shape[1]
    tot = X.sum(axis=1)
    has, first, _ = _first_last_nonzero(X)
    life = n - first
    out = np.where(has & (life > 0), tot / np.maximum(life, 1), 0.0)
    return out.astype(np.float64)


def batch_vse(X: np.ndarray) -> np.ndarray:
    n = X.shape[1]
    tot = X.sum(axis=1)
    has, _, last = _first_last_nonzero(X)
    dist = n - 1 - last
    return np.where(has, tot * (1.0 / (dist + 1.0)), 0.0)


def batch_compound(X: np.ndarray, c1=0.5, c2=0.3, c3=0.2) -> np.ndarray:
    n = X.shape[1]
    tot = X.sum(axis=1)
    rn = max(1, int(n * 0.2))
    rec = X[:, -rn:].sum(axis=1)
    has, _, last = _first_last_nonzero(X)
    dist = n - 1 - last
    recency = 1.0 / (dist + 1.0)
    s = c1 * (tot / n) + c2 * (rec / rn) + c3 * recency
    return np.where(has, s, 0.0)


def batch_pfrf(X: np.ndarray, a=1.2, b=0.8) -> np.ndarray:
    w = np.ones(X.shape[0])
    for j in range(X.shape[1]):
        w = w * np.where(X[:, j] > 0, a, b)
    return w


def _dwt_safe_level(n: int, level: int, wavelet: str = 'db4') -> int:
    """Replica of DWTAssessment._safe_level."""
    fl = pywt.Wavelet(wavelet).dec_len
    if fl <= 1 or n <= 0:
        return 1
    ms = int(math.floor(math.log2(n / (fl - 1)))) if n >= fl else 1
    return max(1, min(level, ms))


def _waf_rows(C: np.ndarray) -> np.ndarray:
    """sum_i 2^-i |C[t-i]|, newest first — replica of _apply_weighted_af."""
    n = C.shape[1]
    out = np.zeros(C.shape[0])
    for i in range(n):
        out = out + (2.0 ** (-i)) * np.abs(C[:, n - 1 - i])
    return out


def make_batch_dwt(level: int = 3, wavelet: str = 'db4', mode: str = 'symmetric',
                   detail_weight: float = 0.1) -> Callable:
    def f(X):
        L = X.shape[1]
        lv = _dwt_safe_level(L, level, wavelet)
        min_len = 2 ** lv
        Xp = np.pad(X, ((0, 0), (min_len - L, 0)), mode='edge') if L < min_len else X
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning, module='pywt')
            c = pywt.wavedec(Xp, wavelet, level=lv, mode=mode, axis=-1)
        return _waf_rows(c[0]) + detail_weight * _waf_rows(c[-1])
    f.__name__ = f'dwt_L{level}'
    return f


def _pow2_target(L: int, level: int) -> int:
    return max(2 ** (level + 1), 2 ** int(np.ceil(np.log2(max(L, 2)))))


def _dtcwt_forward_rows(transform, Xp: np.ndarray, level: int):
    """dtcwt 1-D transforms COLUMNS, so feed X.T and return per-row bands."""
    p = transform.forward(np.ascontiguousarray(Xp.T), nlevels=level)
    # C-contiguous rows so that numpy reductions use the same (pairwise)
    # summation order as the 1-D per-item code -> bit-identical sums
    low = np.ascontiguousarray(np.asarray(p.lowpass).T)      # (n_items, n_low)
    highs = [np.ascontiguousarray(np.asarray(h).T) for h in p.highpasses]
    return low, highs


def make_batch_dtcwt_af(level: int = 2, biort='near_sym_a', qshift='qshift_a',
                        detail_weight: float = 0.1) -> Callable:
    tr = dtcwt.Transform1d(biort=biort, qshift=qshift)

    def core(Xp):
        low, highs = _dtcwt_forward_rows(tr, Xp, level)
        return _waf_rows(np.abs(low)) + detail_weight * _waf_rows(np.abs(highs[0]))

    def f(X):
        L = X.shape[1]
        # DTCWTAssessment: target = max(2^(J+1), 2^ceil(log2 L)), 'edge' on the left
        tgt = max(2 ** (level + 1), 2 ** int(np.ceil(np.log2(L))))
        Xp = np.pad(X, ((0, 0), (tgt - L, 0)), mode='edge') if L < tgt else X
        return core(Xp)
    f.__name__ = f'dtcwt_af_L{level}'
    f.core, f.pad_mode = core, 'edge'
    f.target_len = lambda L: max(2 ** (level + 1), 2 ** int(np.ceil(np.log2(L))))
    return f


def make_batch_wspi(level: int = 3, alpha: float = 1.0, beta: float = 1.0,
                    use_R: bool = True, use_WE: bool = True,
                    biort='near_sym_a', qshift='qshift_a') -> Callable:
    tr = dtcwt.Transform1d(biort=biort, qshift=qshift)

    def f(X):
        L = X.shape[1]
        tgt = _pow2_target(L, level)
        # WSPIAssessment pads with 'reflect' on the left
        Xp = np.pad(X, ((0, 0), (tgt - L, 0)), mode='reflect') if L < tgt else X
        return core(Xp)

    def core(Xp):
        low, highs = _dtcwt_forward_rows(tr, Xp, level)
        lm = np.abs(low)
        n = lm.shape[1]
        w = 2.0 ** -np.arange(n)[::-1]
        mu = (lm * w).sum(axis=1) / w.sum()
        e_low = (lm ** 2).sum(axis=1)
        e_h = np.stack([(np.abs(h) ** 2).sum(axis=1) for h in highs], axis=1)
        e_tot = e_low + e_h.sum(axis=1)
        R = np.where(e_tot > 0, e_low / np.where(e_tot > 0, e_tot, 1.0), 0.0)
        E = np.concatenate([e_low[:, None], e_h], axis=1)      # (n, J+1)
        tot = E.sum(axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            P = E / np.where(tot > 0, tot, 1.0)[:, None]
            term = np.where(P > 0, P * np.log2(np.where(P > 0, P, 1.0)), 0.0)
        ent = -term.sum(axis=1)
        max_ent = np.log2(E.shape[1])
        WE = np.where(tot > 0, ent / max_ent if max_ent > 0 else 0.0, 0.0)
        expo = np.zeros(Xp.shape[0])
        if use_R:
            expo = expo + alpha * R
        if use_WE:
            expo = expo - beta * WE
        return mu * np.exp(expo)
    f.__name__ = f'wspi_L{level}'
    f.core, f.pad_mode = core, 'reflect'
    f.target_len = lambda L: _pow2_target(L, level)
    return f


def make_generic(method_obj) -> Callable:
    """Fallback: call the original object's assess_single row by row."""
    def f(X):
        out = np.empty(X.shape[0])
        for i in range(X.shape[0]):
            try:
                s = method_obj.assess_single(X[i])
                out[i] = float(np.mean(s) if isinstance(s, (list, np.ndarray)) else s)
            except Exception:
                out[i] = 0.0
        return out
    f.__name__ = f'generic_{getattr(method_obj, "name", "method")}'
    return f


@dataclass
class FastMethod:
    name: str
    window_slots: int
    min_obs: int
    scorer: Callable


def build_default_methods(dtcwt_level: int = 2, wspi_level: int = 3,
                          dwt_level: int = 3, window_override: Optional[Dict[str, int]] = None,
                          names: Optional[List[str]] = None) -> Dict[str, FastMethod]:
    """The 9 methods of the paper with the settings of the V4 runs.

    dtcwt_level=2 reproduces the V4 runs (EvaluationConfig resolved
    'auto' to 2 with window_size=30).  WSPI hard-codes J=3.
    """
    from .method_configs import METHOD_CONFIGS as MC
    specs = {
        'AF': batch_af, 'EWMA': batch_ewma, 'RRD': batch_rrd, 'VSE': batch_vse,
        'CompoundPop': batch_compound, 'PFRF': batch_pfrf,
        'DWT+AF': make_batch_dwt(level=dwt_level),
        'DTCWT+AF': make_batch_dtcwt_af(level=dtcwt_level),
        'WSPI': make_batch_wspi(level=wspi_level),
    }
    window_override = window_override or {}
    out = {}
    for n, fn in specs.items():
        if names and n not in names:
            continue
        w = window_override.get(n, MC[n].window_slots)
        out[n] = FastMethod(n, w, MC[n].min_observations, fn)
    return out


# =============================================================================
# Evaluator
# =============================================================================

class FastEvaluator:
    def __init__(self, data: pd.DataFrame, dataset_min_obs: int,
                 mode: str = 'compat', seed: int = 42,
                 rsi_by_item: bool = False, robustness: bool = True,
                 robustness_sample_size: int = 50, spike_multiplier: float = 10.0,
                 item_col: str = 'item_id', time_col: str = 'timestamp',
                 count_col: str = 'count', verbose: bool = True,
                 tie_info: bool = False):
        assert mode in ('compat', 'dense')
        self.mode = mode
        self.seed = seed
        self.rsi_by_item = rsi_by_item
        self.robustness = robustness
        self.sample_size = robustness_sample_size
        self.spike_multiplier = spike_multiplier
        self.verbose = verbose
        # tie_info=True adds a diagnostic column 'ties_top21': True when two
        # of the 21 highest scores are exactly equal, i.e. when the top-K
        # sets / orders (K<=20) depend on np.argsort's tie-breaking.
        self.tie_info = tie_info

        df = data[[time_col, item_col, count_col]].copy()
        df.columns = ['timestamp', 'item_id', 'count']
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # ---- item selection: replica of _select_items_from_data (num_items=None)
        start, end = df['timestamp'].min(), df['timestamp'].max()
        ev = df[(df['timestamp'] >= start) & (df['timestamp'] <= end)]
        ic = ev.groupby('item_id')['count'].sum()
        ic = ic[ic >= dataset_min_obs]
        self.items = ic.index.values
        df = df[df['item_id'].isin(self.items)]

        # ---- slot size: replica of the "granularity-fix"
        ts = df['timestamp'].drop_duplicates().sort_values()
        dmin = int(round(ts.diff().dropna().dt.total_seconds().median() / 60.0))
        self.slot = pd.Timedelta(minutes=dmin)
        self.start = start                  # config.start_date None -> data min
        self.end = end
        self.abs_min = df['timestamp'].min()
        self.num_slots = int((self.end - self.start).total_seconds() / self.slot.total_seconds())

        # ---- matrix
        s = (df['timestamp'] - self.start) / self.slot
        if not np.allclose(s, np.round(s)):
            raise ValueError('timestamps are not on a regular slot grid')
        sidx = np.round(s).astype(np.int64).values
        pos = {iid: k for k, iid in enumerate(self.items)}
        iidx = df['item_id'].map(pos).values
        S = self.num_slots + 2
        n = len(self.items)
        self.V = np.zeros((n, S), dtype=np.float64)       # zero-filled values
        self.M = np.zeros((n, S), dtype=bool)             # row present?
        if pd.Series(list(zip(iidx, sidx))).duplicated().any():
            raise ValueError('duplicate (item, timestamp) rows')
        self.V[iidx, sidx] = df['count'].values.astype(np.float64)
        self.M[iidx, sidx] = True
        # cumulative presence counts, leading zero column: C[:, s] = #rows in [0, s)
        self.C = np.zeros((n, S + 1), dtype=np.int64)
        np.cumsum(self.M, axis=1, out=self.C[:, 1:])
        self.any_row = self.M.any(axis=0)
        self.any_cum = np.concatenate([[0], np.cumsum(self.any_row)])
        # compressed (present-rows-only) values per item, left aligned
        maxc = int(self.C[:, -1].max())
        self.P = np.zeros((n, maxc + 1), dtype=np.float64)
        for k in range(n):
            vals = self.V[k, self.M[k]]
            self.P[k, :len(vals)] = vals
        if self.verbose:
            print(f'[fast] items={n} slots={self.num_slots + 1} slot={dmin}min '
                  f'mode={mode} rows={int(self.M.sum())}')

    # -------------------------------------------------------------------------
    @classmethod
    def from_csv(cls, path, dataset_min_obs: int, item_as_str: bool = True, **kw):
        df = pd.read_csv(path)
        if item_as_str:
            df['item_id'] = df['item_id'].astype(str)
        return cls(df, dataset_min_obs, **kw)

    # -------------------------------------------------------------------------
    def window_bounds(self, k: int, W: int):
        """Slot range [lo, hi) of the training slice of window k."""
        if self.mode == 'compat':
            lo = k - (W - 1)
        else:
            lo = k - W
        # old code clamps train_start to absolute_min (== grid index 0 here,
        # because start_date=None makes data_start == data min)
        lo = max(lo, 0)
        return lo, k

    def series_lengths(self, k: int, W: int) -> np.ndarray:
        lo, hi = self.window_bounds(k, W)
        if self.mode == 'compat':
            return self.C[:, hi] - self.C[:, lo]
        return np.full(len(self.items), hi - lo)

    def _gather(self, idx: np.ndarray, k: int, W: int, L: int) -> np.ndarray:
        lo, hi = self.window_bounds(k, W)
        if self.mode == 'compat':
            a = self.C[idx, lo]
            return self.P[idx[:, None], a[:, None] + np.arange(L)[None, :]]
        return self.V[idx, lo:hi]

    def _pad_index(self, L_of: np.ndarray, tgt: int, mode: str) -> np.ndarray:
        """Per-row source index so that X[r, idx] == np.pad(x_r, (tgt-L_r, 0), mode)."""
        p = (tgt - L_of)[:, None]
        j = np.arange(tgt)[None, :]
        if mode == 'edge':
            return np.maximum(j - p, 0)
        if mode == 'reflect':
            if np.any(p > L_of[:, None] - 1):
                return None                      # multi-bounce reflect: fall back
            return np.abs(j - p)
        return None

    def _score(self, fm: FastMethod, idx: np.ndarray, k: int, lengths: np.ndarray):
        scores = np.empty(len(idx))
        L_of = lengths[idx]
        core = getattr(fm.scorer, 'core', None)
        if core is not None and self.mode == 'compat':
            # rows of different length but the same padded length -> one transform
            tg = np.array([fm.scorer.target_len(int(L)) for L in L_of])
            done = np.zeros(len(idx), dtype=bool)
            for T in np.unique(tg):
                sel = np.where(tg == T)[0]
                Ls = L_of[sel]
                if Ls.max() > T:
                    continue
                src = self._pad_index(Ls, int(T), fm.scorer.pad_mode)
                if src is None:
                    continue
                lo, _ = self.window_bounds(k, fm.window_slots)
                a = self.C[idx[sel], lo]
                Xp = self.P[idx[sel][:, None], a[:, None] + src]
                try:
                    scores[sel] = core(Xp)
                    done[sel] = True
                except Exception:
                    pass
            if done.all():
                return scores
            rest = np.where(~done)[0]
        else:
            rest = np.arange(len(idx))
        for L in np.unique(L_of[rest]):
            sel = rest[L_of[rest] == L]
            X = self._gather(idx[sel], k, fm.window_slots, int(L))
            try:
                scores[sel] = fm.scorer(X)
            except Exception:
                # per-row retry so one bad row does not zero the group
                for j, r in enumerate(sel):
                    try:
                        scores[r] = fm.scorer(X[j:j + 1])[0]
                    except Exception:
                        scores[r] = 0.0
        # evaluator: non-finite scores are kept as returned by the method
        return scores

    # -------------------------------------------------------------------------
    def _robustness(self, fm: FastMethod, idx: np.ndarray, k: int,
                    scores: np.ndarray, rng: np.random.RandomState) -> float:
        lo, hi = self.window_bounds(k, fm.window_slots)
        if self.mode == 'compat':
            # pivot_table(train_data): columns = slots where ANY selected item
            # has a row; missing cells -> 0
            cols = np.where(self.any_row[lo:hi])[0] + lo
            matrix = self.V[idx][:, cols]
        else:
            matrix = self.V[idx, lo:hi]
        if matrix.shape[0] < 2:
            return float('nan')
        # --- replica of RobustnessScenario.select_stable_candidates -------
        ni = matrix.shape[0]
        if ni < self.sample_size:
            targets = list(range(ni))
        else:
            means = np.mean(matrix, axis=1)
            nzi = np.where(means > 0)[0]
            if len(nzi) < self.sample_size:
                targets = nzi.tolist()
            else:
                thr = np.percentile(means[nzi], 50)
                cand = [i for i in nzi if means[i] <= thr]
                if len(cand) > self.sample_size:
                    targets = rng.choice(cand, self.sample_size, replace=False).tolist()
                else:
                    targets = cand
        if not targets:
            return float('nan')
        # --- replica of inject_spike (batched) -----------------------------
        T = matrix[targets].copy()
        sp = T.mean(axis=1) * self.spike_multiplier
        sp = np.where(sp < 1, 1.0, sp)
        T[:, -1] += sp
        try:
            ns = fm.scorer(T)
        except Exception:
            ns = np.array([fm.scorer(T[j:j + 1])[0] for j in range(len(T))])
        d = []
        for j, t in enumerate(targets):
            noisy = scores.copy()
            noisy[t] = float(ns[j])
            d.append(float(calculate_rank_distortion(scores, noisy, t)))
        return float(np.mean(d)) if d else float('nan')

    # -------------------------------------------------------------------------
    def run_method(self, fm: FastMethod, out_dir: Optional[Path] = None) -> pd.DataFrame:
        t0 = time.time()
        rng = np.random.RandomState(self.seed)
        prev: Dict[int, list] = {}
        recs = []
        W = fm.window_slots
        for k in range(self.num_slots + 1):
            lo, hi = self.window_bounds(k, W)
            # skip if no selected item has a row in train slice or test slot
            if self.any_cum[hi] - self.any_cum[lo] == 0 or not self.any_row[k]:
                continue
            lengths = self.C[:, hi] - self.C[:, lo]          # observed rows
            idx = np.where(lengths >= fm.min_obs)[0]
            if len(idx) == 0:
                continue
            sl = self.series_lengths(k, W)
            sc = self._score(fm, idx, k, sl)
            act = self.V[idx, k]                              # 0 if no row
            if len(sc) < 2:
                continue
            rec = {'window_id': k,
                   'timestamp': int((self.start + k * self.slot).timestamp() * 1000),
                   'method': fm.name, 'num_items': len(sc)}
            for K in K_VALUES:
                rec[f'ndcg@{K}'] = calculate_ndcg(sc, act, k=K)
                rec[f'coverage@{K}'] = calculate_coverage(sc, act, k=K)
            dg = calculate_diagnostics(sc, act)
            rec['kendall_tau'] = dg['kendall_tau']
            rec['spearman_rho'] = dg['spearman_rho']
            rec['mae'] = dg['mae']
            for K in K_VALUES:
                top = np.argsort(sc)[-K:][::-1].tolist()
                if self.rsi_by_item:
                    top = [int(idx[t]) for t in top]
                p = prev.get(K)
                rec[f'rsi@{K}'] = float('nan') if p is None else calculate_rsi(p, top)
                prev[K] = top
            if self.tie_info:
                top21 = np.sort(sc)[::-1][:21]
                rec['ties_top21'] = bool(np.any(top21[1:] == top21[:-1]))
            rec['robustness_distortion'] = (
                self._robustness(fm, idx, k, sc, rng) if self.robustness else float('nan'))
            recs.append(rec)
        cols = PROTOCOL_COLUMNS + (['ties_top21'] if self.tie_info else [])
        df = pd.DataFrame(recs, columns=cols)
        dur = time.time() - t0
        if self.verbose:
            print(f'[fast] {fm.name:<12} W={W:<4} windows={len(df):<6} {dur:8.1f}s')
        if out_dir is not None:
            pdir = Path(out_dir) / 'protocol'
            pdir.mkdir(parents=True, exist_ok=True)
            df.to_csv(pdir / f'{fm.name}_protocol.csv', index=False, encoding='utf-8')
        df.attrs['duration_s'] = dur
        return df

    def run(self, methods: Dict[str, FastMethod], out_dir=None) -> Dict[str, pd.DataFrame]:
        res, rt = {}, {}
        for name, fm in methods.items():
            res[name] = self.run_method(fm, out_dir)
            rt[name] = res[name].attrs['duration_s']
        if out_dir is not None:
            md = Path(out_dir) / 'metadata'
            md.mkdir(parents=True, exist_ok=True)
            meta = dict(mode=self.mode, seed=self.seed, rsi_by_item=self.rsi_by_item,
                        robustness=self.robustness, slot_minutes=self.slot.total_seconds() / 60,
                        num_items=len(self.items), num_slots=self.num_slots + 1,
                        methods={n: dict(window_slots=m.window_slots, min_obs=m.min_obs,
                                         scorer=m.scorer.__name__) for n, m in methods.items()},
                        runtime_seconds=rt,
                        numpy=np.__version__, pywt=pywt.__version__, dtcwt=dtcwt.__version__,
                        created=time.strftime('%Y-%m-%d %H:%M:%S'))
            (md / 'fast_run.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
        return res
