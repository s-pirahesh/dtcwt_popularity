r"""
Corrected evaluation protocol V5 (revision-srep-v5, task T1.4)
==============================================================
Builds on ``evaluation/fast_evaluator.py`` (which is NOT modified: its
``compat`` mode keeps reproducing the V4 runs, gate G1).

What changes with respect to the V4 protocol (all approved by Sajjad,
24 Sep 2026 — see 07_Task_Tracker.md, T1.4):

1. Exact window.  The training slice of window k is ``[k - W, k)``, i.e.
   exactly W slots (64 for the wavelet-based methods, 7 for the
   baselines), instead of ``[k - (W-1), k)``.
2. Zero-filled slots.  A slot in which the item has no row in the CSV is a
   zero, not a skipped sample.  Every item therefore gets a regular series
   of length W.  Eligibility is unchanged: the item needs at least
   ``min_observations`` observed rows in the slice (3 for the baselines,
   32 for the wavelet-based methods).
3. Same level and padding for the three wavelet-based methods: J = 3 for
   DWT+AF, DTCWT+AF and WSPI (no automatic level reduction).  A series is
   padded (``reflect``, on the left) only when it is shorter than W, which
   happens only in the first windows (k < W).  From window W on there is no
   padding at all.  The baselines are never padded.
4. RSI compares top-K sets by item id (not by array position).
5. Fixed tie-breaking.  Every top-K / rank decision uses a stable sort on
   the score (descending); equal scores are ordered by the fixed item order
   (item ids sorted as strings, i.e. the row order of the item matrix).
   This rule gives no method extra information and does not depend on the
   NumPy version or the CPU.  NDCG@K, Coverage@K, RSI@K and the rank
   distortion use it.  Non-finite scores are ranked last (counted in
   ``n_nonfinite``).  Kendall tau, Spearman rho and MAE are tie-aware and
   are computed with the unchanged ``metrics.calculate_diagnostics``.
6. Robustness test: target sampling with a fixed seed (default 42, one
   RandomState per method, restarted for every method); the clean and the
   noisy score come from the same zero-filled W-slot matrix and the same
   scorer (same padding).

Extra diagnostic columns in every protocol CSV:
    ties_top21   True if two of the 21 highest scores are exactly equal
    padded       True if the series of this window were padded (k < W)
    n_nonfinite  number of NaN / inf scores in the window

Outputs (``out_dir``):
    protocol/<method>_protocol.csv          window-by-window
    comparison/summary_all_windows.csv      mean/std over each method's windows
    comparison/summary_common_windows.csv   mean/std over the windows that
                                            ALL methods of the run evaluate
    metadata/protocol_v5_run.json

Author: Sajjad (with assistance), September 2026
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import pywt
import dtcwt

from .fast_evaluator import (
    FastEvaluator, FastMethod, PROTOCOL_COLUMNS, K_VALUES,
    batch_af, batch_ewma, batch_rrd, batch_vse, batch_compound, batch_pfrf,
    make_batch_dtcwt_af, make_batch_wspi, _waf_rows,
)
from .metrics import calculate_diagnostics, calculate_rsi

V5_EXTRA_COLUMNS = ['ties_top21', 'padded', 'n_nonfinite']
METRIC_COLUMNS = [c for c in PROTOCOL_COLUMNS
                  if c not in ('window_id', 'timestamp', 'method', 'num_items')]
WAVELET_METHODS = ('DWT+AF', 'DTCWT+AF', 'WSPI')


# =============================================================================
# Stable ranking and tie-stable metric versions
# =============================================================================

def rank_key(scores: np.ndarray) -> np.ndarray:
    """Non-finite scores -> -inf, so they are ranked last."""
    s = np.asarray(scores, dtype=np.float64)
    return np.where(np.isfinite(s), s, -np.inf)


def stable_order(scores: np.ndarray) -> np.ndarray:
    """Indices by descending score; ties -> lower index (item order) first."""
    return np.argsort(-rank_key(scores), kind='stable')


def ndcg_stable(order: np.ndarray, actual: np.ndarray, k: int) -> float:
    """Same formula as metrics.calculate_ndcg, with a given (stable) order."""
    n = len(order)
    if n == 0 or k <= 0:
        return 0.0
    k = min(k, n)
    rel = np.log10(1 + actual)
    pred_rel = rel[order[:k]]
    discounts = np.log2(np.arange(k) + 2)
    dcg = np.sum((np.power(2, pred_rel) - 1) / discounts)
    ideal_rel = np.sort(rel)[-k:][::-1]          # values only: tie-free
    idcg = np.sum((np.power(2, ideal_rel) - 1) / discounts)
    if idcg == 0:
        return 0.0
    return float(dcg / idcg)


def coverage_stable(order: np.ndarray, actual: np.ndarray, k: int) -> float:
    """Same formula as metrics.calculate_coverage, with a given order."""
    total = np.sum(actual)
    if total == 0:
        return 0.0
    k = min(k, len(order))
    # metrics.py sums in ascending-score order -> same summation order
    return float(np.sum(actual[order[:k][::-1]]) / total)


def rank_distortion_stable(clean: np.ndarray, noisy: np.ndarray, target: int) -> int:
    """Same as metrics.calculate_rank_distortion, with the stable order."""
    rc = int(np.where(stable_order(clean) == target)[0][0]) + 1
    rn = int(np.where(stable_order(noisy) == target)[0][0]) + 1
    return abs(rn - rc)


def has_ties_top(scores: np.ndarray, n: int = 21) -> bool:
    top = np.sort(rank_key(scores))[::-1][:n]
    return bool(np.any(top[1:] == top[:-1]))


# =============================================================================
# Wavelet scorers with one padding / level policy
# =============================================================================

def _pad_left_reflect(X: np.ndarray, target: int) -> np.ndarray:
    L = X.shape[1]
    if L >= target:
        return X
    return np.pad(X, ((0, 0), (target - L, 0)), mode='reflect')


def _target_len(window: int, level: int) -> int:
    return max(window, 2 ** (level + 1))


def make_v5_dwt(window: int, level: int = 3, wavelet: str = 'db4',
                mode: str = 'symmetric', detail_weight: float = 0.1) -> Callable:
    """DWT+AF, fixed level (no _safe_level reduction), reflect padding to W."""
    tgt = _target_len(window, level)

    def f(X):
        Xp = _pad_left_reflect(X, tgt)
        c = pywt.wavedec(Xp, wavelet, level=level, mode=mode, axis=-1)
        return _waf_rows(c[0]) + detail_weight * _waf_rows(c[-1])
    f.__name__ = f'v5_dwt_L{level}_W{window}'
    return f


def make_v5_dtcwt_af(window: int, level: int = 3) -> Callable:
    core = make_batch_dtcwt_af(level=level).core
    tgt = _target_len(window, level)

    def f(X):
        return core(_pad_left_reflect(X, tgt))
    f.__name__ = f'v5_dtcwt_af_L{level}_W{window}'
    return f


def make_v5_wspi(window: int, level: int = 3, alpha: float = 1.0, beta: float = 1.0,
                 use_R: bool = True, use_WE: bool = True) -> Callable:
    core = make_batch_wspi(level=level, alpha=alpha, beta=beta,
                           use_R=use_R, use_WE=use_WE).core
    tgt = _target_len(window, level)

    def f(X):
        return core(_pad_left_reflect(X, tgt))
    f.__name__ = f'v5_wspi_L{level}_W{window}'
    return f


def build_v5_methods(level: int = 3, window_override: Optional[Dict[str, int]] = None,
                     names: Optional[List[str]] = None) -> Dict[str, FastMethod]:
    """The 9 methods of the paper under protocol V5."""
    from .method_configs import METHOD_CONFIGS as MC
    window_override = window_override or {}
    out = {}
    baselines = {'AF': batch_af, 'EWMA': batch_ewma, 'RRD': batch_rrd,
                 'VSE': batch_vse, 'CompoundPop': batch_compound, 'PFRF': batch_pfrf}
    wavelets = {'DWT+AF': make_v5_dwt, 'DTCWT+AF': make_v5_dtcwt_af, 'WSPI': make_v5_wspi}
    for n in list(baselines) + list(wavelets):
        if names and n not in names:
            continue
        w = window_override.get(n, MC[n].window_slots)
        fn = baselines[n] if n in baselines else wavelets[n](window=w, level=level)
        out[n] = FastMethod(n, w, MC[n].min_observations, fn)
    return out


# =============================================================================
# Evaluator
# =============================================================================

class ProtocolV5Evaluator(FastEvaluator):
    """Dense, exact-W evaluator with stable tie-breaking (protocol V5)."""

    def __init__(self, data: pd.DataFrame, dataset_min_obs: int, seed: int = 42,
                 robustness: bool = True, **kw):
        kw.pop('mode', None)
        kw.pop('rsi_by_item', None)
        kw.pop('tie_info', None)
        super().__init__(data, dataset_min_obs, mode='dense', seed=seed,
                         rsi_by_item=True, robustness=robustness, tie_info=True, **kw)

    @classmethod
    def from_csv(cls, path, dataset_min_obs: int, item_as_str: bool = True, **kw):
        df = pd.read_csv(path)
        if item_as_str:
            df['item_id'] = df['item_id'].astype(str)
        return cls(df, dataset_min_obs, **kw)

    # -------------------------------------------------------------------------
    def _matrix(self, idx: np.ndarray, k: int, W: int) -> np.ndarray:
        lo, hi = self.window_bounds(k, W)          # dense: [max(k-W,0), k)
        return self.V[idx, lo:hi]

    @staticmethod
    def _safe_score(scorer: Callable, X: np.ndarray) -> np.ndarray:
        try:
            return np.asarray(scorer(X), dtype=np.float64)
        except Exception:
            out = np.empty(X.shape[0])
            for j in range(X.shape[0]):
                try:
                    out[j] = scorer(X[j:j + 1])[0]
                except Exception:
                    out[j] = 0.0
            return out

    def _robustness_v5(self, fm: FastMethod, X: np.ndarray, scores: np.ndarray,
                       rng: np.random.RandomState) -> float:
        """Replica of RobustnessScenario (candidate choice + spike), with the
        clean and noisy series taken from the same matrix X."""
        ni = X.shape[0]
        if ni < 2:
            return float('nan')
        if ni < self.sample_size:
            targets = list(range(ni))
        else:
            means = np.mean(X, axis=1)
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
        T = X[targets].copy()
        sp = T.mean(axis=1) * self.spike_multiplier
        sp = np.where(sp < 1, 1.0, sp)
        T[:, -1] += sp
        ns = self._safe_score(fm.scorer, T)
        d = []
        for j, t in enumerate(targets):
            noisy = scores.copy()
            noisy[t] = float(ns[j])
            d.append(rank_distortion_stable(scores, noisy, t))
        return float(np.mean(d)) if d else float('nan')

    # -------------------------------------------------------------------------
    def run_method(self, fm: FastMethod, out_dir: Optional[Path] = None) -> pd.DataFrame:
        t0 = time.time()
        rng = np.random.RandomState(self.seed)
        prev: Dict[int, set] = {}
        recs = []
        W = fm.window_slots
        for k in range(self.num_slots + 1):
            lo, hi = self.window_bounds(k, W)
            if self.any_cum[hi] - self.any_cum[lo] == 0 or not self.any_row[k]:
                continue
            observed = self.C[:, hi] - self.C[:, lo]
            idx = np.where(observed >= fm.min_obs)[0]
            if len(idx) < 2:
                continue
            X = self._matrix(idx, k, W)
            sc = self._safe_score(fm.scorer, X)
            act = self.V[idx, k]
            order = stable_order(sc)
            rec = {'window_id': k,
                   'timestamp': int((self.start + k * self.slot).timestamp() * 1000),
                   'method': fm.name, 'num_items': len(sc)}
            for K in K_VALUES:
                rec[f'ndcg@{K}'] = ndcg_stable(order, act, K)
                rec[f'coverage@{K}'] = coverage_stable(order, act, K)
            dg = calculate_diagnostics(sc, act)
            rec['kendall_tau'] = dg['kendall_tau']
            rec['spearman_rho'] = dg['spearman_rho']
            rec['mae'] = dg['mae']
            for K in K_VALUES:
                top = [int(i) for i in idx[order[:K]]]        # item ids (matrix rows)
                p = prev.get(K)
                rec[f'rsi@{K}'] = float('nan') if p is None else calculate_rsi(p, top)
                prev[K] = top
            rec['robustness_distortion'] = (
                self._robustness_v5(fm, X, sc, rng) if self.robustness else float('nan'))
            rec['ties_top21'] = has_ties_top(sc, 21)
            rec['padded'] = bool(X.shape[1] < W)
            rec['n_nonfinite'] = int((~np.isfinite(sc)).sum())
            recs.append(rec)
        df = pd.DataFrame(recs, columns=PROTOCOL_COLUMNS + V5_EXTRA_COLUMNS)
        dur = time.time() - t0
        if self.verbose:
            print(f'[v5] {fm.name:<12} W={W:<4} windows={len(df):<6} {dur:8.1f}s')
        if out_dir is not None:
            pdir = Path(out_dir) / 'protocol'
            pdir.mkdir(parents=True, exist_ok=True)
            df.to_csv(pdir / f'{fm.name}_protocol.csv', index=False, encoding='utf-8')
        df.attrs['duration_s'] = dur
        return df

    # -------------------------------------------------------------------------
    def run(self, methods: Dict[str, FastMethod], out_dir=None,
            extra_meta: Optional[dict] = None) -> Dict[str, pd.DataFrame]:
        res, rt = {}, {}
        for name, fm in methods.items():
            res[name] = self.run_method(fm, out_dir)
            rt[name] = res[name].attrs['duration_s']
        if out_dir is not None:
            out_dir = Path(out_dir)
            cdir = out_dir / 'comparison'
            cdir.mkdir(parents=True, exist_ok=True)
            summarize(res, common=False).to_csv(cdir / 'summary_all_windows.csv',
                                                index=False, encoding='utf-8')
            summarize(res, common=True).to_csv(cdir / 'summary_common_windows.csv',
                                               index=False, encoding='utf-8')
            md = out_dir / 'metadata'
            md.mkdir(parents=True, exist_ok=True)
            meta = dict(protocol='v5', mode='dense', seed=self.seed, rsi_by_item=True,
                        tie_rule='stable sort, ties by fixed item order (item ids sorted as str)',
                        nonfinite_rule='ranked last',
                        robustness=self.robustness, slot_minutes=self.slot.total_seconds() / 60,
                        num_items=len(self.items), num_slots=self.num_slots + 1,
                        methods={n: dict(window_slots=m.window_slots, min_obs=m.min_obs,
                                         scorer=m.scorer.__name__) for n, m in methods.items()},
                        runtime_seconds=rt,
                        numpy=np.__version__, pywt=pywt.__version__, dtcwt=dtcwt.__version__,
                        created=time.strftime('%Y-%m-%d %H:%M:%S'))
            if extra_meta:
                meta.update(extra_meta)
            (md / 'protocol_v5_run.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
        return res


def summarize(res: Dict[str, pd.DataFrame], common: bool) -> pd.DataFrame:
    """Mean and SD of every metric per method.  common=True restricts every
    method to the window_ids that all methods in ``res`` evaluate."""
    ids = None
    if common:
        for df in res.values():
            s = set(df['window_id'].tolist())
            ids = s if ids is None else ids & s
    rows = []
    for name, df in res.items():
        d = df[df['window_id'].isin(ids)] if common else df
        r = {'method': name, 'n_windows': len(d),
             'first_window': int(d['window_id'].min()) if len(d) else None,
             'last_window': int(d['window_id'].max()) if len(d) else None}
        for c in METRIC_COLUMNS:
            r[f'{c}_mean'] = d[c].mean()
            r[f'{c}_sd'] = d[c].std()
        r['ties_top21_share'] = d['ties_top21'].mean() if len(d) else np.nan
        r['padded_share'] = d['padded'].mean() if len(d) else np.nan
        r['n_nonfinite_total'] = int(d['n_nonfinite'].sum())
        rows.append(r)
    return pd.DataFrame(rows)
