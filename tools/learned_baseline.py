r"""
Learned baselines for the prediction runs: GBM and GBM+W
========================================================
A light learned forecaster, to answer the question whether a model that is
trained on the data beats the training-free methods.  It is trained once per
scenario and horizon on the tuning windows and scored, like every other method,
by ``ProtocolV5Evaluator`` (``tools/run_prediction.py``).

Model.  ``sklearn.ensemble.HistGradientBoostingRegressor`` with fixed settings
(no tuning): max_iter 300, learning_rate 0.05, max_leaf_nodes 31, no early
stopping, random_state 42.  Fit and prediction run on one thread, so the result
does not depend on the number of cores and parallel jobs do not compete.

Target.  log(1 + y), y = the ground truth of the window (sum of the h slots
k..k+h-1).  The score is exp(prediction) - 1, clipped at zero (same order as the
prediction; on the scale of the ground truth, so that the MAE column is
meaningful).

Features (only the observed window, the same zero-filled N = 64 slot matrix as
every method; no padding):
    GBM     log(1 + sum of the last n slots), n = 1, 2, 4, 8, 16, 32, 64
            (all slots if the window is shorter); least-squares slope of the
            last 16 slots; share of zero slots
    GBM+W   the GBM features and the three WSPI features mu_L, R and W_E
            (DTCWT near_sym_a / qshift_a, J = 3, protocol-V5 padding)

Training data.  The windows are those that every method of the prediction runs
evaluates (``iter_eligible_windows``: N = 64, 32 observed slots, the causal item
rule).  They are split in time as in the first paper (``split_ids`` of
``tools/run_param_grid.py``): the first 30 % are the tuning part, the rest the
test part.  The model is trained on the item rows of the tuning windows k with
k + h - 1 < (first test window), so that no training target reaches a slot of
the test part.  If there are more than 200 000 rows, 200 000 are drawn at random
(seed 42; the same rows for GBM and GBM+W).  The model then scores every window,
including the tuning windows: those scores are in-sample and only the test part
may be compared (``tools/run_prediction_stats.py --part test``).

    from tools.learned_baseline import train_learned
    scorers, info = train_learned(evaluator, ['GBM', 'GBM+W'], horizon=8)

Author: Sajjad Pirahesh
Date: November 2026
"""
from __future__ import annotations

import time
from typing import Dict, List, Sequence, Tuple

import numpy as np
import dtcwt

from evaluation.protocol_v5 import _pad_left_reflect
from tools.prediction_protocol import (LEARNED_METHODS, LEVEL, MIN_OBS, WINDOW, CoefFeatures,
                                       iter_eligible_windows, recency_mean_rows)
from tools.run_param_grid import TUNE_SHARE, split_ids

GBM_PARAMS = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                  early_stopping=False, random_state=42)
MAX_TRAIN_ROWS = 200_000
SAMPLE_SEED = 42
SUM_LENGTHS = (1, 2, 4, 8, 16, 32, 64)
SLOPE_LENGTH = 16
VALUE_FEATURES = tuple(f'log1p_sum{n}' for n in SUM_LENGTHS) + (f'slope{SLOPE_LENGTH}',
                                                                  'zero_share')
WAVELET_FEATURES = ('mu_L', 'R', 'W_E')


# =============================================================================
# Features
# =============================================================================

def value_features(X: np.ndarray) -> np.ndarray:
    """The nine GBM features of every row of X (rows: items, columns: slots)."""
    X = np.asarray(X, dtype=np.float64)
    nr, n = X.shape
    cols = []
    for L in SUM_LENGTHS:
        cols.append(np.log1p(np.clip(X[:, -L:].sum(axis=1), 0.0, None)))
    L = min(SLOPE_LENGTH, n)
    if L >= 2:
        t = np.arange(L, dtype=np.float64)
        t = t - t.mean()
        Y = X[:, -L:]
        slope = (Y - Y.mean(axis=1, keepdims=True)) @ t / float(t @ t)
    else:
        slope = np.zeros(nr)
    cols.append(slope)
    cols.append((X == 0).mean(axis=1) if n else np.zeros(nr))
    return np.column_stack(cols)


class WaveletFeatures:
    """mu_L, R and W_E of WSPI (protocol-V5 padding) for every row."""

    def __init__(self, window: int = WINDOW, level: int = LEVEL):
        self.tr = dtcwt.Transform1d(biort='near_sym_a', qshift='qshift_a')
        self.level = level
        self.target = max(window, 2 ** (level + 1))

    def __call__(self, X: np.ndarray) -> np.ndarray:
        F = CoefFeatures(_pad_left_reflect(np.asarray(X, dtype=np.float64), self.target),
                         self.tr, self.level)
        return np.column_stack([recency_mean_rows(F.a), F.R, F.WE])


def feature_names(name: str) -> Tuple[str, ...]:
    return VALUE_FEATURES + (WAVELET_FEATURES if name == 'GBM+W' else ())


# =============================================================================
# Windows and training rows
# =============================================================================

def split_windows(window_ids: Sequence[int], horizon: int):
    """(tuning ids, test ids, training ids).  Training ids are the tuning
    windows whose ground truth ends before the first test window."""
    tune, test = split_ids(window_ids)
    if not test:
        raise ValueError('no test windows')
    train = [k for k in tune if k + int(horizon) - 1 < test[0]]
    return tune, test, train


def _sample_mask(counts: np.ndarray, max_rows: int, seed: int):
    """Per-window boolean masks selecting at most max_rows rows overall."""
    total = int(counts.sum())
    if total <= max_rows:
        return None, total
    rng = np.random.RandomState(seed)
    chosen = np.zeros(total, dtype=bool)
    chosen[rng.choice(total, max_rows, replace=False)] = True
    return np.split(chosen, np.cumsum(counts)[:-1]), total


def training_rows(fe, train_ids: Sequence[int], with_wavelet: Dict[str, bool],
                  window: int = WINDOW, min_obs: int = MIN_OBS,
                  max_rows: int = MAX_TRAIN_ROWS, seed: int = SAMPLE_SEED):
    """Feature matrices (one per entry of ``with_wavelet``), the target
    log(1 + y) and a summary, for the item rows of the training windows."""
    train = set(int(k) for k in train_ids)
    wins = [(k, idx) for k, idx in iter_eligible_windows(fe, window, min_obs) if k in train]
    counts = np.array([len(idx) for _, idx in wins], dtype=np.int64)
    masks, total = _sample_mask(counts, max_rows, seed)
    wf = WaveletFeatures(window) if any(with_wavelet.values()) else None
    parts = {n: [] for n in with_wavelet}
    target = []
    for j, (k, idx) in enumerate(wins):
        sel = idx if masks is None else idx[masks[j]]
        if len(sel) == 0:
            continue
        X = fe._matrix(sel, k, window)
        y = fe.V[sel, k] if fe.horizon == 1 else fe.V[sel, k:k + fe.horizon].sum(axis=1)
        vf = value_features(X)
        w = wf(X) if wf is not None else None
        for n, use_w in with_wavelet.items():
            parts[n].append(np.column_stack([vf, w]) if use_w else vf)
        target.append(np.log1p(np.clip(np.asarray(y, dtype=np.float64), 0.0, None)))
    feats = {n: np.concatenate(p, axis=0) for n, p in parts.items()}
    y = np.concatenate(target)
    info = dict(n_train_windows=len(wins),
                first_train_window=int(wins[0][0]) if wins else None,
                last_train_window=int(wins[-1][0]) if wins else None,
                n_rows_available=int(total), n_rows_used=int(len(y)),
                sampled=masks is not None, max_rows=int(max_rows), sample_seed=int(seed))
    return feats, y, info


# =============================================================================
# The scorer
# =============================================================================

class LearnedScorer:
    """Batch scorer of a trained GBM: exp(prediction) - 1, clipped at zero."""

    def __init__(self, name: str, model, horizon: int, window: int = WINDOW):
        if name not in LEARNED_METHODS:
            raise ValueError(name)
        self.name = name
        self.model = model
        self.horizon = int(horizon)
        self.wf = WaveletFeatures(window) if name == 'GBM+W' else None
        self.__name__ = f"{name.lower().replace('+', '_')}_h{horizon}"

    def features(self, X: np.ndarray) -> np.ndarray:
        vf = value_features(X)
        return np.column_stack([vf, self.wf(X)]) if self.wf is not None else vf

    def __call__(self, X):
        from threadpoolctl import threadpool_limits
        F = self.features(np.asarray(X, dtype=np.float64))
        with threadpool_limits(limits=1):
            p = self.model.predict(F)
        return np.clip(np.expm1(np.asarray(p, dtype=np.float64)), 0.0, None)


def fit_gbm(F: np.ndarray, y: np.ndarray):
    """HistGradientBoostingRegressor with the fixed settings, on one thread."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from threadpoolctl import threadpool_limits
    model = HistGradientBoostingRegressor(**GBM_PARAMS)
    with threadpool_limits(limits=1):
        model.fit(F, y)
    return model


def train_learned(fe, names: List[str], horizon: int, window: int = WINDOW,
                  min_obs: int = MIN_OBS, max_rows: int = MAX_TRAIN_ROWS,
                  seed: int = SAMPLE_SEED):
    """Train the named learned methods on the tuning windows of ``fe``.

    Returns ({name: LearnedScorer}, {name: info dict}).  ``fe`` is the
    evaluator of the run (its horizon must equal ``horizon``)."""
    import sklearn
    names = [n for n in names if n in LEARNED_METHODS]
    if not names:
        return {}, {}
    if int(fe.horizon) != int(horizon):
        raise ValueError('the evaluator has another horizon')
    t0 = time.time()
    ids = [k for k, _ in iter_eligible_windows(fe, window, min_obs)]
    tune, test, train = split_windows(ids, horizon)
    feats, y, rows = training_rows(fe, train, {n: n == 'GBM+W' for n in names},
                                   window, min_obs, max_rows, seed)
    t_rows = time.time() - t0
    scorers, info = {}, {}
    for n in names:
        t1 = time.time()
        model = fit_gbm(feats[n], y)
        scorers[n] = LearnedScorer(n, model, horizon, window)
        info[n] = dict(method=n, horizon=int(horizon), model='HistGradientBoostingRegressor',
                       params=GBM_PARAMS, threads=1, target='log1p(sum of the h future slots)',
                       score='expm1(prediction), clipped at 0',
                       features=list(feature_names(n)), tune_share=TUNE_SHARE,
                       n_windows=len(ids), n_tune_windows=len(tune), n_test_windows=len(test),
                       first_test_window=int(test[0]),
                       train_rule='tuning windows k with k + h - 1 < first test window',
                       **rows, target_mean=float(y.mean()) if len(y) else float('nan'),
                       rows_seconds=round(t_rows, 2), fit_seconds=round(time.time() - t1, 2),
                       n_iter=int(model.n_iter_), sklearn=sklearn.__version__)
    return scorers, info
