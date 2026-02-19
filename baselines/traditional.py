"""
Popularity Assessment Baseline Methods
=======================================
General-purpose popularity scoring methods for distributed systems.
Based on definitions from: Hamdeni et al. (2016) and related works.

Methods:
  - AF          : Access Frequency (exponential decay weighting) [Chang & Chang, 2008]
  - EWMA        : Exponentially Weighted Moving Average [Gui & Chen, 2020]
  - RRD         : Requests / Lifetime ratio [Al Mistarihi & Yong, 2008]
  - VSE         : Volume + Recency combined score [Mansouri & Asadi, 2014]
  - CompoundPop : Three-factor compound popularity [Ye et al., 2014]
  - PFRF        : Period-based popularity weighting [Lee et al., 2012]

Removed:
  - LRU      — cache-specific eviction policy, not a general popularity metric
  - MeanFreq — replaced by RRD (same concept but normalised by lifetime)
"""
import numpy as np
from typing import List, Dict


class TraditionalBaselines:
    """
    General-purpose popularity scoring methods.
    All methods output a scalar popularity score for a given time series.
    """

    @staticmethod
    def access_frequency(time_series: np.ndarray) -> float:
        """
        Weighted Access Frequency (AF).
        Ref: Chang & Chang (2008), Eq(7).
        Recent accesses have higher weight: 2^0, 2^-1, 2^-2, ...
        """
        if len(time_series) == 0:
            return 0.0
        reversed_ts = time_series[::-1]
        score = 0.0
        for i, val in enumerate(reversed_ts):
            score += (2.0 ** (-i)) * val
        return float(score)

    @staticmethod
    def ewma_score(time_series: np.ndarray, alpha: float = 0.2) -> float:
        """
        Exponentially Weighted Moving Average (EWMA).
        Ref: Gui & Chen (2020), Eq(13).
        P(t) = alpha * R(t) + (1-alpha) * P(t-1)
        Recent periods have higher influence.
        """
        if len(time_series) == 0:
            return 0.0
        ewma = float(time_series[0])
        for val in time_series[1:]:
            ewma = alpha * val + (1 - alpha) * ewma
        return float(ewma)

    @staticmethod
    def rrd_score(time_series: np.ndarray) -> float:
        """
        Requests per unit Lifetime (RRD — Relative Request Density).
        Ref: Al Mistarihi & Yong (2008), Eq(3).
        Formula: total_requests / lifetime_slots
        Normalizes frequency by data lifetime to avoid bias for older data.
        """
        if len(time_series) == 0:
            return 0.0
        total_requests = float(np.sum(time_series))
        # Lifetime = number of periods from first non-zero access to end
        nonzero_indices = np.nonzero(time_series)[0]
        if len(nonzero_indices) == 0:
            return 0.0
        first_access = nonzero_indices[0]
        lifetime = len(time_series) - first_access
        if lifetime <= 0:
            return 0.0
        return total_requests / lifetime

    @staticmethod
    def vse_score(time_series: np.ndarray) -> float:
        """
        Volume + Recency Score (VSE — Value-based Score with Elapsed time).
        Ref: Mansouri & Asadi (2014), Eq(6).
        Formula: total_requests * recency_weight
        Recency weight = 1 / (slots_since_last_access + 1)
        Combines total demand with temporal recency.
        """
        if len(time_series) == 0:
            return 0.0
        nonzero_indices = np.nonzero(time_series)[0]
        if len(nonzero_indices) == 0:
            return 0.0
        # Recency: distance from last access to end of window
        last_access = nonzero_indices[-1]
        dist = len(time_series) - 1 - last_access
        recency_weight = 1.0 / (dist + 1.0)
        total_requests = float(np.sum(time_series))
        return total_requests * recency_weight

    @staticmethod
    def compound_pop_score(time_series: np.ndarray,
                           cons1: float = 0.5,
                           cons2: float = 0.3,
                           cons3: float = 0.2) -> float:
        """
        Compound Popularity Score (three-factor model).
        Ref: Ye et al. (2014), Eq(8).
        Combines: (1) total requests, (2) recent-period requests, (3) recency.
        Formula: cons1*total + cons2*recent + cons3*(1/(dist+1))
        Normalised by window size for scale invariance.
        """
        if len(time_series) == 0:
            return 0.0
        n = len(time_series)
        total_requests = float(np.sum(time_series))
        # Recent period: last 20% of window
        recent_n = max(1, int(n * 0.2))
        recent_requests = float(np.sum(time_series[-recent_n:]))
        # Recency weight
        nonzero_indices = np.nonzero(time_series)[0]
        if len(nonzero_indices) == 0:
            return 0.0
        last_access = nonzero_indices[-1]
        dist = n - 1 - last_access
        recency = 1.0 / (dist + 1.0)
        # Normalize by window size
        score = (cons1 * (total_requests / n) +
                 cons2 * (recent_requests / recent_n) +
                 cons3 * recency)
        return float(score)

    @staticmethod
    def pfrf_score(time_series: np.ndarray,
                   a: float = 1.2,
                   b: float = 0.8) -> float:
        """
        PFRF Period-based Popularity Weight.
        Ref: Lee et al. (2012), Eq(11) — Popular File Replicate First.

        Divides the window into equal-length periods and iteratively
        updates a weight: if a period has requests → multiply by `a`
        (boost), else multiply by `b` (decay). Final score is the
        accumulated weight, reflecting recent active periods more.

        Args:
            time_series: 1-D array of request counts per time slot.
            a: boost factor for active periods (default 1.2, > 1).
            b: decay factor for inactive periods (default 0.8, < 1).

        Returns:
            Cumulative popularity weight.
        """
        if len(time_series) == 0:
            return 0.0
        # Treat each slot as one period (generalisation of the formula)
        weight = 1.0
        for count in time_series:
            if count > 0:
                weight *= a
            else:
                weight *= b
        return float(weight)

    @staticmethod
    def batch_assess_all(time_series_list: List[np.ndarray]) -> Dict[str, np.ndarray]:
        """Compute all baseline scores for a list of time series."""
        n = len(time_series_list)
        results = {
            'AF':          np.zeros(n),
            'EWMA':        np.zeros(n),
            'RRD':         np.zeros(n),
            'VSE':         np.zeros(n),
            'CompoundPop': np.zeros(n),
            'PFRF':        np.zeros(n),
        }
        for i, ts in enumerate(time_series_list):
            results['AF'][i]          = TraditionalBaselines.access_frequency(ts)
            results['EWMA'][i]        = TraditionalBaselines.ewma_score(ts)
            results['RRD'][i]         = TraditionalBaselines.rrd_score(ts)
            results['VSE'][i]         = TraditionalBaselines.vse_score(ts)
            results['CompoundPop'][i] = TraditionalBaselines.compound_pop_score(ts)
            results['PFRF'][i]        = TraditionalBaselines.pfrf_score(ts)
        return results
