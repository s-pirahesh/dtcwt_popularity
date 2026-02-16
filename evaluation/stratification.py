"""
Stratification System — dataset-adaptive item categorisation.

Root cause of the old bug
--------------------------
The previous implementation computed per-item popularity as the SUM of counts
over the entire training window.  For datasets with high per-slot counts
(Uber: ~100-2000 trips/hour, YouTube: ~1000-100000 views/hour) and a
window of 30 slots, almost every item ended up in the 'high' stratum,
making stratified analysis useless.

Fix
---
Popularity is now expressed as the MEAN count per time-slot
(sum / num_slots).  This is the natural unit of each dataset:
  - MovieLens  -> mean ratings/day
  - Uber       -> mean trips/hour per zone
  - YouTube    -> mean views/hour per video

Since the metric no longer scales with window_size, thresholds are
meaningful and stable across runs with different window configurations.

The same fix is applied in get_stratum_label() (called from temporal_evaluator)
where train_count was also a raw sum.

Author: Sajjad
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional


class StratificationSystem:
    """
    Dataset-adaptive item stratification based on mean-per-slot popularity.

    Strata:
      cold_start : mean < Q1  (sparse / new items)
      low        : Q1 <= mean < Q2
      medium     : Q2 <= mean < Q3
      high       : mean >= Q3  (consistently popular)

    Thresholds are either supplied via config.strata_thresholds
    (recommended, dataset-specific) or computed automatically from
    the first window's quartiles.
    """

    def __init__(self, config):
        self.config   = config
        self.thresholds: Optional[List[float]] = None
        self.stratum_stats: Dict = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def stratify_items(self, window_data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Categorise items by their mean-per-slot count in window_data.

        Args:
            window_data: DataFrame with columns ['item_id', 'count'].
                         Each row is one (item, time-slot) observation.

        Returns:
            dict: {stratum_name: ndarray of item_ids}
        """
        # --- mean count per slot per item --------------------------------
        # groupby gives sum; divide by number of distinct time-slots so
        # the metric is independent of window length.
        item_sum   = window_data.groupby('item_id')['count'].sum()
        num_slots  = window_data.groupby('item_id')['count'].count()
        item_means = item_sum / num_slots.clip(lower=1)   # mean per slot

        # --- resolve thresholds ------------------------------------------
        if self.config.strata_thresholds is not None:
            self.thresholds = list(self.config.strata_thresholds)
        elif self.thresholds is None:
            # Auto-compute from this window's quartiles (first call only)
            self.thresholds = self._calculate_thresholds(item_means)

        t0, t1, t2 = self.thresholds

        strata = {
            'cold_start': item_means[item_means  <  t0].index.values,
            'low':        item_means[(item_means >= t0) & (item_means < t1)].index.values,
            'medium':     item_means[(item_means >= t1) & (item_means < t2)].index.values,
            'high':       item_means[item_means  >= t2].index.values,
        }

        self._save_stratum_stats(item_means, strata)
        return strata

    def get_stratum_label(self, mean_count: float) -> int:
        """
        Return numeric stratum label for a pre-computed mean-per-slot value.

        Args:
            mean_count: mean count per time-slot for one item in one window.
                        Caller must pass (sum / num_slots), NOT the raw sum.

        Returns:
            int: 0=cold_start, 1=low, 2=medium, 3=high
        """
        if self.thresholds is None:
            raise ValueError(
                "Thresholds not set. Call stratify_items() before get_stratum_label()."
            )
        t0, t1, t2 = self.thresholds
        if mean_count < t0:
            return 0
        if mean_count < t1:
            return 1
        if mean_count < t2:
            return 2
        return 3

    def get_stratum_name(self, label: int) -> str:
        """Convert numeric label (0-3) to stratum name string."""
        return ['cold_start', 'low', 'medium', 'high'][label]

    def filter_by_stratum(
        self,
        items: np.ndarray,
        stratum_name: str,
        strata: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """Return the intersection of items with stratum_name's item set."""
        return np.intersect1d(items, strata[stratum_name])

    def get_stratum_summary(self) -> pd.DataFrame:
        """Return a DataFrame with per-stratum statistics."""
        if not self.stratum_stats:
            return pd.DataFrame()
        df = pd.DataFrame(self.stratum_stats).T
        df.index.name = 'stratum'
        return df.reset_index()

    def save_thresholds(self, filepath):
        """Persist thresholds and stratum stats to a JSON file."""
        import json

        def _convert(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: _convert(v) for k, v in obj.items()}
            return obj

        data = _convert({
            'thresholds':    self.thresholds,
            'stratum_stats': self.stratum_stats,
            'metric':        'mean_per_slot',
        })
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _calculate_thresholds(self, item_means: pd.Series) -> List[float]:
        """
        Compute [Q25, Q50, Q90] from the distribution of per-slot means.
        Only called when config.strata_thresholds is None.

        Q90 (not Q75) is used for the high boundary so that 'high' always
        represents the top ~10% of items — consistent with the popularity
        literature and avoids the artificial 25/25/25/25 split that Q75
        would produce on power-law distributions (Uber, YouTube).
        """
        q1 = float(item_means.quantile(0.25))
        q2 = float(item_means.quantile(0.50))
        q3 = float(item_means.quantile(0.90))   # top 10% = high

        thresholds = [q1, q2, q3]

        if self.config.verbose:
            print(
                f"Auto thresholds (mean/slot): "
                f"cold<{q1:.2f}, low<{q2:.2f}, medium<{q3:.2f}, high>={q3:.2f}"
            )
        return thresholds

    def _save_stratum_stats(
        self, item_means: pd.Series, strata: Dict[str, np.ndarray]
    ):
        """Store summary statistics for each stratum (uses mean-per-slot)."""
        self.stratum_stats = {}
        total = len(item_means)

        for name in ['cold_start', 'low', 'medium', 'high']:
            items = strata.get(name, np.array([]))
            if len(items) == 0:
                continue
            vals = item_means[items]
            self.stratum_stats[name] = {
                'num_items':    len(items),
                'percentage':   len(items) / total * 100,
                'min_mean':     float(vals.min()),
                'max_mean':     float(vals.max()),
                'mean_mean':    float(vals.mean()),
                'median_mean':  float(vals.median()),
                'std_mean':     float(vals.std()),
            }

        if self.config.verbose:
            self._print_stratum_stats()

    def _print_stratum_stats(self):
        """Print per-stratum statistics table."""
        print("\n" + "=" * 72)
        print("STRATIFICATION STATISTICS  (metric: mean count per time-slot)")
        print("=" * 72)
        print(
            f"{'Stratum':<12} {'Items':>6} {'%':>6}  "
            f"{'Min/slot':>9} {'Max/slot':>9} {'Mean/slot':>10} {'Median/slot':>12}"
        )
        print("-" * 72)
        for name in ['cold_start', 'low', 'medium', 'high']:
            if name not in self.stratum_stats:
                continue
            s = self.stratum_stats[name]
            print(
                f"{name:<12} {s['num_items']:>6} {s['percentage']:>6.1f}  "
                f"{s['min_mean']:>9.2f} {s['max_mean']:>9.2f} "
                f"{s['mean_mean']:>10.2f} {s['median_mean']:>12.2f}"
            )
        print("=" * 72 + "\n")
