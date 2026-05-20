# -*- coding: utf-8 -*-
"""
Temporal Evaluator - Main Evaluation Engine
Implements the 4-Layer "Frozen Evaluation Protocol":
  Layer 1 - Decision:    NDCG@K, Coverage@K
  Layer 2 - Diagnostic:  Kendall's Tau, Spearman's Rho, MAE
  Layer 3 - Stability:   RSI (Ranking Stability Index) @K
  Layer 4 - Robustness:  Rank Distortion (Noise Injection)

Author: Sajjad
Date: February 2025 (refactored)
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time
from datetime import datetime, timedelta
import multiprocessing as mp
from functools import partial
from collections import defaultdict

from .evaluation_config import EvaluationConfig
from .stratification import StratificationSystem
from .metrics import (
    calculate_ndcg,
    calculate_coverage,
    calculate_rsi,
    calculate_rank_distortion,
    calculate_diagnostics,
)
from .wavelet_validator import WaveletWindowValidator
from .storage import StorageSystem
from .method_configs import get_method_config
from .time_utils import create_time_helper, TimeSlotHelper
from .scenarios import RobustnessScenario


class TemporalEvaluator:
    """
    Temporal evaluation with sliding window and 4-Layer Frozen Evaluation Protocol.

    Features:
    - True sliding window (step=1)
    - Automatic stratification (StratificationSystem preserved)
    - 4-Layer metrics: Decision / Diagnostic / Stability / Robustness
    - Noise-injection robustness test via RobustnessScenario
    - Parquet storage (StorageSystem preserved)
    """

    # K values used across Decision & Stability layers
    K_VALUES: List[int] = [5, 10, 20]

    def __init__(self, data_loader, methods: Dict, config: EvaluationConfig):
        self.data_loader = data_loader
        self.methods = methods
        self.config = config

        # --- Subsystems (MUST NOT be deleted) ---------------------------------
        self.stratification = StratificationSystem(config)
        self.storage = StorageSystem(config)

        # Time slot helper
        self.time_helper = create_time_helper(config)

        # --- Frozen Evaluation Protocol additions -----------------------------
        # Layer 4: Robustness scenario
        self.scenario = RobustnessScenario(sample_size=50, spike_multiplier=10.0)

        # Layer 3: per-method state for RSI; reset per method in evaluate()
        #   Structure: prev_top_k[method_name][k] -> List[int] of item indices
        self.prev_top_k: Dict[str, Dict[int, List[int]]] = defaultdict(dict)

        # Wavelet validation
        self._validate_wavelet_config()

        # Data placeholders
        self.data = None
        self.items = None
        self.num_windows = 0

        # Runtime stats
        self.runtime_stats = {
            'start_time': None,
            'end_time': None,
            'total_windows_processed': 0,
            'total_items_processed': 0,
            'total_calculations': 0,
        }

    # ==========================================================================
    # Wavelet config validation (UNCHANGED)
    # ==========================================================================

    def _validate_wavelet_config(self):
        if self.config.verbose:
            print("\n" + "=" * 70)
            print("WAVELET CONFIGURATION VALIDATION")
            print("=" * 70)

        WaveletWindowValidator.print_validation_report(
            self.config.window_size,
            self.config.wavelet_config['dwt']['level'],
            'dwt'
        )
        WaveletWindowValidator.print_validation_report(
            self.config.window_size,
            self.config.wavelet_config['dtcwt']['level'],
            'dtcwt'
        )

    # ==========================================================================
    # Data preparation (MUST NOT be removed)
    # ==========================================================================

    def prepare_data(self):
        """Load and prepare data for evaluation."""
        if self.config.verbose:
            print("\n" + "=" * 70)
            print("DATA PREPARATION")
            print("=" * 70)

        print("Loading data...")
        self.data = self.data_loader.load_data()

        if self.config.verbose:
            print(f"  Total records: {len(self.data):,}")
            print(f"  Date range: {self.data['timestamp'].min()} to {self.data['timestamp'].max()}")

        # Evaluation range
        self.data_start = (pd.to_datetime(self.config.start_date)
                           if self.config.start_date else self.data['timestamp'].min())
        self.data_end   = (pd.to_datetime(self.config.end_date)
                           if self.config.end_date else self.data['timestamp'].max())

        if self.config.verbose:
            print(f"  Evaluation range: {self.data_start.date()} to {self.data_end.date()}")
            print(f"  Full data range: {self.data['timestamp'].min().date()} to "
                  f"{self.data['timestamp'].max().date()}")

        eval_data = self.data[
            (self.data['timestamp'] >= self.data_start) &
            (self.data['timestamp'] <= self.data_end)
        ]

        if self.config.verbose:
            print(f"  Records in eval range: {len(eval_data):,}")

        self.items = self._select_items_from_data(eval_data)

        if self.config.verbose:
            print(f"  Selected items: {len(self.items):,}")

        # Keep pre-range data for selected items
        self.data = self.data[self.data['item_id'].isin(self.items)]

        if self.config.verbose:
            print(f"  Final records (all dates): {len(self.data):,}")

        num_slots = self.time_helper.count_slots(self.data_start, self.data_end)
        self.num_windows = num_slots + 1

        if self.config.verbose:
            unit = self.time_helper.get_unit_name(plural=True)
            print(f"  Evaluation {unit}: {self.num_windows}")
            print(f"  Number of windows: {self.num_windows} "
                  f"(one per {self.time_helper.get_unit_name()})")
            print(f"  Total calculations: "
                  f"{self.num_windows * len(self.items) * len(self.methods):,}")

        print("=" * 70 + "\n")

    # ==========================================================================
    # Private helpers (UNCHANGED)
    # ==========================================================================

    def _apply_time_filter(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.config.start_date:
            data = data[data['timestamp'] >= pd.to_datetime(self.config.start_date)]
        if self.config.end_date:
            data = data[data['timestamp'] <= pd.to_datetime(self.config.end_date)]
        return data

    def _select_items_from_data(self, data: pd.DataFrame) -> np.ndarray:
        item_counts = data.groupby('item_id')['count'].sum()
        item_counts = item_counts[item_counts >= self.config.min_observations]

        if self.config.num_items is None:
            return item_counts.index.values
        elif self.config.item_selection == 'top':
            return item_counts.nlargest(self.config.num_items).index.values
        elif self.config.item_selection == 'random':
            n = min(self.config.num_items, len(item_counts))
            rng = np.random.RandomState(42)   # fixed seed = reproducible across runs
            return rng.choice(item_counts.index.values, size=n, replace=False)
        elif self.config.item_selection == 'stratified':
            return self._stratified_sampling(item_counts, self.config.num_items)
        else:
            raise ValueError(f"Invalid item_selection: {self.config.item_selection}")

    def _stratified_sampling(self, item_counts: pd.Series, n: int) -> np.ndarray:
        """
        Proportional stratified sampling with a fixed seed (42) so that
        temporal and incremental modes always select the same items.
        Uses 4 popularity quartiles (cold_start / low / medium / high).
        """
        rng = np.random.RandomState(42)   # deterministic — matches loader random_state=42

        # Re-use stratification thresholds already set on self.stratification
        strata = self.stratification.stratify_items(
            pd.DataFrame({'item_id': item_counts.index, 'count': item_counts.values})
        )
        total_items = sum(len(items) for items in strata.values())
        selected = []
        for _, stratum_items in strata.items():
            if len(stratum_items) == 0:
                continue
            ratio = len(stratum_items) / total_items
            n_sample = min(int(n * ratio), len(stratum_items))
            chosen = rng.choice(stratum_items, size=n_sample, replace=False)
            selected.extend(chosen.tolist())

        # Fill up to n if rounding left gaps
        all_items = [item for items in strata.values() for item in items]
        remaining = [i for i in all_items if i not in set(selected)]
        if len(selected) < n and remaining:
            extra = rng.choice(remaining,
                               size=min(n - len(selected), len(remaining)),
                               replace=False)
            selected.extend(extra.tolist())

        return np.array(selected[:n])

    # ==========================================================================
    # Top-level evaluate()
    # ==========================================================================

    def evaluate(self):
        if self.data is None:
            self.prepare_data()

        if self.config.verbose:
            print(f"\n{'=' * 70}")
            print("ASSESSMENT MODE (not prediction)")
            print(f"{'=' * 70}")
            unit = self.time_helper.get_unit_name()
            print(f"  Each {unit} = one window")
            print(f"  Training: window_size {self.time_helper.get_unit_name(plural=True)} up to target")
            print(f"  Test: actual count on target {unit}")
            print(f"  Use pre-range data: {self.config.use_pre_range_data}")
            print(f"{'=' * 70}\n")

        self.runtime_stats['start_time'] = time.time()

        for method_name, method in self.methods.items():
            if self.config.verbose:
                print(f"\n{'=' * 70}")
                print(f"EVALUATING: {method_name}")
                print(f"{'=' * 70}")

            # Reset RSI state for each fresh method evaluation
            self.prev_top_k[method_name] = {}
            self._evaluate_method(method_name, method)

        self.runtime_stats['end_time'] = time.time()
        self.runtime_stats['total_duration'] = (
            self.runtime_stats['end_time'] - self.runtime_stats['start_time']
        )

        self._compare_methods()
        self._save_runtime_stats()

        if self.config.verbose:
            self._print_summary()

    # ==========================================================================
    # _evaluate_method — REWRITTEN for the 4-Layer Frozen Evaluation Protocol
    # ==========================================================================

    def _evaluate_method(self, method_name: str, method):
        """
        Evaluate one method across all sliding windows using the
        4-Layer Frozen Evaluation Protocol.

        Layer 1 – Decision:    NDCG@K, Coverage@K          (K in {5, 10, 20})
        Layer 2 – Diagnostic:  Kendall tau, Spearman rho, MAE
        Layer 3 – Stability:   RSI@K                   (K in {5, 10, 20})
        Layer 4 – Robustness:  Mean Rank Distortion (noise injection)

        Legacy outputs (detailed scores & stratum summaries) are preserved via
        _evaluate_window and _calculate_stratum_summary.
        """
        # -- Method-level config --
        try:
            method_config      = get_method_config(method_name)
            method_window_size = method_config.window_slots   # slots, dataset-agnostic
            method_min_obs     = method_config.min_observations
        except KeyError:
            method_window_size = self.config.window_size
            method_min_obs     = self.config.min_observations

        # window_slots is already in the correct unit for add_slots() —
        # no day→hour conversion needed.  7 slots = 7 hours on hourly datasets,
        # 7 days on daily datasets, etc.
        if self.config.verbose:
            print(f"  Window size:      {method_window_size} slots")
            print(f"  Min observations: {method_min_obs}")
            print(f"  Use pre-range:    {self.config.use_pre_range_data}")

        # -- Accumulators --
        all_results: List[Dict]      = []   # per-item, for detailed storage
        stratum_summaries: List[Dict] = []  # per-stratum, for legacy summary storage
        protocol_records: List[Dict]  = []  # per-window 4-layer metrics

        # -- Date limits --
        min_date   = self.data['timestamp'].min()
        eval_start = self.data_start
        eval_end   = self.data_end

        num_slots   = self.time_helper.count_slots(eval_start, eval_end)
        num_windows = num_slots + 1

        if self.config.verbose:
            unit = self.time_helper.get_unit_name(plural=True)
            print(f"  Total windows: {num_windows} (one per {self.time_helper.get_unit_name()})")
            if self.config.use_pre_range_data:
                pre_slots = self.time_helper.count_slots(min_date, eval_start)
                print(f"  Pre-range data: {self.time_helper.format_window_size(pre_slots)}")

        if self.config.progress_bar:
            pbar = tqdm(total=num_windows, desc=method_name)

        # ======================================================================
        # Sliding window loop
        # ======================================================================
        for slot_idx in range(num_windows):

            # ---- Window boundaries -------------------------------------------
            target_date = self.time_helper.add_slots(eval_start, slot_idx)
            train_end   = target_date
            train_start = self.time_helper.add_slots(train_end, -(method_window_size - 1))

            if train_start < min_date:
                train_start = (min_date if self.config.use_pre_range_data
                               else max(train_start, eval_start))

            test_start = target_date
            test_end   = self.time_helper.add_slots(target_date, 1)

            train_window = self.data[
                (self.data['timestamp'] >= train_start) &
                (self.data['timestamp'] < train_end)
            ]
            test_window = self.data[
                (self.data['timestamp'] >= test_start) &
                (self.data['timestamp'] < test_end)
            ]

            if len(train_window) == 0 or len(test_window) == 0:
                if self.config.progress_bar:
                    pbar.update(1)
                continue

            # ---- Stratify ----------------------------------------------------
            strata = self.stratification.stratify_items(train_window)

            # ---- Phase 1: Clean evaluation (legacy detailed + stratum) -------
            window_results = self._evaluate_window(
                method, method_name, slot_idx,
                train_window, test_window, test_start, strata, method_min_obs
            )
            all_results.extend(window_results)
            stratum_summaries.extend(
                self._calculate_stratum_summary(window_results, slot_idx, test_start)
            )

            # ---- 4-Layer Protocol (requires >= 2 items) ----------------------
            if len(window_results) >= 2:
                timestamp_ms = int(test_start.timestamp() * 1000)

                scores_clean = np.array([r['popularity_score'] for r in window_results],
                                        dtype=np.float64)
                actuals      = np.array([r['actual_count']     for r in window_results],
                                        dtype=np.float64)
                item_ids     = [r['item_id'] for r in window_results]

                # ---- Layer 1: Decision ----------------------------------------
                decision: Dict[str, float] = {}
                for k in self.K_VALUES:
                    decision[f'ndcg@{k}'] = calculate_ndcg(scores_clean, actuals, k=k)
                    decision[f'coverage@{k}']  = calculate_coverage(scores_clean, actuals, k=k)

                # ---- Layer 2: Diagnostics -------------------------------------
                diag = calculate_diagnostics(scores_clean, actuals)

                # ---- Layer 3: Stability (RSI) ---------------------------------
                stability: Dict[str, float] = {}
                for k in self.K_VALUES:
                    top_k_now = np.argsort(scores_clean)[-k:][::-1].tolist()
                    prev      = self.prev_top_k[method_name].get(k)

                    if prev is None:
                        # First window — no prior top-K to compare with
                        stability[f'rsi@{k}'] = float('nan')
                    else:
                        stability[f'rsi@{k}'] = calculate_rsi(prev, top_k_now)

                    self.prev_top_k[method_name][k] = top_k_now  # store for next window

                # ---- Layer 4: Robustness (Noise Injection) --------------------
                robustness_distortion = float('nan')
                try:
                    # Build item × time-slot matrix from train_window
                    pivot = (
                        train_window
                        .pivot_table(index='item_id', columns='timestamp',
                                     values='count', aggfunc='sum', fill_value=0)
                    )
                    present = [iid for iid in item_ids if iid in pivot.index]

                    if len(present) >= 2:
                        matrix = pivot.loc[present].values.astype(np.float64)  # N × T

                        # Select low-popularity stable candidates for spike injection
                        target_local_idxs = self.scenario.select_stable_candidates(matrix)

                        if target_local_idxs:
                            id_to_pos = {iid: pos for pos, iid in enumerate(item_ids)}
                            scores_present = np.array(
                                [scores_clean[id_to_pos[iid]] for iid in present],
                                dtype=np.float64
                            )

                            distortions: List[float] = []
                            for local_idx in target_local_idxs:
                                target_iid  = present[local_idx]
                                noisy_ts    = self.scenario.inject_spike(matrix[local_idx])

                                # Phase 2: re-score target item with noisy time series
                                try:
                                    if hasattr(method, 'assess_single'):
                                        ns = float(method.assess_single(noisy_ts))
                                    elif hasattr(method, 'calculate'):
                                        ns = float(method.calculate(noisy_ts))
                                    else:
                                        ns = float(method.predict(
                                            pd.DataFrame({'count': noisy_ts})
                                        ))
                                except Exception:
                                    continue

                                # Build noisy scores (only the target item changes)
                                scores_noisy             = scores_present.copy()
                                scores_noisy[local_idx]  = ns

                                d = calculate_rank_distortion(
                                    scores_present, scores_noisy, local_idx
                                )
                                distortions.append(float(d))

                            if distortions:
                                robustness_distortion = float(np.mean(distortions))

                except Exception as e:
                    if self.config.verbose:
                        print(f"    [Robustness] Window {slot_idx}: {e}")

                # ---- Assemble protocol record ---------------------------------
                window_metrics: Dict = {
                    'window_id':  slot_idx,
                    'timestamp':  timestamp_ms,
                    'method':     method_name,
                    'num_items':  len(window_results),
                    # Layer 1
                    **decision,
                    # Layer 2
                    'kendall_tau':  diag['kendall_tau'],
                    'spearman_rho': diag['spearman_rho'],
                    'mae':          diag['mae'],
                    # Layer 3
                    **stability,
                    # Layer 4
                    'robustness_distortion': robustness_distortion,
                }
                protocol_records.append(window_metrics)

            # ---- Runtime tracking --------------------------------------------
            self.runtime_stats['total_windows_processed'] += 1
            self.runtime_stats['total_items_processed']   += len(window_results)
            self.runtime_stats['total_calculations']      += 1

            if self.config.progress_bar:
                pbar.update(1)

            if self.config.verbose and slot_idx % self.config.log_interval == 0:
                print(f"  Window {slot_idx}/{num_windows}: {len(window_results)} items")

        if self.config.progress_bar:
            pbar.close()

        # -- Save results -------------------------------------------------------
        if self.config.verbose:
            print("  Saving results...")

        # Legacy outputs (DetailedScores + StratumSummary)
        self.storage.save_detailed_scores(method_name, all_results)
        self.storage.save_stratum_summary(method_name, stratum_summaries)

        # New: 4-Layer protocol metrics (one row per window)
        if protocol_records:
            self._save_protocol_metrics(method_name, protocol_records)

        if self.config.verbose:
            print(f"  Completed: {len(all_results):,} item-records | "
                  f"{len(protocol_records)} protocol windows")

    # ==========================================================================
    # Protocol metrics persistence helper
    # ==========================================================================

    def _save_protocol_metrics(self, method_name: str, records: List[Dict]):
        """
        Save 4-layer Frozen Evaluation Protocol metrics.
        Output: <output_dir>/protocol/<method_name>_protocol.parquet  (or .csv)
        """
        df = pd.DataFrame(records)

        if 'timestamp' in df.columns:
            df['timestamp'] = df['timestamp'].astype('int64')

        protocol_dir = self.config.output_dir / 'protocol'
        protocol_dir.mkdir(parents=True, exist_ok=True)

        if self.storage.has_pyarrow:
            fp = protocol_dir / f"{method_name}_protocol.parquet"
            df.to_parquet(fp, compression=self.storage.compression,
                          engine='pyarrow', index=False)
        else:
            fp = protocol_dir / f"{method_name}_protocol.csv"
            df.to_csv(fp, index=False, encoding='utf-8')

        if self.config.verbose:
            size_mb = fp.stat().st_size / (1024 * 1024)
            print(f"  Protocol metrics: {fp.name} ({size_mb:.2f} MB)")

    # ==========================================================================
    # _evaluate_window — UNCHANGED logic, metrics updated to use metrics.py
    # ==========================================================================

    def _evaluate_window(self, method, method_name: str, window_idx: int,
                         train_window: pd.DataFrame, test_window: pd.DataFrame,
                         test_start: datetime, strata: Dict,
                         min_observations: int) -> List[Dict]:
        """Per-item evaluation for a single window (legacy detailed scores)."""
        results = []
        timestamp_ms = int(test_start.timestamp() * 1000)

        for item_id in self.items:
            item_train = train_window[train_window['item_id'] == item_id]

            if len(item_train) < min_observations:
                continue

            try:
                if hasattr(method, 'assess_single'):
                    score = method.assess_single(item_train['count'].values)
                elif hasattr(method, 'calculate'):
                    score = method.calculate(item_train['count'].values)
                else:
                    score = method.predict(item_train)

                score = float(np.mean(score) if isinstance(score, (list, np.ndarray))
                              else score)
            except Exception as e:
                if self.config.verbose:
                    print(f"    Warning: Failed to score item {item_id}: {e}")
                score = 0.0

            item_test    = test_window[test_window['item_id'] == item_id]
            actual_count = int(item_test['count'].sum()) if len(item_test) > 0 else 0

            # Mean-per-slot: independent of window length, comparable across datasets.
            # NYC Yellow Taxi / YouTube have high per-slot counts; using sum caused
            # everything to land in 'high' stratum.  Mean fixes the scale.
            n_slots      = max(len(item_train), 1)
            train_mean   = item_train['count'].sum() / n_slots
            stratum_label = self.stratification.get_stratum_label(train_mean)
            train_count  = float(train_mean)   # store mean for downstream use

            results.append({
                'window_id':       window_idx,
                'timestamp':       timestamp_ms,
                'item_id':         str(item_id),
                'stratum':         stratum_label,
                'popularity_score': score,
                'actual_count':    actual_count,
                'train_count':     train_count,
            })

        # Attach per-window aggregate rank columns to each record
        if results:
            scores  = np.array([r['popularity_score'] for r in results])
            actuals = np.array([r['actual_count']     for r in results])

            diag = calculate_diagnostics(scores, actuals)

            rank_pred   = np.argsort(np.argsort(-scores))   + 1   # 1-based
            rank_actual = np.argsort(np.argsort(-actuals))  + 1

            for i, r in enumerate(results):
                r['mae']            = diag['mae']
                r['rank_predicted'] = int(rank_pred[i])
                r['rank_actual']    = int(rank_actual[i])

        return results

    # ==========================================================================
    # _calculate_stratum_summary — updated to use metrics.py functions
    #   StratificationSystem is fully preserved.
    # ==========================================================================

    def _calculate_stratum_summary(self, window_results: List[Dict],
                                   window_idx: int, timestamp: datetime) -> List[Dict]:
        """Per-stratum aggregated metrics (legacy summary storage)."""
        if not window_results:
            return []

        df = pd.DataFrame(window_results)
        timestamp_ms = int(timestamp.timestamp() * 1000)
        summaries    = []

        for stratum_label in range(4):
            stratum_data = df[df['stratum'] == stratum_label]

            empty_row = {
                'window_id':   window_idx,
                'timestamp':   timestamp_ms,
                'stratum':     stratum_label,
                'stratum_name': self.stratification.get_stratum_name(stratum_label),
                'num_items':   0,
                'mean_mae':    0.0,
                'mean_mape':   0.0,
                'spearman_corr': 0.0,
                'kendall_tau': 0.0,
                'ndcg':        0.0,
                'coverage':    0.0,
            }

            if len(stratum_data) == 0:
                summaries.append(empty_row)
                continue

            scores  = stratum_data['popularity_score'].values.astype(np.float64)
            actuals = stratum_data['actual_count'].values.astype(np.float64)

            try:
                diag     = calculate_diagnostics(scores, actuals)
                ndcg_val = calculate_ndcg(scores, actuals, k=10)

                with np.errstate(divide='ignore', invalid='ignore'):
                    mape = float(np.mean(
                        np.where(actuals != 0,
                                 np.abs((actuals - scores) / actuals) * 100,
                                 0.0)
                    ))

                coverage = float(np.mean(scores > 0))

                summaries.append({
                    'window_id':    window_idx,
                    'timestamp':    timestamp_ms,
                    'stratum':      stratum_label,
                    'stratum_name': self.stratification.get_stratum_name(stratum_label),
                    'num_items':    len(stratum_data),
                    'mean_mae':     diag['mae'],
                    'mean_mape':    mape,
                    'spearman_corr': diag['spearman_rho'],
                    'kendall_tau':  diag['kendall_tau'],
                    'ndcg':         ndcg_val,
                    'coverage':     coverage,
                })

            except Exception:
                summaries.append({**empty_row, 'num_items': len(stratum_data)})

        return summaries

    # ==========================================================================
    # Method comparison & reporting (UNCHANGED)
    # ==========================================================================

    def _compare_methods(self):
        if self.config.verbose:
            print(f"\n{'=' * 70}")
            print("COMPARING METHODS")
            print(f"{'=' * 70}")

        method_summaries = {}
        for method_name in self.methods.keys():
            try:
                method_summaries[method_name] = self.storage.load_stratum_summary(method_name)
            except FileNotFoundError:
                if self.config.verbose:
                    print(f"  Warning: No summary found for {method_name}")

        if not method_summaries:
            return

        comparison_data = []
        for method_name, summary_df in method_summaries.items():
            try:
                comparison_data.append({
                    'method':         method_name,
                    'mean_mae':       summary_df['mean_mae'].mean(),
                    'mean_mape':      summary_df['mean_mape'].mean(),
                    'mean_spearman':  summary_df['spearman_corr'].mean(),
                    'mean_kendall':   summary_df['kendall_tau'].mean(),
                    'mean_ndcg':      summary_df['ndcg'].mean(),
                    'mean_coverage':  summary_df['coverage'].mean(),
                })
            except KeyError as e:
                if self.config.verbose:
                    print(f"  Warning: {method_name} missing column {e}")
                    print(f"  Available: {list(summary_df.columns)}")
                    print("  Delete 'results/' and re-run to regenerate.")

        comparison_df = pd.DataFrame(comparison_data)
        if comparison_df.empty:
            if self.config.verbose:
                print("  No valid summaries — please delete 'results/' and re-run.")
            return

        comparison_df = comparison_df.sort_values('mean_spearman', ascending=False)
        self.storage.save_method_comparison(comparison_df)

        if self.config.verbose:
            print("\nOverall Performance (averaged across all windows & strata):")
            print(comparison_df.to_string(index=False))
            print(f"{'=' * 70}\n")

    def _save_runtime_stats(self):
        stats = {
            **self.runtime_stats,
            'config':              self.config.to_dict(),
            'num_methods':         len(self.methods),
            'num_items':           len(self.items),
            'num_windows':         self.num_windows,
            'avg_time_per_window': (
                self.runtime_stats['total_duration'] / self.num_windows
                if self.num_windows > 0 else 0
            ),
        }
        self.storage.save_runtime_stats(stats)
        self.config.save_config(self.config.output_dir / 'metadata' / 'config.json')
        self.stratification.save_thresholds(
            self.config.output_dir / 'metadata' / 'thresholds.json'
        )

    def _print_summary(self):
        duration = self.runtime_stats['total_duration']
        h  = int(duration // 3600)
        m  = int((duration % 3600) // 60)
        s  = int(duration % 60)

        print(f"\n{'=' * 70}")
        print("EVALUATION COMPLETED")
        print(f"{'=' * 70}")
        print(f"Total Duration:     {h}h {m}m {s}s")
        print(f"Windows Processed:  {self.runtime_stats['total_windows_processed']:,}")
        print(f"Items Processed:    {self.runtime_stats['total_items_processed']:,}")
        print(f"Total Calculations: {self.runtime_stats['total_calculations']:,}")
        print(f"Output Directory:   {self.config.output_dir}")
        print(f"{'=' * 70}\n")

        self.storage.print_storage_summary()
