"""
Incremental Temporal Evaluator
Implements the 4-Layer "Frozen Evaluation Protocol" with incremental
(crash-safe, low-memory) storage.

Layers:
  1 - Decision:    NDCG@K, CHR@K          (K ∈ {5, 10, 20})
  2 - Diagnostic:  Kendall τ, Spearman ρ, MAE
  3 - Stability:   RSI (Ranking Stability Index) @K
  4 - Robustness:  Rank Distortion under Noise Injection

Author: Sajjad
Date: February 2025 (refactored)
"""

import gc
import json
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from collections import defaultdict
from tqdm import tqdm

from .evaluation_config import EvaluationConfig
from .incremental_storage import IncrementalStorage
from .method_configs import get_method_config, METHOD_CONFIGS
from .time_utils import create_time_helper, TimeSlotHelper
from .metrics import (
    calculate_ndcg,
    calculate_hit_rate,
    calculate_rsi,
    calculate_rank_distortion,
    calculate_diagnostics,
    MetricsCalculator,          # kept for backward-compat (shim in metrics.py)
)
from .stratification import StratificationSystem
from .scenarios import RobustnessScenario


class IncrementalTemporalEvaluator:
    """
    Temporal evaluation with incremental saving and 4-Layer Frozen Evaluation Protocol.

    Advantages:
    - Memory usage < 200 MB (batch writing)
    - Crash-safe (continuous flush to disk)
    - Method-specific window sizes
    - Progress tracking with tqdm
    """

    # K values for Decision (NDCG/CHR) and Stability (RSI) layers
    K_VALUES: List[int] = [5, 10, 20]

    def __init__(self,
                 config: EvaluationConfig,
                 methods: Dict,
                 data: pd.DataFrame,
                 items,                   # ignored — evaluator re-selects internally
                 storage_path: Path):
        """
        Args:
            config:       EvaluationConfig instance
            methods:      {method_name: method_instance}
            data:         FULL dataset — all items, all dates (including pre-range).
                          Pass data_loader.load_data() directly, NOT the filtered
                          output of load_for_temporal_evaluation().
            items:        Ignored. Items are always re-derived here using the same
                          logic as TemporalEvaluator so both modes are identical.
            storage_path: Root path for all output files
        """
        self.config = config
        self.methods = methods
        self.data = data
        self.storage_path = Path(storage_path)

        # --- Temporal range (must come FIRST — used by item selection) --------
        self.data_start = (pd.to_datetime(config.start_date)
                           if config.start_date else self.data['timestamp'].min())
        self.data_end   = (pd.to_datetime(config.end_date)
                           if config.end_date else self.data['timestamp'].max())

        # --- Stratification (must come BEFORE item selection) -----------------
        self.stratification = StratificationSystem(config)
        self._initialize_stratification()

        # --- Item selection: EXACT copy of temporal_evaluator logic -----------
        # The 'items' argument is kept for API compatibility but NOT used.
        # We always re-derive items here so both modes select identically.
        self.items = self._select_items_from_data()

        # --- Trim data to selected items (mirrors temporal_evaluator.prepare_data) ---
        # This is critical for performance: keeps only rows for the 10 selected items
        # instead of filtering the full dataset on every window iteration.
        # Also ensures self.data is IDENTICAL to what temporal_evaluator uses.
        self.data = self.data[self.data['item_id'].isin(self.items)].copy()

        # --- Populate config.methods so config.json shows the method list -----
        # config.to_dict() includes 'methods' field; set it here so it is
        # non-None when _save_config_json() is called at end of evaluate().
        self.config.methods = list(methods.keys())

        # --- Incremental storage ----------------------------------------------
        self.storage = IncrementalStorage(self.storage_path, buffer_size=1000)

        # --- Frozen Evaluation Protocol additions -----------------------------
        # Layer 4: Robustness scenario
        self.scenario = RobustnessScenario(sample_size=50, spike_multiplier=10.0)

        # Layer 3: per-method top-K state for RSI computation
        #   prev_top_k[method_name][k] -> List[int] of item-position indices
        self.prev_top_k: Dict[str, Dict[int, List[int]]] = defaultdict(dict)

        # --- Time-slot helper (granularity-aware window sizing) --------------
        # This is the same helper used by temporal_evaluator.  Without it,
        # _calculate_windows would always step by calendar days even for
        # hourly datasets (YouTube), producing 29 windows instead of ~697.
        self.time_helper = create_time_helper(config)

        # --- Runtime stats ----------------------------------------------------
        self.runtime_stats = {
            'start_time':   None,
            'end_time':     None,
            'total_duration': 0,
            'methods_stats': {}
        }

    # ==========================================================================
    # Top-level evaluate() — UNCHANGED structure, protocol hooks added
    # ==========================================================================

    def evaluate(self):
        """Run full evaluation (assessment mode) for all methods."""
        self.runtime_stats['start_time'] = time.time()

        print(f"\n{'=' * 70}")
        print("INCREMENTAL TEMPORAL EVALUATION — 4-LAYER FROZEN PROTOCOL")
        print(f"{'=' * 70}")
        print(f"Total methods:   {len(self.methods)}")
        print(f"Total items:     {len(self.items)}")
        print(f"K values:        {self.K_VALUES}")
        print(f"Eval range:      {self.data_start.date()} → {self.data_end.date()}")
        print(f"Full data range: {self.data['timestamp'].min().date()} → "
              f"{self.data['timestamp'].max().date()}")
        print(f"Pre-range data:  {self.config.use_pre_range_data}")
        print(f"Storage:         {self.storage_path}")
        print(f"{'=' * 70}\n")

        for method_name, method in self.methods.items():
            try:
                print(f"\n{'=' * 70}")
                print(f"METHOD: {method_name}")
                print(f"{'=' * 70}")

                method_start = time.time()

                # Reset RSI state for each fresh method run
                self.prev_top_k[method_name] = {}

                self._evaluate_method_incremental(method_name, method)

                duration = time.time() - method_start
                self.runtime_stats['methods_stats'][method_name] = {
                    'duration': duration,
                    'status': 'completed'
                }
                print(f"✓ Completed in {duration / 60:.1f} minutes")

            except Exception as exc:
                import traceback
                print(f"✗ Error: {exc}")
                print(traceback.format_exc())
                self.runtime_stats['methods_stats'][method_name] = {
                    'duration': 0,
                    'status': 'failed',
                    'error': str(exc)
                }

        # Flush & persist metadata
        print(f"\n{'=' * 70}")
        print("FINALIZING")
        print(f"{'=' * 70}")

        self.storage.flush_all()

        metadata = {
            'dataset':   self.config.dataset_name if hasattr(self.config, 'dataset_name') else 'unknown',
            'num_items': len(self.items),
            'date_range': {
                'start': str(self.data_start.date()),
                'end':   str(self.data_end.date())
            },
            'config': {
                'window_size':         self.config.window_size,
                'prediction_horizon':  self.config.prediction_horizon,
                'min_observations':    self.config.min_observations,
                'k_values':            self.K_VALUES,
            },
            'runtime_stats': self.runtime_stats,
        }
        self.storage.save_metadata(metadata)

        self.runtime_stats['end_time']       = time.time()
        self.runtime_stats['total_duration'] = (
            self.runtime_stats['end_time'] - self.runtime_stats['start_time']
        )

        print(f"\nTotal duration: {self.runtime_stats['total_duration'] / 60:.1f} minutes")
        print(f"Storage stats:  {self.storage.get_stats()}")

        self._save_config_json()
        self._save_thresholds_json()
        self._save_runtime_stats_json()

        print(f"✓ All results saved to: {self.storage_path}")
        print(f"{'=' * 70}\n")

    # ==========================================================================
    # _evaluate_method_incremental — REWRITTEN for 4-Layer Protocol
    # ==========================================================================

    def _evaluate_method_incremental(self, method_name: str, method):
        """
        Evaluate one method across all sliding windows with incremental saving.

        For each window:
          • Phase 1 – Clean evaluation  → detailed & summary records (legacy)
          • 4-Layer Protocol            → one protocol record per window

        Protocol records are flushed incrementally to:
          <storage_path>/protocol/<method_name>_protocol.csv
        """
        # -- Method config -----------------------------------------------------
        try:
            method_config  = get_method_config(method_name)
            window_slots   = method_config.window_slots   # slots, dataset-agnostic
            min_obs        = method_config.min_observations
        except KeyError:
            print(f"  Warning: No config for {method_name}, using defaults")
            window_slots   = self.config.window_size
            min_obs        = self.config.min_observations

        print(f"  Window size:      {window_slots} slots")
        print(f"  Min observations: {min_obs}")
        print(f"  Pre-range data:   {self.config.use_pre_range_data}")

        # -- Window list -------------------------------------------------------
        windows = self._calculate_windows(window_slots)
        unit = self.time_helper.get_unit_name()
        print(f"  Total windows:    {len(windows)} (one per {unit} in range)")

        if not windows:
            print("  Warning: No valid windows for this time range")
            return

        # -- Protocol output buffer --------------------------------------------
        protocol_dir = self.storage_path / 'protocol'
        protocol_dir.mkdir(parents=True, exist_ok=True)
        protocol_path = protocol_dir / f"{method_name}_protocol.csv"

        # Write header on first call
        protocol_header_written = protocol_path.exists()

        # ======================================================================
        # Sliding window loop
        # ======================================================================
        for window_idx, (train_start, train_end, test_start, test_end) in enumerate(
            tqdm(windows, desc=f"  {method_name}", ncols=70)
        ):
            train_data = self.data[
                (self.data['timestamp'] >= train_start) &
                (self.data['timestamp'] <  train_end)
            ]
            test_data = self.data[
                (self.data['timestamp'] >= test_start) &
                (self.data['timestamp'] <  test_end)
            ]

            if len(train_data) == 0 or len(test_data) == 0:
                continue

            # ------------------------------------------------------------------
            # Phase 1 — Clean evaluation (legacy detailed + summary records)
            # ------------------------------------------------------------------
            results = self._evaluate_window(
                method, method_name, window_idx,
                train_data, test_data, test_start, min_obs
            )

            if results['detailed']:
                self.storage.append_detailed(method_name, results['detailed'])
            if results['summary']:
                self.storage.append_summary(method_name, results['summary'])

            # ------------------------------------------------------------------
            # 4-Layer Frozen Evaluation Protocol
            # ------------------------------------------------------------------
            if results['detailed']:
                scores_clean = np.array([r['popularity_score'] for r in results['detailed']],
                                        dtype=np.float64)
                actuals      = np.array([r['actual_count']     for r in results['detailed']],
                                        dtype=np.float64)
                timestamp_ms = int(test_start.timestamp() * 1000)

                if len(scores_clean) >= 2:

                    # ---- Layer 1: Decision -----------------------------------
                    decision: Dict[str, float] = {}
                    for k in self.K_VALUES:
                        decision[f'ndcg@{k}'] = calculate_ndcg(scores_clean, actuals, k=k)
                        decision[f'chr@{k}']  = calculate_hit_rate(scores_clean, actuals, k=k)

                    # ---- Layer 2: Diagnostics --------------------------------
                    diag = calculate_diagnostics(scores_clean, actuals)

                    # ---- Layer 3: Stability (RSI) ----------------------------
                    stability: Dict[str, float] = {}
                    for k in self.K_VALUES:
                        top_k_now = np.argsort(scores_clean)[-k:][::-1].tolist()
                        prev      = self.prev_top_k[method_name].get(k)

                        if prev is None:
                            stability[f'rsi@{k}'] = float('nan')
                        else:
                            stability[f'rsi@{k}'] = calculate_rsi(prev, top_k_now)

                        self.prev_top_k[method_name][k] = top_k_now

                    # ---- Layer 4: Robustness (Noise Injection) ---------------
                    robustness_distortion = float('nan')
                    try:
                        pivot = (
                            train_data
                            .pivot_table(index='item_id', columns='timestamp',
                                         values='count', aggfunc='sum', fill_value=0)
                        )
                        # Item IDs from detailed records (in same order as scores_clean)
                        det_item_ids = [r['item_id'] for r in results['detailed']]
                        present = [iid for iid in det_item_ids if iid in pivot.index]

                        if len(present) >= 2:
                            matrix = pivot.loc[present].values.astype(np.float64)  # N × T
                            target_local_idxs = self.scenario.select_stable_candidates(matrix)

                            if target_local_idxs:
                                id_to_pos = {iid: pos for pos, iid in enumerate(det_item_ids)}
                                scores_present = np.array(
                                    [scores_clean[id_to_pos[iid]] for iid in present],
                                    dtype=np.float64
                                )

                                distortions: List[float] = []
                                for local_idx in target_local_idxs:
                                    noisy_ts = self.scenario.inject_spike(matrix[local_idx])

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

                                    scores_noisy             = scores_present.copy()
                                    scores_noisy[local_idx]  = ns

                                    d = calculate_rank_distortion(
                                        scores_present, scores_noisy, local_idx
                                    )
                                    distortions.append(float(d))

                                if distortions:
                                    robustness_distortion = float(np.mean(distortions))

                    except Exception as e:
                        pass   # silently skip robustness for this window

                    # ---- Assemble and write protocol record ------------------
                    record: Dict = {
                        'window_id':  window_idx,
                        'timestamp':  timestamp_ms,
                        'method':     method_name,
                        'num_items':  len(scores_clean),
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

                    # Incremental CSV append (low memory)
                    rec_df = pd.DataFrame([record])
                    rec_df.to_csv(
                        protocol_path,
                        mode='a',
                        header=not protocol_header_written,
                        index=False,
                        encoding='utf-8'
                    )
                    protocol_header_written = True

            # -- Memory management --------------------------------------------
            del train_data, test_data, results
            if (window_idx + 1) % 50 == 0:
                gc.collect()

        # Flush remaining legacy buffers
        self.storage.flush_all()

    # ==========================================================================
    # Window generation (UNCHANGED)
    # ==========================================================================

    def _calculate_windows(self, window_slots: int) -> List[tuple]:
        """
        Build list of (train_start, train_end, test_start, test_end) tuples.
        One window per TIME-SLOT in the evaluation range.

        Args:
            window_slots: method window size in SLOTS (dataset-agnostic).
                          7 = 7 slots on any dataset (hours, days, 5-min, etc.)
                          This value comes directly from MethodConfig.window_slots
                          with NO day→hour conversion.
        """
        windows       = []
        absolute_min  = self.data['timestamp'].min()

        # window_slots is already in the correct unit — use directly.
        num_slots     = self.time_helper.count_slots(self.data_start, self.data_end)

        for slot_idx in range(num_slots + 1):
            target_date = self.time_helper.add_slots(self.data_start, slot_idx)

            train_end   = target_date
            train_start = self.time_helper.add_slots(train_end, -(window_slots - 1))

            if train_start < absolute_min:
                train_start = (absolute_min if self.config.use_pre_range_data
                               else max(train_start, self.data_start))

            test_start = target_date
            test_end   = self.time_helper.add_slots(target_date, 1)

            windows.append((train_start, train_end, test_start, test_end))

        return windows

    # ==========================================================================
    # _evaluate_window — updated to use metrics.py functions
    # ==========================================================================

    def _evaluate_window(self, method, method_name: str, window_idx: int,
                         train_data: pd.DataFrame, test_data: pd.DataFrame,
                         timestamp: datetime, min_observations: int) -> Dict:
        """
        Per-item evaluation for a single window.

        Mirrors temporal_evaluator._evaluate_window exactly so results are
        identical between --incremental and standard mode:
          - Iterates over self.items (fixed selected set, not window-present items)
          - Dispatch: assess_single → calculate → predict  (same order)
          - Fallback score 0.0 on exception (not mean(ts))
          - Mean-per-slot for train_count and stratum

        Returns:
            {'detailed': List[Dict], 'summary': List[Dict]}
        """
        results      = {'detailed': [], 'summary': []}
        timestamp_ms = int(timestamp.timestamp() * 1000)

        test_grouped = test_data.groupby('item_id')['count'].sum()

        detailed_list = []

        for item_id in self.items:
            item_train = train_data[train_data['item_id'] == item_id]

            if len(item_train) < min_observations:
                continue

            ts = item_train['count'].values

            # --- Score (same dispatch chain as temporal_evaluator) -----------
            try:
                with warnings.catch_warnings():
                    # Suppress pywt boundary-effect warnings that fire on
                    # short series in early windows (pre-range data < 56 pts).
                    # DWTAssessment._safe_level() already handles this correctly;
                    # this catch is a belt-and-suspenders guard.
                    warnings.filterwarnings('ignore', category=UserWarning,
                                            module='pywt')
                    if hasattr(method, 'assess_single'):
                        score = method.assess_single(ts)
                    elif hasattr(method, 'calculate'):
                        score = method.calculate(ts)
                    else:
                        score = method.predict(item_train)

                score = float(np.mean(score)
                             if isinstance(score, (list, np.ndarray))
                             else score)
            except Exception:
                score = 0.0   # same as temporal_evaluator

            actual_count = int(test_grouped.get(item_id, 0))

            # Mean-per-slot: window-length independent, comparable across datasets
            n_slots     = max(len(item_train), 1)
            train_mean  = item_train['count'].sum() / n_slots
            stratum_lbl = self.stratification.get_stratum_label(train_mean)

            detailed_list.append({
                'window_id':        window_idx,
                'timestamp':        timestamp_ms,
                'item_id':          str(item_id),
                'stratum':          stratum_lbl,
                'popularity_score': score,
                'actual_count':     actual_count,
                'train_count':      float(train_mean),
            })

        if not detailed_list:
            return results

        # --- Window-level rank + diagnostics (same as temporal_evaluator) ----
        scores_arr  = np.array([r['popularity_score'] for r in detailed_list], dtype=np.float64)
        actuals_arr = np.array([r['actual_count']     for r in detailed_list], dtype=np.float64)

        diag = calculate_diagnostics(scores_arr, actuals_arr)

        rank_pred   = (np.argsort(np.argsort(-scores_arr))  + 1)
        rank_actual = (np.argsort(np.argsort(-actuals_arr)) + 1)

        for i, r in enumerate(detailed_list):
            score_i  = scores_arr[i]
            actual_i = actuals_arr[i]
            error_i  = abs(score_i - actual_i)
            r['mae']            = float(diag['mae'])
            r['rank_predicted'] = int(rank_pred[i])
            r['rank_actual']    = int(rank_actual[i])
            r['squared_error']  = float(error_i ** 2)
            r['mape']           = float(error_i / actual_i * 100) if actual_i > 0 else 0.0

        results['detailed'] = detailed_list

        # --- Build per-stratum summary records -------------------------------
        # --- Build per-stratum summary from detailed_list -------------------
        for stratum_label in range(4):
            stratum_name = self.stratification.get_stratum_name(stratum_label)
            stratum_items = [r for r in detailed_list if r['stratum'] == stratum_label]

            empty_row = {
                'window_id':     window_idx,
                'timestamp':     timestamp_ms,
                'stratum':       stratum_label,
                'stratum_name':  stratum_name,
                'num_items':     0,
                'mean_mae':      0.0,
                'mean_mape':     0.0,
                'spearman_corr': 0.0,
                'kendall_tau':   0.0,
                'ndcg':          0.0,
                'coverage':      0.0,
            }

            if not stratum_items:
                results['summary'].append(empty_row)
                continue

            s_scores  = np.array([r['popularity_score'] for r in stratum_items], dtype=np.float64)
            s_actuals = np.array([r['actual_count']     for r in stratum_items], dtype=np.float64)

            try:
                s_diag   = calculate_diagnostics(s_scores, s_actuals)
                ndcg_val = calculate_ndcg(s_scores, s_actuals, k=10)

                with np.errstate(divide='ignore', invalid='ignore'):
                    mape = float(np.mean(
                        np.where(s_actuals != 0,
                                 np.abs((s_actuals - s_scores) / s_actuals) * 100,
                                 0.0)
                    ))
                coverage = float(np.mean(s_scores > 0))

                results['summary'].append({
                    'window_id':     window_idx,
                    'timestamp':     timestamp_ms,
                    'stratum':       stratum_label,
                    'stratum_name':  stratum_name,
                    'num_items':     len(stratum_items),
                    'mean_mae':      s_diag['mae'],
                    'mean_mape':     mape,
                    'spearman_corr': s_diag['spearman_rho'],
                    'kendall_tau':   s_diag['kendall_tau'],
                    'ndcg':          ndcg_val,
                    'coverage':      coverage,
                })

            except Exception:
                results['summary'].append({**empty_row, 'num_items': len(stratum_items)})

        return results

    # ==========================================================================
    # Stratification initialisation (UNCHANGED)
    # ==========================================================================

    # ==========================================================================
    # Item selection — EXACT copy of temporal_evaluator._select_items_from_data
    # ==========================================================================

    def _select_items_from_data(self) -> np.ndarray:
        """
        Select items using the same logic as TemporalEvaluator._select_items_from_data.
        Uses eval_range data only for item counting, then keeps all dates for selected items.
        """
        # Step 1: filter to eval range only (for counting, same as temporal)
        eval_data = self.data[
            (self.data['timestamp'] >= self.data_start) &
            (self.data['timestamp'] <= self.data_end)
        ]

        # Step 2: sum counts per item in eval range, apply min_observations filter
        item_counts = eval_data.groupby('item_id')['count'].sum()
        item_counts = item_counts[item_counts >= self.config.min_observations]

        if self.config.num_items is None:
            return item_counts.index.values
        elif self.config.item_selection == 'top':
            return item_counts.nlargest(self.config.num_items).index.values
        elif self.config.item_selection == 'random':
            n = min(self.config.num_items, len(item_counts))
            rng = np.random.RandomState(42)
            return rng.choice(item_counts.index.values, size=n, replace=False)
        elif self.config.item_selection == 'stratified':
            return self._stratified_sampling(item_counts, self.config.num_items)
        else:
            raise ValueError(f"Invalid item_selection: {self.config.item_selection}")

    def _stratified_sampling(self, item_counts: pd.Series, n: int) -> np.ndarray:
        """
        EXACT copy of temporal_evaluator._stratified_sampling.
        Proportional stratified sampling with fixed seed=42.
        """
        rng = np.random.RandomState(42)

        strata = self.stratification.stratify_items(
            pd.DataFrame({'item_id': item_counts.index, 'count': item_counts.values})
        )
        total_items = sum(len(items) for items in strata.values())
        selected = []
        for _, stratum_items in strata.items():
            if len(stratum_items) == 0:
                continue
            ratio    = len(stratum_items) / total_items
            n_sample = min(int(n * ratio), len(stratum_items))
            chosen   = rng.choice(stratum_items, size=n_sample, replace=False)
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

    def _initialize_stratification(self):
        """
        Compute stratification thresholds — mirrors temporal_evaluator:
        - If strata_thresholds in config: use fixed values
        - Otherwise: auto-compute from EVAL RANGE (not full dataset)
        """
        if self.config.strata_thresholds is not None:
            self.stratification.thresholds = list(self.config.strata_thresholds)
        else:
            # Use eval range only — same as temporal which calls stratify_items(eval_data)
            eval_data = self.data[
                (self.data['timestamp'] >= self.data_start) &
                (self.data['timestamp'] <= self.data_end)
            ]
            item_counts = eval_data.groupby('item_id')['count'].sum()
            q1 = item_counts.quantile(0.25)
            q2 = item_counts.quantile(0.50)
            q3 = item_counts.quantile(0.75)
            self.stratification.thresholds = [q1, q2, q3]

        print(f"Stratification thresholds initialized:")
        print(f"  Cold-start : < {self.stratification.thresholds[0]:.0f}")
        print(f"  Low        : {self.stratification.thresholds[0]:.0f} "
              f"– {self.stratification.thresholds[1]:.0f}")
        print(f"  Medium     : {self.stratification.thresholds[1]:.0f} "
              f"– {self.stratification.thresholds[2]:.0f}")
        print(f"  High       : >= {self.stratification.thresholds[2]:.0f}")
        print()

    # ==========================================================================
    # JSON persistence helpers (UNCHANGED)
    # ==========================================================================

    def _save_config_json(self):
        # Use EXACTLY the same config.save_config() as temporal_evaluator.
        # This guarantees both modes produce identical config.json files.
        # config.to_dict() contains ALL fields: start_date, end_date,
        # time_granularity, use_pre_range_data, strata_thresholds, wavelet_config,
        # k_list, robustness_sample_size, spike_multiplier, etc.
        config_path = self.storage_path / 'metadata' / 'config.json'
        config_path.parent.mkdir(parents=True, exist_ok=True)
        self.config.save_config(config_path)
        if self.config.verbose:
            print("  ✓ Saved config.json")

    def _save_thresholds_json(self):
        thresholds_dict = {
            'thresholds':  self.stratification.thresholds if hasattr(self.stratification, 'thresholds') else [],
            'strata_names': self.config.strata_names,
            'total_items': len(self.items)
        }
        thresholds_path = self.storage_path / 'metadata' / 'thresholds.json'
        with open(thresholds_path, 'w', encoding='utf-8') as f:
            json.dump(thresholds_dict, f, indent=2)
        if self.config.verbose:
            print("  ✓ Saved thresholds.json")

    def _save_runtime_stats_json(self):
        runtime_dict = {
            'total_duration':         self.runtime_stats.get('total_duration', 0),
            'total_duration_minutes': self.runtime_stats.get('total_duration', 0) / 60,
            'start_time':             self.runtime_stats.get('start_time'),
            'end_time':               self.runtime_stats.get('end_time'),
            'methods_stats':          self.runtime_stats.get('methods_stats', {})
        }
        runtime_path = self.storage_path / 'metadata' / 'runtime_stats.json'
        with open(runtime_path, 'w', encoding='utf-8') as f:
            json.dump(runtime_dict, f, indent=2)
        if self.config.verbose:
            print("  ✓ Saved runtime_stats.json")

    # ==========================================================================
    # Legacy _save_runtime_stats (kept for backward-compat)
    # ==========================================================================

    def _save_runtime_stats(self):
        stats = {
            **self.runtime_stats,
            'dataset_name':      self.config.dataset_name if hasattr(self.config, 'dataset_name') else 'unknown',
            'window_size':       self.config.window_size,
            'prediction_horizon': self.config.prediction_horizon,
            'time_granularity':  self.config.time_granularity if hasattr(self.config, 'time_granularity') else 'daily',
            'num_items':         len(self.items),
            'item_selection':    self.config.item_selection if hasattr(self.config, 'item_selection') else 'top',
            'min_observations':  self.config.min_observations,
            'start_date':        str(self.data_start.date()),
            'end_date':          str(self.data_end.date()),
            'use_pre_range_data': self.config.use_pre_range_data if hasattr(self.config, 'use_pre_range_data') else True,
            'methods':           list(self.methods.keys()),
            'strata_names':      self.config.strata_names,
        }
        self.storage.save_runtime_stats(stats)
        self.config.save_config(self.config.output_dir / 'metadata' / 'config.json')
        self.stratification.save_thresholds(
            self.config.output_dir / 'metadata' / 'thresholds.json'
        )


if __name__ == '__main__':
    print("\n✓ IncrementalTemporalEvaluator module loaded successfully")
    print("\nUsage:")
    print("  from evaluation.incremental_evaluator import IncrementalTemporalEvaluator")
    print("  evaluator = IncrementalTemporalEvaluator(...)")
    print("  evaluator.evaluate()")
