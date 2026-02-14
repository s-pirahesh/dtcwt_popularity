# -*- coding: utf-8 -*-
"""
Results Analyzer - تحلیل نتایج ذخیره‌شده
=========================================
دو حالت کار:
  display-only  : خواندن متریک‌های از پیش محاسبه‌شده (protocol/*.parquet/csv)
  recompute     : بازمحاسبه کامل 4-Layer Protocol از raw detailed scores

Author: Sajjad
Date: February 2025 (refactored for Frozen Evaluation Protocol)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
import json

from .metrics import (
    calculate_ndcg,
    calculate_hit_rate,
    calculate_rsi,
    calculate_rank_distortion,
    calculate_diagnostics,
    MetricsCalculator,
)
from .scenarios import RobustnessScenario


class ResultsAnalyzer:
    """
    تحلیل نتایج ذخیره‌شده بدون شبیه‌سازی مجدد.

    Features:
    - بارگذاری نتایج از Parquet/CSV
    - display-only: خواندن protocol metrics از پیش محاسبه‌شده
    - recompute: بازمحاسبه 4-Layer از raw detailed scores
    - مقایسه روش‌ها با جدول multi-K
    - فیلتر: stratum, time range, top-k%
    """

    K_VALUES: List[int] = [5, 10, 20]

    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        if not self.run_dir.exists():
            raise ValueError(f"Run directory not found: {run_dir}")

        self.config         = self._load_config()
        self.thresholds     = self._load_thresholds()
        self.runtime_stats  = self._load_runtime_stats()
        self.available_methods = self._detect_methods()

        # override K_VALUES from config if present
        if 'k_list' in self.config:
            self.K_VALUES = self.config['k_list']

        self._cache: Dict[str, pd.DataFrame] = {}

    # =========================================================================
    # Metadata loading
    # =========================================================================

    def _load_config(self) -> dict:
        p = self.run_dir / 'metadata' / 'config.json'
        return json.load(open(p)) if p.exists() else {}

    def _load_thresholds(self) -> dict:
        p = self.run_dir / 'metadata' / 'thresholds.json'
        return json.load(open(p)) if p.exists() else {}

    def _load_runtime_stats(self) -> dict:
        p = self.run_dir / 'metadata' / 'runtime_stats.json'
        return json.load(open(p)) if p.exists() else {}

    def _detect_methods(self) -> List[str]:
        """شناسایی روش‌های موجود از detailed/ یا protocol/"""
        methods = set()

        detailed_dir = self.run_dir / 'detailed'
        if detailed_dir.exists():
            for f in detailed_dir.glob('*_scores.parquet'):
                methods.add(f.stem.replace('_scores', ''))

        protocol_dir = self.run_dir / 'protocol'
        if protocol_dir.exists():
            for f in list(protocol_dir.glob('*_protocol.parquet')) + \
                     list(protocol_dir.glob('*_protocol.csv')):
                methods.add(f.stem.replace('_protocol', ''))

        return sorted(methods)

    # =========================================================================
    # Raw data loading
    # =========================================================================

    def load_detailed_scores(self, method_name: str,
                             use_cache: bool = True) -> pd.DataFrame:
        """بارگذاری detailed scores (Parquet)"""
        if use_cache and method_name in self._cache:
            return self._cache[method_name]

        fp = self.run_dir / 'detailed' / f'{method_name}_scores.parquet'
        if not fp.exists():
            raise FileNotFoundError(f"Detailed scores not found: {fp}")

        df = pd.read_parquet(fp)
        if 'timestamp' in df.columns:
            df['date'] = pd.to_datetime(df['timestamp'], unit='ms')

        if use_cache:
            self._cache[method_name] = df
        return df

    def load_stratum_summary(self, method_name: str) -> pd.DataFrame:
        """بارگذاری stratum summary (Parquet)"""
        fp = self.run_dir / 'summary' / f'{method_name}_stratum_summary.parquet'
        if not fp.exists():
            raise FileNotFoundError(f"Stratum summary not found: {fp}")
        df = pd.read_parquet(fp)
        if 'timestamp' in df.columns:
            df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df

    def load_protocol_metrics(self, method_name: str) -> Optional[pd.DataFrame]:
        """
        بارگذاری 4-Layer Protocol metrics از پیش محاسبه‌شده.
        Returns None if not found (run may be old/without protocol output).
        """
        proto_dir = self.run_dir / 'protocol'
        for ext in ['parquet', 'csv']:
            fp = proto_dir / f'{method_name}_protocol.{ext}'
            if fp.exists():
                df = pd.read_parquet(fp) if ext == 'parquet' else pd.read_csv(fp)
                if 'timestamp' in df.columns:
                    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
                return df
        return None

    def load_method_comparison(self) -> pd.DataFrame:
        for fmt in ['csv', 'parquet']:
            fp = self.run_dir / 'comparison' / f'method_comparison.{fmt}'
            if fp.exists():
                return pd.read_csv(fp) if fmt == 'csv' else pd.read_parquet(fp)
        raise FileNotFoundError("Method comparison file not found")

    # =========================================================================
    # Filter helpers
    # =========================================================================

    def filter_by_percentile(self, df: pd.DataFrame,
                             top_percent: float = 20.0) -> pd.DataFrame:
        threshold = df.groupby('item_id')['actual_count'].sum().quantile(
            1 - top_percent / 100
        )
        top_items = df.groupby('item_id')['actual_count'].sum()
        top_items = top_items[top_items >= threshold].index
        return df[df['item_id'].isin(top_items)]

    def filter_by_stratum(self, df: pd.DataFrame,
                          stratum: str) -> pd.DataFrame:
        stratum_map = {'cold_start': 0, 'low': 1, 'medium': 2, 'high': 3}
        sid = stratum_map.get(stratum)
        if sid is None:
            raise ValueError(f"Invalid stratum: {stratum}")
        return df[df['stratum'] == sid]

    def filter_by_time_range(self, df: pd.DataFrame,
                             start_date: str = None,
                             end_date: str = None) -> pd.DataFrame:
        if 'date' not in df.columns:
            df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
        if start_date:
            df = df[df['date'] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df['date'] <= pd.to_datetime(end_date)]
        return df

    # =========================================================================
    # display-only: read pre-computed protocol metrics
    # =========================================================================

    def get_protocol_summary(self, method_name: str) -> Optional[Dict]:
        """
        خواندن خلاصه آماری متریک‌های Protocol از فایل‌های ذخیره‌شده.
        Returns None if protocol file does not exist.
        """
        df = self.load_protocol_metrics(method_name)
        if df is None or len(df) == 0:
            return None

        summary = {'method': method_name, 'windows': len(df)}

        for k in self.K_VALUES:
            for metric in [f'ndcg@{k}', f'chr@{k}', f'rsi@{k}']:
                if metric in df.columns:
                    summary[metric] = float(df[metric].mean(skipna=True))

        for col in ['kendall_tau', 'spearman_rho', 'mae', 'robustness_distortion']:
            if col in df.columns:
                summary[col] = float(df[col].mean(skipna=True))

        return summary

    # =========================================================================
    # recompute: re-derive 4-Layer Protocol from raw detailed scores
    # =========================================================================

    def recompute_protocol_metrics(self, method_name: str,
                                   filter_top_percent: Optional[float] = None,
                                   filter_stratum: Optional[str] = None,
                                   start_date: Optional[str] = None,
                                   end_date: Optional[str] = None) -> pd.DataFrame:
        """
        بازمحاسبه کامل 4-Layer Protocol از raw detailed scores.

        برای هر window_id:
          - Layer 1: NDCG@K, CHR@K
          - Layer 2: Kendall τ, Spearman ρ, MAE
          - Layer 3: RSI@K (Jaccard با window قبلی)
          - Layer 4: Rank Distortion (با RobustnessScenario روی اسکور‌ها)

        Returns:
            DataFrame با یک سطر برای هر window_id
        """
        df = self.load_detailed_scores(method_name)

        # اعمال فیلترها
        if filter_top_percent:
            df = self.filter_by_percentile(df, filter_top_percent)
        if filter_stratum:
            df = self.filter_by_stratum(df, filter_stratum)
        if start_date or end_date:
            df = self.filter_by_time_range(df, start_date, end_date)

        if len(df) == 0:
            return pd.DataFrame()

        scenario = RobustnessScenario(sample_size=50, spike_multiplier=10.0)
        prev_top_k: Dict[int, List] = {}
        records = []

        for window_id, grp in df.groupby('window_id', sort=True):
            scores  = grp['popularity_score'].values.astype(np.float64)
            actuals = grp['actual_count'].values.astype(np.float64)
            ts_ms   = int(grp['timestamp'].iloc[0])

            if len(scores) < 2:
                continue

            row: Dict = {
                'window_id': window_id,
                'timestamp': ts_ms,
                'date':      pd.to_datetime(ts_ms, unit='ms'),
                'method':    method_name,
                'num_items': len(scores),
            }

            # ---- Layer 1: Decision ------------------------------------------
            for k in self.K_VALUES:
                row[f'ndcg@{k}'] = calculate_ndcg(scores, actuals, k=k)
                row[f'chr@{k}']  = calculate_hit_rate(scores, actuals, k=k)

            # ---- Layer 2: Diagnostics ----------------------------------------
            diag = calculate_diagnostics(scores, actuals)
            row['kendall_tau']  = diag['kendall_tau']
            row['spearman_rho'] = diag['spearman_rho']
            row['mae']          = diag['mae']

            # ---- Layer 3: Stability (RSI) ------------------------------------
            for k in self.K_VALUES:
                top_k_now = np.argsort(scores)[-k:][::-1].tolist()
                prev = prev_top_k.get(k)
                row[f'rsi@{k}'] = float('nan') if prev is None \
                    else calculate_rsi(prev, top_k_now)
                prev_top_k[k] = top_k_now

            # ---- Layer 4: Robustness (synthetic — on score vector) ----------
            # چون داده‌های اولیه time-series موجود نیست، از بردار امتیازها
            # به عنوان یک time-series تک‌بُعدی تقریب می‌زنیم.
            try:
                scores_2d = scores.reshape(1, -1)   # 1 × N  (N = items as "time")
                target_idxs = scenario.select_stable_candidates(scores_2d.T)

                if target_idxs:
                    distortions = []
                    for idx in target_idxs:
                        noisy = scores.copy()
                        mean_v = np.mean(scores)
                        noisy[idx] += mean_v * scenario.spike_multiplier
                        distortions.append(float(
                            calculate_rank_distortion(scores, noisy, idx)
                        ))
                    row['robustness_distortion'] = float(np.mean(distortions))
                else:
                    row['robustness_distortion'] = float('nan')
            except Exception:
                row['robustness_distortion'] = float('nan')

            records.append(row)

        return pd.DataFrame(records)

    # =========================================================================
    # Aggregate metrics (used by both display and recompute paths)
    # =========================================================================

    def calculate_overall_metrics(self, method_name: str,
                                  filter_top_percent: Optional[float] = None,
                                  filter_stratum: Optional[str] = None,
                                  start_date: Optional[str] = None,
                                  end_date: Optional[str] = None,
                                  recompute: bool = False) -> Dict:
        """
        محاسبه / خواندن معیارهای کلی یک روش.

        Args:
            recompute: اگر True بازمحاسبه از raw scores انجام می‌شود؛
                       اگر False از protocol metrics ذخیره‌شده می‌خواند
                       (در صورت نبود، fallback به raw scores).
        """
        if recompute or any([filter_top_percent, filter_stratum,
                              start_date, end_date]):
            # بازمحاسبه از raw scores (تنها راه برای اعمال فیلتر)
            proto_df = self.recompute_protocol_metrics(
                method_name, filter_top_percent, filter_stratum,
                start_date, end_date
            )
        else:
            # سعی در خواندن protocol metrics ذخیره‌شده
            proto_df = self.load_protocol_metrics(method_name)
            if proto_df is None:
                # fallback
                proto_df = self.recompute_protocol_metrics(method_name)

        if proto_df is None or len(proto_df) == 0:
            return {}

        result = {'num_samples': len(proto_df)}
        for col in proto_df.columns:
            if col not in ('window_id', 'timestamp', 'date', 'method', 'num_items'):
                try:
                    result[col] = float(proto_df[col].mean(skipna=True))
                except Exception:
                    pass

        # backward-compat aliases
        result.setdefault('spearman', result.get('spearman_rho', 0.0))
        result.setdefault('kendall',  result.get('kendall_tau', 0.0))
        result.setdefault('ndcg',     result.get('ndcg@10', 0.0))
        result.setdefault('coverage', 0.0)
        result.setdefault('rmse',     0.0)
        result.setdefault('mape',     0.0)

        return result

    def compare_methods(self,
                        filter_top_percent: Optional[float] = None,
                        filter_stratum: Optional[str] = None,
                        start_date: Optional[str] = None,
                        end_date: Optional[str] = None,
                        recompute: bool = False) -> pd.DataFrame:
        """
        مقایسه همه روش‌ها در یک جدول.

        Args:
            recompute: بازمحاسبه از raw scores (به‌جای خواندن فایل ذخیره‌شده)
        """
        rows = []
        for method_name in self.available_methods:
            m = self.calculate_overall_metrics(
                method_name, filter_top_percent, filter_stratum,
                start_date, end_date, recompute=recompute
            )
            if m:
                rows.append({'method': method_name, **m})

        df = pd.DataFrame(rows)
        if len(df) > 0 and 'spearman_rho' in df.columns:
            df = df.sort_values('spearman_rho', ascending=False)
        elif len(df) > 0 and 'spearman' in df.columns:
            df = df.sort_values('spearman', ascending=False)
        return df

    # =========================================================================
    # Temporal evolution
    # =========================================================================

    def get_temporal_evolution(self, method_name: str,
                               metric: str = 'spearman_rho',
                               window_agg: str = 'mean',
                               recompute: bool = False) -> pd.DataFrame:
        """
        تکامل زمانی یک معیار (از protocol metrics یا bازمحاسبه).

        Args:
            metric: نام ستون (مثلاً 'spearman_rho', 'ndcg@10', 'rsi@10')
            recompute: بازمحاسبه از raw scores
        """
        if recompute:
            df = self.recompute_protocol_metrics(method_name)
        else:
            df = self.load_protocol_metrics(method_name)
            if df is None:
                df = self.recompute_protocol_metrics(method_name)

        if df is None or len(df) == 0:
            raise ValueError(f"No protocol data for {method_name}")

        if metric not in df.columns:
            # fallback: try from detailed
            raw = self.load_detailed_scores(method_name)
            if metric not in raw.columns:
                raise ValueError(f"Metric '{metric}' not found")
            agg_fn = {'mean': 'mean', 'median': 'median', 'std': 'std'}[window_agg]
            result = raw.groupby('window_id')[metric].agg(agg_fn)
            timestamps = raw.groupby('window_id')['date'].first()
            return pd.DataFrame({'date': timestamps, metric: result}).reset_index(drop=True)

        agg_fn = {'mean': 'mean', 'median': 'median', 'std': 'std'}[window_agg]
        result = df.groupby('window_id')[metric].agg(agg_fn)
        timestamps = df.groupby('window_id')['date'].first()
        return pd.DataFrame({'date': timestamps, metric: result}).reset_index(drop=True)

    def get_stratum_comparison(self, method_name: str,
                               metric: str = 'spearman_corr') -> pd.DataFrame:
        summary = self.load_stratum_summary(method_name)
        if metric not in summary.columns:
            raise ValueError(f"Metric not found: {metric}")
        result = summary.groupby('stratum_name')[metric].agg(['mean', 'std', 'min', 'max'])
        return result.reset_index()

    # =========================================================================
    # Print summary
    # =========================================================================

    def print_summary(self):
        print("\n" + "="*70)
        print("RESULTS SUMMARY")
        print("="*70)
        print(f"Run Directory:     {self.run_dir}")
        print(f"Dataset:           {self.config.get('dataset_name', 'N/A')}")
        print(f"Window Size:       {self.config.get('window_size', 'N/A')} days")
        print(f"Items:             {self.config.get('num_items', 'all')}")
        print(f"K values:          {self.config.get('k_list', self.K_VALUES)}")
        print(f"Available Methods: {', '.join(self.available_methods)}")

        # بررسی وجود protocol output
        proto_dir = self.run_dir / 'protocol'
        has_proto = proto_dir.exists() and any(proto_dir.iterdir())
        print(f"Protocol data:     {'✓ found' if has_proto else '✗ not found (use --recompute)'}")

        if self.runtime_stats:
            dur = self.runtime_stats.get('total_duration', 0)
            h, m = int(dur // 3600), int((dur % 3600) // 60)
            print(f"Total Duration:    {h}h {m}m")
        print("="*70 + "\n")