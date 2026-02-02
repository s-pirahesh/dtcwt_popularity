# -*- coding: utf-8 -*-
"""
Storage System with Smart Format Handling
- Intermediate results: Always Parquet (high volume)
- Final results: Configurable CSV or Parquet
- Reading: Supports both formats automatically
Author: Sajjad
Date: February 2025
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
import json


class StorageSystem:
    """
    سیستم ذخیره‌سازی هوشمند
    
    Rules:
    - نتایج میانی (detailed, summary): همیشه Parquet
    - نتایج نهایی (comparison): قابل انتخاب (CSV یا Parquet)
    - خواندن: خودکار هر دو فرمت را تشخیص می‌دهد
    """
    
    def __init__(self, config):
        """
        Args:
            config: EvaluationConfig instance
        """
        self.config = config
        self.output_dir = config.output_dir
        self.final_format = config.final_format
        self.compression = config.compression
        
        # بررسی در دسترس بودن pyarrow
        try:
            import pyarrow
            self.has_pyarrow = True
        except ImportError:
            self.has_pyarrow = False
            print("Warning: pyarrow not installed. Install with: pip install pyarrow")
    
    def save_detailed_scores(self, method_name: str, results: List[Dict]):
        """
        ذخیره نتایج تفصیلی - همیشه Parquet
        
        Args:
            method_name: نام روش
            results: لیست dictionary های نتایج
        """
        if not self.config.save_detailed:
            return
        
        df = pd.DataFrame(results)
        
        # تبدیل timestamp به int64
        if 'timestamp' in df.columns:
            df['timestamp'] = df['timestamp'].astype('int64')
        
        # ذخیره - همیشه Parquet
        filename = f"{method_name}_scores.parquet"
        filepath = self.output_dir / 'detailed' / filename
        
        self._save_parquet(df, filepath)
        
        if self.config.verbose:
            file_size = filepath.stat().st_size / (1024 * 1024)  # MB
            print(f"  Saved: {filename} ({file_size:.1f} MB)")
    
    def save_stratum_summary(self, method_name: str, summary: List[Dict]):
        """
        ذخیره خلاصه stratum - همیشه Parquet
        
        Args:
            method_name: نام روش
            summary: لیست dictionary های خلاصه
        """
        if not self.config.save_summary:
            return
        
        df = pd.DataFrame(summary)
        
        # تبدیل timestamp
        if 'timestamp' in df.columns:
            df['timestamp'] = df['timestamp'].astype('int64')
        
        # ذخیره - همیشه Parquet
        filename = f"{method_name}_stratum_summary.parquet"
        filepath = self.output_dir / 'summary' / filename
        
        self._save_parquet(df, filepath)
    
    def save_method_comparison(self, comparison_df: pd.DataFrame):
        """
        ذخیره مقایسه روش‌ها - قابل انتخاب (CSV یا Parquet)
        
        Args:
            comparison_df: DataFrame مقایسه
        """
        filename = f"method_comparison.{self.final_format}"
        filepath = self.output_dir / 'comparison' / filename
        
        if self.final_format == 'parquet':
            self._save_parquet(comparison_df, filepath)
        else:  # csv
            comparison_df.to_csv(filepath, index=False, encoding='utf-8')
    
    def _save_parquet(self, df: pd.DataFrame, filepath: Path):
        """
        ذخیره Parquet با بررسی pyarrow
        
        Args:
            df: DataFrame
            filepath: مسیر فایل
        """
        if not self.has_pyarrow:
            raise ImportError(
                "pyarrow is required for Parquet format. "
                "Install it with: pip install pyarrow"
            )
        
        df.to_parquet(
            filepath,
            compression=self.compression,
            engine='pyarrow',
            index=False
        )
    
    def load_detailed_scores(self, method_name: str) -> pd.DataFrame:
        """
        بارگذاری نتایج تفصیلی - خودکار Parquet
        
        Args:
            method_name: نام روش
        
        Returns:
            DataFrame
        """
        filename = f"{method_name}_scores.parquet"
        filepath = self.output_dir / 'detailed' / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        return pd.read_parquet(filepath, engine='pyarrow')
    
    def load_stratum_summary(self, method_name: str) -> pd.DataFrame:
        """
        بارگذاری خلاصه stratum - خودکار Parquet
        
        Args:
            method_name: نام روش
        
        Returns:
            DataFrame
        """
        filename = f"{method_name}_stratum_summary.parquet"
        filepath = self.output_dir / 'summary' / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        return pd.read_parquet(filepath, engine='pyarrow')
    
    def load_method_comparison(self) -> pd.DataFrame:
        """
        بارگذاری مقایسه روش‌ها - خودکار (CSV یا Parquet)
        
        Returns:
            DataFrame
        """
        # سعی کردن هر دو فرمت
        for fmt in ['parquet', 'csv']:
            filename = f"method_comparison.{fmt}"
            filepath = self.output_dir / 'comparison' / filename
            
            if filepath.exists():
                if fmt == 'parquet':
                    return pd.read_parquet(filepath, engine='pyarrow')
                else:
                    return pd.read_csv(filepath, encoding='utf-8')
        
        raise FileNotFoundError("Method comparison file not found (tried both CSV and Parquet)")
    
    def save_metadata(self, metadata: Dict):
        """
        ذخیره اطلاعات metadata - همیشه JSON
        
        Args:
            metadata: dictionary اطلاعات
        """
        filepath = self.output_dir / 'metadata' / 'run_metadata.json'
        
        # تبدیل numpy types به Python types
        metadata = self._convert_to_json_serializable(metadata)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    def save_runtime_stats(self, stats: Dict):
        """
        ذخیره آمار زمان اجرا - همیشه JSON
        
        Args:
            stats: dictionary آمار
        """
        filepath = self.output_dir / 'metadata' / 'runtime_stats.json'
        
        stats = self._convert_to_json_serializable(stats)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
    
    def _convert_to_json_serializable(self, obj):
        """تبدیل numpy types به Python types"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, Path):
            return str(obj)
        return obj
    
    def get_file_sizes(self) -> Dict[str, float]:
        """
        محاسبه اندازه فایل‌ها
        
        Returns:
            dict: {file_path: size_in_mb}
        """
        sizes = {}
        
        for subdir in ['detailed', 'summary', 'comparison', 'metadata']:
            path = self.output_dir / subdir
            if not path.exists():
                continue
            
            # بررسی هر دو فرمت
            for file in path.glob('*'):
                if file.is_file():
                    size_mb = file.stat().st_size / (1024 * 1024)
                    sizes[str(file.relative_to(self.output_dir))] = size_mb
        
        return sizes
    
    def print_storage_summary(self):
        """چاپ خلاصه فضای ذخیره‌سازی"""
        sizes = self.get_file_sizes()
        
        if not sizes:
            print("No files found.")
            return
        
        print("\n" + "="*70)
        print("STORAGE SUMMARY")
        print("="*70)
        
        total_size = 0
        
        for category in ['detailed', 'summary', 'comparison']:
            category_files = {k: v for k, v in sizes.items() if k.startswith(category)}
            
            if not category_files:
                continue
            
            category_size = sum(category_files.values())
            total_size += category_size
            
            print(f"\n{category.upper()}:")
            for filename, size in sorted(category_files.items()):
                # نمایش فرمت
                fmt = "Parquet" if filename.endswith('.parquet') else "CSV"
                print(f"  {filename:<40} {size:>8.2f} MB ({fmt})")
            print(f"  {'Subtotal:':<40} {category_size:>8.2f} MB")
        
        print(f"\n{'TOTAL:':<40} {total_size:>8.2f} MB")
        print("="*70 + "\n")
        
        # نمایش فرمت‌ها
        print("FORMAT DETAILS:")
        print(f"  Intermediate (detailed/summary): Parquet (compressed)")
        print(f"  Final (comparison): {self.final_format.upper()}")
        print("="*70 + "\n")
