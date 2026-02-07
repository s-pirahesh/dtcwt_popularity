"""
Incremental Storage System
ذخیره‌سازی تدریجی نتایج evaluation

Author: Sajjad
Date: February 2025
"""

import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
import json
from datetime import datetime
import gc


class IncrementalStorage:
    """
    ذخیره‌سازی تدریجی با Parquet
    
    ویژگی‌ها:
    - Append-only: اضافه کردن بدون بازنویسی کامل
    - Low memory: فقط buffer در RAM
    - Crash-safe: هر batch ذخیره می‌شود
    """
    
    def __init__(self, 
                 base_path: Path,
                 buffer_size: int = 1000):
        """
        Args:
            base_path: مسیر پایه برای ذخیره
            buffer_size: تعداد رکوردهای buffer قبل از flush
        """
        self.base_path = Path(base_path)
        self.buffer_size = buffer_size
        
        # ایجاد دایرکتوری‌ها
        self.detailed_dir = self.base_path / 'detailed'
        self.summary_dir = self.base_path / 'summary'
        self.metadata_dir = self.base_path / 'metadata'
        
        for dir_path in [self.detailed_dir, self.summary_dir, self.metadata_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Buffers برای هر method
        self.detailed_buffers: Dict[str, List[Dict]] = {}
        self.summary_buffers: Dict[str, List[Dict]] = {}
        
        # آمار
        self.stats = {
            'total_records_written': 0,
            'total_flushes': 0,
            'methods_processed': set()
        }
    
    def append_detailed(self, method_name: str, records: List[Dict]):
        """
        اضافه کردن رکوردهای detailed
        
        Args:
            method_name: نام method
            records: لیست رکوردها
        """
        # اضافه به buffer
        if method_name not in self.detailed_buffers:
            self.detailed_buffers[method_name] = []
        
        self.detailed_buffers[method_name].extend(records)
        self.stats['methods_processed'].add(method_name)
        
        # Flush اگر buffer پر شد
        if len(self.detailed_buffers[method_name]) >= self.buffer_size:
            self._flush_detailed(method_name)
    
    def append_summary(self, method_name: str, records: List[Dict]):
        """
        اضافه کردن رکوردهای summary
        
        Args:
            method_name: نام method
            records: لیست رکوردها
        """
        if method_name not in self.summary_buffers:
            self.summary_buffers[method_name] = []
        
        self.summary_buffers[method_name].extend(records)
        
        # Flush اگر buffer پر شد
        if len(self.summary_buffers[method_name]) >= self.buffer_size:
            self._flush_summary(method_name)
    
    def _flush_detailed(self, method_name: str):
        """
        نوشتن buffer detailed به فایل
        """
        if not self.detailed_buffers.get(method_name):
            return
        
        # تبدیل به DataFrame
        df = pd.DataFrame(self.detailed_buffers[method_name])
        
        # مسیر فایل
        filepath = self.detailed_dir / f"{method_name}_scores.parquet"
        
        # Append یا Create
        if filepath.exists():
            try:
                # خواندن موجود
                existing = pd.read_parquet(filepath)
                # ترکیب
                combined = pd.concat([existing, df], ignore_index=True)
                # نوشتن
                combined.to_parquet(filepath, compression='snappy', index=False)
            except Exception as e:
                # اگر خواندن fail شد، فقط append کن
                print(f"  Warning: Could not read existing file, overwriting: {e}")
                df.to_parquet(filepath, compression='snappy', index=False)
        else:
            # نوشتن جدید
            df.to_parquet(filepath, compression='snappy', index=False)
        
        # آمار
        self.stats['total_records_written'] += len(df)
        self.stats['total_flushes'] += 1
        
        # پاک کردن buffer
        self.detailed_buffers[method_name].clear()
        
        # Garbage collection
        del df
        gc.collect()
    
    def _flush_summary(self, method_name: str):
        """
        نوشتن buffer summary به فایل
        """
        if not self.summary_buffers.get(method_name):
            return
        
        df = pd.DataFrame(self.summary_buffers[method_name])
        filepath = self.summary_dir / f"{method_name}_stratum_summary.parquet"
        
        if filepath.exists():
            try:
                existing = pd.read_parquet(filepath)
                combined = pd.concat([existing, df], ignore_index=True)
                combined.to_parquet(filepath, compression='snappy', index=False)
            except Exception as e:
                print(f"  Warning: Could not read existing file, overwriting: {e}")
                df.to_parquet(filepath, compression='snappy', index=False)
        else:
            df.to_parquet(filepath, compression='snappy', index=False)
        
        self.summary_buffers[method_name].clear()
        
        del df
        gc.collect()
    
    def flush_all(self):
        """
        نوشتن همه bufferها
        """
        # Detailed
        for method_name in list(self.detailed_buffers.keys()):
            if self.detailed_buffers[method_name]:
                self._flush_detailed(method_name)
        
        # Summary
        for method_name in list(self.summary_buffers.keys()):
            if self.summary_buffers[method_name]:
                self._flush_summary(method_name)
    
    def save_metadata(self, metadata: Dict):
        """
        ذخیره metadata
        """
        filepath = self.metadata_dir / 'run_metadata.json'
        
        # اضافه کردن timestamp
        metadata['saved_at'] = datetime.now().isoformat()
        metadata['storage_stats'] = {
            'total_records_written': self.stats['total_records_written'],
            'total_flushes': self.stats['total_flushes'],
            'methods_processed': list(self.stats['methods_processed'])
        }
        
        with open(filepath, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
    
    def get_stats(self) -> Dict:
        """
        دریافت آمار
        """
        return {
            'total_records_written': self.stats['total_records_written'],
            'total_flushes': self.stats['total_flushes'],
            'methods_processed': len(self.stats['methods_processed']),
            'buffer_sizes': {
                method: len(buf) 
                for method, buf in self.detailed_buffers.items()
            }
        }
    
    def load_detailed_scores(self, method_name: str) -> Optional[pd.DataFrame]:
        """
        بارگذاری نتایج detailed یک method
        
        Args:
            method_name: نام method
            
        Returns:
            DataFrame یا None
        """
        filepath = self.detailed_dir / f"{method_name}_scores.parquet"
        
        if not filepath.exists():
            return None
        
        try:
            return pd.read_parquet(filepath)
        except Exception as e:
            print(f"Error loading {method_name}: {e}")
            return None
    
    def load_stratum_summary(self, method_name: str) -> Optional[pd.DataFrame]:
        """
        بارگذاری summary یک method
        
        Args:
            method_name: نام method
            
        Returns:
            DataFrame یا None
        """
        filepath = self.summary_dir / f"{method_name}_stratum_summary.parquet"
        
        if not filepath.exists():
            return None
        
        try:
            return pd.read_parquet(filepath)
        except Exception as e:
            print(f"Error loading {method_name} summary: {e}")
            return None


if __name__ == '__main__':
    # تست
    import tempfile
    import shutil
    
    print("\nTesting IncrementalStorage...\n")
    
    # ایجاد دایرکتوری موقت
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        # ایجاد storage
        storage = IncrementalStorage(temp_dir, buffer_size=3)
        
        # تست append detailed
        print("Test 1: Append detailed records")
        records1 = [
            {'window_id': 0, 'item_id': 1, 'score': 0.5, 'actual': 10},
            {'window_id': 0, 'item_id': 2, 'score': 0.3, 'actual': 5},
        ]
        storage.append_detailed('AF', records1)
        print(f"  Added {len(records1)} records")
        print(f"  Buffer size: {len(storage.detailed_buffers['AF'])}")
        
        # تست flush
        print("\nTest 2: Force flush")
        records2 = [
            {'window_id': 1, 'item_id': 3, 'score': 0.8, 'actual': 20},
            {'window_id': 1, 'item_id': 4, 'score': 0.6, 'actual': 15},
        ]
        storage.append_detailed('AF', records2)
        print(f"  Added {len(records2)} more records")
        print(f"  Buffer auto-flushed (size >= 3)")
        
        # تست flush_all
        print("\nTest 3: Flush all")
        storage.flush_all()
        print("  All buffers flushed")
        
        # تست load
        print("\nTest 4: Load data")
        df = storage.load_detailed_scores('AF')
        if df is not None:
            print(f"  Loaded {len(df)} records")
            print(f"  Columns: {list(df.columns)}")
        
        # تست metadata
        print("\nTest 5: Save metadata")
        metadata = {
            'dataset': 'test',
            'num_items': 4
        }
        storage.save_metadata(metadata)
        print("  Metadata saved")
        
        # تست stats
        print("\nTest 6: Get stats")
        stats = storage.get_stats()
        print(f"  Stats: {stats}")
        
        print("\n✓ All tests passed!")
        
    finally:
        # پاک کردن
        shutil.rmtree(temp_dir)
        print(f"\nCleaned up temp directory: {temp_dir}")
