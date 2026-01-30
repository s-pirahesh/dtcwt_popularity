"""
Data Converters: خام → استاندارد
تبدیل دیتاست‌های خام به فرمت CSV استاندارد
"""
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, List, Dict, Optional, Any
from tqdm import tqdm
import logging


class BaseConverter(ABC):
    """
    کلاس پایه برای تبدیل دیتاست‌های خام به فرمت استاندارد
    
    فرمت استاندارد خروجی:
        - timestamp: زمان دسترسی (datetime)
        - item_id: شناسه آیتم (str/int)
        - count: تعداد دسترسی یا رتبه (int/float) - اختیاری
        - ستون‌های اضافی دیتاست (category, tags, ...)
    
    پشتیبانی:
        - تک‌فایل یا چندفایل
        - ترکیب خودکار چند فایل
        - Validation و لاگ‌گیری
        - Progress bar
    """
    
    def __init__(self, 
                 output_format: str = 'csv',
                 verbose: bool = True,
                 validate_output: bool = True):
        """
        مقداردهی اولیه
        
        Args:
            output_format: فرمت خروجی ('csv', 'parquet', 'feather')
            verbose: نمایش پیام‌های پیشرفت
            validate_output: اعتبارسنجی خروجی
        """
        self.output_format = output_format
        self.verbose = verbose
        self.validate_output = validate_output
        
        # تنظیم logging
        self.logger = logging.getLogger(self.__class__.__name__)
        if verbose:
            self.logger.setLevel(logging.INFO)
    
    def convert(self, 
                input_path: Union[str, Path, List[Union[str, Path]]],
                output_path: Union[str, Path],
                **kwargs) -> pd.DataFrame:
        """
        تبدیل اصلی: خام → استاندارد
        
        Args:
            input_path: مسیر فایل(ها) خام - می‌تواند تک‌فایل یا لیست باشد
            output_path: مسیر ذخیره فایل استاندارد
            **kwargs: پارامترهای اضافی
            
        Returns:
            DataFrame تبدیل شده
        """
        self.log(f"شروع تبدیل: {self.__class__.__name__}")
        
        # تبدیل به Path
        if isinstance(input_path, (str, Path)):
            input_paths = [Path(input_path)]
        else:
            input_paths = [Path(p) for p in input_path]
        
        output_path = Path(output_path)
        
        # بررسی وجود فایل‌های ورودی
        self._check_input_files(input_paths)
        
        # تبدیل
        if len(input_paths) == 1:
            # تک‌فایل
            self.log(f"تبدیل تک‌فایل: {input_paths[0].name}")
            df = self._convert_single_file(input_paths[0], **kwargs)
        else:
            # چندفایل - ترکیب
            self.log(f"تبدیل و ترکیب {len(input_paths)} فایل")
            df = self._convert_multiple_files(input_paths, **kwargs)
        
        # Validation
        if self.validate_output:
            self.log("اعتبارسنجی خروجی...")
            self._validate_output(df)
        
        # ذخیره
        self.log(f"ذخیره در: {output_path}")
        self._save_output(df, output_path)
        
        # آمار
        self._print_statistics(df)
        
        self.log("✓ تبدیل با موفقیت انجام شد")
        
        return df
    
    @abstractmethod
    def _convert_single_file(self, 
                            file_path: Path,
                            **kwargs) -> pd.DataFrame:
        """
        تبدیل یک فایل خام به DataFrame استاندارد
        
        این متد باید توسط کلاس‌های فرزند پیاده‌سازی شود
        
        Args:
            file_path: مسیر فایل خام
            **kwargs: پارامترهای اضافی
            
        Returns:
            DataFrame با ستون‌های استاندارد
        """
        pass
    
    def _convert_multiple_files(self,
                                file_paths: List[Path],
                                **kwargs) -> pd.DataFrame:
        """
        تبدیل و ترکیب چند فایل
        
        Args:
            file_paths: لیست مسیر فایل‌ها
            **kwargs: پارامترهای اضافی
            
        Returns:
            DataFrame ترکیب شده
        """
        dfs = []
        
        # تبدیل هر فایل
        for file_path in tqdm(file_paths, 
                             desc="Converting files",
                             disable=not self.verbose):
            df = self._convert_single_file(file_path, **kwargs)
            dfs.append(df)
        
        # ترکیب
        self.log("ترکیب فایل‌ها...")
        combined_df = pd.concat(dfs, ignore_index=True)
        
        # مرتب‌سازی بر اساس زمان
        if 'timestamp' in combined_df.columns:
            combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)
        
        return combined_df
    
    def _check_input_files(self, file_paths: List[Path]):
        """بررسی وجود فایل‌های ورودی"""
        for file_path in file_paths:
            if not file_path.exists():
                raise FileNotFoundError(f"فایل یافت نشد: {file_path}")
    
    def _validate_output(self, df: pd.DataFrame):
        """
        اعتبارسنجی DataFrame خروجی
        
        بررسی:
            - وجود ستون‌های الزامی
            - نوع داده‌ها
            - مقادیر null
        """
        # ستون‌های الزامی
        required_columns = ['timestamp', 'item_id']
        missing = set(required_columns) - set(df.columns)
        
        if missing:
            raise ValueError(f"ستون‌های الزامی وجود ندارند: {missing}")
        
        # بررسی timestamp
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            self.logger.warning("ستون timestamp از نوع datetime نیست، در حال تبدیل...")
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # بررسی null
        null_counts = df[required_columns].isnull().sum()
        if null_counts.any():
            self.logger.warning(f"مقادیر null در ستون‌های الزامی:\n{null_counts[null_counts > 0]}")
        
        # بررسی تعداد رکوردها
        if len(df) == 0:
            raise ValueError("DataFrame خروجی خالی است!")
        
        self.log(f"✓ Validation موفق: {len(df)} رکورد")
    
    def _save_output(self, df: pd.DataFrame, output_path: Path):
        """
        ذخیره DataFrame به فرمت مورد نظر
        
        Args:
            df: DataFrame برای ذخیره
            output_path: مسیر خروجی
        """
        # ایجاد پوشه در صورت عدم وجود
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # ذخیره بر اساس فرمت
        if self.output_format == 'csv':
            df.to_csv(output_path, index=False)
        elif self.output_format == 'parquet':
            df.to_parquet(output_path, index=False)
        elif self.output_format == 'feather':
            df.to_feather(output_path)
        else:
            raise ValueError(f"فرمت نامعتبر: {self.output_format}")
    
    def _print_statistics(self, df: pd.DataFrame):
        """نمایش آمار کلی DataFrame"""
        if not self.verbose:
            return
        
        stats = {
            'تعداد کل رکوردها': len(df),
            'تعداد آیتم‌های یکتا': df['item_id'].nunique(),
            'محدوده زمانی': f"{df['timestamp'].min()} تا {df['timestamp'].max()}",
        }
        
        if 'count' in df.columns:
            stats['مجموع دسترسی‌ها'] = df['count'].sum()
            stats['میانگین دسترسی در رکورد'] = df['count'].mean()
        
        self.log("\n" + "="*50)
        self.log("آمار داده‌های تبدیل شده:")
        for key, value in stats.items():
            self.log(f"  {key}: {value}")
        self.log("="*50)
    
    def log(self, message: str):
        """لاگ پیام"""
        if self.verbose:
            self.logger.info(message)
    
    @staticmethod
    def parse_timestamp(timestamp_str: str,
                       format: Optional[str] = None,
                       unit: Optional[str] = None) -> pd.Timestamp:
        """
        تبدیل رشته یا عدد به timestamp
        
        Args:
            timestamp_str: رشته یا عدد timestamp
            format: فرمت تاریخ (مثل '%Y-%m-%d %H:%M:%S')
            unit: واحد برای timestamp عددی ('s', 'ms', 'us', 'ns')
            
        Returns:
            pd.Timestamp
        """
        if format:
            return pd.to_datetime(timestamp_str, format=format)
        elif unit:
            return pd.to_datetime(timestamp_str, unit=unit)
        else:
            return pd.to_datetime(timestamp_str)
    
    @staticmethod
    def aggregate_duplicates(df: pd.DataFrame,
                           group_by: List[str] = ['timestamp', 'item_id'],
                           agg_func: str = 'sum') -> pd.DataFrame:
        """
        ترکیب رکوردهای تکراری
        
        Args:
            df: DataFrame ورودی
            group_by: ستون‌های گروه‌بندی
            agg_func: تابع ترکیب ('sum', 'count', 'mean')
            
        Returns:
            DataFrame بدون تکرار
        """
        if 'count' in df.columns:
            agg_dict = {col: 'first' for col in df.columns if col not in group_by + ['count']}
            agg_dict['count'] = agg_func
            return df.groupby(group_by, as_index=False).agg(agg_dict)
        else:
            return df.drop_duplicates(subset=group_by)


class ConverterFactory:
    """
    Factory برای ایجاد converter مناسب
    """
    
    _converters = {}
    
    @classmethod
    def register(cls, dataset_name: str, converter_class):
        """ثبت converter جدید"""
        cls._converters[dataset_name.lower()] = converter_class
    
    @classmethod
    def create(cls, dataset_name: str, **kwargs) -> BaseConverter:
        """
        ایجاد converter مناسب
        
        Args:
            dataset_name: نام دیتاست
            **kwargs: پارامترهای converter
            
        Returns:
            نمونه‌ای از Converter
        """
        dataset_name = dataset_name.lower()
        
        if dataset_name not in cls._converters:
            raise ValueError(
                f"Converter برای '{dataset_name}' یافت نشد. "
                f"دیتاست‌های موجود: {list(cls._converters.keys())}"
            )
        
        converter_class = cls._converters[dataset_name]
        return converter_class(**kwargs)
    
    @classmethod
    def list_converters(cls) -> List[str]:
        """لیست دیتاست‌های پشتیبانی شده"""
        return list(cls._converters.keys())
