"""
Enhanced Base Converter with Metadata Support

Features:
- Generic parameters (applicable to all datasets)
- Dataset-specific parameters (with prefix)
- Metadata-driven argument generation
- Config file support
"""
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, List, Dict, Optional, Any, Tuple
from tqdm import tqdm
import logging


class BaseConverter(ABC):
    """
    کلاس پایه برای تبدیل دیتاست‌های خام به فرمت استاندارد
    
    فرمت استاندارد خروجی:
        - timestamp: زمان دسترسی (datetime)
        - item_id: شناسه آیتم (str/int)
        - count: تعداد دسترسی (int) - اختیاری
        - ستون‌های اضافی دیتاست‌محور
    
    Generic Parameters (همه datasets):
        - output_format: 'csv', 'parquet', 'feather'
        - verbose: نمایش پیام‌ها
        - validate_output: اعتبارسنجی خروجی
    
    Dataset-Specific Parameters:
        هر converter باید get_specific_params() را override کند
    """
    
    # ==========================================
    # Metadata (باید در subclass set شوند)
    # ==========================================
    DATASET_NAME: Optional[str] = None
    SUPPORTED_FILE_TYPES: List[str] = []
    DESCRIPTION: str = ""
    
    # ==========================================
    # Generic Parameters (مشترک بین همه)
    # ==========================================
    GENERIC_PARAMS = {
        'output_format': {
            'type': str,
            'default': 'csv',
            'choices': ['csv', 'parquet', 'feather'],
            'help': 'Output file format'
        },
        'verbose': {
            'type': bool,
            'default': True,
            'help': 'Display progress messages'
        },
        'validate_output': {
            'type': bool,
            'default': True,
            'help': 'Validate output data'
        }
    }
    
    def __init__(self, 
                 output_format: str = 'csv',
                 verbose: bool = True,
                 validate_output: bool = True,
                 **kwargs):
        """
        مقداردهی اولیه
        
        Args:
            output_format: فرمت خروجی
            verbose: نمایش پیام‌ها
            validate_output: اعتبارسنجی
            **kwargs: پارامترهای اختصاصی converter
        """
        self.output_format = output_format
        self.verbose = verbose
        self.validate_output = validate_output
        
        # ذخیره پارامترهای اضافی
        self.extra_params = kwargs
        
        # تنظیم logging
        self.logger = logging.getLogger(self.__class__.__name__)
        if verbose:
            self.logger.setLevel(logging.INFO)
    
    # ==========================================
    # Metadata Methods (برای argument generation)
    # ==========================================
    
    @classmethod
    def get_specific_params(cls) -> Dict[str, Dict[str, Any]]:
        """
        پارامترهای اختصاصی این converter
        
        باید در subclass override شود
        
        Returns:
            {
                'param_name': {
                    'type': str/int/float/bool,
                    'default': value,
                    'help': 'description',
                    'choices': [...] (optional)
                }
            }
        """
        return {}
    
    @classmethod
    def get_all_params(cls) -> Dict[str, Dict[str, Any]]:
        """ترکیب generic + specific parameters"""
        params = cls.GENERIC_PARAMS.copy()
        params.update(cls.get_specific_params())
        return params
    
    @classmethod
    def add_arguments(cls, parser, prefix: bool = True):
        """
        اضافه کردن arguments به argparse
        
        Args:
            parser: argparse.ArgumentParser
            prefix: استفاده از prefix برای dataset-specific args
        """
        import argparse
        
        # فقط specific parameters را اضافه می‌کنیم
        # Generic parameters در prepare_data.py اضافه می‌شوند
        specific_params = cls.get_specific_params()
        
        if not specific_params:
            return  # اگر parameter خاصی نداریم، skip
        
        # ایجاد argument group
        if cls.DATASET_NAME:
            group_name = f'{cls.DATASET_NAME.upper()} Options'
            group = parser.add_argument_group(group_name, cls.DESCRIPTION)
        else:
            group = parser
        
        # اضافه کردن فقط dataset-specific arguments
        for param_name, param_spec in specific_params.items():
            # Dataset-specific: با prefix
            if prefix and cls.DATASET_NAME:
                arg_flag = f'--{cls.DATASET_NAME}-{param_name.replace("_", "-")}'
            else:
                arg_flag = f'--{param_name.replace("_", "-")}'
            
            # ساخت kwargs برای add_argument
            arg_kwargs = {
                'help': param_spec.get('help', '')
            }
            
            param_type = param_spec.get('type')
            
            if param_type == bool:
                # Boolean: استفاده از store_true
                arg_kwargs['action'] = 'store_true'
                if param_spec.get('default', False):
                    # اگر default=True، از store_false استفاده کن
                    arg_kwargs['action'] = 'store_false'
                    arg_flag = f'--no-{arg_flag[2:]}'
            else:
                # Non-boolean: type و default
                arg_kwargs['type'] = param_type
                arg_kwargs['default'] = param_spec.get('default')
                
                if 'choices' in param_spec:
                    arg_kwargs['choices'] = param_spec['choices']
            
            # اضافه کردن به parser
            group.add_argument(arg_flag, **arg_kwargs)
    
    @classmethod
    def extract_params_from_args(cls, args, prefix: bool = True) -> Dict[str, Any]:
        """
        استخراج پارامترهای این converter از argparse args
        
        Args:
            args: argparse.Namespace
            prefix: آیا از prefix استفاده شده
            
        Returns:
            Dictionary پارامترها
        """
        params = {}
        
        # استخراج generic parameters
        for param_name in cls.GENERIC_PARAMS.keys():
            if hasattr(args, param_name):
                value = getattr(args, param_name)
                if value is not None:
                    params[param_name] = value
        
        # استخراج specific parameters
        specific_params = cls.get_specific_params()
        for param_name in specific_params.keys():
            # تعیین attribute name
            if prefix and cls.DATASET_NAME:
                attr_name = f'{cls.DATASET_NAME}_{param_name}'
            else:
                attr_name = param_name
            
            # دریافت مقدار
            if hasattr(args, attr_name):
                value = getattr(args, attr_name)
                if value is not None:
                    params[param_name] = value
        
        return params
    
    # ==========================================
    # Core Conversion Methods
    # ==========================================
    
    def convert(self, 
                input_path: Union[str, Path, List[Union[str, Path]]],
                output_path: Union[str, Path],
                **kwargs) -> pd.DataFrame:
        """
        تبدیل اصلی: خام → استاندارد
        
        Args:
            input_path: مسیر فایل(ها) خام
            output_path: مسیر ذخیره
            **kwargs: پارامترهای اضافی
            
        Returns:
            DataFrame تبدیل شده
        """
        self.log(f"Start conversion: {self.__class__.__name__}")
        
        # ==========================================
        # مرحله 1: آماده‌سازی مسیرها
        # ==========================================
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # ==========================================
        # مرحله 2: پردازش فایل(ها)
        # ==========================================
        if isinstance(input_path, (list, tuple)):
            # چند فایل
            self.log(f"Processing {len(input_path)} file(s)...")
            dfs = []
            for file_path in tqdm(input_path, disable=not self.verbose, desc="Converting"):
                df = self._convert_single_file(Path(file_path), **kwargs)
                dfs.append(df)
            
            # ترکیب
            self.log("Combining files...")
            df_combined = pd.concat(dfs, ignore_index=True)
            
            # حذف تکراری
            df_combined = self.aggregate_duplicates(
                df_combined,
                group_by=['timestamp', 'item_id']
            )
        else:
            # تک فایل
            df_combined = self._convert_single_file(Path(input_path), **kwargs)
        
        # ==========================================
        # مرحله 3: اعتبارسنجی
        # ==========================================
        if self.validate_output:
            self.log("Validating output...")
            self._validate_output(df_combined)
        
        # ==========================================
        # مرحله 4: ذخیره
        # ==========================================
        self.log(f"Saving to: {output_path}")
        self._save_output(df_combined, output_path)
        
        self.log(f"OK: Conversion completed: {len(df_combined):,} records")
        
        return df_combined
    
    @abstractmethod
    def _convert_single_file(self, 
                            file_path: Path,
                            **kwargs) -> pd.DataFrame:
        """
        تبدیل یک فایل (باید در subclass پیاده‌سازی شود)
        
        Args:
            file_path: مسیر فایل
            **kwargs: پارامترهای اضافی
            
        Returns:
            DataFrame با ستون‌های استاندارد:
            - timestamp (datetime64[ns])
            - item_id (str یا int)
            - count (int) - اختیاری
        """
        pass
    
    # ==========================================
    # Helper Methods
    # ==========================================
    
    def log(self, message: str):
        """چاپ پیام"""
        if self.verbose:
            print(message)
    
    def aggregate_duplicates(self,
                           df: pd.DataFrame,
                           group_by: List[str],
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
    
    def parse_timestamp(self,
                       value: Any,
                       format: Optional[str] = None,
                       unit: Optional[str] = None) -> pd.Timestamp:
        """
        تبدیل به timestamp
        
        Args:
            value: مقدار (string, int, datetime)
            format: فرمت (برای string)
            unit: واحد (برای int: 's', 'ms', 'us')
            
        Returns:
            pd.Timestamp
        """
        if unit:
            return pd.to_datetime(value, unit=unit)
        elif format:
            return pd.to_datetime(value, format=format)
        else:
            return pd.to_datetime(value)
    
    def _validate_output(self, df: pd.DataFrame):
        """
        اعتبارسنجی DataFrame خروجی
        
        بررسی:
        - وجود ستون‌های ضروری
        - نوع داده‌ها
        - مقادیر null
        """
        # ستون‌های ضروری
        required_cols = ['timestamp', 'item_id']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # نوع timestamp
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            raise TypeError("Column 'timestamp' must be datetime64")
        
        # بررسی null
        null_counts = df[required_cols].isnull().sum()
        if null_counts.any():
            self.logger.warning(f"Null values found:\n{null_counts[null_counts > 0]}")
        
        self.log("OK: Validation passed")
    
    def _save_output(self, df: pd.DataFrame, output_path: Path):
        """ذخیره DataFrame"""
        if self.output_format == 'csv':
            df.to_csv(output_path, index=False)
        elif self.output_format == 'parquet':
            df.to_parquet(output_path, index=False)
        elif self.output_format == 'feather':
            df.to_feather(output_path)
        else:
            raise ValueError(f"Unsupported format: {self.output_format}")


class ConverterFactory:
    """
    Factory برای ایجاد converter مناسب
    """
    
    _converters: Dict[str, type] = {}
    
    @classmethod
    def register(cls, dataset_name: str, converter_class: type):
        """
        ثبت converter جدید
        
        Args:
            dataset_name: نام دیتاست
            converter_class: کلاس Converter
        """
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
    
    @classmethod
    def get_converter_class(cls, dataset_name: str) -> type:
        """دریافت کلاس converter"""
        dataset_name = dataset_name.lower()
        if dataset_name not in cls._converters:
            raise ValueError(f"Converter '{dataset_name}' not found")
        return cls._converters[dataset_name]
    
    @classmethod
    def get_all_params(cls, dataset_name: str) -> Dict[str, Dict[str, Any]]:
        """دریافت تمام پارامترهای یک converter"""
        converter_class = cls.get_converter_class(dataset_name)
        return converter_class.get_all_params()
