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
    Base class for converting raw datasets to the standard format
    
    Standard output format:
        - timestamp: access time (datetime)
        - item_id: item identifier (str/int)
        - count: number of accesses (int) - optional
        - extra dataset-specific columns
    
    Generic parameters (all datasets):
        - output_format: 'csv', 'parquet', 'feather'
        - verbose: print messages
        - validate_output: validate the output
    
    Dataset-Specific Parameters:
        Each converter must override get_specific_params()
    """
    
    # ==========================================
    # Metadata (must be set in the subclass)
    # ==========================================
    DATASET_NAME: Optional[str] = None
    SUPPORTED_FILE_TYPES: List[str] = []
    DESCRIPTION: str = ""
    
    # ==========================================
    # Generic parameters (shared by all)
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
        Initialisation
        
        Args:
            output_format: output format
            verbose: print messages
            validate_output: validation
            **kwargs: converter-specific parameters
        """
        self.output_format = output_format
        self.verbose = verbose
        self.validate_output = validate_output
        
        # Store the extra parameters
        self.extra_params = kwargs
        
        # Logging setup
        self.logger = logging.getLogger(self.__class__.__name__)
        if verbose:
            self.logger.setLevel(logging.INFO)
    
    # ==========================================
    # Metadata methods (for argument generation)
    # ==========================================
    
    @classmethod
    def get_specific_params(cls) -> Dict[str, Dict[str, Any]]:
        """
        Specific parameters of this converter
        
        Must be overridden in the subclass
        
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
        """Combine the generic and specific parameters"""
        params = cls.GENERIC_PARAMS.copy()
        params.update(cls.get_specific_params())
        return params
    
    @classmethod
    def add_arguments(cls, parser, prefix: bool = True):
        """
        Add the arguments to argparse
        
        Args:
            parser: argparse.ArgumentParser
            prefix: use a prefix for the dataset-specific arguments
        """
        import argparse
        
        # Only the specific parameters are added here
        # The generic parameters are added in prepare_data.py
        specific_params = cls.get_specific_params()
        
        if not specific_params:
            return  # no specific parameters: skip
        
        # Create the argument group
        if cls.DATASET_NAME:
            group_name = f'{cls.DATASET_NAME.upper()} Options'
            group = parser.add_argument_group(group_name, cls.DESCRIPTION)
        else:
            group = parser
        
        # Add only the dataset-specific arguments
        for param_name, param_spec in specific_params.items():
            # Dataset-specific: with a prefix
            if prefix and cls.DATASET_NAME:
                arg_flag = f'--{cls.DATASET_NAME}-{param_name.replace("_", "-")}'
            else:
                arg_flag = f'--{param_name.replace("_", "-")}'
            
            # Build the kwargs of add_argument
            arg_kwargs = {
                'help': param_spec.get('help', '')
            }
            
            param_type = param_spec.get('type')
            
            if param_type == bool:
                # Boolean: use store_true
                arg_kwargs['action'] = 'store_true'
                if param_spec.get('default', False):
                    # if default=True, use store_false
                    arg_kwargs['action'] = 'store_false'
                    arg_flag = f'--no-{arg_flag[2:]}'
            else:
                # Non-boolean: type and default
                arg_kwargs['type'] = param_type
                arg_kwargs['default'] = param_spec.get('default')
                
                if 'choices' in param_spec:
                    arg_kwargs['choices'] = param_spec['choices']
            
            # Add to the parser
            group.add_argument(arg_flag, **arg_kwargs)
    
    @classmethod
    def extract_params_from_args(cls, args, prefix: bool = True) -> Dict[str, Any]:
        """
        Extract the parameters of this converter from the argparse arguments
        
        Args:
            args: argparse.Namespace
            prefix: whether a prefix was used
            
        Returns:
            Dictionary of parameters
        """
        params = {}
        
        # Extract the generic parameters
        for param_name in cls.GENERIC_PARAMS.keys():
            if hasattr(args, param_name):
                value = getattr(args, param_name)
                if value is not None:
                    params[param_name] = value
        
        # Extract the specific parameters
        specific_params = cls.get_specific_params()
        for param_name in specific_params.keys():
            # Attribute name
            if prefix and cls.DATASET_NAME:
                attr_name = f'{cls.DATASET_NAME}_{param_name}'
            else:
                attr_name = param_name
            
            # Read the value
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
        Main conversion: raw -> standard
        
        Args:
            input_path: path of the raw file(s)
            output_path: output path
            **kwargs: extra parameters
            
        Returns:
            converted DataFrame
        """
        self.log(f"Start conversion: {self.__class__.__name__}")
        
        # ==========================================
        # Step 1: prepare the paths
        # ==========================================
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # ==========================================
        # Step 2: process the file(s)
        # ==========================================
        if isinstance(input_path, (list, tuple)):
            # several files
            self.log(f"Processing {len(input_path)} file(s)...")
            dfs = []
            for file_path in tqdm(input_path, disable=not self.verbose, desc="Converting"):
                df = self._convert_single_file(Path(file_path), **kwargs)
                dfs.append(df)
            
            # combine
            self.log("Combining files...")
            df_combined = pd.concat(dfs, ignore_index=True)
            
            # remove duplicates
            df_combined = self.aggregate_duplicates(
                df_combined,
                group_by=['timestamp', 'item_id']
            )
        else:
            # single file
            df_combined = self._convert_single_file(Path(input_path), **kwargs)
        
        # ==========================================
        # Step 3: validation
        # ==========================================
        if self.validate_output:
            self.log("Validating output...")
            self._validate_output(df_combined)
        
        # ==========================================
        # Step 4: save
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
        Convert one file (must be implemented in the subclass)
        
        Args:
            file_path: file path
            **kwargs: extra parameters
            
        Returns:
            DataFrame with the standard columns:
            - timestamp (datetime64[ns])
            - item_id (str or int)
            - count (int) - optional
        """
        pass
    
    # ==========================================
    # Helper Methods
    # ==========================================
    
    def log(self, message: str):
        """Print a message"""
        if self.verbose:
            print(message)
    
    def aggregate_duplicates(self,
                           df: pd.DataFrame,
                           group_by: List[str],
                           agg_func: str = 'sum') -> pd.DataFrame:
        """
        Merge duplicate records
        
        Args:
            df: input DataFrame
            group_by: grouping columns
            agg_func: aggregation function ('sum', 'count', 'mean')
            
        Returns:
            DataFrame without duplicates
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
        Convert to a timestamp
        
        Args:
            value: value (string, int, datetime)
            format: format (for strings)
            unit: unit (for int: 's', 'ms', 'us')
            
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
        Validate the output DataFrame
        
        Checks:
        - required columns are present
        - data types
        - null values
        """
        # Required columns
        required_cols = ['timestamp', 'item_id']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Type of timestamp
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            raise TypeError("Column 'timestamp' must be datetime64")
        
        # Check for nulls
        null_counts = df[required_cols].isnull().sum()
        if null_counts.any():
            self.logger.warning(f"Null values found:\n{null_counts[null_counts > 0]}")
        
        self.log("OK: Validation passed")
    
    def _save_output(self, df: pd.DataFrame, output_path: Path):
        """Save the DataFrame"""
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
    Factory that creates the right converter
    """
    
    _converters: Dict[str, type] = {}
    
    @classmethod
    def register(cls, dataset_name: str, converter_class: type):
        """
        Register a new converter
        
        Args:
            dataset_name: dataset name
            converter_class: converter class
        """
        cls._converters[dataset_name.lower()] = converter_class
    
    @classmethod
    def create(cls, dataset_name: str, **kwargs) -> BaseConverter:
        """
        Create the right converter
        
        Args:
            dataset_name: dataset name
            **kwargs: converter parameters
            
        Returns:
            a Converter instance
        """
        dataset_name = dataset_name.lower()
        
        if dataset_name not in cls._converters:
            raise ValueError(
                f"No converter found for '{dataset_name}'. "
                f"Available datasets: {list(cls._converters.keys())}"
            )
        
        converter_class = cls._converters[dataset_name]
        return converter_class(**kwargs)
    
    @classmethod
    def list_converters(cls) -> List[str]:
        """List the supported datasets"""
        return list(cls._converters.keys())
    
    @classmethod
    def get_converter_class(cls, dataset_name: str) -> type:
        """Return the converter class"""
        dataset_name = dataset_name.lower()
        if dataset_name not in cls._converters:
            raise ValueError(f"Converter '{dataset_name}' not found")
        return cls._converters[dataset_name]
    
    @classmethod
    def get_all_params(cls, dataset_name: str) -> Dict[str, Dict[str, Any]]:
        """Return all parameters of a converter"""
        converter_class = cls.get_converter_class(dataset_name)
        return converter_class.get_all_params()
