#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data preparation script V2
Converts the raw datasets to the standard format

Features:
- Dynamic argument generation (each converter builds its own arguments)
- Config file support (YAML)
- Generic + Dataset-specific parameters
- Backward compatible

Usage:
    # MovieLens - basic
    python prepare_data.py --dataset movielens \\
        --input data/raw/movielens/ratings.csv \\
        --output data/datasets/movielens.csv
    
    # MovieLens - with options
    python prepare_data.py --dataset movielens \\
        --movielens-aggregate-by day \\
        --movielens-keep-rating \\
        --movielens-min-rating 4.0
    
    # Yellow Taxi - with options
    python prepare_data.py --dataset yellow_taxi \\
        --input "data/raw/yellow_taxi/yellow_*.parquet" \\
        --output data/datasets/yellow_taxi_15min.csv \\
        --yellow-taxi-granularity 15min \\
        --yellow-taxi-min-trips-per-location 200 \\
        --yellow-taxi-extract-features
    
    # From a config file
    python prepare_data.py --config configs/yellow_taxi_hourly.yaml
    
    # List the datasets
    python prepare_data.py --list
"""
import argparse
import sys
from pathlib import Path
from glob import glob
import yaml
from typing import Dict, Any, Optional

# Fix encoding for Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add the project to the path
sys.path.insert(0, str(Path(__file__).parent))

# Import the converters (this registers them automatically)
from data.converters.base_converter import ConverterFactory
from data.converters import movielens_converter
from data.converters import yellow_taxi_converter
from data.converters import youtube_converter

# Dataset configuration (for --all and the default paths)
DATASET_CONFIGS = {
    'movielens': {
        'description': 'MovieLens ratings dataset',
        'input': 'data/raw/movielens/ratings.csv',
        'output': 'data/datasets/movielens.csv'
    },
    'yellow_taxi': {
        'description': 'NYC Yellow Taxi trip data',
        'input': 'data/raw/yellow_taxi/yellow_*.parquet',
        'output': 'data/datasets/yellow_taxi_15min.csv'
    },
    'youtube': {  
        'description': 'YouTube hourly video views converter',
        'input': 'data/raw/youtube/count_observation_upload.csv',
        'output': 'data/datasets/youtube_hourly.csv'
    }
    # Add further datasets here
}


def load_config_file(config_path: str) -> Dict[str, Any]:
    """
    Read a config file (YAML)
    
    Args:
        config_path: path of the config file
        
    Returns:
        Dictionary of settings
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def parse_args():
    """Parse the command-line arguments"""
    
    # ==========================================
    # Main parser
    # ==========================================
    parser = argparse.ArgumentParser(
        description='Convert the raw datasets to the standard format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # MovieLens with aggregation
  python prepare_data.py --dataset movielens --movielens-aggregate-by day --movielens-keep-rating
  
  # NYC Yellow Taxi with features
  python prepare_data.py --dataset yellow_taxi --yellow-taxi-granularity hourly --yellow-taxi-extract-features
  
  # From a config file
  python prepare_data.py --config configs/movielens_daily.yaml
  
  # List the datasets
  python prepare_data.py --list
        """
    )
    
    # ==========================================
    # Core Arguments
    # ==========================================
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        choices=ConverterFactory.list_converters(),
        help='dataset name'
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        help='input path (may contain a wildcard)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='output CSV path'
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='path of a config file (YAML)'
    )
    
    parser.add_argument(
        '--all', '-a',
        action='store_true',
        help='convert all datasets (from DATASET_CONFIGS)'
    )
    
    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='list the available datasets'
    )
    
    # ==========================================
    # Generic parameters (shared by all datasets)
    # ==========================================
    generic_group = parser.add_argument_group('Generic Options')
    
    generic_group.add_argument(
        '--output-format', '-f',
        type=str,
        default='csv',
        choices=['csv', 'parquet', 'feather'],
        help='output format (default: csv)'
    )
    
    generic_group.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='quiet mode (no messages)'
    )
    
    generic_group.add_argument(
        '--no-validate',
        action='store_true',
        help='disable validation'
    )
    
    # ==========================================
    # Dataset-Specific Arguments
    # All converters are added automatically
    # ==========================================
    for dataset_name in ConverterFactory.list_converters():
        converter_class = ConverterFactory.get_converter_class(dataset_name)
        converter_class.add_arguments(parser, prefix=True)
    
    return parser.parse_args()


def list_datasets():
    """List the datasets"""
    print("\n" + "=" * 70)
    print("Available datasets for conversion:")
    print("=" * 70)
    
    for dataset_name in ConverterFactory.list_converters():
        converter_class = ConverterFactory.get_converter_class(dataset_name)
        
        # Information from the config (if present)
        config = DATASET_CONFIGS.get(dataset_name, {})
        
        print(f"\nDataset: {dataset_name.upper()}")
        print(f"   {converter_class.DESCRIPTION}")
        
        if config:
            print(f"   Input (default):  {config.get('input', 'N/A')}")
            print(f"   Output (default): {config.get('output', 'N/A')}")
        
        # Show the parameters
        params = converter_class.get_specific_params()
        if params:
            print(f"   Parameters:")
            for param_name, param_spec in params.items():
                default = param_spec.get('default', 'None')
                print(f"     --{dataset_name}-{param_name.replace('_', '-')}: {param_spec.get('help', '')} (default: {default})")
    
    print("\n" + "=" * 70)
    print(f"Total: {len(ConverterFactory.list_converters())} datasets")
    print("=" * 70 + "\n")


def convert_single_dataset(dataset_name: str,
                           input_path: str,
                           output_path: str,
                           converter_params: Dict[str, Any],
                           verbose: bool = True) -> bool:
    """
    Convert one dataset
    
    Args:
        dataset_name: dataset name
        input_path: input path
        output_path: output path
        converter_params: converter parameters
        verbose: print messages
        
    Returns:
        True on success, False on failure
    """
    print("\n" + "=" * 70)
    print(f"Converting dataset: {dataset_name.upper()}")
    print("=" * 70 + "\n")
    
    # Show the parameters
    if converter_params and verbose:
        print("Converter parameters:")
        for key, value in converter_params.items():
            if value is not None and value != False and value != 'none':
                print(f"  {key}: {value}")
        print()
    
    # Check for a wildcard in the input path
    if '*' in input_path or '?' in input_path:
        input_files = sorted(glob(input_path))
        if not input_files:
            print(f"ERROR: No files found matching pattern '{input_path}'.")
            return False
        print(f"OK: Found {len(input_files)} files.")
        input_path = input_files
    
    try:
        # Create the converter
        converter = ConverterFactory.create(dataset_name, **converter_params)
        
        # Convert
        df = converter.convert(input_path, output_path)
        
        print(f"\nOK: Conversion completed: {len(df):,} records.")
        print(f"  Output: {output_path}\n")
        
        return True
    
    except Exception as e:
        print(f"\nERROR: Conversion failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def convert_all_datasets(verbose: bool = True) -> Dict[str, bool]:
    """
    Convert all datasets
    
    Args:
        verbose: print messages
        
    Returns:
        Dictionary of results {dataset_name: success}
    """
    print("\n" + "=" * 70)
    print("Starting conversion for all datasets")
    print("=" * 70 + "\n")
    
    results = {}
    
    for dataset_name, config in DATASET_CONFIGS.items():
        # Use the default values of the config
        success = convert_single_dataset(
            dataset_name=dataset_name,
            input_path=config['input'],
            output_path=config['output'],
            converter_params={
                'verbose': verbose,
                'output_format': 'csv',
                'validate_output': True
            },
            verbose=verbose
        )
        results[dataset_name] = success
    
    # Summary of the results
    print("\n" + "=" * 70)
    print("Summary:")
    print("=" * 70)
    
    for dataset_name, success in results.items():
        status = "OK" if success else "FAILED"
        print(f"  {dataset_name:<20} {status}")
    
    total = len(results)
    successful = sum(results.values())
    
    print("=" * 70)
    print(f"Total: {successful}/{total} succeeded")
    print("=" * 70 + "\n")
    
    return results


def main():
    """Main function"""
    args = parse_args()
    
    # ==========================================
    # Mode: list
    # ==========================================
    if args.list:
        list_datasets()
        return 0
    
    # ==========================================
    # Mode: from a config file
    # ==========================================
    if args.config:
        print(f"Loading config from: {args.config}")
        config = load_config_file(args.config)
        
        dataset_name = config.get('dataset')
        input_path = config.get('input')
        output_path = config.get('output')
        converter_params = config.get('converter_params', {})
        
        # Add the generic parameters
        converter_params['verbose'] = not args.quiet
        converter_params['output_format'] = args.output_format
        converter_params['validate_output'] = not args.no_validate
        
        success = convert_single_dataset(
            dataset_name=dataset_name,
            input_path=input_path,
            output_path=output_path,
            converter_params=converter_params,
            verbose=not args.quiet
        )
        
        return 0 if success else 1
    
    # ==========================================
    # Mode: convert all
    # ==========================================
    if args.all:
        results = convert_all_datasets(verbose=not args.quiet)
        all_success = all(results.values())
        return 0 if all_success else 1
    
    # ==========================================
    # Mode: convert one dataset
    # ==========================================
    if not args.dataset:
        print("ERROR: You must specify --dataset, --config, or --all.")
        print("   For help: python prepare_data.py --help")
        return 1
    
    # Set input/output
    if not args.input or not args.output:
        # Use the default config
        if args.dataset in DATASET_CONFIGS:
            config = DATASET_CONFIGS[args.dataset]
            input_path = args.input or config['input']
            output_path = args.output or config['output']
        else:
            print(f"ERROR: Dataset '{args.dataset}' not found in DATASET_CONFIGS.")
            print("   Please specify --input and --output.")
            return 1
    else:
        input_path = args.input
        output_path = args.output
    
    # Extract the converter parameters
    converter_class = ConverterFactory.get_converter_class(args.dataset)
    converter_params = converter_class.extract_params_from_args(args, prefix=True)
    
    # Add the generic parameters
    converter_params['verbose'] = not args.quiet
    converter_params['output_format'] = args.output_format
    converter_params['validate_output'] = not args.no_validate
    
    # Convert
    success = convert_single_dataset(
        dataset_name=args.dataset,
        input_path=input_path,
        output_path=output_path,
        converter_params=converter_params,
        verbose=not args.quiet
    )
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
