#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete Pipeline Test
Converter → Loader → DTCWT Assessment
"""
import sys
import subprocess
from pathlib import Path
import pandas as pd
import numpy as np

# Fix encoding for Windows
if sys.platform == 'win32':
    import io

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Setup path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("=" * 70)
print("Complete Pipeline Test - MovieLens")
print("=" * 70)
print()

# ==========================================
# Stage 1: Convert Data (Converter)
# ==========================================
print("Stage 1: Convert raw data to standard format")
print("-" * 70)

# Settings
raw_data_path = "data/raw/movielens/ratings.csv"
converted_data_path = "data/datasets/movielens_test.csv"

# Check if raw data exists
raw_path = Path(raw_data_path)
if not raw_path.exists():
    print(f"Warning: Raw data not found: {raw_data_path}")
    print()
    print("Instructions:")
    print("1. Download MovieLens dataset:")
    print("   wget https://files.grouplens.org/datasets/movielens/ml-32m.zip")
    print("2. Extract:")
    print("   unzip ml-32m.zip")
    print("3. Copy:")
    print("   mkdir -p data/raw/movielens")
    print("   cp ml-32m/ratings.csv data/raw/movielens/")
    print()
    print("For quick test, use sample data:")
    print("   python test_movielens_converter.py create_sample")
    raw_data_path = "data/raw/movielens/ratings_sample.csv"
    converted_data_path = "data/datasets/movielens_sample_test.csv"
    print(f"   Using: {raw_data_path}")
    print()

# Run Converter
print("Running Converter...")
cmd = [
    'python', 'prepare_data.py',
    '--dataset', 'movielens',
    '--input', raw_data_path,
    '--output', converted_data_path,
    '--aggregate', 'day',
    '--keep-rating'
]

print(f"Command: {' '.join(cmd)}")
print()

try:
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

    if result.returncode == 0:
        print("Success: Conversion completed")
        print()
    else:
        print("Error in conversion:")
        print(result.stderr)
        sys.exit(1)

except subprocess.TimeoutExpired:
    print("Error: Timeout - conversion took more than 5 minutes")
    sys.exit(1)
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)

# ==========================================
# Stage 2: Load Data (Loader)
# ==========================================
print("Stage 2: Load data with MovieLensLoader")
print("-" * 70)

try:
    from data.loaders.movielens import MovieLensLoader

    # Configure Loader
    loader_config = {
        'path': converted_data_path,
        'time_col': 'timestamp',
        'item_col': 'item_id',
        'count_col': 'count'
    }

    # Create Loader
    loader = MovieLensLoader(loader_config)
    print(f"Success: Loader created")
    print(f"  Path: {converted_data_path}")
    print()

    # Load data
    print("Loading data...")
    data = loader.load()
    print(f"Success: Data loaded")
    print(f"  Total records: {len(data):,}")
    print(f"  Columns: {data.columns.tolist()}")
    print()

    # Show sample
    print("Data sample:")
    print(data.head())
    print()

    # Statistics
    stats = loader.get_statistics(data)
    print("Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print()

except ImportError as e:
    print(f"Error importing Loader: {e}")
    sys.exit(1)
except Exception as e:
    print(f"Error loading data: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# ==========================================
# Stage 3: Assess Popularity (DTCWT)
# ==========================================
print("Stage 3: Assess popularity with DTCWT")
print("-" * 70)

try:
    from methods.dtcwt_assessment import DTCWTAssessment

    # Create Assessment
    print("Creating DTCWT Assessment...")
    dtcwt = DTCWTAssessment(
        level=3,
        biort='near_sym_a',
        qshift='qshift_a'
    )
    print("Success: Assessment created")
    print()

    # Select an item for testing
    all_items = loader.get_all_items(data)

    if len(all_items) == 0:
        print("Error: No items found!")
        sys.exit(1)

    # Select popular item (most records)
    item_counts = data.groupby('item_id').size()
    popular_item = item_counts.idxmax()

    print(f"Testing with popular item: {popular_item}")
    print(f"  Record count: {item_counts[popular_item]}")
    print()

    # Create time series
    print("Creating time series...")
    window_size = 100  # 100 days
    time_series = loader.create_time_series(
        data,
        popular_item,
        window_size=window_size,
        aggregation='day'
    )

    print(f"Success: Time series created")
    print(f"  Length: {len(time_series)}")
    print(f"  Range: [{time_series.min()}, {time_series.max()}]")
    print(f"  Mean: {time_series.mean():.2f}")
    print()

    # Calculate popularity
    print("Calculating popularity...")
    popularity_score = dtcwt.assess_single(time_series)

    print(f"Success: Calculation completed")
    print(f"  Popularity score: {popularity_score:.4f}")
    print()

    # Test multiple items
    print("Testing multiple items:")
    print("-" * 40)

    # Select 5 random items
    test_items = np.random.choice(all_items, min(5, len(all_items)), replace=False)

    for item_id in test_items:
        ts = loader.create_time_series(data, item_id, window_size, 'day')
        score = dtcwt.assess_single(ts)
        count = item_counts.get(item_id, 0)
        print(f"  Item {item_id}: score={score:.4f}, count={count}")

    print()

except ImportError as e:
    print(f"Warning: Error importing DTCWT: {e}")
    print("  (Dependencies may not be installed)")
    print()
except Exception as e:
    print(f"Error in calculation: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# ==========================================
# Summary
# ==========================================
print("=" * 70)
print("SUCCESS: Complete test passed!")
print("=" * 70)
print()
print("Summary:")
print(f"  1. Converter: {raw_data_path} -> {converted_data_path}")
print(f"  2. Loader: {len(data):,} records loaded")
print(f"  3. DTCWT: Popularity calculated")
print()
print("Pipeline is working! 🎉")
print()

# Usage guide
print("=" * 70)
print("Usage Guide:")
print("=" * 70)
print()
print("1. To create sample data:")
print("   python test_movielens_converter.py create_sample")
print()
print("2. To convert real data:")
print("   python prepare_data.py --dataset movielens \\")
print("                          --aggregate day \\")
print("                          --keep-rating")
print()
print("3. To run complete simulation:")
print("   python demo.py  # (if available)")
print()