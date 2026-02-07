# Content Popularity Prediction using Multi-scale Frequency Decomposition

A comprehensive framework for measuring and predicting content popularity in distributed systems using wavelet-based signal processing and graph analysis.

## Overview

This project implements advanced popularity assessment methods combining:
- **Wavelet Transforms**: DWT and Dual-Tree Complex Wavelet Transform (DTCWT)  
- **Graph Signal Processing**: Content relationship graphs with spectral analysis
- **Statistical Methods**: ARMA, EWMA, and traditional caching policies (LRU, LFU)
- **Hybrid Approaches**: Combining multiple methods for enhanced prediction

## Features

- **Multiple Datasets**: MovieLens, Youku, YouTube
- **Flexible Time Granularity**: Daily, hourly, minute-level, or custom intervals
- **Two Evaluation Modes**:
  - Standard: Full in-memory evaluation
  - Incremental: Memory-efficient, crash-safe evaluation
- **Comprehensive Metrics**: Spearman correlation, Kendall tau, stratified analysis
- **Parallel Processing**: Multi-core support for faster evaluation
- **Rich Visualization**: Performance comparison charts and statistics

## Installation

### Prerequisites

```bash
Python 3.8+
pip install -r requirements.txt
```

### Dependencies

```bash
pip install numpy pandas scipy scikit-learn
pip install pywt dtcwt  # Wavelet transforms
pip install tqdm  # Progress bars
pip install matplotlib seaborn  # Visualization
```

## Quick Start

### 1. Prepare Data

Convert MovieLens dataset:

```bash
python data/convert_movielens.py \
  --input data/raw/ml-25m \
  --output data/processed/movielens_daily.csv \
  --granularity daily
```

### 2. Run Evaluation

Standard mode (all methods):

```bash
python experiments/run_popularity_assessment.py movielens \
  --num-items 500 \
  --start-date 2023-06-01 \
  --end-date 2023-08-31
```

Incremental mode (memory efficient):

```bash
python experiments/run_popularity_assessment.py movielens \
  --num-items 500 \
  --start-date 2023-06-01 \
  --end-date 2023-08-31 \
  --incremental
```

### 3. Analyze Results

```bash
python experiments/analyze_results.py results/movielens/w30_h7_n500_top_20260207_140000
```

## Data Preparation

### MovieLens Dataset

Convert MovieLens 25M dataset to temporal format:

```bash
python data/convert_movielens.py \
  --input data/raw/ml-25m \
  --output data/processed/movielens_daily.csv \
  --granularity daily
```

**Granularity Options:**
- `daily`: One record per day (default)
- `hourly`: One record per hour
- `weekly`: One record per week

### Youku Dataset

Convert Youku video dataset:

```bash
python data/convert_youku.py \
  --input data/raw/youku \
  --output data/processed/youku_5min.csv \
  --granularity 5min
```

**Granularity Options:**
- `5min`: 5-minute intervals
- `15min`: 15-minute intervals
- `hourly`: Hourly aggregation

### Custom Dataset

Your dataset should have these columns:
- `item_id`: Content identifier (integer or string)
- `timestamp`: Date/time (datetime64[ns])
- `count`: Access count (integer)

Example format:
```csv
item_id,timestamp,count
1,2023-06-01 00:00:00,45
1,2023-06-02 00:00:00,52
2,2023-06-01 00:00:00,23
```

## Evaluation

### Standard Mode

Full in-memory evaluation with all features:

```bash
python experiments/run_popularity_assessment.py DATASET [OPTIONS]
```

**Arguments:**

- `DATASET`: Dataset name (`movielens`, `youtube07`, `youku`)
- `--num-items N`: Number of items to evaluate (default: all)
- `--start-date YYYY-MM-DD`: Start date for evaluation
- `--end-date YYYY-MM-DD`: End date for evaluation
- `--window-size N`: Training window size in time slots (default: 30)
- `--horizon N`: Prediction horizon (legacy, not used in assessment mode)
- `--methods M1 M2 ...`: Specific methods to evaluate (default: all)
- `--item-selection {top,random,stratified}`: Item selection strategy (default: top)
- `--cores N`: Number of CPU cores to use (default: all)
- `--format {csv,parquet}`: Output format (default: csv)
- `--data-path PATH`: Custom data file path
- `--quiet`: Suppress verbose output

**Available Methods:**
- `AF`: Access Frequency (baseline)
- `LRU`: Least Recently Used
- `LFU`: Least Frequently Used
- `EWMA`: Exponentially Weighted Moving Average
- `Statistical`: ARMA-based statistical model
- `DWT+AF`: Discrete Wavelet Transform + Access Frequency
- `DTCWT+AF`: Dual-Tree Complex Wavelet Transform + Access Frequency
- `Hybrid V3.0`: Wavelet-Statistical hybrid
- `Hybrid V3.1`: Advanced hybrid with graph features

**Examples:**

Evaluate top 1000 items for 3 months:
```bash
python experiments/run_popularity_assessment.py movielens \
  --num-items 1000 \
  --start-date 2023-06-01 \
  --end-date 2023-08-31
```

Evaluate specific methods only:
```bash
python experiments/run_popularity_assessment.py movielens \
  --num-items 500 \
  --methods AF LRU LFU DTCWT+AF \
  --cores 4
```

Random sampling with custom date range:
```bash
python experiments/run_popularity_assessment.py movielens \
  --num-items 200 \
  --item-selection random \
  --start-date 2023-07-01 \
  --end-date 2023-07-31
```

### Incremental Mode

Memory-efficient evaluation with continuous saving:

```bash
python experiments/run_popularity_assessment.py DATASET --incremental [OPTIONS]
```

**Benefits:**
- Low memory usage (<200 MB)
- Crash-safe: Results saved continuously to SQLite database
- Progress tracking: Resume from last saved state
- Suitable for large-scale evaluations

**Example:**

```bash
python experiments/run_popularity_assessment.py movielens \
  --num-items 5000 \
  --start-date 2023-01-01 \
  --end-date 2023-12-31 \
  --incremental \
  --cores 8
```

### Time Granularity

The framework automatically handles different time granularities:

**Daily (MovieLens):**
```bash
# Window size = 30 days, evaluates 92 time slots for a 92-day period
python experiments/run_popularity_assessment.py movielens \
  --window-size 30
```

**5-Minute (Youku):**
```bash
# Window size = 288 slots (24 hours), evaluates video-level popularity
python experiments/run_popularity_assessment.py youku \
  --window-size 288
```

**Weekly:**
```bash
# Convert data to weekly first, then window size = 8 weeks
python data/convert_movielens.py --granularity weekly
python experiments/run_popularity_assessment.py movielens \
  --window-size 8
```

## Analysis

### Analyzing Results

After evaluation, analyze results using:

```bash
python experiments/analyze_results.py RESULTS_PATH [OPTIONS]
```

**Arguments:**
- `RESULTS_PATH`: Path to results directory
- `--output DIR`: Output directory for analysis (default: same as results)
- `--format {text,html,both}`: Report format (default: both)
- `--plot`: Generate comparison plots
- `--export-csv`: Export summary to CSV

**Examples:**

Basic analysis:
```bash
python experiments/analyze_results.py results/movielens/w30_h7_n500_top_20260207_140000
```

Full analysis with plots:
```bash
python experiments/analyze_results.py results/movielens/w30_h7_n500_top_20260207_140000 \
  --plot \
  --format both \
  --export-csv
```

### Comparing Multiple Experiments

Compare results from different runs:

```bash
python experiments/compare_experiments.py \
  results/movielens/experiment1 \
  results/movielens/experiment2 \
  results/movielens/experiment3 \
  --output comparison_report.html
```

### Visualization

Generate visualization plots:

```bash
python experiments/visualize_results.py RESULTS_PATH [OPTIONS]
```

**Available Plots:**
- Performance comparison (Spearman correlation by method)
- Stratified analysis (performance per content popularity stratum)
- Temporal analysis (performance over time)
- Runtime comparison (execution time by method)

**Example:**

```bash
python experiments/visualize_results.py results/movielens/w30_h7_n500_top_20260207_140000 \
  --plots performance stratified temporal \
  --output figures/
```

## Project Structure

```
dtcwt_popularity/
├── data/
│   ├── loaders/              # Data loading utilities
│   │   ├── movielens_loader.py
│   │   ├── youku_loader.py
│   │   └── base_loader.py
│   ├── convert_movielens.py # MovieLens conversion script
│   ├── convert_youku.py     # Youku conversion script
│   └── raw/                 # Raw datasets (not included)
│
├── evaluation/
│   ├── temporal_evaluator.py      # Standard evaluation engine
│   ├── incremental_evaluator.py  # Incremental evaluation engine
│   ├── evaluation_config.py      # Configuration classes
│   ├── metrics.py                 # Evaluation metrics
│   ├── stratification.py          # Stratification system
│   ├── time_utils.py              # Time slot utilities
│   └── method_configs.py          # Method-specific configurations
│
├── methods/
│   ├── base_method.py           # Base assessment method
│   ├── dtcwt_assessment.py      # DTCWT-based method
│   ├── dwt_assessment.py        # DWT-based method
│   ├── statistical_assessment.py # Statistical methods
│   ├── hybrid_assessment.py     # Hybrid methods
│   └── baselines/
│       └── __init__.py          # Traditional baselines (LRU, LFU, etc.)
│
├── experiments/
│   ├── run_popularity_assessment.py  # Main evaluation script
│   ├── analyze_results.py            # Results analysis
│   ├── compare_experiments.py        # Multi-experiment comparison
│   └── visualize_results.py          # Visualization
│
├── results/                    # Evaluation results (auto-generated)
│
├── README.md                   # This file
├── راهنما.md                   # Persian guide
└── requirements.txt            # Python dependencies
```

## Output Structure

### Standard Mode

```
results/movielens/w30_h7_n500_top_20260207_140000/
├── config.json              # Experiment configuration
├── thresholds.json          # Stratification thresholds
├── runtime_stats.json       # Runtime statistics
├── metadata/
│   ├── config.json
│   ├── thresholds.json
│   └── run_metadata.json
└── results/
    ├── AF_results.csv
    ├── LRU_results.csv
    ├── DTCWT+AF_results.csv
    └── ...
```

### Incremental Mode

```
results/w30_h7_n5000_inc_20260207_140000/
├── config.json              # Experiment configuration
├── thresholds.json          # Stratification thresholds
├── runtime_stats.json       # Runtime statistics
├── metadata/
│   └── run_metadata.json
└── results.db               # SQLite database with all results
```

## Troubleshooting

### Memory Issues

Use incremental mode for large evaluations:
```bash
python experiments/run_popularity_assessment.py movielens \
  --num-items 10000 \
  --incremental
```

### Import Errors

Install missing dependencies:
```bash
pip install pywt dtcwt scipy
```

### Data Format Issues

Ensure your data has correct columns and types:
```python
import pandas as pd
df = pd.read_csv('your_data.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['count'] = df['count'].astype(int)
```

### Time Granularity Mismatch

Ensure dataset and configuration match:
- Daily data → `time_granularity='daily'`
- Hourly data → `time_granularity='hourly'`
- 5-min data → `time_granularity='custom'`, `slot_duration_minutes=5`

## Performance Tips

1. **Use Parallel Processing:**
   ```bash
   --cores 8  # Use 8 CPU cores
   ```

2. **Select Fewer Methods:**
   ```bash
   --methods AF DTCWT+AF  # Only essential methods
   ```

3. **Use Incremental Mode for Large Datasets:**
   ```bash
   --incremental  # <200 MB memory usage
   ```

4. **Optimize Window Sizes:**
   - Smaller windows = faster but less accurate
   - Larger windows = slower but more accurate

## Citation

If you use this framework in your research, please cite:

```bibtex
@phdthesis{popularity_wavelets_2025,
  title={Measuring and Predicting Data Popularity Rate Based on Multi-scale Frequency Decomposition in Distributed Systems},
  author={Sajjad},
  year={2025}
}
```

## License

This project is for academic research purposes.

## Contact

For questions or issues, please open an issue on GitHub.

---

**Last Updated:** February 2025  
**Version:** 1.0.0
