"""
Global configuration for DTCWT popularity assessment
Version 3.1
Author: Sajjad
Date: January 2025
"""
import numpy as np
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data" / "datasets"
RESULTS_DIR = PROJECT_ROOT / "results"
CACHE_DIR = PROJECT_ROOT / "results" / "feature_cache"

# Ensure directories exist
RESULTS_DIR.mkdir(exist_ok=True, parents=True)
(RESULTS_DIR / "tables").mkdir(exist_ok=True, parents=True)
(RESULTS_DIR / "figures").mkdir(exist_ok=True, parents=True)
(RESULTS_DIR / "raw_data").mkdir(exist_ok=True, parents=True)
CACHE_DIR.mkdir(exist_ok=True, parents=True)

# Wavelet parameters
WAVELET_CONFIG = {
    'dwt_wavelet': 'db4',           # Daubechies 4 for DWT
    'decomposition_level': 3,        # 3 levels of decomposition
    'dtcwt_biort': 'near_sym_a',    # DTCWT biorthogonal filter
    'dtcwt_qshift': 'qshift_a',     # DTCWT Q-shift filter
}

# Assessment parameters
ASSESSMENT_CONFIG = {
    # Statistical method weights
    'stat_alpha': 1.0,    # weight for mean
    'stat_beta': 0.5,     # weight for skewness
    'stat_gamma': 0.3,    # weight for kurtosis
    
    # Hybrid method weights (basic)
    'hybrid_alpha': 1.0,
    'hybrid_beta': 0.5,
    'hybrid_gamma': 0.3,
    'hybrid_noise_penalty': 0.1,
    
    # NEW in V3.1: Advanced feature weights
    'hybrid_entropy_weight': 0.2,   # Shannon entropy weight
    'hybrid_hurst_weight': 0.3,     # Hurst exponent weight
    
    # Window settings
    'window_size': 30,           # days for time window
    'prediction_horizon': 7,     # predict 7 days ahead
}

# NEW in V3.1: Advanced features configuration
ADVANCED_FEATURES_CONFIG = {
    'shannon_entropy': True,      # Enable Shannon entropy
    'hurst_exponent': True,       # Enable Hurst exponent
    'sample_entropy': False,      # Optional, slower
    'permutation_entropy': False, # Optional
}

# NEW in V3.1: Caching configuration
CACHE_CONFIG = {
    'enabled': True,
    'memory_cache_size': 1000,    # LRU cache size
    'disk_cache_enabled': False,  # For very large datasets
    'disk_cache_dir': str(CACHE_DIR),
}

# NEW in V3.1: MLflow configuration (OPTIONAL)
MLFLOW_CONFIG = {
    'enabled': False,  # Set to True to enable MLflow tracking
    'tracking_uri': 'file:./mlruns',
    'experiment_name': 'DTCWT_Popularity_Assessment',
}

# Evaluation parameters
EVAL_CONFIG = {
    'replication_ratios': [0.05, 0.10, 0.20],  # Top 5%, 10%, 20%
    'train_test_split': 0.8,
    'cross_validation_folds': 5,
    'random_seed': 42,
}

# Dataset configurations
DATASETS = {
    'youtube07': {
        'path': DATA_DIR / 'youtube07.csv',
        'time_col': 'timestamp',
        'item_col': 'video_id',
        'count_col': 'view_count',
    },
    'movielens': {
        'path': DATA_DIR / 'movielens.csv',
        'time_col': 'timestamp',
        'item_col': 'item_id',
        'count_col': 'count',
    },
    'foursquare': {
        'path': DATA_DIR / 'foursquare.csv',
        'time_col': 'timestamp',
        'item_col': 'venue_id',
        'count_col': 'checkin_count',
    },
    'higgs_twitter': {
        'path': DATA_DIR / 'higgs_twitter.csv',
        'time_col': 'timestamp',
        'item_col': 'tweet_id',
        'count_col': 'retweet_count',
    },
    'uber': {
        'name': 'uber',
        'path': DATA_DIR / 'uber.csv',
        'time_col': 'timestamp',
        'item_col': 'item_id',
        'count_col': 'count',
        'description': 'NYC Yellow Taxi Trip Records',
        'granularity': 'hourly',  # Granularity from converter
        'window_size_default': 30,  # 30 time slots (NOT hours!)
        'num_locations': 263,  # NYC taxi zones
        'date_range': '2009-present',
        'source': 'NYC TLC'
    },
    'youku': {
        'path': DATA_DIR / 'youku.csv',
        'time_col': 'timestamp',
        'item_col': 'video_id',
        'count_col': 'view_count',
    },
}

# Machine Learning parameters (for prediction)
ML_CONFIG = {
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 5,
        'random_state': 42,
    },
}

# Logging
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': RESULTS_DIR / 'experiment.log',
}
