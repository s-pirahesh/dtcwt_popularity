"""
Global configuration — DTCWT Popularity Assessment Framework
Version 4.0  (Frozen Evaluation Protocol + WSPI)

Chapter 3 method lineup:
  Baselines   : AF, MeanFreq, EWMA, RRD, VSE, CompoundPop  (7-slot window)
  Section 3-2 : DWT+AF   — Trend-Shock Model (64-day window)
  Section 3-3 : DTCWT+AF — Stable Model      (64-day window)
  Section 3-4 : WSPI     — Proposed Method   (64-day window, frozen params)

Note: 'Statistical' (skewness/kurtosis) removed.
      'Hybrid V3.0' / 'Hybrid V3.1' replaced by 'WSPI'.

Author: Sajjad
"""

import numpy as np
from pathlib import Path

# =============================================================================
# Project paths
# =============================================================================

PROJECT_ROOT = Path(__file__).parent
DATA_DIR     = PROJECT_ROOT / "data" / "datasets"
RESULTS_DIR  = PROJECT_ROOT / "results"

# Ensure base directories exist
RESULTS_DIR.mkdir(exist_ok=True, parents=True)
(RESULTS_DIR / "tables").mkdir(exist_ok=True, parents=True)
(RESULTS_DIR / "figures").mkdir(exist_ok=True, parents=True)

# =============================================================================
# Wavelet configuration
# =============================================================================
# Nested structure matches EvaluationConfig.wavelet_config exactly.
# Evaluators and run_popularity_assessment.py read from:
#   config.wavelet_config['dwt']['wavelet']
#   config.wavelet_config['dtcwt']['biort']  etc.

WAVELET_CONFIG = {
    # ---- Section 3-2: DWT (Trend-Shock Model) --------------------------------
    # db4 gives good time-frequency resolution for popularity signals.
    # level='auto' → resolved at runtime: min(floor(log2(window_size))-1, 5)
    'dwt': {
        'wavelet': 'db4',       # Daubechies-4 wavelet
        'level':   'auto',      # decomposition levels (resolved at runtime)
        'mode':    'symmetric', # signal extension mode
    },

    # ---- Section 3-3 / 3-4: DTCWT (Stable Model + WSPI) --------------------
    # near_sym_a / qshift_a give the best shift-invariance properties.
    # level='auto' → resolved at runtime: min(floor(log2(window_size))-1, 4)
    'dtcwt': {
        'biort':  'near_sym_a', # biorthogonal filter pair
        'qshift': 'qshift_a',   # Q-shift filter pair
        'level':  'auto',       # decomposition levels (resolved at runtime)
    },

    # ---- Flat aliases (kept for legacy code that reads WAVELET_CONFIG directly)
    'dwt_wavelet':        'db4',
    'decomposition_level': 3,
    'dtcwt_biort':        'near_sym_a',
    'dtcwt_qshift':       'qshift_a',
}

# =============================================================================
# WSPI configuration — Section 3-4 (Frozen Parameters)
# =============================================================================
# P_WSPI = mu_L * exp( clip( alpha*S_L + beta*R - gamma*WE, -3, 3 ) )
#
#   mu_L  — trend volume   : WeightedMean(|Low|) with 2^{-i} decay weights
#   S_L   — normalised slope: Slope(|Low|) / (Mean(|Low|) + eps)
#   R     — energy ratio   : E_low / (E_low + sum(E_high))
#   WE    — wavelet entropy : -sum(p_i * log2(p_i)), normalised
#
# Parameters are FROZEN for the dissertation evaluation.
# Do not change without re-running all experiments.

WSPI_CONFIG = {
    'alpha_slope':    1.0,   # weight for normalised trend slope (S_L)
    'beta_ratio':     0.5,   # weight for energy stability ratio  (R)
    'gamma_entropy':  0.5,   # penalty weight for wavelet entropy  (WE)
    'clamp_min':     -3.0,   # lower bound for exponent argument
    'clamp_max':      3.0,   # upper bound → multiplier in [~0.05, ~20]
    'eps':            1e-8,  # numerical stability in slope normalisation
}

# =============================================================================
# Frozen 4-Layer Evaluation Protocol configuration
# =============================================================================

FROZEN_PROTOCOL_CONFIG = {
    # Layer 1 — Decision
    'k_list':                 [5, 10, 20],  # K values for NDCG@K, CHR@K, RSI@K

    # Layer 4 — Robustness
    'robustness_sample_size': 50,           # number of stable items to test
    'spike_multiplier':       10.0,         # noise spike magnitude (×mean)
}


# =============================================================================
# Stratification thresholds — mean count per time-slot
# =============================================================================
# All values are in MEAN COUNT PER SLOT (not cumulative sum).
# This makes thresholds independent of window_size and comparable
# across datasets with different per-slot magnitudes.
#
# Dataset         Unit            cold  low   med   high
# --------------  --------------  ----  ----  ----  -----
# MovieLens       ratings/day      < 1   1-5  5-20   >= 20
# NYC Yellow Taxi trips/hour       < 5  5-50 50-300  >= 300
# YouTube/Youku   views/hour       < 50 50-500 500-5000 >= 5000

STRATA_THRESHOLDS = {
    'movielens': [1, 5, 20],      # mean ratings/day
    'yellow_taxi': [5, 30, 100],  # mean trips/hour per zone (hourly, NYC TLC)
    'youtube':   [50, 500, 5000], # mean views/hour
    'youku':     [50, 500, 5000], # mean views/5-min-slot
}

# =============================================================================
# Evaluation parameters
# =============================================================================

EVAL_CONFIG = {
    'replication_ratios': [0.05, 0.10, 0.20],  # Top 5 %, 10 %, 20 % thresholds
    'random_seed':        42,
}

# =============================================================================
# Dataset configurations
# =============================================================================

DATASETS = {
    # ---------- Primary datasets (implemented) --------------------------------
    'movielens': {
        'path':        DATA_DIR / 'movielens.csv',
        'time_col':    'timestamp',
        'item_col':    'item_id',
        'count_col':   'count',
        'description': 'MovieLens 25M — daily rating counts',
        'granularity': 'daily',
    },

    'yellow_taxi': {
        'name':        'yellow_taxi',
        'path':        DATA_DIR / 'yellow_taxi.csv',
        'time_col':    'timestamp',
        'item_col':    'item_id',
        'count_col':   'count',
        'description': 'NYC Yellow Taxi Trip Records — hourly zone counts',
        'granularity': 'hourly',
        'num_locations': 263,   # NYC taxi zones
        'source':      'NYC TLC',
    },

    'youtube': {
        'name':        'youtube',
        'path':        DATA_DIR / 'youtube_hourly.csv',
        'time_col':    'timestamp',
        'item_col':    'item_id',
        'count_col':   'count',
        'description': 'YouTube hourly video views',
        'granularity': 'hourly',
        'source':      'Kaggle',
    },

    # ---------- Legacy / not yet implemented ----------------------------------
    'youku': {
        'path':      DATA_DIR / 'youku.csv',
        'time_col':  'timestamp',
        'item_col':  'video_id',
        'count_col': 'view_count',
    },

    'youtube07': {
        'path':      DATA_DIR / 'youtube07.csv',
        'time_col':  'timestamp',
        'item_col':  'video_id',
        'count_col': 'view_count',
    },
}

# =============================================================================
# Logging
# =============================================================================

LOGGING_CONFIG = {
    'level':  'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file':   RESULTS_DIR / 'experiment.log',
}
