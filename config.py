"""
Global configuration of the project.

Only the flat entries of WAVELET_CONFIG are read by the code: by the
per-item classes in ``methods/`` and by ``tools/check_dtcwt_shapes.py``
(filter names, DWT wavelet, default level J = 3).  The runs of
the paper take their settings from ``evaluation/protocol_v5.py`` and
``evaluation/method_configs.py`` (windows: baselines 7 slots, the three
wavelet-based methods 64 slots).

The other blocks below are kept from the first submission and are not read
by any code.

Author: Sajjad Pirahesh
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
# The nested entries 'dwt' and 'dtcwt' are not read by any code; the flat
# entries at the end are.

WAVELET_CONFIG = {
    # ---- DWT (DWT+AF) ----------------------------------------------------------
    # db4 gives good time-frequency resolution for popularity signals.
    'dwt': {
        'wavelet': 'db4',       # Daubechies-4 wavelet
        'level':   'auto',      # not read
        'mode':    'symmetric', # signal extension mode
    },

    # ---- DTCWT (DTCWT+AF and WSPI) --------------------------------------------
    # near_sym_a / qshift_a give the best shift-invariance properties.
    'dtcwt': {
        'biort':  'near_sym_a', # biorthogonal filter pair
        'qshift': 'qshift_a',   # Q-shift filter pair
        'level':  'auto',       # not read
    },

    # ---- Flat entries (read by the per-item classes in methods/) ------------
    'dwt_wavelet':        'db4',
    'decomposition_level': 3,
    'dtcwt_biort':        'near_sym_a',
    'dtcwt_qshift':       'qshift_a',
}

# =============================================================================
# Earlier WSPI parameters (first submission) — NOT READ BY ANY CODE
# =============================================================================
# Earlier formula: mu_L * exp( clip( alpha*S_L + beta*R - gamma*WE, -3, 3 ) ).
# The index of the paper is P = mu_L * exp(alpha*R - beta*W_E) with
# alpha = beta = 1 (methods/wspi_assessment.py, evaluation/fast_evaluator.py).

WSPI_CONFIG = {
    'alpha_slope':    1.0,   # weight for normalised trend slope (S_L)
    'beta_ratio':     0.5,   # weight for energy stability ratio  (R)
    'gamma_entropy':  0.5,   # penalty weight for wavelet entropy  (WE)
    'clamp_min':     -3.0,   # lower bound for exponent argument
    'clamp_max':      3.0,   # upper bound → multiplier in [~0.05, ~20]
    'eps':            1e-8,  # numerical stability in slope normalisation
}

# =============================================================================
# Evaluation settings of the first submission — not read by any code
# =============================================================================

FROZEN_PROTOCOL_CONFIG = {
    # Layer 1 — Decision
    'k_list':                 [5, 10, 20],  # K values for NDCG@K, CHR@K, RSI@K

    # Layer 4 — Robustness
    'robustness_sample_size': 50,           # number of stable items to test
    'spike_multiplier':       10.0,         # noise spike magnitude (×mean)
}


# =============================================================================
# Stratification thresholds of the first submission — not read by any code
# (the evaluation of the paper has no strata)
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
# Evaluation parameters of the first submission — not read by any code
# =============================================================================

EVAL_CONFIG = {
    'replication_ratios': [0.05, 0.10, 0.20],  # Top 5 %, 10 %, 20 % thresholds
    'random_seed':        42,
}

# =============================================================================
# Dataset descriptions — not read by any code (prepare_data.py and
# data/README.md describe the datasets of the paper)
# =============================================================================

DATASETS = {
    # ---------- Primary datasets (implemented) --------------------------------
    'movielens': {
        'path':        DATA_DIR / 'movielens.csv',
        'time_col':    'timestamp',
        'item_col':    'item_id',
        'count_col':   'count',
        'description': 'MovieLens 32M — daily rating counts',
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

    # ---------- Not used in the paper ------------------------------------------
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
# Logging — not read by any code
# =============================================================================

LOGGING_CONFIG = {
    'level':  'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file':   RESULTS_DIR / 'experiment.log',
}
