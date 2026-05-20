# -*- coding: utf-8 -*-
"""
Evaluation Module — Frozen 4-Layer Evaluation Protocol
=======================================================
Public API for the comprehensive temporal evaluation framework.

Layers:
  1. Decision:    NDCG@K, CHR@K
  2. Diagnostic:  Kendall τ, Spearman ρ, MAE
  3. Stability:   RSI@K
  4. Robustness:  Rank Distortion (Noise Injection)
"""

# --- Config & dataset helpers ------------------------------------------------
from .evaluation_config import (
    EvaluationConfig,
    get_movielens_config,
    get_youtube_config,
    get_youku_config,
    get_yellow_taxi_config,
)

# --- Core subsystems ---------------------------------------------------------
from .stratification import StratificationSystem
from .storage import StorageSystem
from .wavelet_validator import WaveletWindowValidator

# --- Metric functions (Frozen Evaluation Protocol) ---------------------------
from .metrics import (
    calculate_ndcg,
    calculate_hit_rate,
    calculate_rsi,
    calculate_rank_distortion,
    calculate_diagnostics,
    MetricsCalculator,          # backward-compat shim
)

# --- Robustness scenario -----------------------------------------------------
from .scenarios import RobustnessScenario

# --- Evaluators --------------------------------------------------------------
from .temporal_evaluator import TemporalEvaluator

# --- Analysis & visualisation ------------------------------------------------
from .results_analyzer import ResultsAnalyzer
from .visualizer import ResultsVisualizer


__all__ = [
    # Config
    'EvaluationConfig',
    'get_movielens_config',
    'get_youtube_config',
    'get_youku_config',
    'get_yellow_taxi_config',
    # Core subsystems
    'StratificationSystem',
    'StorageSystem',
    'WaveletWindowValidator',
    # Metric functions
    'calculate_ndcg',
    'calculate_hit_rate',
    'calculate_rsi',
    'calculate_rank_distortion',
    'calculate_diagnostics',
    'MetricsCalculator',
    # Robustness
    'RobustnessScenario',
    # Evaluators
    'TemporalEvaluator',
    # Analysis
    'ResultsAnalyzer',
    'ResultsVisualizer',
]