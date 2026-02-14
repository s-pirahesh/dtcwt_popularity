# Changelog

All notable changes to the DTCWT Popularity Assessment project will be documented in this file.

---

## [4.0.0] - 2026-02-14  ← Frozen Evaluation Protocol

### Summary
Complete overhaul of the evaluation layer. The project now implements a rigorous
**4-Layer Frozen Evaluation Protocol** (Decision → Diagnostic → Stability → Robustness)
and introduces **WSPI** (Wavelet Structural Popularity Index) as the primary proposed method.

### Added — Core Protocol
- **`evaluation/metrics.py`** — Five new standalone metric functions:
  - `calculate_ndcg(scores, actuals, k)` — log-relevance NDCG@K (dampens viral outliers)
  - `calculate_hit_rate(scores, actuals, k)` — Cache Hit Ratio under static placement
  - `calculate_rsi(top_k_t1, top_k_t2)` — Ranking Stability Index (Jaccard similarity)
  - `calculate_rank_distortion(scores_clean, scores_noisy, idx)` — Robustness ΔRank
  - `calculate_diagnostics(scores, actuals)` — Kendall τ, Spearman ρ, MAE in one call
  - `MetricsCalculator` shim retained for backward compatibility

- **`evaluation/scenarios.py`** — `RobustnessScenario` class:
  - `select_stable_candidates(matrix)` — identifies low-popularity stable items
  - `inject_spike(ts, multiplier)` — synthetic 10× noise injection

### Added — Configuration
- **`evaluation/evaluation_config.py`** — new fields on `EvaluationConfig`:
  - `k_list: List[int] = [5, 10, 20]` — K values for NDCG/CHR/RSI
  - `robustness_sample_size: int = 50` — items per robustness test
  - `spike_multiplier: float = 10.0` — noise magnitude
  - `to_dict()` and `__repr__` updated to include protocol parameters

### Changed — Evaluators
- **`evaluation/temporal_evaluator.py`** — full rewrite of `_evaluate_method`:
  - Layer 1: NDCG@{5,10,20}, CHR@{5,10,20}
  - Layer 2: Kendall τ, Spearman ρ, MAE
  - Layer 3: RSI@{5,10,20} with stateful top-K tracking across windows
  - Layer 4: Noise injection robustness test
  - New `_save_protocol_metrics()` → saves `protocol/<method>_protocol.parquet`

- **`evaluation/incremental_evaluator.py`** — same 4-Layer protocol added:
  - Protocol output written incrementally as CSV (crash-safe)
  - `self.prev_top_k` state maintained across windows per method
  - RSI resets cleanly between methods

### Changed — Runner
- **`experiments/run_popularity_assessment.py`**:
  - Module docstring rewritten in English, aligned with Chapter 3
  - `create_methods_dict()` rewritten:
    - `Hybrid V3.0`, `Hybrid V3.1`, and `Statistical` removed
    - **WSPI** confirmed as the sole proposed method (Section 3-4)
    - Section 3-2 (`DWT+AF`) and Section 3-3 (`DTCWT+AF`) retained as intermediate models
    - All inline comments in English
  - All Persian `print()` strings replaced with English
  - CLI epilog rewritten with Chapter 3 method lineup and English examples

### Changed — Analysis & Display
- **`evaluation/results_analyzer.py`** — dual-path architecture:
  - `load_protocol_metrics(method)` — reads pre-computed protocol files
  - `recompute_protocol_metrics(method, ...)` — full 4-Layer recalculation from raw scores
  - `calculate_overall_metrics(..., recompute=False)` — recompute flag on all methods
  - `compare_methods(..., recompute=False)` — multi-K comparison table
  - Backward-compat aliases: `spearman`, `kendall`, `ndcg` still work

- **`experiments/analyze_results.py`** — redesigned:
  - Default mode: display pre-computed protocol metrics (fast, no recalculation)
  - `--recompute` flag: recalculate all metrics from raw scores
  - `--save-recomputed` flag: persist recomputed results to `protocol/`
  - All analysis modes updated for multi-K tables

- **`experiments/show_results.py`** — display-only (no computation):
  - Displays 4 sections matching protocol layers
  - Explicitly redirects to `analyze_results.py --recompute` for missing data

### Changed — Public API
- **`evaluation/__init__.py`** — now exports:
  - All five metric functions
  - `RobustnessScenario`

### Storage Changes
| Directory | Format | Content |
|-----------|--------|---------|
| `detailed/` | Parquet | Per-item scores (unchanged) |
| `summary/` | Parquet | Stratum summaries (unchanged) |
| `protocol/` | Parquet/CSV | **NEW** — 4-Layer per-window metrics |
| `comparison/` | CSV/Parquet | Method comparison (unchanged) |
| `metadata/` | JSON | Config now includes `k_list`, `robustness_*` |

### Removed
- `Hybrid V3.0` and `Hybrid V3.1` method names (replaced by `WSPI`)
- `Statistical` method (skewness/kurtosis) — not part of the Chapter 3 framework
- Direct `MetricsCalculator` calls in evaluators (replaced by individual functions)
- Persian comments and print strings in all Python files (now English-only)

---

## [3.1.0] - 2025-01-24

### Added
- **Advanced Features Module**: Shannon Entropy and Hurst Exponent
- **Feature Caching**: LRU cache for improved performance (3-5x speedup)
- **MLflow Integration**: Optional experiment tracking (disabled by default)
- **Comprehensive Documentation**: 
  - README.md with full project overview
  - QUICKSTART.md for getting started quickly
  - Extensive docstrings in all modules
- **Sample Data Generator**: Create synthetic datasets for testing
- **Demo Script**: Interactive demonstration of all methods
- **Installation Test**: Verify setup with test_installation.py

### Enhanced
- **Hybrid Assessment V3.1**: Now includes Shannon Entropy and Hurst Exponent
  - Expected +2-3% improvement over V3.0
  - Configurable weights for all components
- **Better Error Handling**: Graceful fallbacks when libraries unavailable
- **Improved Modularity**: Clean separation of concerns
- **Type Hints**: Better code documentation and IDE support

### Performance
- Feature caching reduces repeated computations
- Batch processing optimizations
- Memory-efficient implementations

## [3.0.0] - 2025-01-20

### Added
- **Hybrid Assessment Method**: Combination of DTCWT + Statistical features
- **Noise Estimation**: Using wavelet detail coefficients
- **Comprehensive Evaluation**: Multiple metrics (Hit Rate, F1, NDCG, etc.)
- **Batch Processing**: Efficient assessment of multiple time series
- **Advanced Baselines**: ARMA and LSTM for prediction comparison

### Changed
- Refactored assessment methods into separate modules
- Improved configuration management
- Enhanced visualization capabilities

## [2.0.0] - 2025-01-15

### Added
- **DTCWT Assessment**: Dual-Tree Complex Wavelet Transform implementation
- **Statistical Assessment**: Skewness + Kurtosis based method
- **Dataset Loaders**: Modular loader system for different datasets
- **Time Series Preprocessing**: Normalization, smoothing, detrending

### Performance
- 10-15% improvement over DWT through better shift invariance
- Faster computation with optimized wavelet transforms

## [1.0.0] - 2025-01-10

### Added
- **DWT Assessment**: Initial wavelet-based popularity assessment
- **Traditional Baselines**: AF, LRU, LFU, EWMA implementations
- **Basic Evaluation**: Hit rate and precision metrics
- **YouTube-07 Loader**: Initial dataset support

### Features
- Multi-level wavelet decomposition
- AF formula with 2^-i weighting
- Basic comparison with traditional methods

---

## Version Naming Convention

- **Major.Minor.Patch** (e.g., 3.1.0)
  - Major: Significant new features or breaking changes
  - Minor: New features, backward compatible
  - Patch: Bug fixes, minor improvements

## Future Versions (Planned)

### [3.2.0] - Planned
- Sub-graph analysis for scalability
- GraphSAGE baseline integration
- Real-time ICN simulation
- Performance benchmarking suite

### [4.0.0] - Planned
- Full ICN integration
- Distributed caching strategies
- Multi-objective optimization
- Production deployment tools

---

**Note**: Dates are approximate and subject to research progress.
