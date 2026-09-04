# -*- coding: utf-8 -*-
"""
Popularity Assessment — Main Evaluation Pipeline
=================================================
Evaluates content popularity using the Frozen 4-Layer Protocol.

Three assessment models from Chapter 3:
  Baselines : AF, EWMA, RRD, VSE, CompoundPop, PFRF  (7-slot window)
  Section 3-2: DWT+AF   — Trend-Shock Model  (64-day window)
  Section 3-3: DTCWT+AF — Stable DTCWT Model (64-day window)
  Section 3-4: WSPI     — Proposed Method     (64-day window, frozen params)

Optional Group 5 — explicit value-forecasting baselines (defense-gap
experiment, run ONLY when named in --methods):
  Persistence, Holt, ARYW, ARIMA

Workflow:
  1. This script  → compute scores + 4-Layer metrics → results/
  2. analyze_results.py → display or --recompute metrics
  3. show_results.py    → textual + graphical display

Selective / cached comparison (no re-run of finished methods):
  python experiments/prepare_compare_folder.py results/<ds>/main_<ts>
  python experiments/run_popularity_assessment.py <ds> \
      --resume results/<ds>/predcmp_<ts> --methods WSPI AF ARIMA Holt

Author: Sajjad
Date: February 2026
"""

import sys
import argparse
from pathlib import Path

# Make stdout/stderr UTF-8 safe BEFORE importing modules that print Unicode
# (e.g. methods/__init__.py prints a checkmark). On Windows a redirected
# stdout defaults to cp1252 and would otherwise crash with UnicodeEncodeError.
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation import (
    EvaluationConfig,
    get_movielens_config,
    get_youtube_config,
    get_youku_config,
    get_yellow_taxi_config,
    TemporalEvaluator
)

try:
    from evaluation.incremental_evaluator import IncrementalTemporalEvaluator
    INCREMENTAL_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Warning: Incremental evaluator not available: {e}")
    IncrementalTemporalEvaluator = None
    INCREMENTAL_AVAILABLE = False
from data.loaders import MovieLensLoader, YellowTaxiLoader, YouTubeLoader
# YouTubeLoader and YoukuLoader are not yet fully implemented — MovieLens and NYC Yellow Taxi are the primary datasets
# Import assessment methods (may require optional dependencies: pywt, dtcwt)
try:
    from methods import (
        DTCWTAssessment,
        DWTAssessment,
        HybridAssessment,
    )
    from methods.wspi_assessment import WSPIAssessment
    METHODS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some methods not available: {e}")
    print("Install dependencies: pip install pywt dtcwt")
    DTCWTAssessment = None
    DWTAssessment = None
    HybridAssessment = None
    WSPIAssessment = None
    METHODS_AVAILABLE = False

# Import explicit forecasting baselines (Group 5 — defense-gap experiment)
try:
    from methods.forecasting_baselines import (
        PersistenceForecast,
        HoltForecast,
        ARYWForecast,
        ARIMAForecast,
    )
    FORECASTERS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Forecasting baselines not available: {e}")
    PersistenceForecast = HoltForecast = ARYWForecast = ARIMAForecast = None
    FORECASTERS_AVAILABLE = False

# Import WSPI-F (Group 6 — Level-2 module: coefficient-domain forecasting)
try:
    from methods.wspi_forecast import WSPIForecast
    WSPIF_AVAILABLE = True
except ImportError as e:
    print(f"Warning: WSPI-F not available: {e}")
    WSPIForecast = None
    WSPIF_AVAILABLE = False

# Import WSPI-F2 / WSPI-FT (Group 7 — second-generation Level-2 module)
try:
    from methods.wspi_forecast2 import WSPIForecast2
    WSPIF2_AVAILABLE = True
except ImportError as e:
    print(f"Warning: WSPI-F2 not available: {e}")
    WSPIForecast2 = None
    WSPIF2_AVAILABLE = False

# Group 5, 6 & 7 methods are OPT-IN: they never run unless named in --methods.
FORECASTING_METHODS = ['Persistence', 'Holt', 'ARYW', 'ARIMA',
                       'WSPI-F', 'WSPI-F-YW', 'WSPI-F2', 'WSPI-FT']

# Import baselines
try:
    from baselines import (
        AFMethod,
        EWMAMethod,
        RRDMethod,
        VSEMethod,
        CompoundPopMethod,
        PFRFMethod,
        get_all_baseline_methods,
    )
    BASELINES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Baselines not available: {e}")
    AFMethod = EWMAMethod = RRDMethod = VSEMethod = CompoundPopMethod = PFRFMethod = None
    get_all_baseline_methods = None
    BASELINES_AVAILABLE = False
from config import WAVELET_CONFIG, DATASETS


def create_methods_dict(config: EvaluationConfig) -> dict:
    """
    Build the methods dictionary for the Frozen Evaluation Protocol.

    Chapter 3 method lineup:
      Baselines   — AF, EWMA, RRD, VSE, CompoundPop, PFRF  (7-slot window)
      Section 3-2 — DWT+AF   (Trend-Shock Model) (64-day window)
      Section 3-3 — DTCWT+AF (Stable Model)      (64-day window)
      Section 3-4 — WSPI     (Proposed, frozen)  (64-day window)
      Group 5     — Persistence, Holt, ARYW, ARIMA (explicit forecasters,
                    opt-in via --methods only)

    Note: 'Statistical' (skewness/kurtosis) has been removed from the
    evaluation lineup as it does not belong to the Chapter 3 framework.

    Args:
        config: EvaluationConfig instance

    Returns:
        dict: {method_name: method_instance}
    """
    methods = {}

    # --- Group 1: General Popularity Baselines (7-slot window) ---------------
    if BASELINES_AVAILABLE:
        methods['AF']          = AFMethod()

        methods['EWMA']        = EWMAMethod(alpha=0.2)
        methods['RRD']         = RRDMethod()
        methods['VSE']         = VSEMethod()
        methods['CompoundPop'] = CompoundPopMethod()
        methods['PFRF']        = PFRFMethod()

    # --- Group 2 & 3 & 4: Chapter 3 models -----------------------------------
    if METHODS_AVAILABLE:
        dwt_level   = config.wavelet_config['dwt']['level']
        dtcwt_level = config.wavelet_config['dtcwt']['level']

        # Section 3-2: Trend-Shock Model (DWT)
        # Score_DWT = WAF(cA_L) + beta * WAF(cD_1)
        methods['DWT+AF'] = DWTAssessment(
            wavelet=config.wavelet_config['dwt']['wavelet'],
            level=dwt_level,
            mode=config.wavelet_config['dwt']['mode']
        )

        # Section 3-3: Stable DTCWT Model
        # Score_DTCWT = WAF(M_trend) + beta * WAF(M_shock)
        methods['DTCWT+AF'] = DTCWTAssessment(
            level=dtcwt_level,
            biort=config.wavelet_config['dtcwt']['biort'],
            qshift=config.wavelet_config['dtcwt']['qshift']
        )

        # Section 3-4: WSPI — Proposed Method (final, evidence-based form)
        # WSPI = mu_L * exp(alpha*R - beta*WE)   with alpha = beta = 1
        # (slope term and clip removed; see methods/wspi_assessment.py)
        if WSPIAssessment is not None:
            methods['WSPI'] = WSPIAssessment(alpha=1.0, beta=1.0)

    # --- Group 5: Explicit value-forecasting baselines (opt-in) --------------
    # Score = sum of forecasted next-horizon values, so the horizon must
    # match the evaluation's prediction horizon.
    if FORECASTERS_AVAILABLE:
        horizon = int(getattr(config, 'prediction_horizon', 7) or 7)
        methods['Persistence'] = PersistenceForecast(horizon=horizon)
        methods['Holt']        = HoltForecast(horizon=horizon)
        methods['ARYW']        = ARYWForecast(order=2, horizon=horizon)
        methods['ARIMA']       = ARIMAForecast(p=1, q=1, horizon=horizon)

    # --- Group 6: WSPI-F — forecasted WSPI (opt-in) --------------------------
    if WSPIF_AVAILABLE:
        horizon = int(getattr(config, 'prediction_horizon', 7) or 7)
        methods['WSPI-F']    = WSPIForecast(predictor='nlms', horizon=horizon)
        methods['WSPI-F-YW'] = WSPIForecast(predictor='aryw', horizon=horizon)

    # --- Group 7: WSPI-F2 / WSPI-FT — bounded + gated variants (opt-in) ------
    # WSPI-F2 : AR(2)-YW + clamp + structural gate (reduces to WSPI at g=0)
    # WSPI-FT : Theil-Sen robust regression + clamp (distinct predictor)
    # Defaults are frozen in methods/wspi_forecast2.py; the horizon must match
    # the evaluation horizon exactly, as for every other forecasting method.
    if WSPIF2_AVAILABLE:
        horizon = int(getattr(config, 'prediction_horizon', 7) or 7)
        methods['WSPI-F2'] = WSPIForecast2(
            predictor='aryw', horizon=horizon,
            clamp=4.0, phi_d=1.0, use_gate=True, gate_gamma=0.3)
        methods['WSPI-FT'] = WSPIForecast2(
            predictor='theilsen', horizon=horizon,
            clamp=4.0, phi_d=1.0, use_gate=False)

    # --- Filter by --methods CLI flag ----------------------------------------
    if config.methods is not None:
        methods = {k: v for k, v in methods.items() if k in config.methods}
    else:
        # No explicit selection: exclude opt-in forecasters (Group 5)
        methods = {k: v for k, v in methods.items()
                   if k not in FORECASTING_METHODS}

    return methods


def get_data_loader(dataset_name: str, data_path: str = None):
    """
    Create the appropriate data loader for the given dataset.

    Args:
        dataset_name: one of 'movielens', 'yellow_taxi', 'youtube'
        data_path:    optional override for the file path in config

    Returns:
        DataLoader instance
    """
    if dataset_name == 'movielens':
        config = DATASETS['movielens'].copy()
        if data_path:
            config['path'] = Path(data_path)
        return MovieLensLoader(config)
    elif dataset_name == 'yellow_taxi':
        config = DATASETS.get('yellow_taxi', {
            'name': 'yellow_taxi',
            'path': data_path or Path('./data/yellow_taxi/yellow_taxi.csv'),
            'time_col': 'timestamp',
            'item_col': 'item_id',
            'count_col': 'count'
        }).copy()
        if data_path:
            config['path'] = Path(data_path)
        return YellowTaxiLoader(config)
    elif dataset_name == 'youtube':
        config = DATASETS['youtube'].copy()
        if data_path:
            config['path'] = Path(data_path)
        return YouTubeLoader(config)
    elif dataset_name in ('youtube07', 'youku'):
        raise NotImplementedError(
            f"Loader for '{dataset_name}' is not yet implemented. "
            "Use: --dataset movielens"
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")



def run_temporal_evaluation(dataset_name: str,
                           num_items: int = None,
                           start_date: str = None,
                           end_date: str = None,
                           window_size: int = 30,
                           prediction_horizon: int = 7,
                           methods: list = None,
                           final_format: str = 'csv',
                           parallel: bool = True,
                           num_cores: int = -1,
                           data_path: str = None,
                           incremental: bool = False,
                           **kwargs):
    """
    Run a full temporal evaluation with the Frozen 4-Layer Protocol.

    Args:
        dataset_name:       'movielens', 'yellow_taxi', 'youtube'
        num_items:          number of items (None = all)
        start_date:         'YYYY-MM-DD' or None (dataset start)
        end_date:           'YYYY-MM-DD' or None (dataset end)
        window_size:        training window in days (default 30)
        prediction_horizon: assessment horizon in days (default 7)
        methods:            list of method names (None = all)
        final_format:       'csv' or 'parquet' for comparison output
        parallel:           enable parallel window processing
        num_cores:          CPU cores (-1 = all available)
        data_path:          override dataset file path
        incremental:        use IncrementalTemporalEvaluator (crash-safe)
        **kwargs:           forwarded to get_*_config (k_list, etc.)
    """

    print("="*70)
    print("EXPERIMENT: TEMPORAL EVALUATION — FROZEN 4-LAYER PROTOCOL")
    print("="*70)
    print(f"Dataset:         {dataset_name}")
    print(f"Items:           {num_items or 'all'}")
    print(f"Time Range:      {start_date or 'start'} to {end_date or 'end'}")
    print(f"Window Size:     {window_size} days")
    print(f"Horizon:         {prediction_horizon} days")
    print(f"Final Format:    {final_format.upper()}")
    print(f"Parallel:        {parallel}")
    print(f"Protocol:        Frozen 4-Layer (Decision, Diagnostic, Stability, Robustness)")
    print("="*70 + "\n")

    # 1. Build config
    if dataset_name == 'movielens':
        config = get_movielens_config(
            num_items=num_items,
            start_date=start_date,
            end_date=end_date,
            window_size=window_size,
            prediction_horizon=prediction_horizon,
            methods=methods,
            final_format=final_format,
            parallel=parallel,
            num_cores=num_cores,
            **kwargs
        )
    elif dataset_name == 'yellow_taxi':
        config = get_yellow_taxi_config(
            num_items=num_items,
            start_date=start_date,
            end_date=end_date,
            window_size=window_size or 30,
            prediction_horizon=prediction_horizon or 7,
            methods=methods,
            final_format=final_format,
            parallel=parallel,
            num_cores=num_cores,
            **kwargs
        )
    elif dataset_name == 'youtube':
        config = get_youtube_config(
            num_items=num_items,
            start_date=start_date,
            end_date=end_date,
            window_size=window_size,
            prediction_horizon=prediction_horizon,
            methods=methods,
            final_format=final_format,
            parallel=parallel,
            num_cores=num_cores,
            **kwargs
        )
        loader = YouTubeLoader(DATASETS['youtube'])
    elif dataset_name == 'youtube07':
        config = get_youtube_config(
            num_items=num_items,
            start_date=start_date,
            end_date=end_date,
            window_size=window_size,
            prediction_horizon=prediction_horizon,
            methods=methods,
            final_format=final_format,
            parallel=parallel,
            num_cores=num_cores,
            **kwargs
        )
    elif dataset_name == 'youku':
        config = get_youku_config(
            num_items=num_items,
            start_date=start_date,
            end_date=end_date,
            window_size=window_size,
            prediction_horizon=prediction_horizon,
            methods=methods,
            final_format=final_format,
            parallel=parallel,
            num_cores=num_cores,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    if config.verbose:
        print(config)

    print(f"Protocol:        Frozen 4-Layer (Decision, Diagnostic, Stability, Robustness)")
    print(f"Metrics:         NDCG@{config.k_list}, CHR, RSI")
    print(f"Robustness:      {config.robustness_sample_size} items, {config.spike_multiplier}x spike")
    print()

    # 2. Load data
    data_loader = get_data_loader(dataset_name, data_path=data_path)
    # record the data file path on the config so it lands in run_metadata.json
    config.data_path = data_path

    # 3. Build methods dict
    methods_dict = create_methods_dict(config)

    print(f"Methods to evaluate: {list(methods_dict.keys())}\n")

    # 4. Create evaluator
    if incremental and INCREMENTAL_AVAILABLE:
        print("Using INCREMENTAL evaluation mode")
        print("  Memory efficient (<200 MB)")
        print("  Crash-safe (continuous saving)")
        print("  Method-specific window sizes\n")

        # Load FULL dataset (no item filtering) — incremental evaluator
        # will select items itself using the same logic as TemporalEvaluator,
        # so both modes always work with identical item sets.
        full_data = data_loader.load_data()

        storage_path = config.output_dir

        evaluator = IncrementalTemporalEvaluator(
            config=config,
            methods=methods_dict,
            data=full_data,
            items=None,          # ignored; evaluator re-selects internally
            storage_path=storage_path
        )
    else:
        if incremental and not INCREMENTAL_AVAILABLE:
            print("Warning: Incremental mode requested but not available, using standard mode")

        print("Using STANDARD evaluation mode\n")

        evaluator = TemporalEvaluator(
            data_loader=data_loader,
            methods=methods_dict,
            config=config
        )

    # 5. Run evaluation
    evaluator.evaluate()

    # 6. Summary
    print("\n" + "="*70)
    print("EVALUATION COMPLETED SUCCESSFULLY")
    print("="*70)
    print(f"Results saved to: {config.output_dir}")
    print("\nOutput structure:")
    print(f"  {config.output_dir}/")
    print(f"    ├── detailed/          # Detailed scores per window/item (Parquet)")
    print(f"    ├── summary/           # Stratum summaries (Parquet)")
    print(f"    ├── protocol/          # 4-Layer Protocol metrics per window (Parquet/CSV)")
    print(f"    ├── comparison/        # Method comparison")
    print(f"    ├── metadata/          # Config and runtime statistics")
    print(f"    └── visualization/     # (reserved for plots)")
    print("="*70 + "\n")

    return evaluator


def main():
    """Entry point — parse CLI arguments and run evaluation."""

    parser = argparse.ArgumentParser(
        description='Content Popularity Assessment — Frozen 4-Layer Evaluation Protocol',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Chapter 3 methods:
  Baselines   : AF, EWMA, RRD, VSE, CompoundPop, PFRF  (7-slot window)
  Section 3-2 : DWT+AF   — Trend-Shock Model   (64-day window)
  Section 3-3 : DTCWT+AF — Stable DTCWT Model  (64-day window)
  Section 3-4 : WSPI     — Proposed Method      (64-day window)
  Group 5     : Persistence Holt ARYW ARIMA (explicit forecasters, OPT-IN:
                only run when named in --methods; ARIMA is slow)
  Group 6     : WSPI-F WSPI-F-YW (forecasted WSPI — coefficient-domain
                prediction; OPT-IN like Group 5)

Examples:

  # Quick test (100 items, 1 month)
  python run_popularity_assessment.py movielens \\
      --num-items 100 \\
      --start-date 2023-08-01 --end-date 2023-08-31 \\
      --incremental

  # Medium run (500 items, 3 months)
  python run_popularity_assessment.py movielens \\
      --num-items 500 \\
      --start-date 2023-01-01 --end-date 2023-03-31

  # Full evaluation (1000 items, 1 year)
  python run_popularity_assessment.py movielens \\
      --num-items 1000 \\
      --start-date 2023-01-01 --end-date 2023-12-31

  # Custom K values for NDCG/CHR/RSI
  python run_popularity_assessment.py movielens \\
      --num-items 500 --k-list 5 10 20 50

  # Proposed method only
  python run_popularity_assessment.py movielens \\
      --num-items 500 --methods WSPI DTCWT+AF AF

  # Cached comparison: run ONLY the new forecasters, reuse cached results
  # (first: python experiments/prepare_compare_folder.py results/<ds>/main_<ts>)
  python run_popularity_assessment.py youtube \\
      --resume results/youtube/predcmp_<ts> \\
      --methods WSPI AF Persistence Holt ARYW ARIMA

Window formula:
  num_windows = (end_date - start_date) - window_size - horizon + 1
        """
    )

    parser.add_argument('dataset', type=str,
                        choices=['movielens', 'youtube07', 'youku', 'yellow_taxi', 'youtube'],
                        help='Dataset name')

    parser.add_argument('--num-items', type=int, default=None,
                        help='Number of items to evaluate (default: all)')

    parser.add_argument('--start-date', type=str, default=None,
                        help='Start date YYYY-MM-DD (default: dataset start)')

    parser.add_argument('--end-date', type=str, default=None,
                        help='End date YYYY-MM-DD (default: dataset end)')

    parser.add_argument('--window-size', type=int, default=30,
                        help='Training window size in days (default: 30)')

    parser.add_argument('--horizon', type=int, default=7,
                        help='Assessment horizon in days (default: 7)')

    parser.add_argument('--methods', type=str, nargs='+', default=None,
                        help='Methods to evaluate (default: all EXCEPT opt-in Groups 5-6). '
                             'Choices: AF EWMA RRD VSE CompoundPop PFRF DWT+AF DTCWT+AF WSPI '
                             'Persistence Holt ARYW ARIMA WSPI-F WSPI-F-YW')

    parser.add_argument('--format', type=str, default='csv',
                        choices=['csv', 'parquet'],
                        help='Comparison output format (default: csv). Intermediate always Parquet.')

    parser.add_argument('--no-parallel', action='store_true',
                        help='Disable parallel processing')

    parser.add_argument('--cores', type=int, default=-1,
                        help='CPU cores to use (default: all, -1)')

    parser.add_argument('--item-selection', type=str, default='top',
                        choices=['top', 'random', 'stratified'],
                        help='Item selection strategy (default: top)')

    parser.add_argument('--data-path', type=str, default=None,
                        help='Override dataset file path from config')

    parser.add_argument('--incremental', action='store_true',
                        help='Use incremental evaluation (memory-efficient, crash-safe)')

    parser.add_argument('--quiet', action='store_true',
                        help='Suppress verbose config output')

    parser.add_argument('--resume', type=str, default=None,
                        help='Parallel mode: continue an interrupted run by giving its folder path '
                             '(re-runs only methods not marked done in that folder).')

    parser.add_argument('--k-list', type=int, nargs='+', default=None,
                        metavar='K',
                        help='K values for NDCG/CHR/RSI (default: 5 10 20). '
                             'Example: --k-list 5 10 20 50')

    args = parser.parse_args()

    # ---- Parallel path: one process per method, one shared folder ----------
    if not args.no_parallel:
        from experiments.parallel_engine import run_methods_parallel
        import evaluation.method_configs as _mc
        # Default lineup EXCLUDES the opt-in Group 5 forecasters, so existing
        # commands keep their original behaviour (and ARIMA never runs by accident).
        method_names = args.methods or [
            m for m in _mc.METHOD_CONFIGS.keys() if m not in FORECASTING_METHODS
        ]
        run_methods_parallel(
            args.dataset, method_names,
            cores=args.cores, tag='main',
            specs=None,
            data_path=args.data_path, num_items=args.num_items,
            start_date=args.start_date, end_date=args.end_date,
            window_size=args.window_size, prediction_horizon=args.horizon,
            item_selection=args.item_selection,
            k_list=args.k_list, resume=args.resume,
        )
        return

    # ---- Serial path (original, unchanged) ---------------------------------
    run_temporal_evaluation(
        dataset_name=args.dataset,
        num_items=args.num_items,
        start_date=args.start_date,
        end_date=args.end_date,
        window_size=args.window_size,
        prediction_horizon=args.horizon,
        methods=args.methods,
        final_format=args.format,
        parallel=not args.no_parallel,
        num_cores=args.cores,
        data_path=args.data_path,
        incremental=args.incremental,
        item_selection=args.item_selection,
        verbose=not args.quiet,
        **(({'k_list': args.k_list} if args.k_list else {}))
    )


if __name__ == '__main__':
    main()
