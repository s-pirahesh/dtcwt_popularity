"""
Experiment 1: Assessment Method Comparison
Main experiment comparing all assessment methods
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import *
from data.loaders.youtube07 import YouTube07Loader
from data.loaders.youku import YoukuLoader
from methods.dwt_assessment import DWTAssessment
from methods.dtcwt_assessment import DTCWTAssessment
from methods.statistical_assessment import StatisticalAssessment
from methods.hybrid_assessment import HybridAssessment, HybridAssessmentV30
from baselines.traditional import TraditionalBaselines
from evaluation.metrics import AssessmentMetrics


def run_assessment_experiment(dataset_name: str = 'youtube07',
                              num_items: int = 1000,
                              use_cache: bool = True):
    """
    Run complete assessment comparison experiment
    
    Args:
        dataset_name: Name of dataset to use
        num_items: Number of items to test (for quick testing)
        use_cache: Enable feature caching
        
    Returns:
        Results dictionary
    """
    print(f"\n{'='*70}")
    print(f"Experiment 1: Assessment Method Comparison")
    print(f"Dataset: {dataset_name}")
    print(f"Number of items: {num_items}")
    print(f"{'='*70}\n")
    
    # 1. Load dataset
    print("Step 1: Loading dataset...")
    
    if dataset_name == 'youtube07':
        loader = YouTube07Loader(DATASETS['youtube07'])
    elif dataset_name == 'youku':
        loader = YoukuLoader(DATASETS['youku'])
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    try:
        data = loader.load()
        print(f"✓ Dataset loaded: {len(data)} records")
    except FileNotFoundError:
        print(f"✗ Dataset not found at {DATASETS[dataset_name]['path']}")
        print("  Please download the dataset first.")
        return None
    
    # 2. Create time series
    print("\nStep 2: Creating time series...")
    
    items = loader.get_all_items(data)
    if num_items > 0:
        items = items[:num_items]
    
    window_size = ASSESSMENT_CONFIG['window_size']
    horizon = ASSESSMENT_CONFIG['prediction_horizon']
    
    time_series_dict = {}
    
    for item in tqdm(items, desc="Processing items"):
        ts = loader.create_time_series(data, item, window_size + horizon)
        if len(ts) >= window_size + horizon and np.sum(ts) > 0:
            time_series_dict[item] = ts
    
    print(f"✓ Created {len(time_series_dict)} valid time series")
    
    if len(time_series_dict) == 0:
        print("✗ No valid time series found!")
        return None
    
    # 3. Split data
    print("\nStep 3: Splitting into current and future windows...")
    
    current_series = {item: ts[:window_size] for item, ts in time_series_dict.items()}
    future_accesses = {item: np.sum(ts[window_size:]) for item, ts in time_series_dict.items()}
    
    print(f"✓ Current window: {window_size} time steps")
    print(f"✓ Prediction horizon: {horizon} time steps")
    
    # 4. Initialize methods
    print("\nStep 4: Initializing assessment methods...")
    
    methods = {
        # Baselines
        'AF': None,
        'LRU': None,
        'LFU': None,
        'EWMA': None,
        
        # Proposed methods
        'DWT+AF': DWTAssessment(),
        'DTCWT+AF': DTCWTAssessment(),
        'Statistical': StatisticalAssessment(),
        'Hybrid V3.0': HybridAssessmentV30(),
        'Hybrid V3.1': HybridAssessment(enable_cache=use_cache),
    }
    
    print(f"✓ Initialized {len(methods)} methods")
    
    # 5. Assess popularity
    print("\nStep 5: Assessing popularity with all methods...")
    
    scores = {}
    ts_list = list(current_series.values())
    item_ids = list(current_series.keys())
    
    # Baselines
    print("  Computing baseline scores...")
    baseline_results = TraditionalBaselines.batch_assess_all(ts_list)
    scores.update(baseline_results)
    
    # Proposed methods
    for name, method in methods.items():
        if method is not None:
            print(f"  Computing {name} scores...")
            if hasattr(method, 'batch_assess'):
                method_scores = method.batch_assess(ts_list)
            else:
                method_scores = np.array([method.assess_single(ts) for ts in ts_list])
            scores[name] = method_scores
    
    print(f"✓ Computed scores for all methods")
    
    # 6. Rank items
    print("\nStep 6: Ranking items by predicted popularity...")
    
    rankings = {}
    for method_name, method_scores in scores.items():
        sorted_indices = np.argsort(method_scores)[::-1]
        rankings[method_name] = [item_ids[i] for i in sorted_indices]
    
    print(f"✓ Created rankings for all methods")
    
    # 7. Evaluate
    print("\nStep 7: Evaluating methods...")
    print(f"\n{'Method':<20} {'Top-K':<10} {'Hit Rate':<12} {'Precision':<12} {'FP Rate':<12} {'F1':<12}")
    print("-" * 78)
    
    results = {}
    
    for method_name, ranking in rankings.items():
        method_results = AssessmentMetrics.compute_all_metrics(
            ranking, 
            future_accesses,
            top_k_ratios=EVAL_CONFIG['replication_ratios']
        )
        results[method_name] = method_results
        
        # Print results
        for top_k, metrics in method_results.items():
            print(f"{method_name:<20} {top_k:<10} "
                  f"{metrics['hit_rate']:<12.4f} "
                  f"{metrics['precision']:<12.4f} "
                  f"{metrics['fp_rate']:<12.4f} "
                  f"{metrics['f1']:<12.4f}")
    
    # 8. Save results
    print("\nStep 8: Saving results...")
    
    results_df = create_results_dataframe(results)
    output_path = RESULTS_DIR / 'tables' / f'{dataset_name}_assessment_results.csv'
    results_df.to_csv(output_path, index=False)
    
    print(f"✓ Results saved to: {output_path}")
    
    # 9. Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    # Find best method for each metric
    for top_k in results[list(results.keys())[0]].keys():
        print(f"\n{top_k}:")
        
        # Hit Rate
        best_hr = max(results.items(), 
                     key=lambda x: x[1][top_k]['hit_rate'])
        print(f"  Best Hit Rate: {best_hr[0]} ({best_hr[1][top_k]['hit_rate']:.4f})")
        
        # F1 Score
        best_f1 = max(results.items(),
                     key=lambda x: x[1][top_k]['f1'])
        print(f"  Best F1 Score:  {best_f1[0]} ({best_f1[1][top_k]['f1']:.4f})")
    
    print("\n" + "="*70 + "\n")
    
    return results


def create_results_dataframe(results: dict) -> pd.DataFrame:
    """
    Convert nested results dict to pandas DataFrame
    
    Args:
        results: Nested dictionary of results
        
    Returns:
        DataFrame with all results
    """
    rows = []
    
    for method, top_k_results in results.items():
        for top_k, metrics in top_k_results.items():
            row = {
                'Method': method,
                'Top_K': top_k,
                'Hit_Rate': metrics['hit_rate'],
                'Precision': metrics['precision'],
                'FP_Rate': metrics['fp_rate'],
                'NDCG': metrics['ndcg'],
                'F1': metrics['f1'],
            }
            rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Sort by method and top_k
    df = df.sort_values(['Top_K', 'Hit_Rate'], ascending=[True, False])
    
    return df


def compare_methods(results: dict, top_k: str = 'top_10%'):
    """
    Compare methods and compute improvements
    
    Args:
        results: Results dictionary
        top_k: Which top-k to analyze
    """
    print(f"\nMethod Comparison ({top_k}):")
    print("-" * 60)
    
    # Get baseline (AF)
    baseline_hr = results['AF'][top_k]['hit_rate']
    
    # Compare each method to baseline
    for method, metrics in results.items():
        hr = metrics[top_k]['hit_rate']
        improvement = ((hr - baseline_hr) / baseline_hr) * 100
        
        print(f"{method:<20}: {hr:.4f} ({improvement:+.1f}% vs AF)")


if __name__ == '__main__':
    # Run experiment
    results = run_assessment_experiment(
        dataset_name='youtube07',
        num_items=1000,  # Use 1000 items for quick test, set to 0 for all
        use_cache=True
    )
    
    if results:
        # Compare methods
        compare_methods(results, 'top_10%')
