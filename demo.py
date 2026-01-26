#!/usr/bin/env python3
"""
Quick Demo: DTCWT-based Popularity Assessment
Uses synthetic data to demonstrate the system
"""
import numpy as np
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from methods.dwt_assessment import DWTAssessment
from methods.dtcwt_assessment import DTCWTAssessment
from methods.statistical_assessment import StatisticalAssessment
from methods.hybrid_assessment import HybridAssessment
from baselines.traditional import TraditionalBaselines


def generate_synthetic_time_series(n_items=100, length=30, seed=42):
    """
    Generate synthetic time series with different patterns
    
    Returns:
        dict: {item_id: time_series}
    """
    np.random.seed(seed)
    
    time_series_dict = {}
    
    for i in range(n_items):
        # Different patterns
        pattern_type = i % 4
        
        if pattern_type == 0:
            # Trending up
            ts = np.cumsum(np.random.poisson(5, length)) + np.random.randn(length) * 2
        elif pattern_type == 1:
            # Stable
            ts = np.random.poisson(10, length) + np.random.randn(length) * 2
        elif pattern_type == 2:
            # Bursty
            ts = np.random.poisson(3, length)
            ts[np.random.choice(length, size=5)] += np.random.poisson(20, 5)
        else:
            # Declining
            ts = np.maximum(0, 20 - np.cumsum(np.random.poisson(1, length)))
        
        ts = np.maximum(0, ts)  # Ensure non-negative
        time_series_dict[f'item_{i:03d}'] = ts
    
    return time_series_dict


def demo_basic_assessment():
    """Demonstrate basic popularity assessment"""
    print("\n" + "="*70)
    print("DEMO 1: Basic Popularity Assessment")
    print("="*70 + "\n")
    
    # Create sample time series
    print("Creating sample time series...")
    ts_trending = np.array([5, 8, 12, 15, 20, 25, 30, 35, 40, 45,
                           50, 55, 60, 65, 70, 75, 80, 85, 90, 95,
                           100, 105, 110, 115, 120, 125, 130, 135, 140, 145])
    
    ts_stable = np.array([10]*30) + np.random.randn(30) * 2
    
    ts_bursty = np.array([5]*30)
    ts_bursty[[5, 10, 15, 20, 25]] = [50, 60, 55, 70, 65]
    
    print("✓ Created 3 time series: trending, stable, bursty\n")
    
    # Initialize methods
    print("Initializing assessment methods...")
    dwt = DWTAssessment()
    dtcwt = DTCWTAssessment()
    stat = StatisticalAssessment()
    hybrid = HybridAssessment()
    print("✓ Methods initialized\n")
    
    # Assess each time series
    print(f"{'Method':<20} {'Trending':<12} {'Stable':<12} {'Bursty':<12}")
    print("-" * 56)
    
    for method_name, method in [
        ('AF', None),
        ('DWT+AF', dwt),
        ('DTCWT+AF', dtcwt),
        ('Statistical', stat),
        ('Hybrid V3.1', hybrid)
    ]:
        if method is None:
            # Baseline AF
            score_trend = TraditionalBaselines.access_frequency(ts_trending)
            score_stable = TraditionalBaselines.access_frequency(ts_stable)
            score_bursty = TraditionalBaselines.access_frequency(ts_bursty)
        else:
            score_trend = method.assess_single(ts_trending)
            score_stable = method.assess_single(ts_stable)
            score_bursty = method.assess_single(ts_bursty)
        
        print(f"{method_name:<20} {score_trend:<12.2f} {score_stable:<12.2f} {score_bursty:<12.2f}")
    
    print("\n" + "="*70 + "\n")


def demo_batch_assessment():
    """Demonstrate batch assessment with ranking"""
    print("\n" + "="*70)
    print("DEMO 2: Batch Assessment and Ranking")
    print("="*70 + "\n")
    
    # Generate synthetic data
    print("Generating 100 synthetic time series...")
    ts_dict = generate_synthetic_time_series(n_items=100, length=30)
    print(f"✓ Generated {len(ts_dict)} time series\n")
    
    # Convert to list
    items = list(ts_dict.keys())
    ts_list = list(ts_dict.values())
    
    # Initialize methods
    print("Assessing with Hybrid V3.1 method...")
    hybrid = HybridAssessment(enable_cache=True)
    
    # Batch assess
    scores = hybrid.batch_assess(ts_list, item_ids=items)
    print(f"✓ Computed {len(scores)} popularity scores\n")
    
    # Rank items
    sorted_indices = np.argsort(scores)[::-1]
    ranking = [items[i] for i in sorted_indices]
    
    # Show top-10
    print("Top 10 Most Popular Items:")
    print(f"{'Rank':<6} {'Item ID':<12} {'Score':<12} {'Pattern':<15}")
    print("-" * 50)
    
    for rank, idx in enumerate(sorted_indices[:10], 1):
        item_id = items[idx]
        score = scores[idx]
        
        # Determine pattern
        ts = ts_list[idx]
        trend = np.polyfit(range(len(ts)), ts, 1)[0]
        if trend > 1:
            pattern = "Trending ↗"
        elif trend < -1:
            pattern = "Declining ↘"
        else:
            pattern = "Stable →"
        
        print(f"{rank:<6} {item_id:<12} {score:<12.2f} {pattern:<15}")
    
    print("\n" + "="*70 + "\n")


def demo_component_analysis():
    """Demonstrate component analysis of hybrid method"""
    print("\n" + "="*70)
    print("DEMO 3: Hybrid Method Component Analysis")
    print("="*70 + "\n")
    
    # Create a sample time series
    ts = np.array([10, 12, 15, 18, 22, 28, 35, 42, 50, 58,
                   65, 72, 78, 83, 87, 90, 92, 93, 94, 95,
                   95, 95, 94, 93, 92, 90, 88, 85, 82, 80])
    
    print("Sample time series: Rising then plateauing pattern")
    print(f"Values: {ts[:10]}... (30 points total)\n")
    
    # Initialize hybrid method
    hybrid = HybridAssessment()
    
    # Analyze components
    print("Analyzing components...")
    components = hybrid.analyze_components(ts)
    
    print("\nComponent Breakdown:")
    print(f"{'Component':<25} {'Contribution':<15}")
    print("-" * 40)
    
    for component, value in components.items():
        print(f"{component:<25} {value:<15.4f}")
    
    print("\nInterpretation:")
    print(f"- DTCWT captures multi-scale patterns")
    print(f"- Skewness detects trend direction")
    print(f"- Kurtosis identifies burstiness")
    print(f"- Entropy measures complexity")
    print(f"- Hurst indicates persistence")
    print(f"- Total score reflects overall popularity")
    
    print("\n" + "="*70 + "\n")


def demo_comparison():
    """Compare all methods side by side"""
    print("\n" + "="*70)
    print("DEMO 4: Method Comparison on Diverse Patterns")
    print("="*70 + "\n")
    
    # Create diverse time series
    patterns = {
        'Linear Growth': np.linspace(10, 100, 30),
        'Exponential': np.exp(np.linspace(0, 3, 30)),
        'Periodic': 50 + 30 * np.sin(np.linspace(0, 4*np.pi, 30)),
        'Random Walk': np.cumsum(np.random.randn(30)) + 50,
        'Step Function': np.concatenate([np.ones(15)*20, np.ones(15)*80]),
    }
    
    # Initialize all methods
    methods = {
        'AF': None,
        'LFU': None,
        'EWMA': None,
        'DWT+AF': DWTAssessment(),
        'DTCWT+AF': DTCWTAssessment(),
        'Statistical': StatisticalAssessment(),
        'Hybrid': HybridAssessment(),
    }
    
    # Compute scores
    results = {pattern: {} for pattern in patterns.keys()}
    
    for pattern_name, ts in patterns.items():
        ts = np.maximum(0, ts)  # Ensure non-negative
        
        for method_name, method in methods.items():
            if method is None:
                if method_name == 'AF':
                    score = TraditionalBaselines.access_frequency(ts)
                elif method_name == 'LFU':
                    score = TraditionalBaselines.lfu_score(ts)
                else:  # EWMA
                    score = TraditionalBaselines.ewma_score(ts)
            else:
                score = method.assess_single(ts)
            
            results[pattern_name][method_name] = score
    
    # Display results
    print(f"{'Pattern':<20} " + " ".join([f"{m:<12}" for m in methods.keys()]))
    print("-" * (20 + 13*len(methods)))
    
    for pattern_name, scores in results.items():
        row = f"{pattern_name:<20}"
        for method_name in methods.keys():
            row += f" {scores[method_name]:<12.2f}"
        print(row)
    
    print("\nObservations:")
    print("- Hybrid method consistently provides balanced scores")
    print("- DTCWT captures complex patterns better than DWT")
    print("- Statistical method sensitive to distribution shape")
    print("- Baselines (AF, LFU) are simple but less discriminative")
    
    print("\n" + "="*70 + "\n")


def main():
    """Run all demos"""
    print("\n" + "🎯 "*20)
    print("DTCWT-based Popularity Assessment System - Demo")
    print("Version 3.1 - PhD Research Implementation")
    print("🎯 "*20)
    
    try:
        demo_basic_assessment()
        demo_batch_assessment()
        demo_component_analysis()
        demo_comparison()
        
        print("\n✅ All demos completed successfully!")
        print("\nNext steps:")
        print("1. Prepare your dataset (see README.md)")
        print("2. Run full experiment: python experiments/exp1_assessment_comparison.py")
        print("3. Customize parameters in config.py")
        print("4. Analyze results in results/tables/")
        
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
