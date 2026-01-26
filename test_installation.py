#!/usr/bin/env python3
"""
Test Script: Verify DTCWT Popularity Assessment Installation
Quick smoke test to ensure all modules work correctly
"""
import sys
import numpy as np
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    
    try:
        import config
        print("  ✓ config")
        
        from data.loaders import BaseDataLoader
        print("  ✓ data.loaders.BaseDataLoader")
        
        from data.preprocessors.time_series import TimeSeriesPreprocessor
        print("  ✓ data.preprocessors.TimeSeriesPreprocessor")
        
        from methods.dwt_assessment import DWTAssessment
        print("  ✓ methods.dwt_assessment")
        
        from methods.dtcwt_assessment import DTCWTAssessment
        print("  ✓ methods.dtcwt_assessment")
        
        from methods.statistical_assessment import StatisticalAssessment
        print("  ✓ methods.statistical_assessment")
        
        from methods.advanced_features import AdvancedFeatures
        print("  ✓ methods.advanced_features")
        
        from methods.hybrid_assessment import HybridAssessment
        print("  ✓ methods.hybrid_assessment")
        
        from baselines.traditional import TraditionalBaselines
        print("  ✓ baselines.traditional")
        
        from evaluation.metrics import AssessmentMetrics
        print("  ✓ evaluation.metrics")
        
        from utils.feature_cache import FeatureCache
        print("  ✓ utils.feature_cache")
        
        print("\n✅ All imports successful!\n")
        return True
    
    except ImportError as e:
        print(f"\n❌ Import failed: {e}\n")
        return False


def test_basic_functionality():
    """Test basic functionality of each method"""
    print("Testing basic functionality...")
    
    # Create a simple time series
    ts = np.array([10, 12, 15, 18, 22, 25, 30, 35, 40, 45,
                   50, 55, 60, 65, 70, 75, 80, 85, 90, 95,
                   100, 105, 110, 115, 120, 125, 130, 135, 140, 145])
    
    try:
        # Test DWT
        from methods.dwt_assessment import DWTAssessment
        dwt = DWTAssessment()
        score = dwt.assess_single(ts)
        assert isinstance(score, (int, float)), "DWT score should be numeric"
        print(f"  ✓ DWT: score = {score:.2f}")
        
        # Test DTCWT
        from methods.dtcwt_assessment import DTCWTAssessment
        dtcwt = DTCWTAssessment()
        score = dtcwt.assess_single(ts)
        assert isinstance(score, (int, float)), "DTCWT score should be numeric"
        print(f"  ✓ DTCWT: score = {score:.2f}")
        
        # Test Statistical
        from methods.statistical_assessment import StatisticalAssessment
        stat = StatisticalAssessment()
        score = stat.assess_single(ts)
        assert isinstance(score, (int, float)), "Statistical score should be numeric"
        print(f"  ✓ Statistical: score = {score:.2f}")
        
        # Test Advanced Features
        from methods.advanced_features import AdvancedFeatures
        features = AdvancedFeatures.compute_all_features(ts)
        assert 'shannon_entropy' in features, "Should compute Shannon entropy"
        assert 'hurst_exponent' in features, "Should compute Hurst exponent"
        print(f"  ✓ Advanced Features: entropy = {features['shannon_entropy']:.3f}, hurst = {features['hurst_exponent']:.3f}")
        
        # Test Hybrid
        from methods.hybrid_assessment import HybridAssessment
        hybrid = HybridAssessment()
        score = hybrid.assess_single(ts)
        assert isinstance(score, (int, float)), "Hybrid score should be numeric"
        print(f"  ✓ Hybrid: score = {score:.2f}")
        
        # Test Baselines
        from baselines.traditional import TraditionalBaselines
        af_score = TraditionalBaselines.access_frequency(ts)
        assert isinstance(af_score, (int, float)), "AF score should be numeric"
        print(f"  ✓ Baselines: AF = {af_score:.2f}")
        
        # Test Preprocessor
        from data.preprocessors.time_series import TimeSeriesPreprocessor
        normalized = TimeSeriesPreprocessor.normalize(ts, method='zscore')
        assert len(normalized) == len(ts), "Normalized series should have same length"
        print(f"  ✓ Preprocessor: normalized shape = {normalized.shape}")
        
        # Test Caching
        from utils.feature_cache import FeatureCache
        cache = FeatureCache(memory_size=100)
        cache.put('test_key', {'value': 42})
        cached_value = cache.get('test_key')
        assert cached_value['value'] == 42, "Cache should retrieve stored value"
        print(f"  ✓ Cache: stored and retrieved value")
        
        print("\n✅ All functionality tests passed!\n")
        return True
    
    except Exception as e:
        print(f"\n❌ Functionality test failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_batch_processing():
    """Test batch processing capabilities"""
    print("Testing batch processing...")
    
    try:
        # Create multiple time series
        n_series = 10
        ts_list = [np.random.poisson(20, 30) + np.arange(30) for _ in range(n_series)]
        
        from methods.hybrid_assessment import HybridAssessment
        hybrid = HybridAssessment()
        
        # Batch assessment
        scores = hybrid.batch_assess(ts_list)
        
        assert len(scores) == n_series, f"Should return {n_series} scores"
        assert all(isinstance(s, (int, float)) for s in scores), "All scores should be numeric"
        
        print(f"  ✓ Batch processed {n_series} time series")
        print(f"    Mean score: {np.mean(scores):.2f}")
        print(f"    Std score: {np.std(scores):.2f}")
        
        print("\n✅ Batch processing test passed!\n")
        return True
    
    except Exception as e:
        print(f"\n❌ Batch processing test failed: {e}\n")
        return False


def test_dependencies():
    """Test that all required dependencies are installed"""
    print("Testing dependencies...")
    
    required_packages = [
        ('numpy', 'numpy'),
        ('scipy', 'scipy'),
        ('pandas', 'pandas'),
        ('pywt', 'PyWavelets'),
        ('dtcwt', 'dtcwt'),
        ('sklearn', 'scikit-learn'),
        ('matplotlib', 'matplotlib'),
    ]
    
    all_present = True
    
    for module_name, package_name in required_packages:
        try:
            __import__(module_name)
            print(f"  ✓ {package_name}")
        except ImportError:
            print(f"  ✗ {package_name} - MISSING")
            all_present = False
    
    if all_present:
        print("\n✅ All dependencies installed!\n")
    else:
        print("\n⚠️  Some dependencies missing. Run: pip install -r requirements.txt\n")
    
    return all_present


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("DTCWT Popularity Assessment - Installation Test")
    print("Version 3.1")
    print("="*70 + "\n")
    
    tests = [
        ("Dependencies", test_dependencies),
        ("Imports", test_imports),
        ("Basic Functionality", test_basic_functionality),
        ("Batch Processing", test_batch_processing),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"{'='*70}")
        print(f"Test: {test_name}")
        print(f"{'='*70}\n")
        
        result = test_func()
        results.append((test_name, result))
    
    # Summary
    print("="*70)
    print("TEST SUMMARY")
    print("="*70 + "\n")
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:<30} {status}")
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        print("\n" + "🎉"*20)
        print("ALL TESTS PASSED! System is ready to use.")
        print("🎉"*20 + "\n")
        print("Next steps:")
        print("1. Run demo: python demo.py")
        print("2. Run experiments: python experiments/exp1_assessment_comparison.py")
        print("3. Check README.md for detailed usage instructions")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        print("   Make sure all dependencies are installed: pip install -r requirements.txt")
    
    print()


if __name__ == '__main__':
    main()
