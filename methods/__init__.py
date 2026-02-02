# -*- coding: utf-8 -*-
"""
Methods Package
روش‌های مختلف تخمین محبوبیت

Available methods (بر اساس dependencies نصب شده):
- DTCWTAssessment: Dual-Tree Complex Wavelet Transform (requires: dtcwt)
- DWTAssessment: Discrete Wavelet Transform (requires: pywt)
- HybridAssessment: Hybrid approaches
- StatisticalAssessment: Statistical features

Author: Sajjad
Date: February 2025
"""

import warnings

# لیست methods موجود
__all__ = []

# Import DWTAssessment (نیاز به pywt)
try:
    from .dwt_assessment import DWTAssessment
    __all__.append('DWTAssessment')
except ImportError as e:
    DWTAssessment = None
    warnings.warn(f"DWTAssessment not available: {e}", ImportWarning)

# Import DTCWTAssessment (نیاز به dtcwt)
try:
    from .dtcwt_assessment import DTCWTAssessment
    __all__.append('DTCWTAssessment')
except ImportError as e:
    DTCWTAssessment = None
    warnings.warn(f"DTCWTAssessment not available: {e}", ImportWarning)

# Import HybridAssessment
try:
    from .hybrid_assessment import HybridAssessment
    __all__.append('HybridAssessment')
except ImportError as e:
    HybridAssessment = None
    warnings.warn(f"HybridAssessment not available: {e}", ImportWarning)

# Import StatisticalAssessment
try:
    from .statistical_assessment import StatisticalAssessment
    __all__.append('StatisticalAssessment')
except ImportError as e:
    StatisticalAssessment = None
    warnings.warn(f"StatisticalAssessment not available: {e}", ImportWarning)

__version__ = '1.0.0'

# نمایش methods موجود
if __all__:
    print(f"✓ Loaded {len(__all__)} methods: {', '.join(__all__)}")
else:
    print("⚠️  No methods loaded. Install dependencies: pip install pywt dtcwt")
