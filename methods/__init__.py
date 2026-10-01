# -*- coding: utf-8 -*-
"""
Methods Package
Popularity estimation methods

Per-item classes (depending on the installed dependencies):
- DTCWTAssessment: DTCWT+AF of the first submission (requires: dtcwt)
- DWTAssessment: DWT+AF of the first submission (requires: pywt)
- HybridAssessment: earlier WSPI formula with slope and clip (not used)
- StatisticalAssessment: early statistical score (not used)

The WSPI index of the paper is ``methods/wspi_assessment.WSPIAssessment``.
The results of the paper come from the batched scorers of
``evaluation/protocol_v5.py``.

Author: Sajjad Pirahesh
Date: February 2025
"""

import warnings

# List of the available methods
__all__ = []

# Import DWTAssessment (needs pywt)
try:
    from .dwt_assessment import DWTAssessment
    __all__.append('DWTAssessment')
except ImportError as e:
    DWTAssessment = None
    warnings.warn(f"DWTAssessment not available: {e}", ImportWarning)

# Import DTCWTAssessment (needs dtcwt)
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

# Show the available methods
if __all__:
    print(f"✓ Loaded {len(__all__)} methods: {', '.join(__all__)}")
else:
    print("⚠️  No methods loaded. Install dependencies: pip install pywt dtcwt")
