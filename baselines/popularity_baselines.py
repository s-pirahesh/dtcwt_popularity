"""
Popularity Baseline Wrappers
============================
Wraps TraditionalBaselines static methods as BaseMethod-compatible objects.

Methods provided:
  - AFMethod          : Weighted Access Frequency
  - EWMAMethod        : Exponentially Weighted Moving Average
  - RRDMethod         : Requests / Lifetime ratio
  - VSEMethod         : Volume + Recency combined
  - CompoundPopMethod : Three-factor compound popularity
  - PFRFMethod        : Period-based popularity weighting (PFRF)

Author: Sajjad
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from methods.base_method import BaseMethod
from .traditional import TraditionalBaselines


class AFMethod(BaseMethod):
    """Weighted Access Frequency (AF). Ref: Chang & Chang (2008)."""
    def __init__(self):
        super().__init__(name='AF')

    def assess_single(self, time_series: np.ndarray) -> float:
        return TraditionalBaselines.access_frequency(time_series)


class EWMAMethod(BaseMethod):
    """Exponentially Weighted Moving Average. Ref: Gui & Chen (2020)."""
    def __init__(self, alpha: float = 0.2):
        super().__init__(name='EWMA')
        self.alpha = alpha

    def assess_single(self, time_series: np.ndarray) -> float:
        return TraditionalBaselines.ewma_score(time_series, alpha=self.alpha)


class RRDMethod(BaseMethod):
    """Requests / Lifetime ratio. Ref: Al Mistarihi & Yong (2008)."""
    def __init__(self):
        super().__init__(name='RRD')

    def assess_single(self, time_series: np.ndarray) -> float:
        return TraditionalBaselines.rrd_score(time_series)


class VSEMethod(BaseMethod):
    """Volume + Recency Score. Ref: Mansouri & Asadi (2014)."""
    def __init__(self):
        super().__init__(name='VSE')

    def assess_single(self, time_series: np.ndarray) -> float:
        return TraditionalBaselines.vse_score(time_series)


class CompoundPopMethod(BaseMethod):
    """Three-factor compound popularity. Ref: Ye et al. (2014)."""
    def __init__(self, cons1: float = 0.5, cons2: float = 0.3, cons3: float = 0.2):
        super().__init__(name='CompoundPop')
        self.cons1 = cons1
        self.cons2 = cons2
        self.cons3 = cons3

    def assess_single(self, time_series: np.ndarray) -> float:
        return TraditionalBaselines.compound_pop_score(
            time_series, self.cons1, self.cons2, self.cons3
        )


class PFRFMethod(BaseMethod):
    """Period-based popularity weighting. Ref: Lee et al. (2012)."""
    def __init__(self, a: float = 1.2, b: float = 0.8):
        super().__init__(name='PFRF')
        self.a = a
        self.b = b

    def assess_single(self, time_series: np.ndarray) -> float:
        return TraditionalBaselines.pfrf_score(time_series, self.a, self.b)


def get_all_baseline_methods() -> dict:
    """
    Return {method_name: method_instance} for all baseline methods.
    Ready to pass directly to the evaluation framework.

    Example:
        from baselines import get_all_baseline_methods
        from methods.hybrid_assessment import HybridAssessment
        from methods.dtcwt_assessment import DTCWTAssessment
        from methods.dwt_assessment import DWTAssessment

        methods = get_all_baseline_methods()
        methods.update({
            'DWT+AF':   DWTAssessment(),
            'DTCWT+AF': DTCWTAssessment(),
            'WSPI':     HybridAssessment(),
        })
        evaluator = TemporalEvaluator(loader, methods, config)
    """
    return {
        'AF':          AFMethod(),
        'EWMA':        EWMAMethod(),
        'RRD':         RRDMethod(),
        'VSE':         VSEMethod(),
        'CompoundPop': CompoundPopMethod(),
        'PFRF':        PFRFMethod(),
    }
