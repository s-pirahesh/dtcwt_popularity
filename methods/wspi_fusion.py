"""
WSPI Fusion Candidates (variance-equalized, new methods — NOT replacements)
===========================================================================
Motivated by the formal analysis (formal_analysis_formulation.md):

    rank(WSPI) == rank( log(mu_L) + alpha*S_L + beta*R - gamma*WE )

i.e. the multiplicative-exp form is an ADDITIVE model in log-space. Because
popularity magnitude mu_L is heavy-tailed, std(log mu_L) ~ 2.2 dominates the
bounded structural terms (std ~ 0.04-0.1), so structure controls only ~0.2%
of the ranking. The cure is VARIANCE EQUALIZATION before combining.

These candidates equalize the scales with GLOBAL normalizers (calibrated once
on the dataset), so the existing per-item assess_single(ts)->float API and the
evaluation pipeline are untouched. Each feature contributes on a comparable
scale, and lam controls how much the structure is allowed to matter.

Two fusion mechanisms:
  - mode='z'        : score = N(log mu_L) + lam*( a*N(rho1) + b*N(R) - c*N(WE) )
                      where N(x) = (x - mean_g)/std_g          (z-standardization)
  - mode='quantile' : same, but N(x) = global empirical CDF of x in [0,1]
                      (rank/quantile fusion; fully scale-free, distribution-free)

Feature set is selected by use_rho1:
  - use_rho1=False -> {R, WE}            (coefficients a unused; named b, c)
  - use_rho1=True  -> {rho1, R, WE}      (coefficients a, b, c, IN ORDER)

rho1 = lag-1 autocorrelation of the RAW windowed series (time-domain
persistence; high -> smooth/persistent, low/negative -> noisy/mean-reverting).
It complements R and WE, which live in the wavelet (frequency) domain.

IMPORTANT (read before interpreting results): the goal is NOT to maximise the
structure's share. Pushing structure too high (large lam) re-ranks by structure
and DESTROYS correlation with actual popularity. The target is a small-to-mid
lam that improves RSI / dRank without wrecking NDCG / Spearman. That is why
both a full (lam=1.0) and a soft (lam=0.3) variant are provided.

Author: Sajjad (with assistance)
"""
from typing import Iterable, List, Dict, Optional
import numpy as np

from methods.wspi_candidates import _CandidateBase


def _lag1_autocorr(ts: np.ndarray) -> float:
    """Lag-1 autocorrelation of a raw series; 0.0 for degenerate input."""
    x = np.asarray(ts, dtype=np.float64).ravel()
    if x.size < 3:
        return 0.0
    x0, x1 = x[:-1], x[1:]
    s0, s1 = np.std(x0), np.std(x1)
    if s0 < 1e-12 or s1 < 1e-12:      # flat series -> undefined; neutral
        return 0.0
    return float(np.clip(np.corrcoef(x0, x1)[0, 1], -1.0, 1.0))


class _Normalizer:
    """Global feature normalizer, calibrated once. Supports z and quantile."""

    def __init__(self, mode: str = 'z'):
        assert mode in ('z', 'quantile')
        self.mode = mode
        self.mean: Dict[str, float] = {}
        self.std:  Dict[str, float] = {}
        self.sorted: Dict[str, np.ndarray] = {}

    def fit(self, feats: Dict[str, np.ndarray]):
        for k, v in feats.items():
            v = np.asarray(v, dtype=np.float64)
            v = v[np.isfinite(v)]
            if v.size == 0:
                v = np.array([0.0, 1.0])
            self.mean[k] = float(np.mean(v))
            self.std[k]  = float(np.std(v)) or 1.0
            self.sorted[k] = np.sort(v)
        return self

    def apply(self, key: str, x: float) -> float:
        if self.mode == 'z':
            return (x - self.mean[key]) / self.std[key]
        # quantile -> empirical CDF position in [0, 1], centred to [-0.5, 0.5]
        arr = self.sorted[key]
        pos = float(np.searchsorted(arr, x, side='right')) / max(len(arr), 1)
        return pos - 0.5


# Feature keys
_KEYS = ('logmu', 'rho1', 'R', 'WE')


class WSPIFusion(_CandidateBase):
    """
    Variance-equalized additive fusion of magnitude and structure.

    Parameters
    ----------
    mode      : 'z' or 'quantile'
    use_rho1  : include the lag-1 autocorrelation persistence term
    lam       : overall weight on the (signed) structural block
    a, b, c   : per-term weights for rho1, R, WE respectively (IN ORDER)
    normalizer: a fitted _Normalizer (shared across candidates); if None the
                method falls back to raw features (NOT recommended).
    """

    def __init__(self, mode: str = 'z', use_rho1: bool = False,
                 lam: float = 1.0, a: float = 1.0, b: float = 1.0,
                 c: float = 1.0, normalizer: Optional[_Normalizer] = None,
                 name: str = 'WSPI-FUSE'):
        super().__init__(alpha_slope=0.0, beta_ratio=0.0, gamma_entropy=0.0)
        self.mode = mode
        self.use_rho1 = use_rho1
        self.lam = lam
        self.a, self.b, self.c = a, b, c
        self.norm = normalizer
        self.name = name

    # -- raw feature vector for one series -------------------------------
    def _raw_features(self, ts: np.ndarray) -> Dict[str, float]:
        mu_L, _Sm, R, WE = self._extract_features(ts)
        logmu = float(np.log(mu_L + 1e-9))
        return {'logmu': logmu, 'rho1': _lag1_autocorr(ts), 'R': R, 'WE': WE}

    # -- calibration: build a shared normalizer from many series ---------
    @staticmethod
    def calibrate(series_iter: Iterable[np.ndarray], mode: str,
                  probe: Optional['WSPIFusion'] = None,
                  max_items: int = 2000) -> _Normalizer:
        ex = probe or WSPIFusion(mode=mode)
        buf = {k: [] for k in _KEYS}
        n = 0
        for ts in series_iter:
            ts = np.asarray(ts, dtype=np.float64).ravel()
            if ts.size < 32:                  # same min_observations as WSPI
                continue
            try:
                f = ex._raw_features(ts)
            except Exception:
                continue
            for k in _KEYS:
                buf[k].append(f[k])
            n += 1
            if n >= max_items:
                break
        feats = {k: np.array(v) for k, v in buf.items()}
        return _Normalizer(mode).fit(feats)

    # -- scoring ---------------------------------------------------------
    def _fuse(self, mu_L, Sm, R, WE):     # not used; we override assess_single
        raise NotImplementedError

    def assess_single(self, time_series: np.ndarray) -> float:
        if len(time_series) == 0:
            return 0.0
        try:
            f = self._raw_features(time_series)
            if self.norm is None:
                struct = (self.b * f['R'] - self.c * f['WE']
                          + (self.a * f['rho1'] if self.use_rho1 else 0.0))
                return float(f['logmu'] + self.lam * struct)
            N = self.norm.apply
            base = N('logmu', f['logmu'])
            struct = self.b * N('R', f['R']) - self.c * N('WE', f['WE'])
            if self.use_rho1:
                struct += self.a * N('rho1', f['rho1'])
            return float(base + self.lam * struct)
        except Exception:
            # Keep a usable magnitude ordering even on failure
            try:
                return float(np.log(np.mean(time_series) + 1e-9))
            except Exception:
                return 0.0
