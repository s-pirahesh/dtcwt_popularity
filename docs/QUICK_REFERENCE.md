# Quick Reference — Frozen 4-Layer Evaluation Protocol

## 🎯 ساختار ذخیره‌سازی

| دایرکتوری | فرمت | محتوا | حجم تقریبی |
|-----------|------|-------|------------|
| `detailed/` | Parquet | امتیازهای تفصیلی per-item | ~70 MB/method |
| `summary/` | Parquet | خلاصه per-stratum | ~1 MB/method |
| `protocol/` | CSV/Parquet | **متریک‌های 4-Layer per-window** | ~5 MB/method |
| `comparison/` | CSV | مقایسه نهایی روش‌ها | ~10 KB |
| `metadata/` | JSON | پیکربندی و آمار اجرا | ~5 KB |

---

## 🚀 دستورات اصلی

### ۱. اجرای ارزیابی
```bash
# تست سریع (100 آیتم، حافظه‌کارآمد)
python experiments/run_popularity_assessment.py movielens \
    --num-items 100 \
    --start-date 2023-08-01 \
    --end-date 2023-08-31 \
    --incremental

# اجرای کامل
python experiments/run_popularity_assessment.py movielens \
    --num-items 1000 \
    --start-date 2023-01-01 \
    --end-date 2023-12-31 \
    --incremental

# K values سفارشی (پیش‌فرض: 5 10 20)
python experiments/run_popularity_assessment.py movielens \
    --num-items 500 \
    --k-list 5 10 20 50

# فقط روش‌های خاص
python experiments/run_popularity_assessment.py movielens \
    --num-items 500 \
    --methods WSPI AF DTCWT+AF
```

### ۲. تحلیل نتایج
```bash
# لیست run های موجود
python experiments/analyze_results.py --list movielens

# نمایش متریک‌های از پیش محاسبه‌شده (پیش‌فرض، سریع)
python experiments/analyze_results.py RESULTS_PATH

# بازمحاسبه کامل 4-Layer از raw scores
python experiments/analyze_results.py RESULTS_PATH --recompute

# بازمحاسبه + ذخیره در protocol/
python experiments/analyze_results.py RESULTS_PATH --recompute --save-recomputed

# با فیلتر stratum
python experiments/analyze_results.py RESULTS_PATH \
    --recompute --stratum high --mode detailed

# نمایش با فیلتر تاریخ
python experiments/analyze_results.py RESULTS_PATH \
    --recompute \
    --start-date 2023-03-01 \
    --end-date 2023-06-30
```

### ۳. نمایش نتایج
```bash
# نمایش متنی (فقط از protocol files)
python experiments/show_results.py RESULTS_PATH

# نمودارها
python experiments/show_results.py RESULTS_PATH --graphical --show

# هر دو + آمار تفصیلی
python experiments/show_results.py RESULTS_PATH --both --detailed
```

---

## 📊 متریک‌های 4-Layer Protocol

### Layer 1 — Decision (کیفیت رتبه‌بندی)
| متریک | توضیح | بهتر |
|-------|-------|------|
| `ndcg@5/10/20` | Normalized DCG با log-relevance | بالاتر |
| `chr@5/10/20` | Cache Hit Ratio (static placement) | بالاتر |

### Layer 2 — Diagnostic (آمار کلی)
| متریک | توضیح | بهتر |
|-------|-------|------|
| `spearman_rho` | همبستگی رتبه‌ای یکنواخت | بالاتر |
| `kendall_tau` | همبستگی جفت‌های مرتب | بالاتر |
| `mae` | میانگین خطای مطلق (برای baselineها) | پایین‌تر |

### Layer 3 — Stability (پایداری زمانی)
| متریک | توضیح | بهتر |
|-------|-------|------|
| `rsi@5/10/20` | Ranking Stability Index (Jaccard) | بالاتر |

### Layer 4 — Robustness (مقاومت در برابر نویز)
| متریک | توضیح | بهتر |
|-------|-------|------|
| `robustness_distortion` | میانگین ΔRank پس از تزریق 10× نویز | پایین‌تر |

---

## 🔬 Assessment Methods (Chapter 3)

| Name | Section | Category | Window | min_obs | Formula |
|------|---------|----------|--------|---------|---------|
| `AF` | — | Baseline | 7 days | 3 | count sum |
| `LFU` | — | Baseline | 7 days | 3 | frequency rank |
| `LRU` | — | Baseline | 7 days | 3 | recency score |
| `EWMA` | — | Baseline | 7 days | 3 | α=0.3 exponential smooth |
| `DWT+AF` | 3-2 | Trend-Shock Model | 64 days | 32 | WAF(cA_L) + β·WAF(cD_1) |
| `DTCWT+AF` | 3-3 | Stable DTCWT Model | 64 days | 32 | WAF(M_trend) + β·WAF(M_shock) |
| **`WSPI`** | **3-4** | **Proposed** | 64 days | 32 | μ_L·exp(clip(α·S_L + β·R − γ·WE, −3, 3)) |

**WSPI Frozen Parameters (Section 3-4):**
```python
HybridAssessment(alpha_slope=1.0, beta_ratio=0.5, gamma_entropy=0.5)
# alpha=1.0  (trend slope weight)
# beta=0.5   (energy-ratio weight)
# gamma=0.5  (wavelet entropy penalty — disorder)
```

> `Statistical` (skewness/kurtosis) removed — not part of Chapter 3 framework.

---

## 📁 ساختار خروجی کامل

```
results/movielens/RUN_NAME/
├── detailed/
│   ├── AF_scores.parquet
│   ├── DTCWT+AF_scores.parquet
│   ├── WSPI_scores.parquet
│   └── ...
├── summary/
│   ├── AF_stratum_summary.parquet
│   └── ...
├── protocol/                       ← جدید: 4-Layer per-window
│   ├── AF_protocol.csv
│   ├── DTCWT+AF_protocol.csv
│   ├── WSPI_protocol.csv
│   └── ...
├── comparison/
│   └── method_comparison.csv
├── metadata/
│   ├── config.json                 ← شامل k_list, robustness_*
│   ├── thresholds.json
│   └── runtime_stats.json
└── visualization/
    └── *.png
```

---

## 💻 Python API

```python
from evaluation import (
    ResultsAnalyzer,
    calculate_ndcg, calculate_hit_rate, calculate_rsi,
    calculate_rank_distortion, calculate_diagnostics,
    RobustnessScenario
)

# بارگذاری analyzer
analyzer = ResultsAnalyzer('results/movielens/RUN_NAME')

# خواندن protocol metrics از پیش محاسبه‌شده
proto_df = analyzer.load_protocol_metrics('WSPI')

# بازمحاسبه 4-Layer از raw scores
recomp_df = analyzer.recompute_protocol_metrics('WSPI',
    filter_stratum='high', start_date='2023-01-01')

# مقایسه همه روش‌ها
comparison = analyzer.compare_methods(recompute=True)

# تکامل زمانی NDCG@10
evo = analyzer.get_temporal_evolution('WSPI', metric='ndcg@10')

# مستقیم از توابع metrics
import numpy as np
scores  = np.array([...])
actuals = np.array([...])

ndcg   = calculate_ndcg(scores, actuals, k=10)
chr10  = calculate_hit_rate(scores, actuals, k=10)
diag   = calculate_diagnostics(scores, actuals)
# diag = {'kendall_tau': ..., 'spearman_rho': ..., 'mae': ...}
```

---

## ⚙️ پیکربندی (EvaluationConfig)

```python
from evaluation import get_movielens_config

config = get_movielens_config(
    num_items=1000,
    window_size=30,
    # Frozen Evaluation Protocol parameters:
    k_list=[5, 10, 20],          # K برای NDCG/CHR/RSI
    robustness_sample_size=50,   # تعداد آیتم‌های تست robustness
    spike_multiplier=10.0,       # بزرگی نویز (10× میانگین)
)
```

---

## 🔄 جریان کار (workflow)

```
1. [run_popularity_assessment.py]  →  detailed/ + summary/ + protocol/
         ↓ (یک بار، زمان‌بر)
2. [analyze_results.py]
   --recompute  →  بازمحاسبه 4-Layer از raw  →  نمایش متنی
   (default)    →  خواندن protocol/ مستقیم  →  نمایش متنی
         ↓ (چندبار، سریع)
3. [show_results.py]              →  نمایش متنی + نمودار
         ↓ (فقط display)
```

---

## ⏱️ زمان اجرای تقریبی

| مرحله | برنامه | زمان |
|-------|--------|------|
| ارزیابی ۱۰۰۰ آیتم، ۱ سال | `run_popularity_assessment.py` | ~10 ساعت |
| تحلیل (display-only) | `analyze_results.py` | ~3 ثانیه |
| تحلیل (recompute) | `analyze_results.py --recompute` | ~30 ثانیه |
| نمایش | `show_results.py` | ~5 ثانیه |

**یک بار محاسبه، بارها تحلیل!** 🎯
