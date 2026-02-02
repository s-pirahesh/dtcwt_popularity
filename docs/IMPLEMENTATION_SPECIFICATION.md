# دستورالعمل کامل پیاده‌سازی سیستم ارزیابی محبوبیت
# Complete Implementation Specification for Popularity Assessment System

**نسخه:** 1.0
**تاریخ:** 2 فوریه 2025
**نویسنده:** Sajjad

---

## 📋 فهرست مطالب

1. [نمای کلی سیستم](#1-نمای-کلی-سیستم)
2. [معماری سیستم](#2-معماری-سیستم)
3. [ماژول‌های اصلی](#3-ماژول‌های-اصلی)
4. [ساختار فایل‌ها](#4-ساختار-فایل‌ها)
5. [جریان داده](#5-جریان-داده)
6. [مشخصات فنی](#6-مشخصات-فنی)
7. [استانداردهای کدنویسی](#7-استانداردهای-کدنویسی)
8. [تست و اعتبارسنجی](#8-تست-و-اعتبارسنجی)

---

## 1. نمای کلی سیستم

### 1.1 هدف
سیستم ارزیابی و تخمین محبوبیت محتوا در سیستم‌های توزیع شده با استفاده از روش‌های مختلف شامل Graph Signal Processing و Wavelet Analysis.

### 1.2 قابلیت‌های اصلی
- ✅ محاسبه امتیاز محبوبیت با روش‌های مختلف
- ✅ ارزیابی زمانی با Sliding Window
- ✅ ذخیره نتایج با timestamp منحصر به فرد
- ✅ فیلتر تاریخ برای انتخاب بازه زمانی
- ✅ تحلیل نتایج بدون شبیه‌سازی مجدد
- ✅ نمایش متنی و گرافیکی نتایج

### 1.3 دیتاست‌های پشتیبانی شده
- MovieLens 25M
- YouTube
- Youku

---

## 2. معماری سیستم

### 2.1 معماری کلی (3-Tier)

```
┌─────────────────────────────────────────────────┐
│          Presentation Layer (نمایش)             │
│  - show_results.py (متنی + گرافیکی)            │
│  - ResultsVisualizer (نمودارها)                │
└─────────────────────────────────────────────────┘
                        ⬇
┌─────────────────────────────────────────────────┐
│         Business Logic Layer (منطق)             │
│  - run_popularity_assessment.py (محاسبات)      │
│  - analyze_results.py (تحلیل)                  │
│  - ResultsAnalyzer (پردازش)                    │
└─────────────────────────────────────────────────┘
                        ⬇
┌─────────────────────────────────────────────────┐
│           Data Layer (داده)                     │
│  - Data Loaders (بارگذاری)                     │
│  - Storage System (ذخیره‌سازی)                 │
│  - Parquet/CSV Files                           │
└─────────────────────────────────────────────────┘
```

### 2.2 الگوهای طراحی (Design Patterns)

1. **Strategy Pattern**: روش‌های مختلف محاسبه محبوبیت
2. **Factory Pattern**: ایجاد data loaders
3. **Observer Pattern**: گزارش پیشرفت
4. **Template Method**: جریان ارزیابی
5. **Singleton Pattern**: پیکربندی سیستم

---

## 3. ماژول‌های اصلی

### 3.1 ماژول Data Loading

**مسئولیت:** بارگذاری و پیش‌پردازش دیتاست‌ها

**فایل‌ها:**
```
data/loaders/
├── __init__.py
├── base_loader.py          # کلاس پایه
├── movielens_loader.py     # MovieLens
├── youtube_loader.py       # YouTube
└── youku_loader.py         # Youku
```

**کلاس‌های کلیدی:**
- `BaseDataLoader`: کلاس انتزاعی پایه
- `MovieLensLoader`: پیاده‌سازی برای MovieLens
- `YouTubeLoader`: پیاده‌سازی برای YouTube
- `YoukuLoader`: پیاده‌سازی برای Youku

**قراردادها:**
```python
class BaseDataLoader:
    def load_data(self) -> pd.DataFrame
    def filter_by_date(self, start_date, end_date) -> pd.DataFrame
    def get_date_range(self) -> Tuple[datetime, datetime]
    def aggregate_by_day(self) -> pd.DataFrame
```

---

### 3.2 ماژول Popularity Methods

**مسئولیت:** پیاده‌سازی روش‌های تخمین محبوبیت

**فایل‌ها:**
```
methods/
├── __init__.py
├── base_method.py          # کلاس پایه
├── access_frequency.py     # AF
├── lfu.py                  # LFU
├── lru.py                  # LRU
├── ewma.py                 # EWMA
├── wavelet_methods.py      # DWT + DTCWT
├── statistical.py          # Statistical features
└── hybrid.py               # Hybrid approaches
```

**کلاس‌های کلیدی:**
- `BasePopularityMethod`: کلاس پایه
- `AccessFrequency`: محاسبه بر اساس تعداد دسترسی
- `WaveletMethod`: روش‌های Wavelet-based
- `StatisticalMethod`: ویژگی‌های آماری

**قراردادها:**
```python
class BasePopularityMethod:
    def calculate_score(self, history: np.ndarray) -> float
    def get_name(self) -> str
    def validate_window_size(self, window_size: int) -> bool
```

---

### 3.3 ماژول Evaluation

**مسئولیت:** ارزیابی و محاسبه معیارها

**فایل‌ها:**
```
evaluation/
├── __init__.py
├── evaluation_config.py    # پیکربندی
├── temporal_evaluator.py   # ارزیابی زمانی
├── stratification.py       # طبقه‌بندی
├── metrics.py           # معیارهای ارزیابی
├── wavelet_validator.py    # اعتبارسنجی wavelet
├── storage.py              # ذخیره‌سازی
├── results_analyzer.py     # تحلیل نتایج
└── visualizer.py           # نمایش گرافیکی
```

**کلاس‌های کلیدی:**
- `EvaluationConfig`: پیکربندی ارزیابی
- `TemporalEvaluator`: اجرای ارزیابی زمانی
- `StratificationSystem`: طبقه‌بندی آیتم‌ها
- `MetricsCalculator`: محاسبه معیارها
- `StorageSystem`: ذخیره نتایج
- `ResultsAnalyzer`: تحلیل نتایج
- `ResultsVisualizer`: نمایش گرافیکی

---

### 3.4 ماژول Experiments

**مسئولیت:** اسکریپت‌های اجرایی

**فایل‌ها:**
```
experiments/
├── run_popularity_assessment.py   # محاسبات اصلی
├── analyze_results.py             # تحلیل نتایج
├── show_results.py                # نمایش نتایج
├── test_exp2_system.py            # تست سیستم
├── test_unified_storage.py        # تست ذخیره‌سازی
└── test_analysis_system.py        # تست تحلیل
```

---

## 4. ساختار فایل‌ها

### 4.1 ساختار کلی پروژه

```
dtcwt_popularity/
│
├── data/                          # داده‌ها و بارگذاری
│   ├── loaders/
│   │   ├── __init__.py
│   │   ├── base_loader.py
│   │   ├── movielens_loader.py
│   │   ├── youtube_loader.py
│   │   └── youku_loader.py
│   │
│   ├── raw/                       # داده‌های خام
│   └── processed/                 # داده‌های پردازش شده
│
├── methods/                       # روش‌های محاسبه محبوبیت
│   ├── __init__.py
│   ├── base_method.py
│   ├── access_frequency.py
│   ├── lfu.py
│   ├── lru.py
│   ├── ewma.py
│   ├── wavelet_methods.py
│   ├── statistical.py
│   └── hybrid.py
│
├── evaluation/                    # سیستم ارزیابی
│   ├── __init__.py
│   ├── evaluation_config.py
│   ├── temporal_evaluator.py
│   ├── stratification.py
│   ├── metrics.py
│   ├── wavelet_validator.py
│   ├── storage.py
│   ├── results_analyzer.py
│   └── visualizer.py
│
├── experiments/                   # اسکریپت‌های اجرایی
│   ├── run_popularity_assessment.py
│   ├── analyze_results.py
│   ├── show_results.py
│   ├── test_exp2_system.py
│   ├── test_unified_storage.py
│   └── test_analysis_system.py
│
├── results/                       # نتایج
│   └── {dataset}/
│       └── {run_name_timestamp}/
│           ├── detailed/
│           ├── summary/
│           ├── comparison/
│           ├── metadata/
│           └── visualization/
│
├── docs/                          # مستندات
│   ├── DATE_FILTERING_GUIDE.md
│   ├── WORKFLOW_DIAGRAM.md
│   ├── ANALYSIS_GUIDE.md
│   └── ...
│
├── requirements.txt               # وابستگی‌ها
└── README.md                      # راهنمای اصلی
```

### 4.2 نامگذاری فایل‌ها

**قوانین:**
- کلاس‌ها: `PascalCase` (مثل `BaseDataLoader`)
- فایل‌ها: `snake_case` (مثل `base_loader.py`)
- ثابت‌ها: `UPPER_CASE` (مثل `DEFAULT_WINDOW_SIZE`)
- متغیرها: `snake_case` (مثل `window_size`)

---

## 5. جریان داده

### 5.1 جریان اصلی محاسبات

```
Dataset → DataLoader → Filter by Date → Select Items → Stratification
                                                              ↓
                                                        Sliding Window
                                                              ↓
                                            Calculate Popularity (all methods)
                                                              ↓
                                            Calculate Metrics (per window)
                                                              ↓
                            StorageSystem (Parquet + CSV/Parquet + JSON)
                                                              ↓
                                            Results with Timestamp
```

### 5.2 جریان تحلیل

```
Results Directory → ResultsAnalyzer → Load (Parquet/CSV)
                                              ↓
                                        Apply Filters
                                              ↓
                                    Calculate Metrics
                                              ↓
                                    Compare Methods
                                              ↓
                            TextualOutput + GraphicalOutput
```

### 5.3 جریان داده در Sliding Window

```
Time Series Data (T days)
        ↓
For each window:
    Train Window: [day_i, day_i+window_size]
    Test Day: day_i+window_size+horizon
        ↓
    Calculate popularity scores for all items
        ↓
    Compare with actual popularity
        ↓
    Calculate metrics (Spearman, MAE, NDCG, etc.)
        ↓
    Save to storage
        ↓
Next window (i = i + 1)
```

---

## 6. مشخصات فنی

### 6.1 ورودی‌ها

**پارامترهای اجباری:**
- `dataset`: نام دیتاست (movielens, youtube07, youku)

**پارامترهای اختیاری:**
- `num_items`: تعداد آیتم‌ها (پیش‌فرض: همه)
- `start_date`: تاریخ شروع (فرمت: YYYY-MM-DD)
- `end_date`: تاریخ پایان (فرمت: YYYY-MM-DD)
- `window_size`: اندازه پنجره (پیش‌فرض: 30 روز)
- `prediction_horizon`: افق پیش‌بینی (پیش‌فرض: 7 روز)
- `methods`: لیست روش‌ها (پیش‌فرض: همه)
- `final_format`: فرمت نهایی (csv یا parquet، پیش‌فرض: csv)
- `item_selection`: روش انتخاب (top, random, stratified)

### 6.2 خروجی‌ها

**ساختار خروجی:**
```
results/{dataset}/w{window}_h{horizon}_n{items}_{selection}_{timestamp}/
├── detailed/                      # نتایج تفصیلی (Parquet)
│   ├── {method}_scores.parquet   # ~70 MB per method
│   └── ...
│
├── summary/                       # خلاصه (Parquet)
│   ├── {method}_stratum_summary.parquet  # ~1 MB per method
│   └── ...
│
├── comparison/                    # مقایسه (CSV یا Parquet)
│   └── method_comparison.{csv|parquet}   # ~10 KB
│
├── metadata/                      # اطلاعات (JSON)
│   ├── config.json
│   ├── runtime_stats.json
│   └── thresholds.json
│
└── visualization/                 # نمودارها (PNG)
    ├── temporal_evolution_*.png
    ├── method_comparison.png
    └── ...
```

### 6.3 فرمت‌های داده

**Parquet Schema - Detailed Scores:**
```python
{
    'window_id': int32,           # شماره window
    'timestamp': int64,           # Unix milliseconds
    'item_id': int32,             # شناسه آیتم
    'stratum': int8,              # 0-3 (cold/low/med/high)
    'popularity_score': float32,  # امتیاز محبوبیت
    'actual_count': int32,        # تعداد واقعی
    'train_count': int32,         # تعداد در train
    'mae': float32,               # Mean Absolute Error
    'mape': float32,              # MAPE
    'rank_predicted': int32,      # رتبه پیش‌بینی شده
    'rank_actual': int32,         # رتبه واقعی
}
```

**CSV Schema - Comparison:**
```python
{
    'method': str,                # نام روش
    'spearman': float,            # Spearman correlation
    'kendall': float,             # Kendall's tau
    'mae': float,                 # Mean Absolute Error
    'rmse': float,                # Root Mean Squared Error
    'mape': float,                # MAPE (%)
    'ndcg': float,                # NDCG
    'coverage': float,            # Coverage
    'num_samples': int,           # تعداد نمونه‌ها
}
```

### 6.4 معیارهای ارزیابی

**معیارهای Ranking:**
- Spearman Correlation
- Kendall's Tau
- NDCG (Normalized Discounted Cumulative Gain)

**معیارهای Error:**
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- MAPE (Mean Absolute Percentage Error)

**سایر معیارها:**
- Coverage
- Rank Error (mean, median)
- Temporal Stability

---

## 7. استانداردهای کدنویسی

### 7.1 Python Style Guide

**پایبندی به PEP 8:**
- خطوط حداکثر 100 کاراکتر
- 4 فاضله برای indent
- docstrings برای همه توابع و کلاس‌ها

**Type Hints:**
```python
def calculate_score(self, 
                   history: np.ndarray, 
                   window_size: int = 30) -> float:
    """
    محاسبه امتیاز محبوبیت
    
    Args:
        history: تاریخچه دسترسی
        window_size: اندازه پنجره
    
    Returns:
        امتیاز محبوبیت
    """
    pass
```

### 7.2 Docstring Format

**استفاده از Google Style:**
```python
def function_name(param1: type1, param2: type2) -> return_type:
    """
    خلاصه تابع در یک خط
    
    توضیحات تفصیلی در چند خط...
    
    Args:
        param1: توضیح پارامتر اول
        param2: توضیح پارامتر دوم
    
    Returns:
        توضیح خروجی
    
    Raises:
        ValueError: در چه شرایطی
    
    Examples:
        >>> function_name(1, 2)
        3
    """
    pass
```

### 7.3 Error Handling

**استفاده از exceptions مشخص:**
```python
try:
    result = calculate_popularity(data)
except ValueError as e:
    logger.error(f"Invalid data: {e}")
    raise
except FileNotFoundError as e:
    logger.error(f"File not found: {e}")
    raise
```

### 7.4 Logging

**استفاده از logging module:**
```python
import logging

logger = logging.getLogger(__name__)

def process_data():
    logger.info("Starting data processing")
    try:
        # ...
        logger.debug(f"Processed {n} items")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        raise
```

---

## 8. تست و اعتبارسنجی

### 8.1 سطوح تست

**Unit Tests:**
- تست هر تابع به صورت مجزا
- استفاده از pytest
- Coverage حداقل 70%

**Integration Tests:**
- تست جریان کامل
- تست ارتباط ماژول‌ها

**System Tests:**
- تست end-to-end
- موجود: `test_exp2_system.py`, `test_unified_storage.py`

### 8.2 Test Cases

**تست‌های ضروری:**
1. بارگذاری صحیح دیتاست
2. فیلتر تاریخ
3. محاسبه صحیح معیارها
4. ذخیره و بازیابی
5. تحلیل نتایج
6. تولید نمودارها

### 8.3 اعتبارسنجی نتایج

**بررسی‌های خودکار:**
- مقادیر Spearman بین -1 و 1
- MAE همیشه مثبت
- تعداد windows صحیح
- اندازه فایل‌ها منطقی

---

## 9. مشخصات عملکرد

### 9.1 زمان اجرا

**تخمین:**
```
زمان (ساعت) ≈ (num_windows × num_items × num_methods) / 100,000
```

**مثال:**
- 1000 items, 329 windows, 9 methods → ~8 ساعت

### 9.2 حافظه

**استفاده از RAM:**
- حداکثر: ~8 GB برای 1000 items
- توصیه: 16 GB RAM

### 9.3 فضای دیسک

**تخمین:**
```
حجم (MB) ≈ num_methods × (70 MB + 1 MB) + 10 MB
```

**مثال:**
- 9 methods → ~650 MB

---

## 10. نقشه راه پیاده‌سازی

### فاز 1: ساختار پایه ✅
- [x] ساختار دایرکتوری
- [x] کلاس‌های پایه
- [x] Data loaders

### فاز 2: روش‌های محاسبه ✅
- [x] Access Frequency
- [x] LFU, LRU
- [x] EWMA
- [x] Wavelet methods
- [x] Statistical methods

### فاز 3: سیستم ارزیابی ✅
- [x] Temporal Evaluator
- [x] Stratification
- [x] Metrics
- [x] Storage System

### فاز 4: تحلیل و نمایش ✅
- [x] Results Analyzer
- [x] Visualizer
- [x] Show Results

### فاز 5: تست و مستندات ✅
- [x] Unit tests
- [x] Integration tests
- [x] مستندات جامع

### فاز 6: بهینه‌سازی (آینده)
- [ ] Parallel processing
- [ ] Caching
- [ ] GPU acceleration (optional)

---

## 11. نکات مهم پیاده‌سازی

### 11.1 وابستگی‌ها

**Python Packages:**
```
numpy>=1.21.0
pandas>=1.3.0
scipy>=1.7.0
scikit-learn>=0.24.0
matplotlib>=3.5.0
seaborn>=0.11.0
pyarrow>=10.0.0
dtcwt>=0.12.0
pywt>=1.1.1
tqdm>=4.62.0
```

### 11.2 Compatibility

**Python Version:** 3.8+
**OS:** Linux, macOS, Windows

### 11.3 Best Practices

1. **همیشه از type hints استفاده کنید**
2. **docstrings کامل بنویسید**
3. **error handling مناسب داشته باشید**
4. **logging را فراموش نکنید**
5. **کد را modular نگه دارید**
6. **از design patterns استفاده کنید**

---

## 12. Checklist پیاده‌سازی

قبل از commit:
- [ ] همه تست‌ها pass می‌کنند
- [ ] docstrings کامل هستند
- [ ] type hints اضافه شده
- [ ] logging مناسب است
- [ ] error handling کامل است
- [ ] کد PEP 8 compliant است
- [ ] مستندات به‌روز است

---

## پیوست A: مثال پیاده‌سازی کامل

**نمونه کلاس BasePopularityMethod:**
```python
from abc import ABC, abstractmethod
import numpy as np
from typing import Optional

class BasePopularityMethod(ABC):
    """
    کلاس پایه برای همه روش‌های تخمین محبوبیت
    """
    
    def __init__(self, name: str):
        """
        Args:
            name: نام روش
        """
        self.name = name
    
    @abstractmethod
    def calculate_score(self, 
                       history: np.ndarray,
                       current_time: int) -> float:
        """
        محاسبه امتیاز محبوبیت
        
        Args:
            history: آرایه تاریخچه دسترسی
            current_time: زمان فعلی
        
        Returns:
            امتیاز محبوبیت
        """
        pass
    
    def validate_window_size(self, window_size: int) -> bool:
        """
        اعتبارسنجی اندازه پنجره
        
        Args:
            window_size: اندازه پنجره
        
        Returns:
            True اگر معتبر باشد
        """
        return window_size > 0
    
    def get_name(self) -> str:
        """
        دریافت نام روش
        
        Returns:
            نام روش
        """
        return self.name
```

---

**تاریخ آخرین به‌روزرسانی:** 2 فوریه 2025
**نسخه:** 1.0
**وضعیت:** ✅ تایید شده برای پیاده‌سازی
