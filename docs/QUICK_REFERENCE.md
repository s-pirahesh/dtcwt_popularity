# Quick Reference - وحدت رویه نهایی

## 🎯 قوانین ذخیره‌سازی (یک نگاه)

| نوع | فرمت | تغییرپذیری | حجم |
|-----|------|-----------|-----|
| **detailed** | Parquet | ❌ نه | 70 MB |
| **summary** | Parquet | ❌ نه | 1 MB |
| **comparison** | CSV/Parquet | ✅ بله | 10 KB |
| **metadata** | JSON | ❌ نه | 5 KB |

---

## 🚀 دستورات اصلی

```bash
# شبیه‌سازی (CSV - پیش‌فرض)
python run_popularity_assessment.py movielens --num-items 100

# شبیه‌سازی (Parquet)
python run_popularity_assessment.py movielens --num-items 100 --format parquet

# با فیلتر تاریخ (1 سال)
python run_popularity_assessment.py movielens \
    --num-items 1000 \
    --start-date 2023-01-01 \
    --end-date 2023-12-31

# تحلیل (خودکار)
python analyze_results.py --list movielens
python analyze_results.py RESULTS_PATH

# نمایش
python show_results.py RESULTS_PATH --both

# تست
python experiments/test_unified_storage.py
```

---

## 🗓️ فیلتر تاریخ (جدید!)

```bash
# تست سریع (1 ماه - 30 دقیقه)
python run_popularity_assessment.py movielens \
    --num-items 100 \
    --start-date 2023-08-01 \
    --end-date 2023-08-31

# یک سه‌ماهه (2 ساعت)
python run_popularity_assessment.py movielens \
    --num-items 500 \
    --start-date 2023-01-01 \
    --end-date 2023-03-31

# یک سال کامل (8 ساعت)
python run_popularity_assessment.py movielens \
    --num-items 1000 \
    --start-date 2023-01-01 \
    --end-date 2023-12-31

# بدون فیلتر (همه داده‌ها)
python run_popularity_assessment.py movielens \
    --num-items 1000
```

### محاسبه تعداد Windows:
```
num_windows = (end_date - start_date) - window_size - horizon + 1
```

---

## 📁 ساختار خروجی

```
results/movielens/RUN_NAME/
├── detailed/*.parquet          [Parquet Only - 70 MB each]
├── summary/*.parquet           [Parquet Only - 1 MB each]
├── comparison/*.{csv|parquet}  [Configurable - 10 KB]
├── metadata/*.json             [JSON Only - 5 KB]
└── visualization/*.png         [PNG - 1 MB each]
```

---

## 💻 Python API

```python
from evaluation import ResultsAnalyzer

# بارگذاری
analyzer = ResultsAnalyzer('results/movielens/...')

# نتایج میانی (Parquet)
detailed = analyzer.load_detailed_scores('DTCWT+AF')
summary = analyzer.load_stratum_summary('DTCWT+AF')

# نتایج نهایی (خودکار CSV یا Parquet)
comparison = analyzer.load_method_comparison()

# تحلیل
metrics = analyzer.calculate_overall_metrics('DTCWT+AF', filter_top_percent=20)
comp = analyzer.compare_methods(filter_stratum='cold_start')
```

---

## ✅ تست‌ها

```bash
# تست سیستم اصلی
python experiments/test_exp2_system.py          # 7/7 PASS ✅

# تست وحدت رویه
python experiments/test_unified_storage.py      # 3/3 PASS ✅

# تست تحلیل
python experiments/test_analysis_system.py      # 3/3 PASS ✅
```

---

## 🎨 مثال کامل

```bash
# 1. شبیه‌سازی با فیلتر تاریخ
python run_popularity_assessment.py movielens \
    --num-items 1000 \
    --start-date 2023-01-01 \
    --end-date 2023-12-31 \
    --format csv

# نتیجه:
# results/movielens/w30_h7_n1000_top_20250202_143052/

# 2. لیست
python analyze_results.py --list movielens

# 3. تحلیل کامل
python analyze_results.py results/movielens/w30_h7_n1000_top_20250202_143052/ \
    --mode detailed \
    --visualize

# 4. تحلیل با فیلتر
python analyze_results.py results/movielens/w30_h7_n1000_top_20250202_143052/ \
    --top-percent 20 \
    --stratum cold_start

# 5. نمایش
python show_results.py results/movielens/w30_h7_n1000_top_20250202_143052/ \
    --both --show
```

---

## 📊 فشرده‌سازی

| داده | CSV | Parquet | صرفه‌جویی |
|------|-----|---------|----------|
| 10M records | 1.5 GB | 200 MB | **85%** ✅ |
| 100 records | 10 KB | 8 KB | 20% |

---

## 🔍 عیب‌یابی

### خطا: "pyarrow not found"
```bash
pip install pyarrow --break-system-packages
```

### خطا: "Invalid final_format"
```bash
# فقط مجاز: csv, parquet
--format csv     ✅
--format parquet ✅
--format hdf5    ❌
```

### خطا: "بازه زمانی کافی نیست"
```bash
# بازه باید بزرگتر از window_size + horizon باشد
# مثال خطا:
--start-date 2023-01-01 --end-date 2023-01-20  # 20 روز
--window-size 30                                # نیاز به حداقل 30+7=37 روز

# درست:
--start-date 2023-01-01 --end-date 2023-02-15  # 45 روز ✅
```

---

## 📚 مستندات

1. **UNIFIED_FINAL_SUMMARY.md** - خلاصه تغییرات
2. **STORAGE_STRATEGY_UNIFIED.md** - استراتژی کامل
3. **ANALYSIS_GUIDE.md** - راهنمای تحلیل
4. **DATE_FILTERING_GUIDE.md** - راهنمای فیلتر تاریخ (جدید!)
5. **WORKFLOW_DIAGRAM.md** - نمودارهای جریان کار
6. این فایل - Quick Reference

---

**همه چیز آماده! وحدت رویه کامل! فیلتر تاریخ کامل!** ✅🎉

---

## 📁 ساختار خروجی

```
results/movielens/RUN_NAME/
├── detailed/*.parquet          [Parquet Only - 70 MB each]
├── summary/*.parquet           [Parquet Only - 1 MB each]
├── comparison/*.{csv|parquet}  [Configurable - 10 KB]
├── metadata/*.json             [JSON Only - 5 KB]
└── visualization/*.png         [PNG - 1 MB each]
```

---

## 💻 Python API

```python
from evaluation import ResultsAnalyzer

# بارگذاری
analyzer = ResultsAnalyzer('results/movielens/...')

# نتایج میانی (Parquet)
detailed = analyzer.load_detailed_scores('DTCWT+AF')
summary = analyzer.load_stratum_summary('DTCWT+AF')

# نتایج نهایی (خودکار CSV یا Parquet)
comparison = analyzer.load_method_comparison()

# تحلیل
metrics = analyzer.calculate_overall_metrics('DTCWT+AF', filter_top_percent=20)
comp = analyzer.compare_methods(filter_stratum='cold_start')
```

---

## ✅ تست‌ها

```bash
# تست سیستم اصلی
python experiments/test_exp2_system.py          # 7/7 PASS ✅

# تست وحدت رویه
python experiments/test_unified_storage.py      # 3/3 PASS ✅

# تست تحلیل
python experiments/test_analysis_system.py      # 3/3 PASS ✅
```

---

## 🎨 مثال کامل

```bash
# 1. شبیه‌سازی
python exp2_temporal_evaluation.py movielens \
    --num-items 1000 \
    --start-date 2022-01-01 \
    --end-date 2023-12-31 \
    --format csv

# نتیجه:
# results/movielens/w30_h7_n1000_top_20250202_143052/

# 2. لیست
python analyze_results.py --list movielens

# 3. تحلیل کامل
python analyze_results.py results/movielens/w30_h7_n1000_top_20250202_143052/ \
    --mode detailed \
    --visualize

# 4. تحلیل با فیلتر
python analyze_results.py results/movielens/w30_h7_n1000_top_20250202_143052/ \
    --top-percent 20 \
    --stratum cold_start
```

---

## 📊 فشرده‌سازی

| داده | CSV | Parquet | صرفه‌جویی |
|------|-----|---------|----------|
| 10M records | 1.5 GB | 200 MB | **85%** ✅ |
| 100 records | 10 KB | 8 KB | 20% |

---

## 🔍 عیب‌یابی

### خطا: "pyarrow not found"
```bash
pip install pyarrow --break-system-packages
```

### خطا: "Invalid final_format"
```bash
# فقط مجاز: csv, parquet
--format csv     ✅
--format parquet ✅
--format hdf5    ❌
```

### خطا: "File not found"
```bash
# چک کنید:
ls results/movielens/RUN_NAME/detailed/       # باید *.parquet باشد
ls results/movielens/RUN_NAME/comparison/     # باید *.csv یا *.parquet باشد
```

---

## 📚 مستندات

1. **UNIFIED_FINAL_SUMMARY.md** - خلاصه تغییرات
2. **STORAGE_STRATEGY_UNIFIED.md** - استراتژی کامل
3. **ANALYSIS_GUIDE.md** - راهنمای تحلیل
4. این فایل - Quick Reference

---

**همه چیز آماده! وحدت رویه کامل! ✅🎉**
