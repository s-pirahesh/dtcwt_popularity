# فایل‌های پروژه - ساختار کامل
# Project Files - Complete Structure

**تاریخ:** 2 فوریه 2025  
**نسخه:** 1.0  
**وضعیت:** ✅ آماده برای استفاده  

---

## 📋 فهرست مطالب

1. [ساختار فولدرها](#ساختار-فولدرها)
2. [فایل‌های جدید](#فایل‌های-جدید)
3. [فایل‌های اصلاح شده](#فایل‌های-اصلاح-شده)
4. [نحوه استفاده](#نحوه-استفاده)
5. [نکات مهم](#نکات-مهم)

---

## 📁 ساختار فولدرها

```
project_files/
│
├── data/                          # لایه داده (جدید)
│   └── loaders/
│       ├── __init__.py            # ✨ جدید
│       ├── base_loader.py         # ✨ جدید
│       └── movielens_loader.py    # ✨ جدید
│
├── evaluation/                    # لایه ارزیابی (اصلاح شده)
│   ├── __init__.py                # 🔧 اصلاح شده
│   ├── metrics.py                 # 🔧 تغییر نام (از metrics_v2.py)
│   ├── temporal_evaluator.py     # 🔧 اصلاح شده
│   └── results_analyzer.py       # 🔧 اصلاح شده
│
├── experiments/                   # اسکریپت‌های اجرایی (اصلاح شده)
│   ├── run_popularity_assessment.py  # 🔧 اصلاح شده
│   ├── analyze_results.py        # موجود
│   └── show_results.py            # موجود
│
├── docs/                          # مستندات
│   ├── IMPLEMENTATION_SPECIFICATION.md
│   ├── DATE_FILTERING_GUIDE.md
│   ├── WORKFLOW_DIAGRAM.md
│   ├── NAMING_UPDATE_FINAL_REPORT.md
│   ├── IMPLEMENTATION_PROGRESS.md
│   └── QUICK_REFERENCE.md
│
├── IMPLEMENTATION_GUIDE.md        # دستورالعمل اصلی
└── README.md                      # این فایل
```

**راهنمای نمادها:**
- ✨ = فایل جدید
- 🔧 = فایل اصلاح شده
- 📄 = فایل موجود (بدون تغییر)

---

## ✨ فایل‌های جدید

### 1. data/loaders/

#### base_loader.py (280 خط)
**کلاس پایه انتزاعی برای بارگذاری دیتاست‌ها**

**قابلیت‌ها:**
- `load_data()`: بارگذاری داده‌های خام (abstract)
- `filter_by_date()`: فیلتر تاریخی
- `aggregate_by_day()`: تجمیع روزانه
- `get_item_list()`: انتخاب آیتم‌ها (top/random/stratified)
- `prepare_temporal_data()`: آماده‌سازی کامل
- `validate_data()`: اعتبارسنجی

**استفاده:**
```python
from data.loaders import BaseLoader

class MyLoader(BaseLoader):
    def load_data(self):
        # پیاده‌سازی
        pass
```

#### movielens_loader.py (220 خط)
**بارگذاری MovieLens 25M Dataset**

**قابلیت‌ها:**
- بارگذاری 25 میلیون rating
- تبدیل timestamp (Unix → datetime)
- فیلتر تاریخی
- آمار و metadata
- ایجاد نمونه (sample)

**استفاده:**
```python
from data.loaders import get_movielens_loader

loader = get_movielens_loader()
data = loader.prepare_temporal_data(
    start_date='2023-01-01',
    end_date='2023-12-31',
    num_items=1000
)
```

#### __init__.py (20 خط)
**Package initialization**

**Export ها:**
- `BaseLoader`
- `MovieLensLoader`
- `get_movielens_loader`

---

## 🔧 فایل‌های اصلاح شده

### 1. evaluation/metrics.py
**تغییر نام:** `metrics_v2.py` → `metrics.py`

**دلیل:** حذف نسخه‌گذاری نامناسب از نام فایل

**تغییرات در کد:** هیچ (فقط نام تغییر کرد)

---

### 2. evaluation/__init__.py
**تغییر:** اصلاح import

```python
# قبل:
from .metrics_v2 import MetricsCalculator

# بعد:
from .metrics import MetricsCalculator
```

---

### 3. evaluation/temporal_evaluator.py
**تغییر:** اصلاح import

```python
# قبل:
from .metrics_v2 import MetricsCalculator

# بعد:
from .metrics import MetricsCalculator
```

---

### 4. evaluation/results_analyzer.py
**تغییر:** اصلاح import

```python
# قبل:
from .metrics_v2 import MetricsCalculator

# بعد:
from .metrics import MetricsCalculator
```

---

### 5. experiments/run_popularity_assessment.py
**تغییرات:**
- اصلاح help text (فارسی)
- به‌روزرسانی مثال‌ها
- بهبود توضیحات فیلتر تاریخ

---

## 📚 مستندات

### 1. IMPLEMENTATION_GUIDE.md
**دستورالعمل کامل پیاده‌سازی (675 خط)**

محتوا:
- معماری کلی سیستم
- ساختار کامل پروژه
- جزئیات کلاس‌ها و توابع
- جریان داده
- دستورالعمل گام‌به‌گام

### 2. docs/IMPLEMENTATION_SPECIFICATION.md
**مشخصات فنی کامل**

### 3. docs/DATE_FILTERING_GUIDE.md
**راهنمای کامل فیلتر تاریخ**

محتوا:
- نحوه استفاده از فیلتر تاریخ
- مثال‌های عملی
- محاسبه تعداد windows
- تخمین زمان اجرا

### 4. docs/WORKFLOW_DIAGRAM.md
**نمودارهای جریان کار (Mermaid)**

محتوا:
- نمودار جریان کامل
- نمودار ساده‌شده
- Timeline
- Sequence diagram

### 5. docs/NAMING_UPDATE_FINAL_REPORT.md
**گزارش اصلاح نامگذاری**

### 6. docs/IMPLEMENTATION_PROGRESS.md
**گزارش پیشرفت پیاده‌سازی**

### 7. docs/QUICK_REFERENCE.md
**مرجع سریع دستورات**

---

## 🚀 نحوه استفاده

### مرحله 1: کپی فایل‌ها

```bash
# کپی به پروژه اصلی
cp -r project_files/data ~/dtcwt_popularity/
cp -r project_files/evaluation/* ~/dtcwt_popularity/evaluation/
cp -r project_files/experiments/* ~/dtcwt_popularity/experiments/
cp project_files/IMPLEMENTATION_GUIDE.md ~/dtcwt_popularity/
```

### مرحله 2: نصب وابستگی‌ها

```bash
pip install numpy pandas scipy scikit-learn matplotlib seaborn pyarrow pywt dtcwt tqdm --break-system-packages
```

### مرحله 3: تست

```bash
# تست import
cd ~/dtcwt_popularity
python -c "from data.loaders import get_movielens_loader; print('✓ OK')"
python -c "from evaluation import MetricsCalculator; print('✓ OK')"
```

### مرحله 4: اجرای تست سریع

```bash
python experiments/run_popularity_assessment.py movielens \
    --num-items 100 \
    --start-date 2023-08-01 \
    --end-date 2023-08-31
```

---

## 📊 خلاصه تغییرات

### آمار فایل‌ها:

| نوع | تعداد | خطوط کد |
|-----|-------|---------|
| **فایل‌های جدید** | 3 | ~520 |
| **فایل‌های اصلاح شده** | 5 | - |
| **مستندات** | 7 | 3000+ |

### فایل‌های جدید:
1. ✨ `data/loaders/base_loader.py` (280 خط)
2. ✨ `data/loaders/movielens_loader.py` (220 خط)
3. ✨ `data/loaders/__init__.py` (20 خط)

### فایل‌های اصلاح شده:
1. 🔧 `evaluation/metrics_v2.py` → `metrics.py`
2. 🔧 `evaluation/__init__.py`
3. 🔧 `evaluation/temporal_evaluator.py`
4. 🔧 `evaluation/results_analyzer.py`
5. 🔧 `experiments/run_popularity_assessment.py`

---

## ⚠️ نکات مهم

### 1. تغییر نام metrics_v2.py
```bash
# در پروژه اصلی:
cd ~/dtcwt_popularity/evaluation
rm metrics_v2.py  # حذف فایل قدیمی
# سپس فایل جدید metrics.py را کپی کنید
```

### 2. فایل قدیمی __init__.py
```bash
# data/loaders/__init__.py قدیمی را backup کنید:
cd ~/dtcwt_popularity/data/loaders
mv __init__.py __init__.py.old
# سپس فایل جدید را کپی کنید
```

### 3. تست بعد از کپی
```bash
# همیشه تست کنید:
python -c "from evaluation import MetricsCalculator"
python -c "from data.loaders import MovieLensLoader"
```

### 4. Git
```bash
# Commit تغییرات:
git add data/loaders/
git add evaluation/
git add experiments/
git commit -m "feat: add data loaders and fix naming conventions"
```

---

## 🎯 Phase های بعدی

### Phase 2: Methods Layer (در حال انجام)
فایل‌هایی که باید پیاده‌سازی شوند:

1. `methods/base_method.py`
2. `methods/access_frequency.py`
3. `methods/wavelet_af.py`
4. `methods/lfu.py`
5. `methods/lru.py`
6. `methods/ewma.py`

**زمان تخمینی:** ~10 ساعت

---

## 📞 پشتیبانی

### مشکلات رایج:

**1. ImportError:**
```bash
# حل:
pip install pyarrow --break-system-packages
```

**2. فایل قدیمی metrics_v2:**
```bash
# حذف کنید:
rm evaluation/metrics_v2.py
```

**3. مسیرهای اشتباه:**
```bash
# مطمئن شوید در دایرکتوری صحیح هستید:
cd ~/dtcwt_popularity
```

---

## ✅ Checklist کپی

قبل از شروع:
- [ ] پروژه را backup کنید
- [ ] Git status بررسی کنید

بعد از کپی:
- [ ] همه فایل‌ها کپی شدند
- [ ] فایل قدیمی metrics_v2.py حذف شد
- [ ] Import ها کار می‌کنند
- [ ] تست‌ها موفق هستند
- [ ] تغییرات commit شدند

---

## 📈 پیشرفت پروژه

```
Phase 1: Data Layer        ████████░░ 100% ✅
Phase 2: Methods Layer     ██░░░░░░░░  20% 🔄
Phase 3: Integration       ░░░░░░░░░░   0% ⏳
Phase 4: Testing           ░░░░░░░░░░   0% ⏳

کل پروژه:                 ███░░░░░░░  30% 📊
```

---

## 📚 مراجع

1. **IMPLEMENTATION_GUIDE.md** - دستورالعمل کامل
2. **docs/QUICK_REFERENCE.md** - مرجع سریع
3. **docs/DATE_FILTERING_GUIDE.md** - راهنمای فیلتر تاریخ
4. **docs/WORKFLOW_DIAGRAM.md** - نمودارها

---

**تاریخ ایجاد:** 2 فوریه 2025  
**آخرین به‌روزرسانی:** 2 فوریه 2025  
**نسخه:** 1.0  
**وضعیت:** ✅ آماده برای استفاده  

---

**همه فایل‌ها آماده و مستندسازی شده است!** 🎉✨
