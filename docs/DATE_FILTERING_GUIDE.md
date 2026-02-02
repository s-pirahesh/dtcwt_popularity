# راهنمای کامل - فیلتر تاریخ در محاسبات محبوبیت

## 🗓️ فیلتر تاریخ - قابلیت‌ها

در `run_popularity_assessment.py` می‌توانید بازه زمانی دلخواه را انتخاب کنید:

### روش‌های تنظیم:

1. **از خط فرمان (Command Line)** - توصیه می‌شود ✅
2. **از فایل پیکربندی (Config File)**
3. **به صورت پیش‌فرض (از ابتدا تا انتها)**

---

## 📝 نحوه استفاده

### 1. از خط فرمان (ساده‌ترین)

```bash
python run_popularity_assessment.py movielens \
    --num-items 1000 \
    --start-date 2023-01-01 \
    --end-date 2023-12-31
```

**پارامترها:**
- `--start-date`: تاریخ شروع (فرمت: YYYY-MM-DD)
- `--end-date`: تاریخ پایان (فرمت: YYYY-MM-DD)

---

## 🎯 مثال‌های عملی

### مثال 1: یک سال کامل (2023)
```bash
python run_popularity_assessment.py movielens \
    --num-items 1000 \
    --start-date 2023-01-01 \
    --end-date 2023-12-31 \
    --window-size 30 \
    --horizon 7

# نتیجه:
# - تعداد روزها: 365
# - تعداد windows: 365 - 30 - 7 + 1 = 329
# - زمان تخمینی: ~8 ساعت
```

### مثال 2: سه ماهه اول 2023 (Q1)
```bash
python run_popularity_assessment.py movielens \
    --num-items 500 \
    --start-date 2023-01-01 \
    --end-date 2023-03-31 \
    --window-size 14 \
    --horizon 3

# نتیجه:
# - تعداد روزها: 90
# - تعداد windows: 90 - 14 - 3 + 1 = 74
# - زمان تخمینی: ~2 ساعت
```

### مثال 3: یک ماه خاص (ژانویه 2023)
```bash
python run_popularity_assessment.py movielens \
    --num-items 100 \
    --start-date 2023-01-01 \
    --end-date 2023-01-31 \
    --window-size 7 \
    --horizon 3

# نتیجه:
# - تعداد روزها: 31
# - تعداد windows: 31 - 7 - 3 + 1 = 22
# - زمان تخمینی: ~30 دقیقه
```

### مثال 4: بازه دلخواه (6 ماه)
```bash
python run_popularity_assessment.py movielens \
    --num-items 1000 \
    --start-date 2022-07-01 \
    --end-date 2022-12-31 \
    --window-size 30 \
    --horizon 7

# نتیجه:
# - تعداد روزها: 184
# - تعداد windows: 184 - 30 - 7 + 1 = 148
# - زمان تخمینی: ~4 ساعت
```

### مثال 5: بدون فیلتر (همه داده‌ها)
```bash
python run_popularity_assessment.py movielens \
    --num-items 1000 \
    --window-size 30 \
    --horizon 7

# استفاده از همه داده‌های موجود در دیتاست
# MovieLens: از 1995 تا 2018 (~8500 روز)
```

---

## 📊 دیتاست‌های مختلف

### MovieLens 25M
```bash
# بازه زمانی: 1995-01-09 تا 2018-09-26

# مثال: سال 2018
python run_popularity_assessment.py movielens \
    --start-date 2018-01-01 \
    --end-date 2018-09-26 \
    --num-items 1000

# مثال: 5 سال اخیر
python run_popularity_assessment.py movielens \
    --start-date 2014-01-01 \
    --end-date 2018-09-26 \
    --num-items 1000

# مثال: تست سریع (1 ماه)
python run_popularity_assessment.py movielens \
    --start-date 2018-08-01 \
    --end-date 2018-08-31 \
    --num-items 100
```

### YouTube
```bash
# بازه زمانی: معمولاً چند ماه

# مثال: 3 ماه
python run_popularity_assessment.py youtube07 \
    --start-date 2007-01-01 \
    --end-date 2007-03-31 \
    --num-items 500
```

### Youku
```bash
# بازه زمانی: معمولاً چند ماه

# مثال: یک فصل
python run_popularity_assessment.py youku \
    --start-date 2023-01-01 \
    --end-date 2023-03-31 \
    --num-items 500
```

---

## 🧮 محاسبه تعداد Windows

فرمول محاسبه:
```
num_windows = (end_date - start_date) - window_size - horizon + 1
```

### جدول محاسبه سریع

| بازه زمانی | روزها | window=30, h=7 | window=14, h=3 | window=7, h=3 |
|------------|-------|----------------|----------------|---------------|
| 1 ماه | 30 | ندارد | 14 | 21 |
| 3 ماه | 90 | 54 | 74 | 81 |
| 6 ماه | 180 | 144 | 164 | 171 |
| 1 سال | 365 | 329 | 349 | 356 |
| 2 سال | 730 | 694 | 714 | 721 |
| 5 سال | 1825 | 1789 | 1809 | 1816 |

---

## ⚡ تخمین زمان اجرا

### فرمول تخمین:
```
زمان (ساعت) ≈ (num_windows × num_items × num_methods) / 100,000
```

### جدول تخمینی

| Items | Windows | Methods | زمان تخمینی |
|-------|---------|---------|-------------|
| 100 | 50 | 9 | ~30 دقیقه |
| 500 | 150 | 9 | ~2 ساعت |
| 1000 | 329 | 9 | ~8 ساعت |
| 1000 | 1789 | 9 | ~40 ساعت |

**نکته:** زمان واقعی بسته به سخت‌افزار متفاوت است.

---

## 🎯 توصیه‌های عملی

### برای تست و توسعه:
```bash
# تست سریع - 30 دقیقه
python run_popularity_assessment.py movielens \
    --num-items 100 \
    --start-date 2023-08-01 \
    --end-date 2023-08-31 \
    --window-size 7 \
    --horizon 3
```

### برای آزمایش الگوریتم:
```bash
# آزمایش متوسط - 2 ساعت
python run_popularity_assessment.py movielens \
    --num-items 500 \
    --start-date 2023-01-01 \
    --end-date 2023-03-31 \
    --window-size 14 \
    --horizon 3
```

### برای نتایج نهایی مقاله:
```bash
# ارزیابی کامل - 8-10 ساعت
python run_popularity_assessment.py movielens \
    --num-items 1000 \
    --start-date 2023-01-01 \
    --end-date 2023-12-31 \
    --window-size 30 \
    --horizon 7
```

---

## 🔍 بررسی تاریخ‌های موجود در دیتاست

### روش 1: استفاده از prepare_data.py
```bash
python data_loaders/prepare_data.py --dataset movielens --stats
```

### روش 2: در Python
```python
import pandas as pd

# بارگذاری MovieLens
df = pd.read_csv('data/movielens/ratings.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

# بازه زمانی
print(f"Start: {df['timestamp'].min()}")
print(f"End:   {df['timestamp'].max()}")
print(f"Days:  {(df['timestamp'].max() - df['timestamp'].min()).days}")

# خروجی نمونه:
# Start: 1995-01-09
# End:   2018-09-26
# Days:  8626
```

---

## ⚙️ تنظیم در Config File

اگر می‌خواهید در کد تنظیم کنید:

```python
from evaluation import get_movielens_config

config = get_movielens_config(
    num_items=1000,
    start_date='2023-01-01',
    end_date='2023-12-31',
    window_size=30,
    prediction_horizon=7
)
```

---

## 📋 Checklist قبل از اجرا

- [ ] تاریخ شروع صحیح است؟ (YYYY-MM-DD)
- [ ] تاریخ پایان صحیح است؟ (YYYY-MM-DD)
- [ ] بازه زمانی کافی است؟ (حداقل window_size + horizon)
- [ ] تعداد windows منطقی است؟
- [ ] زمان تخمینی قابل قبول است؟
- [ ] فضای دیسک کافی است؟ (~1GB per 1000 items)

---

## ❌ خطاهای رایج

### 1. بازه زمانی کوتاه
```bash
# ❌ خطا
python run_popularity_assessment.py movielens \
    --start-date 2023-01-01 \
    --end-date 2023-01-20 \
    --window-size 30

# Error: بازه زمانی کافی نیست (20 < 30 + 7)
```

### 2. فرمت تاریخ اشتباه
```bash
# ❌ خطا
--start-date 01-01-2023    # باید YYYY-MM-DD باشد

# ✅ صحیح
--start-date 2023-01-01
```

### 3. تاریخ خارج از بازه دیتاست
```bash
# ❌ MovieLens تا 2018 داده دارد
--start-date 2020-01-01    # خارج از بازه

# ✅ صحیح
--start-date 2018-01-01
```

---

## 🔄 مقایسه بازه‌های مختلف

### سناریو: تاثیر فصل‌ها

```bash
# زمستان 2023
python run_popularity_assessment.py movielens \
    --start-date 2023-01-01 --end-date 2023-03-31 \
    --num-items 1000

# بهار 2023
python run_popularity_assessment.py movielens \
    --start-date 2023-04-01 --end-date 2023-06-30 \
    --num-items 1000

# تابستان 2023
python run_popularity_assessment.py movielens \
    --start-date 2023-07-01 --end-date 2023-09-30 \
    --num-items 1000

# پاییز 2023
python run_popularity_assessment.py movielens \
    --start-date 2023-10-01 --end-date 2023-12-31 \
    --num-items 1000

# مقایسه نتایج با analyze_results.py
```

---

## 💡 نکات پیشرفته

### 1. اجرای Parallel برای بازه‌های مختلف
```bash
# Terminal 1
python run_popularity_assessment.py movielens \
    --start-date 2022-01-01 --end-date 2022-12-31 \
    --num-items 1000 &

# Terminal 2
python run_popularity_assessment.py movielens \
    --start-date 2023-01-01 --end-date 2023-12-31 \
    --num-items 1000 &
```

### 2. تست سریع با بازه کوچک
```bash
# قبل از اجرای کامل، تست کنید:
python run_popularity_assessment.py movielens \
    --start-date 2023-01-01 --end-date 2023-01-07 \
    --num-items 50 \
    --window-size 3 \
    --horizon 1

# اگر موفق بود، بازه را بزرگ کنید
```

---

## 📊 خلاصه دستورات

```bash
# تست سریع (30 دقیقه)
python run_popularity_assessment.py movielens \
    --num-items 100 \
    --start-date 2023-08-01 \
    --end-date 2023-08-31

# یک سه‌ماهه (2 ساعت)
python run_popularity_assessment.py movielens \
    --num-items 500 \
    --start-date 2023-01-01 \
    --end-date 2023-03-31

# یک سال (8 ساعت)
python run_popularity_assessment.py movielens \
    --num-items 1000 \
    --start-date 2023-01-01 \
    --end-date 2023-12-31

# همه داده‌ها (روزها)
python run_popularity_assessment.py movielens \
    --num-items 1000
```

---

**فیلتر تاریخ: کنترل کامل بر بازه زمانی محاسبات!** 🗓️✨

**نکته مهم:** همیشه قبل از اجرای طولانی، با بازه کوچک تست کنید!
