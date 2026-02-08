# راهنمای MovieLens Converter

## 📚 معرفی

MovieLens Converter برای تبدیل دیتاست MovieLens (ratings.csv) به فرمت استاندارد برای تحلیل popularity طراحی شده است.

**Dataset**: MovieLens (ml-25m, ml-32m)  
**Source**: https://grouplens.org/datasets/movielens/  
**Size**: 32M ratings, 87K movies, 200K users  
**Period**: 1995-2023

---

## 📊 فرمت داده

### Input (ratings.csv)

```csv
userId,movieId,rating,timestamp
1,1,4.0,964982703
1,3,4.0,964981247
1,6,4.0,964982224
```

**ستون‌ها:**
- `userId`: شناسه کاربر (int)
- `movieId`: شناسه فیلم (int)
- `rating`: امتیاز (0.5 to 5.0)
- `timestamp`: Unix timestamp (seconds since 1970)

### Output (استاندارد)

**حالت پایه:**
```csv
timestamp,item_id,count
2024-01-15 12:34:56,123,1
2024-01-15 12:45:23,456,1
```

**با rating:**
```csv
timestamp,item_id,count,rating
2024-01-15 12:34:56,123,1,4.5
```

**با aggregation:**
```csv
timestamp,item_id,count,rating
2024-01-15 00:00:00,123,245,4.3
2024-01-15 00:00:00,456,189,3.8
```

---

## 🔧 Parameters

### `--movielens-aggregate-by`

**نوع**: string  
**مقادیر مجاز**: `hour`, `day`, `week`, `none`  
**پیش‌فرض**: `None` (بدون تجمیع)

**توضیح**: تجمیع ratings بر اساس بازه زمانی

**مثال‌ها:**

```bash
# بدون تجمیع (هر rating جداگانه)
python prepare_data.py --dataset movielens

# تجمیع ساعتی
python prepare_data.py --dataset movielens \
    --movielens-aggregate-by hour

# تجمیع روزانه (پیشنهادی برای DTCWT)
python prepare_data.py --dataset movielens \
    --movielens-aggregate-by day

# تجمیع هفتگی
python prepare_data.py --dataset movielens \
    --movielens-aggregate-by week
```

**تأثیر روی output:**

| Aggregation | Input Records | Output Records (approx) | Window Size (DTCWT) |
|-------------|---------------|-------------------------|---------------------|
| None | 32M | 32M | 30-64 ratings |
| hour | 32M | ~8M | 720-1440 hours |
| day | 32M | ~300K | 30-64 days |
| week | 32M | ~50K | 8-12 weeks |

**پیشنهاد**: برای popularity analysis، `day` بهترین گزینه است.

---

### `--movielens-keep-rating`

**نوع**: flag (boolean)  
**پیش‌فرض**: `False`

**توضیح**: نگهداری ستون rating در خروجی

**استفاده:**

```bash
# با نگهداری rating
python prepare_data.py --dataset movielens \
    --movielens-keep-rating
```

**Output:**
```csv
timestamp,item_id,count,rating
2024-01-15 00:00:00,123,245,4.3
```

**کاربرد**:
- مقایسه popularity با quality
- تحلیل همبستگی rating و popularity
- Baseline methods که از rating استفاده می‌کنند

---

### `--movielens-keep-user`

**نوع**: flag (boolean)  
**پیش‌فرض**: `False`

**توضیح**: نگهداری ستون user_id در خروجی

**استفاده:**

```bash
# با نگهداری user
python prepare_data.py --dataset movielens \
    --movielens-keep-user
```

**Output:**
```csv
timestamp,item_id,count,user_id
2024-01-15 12:34:56,123,1,456
```

**با aggregation:**
```csv
timestamp,item_id,count,user_count
2024-01-15 00:00:00,123,245,198
```

**توضیح**: با aggregation، به جای user_id منفرد، تعداد کاربران یکتا (`user_count`) ذخیره می‌شود.

**کاربرد**:
- تحلیل user behavior
- Graph-based methods
- Collaborative filtering baselines

---

### `--movielens-min-rating`

**نوع**: float  
**پیش‌فرض**: `None` (بدون فیلتر)  
**محدوده**: 0.5 to 5.0

**توضیح**: فقط ratings بالاتر از این مقدار نگه‌داری می‌شوند

**استفاده:**

```bash
# فقط ratings بالا (>= 4.0)
python prepare_data.py --dataset movielens \
    --movielens-min-rating 4.0

# فقط ratings خوب (>= 3.5)
python prepare_data.py --dataset movielens \
    --movielens-min-rating 3.5 \
    --movielens-keep-rating
```

**تأثیر:**

| min_rating | Records Kept (approx) | Percentage |
|------------|----------------------|------------|
| None | 32M | 100% |
| 3.0 | 24M | 75% |
| 3.5 | 18M | 56% |
| 4.0 | 12M | 37% |
| 4.5 | 5M | 16% |

**کاربرد**:
- فوکوس روی محتوای با کیفیت
- کاهش noise (ratings بسیار پایین)
- تحلیل implicit feedback (rating بالا = like)

---

## 📖 Use Cases

### Use Case 1: Basic Popularity Analysis

```bash
python prepare_data.py --dataset movielens \
    --input data/raw/movielens/ratings.csv \
    --output data/datasets/movielens_basic.csv
```

**Output:**
```csv
timestamp,item_id,count
1995-01-09 11:46:43,1,1
1995-01-09 11:47:27,3,1
```

**مناسب برای:**
- تست اولیه
- Fine-grained temporal analysis

---

### Use Case 2: Daily Aggregation (پیشنهادی)

```bash
python prepare_data.py --dataset movielens \
    --movielens-aggregate-by day \
    --output data/datasets/movielens_daily.csv
```

**Output:**
```csv
timestamp,item_id,count
1995-01-09 00:00:00,1,12
1995-01-09 00:00:00,3,8
```

**مناسب برای:**
- DTCWT analysis (window: 30-64 days)
- DWT analysis
- Statistical methods

---

### Use Case 3: High-Quality Content Focus

```bash
python prepare_data.py --dataset movielens \
    --movielens-aggregate-by day \
    --movielens-keep-rating \
    --movielens-min-rating 4.0 \
    --output data/datasets/movielens_high_quality.csv
```

**Output:**
```csv
timestamp,item_id,count,rating
1995-01-09 00:00:00,1,8,4.5
1995-01-09 00:00:00,3,5,4.2
```

**مناسب برای:**
- Recommendation systems (فوکوس روی liked items)
- Quality-weighted popularity

---

### Use Case 4: User-Aware Analysis

```bash
python prepare_data.py --dataset movielens \
    --movielens-aggregate-by day \
    --movielens-keep-user \
    --movielens-keep-rating \
    --output data/datasets/movielens_user_aware.csv
```

**Output:**
```csv
timestamp,item_id,count,rating,user_count
1995-01-09 00:00:00,1,12,4.3,10
1995-01-09 00:00:00,3,8,3.9,7
```

**توضیح**: `user_count` نشان می‌دهد چند کاربر یکتا در این روز به این فیلم rating دادند.

**مناسب برای:**
- Diversity analysis
- User engagement metrics

---

## 🎯 پیشنهادات برای PhD Research

### Scenario 1: تست سریع الگوریتم

```bash
# Sample 1M ratings
head -n 1000000 data/raw/movielens/ratings.csv > data/raw/movielens/ratings_sample.csv

# تبدیل
python prepare_data.py --dataset movielens \
    --input data/raw/movielens/ratings_sample.csv \
    --output data/datasets/movielens_sample.csv \
    --movielens-aggregate-by day
```

---

### Scenario 2: DTCWT Evaluation

```bash
python prepare_data.py --dataset movielens \
    --movielens-aggregate-by day \
    --output data/datasets/movielens_for_dtcwt.csv
```

**Window size پیشنهادی**: 30, 42, 56, 64 days

---

### Scenario 3: Baseline Comparison

```bash
# برای ARMA, LSTM
python prepare_data.py --dataset movielens \
    --movielens-aggregate-by day \
    --output data/datasets/movielens_baselines.csv

# برای Collaborative Filtering
python prepare_data.py --dataset movielens \
    --movielens-aggregate-by day \
    --movielens-keep-user \
    --movielens-keep-rating \
    --output data/datasets/movielens_cf.csv
```

---

### Scenario 4: مقایسه Different Aggregations

```bash
#!/bin/bash
# compare_aggregations.sh

# No aggregation
python prepare_data.py --dataset movielens \
    --output data/datasets/movielens_raw.csv

# Hourly
python prepare_data.py --dataset movielens \
    --movielens-aggregate-by hour \
    --output data/datasets/movielens_hourly.csv

# Daily
python prepare_data.py --dataset movielens \
    --movielens-aggregate-by day \
    --output data/datasets/movielens_daily.csv

# Weekly
python prepare_data.py --dataset movielens \
    --movielens-aggregate-by week \
    --output data/datasets/movielens_weekly.csv
```

---

## 📊 آمار Dataset

### حجم داده

```python
import pandas as pd

# خواندن
df = pd.read_csv('data/datasets/movielens_daily.csv')

print(f"Records: {len(df):,}")
print(f"Unique movies: {df['item_id'].nunique():,}")
print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
print(f"Total ratings: {df['count'].sum():,}")
```

**خروجی نمونه:**
```
Records: 2,812,458
Unique movies: 86,537
Date range: 1995-01-09 to 2023-10-16
Total ratings: 32,000,263
```

---

### توزیع Ratings

```python
# اگر rating داشتیم
if 'rating' in df.columns:
    print(df['rating'].describe())
    print(df['rating'].value_counts().sort_index())
```

---

## 🔄 Integration با Framework

### با MovieLensLoader

```python
from data.loaders.movielens_loader import MovieLensLoader

# Load converted data
loader = MovieLensLoader('data/datasets/movielens_daily.csv')
data = loader.load_data()

# Get time series for a movie
movie_id = '123'
ts = loader.get_popularity_time_series(movie_id)
print(ts.head())
```

---

### Evaluation

```bash
python experiments/run_popularity_assessment.py movielens \
    --num-items 1000 \
    --data-path data/datasets/movielens_daily.csv \
    --start-date 2020-01-01 \
    --end-date 2023-12-31 \
    --window-size 30 \
    --incremental
```

---

## 🐛 Troubleshooting

### خطا: "File not found"

```bash
# بررسی مسیر
ls -lh data/raw/movielens/ratings.csv
```

---

### خطا: "Memory error"

```bash
# استفاده از sample
head -n 10000000 data/raw/movielens/ratings.csv > ratings_10m.csv
python prepare_data.py --dataset movielens \
    --input ratings_10m.csv \
    --output movielens_10m.csv
```

---

### Output خیلی بزرگ

```bash
# استفاده از aggregation
python prepare_data.py --dataset movielens \
    --movielens-aggregate-by week  # به جای day
```

---

### Rating distribution نامناسب

```bash
# فیلتر low ratings
python prepare_data.py --dataset movielens \
    --movielens-min-rating 3.0
```

---

## 💡 Best Practices

### 1. همیشه با sample شروع کنید

```bash
# تست سریع
head -n 100000 ratings.csv > ratings_sample.csv
python prepare_data.py --dataset movielens --input ratings_sample.csv ...
```

---

### 2. Config file برای experiments

```yaml
# experiment_1.yaml
dataset: movielens
input: data/raw/movielens/ratings.csv
output: data/datasets/movielens_exp1.csv
converter_params:
  aggregate_by: day
  keep_rating: true
  min_rating: 3.5
```

---

### 3. Validate output

```python
import pandas as pd

df = pd.read_csv('output.csv')
assert df['timestamp'].dtype == 'datetime64[ns]' or df['timestamp'].dtype == 'object'
assert 'item_id' in df.columns
assert 'count' in df.columns
assert df['count'].min() >= 0
print("✓ Validation passed")
```

---

## 📚 مراجع

- **MovieLens Dataset**: https://grouplens.org/datasets/movielens/
- **Paper**: F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets
- **Documentation**: https://files.grouplens.org/datasets/movielens/ml-32m-README.html

---

**تاریخ به‌روزرسانی**: 2026-02-08  
**نسخه**: 2.0
