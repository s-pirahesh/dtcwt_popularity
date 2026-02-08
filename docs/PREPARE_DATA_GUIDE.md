# راهنمای جامع prepare_data.py

## 📚 معرفی

`prepare_data.py` نقطه ورودی اصلی برای تبدیل دیتاست‌های خام به فرمت استاندارد است.

**فرمت استاندارد خروجی:**
```csv
timestamp,item_id,count
2024-01-01 00:00:00,123,45
2024-01-01 00:00:00,456,38
```

---

## 🎯 ویژگی‌های کلیدی

### ✨ Dynamic Parameter System
- هر دیتاست parameters اختصاصی خودش را دارد
- با prefix مشخص می‌شوند (`--movielens-`, `--uber-`)
- بدون conflict بین datasets مختلف

### ✨ Config File Support
- استفاده از YAML برای تنظیمات
- Reproducible و version controllable
- مناسب برای experiments

### ✨ Multi-File Support
- پردازش چند فایل به صورت همزمان
- Wildcard support (`*.parquet`, `ratings_*.csv`)
- Auto-merge و deduplication

---

## 🚀 استفاده سریع

### لیست دیتاست‌های موجود

```bash
python prepare_data.py --list
```

**خروجی:**
```
======================================================================
دیتاست‌های قابل تبدیل:
======================================================================

📊 MOVIELENS
   MovieLens ratings dataset converter
   Input (default):  data/raw/movielens/ratings.csv
   Output (default): data/datasets/movielens.csv
   Parameters:
     --movielens-aggregate-by: Temporal aggregation level (default: None)
     --movielens-keep-rating: Keep rating column in output (default: False)
     --movielens-keep-user: Keep user ID column in output (default: False)
     --movielens-min-rating: Minimum rating to include (default: None)

📊 UBER
   NYC Yellow Taxi trip data converter
   Input (default):  data/raw/uber/yellow_*.parquet
   Output (default): data/datasets/uber_15min.csv
   Parameters:
     --uber-granularity: Time slot size for aggregation (default: 15min)
     --uber-min-trips-per-location: Minimum total trips (default: 100)
     --uber-extract-features: Extract additional features (default: False)

======================================================================
Total: 2 datasets
======================================================================
```

---

## 📖 راهنمای استفاده

### 1️⃣ استفاده پایه (با مقادیر پیش‌فرض)

```bash
# MovieLens - استفاده از input/output پیش‌فرض
python prepare_data.py --dataset movielens

# Uber - مشخص کردن input/output
python prepare_data.py --dataset uber \
    --input data/raw/uber/yellow_2024_01.parquet \
    --output data/datasets/uber_jan.csv
```

---

### 2️⃣ استفاده با Parameters

#### MovieLens

```bash
# با تجمیع روزانه
python prepare_data.py --dataset movielens \
    --movielens-aggregate-by day

# نگهداری rating و فیلتر >= 4.0
python prepare_data.py --dataset movielens \
    --movielens-keep-rating \
    --movielens-min-rating 4.0

# کامل: aggregation + rating + user
python prepare_data.py --dataset movielens \
    --input data/raw/movielens/ratings.csv \
    --output data/datasets/movielens_daily_rated.csv \
    --movielens-aggregate-by day \
    --movielens-keep-rating \
    --movielens-keep-user \
    --movielens-min-rating 3.0
```

#### Uber

```bash
# با hourly aggregation
python prepare_data.py --dataset uber \
    --uber-granularity hourly

# با استخراج features
python prepare_data.py --dataset uber \
    --uber-extract-features \
    --uber-min-trips-per-location 200

# کامل: چند فایل + features
python prepare_data.py --dataset uber \
    --input "data/raw/uber/yellow_2024-*.parquet" \
    --output data/datasets/uber_q1_hourly.csv \
    --uber-granularity hourly \
    --uber-min-trips-per-location 500 \
    --uber-extract-features
```

---

### 3️⃣ استفاده از Config File

#### ایجاد Config File

**configs/movielens_daily.yaml:**
```yaml
dataset: movielens
input: data/raw/movielens/ratings.csv
output: data/datasets/movielens_daily.csv

converter_params:
  aggregate_by: day
  keep_rating: true
  keep_user: false
  min_rating: 3.5
```

**configs/uber_hourly.yaml:**
```yaml
dataset: uber
input: data/raw/uber/yellow_2024-*.parquet
output: data/datasets/uber_2024_hourly.csv

converter_params:
  granularity: hourly
  min_trips_per_location: 300
  extract_features: true
```

#### اجرا از Config

```bash
# MovieLens
python prepare_data.py --config configs/movielens_daily.yaml

# Uber
python prepare_data.py --config configs/uber_hourly.yaml

# Override پارامترها
python prepare_data.py --config configs/uber_hourly.yaml \
    --uber-min-trips-per-location 500
```

---

### 4️⃣ تبدیل همه دیتاست‌ها

```bash
# تبدیل تمام datasets با تنظیمات پیش‌فرض
python prepare_data.py --all

# با format خاص
python prepare_data.py --all --output-format parquet
```

---

## 🔧 Parameters عمومی (Generic)

این parameters برای **همه** datasets اعمال می‌شوند:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--output-format` | str | csv | فرمت خروجی (csv/parquet/feather) |
| `--quiet` | flag | False | حالت ساکت (بدون پیام) |
| `--no-validate` | flag | False | غیرفعال کردن اعتبارسنجی |

**مثال:**
```bash
python prepare_data.py --dataset movielens \
    --output-format parquet \
    --quiet
```

---

## 📊 Parameters اختصاصی Datasets

### MovieLens

| Parameter | Type | Choices | Default | Description |
|-----------|------|---------|---------|-------------|
| `--movielens-aggregate-by` | str | hour/day/week/none | None | تجمیع زمانی |
| `--movielens-keep-rating` | flag | - | False | نگهداری ستون rating |
| `--movielens-keep-user` | flag | - | False | نگهداری ستون user_id |
| `--movielens-min-rating` | float | - | None | حداقل rating (فیلتر) |

**Output بدون parameters:**
```csv
timestamp,item_id,count
2024-01-01 12:34:56,123,1
```

**Output با `--keep-rating`:**
```csv
timestamp,item_id,count,rating
2024-01-01 12:34:56,123,1,4.5
```

**Output با `--aggregate-by day --keep-rating`:**
```csv
timestamp,item_id,count,rating
2024-01-01 00:00:00,123,245,4.3
```

---

### Uber

| Parameter | Type | Choices | Default | Description |
|-----------|------|---------|---------|-------------|
| `--uber-granularity` | str | 5min/15min/30min/hourly/daily | 15min | اندازه time slot |
| `--uber-min-trips-per-location` | int | - | 100 | حداقل سفرها (فیلتر) |
| `--uber-extract-features` | flag | - | False | استخراج features اضافی |

**Output بدون features:**
```csv
timestamp,location_id,item_id,count
2024-01-01 00:00:00,107,107,45
```

**Output با `--extract-features`:**
```csv
timestamp,location_id,item_id,count,fare_amount,trip_distance,passenger_count
2024-01-01 00:00:00,107,107,45,15.23,3.45,1.8
```

---

## 🎯 Use Cases متداول

### Use Case 1: تست سریع با یک ماه داده

```bash
# MovieLens - sample
python prepare_data.py --dataset movielens \
    --input data/raw/movielens/ratings_sample.csv \
    --output data/datasets/movielens_test.csv

# Uber - یک ماه
python prepare_data.py --dataset uber \
    --input data/raw/uber/yellow_2024_01.parquet \
    --output data/datasets/uber_jan_test.csv \
    --uber-granularity hourly \
    --uber-min-trips-per-location 50
```

---

### Use Case 2: Evaluation آماده‌سازی

```bash
# MovieLens با aggregation برای DTCWT
python prepare_data.py --dataset movielens \
    --movielens-aggregate-by day \
    --output data/datasets/movielens_daily_for_eval.csv

# Uber برای evaluation
python prepare_data.py --dataset uber \
    --input "data/raw/uber/yellow_2024-0[1-3].parquet" \
    --output data/datasets/uber_q1_15min.csv \
    --uber-granularity 15min \
    --uber-min-trips-per-location 500
```

---

### Use Case 3: تحلیل با Features

```bash
# Uber با features برای baseline comparison
python prepare_data.py --dataset uber \
    --uber-extract-features \
    --uber-granularity hourly \
    --output data/datasets/uber_with_features.csv
```

---

### Use Case 4: Batch Processing

```bash
#!/bin/bash
# process_all.sh

# MovieLens variants
python prepare_data.py --config configs/movielens_basic.yaml
python prepare_data.py --config configs/movielens_daily.yaml
python prepare_data.py --config configs/movielens_high_rated.yaml

# Uber variants
python prepare_data.py --config configs/uber_15min.yaml
python prepare_data.py --config configs/uber_hourly.yaml
python prepare_data.py --config configs/uber_daily.yaml
```

---

## 📁 ساختار فایل‌ها

### Input Files

```
data/raw/
├── movielens/
│   ├── ratings.csv
│   └── movies.csv
│
└── uber/
    ├── yellow_tripdata_2024-01.parquet
    ├── yellow_tripdata_2024-02.parquet
    └── yellow_tripdata_2024-03.parquet
```

### Output Files

```
data/datasets/
├── movielens.csv
├── movielens_daily.csv
├── movielens_high_rated.csv
├── uber_15min.csv
├── uber_hourly.csv
└── uber_with_features.csv
```

### Config Files

```
configs/
├── movielens_basic.yaml
├── movielens_daily.yaml
├── movielens_high_rated.yaml
├── uber_15min.yaml
├── uber_hourly.yaml
└── uber_daily.yaml
```

---

## 🐛 Troubleshooting

### خطا: "Converter not found"

```bash
$ python prepare_data.py --dataset youtube
❌ خطا: Converter برای 'youtube' یافت نشد
```

**راه‌حل:**
```bash
# لیست converters موجود
python prepare_data.py --list
```

---

### خطا: "No files found"

```bash
$ python prepare_data.py --dataset uber --input "data/raw/uber/*.parquet"
❌ هیچ فایلی با الگوی 'data/raw/uber/*.parquet' یافت نشد!
```

**راه‌حل:**
```bash
# بررسی وجود فایل‌ها
ls data/raw/uber/*.parquet

# در Windows از quotes استفاده کنید
python prepare_data.py --dataset uber --input "data\raw\uber\*.parquet"
```

---

### خطا: "Missing required columns"

```bash
❌ خطا: Missing required columns: ['tpep_pickup_datetime']
```

**راه‌حل:**
- فایل input را بررسی کنید
- مطمئن شوید فرمت صحیح است
- برای Uber: فقط Yellow Taxi Parquet files

---

### Warning: "Null values found"

```bash
⚠️  Null values found:
timestamp    0
item_id      5
```

**راه‌حل:**
- این warning معمولاً مشکلی ایجاد نمی‌کند
- رکوردهای null به صورت خودکار حذف می‌شوند
- برای بررسی بیشتر: `--no-validate` را حذف کنید

---

## 💡 Best Practices

### 1. استفاده از Config Files برای Production

```yaml
# ✅ خوب: reproducible و documented
dataset: uber
input: data/raw/uber/yellow_2024-*.parquet
output: data/datasets/uber_hourly_final.csv
converter_params:
  granularity: hourly
  min_trips_per_location: 500
  extract_features: false

# ❌ بد: hard to reproduce
$ python prepare_data.py --dataset uber --uber-granularity hourly ...
```

---

### 2. تست با Sample Data اول

```bash
# اول sample
python prepare_data.py --dataset uber \
    --input data/raw/uber/yellow_2024_01.parquet \
    --output data/datasets/uber_test.csv

# سپس full dataset
python prepare_data.py --dataset uber \
    --input "data/raw/uber/yellow_2024-*.parquet" \
    --output data/datasets/uber_full.csv
```

---

### 3. Version Control برای Configs

```bash
git add configs/
git commit -m "Add Uber hourly conversion config"
```

---

### 4. Logging Output

```bash
# ذخیره logs
python prepare_data.py --config configs/uber_hourly.yaml 2>&1 | tee logs/uber_conversion.log
```

---

## 📚 منابع اضافی

- **راهنمای MovieLens Converter**: `docs/converters/MOVIELENS_CONVERTER.md`
- **راهنمای Uber Converter**: `docs/converters/UBER_CONVERTER.md`
- **راهنمای توسعه Converter جدید**: `docs/CONVERTER_DEVELOPMENT.md`

---

## 🔄 Workflow پیشنهادی

```bash
# 1. لیست datasets
python prepare_data.py --list

# 2. تست با sample
python prepare_data.py --dataset uber \
    --input data/raw/uber/yellow_2024_01.parquet \
    --output data/datasets/test.csv \
    --uber-min-trips-per-location 50

# 3. بررسی output
head -20 data/datasets/test.csv

# 4. ایجاد config نهایی
cat > configs/uber_final.yaml << EOF
dataset: uber
input: data/raw/uber/yellow_2024-*.parquet
output: data/datasets/uber_15min_final.csv
converter_params:
  granularity: 15min
  min_trips_per_location: 200
  extract_features: false
EOF

# 5. اجرای نهایی
python prepare_data.py --config configs/uber_final.yaml

# 6. validation
python -c "import pandas as pd; df = pd.read_csv('data/datasets/uber_15min_final.csv'); print(df.info()); print(df.head())"
```

---

**تاریخ به‌روزرسانی**: 2026-02-08  
**نسخه**: 2.0
