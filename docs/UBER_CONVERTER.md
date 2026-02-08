# راهنمای Uber/NYC Yellow Taxi Converter

## 📚 معرفی

Uber Converter برای تبدیل داده‌های NYC Yellow Taxi Trip Records به فرمت استاندارد برای تحلیل location popularity طراحی شده است.

**Dataset**: NYC Yellow Taxi Trip Records  
**Source**: NYC TLC (https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)  
**Format**: Parquet files  
**Update**: Monthly  
**Size**: ~3-4M trips per month

---

## 📊 فرمت داده

### Input (Parquet)

**ستون‌های کلیدی:**
```
tpep_pickup_datetime    : زمان pickup (datetime)
PULocationID           : شناسه منطقه pickup (1-263)
fare_amount            : کرایه (دلار)
trip_distance          : مسافت (مایل)
passenger_count        : تعداد مسافر
total_amount           : هزینه کل
```

**مثال رکوردها:**
```
tpep_pickup_datetime    PULocationID  fare_amount  trip_distance  passenger_count
2024-01-01 00:05:23     107           15.50        3.2            1
2024-01-01 00:12:45     161           22.00        5.8            2
```

---

### Output (استاندارد)

**بدون features:**
```csv
timestamp,location_id,item_id,count
2024-01-01 00:00:00,107,107,45
2024-01-01 00:00:00,161,52,52
2024-01-01 00:15:00,107,107,38
```

**با features:**
```csv
timestamp,location_id,item_id,count,fare_amount,trip_distance,passenger_count
2024-01-01 00:00:00,107,107,45,15.23,3.45,1.8
2024-01-01 00:00:00,161,161,52,18.50,4.12,1.9
```

**توضیح ستون‌ها:**
- `timestamp`: زمان (aggregated به time slot)
- `location_id`: شناسه zone (1-263)
- `item_id`: همان location_id (برای سازگاری با framework)
- `count`: تعداد trips در این time slot
- `fare_amount`: میانگین کرایه (optional)
- `trip_distance`: میانگین مسافت (optional)
- `passenger_count`: میانگین تعداد مسافر (optional)

---

## 🔧 Parameters

### `--uber-granularity`

**نوع**: string  
**مقادیر مجاز**: `5min`, `15min`, `30min`, `hourly`, `daily`  
**پیش‌فرض**: `15min`

**توضیح**: اندازه time slot برای تجمیع trips

**مقایسه granularities:**

| Granularity | Slots/Day | Window Size (1 day) | Use Case |
|-------------|-----------|---------------------|----------|
| 5min | 288 | 288 slots | Very fine-grained, short-term patterns |
| **15min** | 96 | 96 slots | **پیشنهادی**: تعادل خوب |
| 30min | 48 | 48 slots | Medium granularity |
| hourly | 24 | 24 slots | تحلیل روزانه |
| daily | 1 | 30-90 days | Long-term trends |

**استفاده:**

```bash
# 15-minute (پیشنهادی)
python prepare_data.py --dataset uber \
    --uber-granularity 15min

# Hourly (برای window بزرگ‌تر)
python prepare_data.py --dataset uber \
    --uber-granularity hourly

# Daily (برای تحلیل بلندمدت)
python prepare_data.py --dataset uber \
    --uber-granularity daily
```

**تأثیر روی output:**

| Input | Granularity | Output Records | Reduction |
|-------|-------------|----------------|-----------|
| 3M trips/month | 5min | ~1.2M | 60% |
| 3M trips/month | 15min | ~400K | 87% |
| 3M trips/month | hourly | ~150K | 95% |
| 3M trips/month | daily | ~8K | 99.7% |

**پیشنهاد برای PhD:**
- تست اولیه: `hourly` (سریع)
- Evaluation نهایی: `15min` (بهترین تعادل)

---

### `--uber-min-trips-per-location`

**نوع**: integer  
**پیش‌فرض**: `100`  
**محدوده معمول**: 50-1000

**توضیح**: locations با کمتر از این تعداد trip (total) حذف می‌شوند

**منطق**: 
- NYC دارای 263 taxi zone است
- بسیاری از zones کم‌استفاده هستند (مثلاً فرودگاه‌ها، مناطق صنعتی)
- فیلتر کردن این zones → داده تمیزتر و تحلیل دقیق‌تر

**تأثیر:**

| min_trips | Locations Kept (از 263) | Data Kept |
|-----------|--------------------------|-----------|
| 50 | ~240 | 99.5% |
| 100 | ~220 | 99.2% |
| 200 | ~200 | 98.5% |
| 500 | ~150 | 97% |
| 1000 | ~100 | 95% |

**استفاده:**

```bash
# تست (keep بیشتر zones)
python prepare_data.py --dataset uber \
    --uber-min-trips-per-location 50

# Production (فوکوس روی active zones)
python prepare_data.py --dataset uber \
    --uber-min-trips-per-location 500
```

**پیشنهاد:**
- تست: 50-100
- Final evaluation: 200-500

---

### `--uber-extract-features`

**نوع**: flag (boolean)  
**پیش‌فرض**: `False`

**توضیح**: استخراج features اضافی (میانگین fare, distance, passenger count)

**Features استخراج‌شده:**
- `fare_amount`: میانگین کرایه (دلار)
- `trip_distance`: میانگین مسافت (مایل)
- `passenger_count`: میانگین تعداد مسافر
- `total_amount`: میانگین هزینه کل (optional)

**استفاده:**

```bash
# بدون features (فقط count)
python prepare_data.py --dataset uber

# با features
python prepare_data.py --dataset uber \
    --uber-extract-features
```

**کاربرد features:**

1. **Baseline Methods**: 
   - ARIMA با exogenous variables
   - Regression models

2. **Feature Engineering**:
   - Correlation analysis (fare vs popularity)
   - Peak hour detection (passenger count patterns)

3. **Dataset Understanding**:
   - توزیع کرایه در مناطق مختلف
   - الگوهای مسافت

**هشدار**: با features، حجم فایل ~30-40% بزرگ‌تر می‌شود.

---

## 📖 Use Cases

### Use Case 1: تست سریع (یک ماه)

```bash
# دانلود
wget https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet

# تبدیل
python prepare_data.py --dataset uber \
    --input yellow_tripdata_2024-01.parquet \
    --output data/datasets/uber_jan_test.csv \
    --uber-granularity hourly \
    --uber-min-trips-per-location 50
```

**نتیجه**: ~150K records, ~200 locations

---

### Use Case 2: یک ماه با 15min (پیشنهادی)

```bash
python prepare_data.py --dataset uber \
    --input yellow_tripdata_2024-01.parquet \
    --output data/datasets/uber_jan_15min.csv \
    --uber-granularity 15min \
    --uber-min-trips-per-location 200
```

**نتیجه**: ~400K records, ~200 locations  
**Window size پیشنهادی**: 96 slots (24 hours)

---

### Use Case 3: سه‌ماهه (Q1)

```bash
# دانلود
for month in {01..03}; do
    wget https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-${month}.parquet
done

# تبدیل
python prepare_data.py --dataset uber \
    --input "yellow_tripdata_2024-*.parquet" \
    --output data/datasets/uber_q1_hourly.csv \
    --uber-granularity hourly \
    --uber-min-trips-per-location 500
```

**نتیجه**: ~500K records, ~150 locations  
**Window size پیشنهادی**: 168 slots (7 days)

---

### Use Case 4: با Features (برای baselines)

```bash
python prepare_data.py --dataset uber \
    --input "yellow_tripdata_2024-*.parquet" \
    --output data/datasets/uber_q1_features.csv \
    --uber-granularity hourly \
    --uber-min-trips-per-location 300 \
    --uber-extract-features
```

**Output:**
```csv
timestamp,location_id,item_id,count,fare_amount,trip_distance,passenger_count
2024-01-01 00:00:00,107,107,45,15.23,3.45,1.8
```

**کاربرد**: regression models, feature importance analysis

---

## 🎯 پیشنهادات برای PhD Research

### Scenario 1: Initial Testing

```bash
# یک هفته برای test
python prepare_data.py --dataset uber \
    --input yellow_tripdata_2024-01.parquet \
    --output data/datasets/uber_week1.csv \
    --uber-granularity 15min \
    --uber-min-trips-per-location 50

# فیلتر در Python برای یک هفته
import pandas as pd
df = pd.read_csv('data/datasets/uber_week1.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df[(df['timestamp'] >= '2024-01-01') & (df['timestamp'] < '2024-01-08')]
df.to_csv('data/datasets/uber_week1_filtered.csv', index=False)
```

---

### Scenario 2: DTCWT Evaluation

```bash
# یک ماه، 15min
python prepare_data.py --dataset uber \
    --uber-granularity 15min \
    --uber-min-trips-per-location 200 \
    --output data/datasets/uber_dtcwt.csv
```

**Window sizes پیشنهادی:**
- Short-term: 96 slots (24h)
- Medium-term: 672 slots (7 days)
- Long-term: 2880 slots (30 days)

---

### Scenario 3: Baseline Comparison

```bash
# DWT+AF, DTCWT+AF (count-based)
python prepare_data.py --dataset uber \
    --uber-granularity 15min \
    --uber-min-trips-per-location 500 \
    --output data/datasets/uber_baselines.csv

# ARMA, LSTM (با features)
python prepare_data.py --dataset uber \
    --uber-granularity hourly \
    --uber-min-trips-per-location 500 \
    --uber-extract-features \
    --output data/datasets/uber_baselines_features.csv
```

---

### Scenario 4: مقایسه Granularities

```bash
#!/bin/bash
# compare_granularities.sh

MONTHS="yellow_tripdata_2024-0[1-3].parquet"

# 15min
python prepare_data.py --dataset uber \
    --input "$MONTHS" \
    --output data/datasets/uber_q1_15min.csv \
    --uber-granularity 15min \
    --uber-min-trips-per-location 500

# Hourly
python prepare_data.py --dataset uber \
    --input "$MONTHS" \
    --output data/datasets/uber_q1_hourly.csv \
    --uber-granularity hourly \
    --uber-min-trips-per-location 500

# Daily
python prepare_data.py --dataset uber \
    --input "$MONTHS" \
    --output data/datasets/uber_q1_daily.csv \
    --uber-granularity daily \
    --uber-min-trips-per-location 500
```

---

## 📊 آمار Dataset

### دانلود داده

```bash
# لیست فایل‌های موجود
# https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page

# مثال: ژانویه 2024
wget https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet
```

### بررسی Output

```python
import pandas as pd
import numpy as np

# خواندن
df = pd.read_csv('data/datasets/uber_15min.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

print("=" * 60)
print("Dataset Statistics")
print("=" * 60)
print(f"Total records: {len(df):,}")
print(f"Unique locations: {df['location_id'].nunique()}")
print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
print(f"Days: {(df['timestamp'].max() - df['timestamp'].min()).days}")
print(f"Total trips: {df['count'].sum():,}")
print(f"Avg trips/slot: {df['count'].mean():.2f}")

# Top locations
top_locs = df.groupby('location_id')['count'].sum().sort_values(ascending=False).head(10)
print("\nTop 10 Locations:")
print(top_locs)

# Temporal patterns
df['hour'] = df['timestamp'].dt.hour
hourly_avg = df.groupby('hour')['count'].mean()
print("\nAverage trips by hour:")
print(hourly_avg)

# Visualize
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))
hourly_avg.plot(kind='bar')
plt.title('Average Trips by Hour of Day')
plt.xlabel('Hour')
plt.ylabel('Average Trips')
plt.tight_layout()
plt.savefig('uber_hourly_pattern.png')
print("\nPlot saved: uber_hourly_pattern.png")
```

---

### Taxi Zones (NYC)

NYC دارای 263 taxi zone است. برخی مهم‌ترین‌ها:

**Manhattan (high activity):**
- 13: Battery Park
- 87: Chinatown  
- 107: East Harlem South
- 161: Midtown Center
- 237: Upper East Side South

**Airports:**
- 132: JFK Airport
- 138: LaGuardia Airport

**بررسی zones:**
```bash
# دانلود shapefile
wget https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip
unzip taxi_zones.zip
```

---

## 🔄 Integration با Framework

### با UberLoader

```python
from data.loaders.uber_loader import UberLoader

# Load converted data
loader = UberLoader('data/datasets/uber_15min.csv')
data = loader.load_data()

# Get time series for a location
location_id = 107  # East Harlem South
ts = loader.get_popularity_time_series(location_id)

print(ts.head(20))
```

---

### Evaluation

```bash
python experiments/run_popularity_assessment.py uber \
    --num-items 500 \
    --data-path data/datasets/uber_15min.csv \
    --start-date 2024-01-01 \
    --end-date 2024-01-31 \
    --window-size 96 \
    --incremental
```

**پارامترهای پیشنهادی:**
- `--window-size 96`: 24 hours در 15min slots
- `--window-size 168`: 7 days در hourly slots
- `--num-items 500`: top 500 locations

---

## 🐛 Troubleshooting

### خطا: "pyarrow not found"

```bash
pip install pyarrow --break-system-packages
```

---

### خطا: "No files found"

```bash
# در Linux/Mac
ls data/raw/uber/*.parquet

# در Windows
dir data\raw\uber\*.parquet

# wildcard در quotes
python prepare_data.py --input "data/raw/uber/*.parquet" ...
```

---

### خطا: "Missing required columns"

فایل باید Yellow Taxi Parquet باشد (نه Green Taxi یا FHV).

**بررسی:**
```python
import pyarrow.parquet as pq
table = pq.read_table('file.parquet')
print(table.column_names)
# باید شامل: tpep_pickup_datetime, PULocationID
```

---

### Memory Error

```bash
# استفاده از یک فایل کوچک‌تر
python prepare_data.py --dataset uber \
    --input yellow_tripdata_2024-01.parquet \  # فقط یک ماه
    --uber-granularity hourly  # granularity بزرگ‌تر
```

---

### Output خیلی بزرگ

```bash
# افزایش min_trips
--uber-min-trips-per-location 1000

# granularity بزرگ‌تر
--uber-granularity daily

# فیلتر زمانی (بعد از convert)
import pandas as pd
df = pd.read_csv('output.csv')
df = df[df['timestamp'] >= '2024-01-01']
df = df[df['timestamp'] < '2024-02-01']
df.to_csv('filtered.csv', index=False)
```

---

## 💡 Best Practices

### 1. شروع با یک ماه

```bash
# اول یک ماه test
python prepare_data.py --dataset uber \
    --input yellow_2024_01.parquet \
    --output test.csv

# بعد scale up
python prepare_data.py --dataset uber \
    --input "yellow_2024-*.parquet" \
    --output full.csv
```

---

### 2. Config file برای experiments

```yaml
# uber_exp1.yaml
dataset: uber
input: data/raw/uber/yellow_2024-*.parquet
output: data/datasets/uber_exp1.csv
converter_params:
  granularity: 15min
  min_trips_per_location: 300
  extract_features: false
```

---

### 3. Validate temporal patterns

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('uber_15min.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['hour'] = df['timestamp'].dt.hour
df['dayofweek'] = df['timestamp'].dt.dayofweek

# Heatmap: hour vs dayofweek
pivot = df.groupby(['hour', 'dayofweek'])['count'].mean().unstack()
plt.figure(figsize=(10, 6))
plt.imshow(pivot, cmap='YlOrRd', aspect='auto')
plt.colorbar(label='Avg Trips')
plt.xlabel('Day of Week (0=Monday)')
plt.ylabel('Hour of Day')
plt.title('NYC Taxi Trips Heatmap')
plt.tight_layout()
plt.savefig('uber_heatmap.png')
```

---

## 📚 مراجع

- **NYC TLC Data**: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
- **Data Dictionary**: https://www.nyc.gov/assets/tlc/downloads/pdf/data_dictionary_trip_records_yellow.pdf
- **Taxi Zones**: https://data.cityofnewyork.us/Transportation/NYC-Taxi-Zones/d3c5-ddgc

---

**تاریخ به‌روزرسانی**: 2026-02-08  
**نسخه**: 2.0
