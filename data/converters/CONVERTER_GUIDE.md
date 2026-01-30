# راهنمای توسعه Converter

## 📚 معرفی

این راهنما نحوه ایجاد Converter جدید برای دیتاست‌های مختلف را توضیح می‌دهد.

---

## 🎯 ساختار کلی

هر Converter باید:
1. از `BaseConverter` ارث‌بری کند
2. متد `_convert_single_file()` را پیاده‌سازی کند
3. در `ConverterFactory` ثبت شود

---

## 🚀 مراحل ساخت Converter جدید

### مرحله ۱: ایجاد فایل

```bash
# ساخت فایل جدید
touch data/converters/my_dataset_converter.py
```

### مرحله ۲: پیاده‌سازی کلاس

```python
"""
Converter برای دیتاست من
"""
import pandas as pd
from pathlib import Path
from . import BaseConverter, ConverterFactory


class MyDatasetConverter(BaseConverter):
    """Converter برای دیتاست من"""
    
    def _convert_single_file(self, file_path: Path, **kwargs) -> pd.DataFrame:
        """تبدیل یک فایل"""
        
        # 1. خواندن فایل خام
        raw_df = pd.read_csv(file_path, ...)
        
        # 2. تبدیل timestamp
        raw_df['timestamp'] = pd.to_datetime(raw_df['time_col'])
        
        # 3. استخراج item_id
        raw_df['item_id'] = raw_df['id_col']
        
        # 4. استخراج count (اختیاری)
        raw_df['count'] = raw_df['access_col']
        
        # 5. انتخاب ستون‌های استاندارد
        df = raw_df[['timestamp', 'item_id', 'count']].copy()
        
        # 6. پاکسازی
        df = df.dropna()
        df = df.sort_values('timestamp')
        
        return df

# ثبت در Factory
ConverterFactory.register('my_dataset', MyDatasetConverter)
```

### مرحله ۳: استفاده

```bash
python prepare_data.py --dataset my_dataset \
                       --input data/raw/my_data.txt \
                       --output data/datasets/my_data.csv
```

---

## 📖 سناریوهای مختلف

### سناریو 1: فایل CSV ساده

```python
def _convert_single_file(self, file_path: Path, **kwargs) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['date'])
    df['item_id'] = df['video_id']
    return df[['timestamp', 'item_id']]
```

### سناریو 2: فایل با separator خاص

```python
def _convert_single_file(self, file_path: Path, **kwargs) -> pd.DataFrame:
    df = pd.read_csv(file_path, sep='\t', header=None,
                     names=['time', 'user', 'item', 'rating'])
    df['timestamp'] = pd.to_datetime(df['time'], unit='s')
    df['item_id'] = df['item']
    df['count'] = df['rating']
    return df[['timestamp', 'item_id', 'count']]
```

### سناریو 3: Unix timestamp

```python
def _convert_single_file(self, file_path: Path, **kwargs) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    # تبدیل Unix timestamp (seconds)
    df['timestamp'] = pd.to_datetime(df['unix_time'], unit='s')
    # یا milliseconds
    # df['timestamp'] = pd.to_datetime(df['unix_time'], unit='ms')
    df['item_id'] = df['item']
    return df[['timestamp', 'item_id']]
```

### سناریو 4: فرمت تاریخ خاص

```python
def _convert_single_file(self, file_path: Path, **kwargs) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    # فرمت: "2024-01-15 14:30:45"
    df['timestamp'] = pd.to_datetime(df['date'], 
                                     format='%Y-%m-%d %H:%M:%S')
    df['item_id'] = df['item']
    return df[['timestamp', 'item_id']]
```

### سناریو 5: فایل فشرده

```python
def _convert_single_file(self, file_path: Path, **kwargs) -> pd.DataFrame:
    df = pd.read_csv(file_path, compression='gzip')
    # یا compression='zip', 'bz2', 'xz'
    df['timestamp'] = pd.to_datetime(df['time'])
    df['item_id'] = df['item']
    return df[['timestamp', 'item_id']]
```

### سناریو 6: بدون header

```python
def _convert_single_file(self, file_path: Path, **kwargs) -> pd.DataFrame:
    df = pd.read_csv(file_path, header=None,
                     names=['time', 'item', 'count'])
    df['timestamp'] = pd.to_datetime(df['time'])
    df['item_id'] = df['item']
    return df[['timestamp', 'item_id', 'count']]
```

### سناریو 7: ترکیب تکراری‌ها

```python
def _convert_single_file(self, file_path: Path, **kwargs) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['time'])
    df['item_id'] = df['item']
    df['count'] = 1  # هر رکورد = 1 دسترسی
    
    # ترکیب رکوردهای تکراری
    df = self.aggregate_duplicates(
        df,
        group_by=['timestamp', 'item_id'],
        agg_func='sum'  # جمع count ها
    )
    
    return df
```

### سناریو 8: فیلتر کردن داده‌ها

```python
def _convert_single_file(self, file_path: Path, **kwargs) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['time'])
    df['item_id'] = df['item']
    df['count'] = df['views']
    
    # فیلتر: فقط ویدیوهای با بیش از 100 بازدید
    df = df[df['count'] > 100]
    
    # فیلتر: فقط سال 2024
    df = df[df['timestamp'].dt.year == 2024]
    
    return df[['timestamp', 'item_id', 'count']]
```

### سناریو 9: ستون‌های اضافی

```python
def _convert_single_file(self, file_path: Path, **kwargs) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['time'])
    df['item_id'] = df['video_id']
    df['count'] = df['views']
    
    # ستون‌های اضافی
    df['category'] = df['cat']
    df['uploader'] = df['user_id']
    
    return df[['timestamp', 'item_id', 'count', 'category', 'uploader']]
```

### سناریو 10: چند فایل - NYC Taxi

```python
class NYCTaxiConverter(BaseConverter):
    """
    NYC Taxi - هر ماه یک فایل
    مثال: yellow_tripdata_2024-01.csv, yellow_tripdata_2024-02.csv, ...
    """
    
    def _convert_single_file(self, file_path: Path, **kwargs) -> pd.DataFrame:
        # خواندن فایل یک ماه
        df = pd.read_csv(file_path)
        
        # تبدیل timestamp
        df['timestamp'] = pd.to_datetime(df['tpep_pickup_datetime'])
        
        # location به عنوان item_id
        df['item_id'] = df['PULocationID'].astype(str)
        
        # تعداد سفرها
        df['count'] = 1
        
        # ترکیب: تعداد سفرها از هر مکان در هر ساعت
        df['hour'] = df['timestamp'].dt.floor('H')
        df = df.groupby(['hour', 'item_id'], as_index=False).agg({
            'count': 'sum'
        })
        df.rename(columns={'hour': 'timestamp'}, inplace=True)
        
        return df[['timestamp', 'item_id', 'count']]
    
    # BaseConverter خودش چند فایل را ترکیب می‌کند!
```

استفاده:
```bash
# همه فایل‌های 2024
python prepare_data.py --dataset nyc_taxi \
                       --input "data/raw/taxi/yellow_tripdata_2024-*.csv" \
                       --output data/datasets/nyc_taxi_2024.csv

# فقط چند ماه خاص
python prepare_data.py --dataset nyc_taxi \
                       --input data/raw/taxi/yellow_tripdata_2024-0[1-3].csv \
                       --output data/datasets/nyc_taxi_q1.csv
```

---

## 🔧 توابع کمکی BaseConverter

### aggregate_duplicates()
```python
# ترکیب رکوردهای تکراری
df = self.aggregate_duplicates(
    df,
    group_by=['timestamp', 'item_id'],
    agg_func='sum'  # 'sum', 'count', 'mean'
)
```

### parse_timestamp()
```python
# تبدیل timestamp
timestamp = self.parse_timestamp(
    '2024-01-15 14:30:45',
    format='%Y-%m-%d %H:%M:%S'
)

# Unix timestamp
timestamp = self.parse_timestamp(1705328445, unit='s')
```

### log()
```python
# لاگ پیام
self.log("پردازش آیتم 1000")
```

---

## ✅ چک‌لیست

قبل از commit، مطمئن شوید:

- [ ] کلاس از `BaseConverter` ارث‌بری می‌کند
- [ ] `_convert_single_file()` پیاده‌سازی شده
- [ ] خروجی دارای ستون‌های `timestamp` و `item_id` است
- [ ] `timestamp` از نوع datetime است
- [ ] در `ConverterFactory` ثبت شده
- [ ] در `DATASET_CONFIGS` در `prepare_data.py` اضافه شده
- [ ] تست شده با `python prepare_data.py --dataset your_dataset`
- [ ] مستندات نوشته شده (docstring)

---

## 🐛 عیب‌یابی

### خطا: "Converter not found"
```python
# فراموش نکنید ثبت کنید:
ConverterFactory.register('my_dataset', MyDatasetConverter)
```

### خطا: "timestamp is not datetime"
```python
# مطمئن شوید تبدیل کرده‌اید:
df['timestamp'] = pd.to_datetime(df['time_col'])
```

### خطا: "Missing required columns"
```python
# حتماً این ستون‌ها باشند:
return df[['timestamp', 'item_id']]
# یا با count:
return df[['timestamp', 'item_id', 'count']]
```

