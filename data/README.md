# Data Directory

این پوشه برای داده‌های خام و تبدیل شده پروژه است.

---

## ⚠️ نکته مهم: داده‌ها در Git نیستند!

**پوشه‌های `data/raw/` و `data/datasets/` در `.gitignore` قرار دارند و در repository آپلود نمی‌شوند.**

**چرا؟**
- 🔴 حجم بالا (چندین GB)
- 🔴 محدودیت GitHub (max 100MB per file)
- 🔴 Performance مشکل می‌شود

**چطور دریافت کنم؟**
- 👉 دیتاست‌ها را از منابع اصلی دانلود کنید (راهنما در پایین)
- 👉 با `prepare_data.py` تبدیل کنید

---

## 📁 ساختار

```
data/
├── raw/              # داده‌های خام (در git نیست)
│   ├── movielens/
│   │   └── ratings.csv
│   ├── youtube07/
│   ├── foursquare/
│   ├── higgs_twitter/
│   └── taxi/
│
├── datasets/         # داده‌های تبدیل شده (در git نیست)
│   ├── movielens.csv
│   ├── youtube07.csv
│   └── ...
│
├── converters/       # کدهای تبدیل (در git هست)
│   ├── __init__.py
│   ├── movielens_converter.py
│   └── ...
│
└── loaders/         # کدهای بارگذاری (در git هست)
    ├── __init__.py
    └── ...
```

---

## 📥 دانلود دیتاست‌ها

### 1. MovieLens (ml-32m)
```bash
# دانلود
wget https://files.grouplens.org/datasets/movielens/ml-32m.zip

# استخراج
unzip ml-32m.zip

# کپی
mkdir -p data/raw/movielens
cp ml-32m/ratings.csv data/raw/movielens/
```

**منبع:** https://grouplens.org/datasets/movielens/

---

### 2. YouTube-07
```bash
mkdir -p data/raw/youtube07
# دانلود از منبع اصلی
```

**منبع:** http://trace.eas.asu.edu/yudata/YouTube-07/

---

### 3. Foursquare
```bash
mkdir -p data/raw/foursquare
# دانلود از منبع اصلی
```

**منبع:** https://sites.google.com/site/yangdingqi/home/foursquare-dataset

---

### 4. Higgs Twitter
```bash
mkdir -p data/raw/higgs_twitter
# دانلود از منبع اصلی
```

**منبع:** https://snap.stanford.edu/data/higgs-twitter.html

---

### 5. NYC Taxi
```bash
mkdir -p data/raw/taxi

# دانلود فایل‌های ماهانه (مثال: سال 2024)
for month in {01..12}; do
    wget "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-${month}.parquet"
done
```

**منبع:** https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page

---

## 🔄 تبدیل داده‌ها

بعد از دانلود، دیتاست‌ها را تبدیل کنید:

```bash
# تک دیتاست
python prepare_data.py --dataset movielens \
                       --input data/raw/movielens/ratings.csv \
                       --output data/datasets/movielens.csv

# همه دیتاست‌ها
python prepare_data.py --all
```

---

## ⚠️ نکات مهم

### 1. Git Ignore
**فایل‌های خام و تبدیل شده در git نیستند!**

- ✅ کدها در git هستند
- ❌ داده‌های خام در git نیستند (`data/raw/`)
- ❌ داده‌های تبدیل شده در git نیستند (`data/datasets/*.csv`)

### 2. حجم
دیتاست‌ها حجیم هستند:
- MovieLens: ~700 MB
- YouTube-07: ~1-2 GB
- NYC Taxi: ~5-10 GB

### 3. License
هر دیتاست دارای مجوز خاص خود است. قبل از استفاده بخوانید.

---

## 📊 وضعیت دیتاست‌ها

| دیتاست | وضعیت | Converter | حجم |
|--------|-------|-----------|-----|
| MovieLens | ✅ آماده | ✅ | 700 MB |
| YouTube-07 | ⏳ در انتظار | ❌ | 1-2 GB |
| Foursquare | ⏳ در انتظار | ❌ | ? |
| Higgs Twitter | ⏳ در انتظار | ❌ | ? |
| NYC Taxi | ⏳ در انتظار | ❌ | 5-10 GB |

---

## 🔍 تست

برای تست، از داده‌های نمونه استفاده کنید:

```bash
# ایجاد داده نمونه MovieLens
python test_movielens_converter.py create_sample

# تبدیل نمونه
python prepare_data.py --dataset movielens \
                       --input data/raw/movielens/ratings_sample.csv \
                       --output data/datasets/movielens_sample.csv
```

---

## 📚 مستندات

برای جزئیات بیشتر هر دیتاست:
- `data/converters/MOVIELENS_GUIDE.md`
- `data/converters/CONVERTER_GUIDE.md`

---

**به‌روزرسانی:** ۱۱ بهمن ۱۴۰۴
