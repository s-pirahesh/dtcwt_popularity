# راهنمای اجرای کامل — WSPI (نسخهٔ نهایی، اجرای موازی)

این راهنما همهٔ دستورهای لازم برای اجرای دوبارهٔ کلِ آزمایش‌ها با فرمولِ نهاییِ WSPI را دارد. همه‌چیز **موازی** و **resumable** است و خروجی هر اجرا در **یک پوشهٔ واحد** می‌نشیند.

---

## فرمولِ نهایی

```
WSPI = μ_L · exp(α·R − β·WE)        α = β = 1
```
جملهٔ شیب (S_L) و عملگرِ clip حذف شده‌اند (با مدرک، در `methods/wspi_assessment.py` توضیح داده شده).

---

## ۱) جای‌گذاری فایل‌ها

این فایل‌ها را در مسیرهای زیر بگذار (جایگزین/جدید):

```
methods\wspi_assessment.py            (جدید — کلاسِ کانونیِ WSPI)
experiments\parallel_engine.py        (جدید — موتورِ موازیِ مشترک)
experiments\run_popularity_assessment.py   (ویرایش‌شده — حالا --parallel دارد)
experiments\run_wspi_ablation.py      (جدید — ابلیشن موازی)
experiments\run_wspi_sensitivity.py   (جدید — حساسیت موازی)
```

اگر ترجیح می‌دهی `run_popularity_assessment.py` را دستی وصله کنی به‌جای جایگزینی، چهار تغییر این‌هاست:
1. بالای فایل، قبل از import‌ها: یک محافظِ `sys.stdout.reconfigure(encoding='utf-8')`.
2. در import متدها: `from methods.wspi_assessment import WSPIAssessment`.
3. در `create_methods_dict`، ساختِ `methods['WSPI']` به `WSPIAssessment(alpha=1.0, beta=1.0)` تغییر کرده.
4. در `main()`: یک شاخهٔ `--parallel` که به `parallel_engine.run_methods_parallel` می‌رود + آرگومانِ `--fresh`.

> فایل‌های قدیمیِ exploratory (`wspi_fusion.py`, `wspi_candidates.py`, `run_fusion_candidates.py`, `run_parallel.py`) دیگر برای اجرای نهایی لازم نیستند. نگه‌داشتنشان ضرری ندارد.

---

## ۲) قبل از اجرای کامل — یک تستِ کوتاه (مهم)

اول روی ماشینِ خودت با چند آیتم تست کن تا مطمئن شوی همه‌چیز سبز است:
```
python experiments\run_popularity_assessment.py youtube --num-items 100 --cores 3
```
اگر بدونِ خطا تمام شد و جدولِ خلاصه را چاپ کرد، برو سراغ اجرای کامل.

---

## ۳) اجرای اصلیِ سناریوها (۹ روش، موازی، یک پوشه)

`--parallel` به‌صورت پیش‌فرض روشن است (برای سریال: `--no-parallel`). `--cores` را به تعداد هسته‌های ماشینت بگذار.

**YouTube (ساعتی):**
```
python experiments\run_popularity_assessment.py youtube --cores 8
```

**Yellow Taxi (ساعتی):**
```
python experiments\run_popularity_assessment.py yellow_taxi --cores 8 --data-path data\datasets\yellow_taxi_2025_all_hourly.csv
```

**دانه‌بندی‌های ریز (۳۰ و ۵ دقیقه)** — اگر فایلِ resample‌شده را داری، فقط `--data-path` را عوض کن:
```
python experiments\run_popularity_assessment.py yellow_taxi --cores 8 --data-path data\datasets\yellow_taxi_2025_all_30min.csv
python experiments\run_popularity_assessment.py yellow_taxi --cores 8 --data-path data\datasets\yellow_taxi_2025_all_5min.csv
```

خروجی هر اجرا:
```
results\<dataset>\main_<timestamp>\                       ← پوشهٔ واحد
        ├── detailed\ protocol\ summary\ metadata\
        └── comparison\main_summary.csv                  ← خلاصهٔ همهٔ روش‌ها
results\tables\main_<dataset>_<timestamp>.csv            ← یک کپیِ خلاصه
```

---

## ۴) ابلیشن (موازی) — مورد ۵ داور

```
python experiments\run_wspi_ablation.py youtube --cores 8
python experiments\run_wspi_ablation.py yellow_taxi --cores 8 --data-path data\datasets\yellow_taxi_2025_all_hourly.csv
```
واریانت‌ها: `WSPI` (کامل)، `WSPI-noR`، `WSPI-noWE`، `WSPI-DWT`.
خلاصه در `results\<dataset>\ablation_<ts>\comparison\ablation_summary.csv`.

تفسیر: حذف R یا WE باید عملکرد را بدتر کند (سهمِ واقعی)؛ `WSPI-DWT` باید RSI را پایین و ΔRank را بالا ببرد (اهمیتِ shift-invariance).

---

## ۵) حساسیتِ ضرایب (موازی) — مورد ۴ داور

```
python experiments\run_wspi_sensitivity.py youtube --cores 8
python experiments\run_wspi_sensitivity.py yellow_taxi --cores 8 --data-path data\datasets\yellow_taxi_2025_all_hourly.csv
```
α را در {۰، ۰.۲۵، ۰.۵، ۱، ۱.۵، ۲} (با β=۱) و β را در همان مقادیر (با α=۱) جارو می‌کند.
خلاصه در `results\<dataset>\sensitivity_<ts>\comparison\sensitivity_summary.csv`.

تفسیر: نشان می‌دهد عملکرد حولِ α=β=۱ پایدار است → انتخابِ ضرایب توجیهِ اصولی دارد (نه tuning روی داده).

---

## ۶) resume و قطعِ برق/ری‌استارت

- هر اجرا **resumable** است: هر روشی که تمام شود یک marker می‌گیرد. اگر ویندوز ری‌استارت شد، فقط روش‌های در-حال-اجرا از بین می‌روند.
- **همان دستور را دوباره بزن** — از همان پوشه ادامه می‌دهد و فقط روش‌های ناتمام را از نو اجرا می‌کند.
- برای شروعِ کاملاً از صفر (پوشهٔ جدید، نادیده‌گرفتنِ markerها): `--fresh` را اضافه کن.
- markerها و لاگِ هر روش این‌جاست: `results\_parallel_state\<dataset>_<tag>\`. اگر روشی failed شد، لاگش (`<method>.log`) خطا را نشان می‌دهد.

برای اطمینان بیشتر، حین اجرا آپدیتِ ویندوز را Pause کن و خوابِ سیستم را خاموش کن:
```
powercfg /change standby-timeout-ac 0
powercfg /change monitor-timeout-ac 0
```

---

## ۷) ترتیب پیشنهادی

1. تستِ کوتاه (`--num-items 100`).
2. اجرای اصلیِ هر چهار سناریو (بخش ۳).
3. ابلیشن (بخش ۴).
4. حساسیت (بخش ۵).

می‌توانی اجراهای دیتاست‌های مختلف را در ترمینال‌های جدا هم‌زمان بزنی (هر کدام پوشه و state جدا دارد). وقتی CSVهای خلاصه آماده شد، تحلیلِ نهایی و نگارشِ v4.6 را با هم انجام می‌دهیم.
