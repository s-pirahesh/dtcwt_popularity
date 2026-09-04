# راهنمای اجرای آزمایش‌های پیش‌بینی و رسم نمودارها — نسخه ۲

**پروژه:** `D:\Research\Thesis Research\Simulations\Popularity With Wavelets\Dtcwt popularity\dtcwt_popularity`
**به‌روزرسانی:** ۲۰۲۶-۰۸-۱۹ — افزوده شدن Group 7 (`WSPI-F2`، `WSPI-FT`) و انتخاب‌پذیری نمودارها

> **قانون طلایی:** هیچ روش قبلی دوباره اجرا نمی‌شود. با `--resume` هر روشِ `done` بی‌درنگ رد می‌شود. پوشه‌های `main_<ts>` مقدس‌اند و دست نمی‌خورند.

---

## ۰ — چه چیزی در این نسخه تغییر کرده

| فایل | وضعیت |
|---|---|
| `methods\wspi_forecast2.py` | **جدید** — `WSPI-F2` و `WSPI-FT` |
| `evaluation\method_configs.py` | جایگزین شد — دو ورودی Group 7 اضافه شد (۱۷ روش) |
| `experiments\run_popularity_assessment.py` | جایگزین شد — import و ثبت Group 7 |
| `experiments\plot_prediction_comparison.py` | جایگزین شد — `--methods / --exclude / --preset / --top-n / --from-csv` |
| `methods\wspi_forecast.py` | **دست نخورد** (`WSPI-F`، `WSPI-F-YW` سر جایشان) |
| `methods\wspi_assessment.py` | **دست نخورد** (WSPI اصلی) |

### صحت‌سنجی نصب (۲۰ ثانیه)

```bat
cd /d "D:\Research\Thesis Research\Simulations\Popularity With Wavelets\Dtcwt popularity\dtcwt_popularity"
.venv\Scripts\activate

python -c "from evaluation.method_configs import METHOD_CONFIGS as M; print(len(M),'methods'); print([m for m in M if m.startswith('WSPI')])"
```
انتظار: `17 methods` و `['WSPI', 'WSPI-F', 'WSPI-F-YW', 'WSPI-F2', 'WSPI-FT']`

```bat
:: تست خودکار ماژول جدید — کران ریاضی را با brute force چک می‌کند
python -m methods.wspi_forecast2
```
انتظار: جدول `m / observed max / bound` و در پایان `[OK] bound holds for every m`.

---

## ۱ — دو روش جدید چه هستند و چه انتظاری باید داشت

هر دو **بدون آموزش، فرم بسته، هم‌مرتبه WSPI** (اندازه‌گیری‌شده: ۱.۱۶ برابر هزینه WSPI).

| | `WSPI-F2` | `WSPI-FT` |
|---|---|---|
| پیش‌بین ضرایب | AR(2) Yule–Walker روی نموها | Theil–Sen (میانه شیب‌های زوجی) |
| مهار | ضربی، `c = 4` | ضربی، `c = 4` |
| دروازه ساختاری | `g = (R·(1−W_E))^0.3` | ندارد |
| سؤالی که جواب می‌دهد | «آیا دروازه‌بندی ساختاری کمک می‌کند؟» | «آیا پیش‌بین مقاوم از خودبازگشتی بهتر است؟» |
| ρ با WSPI-F-YW (اندازه‌گیری‌شده) | ۰.۹۹۸ | ۰.۹۹۵ |

### ⚠️ انتظار واقع‌بینانه (بخوانید تا از نتیجه جا نخورید)

روی جمعیت ۶۰۰ آیتمی مصنوعی اندازه‌گیری شد:

- **`WSPI-F2` با احتمال زیاد در حد نوسان کنار `WSPI-F-YW` می‌نشیند** (ρ = ۰.۹۹۸). ارزشش عدد جدید نیست؛ **کران اثبات‌پذیر** و **شکل توزیع دروازه** است. اگر روی تاکسی هم همین شد، صادقانه بنویسید «دروازه ساختاری روی این دو دیتاست فعال نمی‌شود چون …».
- **`WSPI-FT` تنها روشی است که می‌تواند عدد را واقعاً جابه‌جا کند** (ρ = ۰.۹۹۲ با WSPI). این را جدی بگیرید.

### چرا اصلاً لازم بودند (عدد کلیدی تشخیص)

وزن‌های بازگشتی هندسی با نسبت ۱/۲ هستند، پس آخرین عضو دنباله وزن ۱ از مجموع ≈۲ می‌گیرد:

> **یک ضریب پیش‌بینی‌شده دقیقاً ۵۰٪ کل `μ_L` را تعیین می‌کند** (اندازه‌گیری: ۰.۵۰۰۰ برای پنجره ۶۴ اسلاتی، ۰.۵۰۲۰ برای ۳۲).

با این اهرم، یک برون‌یاب بی‌مهار یک پنجره جهش‌دار را به انفجار رتبه تبدیل می‌کند. روی ۱۵۰۰ پنجره جهش‌دار تصادفی:

| روش | میانه نسبت به WSPI | صدک ۹۹ | **بیشینه** | کران اثبات‌پذیر |
|---|---|---|---|---|
| `WSPI-F` (NLMS) | ۱.۴۵ | ۳.۱۰ | **۶.۹۹** | ندارد |
| `WSPI-F-YW` | ۰.۷۷ | ۱.۰۶ | ۱.۱۹ | ندارد |
| **`WSPI-F2`** | ۰.۹۱ | ۱.۰۴ | **۱.۱۰** | **۴.۵۰** |
| **`WSPI-FT`** | ۰.۷۲ | ۱.۰۴ | **۱.۵۰** | **۴.۵۰** |

**معیار پذیرش ۲ سند طراحی محقق شد:** تورم ۷ برابر حذف شد و جایش کران قابل اثبات نشست.

نکته صادقانه: `WSPI-F-YW` هم عملاً بیشینه ۱.۱۹ داشت — یعنی **مشکل جهش منحصر به NLMS بود**، نه به معماری. `WSPI-F2` تضمین ریاضی اضافه می‌کند، نه نجات یک روش خراب.

---

## ۲ — اجرای YouTube (فقط دو روش جدید)

پوشه مقایسه **از قبل وجود دارد** و هر ۱۵ روش در آن `done` هستند:

```
results\youtube\predcmp_20260815_232251
```

پس نیازی به `prepare_compare_folder.py` نیست — مستقیم `--resume`:

```bat
python experiments\run_popularity_assessment.py youtube --cores 6 ^
    --resume results\youtube\predcmp_20260815_232251 ^
    --window-size 30 --horizon 7 ^
    --methods WSPI-F2 WSPI-FT
```

> **تنظیمات از `metadata\run_metadata.json` همان اجرا خوانده شده‌اند:** `window_size = 30`، `prediction_horizon = 7`، `k = [5,10,20]`، `num_items = 1485`. اگر این‌ها را عوض کنید پنجره‌های روش جدید با کش هم‌تراز نمی‌شود و مقایسه بی‌معنا می‌شود.

**برآورد زمان:** WSPI روی این اجرا حدود چند دقیقه بود؛ دو روش جدید ۱.۱۶ برابر WSPI هستند و موازی اجرا می‌شوند → **کمتر از ۱۵ دقیقه**. (برای مقایسه: ARIMA روی همین اجرا **۴۴٬۶۴۸ ثانیه = ۱۲.۴ ساعت** طول کشید — این عدد از `runtime_stats` خودتان است و همان «عدد طلایی دفاع» برای جدول هزینه است.)

سپس نمودارها:

```bat
python experiments\plot_prediction_comparison.py results\youtube\predcmp_20260815_232251 ^
    --tag youtube --preset core
python experiments\plot_prediction_comparison.py results\youtube\predcmp_20260815_232251 ^
    --tag youtube --preset forecast
python experiments\plot_prediction_comparison.py results\youtube\predcmp_20260815_232251 ^
    --tag youtube --preset ablation
```

---

## ۳ — «اوبر»: در این پروژه دیتاست جداگانه‌ای وجود ندارد

بررسی شد. یافته:

- در `config.py` هیچ ورودی `uber` وجود ندارد.
- در `data\datasets\` هیچ فایل `uber*.csv` نیست و در `data\raw\` هم داده خام اوبر نیست.
- فقط `docs\UBER_CONVERTER.md` هست و **عنوان خودش این است: «راهنمای Uber/NYC Yellow Taxi Converter»** — یعنی در این پروژه Uber همان **NYC Yellow Taxi** است و مبدلش الان `data\converters\yellow_taxi_converter.py` نام دارد.
- فقط دو فایل `__pycache__\uber_converter.pyc` و `uber_loader.pyc` باقی مانده‌اند؛ سورس `.py` آن‌ها حذف شده.

**نتیجه:** «اجرای اوبر» = **اجرای `yellow_taxi`** که همین حالا آماده است و کش دارد. اگر منظورتان یک دیتاست اوبرِ واقعاً جدا بود، مسیر فایل خامش را بدهید تا مبدل و ثبت دیتاست را بنویسم.

### ۳-۱ تاکسی ساعتی (سناریوی «ذات متفاوت» — تقاضای مکانی، فصلی‌تر و کم‌جهش‌تر از YouTube)

```bat
:: الف) پوشه مقایسه از کش اجرای اصلی
python experiments\prepare_compare_folder.py results\yellow_taxi\main_20260612_140633_hourly
:: خروجی، مسیر predcmp_<ts> جدید را چاپ می‌کند — کپی کنید
```

```bat
:: ب) اجرای همه روش‌ها؛ ۹ روش کش‌شده صفر ثانیه می‌گیرند
python experiments\run_popularity_assessment.py yellow_taxi --cores 6 ^
    --resume results\yellow_taxi\predcmp_<ts> ^
    --data-path data\datasets\yellow_taxi_2025_all_hourly.csv ^
    --window-size 30 --horizon 7 ^
    --methods AF EWMA RRD VSE CompoundPop PFRF DWT+AF DTCWT+AF WSPI ^
              Persistence Holt ARYW WSPI-F WSPI-F-YW WSPI-F2 WSPI-FT
```

**ARIMA را عمداً از این لیست حذف کرده‌ام.** روی YouTube ‏۱۲.۴ ساعت گرفت و تاکسی ساعتی بزرگ‌تر است (≈۱۵ ساعت). اگر لازمش دارید، **جداگانه و شبانه**:

```bat
python experiments\run_popularity_assessment.py yellow_taxi --cores 6 ^
    --resume results\yellow_taxi\predcmp_<ts> ^
    --data-path data\datasets\yellow_taxi_2025_all_hourly.csv ^
    --window-size 30 --horizon 7 --methods ARIMA
```

قبل از اجرای شبانه:
```bat
powercfg /change standby-timeout-ac 0
powercfg /change monitor-timeout-ac 0
```
اگر ویندوز ری‌استارت شد، **همان دستور `--resume` را دوباره بزنید** — از همان‌جا ادامه می‌دهد.

### ۳-۲ تاکسی ۳۰ دقیقه و ۵ دقیقه (بدون ARIMA)

```bat
python experiments\prepare_compare_folder.py results\yellow_taxi\main_20260620_092517_30min
python experiments\run_popularity_assessment.py yellow_taxi --cores 6 ^
    --resume results\yellow_taxi\predcmp_<ts30> ^
    --data-path data\datasets\yellow_taxi_2025_all_30min.csv ^
    --window-size 30 --horizon 7 ^
    --methods AF EWMA RRD VSE CompoundPop PFRF DWT+AF DTCWT+AF WSPI ^
              Persistence Holt ARYW WSPI-F WSPI-F-YW WSPI-F2 WSPI-FT

python experiments\prepare_compare_folder.py results\yellow_taxi\main_20260620_170044_5min
python experiments\run_popularity_assessment.py yellow_taxi --cores 6 ^
    --resume results\yellow_taxi\predcmp_<ts5> ^
    --data-path data\datasets\yellow_taxi_2025_all_5min.csv ^
    --window-size 30 --horizon 7 ^
    --methods AF EWMA RRD VSE CompoundPop PFRF DWT+AF DTCWT+AF WSPI ^
              Persistence Holt ARYW WSPI-F WSPI-F-YW WSPI-F2 WSPI-FT
```

> ⚠️ **قبل از هر اجرا `metadata\run_metadata.json` همان پوشه main را باز کنید** و `window_size` / `prediction_horizon` / `k_values` واقعی‌اش را در دستور تکرار کنید. اجرای تاکسی ساعتی `window_size = 30، horizon = 7` بود ولی `min_observations` آن ۲۴ است در حالی که YouTube ‏۵۰ است — این عدد خودکار از دانه‌بندی می‌آید، ولی اگر ستون `windows` در خروجی بین روش‌ها یکسان نشد، اولین جای مشکوک همین است.

---

## ۴ — نمودارهای انتخابی (مشکل ۱۷ روش)

با ۱۷ روش، نمودار تک‌قابی خوانا نیست. چهار سازوکار انتخاب اضافه شد. ترتیب اولویت: `--methods` > `--preset` > همه؛ بعد `--exclude`؛ بعد `--top-n`.

```bat
:: دیدن مجموعه‌های آماده
python experiments\plot_prediction_comparison.py --list-presets
```

| preset | روش‌ها | کجای رساله |
|---|---|---|
| `core` | WSPI, WSPI-F-YW, WSPI-F2, WSPI-FT, DTCWT+AF, ARYW | **متن اصلی** (۶ روش، خوانا) |
| `forecast` | WSPI + هر پنج مشتق + Persistence, Holt, ARYW, ARIMA | زیربخش ۴-۵-۴ |
| `wavelet` | فقط خانواده موجکی (۷ روش) | تحلیل سهم مؤلفه‌ها |
| `ablation` | WSPI, WSPI-F, WSPI-F-YW, WSPI-F2, WSPI-FT | مطالعه فرسایشی پیش‌بین |
| `classic` | WSPI + شش مبنای کلاسیک | فصل مبناها |
| `all` | همه (رفتار قبلی) | پیوست |

```bat
:: انتخاب دستی با ترتیب دلخواه
python experiments\plot_prediction_comparison.py <folder> --tag youtube ^
    --methods WSPI WSPI-F-YW WSPI-FT ARIMA Persistence

:: حذف چند روش
python experiments\plot_prediction_comparison.py <folder> --tag youtube --exclude PFRF Holt

:: فقط ۸ روش برتر هر نمودار بر اساس معیار خودش
python experiments\plot_prediction_comparison.py <folder> --tag youtube --preset forecast --top-n 8

:: رسم مجدد از روی CSV خلاصه، بدون نیاز به پوشه protocol
python experiments\plot_prediction_comparison.py --from-csv results\youtube\predcmp_20260815_232251\figures_prediction\prediction_summary_youtube.csv --tag youtube --preset core
```

**رفتارهایی که عمداً این‌طور طراحی شده‌اند:**

- **روش‌های پیشنهادی هرگز با `--top-n` حذف نمی‌شوند** (`ALWAYS_KEEP`). اگر WSPI در یک معیار بازنده باشد باز هم دیده می‌شود — پنهان کردنش دستچین‌کردن نتیجه است.
- **اگر `--top-n` چیزی را حذف کند، روی خود نمودار نوشته می‌شود** (`top-8 shown, 5 method(s) not shown`). نموداری که بی‌صدا نصف میدان را پنهان کند، نمودار دروغ‌گوست.
- **هر انتخاب فایل جدا می‌سازد** (`fig_pred_ndcg10_youtube_core.png`)، پس `core` و `forecast` همدیگر را بازنویسی نمی‌کنند. با `--suffix` قابل تغییر است.
- کدگذاری رنگی، برچسب مقدار سر میله، پررنگی نام روش‌های پیشنهادی و `dpi=200` **همه دست‌نخورده باقی مانده‌اند**؛ `WSPI-F2` و `WSPI-FT` به مجموعه آبی/پررنگ اضافه شدند.

---

## ۵ — بعد از اجرا: سه چیزی که باید فوراً چک کنید

1. **ستون `windows`** در `prediction_summary_*.csv` برای هر ۱۷ روش. روش‌های ۶۴-اسلاتی باید عدد یکسان بگیرند (روی YouTube: ۶۵۲). اگر `WSPI-F2` عددی غیر از `WSPI` گرفت، تنظیمات اجرا هم‌تراز نیست — نتیجه را دور بریزید و با فلگ‌های درست دوباره اجرا کنید.
2. **`WSPI-F2` در برابر `WSPI-F-YW`**: اگر هر چهار معیار تا سه رقم اعشار یکی شد، یعنی دروازه روی این دیتاست خاموش مانده. این **نتیجه است، نه شکست** — توزیع `g` را گزارش کنید (`WSPIForecast2.gate_value`).
3. **`WSPI-FT`**: این جایی است که ممکن است عدد جدید ببینید. اگر RSI@10 آن از WSPI بالاتر رفت، سرفصل تازه‌ای برای ۴-۵-۵ دارید.

### آزمون معناداری (۱۰ دقیقه، ولی برای دفاع ضروری)

اختلاف‌های ۰.۵٪ در NDCG بین WSPI و مشتقاتش **در حد پراکندگی کل جدول** است. قبل از نوشتن «بهتر است» در رساله، آزمون زوجی روی پنجره‌ها بگیرید:

```bat
python -c "import pandas as pd; from scipy.stats import wilcoxon; import glob; d=r'results\youtube\predcmp_20260815_232251\protocol'; a=pd.read_parquet(glob.glob(d+r'\WSPI_protocol*')[0]); b=pd.read_parquet(glob.glob(d+r'\WSPI-F-YW_protocol*')[0]); print(a.columns.tolist())"
```
(اول نام ستون‌ها را ببینید، بعد ستون `ndcg@10` هر دو را روی `window` جفت کنید و `wilcoxon` بگیرید. اگر ساختار ستون‌ها را برایم بفرستید، اسکریپتش را می‌نویسم.)

---

## ۶ — عیب‌یابی

| نشانه | علت و راه‌حل |
|---|---|
| `Warning: WSPI-F2 not available` | `methods\wspi_forecast2.py` سر جایش نیست یا `dtcwt` نصب نیست. |
| `17 methods` چاپ نمی‌شود | `evaluation\method_configs.py` جایگزین نشده. |
| روش‌های قدیمی زیر `to run now` می‌آیند | مسیر `--resume` پوشه predcmp نیست. |
| `Unknown preset 'x'` | `--list-presets` را بزنید. |
| نمودار انتخابی خالی | نام‌ها را دقیق بنویسید (`WSPI-F-YW` نه `WSPI_F_YW`)؛ اسکریپت نام‌های ناموجود را با `!` گزارش می‌کند. |
| `WSPI-F2` و `WSPI` دقیقاً یکی شدند | یعنی `g ≈ 0` در همه پنجره‌ها. باگ نیست — `gate_gamma` را در `run_popularity_assessment.py` از ۰.۳ به ۰.۱۵ کم کنید و با نام جدید دوباره اجرا کنید. |
| ستون `windows` نایکسان | `--window-size` / `--horizon` / `--data-path` با اجرای اصلی فرق دارد. |

---

## ۷ — چک‌لیست

1. ☐ `python -m methods.wspi_forecast2` سبز شد (`[OK] bound holds`).
2. ☐ `17 methods` تأیید شد.
3. ☐ YouTube: دو روش جدید روی `predcmp_20260815_232251` اجرا شدند و ستون `windows` آن‌ها ۶۵۲ شد.
4. ☐ نمودارهای `--preset core` و `--preset ablation` تولید شدند.
5. ☐ تاکسی ساعتی: پوشه predcmp ساخته و ۱۶ روش (بدون ARIMA) اجرا شد.
6. ☐ تاکسی ۳۰ و ۵ دقیقه اجرا شدند.
7. ☐ آزمون زوجی معناداری بین WSPI و بهترین مشتقش گرفته شد.
8. ☐ توزیع دروازه `g` استخراج و به‌عنوان شکل فصل ۴ ذخیره شد.

---

**اسناد مرتبط:** `Diagnosis_WSPI_F_YouTube.md` (تشخیص و استدلال طراحی) · `Thesis_v2_Prediction_Revisions.md` (متن فارسی رساله) · `SELECTIVE_RUN_GUIDE.md` (مکانیزم کش)
