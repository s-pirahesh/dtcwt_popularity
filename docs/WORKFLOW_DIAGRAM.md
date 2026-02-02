# جریان کار کامل سیستم ارزیابی محبوبیت

## 🔄 نمودار اصلی - جریان کامل

```mermaid
graph TB
    %% ========== ورودی‌ها ==========
    subgraph INPUT["📥 ورودی‌ها"]
        DS[("Dataset<br/>MovieLens/YouTube/Youku<br/>14.6M records")]
        CONFIG["پیکربندی<br/>• window_size: 30<br/>• horizon: 7<br/>• num_items: 1000"]
        METHODS["روش‌های ارزیابی<br/>• AF (Access Frequency)<br/>• DTCWT+AF<br/>• DWT+AF<br/>• Statistical<br/>• Hybrid"]
    end
    
    %% ========== محاسبات ==========
    subgraph COMPUTE["⚙️ محاسبات محبوبیت<br/>run_popularity_assessment.py"]
        PREP["آماده‌سازی<br/>• فیلتر زمانی<br/>• انتخاب آیتم‌ها<br/>• Stratification"]
        SLIDE["Sliding Window<br/>• step=1<br/>• 9,964 windows<br/>• 1,000 items/window"]
        CALC["محاسبه امتیاز محبوبیت<br/>برای هر روش<br/>9 methods × 9,964 windows"]
        METRICS["محاسبه معیارها<br/>• Spearman<br/>• MAE, RMSE<br/>• NDCG"]
    end
    
    %% ========== ذخیره‌سازی ==========
    subgraph STORAGE["💾 ذخیره با timestamp منحصر به فرد"]
        TIMESTAMP["w30_h7_n1000_top_20250202_143052"]
        
        subgraph INTER["نتایج میانی - Parquet"]
            DETAIL["detailed/<br/>• AF_scores.parquet<br/>• DTCWT_AF_scores.parquet<br/>• DWT_AF_scores.parquet<br/>• ... (9 files × 70MB)"]
            SUMM["summary/<br/>• stratum_summary.parquet<br/>• ... (9 files × 1MB)"]
        end
        
        subgraph FINAL["نتایج نهایی - CSV/Parquet"]
            COMP["comparison/<br/>• method_comparison.csv<br/>  (10 KB)"]
        end
        
        META["metadata/<br/>• config.json<br/>• runtime_stats.json<br/>• thresholds.json"]
    end
    
    %% ========== تحلیل ==========
    subgraph ANALYSIS["📊 تحلیل<br/>analyze_results.py"]
        LOAD["بارگذاری خودکار<br/>• خواندن Parquet<br/>• خواندن CSV<br/>• Cache در RAM"]
        
        FILTER["فیلترها<br/>• top-k% (مثلاً 20%)<br/>• stratum (cold/low/med/high)<br/>• time range"]
        
        ANALYZE["تحلیل‌ها<br/>• محاسبه معیارها<br/>• مقایسه روش‌ها<br/>• تحلیل زمانی<br/>• مقایسه strata"]
        
        EXPORT["خروجی تحلیل<br/>• جداول DataFrame<br/>• آمار آماری<br/>• نتایج مقایسه"]
    end
    
    %% ========== نمایش ==========
    subgraph DISPLAY["🎨 نمایش<br/>show_results.py"]
        subgraph TEXT["نمایش متنی"]
            TABLE["جداول مقایسه<br/>• روش‌ها<br/>• معیارها<br/>• رتبه‌بندی"]
            STATS["آمار تفصیلی<br/>• هر روش<br/>• هر stratum"]
            BEST["بهترین روش‌ها<br/>• بالاترین Spearman<br/>• کمترین MAE<br/>• بالاترین NDCG"]
        end
        
        subgraph GRAPH["نمایش گرافیکی (300 DPI)"]
            TEMPORAL["temporal_evolution_mae.png<br/>تکامل زمانی خطا"]
            METHOD["method_comparison.png<br/>مقایسه روش‌ها"]
            STRATA["stratum_comparison.png<br/>عملکرد در strata"]
            HEAT["ranking_heatmap.png<br/>تکامل ranking"]
        end
    end
    
    %% ========== خروجی ==========
    subgraph OUTPUT["📤 خروجی نهایی"]
        PAPER["مقاله/پایان‌نامه<br/>• جداول<br/>• نمودارها<br/>• آمار"]
        PRESENT["ارائه<br/>• اسلایدها<br/>• نمودارهای با کیفیت"]
    end
    
    %% ========== روابط ==========
    DS --> PREP
    CONFIG --> PREP
    METHODS --> PREP
    
    PREP --> SLIDE
    SLIDE --> CALC
    CALC --> METRICS
    
    METRICS --> TIMESTAMP
    TIMESTAMP --> DETAIL
    TIMESTAMP --> SUMM
    TIMESTAMP --> COMP
    TIMESTAMP --> META
    
    DETAIL --> LOAD
    SUMM --> LOAD
    COMP --> LOAD
    META --> LOAD
    
    LOAD --> FILTER
    FILTER --> ANALYZE
    ANALYZE --> EXPORT
    
    EXPORT --> TABLE
    EXPORT --> STATS
    EXPORT --> BEST
    
    EXPORT --> TEMPORAL
    EXPORT --> METHOD
    EXPORT --> STRATA
    EXPORT --> HEAT
    
    TABLE --> PAPER
    STATS --> PAPER
    TEMPORAL --> PAPER
    METHOD --> PAPER
    STRATA --> PAPER
    
    TEMPORAL --> PRESENT
    METHOD --> PRESENT
    STRATA --> PRESENT
    
    %% ========== استایل‌ها ==========
    classDef inputStyle fill:#e1f5ff,stroke:#01579b,stroke-width:3px
    classDef computeStyle fill:#fff9c4,stroke:#f57f17,stroke-width:3px
    classDef storageStyle fill:#f3e5f5,stroke:#4a148c,stroke-width:3px
    classDef analysisStyle fill:#e8f5e9,stroke:#1b5e20,stroke-width:3px
    classDef displayStyle fill:#ffe0b2,stroke:#e65100,stroke-width:3px
    classDef outputStyle fill:#ffebee,stroke:#b71c1c,stroke-width:3px
    
    class DS,CONFIG,METHODS inputStyle
    class PREP,SLIDE,CALC,METRICS computeStyle
    class TIMESTAMP,DETAIL,SUMM,COMP,META storageStyle
    class LOAD,FILTER,ANALYZE,EXPORT analysisStyle
    class TABLE,STATS,BEST,TEMPORAL,METHOD,STRATA,HEAT displayStyle
    class PAPER,PRESENT outputStyle
```

---

## 📋 نمودار ساده‌شده

```mermaid
flowchart LR
    A["📥 Dataset<br/>+ Config"] 
    B["⚙️ run_popularity_<br/>assessment.py<br/>(محاسبات)"]
    C["💾 نتایج با timestamp<br/>w30_h7_n1000_top_<br/>20250202_143052"]
    D["📊 analyze_results.py<br/>(تحلیل)"]
    E["🎨 show_results.py<br/>(نمایش)"]
    F["📤 مقاله/<br/>پایان‌نامه"]
    
    A --> B
    B --> |"حجم زیاد<br/>~630 MB"| C
    C --> |"بارگذاری<br/>سریع"| D
    D --> |"تحلیل<br/>~5 sec"| E
    E --> |"جداول<br/>نمودارها"| F
    
    style A fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    style B fill:#fff9c4,stroke:#f57f17,stroke-width:2px
    style C fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    style D fill:#e8f5e9,stroke:#388e3c,stroke-width:2px
    style E fill:#ffe0b2,stroke:#e64a19,stroke-width:2px
    style F fill:#ffebee,stroke:#c62828,stroke-width:2px
```

---

## 🔁 نمودار زمان‌بندی (Timeline)

```mermaid
gantt
    title جدول زمانی انجام کار
    dateFormat HH:mm
    axisFormat %H:%M
    
    section محاسبات
    run_popularity_assessment    :a1, 00:00, 10h
    
    section ذخیره
    ذخیره Parquet (630 MB)      :a2, after a1, 2m
    
    section تحلیل
    analyze - همه داده           :a3, after a2, 5s
    analyze - top 20%            :a4, after a3, 3s
    analyze - cold-start         :a5, after a4, 4s
    analyze - time range         :a6, after a5, 4s
    
    section نمایش
    show - متنی                  :a7, after a6, 2s
    show - گرافیکی               :a8, after a7, 8s
    show - تفصیلی                :a9, after a8, 3s
```

---

## 📊 نمودار جریان داده (Sequence Diagram)

```mermaid
sequenceDiagram
    participant U as 👤 محقق
    participant R as ⚙️ Run
    participant S as 💾 Storage
    participant A as 📊 Analyze
    participant V as 🎨 Show
    
    U->>R: اجرا با پارامترها
    Note over R: window=30, items=1000
    
    R->>R: محاسبات Sliding Window
    Note over R: 9,964 windows × 1,000 items<br/>= 9.9M محاسبه
    
    R->>S: ذخیره با timestamp
    Note over S: w30_h7_n1000_top_20250202_143052/<br/>• detailed: 9 × 70MB = 630MB<br/>• summary: 9 × 1MB = 9MB<br/>• comparison: 10KB
    
    Note over U: ⏰ 10 ساعت بعد...
    
    U->>A: تحلیل - همه داده
    A->>S: بارگذاری Parquet
    S-->>A: DataFrame (RAM)
    A->>A: محاسبه معیارها
    A-->>U: جداول مقایسه
    Note over U: ⏱️ 5 ثانیه
    
    U->>A: تحلیل - top 20%
    A->>A: فیلتر از cache
    A-->>U: نتایج فیلتر شده
    Note over U: ⏱️ 3 ثانیه
    
    U->>V: نمایش متنی
    V->>A: درخواست داده
    A-->>V: DataFrame
    V-->>U: جداول متنی
    Note over U: ⏱️ 2 ثانیه
    
    U->>V: نمایش گرافیکی
    V->>V: تولید نمودارها
    V-->>U: PNG files (300 DPI)
    Note over U: ⏱️ 8 ثانیه
    
    U->>U: استفاده در مقاله ✓
```

---

## 📁 ساختار فایل‌ها

```mermaid
graph TD
    ROOT["📁 results/movielens/"]
    
    RUN1["📂 w30_h7_n1000_top_20250201_143052/<br/>⏱️ اجرا: 2025-02-01 14:30"]
    RUN2["📂 w14_h3_n500_str_20250202_091523/<br/>⏱️ اجرا: 2025-02-02 09:15"]
    RUN3["📂 w60_h14_nall_ran_20250203_153015/<br/>⏱️ اجرا: 2025-02-03 15:30"]
    
    ROOT --> RUN1
    ROOT --> RUN2
    ROOT --> RUN3
    
    D1["📊 detailed/<br/>9 methods × 70MB<br/>= 630 MB"]
    S1["📈 summary/<br/>9 methods × 1MB<br/>= 9 MB"]
    C1["📄 comparison/<br/>method_comparison.csv<br/>= 10 KB"]
    M1["⚙️ metadata/<br/>JSON files<br/>= 8 KB"]
    V1["🎨 visualization/<br/>PNG files (300 DPI)<br/>= 20 MB"]
    
    RUN1 --> D1
    RUN1 --> S1
    RUN1 --> C1
    RUN1 --> M1
    RUN1 --> V1
    
    style ROOT fill:#e3f2fd,stroke:#1565c0,stroke-width:3px
    style RUN1 fill:#c5e1a5,stroke:#558b2f,stroke-width:2px
    style RUN2 fill:#c5e1a5,stroke:#558b2f,stroke-width:2px
    style RUN3 fill:#c5e1a5,stroke:#558b2f,stroke-width:2px
    style D1 fill:#fff9c4,stroke:#f57f17,stroke-width:2px
    style S1 fill:#ffccbc,stroke:#d84315,stroke-width:2px
    style C1 fill:#b2dfdb,stroke:#00695c,stroke-width:2px
    style M1 fill:#d1c4e9,stroke:#512da8,stroke-width:2px
    style V1 fill:#f8bbd0,stroke:#c2185b,stroke-width:2px
```

---

## 🔄 مثال عملی کامل

```mermaid
flowchart TB
    START(["شروع پروژه"])
    
    %% مرحله 1: محاسبات
    STEP1["<b>مرحله 1: محاسبات</b><br/>python run_popularity_assessment.py movielens \<br/>  --num-items 1000 \<br/>  --window-size 30 \<br/>  --horizon 7"]
    
    WAIT1["⏰ انتظار 10 ساعت"]
    
    RESULT1["✅ نتیجه:<br/>results/movielens/w30_h7_n1000_top_20250202_143052/<br/>حجم: ~650 MB"]
    
    %% مرحله 2: تحلیل‌های مختلف
    STEP2A["<b>تحلیل 1: همه داده</b><br/>python analyze_results.py RESULTS_PATH<br/>⏱️ 5 ثانیه"]
    
    STEP2B["<b>تحلیل 2: Top 20%</b><br/>python analyze_results.py RESULTS_PATH \<br/>  --top-percent 20<br/>⏱️ 3 ثانیه"]
    
    STEP2C["<b>تحلیل 3: Cold-Start</b><br/>python analyze_results.py RESULTS_PATH \<br/>  --stratum cold_start<br/>⏱️ 4 ثانیه"]
    
    %% مرحله 3: نمایش
    STEP3A["<b>نمایش متنی</b><br/>python show_results.py RESULTS_PATH<br/>⏱️ 2 ثانیه"]
    
    STEP3B["<b>نمایش گرافیکی</b><br/>python show_results.py RESULTS_PATH \<br/>  --graphical<br/>⏱️ 8 ثانیه"]
    
    STEP3C["<b>هر دو</b><br/>python show_results.py RESULTS_PATH \<br/>  --both --show<br/>⏱️ 10 ثانیه"]
    
    %% خروجی نهایی
    OUTPUT["📤 خروجی:<br/>• جداول مقایسه<br/>• نمودارهای 300 DPI<br/>• آمار تفصیلی"]
    
    PAPER["📄 استفاده در:<br/>• مقاله<br/>• پایان‌نامه<br/>• ارائه"]
    
    %% روابط
    START --> STEP1
    STEP1 --> WAIT1
    WAIT1 --> RESULT1
    
    RESULT1 --> STEP2A
    RESULT1 --> STEP2B
    RESULT1 --> STEP2C
    
    STEP2A --> STEP3A
    STEP2B --> STEP3B
    STEP2C --> STEP3C
    
    STEP3A --> OUTPUT
    STEP3B --> OUTPUT
    STEP3C --> OUTPUT
    
    OUTPUT --> PAPER
    
    %% استایل‌ها
    style START fill:#4caf50,stroke:#2e7d32,color:#fff,stroke-width:3px
    style STEP1 fill:#fff9c4,stroke:#f57f17,stroke-width:2px
    style WAIT1 fill:#ffccbc,stroke:#d84315,stroke-width:2px
    style RESULT1 fill:#c5e1a5,stroke:#558b2f,stroke-width:2px
    style STEP2A fill:#b2dfdb,stroke:#00695c,stroke-width:2px
    style STEP2B fill:#b2dfdb,stroke:#00695c,stroke-width:2px
    style STEP2C fill:#b2dfdb,stroke:#00695c,stroke-width:2px
    style STEP3A fill:#f8bbd0,stroke:#c2185b,stroke-width:2px
    style STEP3B fill:#f8bbd0,stroke:#c2185b,stroke-width:2px
    style STEP3C fill:#f8bbd0,stroke:#c2185b,stroke-width:2px
    style OUTPUT fill:#ce93d8,stroke:#7b1fa2,stroke-width:2px
    style PAPER fill:#ef5350,stroke:#c62828,color:#fff,stroke-width:3px
```

---

## 📝 خلاصه دستورات

### 1️⃣ محاسبات (یک بار - زمان‌بر)
```bash
python run_popularity_assessment.py movielens \
    --num-items 1000 \
    --window-size 30 \
    --horizon 7 \
    --format csv

# ⏱️ زمان: ~10 ساعت
# 💾 خروجی: results/movielens/w30_h7_n1000_top_20250202_143052/
# 📦 حجم: ~650 MB
```

### 2️⃣ تحلیل (بارها - سریع)
```bash
# همه داده
python analyze_results.py results/movielens/w30_h7_n1000_top_20250202_143052/

# با فیلتر
python analyze_results.py results/movielens/w30_h7_n1000_top_20250202_143052/ \
    --top-percent 20

# ⏱️ زمان هر تحلیل: ~5 ثانیه
```

### 3️⃣ نمایش (بارها - سریع)
```bash
# متنی
python show_results.py results/movielens/w30_h7_n1000_top_20250202_143052/

# گرافیکی
python show_results.py results/movielens/w30_h7_n1000_top_20250202_143052/ \
    --graphical

# هر دو
python show_results.py results/movielens/w30_h7_n1000_top_20250202_143052/ \
    --both --show

# ⏱️ زمان: ~10 ثانیه
```

---

## 📊 مقایسه زمان و کارایی

| مرحله | برنامه | دفعات اجرا | زمان/اجرا | مجموع |
|-------|---------|-----------|----------|--------|
| **1. محاسبات** | run_popularity_assessment.py | 1 بار | 10 ساعت | 10 ساعت |
| **2. ذخیره** | خودکار | 1 بار | 2 دقیقه | 2 دقیقه |
| **3. تحلیل** | analyze_results.py | N بار | 5 ثانیه | 5N ثانیه |
| **4. نمایش** | show_results.py | M بار | 10 ثانیه | 10M ثانیه |

### مثال: 10 تحلیل مختلف
- **قبل (بدون ذخیره):** 10 × 10 ساعت = **100 ساعت** ❌
- **بعد (با ذخیره):** 10 ساعت + (10 × 5 ثانیه) = **10 ساعت** ✅
- **صرفه‌جویی:** **90 ساعت (90%)** 🎯

---

## 🎯 نکات کلیدی

### ✅ نامگذاری با Timestamp
- هر اجرا نتایج منحصر به فرد دارد
- امکان مقایسه runهای مختلف
- بدون نگرانی از بازنویسی

### ✅ جداسازی مسئولیت‌ها
1. **run_popularity_assessment.py** → محاسبات (زمان‌بر)
2. **analyze_results.py** → تحلیل (سریع)
3. **show_results.py** → نمایش (سریع)

### ✅ ذخیره‌سازی هوشمند
- میانی: **Parquet** (فشرده‌سازی 85%)
- نهایی: **CSV** یا **Parquet** (قابل انتخاب)
- Metadata: **JSON** (قابل خواندن)

---

**یک بار محاسبه، بارها تحلیل و نمایش!** 🎯✨
