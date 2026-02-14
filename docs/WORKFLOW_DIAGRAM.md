# جریان کار کامل — Frozen 4-Layer Evaluation Protocol

## 🔄 نمودار اصلی

```mermaid
graph TB
    subgraph INPUT["📥 ورودی‌ها"]
        DS[("Dataset\nMovieLens / Uber / YouTube\n14.6M+ records")]
        CONFIG["پیکربندی\n• window_size: 30\n• k_list: [5,10,20]\n• robustness_sample: 50\n• spike_multiplier: 10×"]
        METHODS["روش‌ها\n• AF / LFU / LRU / EWMA\n• Statistical\n• DWT+AF / DTCWT+AF\n• WSPI (Proposed)"]
    end

    subgraph COMPUTE["⚙️ ارزیابی — run_popularity_assessment.py"]
        PREP["آماده‌سازی\n• فیلتر زمانی\n• انتخاب آیتم‌ها\n• Stratification"]
        SLIDE["Sliding Window\n(یک روز = یک window)"]
        SCORE["محاسبه امتیاز\n(هر روش × هر window)"]

        subgraph PROTO["4-Layer Frozen Protocol"]
            L1["Layer 1 — Decision\nNDCG@K, CHR@K"]
            L2["Layer 2 — Diagnostic\nKendall τ, Spearman ρ, MAE"]
            L3["Layer 3 — Stability\nRSI@K (Jaccard)"]
            L4["Layer 4 — Robustness\n10× Spike → ΔRank"]
        end
    end

    subgraph STORAGE["💾 ذخیره‌سازی"]
        DETAIL["detailed/\n*_scores.parquet"]
        SUMM["summary/\n*_stratum_summary.parquet"]
        PROTO_OUT["protocol/ ← جدید\n*_protocol.csv"]
        META["metadata/\nconfig.json + thresholds.json"]
    end

    subgraph ANALYZE["📊 تحلیل — analyze_results.py"]
        DISP["Display-only\n(پیش‌فرض - سریع)"]
        RECOMP["--recompute\n(بازمحاسبه از raw)"]
        SAVR["--save-recomputed\n(ذخیره نتایج)"]
    end

    subgraph SHOW["🖥️ نمایش — show_results.py"]
        TEXT["متنی\nجدول multi-K"]
        GRAPH["گرافیکی\nنمودارهای 4-Layer"]
    end

    INPUT --> COMPUTE
    COMPUTE --> STORAGE
    STORAGE --> ANALYZE
    ANALYZE --> SHOW
```

---

## 📊 4-Layer Protocol — جزئیات

```mermaid
graph LR
    subgraph W["برای هر Window t"]
        S["scores_clean\n(امتیازهای مدل)"]
        A["actuals\n(بازدید واقعی)"]

        S & A --> L1
        S & A --> L2
        S --> L3
        S --> L4

        subgraph L1["Layer 1 — Decision"]
            direction TB
            N5["NDCG@5"]
            N10["NDCG@10"]
            N20["NDCG@20"]
            C5["CHR@5"]
            C10["CHR@10"]
            C20["CHR@20"]
        end

        subgraph L2["Layer 2 — Diagnostic"]
            direction TB
            KT["Kendall τ"]
            SR["Spearman ρ"]
            MAE["MAE"]
        end

        subgraph L3["Layer 3 — Stability"]
            direction TB
            TOP["Top-K now"]
            PREV["Top-K (t-1)"]
            RSI["RSI = |A∩B|/|A∪B|"]
            TOP & PREV --> RSI
        end

        subgraph L4["Layer 4 — Robustness"]
            direction TB
            SEL["انتخاب 50 آیتم پایدار"]
            INJ["تزریق 10× نویز"]
            DR["ΔRank = |rank_clean - rank_noisy|"]
            SEL --> INJ --> DR
        end
    end
```

---

## 🗂️ ساختار خروجی

```
results/
└── movielens/
    └── w30_nall_top_20260214_143052/
        ├── detailed/
        │   ├── AF_scores.parquet           # امتیاز + actual + stratum per item
        │   ├── DTCWT+AF_scores.parquet
        │   ├── WSPI_scores.parquet
        │   └── ...
        ├── summary/
        │   ├── AF_stratum_summary.parquet  # میانگین per stratum per window
        │   └── ...
        ├── protocol/                       ← جدید (4-Layer)
        │   ├── AF_protocol.csv             # ndcg@K, chr@K, rsi@K, τ, ρ, ΔRank
        │   ├── DTCWT+AF_protocol.csv
        │   ├── WSPI_protocol.csv
        │   └── ...
        ├── comparison/
        │   └── method_comparison.csv
        ├── metadata/
        │   ├── config.json                 # شامل k_list, robustness_*
        │   ├── thresholds.json
        │   └── runtime_stats.json
        └── visualization/
            └── *.png
```

---

## 🔄 جریان تحلیل (دو حالت)

```mermaid
flowchart TD
    RES["نتایج موجود\nresults/…/"]

    RES --> Q{{"آیا protocol/ موجود است؟"}}

    Q -- بله --> FAST["analyze_results.py\n(بدون --recompute)\nخواندن مستقیم"]
    Q -- خیر --> SLOW["analyze_results.py\n--recompute\nبازمحاسبه از detailed/"]

    SLOW -- "--save-recomputed" --> SAVE["ذخیره در protocol/\nبرای بار بعد"]
    FAST --> SHOW["show_results.py\nنمایش متنی + گرافیکی"]
    SAVE --> SHOW
```

---

## ⚙️ مسئولیت فایل‌ها

| فایل | مسئولیت | محاسبه؟ |
|------|---------|---------|
| `run_popularity_assessment.py` | ارزیابی کامل + ذخیره 4-Layer | ✅ بله |
| `analyze_results.py` | تحلیل، فیلتر، بازمحاسبه | 🔁 اختیاری (--recompute) |
| `show_results.py` | نمایش خالص | ❌ خیر |

---

## 📋 ستون‌های protocol CSV

```
window_id, timestamp, method, num_items,
ndcg@5, ndcg@10, ndcg@20,
chr@5,  chr@10,  chr@20,
kendall_tau, spearman_rho, mae,
rsi@5,  rsi@10,  rsi@20,
robustness_distortion
```

---

## ⏱️ زمان‌بندی (۱۰۰۰ آیتم، ۱ سال)

```
run_popularity_assessment.py  →  ~10 ساعت (یک بار)
analyze_results.py (display)  →   3 ثانیه
analyze_results.py --recompute → 30 ثانیه
show_results.py               →   5 ثانیه
```
