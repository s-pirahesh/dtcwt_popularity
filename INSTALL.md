# دستورالعمل نصب و استفاده
# Installation & Usage Instructions

**تاریخ:** 2 فوریه 2025  
**وضعیت:** ✅ آماده برای استفاده  

---

## 📦 محتویات بسته

این بسته شامل **19 فایل** است:

- **10 فایل Python** (کد)
- **9 فایل Markdown** (مستندات)

---

## 🚀 نصب سریع

### مرحله 1: دانلود فایل‌ها

همه فایل‌ها در پوشه زیر آماده‌اند:
```
/mnt/user-data/outputs/project_files/
```

### مرحله 2: کپی به پروژه

```bash
# مسیر پروژه شما
PROJECT_DIR=~/dtcwt_popularity

# کپی فایل‌های جدید (Data Layer)
cp -r project_files/data/ $PROJECT_DIR/

# کپی فایل‌های اصلاح شده (Evaluation)
cp project_files/evaluation/metrics.py $PROJECT_DIR/evaluation/
cp project_files/evaluation/__init__.py $PROJECT_DIR/evaluation/
cp project_files/evaluation/temporal_evaluator.py $PROJECT_DIR/evaluation/
cp project_files/evaluation/results_analyzer.py $PROJECT_DIR/evaluation/

# کپی فایل‌های Experiments
cp project_files/experiments/run_popularity_assessment.py $PROJECT_DIR/experiments/

# کپی دستورالعمل
cp project_files/IMPLEMENTATION_GUIDE.md $PROJECT_DIR/

# کپی مستندات (اختیاری)
mkdir -p $PROJECT_DIR/docs
cp -r project_files/docs/* $PROJECT_DIR/docs/
```

### مرحله 3: حذف فایل قدیمی

```bash
cd $PROJECT_DIR/evaluation

# حذف فایل قدیمی metrics_v2.py
rm -f metrics_v2.py

# بررسی
ls -la | grep metrics
# باید فقط metrics.py را نشان دهد
```

### مرحله 4: تست

```bash
cd $PROJECT_DIR

# تست import ها
python3 << EOF
from data.loaders import get_movielens_loader
from evaluation import MetricsCalculator
print('✅ همه import ها موفق بود!')
EOF
```

---

## 📋 دستورات تفصیلی

### برای کاربران Linux/Mac:

```bash
#!/bin/bash

# تنظیمات
SOURCE="project_files"
TARGET="$HOME/dtcwt_popularity"

echo "🔧 شروع نصب..."

# 1. Backup قدیمی (اختیاری)
if [ -d "$TARGET" ]; then
    echo "📦 ایجاد backup..."
    cp -r "$TARGET" "$TARGET.backup.$(date +%Y%m%d_%H%M%S)"
fi

# 2. کپی data/loaders (جدید)
echo "📁 کپی data/loaders..."
mkdir -p "$TARGET/data"
cp -r "$SOURCE/data/loaders" "$TARGET/data/"

# 3. کپی evaluation (اصلاح شده)
echo "📁 کپی evaluation..."
cp "$SOURCE/evaluation/metrics.py" "$TARGET/evaluation/"
cp "$SOURCE/evaluation/__init__.py" "$TARGET/evaluation/"
cp "$SOURCE/evaluation/temporal_evaluator.py" "$TARGET/evaluation/"
cp "$SOURCE/evaluation/results_analyzer.py" "$TARGET/evaluation/"

# 4. حذف فایل قدیمی
echo "🗑️  حذف metrics_v2.py..."
rm -f "$TARGET/evaluation/metrics_v2.py"

# 5. کپی experiments
echo "📁 کپی experiments..."
cp "$SOURCE/experiments/run_popularity_assessment.py" "$TARGET/experiments/"

# 6. کپی مستندات
echo "📚 کپی مستندات..."
cp "$SOURCE/IMPLEMENTATION_GUIDE.md" "$TARGET/"
mkdir -p "$TARGET/docs"
cp -r "$SOURCE/docs/"* "$TARGET/docs/"

# 7. تست
echo "🧪 تست..."
cd "$TARGET"
if python3 -c "from data.loaders import MovieLensLoader; from evaluation import MetricsCalculator" 2>/dev/null; then
    echo "✅ نصب موفق!"
else
    echo "❌ خطا در import ها!"
    exit 1
fi

echo ""
echo "🎉 نصب کامل شد!"
echo ""
echo "مراحل بعدی:"
echo "  1. cd $TARGET"
echo "  2. python3 experiments/run_popularity_assessment.py --help"
```

### برای کاربران Windows:

```powershell
# PowerShell Script

$SOURCE = "project_files"
$TARGET = "$HOME\dtcwt_popularity"

Write-Host "🔧 شروع نصب..." -ForegroundColor Green

# 1. کپی data/loaders
Write-Host "📁 کپی data/loaders..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path "$TARGET\data"
Copy-Item -Recurse -Force "$SOURCE\data\loaders" "$TARGET\data\"

# 2. کپی evaluation
Write-Host "📁 کپی evaluation..." -ForegroundColor Yellow
Copy-Item -Force "$SOURCE\evaluation\metrics.py" "$TARGET\evaluation\"
Copy-Item -Force "$SOURCE\evaluation\__init__.py" "$TARGET\evaluation\"
Copy-Item -Force "$SOURCE\evaluation\temporal_evaluator.py" "$TARGET\evaluation\"
Copy-Item -Force "$SOURCE\evaluation\results_analyzer.py" "$TARGET\evaluation\"

# 3. حذف فایل قدیمی
Write-Host "🗑️  حذف metrics_v2.py..." -ForegroundColor Yellow
Remove-Item -Force "$TARGET\evaluation\metrics_v2.py" -ErrorAction SilentlyContinue

# 4. کپی experiments
Write-Host "📁 کپی experiments..." -ForegroundColor Yellow
Copy-Item -Force "$SOURCE\experiments\run_popularity_assessment.py" "$TARGET\experiments\"

# 5. کپی مستندات
Write-Host "📚 کپی مستندات..." -ForegroundColor Yellow
Copy-Item -Force "$SOURCE\IMPLEMENTATION_GUIDE.md" "$TARGET\"
New-Item -ItemType Directory -Force -Path "$TARGET\docs"
Copy-Item -Recurse -Force "$SOURCE\docs\*" "$TARGET\docs\"

Write-Host "✅ نصب کامل شد!" -ForegroundColor Green
```

---

## ✅ Checklist نصب

بعد از نصب، موارد زیر را بررسی کنید:

### فایل‌ها:
- [ ] `data/loaders/base_loader.py` وجود دارد
- [ ] `data/loaders/movielens_loader.py` وجود دارد
- [ ] `data/loaders/__init__.py` وجود دارد
- [ ] `evaluation/metrics.py` وجود دارد
- [ ] `evaluation/metrics_v2.py` حذف شد ❌

### Import ها:
- [ ] `from data.loaders import MovieLensLoader` کار می‌کند
- [ ] `from evaluation import MetricsCalculator` کار می‌کند

### تست:
- [ ] Help text فارسی است:
```bash
python experiments/run_popularity_assessment.py --help
```

---

## 🐛 عیب‌یابی

### مشکل 1: ImportError
```
ModuleNotFoundError: No module named 'data.loaders'
```

**حل:**
```bash
# مطمئن شوید در دایرکتوری صحیح هستید
cd ~/dtcwt_popularity
pwd  # باید ~/dtcwt_popularity را نشان دهد

# بررسی فایل‌ها
ls data/loaders/
# باید: __init__.py, base_loader.py, movielens_loader.py
```

### مشکل 2: metrics_v2 یافت نشد
```
ModuleNotFoundError: No module named 'evaluation.metrics_v2'
```

**حل:**
```bash
# فایل قدیمی را حذف کنید
rm -f evaluation/metrics_v2.py

# فایل جدید را کپی کنید
cp project_files/evaluation/metrics.py evaluation/

# فایل‌های اصلاح شده را هم کپی کنید
cp project_files/evaluation/__init__.py evaluation/
cp project_files/evaluation/temporal_evaluator.py evaluation/
cp project_files/evaluation/results_analyzer.py evaluation/
```

### مشکل 3: فایل قدیمی __init__.py
```bash
# backup کنید
mv data/loaders/__init__.py data/loaders/__init__.py.old

# فایل جدید را کپی کنید
cp project_files/data/loaders/__init__.py data/loaders/
```

---

## 📊 اطلاعات فایل‌ها

### فایل‌های جدید:
```
data/loaders/__init__.py              24 خط
data/loaders/base_loader.py          300 خط
data/loaders/movielens_loader.py     262 خط
```

### فایل‌های اصلاح شده:
```
evaluation/metrics.py                292 خط (تغییر نام)
evaluation/__init__.py                28 خط (اصلاح import)
evaluation/temporal_evaluator.py     565 خط (اصلاح import)
evaluation/results_analyzer.py       405 خط (اصلاح import)
experiments/run_popularity_assessment.py  348 خط (بهبود help)
```

### مستندات:
```
IMPLEMENTATION_GUIDE.md              674 خط
README.md                            402 خط
FILE_LIST.md                         221 خط
docs/IMPLEMENTATION_SPECIFICATION.md 721 خط
docs/DATE_FILTERING_GUIDE.md         412 خط
docs/WORKFLOW_DIAGRAM.md             423 خط
docs/NAMING_UPDATE_FINAL_REPORT.md   299 خط
docs/IMPLEMENTATION_PROGRESS.md      242 خط
docs/QUICK_REFERENCE.md              331 خط
```

**جمع:** 19 فایل

---

## 🎯 مراحل بعدی

بعد از نصب موفق:

### 1. تست سریع:
```bash
python experiments/run_popularity_assessment.py movielens \
    --num-items 100 \
    --start-date 2023-08-01 \
    --end-date 2023-08-31
```

### 2. مطالعه مستندات:
- `README.md` - راهنمای کلی
- `IMPLEMENTATION_GUIDE.md` - دستورالعمل کامل
- `docs/QUICK_REFERENCE.md` - مرجع سریع

### 3. Phase بعدی:
به `docs/IMPLEMENTATION_PROGRESS.md` مراجعه کنید.

---

## 📞 پشتیبانی

برای مشکلات:
1. `README.md` را بخوانید
2. `docs/QUICK_REFERENCE.md` را بررسی کنید
3. لاگ‌های error را بررسی کنید

---

**موفق باشید!** 🎉✨
