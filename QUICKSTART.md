# ⚡ Quick Start Guide

Get up and running with DTCWT Popularity Assessment in 5 minutes!

---

## 🚀 Installation (1 minute)

```bash
# Clone the repository
git clone <your-repo-url>
cd dtcwt_popularity

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🎮 Run Demo (30 seconds)

```bash
python demo.py
```

This will:
- Show different popularity patterns
- Compare all assessment methods
- Display advanced features
- Save visualization to `results/figures/`

**Expected output:**
```
Pattern: Increasing Trend
AF (Baseline):            365.000
DTCWT + AF:              1234.567
Hybrid V3.1:             1456.789
```

---

## 📊 Test on Sample Data (2 minutes)

Create a simple test:

```python
# test_basic.py
import numpy as np
from methods.hybrid_assessment import HybridAssessment

# Create sample time series (30 days of access data)
time_series = np.array([
    10, 15, 22, 35, 55, 85, 130, 195, 280, 385,
    510, 650, 810, 985, 1175, 1380, 1600, 1835, 2085, 2350,
    2630, 2925, 3235, 3560, 3900, 4255, 4625, 5010, 5410, 5825
])

# Assess popularity
hybrid = HybridAssessment()
score = hybrid.assess(time_series)
features = hybrid.get_comprehensive_features(time_series)

print(f"Popularity Score: {score:.2f}")
print(f"\nKey Features:")
print(f"  Mean: {features['mean']:.2f}")
print(f"  Skewness: {features['skewness']:.2f}")
print(f"  Hurst Exponent: {features['hurst_exponent']:.2f}")
print(f"  Shannon Entropy: {features['shannon_entropy']:.2f}")
```

Run it:
```bash
python test_basic.py
```

---

## 📈 Compare Methods (1 minute)

```python
# compare.py
import numpy as np
from methods import *
from baselines.traditional import TraditionalBaselines

# Sample data
ts = np.array([50, 65, 80, 95, 110, 125, 140, 155, 170, 185,
               200, 215, 230, 245, 260, 275, 290, 305, 320, 335,
               350, 365, 380, 395, 410, 425, 440, 455, 470, 485])

# Compare all methods
methods = {
    'AF': TraditionalBaselines.access_frequency(ts),
    'EWMA': TraditionalBaselines.ewma_score(ts),
    'DWT': DWTAssessment().assess(ts),
    'DTCWT': DTCWTAssessment().assess(ts),
    'Statistical': StatisticalAssessment().assess(ts),
    'Hybrid': HybridAssessment().assess(ts),
}

for name, score in methods.items():
    print(f"{name:15} {score:10.2f}")
```

---

## 🔬 Run Full Experiment (Optional)

**Note:** Requires dataset files in `data/datasets/`

```bash
python experiments/exp1_assessment_comparison.py
```

This will:
1. Load YouTube-07 dataset
2. Create time series for all items
3. Compare all methods
4. Generate results table
5. Create comparison plot

**Output:**
- `results/tables/youtube07_assessment_results.csv`
- `results/figures/youtube07_comparison.png`

---

## 📦 Prepare Your Dataset

### Option 1: Use Sample Data

Create a CSV file `data/datasets/my_data.csv`:

```csv
timestamp,item_id,count
2024-01-01,video_1,100
2024-01-01,video_2,50
2024-01-02,video_1,120
2024-01-02,video_2,55
...
```

### Option 2: Generate Synthetic Data

```python
# generate_data.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Generate 100 items, 30 days
dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(30)]
items = [f'item_{i}' for i in range(100)]

data = []
for item in items:
    # Random growth pattern
    base = np.random.randint(10, 100)
    growth = np.random.uniform(1.01, 1.10)
    
    for i, date in enumerate(dates):
        count = int(base * (growth ** i) + np.random.randint(-5, 5))
        data.append({
            'timestamp': date,
            'item_id': item,
            'count': max(0, count)
        })

df = pd.DataFrame(data)
df.to_csv('data/datasets/synthetic_data.csv', index=False)
print(f"Generated {len(df)} records")
```

Then use it:

```python
from config import DATASETS
from data.loaders.youtube07 import YouTube07Loader

# Update config
DATASETS['synthetic'] = {
    'path': 'data/datasets/synthetic_data.csv',
    'time_col': 'timestamp',
    'item_col': 'item_id',
    'count_col': 'count',
}

# Load and use
loader = YouTube07Loader(DATASETS['synthetic'])
data = loader.load()
```

---

## 🎯 Next Steps

1. **Understand the Methods**
   - Read `README.md` for detailed explanations
   - Check method docstrings in `methods/`

2. **Customize Configuration**
   - Edit `config.py` to adjust weights and parameters
   - Try different wavelet families

3. **Add Your Dataset**
   - Create a custom loader in `data/loaders/`
   - Follow the `BaseDataLoader` interface

4. **Run Experiments**
   - `exp1_assessment_comparison.py` - Compare methods
   - `exp2_ablation_study.py` - Feature importance (if implemented)
   - `exp3_prediction_comparison.py` - Prediction task (if implemented)

5. **Analyze Results**
   - Check `results/tables/` for CSV files
   - View `results/figures/` for plots
   - Use pandas for custom analysis

---

## 💡 Common Use Cases

### Use Case 1: Rank Items by Popularity

```python
from methods.hybrid_assessment import HybridAssessment
import numpy as np

# Your time series data
items = ['video_1', 'video_2', 'video_3', 'video_4', 'video_5']
time_series_dict = {
    'video_1': np.array([...]),  # 30 days of data
    'video_2': np.array([...]),
    'video_3': np.array([...]),
    'video_4': np.array([...]),
    'video_5': np.array([...]),
}

# Assess all items
hybrid = HybridAssessment()
scores = {}

for item, ts in time_series_dict.items():
    scores[item] = hybrid.assess(ts)

# Rank by score
ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

print("Top 3 most popular items:")
for item, score in ranked[:3]:
    print(f"  {item}: {score:.2f}")
```

### Use Case 2: Feature Extraction for ML

```python
from methods.hybrid_assessment import HybridAssessment
import numpy as np

# Extract features for machine learning
hybrid = HybridAssessment()
time_series = np.array([...])  # Your data

# Get feature vector
features = hybrid.extract_ml_features(time_series)
# features = [dtcwt_score, mean, std, skewness, kurtosis, 
#            noise, entropy, hurst, trend, monotonicity]

print(f"Feature vector: {features}")
print(f"Feature dimension: {len(features)}")
```

### Use Case 3: Identify Viral Content

```python
from methods.advanced_features import AdvancedFeatures
import numpy as np

time_series = np.array([...])  # Your data

# Check for viral characteristics
hurst = AdvancedFeatures.hurst_exponent(time_series)
entropy = AdvancedFeatures.shannon_entropy(time_series)

if hurst > 0.7 and entropy > 2.0:
    print("⚡ Viral content detected!")
    print(f"  High persistence (H={hurst:.2f})")
    print(f"  High complexity (E={entropy:.2f})")
else:
    print("📊 Normal content")
```

---

## 🐛 Troubleshooting

**Error: `ModuleNotFoundError: No module named 'dtcwt'`**
```bash
pip install dtcwt
```

**Error: `FileNotFoundError: Dataset file not found`**
- Make sure dataset is in `data/datasets/`
- Check path in `config.py`

**Warning: `dtcwt library not available`**
- DWT will be used as fallback
- Install: `pip install dtcwt`

**Low scores for all methods**
- Check if time series has sufficient data (>20 points)
- Verify data is not all zeros
- Try different normalization in preprocessing

---

## 📚 Learn More

- **Full Documentation**: See `README.md`
- **Method Details**: Read docstrings in `methods/`
- **Configuration**: Check `config.py` comments
- **Examples**: Look at `demo.py` and `experiments/`

---

**Ready to go! 🚀**

Start with `python demo.py` and explore from there!
