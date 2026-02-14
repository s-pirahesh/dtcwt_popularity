# ⚡ Quick Start Guide — Frozen 4-Layer Evaluation Protocol

Get up and running with WSPI Popularity Assessment in 5 minutes!

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
- Compare all assessment methods including WSPI
- Display advanced features
- Save visualization to `results/figures/`

**Expected output:**
```
Pattern: Increasing Trend
AF (Baseline):            365.000
DTCWT+AF:               1234.567
WSPI (Proposed):         1456.789
```

---

## 📊 Test on Sample Data (2 minutes)

```python
# test_basic.py
import numpy as np
from methods.hybrid_assessment import HybridAssessment
from evaluation.metrics import calculate_ndcg, calculate_hit_rate, calculate_diagnostics

# Create sample time series (64 days — WSPI requires min 32 observations)
time_series = np.array([
    10, 15, 22, 35, 55, 85, 130, 195, 280, 385,
    510, 650, 810, 985, 1175, 1380, 1600, 1835, 2085, 2350,
    2630, 2925, 3235, 3560, 3900, 4255, 4625, 5010, 5410, 5825,
    6255, 6700, 7160, 7635, 8125, 8630, 9150, 9685, 10235, 10800,
    11380, 11975, 12585, 13210, 13850, 14505, 15175, 15860, 16560, 17275,
    18005, 18750, 19510, 20285, 21075, 21880, 22700, 23535, 24385, 25250,
    26130, 27025, 27935, 28860
])

# WSPI (Proposed Method — Frozen Parameters)
wspi = HybridAssessment(alpha_slope=1.0, beta_ratio=0.5, gamma_entropy=0.5)
score = wspi.assess(time_series)
print(f"WSPI Score: {score:.4f}")

# Evaluate metrics (example with random actuals)
scores_all  = np.array([score, score*0.8, score*0.6, score*0.4, score*0.2])
actuals_all = np.array([1000, 800, 500, 200, 100])

print(f"NDCG@3:      {calculate_ndcg(scores_all, actuals_all, k=3):.4f}")
print(f"CHR@3:       {calculate_hit_rate(scores_all, actuals_all, k=3):.4f}")
diag = calculate_diagnostics(scores_all, actuals_all)
print(f"Spearman ρ:  {diag['spearman_rho']:.4f}")
```

---

## 📈 Compare Methods (1 minute)

```python
# compare.py
import numpy as np
from methods.hybrid_assessment import HybridAssessment
from methods.dtcwt_assessment import DTCWTAssessment
from methods.dwt_assessment import DWTAssessment
from baselines import AccessFrequency, EWMA

ts = np.random.randint(10, 500, size=64).astype(float)

methods = {
    'AF':       AccessFrequency().assess_single(ts),
    'EWMA':     EWMA(alpha=0.3).assess_single(ts),
    'DWT+AF':   DWTAssessment().assess_single(ts),
    'DTCWT+AF': DTCWTAssessment().assess_single(ts),
    'WSPI':     HybridAssessment(alpha_slope=1.0, beta_ratio=0.5,
                                  gamma_entropy=0.5).assess_single(ts),
}

print(f"{'Method':<15} {'Score':>10}")
print("-"*25)
for name, score in methods.items():
    print(f"{name:<15} {score:>10.4f}")
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

1. **Run evaluation on MovieLens**
   ```bash
   python experiments/run_popularity_assessment.py movielens \
       --num-items 100 --start-date 2023-08-01 --end-date 2023-08-31 \
       --incremental
   ```

2. **Analyze results**
   ```bash
   # Display pre-computed metrics
   python experiments/analyze_results.py results/movielens/RUN_NAME
   
   # Recompute and save for future fast display
   python experiments/analyze_results.py results/movielens/RUN_NAME \
       --recompute --save-recomputed
   ```

3. **Show results graphically**
   ```bash
   python experiments/show_results.py results/movielens/RUN_NAME --both --show
   ```

4. **Read documentation**
   - `docs/QUICK_REFERENCE.md` — commands, metrics table, API reference
   - `docs/WORKFLOW_DIAGRAM.md` — pipeline architecture diagrams
   - `CHANGELOG.md` — full history of changes

5. **Customize configuration**
   ```python
   from evaluation import get_movielens_config
   config = get_movielens_config(
       num_items=500,
       k_list=[5, 10, 20, 50],        # custom K values
       robustness_sample_size=100,     # more noise injection samples
       spike_multiplier=5.0,           # smaller spike
   )
   ```

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
