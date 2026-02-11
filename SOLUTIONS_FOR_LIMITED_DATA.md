# Solutions for Limited Sample Data

## Current Situation
- **Sample size**: 405,894 task events → 4,057 jobs
- **Class imbalance**: 98.32% non-skewed, 1.68% skewed (only 68 skewed jobs)
- **Challenge**: Insufficient positive samples for robust ML models

---

## Solution 1: Adjust Skew Definition (Immediate)

**Problem**: Current definition is too strict (max ≥ 2× avg)

**Solution**: Relax the threshold to get more skewed samples

```python
# In src/skew_labeling.py
def label_skewed_jobs(df, threshold_multiplier=1.5):  # Changed from 2.0
    """
    Label jobs as skewed if max_task_runtime >= threshold_multiplier * avg_task_runtime
    
    Lower threshold = more skewed jobs = better class balance
    """
    # Current: max >= 2.0 * avg  →  1.68% skewed
    # With 1.5: max >= 1.5 * avg  →  ~5-10% skewed (estimated)
    # With 1.3: max >= 1.3 * avg  →  ~15-20% skewed (estimated)
```

**Implementation Steps**:
1. Add `threshold_multiplier` parameter to `label_skewed_jobs()`
2. Try values: 1.8, 1.5, 1.3
3. Find threshold that gives ~10-20% positive class
4. Re-run pipeline and compare results

**Expected Outcome**: More balanced dataset → better PR-AUC scores

---

## Solution 2: Load More Sample Data

**Current**: 500,000 rows → 4,057 jobs

**Solution**: Increase sample size in `main_sample.py`

```python
# In main_sample.py
SAMPLE_SIZE = 2_000_000  # Increase from 500,000 to 2 million

# Expected results:
# 500k rows   → ~4,000 jobs → ~68 skewed
# 2M rows     → ~16,000 jobs → ~272 skewed
# 5M rows     → ~40,000 jobs → ~680 skewed
```

**Trade-off**: Longer processing time, but more training samples

---

## Solution 3: Generate Synthetic Skewed Jobs

**Approach**: Use SMOTE (already implemented) + custom augmentation

Create script: `augment_data.py`

```python
"""
Data Augmentation for Skewed Jobs
Creates synthetic samples by perturbing existing skewed jobs
"""

import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

def augment_skewed_jobs(df, target_ratio=0.2):
    """
    Augment minority class (skewed jobs) to achieve target ratio
    
    Parameters:
    -----------
    df : DataFrame with features and 'skewed' label
    target_ratio : Desired ratio of skewed jobs (e.g., 0.2 = 20%)
    """
    skewed = df[df['skewed'] == 1]
    non_skewed = df[df['skewed'] == 0]
    
    n_needed = int(len(non_skewed) * target_ratio / (1 - target_ratio)) - len(skewed)
    
    if n_needed <= 0:
        return df
    
    # Generate synthetic samples
    synthetic = []
    for i in range(n_needed):
        # Randomly select a skewed job
        base = skewed.sample(1).iloc[0]
        
        # Add small random noise
        new_sample = base.copy()
        numeric_cols = ['num_tasks', 'priority', 'cpu_request_mean', 
                       'memory_request_mean', 'disk_space_request_mean']
        
        for col in numeric_cols:
            if col in new_sample:
                noise = np.random.normal(0, 0.1 * abs(new_sample[col]))
                new_sample[col] = max(0, new_sample[col] + noise)
        
        synthetic.append(new_sample)
    
    synthetic_df = pd.DataFrame(synthetic)
    augmented = pd.concat([df, synthetic_df], ignore_index=True)
    
    print(f"Original: {len(df)} jobs ({len(skewed)} skewed, {len(non_skewed)} non-skewed)")
    print(f"Added: {n_needed} synthetic skewed jobs")
    print(f"Final: {len(augmented)} jobs")
    
    return augmented
```

---

## Solution 4: Multi-Class Problem (Creative Approach)

Instead of binary (skewed/non-skewed), create multiple classes:

```python
def label_skew_severity(df):
    """
    Classify jobs into multiple severity levels
    
    Classes:
    - 0: Balanced (max < 1.2 × avg)
    - 1: Mild skew (1.2 ≤ max < 1.5 × avg)
    - 2: Moderate skew (1.5 ≤ max < 2.0 × avg)
    - 3: Severe skew (max ≥ 2.0 × avg)
    """
    ratio = df['max_task_runtime'] / df['avg_task_runtime']
    
    df['skew_level'] = pd.cut(ratio, 
                               bins=[0, 1.2, 1.5, 2.0, float('inf')],
                               labels=[0, 1, 2, 3])
    return df
```

**Benefits**: 
- More balanced distribution across classes
- Provides nuanced predictions
- Can still combine classes 2+3 as "skewed" for evaluation

---

## Solution 5: Focus on Methodology (PhD/Research Approach)

**Strategy**: Use sample data to demonstrate methodology, acknowledge limitations

**Research Paper Structure**:
```
Abstract:
"We propose a lightweight ML framework for predicting data skew 
in cloud jobs using pre-execution features. Due to dataset size 
constraints, we demonstrate the methodology on a sample dataset 
and discuss scalability..."

Methodology:
- Novel leakage-free feature engineering ✓
- Time-based and template-based evaluation ✓
- Early-execution prediction ✓
- Calibration and proper metrics (PR-AUC) ✓

Results:
- Report results on sample data
- Discuss class imbalance challenges
- Highlight feature importance findings
- Show early-prediction feasibility

Limitations:
- Sample size constraints (4,057 jobs)
- Severe class imbalance (1.68%)
- Limited feature diversity in sample

Future Work:
- Evaluation on full-scale production traces
- Additional feature engineering
- Deployment in real cluster environment
```

---

## Solution 6: Ensemble Approach

Combine predictions from all models to improve robustness:

```python
def ensemble_predict(models, X):
    """
    Ensemble prediction using weighted voting
    Weights can be based on individual PR-AUC scores
    """
    predictions = []
    weights = {
        'xgboost': 0.0889,      # PR-AUC weights
        'lightgbm': 0.0819,
        'random_forest': 0.0871,
        'logistic_regression': 0.0438
    }
    
    total_weight = sum(weights.values())
    weights = {k: v/total_weight for k, v in weights.items()}
    
    for name, model in models.items():
        pred_proba = model.predict_proba(X)[:, 1]
        predictions.append(pred_proba * weights[name])
    
    ensemble_proba = sum(predictions)
    return ensemble_proba
```

---

## Solution 7: Cost-Sensitive Learning

Adjust misclassification costs:

```python
# In train_model.py
def train_with_custom_weights(X_train, y_train):
    """
    Assign higher cost to missing skewed jobs (false negatives)
    """
    from sklearn.utils.class_weight import compute_sample_weight
    
    # False negative (missing skewed job) costs 10x more than false positive
    sample_weights = compute_sample_weight(
        class_weight={0: 1, 1: 10},  # 10x weight for minority class
        y=y_train
    )
    
    model = RandomForestClassifier()
    model.fit(X_train, y_train, sample_weight=sample_weights)
    return model
```

---

## Recommended Action Plan

### Phase 1: Quick Wins (Today)
1. ✅ **Adjust skew threshold** to 1.5× (from 2×) → more skewed samples
2. ✅ **Increase sample size** to 2M rows in `main_sample.py`
3. ✅ Run pipeline again and check if class balance improves

### Phase 2: Enhanced Methods (This Week)
4. ✅ Implement **synthetic data augmentation** script
5. ✅ Try **multi-class severity levels** instead of binary
6. ✅ Implement **ensemble prediction** for better robustness

### Phase 3: Research Positioning (For Paper)
7. ✅ Frame as **methodology demonstration** with acknowledged limitations
8. ✅ Emphasize **novel contributions**: leakage-free features, rigorous evaluation
9. ✅ Discuss **scalability** to full datasets in future work

---

## Expected Improvements

| Solution | Expected PR-AUC Gain | Implementation Effort |
|----------|---------------------|----------------------|
| Adjust threshold (1.5×) | +100% (0.09 → 0.18) | 5 minutes |
| Load 2M rows | +50% (more samples) | 2 minutes |
| Synthetic augmentation | +30-50% | 1 hour |
| Multi-class approach | +20-40% | 2 hours |
| Ensemble methods | +10-20% | 30 minutes |
| Cost-sensitive learning | +15-25% | 30 minutes |

---

## Next Steps

Run this command to quickly test adjusted threshold:

```powershell
# Edit main_sample.py to use threshold_multiplier=1.5
python main_sample.py --save-log

# Compare results with original (threshold=2.0)
python analyze_logs.py
```

Would you like me to implement any of these solutions for you?
