# Alternative Datasets for Data Skew Prediction

## Overview
If you cannot access the full Google Cluster Traces dataset, here are alternative publicly available datasets suitable for workload/skew analysis.

---

## ✅ Option 1: Alibaba Cluster Trace (RECOMMENDED)

**Source**: Alibaba Cloud  
**URL**: https://github.com/alibaba/clusterdata  
**Size**: Multiple versions available
- **2017 version**: ~1300 machines, 12 days of data
- **2018 version**: ~4000 machines, 8 days of data  
- **2020 version**: Microservices traces

**Data Format**:
- Similar to Google traces
- Contains: job_id, task_id, resource requests, execution times
- Better class balance than Google sample

**Advantages**:
- ✅ Similar structure to Google traces
- ✅ Well-documented
- ✅ Manageable size (~50GB compressed)
- ✅ More skewed jobs (better class balance)
- ✅ Actively maintained

**How to Use**:
```powershell
# Download
wget https://github.com/alibaba/clusterdata/blob/master/cluster-trace-v2018/data.tar.gz

# Your existing pipeline needs minimal changes:
# 1. Update data_loader.py to handle Alibaba format
# 2. Column mapping: Alibaba → Google format
# 3. Same feature engineering pipeline
```

**Compatibility**: ~80% compatible with your current code (minor format differences)

---

## ✅ Option 2: Azure Public Dataset

**Source**: Microsoft Azure  
**URL**: https://github.com/Azure/AzurePublicDataset  
**Size**: ~2TB raw, sample versions available

**Data Format**:
- VM traces and metrics
- Contains execution times, resource allocations
- Job-level and task-level data

**Advantages**:
- ✅ Production traces from Azure cloud
- ✅ Rich resource utilization data
- ✅ Sample versions available (50GB-200GB)
- ✅ Good for skew analysis

**Compatibility**: ~60% compatible (requires data loader modifications)

---

## ✅ Option 3: Bitbrains Traces (SMALL & MANAGEABLE)

**Source**: VU University Amsterdam  
**URL**: http://gwa.ewi.tudelft.nl/datasets/gwa-t-12-bitbrains  
**Size**: ~2GB (SMALL - easy to work with!)

**Data Format**:
- Fastcache and Rnd traces
- 1,750 VMs over 2 months
- CPU, memory, disk metrics

**Advantages**:
- ✅ **Very small** - downloads in minutes
- ✅ Well-structured CSV files
- ✅ Easy to process
- ✅ Contains resource utilization
- ✅ Suitable for master's thesis / small projects

**Disadvantages**:
- ❌ VM-level (not job/task level) - requires adaptation
- ❌ Smaller scale than Google/Alibaba

**Use Case**: Good for **proof-of-concept** and methodology demonstration

---

## ✅ Option 4: Facebook Hadoop Traces

**Source**: Facebook (archived)  
**URL**: https://github.com/SWIMProjectUCB/SWIM/wiki/Workloads-repository  
**Size**: Various (100MB - 10GB)

**Data Format**:
- Hadoop job traces
- Job submission times, durations, tasks
- MapReduce workloads

**Advantages**:
- ✅ Job-level data (perfect fit!)
- ✅ Task execution times available
- ✅ Real production workloads
- ✅ Multiple traces available

**Note**: Some traces are older (2009-2011) but still valid for research

---

## ✅ Option 5: Generate Synthetic Data (CUSTOM)

**Approach**: Create realistic synthetic dataset based on your sample

**Tool**: Create `generate_synthetic_dataset.py`

```python
"""
Generate synthetic workload traces with controlled skew characteristics
"""

import pandas as pd
import numpy as np
from scipy import stats

def generate_synthetic_jobs(n_jobs=50000, skew_ratio=0.15):
    """
    Generate synthetic job dataset with controllable parameters
    
    Parameters:
    -----------
    n_jobs : int
        Number of jobs to generate
    skew_ratio : float
        Proportion of skewed jobs (e.g., 0.15 = 15%)
    """
    np.random.seed(42)
    
    data = []
    
    for job_id in range(n_jobs):
        # Determine if job is skewed
        is_skewed = np.random.random() < skew_ratio
        
        # Number of tasks (log-normal distribution)
        num_tasks = int(np.random.lognormal(mean=4, sigma=1.5))
        num_tasks = max(10, min(num_tasks, 10000))
        
        # Scheduling class (categorical)
        scheduling_class = np.random.choice([0, 1, 2, 3], 
                                           p=[0.3, 0.4, 0.2, 0.1])
        
        # Priority
        priority = np.random.choice(range(12))
        
        # Resource requests
        cpu_mean = np.random.uniform(0.1, 0.9)
        cpu_std = np.random.uniform(0.01, 0.2)
        memory_mean = np.random.uniform(0.1, 0.8)
        memory_std = np.random.uniform(0.01, 0.15)
        disk_mean = np.random.uniform(0.0, 0.3)
        disk_std = np.random.uniform(0.0, 0.1)
        
        # Generate task runtimes
        if is_skewed:
            # Skewed job: one or more tasks take much longer
            avg_runtime = np.random.uniform(1e8, 1e9)
            # Create skew by making max 2-5x larger than avg
            max_runtime = avg_runtime * np.random.uniform(2.0, 5.0)
            std_runtime = avg_runtime * np.random.uniform(0.5, 1.5)
        else:
            # Balanced job: max close to avg
            avg_runtime = np.random.uniform(1e8, 1e9)
            max_runtime = avg_runtime * np.random.uniform(1.0, 1.4)
            std_runtime = avg_runtime * np.random.uniform(0.1, 0.5)
        
        data.append({
            'job_id': job_id,
            'num_tasks': num_tasks,
            'scheduling_class': scheduling_class,
            'priority': priority,
            'cpu_request_mean': cpu_mean,
            'cpu_request_std': cpu_std,
            'memory_request_mean': memory_mean,
            'memory_request_std': memory_std,
            'disk_space_request_mean': disk_mean,
            'disk_space_request_std': disk_std,
            'different_machine_constraint_mean': np.random.uniform(0, 0.3),
            'avg_task_runtime': avg_runtime,
            'max_task_runtime': max_runtime,
            'std_task_runtime': std_runtime,
            'skewed': 1 if is_skewed else 0
        })
    
    df = pd.DataFrame(data)
    
    print(f"Generated {n_jobs} jobs")
    print(f"Skewed: {df['skewed'].sum()} ({df['skewed'].mean()*100:.2f}%)")
    print(f"Non-skewed: {(1-df['skewed']).sum()} ({(1-df['skewed']).mean()*100:.2f}%)")
    
    return df

# Usage
synthetic_data = generate_synthetic_jobs(n_jobs=50000, skew_ratio=0.15)
synthetic_data.to_csv('data/processed/synthetic_jobs.csv', index=False)
```

**Advantages**:
- ✅ **Complete control** over class balance
- ✅ **Instant availability** - no download needed
- ✅ Can generate any size dataset
- ✅ Perfect for ablation studies

**Disadvantages**:
- ❌ Not real-world data
- ❌ May not capture all complexities
- ❌ Less credible for publication

**Best Use**: Supplement real data, test methodology, ablation studies

---

## ✅ Option 6: Grid Workload Archive (GWA)

**Source**: TU Delft / Multiple sources  
**URL**: http://gwa.ewi.tudelft.nl/  
**Size**: Various (100MB - 50GB)

**Collections Include**:
- DAS-2 traces
- LANL traces  
- LCG traces
- SHARCNET traces

**Advantages**:
- ✅ Multiple diverse datasets
- ✅ Academic-friendly (widely used in papers)
- ✅ Well-documented
- ✅ Free download

**Format**: Standard Workload Format (SWF) - needs parsing

---

## 📊 Dataset Comparison Table

| Dataset | Size | Jobs | Timespan | Download Time | Code Changes | Recommended? |
|---------|------|------|----------|---------------|--------------|--------------|
| **Alibaba 2018** | ~50GB | ~thousands | 8 days | ~2 hours | Minimal | ⭐⭐⭐⭐⭐ |
| **Azure** | 50-200GB | thousands | varies | ~4 hours | Moderate | ⭐⭐⭐⭐ |
| **Bitbrains** | ~2GB | 1,750 VMs | 2 months | 10 min | Moderate | ⭐⭐⭐ |
| **Facebook Hadoop** | 1-10GB | thousands | varies | 30 min | Minimal | ⭐⭐⭐⭐ |
| **Synthetic** | Any | Any | N/A | Instant | None | ⭐⭐⭐ |
| **GWA Archive** | varies | varies | varies | varies | Significant | ⭐⭐⭐ |

---

## 🎯 My Recommendation

### For Immediate Results (Today):
1. **Generate synthetic data** (50,000 jobs, 15% skew ratio)
2. Test your complete pipeline
3. Validate methodology works

### For Research Paper (This Week):
1. **Download Alibaba 2018 traces** (~50GB)
2. Modify `data_loader.py` for Alibaba format
3. Run full pipeline on real production data
4. Compare results: Google sample vs Alibaba vs Synthetic

### For Best Academic Impact:
1. Use **combination**: Real data (Alibaba) + Synthetic
2. Show methodology works on multiple datasets
3. Discuss generalization across different cloud environments

---

## 🚀 Quick Start: Alibaba Dataset

### Step 1: Download
```powershell
# Option A: Direct download
Invoke-WebRequest -Uri "https://github.com/alibaba/clusterdata/releases/download/v2018/batch_task.tar.gz" -OutFile "alibaba_traces.tar.gz"

# Option B: Clone repository
git clone https://github.com/alibaba/clusterdata.git
```

### Step 2: Extract
```powershell
tar -xzf alibaba_traces.tar.gz -C data/raw/
```

### Step 3: Modify Data Loader

Create `src/data_loader_alibaba.py`:

```python
def load_alibaba_traces(file_path="data/raw/batch_task.csv"):
    """
    Load Alibaba cluster traces and convert to Google format
    
    Alibaba columns → Google columns:
    - task_name → job_id
    - instance_num → task_id  
    - start_time → time (submit)
    - end_time → time (finish)
    - plan_cpu, plan_mem → resource_request
    """
    df = pd.read_csv(file_path)
    
    # Map to your expected format
    df_mapped = df.rename(columns={
        'task_name': 'job_id',
        'instance_num': 'task_id',
        'start_time': 'submit_time',
        'end_time': 'finish_time',
        'plan_cpu': 'cpu_request',
        'plan_mem': 'memory_request'
    })
    
    return df_mapped
```

### Step 4: Run Pipeline
```powershell
# Update main_sample.py to use load_alibaba_traces()
python main_sample.py --save-log
```

---

## 💡 Bonus: Data Preprocessing Tips

Regardless of which dataset you choose:

### 1. Balance the Dataset
```python
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

# Combine over and under sampling
over = SMOTE(sampling_strategy=0.3)  # Oversample minority to 30%
under = RandomUnderSampler(sampling_strategy=0.7)  # Undersample majority
pipeline = Pipeline([('over', over), ('under', under)])
X_resampled, y_resampled = pipeline.fit_resample(X, y)
```

### 2. Feature Engineering from New Data
```python
# Extract additional features from Alibaba/Azure data
def extract_alibaba_features(df):
    """Extract pre-execution features from Alibaba traces"""
    features = df.groupby('job_id').agg({
        'instance_num': 'count',  # num_tasks
        'plan_cpu': ['mean', 'std', 'max'],
        'plan_mem': ['mean', 'std', 'max'],
        'plan_disk': ['mean', 'std'],
        'task_type': lambda x: x.mode()[0]  # Most common type
    })
    return features
```

---

## 📚 Additional Resources

1. **Comprehensive Workload Archive**:
   - http://portal.nersc.gov/project/m888/apex/dataset.html

2. **MLCommons Storage Benchmark**:
   - https://github.com/mlcommons/storage

3. **CloudSuite Benchmarks**:
   - http://cloudsuite.ch/

4. **Research Paper Datasets**:
   - Check recent SIGMOD/VLDB papers on workload analysis
   - Many authors share datasets on GitHub

---

## ✅ Action Items

Pick ONE approach to start TODAY:

- [ ] **Quick Win**: Generate synthetic data (30 minutes)
- [ ] **Best Balance**: Download Alibaba 2018 (~2 hours)  
- [ ] **Lightweight**: Download Bitbrains (~10 minutes)
- [ ] **Adjust current data**: Change threshold to 1.5× and load 2M rows

Let me know which approach you'd like to pursue and I'll help you implement it!
