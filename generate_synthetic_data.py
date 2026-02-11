"""
Synthetic Data Generator for Data Skew Prediction

Generates realistic synthetic job traces with controllable parameters.
Useful when real data is limited or unavailable.

Usage:
    python generate_synthetic_data.py --jobs 50000 --skew-ratio 0.15
    python generate_synthetic_data.py --help
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path


def generate_synthetic_jobs(n_jobs=50000, skew_ratio=0.15, seed=42):
    """
    Generate synthetic job dataset with controlled skew characteristics.
    
    Parameters:
    -----------
    n_jobs : int
        Number of jobs to generate
    skew_ratio : float
        Proportion of skewed jobs (e.g., 0.15 = 15%)
    seed : int
        Random seed for reproducibility
    
    Returns:
    --------
    pd.DataFrame with columns matching processed job data format
    """
    np.random.seed(seed)
    
    print(f"Generating {n_jobs:,} synthetic jobs...")
    print(f"Target skew ratio: {skew_ratio*100:.1f}%")
    print()
    
    data = []
    
    for job_id in range(n_jobs):
        # Determine if job is skewed
        is_skewed = np.random.random() < skew_ratio
        
        # Number of tasks (log-normal distribution)
        # CORRELATION: Skewed jobs tend to have MORE tasks (higher chance of stragglers)
        if is_skewed:
            num_tasks = int(np.random.lognormal(mean=5.0, sigma=1.3))  # More tasks
        else:
            num_tasks = int(np.random.lognormal(mean=3.5, sigma=1.2))  # Fewer tasks
        num_tasks = max(10, min(num_tasks, 10000))
        
        # Scheduling class (categorical: 0-3)
        # CORRELATION: Skewed jobs more common in best-effort/free tiers
        if is_skewed:
            scheduling_class = np.random.choice([0, 1, 2, 3], 
                                               p=[0.35, 0.45, 0.15, 0.05])  # More 0/1
        else:
            scheduling_class = np.random.choice([0, 1, 2, 3], 
                                               p=[0.25, 0.35, 0.25, 0.15])  # More balanced
        
        # Priority (0-11, lower = higher priority)
        # CORRELATION: Skewed jobs tend to have lower priority (higher numbers)
        if is_skewed:
            priority = np.random.choice(range(12), 
                                       p=[0.02, 0.02, 0.03, 0.04, 0.05,
                                          0.08, 0.10, 0.15, 0.20,
                                          0.12, 0.10, 0.09])  # Skewed toward high priority values
        else:
            priority = np.random.choice(range(12), 
                                       p=[0.08, 0.08, 0.08, 0.08, 0.08,
                                          0.12, 0.12, 0.12, 0.12,
                                          0.04, 0.04, 0.04])  # More uniform
        
        # Resource requests (normalized 0-1)
        # CORRELATION: Skewed jobs have HIGHER VARIANCE in resource requests
        base_cpu = 0.3 if scheduling_class >= 2 else 0.5
        base_mem = 0.4 if scheduling_class >= 2 else 0.6
        
        cpu_mean = np.clip(np.random.normal(base_cpu, 0.2), 0.01, 0.99)
        memory_mean = np.clip(np.random.normal(base_mem, 0.2), 0.01, 0.99)
        disk_mean = np.random.uniform(0.0, 0.4)
        
        # CORRELATION: Skewed jobs have higher resource variance (std)
        if is_skewed:
            cpu_std = np.random.uniform(0.15, 0.40)  # High variance
            memory_std = np.random.uniform(0.12, 0.35)  # High variance
            disk_std = np.random.uniform(0.05, 0.20)  # Higher variance
        else:
            cpu_std = np.random.uniform(0.01, 0.15)  # Low variance
            memory_std = np.random.uniform(0.01, 0.12)  # Low variance
            disk_std = np.random.uniform(0.0, 0.08)  # Lower variance
        
        # Machine constraints (0-1, higher = more constrained)
        # CORRELATION: Skewed jobs tend to be less constrained (more flexibility = more variance)
        if is_skewed:
            constraint_mean = np.random.uniform(0.0, 0.3)  # Less constrained
        else:
            constraint_mean = np.random.uniform(0.2, 0.6)  # More constrained
        
        # Generate task runtimes based on skew status
        if is_skewed:
            # Skewed job: create tasks with high variance
            avg_runtime = np.random.uniform(1e8, 1e9)  # 100M - 1B nanoseconds
            
            # Skew multiplier: how much longer is the slowest task
            skew_multiplier = np.random.uniform(2.0, 5.0)
            max_runtime = avg_runtime * skew_multiplier
            
            # High standard deviation for skewed jobs
            std_runtime = avg_runtime * np.random.uniform(0.5, 1.5)
        else:
            # Balanced job: tasks have similar runtimes
            avg_runtime = np.random.uniform(1e8, 1e9)
            
            # Max close to average for balanced jobs
            max_multiplier = np.random.uniform(1.0, 1.4)
            max_runtime = avg_runtime * max_multiplier
            
            # Low standard deviation for balanced jobs
            std_runtime = avg_runtime * np.random.uniform(0.1, 0.5)
        
        # Submit time (for time-based split)
        # Distribute jobs over 7 days = 604800 seconds
        submit_time = np.random.uniform(0, 604800)
        
        # Template ID (for template-based split)
        # Create ~100 different job templates
        template_id = np.random.randint(0, 100)
        
        data.append({
            'job_id': f'j_{job_id}',
            'num_tasks': num_tasks,
            'scheduling_class': int(scheduling_class),
            'priority': int(priority),
            'cpu_request_mean': cpu_mean,
            'cpu_request_std': cpu_std,
            'memory_request_mean': memory_mean,
            'memory_request_std': memory_std,
            'disk_space_request_mean': disk_mean,
            'disk_space_request_std': disk_std,
            'different_machine_constraint_mean': constraint_mean,
            'avg_task_runtime': avg_runtime,
            'max_task_runtime': max_runtime,
            'std_task_runtime': std_runtime,
            'submit_time': submit_time,
            'template_id': template_id,
            'skewed': 1 if is_skewed else 0
        })
        
        # Progress indicator
        if (job_id + 1) % 10000 == 0:
            print(f"  Generated {job_id + 1:,} / {n_jobs:,} jobs...")
    
    df = pd.DataFrame(data)
    
    # Print statistics
    print("\n" + "=" * 80)
    print("GENERATION COMPLETE")
    print("=" * 80)
    print(f"Total jobs: {len(df):,}")
    print(f"Skewed jobs: {df['skewed'].sum():,} ({df['skewed'].mean()*100:.2f}%)")
    print(f"Non-skewed jobs: {(df['skewed']==0).sum():,} ({(df['skewed']==0).mean()*100:.2f}%)")
    
    print("\n" + "=" * 80)
    print("FEATURE STATISTICS")
    print("=" * 80)
    print(f"Tasks per job: {df['num_tasks'].min():.0f} - {df['num_tasks'].max():.0f} "
          f"(median: {df['num_tasks'].median():.0f})")
    print(f"Scheduling classes: {sorted(df['scheduling_class'].unique())}")
    print(f"Priority range: {df['priority'].min():.0f} - {df['priority'].max():.0f}")
    print(f"Job templates: {df['template_id'].nunique()}")
    
    print("\n" + "=" * 80)
    print("SKEW CHARACTERISTICS")
    print("=" * 80)
    
    skewed_df = df[df['skewed'] == 1]
    non_skewed_df = df[df['skewed'] == 0]
    
    def runtime_ratio(row):
        return row['max_task_runtime'] / row['avg_task_runtime']
    
    print("Skewed jobs:")
    print(f"  Runtime ratio (max/avg): {skewed_df.apply(runtime_ratio, axis=1).mean():.2f}x "
          f"(range: {skewed_df.apply(runtime_ratio, axis=1).min():.2f}x - "
          f"{skewed_df.apply(runtime_ratio, axis=1).max():.2f}x)")
    
    print("Non-skewed jobs:")
    print(f"  Runtime ratio (max/avg): {non_skewed_df.apply(runtime_ratio, axis=1).mean():.2f}x "
          f"(range: {non_skewed_df.apply(runtime_ratio, axis=1).min():.2f}x - "
          f"{non_skewed_df.apply(runtime_ratio, axis=1).max():.2f}x)")
    
    return df


def save_dataset(df, output_path="data/processed/synthetic_jobs.csv"):
    """
    Save generated dataset to CSV file.
    """
    # Create directory if needed
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_path, index=False)
    
    print("\n" + "=" * 80)
    print("SAVED")
    print("=" * 80)
    print(f"Dataset saved to: {output_path}")
    print(f"File size: {Path(output_path).stat().st_size / 1024 / 1024:.2f} MB")
    
    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("1. Load this data in your pipeline:")
    print(f"   df = pd.read_csv('{output_path}')")
    print()
    print("2. Skip feature engineering (already done):")
    print("   X, y = prepare_features_for_training(df, mode='pre_exec')")
    print()
    print("3. Train models:")
    print("   models, scalers = train_all_models(X_train, y_train)")
    print()
    print("4. Or create a custom script:")
    print("   python train_on_synthetic.py")


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic job traces for data skew prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 50k jobs with 15% skew
  python generate_synthetic_data.py --jobs 50000 --skew-ratio 0.15
  
  # Generate 100k jobs with 20% skew
  python generate_synthetic_data.py --jobs 100000 --skew-ratio 0.20
  
  # Generate with custom output path
  python generate_synthetic_data.py --output data/my_synthetic.csv
        """
    )
    
    parser.add_argument(
        "--jobs",
        type=int,
        default=50000,
        help="Number of jobs to generate (default: 50000)"
    )
    
    parser.add_argument(
        "--skew-ratio",
        type=float,
        default=0.15,
        help="Proportion of skewed jobs, 0-1 (default: 0.15 = 15%%)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/synthetic_jobs.csv",
        help="Output CSV file path (default: data/processed/synthetic_jobs.csv)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if args.jobs < 100:
        print("ERROR: Must generate at least 100 jobs")
        return
    
    if not 0 < args.skew_ratio < 1:
        print("ERROR: Skew ratio must be between 0 and 1")
        return
    
    # Generate data
    df = generate_synthetic_jobs(
        n_jobs=args.jobs,
        skew_ratio=args.skew_ratio,
        seed=args.seed
    )
    
    # Save to file
    save_dataset(df, args.output)
    
    print("\n✅ Synthetic dataset generation complete!")


if __name__ == "__main__":
    main()
