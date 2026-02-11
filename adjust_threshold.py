"""
Quick Fix: Adjust Skew Threshold for Better Class Balance

This script allows you to experiment with different skew thresholds
to find one that gives better class balance.

Current problem:
- Threshold = 2.0 (max >= 2x avg) → 1.68% skewed (too few)
- Need: 10-20% skewed for better model training

Usage:
    python adjust_threshold.py --threshold 1.5
    python adjust_threshold.py --threshold 1.3 --sample-size 1000000
"""

import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import pandas as pd
from data_loader import get_sample_data
from preprocessing import clean_task_events, extract_task_runtimes
from feature_engineering import extract_pre_execution_features, encode_categorical_features
from skew_labeling import compute_job_runtime_stats


def test_threshold(threshold: float, sample_size: int = 500000):
    """
    Test different skew thresholds to find optimal class balance
    """
    print("=" * 80)
    print(f"TESTING SKEW THRESHOLD = {threshold:.2f}")
    print("=" * 80)
    
    # Load data
    print(f"\nLoading {sample_size:,} rows...")
    df = get_sample_data(n_rows=sample_size)
    print(f"Loaded {len(df):,} task events")
    
    # Clean and extract runtimes
    df_clean = clean_task_events(df)
    runtimes = extract_task_runtimes(df_clean)
    
    # Compute runtime statistics
    job_stats = compute_job_runtime_stats(runtimes)
    
    # Apply custom threshold
    job_stats['skewed'] = (
        job_stats['max_task_runtime'] >= threshold * job_stats['avg_task_runtime']
    ).astype(int)
    
    # Calculate statistics
    total_jobs = len(job_stats)
    skewed_jobs = job_stats['skewed'].sum()
    skewed_ratio = skewed_jobs / total_jobs
    
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Total jobs: {total_jobs:,}")
    print(f"Skewed jobs: {skewed_jobs:,} ({skewed_ratio*100:.2f}%)")
    print(f"Non-skewed jobs: {total_jobs - skewed_jobs:,} ({(1-skewed_ratio)*100:.2f}%)")
    
    # Evaluate balance
    print("\n" + "=" * 80)
    print("EVALUATION")
    print("=" * 80)
    
    if skewed_ratio < 0.05:
        status = "❌ TOO FEW SKEWED JOBS"
        recommendation = "Try LOWER threshold (e.g., 1.3 or 1.5)"
    elif skewed_ratio < 0.10:
        status = "⚠️  STILL IMBALANCED"
        recommendation = "Try slightly LOWER threshold (e.g., 1.3)"
    elif skewed_ratio <= 0.30:
        status = "✅ GOOD BALANCE"
        recommendation = "This threshold looks promising! Run full pipeline."
    else:
        status = "⚠️  TOO MANY SKEWED JOBS"
        recommendation = "Try HIGHER threshold (e.g., 1.8)"
    
    print(f"Status: {status}")
    print(f"Recommendation: {recommendation}")
    
    # Show sample statistics
    print("\n" + "=" * 80)
    print("SAMPLE STATISTICS")
    print("=" * 80)
    
    skewed_subset = job_stats[job_stats['skewed'] == 1]
    non_skewed_subset = job_stats[job_stats['skewed'] == 0]
    
    if len(skewed_subset) > 0:
        print(f"\nSkewed jobs:")
        print(f"  Avg runtime ratio: {(skewed_subset['max_task_runtime'] / skewed_subset['avg_task_runtime']).mean():.2f}x")
        print(f"  Median runtime ratio: {(skewed_subset['max_task_runtime'] / skewed_subset['avg_task_runtime']).median():.2f}x")
    
    if len(non_skewed_subset) > 0:
        print(f"\nNon-skewed jobs:")
        print(f"  Avg runtime ratio: {(non_skewed_subset['max_task_runtime'] / non_skewed_subset['avg_task_runtime']).mean():.2f}x")
        print(f"  Median runtime ratio: {(non_skewed_subset['max_task_runtime'] / non_skewed_subset['avg_task_runtime']).median():.2f}x")
    
    print("\n" + "=" * 80)
    
    return {
        'threshold': threshold,
        'total_jobs': total_jobs,
        'skewed_jobs': skewed_jobs,
        'skewed_ratio': skewed_ratio,
        'status': status
    }


def compare_thresholds(thresholds, sample_size=500000):
    """
    Compare multiple thresholds side-by-side
    """
    results = []
    
    for threshold in thresholds:
        result = test_threshold(threshold, sample_size)
        results.append(result)
    
    # Print comparison table
    print("\n" + "=" * 80)
    print("THRESHOLD COMPARISON")
    print("=" * 80)
    print(f"{'Threshold':<12} {'Total Jobs':<12} {'Skewed':<12} {'Ratio':<12} {'Status':<20}")
    print("-" * 80)
    
    for r in results:
        print(f"{r['threshold']:<12.2f} {r['total_jobs']:<12,} {r['skewed_jobs']:<12,} "
              f"{r['skewed_ratio']:<12.2%} {r['status']:<20}")
    
    print("-" * 80)
    
    # Find best threshold
    best = min(results, key=lambda x: abs(x['skewed_ratio'] - 0.15))  # Target 15%
    
    print(f"\n🎯 RECOMMENDED THRESHOLD: {best['threshold']:.2f}")
    print(f"   This gives ~{best['skewed_ratio']*100:.1f}% skewed jobs (closest to 15% target)")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Test different skew thresholds to find optimal class balance"
    )
    parser.add_argument(
        "--threshold", 
        type=float, 
        help="Single threshold to test (e.g., 1.5)"
    )
    parser.add_argument(
        "--compare", 
        action="store_true",
        help="Compare multiple thresholds (1.3, 1.5, 1.8, 2.0)"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=500000,
        help="Number of rows to sample (default: 500000)"
    )
    
    args = parser.parse_args()
    
    if args.compare:
        # Compare multiple thresholds
        thresholds = [1.3, 1.5, 1.8, 2.0]
        compare_thresholds(thresholds, args.sample_size)
    elif args.threshold:
        # Test single threshold
        test_threshold(args.threshold, args.sample_size)
    else:
        # Default: compare common thresholds
        print("No threshold specified. Running comparison mode...\n")
        thresholds = [1.3, 1.5, 1.8, 2.0]
        compare_thresholds(thresholds, args.sample_size)


if __name__ == "__main__":
    main()
