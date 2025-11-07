#!/usr/bin/env python3
"""
Evaluate clustering quality of raw 13D ECG features.

Metrics:
- Calinski-Harabasz Index (higher is better)
- Between-Within Ratio (higher is better)
- Silhouette Coefficient (range [-1, 1], higher is better) [sampled]
- Davies-Bouldin Index (lower is better) [sampled]

Usage:
    python evaluate_raw_13d.py --n-patients 5000
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import (
    calinski_harabasz_score,
    silhouette_score,
    davies_bouldin_score
)


def calculate_between_within_ratio(X, labels):
    """
    Calculate Between-cluster / Within-cluster variance ratio.

    Args:
        X: array (N, D) - data points
        labels: array (N,) - cluster labels

    Returns:
        float: between/within ratio (higher is better)
    """
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    if n_clusters < 2:
        return 0.0

    # Overall mean
    overall_mean = X.mean(axis=0)

    # Within-cluster variance
    within_variance = 0.0
    for label in unique_labels:
        cluster_points = X[labels == label]
        cluster_mean = cluster_points.mean(axis=0)
        within_variance += np.sum((cluster_points - cluster_mean) ** 2)

    # Between-cluster variance
    between_variance = 0.0
    for label in unique_labels:
        cluster_points = X[labels == label]
        n_points = len(cluster_points)
        cluster_mean = cluster_points.mean(axis=0)
        between_variance += n_points * np.sum((cluster_mean - overall_mean) ** 2)

    # Avoid division by zero
    if within_variance == 0:
        return float('inf')

    return between_variance / within_variance


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Evaluate clustering quality of raw 13D ECG features')
    parser.add_argument('--n-patients', type=int, default=5000,
                        help='Number of patients to sample for expensive metrics (default: 5000)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    args = parser.parse_args()

    # Load data
    data_path = '/Users/arjuna/Progetti/siamese/data/ECG/02filter.csv'
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)

    # ECG features (13D)
    ecg_features = [
        'VentricularRate', 'PRInterval', 'QRSDuration',
        'QTInterval', 'QTCorrected', 'PAxis', 'RAxis',
        'TAxis', 'QOnset', 'QOffset', 'POnset',
        'POffset', 'TOffset'
    ]

    print(f"\nDataset shape: {df.shape}")
    print(f"ECG features (13D): {ecg_features}")

    # Extract features and labels
    X_full = df[ecg_features].values.astype(np.float32)
    patient_ids_full = df['PatientID'].values

    # Check for NaN/Inf
    if np.any(np.isnan(X_full)) or np.any(np.isinf(X_full)):
        print("\nWARNING: Data contains NaN or Inf values. Removing...")
        valid_mask = ~(np.isnan(X_full).any(axis=1) | np.isinf(X_full).any(axis=1))
        X_full = X_full[valid_mask]
        patient_ids_full = patient_ids_full[valid_mask]

    n_samples_full = len(X_full)
    n_patients_full = len(np.unique(patient_ids_full))

    print(f"\nFull dataset:")
    print(f"  Samples:  {n_samples_full}")
    print(f"  Patients: {n_patients_full}")

    if n_samples_full < 2 or n_patients_full < 2:
        print("\nERROR: Not enough samples or patients for clustering evaluation.")
        return

    # Sample patients for expensive metrics (Silhouette, Davies-Bouldin)
    unique_patients = np.unique(patient_ids_full)
    n_patients_sample = min(args.n_patients, len(unique_patients))

    np.random.seed(args.seed)
    sampled_patients = np.random.choice(unique_patients, size=n_patients_sample, replace=False)

    # Get all samples from sampled patients
    sample_mask = np.isin(patient_ids_full, sampled_patients)
    X_sample = X_full[sample_mask]
    patient_ids_sample = patient_ids_full[sample_mask]
    n_samples_sample = len(X_sample)

    print(f"\nSampled dataset (for Silhouette & Davies-Bouldin):")
    print(f"  Patients: {n_patients_sample}")
    print(f"  Samples:  {n_samples_sample}")
    print(f"  Ratio:    {n_samples_sample/n_samples_full*100:.1f}% of full dataset")

    # Calculate metrics (ALL ON SAMPLED DATASET)
    print("\n" + "=" * 80)
    print("CLUSTERING QUALITY METRICS - RAW 13D ECG FEATURES")
    print("=" * 80)

    # 1. Calinski-Harabasz Index (SAMPLED)
    print(f"\n1. Calinski-Harabasz Index [SAMPLED: {n_patients_sample} patients, {n_samples_sample} samples]")
    print("   Computing...")
    ch_score = calinski_harabasz_score(X_sample, patient_ids_sample)
    print(f"   Score: {ch_score:.2f}  (↑ higher is better)")
    print("   Interpretation:")
    if ch_score > 100:
        print("   → Excellent cluster separation")
    elif ch_score > 50:
        print("   → Good cluster separation")
    else:
        print("   → Poor cluster separation")

    # 2. Between-Within Ratio (SAMPLED)
    print(f"\n2. Between-Within Ratio [SAMPLED: {n_patients_sample} patients, {n_samples_sample} samples]")
    print("   Computing...")
    bw_ratio = calculate_between_within_ratio(X_sample, patient_ids_sample)
    print(f"   Ratio: {bw_ratio:.4f}  (↑ higher is better)")
    print("   Interpretation:")
    if bw_ratio > 1.0:
        print("   → Between-cluster variance > Within-cluster variance (good)")
    else:
        print("   → Within-cluster variance dominates (poor separation)")

    # 3. Silhouette Coefficient (SAMPLED)
    print(f"\n3. Silhouette Coefficient [SAMPLED: {n_patients_sample} patients, {n_samples_sample} samples]")
    print("   Computing... (may take 1-2 minutes)")
    silhouette = silhouette_score(X_sample, patient_ids_sample)
    print(f"   Score: {silhouette:.4f}  (↑ higher is better, range [-1, 1])")
    print("   Interpretation:")
    if silhouette > 0.5:
        print("   → Strong cluster structure")
    elif silhouette > 0.25:
        print("   → Moderate cluster structure")
    elif silhouette > 0:
        print("   → Weak cluster structure")
    else:
        print("   → No cluster structure / overlapping clusters")

    # 4. Davies-Bouldin Index (SAMPLED)
    print(f"\n4. Davies-Bouldin Index [SAMPLED: {n_patients_sample} patients, {n_samples_sample} samples]")
    print("   Computing... (may take 30-60 seconds)")
    db_index = davies_bouldin_score(X_sample, patient_ids_sample)
    print(f"   Score: {db_index:.4f}  (↓ lower is better)")
    print("   Interpretation:")
    if db_index < 0.5:
        print("   → Excellent cluster separation")
    elif db_index < 1.0:
        print("   → Good cluster separation")
    else:
        print("   → Poor cluster separation")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Full dataset:    {n_samples_full} samples, {n_patients_full} patients")
    print(f"Sampled dataset: {n_samples_sample} samples, {n_patients_sample} patients (all metrics computed on this)")
    print(f"Features:        13D ECG (raw, normalized)")
    print("\nMetrics (all on sampled dataset):")
    print(f"  Calinski-Harabasz Index: {ch_score:.2f}")
    print(f"  Between-Within Ratio:    {bw_ratio:.4f}")
    print(f"  Silhouette Coefficient:  {silhouette:.4f}")
    print(f"  Davies-Bouldin Index:    {db_index:.4f}")

    # Save results
    results = {
        'dataset_info': {
            'full_dataset_samples': n_samples_full,
            'full_dataset_patients': n_patients_full,
            'sampled_dataset_samples': n_samples_sample,
            'sampled_dataset_patients': n_patients_sample,
            'sample_ratio_percent': round(n_samples_sample/n_samples_full*100, 2)
        },
        'metrics': {
            'calinski_harabasz': float(ch_score),
            'between_within_ratio': float(bw_ratio),
            'silhouette': float(silhouette),
            'davies_bouldin': float(db_index)
        },
        'config': {
            'n_patients_requested': args.n_patients,
            'random_seed': args.seed
        }
    }

    import json
    output_path = '/Users/arjuna/Progetti/siamese/results/raw_13d_metrics.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Results saved to: {output_path}")


if __name__ == '__main__':
    main()
