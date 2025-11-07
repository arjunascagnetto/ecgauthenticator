import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.ecg_encoder import ECGEncoder
from src.ecg_metric_dataset import ECGPairDataset
from src.evaluation_utils import compute_clustering_metrics, compute_intra_inter_distances

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_embeddings(encoder: torch.nn.Module, dataset: ECGPairDataset,
                       device: torch.device, batch_size: int = 512) -> tuple:
    """
    Genera embeddings per tutti i samples nel dataset.

    Returns:
        (embeddings, patient_ids) - (N, D) e (N,)
    """
    encoder.eval()
    all_embeddings = []
    all_patient_ids = []

    num_batches = (len(dataset.features) + batch_size - 1) // batch_size

    with torch.no_grad():
        for i in tqdm(range(num_batches), desc="Generating embeddings"):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(dataset.features))

            x = torch.from_numpy(dataset.features[start_idx:end_idx]).to(device)
            embeddings = encoder(x).cpu().numpy()

            all_embeddings.extend(embeddings)
            all_patient_ids.extend(dataset.patient_ids[start_idx:end_idx])

    return np.array(all_embeddings), np.array(all_patient_ids)


def evaluate_on_set(encoder: torch.nn.Module, csv_path: str, device: torch.device,
                   set_name: str = "Test") -> dict:
    """
    Valuta il modello su un dataset (train/val/test).
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluating on {set_name} set: {csv_path}")
    logger.info(f"{'='*60}")

    # Load dataset
    dataset = ECGPairDataset(csv_path, mining_strategy="random")

    # Generate embeddings
    logger.info(f"Generating embeddings for {len(dataset.features)} samples...")
    embeddings, patient_ids = generate_embeddings(encoder, dataset, device, batch_size=512)

    # Compute metrics
    logger.info("Computing clustering metrics...")
    metrics = compute_clustering_metrics(embeddings, patient_ids, sample_size=5000)

    logger.info("Computing intra/inter distances...")
    distances = compute_intra_inter_distances(embeddings, patient_ids)

    # Combine results
    results = {
        'set_name': set_name,
        'num_samples': len(embeddings),
        'num_patients': len(np.unique(patient_ids)),
        'embedding_dim': embeddings.shape[1],
        'clustering_metrics': metrics,
        'distance_metrics': distances
    }

    # Log results
    logger.info(f"\n{set_name} Set Results:")
    logger.info(f"  Samples: {results['num_samples']}")
    logger.info(f"  Unique patients: {results['num_patients']}")
    logger.info(f"  Embedding dim: {results['embedding_dim']}")
    logger.info(f"\n  Clustering Metrics:")
    for metric, value in metrics.items():
        logger.info(f"    {metric}: {value:.4f}")
    logger.info(f"\n  Distance Metrics:")
    for metric, value in distances.items():
        logger.info(f"    {metric}: {value:.4f}")

    return results


def compare_with_baselines(encoder: torch.nn.Module, device: torch.device) -> dict:
    """
    Confronta il metric learning encoder con i baselines (raw 13D, autoencoder 32D)
    """
    logger.info("\n" + "="*60)
    logger.info("COMPARING WITH BASELINES")
    logger.info("="*60)

    test_csv = '/Users/arjuna/Progetti/siamese/data/ECG/test.csv'
    dataset = ECGPairDataset(test_csv, mining_strategy="random")

    # Metric Learning embeddings
    logger.info("\n1. Metric Learning (32D)...")
    ml_embeddings, patient_ids = generate_embeddings(encoder, dataset, device)
    ml_metrics = compute_clustering_metrics(ml_embeddings, patient_ids, sample_size=5000)

    # Raw 13D baseline
    logger.info("\n2. Raw 13D Features (baseline)...")
    raw_13d = (dataset.features - dataset.scaler_mean) / (dataset.scaler_std + 1e-8)
    raw_metrics = compute_clustering_metrics(raw_13d, patient_ids, sample_size=5000)

    # Load autoencoder 32D baseline
    logger.info("\n3. Autoencoder 32D (baseline)...")
    ae_model_path = '/Users/arjuna/Progetti/siamese/models/autoencoder_best.pth'
    if os.path.exists(ae_model_path):
        from src.autoencoder import Autoencoder
        ae = Autoencoder(input_dim=13, hidden_dim=20, latent_dim=32).to(device)
        ae.load_state_dict(torch.load(ae_model_path))

        ae_embeddings = []
        with torch.no_grad():
            for i in tqdm(range(0, len(dataset.features), 512), desc="AE embeddings"):
                x = torch.from_numpy(dataset.features[i:i+512]).to(device)
                latent = ae.encode(x).cpu().numpy()
                ae_embeddings.extend(latent)
        ae_embeddings = np.array(ae_embeddings)
        ae_metrics = compute_clustering_metrics(ae_embeddings, patient_ids, sample_size=5000)
    else:
        logger.warning("Autoencoder model not found, skipping...")
        ae_metrics = None

    # Comparison table
    comparison = {
        'metric_learning_32d': ml_metrics,
        'raw_13d': raw_metrics,
        'autoencoder_32d': ae_metrics
    }

    logger.info("\n" + "-"*60)
    logger.info("COMPARISON SUMMARY")
    logger.info("-"*60)

    metrics_to_compare = ['calinski_harabasz', 'davies_bouldin', 'silhouette', 'between_within_ratio']

    for metric_name in metrics_to_compare:
        logger.info(f"\n{metric_name}:")
        logger.info(f"  Metric Learning:  {ml_metrics.get(metric_name, 'N/A'):.4f}")
        logger.info(f"  Raw 13D:           {raw_metrics.get(metric_name, 'N/A'):.4f}")
        if ae_metrics:
            logger.info(f"  Autoencoder 32D:   {ae_metrics.get(metric_name, 'N/A'):.4f}")

    return comparison


def main():
    """Main evaluation script"""

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    MODEL_PATH = '/Users/arjuna/Progetti/siamese/models/ecg_encoder_best.pth'
    RESULTS_DIR = Path('/Users/arjuna/Progetti/siamese/results')
    RESULTS_DIR.mkdir(exist_ok=True)

    logger.info(f"Device: {DEVICE}")

    # Load encoder
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model not found: {MODEL_PATH}")
        return

    logger.info(f"Loading model from {MODEL_PATH}")
    encoder = ECGEncoder(input_dim=13, hidden_dim=20, embedding_dim=32).to(DEVICE)
    encoder.load_state_dict(torch.load(MODEL_PATH))

    # Evaluate on all sets
    results = {}

    train_csv = '/Users/arjuna/Progetti/siamese/data/ECG/train.csv'
    results['train'] = evaluate_on_set(encoder, train_csv, DEVICE, "Train")

    val_csv = '/Users/arjuna/Progetti/siamese/data/ECG/val.csv'
    results['val'] = evaluate_on_set(encoder, val_csv, DEVICE, "Validation")

    test_csv = '/Users/arjuna/Progetti/siamese/data/ECG/test.csv'
    results['test'] = evaluate_on_set(encoder, test_csv, DEVICE, "Test")

    # Compare with baselines
    comparison = compare_with_baselines(encoder, DEVICE)
    results['comparison'] = comparison

    # Save results
    output_path = RESULTS_DIR / 'metric_learning_evaluation.json'
    logger.info(f"\nSaving results to {output_path}")

    # Convert numpy types to native Python types for JSON serialization
    def convert_to_native(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(item) for item in obj]
        return obj

    results_serializable = convert_to_native(results)

    with open(output_path, 'w') as f:
        json.dump(results_serializable, f, indent=2)

    logger.info(f"Results saved successfully!")


if __name__ == '__main__':
    main()
