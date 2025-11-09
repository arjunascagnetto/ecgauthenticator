#!/usr/bin/env python3
"""
Evaluation script for ranking metrics (AUC, Rank@K).
Calculates authentication metrics across training epochs.

Computes:
- AUC: Area Under ROC Curve for same/different patient discrimination
- Rank@1, Rank@5, Rank@10: % of correct matches in top-K neighbors
"""

import os
import sys
import yaml
import json
import torch
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, List, Optional
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns
from tqdm import tqdm

# Optional Qt5 import
try:
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QTabWidget, QLabel, QPushButton, QGridLayout, QTextEdit
    )
    from PyQt5.QtCore import Qt
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    HAS_QT5 = True
except ImportError:
    HAS_QT5 = False

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from src.ecg_encoder import ECGEncoder
from src.ecg_metric_dataset import ECGPairDataset


# ============================================================================
# LOGGING
# ============================================================================
def setup_logging(log_file: Optional[Path] = None, level: str = "INFO"):
    """Setup logging configuration."""
    logger = logging.getLogger(__name__)
    logger.setLevel(getattr(logging, level))

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


logger = logging.getLogger(__name__)


# ============================================================================
# CONFIG LOADING
# ============================================================================
def load_config(config_path: Path = None) -> Dict:
    """Load evaluation configuration from YAML."""
    if config_path is None:
        config_path = Path(__file__).parent / "config_evaluation.yaml"

    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    logger.info(f"Config loaded from {config_path}")
    return config


# ============================================================================
# RUN DETECTION
# ============================================================================
def find_latest_run(base_dir: Path = None) -> Path:
    """Find latest training run."""
    if base_dir is None:
        base_dir = Path(__file__).parent

    runs_dir = base_dir / "runs_v2"
    if not runs_dir.exists():
        raise FileNotFoundError(f"Runs directory not found: {runs_dir}")

    # Find all run directories
    run_dirs = sorted(
        [d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith("run_")],
        key=lambda d: d.stat().st_mtime,
        reverse=True
    )

    if not run_dirs:
        raise FileNotFoundError("No training runs found")

    latest_run = run_dirs[0]
    logger.info(f"Found latest run: {latest_run.name}")
    return latest_run


def get_run_dir(config: Dict, base_dir: Path = None) -> Path:
    """Get run directory from config or find latest."""
    if base_dir is None:
        base_dir = Path(__file__).parent

    run_id = config['evaluation'].get('run_id')

    if run_id is None:
        run_dir = find_latest_run(base_dir)
    else:
        run_dir = base_dir / "runs_v2" / run_id
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")

    logger.info(f"Using run directory: {run_dir}")
    return run_dir


# ============================================================================
# MODEL SELECTION
# ============================================================================
def get_models_to_evaluate(run_dir: Path, config: Dict) -> List[Tuple[int, Path]]:
    """
    Get list of (epoch, model_path) to evaluate.

    Returns:
        List of (epoch_number, model_path) tuples
    """
    models_dir = run_dir / "models"
    if not models_dir.exists():
        raise FileNotFoundError(f"Models directory not found: {models_dir}")

    # Find all epoch models
    epoch_models = sorted(
        [
            (int(f.stem.split('_')[-1]), f)
            for f in models_dir.glob("model_epoch_*.pth")
        ],
        key=lambda x: x[0]
    )

    if not epoch_models:
        raise FileNotFoundError(f"No epoch models found in {models_dir}")

    # Filter by interval
    model_interval = config['evaluation'].get('model_interval')
    if model_interval is None:
        selected_models = epoch_models
    else:
        selected_models = [m for m in epoch_models if m[0] % model_interval == 0]

    logger.info(f"Selected {len(selected_models)} models for evaluation")
    logger.info(f"Epochs: {[e for e, _ in selected_models]}")

    return selected_models


# ============================================================================
# DATA LOADING
# ============================================================================
def load_datasets(config: Dict, run_dir: Path) -> Dict:
    """Load train/val/test datasets."""
    base_dir = run_dir.parent.parent
    config_file = list(run_dir.glob('config_*.yaml'))[0]

    with open(config_file, 'r') as f:
        train_config = yaml.safe_load(f)

    datasets = {}

    if config['evaluation']['splits'].get('train', False):
        train_csv = base_dir / train_config['data']['train_csv']
        datasets['train'] = ECGPairDataset(str(train_csv), mining_strategy='random')
        logger.info(f"Loaded train dataset: {len(datasets['train'].features)} samples")

    if config['evaluation']['splits'].get('val', True):
        val_csv = base_dir / train_config['data']['val_csv']
        datasets['val'] = ECGPairDataset(str(val_csv), mining_strategy='random')
        logger.info(f"Loaded val dataset: {len(datasets['val'].features)} samples")

    if config['evaluation']['splits'].get('test', True):
        test_csv = base_dir / train_config['data'].get('test_csv', 'data/ECG/test.csv')
        test_csv = base_dir / test_csv
        if test_csv.exists():
            datasets['test'] = ECGPairDataset(str(test_csv), mining_strategy='random')
            logger.info(f"Loaded test dataset: {len(datasets['test'].features)} samples")
        else:
            logger.warning(f"Test CSV not found: {test_csv}")

    return datasets


def sample_patients(dataset: ECGPairDataset, num_patients: int = 1000, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample N patients and return their features and patient IDs.

    Returns:
        (features, patient_ids)
    """
    np.random.seed(seed)

    unique_patients = np.unique(dataset.patient_ids)
    if len(unique_patients) <= num_patients:
        sampled_patients = unique_patients
        logger.info(f"Using all {len(unique_patients)} patients (fewer than requested {num_patients})")
    else:
        sampled_patients = np.random.choice(unique_patients, num_patients, replace=False)
        logger.info(f"Sampled {num_patients} patients from {len(unique_patients)} total")

    # Get indices for sampled patients
    mask = np.isin(dataset.patient_ids, sampled_patients)
    indices = np.where(mask)[0]

    features = dataset.features[indices]
    patient_ids = dataset.patient_ids[indices]

    logger.info(f"Sampled {len(features)} ECG samples from {len(np.unique(patient_ids))} patients")

    return features, patient_ids


# ============================================================================
# EMBEDDING COMPUTATION
# ============================================================================
def compute_embeddings(
    model_path: Path,
    features: np.ndarray,
    config: Dict,
    device: torch.device
) -> np.ndarray:
    """
    Compute embeddings for given features using a trained model.

    Returns:
        (N, embedding_dim) embeddings
    """
    # Load model config
    run_dir = model_path.parent.parent
    config_file = list(run_dir.glob('config_*.yaml'))[0]
    with open(config_file, 'r') as f:
        train_config = yaml.safe_load(f)

    # Create encoder
    encoder = ECGEncoder(
        input_dim=train_config['encoder']['input_dim'],
        hidden_dims=train_config['encoder'].get('hidden_dims', [20]),
        embedding_dim=train_config['encoder']['embedding_dim'],
        dropout=train_config['encoder']['dropout'],
        normalize=train_config['encoder']['normalize']
    ).to(device)

    # Load weights
    encoder.load_state_dict(torch.load(model_path, map_location=device))
    encoder.eval()

    # Compute embeddings
    all_embeddings = []
    batch_size = config['evaluation']['batch_size']

    with torch.no_grad():
        for i in tqdm(range(0, len(features), batch_size), desc="Computing embeddings", leave=False):
            batch_end = min(i + batch_size, len(features))
            batch = torch.from_numpy(features[i:batch_end]).float().to(device)
            embeddings = encoder(batch).cpu().numpy()
            all_embeddings.extend(embeddings)

    return np.array(all_embeddings)


# ============================================================================
# DISTANCE MATRIX
# ============================================================================
def compute_distance_matrix(embeddings: np.ndarray, metric: str = "cosine") -> np.ndarray:
    """
    Compute pairwise distance matrix.

    Args:
        embeddings: (N, D) array
        metric: "cosine" or "euclidean"

    Returns:
        (N, N) distance matrix
    """
    if metric == "cosine":
        # Cosine distance = 1 - cosine_similarity
        from sklearn.metrics.pairwise import cosine_distances
        return cosine_distances(embeddings)
    elif metric == "euclidean":
        from sklearn.metrics.pairwise import euclidean_distances
        return euclidean_distances(embeddings)
    else:
        raise ValueError(f"Unknown metric: {metric}")


# ============================================================================
# RANK METRICS
# ============================================================================
def compute_rank_metrics(
    distance_matrix: np.ndarray,
    patient_ids: np.ndarray
) -> Dict[str, float]:
    """
    Compute Rank@K metrics.

    Args:
        distance_matrix: (N, N) distance matrix
        patient_ids: (N,) patient IDs for each sample

    Returns:
        Dict with rank@1, rank@5, rank@10
    """
    ranks = {1: [], 5: [], 10: []}

    for i in range(len(patient_ids)):
        query_patient = patient_ids[i]
        distances = distance_matrix[i]

        # Sort by distance (ascending)
        sorted_indices = np.argsort(distances)

        # Skip self (first index)
        sorted_indices = sorted_indices[1:]

        for k in [1, 5, 10]:
            top_k_indices = sorted_indices[:k]
            top_k_patients = patient_ids[top_k_indices]

            # Check if any of top-k are same patient
            is_correct = np.any(top_k_patients == query_patient)
            ranks[k].append(1.0 if is_correct else 0.0)

    results = {
        'rank@1': float(np.mean(ranks[1])) * 100,
        'rank@5': float(np.mean(ranks[5])) * 100,
        'rank@10': float(np.mean(ranks[10])) * 100,
    }

    return results


# ============================================================================
# AUC COMPUTATION
# ============================================================================
def compute_auc(
    distance_matrix: np.ndarray,
    patient_ids: np.ndarray
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Compute AUC for same vs different patient discrimination.

    Args:
        distance_matrix: (N, N) distance matrix
        patient_ids: (N,) patient IDs

    Returns:
        (auc_score, fpr, tpr)
    """
    from sklearn.metrics import roc_curve, auc

    # Create labels: 1 if same patient, 0 if different
    all_distances = []
    all_labels = []

    for i in range(len(patient_ids)):
        query_patient = patient_ids[i]
        distances = distance_matrix[i]

        # Skip self
        for j in range(len(patient_ids)):
            if i != j:
                is_same = (patient_ids[j] == query_patient)
                all_distances.append(distances[j])
                all_labels.append(1.0 if is_same else 0.0)

    # Convert to numpy arrays
    all_distances = np.array(all_distances)
    all_labels = np.array(all_labels)

    # Compute ROC curve (using negative distance as score, since lower distance = more similar)
    fpr, tpr, _ = roc_curve(all_labels, -all_distances)
    auc_score = auc(fpr, tpr)

    return auc_score, fpr, tpr


# ============================================================================
# PLOTTING
# ============================================================================
def plot_metrics_evolution(
    epochs: List[int],
    metrics: Dict[str, List[float]],
    output_path: Path
):
    """Plot evolution of rank metrics across epochs."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Ranking Metrics Evolution Across Epochs', fontsize=14, fontweight='bold')

    metric_names = ['auc', 'rank@1', 'rank@5', 'rank@10']
    axes_flat = axes.flatten()

    for idx, metric in enumerate(metric_names):
        ax = axes_flat[idx]
        values = metrics.get(metric, [])

        ax.plot(epochs, values, 'o-', linewidth=2, markersize=8, color='steelblue')
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel(f'{metric.upper()} (%)', fontsize=11)
        ax.set_title(f'{metric.upper()} vs Epoch', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Add value labels on points
        for epoch, value in zip(epochs, values):
            ax.text(epoch, value + 1, f'{value:.1f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved metrics plot to {output_path}")
    plt.close()


def plot_roc_curves(
    fpr_dict: Dict[int, np.ndarray],
    tpr_dict: Dict[int, np.ndarray],
    auc_dict: Dict[int, float],
    output_path: Path
):
    """Plot ROC curves for selected epochs."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot for every Nth epoch to avoid crowding
    epochs = sorted(fpr_dict.keys())
    step = max(1, len(epochs) // 5)
    selected_epochs = epochs[::step]

    for epoch in selected_epochs:
        fpr = fpr_dict[epoch]
        tpr = tpr_dict[epoch]
        auc_score = auc_dict[epoch]

        ax.plot(fpr, tpr, label=f'Epoch {epoch} (AUC={auc_score:.3f})', linewidth=2)

    ax.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title('ROC Curves Evolution', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved ROC plot to {output_path}")
    plt.close()


# ============================================================================
# STATISTICS OUTPUT
# ============================================================================
def save_statistics(
    epochs: List[int],
    metrics: Dict[str, List[float]],
    output_path: Path
):
    """Save statistics to text file."""
    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("RANKING METRICS EVALUATION SUMMARY\n")
        f.write("=" * 80 + "\n\n")

        # Overall statistics
        f.write("OVERALL STATISTICS:\n")
        f.write("-" * 80 + "\n")

        for metric in ['auc', 'rank@1', 'rank@5', 'rank@10']:
            values = np.array(metrics.get(metric, []))
            if len(values) > 0:
                f.write(f"\n{metric.upper()}:\n")
                f.write(f"  Mean:     {np.mean(values):.2f}%\n")
                f.write(f"  Std:      {np.std(values):.2f}%\n")
                f.write(f"  Min:      {np.min(values):.2f}% (Epoch {epochs[np.argmin(values)]})\n")
                f.write(f"  Max:      {np.max(values):.2f}% (Epoch {epochs[np.argmax(values)]})\n")

        # Per-epoch details
        f.write("\n" + "=" * 80 + "\n")
        f.write("PER-EPOCH DETAILS:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Epoch':<10} {'AUC':<12} {'Rank@1':<12} {'Rank@5':<12} {'Rank@10':<12}\n")
        f.write("-" * 80 + "\n")

        for epoch, auc_val, r1, r5, r10 in zip(
            epochs,
            metrics.get('auc', []),
            metrics.get('rank@1', []),
            metrics.get('rank@5', []),
            metrics.get('rank@10', [])
        ):
            f.write(f"{epoch:<10d} {auc_val:<12.2f} {r1:<12.2f} {r5:<12.2f} {r10:<12.2f}\n")

    logger.info(f"Saved statistics to {output_path}")


# ============================================================================
# RESULTS TO CSV
# ============================================================================
def save_results_csv(
    epochs: List[int],
    metrics: Dict[str, List[float]],
    output_path: Path
):
    """Save results to CSV."""
    df = pd.DataFrame({
        'epoch': epochs,
        'auc': metrics.get('auc', []),
        'rank@1': metrics.get('rank@1', []),
        'rank@5': metrics.get('rank@5', []),
        'rank@10': metrics.get('rank@10', []),
    })

    df.to_csv(output_path, index=False)
    logger.info(f"Saved results to {output_path}")


# ============================================================================
# Qt5 GUI (OPTIONAL)
# ============================================================================
if HAS_QT5:
    class EvaluationResultsViewer(QMainWindow):
        def __init__(self, epochs: List[int], metrics: Dict[str, List[float]], stats_text: str):
            super().__init__()
            self.setWindowTitle("Ranking Metrics Evaluation Results")
            self.setGeometry(100, 100, 1400, 800)

            # Main widget
            main_widget = QWidget()
            self.setCentralWidget(main_widget)
            layout = QHBoxLayout()

            # Left: Plots
            self.fig = Figure(figsize=(10, 8))
            self.canvas = FigureCanvas(self.fig)
            layout.addWidget(self.canvas, stretch=3)

            # Right: Statistics
            stats_widget = QWidget()
            stats_layout = QVBoxLayout()
            stats_label = QLabel("Statistics:")
            stats_text_edit = QTextEdit()
            stats_text_edit.setText(stats_text)
            stats_text_edit.setReadOnly(True)
            stats_layout.addWidget(stats_label)
            stats_layout.addWidget(stats_text_edit)
            stats_widget.setLayout(stats_layout)
            layout.addWidget(stats_widget, stretch=1)

            main_widget.setLayout(layout)

            # Plot metrics
            self.plot_metrics(epochs, metrics)

        def plot_metrics(self, epochs: List[int], metrics: Dict[str, List[float]]):
            """Plot metrics on figure."""
            self.fig.clear()
            axes = self.fig.subplots(2, 2)

            metric_names = ['auc', 'rank@1', 'rank@5', 'rank@10']
            colors = ['steelblue', 'orange', 'green', 'red']

            for idx, (metric, color) in enumerate(zip(metric_names, colors)):
                ax = axes[idx // 2, idx % 2]
                values = metrics.get(metric, [])

                ax.plot(epochs, values, 'o-', linewidth=2, markersize=6, color=color)
                ax.set_xlabel('Epoch')
                ax.set_ylabel(f'{metric.upper()} (%)')
                ax.set_title(f'{metric.upper()} Evolution')
                ax.grid(True, alpha=0.3)

            self.fig.tight_layout()
            self.canvas.draw()


# ============================================================================
# MAIN EVALUATION LOOP
# ============================================================================
def main():
    """Main evaluation loop."""
    # Load config
    config = load_config()

    # Setup logging
    log_level = config['logging'].get('level', 'INFO')
    logger = setup_logging(level=log_level)

    logger.info("=" * 80)
    logger.info("RANKING METRICS EVALUATION")
    logger.info("=" * 80)

    # Get run directory
    base_dir = Path(__file__).parent
    run_dir = get_run_dir(config, base_dir)
    logger.info(f"Run directory: {run_dir}")

    # Load training config
    config_file = list(run_dir.glob('config_*.yaml'))[0]
    with open(config_file, 'r') as f:
        train_config = yaml.safe_load(f)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    # Create output directory
    output_dir = run_dir / config['output'].get('output_dir', 'evaluation_results')
    output_dir.mkdir(exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Load datasets
    logger.info("\nLoading datasets...")
    datasets = load_datasets(config, run_dir)

    if not datasets:
        logger.error("No datasets loaded!")
        return

    # Determine which dataset to evaluate
    eval_split = None
    if 'test' in datasets:
        eval_split = 'test'
    elif 'val' in datasets:
        eval_split = 'val'
    elif 'train' in datasets:
        eval_split = 'train'

    logger.info(f"Evaluating on '{eval_split}' split")
    dataset = datasets[eval_split]

    # Sample patients
    num_patients = config['evaluation'].get('num_patients', 1000)
    num_patients = None if num_patients == 0 else num_patients
    features, patient_ids = sample_patients(dataset, num_patients or len(np.unique(dataset.patient_ids)))

    # Get models
    logger.info("\nFinding models to evaluate...")
    models = get_models_to_evaluate(run_dir, config)

    # Evaluation loop
    logger.info(f"\nEvaluating {len(models)} models...")
    epochs = []
    results = {
        'auc': [],
        'rank@1': [],
        'rank@5': [],
        'rank@10': []
    }
    fpr_dict = {}
    tpr_dict = {}
    auc_dict = {}

    for epoch, model_path in tqdm(models, desc="Evaluating models"):
        logger.info(f"\n{'='*80}")
        logger.info(f"Epoch {epoch}: {model_path.name}")
        logger.info('='*80)

        # Compute embeddings
        embeddings = compute_embeddings(model_path, features, config, device)
        logger.info(f"Embeddings shape: {embeddings.shape}")

        # Compute distances
        logger.info("Computing distance matrix...")
        distance_metric = config['evaluation'].get('distance_metric', 'cosine')
        distance_matrix = compute_distance_matrix(embeddings, metric=distance_metric)
        logger.info(f"Distance matrix shape: {distance_matrix.shape}")

        # Compute metrics
        logger.info("Computing rank metrics...")
        rank_metrics = compute_rank_metrics(distance_matrix, patient_ids)

        logger.info("Computing AUC...")
        auc_score, fpr, tpr = compute_auc(distance_matrix, patient_ids)

        # Store results
        epochs.append(epoch)
        results['auc'].append(auc_score * 100)
        results['rank@1'].append(rank_metrics['rank@1'])
        results['rank@5'].append(rank_metrics['rank@5'])
        results['rank@10'].append(rank_metrics['rank@10'])
        fpr_dict[epoch] = fpr
        tpr_dict[epoch] = tpr
        auc_dict[epoch] = auc_score

        # Log results
        logger.info(f"AUC:      {auc_score*100:.2f}%")
        logger.info(f"Rank@1:   {rank_metrics['rank@1']:.2f}%")
        logger.info(f"Rank@5:   {rank_metrics['rank@5']:.2f}%")
        logger.info(f"Rank@10:  {rank_metrics['rank@10']:.2f}%")

    # Save results
    logger.info(f"\n{'='*80}")
    logger.info("SAVING RESULTS")
    logger.info('='*80)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if config['output'].get('save_results_csv', True):
        csv_path = output_dir / f"rank_metrics_{timestamp}.csv"
        save_results_csv(epochs, results, csv_path)

    if config['output'].get('save_statistics_txt', True):
        stats_path = output_dir / f"rank_metrics_statistics_{timestamp}.txt"
        save_statistics(epochs, results, stats_path)

    if config['output'].get('save_plots_png', True):
        plot_path = output_dir / f"rank_metrics_evolution_{timestamp}.png"
        plot_metrics_evolution(epochs, results, plot_path)

        if config['evaluation'].get('compute_roc_curve', True):
            roc_path = output_dir / f"rank_metrics_roc_{timestamp}.png"
            plot_roc_curves(fpr_dict, tpr_dict, auc_dict, roc_path)

    # Print summary table
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY TABLE")
    logger.info("=" * 80)
    summary_df = pd.DataFrame({
        'Epoch': epochs,
        'AUC (%)': [f"{v:.2f}" for v in results['auc']],
        'Rank@1 (%)': [f"{v:.2f}" for v in results['rank@1']],
        'Rank@5 (%)': [f"{v:.2f}" for v in results['rank@5']],
        'Rank@10 (%)': [f"{v:.2f}" for v in results['rank@10']],
    })
    logger.info("\n" + summary_df.to_string(index=False))

    # Show GUI if requested and available
    if config['output'].get('show_gui', True) and HAS_QT5:
        logger.info("\nOpening Qt5 GUI...")
        try:
            app = QApplication(sys.argv)
            viewer = EvaluationResultsViewer(epochs, results, summary_df.to_string(index=False))
            viewer.show()
            sys.exit(app.exec_())
        except Exception as e:
            logger.error(f"Error opening GUI: {e}")
    else:
        logger.info("\nEvaluation complete!")
        logger.info(f"Results saved to {output_dir}")


if __name__ == '__main__':
    main()
