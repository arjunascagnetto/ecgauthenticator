import os
import sys
import json
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Tuple, Any
import logging
from datetime import datetime
import signal

# Aggiungi src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.ecg_encoder import ECGEncoder
from src.adaptive_contrastive_loss import (ContrastiveLoss, AdaptiveContrastiveLoss,
                                            CurriculumContrastiveLoss, MultiSimilarityLoss)
from src.mining_strategies import RandomMining, SemiHardMining, BatchHardMining, AdaptiveMining
from src.ecg_metric_dataset import ECGPairDataset, PKSampler
from src.evaluation_utils import compute_clustering_metrics

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/Users/arjuna/Progetti/siamese/logs/train_metric_ecg.log')
    ]
)
logger = logging.getLogger(__name__)


def load_training_config(config_path: str = 'train_configs.yaml') -> Dict[str, Any]:
    """
    Carica configurazione di training da YAML.
    Se il file non esiste, ritorna una configurazione di default.
    """
    # Cerca il file nella directory principale del progetto
    # Se chiamato da src/, va una directory sopra
    script_dir = Path(__file__).parent.parent  # da src/ torna a root
    config_file = script_dir / config_path

    if config_file.exists():
        print(f"Loading config from {config_file}...")
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        if config:
            return config

    # Default config
    print(f"Config file not found, using defaults")
    return {
        'training': {
            'use_curriculum': False,
            'max_epochs': 50,
            'patience': 15,
            'early_stop_ch_threshold': 5.0,
        },
        'optimizer': {
            'learning_rate': 5.0e-4,
            'weight_decay': 1.0e-5,
            'lr_scheduler_factor': 0.5,
            'lr_scheduler_patience': 5,
        },
        'loss': {
            'loss_type': 'curriculum_contrastive',
            'margin_init': 1.0,
            'margin_final': 0.5,
            'alpha': 2.0,
            'beta': 50.0,
            'lambda_param': 0.5,
        },
        'mining': {
            'strategy': 'hard',
            'start_epoch': 1,
            'random_epochs': 10,
            'semihard_epochs': 25,
            'hardmining_epochs': 50,
        },
        'batch': {
            'use_pk_sampling': True,
            'batch_size': 256,
            'num_patients_per_batch': 64,
            'num_ecg_per_patient': 4,
            'shuffle': True,
        },
        'encoder': {
            'input_dim': 13,
            'hidden_dims': [20],
            'embedding_dim': 32,
            'dropout': 0.2,
            'normalize': True,
        },
        'validation': {
            'sample_size': 5000,
            'compute_silhouette': False,
            'compute_bw_ratio': False,
        },
        # Performance optimization flags
        'performance': {
            'compute_train_batch_metrics': True,
            'compute_val_batch_metrics': True,
            'compute_val_global_distances': True,
            'compute_val_db_score': True,
            'validation_frequency': 1,
        },
        'test': {
            'compute_bw_ratio': False,
            'compute_silhouette': False,
            'compute_db': True,
            'compute_global_distances': True,
        },
        'data': {
            'train_csv': '/Users/arjuna/Progetti/siamese/data/ECG/train.csv',
            'val_csv': '/Users/arjuna/Progetti/siamese/data/ECG/val.csv',
            'test_csv': '/Users/arjuna/Progetti/siamese/data/ECG/test.csv',
        },
        'output': {
            'base_dir': '/Users/arjuna/Progetti/siamese',
            'runs_dir': 'runs',
        },
        'device': 'cpu',
    }


def get_mining_strategy_for_epoch(epoch: int, random_epochs: int, semihard_epochs: int, hardmining_epochs: int) -> str:
    """
    Ritorna mining strategy per l'epoca corrente basato su progressione configurata.

    Fasi:
    - Epoch 1 to random_epochs: "random"
    - Epoch (random_epochs+1) to semihard_epochs: "semi-hard"
    - Epoch (semihard_epochs+1) to hardmining_epochs: "hard"
    - Epoch > hardmining_epochs: "hard" (rimani su hard)

    Args:
        epoch: epoca corrente (1-indexed)
        random_epochs: fine della fase random
        semihard_epochs: fine della fase semi-hard
        hardmining_epochs: fine della fase hard (di solito = max_epochs)

    Returns:
        mining_strategy: "random" | "semi-hard" | "hard"
    """
    if epoch <= random_epochs:
        return "random"
    elif epoch <= semihard_epochs:
        return "semi-hard"
    else:  # epoch > semihard_epochs
        return "hard"


def create_random_pairs_batch(batch_indices: list, features: np.ndarray, patient_ids: np.ndarray,
                              device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
    """
    Crea pairs casuali per un batch di indici (random mining).
    Molto veloce perché non usa Dataset.__getitem__

    Returns:
        anchors: tensor (N, 13)
        others: tensor (N, 13)
        labels: tensor (N,)
        anchor_patient_ids: array (N,) - patient IDs degli anchors
        other_patient_ids: array (N,) - patient IDs degli others
    """
    batch_patient_ids = patient_ids[batch_indices]
    batch_features = features[batch_indices]

    anchors = []
    others = []
    labels = []
    anchor_patient_ids = []
    other_patient_ids = []

    for i, idx in enumerate(batch_indices):
        anchor_patient = batch_patient_ids[i]

        # Positivo casuale: stesso paziente
        pos_mask = (batch_patient_ids == anchor_patient)
        pos_indices = np.where(pos_mask)[0]
        pos_indices = pos_indices[pos_indices != i]

        if len(pos_indices) > 0:
            pos_i = np.random.choice(pos_indices)
            anchors.append(batch_features[i])
            others.append(batch_features[pos_i])
            labels.append(0)  # positivo
            anchor_patient_ids.append(anchor_patient)
            other_patient_ids.append(batch_patient_ids[pos_i])

        # Negativo casuale: paziente diverso
        neg_mask = (batch_patient_ids != anchor_patient)
        neg_indices = np.where(neg_mask)[0]

        if len(neg_indices) > 0:
            neg_i = np.random.choice(neg_indices)
            anchors.append(batch_features[i])
            others.append(batch_features[neg_i])
            labels.append(1)  # negativo
            anchor_patient_ids.append(anchor_patient)
            other_patient_ids.append(batch_patient_ids[neg_i])

    anchors = torch.from_numpy(np.array(anchors)).float().to(device)
    others = torch.from_numpy(np.array(others)).float().to(device)
    labels = torch.from_numpy(np.array(labels)).float().to(device)
    anchor_patient_ids = np.array(anchor_patient_ids)
    other_patient_ids = np.array(other_patient_ids)

    return anchors, others, labels, anchor_patient_ids, other_patient_ids


def train_epoch(encoder: nn.Module, loss_fn: nn.Module, optimizer: optim.Optimizer,
                features: np.ndarray, patient_ids: np.ndarray, batch_indices_list: list,
                device: torch.device, epoch: int, mining_strategy: str = "random",
                compute_batch_metrics_flag: bool = True) -> Tuple[float, list]:
    """
    Esegui un'epoca di training e calcola metriche per ogni batch.

    Returns:
        avg_loss: loss medio su tutti i batch
        batch_metrics_list: lista di dict con metriche per ogni batch
    """
    from src.evaluation_utils import compute_batch_metrics

    encoder.train()
    total_loss = 0.0
    num_batches = 0
    batch_metrics_list = []

    pbar = tqdm(batch_indices_list, desc=f"Training [{mining_strategy}]",
                position=0, leave=True, ncols=100)

    for batch_idx, batch_indices in enumerate(pbar):
        anchors, others, labels, anchor_patient_ids, other_patient_ids = create_random_pairs_batch(
            batch_indices, features, patient_ids, device)

        optimizer.zero_grad()

        # Forward pass
        anchor_embeddings = encoder(anchors)
        other_embeddings = encoder(others)

        # Compute loss
        loss = loss_fn(anchor_embeddings, other_embeddings, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        # Calcola % negativi attivi (solo per contrastive loss con margin esplicito)
        active_negatives_pct = 0.0
        if isinstance(loss_fn, (ContrastiveLoss, AdaptiveContrastiveLoss, CurriculumContrastiveLoss)):
            # Calcola distanze cosine tra i pair
            cosine_sim = F.cosine_similarity(anchor_embeddings, other_embeddings, dim=1)
            cosine_dist = 1 - cosine_sim

            # Seleziona solo i negativi (label=1)
            neg_mask = (labels == 1.0)
            if neg_mask.sum() > 0:
                neg_distances = cosine_dist[neg_mask]
                # Margin corrente (dipende dal tipo di loss)
                if isinstance(loss_fn, AdaptiveContrastiveLoss):
                    margin = loss_fn.current_margin
                elif isinstance(loss_fn, CurriculumContrastiveLoss):
                    if loss_fn.current_phase == "warmup":
                        margin = loss_fn.margin_init
                    elif loss_fn.current_phase == "transition":
                        margin = loss_fn.margin_init * 0.75
                    else:  # hard
                        margin = loss_fn.margin_init * 0.5
                else:  # standard contrastive
                    margin = loss_fn.margin

                # Negativi attivi: quelli con d_neg < margin (violano margin)
                active_negatives = (neg_distances < margin).sum().item()
                active_negatives_pct = (active_negatives / neg_mask.sum().item()) * 100

        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'active_neg%': f'{active_negatives_pct:.1f}%'})

        # Calcola metriche batch (se abilitato nel config)
        if compute_batch_metrics_flag:
            # Combina embeddings e patient ids
            batch_embeddings = torch.cat([anchor_embeddings, other_embeddings], dim=0).detach().cpu().numpy()
            batch_patient_ids_combined = np.concatenate([anchor_patient_ids, other_patient_ids])

            batch_metrics = compute_batch_metrics(batch_embeddings, batch_patient_ids_combined)
            batch_metrics['loss'] = loss.item()
            batch_metrics['active_negatives_pct'] = active_negatives_pct
            batch_metrics_list.append(batch_metrics)
        else:
            # Se disabilitato: salva solo loss e active_negatives_pct (leggero)
            batch_metrics_list.append({
                'loss': loss.item(),
                'active_negatives_pct': active_negatives_pct
            })

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss, batch_metrics_list


@torch.no_grad()
def validate(encoder: nn.Module, val_dataset: ECGPairDataset,
             device: torch.device, sample_size: int = 5000, compute_silhouette: bool = False,
             compute_bw_ratio: bool = True, compute_db: bool = True,
             compute_global_distances: bool = True) -> Dict:
    """
    Valuta il modello su validation set e calcola clustering metrics.

    Args:
        compute_silhouette: se False, skip Silhouette (lentissimo)
        compute_bw_ratio: se False, skip Between-Within Ratio
        compute_db: se False, skip Davies-Bouldin (costoso)
        compute_global_distances: se False, skip distanze globali intra/inter
    """
    encoder.eval()

    # Genera embeddings per tutti i samples
    print(f"Computing embeddings for validation (sample_size={sample_size})...")

    all_embeddings = []
    all_patient_ids = []

    # Sample da validation set
    if sample_size is None:
        sample_size = len(val_dataset.features)

    indices = np.random.choice(len(val_dataset.features), min(sample_size, len(val_dataset.features)),
                              replace=False)

    # Batch processing per velocità
    batch_size = 256
    for batch_start in tqdm(range(0, len(indices), batch_size),
                            desc="Generating embeddings", position=0, leave=True, ncols=100):
        batch_end = min(batch_start + batch_size, len(indices))
        batch_indices = indices[batch_start:batch_end]

        x = torch.from_numpy(val_dataset.features[batch_indices]).to(device)
        embeddings = encoder(x).cpu().numpy()

        all_embeddings.extend(embeddings)
        all_patient_ids.extend(val_dataset.patient_ids[batch_indices])

    all_embeddings = np.array(all_embeddings)
    all_patient_ids = np.array(all_patient_ids)

    # Calcola clustering metrics (Davis-Bouldin è opzionale per ottimizzazione)
    print("Computing validation metrics...")
    metrics = compute_clustering_metrics(all_embeddings, all_patient_ids, sample_size=5000,
                                        compute_silhouette=compute_silhouette,
                                        compute_bw_ratio=compute_bw_ratio, compute_db=compute_db)

    # Calcola distanze intra/inter globali su tutto il validation set (se abilitato)
    if compute_global_distances:
        print("Computing global intra/inter distances...")
        from src.evaluation_utils import compute_intra_inter_distances
        distance_metrics = compute_intra_inter_distances(all_embeddings, all_patient_ids)
        metrics['d_intra_global'] = distance_metrics['intra_patient_mean']
        metrics['d_inter_global'] = distance_metrics['inter_patient_mean']
    else:
        # Se disabilitato: set defaults a 0
        metrics['d_intra_global'] = 0.0
        metrics['d_inter_global'] = 0.0

    return metrics


@torch.no_grad()
def validate_with_batch_metrics(encoder: nn.Module, val_dataset: ECGPairDataset,
                                device: torch.device, sample_size: int = 5000,
                                compute_silhouette: bool = False,
                                compute_bw_ratio: bool = True, compute_batch_metrics_flag: bool = True,
                                compute_db: bool = True, compute_global_distances: bool = True) -> Tuple[Dict, list]:
    """
    Valuta il modello su validation set e calcola clustering metrics sia aggregate che per batch.

    Returns:
        metrics: dict con metriche aggregate
        batch_metrics_list: lista di dict con metriche per ogni batch
    """
    from src.evaluation_utils import compute_batch_metrics

    encoder.eval()

    # Genera embeddings per tutti i samples
    print(f"Computing embeddings for validation (sample_size={sample_size})...")

    all_embeddings = []
    all_patient_ids = []
    batch_metrics_list = []

    # Sample da validation set
    if sample_size is None:
        sample_size = len(val_dataset.features)

    indices = np.random.choice(len(val_dataset.features), min(sample_size, len(val_dataset.features)),
                              replace=False)

    # Batch processing per velocità
    batch_size = 256
    for batch_idx, batch_start in enumerate(tqdm(range(0, len(indices), batch_size),
                            desc="Generating embeddings", position=0, leave=True, ncols=100)):
        batch_end = min(batch_start + batch_size, len(indices))
        batch_indices = indices[batch_start:batch_end]

        x = torch.from_numpy(val_dataset.features[batch_indices]).to(device)
        embeddings = encoder(x).cpu().numpy()

        all_embeddings.extend(embeddings)
        all_patient_ids.extend(val_dataset.patient_ids[batch_indices])

        # Calcola metriche batch (se abilitato nel config)
        if compute_batch_metrics_flag:
            batch_metrics = compute_batch_metrics(embeddings, val_dataset.patient_ids[batch_indices])
            batch_metrics_list.append(batch_metrics)

    all_embeddings = np.array(all_embeddings)
    all_patient_ids = np.array(all_patient_ids)

    # Calcola clustering metrics aggregate (Davies-Bouldin è opzionale per ottimizzazione)
    print("Computing validation metrics...")
    metrics = compute_clustering_metrics(all_embeddings, all_patient_ids, sample_size=5000,
                                        compute_silhouette=compute_silhouette,
                                        compute_bw_ratio=compute_bw_ratio, compute_db=compute_db)

    # Calcola distanze intra/inter globali su tutto il validation set (se abilitato)
    if compute_global_distances:
        print("Computing global intra/inter distances...")
        from src.evaluation_utils import compute_intra_inter_distances
        distance_metrics = compute_intra_inter_distances(all_embeddings, all_patient_ids)
        metrics['d_intra_global'] = distance_metrics['intra_patient_mean']
        metrics['d_inter_global'] = distance_metrics['inter_patient_mean']
    else:
        # Se disabilitato: set defaults a 0
        metrics['d_intra_global'] = 0.0
        metrics['d_inter_global'] = 0.0

    return metrics, batch_metrics_list


class GracefulShutdown:
    """Gestore per Ctrl+C con graceful shutdown e double-tap hard shutdown"""
    def __init__(self):
        self.interrupted = False
        self.interrupt_count = 0
        signal.signal(signal.SIGINT, self._handle_interrupt)

    def _handle_interrupt(self, signum, frame):
        """Handler per SIGINT (Ctrl+C)"""
        self.interrupt_count += 1

        if self.interrupt_count == 1:
            self.interrupted = True
            print("\n\n" + "="*90)
            print("⚠️  CTRL+C PRESSED (1st) - Saving metrics and shutting down gracefully...")
            print("     Press CTRL+C again for immediate exit without saving")
            print("="*90 + "\n")
        else:
            # Secondo Ctrl+C: shutdown immediato
            print("\n\n" + "="*90)
            print("🛑 CTRL+C PRESSED (2nd) - IMMEDIATE SHUTDOWN (no saving)")
            print("="*90 + "\n")
            import sys
            sys.exit(1)


def main():
    """Main training loop"""

    # Crea timestamp per i file di output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Setup graceful shutdown handler (Ctrl+C)
    shutdown = GracefulShutdown()

    # Carica configurazione da YAML (o usa defaults)
    config = load_training_config('train_configs.yaml')

    # Estrai parametri dalla configurazione
    USE_CURRICULUM = config['training']['use_curriculum']
    NUM_EPOCHS = config['training']['max_epochs']
    PATIENCE = config['training']['patience']
    EARLY_STOP_CH_THRESHOLD = config['training']['early_stop_ch_threshold']

    LEARNING_RATE = config['optimizer']['learning_rate']
    WEIGHT_DECAY = config['optimizer']['weight_decay']
    LR_SCHEDULER_FACTOR = config['optimizer']['lr_scheduler_factor']
    LR_SCHEDULER_PATIENCE = config['optimizer']['lr_scheduler_patience']

    LOSS_TYPE = config['loss']['loss_type']
    MARGIN_INIT = config['loss']['margin_init']
    MARGIN_FINAL = config['loss']['margin_final']
    LOSS_ALPHA = config['loss'].get('alpha', 2.0)
    LOSS_BETA = config['loss'].get('beta', 50.0)
    LOSS_LAMBDA = config['loss'].get('lambda_param', 0.5)

    # Mining strategy (progressive o statico)
    USE_PROGRESSIVE_MINING = config['mining'].get('use_progressive', False)
    RANDOM_EPOCHS = config['mining']['random_epochs']
    SEMIHARD_EPOCHS = config['mining']['semihard_epochs']
    HARDMINING_EPOCHS = config['mining']['hardmining_epochs']

    # Per backward compatibility
    MINING_STRATEGY = config['mining'].get('strategy', 'hard')
    MINING_START_EPOCH = config['mining'].get('start_epoch', 1)

    USE_PK_SAMPLING = config['batch'].get('use_pk_sampling', True)
    NUM_PATIENTS = config['batch']['num_patients_per_batch']
    NUM_ECG = config['batch']['num_ecg_per_patient']

    # Se use_pk_sampling è attivo, batch_size = P * K
    if USE_PK_SAMPLING:
        BATCH_SIZE = NUM_PATIENTS * NUM_ECG
    else:
        BATCH_SIZE = config['batch']['batch_size']

    ENCODER_INPUT = config['encoder']['input_dim']
    ENCODER_HIDDEN_DIMS = config['encoder'].get('hidden_dims', [20])  # Default [20] per backward compat
    ENCODER_EMBEDDING = config['encoder']['embedding_dim']
    ENCODER_DROPOUT = config['encoder']['dropout']
    ENCODER_NORMALIZE = config['encoder']['normalize']

    VALIDATION_SAMPLE_SIZE = config['validation']['sample_size']
    COMPUTE_SILHOUETTE = config['validation']['compute_silhouette']
    COMPUTE_BW_RATIO = config['validation']['compute_bw_ratio']

    # Test set configuration flags
    test_config = config.get('test', {})
    TEST_COMPUTE_BW_RATIO = test_config.get('compute_bw_ratio', False)
    TEST_COMPUTE_SILHOUETTE = test_config.get('compute_silhouette', False)
    TEST_COMPUTE_DB = test_config.get('compute_db', True)
    TEST_COMPUTE_GLOBAL_DISTANCES = test_config.get('compute_global_distances', True)

    # Performance optimization flags (da config 'performance')
    perf_config = config.get('performance', {})
    COMPUTE_TRAIN_BATCH_METRICS = perf_config.get('compute_train_batch_metrics', True)
    COMPUTE_VAL_BATCH_METRICS = perf_config.get('compute_val_batch_metrics', True)
    COMPUTE_VAL_GLOBAL_DISTANCES = perf_config.get('compute_val_global_distances', True)
    COMPUTE_VAL_DB_SCORE = perf_config.get('compute_val_db_score', True)
    VALIDATION_FREQUENCY = perf_config.get('validation_frequency', 1)

    # Data paths
    TRAIN_CSV = config['data']['train_csv']
    VAL_CSV = config['data']['val_csv']
    TEST_CSV = config['data']['test_csv']

    # Output paths
    BASE_DIR = Path(config['output']['base_dir'])
    RUNS_DIR = BASE_DIR / config['output']['runs_dir']

    DEVICE = torch.device(config['device'])
    if DEVICE.type == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, switching to CPU")
        DEVICE = torch.device('cpu')

    # Crea struttura folder per il run: runs/run_TIMESTAMP/
    RUN_DIR = RUNS_DIR / f'run_{timestamp}'
    MODEL_DIR = RUN_DIR / 'models'
    LOGS_DIR = RUN_DIR / 'logs'
    RESULTS_DIR = RUN_DIR / 'results'

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)
    LOGS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    # Salva copia della config nel run folder
    config_copy_path = RUN_DIR / f'config_{timestamp}.yaml'
    with open(config_copy_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    logger.info(f"Config saved to {config_copy_path}")

    # Aggiungi file handler per il run specifico
    run_log_path = LOGS_DIR / f'train_{timestamp}.log'
    run_handler = logging.FileHandler(run_log_path)
    run_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(run_handler)

    logger.info(f"Device: {DEVICE}")
    if USE_PROGRESSIVE_MINING:
        logger.info(f"Mining strategy: PROGRESSIVE (random→semi-hard→hard)")
        logger.info(f"  - Random epochs: 1-{RANDOM_EPOCHS}")
        logger.info(f"  - Semi-hard epochs: {RANDOM_EPOCHS+1}-{SEMIHARD_EPOCHS}")
        logger.info(f"  - Hard epochs: {SEMIHARD_EPOCHS+1}-{HARDMINING_EPOCHS}")
    else:
        logger.info(f"Mining strategy: STATIC ({MINING_STRATEGY})")
        logger.info(f"Mining start epoch: {MINING_START_EPOCH}")
    logger.info(f"Batch size: {BATCH_SIZE}")
    logger.info(f"Learning rate: {LEARNING_RATE}")
    logger.info(f"Margin init: {MARGIN_INIT}")

    # Load data
    print("Loading train/val data...")
    train_dataset = ECGPairDataset(TRAIN_CSV, mining_strategy="random")
    val_dataset = ECGPairDataset(VAL_CSV, mining_strategy="random")

    # Estrai features e patient_ids per training veloce
    train_features = train_dataset.features
    train_patient_ids = train_dataset.patient_ids

    print(f"Train samples: {len(train_features)}")
    print(f"Val samples: {len(val_dataset.features)}")

    # Create encoder e loss
    encoder = ECGEncoder(input_dim=ENCODER_INPUT, hidden_dims=ENCODER_HIDDEN_DIMS,
                        embedding_dim=ENCODER_EMBEDDING, dropout=ENCODER_DROPOUT,
                        normalize=ENCODER_NORMALIZE).to(DEVICE)

    # Select loss function based on configuration
    if LOSS_TYPE == 'contrastive':
        loss_fn = ContrastiveLoss(margin=MARGIN_INIT)
    elif LOSS_TYPE == 'adaptive_contrastive':
        loss_fn = AdaptiveContrastiveLoss(margin_init=MARGIN_INIT, margin_final=MARGIN_FINAL,
                                          total_epochs=NUM_EPOCHS)
    elif LOSS_TYPE == 'curriculum_contrastive':
        loss_fn = CurriculumContrastiveLoss(margin_init=MARGIN_INIT, total_epochs=NUM_EPOCHS,
                                           random_epochs=RANDOM_EPOCHS, semihard_epochs=SEMIHARD_EPOCHS)
    elif LOSS_TYPE == 'multi_similarity':
        loss_fn = MultiSimilarityLoss(alpha=LOSS_ALPHA, beta=LOSS_BETA, lambda_param=LOSS_LAMBDA)
    else:
        raise ValueError(f"Unknown loss type: {LOSS_TYPE}")

    logger.info(f"Loss function: {LOSS_TYPE}")

    # Optimizer
    optimizer = optim.AdamW(encoder.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                      factor=LR_SCHEDULER_FACTOR,
                                                      patience=LR_SCHEDULER_PATIENCE, verbose=True)

    # Training loop
    best_ch_score = 0.0
    best_epoch = 0
    patience_counter = 0
    history = {
        'epoch': [],
        'train_loss_mean': [], 'train_loss_std': [],
        'train_d_intra_mean': [], 'train_d_intra_std': [],
        'train_d_inter_mean': [], 'train_d_inter_std': [],
        'train_db_mean': [], 'train_db_std': [],
        'train_ch_mean': [], 'train_ch_std': [],
        'train_active_negatives_pct_mean': [], 'train_active_negatives_pct_std': [],
        'val_d_intra_mean': [], 'val_d_intra_std': [],
        'val_d_inter_mean': [], 'val_d_inter_std': [],
        'val_db_mean': [], 'val_db_std': [],
        'val_ch_mean': [], 'val_ch_std': [],
        'val_ch_score': [], 'val_db_index': [], 'val_bw_ratio': [], 'val_silhouette': [],
        'val_d_intra_global': [], 'val_d_inter_global': [],
        'mining_strategy': []
    }

    # Storage per batch metrics
    all_batch_metrics_train = []  # lista di (epoch, batch_idx, metrics)
    all_batch_metrics_val = []    # lista di (epoch, batch_idx, metrics)

    # Print training header
    print("\n" + "="*90)
    print("METRIC LEARNING TRAINING - ECG Embedding Clustering")
    print("="*90)
    print(f"Device: {DEVICE}")
    print(f"Batch size: {BATCH_SIZE} (P=64 x K=4)")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Max epochs: {NUM_EPOCHS}")
    print("="*90 + "\n")

    for epoch in range(1, NUM_EPOCHS + 1):
        # Check per graceful shutdown
        if shutdown.interrupted:
            logger.info("Training interrupted by user (Ctrl+C)")
            break

        # Aggiorna epoch per loss functions che lo supportano (adaptive e curriculum)
        if hasattr(loss_fn, 'update_epoch'):
            loss_fn.update_epoch(epoch)

        # Scegli mining strategy basata su configurazione
        if USE_PROGRESSIVE_MINING:
            # Mining progressivo: random → semi-hard → hard
            mining_strategy = get_mining_strategy_for_epoch(epoch, RANDOM_EPOCHS, SEMIHARD_EPOCHS, HARDMINING_EPOCHS)
        else:
            # Mining statico (backward compatible)
            if epoch < MINING_START_EPOCH:
                mining_strategy = "random"
            else:
                mining_strategy = MINING_STRATEGY

        # Crea batch indices direttamente con PKSampler
        pk_sampler = PKSampler(train_patient_ids, P=NUM_PATIENTS, K=NUM_ECG, shuffle=True)
        batch_indices_list = list(pk_sampler)

        # Train epoch (con flag per computing batch metrics)
        train_loss, batch_metrics_train = train_epoch(encoder, loss_fn, optimizer, train_features, train_patient_ids,
                                                      batch_indices_list, DEVICE, epoch, mining_strategy,
                                                      compute_batch_metrics_flag=COMPUTE_TRAIN_BATCH_METRICS)

        # Salva batch metrics train
        for batch_idx, batch_metrics in enumerate(batch_metrics_train):
            all_batch_metrics_train.append({
                'epoch': epoch,
                'batch_idx': batch_idx,
                **batch_metrics
            })

        # Aggregate batch metrics train (handle caso quando compute_train_batch_metrics=False)
        train_metrics_agg = {}

        # Loss è sempre disponibile
        train_metrics_agg['loss'] = np.array([m['loss'] for m in batch_metrics_train])

        # Metriche clustering sono disponibili solo se compute_train_batch_metrics=True
        if batch_metrics_train and 'd_intra' in batch_metrics_train[0]:
            train_metrics_agg['d_intra'] = np.array([m['d_intra'] for m in batch_metrics_train])
            train_metrics_agg['d_inter'] = np.array([m['d_inter'] for m in batch_metrics_train])
            train_metrics_agg['db'] = np.array([m['db'] for m in batch_metrics_train])
            train_metrics_agg['ch'] = np.array([m['ch'] for m in batch_metrics_train])
        else:
            # Se non disponibili: set defaults vuoti (non salvati in history)
            train_metrics_agg['d_intra'] = np.array([])
            train_metrics_agg['d_inter'] = np.array([])
            train_metrics_agg['db'] = np.array([])
            train_metrics_agg['ch'] = np.array([])

        # Validate con batch metrics (con flags di performance)
        # Solo valida se è multiplo di VALIDATION_FREQUENCY
        if epoch % VALIDATION_FREQUENCY == 0:
            val_metrics, batch_metrics_val = validate_with_batch_metrics(
                encoder, val_dataset, DEVICE, sample_size=VALIDATION_SAMPLE_SIZE,
                compute_silhouette=COMPUTE_SILHOUETTE, compute_bw_ratio=COMPUTE_BW_RATIO,
                compute_batch_metrics_flag=COMPUTE_VAL_BATCH_METRICS,
                compute_db=COMPUTE_VAL_DB_SCORE,
                compute_global_distances=COMPUTE_VAL_GLOBAL_DISTANCES)
        else:
            # Se non è tempo di validare: salta
            val_metrics = None
            batch_metrics_val = []

        # Salva batch metrics val
        for batch_idx, batch_metrics in enumerate(batch_metrics_val):
            all_batch_metrics_val.append({
                'epoch': epoch,
                'batch_idx': batch_idx,
                **batch_metrics
            })

        # Extract validation metrics (se validation è stata eseguita in questa epoca)
        if val_metrics is not None:
            ch_score = val_metrics.get('calinski_harabasz', 0.0)
            db_index = val_metrics.get('davies_bouldin', float('inf'))
            bw_ratio = val_metrics.get('between_within_ratio', 0.0)
            silhouette = val_metrics.get('silhouette', None)
            val_d_intra_global = val_metrics.get('d_intra_global', 0.0)
            val_d_inter_global = val_metrics.get('d_inter_global', 0.0)
        else:
            # Validazione saltata (validation_frequency > 1): set defaults
            ch_score = 0.0
            db_index = float('inf')
            bw_ratio = None
            silhouette = None
            val_d_intra_global = 0.0
            val_d_inter_global = 0.0

        # Calcola % negativi attivi media (se disponibile)
        train_active_negatives_pct_mean = 0.0
        if 'active_negatives_pct' in batch_metrics_train[0]:
            active_pct_arr = np.array([m['active_negatives_pct'] for m in batch_metrics_train])
            train_active_negatives_pct_mean = active_pct_arr.mean()

        # Pretty print epoch summary
        print("\n" + "="*90)
        print(f"EPOCH {epoch:2d} | Strategy: {mining_strategy:12s} | Patience: {patience_counter:2d}/{PATIENCE}")
        print("="*90)
        print(f"  Train Loss:          {train_loss:.6f}")
        # Stampa train distances solo se disponibili (compute_train_batch_metrics=True)
        if len(train_metrics_agg['d_intra']) > 0:
            print(f"  Train Distances:")
            print(f"    - Intra-patient:     {train_metrics_agg['d_intra'].mean():.4f} ± {train_metrics_agg['d_intra'].std():.4f}")
            print(f"    - Inter-patient:     {train_metrics_agg['d_inter'].mean():.4f} ± {train_metrics_agg['d_inter'].std():.4f}")
        else:
            print(f"  Train Distances:     (disabled in config)")
        if train_active_negatives_pct_mean > 0:
            print(f"  Active Negatives:    {train_active_negatives_pct_mean:.1f}%")
        if val_metrics is not None:
            print(f"  Val Distances (global):")
            print(f"    - Intra-patient:     {val_d_intra_global:.4f}")
            print(f"    - Inter-patient:     {val_d_inter_global:.4f}")
            print(f"  Val Clustering Metrics:")
            print(f"    - Calinski-Harabasz: {ch_score:8.2f} (target: >15)")
            print(f"    - Davies-Bouldin:    {db_index:8.2f} (target: <2.5)")
        else:
            print(f"  Val Metrics: (skipped - validation_frequency)")
            print(f"    - Next validation: epoch {epoch + (VALIDATION_FREQUENCY - epoch % VALIDATION_FREQUENCY)}")
        if bw_ratio is not None:
            print(f"    - Between-Within:    {bw_ratio:8.2f} (target: >4.0)")
        else:
            print(f"    - Between-Within:    skipped (computed only at final eval)")
        if silhouette is not None:
            print(f"    - Silhouette:        {silhouette:8.3f} (range: [-1, 1])")
        else:
            print(f"    - Silhouette:        skipped (computed only at final eval)")
        print("="*90 + "\n")

        silhouette_str = f"{silhouette:.3f}" if silhouette is not None else "skipped"
        bw_ratio_str = f"{bw_ratio:.2f}" if bw_ratio is not None else "skipped"
        active_neg_str = f"{train_active_negatives_pct_mean:.1f}%" if train_active_negatives_pct_mean > 0 else "N/A"

        # Log message (condizionato a validation eseguita o no)
        if val_metrics is not None:
            # Training metrics da stampare (se disponibili)
            train_d_intra_str = f"{train_metrics_agg['d_intra'].mean():.4f}" if len(train_metrics_agg['d_intra']) > 0 else "N/A"
            train_d_inter_str = f"{train_metrics_agg['d_inter'].mean():.4f}" if len(train_metrics_agg['d_inter']) > 0 else "N/A"

            logger.info(f"Epoch {epoch} | Train Loss: {train_loss:.6f} | "
                       f"d_intra_train: {train_d_intra_str} | "
                       f"d_inter_train: {train_d_inter_str} | "
                       f"Active Neg: {active_neg_str} | "
                       f"d_intra_val: {val_d_intra_global:.4f} | d_inter_val: {val_d_inter_global:.4f} | "
                       f"CH: {ch_score:.2f} | DB: {db_index:.2f} | Strategy: {mining_strategy}")
        else:
            logger.info(f"Epoch {epoch} | Train Loss: {train_loss:.6f} | "
                       f"Active Neg: {active_neg_str} | "
                       f"Val: SKIPPED (validation_frequency) | Strategy: {mining_strategy}")

        # Save history con aggregazione batch metrics
        history['epoch'].append(epoch)
        history['mining_strategy'].append(mining_strategy)

        # Train metrics aggregate
        history['train_loss_mean'].append(float(train_metrics_agg['loss'].mean()))
        history['train_loss_std'].append(float(train_metrics_agg['loss'].std()))

        # Distanze train (disponibili solo se compute_train_batch_metrics=True)
        if len(train_metrics_agg['d_intra']) > 0:
            history['train_d_intra_mean'].append(float(train_metrics_agg['d_intra'].mean()))
            history['train_d_intra_std'].append(float(train_metrics_agg['d_intra'].std()))
            history['train_d_inter_mean'].append(float(train_metrics_agg['d_inter'].mean()))
            history['train_d_inter_std'].append(float(train_metrics_agg['d_inter'].std()))
            history['train_db_mean'].append(float(train_metrics_agg['db'].mean()))
            history['train_db_std'].append(float(train_metrics_agg['db'].std()))
            history['train_ch_mean'].append(float(train_metrics_agg['ch'].mean()))
            history['train_ch_std'].append(float(train_metrics_agg['ch'].std()))
        else:
            # Se disabilitato: NaN per coerenza
            history['train_d_intra_mean'].append(np.nan)
            history['train_d_intra_std'].append(np.nan)
            history['train_d_inter_mean'].append(np.nan)
            history['train_d_inter_std'].append(np.nan)
            history['train_db_mean'].append(np.nan)
            history['train_db_std'].append(np.nan)
            history['train_ch_mean'].append(np.nan)
            history['train_ch_std'].append(np.nan)

        # % negativi attivi (solo se loss contrastive)
        if 'active_negatives_pct' in batch_metrics_train[0]:
            active_pct_arr = np.array([m['active_negatives_pct'] for m in batch_metrics_train])
            history['train_active_negatives_pct_mean'].append(float(active_pct_arr.mean()))
            history['train_active_negatives_pct_std'].append(float(active_pct_arr.std()))
        else:
            history['train_active_negatives_pct_mean'].append(0.0)
            history['train_active_negatives_pct_std'].append(0.0)

        # Val metrics aggregate (da batch_metrics_val, disponibili solo se compute_val_batch_metrics=True)
        if batch_metrics_val and 'd_intra' in batch_metrics_val[0]:
            val_metrics_agg = {
                'd_intra': np.array([m['d_intra'] for m in batch_metrics_val]),
                'd_inter': np.array([m['d_inter'] for m in batch_metrics_val]),
                'db': np.array([m['db'] for m in batch_metrics_val]),
                'ch': np.array([m['ch'] for m in batch_metrics_val])
            }
            history['val_d_intra_mean'].append(float(val_metrics_agg['d_intra'].mean()))
            history['val_d_intra_std'].append(float(val_metrics_agg['d_intra'].std()))
            history['val_d_inter_mean'].append(float(val_metrics_agg['d_inter'].mean()))
            history['val_d_inter_std'].append(float(val_metrics_agg['d_inter'].std()))
            history['val_db_mean'].append(float(val_metrics_agg['db'].mean()))
            history['val_db_std'].append(float(val_metrics_agg['db'].std()))
            history['val_ch_mean'].append(float(val_metrics_agg['ch'].mean()))
            history['val_ch_std'].append(float(val_metrics_agg['ch'].std()))
        else:
            # Se disabilitato o non validato questa epoca: NaN
            history['val_d_intra_mean'].append(np.nan)
            history['val_d_intra_std'].append(np.nan)
            history['val_d_inter_mean'].append(np.nan)
            history['val_d_inter_std'].append(np.nan)
            history['val_db_mean'].append(np.nan)
            history['val_db_std'].append(np.nan)
            history['val_ch_mean'].append(np.nan)
            history['val_ch_std'].append(np.nan)

        # Val metrics dalle metriche aggregate (non batch)
        history['val_ch_score'].append(ch_score)
        history['val_db_index'].append(db_index)
        history['val_bw_ratio'].append(bw_ratio)
        history['val_silhouette'].append(silhouette)
        history['val_d_intra_global'].append(val_d_intra_global)
        history['val_d_inter_global'].append(val_d_inter_global)

        # Salva modello ad ogni epoca
        epoch_model_path = MODEL_DIR / f'encoder_epoch_{epoch:02d}.pth'
        torch.save(encoder.state_dict(), epoch_model_path)
        logger.info(f"Epoch model saved to {epoch_model_path}")

        # Early stopping (solo se validation è stata eseguita)
        if val_metrics is not None and ch_score > best_ch_score:
            best_ch_score = ch_score
            best_epoch = epoch
            patience_counter = 0

            # Salva best model
            model_path = MODEL_DIR / 'encoder_best.pth'
            torch.save(encoder.state_dict(), model_path)
            logger.info(f"Best model saved to {model_path} (CH={ch_score:.2f})")
        else:
            # Incrementa patience solo se validation è stata eseguita
            if val_metrics is not None:
                patience_counter += 1

        # Learning rate scheduling (solo se validation è stata eseguita)
        if val_metrics is not None:
            scheduler.step(ch_score)

            # Collapse detection (solo se validation è stata eseguita)
            if ch_score < EARLY_STOP_CH_THRESHOLD:
                logger.warning(f"COLLAPSE DETECTED at epoch {epoch}! CH={ch_score:.2f} (threshold={EARLY_STOP_CH_THRESHOLD})")
                break

        # Early stop
        if patience_counter >= PATIENCE:
            logger.info(f"Early stopping at epoch {epoch}")
            break

    # Save training history (aggregate per epoca)
    history_df = pd.DataFrame(history)
    history_path = LOGS_DIR / 'train_history.csv'
    history_df.to_csv(history_path, index=False)
    logger.info(f"Training history saved to {history_path}")

    # Save batch metrics train
    if all_batch_metrics_train:
        batch_metrics_train_df = pd.DataFrame(all_batch_metrics_train)
        batch_metrics_train_path = LOGS_DIR / 'batch_metrics_train.csv'
        batch_metrics_train_df.to_csv(batch_metrics_train_path, index=False)
        logger.info(f"Batch metrics train saved to {batch_metrics_train_path}")

    # Save batch metrics validation
    if all_batch_metrics_val:
        batch_metrics_val_df = pd.DataFrame(all_batch_metrics_val)
        batch_metrics_val_path = LOGS_DIR / 'batch_metrics_val.csv'
        batch_metrics_val_df.to_csv(batch_metrics_val_path, index=False)
        logger.info(f"Batch metrics val saved to {batch_metrics_val_path}")

    # Load best model (se esiste)
    model_path = MODEL_DIR / 'encoder_best.pth'
    if model_path.exists():
        encoder.load_state_dict(torch.load(model_path))
        logger.info(f"Loaded best model from {model_path}")
    else:
        logger.warning(f"Best model not found at {model_path}, using current model")

    # Final evaluation on test set (skip se interrotto da utente)
    if shutdown.interrupted:
        print("\n" + "="*90)
        print("⚠️  TEST SET EVALUATION SKIPPED (training interrupted)")
        print("="*90)
        logger.info("Test set evaluation skipped due to user interrupt")
        test_metrics = {}
    else:
        # Final evaluation on test set (con tutte le metriche)
        print("\n" + "="*90)
        print("FINAL EVALUATION ON TEST SET")
        print("="*90)
        logger.info("Final evaluation on test set...")
        test_dataset = ECGPairDataset(TEST_CSV, mining_strategy="random")
        test_metrics = validate(encoder, test_dataset, DEVICE, sample_size=None,
                               compute_silhouette=TEST_COMPUTE_SILHOUETTE,
                               compute_bw_ratio=TEST_COMPUTE_BW_RATIO,
                               compute_db=TEST_COMPUTE_DB,
                               compute_global_distances=TEST_COMPUTE_GLOBAL_DISTANCES)

        logger.info(f"Final metrics: {test_metrics}")

        # Stampa test metrics
        print("\nTest Set Results:")
        print(f"  Calinski-Harabasz: {test_metrics.get('calinski_harabasz', 'N/A'):.2f}" if test_metrics.get('calinski_harabasz') else "  Calinski-Harabasz: N/A")
        if test_metrics.get('davies_bouldin') is not None and test_metrics.get('davies_bouldin') != float('inf'):
            print(f"  Davies-Bouldin:    {test_metrics.get('davies_bouldin', 'N/A'):.2f}")
        else:
            print(f"  Davies-Bouldin:    (disabled in config)")
        if test_metrics.get('d_intra_global'):
            print(f"  Intra-patient:     {test_metrics.get('d_intra_global', 0.0):.4f}")
            print(f"  Inter-patient:     {test_metrics.get('d_inter_global', 0.0):.4f}")
        else:
            print(f"  Distances:         (disabled in config)")
        if test_metrics.get('between_within_ratio'):
            print(f"  Between-Within:    {test_metrics.get('between_within_ratio', 0.0):.2f}")
        else:
            print(f"  Between-Within:    (disabled in config)")
        if test_metrics.get('silhouette') is not None:
            print(f"  Silhouette:        {test_metrics.get('silhouette', 0.0):.3f}")
        else:
            print(f"  Silhouette:        (disabled in config)")

    # Save results (gestendo valori None e inf per JSON)
    def safe_float(v):
        """Converte valori a float, gestendo None e inf"""
        if v is None:
            return None
        if isinstance(v, str):
            return v
        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            return None
        return float(v)

    results = {
        'best_epoch': int(best_epoch),
        'best_ch_score': float(best_ch_score),
        'test_metrics': {k: safe_float(v) for k, v in test_metrics.items()} if test_metrics else {}
    }

    results_path = RESULTS_DIR / 'metrics.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Results saved to {results_path}")

    # Final summary
    print("\n" + "="*90)
    if shutdown.interrupted:
        print("⚠️  TRAINING INTERRUPTED BY USER (Ctrl+C)")
        print(f"✓ Metrics saved in: {LOGS_DIR}/")
        print(f"  - train_history.csv (epochs aggregate)")
        print(f"  - batch_metrics_train.csv (per-batch details)")
        print(f"  - batch_metrics_val.csv (validation per-batch)")
        logger.info("Training interrupted - metrics saved successfully")
    else:
        print("✓ TRAINING COMPLETED SUCCESSFULLY")
        print(f"✓ Results saved to: {results_path}")
        logger.info("Training completed successfully!")
    print("="*90 + "\n")


if __name__ == '__main__':
    main()
