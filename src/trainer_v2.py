"""
Trainer semplificato per metric learning con contrastive loss.
- Contrastive loss con margin
- PK sampling per batch
- Curriculum mining: random → semi-hard → hard
- Metriche: CH, DB, loss, % negativi usati
- Salva modello ogni epoca e best model
"""

import os
import sys
import yaml
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
import signal

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.ecg_encoder import ECGEncoder
from src.ecg_metric_dataset import ECGPairDataset, PKSampler
from src.evaluation_utils import compute_batch_metrics


# Graceful shutdown state
shutdown_requested = False
force_shutdown_count = 0


def signal_handler(signum, frame):
    """Gestisce CTRL+C: graceful alla 1a pressione, force alla 2a"""
    global shutdown_requested, force_shutdown_count

    force_shutdown_count += 1

    if force_shutdown_count == 1:
        shutdown_requested = True
        print("\n\n⚠️  Graceful shutdown requested. Finishing current epoch and saving state...")
        print("   Press CTRL+C again to force immediate shutdown.\n")
    else:
        print("\n\n🛑 Force shutdown! Exiting immediately...\n")
        sys.exit(1)


class ContrastiveLoss(nn.Module):
    """Contrastive loss con margin"""
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, labels):
        """
        Args:
            anchor: (N, D) embeddings
            positive: (N, D) embeddings
            labels: (N,) binary labels (0=same patient, 1=different patient)
        Returns:
            loss, active_negatives_pct
        """
        # Distanza cosine
        cosine_sim = F.cosine_similarity(anchor, positive, dim=1)
        cosine_dist = 1 - cosine_sim

        # Loss
        pos_loss = cosine_dist[labels == 0].mean() if (labels == 0).sum() > 0 else 0
        neg_loss = torch.relu(self.margin - cosine_dist[labels == 1])
        neg_loss = neg_loss.mean() if (labels == 1).sum() > 0 else 0

        loss = pos_loss + neg_loss

        # % negativi attivi (che violano margin)
        if (labels == 1).sum() > 0:
            active_negs = (cosine_dist[labels == 1] < self.margin).sum().float()
            active_negatives_pct = (active_negs / (labels == 1).sum()).item() * 100
        else:
            active_negatives_pct = 0.0

        return loss, active_negatives_pct


def get_mining_strategy(epoch, config_mining):
    """Ritorna mining strategy per epoch"""
    random_epochs = config_mining['random_epochs']
    semihard_epochs = config_mining['semihard_epochs']
    hardmining_epochs = config_mining['hardmining_epochs']

    if epoch <= random_epochs:
        return "random"
    elif epoch <= semihard_epochs:
        return "semi-hard"
    else:
        return "hard"


def create_pairs_batch(batch_indices, features, patient_ids, device, mining_strategy="random",
                       embeddings=None, margin=2.0):
    """
    Crea pairs per batch.
    Per random mining: genera pairs random nel batch
    Per semi-hard/hard: usa embeddings per selezionare pairs
    """
    batch_patient_ids = patient_ids[batch_indices]
    batch_features = features[batch_indices]

    anchors = []
    others = []
    labels = []

    for i, idx in enumerate(batch_indices):
        anchor_patient = batch_patient_ids[i]

        # Positivo: stesso paziente nel batch
        pos_mask = (batch_patient_ids == anchor_patient)
        pos_indices = np.where(pos_mask)[0]
        pos_indices = pos_indices[pos_indices != i]

        if len(pos_indices) > 0:
            pos_i = np.random.choice(pos_indices)
            anchors.append(batch_features[i])
            others.append(batch_features[pos_i])
            labels.append(0)  # positivo

        # Negativo: paziente diverso nel batch
        neg_mask = (batch_patient_ids != anchor_patient)
        neg_indices = np.where(neg_mask)[0]

        if len(neg_indices) > 0:
            if mining_strategy == "random":
                neg_i = np.random.choice(neg_indices)
            else:
                # Semi-hard o hard mining: seleziona base su embeddings
                if embeddings is not None:
                    anchor_emb = embeddings[idx]
                    distances = []
                    for neg_i in neg_indices:
                        d = np.linalg.norm(embeddings[batch_indices[neg_i]] - anchor_emb)
                        distances.append((d, neg_i))

                    distances.sort()

                    if mining_strategy == "semi-hard":
                        # Semi-hard: distanza tra d_pos e d_pos+margin
                        pos_idx = batch_indices[pos_indices[0]]
                        d_pos = np.linalg.norm(embeddings[pos_idx] - anchor_emb)
                        semi_hard = [idx for d, idx in distances if d_pos < d < d_pos + margin]
                        if semi_hard:
                            neg_i = np.random.choice(semi_hard)
                        else:
                            neg_i = distances[-1][1]  # hardest
                    else:  # hard
                        neg_i = distances[-1][1]  # hardest negative
                else:
                    neg_i = np.random.choice(neg_indices)

            anchors.append(batch_features[i])
            others.append(batch_features[neg_i])
            labels.append(1)  # negativo

    anchors = torch.from_numpy(np.array(anchors)).float().to(device)
    others = torch.from_numpy(np.array(others)).float().to(device)
    labels = torch.from_numpy(np.array(labels)).float().to(device)

    return anchors, others, labels


def train_epoch(encoder, loss_fn, optimizer, features, patient_ids, batch_indices_list,
                device, mining_strategy="random", embeddings=None, margin=2.0):
    """Train un'epoca"""
    encoder.train()
    total_loss = 0.0
    total_active_neg_pct = 0.0
    num_batches = 0

    pbar = tqdm(batch_indices_list, desc=f"Train [{mining_strategy}]",
                position=0, leave=True, ncols=100)

    for batch_indices in pbar:
        anchors, others, labels = create_pairs_batch(
            batch_indices, features, patient_ids, device, mining_strategy, embeddings, margin)

        optimizer.zero_grad()

        anchor_embs = encoder(anchors)
        other_embs = encoder(others)

        loss, active_neg_pct = loss_fn(anchor_embs, other_embs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_active_neg_pct += active_neg_pct
        num_batches += 1

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'active_neg%': f'{active_neg_pct:.1f}%'
        })

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_active_neg_pct = total_active_neg_pct / num_batches if num_batches > 0 else 0.0

    return avg_loss, avg_active_neg_pct


@torch.no_grad()
def validate(encoder, val_dataset, device, sample_size=5000):
    """Valida e ritorna metriche: CH, DB"""
    encoder.eval()

    print(f"Computing embeddings for validation...")

    # Sample
    indices = np.random.choice(len(val_dataset.features),
                              min(sample_size, len(val_dataset.features)),
                              replace=False)

    all_embeddings = []
    all_patient_ids = []

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

    # Metriche
    print("Computing validation metrics...")
    metrics = compute_batch_metrics(all_embeddings, all_patient_ids)

    return metrics, all_embeddings


def setup_logging(run_dir):
    """Setup logging"""
    log_file = run_dir / 'train.log'

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file)
        ]
    )

    return logging.getLogger(__name__)


def main():
    global shutdown_requested, force_shutdown_count

    # Setup signal handler per CTRL+C
    signal.signal(signal.SIGINT, signal_handler)

    # Timestamp e directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = Path('/Users/arjuna/Progetti/siamese')
    run_dir = base_dir / 'runs_v2' / f'run_{timestamp}'
    run_dir.mkdir(parents=True, exist_ok=True)

    model_dir = run_dir / 'models'
    model_dir.mkdir(exist_ok=True)

    logger = setup_logging(run_dir)
    logger.info(f"Run directory: {run_dir}")

    # Carica config
    config_path = base_dir / 'train_config_v2.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    logger.info(f"Config loaded from {config_path}")

    # Salva copia config
    with open(run_dir / f'config_{timestamp}.yaml', 'w') as f:
        yaml.dump(config, f)

    # Parametri
    device = torch.device(config['device'])
    if device.type == 'cuda' and not torch.cuda.is_available():
        device = torch.device('cpu')
        logger.warning("CUDA not available, using CPU")

    logger.info(f"Device: {device}")

    # Dataset
    logger.info("Loading datasets...")
    train_dataset = ECGPairDataset(config['data']['train_csv'], mining_strategy="random")
    val_dataset = ECGPairDataset(config['data']['val_csv'], mining_strategy="random")

    train_features = train_dataset.features
    train_patient_ids = train_dataset.patient_ids

    logger.info(f"Train samples: {len(train_features)}")
    logger.info(f"Val samples: {len(val_dataset.features)}")

    # Encoder
    encoder = ECGEncoder(
        input_dim=config['encoder']['input_dim'],
        hidden_dims=config['encoder'].get('hidden_dims', [20]),
        embedding_dim=config['encoder']['embedding_dim'],
        dropout=config['encoder']['dropout'],
        normalize=config['encoder']['normalize']
    ).to(device)

    # Loss e optimizer
    loss_fn = ContrastiveLoss(margin=config['loss']['margin']).to(device)
    optimizer = optim.AdamW(
        encoder.parameters(),
        lr=config['optimizer']['learning_rate'],
        weight_decay=config['optimizer'].get('weight_decay', 1e-5)
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=config['optimizer'].get('lr_scheduler_factor', 0.5),
        patience=config['optimizer'].get('lr_scheduler_patience', 5),
        verbose=True
    )

    logger.info(f"Encoder: {encoder}")
    logger.info(f"Loss: Contrastive (margin={config['loss']['margin']})")

    # Training loop
    best_ch = 0.0
    best_epoch = 0
    patience_counter = 0

    history = {
        'epoch': [],
        'mining_strategy': [],
        'train_loss': [],
        'train_active_neg_pct': [],
        'val_ch': [],
        'val_db': []
    }

    num_epochs = config['training']['max_epochs']
    patience = config['training']['patience']

    print("\n" + "="*90)
    print("METRIC LEARNING TRAINER V2 - Contrastive Loss")
    print("="*90)
    print(f"Max epochs: {num_epochs}")
    print(f"Patience: {patience}")
    print(f"Device: {device}")
    print("="*90 + "\n")

    for epoch in range(1, num_epochs + 1):
        # Check graceful shutdown request
        if shutdown_requested:
            logger.info(f"Graceful shutdown at epoch {epoch}")
            break

        # Mining strategy
        mining_strategy = get_mining_strategy(epoch, config['mining'])
        logger.info(f"Epoch {epoch} - Mining: {mining_strategy}")

        # PK Sampler
        pk_sampler = PKSampler(
            train_patient_ids,
            P=config['batch']['num_patients'],
            K=config['batch']['num_ecg'],
            shuffle=True
        )
        batch_indices_list = list(pk_sampler)

        # Train
        train_loss, train_active_neg_pct = train_epoch(
            encoder, loss_fn, optimizer, train_features, train_patient_ids,
            batch_indices_list, device, mining_strategy
        )

        logger.info(f"Epoch {epoch} - Train loss: {train_loss:.6f}, Active neg: {train_active_neg_pct:.1f}%")

        # Validate
        val_metrics, val_embeddings = validate(encoder, val_dataset, device)

        ch_score = val_metrics['ch']
        db_score = val_metrics['db']

        # Salva metriche
        history['epoch'].append(epoch)
        history['mining_strategy'].append(mining_strategy)
        history['train_loss'].append(train_loss)
        history['train_active_neg_pct'].append(train_active_neg_pct)
        history['val_ch'].append(ch_score)
        history['val_db'].append(db_score)

        # Stampa summary
        print("\n" + "="*90)
        print(f"EPOCH {epoch:3d} | Strategy: {mining_strategy:10s} | Patience: {patience_counter:2d}/{patience}")
        print("="*90)
        print(f"  Train Loss:     {train_loss:.6f}")
        print(f"  Active Neg:     {train_active_neg_pct:.1f}%")
        print(f"  Val CH:         {ch_score:.2f}")
        print(f"  Val DB:         {db_score:.2f}")
        print("="*90 + "\n")

        logger.info(f"Val CH: {ch_score:.2f}, DB: {db_score:.2f}")

        # Salva modello ogni epoca
        epoch_model_path = model_dir / f'model_epoch_{epoch:04d}.pth'
        torch.save(encoder.state_dict(), epoch_model_path)

        # Early stopping e best model
        if ch_score > best_ch:
            best_ch = ch_score
            best_epoch = epoch
            patience_counter = 0

            best_model_path = model_dir / 'model_best.pth'
            torch.save(encoder.state_dict(), best_model_path)
            logger.info(f"New best model saved (CH={ch_score:.2f})")
        else:
            patience_counter += 1

        # Learning rate scheduling
        scheduler.step(ch_score)

        # Early stop
        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch}")
            break

    # Salva history
    history_df = pd.DataFrame(history)
    history_path = run_dir / 'history.csv'
    history_df.to_csv(history_path, index=False)
    logger.info(f"History saved to {history_path}")

    # Salva results
    results = {
        'best_epoch': int(best_epoch),
        'best_ch': float(best_ch),
        'final_db': float(history['val_db'][-1]) if history['val_db'] else 0.0,
        'shutdown_type': 'graceful' if shutdown_requested else 'normal'
    }

    results_path = run_dir / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {results_path}")

    # Stampa finale
    status_msg = "SHUTDOWN (graceful)" if shutdown_requested else "COMPLETED"
    print("\n" + "="*90)
    print(f"✓ TRAINING {status_msg}")
    print(f"✓ Run dir: {run_dir}")
    print(f"✓ Best epoch: {best_epoch} (CH={best_ch:.2f})")
    print("="*90 + "\n")


if __name__ == '__main__':
    main()
